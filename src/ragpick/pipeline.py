"""The :class:`Pipeline`: builds components from config and runs ingest / query."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ragpick import interfaces, tracing
from ragpick.cache import Cache, CachedEmbedder, wrap_generator
from ragpick.components.generators import parse_citations
from ragpick.config import ComponentSpec, RagConfig, TenancyConfig, build_config, load_config
from ragpick.conversation import ConversationManager, make_store, owner_key
from ragpick.errors import ConfigError, IndexNotFoundError, TenantError
from ragpick.filters import acl_filter
from ragpick.models import Answer, RetrievedChunk, StreamEvent, Usage
from ragpick.registry import Registry, registry
from ragpick.tracing import set_attributes, span

logger = logging.getLogger(__name__)

#: Dependencies the pipeline injects into retrievers (and any other component)
#: whose constructor declares a parameter with one of these names. ``index_dir`` is
#: the absolute index directory (a :class:`~pathlib.Path`).
INJECTABLE = ("embedder", "store", "keyword_index", "generator", "index_dir")

_PROTOCOLS: dict[str, type] = {
    "loader": interfaces.Loader,
    "chunker": interfaces.Chunker,
    "embedder": interfaces.Embedder,
    "store": interfaces.VectorStore,
    "keyword_index": interfaces.KeywordIndex,
    "retriever": interfaces.Retriever,
    "reranker": interfaces.Reranker,
    "generator": interfaces.Generator,
}

MANIFEST = "ragpick.json"

#: ``progress(stage, done, total)`` callback used by :meth:`Pipeline.ingest`.
ProgressCallback = Callable[[str, int, int], None]


_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_tenant(tenant: str) -> str:
    """Check a tenant id (letters, digits, ``-``, ``_``; max 64): it becomes a directory name.

    Raises:
        TenantError: the id could escape the index directory or is otherwise invalid.
    """
    if not isinstance(tenant, str) or not _TENANT_RE.match(tenant):
        raise TenantError(
            f"Invalid tenant id {tenant!r}: use 1-64 letters, digits, '-' or '_', "
            "starting with a letter or digit"
        )
    return tenant


def _fingerprint(spec: ComponentSpec) -> dict[str, Any]:
    return {"type": spec.type, "params": spec.params}


class IngestStats(BaseModel):
    """Summary of an ingest run."""

    documents: int
    chunks: int
    seconds: float
    index_dir: str


class Pipeline:
    """A runnable RAG pipeline.

    Example::

        p = Pipeline.from_yaml("rag.yaml")
        p.ingest("./docs")
        answer = p.query("How do I reset my password?", groups=["support"])

    Components are created from the config through the :data:`~ragpick.registry`.
    Components receive ``embedder``, ``store``, ``keyword_index``, ``generator`` and
    ``index_dir`` by constructor parameter name. Pass
    ``components={"generator": my_generator}`` to inject ready-made instances instead.
    """

    def __init__(
        self,
        config: RagConfig,
        *,
        components: Mapping[str, Any] | None = None,
        component_registry: Registry | None = None,
    ) -> None:
        self.config = config
        self._registry = component_registry or registry
        self._injected = dict(components or {})
        self._loaded = False
        self._tenants: OrderedDict[str, Pipeline] = OrderedDict()
        self._conversations: ConversationManager | None = None
        self._tenant_lock = threading.Lock()
        c = config.components
        self.cache: Cache | None = None
        if config.cache.enabled:
            self.cache = Cache(
                config.resolve_path(config.cache.dir) / "cache.sqlite", config.cache.ttl_seconds
            )
        # the generator is built first so loaders (vision) and retrievers can use it
        self.generator: interfaces.Generator = self._build("generator", c.generator)
        if c.generator.type == "fake" and "generator" not in self._injected:
            logger.info("Using the fake generator: answers are extractive placeholders")
        if self.cache is not None and config.cache.generations:
            fingerprint = {
                **_fingerprint(c.generator),
                "prompt": getattr(self.generator, "system_prompt", None),
            }
            self.generator = wrap_generator(self.generator, self.cache, fingerprint)
        self.loader: interfaces.Loader = self._build("loader", c.loader)
        self.chunker: interfaces.Chunker = self._build("chunker", c.chunker)
        self.embedder: interfaces.Embedder = self._build("embedder", c.embedder)
        if self.cache is not None and config.cache.embeddings:
            self.embedder = CachedEmbedder(self.embedder, self.cache, _fingerprint(c.embedder))
        self.store: interfaces.VectorStore = self._build("store", c.store)
        self.keyword_index: interfaces.KeywordIndex | None = None
        retriever_needs = self._registry.accepted_params("retriever", c.retriever.type)
        keyword_spec = c.keyword_index
        if (
            keyword_spec is None
            and retriever_needs is not None
            and "keyword_index" in retriever_needs
        ):
            logger.info("retriever '%s' needs a keyword index; using bm25", c.retriever.type)
            keyword_spec = ComponentSpec(type="bm25")
        if keyword_spec is not None or "keyword_index" in self._injected:
            self.keyword_index = self._build("keyword_index", keyword_spec)
        self.retriever: interfaces.Retriever = self._build("retriever", c.retriever)
        self.reranker: interfaces.Reranker = self._build("reranker", c.reranker)
        self.exporters: list[Any] = [
            self._registry.create("trace_exporter", spec.type, **self._exporter_params(spec))
            for spec in config.tracing.exporters
        ]

    def _exporter_params(self, spec: ComponentSpec) -> dict[str, Any]:
        params = dict(spec.params)
        if isinstance(params.get("path"), str):
            params["path"] = str(self.config.resolve_path(params["path"]))
        return params

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_yaml(cls, path: str | Path, overrides: Mapping[str, Any] | None = None) -> Pipeline:
        """Build a pipeline from a ``rag.yaml`` file."""
        return cls(load_config(path, overrides))

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any], *, base_dir: str | Path | None = None, **kwargs: Any
    ) -> Pipeline:
        """Build a pipeline from a config mapping (same schema as ``rag.yaml``)."""
        config = build_config(raw, base_dir=Path(base_dir) if base_dir else None)
        return cls(config, **kwargs)

    def _build(self, kind: str, spec: ComponentSpec | None) -> Any:
        if kind in self._injected:
            component = self._injected[kind]
        else:
            assert spec is not None
            params = dict(spec.params)
            accepted = self._registry.accepted_params(kind, spec.type)
            for dep in INJECTABLE:
                if dep == kind or dep in params or accepted is None or dep not in accepted:
                    continue
                value = getattr(self, dep, None)
                if value is None:
                    raise ConfigError(f"components.{kind}: '{spec.type}' requires components.{dep}")
                params[dep] = value
            component = self._registry.create(kind, spec.type, **params)
        protocol = _PROTOCOLS.get(kind)
        if protocol is not None and not isinstance(component, protocol):
            raise ConfigError(
                f"components.{kind}: {type(component).__name__} does not implement "
                f"the {protocol.__name__} interface"
            )
        return component

    # ------------------------------------------------------------------ summary
    @property
    def name(self) -> str:
        """Application name from the config."""
        return self.config.name

    @property
    def blueprint(self) -> str | None:
        """Blueprint name from the config."""
        return self.config.blueprint

    @property
    def top_k(self) -> int:
        """Contexts passed to the generator per query."""
        return self.config.query.top_k

    @property
    def model_name(self) -> str:
        """The generator's model name (used for cost estimates)."""
        spec = self.config.components.generator
        nested = spec.params.get("kwargs")
        if "model" not in spec.params and isinstance(nested, dict) and "model" in nested:
            return str(nested["model"])
        return str(spec.params.get("model", spec.type))

    @property
    def requires_ingest(self) -> bool:
        """Whether queries need a prior ingest (``False`` for SQL over an existing DB)."""
        return bool(getattr(self.retriever, "requires_ingest", True))

    def ready(self) -> bool:
        """Whether the pipeline can answer queries now (index present or not needed).

        Multi-tenant pipelines are always ready; each tenant's index is checked per query.
        """
        if self.config.tenancy.enabled:
            return True
        return self.has_index() or not self.requires_ingest

    # ------------------------------------------------------------------ conversations
    @property
    def conversations(self) -> ConversationManager:
        """The pipeline's chat-session manager (created on first use)."""
        if self._conversations is None:
            cfg = self.config.conversation
            gen = self.generator
            real = isinstance(gen, interfaces.TextCompleter) and not getattr(gen, "is_fake", False)
            llm = gen if isinstance(gen, interfaces.TextCompleter) and real else None
            self._conversations = ConversationManager(cfg, make_store(cfg, self.index_dir), llm)
        return self._conversations

    def chat(
        self,
        message: str,
        *,
        session_id: str | None = None,
        owner: str | None = None,
        groups: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        trace: bool | None = None,
        tenant: str | None = None,
    ) -> Answer:
        """Answer a chat message, rewriting follow-ups with the session's history.

        Returns an :class:`Answer` with ``session_id`` (pass it back for the next turn)
        and ``standalone_question``. ``owner`` (e.g. an API key name) scopes sessions.
        """
        return self.conversations.chat(
            self.query,
            message,
            session_id,
            owner_key(owner, tenant),
            groups=groups,
            filters=filters,
            top_k=top_k,
            trace=trace,
            tenant=tenant,
        )

    def chat_stream(
        self,
        message: str,
        *,
        session_id: str | None = None,
        owner: str | None = None,
        groups: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        tenant: str | None = None,
    ) -> Iterator[StreamEvent]:
        """Streaming :meth:`chat`; the final ``answer`` event carries ``session_id``."""
        return self.conversations.chat_stream(
            self.stream,
            message,
            session_id,
            owner_key(owner, tenant),
            groups=groups,
            filters=filters,
            top_k=top_k,
            tenant=tenant,
        )

    # ------------------------------------------------------------------ tenancy
    def for_tenant(self, tenant: str) -> Pipeline:
        """The pipeline for ``tenant`` (isolated index; models and cache are shared).

        Tenant pipelines are created on first use and kept in an LRU of
        ``tenancy.max_loaded`` entries.

        Raises:
            TenantError: tenancy is disabled or the tenant id is invalid.
        """
        if not self.config.tenancy.enabled:
            raise TenantError(
                "A tenant was given, but this pipeline isn't multi-tenant "
                "(set tenancy.enabled: true)"
            )
        tenant = validate_tenant(tenant)
        with self._tenant_lock:
            pipeline = self._tenants.pop(tenant, None)
            if pipeline is None:
                cfg = self.config.model_copy(deep=True)
                cfg.index_dir = str(self.index_dir / "tenants" / tenant)
                cfg.tenancy = TenancyConfig(enabled=False)
                cfg.cache = cfg.cache.model_copy(update={"enabled": False})  # shared below
                cfg.with_base_dir(self.config.base_dir)
                shared = {
                    k: v
                    for k, v in self._injected.items()
                    if k not in ("store", "keyword_index", "retriever")  # per-tenant state
                }
                shared.update(embedder=self.embedder, generator=self.generator)
                pipeline = Pipeline(cfg, components=shared, component_registry=self._registry)
                pipeline.exporters = self.exporters
            self._tenants[tenant] = pipeline
            while len(self._tenants) > self.config.tenancy.max_loaded:
                self._tenants.popitem(last=False)
        return pipeline

    def _resolve(self, tenant: str | None) -> Pipeline | None:
        """The tenant pipeline to delegate to, ``None`` to handle the call here."""
        if tenant is not None:
            return self.for_tenant(tenant)
        if self.config.tenancy.enabled:
            raise TenantError("This pipeline is multi-tenant: pass a tenant id")
        return None

    # ------------------------------------------------------------------ persistence
    @property
    def index_dir(self) -> Path:
        """Absolute index directory."""
        return self.config.index_path

    def _fingerprint(self) -> dict[str, Any]:
        spec = self.config.components.embedder
        return {"embedder": {"type": spec.type, "params": spec.params}}

    def persist(self) -> None:
        """Write the vector store, keyword index and manifest to ``index_dir``."""
        root = self.index_dir
        root.mkdir(parents=True, exist_ok=True)
        self.store.persist(str(root / "store"))
        if self.keyword_index is not None:
            self.keyword_index.persist(str(root / "keyword"))
        manifest = {"format": 1, "name": self.config.name, **self._fingerprint()}
        (root / MANIFEST).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    def has_index(self) -> bool:
        """Whether an index has been persisted to ``index_dir``."""
        return (self.index_dir / MANIFEST).is_file()

    def load(self) -> None:
        """Load a persisted index from ``index_dir``.

        Raises:
            IndexNotFoundError: nothing has been ingested yet.
            ConfigError: the index was built with a different embedder.
        """
        root = self.index_dir
        if not self.has_index():
            raise IndexNotFoundError(f"No index found at {root}. Run `ragpick ingest PATH` first.")
        manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        current = json.loads(json.dumps(self._fingerprint(), default=str))
        if manifest.get("embedder") != current["embedder"]:
            raise ConfigError(
                f"The index at {root} was built with embedder {manifest.get('embedder')}, "
                f"but the config uses {current['embedder']}. Re-ingest, or point index_dir "
                "at a new directory."
            )
        self.store.load(str(root / "store"))
        if self.keyword_index is not None:
            keyword_dir = root / "keyword"
            if not keyword_dir.exists():
                raise IndexNotFoundError(
                    f"The index at {root} has no keyword index (it was built for a different "
                    "retriever). Re-ingest with this config."
                )
            self.keyword_index.load(str(keyword_dir))
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.has_index() and not self.requires_ingest:
            self._loaded = True  # e.g. SQL over an existing database: nothing to load
            return
        self.load()

    # ------------------------------------------------------------------ ingest
    def ingest(
        self,
        path: str | Path | None = None,
        *,
        batch_size: int = 64,
        progress: ProgressCallback | None = None,
        tenant: str | None = None,
    ) -> IngestStats:
        """Load, chunk, embed, index and persist the documents at ``path``.

        Re-ingesting a document replaces its previous chunks; other documents already
        in the index are kept. ``path`` defaults to the config's ``data`` setting.
        With tenancy enabled, ``tenant`` selects the tenant's isolated index.
        """
        if (delegate := self._resolve(tenant)) is not None:
            return delegate.ingest(path, batch_size=batch_size, progress=progress)
        start = time.perf_counter()
        if path is None:
            if self.config.data is None:
                raise ConfigError("No path given and no `data` set in the config")
            source = self.config.resolve_path(self.config.data)
        else:
            source = Path(path)
        if self.has_index() and not self._loaded:
            self.load()
        self._loaded = True

        docs = list(self.loader.load(str(source)))
        if not docs:
            logger.warning("No supported documents found at %s", source)
        chunks = self.chunker.split(docs)
        total = len(chunks)
        if progress:
            progress("embed", 0, total)
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            vectors = self.embedder.embed([c.text for c in batch])
            for chunk, vector in zip(batch, vectors, strict=True):
                chunk.embedding = vector
            if progress:
                progress("embed", min(i + batch_size, total), total)

        for component in (self.retriever, self.reranker, self.generator):
            if isinstance(component, interfaces.DocumentIndexer):
                component.index_documents(docs)

        for component in (self.retriever, self.reranker):
            if isinstance(component, interfaces.ChunkIndexer):
                with span("index_chunks", component=type(component).__name__):
                    component.index_chunks(chunks)

        doc_ids = [d.id for d in docs]
        self.store.delete(doc_ids)
        self.store.add(chunks)
        if self.keyword_index is not None:
            delete = getattr(self.keyword_index, "delete", None)
            if callable(delete):
                delete(doc_ids)
            self.keyword_index.add(chunks)
        self.persist()
        stats = IngestStats(
            documents=len(docs),
            chunks=total,
            seconds=time.perf_counter() - start,
            index_dir=str(self.index_dir),
        )
        logger.info("Ingested %d documents (%d chunks)", stats.documents, stats.chunks)
        return stats

    # ------------------------------------------------------------------ query
    def _retrieval_k(self, top_k: int) -> int:
        k = self.config.components.retriever.params.get("k", top_k)
        return max(int(k), top_k)

    def retrieve(
        self,
        question: str,
        *,
        groups: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        tenant: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve and rerank contexts without generating an answer.

        Access control is always enforced: chunks with ``allowed_groups`` metadata are
        returned only when ``groups`` intersects them (``groups=None`` → public only).
        """
        if (delegate := self._resolve(tenant)) is not None:
            return delegate.retrieve(question, groups=groups, filters=filters, top_k=top_k)
        self._ensure_loaded()
        top_k = top_k or self.config.query.top_k
        combined = acl_filter(groups, filters)
        k = self._retrieval_k(top_k)
        with span("retrieve", retriever=self.config.components.retriever.type, k=k):
            candidates = self.retriever.retrieve(question, k, combined)
            set_attributes(results=len(candidates))
        with span("rerank", reranker=self.config.components.reranker.type, top_n=top_k):
            reranked = self.reranker.rerank(question, candidates, top_k)
            set_attributes(results=len(reranked))
        return reranked

    def query(
        self,
        question: str,
        *,
        groups: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        trace: bool | None = None,
        tenant: str | None = None,
    ) -> Answer:
        """Answer ``question`` from the index (the ``tenant``'s index when multi-tenant).

        ``answer.usage.latency_ms`` covers the whole query (retrieval + generation).
        With ``trace=True`` (or ``tracing.enabled`` in the config) ``answer.trace``
        holds the timed spans of every stage, and configured exporters receive it.
        """
        if (delegate := self._resolve(tenant)) is not None:
            return delegate.query(
                question, groups=groups, filters=filters, top_k=top_k, trace=trace
            )
        if tracing.tracing_active():  # e.g. called by a router: join its trace
            with span("pipeline", app=self.config.name, blueprint=self.config.blueprint):
                return self._query(question, groups, filters, top_k)
        if not (self.config.tracing.enabled if trace is None else trace):
            return self._query(question, groups, filters, top_k)
        with tracing.trace(
            "query", groups=len(groups or []), blueprint=self.config.blueprint
        ) as root:
            answer = self._query(question, groups, filters, top_k)
        answer.trace = root
        for exporter in self.exporters:
            try:
                exporter.export(root)
            except Exception as exc:  # exporters must never break queries
                logger.warning("Trace exporter %s failed: %s", type(exporter).__name__, exc)
        return answer

    def stream(
        self,
        question: str,
        *,
        groups: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        tenant: str | None = None,
    ) -> Iterator[StreamEvent]:
        """Answer ``question`` as a stream of events: contexts, text deltas, final answer.

        Generators without ``stream()`` produce a single delta. Streams are not traced.
        """
        if (delegate := self._resolve(tenant)) is not None:
            yield from delegate.stream(question, groups=groups, filters=filters, top_k=top_k)
            return
        start = time.perf_counter()
        contexts = self.retrieve(question, groups=groups, filters=filters, top_k=top_k)
        yield StreamEvent(type="contexts", contexts=contexts)
        parts: list[str] = []
        usage = Usage()
        if isinstance(self.generator, interfaces.StreamingGenerator):
            for item in self.generator.stream(question, contexts):
                if isinstance(item, Usage):
                    usage = item
                elif item:
                    parts.append(item)
                    yield StreamEvent(type="delta", text=item)
        else:
            answer = self.generator.generate(question, contexts)
            parts.append(answer.text)
            usage = answer.usage
            yield StreamEvent(type="delta", text=answer.text)
        text = "".join(parts).strip()
        usage = usage.model_copy(update={"latency_ms": (time.perf_counter() - start) * 1000})
        final = Answer(
            text=text, citations=parse_citations(text, contexts), contexts=contexts, usage=usage
        )
        yield StreamEvent(type="answer", answer=final)

    def _query(
        self,
        question: str,
        groups: list[str] | None,
        filters: dict[str, Any] | None,
        top_k: int | None,
    ) -> Answer:
        start = time.perf_counter()
        contexts = self.retrieve(question, groups=groups, filters=filters, top_k=top_k)
        with span("generate", generator=self.config.components.generator.type) as s:
            answer = self.generator.generate(question, contexts)
            if s is not None:
                s.attributes.update(
                    prompt_tokens=answer.usage.prompt_tokens,
                    completion_tokens=answer.usage.completion_tokens,
                    cached=answer.usage.cached,
                    citations=len(answer.citations),
                )
        answer.usage.latency_ms = (time.perf_counter() - start) * 1000
        return answer
