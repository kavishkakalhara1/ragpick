"""The :class:`Pipeline`: builds components from config and runs ingest / query."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ragpick import interfaces
from ragpick.config import ComponentSpec, RagConfig, build_config, load_config
from ragpick.errors import ConfigError, IndexNotFoundError
from ragpick.filters import acl_filter
from ragpick.models import Answer, RetrievedChunk
from ragpick.registry import Registry, registry

logger = logging.getLogger(__name__)

#: Dependencies the pipeline injects into retrievers (and any other component)
#: whose constructor declares a parameter with one of these names.
INJECTABLE = ("embedder", "store", "keyword_index")

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
    Retrievers receive ``embedder``, ``store`` and ``keyword_index`` by constructor
    parameter name. Pass ``components={"generator": my_generator}`` to inject
    ready-made instances instead.
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
        c = config.components
        self.loader: interfaces.Loader = self._build("loader", c.loader)
        self.chunker: interfaces.Chunker = self._build("chunker", c.chunker)
        self.embedder: interfaces.Embedder = self._build("embedder", c.embedder)
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
        self.generator: interfaces.Generator = self._build("generator", c.generator)
        if c.generator.type == "fake" and "generator" not in self._injected:
            logger.info("Using the fake generator: answers are extractive placeholders")

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
        if not self._loaded:
            self.load()

    # ------------------------------------------------------------------ ingest
    def ingest(
        self,
        path: str | Path | None = None,
        *,
        batch_size: int = 64,
        progress: ProgressCallback | None = None,
    ) -> IngestStats:
        """Load, chunk, embed, index and persist the documents at ``path``.

        Re-ingesting a document replaces its previous chunks; other documents already
        in the index are kept. ``path`` defaults to the config's ``data`` setting.
        """
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
    ) -> list[RetrievedChunk]:
        """Retrieve and rerank contexts without generating an answer.

        Access control is always enforced: chunks with ``allowed_groups`` metadata are
        returned only when ``groups`` intersects them (``groups=None`` → public only).
        """
        self._ensure_loaded()
        top_k = top_k or self.config.query.top_k
        combined = acl_filter(groups, filters)
        candidates = self.retriever.retrieve(question, self._retrieval_k(top_k), combined)
        return self.reranker.rerank(question, candidates, top_k)

    def query(
        self,
        question: str,
        *,
        groups: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> Answer:
        """Answer ``question`` from the index.

        ``answer.usage.latency_ms`` covers the whole query (retrieval + generation).
        """
        start = time.perf_counter()
        contexts = self.retrieve(question, groups=groups, filters=filters, top_k=top_k)
        answer = self.generator.generate(question, contexts)
        answer.usage.latency_ms = (time.perf_counter() - start) * 1000
        return answer
