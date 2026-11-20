"""Hierarchical (RAPTOR-style) retrieval: a tree of summaries over your chunks.

**Ingest.** Chunks are clustered by embedding similarity (``cluster_size`` per group),
each cluster is summarised (LLM, or extractively offline), the summaries are embedded
and clustered again, up to ``max_levels``. Summary nodes are added to the vector store
and keyword index next to the chunks.

**Query ("collapsed tree").** Hybrid search runs over chunks *and* summaries together,
so a detail question finds a chunk, and a broad question ("what does the warranty
cover overall?") finds a summary that spans several chunks.

**Access control.** Trees are built separately per access scope (public, and each
distinct ``allowed_groups`` set), and summary nodes inherit that scope's
``allowed_groups``, so a summary never mixes restricted and public text.

The leaves are kept in ``<index_dir>/hierarchy/tree.json``; re-ingest rebuilds the tree
and reuses summaries whose members didn't change.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ragpick.components.generators import split_sentences
from ragpick.components.retrievers import HybridRetriever
from ragpick.errors import ConfigError
from ragpick.filters import ACL_KEY
from ragpick.interfaces import Embedder, Generator, KeywordIndex, TextCompleter, VectorStore
from ragpick.models import Chunk, RetrievedChunk
from ragpick.registry import register
from ragpick.tracing import set_attributes, span

logger = logging.getLogger(__name__)

TREE_FILE = "tree.json"

SUMMARY_SYSTEM = (
    "You summarise related passages for a search index. Write 3-6 sentences that keep "
    "the key facts, names, codes and numbers. Use only the passages."
)

SUMMARY_PROMPT = """Passages:
{passages}

Summary:"""

Scope = tuple[str, ...] | None  # None = public; a tuple = allowed_groups (sorted)


def _scope(chunk: Chunk) -> Scope:
    groups = chunk.metadata.get(ACL_KEY)
    if groups is None:
        return None
    values = groups if isinstance(groups, list | tuple) else [groups]
    return tuple(sorted(str(g) for g in values))


def cluster(vectors: np.ndarray[Any, Any], size: int) -> list[list[int]]:
    """Greedy, deterministic clustering: each seed takes its ``size - 1`` nearest neighbours."""
    n = len(vectors)
    if n == 0:
        return []
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms
    sims = unit @ unit.T
    unassigned = list(range(n))
    groups: list[list[int]] = []
    while unassigned:
        seed = unassigned[0]
        rest = unassigned[1:]
        ranked = sorted(rest, key=lambda j: (-sims[seed, j], j))[: size - 1]
        group = [seed, *ranked]
        groups.append(sorted(group))
        taken = set(group)
        unassigned = [i for i in unassigned if i not in taken]
    return groups


def extractive_summary(texts: list[str], max_sentences: int = 6) -> str:
    """Offline summary: the first informative sentence of each passage."""
    picked: list[str] = []
    for text in texts:
        for sentence in split_sentences(text):
            clean = sentence.replace("**", "").strip().lstrip("-*0123456789. ").strip()
            if len(clean.split()) >= 5:
                picked.append(clean)
                break
        if len(picked) >= max_sentences:
            break
    return " ".join(picked)


@register("retriever", "hierarchical")
class HierarchicalRetriever:
    """Collapsed-tree retrieval over chunks and their summary tree; see the module docstring.

    Args:
        cluster_size: members per summary.
        max_levels: maximum tree depth above the chunks.
        summarizer: ``auto`` (LLM when the generator is a real LLM), ``llm`` or
            ``extractive``.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        keyword_index: KeywordIndex,
        generator: Generator,
        index_dir: Path,
        k: int = 10,
        cluster_size: int = 6,
        max_levels: int = 3,
        summarizer: str = "auto",
        rrf_k: int = 60,
    ) -> None:
        if summarizer not in ("auto", "llm", "extractive"):
            raise ConfigError(f"summarizer must be auto, llm or extractive, got {summarizer!r}")
        if cluster_size < 2:
            raise ConfigError("cluster_size must be >= 2")
        if max_levels < 1:
            raise ConfigError("max_levels must be >= 1")
        real = isinstance(generator, TextCompleter) and not getattr(generator, "is_fake", False)
        if summarizer == "llm" and not isinstance(generator, TextCompleter):
            raise ConfigError("summarizer: llm needs a generator that supports complete()")
        self.llm: TextCompleter | None = None
        if isinstance(generator, TextCompleter) and (
            summarizer == "llm" or (summarizer == "auto" and real)
        ):
            self.llm = generator
        self.embedder = embedder
        self.store = store
        self.keyword_index = keyword_index
        self.base = HybridRetriever(embedder, store, keyword_index, k=k, rrf_k=rrf_k)
        self.path = Path(index_dir) / "hierarchy"
        self.k = k
        self.cluster_size = cluster_size
        self.max_levels = max_levels
        self.leaves: dict[str, Chunk] = {}
        self.summaries: dict[str, str] = {}  # signature -> summary text (reuse cache)
        self.node_doc_ids: list[str] = []
        self._loaded = False

    # -------------------------------------------------------------- persistence
    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        file = self.path / TREE_FILE
        if file.is_file():
            data = json.loads(file.read_text(encoding="utf-8"))
            self.leaves = {c["id"]: Chunk.model_validate(c) for c in data["leaves"]}
            self.summaries = dict(data.get("summaries", {}))
            self.node_doc_ids = list(data.get("nodes", []))

    def _persist(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        data = {
            "format": 1,
            "leaves": [c.model_dump() for c in self.leaves.values()],
            "summaries": self.summaries,
            "nodes": self.node_doc_ids,
        }
        (self.path / TREE_FILE).write_text(json.dumps(data), encoding="utf-8")

    # -------------------------------------------------------------- building
    def summarize(self, texts: list[str]) -> str:
        """Summary of a cluster's member texts."""
        if self.llm is not None:
            passages = "\n\n".join(f"- {t[:800]}" for t in texts)
            text = self.llm.complete(SUMMARY_SYSTEM, SUMMARY_PROMPT.format(passages=passages)).text
            if text.strip():
                return text.strip()
        return extractive_summary(texts) or texts[0][:400]

    def _build_scope(self, scope: Scope, items: list[Chunk]) -> list[Chunk]:
        """Summary nodes for one access scope (all levels)."""
        nodes: list[Chunk] = []
        level_items = sorted(items, key=lambda c: c.id)
        for level in range(1, self.max_levels + 1):
            if len(level_items) <= 1:
                break
            vectors = np.asarray([c.embedding for c in level_items], dtype=np.float64)
            groups = [g for g in cluster(vectors, self.cluster_size) if len(g) >= 2]
            if not groups:
                break
            new_nodes: list[Chunk] = []
            pending: list[tuple[str, list[Chunk]]] = []
            for group in groups:
                members = [level_items[i] for i in group]
                signature = hashlib.sha256(
                    json.dumps([level, [(m.id, m.text) for m in members]]).encode("utf-8")
                ).hexdigest()[:16]
                if signature not in self.summaries:
                    self.summaries[signature] = self.summarize([m.text for m in members])
                pending.append((signature, members))
            texts = [self.summaries[sig] for sig, _ in pending]
            vectors_out = self.embedder.embed(texts) if texts else []
            for (signature, members), text, vector in zip(pending, texts, vectors_out, strict=True):
                doc_id = f"summary-{level}-{signature}"
                metadata: dict[str, Any] = {
                    "synthetic": True,
                    "level": level,
                    "children": [m.id for m in members],
                    "source_docs": sorted(
                        {d for m in members for d in m.metadata.get("source_docs", [m.doc_id])}
                    ),
                }
                if scope is not None:
                    metadata[ACL_KEY] = list(scope)
                new_nodes.append(
                    Chunk(
                        id=f"{doc_id}#0",
                        doc_id=doc_id,
                        text=text,
                        metadata=metadata,
                        embedding=vector,
                    )
                )
            nodes.extend(new_nodes)
            # clusters of one carry their item up unchanged
            grouped = {i for g in groups for i in g}
            carried = [c for i, c in enumerate(level_items) if i not in grouped]
            level_items = new_nodes + carried
        return nodes

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Update the leaves with ``chunks`` and rebuild the summary tree."""
        self._load()
        changed = {c.doc_id for c in chunks}
        self.leaves = {cid: c for cid, c in self.leaves.items() if c.doc_id not in changed}
        for c in chunks:
            if c.embedding is not None:
                self.leaves[c.id] = c
        if self.node_doc_ids:
            self.store.delete(self.node_doc_ids)
            delete = getattr(self.keyword_index, "delete", None)
            if callable(delete):
                delete(self.node_doc_ids)
        scopes: dict[Scope, list[Chunk]] = {}
        for leaf in self.leaves.values():
            scopes.setdefault(_scope(leaf), []).append(leaf)
        with span("build_tree", leaves=len(self.leaves), scopes=len(scopes)):
            cached = set(self.summaries)
            nodes = [
                node
                for scope, items in sorted(scopes.items(), key=lambda kv: str(kv[0]))
                for node in self._build_scope(scope, items)
            ]
            used = {n.doc_id.split("-", 2)[2] for n in nodes}
            self.summaries = {sig: text for sig, text in self.summaries.items() if sig in used}
            set_attributes(nodes=len(nodes), reused=len(used & cached), new=len(used - cached))
        if nodes:
            self.store.add(nodes)
            self.keyword_index.add(nodes)
        self.node_doc_ids = [n.doc_id for n in nodes]
        self._persist()

    # -------------------------------------------------------------- query
    def retrieve(
        self, query: str, k: int | None = None, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Hybrid search over chunks and summaries together (collapsed tree)."""
        return self.base.retrieve(query, k or self.k, filters)
