"""Retrievers, including hybrid retrieval with Reciprocal Rank Fusion.

Retrievers receive their dependencies (``embedder``, ``store``, ``keyword_index``)
from the pipeline by constructor parameter name; see :class:`ragpick.Pipeline`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ragpick.components.chunkers import PARENT_ID_KEY, PARENT_TEXT_KEY
from ragpick.interfaces import Embedder, KeywordIndex, Retriever, VectorStore
from ragpick.models import Chunk, RetrievedChunk
from ragpick.registry import register
from ragpick.tracing import set_attributes, span


def embed_query(embedder: Embedder, query: str) -> list[float]:
    """Embed a search query, using ``embedder.embed_query`` when the embedder has one.

    Asymmetric models (E5, Cohere, LangChain/LlamaIndex embeddings) embed queries
    differently from documents; ``embed_query`` is an optional extension of the
    :class:`~ragpick.interfaces.Embedder` protocol.
    """
    method = getattr(embedder, "embed_query", None)
    if callable(method):
        return [float(x) for x in method(query)]
    return embedder.embed([query])[0]


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[RetrievedChunk]],
    rrf_k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked lists with Reciprocal Rank Fusion (Cormack et al., 2009).

    ``score(chunk) = Σ_i weight_i / (rrf_k + rank_i(chunk))`` with 1-based ranks.
    Chunks are identified by id. Ties are broken by best individual rank, then id.
    """
    if rrf_k < 0:
        raise ValueError("rrf_k must be >= 0")
    weights = list(weights) if weights is not None else [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("weights must have one entry per result list")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    chunks: dict[str, RetrievedChunk] = {}
    for weight, results in zip(weights, result_lists, strict=True):
        seen: set[str] = set()
        for rank, item in enumerate(results, start=1):
            cid = item.chunk.id
            if cid in seen:  # ignore duplicates within one list
                continue
            seen.add(cid)
            scores[cid] = scores.get(cid, 0.0) + weight / (rrf_k + rank)
            best_rank[cid] = min(best_rank.get(cid, rank), rank)
            chunks.setdefault(cid, item)
    order = sorted(scores, key=lambda cid: (-scores[cid], best_rank[cid], cid))
    return [
        RetrievedChunk(chunk=chunks[cid].chunk, score=scores[cid], source="fused") for cid in order
    ]


@register("retriever", "vector")
class VectorRetriever:
    """Dense retrieval: embed the query and search the vector store."""

    def __init__(self, embedder: Embedder, store: VectorStore, k: int = 10) -> None:
        self.embedder = embedder
        self.store = store
        self.k = k

    def retrieve(
        self, query: str, k: int | None = None, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Top-``k`` chunks by vector similarity."""
        with span("vector_search", k=k or self.k):
            results = self.store.search(embed_query(self.embedder, query), k or self.k, filters)
            set_attributes(results=len(results))
            return results


@register("retriever", "keyword")
class KeywordRetriever:
    """Lexical retrieval with the keyword index only (e.g. BM25)."""

    def __init__(self, keyword_index: KeywordIndex, k: int = 10) -> None:
        self.keyword_index = keyword_index
        self.k = k

    def retrieve(
        self, query: str, k: int | None = None, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Top-``k`` chunks by keyword score."""
        with span("keyword_search", k=k or self.k):
            results = self.keyword_index.search(query, k or self.k, filters)
            set_attributes(results=len(results))
            return results


@register("retriever", "hybrid")
class HybridRetriever:
    """Vector + keyword retrieval fused with Reciprocal Rank Fusion.

    Each sub-retriever fetches ``candidates`` results (default: ``k``); the fused
    list is truncated to ``k``. Filters are applied inside both searches, so access
    control holds for the fused result.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        keyword_index: KeywordIndex,
        k: int = 20,
        rrf_k: int = 60,
        candidates: int | None = None,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
    ) -> None:
        self.vector = VectorRetriever(embedder, store, k)
        self.keyword = KeywordRetriever(keyword_index, k)
        self.k = k
        self.rrf_k = rrf_k
        self.candidates = candidates
        self.weights = (vector_weight, keyword_weight)

    def retrieve(
        self, query: str, k: int | None = None, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Fused top-``k`` chunks."""
        k = k or self.k
        n = max(self.candidates or k, k)
        dense = self.vector.retrieve(query, n, filters)
        sparse = self.keyword.retrieve(query, n, filters)
        return reciprocal_rank_fusion([dense, sparse], self.rrf_k, self.weights)[:k]


def expand_to_parents(results: Sequence[RetrievedChunk], k: int) -> list[RetrievedChunk]:
    """Replace child chunks with their parent sections, deduplicated, best first.

    A parent's score is its best child's score. Results without parent metadata are
    passed through unchanged.
    """
    parents: list[RetrievedChunk] = []
    seen: set[str] = set()
    for item in results:
        meta = item.chunk.metadata
        parent_id = meta.get(PARENT_ID_KEY)
        if parent_id is None:
            if item.chunk.id not in seen:
                seen.add(item.chunk.id)
                parents.append(item)
        elif parent_id not in seen:
            seen.add(parent_id)
            parent_meta = {
                key: value
                for key, value in meta.items()
                if key not in (PARENT_TEXT_KEY, "chunk_index")
            }
            parent_meta["matched_child"] = item.chunk.id
            parent = Chunk(
                id=str(parent_id),
                doc_id=item.chunk.doc_id,
                text=str(meta.get(PARENT_TEXT_KEY, item.chunk.text)),
                metadata=parent_meta,
            )
            parents.append(RetrievedChunk(chunk=parent, score=item.score, source=item.source))
        if len(parents) >= k:
            break
    return parents


@register("retriever", "parent_child")
class ParentChildRetriever:
    """Search small child chunks, return their larger parent sections.

    ``base`` selects the child search strategy: ``"hybrid"`` (default, BM25 + vector
    with RRF) or ``"vector"``. ``child_k`` children are fetched (default ``4 * k``) so
    that ``k`` distinct parents are usually found. Use with the ``parent_child``
    chunker. Filters (and access control) apply to the child search.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        keyword_index: KeywordIndex,
        k: int = 5,
        base: str = "hybrid",
        child_k: int | None = None,
        rrf_k: int = 60,
    ) -> None:
        if base not in ("hybrid", "vector"):
            raise ValueError(f"base must be 'hybrid' or 'vector', got {base!r}")
        self.k = k
        self.child_k = child_k
        self.base: Retriever = (
            HybridRetriever(embedder, store, keyword_index, k=k, rrf_k=rrf_k)
            if base == "hybrid"
            else VectorRetriever(embedder, store, k=k)
        )

    def retrieve(
        self, query: str, k: int | None = None, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Top-``k`` parent sections."""
        k = k or self.k
        children = self.base.retrieve(query, self.child_k or 4 * k, filters)
        return expand_to_parents(children, k)
