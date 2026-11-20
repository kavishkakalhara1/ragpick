"""Retrievers, including hybrid retrieval with Reciprocal Rank Fusion.

Retrievers receive their dependencies (``embedder``, ``store``, ``keyword_index``)
from the pipeline by constructor parameter name; see :class:`ragpick.Pipeline`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ragpick.interfaces import Embedder, KeywordIndex, VectorStore
from ragpick.models import RetrievedChunk
from ragpick.registry import register


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
        vector = self.embedder.embed([query])[0]
        return self.store.search(vector, k or self.k, filters)


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
        return self.keyword_index.search(query, k or self.k, filters)


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
