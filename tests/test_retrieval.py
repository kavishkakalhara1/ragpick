"""Retrievers, RRF fusion and rerankers."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from conftest import make_chunks
from ragpick.components.embedders import FakeEmbedder
from ragpick.components.keyword import BM25Index
from ragpick.components.rerankers import CrossEncoderReranker, NoopReranker
from ragpick.components.retrievers import (
    HybridRetriever,
    KeywordRetriever,
    VectorRetriever,
    reciprocal_rank_fusion,
)
from ragpick.components.stores import InMemoryVectorStore
from ragpick.errors import MissingDependencyError
from ragpick.filters import acl_filter
from ragpick.interfaces import Reranker, Retriever
from ragpick.models import Chunk, RetrievedChunk


def rc(cid: str, score: float = 0.0, source: str = "vector") -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(id=cid, doc_id=cid, text=cid), score=score, source=source)  # type: ignore[arg-type]


def ids(results: list[RetrievedChunk]) -> list[str]:
    return [r.chunk.id for r in results]


# ------------------------------------------------------------------ RRF
def test_rrf_scores_match_formula() -> None:
    fused = reciprocal_rank_fusion([[rc("a"), rc("b"), rc("c")], [rc("b"), rc("d")]], rrf_k=60)
    scores = {r.chunk.id: r.score for r in fused}
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63)
    assert scores["d"] == pytest.approx(1 / 62)
    assert ids(fused) == ["b", "a", "d", "c"]
    assert all(r.source == "fused" for r in fused)


def test_rrf_tie_breaking_weights_and_duplicates() -> None:
    # a and x tie on score; both rank 1 in their lists -> tie broken by id
    fused = reciprocal_rank_fusion([[rc("x")], [rc("a")]])
    assert ids(fused) == ["a", "x"]
    weighted = reciprocal_rank_fusion([[rc("x")], [rc("a")]], weights=[2.0, 1.0])
    assert ids(weighted) == ["x", "a"]
    dup = reciprocal_rank_fusion([[rc("a"), rc("a"), rc("b")]], rrf_k=0)
    assert {r.chunk.id: r.score for r in dup} == {"a": 1.0, "b": pytest.approx(1 / 3)}
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_validation() -> None:
    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fusion([[rc("a")]], rrf_k=-1)
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([[rc("a")]], weights=[1.0, 2.0])


# ------------------------------------------------------------------ retrievers
CORPUS = {
    "pw#0": "To reset your password tap forgot password on the sign in screen",
    "err#0": "Error E-4471 indicates a faulty battery sensor",
    "bill#0": "Invoices are emailed after every payment",
    "wifi#0": "The thermostat only supports 2.4 GHz wifi networks",
}


@pytest.fixture
def indexes() -> tuple[FakeEmbedder, InMemoryVectorStore, BM25Index]:
    emb = FakeEmbedder(dimension=64)
    chunks = make_chunks(CORPUS, {"bill#0": {"allowed_groups": ["finance"]}})
    store = InMemoryVectorStore()
    store.add(chunks)
    kw = BM25Index()
    kw.add(chunks)
    return emb, store, kw


def test_vector_and_keyword_retrievers(indexes: tuple[Any, Any, Any]) -> None:
    emb, store, kw = indexes
    vec = VectorRetriever(emb, store, k=2)
    assert isinstance(vec, Retriever)
    assert ids(vec.retrieve("reset my password"))[0] == "pw#0"
    assert len(vec.retrieve("reset my password")) == 2
    key = KeywordRetriever(kw, k=3)
    assert ids(key.retrieve("E-4471", 1)) == ["err#0"]


def test_hybrid_retriever_fuses_both(indexes: tuple[Any, Any, Any]) -> None:
    emb, store, kw = indexes
    hybrid = HybridRetriever(emb, store, kw, k=3, rrf_k=60, candidates=4)
    assert isinstance(hybrid, Retriever)
    results = hybrid.retrieve("what does E-4471 mean")
    assert ids(results)[0] == "err#0"
    assert len(results) == 3
    assert all(r.source == "fused" for r in results)
    assert len(hybrid.retrieve("password", 1)) == 1


def test_hybrid_retriever_enforces_acl(indexes: tuple[Any, Any, Any]) -> None:
    emb, store, kw = indexes
    hybrid = HybridRetriever(emb, store, kw, k=10)
    public = ids(hybrid.retrieve("invoices payment", filters=acl_filter([])))
    assert "bill#0" not in public
    finance = ids(hybrid.retrieve("invoices payment", filters=acl_filter(["finance"])))
    assert finance[0] == "bill#0"


# ------------------------------------------------------------------ rerankers
def test_noop_reranker_truncates() -> None:
    reranker = NoopReranker()
    assert isinstance(reranker, Reranker)
    assert ids(reranker.rerank("q", [rc("a"), rc("b"), rc("c")], 2)) == ["a", "b"]


def test_cross_encoder_reranker_with_mock_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class Model:
        def __init__(self, name: str, device: str | None = None) -> None:
            self.name = name

        def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[float]:
            return [float(len(text)) for _, text in pairs]

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=Model))
    reranker = CrossEncoderReranker()
    chunks = [rc("bb", source="keyword"), rc("a"), rc("cccc")]
    out = reranker.rerank("q", chunks, 2)
    assert ids(out) == ["cccc", "bb"]
    assert out[0].score == 4.0
    assert out[1].source == "keyword"
    assert reranker.rerank("q", [], 3) == []


def test_cross_encoder_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(MissingDependencyError, match=r"ragpick\[rerank\]"):
        CrossEncoderReranker()
