"""parent_child blueprint: chunker, parent expansion, retriever, end-to-end with ACL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from conftest import EXAMPLES
from ragpick import Pipeline
from ragpick.components.chunkers import PARENT_ID_KEY, PARENT_TEXT_KEY, ParentChildChunker
from ragpick.components.embedders import FakeEmbedder
from ragpick.components.keyword import BM25Index
from ragpick.components.retrievers import (
    ParentChildRetriever,
    VectorRetriever,
    embed_query,
    expand_to_parents,
)
from ragpick.components.stores import InMemoryVectorStore
from ragpick.filters import acl_filter
from ragpick.models import Chunk, Document, RetrievedChunk

DATA = EXAMPLES / "data"

LONG = "\n\n".join(f"Section {i}. " + " ".join(f"word{i}_{j}" for j in range(60)) for i in range(6))


def test_chunker_produces_children_with_parent_metadata() -> None:
    doc = Document(id="manual.md", text=LONG, metadata={"allowed_groups": ["ops"]})
    children = ParentChildChunker(parent_size=600, child_size=150, child_overlap=20).split([doc])
    assert children
    assert all(len(c.text) <= 150 for c in children)
    parent_ids = {c.metadata[PARENT_ID_KEY] for c in children}
    assert len(parent_ids) >= 2
    for c in children:
        parent_id = c.metadata[PARENT_ID_KEY]
        assert c.id.startswith(parent_id + ".")
        assert c.text in c.metadata[PARENT_TEXT_KEY]
        assert len(c.metadata[PARENT_TEXT_KEY]) <= 600
        assert c.metadata["allowed_groups"] == ["ops"]
        assert c.doc_id == "manual.md"
    assert [c.metadata["chunk_index"] for c in children] == list(range(len(children)))


def test_chunker_validation_and_empty_docs() -> None:
    with pytest.raises(ValueError, match="child_size"):
        ParentChildChunker(parent_size=400, child_size=400)
    chunker = ParentChildChunker()
    assert chunker.split([Document(id="e", text="  \n ")]) == []


def rc(cid: str, parent: str | None, score: float, text: str = "child") -> RetrievedChunk:
    meta: dict[str, Any] = {"lang": "en", "chunk_index": 0}
    if parent is not None:
        meta.update({PARENT_ID_KEY: parent, PARENT_TEXT_KEY: f"PARENT {parent}"})
    chunk = Chunk(id=cid, doc_id=cid.split("#")[0], text=text, metadata=meta)
    return RetrievedChunk(chunk=chunk, score=score, source="fused")


def test_expand_to_parents_dedupes_and_keeps_best_score() -> None:
    results = [
        rc("a#0.1", "a#0", 0.9),
        rc("a#0.0", "a#0", 0.8),  # same parent: dropped
        rc("b#1.0", "b#1", 0.7),
        rc("plain#3", None, 0.6),  # no parent metadata: passed through
        rc("plain#3", None, 0.5),  # duplicate id: dropped
        rc("c#0.0", "c#0", 0.4),
    ]
    parents = expand_to_parents(results, k=3)
    assert [p.chunk.id for p in parents] == ["a#0", "b#1", "plain#3"]
    first = parents[0]
    assert first.chunk.text == "PARENT a#0"
    assert first.score == 0.9
    assert first.source == "fused"
    assert first.chunk.metadata["matched_child"] == "a#0.1"
    assert PARENT_TEXT_KEY not in first.chunk.metadata
    assert "chunk_index" not in first.chunk.metadata
    assert first.chunk.metadata["lang"] == "en"
    assert len(expand_to_parents(results, k=10)) == 4


def _indexes(chunker: ParentChildChunker) -> tuple[FakeEmbedder, InMemoryVectorStore, BM25Index]:
    emb = FakeEmbedder(dimension=64)
    docs = [
        Document(id="public.md", text=LONG),
        Document(
            id="secret.md",
            text=LONG.replace("Section", "Secret"),
            metadata={"allowed_groups": ["hr"]},
        ),
    ]
    chunks = chunker.split(docs)
    for c, v in zip(chunks, emb.embed([c.text for c in chunks]), strict=True):
        c.embedding = v
    store, kw = InMemoryVectorStore(), BM25Index()
    store.add(chunks)
    kw.add(chunks)
    return emb, store, kw


@pytest.mark.parametrize("base", ["hybrid", "vector"])
def test_retriever_returns_parents_and_respects_acl(base: str) -> None:
    emb, store, kw = _indexes(ParentChildChunker(parent_size=600, child_size=150, child_overlap=20))
    retriever = ParentChildRetriever(emb, store, kw, k=2, base=base, child_k=20)
    results = retriever.retrieve("word2_10 word2_11", filters=acl_filter([]))
    assert 1 <= len(results) <= 2
    assert all(r.chunk.doc_id == "public.md" for r in results)
    assert all("#" in r.chunk.id and "." not in r.chunk.id.split("#")[1] for r in results)
    assert "word2_10" in results[0].chunk.text
    hr = retriever.retrieve("Secret 2 word2_10", 4, acl_filter(["hr"]))
    assert any(r.chunk.doc_id == "secret.md" for r in hr)
    with pytest.raises(ValueError, match="base must be"):
        ParentChildRetriever(emb, store, kw, base="graph")


def test_parent_child_end_to_end(tmp_path: Path) -> None:
    pipeline = Pipeline.from_dict(
        {
            "blueprint": "parent_child",
            "index_dir": str(tmp_path / "idx"),
            "components": {
                "chunker": {"params": {"parent_size": 600, "child_size": 150, "child_overlap": 20}}
            },
        }
    )
    assert isinstance(pipeline.retriever, ParentChildRetriever)
    stats = pipeline.ingest(DATA)
    assert stats.chunks > stats.documents  # children
    answer = pipeline.query("What does error E-5120 mean?")
    assert answer.citations
    assert all("." not in cid.split("#")[1] for cid in answer.citations)  # parent ids
    assert any("E-5120" in c.chunk.text for c in answer.contexts)
    question = "When should support escalate an E-3301 ticket to Tier 2?"
    public = pipeline.retrieve(question, top_k=9)
    assert all(c.chunk.doc_id != "internal-escalation.md" for c in public)
    support = pipeline.retrieve(question, groups=["support"], top_k=9)
    assert any(c.chunk.doc_id == "internal-escalation.md" for c in support)
    # persistence: parent text survives a reload
    reloaded = Pipeline.from_dict(pipeline.config.model_dump(mode="json"))
    reloaded.config.with_base_dir(pipeline.config.base_dir)
    assert reloaded.retrieve("E-5120", top_k=1)[0].chunk.text


def test_embed_query_prefers_query_method() -> None:
    class Asymmetric(FakeEmbedder):
        def embed_query(self, query: str) -> list[float]:
            return [9.0] * self.dimension

    emb = Asymmetric(dimension=4)
    assert embed_query(emb, "q") == [9.0] * 4
    assert embed_query(FakeEmbedder(dimension=4), "hello") == FakeEmbedder(dimension=4).embed_one(
        "hello"
    )

    store = InMemoryVectorStore()
    seen: list[list[float]] = []
    store.search = lambda vector, k, filters=None: seen.append(vector) or []  # type: ignore[method-assign]
    VectorRetriever(emb, store).retrieve("q")
    assert seen == [[9.0] * 4]
