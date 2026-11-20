"""Corrective retriever: grading, rewriting, refusal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from conftest import EXAMPLES, make_chunks
from ragpick import Pipeline
from ragpick.components.corrective import (
    CorrectiveRetriever,
    build_base_retriever,
    lexical_relevance,
    parse_relevant,
)
from ragpick.components.embedders import FakeEmbedder
from ragpick.components.generators import FakeGenerator, LLMGenerator
from ragpick.components.keyword import BM25Index
from ragpick.components.retrievers import HybridRetriever, ParentChildRetriever, VectorRetriever
from ragpick.components.stores import InMemoryVectorStore
from ragpick.errors import ConfigError
from ragpick.filters import acl_filter
from ragpick.models import Answer, Completion, RetrievedChunk
from ragpick.tracing import trace

CORPUS = {
    "pw#0": "To reset your password tap forgot password on the sign in screen",
    "err#0": "Error E-4471 indicates a faulty battery sensor",
    "bill#0": "Invoices are emailed after every payment",
    "wifi#0": "The thermostat supports 2.4 GHz wifi networks only",
    "hr#0": "Password policy for employees requires rotation",
}


def indexes() -> tuple[FakeEmbedder, InMemoryVectorStore, BM25Index]:
    chunks = make_chunks(CORPUS, {"hr#0": {"allowed_groups": ["hr"]}})
    store, kw = InMemoryVectorStore(), BM25Index()
    store.add(chunks)
    kw.add(chunks)
    return FakeEmbedder(dimension=64), store, kw


class ScriptedLLM(LLMGenerator):
    """A real-LLM stand-in: replies in order, records prompts."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        self.prompts.append((system, user))
        return Completion(text=self.replies.pop(0) if self.replies else "")


def make(gen: Any, **kw: Any) -> CorrectiveRetriever:
    emb, store, bm25 = indexes()
    return CorrectiveRetriever(emb, store, bm25, gen, **kw)


def test_lexical_relevance_and_parse_relevant() -> None:
    assert lexical_relevance("reset password", "how to reset your password") == 1.0
    assert lexical_relevance("reset password", "reset the router") == 0.5
    assert lexical_relevance("the of", "anything") == 0.0
    assert parse_relevant('{"relevant": [2, 1, 2, 9]}', 3) == [0, 1]
    assert parse_relevant('Sure: {"relevant": []}', 3) == []
    assert parse_relevant("nope", 3) is None
    assert parse_relevant('{"relevant": "all"}', 3) is None
    assert parse_relevant('{"relevant": ["x"]}', 3) is None


def test_build_base_retriever() -> None:
    emb, store, kw = indexes()
    assert isinstance(build_base_retriever("hybrid", emb, store, kw, 3), HybridRetriever)
    assert isinstance(build_base_retriever("vector", emb, store, kw, 3), VectorRetriever)
    assert isinstance(build_base_retriever("parent_child", emb, store, kw, 3), ParentChildRetriever)
    with pytest.raises(ConfigError, match="Unknown base retriever"):
        build_base_retriever("magic", emb, store, kw, 3)
    with pytest.raises(ConfigError, match="needs the pipeline's generator"):
        build_base_retriever("graph", emb, store, kw, 3)


def test_validation_and_mode_selection() -> None:
    with pytest.raises(ConfigError, match="grader"):
        make(FakeGenerator(), grader="vibes")
    with pytest.raises(ConfigError, match="on_failure"):
        make(FakeGenerator(), on_failure="panic")

    class GenerateOnly:
        def generate(self, query: str, contexts: list[RetrievedChunk]) -> Answer:
            return Answer(text="")

    with pytest.raises(ConfigError, match="complete"):
        make(GenerateOnly(), grader="llm")
    assert make(FakeGenerator()).mode == "lexical"  # auto + fake → lexical
    assert make(GenerateOnly()).mode == "lexical"
    assert make(ScriptedLLM([])).mode == "llm"
    assert make(ScriptedLLM([]), grader="lexical").mode == "lexical"


def test_lexical_keeps_relevant_and_refuses_irrelevant() -> None:
    retriever = make(FakeGenerator(), k=3)
    kept = retriever.retrieve("what does error E-4471 mean", filters=acl_filter([]))
    assert [r.chunk.id for r in kept] == ["err#0"]
    with trace("t") as root:
        assert retriever.retrieve("capital of france", filters=acl_filter([])) == []
    correct = root.find("correct")
    assert correct is not None
    assert correct.attributes["strategy"] == "widen"
    assert root.attributes["attempts"] == 2
    keep = make(FakeGenerator(), k=3, on_failure="keep")
    assert len(keep.retrieve("capital of france")) == 3  # ungraded fallback
    no_retry = make(FakeGenerator(), rewrite=False)
    with trace("t") as root2:
        no_retry.retrieve("capital of france")
    assert root2.find("correct") is None


def test_lexical_respects_acl() -> None:
    retriever = make(FakeGenerator(), k=5, threshold=0.5)
    public = [r.chunk.id for r in retriever.retrieve("password", filters=acl_filter([]))]
    assert "hr#0" not in public
    hr = [r.chunk.id for r in retriever.retrieve("password", filters=acl_filter(["hr"]))]
    assert "hr#0" in hr


def test_llm_grading_selects_passages() -> None:
    llm = ScriptedLLM(['{"relevant": [1]}'])
    retriever = make(llm, k=3)
    kept = retriever.retrieve("reset password")
    assert len(kept) == 1
    assert len(llm.prompts) == 1
    system, user = llm.prompts[0]
    assert "JSON" in system
    assert "Question: reset password" in user
    assert "[1]" in user


def test_llm_rewrite_then_grade() -> None:
    llm = ScriptedLLM(['{"relevant": []}', '"battery sensor fault code"', '{"relevant": [1, 2]}'])
    retriever = make(llm, k=3)
    with trace("t") as root:
        kept = retriever.retrieve("why is my thermostat broken")
    assert len(kept) == 2
    assert root.find("rewrite").attributes["query"] == "battery sensor fault code"  # type: ignore[union-attr]
    assert root.find("correct").attributes["strategy"] == "rewrite"  # type: ignore[union-attr]
    assert "why is my thermostat broken" in llm.prompts[1][1]


def test_llm_rewrite_empty_reply_falls_back_to_question() -> None:
    llm = ScriptedLLM(["   ", '{"relevant": []}', "", '{"relevant": []}'])
    retriever = make(llm, k=2)
    assert retriever.rewrite("q?") == "q?"  # blank rewrite → original question
    assert retriever.retrieve("reset password") == []  # grade [] → rewrite "" → grade []
    assert llm.replies == []


def test_llm_unparseable_grade_keeps_all(caplog: pytest.LogCaptureFixture) -> None:
    retriever = make(ScriptedLLM(["I think they are all fine"]), k=3)
    assert len(retriever.retrieve("reset password")) == 3
    assert "Unparseable grader reply" in caplog.text
    assert retriever.grade("q", []) == []


def test_corrective_blueprint_end_to_end(tmp_path: Path) -> None:
    pipeline = Pipeline.from_dict({"blueprint": "corrective", "index_dir": str(tmp_path / "i")})
    assert isinstance(pipeline.retriever, CorrectiveRetriever)
    pipeline.ingest(EXAMPLES / "data")
    good = pipeline.query("What does error E-1042 mean?")
    assert good.citations
    bad = pipeline.query("What is the capital of France?")
    assert bad.text == "I don't know."
    assert bad.contexts == []
    # corrective over parent_child, as the selector configures it
    pc = Pipeline.from_dict(
        {
            "blueprint": "corrective",
            "index_dir": str(tmp_path / "pc"),
            "components": {
                "chunker": {
                    "type": "parent_child",
                    "params": {"parent_size": 600, "child_size": 150},
                },
                "retriever": {"params": {"base": "parent_child"}},
            },
        }
    )
    pc.ingest(EXAMPLES / "data")
    assert pc.query("What does error E-5120 mean?").citations
