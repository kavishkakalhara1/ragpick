"""Pipeline construction, persistence and end-to-end ingest -> query on examples/data."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from conftest import EXAMPLES
from ragpick import Pipeline
from ragpick.components.generators import IDK
from ragpick.components.keyword import BM25Index
from ragpick.components.retrievers import HybridRetriever, VectorRetriever
from ragpick.config import ComponentSpec
from ragpick.errors import ComponentNotFoundError, ConfigError, IndexNotFoundError
from ragpick.models import Answer, RetrievedChunk
from ragpick.registry import Registry, registry

DATA = EXAMPLES / "data"


def make(tmp_path: Path, blueprint: str = "naive", **extra: Any) -> Pipeline:
    raw: dict[str, Any] = {"blueprint": blueprint, "index_dir": str(tmp_path / "index"), **extra}
    return Pipeline.from_dict(raw)


@pytest.mark.parametrize("blueprint", ["naive", "hybrid"])
def test_end_to_end_ingest_and_query(tmp_path: Path, blueprint: str) -> None:
    pipeline = make(tmp_path, blueprint)
    events: list[tuple[str, int, int]] = []
    stats = pipeline.ingest(DATA, progress=lambda s, d, t: events.append((s, d, t)))
    assert stats.documents == 9
    assert stats.chunks >= 9
    assert events[0] == ("embed", 0, stats.chunks)
    assert events[-1] == ("embed", stats.chunks, stats.chunks)
    assert (tmp_path / "index" / "ragpick.json").is_file()

    answer = pipeline.query("How do I reset my password?")
    assert isinstance(answer, Answer)
    assert answer.text != IDK
    assert answer.citations
    assert answer.citations[0].startswith("account-and-password.md#")
    assert len(answer.contexts) <= 5
    assert answer.usage.latency_ms > 0

    # a fresh pipeline (new process) loads the persisted index
    reloaded = make(tmp_path, blueprint)
    again = reloaded.query("How do I reset my password?")
    assert again.citations == answer.citations


def test_hybrid_finds_exact_error_codes(tmp_path: Path) -> None:
    pipeline = make(tmp_path, "hybrid")
    pipeline.ingest(DATA)
    contexts = pipeline.retrieve("E-5120", top_k=3)
    assert contexts[0].chunk.doc_id == "error-codes.md"
    assert isinstance(pipeline.retriever, HybridRetriever)
    assert isinstance(pipeline.keyword_index, BM25Index)


@pytest.mark.parametrize("blueprint", ["naive", "hybrid"])
def test_access_control_end_to_end(tmp_path: Path, blueprint: str) -> None:
    pipeline = make(tmp_path, blueprint)
    pipeline.ingest(DATA)
    question = "When should support escalate an E-3301 ticket to Tier 2?"

    def doc_ids(groups: list[str] | None) -> set[str]:
        contexts = pipeline.retrieve(question, groups=groups, top_k=9)
        return {c.chunk.doc_id for c in contexts}

    assert "internal-escalation.md" not in doc_ids(None)
    assert "internal-escalation.md" not in doc_ids(["hr"])
    assert "internal-escalation.md" in doc_ids(["support"])
    assert "employee-discount-policy.md" not in doc_ids(["support"])
    answer = pipeline.query(question, groups=["support"])
    assert any(c.startswith("internal-escalation.md") for c in answer.citations)
    # restricted docs never leak into answers for other groups
    hr_answer = pipeline.query(question, groups=["hr"])
    assert all(not c.chunk.doc_id.startswith("internal") for c in hr_answer.contexts)


def test_metadata_filters_combined_with_acl(tmp_path: Path) -> None:
    pipeline = make(tmp_path)
    pipeline.ingest(DATA)
    contexts = pipeline.retrieve("warranty", filters={"title": "warranty-and-returns"})
    assert contexts
    assert {c.chunk.doc_id for c in contexts} == {"warranty-and-returns.md"}


def test_reingest_replaces_document_chunks(tmp_path: Path) -> None:
    data = tmp_path / "docs"
    shutil.copytree(DATA, data)
    pipeline = make(tmp_path)
    pipeline.ingest(data)
    n_before = len(pipeline.store)  # type: ignore[arg-type]
    (data / "billing-and-subscriptions.md").write_text(
        "Lumen Plus now costs $5.99.", encoding="utf-8"
    )
    pipeline.ingest(data / "billing-and-subscriptions.md")
    contexts = pipeline.retrieve("Lumen Plus cost", top_k=9)
    billing = [c for c in contexts if c.chunk.doc_id == "billing-and-subscriptions.md"]
    assert [c.chunk.text for c in billing] == ["Lumen Plus now costs $5.99."]
    assert len(pipeline.store) <= n_before  # type: ignore[arg-type]
    # other documents are still present after a partial ingest
    reloaded = make(tmp_path)
    assert any(c.chunk.doc_id == "error-codes.md" for c in reloaded.retrieve("E-1042", top_k=9))


def test_ingest_uses_config_data_path(tmp_path: Path) -> None:
    pipeline = make(tmp_path, data=str(DATA))
    assert pipeline.ingest().documents == 9
    with pytest.raises(ConfigError, match="No path given"):
        make(tmp_path / "other").ingest()


def test_ingest_empty_directory(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "empty").mkdir()
    stats = make(tmp_path).ingest(tmp_path / "empty")
    assert stats.documents == 0
    assert "No supported documents" in caplog.text


def test_query_before_ingest(tmp_path: Path) -> None:
    with pytest.raises(IndexNotFoundError, match="ragpick ingest"):
        make(tmp_path).query("hello")


def test_embedder_mismatch_is_detected(tmp_path: Path) -> None:
    make(tmp_path).ingest(DATA)
    other = make(tmp_path, components={"embedder": {"type": "fake", "params": {"dimension": 32}}})
    with pytest.raises(ConfigError, match="built with embedder"):
        other.query("x")


def test_hybrid_on_naive_index_requires_reingest(tmp_path: Path) -> None:
    make(tmp_path, "naive").ingest(DATA)
    with pytest.raises(IndexNotFoundError, match="no keyword index"):
        make(tmp_path, "hybrid").query("x")


def test_hybrid_retriever_without_keyword_index_defaults_to_bm25(tmp_path: Path) -> None:
    pipeline = make(tmp_path, components={"retriever": {"type": "hybrid", "params": {"k": 10}}})
    assert isinstance(pipeline.keyword_index, BM25Index)
    assert isinstance(pipeline.retriever, HybridRetriever)


def test_injected_components(tmp_path: Path) -> None:
    class EchoGenerator:
        def generate(self, query: str, contexts: list[RetrievedChunk]) -> Answer:
            return Answer(text=f"echo: {query}", contexts=contexts)

    pipeline = Pipeline.from_dict(
        {"blueprint": "naive", "index_dir": str(tmp_path / "i")},
        components={"generator": EchoGenerator()},
    )
    pipeline.ingest(DATA)
    assert pipeline.query("hi").text == "echo: hi"


def test_component_not_implementing_interface(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not implement the Generator interface"):
        Pipeline.from_dict({"blueprint": "naive"}, components={"generator": object()})


def test_unknown_component_type(tmp_path: Path) -> None:
    with pytest.raises(ComponentNotFoundError, match="Available generators: fake, ollama, openai"):
        make(tmp_path, components={"generator": {"type": "gpt5"}})


@pytest.fixture
def scratch_registry() -> Any:
    """The global registry; components added in a test are removed afterwards."""
    added: list[tuple[str, str]] = []

    class Scratch:
        def add(self, kind: str, name: str, factory: Any) -> None:
            registry.add(kind, name, factory)
            added.append((kind, name))

    yield Scratch()
    for kind, name in added:
        registry.remove(kind, name)


def test_custom_retriever_gets_dependencies_injected(scratch_registry: Any) -> None:
    reg = scratch_registry

    class NeedsKeyword:
        def __init__(self, keyword_index: Any) -> None:
            self.keyword_index = keyword_index

        def retrieve(self, query: str, k: int, filters: Any = None) -> list[RetrievedChunk]:
            return []

    reg.add("retriever", "needs_kw", NeedsKeyword)
    config = Pipeline.from_dict({"blueprint": "naive"}).config
    config.components.retriever = ComponentSpec(type="needs_kw")
    # keyword_index auto-defaults to bm25, so the dependency is satisfied
    pipeline = Pipeline(config)
    assert isinstance(pipeline.retriever, NeedsKeyword)
    assert isinstance(pipeline.retriever.keyword_index, BM25Index)


def test_unsatisfied_dependency_is_reported(scratch_registry: Any) -> None:
    reg = scratch_registry

    class Weird:
        def __init__(self, keyword_index: Any) -> None:
            pass

        def rerank(
            self, query: str, chunks: list[RetrievedChunk], top_n: int
        ) -> list[RetrievedChunk]:
            return chunks

    reg.add("reranker", "weird", Weird)
    config = Pipeline.from_dict({"blueprint": "naive"}).config
    config.components.reranker = ComponentSpec(type="weird")
    with pytest.raises(ConfigError, match=r"requires components\.keyword_index"):
        Pipeline(config)


def test_isolated_registry_is_used(tmp_path: Path) -> None:
    reg = Registry()
    assert reg.names("generator") == []
    with pytest.raises(ComponentNotFoundError):
        Pipeline(Pipeline.from_dict({"blueprint": "naive"}).config, component_registry=reg)


def test_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "rag.yaml"
    path.write_text("blueprint: naive\nindex_dir: idx\n", encoding="utf-8")
    pipeline = Pipeline.from_yaml(path, overrides={"query": {"top_k": 2}})
    assert pipeline.index_dir == tmp_path.resolve() / "idx"
    assert pipeline.config.query.top_k == 2
    assert isinstance(pipeline.retriever, VectorRetriever)
    assert registry.has("retriever", "vector")
