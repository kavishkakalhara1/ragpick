"""Tracing and caching."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from conftest import EXAMPLES
from ragpick import Pipeline
from ragpick.cache import (
    Cache,
    CachedCompleterGenerator,
    CachedEmbedder,
    CachedGenerator,
    make_key,
    wrap_generator,
)
from ragpick.cli import app
from ragpick.components.embedders import FakeEmbedder
from ragpick.components.generators import FakeGenerator
from ragpick.interfaces import Embedder, Generator, TextCompleter
from ragpick.models import Answer, Chunk, RetrievedChunk
from ragpick.tracing import (
    ConsoleExporter,
    JsonlExporter,
    Span,
    format_tree,
    set_attributes,
    span,
    trace,
    tracing_active,
)

DATA = EXAMPLES / "data"


# =================================================================== tracing
def test_spans_are_noops_without_a_tracer() -> None:
    assert not tracing_active()
    with span("x", a=1) as s:
        set_attributes(b=2)
    assert s is None


def test_nested_spans_attributes_and_errors() -> None:
    with trace("root", user="u") as root:
        assert tracing_active()
        with span("child", n=1):
            set_attributes(found=3, obj=object(), items=(1, "a"), mapping={"k": Path("p")})
            with span("grandchild"):
                time.sleep(0.001)
        with pytest.raises(ValueError, match="boom"), span("failing"):
            raise ValueError("boom")
    assert not tracing_active()
    assert root.attributes == {"user": "u"}
    child = root.find("child")
    assert child is not None
    assert child.attributes["found"] == 3
    assert isinstance(child.attributes["obj"], str)
    assert child.attributes["items"] == [1, "a"]
    assert child.attributes["mapping"] == {"k": "p"}
    grand = root.find("grandchild")
    assert grand is not None
    assert grand.duration_ms >= 1.0
    assert child.duration_ms >= grand.duration_ms
    assert root.find("failing").error == "ValueError: boom"  # type: ignore[union-attr]
    assert root.find("missing") is None
    assert [(d, s.name) for d, s in root.walk()] == [
        (0, "root"),
        (1, "child"),
        (2, "grandchild"),
        (1, "failing"),
    ]
    text = format_tree(root)
    assert "  child" in text
    assert "ERROR ValueError: boom" in text


def test_root_records_errors() -> None:
    with pytest.raises(RuntimeError), trace("root") as root:
        raise RuntimeError("x")
    assert root.error == "RuntimeError: x"
    assert not tracing_active()


def test_exporters(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    root = Span(name="query", duration_ms=1.5, children=[Span(name="retrieve")])
    path = tmp_path / "deep" / "traces.jsonl"
    exporter = JsonlExporter(str(path))
    exporter.export(root)
    exporter.export(root)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["name"] == "query"
    assert record["children"][0]["name"] == "retrieve"
    assert "timestamp" in record
    with caplog.at_level(logging.INFO, logger="ragpick.tracing"):
        ConsoleExporter().export(root)
    assert "retrieve" in caplog.text


def _pipeline(tmp_path: Path, **extra: Any) -> Pipeline:
    raw: dict[str, Any] = {"blueprint": "hybrid", "index_dir": str(tmp_path / "idx"), **extra}
    return Pipeline.from_dict(raw, base_dir=tmp_path)


def test_pipeline_query_trace(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.ingest(DATA)
    assert pipeline.query("What does E-1042 mean?").trace is None  # off by default
    answer = pipeline.query("What does E-1042 mean?", trace=True)
    root = answer.trace
    assert root is not None
    assert root.name == "query"
    assert [c.name for c in root.children] == ["retrieve", "rerank", "generate"]
    retrieve = root.find("retrieve")
    assert retrieve is not None
    assert retrieve.attributes["retriever"] == "hybrid"
    assert {c.name for c in retrieve.children} == {"vector_search", "keyword_search"}
    generate = root.find("generate")
    assert generate is not None
    assert generate.attributes["prompt_tokens"] > 0
    assert root.duration_ms >= retrieve.duration_ms


def test_tracing_config_exporters(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    pipeline = _pipeline(
        tmp_path,
        tracing={"enabled": True, "exporters": [{"type": "jsonl", "params": {"path": "t.jsonl"}}]},
    )
    pipeline.ingest(DATA)
    assert pipeline.query("hello").trace is not None
    assert pipeline.query("hello", trace=False).trace is None
    assert len((tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    class Broken:
        def export(self, root: Span) -> None:
            raise OSError("disk full")

    pipeline.exporters.append(Broken())
    with caplog.at_level(logging.WARNING):
        assert pipeline.query("hello").text
    assert "disk full" in caplog.text


# =================================================================== cache
def test_cache_get_set_stats_ttl_clear(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "c" / "cache.sqlite")
    key = make_key("ns", {"model": "m"}, "text")
    assert key == make_key("ns", {"model": "m"}, "text")
    assert key != make_key("ns", {"model": "other"}, "text")
    assert cache.get(key) is None
    cache.set(key, "ns", [1.0, 2.0])
    assert cache.get(key) == [1.0, 2.0]
    assert (cache.hits, cache.misses) == (1, 1)
    assert cache.stats() == {"ns": 1, "total": 1}
    expiring = Cache(tmp_path / "c" / "cache.sqlite", ttl_seconds=0.001)
    time.sleep(0.01)
    assert expiring.get(key) is None
    assert cache.clear() == 1
    assert cache.stats() == {"total": 0}
    cache.close()
    expiring.close()


class CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__(dimension=8)
        self.embedded: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return super().embed(texts)


def test_cached_embedder_only_embeds_misses(tmp_path: Path) -> None:
    inner = CountingEmbedder()
    emb = CachedEmbedder(inner, Cache(tmp_path / "c.sqlite"), {"type": "counting"})
    assert isinstance(emb, Embedder)
    assert emb.dimension == 8
    assert emb.model == "fake-hash"  # attribute passthrough
    first = emb.embed(["a", "b"])
    second = emb.embed(["b", "c", "a"])
    assert inner.embedded == ["a", "b", "c"]
    assert second == [first[1], inner.embed_one("c"), first[0]]
    q1 = emb.embed_query("q")
    assert emb.embed_query("q") == q1
    assert inner.embedded.count("q") == 1


class CountingGenerator(FakeGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.completions = 0

    def generate(self, query: str, contexts: list[RetrievedChunk]) -> Answer:
        self.calls += 1
        return super().generate(query, contexts)

    def complete(self, system: str, user: str) -> Any:
        self.completions += 1
        return super().complete(system, user)


CTX = [
    RetrievedChunk(
        chunk=Chunk(id="a#0", doc_id="a", text="Reset via the link."), score=1, source="vector"
    )
]


def test_cached_generator_and_completer(tmp_path: Path) -> None:
    inner = CountingGenerator()
    gen = wrap_generator(inner, Cache(tmp_path / "c.sqlite"), {"type": "counting"})
    assert isinstance(gen, CachedCompleterGenerator)
    assert isinstance(gen, Generator)
    assert isinstance(gen, TextCompleter)
    assert gen.is_fake  # passthrough
    a1 = gen.generate("How to reset?", CTX)
    a2 = gen.generate("How to reset?", CTX)
    assert inner.calls == 1
    assert (a2.text, a2.citations) == (a1.text, a1.citations)
    assert a2.usage.cached
    assert a2.usage.total_tokens == 0
    assert not a1.usage.cached
    gen.generate("How to reset?", [])  # different contexts → miss
    assert inner.calls == 2
    c1 = gen.complete("s", "u")
    c2 = gen.complete("s", "u")
    assert inner.completions == 1
    assert c2.text == c1.text
    assert c2.usage.cached

    class GenerateOnly:
        def generate(self, query: str, contexts: list[RetrievedChunk]) -> Answer:
            return Answer(text="x")

    plain = wrap_generator(GenerateOnly(), Cache(tmp_path / "d.sqlite"), {})
    assert type(plain) is CachedGenerator
    assert not isinstance(plain, TextCompleter)


def test_pipeline_cache_end_to_end(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, cache={"enabled": True, "dir": "cache"})
    assert isinstance(pipeline.embedder, CachedEmbedder)
    assert isinstance(pipeline.generator, CachedCompleterGenerator)
    pipeline.ingest(DATA)
    first = pipeline.query("What does E-1042 mean?")
    second = pipeline.query("What does E-1042 mean?", trace=True)
    assert not first.usage.cached
    assert second.usage.cached
    assert second.text == first.text
    assert second.trace is not None
    assert second.trace.find("generate").attributes["cached"] is True  # type: ignore[union-attr]
    stats = Cache(tmp_path / "cache" / "cache.sqlite").stats()
    assert stats["embed"] >= 9
    assert stats["generate"] == 1
    # re-ingest embeds nothing new
    reingest = _pipeline(tmp_path, cache={"enabled": True, "dir": "cache"})
    inner = reingest.embedder.inner  # type: ignore[attr-defined]
    calls: list[int] = []
    original = inner.embed
    inner.embed = lambda texts: calls.append(len(texts)) or original(texts)
    reingest.ingest(DATA)
    assert calls == []
    # changing generator params changes the fingerprint → miss
    other = _pipeline(
        tmp_path,
        cache={"enabled": True, "dir": "cache"},
        components={"generator": {"type": "fake", "params": {"max_sentences": 1}}},
    )
    assert not other.query("What does E-1042 mean?").usage.cached
    selective = _pipeline(tmp_path / "s", cache={"enabled": True, "embeddings": False})
    assert not isinstance(selective.embedder, CachedEmbedder)
    assert isinstance(selective.generator, CachedGenerator)


def test_cache_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    Path("rag.yaml").write_text("blueprint: naive\ncache: {enabled: true}\n", encoding="utf-8")
    missing = runner.invoke(app, ["cache", "stats"])
    assert missing.exit_code == 1
    assert "No cache" in missing.output
    assert runner.invoke(app, ["ingest", str(DATA)]).exit_code == 0
    stats = runner.invoke(app, ["cache", "stats"])
    assert stats.exit_code == 0, stats.output
    assert "embed" in stats.output
    assert "total" in stats.output
    cleared = runner.invoke(app, ["cache", "clear"])
    assert cleared.exit_code == 0
    assert "Removed" in cleared.output
    query = runner.invoke(app, ["query", "What does E-1042 mean?", "--trace"])
    assert query.exit_code == 0, query.output
    assert "retrieve" in query.output
    again = runner.invoke(app, ["query", "What does E-1042 mean?"])
    assert "cached" in again.output
