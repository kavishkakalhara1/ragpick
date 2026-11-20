"""graph blueprint: extraction, knowledge graph, retrieval, access control."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from conftest import EXAMPLES
from ragpick import Pipeline
from ragpick.components.generators import FakeGenerator, LLMGenerator
from ragpick.components.graph import (
    Extraction,
    GraphRetriever,
    KnowledgeGraph,
    heuristic_extract,
    normalize,
    parse_extraction,
)
from ragpick.errors import ConfigError
from ragpick.eval import evaluate, load_dataset
from ragpick.models import Answer, Chunk, Completion, RetrievedChunk
from ragpick.selector import recommend

DATA = EXAMPLES / "data"


def test_normalize() -> None:
    assert normalize("  LM-300   Pro! ") == "lm-300 pro"
    assert normalize("“Tier 2”") == "tier 2"


def test_heuristic_extract() -> None:
    text = (
        "The LM-300 Pro needs a C-wire; the LM-200 runs on AAA batteries. "
        "Every Lumen Plus plan includes priority support. Click Add to continue."
    )
    ex = heuristic_extract(text)
    names = {e.name for e in ex.entities}
    assert {"LM-300 Pro", "LM-300", "LM-200", "AAA", "Lumen Plus"} <= names
    assert "Add" not in names  # single capitalised words are ignored
    assert "Every Lumen Plus" not in names
    pairs = {(r.source, r.target) for r in ex.relations}
    assert ("LM-300 Pro", "LM-300") in pairs or ("LM-300", "LM-300 Pro") in pairs
    assert all(r.relation == "related to" for r in ex.relations)
    assert len(heuristic_extract(text, max_entities=2).entities) == 2


def test_parse_extraction() -> None:
    reply = (
        'Here: {"entities": [{"name": "LM-300"}], '
        '"relations": [{"source": "LM-300", "target": "C-wire", "relation": "requires"}]}'
    )
    parsed = parse_extraction(reply)
    assert parsed is not None
    assert parsed.entities[0].type == "concept"
    assert parsed.relations[0].relation == "requires"
    assert parse_extraction("no json") is None
    assert parse_extraction('{"entities": "bad"}') is None


def chunk(cid: str, text: str, **meta: Any) -> Chunk:
    return Chunk(id=cid, doc_id=cid.split("#")[0], text=text, metadata=meta)


def test_knowledge_graph_ops(tmp_path: Path) -> None:
    g = KnowledgeGraph()
    g.add(
        chunk("a#0", "LM-300 requires C-wire"),
        Extraction.model_validate(
            {
                "entities": [{"name": "LM-300", "description": "thermostat"}],
                "relations": [{"source": "LM-300", "target": "C-wire", "relation": "Requires"}],
            }
        ),
    )
    g.add(
        chunk("b#0", "C-wire powers HVAC"),
        Extraction.model_validate(
            {
                "relations": [
                    {"source": "C-wire", "target": "HVAC", "relation": "powers"},
                    {"source": "x", "target": "x"},
                    {"source": "a", "target": "HVAC"},
                ]
            }
        ),
    )
    assert set(g.nodes) == {"lm-300", "c-wire", "hvac"}
    assert g.edges[("lm-300", "c-wire")] == {"a#0": "requires"}
    assert g.match("Does the LM-300 need a c-wire?") == ["lm-300", "c-wire"]
    assert g.match("nothing here") == []
    hood = g.neighborhood(["lm-300"], hops=2)
    assert hood == {"lm-300": 0, "c-wire": 1, "hvac": 2}
    assert g.neighborhood(["lm-300"], hops=0) == {"lm-300": 0}
    ranked = dict(g.score_chunks(hood))
    assert ranked["a#0"] > ranked["b#0"] > 0
    facts = g.facts(hood, {"a#0", "b#0"}, limit=10)
    assert facts == [
        "LM-300 → requires → C-wire (source: a#0)",
        "C-wire → powers → HVAC (source: b#0)",
    ]
    assert g.facts(hood, {"b#0"}, limit=10) == ["C-wire → powers → HVAC (source: b#0)"]
    assert g.facts(hood, set(), limit=10) == []
    g.persist(tmp_path)
    loaded = KnowledgeGraph()
    assert loaded.load(tmp_path)
    assert loaded.edges == g.edges
    assert loaded.chunks["a#0"].text == "LM-300 requires C-wire"
    assert not KnowledgeGraph().load(tmp_path / "missing")
    g.remove_docs(["a"])
    assert "lm-300" not in g.nodes
    assert ("lm-300", "c-wire") not in g.edges
    assert "c-wire" in g.nodes  # still supported by b#0
    g.remove_docs(["nope"])
    assert len(g) == 2


def make_pipeline(tmp_path: Path, **retriever: Any) -> Pipeline:
    return Pipeline.from_dict(
        {
            "blueprint": "graph",
            "index_dir": str(tmp_path / "idx"),
            "components": {"retriever": {"params": retriever}} if retriever else {},
        }
    )


def test_graph_blueprint_end_to_end(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path)
    assert isinstance(pipeline.retriever, GraphRetriever)
    assert pipeline.retriever.mode == "heuristic"
    pipeline.ingest(DATA)
    assert (tmp_path / "idx" / "graph" / "graph.json").is_file()
    answer = pipeline.query("Which error appears when the LM-200 battery is low?", trace=True)
    first = answer.contexts[0]
    assert first.chunk.doc_id == "graph"
    assert first.chunk.metadata["synthetic"] is True
    assert "LM-200" in first.chunk.text
    assert answer.trace is not None
    search = answer.trace.find("graph_search")
    assert search is not None
    assert "LM-200" in search.attributes["seeds"]
    # a fresh pipeline loads the persisted graph lazily
    again = make_pipeline(tmp_path)
    assert again.retrieve("LM-200", top_k=3)[0].chunk.doc_id == "graph"
    # no graph match → plain hybrid results, no facts
    plain = again.retrieve("password reset link", top_k=3)
    assert all(c.chunk.doc_id != "graph" for c in plain)
    no_facts = make_pipeline(tmp_path, include_facts=False)
    assert all(c.chunk.doc_id != "graph" for c in no_facts.retrieve("LM-200", top_k=3))


def test_graph_never_leaks_restricted_facts(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "public.md").write_text("The LM-300 Pro supports the RMA process.", encoding="utf-8")
    (docs / "secret.md").write_text(
        "---\nallowed_groups: [hr]\n---\nThe LM-300 Pro bonus code is ZX-9999 for HR staff.",
        encoding="utf-8",
    )
    pipeline = make_pipeline(tmp_path)
    pipeline.ingest(docs)

    def everything(groups: list[str] | None) -> str:
        return " ".join(
            c.chunk.text for c in pipeline.retrieve("LM-300 Pro", groups=groups, top_k=10)
        )

    public = everything(None)
    assert "RMA" in public
    assert "ZX-9999" not in public  # neither the chunk nor a graph fact
    assert "ZX-9999" in everything(["hr"])


def test_reingest_replaces_graph_entries(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("The QX-1 connects to the HVAC unit.", encoding="utf-8")
    pipeline = make_pipeline(tmp_path)
    pipeline.ingest(docs)
    assert "qx-1" in pipeline.retriever.graph.nodes  # type: ignore[attr-defined]
    (docs / "a.md").write_text("The QX-2 connects to the SMS gateway.", encoding="utf-8")
    pipeline.ingest(docs)
    nodes = pipeline.retriever.graph.nodes  # type: ignore[attr-defined]
    assert "qx-1" not in nodes
    assert "qx-2" in nodes


class ExtractingLLM(LLMGenerator):
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, system: str, user: str) -> Completion:
        self.calls += 1
        return Completion(text=self.reply)


def test_llm_extraction(tmp_path: Path) -> None:
    reply = (
        '{"entities": [{"name": "LM-300", "type": "product"}, {"name": "C-wire", "type": "part"}], '
        '"relations": [{"source": "LM-300", "target": "C-wire", "relation": "requires"}]}'
    )
    llm = ExtractingLLM(reply)
    pipeline = Pipeline.from_dict(
        {"blueprint": "graph", "index_dir": str(tmp_path / "i")}, components={"generator": llm}
    )
    assert pipeline.retriever.mode == "llm"  # type: ignore[attr-defined]
    pipeline.ingest(DATA / "getting-started.md")
    assert llm.calls >= 1
    facts = pipeline.retrieve("Does the LM-300 need a C-wire?", top_k=3)[0]
    assert "LM-300 → requires → C-wire" in facts.chunk.text
    bad = ExtractingLLM("not json")
    retriever = Pipeline.from_dict(
        {"blueprint": "graph", "index_dir": str(tmp_path / "j")}, components={"generator": bad}
    ).retriever
    assert retriever.extract("The LM-200 uses AAA cells.").entities  # type: ignore[attr-defined]


def test_graph_retriever_validation(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="extractor"):
        make_pipeline(tmp_path, extractor="magic")
    with pytest.raises(ConfigError, match="hops"):
        make_pipeline(tmp_path, hops=-1)

    class GenerateOnly:
        def generate(self, query: str, contexts: list[RetrievedChunk]) -> Answer:
            return Answer(text="")

    with pytest.raises(ConfigError, match="complete"):
        Pipeline.from_dict(
            {"blueprint": "graph", "components": {"retriever": {"params": {"extractor": "llm"}}}},
            components={"generator": GenerateOnly()},
        )
    assert make_pipeline(tmp_path, extractor="heuristic").retriever.mode == "heuristic"  # type: ignore[attr-defined]
    assert isinstance(FakeGenerator(), LLMGenerator)


def test_synthetic_contexts_are_excluded_from_retrieval_metrics(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path)
    pipeline.ingest(DATA)
    report = evaluate(pipeline, load_dataset(EXAMPLES / "eval.jsonl"))
    assert all("graph" not in r.retrieved_doc_ids for r in report.results)
    assert (report.metrics["hit_rate@k"] or 0) >= 0.9


def test_selector_multi_hop() -> None:
    r = recommend({"multi_hop": True}, installed=lambda m: True)
    assert r.blueprint == "graph"
    assert any("knowledge graph" in reason for reason in r.reasons)
    both = recommend({"multi_hop": True, "strict_grounding": True}, installed=lambda m: True)
    assert both.blueprint == "corrective"
    assert any("bench -b corrective -b graph" in w for w in both.warnings)
