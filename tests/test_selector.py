"""Every selector rule is exercised here."""

from __future__ import annotations

from typing import Any

import pytest

from ragpick.config import Requirements
from ragpick.registry import Registry
from ragpick.selector import RULES, Rule, recommend


def rec(installed: bool = True, store: str = "memory", **req: Any) -> Any:
    return recommend(req, store=store, installed=lambda module: installed)


def test_rule_ids_are_unique() -> None:
    ids = [r.id for r in RULES]
    assert len(ids) == len(set(ids))


def test_defaults_choose_naive() -> None:
    r = rec()
    assert r.blueprint == "naive"
    assert "small_simple_corpus" in r.matched_rules
    assert any("small corpus" in reason for reason in r.reasons)
    assert "hybrid" in r.alternatives


def test_keyword_heavy_chooses_hybrid_even_when_small() -> None:
    r = rec(keyword_heavy=True, corpus_size="small")
    assert r.blueprint == "hybrid"
    assert "keyword_heavy" in r.matched_rules
    assert "small_simple_corpus" not in r.matched_rules
    assert any("BM25" in reason for reason in r.reasons)
    assert "naive" in r.alternatives


@pytest.mark.parametrize("size", ["medium", "large"])
def test_medium_or_large_corpus_chooses_hybrid(size: str) -> None:
    r = rec(corpus_size=size)
    assert r.blueprint == "hybrid"
    assert any(f"A {size} corpus" in reason for reason in r.reasons)


def test_keyword_and_size_reasons_both_reported() -> None:
    r = rec(corpus_size="medium", keyword_heavy=True)
    assert {"keyword_heavy", "medium_or_large_corpus"} <= set(r.matched_rules)
    assert sum("BM25" in x for x in r.reasons) == 2


def test_structured_data_recommends_sql() -> None:
    r = rec(data_types=["CSV"])
    assert r.blueprint == "sql"
    assert any("Structured data (csv)" in reason for reason in r.reasons)
    assert "hybrid" in r.alternatives
    assert not r.warnings
    # reasons for blueprints that lost the vote are not shown
    assert not any("small corpus" in reason for reason in r.reasons)


def test_mixed_data_and_sql_access_control_warn() -> None:
    r = rec(data_types=["markdown", "sql"], access_control=True)
    assert r.blueprint == "sql"
    assert any("second pipeline" in w for w in r.warnings)
    assert any("table_groups" in w for w in r.warnings)
    assert not any("second pipeline" in w for w in rec(data_types=["sqlite"]).warnings)


def test_strict_grounding_chooses_corrective() -> None:
    r = rec(strict_grounding=True, keyword_heavy=True, long_documents=False)
    assert r.blueprint == "corrective"
    assert any("I don't know" in reason for reason in r.reasons)
    assert r.overrides == {}
    assert not r.warnings


def test_corrective_with_long_documents_uses_parent_child_base() -> None:
    r = rec(strict_grounding=True, long_documents=True)
    assert r.blueprint == "corrective"
    assert r.overrides["retriever"] == {"type": "corrective", "params": {"base": "parent_child"}}
    assert r.overrides["chunker"] == {"type": "parent_child"}
    assert any("whole sections" in reason for reason in r.reasons)


def test_corrective_low_latency_warns() -> None:
    r = rec(strict_grounding=True, latency="low")
    assert any("grading call" in w for w in r.warnings)


def test_low_latency_disables_reranker_with_reason() -> None:
    r = rec(latency="low", keyword_heavy=True)
    assert r.overrides == {"reranker": {"type": "none"}}
    assert any("reranker is disabled" in reason for reason in r.reasons)
    assert not any("cross_encoder" in alt for alt in r.alternatives)


def test_rerank_for_precision_alternative() -> None:
    r = rec(keyword_heavy=True)
    assert any("cross_encoder" in alt for alt in r.alternatives)
    assert r.overrides == {}
    assert not any("cross_encoder" in alt for alt in rec().alternatives)


def test_large_corpus_with_memory_store_warns() -> None:
    assert any("qdrant" in w for w in rec(corpus_size="large").warnings)
    assert not any("qdrant" in w for w in rec(corpus_size="large", store="qdrant").warnings)
    assert not any("in-memory" in w for w in rec(corpus_size="medium").warnings)


def test_citations_reason() -> None:
    assert any("Citations" in r for r in rec(needs_citations=True).reasons)
    assert not any("Citations" in r for r in rec(needs_citations=False).reasons)


def test_access_control_with_filtering_store() -> None:
    r = rec(access_control=True, store="qdrant")
    assert "access_control" in r.matched_rules
    assert any("'qdrant' store" in reason for reason in r.reasons)
    assert not r.warnings


def test_access_control_with_store_without_filters_warns() -> None:
    reg = Registry()

    class DumbStore:
        pass

    reg.add("store", "dumb", DumbStore)
    r = recommend(
        {"access_control": True}, store="dumb", component_registry=reg, installed=lambda m: True
    )
    assert "access_control_unsupported_store" in r.matched_rules
    assert any("Store 'dumb' does not support metadata filters" in w for w in r.warnings)
    # unknown stores are treated as unsupported
    r2 = recommend({"access_control": True}, store="nope", installed=lambda m: True)
    assert any("'nope'" in w for w in r2.warnings)
    # without access control, no warning
    assert not recommend({}, store="dumb", component_registry=reg).warnings


def test_long_documents_choose_parent_child() -> None:
    r = rec(long_documents=True, keyword_heavy=True, corpus_size="large")
    assert r.blueprint == "parent_child"
    assert any("parent sections" in reason for reason in r.reasons)
    assert "hybrid" in r.alternatives
    # hybrid's own reasons are hidden because it lost the vote
    assert not any("A large corpus" in reason for reason in r.reasons)


def test_long_documents_with_low_latency_warns() -> None:
    r = rec(long_documents=True, latency="low")
    assert r.blueprint == "parent_child"
    assert any("parent_size" in w for w in r.warnings)
    assert r.overrides == {"reranker": {"type": "none"}}
    assert not any("parent_size" in w for w in rec(long_documents=True).warnings)


def test_structured_data_beats_other_votes() -> None:
    r = rec(data_types=["sql"], long_documents=True, strict_grounding=True)
    assert r.blueprint == "sql"


def test_pdf_extra_warning_and_parent_child_alternative() -> None:
    missing = rec(installed=False, data_types=["pdf"])
    assert any("ragpick[pdf]" in w for w in missing.warnings)
    assert any(alt.startswith("parent_child") for alt in missing.alternatives)
    assert not rec(installed=True, data_types=["pdf"]).warnings
    long_pdfs = rec(data_types=["pdf"], long_documents=True)
    assert long_pdfs.blueprint == "parent_child"
    assert "pdf_parent_child_hint" not in long_pdfs.matched_rules


def test_unknown_data_types_warn() -> None:
    r = rec(data_types=["markdown", "html", "docx"])
    assert any("docx, html" in w for w in r.warnings)
    known = rec(data_types=["markdown", "txt", "csv"], keyword_heavy=True)
    assert len(known.warnings) == 1  # only the "sql is planned" warning
    assert "planned" in known.warnings[0]


def test_default_installed_check_runs() -> None:
    r = recommend(Requirements(data_types=["pdf"]))
    assert r.blueprint == "naive"


def test_custom_rule_table() -> None:
    table = [
        Rule(id="always", when=lambda c: True, blueprint="hybrid", reason="custom", priority=1),
        Rule(id="planned", when=lambda c: True, blueprint="graph", priority=2),
    ]
    r = recommend({}, rules=table)
    assert r.blueprint == "naive"  # graph is planned and has no fallback -> naive
    assert r.matched_rules == ["always", "planned"]
    with pytest.raises(ValueError, match="No selector rule"):
        recommend({}, rules=[Rule(id="x", when=lambda c: True)])
