"""v1.0: long_context retrieval, API stability contract, Router constructors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from conftest import EXAMPLES
from ragpick import Pipeline, Router
from ragpick.components.long_context import TRUNCATED, LongContextRetriever
from ragpick.errors import ConfigError
from ragpick.selector import recommend

DATA = EXAMPLES / "data"


# =================================================================== long_context
def lc(tmp_path: Path, **params: Any) -> Pipeline:
    raw = {"blueprint": "long_context", "index_dir": str(tmp_path / "idx")}
    if params:
        raw["components"] = {"retriever": {"params": params}}
    return Pipeline.from_dict(raw)


def test_include_all_returns_every_visible_document_in_stable_order(tmp_path: Path) -> None:
    p = lc(tmp_path)
    assert isinstance(p.retriever, LongContextRetriever)
    p.ingest(DATA)
    first = p.retrieve("Does the LM-200 need a C-wire?")
    second = p.retrieve("How much does Lumen Plus cost?")
    ids = [c.chunk.doc_id for c in first]
    assert ids == sorted(ids)  # stable order → identical prompt prefix
    assert ids == [c.chunk.doc_id for c in second]
    assert "internal-escalation.md" not in ids  # restricted
    assert len(ids) == 7
    whole = next(c for c in first if c.chunk.doc_id == "getting-started.md")
    assert whole.chunk.id == "getting-started.md#full"
    assert whole.chunk.metadata["whole_document"] is True
    assert "## Schedules" in whole.chunk.text  # the full document, headings and all
    support = [c.chunk.doc_id for c in p.retrieve("escalate E-3301", groups=["support"])]
    assert "internal-escalation.md" in support
    answer = p.query("Does the LM-200 need a C-wire?", trace=True)
    assert answer.citations
    assert answer.trace.find("long_context").attributes["mode"] == "all"  # type: ignore[union-attr]


def test_ranked_mode_and_budget(tmp_path: Path) -> None:
    p = lc(tmp_path, include_all=False, max_docs=2, max_context_tokens=2000)
    p.ingest(DATA)
    contexts = p.retrieve("What does error E-1042 mean?")
    assert len(contexts) == 2
    assert contexts[0].chunk.doc_id == "error-codes.md"
    assert contexts[0].score >= contexts[1].score
    tight = lc(tmp_path / "t", include_all=True, max_context_tokens=150)
    tight.ingest(DATA)
    capped = tight.retrieve("What does error E-1042 mean?")
    assert capped  # corpus exceeds the budget → ranked mode, truncated to fit
    assert capped[0].chunk.text.endswith(TRUNCATED)
    assert capped[0].chunk.metadata["truncated"] is True
    assert sum(len(c.chunk.text) for c in capped) <= 150 * 4 + len(TRUNCATED)


def test_long_context_reingest_and_persistence(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Version one of the policy.", encoding="utf-8")
    p = lc(tmp_path)
    p.ingest(docs)
    (docs / "a.md").write_text(
        "Version two of the policy, with a new refund rule.", encoding="utf-8"
    )
    p.ingest(docs)
    reloaded = lc(tmp_path)
    [ctx] = reloaded.retrieve("policy")
    assert ctx.chunk.text == "Version two of the policy, with a new refund rule."


def test_long_context_validation() -> None:
    with pytest.raises(ConfigError, match="max_docs"):
        Pipeline.from_dict(
            {"blueprint": "long_context", "components": {"retriever": {"params": {"max_docs": 0}}}}
        )
    with pytest.raises(ConfigError, match="max_context_tokens"):
        Pipeline.from_dict(
            {
                "blueprint": "long_context",
                "components": {"retriever": {"params": {"max_context_tokens": 10}}},
            }
        )


def test_selector_long_context() -> None:
    small = recommend({"long_context_model": True}, installed=lambda m: True)
    assert small.blueprint == "long_context"
    large = recommend(
        {"long_context_model": True, "corpus_size": "large"}, installed=lambda m: True
    )
    assert large.blueprint == "hybrid"
    assert any("long_context suits small corpora" in w for w in large.warnings)


# =================================================================== Router constructors
def test_router_from_yaml_and_dict(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text(
        yaml.safe_dump({"blueprint": "naive", "data": str(DATA), "index_dir": "ia"}),
        encoding="utf-8",
    )
    raw = {"name": "r", "router": {"routes": [{"name": "a", "config": "a.yaml"}]}}
    (tmp_path / "router.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    from_file = Router.from_yaml(tmp_path / "router.yaml", overrides={"name": "renamed"})
    assert from_file.name == "renamed"
    from_dict = Router.from_dict(raw, base_dir=tmp_path)
    from_dict.ingest()
    assert from_dict.query("What does error E-1042 mean?").route == "a"
    with pytest.raises(ConfigError, match=r"router\.routes"):
        Router.from_dict({"router": {"routes": []}})
