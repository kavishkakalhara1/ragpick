from __future__ import annotations

import pytest

from ragpick.blueprints import blueprint_defaults, get_blueprint, list_blueprints
from ragpick.config import build_config
from ragpick.errors import BlueprintError


def test_list_blueprints_roadmap_order() -> None:
    names = [bp.name for bp in list_blueprints()]
    stable = [
        "naive", "hybrid", "parent_child", "corrective", "graph", "agentic", "multimodal",
        "conversational", "hierarchical", "sql",
    ]  # fmt: skip
    assert names == [*stable, "long_context"]
    assert [bp.name for bp in list_blueprints(include_planned=False)] == stable
    assert get_blueprint("sql").metadata.data == "structured"
    assert get_blueprint("corrective").metadata.data == "documents"


def test_planned_blueprints_have_descriptions_and_no_defaults() -> None:
    for bp in list_blueprints():
        assert bp.metadata.description
        assert bp.metadata.best_for
        if not bp.is_stable:
            assert bp.defaults == {}


@pytest.mark.parametrize("name", [bp.name for bp in list_blueprints(include_planned=False)])
def test_stable_blueprints_produce_valid_configs(name: str) -> None:
    cfg = build_config({"blueprint": name})
    assert cfg.blueprint == name


def test_defaults_are_copies() -> None:
    first = blueprint_defaults("naive")
    first["components"]["embedder"]["type"] = "mutated"
    assert blueprint_defaults("naive")["components"]["embedder"]["type"] == "fake"


def test_get_unknown_blueprint() -> None:
    with pytest.raises(
        BlueprintError, match="Available blueprints: naive, hybrid, parent_child, corrective"
    ):
        get_blueprint("bogus")
