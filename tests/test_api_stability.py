"""The 1.x stability contract: public API snapshot and deprecation helpers."""

from __future__ import annotations

import json
import warnings

import pytest

from conftest import ROOT
from ragpick import RagpickDeprecationWarning
from ragpick._api_surface import build
from ragpick.deprecation import deprecated, warn_deprecated


def test_public_api_matches_snapshot() -> None:
    """The 1.x contract: changing the public surface must be a deliberate decision.

    If this fails after an intentional change (additions are fine in minor releases),
    run ``python scripts/api_snapshot.py`` and document the change in CHANGELOG.md.
    Removals or incompatible changes need a deprecation period and a major release.
    """
    snapshot = json.loads((ROOT / "tests" / "api_snapshot.json").read_text(encoding="utf-8"))
    current = json.loads(json.dumps(build(), sort_keys=True))
    for section in snapshot:
        assert current[section] == snapshot[section], f"public API section '{section}' changed"
    assert set(current) == set(snapshot)


def test_snapshot_covers_the_contract() -> None:
    surface = build()
    assert {"Pipeline", "Router", "load_config", "load_engine", "register", "registry"} <= set(
        surface["exports"]
    )
    assert "query" in surface["engines"]["Router"]
    assert "for_tenant" in surface["engines"]["Pipeline"]
    assert "components.retriever.params" in surface["config_keys"]["rag.yaml"]
    assert "router.routes" in surface["config_keys"]["router.yaml"]
    assert "--fail-under" in surface["cli"]["eval"]
    assert "feedback export" in surface["cli"]
    assert "POST /query" in surface["http"]
    assert surface["entry_point_group"] == "ragpick.components"
    assert surface["blueprints"]["adaptive"] == "planned"
    assert "retrieve" in surface["protocols"]["Retriever"]


def test_deprecation_helpers() -> None:
    @deprecated(since="1.0", removed_in="2.0", alternative="new_thing()")
    def old_thing(x: int) -> int:
        """Old."""
        return x * 2

    with pytest.warns(
        RagpickDeprecationWarning,
        match=r"old_thing\(\) is deprecated since ragpick 1.0 .* 2.0\. Use new_thing\(\) instead\.",
    ):
        assert old_thing(2) == 4
    assert "deprecated:: 1.0" in (old_thing.__doc__ or "")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_deprecated("the `foo` option", since="1.1", removed_in="2.0")
    assert issubclass(caught[0].category, DeprecationWarning)
    assert "Use" not in str(caught[0].message)
