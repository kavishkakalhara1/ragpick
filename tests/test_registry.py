from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import pytest

from ragpick import registry as global_registry
from ragpick.errors import ComponentConfigError, ComponentNotFoundError
from ragpick.registry import Registry

REGISTRY_MODULE = sys.modules["ragpick.registry"]


class Widget:
    def __init__(self, size: int = 1, *, color: str = "red") -> None:
        self.size = size
        self.color = color


def test_register_decorator_and_create() -> None:
    reg = Registry(load_builtins=False)

    @reg.register("embedder", "widget")
    class _W(Widget):
        pass

    obj = reg.create("embedder", "widget", size=3, color="blue")
    assert isinstance(obj, Widget)
    assert (obj.size, obj.color) == (3, "blue")
    assert reg.names("embedder") == ["widget"]
    assert reg.has("embedder", "widget")
    assert reg.accepted_params("embedder", "widget") == {"size", "color"}


def test_duplicate_registration_rejected_unless_override() -> None:
    reg = Registry(load_builtins=False)
    reg.add("loader", "x", Widget)
    reg.add("loader", "x", Widget)  # same factory: idempotent

    class Other(Widget):
        pass

    with pytest.raises(ValueError, match="already registered"):
        reg.add("loader", "x", Other)
    reg.add("loader", "x", Other, override=True)
    assert reg.get("loader", "x") is Other


def test_unknown_component_lists_available_names() -> None:
    reg = Registry(load_builtins=False)
    reg.add("embedder", "alpha", Widget)
    reg.add("embedder", "beta", Widget)
    with pytest.raises(ComponentNotFoundError, match=r"Unknown embedder 'gamma'.*alpha, beta"):
        reg.get("embedder", "gamma")


def test_unknown_kind() -> None:
    reg = Registry(load_builtins=False)
    with pytest.raises(ComponentNotFoundError, match="Unknown component kind"):
        reg.get("teleporter", "x")


def test_invalid_params_are_reported_with_accepted_names() -> None:
    reg = Registry(load_builtins=False)
    reg.add("embedder", "w", Widget)
    with pytest.raises(ComponentConfigError, match=r"Accepted params: size, color"):
        reg.create("embedder", "w", colour="green")


def test_kwargs_factory_accepts_anything() -> None:
    reg = Registry(load_builtins=False)

    def factory(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    reg.add("generator", "kw", factory)
    assert reg.accepted_params("generator", "kw") is None
    assert reg.create("generator", "kw", a=1) == {"a": 1}


def test_new_kinds_can_be_registered() -> None:
    reg = Registry(load_builtins=False)
    reg.add("router", "simple", Widget)
    assert "router" in reg.kinds()


@dataclass
class FakeEntryPoint:
    name: str
    obj: Any = None
    error: Exception | None = None

    def load(self) -> Any:
        if self.error is not None:
            raise self.error
        return self.obj


def test_entry_point_plugins_are_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    eps = [
        FakeEntryPoint("embedder.plugged", Widget),
        FakeEntryPoint("side_effect_module", object()),
        FakeEntryPoint("store.broken", error=ImportError("boom")),
    ]
    monkeypatch.setattr(
        REGISTRY_MODULE.metadata, "entry_points", lambda group: eps if group else []
    )
    reg = Registry(load_builtins=False)
    assert reg.names("embedder") == ["plugged"]
    assert isinstance(reg.create("embedder", "plugged"), Widget)
    assert reg.names("store") == []


def test_plugin_cannot_shadow_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    class Plugin(Widget):
        pass

    monkeypatch.setattr(
        REGISTRY_MODULE.metadata,
        "entry_points",
        lambda group: [FakeEntryPoint("embedder.taken", Plugin)],
    )
    reg = Registry(load_builtins=False)
    reg.add("embedder", "taken", Widget)
    reg.load_plugins()
    assert reg.get("embedder", "taken") is Widget


def test_global_registry_contains_builtins() -> None:
    assert "fake" in global_registry.names("embedder")
    assert {"vector", "hybrid", "keyword"} <= set(global_registry.names("retriever"))
    assert "bm25" in global_registry.names("keyword_index")
