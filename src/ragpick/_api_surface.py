"""Build a machine-readable description of ragpick's public API (the 1.x contract).

``tests/api_snapshot.json`` stores the released surface; ``tests/test_api_stability.py``
fails when the current surface differs, so incompatible changes can't slip into a
minor release unnoticed. After an *intentional* change (additions are fine in minor
releases), regenerate it with ``python scripts/api_snapshot.py`` and note it in the
CHANGELOG.
"""

from __future__ import annotations

import inspect
from typing import Any

from pydantic import BaseModel

import ragpick
from ragpick import interfaces, models
from ragpick.blueprints import list_blueprints
from ragpick.config import RagConfig, RouterConfig
from ragpick.pipeline import Pipeline
from ragpick.registry import ENTRY_POINT_GROUP, registry
from ragpick.router import Router
from ragpick.server import ENDPOINTS

ENGINE_METHODS = (
    "from_yaml",
    "from_dict",
    "ingest",
    "retrieve",
    "query",
    "stream",
    "chat",
    "chat_stream",
    "record_feedback",
    "ready",
)
PIPELINE_ONLY = ("for_tenant", "load", "persist")


def _signature(func: Any) -> str:
    sig = inspect.signature(func)
    params = [p for p in sig.parameters.values() if p.name not in ("self", "cls")]
    return str(sig.replace(parameters=params))


def _config_keys(model: type[BaseModel], prefix: str = "") -> list[str]:
    keys: list[str] = []
    for name, field in model.model_fields.items():
        path = f"{prefix}{name}"
        keys.append(path)
        annotation = field.annotation
        candidates = [annotation, *getattr(annotation, "__args__", ())]
        for candidate in candidates:
            if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                keys.extend(_config_keys(candidate, f"{path}."))
                break
    return keys


def _cli() -> dict[str, list[str]]:
    import typer.main

    from ragpick.cli import app

    root = typer.main.get_command(app)
    out: dict[str, list[str]] = {}

    def walk(group: Any, prefix: str) -> None:
        for name, command in sorted(getattr(group, "commands", {}).items()):
            path = f"{prefix}{name}"
            if getattr(command, "commands", None):
                walk(command, f"{path} ")
            else:
                out[path] = sorted(opt for p in command.params for opt in getattr(p, "opts", []))

    walk(root, "")
    return out


def build() -> dict[str, Any]:
    """The current public surface as plain JSON-serialisable data."""
    protocols = {
        name: {
            method: _signature(getattr(obj, method))
            for method in sorted(vars(obj))
            if not method.startswith("_") and callable(getattr(obj, method))
        }
        for name, obj in sorted(vars(interfaces).items())
        if inspect.isclass(obj)
        and getattr(obj, "_is_protocol", False)
        and obj.__module__ == interfaces.__name__
    }
    model_fields = {
        name: sorted(obj.model_fields)
        for name, obj in sorted(vars(models).items())
        if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj.__module__ == models.__name__
    }
    return {
        "version_scheme": "semver; public surface of ragpick 1.x",
        "exports": sorted(ragpick.__all__),
        "protocols": protocols,
        "models": model_fields,
        "engines": {
            "Pipeline": {
                m: _signature(getattr(Pipeline, m)) for m in ENGINE_METHODS + PIPELINE_ONLY
            },
            "Router": {m: _signature(getattr(Router, m)) for m in ENGINE_METHODS},
        },
        "config_keys": {
            "rag.yaml": _config_keys(RagConfig),
            "router.yaml": _config_keys(RouterConfig),
        },
        "cli": _cli(),
        "http": sorted(f"{method} {path}" for method, path in ENDPOINTS),
        "registry": {kind: registry.names(kind) for kind in sorted(registry.kinds())},
        "entry_point_group": ENTRY_POINT_GROUP,
        "blueprints": {bp.name: bp.metadata.status for bp in list_blueprints()},
    }
