"""Blueprints: named, tested RAG architectures shipped as YAML.

Each ``<name>.yaml`` in this package has a ``metadata`` header and, for stable
blueprints, a ``defaults`` section that is merged under the user's config.
"""

from __future__ import annotations

from functools import cache
from importlib import resources
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from ragpick.errors import BlueprintError


class BlueprintMeta(BaseModel):
    """Descriptive metadata for a blueprint."""

    name: str
    description: str
    best_for: list[str] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    status: Literal["stable", "planned"] = "stable"
    data: Literal["documents", "structured"] = "documents"
    """What the blueprint answers from; ``ragpick bench`` compares ``documents`` ones."""


class Blueprint(BaseModel):
    """A blueprint: metadata plus default config values."""

    metadata: BlueprintMeta
    defaults: dict[str, Any] = Field(default_factory=dict)

    @property
    def name(self) -> str:
        """Blueprint name."""
        return self.metadata.name

    @property
    def is_stable(self) -> bool:
        """Whether the blueprint can be used today."""
        return self.metadata.status == "stable"


_ORDER = [
    "naive",
    "hybrid",
    "parent_child",
    "corrective",
    "graph",
    "agentic",
    "multimodal",
    "sql",
    "conversational",
]


@cache
def _load_all() -> dict[str, Blueprint]:
    found: dict[str, Blueprint] = {}
    for entry in resources.files(__name__).iterdir():
        if entry.name.endswith(".yaml"):
            data = yaml.safe_load(entry.read_text(encoding="utf-8"))
            bp = Blueprint.model_validate(data)
            found[bp.name] = bp
    order = {n: i for i, n in enumerate(_ORDER)}
    return dict(sorted(found.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0])))


def list_blueprints(*, include_planned: bool = True) -> list[Blueprint]:
    """All shipped blueprints, stable ones first in roadmap order."""
    return [bp for bp in _load_all().values() if include_planned or bp.is_stable]


def get_blueprint(name: str) -> Blueprint:
    """Return blueprint ``name`` (stable or planned).

    Raises:
        BlueprintError: unknown name.
    """
    blueprints = _load_all()
    if name not in blueprints:
        raise BlueprintError(
            f"Unknown blueprint '{name}'. Available blueprints: {', '.join(blueprints)}"
        )
    return blueprints[name]


def blueprint_defaults(name: str) -> dict[str, Any]:
    """Default config values for a *stable* blueprint (a deep copy).

    Raises:
        BlueprintError: unknown or planned blueprint.
    """
    bp = get_blueprint(name)
    if not bp.is_stable:
        stable = ", ".join(b.name for b in list_blueprints(include_planned=False))
        raise BlueprintError(
            f"Blueprint '{name}' is planned for a future release and cannot be used yet. "
            f"Stable blueprints: {stable}"
        )
    return yaml.safe_load(yaml.safe_dump(bp.defaults)) or {}
