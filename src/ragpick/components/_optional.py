"""Helpers for importing optional dependencies with friendly errors."""

from __future__ import annotations

import importlib
from types import ModuleType

from ragpick.errors import MissingDependencyError


def require(module: str, extra: str, component: str, install: str | None = None) -> ModuleType:
    """Import ``module`` or raise :class:`MissingDependencyError`.

    The message suggests ``pip install "ragpick[<extra>]"``, or ``install`` when given
    (for packages that aren't a ragpick extra).
    """
    try:
        return importlib.import_module(module)
    except ImportError:
        raise MissingDependencyError(module.split(".")[0], extra, component, install) from None
