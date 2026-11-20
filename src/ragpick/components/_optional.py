"""Helpers for importing optional dependencies with friendly errors."""

from __future__ import annotations

import importlib
from types import ModuleType

from ragpick.errors import MissingDependencyError


def require(module: str, extra: str, component: str) -> ModuleType:
    """Import ``module`` or raise :class:`MissingDependencyError` naming the extra."""
    try:
        return importlib.import_module(module)
    except ImportError:
        raise MissingDependencyError(module.split(".")[0], extra, component) from None
