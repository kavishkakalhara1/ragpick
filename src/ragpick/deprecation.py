"""Deprecation helpers used to evolve ragpick's public API within a major version.

ragpick 1.x follows semantic versioning (see ``docs/stability.md``): public APIs are
never removed or changed incompatibly in a minor release. Instead they are deprecated
first. They keep working and emit :class:`RagpickDeprecationWarning` (a
``DeprecationWarning``) naming the replacement and the release that removes them.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class RagpickDeprecationWarning(DeprecationWarning):
    """Emitted when deprecated ragpick functionality is used."""


def warn_deprecated(
    what: str, *, since: str, removed_in: str, alternative: str | None = None, stacklevel: int = 3
) -> None:
    """Emit a :class:`RagpickDeprecationWarning` for ``what``."""
    message = f"{what} is deprecated since ragpick {since} and will be removed in {removed_in}."
    if alternative:
        message += f" Use {alternative} instead."
    warnings.warn(message, RagpickDeprecationWarning, stacklevel=stacklevel)


def deprecated(*, since: str, removed_in: str, alternative: str | None = None) -> Callable[[F], F]:
    """Decorator marking a function or method as deprecated (it keeps working)."""

    def decorator(func: F) -> F:
        name = getattr(func, "__qualname__", repr(func))

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warn_deprecated(
                f"{name}()", since=since, removed_in=removed_in, alternative=alternative
            )
            return func(*args, **kwargs)

        note = f"\n\n.. deprecated:: {since}\n   Removed in {removed_in}."
        if alternative:
            note += f" Use {alternative} instead."
        wrapper.__doc__ = (func.__doc__ or "") + note
        return wrapper  # type: ignore[return-value]

    return decorator
