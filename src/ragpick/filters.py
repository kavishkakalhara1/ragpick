"""Metadata filter semantics shared by every store and index.

A filter is a ``dict`` mapping a metadata key to a scalar or a list:

* scalar value → the chunk's metadata value must equal it (or contain it, if the
  metadata value is a list);
* list value → ``in`` match: at least one element must equal (or be contained in)
  the chunk's metadata value;
* a chunk that lacks the key does **not** match.

Access control
--------------
The key :data:`ACL_KEY` (``"allowed_groups"``) is special. A chunk **without**
``allowed_groups`` metadata is public and always matches; a chunk **with** it
matches only when the caller's groups intersect it. An empty group list therefore
means "public chunks only".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

ACL_KEY = "allowed_groups"


def _as_set(value: Any) -> set[Any]:
    if isinstance(value, list | tuple | set | frozenset):
        return {_hashable(v) for v in value}
    return {_hashable(value)}


def _hashable(value: Any) -> Any:
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def matches_filters(metadata: Mapping[str, Any], filters: Mapping[str, Any] | None) -> bool:
    """Return ``True`` when ``metadata`` satisfies every condition in ``filters``."""
    if not filters:
        return True
    for key, wanted in filters.items():
        if key == ACL_KEY:
            if key not in metadata or metadata[key] is None:
                continue  # public chunk
            if not (_as_set(metadata[key]) & _as_set(wanted)):
                return False
            continue
        if key not in metadata:
            return False
        if not (_as_set(metadata[key]) & _as_set(wanted)):
            return False
    return True


def acl_filter(
    groups: list[str] | None, filters: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Combine user ``filters`` with an access-control condition for ``groups``.

    ``groups=None`` is treated like ``[]``: only public chunks are visible. This is
    deliberately secure-by-default.
    """
    combined: dict[str, Any] = dict(filters or {})
    combined[ACL_KEY] = list(groups or [])
    return combined
