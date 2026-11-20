"""Lightweight tracing: nested, timed spans for every query stage.

Components record spans with :func:`span` and annotate the current one with
:func:`set_attributes`. Both are no-ops unless a :class:`Tracer` is active, so
instrumentation costs nothing when tracing is off::

    from ragpick.tracing import span, set_attributes

    with span("grade", contexts=len(chunks)):
        ...
        set_attributes(relevant=3)

Finished traces are handed to exporters (``trace_exporter`` registry kind):
``jsonl`` appends one JSON object per query, ``console`` logs an indented tree.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ragpick.registry import register

logger = logging.getLogger(__name__)


class Span(BaseModel):
    """One timed operation; ``children`` are the operations it contained."""

    name: str
    timestamp: float | None = None
    """Wall-clock start (Unix seconds); set on root spans."""
    start_ms: float = 0.0
    duration_ms: float = 0.0
    attributes: dict[str, Any] = Field(default_factory=dict)
    children: list[Span] = Field(default_factory=list)
    error: str | None = None

    def find(self, name: str) -> Span | None:
        """First span named ``name`` in this subtree (depth-first), or ``None``."""
        if self.name == name:
            return self
        for child in self.children:
            found = child.find(name)
            if found is not None:
                return found
        return None

    def walk(self, depth: int = 0) -> Iterator[tuple[int, Span]]:
        """Yield ``(depth, span)`` for this subtree, depth-first."""
        yield depth, self
        for child in self.children:
            yield from child.walk(depth + 1)


_stack: ContextVar[tuple[Span, ...]] = ContextVar("ragpick_span_stack", default=())
_origin: ContextVar[float] = ContextVar("ragpick_trace_origin", default=0.0)


def _jsonable(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def tracing_active() -> bool:
    """Whether a tracer is recording in the current context."""
    return bool(_stack.get())


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span | None]:
    """Record a child span of the current span (no-op without an active tracer)."""
    stack = _stack.get()
    if not stack:
        yield None
        return
    now = time.perf_counter()
    current = Span(
        name=name,
        start_ms=(now - _origin.get()) * 1000,
        attributes={k: _jsonable(v) for k, v in attributes.items()},
    )
    stack[-1].children.append(current)
    token = _stack.set((*stack, current))
    try:
        yield current
    except BaseException as exc:
        current.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        current.duration_ms = (time.perf_counter() - now) * 1000
        _stack.reset(token)


def set_attributes(**attributes: Any) -> None:
    """Add attributes to the current span (no-op without an active tracer)."""
    stack = _stack.get()
    if stack:
        stack[-1].attributes.update({k: _jsonable(v) for k, v in attributes.items()})


@contextmanager
def trace(name: str, **attributes: Any) -> Iterator[Span]:
    """Start a new root span and make it the current trace."""
    now = time.perf_counter()
    root = Span(
        name=name,
        timestamp=time.time(),
        attributes={k: _jsonable(v) for k, v in attributes.items()},
    )
    stack_token = _stack.set((root,))
    origin_token = _origin.set(now)
    try:
        yield root
    except BaseException as exc:
        root.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        root.duration_ms = (time.perf_counter() - now) * 1000
        _stack.reset(stack_token)
        _origin.reset(origin_token)


def format_tree(root: Span) -> str:
    """Indented text rendering of a trace."""
    lines = []
    for depth, s in root.walk():
        attrs = " ".join(f"{k}={v}" for k, v in s.attributes.items())
        err = f" ERROR {s.error}" if s.error else ""
        lines.append(f"{'  ' * depth}{s.name} {s.duration_ms:.1f}ms {attrs}{err}".rstrip())
    return "\n".join(lines)


@register("trace_exporter", "jsonl")
class JsonlExporter:
    """Appends each trace as one JSON line to ``path``."""

    def __init__(self, path: str = ".ragpick/traces.jsonl") -> None:
        self.path = Path(path)

    def export(self, root: Span) -> None:
        """Append ``root`` to the file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = root.model_dump()
        record["timestamp"] = root.timestamp or time.time()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


@register("trace_exporter", "console")
class ConsoleExporter:
    """Logs each trace as an indented tree (logger ``ragpick.tracing``, INFO level)."""

    def export(self, root: Span) -> None:
        """Log ``root``."""
        logger.info("trace\n%s", format_tree(root))
