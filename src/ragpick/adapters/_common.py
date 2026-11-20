"""Shared helpers for framework adapters."""

from __future__ import annotations

import importlib
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ragpick.errors import ComponentConfigError
from ragpick.filters import matches_filters
from ragpick.models import Chunk, Document, RetrievedChunk


def import_object(path: str) -> Any:
    """Import ``"package.module:Name"`` or ``"package.module.Name"``.

    Raises:
        ComponentConfigError: the module or attribute cannot be imported.
    """
    module_name, sep, attr = path.partition(":")
    if not sep:
        module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ComponentConfigError(
            f"Invalid class_path '{path}': use 'package.module.Class' or 'package.module:Class'"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ComponentConfigError(
            f"Could not import '{path}': {exc}. Install the package that provides it "
            "(for example `pip install langchain-openai` or `pip install llama-index`)."
        ) from None
    try:
        obj = module
        for part in attr.split("."):
            obj = getattr(obj, part)
    except AttributeError:
        raise ComponentConfigError(f"'{module_name}' has no attribute '{attr}'") from None
    return obj


def resolve(
    instance: Any | None,
    class_path: str | None,
    kwargs: Mapping[str, Any] | None,
    what: str,
) -> Any:
    """Return ``instance`` or build ``class_path(**kwargs)``."""
    if instance is not None:
        return instance
    if not class_path:
        raise ComponentConfigError(
            f"{what}: set `class_path` (e.g. 'langchain_openai.OpenAIEmbeddings') "
            "or pass a ready-made object from Python"
        )
    factory = import_object(class_path)
    try:
        return factory(**dict(kwargs or {}))
    except TypeError as exc:
        raise ComponentConfigError(f"{what}: could not create {class_path}: {exc}") from None


def require_method(obj: Any, method: str, what: str) -> None:
    """Raise a clear error if ``obj`` lacks ``method``."""
    if not callable(getattr(obj, method, None)):
        raise ComponentConfigError(
            f"{what}: {type(obj).__name__} has no `{method}()` method; is it the right kind "
            "of object?"
        )


def documents_with_ids(items: list[tuple[str, dict[str, Any]]], root: str) -> list[Document]:
    """Turn ``(text, metadata)`` pairs into documents with stable ids.

    The id is the ``source`` / ``file_path`` metadata relative to ``root``. When one
    source yields several documents (e.g. PDF pages), ``:<n>`` is appended.
    """
    base = Path(root)
    base_dir = base if base.is_dir() else base.parent

    def source_of(meta: dict[str, Any], index: int) -> str:
        source = meta.get("source") or meta.get("file_path") or meta.get("file_name")
        if not source:
            return f"{base.name or 'doc'}-{index}"
        try:
            return Path(str(source)).resolve().relative_to(base_dir.resolve()).as_posix()
        except ValueError:
            return str(source)

    sources = [source_of(meta, i) for i, (_, meta) in enumerate(items)]
    totals = Counter(sources)
    seen: Counter[str] = Counter()
    docs = []
    for (text, meta), source in zip(items, sources, strict=True):
        doc_id = source if totals[source] == 1 else f"{source}:{seen[source]}"
        seen[source] += 1
        docs.append(Document(id=doc_id, text=text, metadata={**meta, "source": str(source)}))
    return docs


def to_retrieved(
    hits: list[tuple[str, dict[str, Any], str | None, float | None]],
    k: int,
    filters: dict[str, Any] | None,
) -> list[RetrievedChunk]:
    """Convert ``(text, metadata, id, score)`` hits, post-filter, and keep ``k``.

    External retrievers cannot apply ragpick filters, so filters (including
    ``allowed_groups`` access control) are applied to their results here.
    Missing scores become ``1 / rank``.
    """
    results: list[RetrievedChunk] = []
    for rank, (text, meta, hit_id, score) in enumerate(hits, start=1):
        if not matches_filters(meta, filters):
            continue
        source = str(meta.get("source") or meta.get("file_path") or hit_id or f"hit-{rank}")
        chunk_id = str(hit_id or f"{source}#{rank}")
        chunk = Chunk(id=chunk_id, doc_id=source, text=text, metadata=dict(meta))
        value = float(score) if score is not None else 1.0 / rank
        results.append(RetrievedChunk(chunk=chunk, score=value, source="vector"))
        if len(results) >= k:
            break
    return results
