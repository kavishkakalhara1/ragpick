"""Persistent cache for embeddings and LLM calls (SQLite, standard library only).

Keys are SHA-256 hashes of a *fingerprint* (component type + params + prompt
version) plus the input, so changing a model or its settings never returns stale
results. Values are JSON. Secrets that appear in params only ever enter the cache
as part of a hash.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from ragpick.interfaces import Embedder, Generator, TextCompleter, VisionCompleter
from ragpick.models import Answer, Completion, RetrievedChunk, Usage
from ragpick.tracing import set_attributes

logger = logging.getLogger(__name__)


def make_key(namespace: str, fingerprint: Any, payload: Any) -> str:
    """Stable hash of ``namespace``, ``fingerprint`` and ``payload``."""
    blob = json.dumps([namespace, fingerprint, payload], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Cache:
    """A small key-value store in a SQLite file, safe to share between threads."""

    def __init__(self, path: str | Path, ttl_seconds: float | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "key TEXT PRIMARY KEY, namespace TEXT NOT NULL, value TEXT NOT NULL, "
            "created REAL NOT NULL)"
        )
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        """Return the cached value, or ``None`` if missing or expired."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value, created FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None or (
            self.ttl_seconds is not None and time.time() - row[1] > self.ttl_seconds
        ):
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def set(self, key: str, namespace: str, value: Any) -> None:
        """Store ``value`` (JSON-serialisable) under ``key``."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, namespace, value, created) VALUES (?,?,?,?)",
                (key, namespace, json.dumps(value), time.time()),
            )
            self._conn.commit()

    def stats(self) -> dict[str, int]:
        """Entry counts per namespace (plus ``total``)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT namespace, COUNT(*) FROM cache GROUP BY namespace"
            ).fetchall()
        counts = {str(ns): int(n) for ns, n in rows}
        counts["total"] = sum(counts.values())
        return counts

    def clear(self) -> int:
        """Delete all entries; returns how many were removed."""
        with self._lock:
            removed = self._conn.execute("DELETE FROM cache").rowcount
            self._conn.commit()
        return int(removed)

    def close(self) -> None:
        """Close the database connection (safe to call more than once)."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Cache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is not None:
            with contextlib.suppress(Exception):  # e.g. interpreter shutdown
                conn.close()


class CachedEmbedder:
    """Wraps an :class:`~ragpick.interfaces.Embedder`; only cache misses are embedded."""

    def __init__(self, inner: Embedder, cache: Cache, fingerprint: Any) -> None:
        self.inner = inner
        self.cache = cache
        self.fingerprint = fingerprint

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def dimension(self) -> int:
        """The wrapped embedder's dimension."""
        return self.inner.dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts``, reusing cached vectors."""
        keys = [make_key("embed", self.fingerprint, t) for t in texts]
        vectors: list[list[float] | None] = [self.cache.get(k) for k in keys]
        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing:
            fresh = self.inner.embed([texts[i] for i in missing])
            for i, vector in zip(missing, fresh, strict=True):
                vectors[i] = vector
                self.cache.set(keys[i], "embed", vector)
        set_attributes(embed_cache_hits=len(texts) - len(missing))
        return [v for v in vectors if v is not None]

    def embed_query(self, query: str) -> list[float]:
        """Embed a query (``embed_query`` if the inner embedder has it), cached."""
        from ragpick.components.retrievers import embed_query

        key = make_key("embed_query", self.fingerprint, query)
        cached = self.cache.get(key)
        if cached is not None:
            set_attributes(query_cache_hit=True)
            return [float(x) for x in cached]
        vector = embed_query(self.inner, query)
        self.cache.set(key, "embed_query", vector)
        return vector


def _context_payload(contexts: Sequence[RetrievedChunk]) -> list[tuple[str, str]]:
    return [(c.chunk.id, c.chunk.text) for c in contexts]


class CachedGenerator:
    """Wraps a :class:`~ragpick.interfaces.Generator`; identical inputs reuse answers.

    A cached answer has ``usage.cached = True`` and zero tokens (it cost nothing).
    Image descriptions (``describe_image``) are cached by image hash when supported.
    """

    def __init__(self, inner: Generator, cache: Cache, fingerprint: Any) -> None:
        self.inner = inner
        self.cache = cache
        self.fingerprint = fingerprint

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def generate(self, query: str, contexts: list[RetrievedChunk]) -> Answer:
        """Return a cached answer or generate (and cache) a new one."""
        start = time.perf_counter()
        key = make_key("generate", self.fingerprint, [query, _context_payload(contexts)])
        cached = self.cache.get(key)
        if cached is not None:
            set_attributes(cache_hit=True)
            usage = Usage(cached=True, latency_ms=(time.perf_counter() - start) * 1000)
            return Answer(
                text=cached["text"], citations=cached["citations"], contexts=contexts, usage=usage
            )
        answer = self.inner.generate(query, contexts)
        self.cache.set(key, "generate", {"text": answer.text, "citations": answer.citations})
        return answer

    def stream(self, query: str, contexts: list[RetrievedChunk]) -> Iterator[str | Usage]:
        """Stream from the cache (one piece) or from the inner generator, caching the result."""
        from ragpick.components.generators import parse_citations

        start = time.perf_counter()
        key = make_key("generate", self.fingerprint, [query, _context_payload(contexts)])
        cached = self.cache.get(key)
        if cached is not None:
            yield str(cached["text"])
            yield Usage(cached=True, latency_ms=(time.perf_counter() - start) * 1000)
            return
        parts: list[str] = []
        inner_stream = getattr(self.inner, "stream", None)
        if callable(inner_stream):
            for item in inner_stream(query, contexts):
                if isinstance(item, str):
                    parts.append(item)
                yield item
        else:
            answer = self.inner.generate(query, contexts)
            parts.append(answer.text)
            yield answer.text
            yield answer.usage
        text = "".join(parts).strip()
        self.cache.set(
            key, "generate", {"text": text, "citations": parse_citations(text, contexts)}
        )


class CachedCompleterGenerator(CachedGenerator):
    """A :class:`CachedGenerator` for generators that also implement ``complete``."""

    def complete(self, system: str, user: str) -> Completion:
        """Return a cached completion or call the model."""
        key = make_key("complete", self.fingerprint, [system, user])
        cached = self.cache.get(key)
        if cached is not None:
            set_attributes(cache_hit=True)
            return Completion(text=str(cached), usage=Usage(cached=True))
        inner: TextCompleter = self.inner  # type: ignore[assignment]
        completion = inner.complete(system, user)
        self.cache.set(key, "complete", completion.text)
        return completion


class _CachedVision:
    """Mixin: cached ``describe_image``, keyed by image hash.

    Defined as a real method (not via ``__getattr__``) because ``isinstance`` checks
    against runtime-checkable protocols ignore ``__getattr__`` on Python 3.12+.
    """

    inner: Any
    cache: Cache
    fingerprint: Any

    def describe_image(self, image: bytes, mime: str, prompt: str) -> Completion:
        """Describe ``image`` via the inner generator, reusing cached descriptions."""
        digest = hashlib.sha256(image).hexdigest()
        key = make_key("describe_image", self.fingerprint, [digest, mime, prompt])
        hit = self.cache.get(key)
        if hit is not None:
            return Completion(text=str(hit), usage=Usage(cached=True))
        completion: Completion = self.inner.describe_image(image, mime, prompt)
        self.cache.set(key, "describe_image", completion.text)
        return completion


class CachedVisionGenerator(_CachedVision, CachedGenerator):
    """A :class:`CachedGenerator` for vision generators without ``complete``."""


class CachedVisionCompleterGenerator(_CachedVision, CachedCompleterGenerator):
    """A :class:`CachedCompleterGenerator` that also caches image descriptions."""


def wrap_generator(inner: Generator, cache: Cache, fingerprint: Any) -> CachedGenerator:
    """Wrap ``inner``, preserving its ``complete`` and ``describe_image`` support."""
    vision = isinstance(inner, VisionCompleter)
    if isinstance(inner, TextCompleter):
        cls: type[CachedGenerator] = (
            CachedVisionCompleterGenerator if vision else CachedCompleterGenerator
        )
    else:
        cls = CachedVisionGenerator if vision else CachedGenerator
    return cls(inner, cache, fingerprint)
