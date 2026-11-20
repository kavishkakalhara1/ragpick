"""Vector stores.

Both stores implement the filter semantics in :mod:`ragpick.filters`, including
``allowed_groups`` access control, and advertise it with ``supports_filters = True``
(the selector warns when a store without filter support is combined with access
control).
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ragpick.components._optional import require
from ragpick.filters import ACL_KEY, matches_filters
from ragpick.models import Chunk, RetrievedChunk
from ragpick.registry import register

logger = logging.getLogger(__name__)

_MANIFEST = "manifest.json"
_CHUNKS = "chunks.jsonl"
_VECTORS = "vectors.npy"


def _normalize(
    matrix: np.ndarray[Any, np.dtype[np.float32]],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    result: np.ndarray[Any, np.dtype[np.float32]] = (matrix / norms).astype(np.float32)
    return result


@register("store", "memory")
class InMemoryVectorStore:
    """Exact cosine-similarity search over a numpy matrix; persists to a directory.

    Suitable for up to roughly a few hundred thousand chunks. Adding a chunk whose id
    already exists replaces it.
    """

    supports_filters = True

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._index: dict[str, int] = {}
        self._vectors: np.ndarray[Any, np.dtype[np.float32]] = np.zeros((0, 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def dimension(self) -> int | None:
        """Vector length, or ``None`` while empty."""
        return int(self._vectors.shape[1]) if self._chunks else None

    def add(self, chunks: list[Chunk]) -> None:
        """Insert or replace chunks; every chunk must carry an embedding."""
        if not chunks:
            return
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.id!r} has no embedding; embed chunks before adding")
        new = _normalize(np.asarray([c.embedding for c in chunks], dtype=np.float32))
        if self._chunks and new.shape[1] != self._vectors.shape[1]:
            raise ValueError(
                f"Embedding dimension {new.shape[1]} does not match store dimension "
                f"{self._vectors.shape[1]} (did the embedder change? re-ingest into a new index)"
            )
        if not self._chunks:
            self._vectors = np.zeros((0, new.shape[1]), dtype=np.float32)
        append_rows: list[tuple[Chunk, np.ndarray[Any, np.dtype[np.float32]]]] = []
        for row, chunk in zip(new, chunks, strict=True):
            stored = chunk.model_copy(update={"embedding": None})
            if chunk.id in self._index:
                i = self._index[chunk.id]
                self._chunks[i] = stored
                self._vectors[i] = row
            else:
                self._index[chunk.id] = len(self._chunks) + len(append_rows)
                append_rows.append((stored, row))
        if append_rows:
            self._chunks.extend(c for c, _ in append_rows)
            self._vectors = np.vstack([self._vectors, np.stack([r for _, r in append_rows])])

    def search(
        self, vector: list[float], k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Top-``k`` chunks by cosine similarity among those matching ``filters``."""
        if not self._chunks or k <= 0:
            return []
        query = _normalize(np.asarray(vector, dtype=np.float32)[None, :])[0]
        scores = self._vectors @ query
        if filters:
            mask = np.fromiter(
                (matches_filters(c.metadata, filters) for c in self._chunks),
                dtype=bool,
                count=len(self._chunks),
            )
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            RetrievedChunk(chunk=self._chunks[i], score=float(scores[i]), source="vector")
            for i in order
            if np.isfinite(scores[i])
        ]

    def delete(self, doc_ids: list[str]) -> None:
        """Remove all chunks of ``doc_ids``."""
        drop = set(doc_ids)
        keep = [i for i, c in enumerate(self._chunks) if c.doc_id not in drop]
        if len(keep) == len(self._chunks):
            return
        self._chunks = [self._chunks[i] for i in keep]
        self._vectors = self._vectors[keep]
        self._index = {c.id: i for i, c in enumerate(self._chunks)}

    def persist(self, path: str) -> None:
        """Write ``manifest.json``, ``chunks.jsonl`` and ``vectors.npy`` into ``path``."""
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        with (root / _CHUNKS).open("w", encoding="utf-8") as fh:
            for chunk in self._chunks:
                fh.write(chunk.model_dump_json(exclude={"embedding"}) + "\n")
        np.save(root / _VECTORS, self._vectors)
        manifest = {"format": 1, "type": "memory", "count": len(self), "dimension": self.dimension}
        (root / _MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")

    def load(self, path: str) -> None:
        """Replace the store contents with the data persisted at ``path``."""
        root = Path(path)
        if not (root / _MANIFEST).is_file():
            raise FileNotFoundError(f"No vector store found at {root}")
        with (root / _CHUNKS).open(encoding="utf-8") as fh:
            self._chunks = [Chunk.model_validate_json(line) for line in fh if line.strip()]
        self._vectors = np.load(root / _VECTORS).astype(np.float32)
        self._index = {c.id: i for i, c in enumerate(self._chunks)}


_NO_ACCESS = "__ragpick_no_access__"


def qdrant_filter(filters: dict[str, Any] | None, models: Any) -> Any:
    """Translate ragpick filters into a Qdrant ``Filter`` (``None`` if empty)."""
    if not filters:
        return None
    must: list[Any] = []
    for key, wanted in filters.items():
        field = f"metadata.{key}"
        values = list(wanted) if isinstance(wanted, list | tuple | set) else [wanted]
        if key == ACL_KEY:
            must.append(
                models.Filter(
                    should=[
                        models.IsEmptyCondition(is_empty=models.PayloadField(key=field)),
                        models.FieldCondition(key=field, match=models.MatchAny(any=values)),
                    ]
                )
            )
        else:
            must.append(models.FieldCondition(key=field, match=models.MatchAny(any=values)))
    return models.Filter(must=must)


@register("store", "qdrant")
class QdrantStore:
    """Qdrant vector store (requires ``ragpick[qdrant]``).

    Connects to ``url`` (server), ``path`` (local on-disk mode) or ``location``
    (e.g. ``":memory:"``). The collection is created on first insert. Data is
    persisted by Qdrant itself, so :meth:`persist`/:meth:`load` are no-ops.
    """

    supports_filters = True

    def __init__(
        self,
        collection: str = "ragpick",
        url: str | None = None,
        path: str | None = None,
        location: str | None = None,
        api_key: str | None = None,
        distance: str = "Cosine",
        client: Any | None = None,
        models: Any | None = None,
    ) -> None:
        self.collection = collection
        self.distance = distance
        if models is None:
            models = require("qdrant_client.models", "qdrant", "QdrantStore")
        self._models = models
        if client is None:
            qdrant_client = require("qdrant_client", "qdrant", "QdrantStore")
            if url is None and path is None and location is None:
                location = ":memory:"
            client = qdrant_client.QdrantClient(
                url=url, path=path, location=location, api_key=api_key
            )
        self._client = client

    @staticmethod
    def point_id(chunk_id: str) -> str:
        """Deterministic UUID for a chunk id (Qdrant ids must be UUIDs or ints)."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ragpick:{chunk_id}"))

    def _ensure_collection(self, dimension: int) -> None:
        if self._client.collection_exists(self.collection):
            return
        m = self._models
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=m.VectorParams(
                size=dimension, distance=getattr(m.Distance, self.distance.upper())
            ),
        )

    def add(self, chunks: list[Chunk]) -> None:
        """Upsert chunks (with embeddings) as Qdrant points."""
        if not chunks:
            return
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.id!r} has no embedding; embed chunks before adding")
        self._ensure_collection(len(chunks[0].embedding or []))
        points = []
        for c in chunks:
            metadata = dict(c.metadata)
            if metadata.get(ACL_KEY) == []:
                # Qdrant's is_empty matches [] too; keep "restricted to nobody" restricted.
                metadata[ACL_KEY] = [_NO_ACCESS]
            payload = {"chunk_id": c.id, "doc_id": c.doc_id, "text": c.text, "metadata": metadata}
            points.append(
                self._models.PointStruct(
                    id=self.point_id(c.id), vector=c.embedding, payload=payload
                )
            )
        self._client.upsert(collection_name=self.collection, points=points)

    def search(
        self, vector: list[float], k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Similarity search with filters pushed down to Qdrant."""
        if not self._client.collection_exists(self.collection):
            return []
        response = self._client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=k,
            query_filter=qdrant_filter(filters, self._models),
            with_payload=True,
        )
        results = []
        for point in response.points:
            payload = point.payload or {}
            metadata = dict(payload.get("metadata") or {})
            if metadata.get(ACL_KEY) == [_NO_ACCESS]:
                metadata[ACL_KEY] = []
            chunk = Chunk(
                id=payload["chunk_id"],
                doc_id=payload["doc_id"],
                text=payload["text"],
                metadata=metadata,
            )
            results.append(RetrievedChunk(chunk=chunk, score=float(point.score), source="vector"))
        return results

    def delete(self, doc_ids: list[str]) -> None:
        """Delete every point whose ``doc_id`` is in ``doc_ids``."""
        if not doc_ids or not self._client.collection_exists(self.collection):
            return
        m = self._models
        selector = m.FilterSelector(
            filter=m.Filter(
                must=[m.FieldCondition(key="doc_id", match=m.MatchAny(any=list(doc_ids)))]
            )
        )
        self._client.delete(collection_name=self.collection, points_selector=selector)

    def persist(self, path: str) -> None:
        """No-op: Qdrant persists data itself."""
        logger.debug("QdrantStore.persist(%s): data is persisted by Qdrant", path)

    def load(self, path: str) -> None:
        """No-op: data lives in the Qdrant collection."""
        logger.debug("QdrantStore.load(%s): data is read from Qdrant", path)
