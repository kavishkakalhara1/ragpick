"""Chroma vector store (``store: {type: chroma}``).

Requires ``chromadb`` (not a ragpick dependency): ``pip install chromadb``.

Chroma metadata values must be scalars and its ``where`` filters can't test list
membership, so ragpick stores, next to the original metadata (JSON, restored on read),
*filter fields* it can query exactly:

* ``m:<key>`` → the value as text, for scalar metadata;
* ``l:<key>:<value>`` → ``True`` for every item of list-valued metadata;
* ``acl:restricted`` and ``acl:<group>`` for ``allowed_groups``.

ragpick filters (equality / ``in`` / list intersection, and the ``allowed_groups``
access-control rule) are translated to ``where`` clauses over those fields, so
filtering happens inside Chroma before ranking.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ragpick.components._optional import require
from ragpick.errors import ConfigError
from ragpick.filters import ACL_KEY
from ragpick.models import Chunk, RetrievedChunk
from ragpick.registry import register

_META = "ragpick:metadata"
_COLLECTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,510}[A-Za-z0-9]$")


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def filter_fields(metadata: dict[str, Any]) -> dict[str, str | bool]:
    """Chroma-filterable fields for ``metadata`` (see module docstring)."""
    fields: dict[str, str | bool] = {}
    for key, value in metadata.items():
        if key == ACL_KEY:
            continue
        if isinstance(value, list | tuple):
            for item in value:
                fields[f"l:{key}:{_text(item)}"] = True
        elif value is not None:
            fields[f"m:{key}"] = _text(value)
    groups = metadata.get(ACL_KEY)
    if groups is None:
        fields["acl:restricted"] = False
    else:
        fields["acl:restricted"] = True
        for group in groups if isinstance(groups, list | tuple) else [groups]:
            fields[f"acl:{_text(group)}"] = True
    return fields


def _any(clauses: list[dict[str, Any]]) -> dict[str, Any]:
    return clauses[0] if len(clauses) == 1 else {"$or": clauses}


def chroma_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate ragpick filters into a Chroma ``where`` clause (``None`` if empty)."""
    if not filters:
        return None
    conditions: list[dict[str, Any]] = []
    for key, wanted in filters.items():
        values = [
            _text(v) for v in (wanted if isinstance(wanted, list | tuple | set) else [wanted])
        ]
        if key == ACL_KEY:
            options: list[dict[str, Any]] = [{"acl:restricted": {"$eq": False}}]
            options += [{f"acl:{v}": {"$eq": True}} for v in values]
        else:
            options = [{f"m:{key}": {"$in": values}}] if values else []
            options += [{f"l:{key}:{v}": {"$eq": True}} for v in values]
            if not options:
                options = [{f"m:{key}": {"$in": ["\x00never"]}}]  # empty `in` list matches nothing
        conditions.append(_any(options))
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


@register("store", "chroma")
class ChromaStore:
    """Chroma collection with cosine distance.

    Args:
        path: local persistent directory (``chromadb.PersistentClient``).
        host / port: a Chroma server (``chromadb.HttpClient``). Without either, an
            in-memory client is used.
        collection: collection name.
    """

    supports_filters = True

    def __init__(
        self,
        path: str | None = None,
        host: str | None = None,
        port: int = 8000,
        collection: str = "ragpick",
        client: Any | None = None,
    ) -> None:
        if not _COLLECTION_RE.match(collection):
            raise ConfigError(
                f"Invalid Chroma collection name {collection!r}: 3-512 characters from "
                "letters, digits, '.', '_' and '-', starting and ending with a letter or digit"
            )
        if client is None:
            chromadb = require("chromadb", "chroma", "ChromaStore", "pip install chromadb")
            if host:
                client = chromadb.HttpClient(host=host, port=port)
            elif path:
                client = chromadb.PersistentClient(path=path)
            else:
                client = chromadb.EphemeralClient()
        self._client = client
        self._collection = client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk]) -> None:
        """Upsert chunks (with embeddings)."""
        if not chunks:
            return
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.id!r} has no embedding; embed chunks before adding")
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=[list(c.embedding or []) for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    _META: json.dumps(c.metadata, default=str),
                    **filter_fields(c.metadata),
                }
                for c in chunks
            ],
        )

    def search(
        self, vector: list[float], k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Cosine search with filters applied by Chroma."""
        if k <= 0 or self._collection.count() == 0:
            return []
        kwargs: dict[str, Any] = {
            "query_embeddings": [list(vector)],
            "n_results": min(k, self._collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        where = chroma_where(filters)
        if where is not None:
            kwargs["where"] = where
        result = self._collection.query(**kwargs)
        out = []
        for cid, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            metadata = json.loads(meta.get(_META, "{}"))
            chunk = Chunk(id=cid, doc_id=str(meta["doc_id"]), text=text, metadata=metadata)
            out.append(RetrievedChunk(chunk=chunk, score=1.0 - float(distance), source="vector"))
        return out

    def delete(self, doc_ids: list[str]) -> None:
        """Delete all chunks of ``doc_ids``."""
        if doc_ids:
            self._collection.delete(where={"doc_id": {"$in": list(doc_ids)}})

    def persist(self, path: str) -> None:
        """No-op: Chroma persists data itself."""

    def load(self, path: str) -> None:
        """No-op: data lives in Chroma."""
