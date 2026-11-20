"""LanceDB vector store (``store: {type: lancedb}``).

Requires ``lancedb`` (not a ragpick dependency): ``pip install lancedb``. Data lives in
``<index_dir>/lancedb`` by default (embedded, no server), or any LanceDB ``uri``
(local path, ``s3://...``, LanceDB Cloud).

Every row has a ``tags`` list column with one ``key=value`` entry per metadata value
(list-valued metadata contributes one entry per item). ragpick filters, including the
``allowed_groups`` access-control rule, become SQL over ``tags`` (``array_has_any``), with
every value quoted and escaped, applied before the vector search ranks results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragpick.components._optional import require
from ragpick.errors import ConfigError
from ragpick.filters import ACL_KEY
from ragpick.models import Chunk, RetrievedChunk
from ragpick.registry import register


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def tags_for(metadata: dict[str, Any]) -> list[str]:
    """``key=value`` tags for ``metadata`` (list values expand to one tag per item)."""
    tags: list[str] = []
    for key, value in metadata.items():
        if value is None:
            continue
        for item in value if isinstance(value, list | tuple) else [value]:
            tags.append(f"{key}={_text(item)}")
    return tags


def sql_string(value: str) -> str:
    """Quote ``value`` as an SQL string literal."""
    if "\x00" in value:
        raise ConfigError("Filter values may not contain NUL characters")
    return "'" + value.replace("'", "''") + "'"


def lance_where(filters: dict[str, Any] | None) -> str | None:
    """Translate ragpick filters into a LanceDB SQL predicate (``None`` if empty)."""
    if not filters:
        return None
    clauses = []
    for key, wanted in filters.items():
        values = [
            _text(v) for v in (wanted if isinstance(wanted, list | tuple | set) else [wanted])
        ]
        array = "[" + ", ".join(sql_string(f"{key}={v}") for v in values) + "]"
        match = f"array_has_any(tags, {array})" if values else "FALSE"
        if key == ACL_KEY:
            clauses.append(f"(acl_restricted = false OR {match})")
        else:
            clauses.append(f"({match})")
    return " AND ".join(clauses)


@register("store", "lancedb")
class LanceDBStore:
    """Embedded LanceDB table with cosine distance.

    Args:
        uri: database location (default ``<index_dir>/lancedb``).
        table: table name (created on first insert).
    """

    supports_filters = True

    def __init__(
        self,
        index_dir: Path,
        uri: str | None = None,
        table: str = "ragpick",
        db: Any | None = None,
    ) -> None:
        if db is None:
            lancedb = require("lancedb", "lancedb", "LanceDBStore", "pip install lancedb")
            db = lancedb.connect(uri or str(Path(index_dir) / "lancedb"))
        self._db = db
        self.table_name = table
        self._table: Any | None = None

    def _open(self) -> Any | None:
        if self._table is None:
            listing = self._db.table_names()
            if self.table_name in listing:
                self._table = self._db.open_table(self.table_name)
        return self._table

    @staticmethod
    def _row(chunk: Chunk) -> dict[str, Any]:
        groups = chunk.metadata.get(ACL_KEY)
        return {
            "id": chunk.id,
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            "metadata": json.dumps(chunk.metadata, default=str),
            "tags": tags_for(chunk.metadata),
            "acl_restricted": groups is not None,
            "vector": [float(x) for x in chunk.embedding or []],
        }

    def add(self, chunks: list[Chunk]) -> None:
        """Upsert chunks (with embeddings)."""
        if not chunks:
            return
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.id!r} has no embedding; embed chunks before adding")
        rows = [self._row(c) for c in chunks]
        table = self._open()
        if table is None:
            self._table = self._db.create_table(self.table_name, data=rows)
            return
        (
            table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

    def search(
        self, vector: list[float], k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Cosine search with filters applied before ranking (prefilter)."""
        table = self._open()
        if table is None or k <= 0:
            return []
        query = table.search([float(x) for x in vector]).metric("cosine").limit(k)
        where = lance_where(filters)
        if where is not None:
            query = query.where(where, prefilter=True)
        out = []
        for row in query.to_list():
            metadata = json.loads(row["metadata"])
            chunk = Chunk(id=row["id"], doc_id=row["doc_id"], text=row["text"], metadata=metadata)
            score = 1.0 - float(row.get("_distance", 0.0))
            out.append(RetrievedChunk(chunk=chunk, score=score, source="vector"))
        return out

    def delete(self, doc_ids: list[str]) -> None:
        """Delete all chunks of ``doc_ids``."""
        table = self._open()
        if table is None or not doc_ids:
            return
        table.delete("doc_id IN (" + ", ".join(sql_string(d) for d in doc_ids) + ")")

    def persist(self, path: str) -> None:
        """No-op: LanceDB writes to disk on every change."""

    def load(self, path: str) -> None:
        """No-op: the table is opened lazily."""
