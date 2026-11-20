"""PostgreSQL + pgvector vector store (``store: {type: pgvector}``).

Requires ``psycopg`` (v3) and the ``vector`` extension in the database; ragpick doesn't
depend on either: ``pip install "psycopg[binary]"``. Vectors are sent as text and cast
to ``vector``, so the ``pgvector`` Python package isn't needed.

Table layout (created on first insert)::

    id text PRIMARY KEY, doc_id text, text text, metadata jsonb, embedding vector(<dim>)

ragpick filters are translated to SQL over ``metadata`` (values compared as text,
list-valued metadata matched by intersection), including the ``allowed_groups``
access-control rule, so filtering happens inside PostgreSQL before ranking.
"""

from __future__ import annotations

import json
import re
from contextlib import closing
from typing import Any

from ragpick.components._optional import require
from ragpick.errors import ConfigError
from ragpick.filters import ACL_KEY
from ragpick.models import Chunk, RetrievedChunk
from ragpick.registry import register

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def _as_text_list(value: Any) -> list[str]:
    values = value if isinstance(value, list | tuple | set) else [value]
    return [v if isinstance(v, str) else json.dumps(v) for v in values]


def filter_sql(filters: dict[str, Any] | None) -> tuple[str, list[Any]]:
    """``(where_clause, params)`` implementing :func:`ragpick.filters.matches_filters`."""
    if not filters:
        return "TRUE", []
    clauses: list[str] = []
    params: list[Any] = []
    for key, wanted in filters.items():
        values = _as_text_list(wanted)
        array_match = (
            "(jsonb_typeof(metadata -> %s) = 'array' AND "
            "ARRAY(SELECT jsonb_array_elements_text(metadata -> %s)) && %s::text[])"
        )
        scalar_match = (
            "(jsonb_typeof(metadata -> %s) <> 'array' AND metadata ->> %s = ANY(%s::text[]))"
        )
        if key == ACL_KEY:
            clauses.append(
                f"(NOT (metadata ? %s) OR jsonb_typeof(metadata -> %s) = 'null' OR "
                f"{array_match} OR {scalar_match})"
            )
            params += [key, key, key, key, values, key, key, values]
        else:
            clauses.append(f"(metadata ? %s AND ({array_match} OR {scalar_match}))")
            params += [key, key, key, values, key, key, values]
    return " AND ".join(clauses), params


@register("store", "pgvector")
class PgVectorStore:
    """Vector store backed by PostgreSQL with the pgvector extension.

    Args:
        dsn: connection string, e.g. ``postgresql://app:${PG_PASSWORD}@db/rag``.
        table: table name (created on first insert).
        create_extension: run ``CREATE EXTENSION IF NOT EXISTS vector`` (needs privileges).
        index: ``hnsw`` (default), ``ivfflat`` or ``none``: approximate index on the vectors.
    """

    supports_filters = True

    def __init__(
        self,
        dsn: str,
        table: str = "ragpick_chunks",
        create_extension: bool = True,
        index: str = "hnsw",
        psycopg: Any | None = None,
    ) -> None:
        if not _IDENT_RE.match(table):
            raise ConfigError(f"Invalid pgvector table name {table!r}: use letters, digits and _")
        if index not in ("hnsw", "ivfflat", "none"):
            raise ConfigError(f"index must be hnsw, ivfflat or none, got {index!r}")
        self.dsn = dsn
        self.table = table
        self.create_extension = create_extension
        self.index = index
        self._pg = psycopg or require(
            "psycopg", "postgres", "PgVectorStore", 'pip install "psycopg[binary]"'
        )
        self._ready = False

    def _connect(self) -> Any:
        return self._pg.connect(self.dsn, autocommit=True)

    def _table_exists(self, cur: Any) -> bool:
        cur.execute("SELECT to_regclass(%s)", (self.table,))
        return cur.fetchone()[0] is not None

    def _ensure_table(self, cur: Any, dimension: int) -> None:
        if self._ready:
            return
        if self.create_extension:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {self.table} ("
            "id text PRIMARY KEY, doc_id text NOT NULL, text text NOT NULL, "
            f"metadata jsonb NOT NULL DEFAULT '{{}}', embedding vector({int(dimension)}) NOT NULL)"
        )
        cur.execute(f"CREATE INDEX IF NOT EXISTS {self.table}_doc_idx ON {self.table} (doc_id)")
        if self.index != "none":
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_vec_idx ON {self.table} "
                f"USING {self.index} (embedding vector_cosine_ops)"
            )
        self._ready = True

    def add(self, chunks: list[Chunk]) -> None:
        """Upsert chunks (with embeddings)."""
        if not chunks:
            return
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.id!r} has no embedding; embed chunks before adding")
        rows = [
            (
                c.id,
                c.doc_id,
                c.text,
                json.dumps(c.metadata, default=str),
                _vector_literal(c.embedding or []),
            )
            for c in chunks
        ]
        with closing(self._connect()) as conn, closing(conn.cursor()) as cur:
            self._ensure_table(cur, len(chunks[0].embedding or []))
            cur.executemany(
                f"INSERT INTO {self.table} (id, doc_id, text, metadata, embedding) "
                "VALUES (%s, %s, %s, %s::jsonb, %s::vector) "
                "ON CONFLICT (id) DO UPDATE SET doc_id = EXCLUDED.doc_id, text = EXCLUDED.text, "
                "metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding",
                rows,
            )

    def search(
        self, vector: list[float], k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Cosine-similarity search with filters applied in SQL."""
        if k <= 0:
            return []
        where, params = filter_sql(filters)
        literal = _vector_literal(vector)
        with closing(self._connect()) as conn, closing(conn.cursor()) as cur:
            if not self._ready and not self._table_exists(cur):
                return []
            cur.execute(
                f"SELECT id, doc_id, text, metadata, 1 - (embedding <=> %s::vector) AS score "
                f"FROM {self.table} WHERE {where} ORDER BY embedding <=> %s::vector LIMIT %s",
                [literal, *params, literal, k],
            )
            rows = cur.fetchall()
        results = []
        for cid, doc_id, text, metadata, score in rows:
            meta = json.loads(metadata) if isinstance(metadata, str) else dict(metadata or {})
            chunk = Chunk(id=cid, doc_id=doc_id, text=text, metadata=meta)
            results.append(RetrievedChunk(chunk=chunk, score=float(score), source="vector"))
        return results

    def delete(self, doc_ids: list[str]) -> None:
        """Delete all chunks of ``doc_ids``."""
        if not doc_ids:
            return
        with closing(self._connect()) as conn, closing(conn.cursor()) as cur:
            if self._ready or self._table_exists(cur):
                cur.execute(f"DELETE FROM {self.table} WHERE doc_id = ANY(%s)", (list(doc_ids),))

    def persist(self, path: str) -> None:
        """No-op: data lives in PostgreSQL."""

    def load(self, path: str) -> None:
        """No-op: data lives in PostgreSQL."""
