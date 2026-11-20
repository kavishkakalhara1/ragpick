"""PostgreSQL backend for the ``sql`` retriever (``database: postgresql://...``).

Requires ``psycopg`` (v3), which isn't a ragpick dependency:
``pip install "psycopg[binary]"``.

Defence in depth (connect with a **read-only role** that can only ``SELECT`` the
tables you want to expose; that's the real boundary):

* every session is ``READ ONLY`` with a ``statement_timeout``;
* statements must be a single ``SELECT``/``WITH`` query;
* administrative/file/network functions (``pg_*``, ``lo_*``, ``dblink``,
  ``set_config``…) are rejected;
* before running, ``EXPLAIN (FORMAT JSON)`` lists every relation the plan reads
  (through CTEs, subqueries and views); any table outside the caller's visible set, or
  in ``pg_catalog``/``information_schema``, is denied. This is how ``table_groups``
  access control works without SQLite's authorizer.
"""

from __future__ import annotations

import json
import re
from contextlib import closing
from typing import Any

from ragpick.components._optional import require
from ragpick.components.sql import SQLQueryError, SQLSafetyError, validate_sql

_DANGEROUS_RE = re.compile(
    r"\b(pg_\w+|lo_\w+|dblink\w*|set_config|current_setting|query_to_xml\w*|"
    r"table_to_xml\w*|cursor_to_xml\w*|database_to_xml\w*)\s*\(",
    re.IGNORECASE,
)
_SYSTEM_SCHEMAS = {"pg_catalog", "information_schema", "pg_toast"}


def quote_ident(name: str) -> str:
    """Quote a PostgreSQL identifier."""
    return '"' + name.replace('"', '""') + '"'


def plan_relations(plan: Any) -> set[tuple[str, str]]:
    """All ``(schema, relation)`` pairs in an ``EXPLAIN (FORMAT JSON)`` result."""
    found: set[tuple[str, str]] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "Relation Name" in node:
                found.add((str(node.get("Schema", "")), str(node["Relation Name"])))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(plan)
    return found


class PostgresDatabase:
    """Read-only PostgreSQL access with plan-based table checks."""

    dialect = "PostgreSQL"

    def __init__(
        self,
        dsn: str,
        timeout_s: float = 5.0,
        schemas: list[str] | None = None,
        psycopg: Any | None = None,
    ) -> None:
        self.dsn = dsn
        self.timeout_s = timeout_s
        self.schemas = list(schemas or ["public"])
        self._pg = psycopg or require(
            "psycopg", "postgres", "The Postgres sql backend", 'pip install "psycopg[binary]"'
        )

    def connect(self) -> Any:
        """Open a session that is read-only and time-limited."""
        conn = self._pg.connect(self.dsn)
        with closing(conn.cursor()) as cur:
            cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
            cur.execute(f"SET statement_timeout = {max(1, int(self.timeout_s * 1000))}")
        conn.commit()
        return conn

    def tables(self) -> list[str]:
        """Tables and views in the configured schemas."""
        with closing(self.connect()) as conn, closing(conn.cursor()) as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = ANY(%s) ORDER BY table_name",
                (self.schemas,),
            )
            return [str(r[0]) for r in cur.fetchall()]

    def schema(self, tables: list[str], sample_rows: int = 3) -> str:
        """Pseudo ``CREATE TABLE`` statements plus sample rows."""
        parts: list[str] = []
        with closing(self.connect()) as conn, closing(conn.cursor()) as cur:
            for table in tables:
                cur.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema = ANY(%s) AND table_name = %s ORDER BY ordinal_position",
                    (self.schemas, table),
                )
                columns = cur.fetchall()
                if not columns:
                    continue
                cols = ",\n  ".join(f"{quote_ident(str(c))} {t}" for c, t in columns)
                parts.append(f"CREATE TABLE {quote_ident(table)} (\n  {cols}\n);")
                if sample_rows > 0:
                    cur.execute(f"SELECT * FROM {quote_ident(table)} LIMIT {int(sample_rows)}")
                    samples = cur.fetchall()
                    if samples:
                        parts.append(f"-- sample rows from {table}: {[str(c) for c, _ in columns]}")
                        parts += [f"--   {list(r)}" for r in samples]
            conn.rollback()
        return "\n".join(parts)

    def _classify(self, exc: Exception) -> Exception:
        name = type(exc).__name__
        if name == "QueryCanceled":
            return SQLSafetyError(f"Query exceeded the {self.timeout_s:g}s timeout")
        if name in ("ReadOnlySqlTransaction", "InsufficientPrivilege"):
            return SQLSafetyError(f"Access denied: {exc}")
        return SQLQueryError(str(exc).strip())

    def run(
        self, sql: str, allowed_tables: set[str], max_rows: int = 50
    ) -> tuple[list[str], list[tuple[Any, ...]], bool, set[str]]:
        """Check the plan's relations, then execute: ``(columns, rows, truncated, tables)``."""
        sql = validate_sql(sql)
        dangerous = _DANGEROUS_RE.search(sql)
        if dangerous:
            raise SQLSafetyError(f"Function not allowed: {dangerous.group(1)}")
        allowed = {t.lower() for t in allowed_tables}
        error_type: type[BaseException] = getattr(self._pg, "Error", Exception)
        with closing(self.connect()) as conn, closing(conn.cursor()) as cur:
            try:
                cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
                raw = cur.fetchone()[0]
                plan = json.loads(raw) if isinstance(raw, str) else raw
                relations = plan_relations(plan)
                denied = sorted(
                    name
                    for schema, name in relations
                    if schema.lower() in _SYSTEM_SCHEMAS or name.lower() not in allowed
                )
                if denied:
                    raise SQLSafetyError(f"Access denied: {', '.join(denied)}")
                cur.execute(sql)
                columns = [str(getattr(d, "name", d[0])) for d in cur.description or []]
                rows = [tuple(r) for r in cur.fetchmany(max_rows + 1)]
            except SQLSafetyError:
                raise
            except error_type as exc:
                raise self._classify(exc) from exc  # type: ignore[arg-type]
            finally:
                conn.rollback()
        return columns, rows[:max_rows], len(rows) > max_rows, {name for _, name in relations}
