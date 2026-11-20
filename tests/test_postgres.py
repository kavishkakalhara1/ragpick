"""Postgres backend for the sql retriever, with a fake psycopg module."""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from ragpick.components.generators import LLMGenerator
from ragpick.components.postgres import PostgresDatabase, plan_relations, quote_ident
from ragpick.components.sql import SQL_SYSTEM, SQLQueryError, SQLRetriever, SQLSafetyError
from ragpick.errors import MissingDependencyError
from ragpick.filters import acl_filter
from ragpick.models import Completion

DSN = "postgresql://reader:secret@db.example/shop"


class FakeError(Exception):
    pass


class QueryCanceled(FakeError):  # noqa: N818 - mirrors psycopg's names
    pass


class ReadOnlySqlTransaction(FakeError):  # noqa: N818 - mirrors psycopg's names
    pass


class UndefinedColumn(FakeError):  # noqa: N818 - mirrors psycopg's names
    pass


PLANS: dict[str, Any] = {
    "SELECT COUNT(*) FROM orders": [
        {
            "Plan": {
                "Node Type": "Aggregate",
                "Plans": [{"Relation Name": "orders", "Schema": "public"}],
            }
        }
    ],
    "SELECT * FROM salaries": [{"Plan": {"Relation Name": "salaries", "Schema": "public"}}],
    "SELECT relname FROM pg_class": [
        {"Plan": {"Relation Name": "pg_class", "Schema": "pg_catalog"}}
    ],
    "WITH x AS (SELECT * FROM orders) SELECT * FROM x": json.dumps(
        [{"Plan": {"Plans": [{"Plan": {"Relation Name": "orders", "Schema": "public"}}]}}]
    ),
}


class FakeCursor:
    def __init__(self, db: FakePsycopg) -> None:
        self.db = db
        self.description: list[Any] | None = None
        self._rows: list[tuple[Any, ...]] = []

    def execute(self, sql: str, params: Any = None) -> None:
        self.db.executed.append(sql)
        if sql.startswith("SET "):
            return
        if "information_schema.tables" in sql:
            self._rows = [("orders",), ("products",), ("salaries",)]
            return
        if "information_schema.columns" in sql:
            table = params[1]
            cols = {
                "orders": [("order_id", "integer"), ("sku", "text")],
                "products": [("sku", "text")],
            }
            self._rows = cols.get(table, [])
            return
        if sql.startswith('SELECT * FROM "orders" LIMIT'):
            self.description = [
                type("Col", (), {"name": "order_id"})(),
                type("Col", (), {"name": "sku"})(),
            ]
            self._rows = [(1001, "LM-200")]
            return
        if sql.startswith('SELECT * FROM "products" LIMIT'):
            self._rows = []
            return
        if sql.startswith("EXPLAIN (FORMAT JSON) "):
            inner = sql.removeprefix("EXPLAIN (FORMAT JSON) ")
            if inner in self.db.errors:
                raise self.db.errors[inner]
            self._rows = [(PLANS[inner],)]
            return
        if sql in self.db.errors_on_run:
            raise self.db.errors_on_run[sql]
        self.description = [("n",)]
        self._rows = [(i,) for i in range(self.db.result_rows)]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    def fetchone(self) -> tuple[Any, ...]:
        return self._rows[0]

    def fetchmany(self, n: int) -> list[tuple[Any, ...]]:
        return self._rows[:n]

    def close(self) -> None:
        pass


class FakeConnection:
    def __init__(self, db: FakePsycopg) -> None:
        self.db = db

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.db)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        self.db.rollbacks += 1

    def close(self) -> None:
        pass


class FakePsycopg:
    Error = FakeError

    def __init__(self) -> None:
        self.executed: list[str] = []
        self.rollbacks = 0
        self.dsns: list[str] = []
        self.errors: dict[str, Exception] = {}
        self.errors_on_run: dict[str, Exception] = {}
        self.result_rows = 1

    def connect(self, dsn: str) -> FakeConnection:
        self.dsns.append(dsn)
        return FakeConnection(self)


@pytest.fixture
def pg() -> FakePsycopg:
    return FakePsycopg()


ALL = {"orders", "products", "salaries"}


def test_helpers() -> None:
    assert quote_ident('we"ird') == '"we""ird"'
    assert plan_relations(
        [{"Plan": {"Relation Name": "a", "Schema": "s", "Plans": [{"Relation Name": "b"}]}}]
    ) == {
        ("s", "a"),
        ("", "b"),
    }


def test_session_is_read_only_with_timeout(pg: FakePsycopg) -> None:
    db = PostgresDatabase(DSN, timeout_s=2.5, psycopg=pg)
    assert db.dialect == "PostgreSQL"
    assert db.tables() == ["orders", "products", "salaries"]
    assert "SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY" in pg.executed
    assert "SET statement_timeout = 2500" in pg.executed
    assert pg.dsns == [DSN]


def test_schema(pg: FakePsycopg) -> None:
    schema = PostgresDatabase(DSN, psycopg=pg).schema(
        ["orders", "products", "missing"], sample_rows=1
    )
    assert 'CREATE TABLE "orders" (\n  "order_id" integer,\n  "sku" text\n);' in schema
    assert "sample rows from orders" in schema
    assert "[1001, 'LM-200']" in schema
    assert "missing" not in schema
    assert "sample rows" not in PostgresDatabase(DSN, psycopg=pg).schema(["orders"], sample_rows=0)


def test_run_checks_plan_relations(pg: FakePsycopg) -> None:
    db = PostgresDatabase(DSN, psycopg=pg)
    columns, rows, truncated, read = db.run("SELECT COUNT(*) FROM orders", ALL)
    assert (columns, rows, truncated, read) == (["n"], [(0,)], False, {"orders"})
    _, _, _, read = db.run("WITH x AS (SELECT * FROM orders) SELECT * FROM x", {"orders"})
    assert read == {"orders"}
    with pytest.raises(SQLSafetyError, match="Access denied: salaries"):
        db.run("SELECT * FROM salaries", {"orders"})
    with pytest.raises(SQLSafetyError, match="Access denied: pg_class"):
        db.run("SELECT relname FROM pg_class", ALL | {"pg_class"})
    assert "SELECT * FROM salaries" not in pg.executed  # never executed, only explained
    pg.result_rows = 5
    _, rows, truncated, _ = db.run("SELECT COUNT(*) FROM orders", ALL, max_rows=3)
    assert (len(rows), truncated) == (3, True)
    assert pg.rollbacks >= 4  # every run ends its transaction


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT pg_terminate_backend(1)",
        "SELECT set_config('x', 'y', false)",
        "SELECT * FROM dblink('host=evil', 'select 1')",
        "SELECT lo_export(1, '/tmp/x')",
    ],
)
def test_dangerous_functions_rejected(pg: FakePsycopg, sql: str) -> None:
    with pytest.raises(SQLSafetyError, match="Function not allowed"):
        PostgresDatabase(DSN, psycopg=pg).run(sql, ALL)
    assert not any(e.startswith("EXPLAIN") for e in pg.executed)


def test_error_classification(pg: FakePsycopg) -> None:
    db = PostgresDatabase(DSN, timeout_s=1, psycopg=pg)
    pg.errors["SELECT COUNT(*) FROM orders"] = UndefinedColumn('column "x" does not exist')
    with pytest.raises(SQLQueryError, match="does not exist"):
        db.run("SELECT COUNT(*) FROM orders", ALL)
    pg.errors.clear()
    pg.errors_on_run["SELECT COUNT(*) FROM orders"] = QueryCanceled("canceling statement")
    with pytest.raises(SQLSafetyError, match="1s timeout"):
        db.run("SELECT COUNT(*) FROM orders", ALL)
    pg.errors_on_run["SELECT COUNT(*) FROM orders"] = ReadOnlySqlTransaction("read-only")
    with pytest.raises(SQLSafetyError, match="Access denied"):
        db.run("SELECT COUNT(*) FROM orders", ALL)
    with pytest.raises(SQLSafetyError, match="Only SELECT"):
        db.run("DELETE FROM orders", ALL)


def test_missing_psycopg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "psycopg", None)
    with pytest.raises(MissingDependencyError, match=r'pip install "psycopg\[binary\]"'):
        PostgresDatabase(DSN)


class SQLWriter(LLMGenerator):
    def __init__(self, sql: str) -> None:
        self.sql = sql
        self.systems: list[str] = []

    def complete(self, system: str, user: str) -> Completion:
        self.systems.append(system)
        return Completion(text=f"```sql\n{self.sql}\n```")


def test_sql_retriever_selects_postgres_backend(
    pg: FakePsycopg, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setitem(sys.modules, "psycopg", pg)
    llm = SQLWriter("SELECT COUNT(*) FROM orders")
    retriever = SQLRetriever(llm, tmp_path, database=DSN, table_groups={"salaries": ["hr"]})
    assert not retriever.requires_ingest
    assert retriever.database is None
    [ctx] = retriever.retrieve("How many orders?", filters=acl_filter([]))
    assert ctx.source == "sql"
    assert ctx.chunk.doc_id == "sql:orders"
    assert llm.systems[0] == SQL_SYSTEM.format(max_rows=50, dialect="PostgreSQL")
    retriever.index_documents([])  # nothing to ingest for Postgres
    other = SQLRetriever(SQLWriter("x"), tmp_path, database="postgres://u@h/db", schemas=["sales"])
    assert other.dsn == "postgres://u@h/db"
    assert other.schemas == ["sales"]
