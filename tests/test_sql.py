"""SQL blueprint: CSV loading, safety layers, retriever, end-to-end."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from conftest import EXAMPLES
from ragpick import Pipeline
from ragpick.components.generators import FakeGenerator, LLMGenerator
from ragpick.components.sql import (
    SQL_SYSTEM,
    CsvLoader,
    NoChunker,
    SQLDatabase,
    SQLQueryError,
    SQLRetriever,
    SQLSafetyError,
    extract_sql,
    format_rows,
    load_csv_table,
    sanitize_identifier,
    validate_sql,
)
from ragpick.errors import ConfigError
from ragpick.filters import acl_filter
from ragpick.models import Completion, Document
from ragpick.tracing import trace

TABLES = EXAMPLES / "tables"


class ScriptedSQLLLM(LLMGenerator):
    """Returns queued SQL for SQL prompts and a fixed answer for RAG prompts."""

    def __init__(self, sql: list[str], answer: str = "There were 4 units [1].") -> None:
        self.sql = list(sql)
        self.answer = answer
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        self.prompts.append((system, user))
        if system.startswith("You translate questions"):
            return Completion(text=f"```sql\n{self.sql.pop(0)}\n```")
        return Completion(text=self.answer)


@pytest.fixture
def db(tmp_path: Path) -> SQLDatabase:
    path = tmp_path / "data.sqlite"
    conn = sqlite3.connect(path)
    for csv_file in sorted(TABLES.glob("*.csv")):
        load_csv_table(conn, csv_file.stem, csv_file.read_text(encoding="utf-8"))
    conn.commit()
    conn.close()
    return SQLDatabase(path, timeout_s=2)


ALL = {"orders", "products", "salaries"}


# ---------------------------------------------------------------- loading
def test_sanitize_identifier() -> None:
    assert sanitize_identifier("Order Date") == "order_date"
    assert sanitize_identifier("2024 sales") == "_2024_sales"
    assert sanitize_identifier("!!!", "col3") == "col3"


def test_load_csv_table_infers_types(tmp_path: Path) -> None:
    conn = sqlite3.connect(":memory:")
    text = "Name,Qty,Price,Note,Name\nA,1,2.5,,x\nB,,3,hi,y\n,,,,\nC,3\n"
    assert load_csv_table(conn, "My Table", text) == 3
    cols = conn.execute('PRAGMA table_info("my_table")').fetchall()
    assert [(c[1], c[2]) for c in cols] == [
        ("name", "TEXT"),
        ("qty", "INTEGER"),
        ("price", "REAL"),
        ("note", "TEXT"),
        ("name_", "TEXT"),
    ]
    assert conn.execute("SELECT qty, price FROM my_table ORDER BY name").fetchall() == [
        (1, 2.5),
        (None, 3.0),
        (3, None),
    ]
    assert load_csv_table(conn, "empty", "") == 0
    assert load_csv_table(conn, "t", "a\n") == 0  # header only: table recreated, no rows
    conn.close()


def test_csv_loader_and_no_chunker(tmp_path: Path) -> None:
    docs = list(CsvLoader().load(str(TABLES)))
    assert [d.id for d in docs] == ["orders.csv", "products.csv", "salaries.csv"]
    assert docs[0].metadata == {
        "format": "csv",
        "table": "orders",
        "source": str(TABLES / "orders.csv"),
    }
    assert [d.id for d in CsvLoader().load(str(TABLES / "products.csv"))] == ["products.csv"]
    with pytest.raises(FileNotFoundError):
        list(CsvLoader().load(str(tmp_path / "nope")))
    assert NoChunker().split(docs) == []


# ---------------------------------------------------------------- safety
def test_extract_and_validate_sql() -> None:
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("Here: ```\nselect 2\n``` done") == "select 2"
    assert extract_sql("SELECT 3;") == "SELECT 3"
    assert validate_sql("  select * from t; ") == "select * from t"
    assert validate_sql("WITH x AS (SELECT 1) SELECT * FROM x")
    for bad, message in [
        ("", "no SQL"),
        ("DELETE FROM orders", "Only SELECT"),
        ("PRAGMA table_info(orders)", "Only SELECT"),
        ("SELECT 1; DROP TABLE orders", "single SQL statement"),
        ("SELECT 'unterminated", "single SQL statement"),
    ]:
        with pytest.raises(SQLSafetyError, match=message):
            validate_sql(bad)


def test_run_query_and_row_cap(db: SQLDatabase) -> None:
    assert db.tables() == ["orders", "products", "salaries"]
    columns, rows, truncated, read = db.run(
        "SELECT sku, SUM(quantity) AS units FROM orders GROUP BY sku ORDER BY units DESC", ALL
    )
    assert columns == ["sku", "units"]
    assert rows[0] == ("LM-300", 8)
    assert not truncated
    assert read == {"orders"}
    _, capped, truncated, _ = db.run("SELECT * FROM orders", ALL, max_rows=5)
    assert len(capped) == 5
    assert truncated
    _, joined, _, read = db.run(
        "SELECT p.name FROM orders o JOIN products p ON p.sku = o.sku WHERE o.order_id = 1001", ALL
    )
    assert joined == [("Lumen LM-200 thermostat",)]
    assert read == {"orders", "products"}


def test_writes_are_impossible(db: SQLDatabase) -> None:
    # a data-modifying CTE passes the SELECT/WITH prefix check but the authorizer denies it
    with pytest.raises(SQLSafetyError, match="Access denied"):
        db.run("WITH x AS (SELECT 1) DELETE FROM orders", ALL)
    # the connection itself is read-only
    conn = db.connect()
    with pytest.raises(sqlite3.OperationalError, match=r"readonly|query_only|read-only"):
        conn.execute("DELETE FROM orders")
    conn.close()
    assert len(db.run("SELECT * FROM orders", ALL)[1]) == 12


def test_forbidden_tables_and_schema_are_denied(db: SQLDatabase) -> None:
    with pytest.raises(SQLSafetyError, match="Access denied: salaries"):
        db.run("SELECT * FROM salaries", {"orders", "products"})
    with pytest.raises(SQLSafetyError, match="Access denied"):
        db.run("SELECT name FROM sqlite_master", ALL)
    with pytest.raises(SQLSafetyError, match="Access denied"):
        db.run("SELECT o.order_id FROM orders o JOIN salaries s ON 1=1", {"orders"})


def test_timeout(db: SQLDatabase) -> None:
    slow = SQLDatabase(db.path, timeout_s=0.05)
    forever = (
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT count(*) FROM n"
    )
    with pytest.raises(SQLSafetyError, match="timeout"):
        slow.run(forever, ALL)


def test_ctes_and_subqueries_are_allowed(db: SQLDatabase) -> None:
    columns, rows, _, read = db.run(
        "WITH units AS (SELECT sku, SUM(quantity) AS n FROM orders GROUP BY sku) "
        "SELECT p.name, u.n FROM units u JOIN products p ON p.sku = u.sku ORDER BY u.n DESC",
        {"orders", "products"},
    )
    assert columns == ["name", "n"]
    assert rows[0] == ("Lumen LM-300 Pro thermostat", 8)
    assert read == {"orders", "products"}
    _, sub, _, _ = db.run("SELECT * FROM (SELECT sku FROM products) AS s LIMIT 1", {"products"})
    assert sub == [("LM-200",)]
    # reads inside a CTE are still checked
    with pytest.raises(SQLSafetyError, match="Access denied: salaries"):
        db.run("WITH x AS (SELECT * FROM salaries) SELECT * FROM x", {"orders"})
    # a CTE that shadows a forbidden table's name never reads the real table
    _, shadow, _, read = db.run(
        "WITH salaries AS (SELECT 1 AS a) SELECT * FROM salaries", {"orders"}
    )
    assert shadow == [(1,)]
    assert read == set()


def test_sql_errors_propagate_for_retry(db: SQLDatabase) -> None:
    with pytest.raises(SQLQueryError, match="no such column"):
        db.run("SELECT nope FROM orders", ALL)


def test_schema_and_format_rows(db: SQLDatabase) -> None:
    schema = db.schema(["products", "missing"], sample_rows=2)
    assert 'CREATE TABLE "products"' in schema
    assert "sample rows from products" in schema
    assert "LM-300" in schema
    assert "sample rows" not in db.schema(["products"], sample_rows=0)
    assert format_rows([], [], False) == "(no columns)"
    assert format_rows(["a"], [], False) == "(0 rows)"
    table = format_rows(["a", "b"], [(1, None)], True)
    assert "| 1 |  |" in table
    assert "more rows exist" in table
    with pytest.raises(ConfigError, match="not found"):
        SQLDatabase(db.path.parent / "missing.sqlite")


# ---------------------------------------------------------------- retriever
def retriever(db: SQLDatabase, llm: Any, tmp_path: Path, **kw: Any) -> SQLRetriever:
    return SQLRetriever(llm, tmp_path / "idx", database=str(db.path), **kw)


def test_retriever_success_and_prompt(db: SQLDatabase, tmp_path: Path) -> None:
    llm = ScriptedSQLLLM(["SELECT SUM(quantity) AS units FROM orders WHERE sku = 'LM-300'"])
    r = retriever(db, llm, tmp_path, table_groups={"salaries": ["hr"]})
    assert not r.requires_ingest
    with trace("t") as root:
        [ctx] = r.retrieve("How many LM-300 units were ordered?", filters=acl_filter([]))
    assert ctx.source == "sql"
    assert ctx.chunk.doc_id == "sql:orders"
    assert ctx.chunk.id.startswith("sql:orders#")
    assert "| 8 |" in ctx.chunk.text
    assert ctx.chunk.metadata["row_count"] == 1
    system, user = llm.prompts[0]
    assert system == SQL_SYSTEM.format(max_rows=50, dialect="SQLite")
    assert 'CREATE TABLE "orders"' in user
    assert "salaries" not in user  # hidden from callers outside hr
    assert root.find("execute_sql").attributes["rows"] == 1  # type: ignore[union-attr]
    assert "SUM(quantity)" in root.find("generate_sql").attributes["sql"]  # type: ignore[union-attr]


def test_retriever_retries_with_error(db: SQLDatabase, tmp_path: Path) -> None:
    llm = ScriptedSQLLLM(["SELECT units FROM orders", "SELECT SUM(quantity) FROM orders"])
    [ctx] = retriever(db, llm, tmp_path).retrieve("total units?")
    assert ctx.source == "sql"
    assert len(llm.prompts) == 2
    assert "no such column: units" in llm.prompts[1][1]
    assert "previous query failed" in llm.prompts[1][1]


def test_retriever_gives_up_and_reports(db: SQLDatabase, tmp_path: Path) -> None:
    llm = ScriptedSQLLLM(["SELECT a FROM orders", "SELECT b FROM orders"])
    [ctx] = retriever(db, llm, tmp_path, max_attempts=2).retrieve("q")
    assert ctx.chunk.id == "sql#error"
    assert ctx.score == 0.0
    assert "no such column: b" in ctx.chunk.text


def test_retriever_does_not_retry_access_denied(db: SQLDatabase, tmp_path: Path) -> None:
    llm = ScriptedSQLLLM(["SELECT * FROM salaries", "SELECT * FROM salaries"])
    r = retriever(db, llm, tmp_path, table_groups={"salaries": ["hr"]})
    [ctx] = r.retrieve("salaries?", filters=acl_filter(["support"]))
    assert ctx.chunk.id == "sql#error"
    assert "Access denied" in ctx.chunk.text
    assert len(llm.prompts) == 1  # no second attempt to probe forbidden tables
    hr = ScriptedSQLLLM(["SELECT AVG(salary_usd) FROM salaries"])
    [ok] = retriever(db, hr, tmp_path, table_groups={"salaries": ["hr"]}).retrieve(
        "avg salary", filters=acl_filter(["hr"])
    )
    assert ok.chunk.doc_id == "sql:salaries"


def test_visible_tables(db: SQLDatabase, tmp_path: Path) -> None:
    r = retriever(
        db,
        ScriptedSQLLLM([]),
        tmp_path,
        tables=["Orders", "salaries"],
        table_groups={"Salaries": ["hr"]},
    )
    assert r.visible_tables(db, acl_filter([])) == ["orders"]
    assert r.visible_tables(db, acl_filter(["hr"])) == ["orders", "salaries"]
    assert r.visible_tables(db, {"allowed_groups": "hr"}) == ["orders", "salaries"]
    nothing = retriever(db, ScriptedSQLLLM([]), tmp_path, tables=["nope"])
    assert nothing.retrieve("q") == []


def test_retriever_requires_real_llm_and_database(db: SQLDatabase, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="needs an LLM"):
        retriever(db, FakeGenerator(), tmp_path).retrieve("q")
    no_db = SQLRetriever(ScriptedSQLLLM([]), tmp_path / "idx")
    assert no_db.requires_ingest
    with pytest.raises(ConfigError, match="No SQL database"):
        no_db.retrieve("q")


def test_index_documents(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    r = SQLRetriever(ScriptedSQLLLM([]), tmp_path / "idx")
    r.index_documents([Document(id="x.md", text="not csv")])
    assert not r.ingested_path.exists()
    r.index_documents(list(CsvLoader().load(str(TABLES))))
    assert SQLDatabase(r.ingested_path).tables() == ["orders", "products", "salaries"]
    existing = SQLRetriever(ScriptedSQLLLM([]), tmp_path / "idx2", database=str(r.ingested_path))
    existing.index_documents(list(CsvLoader().load(str(TABLES))))
    assert "ignored" in caplog.text


# ---------------------------------------------------------------- end to end
def test_sql_blueprint_end_to_end(tmp_path: Path) -> None:
    shutil.copytree(TABLES, tmp_path / "tables")
    llm = ScriptedSQLLLM(
        ["SELECT sku, SUM(quantity) AS units FROM orders GROUP BY sku ORDER BY units DESC LIMIT 1"],
        answer="The LM-300 sold the most units (8) [1].",
    )
    pipeline = Pipeline.from_dict(
        {
            "blueprint": "sql",
            "index_dir": str(tmp_path / "idx"),
            "components": {"retriever": {"params": {"table_groups": {"salaries": ["hr"]}}}},
        },
        components={"generator": llm},
    )
    stats = pipeline.ingest(tmp_path / "tables")
    assert stats.documents == 3
    assert stats.chunks == 0  # nothing goes to the vector store
    answer = pipeline.query("Which product sold the most units?", trace=True)
    assert answer.text == "The LM-300 sold the most units (8) [1]."
    assert answer.citations == [answer.contexts[0].chunk.id]
    assert answer.citations[0].startswith("sql:orders#")
    assert answer.trace is not None
    assert answer.trace.find("execute_sql") is not None


def test_sql_over_existing_database_needs_no_ingest(db: SQLDatabase, tmp_path: Path) -> None:
    llm = ScriptedSQLLLM(["SELECT COUNT(*) FROM products"], answer="4 products [1].")
    pipeline = Pipeline.from_dict(
        {
            "blueprint": "sql",
            "index_dir": str(tmp_path / "never-ingested"),
            "components": {"retriever": {"params": {"database": str(db.path)}}},
        },
        components={"generator": llm},
    )
    answer = pipeline.query("How many products?")
    assert answer.text == "4 products [1]."
    assert not (tmp_path / "never-ingested").exists()
