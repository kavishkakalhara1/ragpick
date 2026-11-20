"""Text-to-SQL over SQLite: the ``sql`` blueprint's loader, chunker and retriever.

The retriever asks the LLM for one ``SELECT`` statement, runs it against a SQLite
database and returns the result rows as a context, so the generator answers (with
citations) from real data instead of guessing.

Safety, enforced by SQLite itself rather than by the prompt:

* the database is opened **read-only** (``mode=ro``) with ``PRAGMA query_only``;
* an **authorizer** allows only ``SELECT``, reads and functions, and only reads of
  tables the caller may see. Everything else (writes, ``ATTACH``, ``PRAGMA``) is denied;
* statements must be a single ``SELECT``/``WITH`` query;
* a **timeout** aborts long queries, and results are capped at ``max_rows``.

Access control is table-level: ``table_groups: {salaries: [hr]}`` hides a table
(from the schema shown to the LLM *and* from the authorizer) unless the caller's
groups intersect.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any, Protocol

from ragpick.errors import ConfigError, RagpickError
from ragpick.filters import ACL_KEY
from ragpick.interfaces import Generator, TextCompleter
from ragpick.models import Chunk, Document, RetrievedChunk
from ragpick.registry import register
from ragpick.tracing import set_attributes, span

logger = logging.getLogger(__name__)

SQL_SYSTEM = """You translate questions into one read-only {dialect} query.

Rules:
1. Write exactly one SELECT statement (WITH ... SELECT is allowed). Never modify data.
2. Use only the tables and columns in the schema. Quote identifiers with double quotes.
3. Add LIMIT {max_rows} unless the query aggregates to a few rows.
4. Reply with the SQL inside a ```sql code block and nothing else."""

SQL_PROMPT = """Schema:
{schema}

Question: {question}"""

SQL_RETRY_PROMPT = """Schema:
{schema}

Question: {question}

Your previous query failed:
```sql
{sql}
```
Error: {error}

Write a corrected query."""

_IDENT_RE = re.compile(r"[^0-9a-zA-Z_]+")
POSTGRES_PREFIXES = ("postgresql://", "postgres://")
_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class SQLSafetyError(RagpickError):
    """A generated statement was rejected (not a single read-only query, or denied)."""


class SQLQueryError(RagpickError):
    """The database rejected a query (e.g. unknown column); the model may retry."""


class SQLBackend(Protocol):
    """A read-only SQL database the ``sql`` retriever can query."""

    dialect: str

    def tables(self) -> list[str]:
        """Queryable table and view names."""
        ...

    def schema(self, tables: list[str], sample_rows: int = 3) -> str:
        """Schema description (and sample rows) for ``tables``."""
        ...

    def run(
        self, sql: str, allowed_tables: set[str], max_rows: int = 50
    ) -> tuple[list[str], list[tuple[Any, ...]], bool, set[str]]:
        """Run a query: ``(columns, rows, truncated, tables_read)``."""
        ...


# --------------------------------------------------------------------------- loading
def sanitize_identifier(name: str, fallback: str = "col") -> str:
    """Turn ``name`` into a safe lowercase SQL identifier."""
    ident = _IDENT_RE.sub("_", name.strip().lower()).strip("_") or fallback
    return f"_{ident}" if ident[0].isdigit() else ident


def _column_type(values: list[str]) -> str:
    present = [v for v in values if v.strip() != ""]
    if not present:
        return "TEXT"
    for caster, sql_type in ((int, "INTEGER"), (float, "REAL")):
        try:
            for v in present:
                caster(v)
        except ValueError:
            continue
        return sql_type
    return "TEXT"


def _convert(value: str, sql_type: str) -> Any:
    if value.strip() == "":
        return None
    if sql_type == "INTEGER":
        return int(value)
    if sql_type == "REAL":
        return float(value)
    return value


def load_csv_table(conn: sqlite3.Connection, table: str, text: str) -> int:
    """(Re)create ``table`` from CSV ``text`` with inferred column types; returns row count."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return 0
    header, body = rows[0], [r for r in rows[1:] if any(cell.strip() for cell in r)]
    columns: list[str] = []
    for i, name in enumerate(header):
        ident = sanitize_identifier(name, f"col{i + 1}")
        while ident in columns:
            ident += "_"
        columns.append(ident)
    width = len(columns)
    body = [(r + [""] * width)[:width] for r in body]
    types = [_column_type([r[i] for r in body]) for i in range(width)]
    table = sanitize_identifier(table, "data")
    cols_sql = ", ".join(f'"{c}" {t}' for c, t in zip(columns, types, strict=True))
    conn.execute(f'DROP TABLE IF EXISTS "{table}"')
    conn.execute(f'CREATE TABLE "{table}" ({cols_sql})')
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f'INSERT INTO "{table}" VALUES ({placeholders})',
        [[_convert(v, t) for v, t in zip(r, types, strict=True)] for r in body],
    )
    return len(body)


@register("loader", "csv")
class CsvLoader:
    """Loads ``.csv`` files (a file or a directory) as documents, one per table.

    The table name is the file name without extension (sanitised); use with the
    ``sql`` retriever, which turns these documents into SQLite tables at ingest.
    """

    def __init__(self, glob: str = "**/*.csv", encoding: str = "utf-8") -> None:
        self.glob = glob
        self.encoding = encoding

    def load(self, path: str) -> Iterator[Document]:
        """Yield one document per CSV file."""
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"Path not found: {root}")
        files = [root] if root.is_file() else sorted(p for p in root.glob(self.glob) if p.is_file())
        base = root.parent if root.is_file() else root
        for file in files:
            text = file.read_text(encoding=self.encoding)
            meta = {
                "format": "csv",
                "table": sanitize_identifier(file.stem, "data"),
                "source": str(file),
            }
            yield Document(id=file.relative_to(base).as_posix(), text=text, metadata=meta)


@register("chunker", "none")
class NoChunker:
    """Produces no chunks: for pipelines whose retriever does not use the vector store."""

    def split(self, docs: Any) -> list[Chunk]:
        """Return an empty list."""
        return []


# --------------------------------------------------------------------------- database
def extract_sql(reply: str) -> str:
    """Take the SQL out of a model reply (code fence or bare statement)."""
    match = _FENCE_RE.search(reply)
    sql = (match.group(1) if match else reply).strip()
    return sql.rstrip(";").strip()


def validate_sql(sql: str) -> str:
    """Accept exactly one ``SELECT``/``WITH`` statement.

    Raises:
        SQLSafetyError: empty, multiple statements, or not a query.
    """
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        raise SQLSafetyError("The model returned no SQL")
    if not sqlite3.complete_statement(stripped + ";") or ";" in stripped:
        raise SQLSafetyError("Only a single SQL statement is allowed")
    first = stripped.split(None, 1)[0].upper()
    if first not in ("SELECT", "WITH"):
        raise SQLSafetyError(f"Only SELECT queries are allowed, got {first}")
    return stripped


class SQLDatabase:
    """Read-only access to a SQLite file with an authorizer, timeout and row cap."""

    dialect = "SQLite"

    def __init__(self, path: Path, timeout_s: float = 5.0) -> None:
        if not path.is_file():
            raise ConfigError(f"SQLite database not found: {path}")
        self.path = path
        self.timeout_s = timeout_s

    def connect(self) -> sqlite3.Connection:
        """Open a read-only connection."""
        conn = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only = ON")
        return conn

    def tables(self) -> list[str]:
        """User tables in the database."""
        with closing(self.connect()) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()
        return [str(r[0]) for r in rows]

    def objects(self) -> list[str]:
        """All tables and views (including SQLite's internal ones)."""
        with closing(self.connect()) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
        return [str(r[0]) for r in rows]

    def schema(self, tables: list[str], sample_rows: int = 3) -> str:
        """``CREATE TABLE`` statements plus a few sample rows for ``tables``."""
        parts = []
        with closing(self.connect()) as conn:
            for table in tables:
                row = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
                ).fetchone()
                if row is None:
                    continue
                parts.append(str(row[0]) + ";")
                if sample_rows > 0:
                    cur = conn.execute(f'SELECT * FROM "{table}" LIMIT ?', (sample_rows,))
                    names = [d[0] for d in cur.description]
                    samples = cur.fetchall()
                    if samples:
                        parts.append(f"-- sample rows from {table}: {names}")
                        parts += [f"--   {list(r)}" for r in samples]
        return "\n".join(parts)

    def run(
        self, sql: str, allowed_tables: set[str], max_rows: int = 50
    ) -> tuple[list[str], list[tuple[Any, ...]], bool, set[str]]:
        """Execute a validated query: ``(columns, rows, truncated, tables_read)``.

        Raises:
            SQLSafetyError: the statement was rejected or touched a forbidden table.
            SQLQueryError: SQL errors (e.g. unknown column), for the retry loop.
        """
        sql = validate_sql(sql)
        tables_read: set[str] = set()
        denied: list[str] = []
        allowed_lower = {t.lower() for t in allowed_tables}
        # Real tables/views: a READ of one of these outside the allow-list is denied.
        # Other names are CTEs/subquery aliases, whose underlying reads are checked
        # separately. A CTE that shadows a forbidden table name is denied (safe side).
        real_lower = {t.lower() for t in self.objects()}

        def authorizer(action: int, arg1: Any, arg2: Any, db: Any, trigger: Any) -> int:
            if action == sqlite3.SQLITE_READ:
                table = str(arg1)
                lowered = table.lower()
                if lowered.startswith("sqlite_") or (
                    lowered in real_lower and lowered not in allowed_lower
                ):
                    denied.append(table)
                    return sqlite3.SQLITE_DENY
                if lowered in real_lower:
                    tables_read.add(table)
                return sqlite3.SQLITE_OK
            if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE):
                return sqlite3.SQLITE_OK
            denied.append(f"action {action}")
            return sqlite3.SQLITE_DENY

        deadline = time.monotonic() + self.timeout_s
        conn = self.connect()
        try:
            conn.set_authorizer(authorizer)
            conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
            try:
                cur = conn.execute(sql)
                columns = [d[0] for d in cur.description or []]
                rows = cur.fetchmany(max_rows + 1)
            except sqlite3.DatabaseError as exc:
                if denied:
                    raise SQLSafetyError(
                        f"Access denied: {', '.join(sorted(set(denied)))}"
                    ) from exc
                if "interrupted" in str(exc).lower():
                    raise SQLSafetyError(f"Query exceeded the {self.timeout_s:g}s timeout") from exc
                raise SQLQueryError(str(exc)) from exc
        finally:
            conn.close()
        return columns, rows[:max_rows], len(rows) > max_rows, tables_read


def format_rows(columns: list[str], rows: list[tuple[Any, ...]], truncated: bool) -> str:
    """Markdown table of the result (with a truncation note)."""
    if not columns:
        return "(no columns)"
    if not rows:
        return "(0 rows)"
    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "---|" * len(columns)
    body = ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows]
    note = [f"(showing the first {len(rows)} rows; more rows exist)"] if truncated else []
    return "\n".join([head, sep, *body, *note])


@register("retriever", "sql")
class SQLRetriever:
    """Text-to-SQL retriever: returns query results as a context.

    Data comes from ``database`` (an existing SQLite file, opened read-only; no ingest
    needed) or from CSV documents ingested with the ``csv`` loader, stored in
    ``<index_dir>/sql/data.sqlite``.

    Args:
        tables: restrict the visible tables (default: all).
        table_groups: table-level access control, e.g. ``{salaries: [hr]}``.
        max_rows: row cap per query. sample_rows: example rows shown to the LLM.
        timeout_s: per-query time limit. max_attempts: tries, the later ones
            including the previous error.
    """

    def __init__(
        self,
        generator: Generator,
        index_dir: Path,
        database: str | None = None,
        tables: list[str] | None = None,
        table_groups: dict[str, list[str]] | None = None,
        schemas: list[str] | None = None,
        max_rows: int = 50,
        sample_rows: int = 3,
        timeout_s: float = 5.0,
        max_attempts: int = 2,
        k: int = 1,
    ) -> None:
        self.generator = generator
        self.dsn = database if database and database.startswith(POSTGRES_PREFIXES) else None
        self.database = Path(database).expanduser() if database and not self.dsn else None
        self.schemas = list(schemas) if schemas else None
        self.ingested_path = Path(index_dir) / "sql" / "data.sqlite"
        self.tables = list(tables) if tables else None
        self.table_groups = {t.lower(): list(g) for t, g in (table_groups or {}).items()}
        self.max_rows = max_rows
        self.sample_rows = sample_rows
        self.timeout_s = timeout_s
        self.max_attempts = max(1, max_attempts)
        self.k = k

    @property
    def requires_ingest(self) -> bool:
        """``False`` when querying an existing ``database`` (SQLite file or Postgres)."""
        return self.database is None and self.dsn is None

    def _db(self) -> SQLBackend:
        if self.dsn is not None:
            from ragpick.components.postgres import PostgresDatabase

            return PostgresDatabase(self.dsn, self.timeout_s, schemas=self.schemas)
        path = self.database or self.ingested_path
        if not path.is_file():
            raise ConfigError(
                f"No SQL database at {path}. Ingest CSV files (loader type: csv) or set "
                "retriever param `database` to an existing SQLite file."
            )
        return SQLDatabase(path, self.timeout_s)

    # -------------------------------------------------------------- ingest
    def index_documents(self, docs: list[Document]) -> None:
        """Load CSV documents into ``<index_dir>/sql/data.sqlite`` (one table each)."""
        csv_docs = [d for d in docs if d.metadata.get("format") == "csv"]
        if self.database is not None or self.dsn is not None:
            if csv_docs:
                logger.warning("retriever.database is set; ingested CSV files are ignored")
            return
        if not csv_docs:
            return
        self.ingested_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.ingested_path)) as conn:
            for doc in csv_docs:
                table = str(doc.metadata.get("table") or Path(doc.id).stem)
                count = load_csv_table(conn, table, doc.text)
                logger.info("Loaded %d rows into table %s", count, table)
            conn.commit()

    # -------------------------------------------------------------- query
    def visible_tables(self, db: SQLBackend, filters: dict[str, Any] | None) -> list[str]:
        """Tables the caller may query (``tables`` allow-list and ``table_groups`` ACL)."""
        groups = (filters or {}).get(ACL_KEY)
        caller = set(groups if isinstance(groups, list) else [groups] if groups else [])
        names = db.tables()
        if self.tables is not None:
            wanted = {t.lower() for t in self.tables}
            names = [n for n in names if n.lower() in wanted]
        return [
            n
            for n in names
            if n.lower() not in self.table_groups or caller & set(self.table_groups[n.lower()])
        ]

    def _llm(self) -> TextCompleter:
        if not isinstance(self.generator, TextCompleter) or getattr(
            self.generator, "is_fake", False
        ):
            raise ConfigError(
                "The sql retriever needs an LLM to write queries: set components.generator "
                "to openai, ollama, langchain or llamaindex (the fake generator cannot write SQL)."
            )
        return self.generator

    def retrieve(
        self, query: str, k: int | None = None, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Generate, run and return one SQL result as a context."""
        llm = self._llm()
        db = self._db()
        tables = self.visible_tables(db, filters)
        if not tables:
            set_attributes(tables=0)
            return []
        schema = db.schema(tables, self.sample_rows)
        system = SQL_SYSTEM.format(max_rows=self.max_rows, dialect=db.dialect)
        prompt = SQL_PROMPT.format(schema=schema, question=query)
        last_error = ""
        sql = ""
        for attempt in range(1, self.max_attempts + 1):
            with span("generate_sql", attempt=attempt):
                sql = extract_sql(llm.complete(system, prompt).text)
                set_attributes(sql=sql)
            with span("execute_sql") as s:
                try:
                    columns, rows, truncated, read = db.run(sql, set(tables), self.max_rows)
                except SQLSafetyError as exc:
                    last_error = str(exc)
                    if s is not None:
                        s.error = last_error
                    if "Access denied" in last_error or "timeout" in last_error:
                        break  # don't let the model probe for forbidden tables
                except SQLQueryError as exc:
                    last_error = str(exc)
                    if s is not None:
                        s.error = last_error
                else:
                    set_attributes(rows=len(rows), truncated=truncated)
                    return [self._context(sql, columns, rows, truncated, read)]
            prompt = SQL_RETRY_PROMPT.format(
                schema=schema, question=query, sql=sql, error=last_error
            )
        logger.warning("SQL generation failed: %s", last_error)
        text = f"The database query failed: {last_error}\nSQL: {sql}"
        chunk = Chunk(
            id="sql#error", doc_id="sql", text=text, metadata={"sql": sql, "error": last_error}
        )
        return [RetrievedChunk(chunk=chunk, score=0.0, source="sql")]

    def _context(
        self,
        sql: str,
        columns: list[str],
        rows: list[tuple[Any, ...]],
        truncated: bool,
        read: set[str],
    ) -> RetrievedChunk:
        digest = hashlib.sha256(sql.encode("utf-8")).hexdigest()[:10]
        used = ",".join(sorted(read)) or "sql"
        text = (
            f"Result of the SQL query below ({len(rows)} rows).\n"
            f"SQL: {sql}\n\n{format_rows(columns, rows, truncated)}"
        )
        meta = {"sql": sql, "row_count": len(rows), "truncated": truncated, "tables": sorted(read)}
        chunk = Chunk(id=f"sql:{used}#{digest}", doc_id=f"sql:{used}", text=text, metadata=meta)
        return RetrievedChunk(chunk=chunk, score=1.0, source="sql")
