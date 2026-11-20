"""Query routing between pipelines."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from conftest import EXAMPLES
from ragpick import Pipeline, Route, Router, load_engine
from ragpick.cli import app
from ragpick.components.generators import IDK, LLMGenerator
from ragpick.components.sql import load_csv_table
from ragpick.config import RouterConfig
from ragpick.errors import ConfigError
from ragpick.eval import evaluate_configs
from ragpick.models import Completion

DATA = EXAMPLES / "data"
TABLES = EXAMPLES / "tables"


class ScriptedLLM(LLMGenerator):
    """Routes, writes SQL and answers from scripted replies keyed by the system prompt."""

    def __init__(self, route: str = "", sql: str = "SELECT COUNT(*) AS n FROM orders") -> None:
        self.route = route
        self.sql = sql
        self.calls: list[str] = []

    def complete(self, system: str, user: str) -> Completion:
        if system.startswith("You route questions"):
            self.calls.append("route")
            return Completion(text=self.route)
        if system.startswith("You translate questions"):
            self.calls.append("sql")
            return Completion(text=f"```sql\n{self.sql}\n```")
        self.calls.append("answer")
        return Completion(text="From the data [1].")


@pytest.fixture
def docs(tmp_path: Path) -> Pipeline:
    p = Pipeline.from_dict({"blueprint": "hybrid", "index_dir": str(tmp_path / "docs")})
    p.ingest(DATA)
    return p


@pytest.fixture
def sql_db(tmp_path: Path) -> Path:
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    for f in TABLES.glob("*.csv"):
        load_csv_table(conn, f.stem, f.read_text(encoding="utf-8"))
    conn.commit()
    conn.close()
    return path


def sql_pipeline(tmp_path: Path, db: Path, llm: Any) -> Pipeline:
    return Pipeline.from_dict(
        {
            "blueprint": "sql",
            "index_dir": str(tmp_path / "sql"),
            "components": {"retriever": {"params": {"database": str(db)}}},
        },
        components={"generator": llm},
    )


def make_router(docs: Pipeline, analytics: Pipeline, **kw: Any) -> Router:
    routes = [
        Route("docs", docs, "Product manuals, troubleshooting, error codes and policies"),
        Route(
            "analytics",
            analytics,
            "Orders, sales, prices and quantities",
            keywords=["how many", "total", "revenue", "orders"],
            groups=kw.pop("analytics_groups", None),
        ),
    ]
    return Router(routes, **kw)


def test_keyword_routing(docs: Pipeline, sql_db: Path, tmp_path: Path) -> None:
    llm = ScriptedLLM()
    router = make_router(docs, sql_pipeline(tmp_path, sql_db, llm), strategy="keywords")
    assert router.strategy == "keywords"
    route, reason = router.choose("How many orders were placed?")
    assert route is not None
    assert route.name == "analytics"
    assert reason.startswith("keywords")
    assert router.choose("What does error E-1042 mean?")[0].name == "docs"  # type: ignore[union-attr]
    assert router.choose("zzz qqq")[1] == "default"
    answer = router.query("How many orders were placed?", trace=True)
    assert answer.route == "analytics"
    assert answer.contexts[0].source == "sql"
    assert answer.trace is not None
    assert answer.trace.find("route").attributes["route"] == "analytics"  # type: ignore[union-attr]
    assert answer.trace.find("pipeline") is not None  # the route's pipeline joined the trace
    assert answer.trace.find("execute_sql") is not None
    assert "route" not in llm.calls  # keywords strategy never asks the LLM
    doc_answer = router.query("What does error E-1042 mean?")
    assert doc_answer.route == "docs"
    assert doc_answer.citations[0].startswith("error-codes.md")


def test_llm_routing_with_keyword_fallback(docs: Pipeline, sql_db: Path, tmp_path: Path) -> None:
    llm = ScriptedLLM(route='{"route": "analytics"}')
    router = make_router(docs, sql_pipeline(tmp_path, sql_db, llm))
    assert router.strategy == "llm"  # auto picked the analytics route's real LLM
    route, reason = router.choose("Anything at all?")
    assert (route.name, reason) == ("analytics", "llm")  # type: ignore[union-attr]
    llm.route = "the docs one, I think"
    assert router.choose("What does error E-1042 mean?")[0].name == "docs"  # type: ignore[union-attr]
    llm.route = '{"route": "nonexistent"}'
    assert router.choose("total revenue")[0].name == "analytics"  # type: ignore[union-attr]


def test_fallback_to_default_route(docs: Pipeline, sql_db: Path, tmp_path: Path) -> None:
    broken = ScriptedLLM(sql="SELECT missing_column FROM orders")
    router = make_router(docs, sql_pipeline(tmp_path, sql_db, broken), strategy="keywords")
    answer = router.query("How many orders mention error E-1042?", trace=True)
    assert answer.route == "docs"  # SQL failed twice → docs answered
    fallback = answer.trace.find("fallback")  # type: ignore[union-attr]
    assert fallback is not None
    assert fallback.attributes == {"from_route": "analytics", "to_route": "docs", "answered": True}
    strict = make_router(
        docs, sql_pipeline(tmp_path, sql_db, broken), strategy="keywords", fallback=False
    )
    assert strict.query("How many orders mention error E-1042?").route == "analytics"


def test_fallback_keeps_first_answer_when_default_also_fails(
    docs: Pipeline, sql_db: Path, tmp_path: Path
) -> None:
    broken = ScriptedLLM(sql="SELECT nope FROM orders")
    router = make_router(docs, sql_pipeline(tmp_path, sql_db, broken), strategy="keywords")
    answer = router.query("how many zzzqqq")
    assert answer.route == "analytics"
    assert answer.contexts[0].chunk.id == "sql#error"


def test_route_groups(docs: Pipeline, sql_db: Path, tmp_path: Path) -> None:
    router = make_router(
        docs,
        sql_pipeline(tmp_path, sql_db, ScriptedLLM()),
        strategy="keywords",
        analytics_groups=["staff"],
    )
    assert router.choose("How many orders?")[0].name == "docs"  # type: ignore[union-attr]
    assert router.choose("How many orders?", groups=["staff"])[0].name == "analytics"  # type: ignore[union-attr]
    only = Router([Route("a", docs, groups=["hr"])])
    assert only.choose("x") == (None, "no route is available to the caller's groups")
    assert only.query("x").text == IDK
    assert only.retrieve("x") == []
    single = Router([Route("a", docs)])
    assert single.choose("x")[1] == "only route available"
    assert single.retrieve("What does E-1042 mean?")


def test_router_validation(docs: Pipeline) -> None:
    with pytest.raises(ConfigError, match="at least one route"):
        Router([])
    with pytest.raises(ConfigError, match="strategy"):
        Router([Route("a", docs)], strategy="vibes")
    with pytest.raises(ConfigError, match="Unknown default"):
        Router([Route("a", docs)], default="b")
    with pytest.raises(ConfigError, match="needs an LLM"):
        Router([Route("a", docs)], strategy="llm")  # only a fake generator available


def write_router(tmp_path: Path, extra: dict[str, Any] | None = None) -> Path:
    (tmp_path / "naive.yaml").write_text(
        yaml.safe_dump(
            {"name": "naive", "blueprint": "naive", "data": str(DATA), "index_dir": "idx-naive"}
        ),
        encoding="utf-8",
    )
    (tmp_path / "hybrid.yaml").write_text(
        yaml.safe_dump(
            {"name": "hybrid", "blueprint": "hybrid", "data": str(DATA), "index_dir": "idx-hybrid"}
        ),
        encoding="utf-8",
    )
    raw: dict[str, Any] = {
        "name": "assistant",
        "router": {
            "strategy": "keywords",
            "default": "general",
            "routes": [
                {"name": "general", "config": "naive.yaml", "description": "general questions"},
                {"name": "codes", "config": "hybrid.yaml", "keywords": ["error", "code"]},
            ],
        },
        **(extra or {}),
    }
    path = tmp_path / "router.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_router_from_config_ingest_and_eval(tmp_path: Path) -> None:
    path = write_router(
        tmp_path, {"tracing": {"exporters": [{"type": "jsonl", "params": {"path": "t.jsonl"}}]}}
    )
    engine = load_engine(path)
    assert isinstance(engine, Router)
    assert engine.name == "assistant"
    assert engine.blueprint == "router(general=naive,codes=hybrid)"
    assert engine.top_k == 5
    assert engine.model_name == "fake"
    assert not engine.ready()
    stats = engine.ingest()
    assert set(stats) == {"general", "codes"}
    assert engine.ready()
    with pytest.raises(ConfigError, match="without a PATH"):
        engine.ingest(DATA)
    answer = engine.query("What does error E-1042 mean?", trace=True)
    assert answer.route == "codes"
    assert (tmp_path / "t.jsonl").read_text(encoding="utf-8").count("\n") == 1
    reports = evaluate_configs([path], EXAMPLES / "eval.jsonl")
    assert reports[0].label == "assistant"
    assert reports[0].blueprint == "router(general=naive,codes=hybrid)"
    assert {r.route for r in reports[0].results} <= {"general", "codes"}
    with pytest.raises(ConfigError, match="omit --data"):
        evaluate_configs([path], EXAMPLES / "eval.jsonl", data=DATA)


def test_router_config_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        RouterConfig.model_validate(
            {"router": {"routes": [{"name": "a", "config": "x"}, {"name": "a", "config": "y"}]}}
        )
    with pytest.raises(ValueError, match="default route"):
        RouterConfig.model_validate(
            {"router": {"default": "zz", "routes": [{"name": "a", "config": "x"}]}}
        )
    path = write_router(tmp_path)
    nested = tmp_path / "nested.yaml"
    nested.write_text(
        yaml.safe_dump({"router": {"routes": [{"name": "inner", "config": "router.yaml"}]}}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="nested routers"):
        load_engine(nested)
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"router": {"routes": []}}), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"router\.routes"):
        load_engine(bad)
    no_data = tmp_path / "nodata.yaml"
    (tmp_path / "plain.yaml").write_text("blueprint: naive\nindex_dir: x\n", encoding="utf-8")
    no_data.write_text(
        yaml.safe_dump({"router": {"routes": [{"name": "p", "config": "plain.yaml"}]}}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="no `data`"):
        load_engine(no_data).ingest()  # type: ignore[call-arg]
    with_llm = tmp_path / "llm.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["router"]["llm"] = {"type": "fake", "params": {"response": '{"route": "codes"}'}}
    raw["router"]["strategy"] = "llm"
    with_llm.write_text(yaml.safe_dump(raw), encoding="utf-8")
    llm_router = load_engine(with_llm)
    assert isinstance(llm_router, Router)
    assert llm_router.choose("anything")[0].name == "codes"  # type: ignore[union-attr]


def test_router_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_router(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    valid = runner.invoke(app, ["validate", "-c", str(path)])
    assert valid.exit_code == 0, valid.output
    assert "valid router (2 routes)" in valid.output
    ingest = runner.invoke(app, ["ingest", "-c", str(path)])
    assert ingest.exit_code == 0, ingest.output
    assert "general:" in ingest.output
    assert "codes:" in ingest.output
    with_path = runner.invoke(app, ["ingest", "examples", "-c", str(path)])
    assert with_path.exit_code == 1
    query = runner.invoke(
        app, ["query", "What does error E-1042 mean?", "-c", str(path), "--trace"]
    )
    assert query.exit_code == 0, query.output
    assert "route: codes" in query.output
    assert "route" in query.output
