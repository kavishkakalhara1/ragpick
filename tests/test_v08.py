"""v0.8: eval thresholds, conversations, Chroma and LanceDB stores."""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from conftest import EXAMPLES, make_chunks
from ragpick import Pipeline, Route, Router
from ragpick.cli import app as cli_app
from ragpick.components.chroma import ChromaStore, chroma_where, filter_fields
from ragpick.components.generators import LLMGenerator
from ragpick.components.lancedb import LanceDBStore, lance_where, sql_string, tags_for
from ragpick.config import ApiKeySpec, ConversationConfig, ServeConfig
from ragpick.conversation import (
    ChatTurn,
    ConversationError,
    ConversationManager,
    MemorySessionStore,
    QuestionRewriter,
    SqliteSessionStore,
    is_followup,
    owner_key,
    validate_session_id,
)
from ragpick.errors import ConfigError, MissingDependencyError
from ragpick.eval import EvalReport
from ragpick.eval.runner import check_thresholds, parse_thresholds
from ragpick.filters import acl_filter, matches_filters
from ragpick.models import Chunk, Completion
from ragpick.selector import recommend
from ragpick.server import App

DATA = EXAMPLES / "data"


# =================================================================== thresholds
def report(label: str, **metrics: float | None) -> EvalReport:
    return EvalReport(
        label=label, blueprint=None, k=3, model="fake", num_examples=1, metrics=metrics, results=[]
    )


def test_parse_and_check_thresholds() -> None:
    assert parse_thresholds(["hit_rate@k=0.8", "latency_p95_ms=1500"]) == {
        "hit_rate@k": 0.8,
        "latency_p95_ms": 1500.0,
    }
    with pytest.raises(ConfigError, match="METRIC=VALUE"):
        parse_thresholds(["accuracy=0.9"])
    with pytest.raises(ConfigError, match="METRIC=VALUE"):
        parse_thresholds(["mrr"])
    with pytest.raises(ConfigError, match="value"):
        parse_thresholds(["mrr=high"])
    reports = [
        report("a", mrr=0.9, latency_p95_ms=100, faithfulness=None),
        report("b", mrr=0.4, latency_p95_ms=900),
    ]
    assert check_thresholds(reports, {"mrr": 0.5}, {"latency_p95_ms": 500}) == [
        "b: mrr = 0.400 < 0.5",
        "b: latency_p95_ms = 900.000 > 500",
    ]
    assert check_thresholds(reports[:1], {"mrr": 0.5}) == []
    missing = check_thresholds(reports[:1], {"faithfulness": 0.7}, {"total_tokens": 10})
    assert missing == [
        "a: faithfulness is not available (required >= 0.7)",
        "a: total_tokens is not available (required <= 10)",
    ]
    with pytest.raises(ConfigError, match="Unknown metric"):
        check_thresholds(reports, {"vibes": 1})


def test_eval_cli_thresholds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    args = [
        "eval",
        str(EXAMPLES / "eval.jsonl"),
        "-c",
        str(EXAMPLES / "hybrid.yaml"),
        "--data",
        str(DATA),
    ]
    ok = runner.invoke(
        cli_app, [*args, "--fail-under", "hit_rate@k=0.5", "--fail-over", "latency_p95_ms=60000"]
    )
    assert ok.exit_code == 0, ok.output
    assert "All 2 thresholds passed" in ok.output
    bad = runner.invoke(cli_app, [*args, "--fail-under", "mrr=0.99"])
    assert bad.exit_code == 1
    assert "mrr = " in bad.output
    assert "< 0.99" in bad.output
    no_judge = runner.invoke(cli_app, [*args, "--fail-under", "faithfulness=0.5"])
    assert no_judge.exit_code == 1
    assert "not available" in no_judge.output
    cfg = tmp_path / "gated.yaml"
    cfg.write_text(
        f"blueprint: hybrid\ndata: {DATA}\neval:\n  fail_under: {{hit_rate@k: 1.1}}\n",
        encoding="utf-8",
    )
    gated = runner.invoke(cli_app, ["eval", str(EXAMPLES / "eval.jsonl"), "-c", str(cfg)])
    assert gated.exit_code == 1
    assert "hit_rate@k" in gated.output
    overridden = runner.invoke(
        cli_app,
        ["eval", str(EXAMPLES / "eval.jsonl"), "-c", str(cfg), "--fail-under", "hit_rate@k=0.1"],
    )
    assert overridden.exit_code == 0, overridden.output  # CLI wins over config
    invalid = runner.invoke(cli_app, [*args, "--fail-under", "accuracy=1"])
    assert invalid.exit_code == 1


# =================================================================== conversations
def turn(q: str, standalone: str | None = None) -> ChatTurn:
    return ChatTurn(question=q, standalone=standalone or q, answer=f"answer to {q}")


@pytest.mark.parametrize(
    "make", [lambda p: MemorySessionStore(), lambda p: SqliteSessionStore(p / "s.sqlite")]
)
def test_session_stores(tmp_path: Path, make: Any) -> None:
    store = make(tmp_path)
    assert store.get("session-1", "alice") == []
    store.append("session-1", "alice", turn("q1"), max_turns=2)
    store.append("session-1", "alice", turn("q2"), max_turns=2)
    store.append("session-1", "alice", turn("q3"), max_turns=2)
    assert [t.question for t in store.get("session-1", "alice")] == ["q2", "q3"]
    with pytest.raises(ConversationError, match="Unknown session"):
        store.get("session-1", "mallory")  # someone else's session
    with pytest.raises(ConversationError, match="Unknown session"):
        store.append("session-1", "mallory", turn("x"), max_turns=5)
    with pytest.raises(ConversationError, match="Unknown session"):
        store.delete("session-1", "mallory")
    assert store.delete("session-1", "alice")
    assert not store.delete("session-1", "alice")
    assert store.get("session-1", "alice") == []


def test_session_ttl_and_lru(tmp_path: Path) -> None:
    for store in (
        MemorySessionStore(ttl_seconds=0.01),
        SqliteSessionStore(tmp_path / "t.sqlite", ttl_seconds=0.01),
    ):
        store.append("session-1", "a", turn("q"), max_turns=5)
        time.sleep(0.03)
        assert store.get("session-1", "a") == []
    lru = MemorySessionStore(max_sessions=2)
    for sid in ("session-a", "session-b", "session-c"):
        lru.append(sid, "o", turn(sid), max_turns=5)
    assert lru.get("session-a", "o") == []
    assert lru.get("session-c", "o")
    persisted = SqliteSessionStore(tmp_path / "p.sqlite")
    persisted.append("session-1", "o", turn("kept"), max_turns=5)
    assert SqliteSessionStore(tmp_path / "p.sqlite").get("session-1", "o")[0].question == "kept"


def test_session_ids_and_owner() -> None:
    assert validate_session_id("abcdefgh") == "abcdefgh"
    for bad in ("short", "x" * 65, "../../etc/passwd", "has space!"):
        with pytest.raises(ConversationError, match="Invalid session id"):
            validate_session_id(bad)
    assert owner_key(None, None) == "local|-"
    assert owner_key("portal", "acme") == "portal|acme"


def test_followup_detection_and_heuristic_rewrite() -> None:
    assert is_followup("and can I cancel it?")
    assert is_followup("what about the LM-200?")
    assert is_followup("Why?")
    assert not is_followup("How do I reset my password on the Lumen app today?")
    rewriter = QuestionRewriter(None, "heuristic")
    history = [turn("How much does Lumen Plus cost?")]
    assert rewriter.rewrite("and can I cancel it?", history) == (
        "and can I cancel it? (regarding: How much does Lumen Plus cost)"
    )
    standalone = "What does error E-1042 mean on the thermostat display?"
    assert rewriter.rewrite(standalone, history) == standalone
    assert rewriter.rewrite("and it?", []) == "and it?"  # no history yet
    assert QuestionRewriter(None, "off").rewrite("and it?", history) == "and it?"


class Rewriting(LLMGenerator):
    def __init__(self, reply: str = "Can I cancel my Lumen Plus subscription?") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, system: str, user: str) -> Completion:
        if system.startswith("You rewrite the user's latest message"):
            self.prompts.append(user)
            return Completion(text=self.reply)
        return Completion(text="Yes, under Account > Subscription [1].")


def test_llm_rewrite() -> None:
    llm = Rewriting()
    rewriter = QuestionRewriter(llm, "auto")
    history = [turn("How much does Lumen Plus cost?")]
    assert (
        rewriter.rewrite("and can I cancel it?", history)
        == "Can I cancel my Lumen Plus subscription?"
    )
    assert "User: How much does Lumen Plus cost?" in llm.prompts[0]
    blank = QuestionRewriter(Rewriting(reply="  "), "auto")
    assert blank.rewrite("and can I cancel it?", history).startswith(
        "and can I cancel it? (regarding:"
    )
    with pytest.raises(ConversationError, match="rewrite: llm"):
        ConversationManager(ConversationConfig(rewrite="llm"), MemorySessionStore(), None)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> Pipeline:
    p = Pipeline.from_dict(
        {"blueprint": "hybrid", "index_dir": str(tmp_path_factory.mktemp("c") / "i")}
    )
    p.ingest(DATA)
    return p


def test_pipeline_chat(pipeline: Pipeline) -> None:
    first = pipeline.chat("How much does Lumen Plus cost?", trace=True)
    assert first.session_id is not None
    assert first.standalone_question == "How much does Lumen Plus cost?"
    assert first.citations[0].startswith("billing-and-subscriptions.md")
    assert first.trace is not None
    second = pipeline.chat("and can I cancel it?", session_id=first.session_id)
    assert second.session_id == first.session_id
    assert "Lumen Plus" in (second.standalone_question or "")
    assert second.citations[0].startswith("billing-and-subscriptions.md")
    history = pipeline.conversations.store.get(first.session_id, "local|-")
    assert [t.question for t in history] == [
        "How much does Lumen Plus cost?",
        "and can I cancel it?",
    ]
    with pytest.raises(ConversationError, match="Unknown session"):
        pipeline.chat("hi there friend", session_id=first.session_id, owner="someone-else")
    with pytest.raises(ConversationError, match="Empty"):
        pipeline.chat("   ")
    events = list(pipeline.chat_stream("what about energy history?", session_id=first.session_id))
    final = events[-1].answer
    assert final is not None
    assert final.session_id == first.session_id
    assert len(pipeline.conversations.store.get(first.session_id, "local|-")) == 3


def test_chat_with_llm_rewrite_and_sqlite_sessions(tmp_path: Path) -> None:
    llm = Rewriting()
    p = Pipeline.from_dict(
        {"blueprint": "conversational", "index_dir": str(tmp_path / "i")},
        components={"generator": llm},
    )
    assert p.config.conversation.store == "sqlite"
    p.ingest(DATA)
    first = p.chat("How much does Lumen Plus cost?")
    second = p.chat("and can I cancel it?", session_id=first.session_id)
    assert second.standalone_question == "Can I cancel my Lumen Plus subscription?"
    assert (tmp_path / "i" / "sessions.sqlite").is_file()
    restarted = Pipeline.from_dict(
        {"blueprint": "conversational", "index_dir": str(tmp_path / "i")},
        components={"generator": llm},
    )
    assert len(restarted.conversations.store.get(first.session_id or "", "local|-")) == 2


def test_router_chat(pipeline: Pipeline, tmp_path: Path) -> None:
    router = Router([Route("docs", pipeline)], sessions_dir=tmp_path)
    a = router.chat("How much does Lumen Plus cost?")
    b = router.chat("and can I cancel it?", session_id=a.session_id)
    assert b.route == "docs"
    assert "regarding" in (b.standalone_question or "")
    streamed = list(router.chat_stream("and energy history?", session_id=a.session_id))
    assert streamed[-1].answer.session_id == a.session_id  # type: ignore[union-attr]


def test_chat_over_http_scopes_sessions_to_keys(pipeline: Pipeline) -> None:
    keys = [
        ApiKeySpec(key="alice-key-0123456789", name="alice"),
        ApiKeySpec(key="mallory-key-012345678", name="mallory"),
    ]
    app = App(pipeline, ServeConfig(api_keys=keys))

    def call(method: str, path: str, key: str, body: dict[str, Any] | None = None) -> Any:
        return app.handle(
            method, path, {"authorization": f"Bearer {key}"}, json.dumps(body or {}).encode()
        )

    first = call(
        "POST", "/chat", "alice-key-0123456789", {"message": "How much does Lumen Plus cost?"}
    )
    assert first.status == 200
    sid = first.body["session_id"]
    second = call(
        "POST",
        "/chat",
        "alice-key-0123456789",
        {"message": "and can I cancel it?", "session_id": sid},
    )
    assert "regarding" in second.body["standalone_question"]
    stolen = call(
        "POST", "/chat", "mallory-key-012345678", {"message": "and what else?", "session_id": sid}
    )
    assert stolen.status == 404  # another key can't continue (or read) Alice's session
    assert call("DELETE", f"/sessions/{sid}", "mallory-key-012345678").status == 404
    assert (
        call(
            "POST", "/chat", "alice-key-0123456789", {"message": "hi", "session_id": "../x"}
        ).status
        == 400
    )
    assert call("GET", f"/sessions/{sid}", "alice-key-0123456789").status == 405
    assert call("DELETE", f"/sessions/{sid}", "alice-key-0123456789").status == 200
    assert call("DELETE", f"/sessions/{sid}", "alice-key-0123456789").status == 404
    streamed = call(
        "POST",
        "/chat",
        "alice-key-0123456789",
        {"message": "How much does Lumen Plus cost?", "stream": True},
    )
    events = [b for b in b"".join(streamed.stream).decode().split("\n\n") if b]
    final = json.loads(events[-1].split("data: ", 1)[1])
    assert final["session_id"]
    assert final["standalone_question"] == "How much does Lumen Plus cost?"
    assert call("POST", "/chat", "alice-key-0123456789", {"query": "wrong field"}).status == 422


def test_chat_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("rag.yaml").write_text("blueprint: hybrid\n", encoding="utf-8")
    runner = CliRunner()
    assert runner.invoke(cli_app, ["ingest", str(DATA)]).exit_code == 0
    script = (
        "How much does Lumen Plus cost?\n\nand can I cancel it?\n/reset\n"
        "What does error E-1042 mean?\n/exit\n"
    )
    result = runner.invoke(cli_app, ["chat"], input=script)
    assert result.exit_code == 0, result.output
    assert "↳ and can I cancel it? (regarding: How much does Lumen Plus cost)" in result.output
    assert "New session." in result.output
    assert "Citations:" in result.output
    assert "Session:" in result.output
    eof = runner.invoke(cli_app, ["chat"], input="How much does Lumen Plus cost?\n")
    assert eof.exit_code == 0


def test_selector_conversational() -> None:
    r = recommend({"conversational": True}, installed=lambda m: True)
    assert r.blueprint == "conversational"
    assert (
        recommend(
            {"conversational": True, "keyword_heavy": True}, installed=lambda m: True
        ).blueprint
        == "hybrid"
    )


# =================================================================== Chroma
def _chroma_matches(where: dict[str, Any] | None, fields: dict[str, Any]) -> bool:
    """Reference evaluator for the Chroma `where` subset ragpick emits."""
    if where is None:
        return True
    if "$and" in where:
        return all(_chroma_matches(w, fields) for w in where["$and"])
    if "$or" in where:
        return any(_chroma_matches(w, fields) for w in where["$or"])
    [(key, cond)] = where.items()
    if key not in fields:
        return False
    if "$eq" in cond:
        return bool(fields[key] == cond["$eq"])
    return fields[key] in cond["$in"]


METAS = [
    {},
    {"lang": "en"},
    {"lang": "de", "year": 2024},
    {"tags": ["a", "b"]},
    {"allowed_groups": ["hr"]},
    {"allowed_groups": ["finance", "admin"], "lang": "en"},
    {"allowed_groups": []},
    {"allowed_groups": None, "tags": ["b"]},
]
FILTERS = [
    None,
    {"lang": "en"},
    {"lang": ["de", "en"]},
    {"year": 2024},
    {"tags": "b"},
    {"tags": ["z"]},
    acl_filter(None),
    acl_filter(["hr"]),
    acl_filter(["admin"], {"lang": "en"}),
    acl_filter(["hr", "finance"]),
]


@pytest.mark.parametrize(("meta", "flt"), list(itertools.product(METAS, FILTERS)))
def test_chroma_translation_matches_ragpick_semantics(
    meta: dict[str, Any], flt: dict[str, Any] | None
) -> None:
    assert _chroma_matches(chroma_where(flt), filter_fields(meta)) == matches_filters(meta, flt)


class FakeCollection:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.last_query: dict[str, Any] = {}

    def upsert(
        self, ids: list[str], embeddings: list[Any], documents: list[str], metadatas: list[Any]
    ) -> None:
        for i, e, d, m in zip(ids, embeddings, documents, metadatas, strict=True):
            self.rows[i] = {"embedding": e, "document": d, "metadata": m}

    def count(self) -> int:
        return len(self.rows)

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.last_query = kwargs
        q = kwargs["query_embeddings"][0]
        hits = []
        for cid, row in self.rows.items():
            if not _chroma_matches(kwargs.get("where"), row["metadata"]):
                continue
            sim = sum(a * b for a, b in zip(q, row["embedding"], strict=True))
            hits.append((1 - sim, cid, row))
        hits.sort()
        hits = hits[: kwargs["n_results"]]
        return {
            "ids": [[h[1] for h in hits]],
            "documents": [[h[2]["document"] for h in hits]],
            "metadatas": [[h[2]["metadata"] for h in hits]],
            "distances": [[h[0] for h in hits]],
        }

    def delete(self, where: dict[str, Any]) -> None:
        drop = set(where["doc_id"]["$in"])
        self.rows = {k: v for k, v in self.rows.items() if v["metadata"]["doc_id"] not in drop}


class FakeChromaClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}
        self.created: dict[str, Any] = {}

    def get_or_create_collection(self, name: str, metadata: dict[str, Any]) -> FakeCollection:
        self.created[name] = metadata
        return self.collections.setdefault(name, FakeCollection())


def test_chroma_store_with_fake_client(acl_chunks: list[Any]) -> None:
    client = FakeChromaClient()
    store = ChromaStore(collection="kb_store", client=client)
    assert client.created["kb_store"] == {"hnsw:space": "cosine"}
    assert store.search([0.0] * 64, 3) == []
    store.add([])
    store.add(acl_chunks)
    q = acl_chunks[0].embedding
    public = [r.chunk.id for r in store.search(q, 10, acl_filter(None))]
    assert public == ["public#0"]
    hr = sorted(r.chunk.id for r in store.search(q, 10, acl_filter(["hr"])))
    assert hr == ["hr#0", "public#0"]
    top = store.search(q, 1)[0]
    assert top.chunk.id == "public#0"
    assert top.chunk.metadata == {}
    assert top.score > 0.99
    restricted = store.search(acl_chunks[1].embedding, 1, acl_filter(["hr"]))[0]
    assert restricted.chunk.metadata == {"allowed_groups": ["hr"]}  # original metadata restored
    store.delete(["hr"])
    assert "hr#0" not in [r.chunk.id for r in store.search(q, 10)]
    store.delete([])
    store.persist("x")
    store.load("x")
    with pytest.raises(ValueError, match="no embedding"):
        store.add([Chunk(id="x", doc_id="x", text="t")])


def test_chroma_client_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    made: list[tuple[str, dict[str, Any]]] = []

    class Module:
        @staticmethod
        def HttpClient(**kw: Any) -> FakeChromaClient:  # noqa: N802 - mirrors chromadb
            made.append(("http", kw))
            return FakeChromaClient()

        @staticmethod
        def PersistentClient(**kw: Any) -> FakeChromaClient:  # noqa: N802
            made.append(("persistent", kw))
            return FakeChromaClient()

        @staticmethod
        def EphemeralClient() -> FakeChromaClient:  # noqa: N802
            made.append(("ephemeral", {}))
            return FakeChromaClient()

    monkeypatch.setitem(sys.modules, "chromadb", Module)
    ChromaStore(host="chroma.local", port=9000)
    ChromaStore(path=str(tmp_path))
    ChromaStore()
    assert [m[0] for m in made] == ["http", "persistent", "ephemeral"]
    assert made[0][1] == {"host": "chroma.local", "port": 9000}
    with pytest.raises(ConfigError, match="collection name"):
        ChromaStore(collection="t")
    monkeypatch.setitem(sys.modules, "chromadb", None)
    with pytest.raises(MissingDependencyError, match="pip install chromadb"):
        ChromaStore()


# =================================================================== LanceDB
def test_lance_helpers() -> None:
    assert tags_for({"lang": "en", "tags": ["a", "b"], "n": 3, "none": None}) == [
        "lang=en",
        "tags=a",
        "tags=b",
        "n=3",
    ]
    assert sql_string("it's") == "'it''s'"
    with pytest.raises(ConfigError, match="NUL"):
        sql_string("a\x00b")
    assert lance_where(None) is None
    where = lance_where(acl_filter(["hr"], {"lang": ["en", "de"]}))
    assert where == (
        "(array_has_any(tags, ['lang=en', 'lang=de'])) AND "
        "(acl_restricted = false OR array_has_any(tags, ['allowed_groups=hr']))"
    )
    assert lance_where(acl_filter([])) == "(acl_restricted = false OR FALSE)"
    injection = lance_where({"lang": "x') OR TRUE --"})
    assert injection == "(array_has_any(tags, ['lang=x'') OR TRUE --']))"


class FakeQuery:
    def __init__(self, table: FakeLanceTable, vector: list[float]) -> None:
        self.table = table
        self.vector = vector
        self.calls: dict[str, Any] = {}

    def metric(self, name: str) -> FakeQuery:
        self.calls["metric"] = name
        return self

    def limit(self, k: int) -> FakeQuery:
        self.calls["limit"] = k
        return self

    def where(self, predicate: str, prefilter: bool) -> FakeQuery:
        self.calls["where"] = (predicate, prefilter)
        return self

    def to_list(self) -> list[dict[str, Any]]:
        self.table.queries.append(self.calls)
        rows = sorted(self.table.rows.values(), key=lambda r: r["id"])
        return [{**r, "_distance": 0.25} for r in rows][: self.calls["limit"]]


class FakeLanceTable:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = {r["id"]: r for r in rows}
        self.queries: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.merged: list[Any] = []

    def search(self, vector: list[float]) -> FakeQuery:
        return FakeQuery(self, vector)

    def merge_insert(self, key: str) -> Any:
        table = self

        class Builder:
            def when_matched_update_all(self) -> Builder:
                return self

            def when_not_matched_insert_all(self) -> Builder:
                return self

            def execute(self, rows: list[dict[str, Any]]) -> None:
                table.merged.append(key)
                for r in rows:
                    table.rows[r["id"]] = r

        return Builder()

    def delete(self, predicate: str) -> None:
        self.deleted.append(predicate)


class FakeLanceDB:
    def __init__(self) -> None:
        self.tables: dict[str, FakeLanceTable] = {}

    def table_names(self) -> list[str]:
        return list(self.tables)

    def open_table(self, name: str) -> FakeLanceTable:
        return self.tables[name]

    def create_table(self, name: str, data: list[dict[str, Any]]) -> FakeLanceTable:
        self.tables[name] = FakeLanceTable(data)
        return self.tables[name]


def test_lancedb_store_with_fake_db(tmp_path: Path) -> None:
    db = FakeLanceDB()
    store = LanceDBStore(tmp_path, table="kb", db=db)
    assert store.search([0.1], 3) == []
    store.add([])
    store.delete(["a"])  # no table yet
    chunks = make_chunks(
        {"a#0": "alpha", "b#0": "beta"}, {"a#0": {"allowed_groups": ["hr"], "lang": "en"}}
    )
    store.add(chunks)
    row = db.tables["kb"].rows["a#0"]
    assert row["tags"] == ["allowed_groups=hr", "lang=en"]
    assert row["acl_restricted"] is True
    assert db.tables["kb"].rows["b#0"]["acl_restricted"] is False
    store.add(make_chunks({"a#0": "alpha v2"}))
    assert db.tables["kb"].merged == ["id"]
    results = store.search([0.1] * 64, 1, acl_filter(["hr"]))
    assert results[0].score == 0.75
    assert results[0].chunk.metadata == {}
    query = db.tables["kb"].queries[-1]
    assert query["metric"] == "cosine"
    assert query["where"][1] is True  # prefilter
    assert "allowed_groups=hr" in query["where"][0]
    store.delete(["a", "o'brien"])
    assert db.tables["kb"].deleted[-1] == "doc_id IN ('a', 'o''brien')"
    reopened = LanceDBStore(tmp_path, table="kb", db=db)
    assert reopened.search([0.1] * 64, 5)
    store.persist("x")
    store.load("x")
    with pytest.raises(ValueError, match="no embedding"):
        store.add([Chunk(id="x", doc_id="x", text="t")])


def test_lancedb_connect(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    uris: list[str] = []

    class Module:
        @staticmethod
        def connect(uri: str) -> FakeLanceDB:
            uris.append(uri)
            return FakeLanceDB()

    monkeypatch.setitem(sys.modules, "lancedb", Module)
    LanceDBStore(tmp_path)
    LanceDBStore(tmp_path, uri="s3://bucket/db")
    assert uris == [str(tmp_path / "lancedb"), "s3://bucket/db"]
    monkeypatch.setitem(sys.modules, "lancedb", None)
    with pytest.raises(MissingDependencyError, match="pip install lancedb"):
        LanceDBStore(tmp_path)
