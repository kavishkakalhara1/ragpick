"""v0.6: GraphRAG global search, agentic retrieval, per-tenant indexes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from conftest import EXAMPLES, make_chunks
from ragpick import Pipeline, Route, Router
from ragpick.cli import app as cli_app
from ragpick.components.agentic import AgenticRetriever, decompose, parse_action
from ragpick.components.embedders import FakeEmbedder
from ragpick.components.generators import FakeGenerator, LLMGenerator
from ragpick.components.graph import Extraction, GraphRetriever, KnowledgeGraph
from ragpick.components.keyword import BM25Index
from ragpick.components.stores import InMemoryVectorStore
from ragpick.config import ApiKeySpec, ServeConfig
from ragpick.errors import ConfigError, TenantError
from ragpick.filters import acl_filter
from ragpick.models import Answer, Chunk, Completion, RetrievedChunk
from ragpick.pipeline import validate_tenant
from ragpick.selector import recommend
from ragpick.server import App

DATA = EXAMPLES / "data"


class Scripted(LLMGenerator):
    """Replies keyed by system-prompt prefix; records calls."""

    def __init__(self, **replies: list[str]) -> None:
        self.replies = {k: list(v) for k, v in replies.items()}
        self.calls: list[str] = []

    def complete(self, system: str, user: str) -> Completion:
        for prefix, kind in (
            ("You plan document searches", "plan"),
            ("You summarise a cluster", "summary"),
            ("You extract a knowledge graph", "extract"),
        ):
            if system.startswith(prefix):
                self.calls.append(kind)
                queue = self.replies.get(kind, [])
                return Completion(text=queue.pop(0) if queue else "")
        self.calls.append("answer")
        return Completion(text="Answer [1].")


# =================================================================== global search
def ext(*relations: tuple[str, str]) -> Extraction:
    return Extraction.model_validate(
        {"relations": [{"source": a, "target": b} for a, b in relations]}
    )


def test_label_propagation_communities() -> None:
    g = KnowledgeGraph()
    rel = [("Alpha", "Beta"), ("Beta", "Gamma"), ("Alpha", "Gamma")]
    g.add(Chunk(id="a#0", doc_id="a", text="t"), ext(*rel))
    g.add(Chunk(id="b#0", doc_id="b", text="t"), ext(("Xray", "Yankee")))
    g.add(Chunk(id="c#0", doc_id="c", text="t"), ext(("Lonely", "Lonely")))  # self-loop ignored
    communities = g.detect_communities()
    assert communities == [["alpha", "beta", "gamma"], ["xray", "yankee"]]
    assert g.detect_communities() == communities  # deterministic
    names, relations, chunks = g.community_material(communities[0])
    assert names == ["Alpha", "Beta", "Gamma"]
    assert len(relations) == 3
    assert chunks == ["a#0"]


def graph_pipeline(tmp_path: Path, generator: Any = None, **params: Any) -> Pipeline:
    raw = {
        "blueprint": "graph",
        "index_dir": str(tmp_path / "idx"),
        "components": {"retriever": {"params": params}} if params else {},
    }
    return Pipeline.from_dict(raw, components={"generator": generator} if generator else None)


def test_global_search_uses_community_summaries(tmp_path: Path) -> None:
    pipeline = graph_pipeline(tmp_path)
    pipeline.ingest(DATA)
    retriever = pipeline.retriever
    assert isinstance(retriever, GraphRetriever)
    communities = retriever.graph.communities
    assert communities
    assert all(c["summary"].startswith("Topic covering") for c in communities)
    answer = pipeline.query("What are the main themes across these documents?", trace=True)
    assert answer.contexts
    assert all(c.chunk.doc_id == "graph:community" for c in answer.contexts)
    assert all(c.chunk.metadata["synthetic"] for c in answer.contexts)
    assert answer.trace is not None
    assert answer.trace.find("graph_global") is not None
    # targeted global question ranks by overlap
    top = pipeline.retrieve("Give an overview of the LM-200 thermostats", top_k=1)
    assert "LM-200" in top[0].chunk.text
    # local questions still use entity search
    local = pipeline.retrieve("What does error E-1042 mean?", top_k=3)
    assert all(c.chunk.doc_id != "graph:community" for c in local)
    # persisted with the graph
    reloaded = graph_pipeline(tmp_path)
    assert (
        reloaded.retrieve("Summarize the main topics", top_k=2)[0].chunk.doc_id == "graph:community"
    )


def test_global_summaries_respect_access_control(tmp_path: Path) -> None:
    pipeline = graph_pipeline(tmp_path)
    pipeline.ingest(DATA)
    public = " ".join(c.chunk.text for c in pipeline.retrieve("main themes", top_k=10))
    support = " ".join(
        c.chunk.text for c in pipeline.retrieve("main themes", top_k=10, groups=["support"])
    )
    # RMA/ADV and Tier 2 only appear in the support-only escalation guide
    assert "RMA" not in public
    assert "Tier 2" not in public
    assert "RMA" in support


def test_global_modes_and_fallback(tmp_path: Path) -> None:
    always_global = graph_pipeline(tmp_path, mode="global")
    always_global.ingest(DATA)
    assert always_global.retrieve("LM-200", top_k=2)[0].chunk.doc_id == "graph:community"
    # nothing matches the question terms → falls back to local search
    assert all(
        c.chunk.doc_id != "graph:community"
        for c in always_global.retrieve("zebra quantum", top_k=2)
    )
    local_only = graph_pipeline(tmp_path / "l", mode="local", communities=False)
    local_only.ingest(DATA)
    assert local_only.retriever.graph.communities == []  # type: ignore[attr-defined]
    assert all(
        c.chunk.doc_id != "graph:community" for c in local_only.retrieve("main themes", top_k=3)
    )
    with pytest.raises(ConfigError, match="mode"):
        graph_pipeline(tmp_path, mode="cosmic")


def test_llm_summaries_are_reused_when_unchanged(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("The QX-1 connects to the HVAC unit.", encoding="utf-8")
    (docs / "b.md").write_text("The ZZ-9 pairs with the SMS gateway.", encoding="utf-8")
    llm = Scripted(
        extract=[
            '{"relations": [{"source": "QX-1", "target": "HVAC", "relation": "connects to"}]}',
            '{"relations": [{"source": "ZZ-9", "target": "SMS", "relation": "pairs with"}]}',
            '{"relations": [{"source": "ZZ-9", "target": "SMS", "relation": "pairs with"}]}',
        ],
        summary=[
            "QX-1 units connect to HVAC systems.",
            "ZZ-9 devices pair with SMS gateways.",
            "unused",
        ],
    )
    pipeline = graph_pipeline(tmp_path, llm)
    pipeline.ingest(docs)
    assert llm.calls.count("summary") == 2
    summaries = {c["summary"] for c in pipeline.retriever.graph.communities}  # type: ignore[attr-defined]
    assert summaries == {
        "QX-1 units connect to HVAC systems.",
        "ZZ-9 devices pair with SMS gateways.",
    }
    pipeline.ingest(docs / "b.md")  # unchanged content → summaries reused, no new LLM call
    assert llm.calls.count("summary") == 2


# =================================================================== agentic
def test_decompose_and_parse_action() -> None:
    assert decompose("How do I reset my password and enable two-factor authentication?") == [
        "How do I reset my password",
        "enable two-factor authentication",
    ]
    assert decompose("LM-200 vs LM-300 Pro warranty") == ["LM-200", "LM-300 Pro warranty"]
    assert decompose("Compare the LM-200 compared to the LM-300") == [
        "Compare the LM-200",
        "the LM-300",
    ]
    assert decompose("cats and dogs") == []
    assert decompose("a b; c d; e f; g h; i j", max_parts=2) == ["a b", "c d"]
    assert parse_action('{"action": "search", "query": "LM-300 warranty"}') == (
        "search",
        "LM-300 warranty",
    )
    assert parse_action('ok {"action": "finish"}') == ("finish", "")
    assert parse_action('{"action": "search", "query": ""}') == ("finish", "")
    assert parse_action("garbage") == ("finish", "")
    assert parse_action('{"action": ["x"]}') == ("finish", "")


CORPUS = {
    "pw#0": "To reset your password tap forgot password on the sign in screen",
    "tfa#0": "Enable two-factor authentication under account security settings",
    "plus#0": "Lumen Plus costs 4.99 dollars per month",
    "hr#0": "Payroll password rotation policy for HR staff",
}


def agent(generator: Any, **kw: Any) -> AgenticRetriever:
    chunks = make_chunks(CORPUS, {"hr#0": {"allowed_groups": ["hr"]}})
    store, kwi = InMemoryVectorStore(), BM25Index()
    store.add(chunks)
    kwi.add(chunks)
    return AgenticRetriever(FakeEmbedder(dimension=64), store, kwi, generator, **kw)


def test_offline_agent_searches_each_part() -> None:
    retriever = agent(FakeGenerator(), k=3)
    assert retriever.mode == "decompose"
    from ragpick.tracing import trace

    with trace("t") as root:
        results = retriever.retrieve(
            "How do I reset my password and how much does Lumen Plus cost?", filters=acl_filter([])
        )
    ids = [r.chunk.id for r in results]
    assert "pw#0" in ids
    assert "plus#0" in ids
    searches = [s for _, s in root.walk() if s.name == "agent_search"]
    assert len(searches) == 3
    assert root.attributes["planner"] == "decompose"
    assert all(r.source == "fused" for r in results)
    assert "hr#0" not in [
        r.chunk.id for r in retriever.retrieve("password and payroll rotation", 10, acl_filter([]))
    ]


def test_llm_agent_plans_follow_up_searches() -> None:
    llm = Scripted(
        plan=[
            '{"action": "search", "query": "two-factor authentication settings"}',
            '{"action": "finish"}',
        ]
    )
    retriever = agent(llm, k=4, max_steps=5)
    assert retriever.mode == "llm"
    ids = [r.chunk.id for r in retriever.retrieve("How do I secure my account login?")]
    assert "tfa#0" in ids
    assert llm.calls == ["plan", "plan"]


def test_llm_agent_stops_on_repeats_idle_and_max_steps() -> None:
    repeat = Scripted(plan=['{"action": "search", "query": "secure my account"}'])
    agent(repeat, max_steps=5).retrieve("secure my account")
    assert repeat.calls == ["plan"]  # repeated query ends the loop
    looping = Scripted(plan=[f'{{"action": "search", "query": "password {i}"}}' for i in range(10)])
    from ragpick.tracing import trace

    with trace("t") as root:
        agent(looping, max_steps=10, per_search_k=2).retrieve("reset password")
    assert root.attributes["searches"] < 10  # stopped once searches stopped adding results
    capped = Scripted(plan=[f'{{"action": "search", "query": "q{i} lumen"}}' for i in range(10)])
    with trace("t") as root2:
        agent(capped, max_steps=2).retrieve("reset password")
    assert root2.attributes["searches"] == 2


def test_agentic_validation_and_blueprint(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="planner"):
        agent(FakeGenerator(), planner="psychic")
    with pytest.raises(ConfigError, match="max_steps"):
        agent(FakeGenerator(), max_steps=0)

    class GenerateOnly:
        def generate(self, query: str, contexts: list[RetrievedChunk]) -> Answer:
            return Answer(text="")

    with pytest.raises(ConfigError, match="complete"):
        agent(GenerateOnly(), planner="llm")
    pipeline = Pipeline.from_dict({"blueprint": "agentic", "index_dir": str(tmp_path / "a")})
    pipeline.ingest(DATA)
    answer = pipeline.query("How do I reset my password and how much does Lumen Plus cost?")
    docs = {c.chunk.doc_id for c in answer.contexts}
    assert {"account-and-password.md", "billing-and-subscriptions.md"} <= docs


def test_agentic_and_corrective_over_graph(tmp_path: Path) -> None:
    for blueprint in ("agentic", "corrective"):
        pipeline = Pipeline.from_dict(
            {
                "blueprint": blueprint,
                "index_dir": str(tmp_path / blueprint),
                "components": {"retriever": {"params": {"base": "graph"}}},
            }
        )
        pipeline.ingest(DATA)
        assert (tmp_path / blueprint / "graph" / "graph.json").is_file()  # index_chunks forwarded
        assert pipeline.retrieve("What does error E-1042 mean?", top_k=3)


def test_selector_agentic_and_tenancy() -> None:
    r = recommend({"complex_questions": True}, installed=lambda m: True)
    assert r.blueprint == "agentic"
    slow = recommend({"complex_questions": True, "latency": "low"}, installed=lambda m: True)
    assert any("max_steps" in w for w in slow.warnings)
    tenant = recommend({"multi_tenant": True}, installed=lambda m: True)
    assert any("tenancy: {enabled: true}" in reason for reason in tenant.reasons)


# =================================================================== tenancy
@pytest.mark.parametrize("bad", ["", "../etc", "a/b", "a b", ".hidden", "-x", "x" * 65, "ä"])
def test_tenant_ids_are_validated(bad: str) -> None:
    with pytest.raises(TenantError, match="Invalid tenant id"):
        validate_tenant(bad)


def test_valid_tenant_ids() -> None:
    for good in ("acme", "Acme-Corp_2", "1", "x" * 64):
        assert validate_tenant(good) == good


def tenant_pipeline(tmp_path: Path, **tenancy: Any) -> Pipeline:
    return Pipeline.from_dict(
        {
            "blueprint": "hybrid",
            "index_dir": str(tmp_path / "idx"),
            "tenancy": {"enabled": True, **tenancy},
            "cache": {"enabled": True, "dir": str(tmp_path / "cache")},
        }
    )


def two_tenant_docs(tmp_path: Path) -> tuple[Path, Path]:
    acme = tmp_path / "acme"
    globex = tmp_path / "globex"
    acme.mkdir()
    globex.mkdir()
    (acme / "pricing.md").write_text(
        "Acme pays 100 dollars per seat for the Gold plan.", encoding="utf-8"
    )
    (globex / "pricing.md").write_text(
        "Globex pays 70 dollars per seat for the Gold plan.", encoding="utf-8"
    )
    return acme, globex


def test_tenants_are_isolated(tmp_path: Path) -> None:
    pipeline = tenant_pipeline(tmp_path)
    acme_docs, globex_docs = two_tenant_docs(tmp_path)
    pipeline.ingest(acme_docs, tenant="acme")
    pipeline.ingest(globex_docs, tenant="globex")
    assert (tmp_path / "idx" / "tenants" / "acme" / "ragpick.json").is_file()
    assert (tmp_path / "idx" / "tenants" / "globex" / "ragpick.json").is_file()
    acme = pipeline.query("How much per seat for Gold?", tenant="acme")
    globex = pipeline.query("How much per seat for Gold?", tenant="globex")
    assert "100" in acme.text
    assert "70" not in acme.text
    assert "70" in globex.text
    assert all(
        "Globex" not in c.chunk.text
        for c in pipeline.retrieve("Gold plan", tenant="acme", top_k=10)
    )
    streamed = list(pipeline.stream("How much per seat for Gold?", tenant="globex"))
    assert "70" in streamed[-1].answer.text  # type: ignore[union-attr]
    # models and the cache are shared, indexes are not
    t = pipeline.for_tenant("acme")
    assert t.embedder is pipeline.embedder
    assert t.generator is pipeline.generator
    assert t.store is not pipeline.for_tenant("globex").store
    assert pipeline.ready()
    with pytest.raises(Exception, match="ragpick ingest"):
        pipeline.query("x", tenant="unknown-tenant")


def test_tenant_required_and_rejected(tmp_path: Path) -> None:
    pipeline = tenant_pipeline(tmp_path)
    with pytest.raises(TenantError, match="pass a tenant"):
        pipeline.query("x")
    with pytest.raises(TenantError, match="pass a tenant"):
        list(pipeline.stream("x"))
    with pytest.raises(TenantError, match="Invalid tenant"):
        pipeline.ingest(DATA, tenant="../escape")
    plain = Pipeline.from_dict({"blueprint": "naive", "index_dir": str(tmp_path / "plain")})
    with pytest.raises(TenantError, match="isn't multi-tenant"):
        plain.retrieve("x", tenant="acme")


def test_tenant_lru(tmp_path: Path) -> None:
    pipeline = tenant_pipeline(tmp_path, max_loaded=2)
    a = pipeline.for_tenant("a")
    pipeline.for_tenant("b")
    assert pipeline.for_tenant("a") is a  # cached; now most recent
    pipeline.for_tenant("c")  # evicts b
    assert list(pipeline._tenants) == ["a", "c"]
    assert pipeline.for_tenant("b") is not None


def test_router_passes_tenant(tmp_path: Path) -> None:
    pipeline = tenant_pipeline(tmp_path)
    acme_docs, _ = two_tenant_docs(tmp_path)
    pipeline.ingest(acme_docs, tenant="acme")
    router = Router([Route("docs", pipeline)])
    assert "100" in router.query("Gold plan price per seat?", tenant="acme").text
    assert router.retrieve("Gold plan", tenant="acme")
    assert list(router.stream("Gold plan", tenant="acme"))[-1].answer.route == "docs"  # type: ignore[union-attr]
    with pytest.raises(ConfigError, match="per tenant"):
        router.ingest(tenant="acme")


def test_server_binds_keys_to_tenants(tmp_path: Path) -> None:
    pipeline = tenant_pipeline(tmp_path)
    acme_docs, globex_docs = two_tenant_docs(tmp_path)
    pipeline.ingest(acme_docs, tenant="acme")
    pipeline.ingest(globex_docs, tenant="globex")
    keys = [
        ApiKeySpec(key="acme-key-0123456789", name="acme-app", tenant="acme"),
        ApiKeySpec(key="admin-key-0123456789", name="admin"),
    ]
    app = App(pipeline, ServeConfig(api_keys=keys))

    def ask(key: str, **body: Any) -> tuple[int, Any]:
        r = app.handle(
            "POST",
            "/query",
            {"authorization": f"Bearer {key}"},
            json.dumps({"question": "Gold plan price per seat?", **body}).encode(),
        )
        return r.status, r.body

    status, body = ask("acme-key-0123456789")
    assert status == 200
    assert "100" in body["text"]
    assert ask("acme-key-0123456789", tenant="globex")[0] == 403  # can't reach another tenant
    assert ask("acme-key-0123456789", tenant="acme")[0] == 200
    status, body = ask("admin-key-0123456789", tenant="globex")
    assert "70" in body["text"]
    status, body = ask("admin-key-0123456789")
    assert status == 400
    assert "pass a tenant" in body["error"]
    assert ask("admin-key-0123456789", tenant="../x")[0] == 400
    assert app.handle("GET", "/health", {}, b"").status == 200


def test_tenant_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    acme_docs, _ = two_tenant_docs(tmp_path)
    Path("rag.yaml").write_text("blueprint: hybrid\ntenancy: {enabled: true}\n", encoding="utf-8")
    runner = CliRunner()
    assert runner.invoke(cli_app, ["ingest", str(acme_docs), "--tenant", "acme"]).exit_code == 0
    result = runner.invoke(cli_app, ["query", "Gold plan price per seat?", "--tenant", "acme"])
    assert result.exit_code == 0, result.output
    assert "100" in result.output
    streamed = runner.invoke(cli_app, ["query", "Gold plan price?", "--tenant", "acme", "--stream"])
    assert "100" in streamed.output
    missing = runner.invoke(cli_app, ["query", "x"])
    assert missing.exit_code == 1
    assert "pass a tenant" in missing.output
    shutil.copy(EXAMPLES / "hybrid.yaml", tmp_path / "h.yaml")
    router_cfg = tmp_path / "router.yaml"
    router_cfg.write_text("router: {routes: [{name: d, config: rag.yaml}]}\n", encoding="utf-8")
    bad = runner.invoke(cli_app, ["ingest", "-c", str(router_cfg), "--tenant", "acme"])
    assert bad.exit_code == 1
    assert "per tenant" in bad.output
