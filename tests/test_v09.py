"""v0.9: hosted rerankers, hierarchical (RAPTOR) retrieval, answer feedback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from typer.testing import CliRunner

from conftest import EXAMPLES
from ragpick import Pipeline, Route, Router
from ragpick.cli import app as cli_app
from ragpick.components.generators import LLMGenerator
from ragpick.components.hierarchical import HierarchicalRetriever, cluster, extractive_summary
from ragpick.components.rerankers import CohereReranker, JinaReranker
from ragpick.config import ApiKeySpec, ServeConfig
from ragpick.errors import ConfigError, MissingCredentialError, ProviderError
from ragpick.eval import evaluate, load_dataset
from ragpick.feedback import (
    AnswerLog,
    FeedbackError,
    FeedbackStore,
    normalize_rating,
    record_from_answer,
)
from ragpick.models import Answer, Chunk, Completion, RetrievedChunk
from ragpick.selector import recommend
from ragpick.server import App

DATA = EXAMPLES / "data"


def rc(cid: str, text: str, source: str = "fused") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id=cid, doc_id=cid.split("#")[0], text=text), score=0.1, source=source
    )  # type: ignore[arg-type]


# =================================================================== rerankers
@pytest.mark.parametrize(
    ("cls", "path", "env"),
    [
        (CohereReranker, "/v2/rerank", "COHERE_API_KEY"),
        (JinaReranker, "/v1/rerank", "JINA_API_KEY"),
    ],
)
def test_hosted_rerankers(cls: Any, path: str, env: str, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.4},
                    {"index": 7, "relevance_score": 0.99},
                ]
            },
        )

    monkeypatch.setenv(env, "secret-key")
    reranker = cls(base_url="https://api.test")
    reranker._client = httpx.Client(
        base_url="https://api.test",
        transport=httpx.MockTransport(handler),
        headers=reranker._client.headers,
    )
    chunks = [rc("a#0", "alpha", "keyword"), rc("b#0", "beta"), rc("c#0", "gamma", "vector")]
    out = reranker.rerank("q", chunks, top_n=2)
    assert [(c.chunk.id, c.score, c.source) for c in out] == [
        ("c#0", 0.9, "vector"),
        ("a#0", 0.4, "keyword"),
    ]
    assert seen["path"] == path
    assert seen["body"] == {
        "model": reranker.model,
        "query": "q",
        "documents": ["alpha", "beta", "gamma"],
        "top_n": 2,
    }
    assert seen["auth"] == "Bearer secret-key"
    assert reranker.rerank("q", [], 3) == []
    monkeypatch.delenv(env)
    with pytest.raises(MissingCredentialError, match=env):
        cls()


def test_reranker_errors() -> None:
    def make(handler: Any) -> CohereReranker:
        return CohereReranker(
            client=httpx.Client(base_url="https://x", transport=httpx.MockTransport(handler))
        )

    with pytest.raises(ProviderError, match="HTTP 429"):
        make(lambda r: httpx.Response(429, text="rate limited")).rerank("q", [rc("a#0", "x")], 1)
    with pytest.raises(ProviderError, match="rerank failed"):
        make(lambda r: httpx.Response(200, json={"unexpected": True})).rerank(
            "q", [rc("a#0", "x")], 1
        )

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderError, match="refused"):
        make(down).rerank("q", [rc("a#0", "x")], 1)
    assert JinaReranker(client=httpx.Client()).model == "jina-reranker-v2-base-multilingual"


def test_reranker_in_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "k")
    p = Pipeline.from_dict(
        {
            "blueprint": "hybrid",
            "index_dir": str(tmp_path / "i"),
            "components": {"reranker": {"type": "cohere", "params": {"model": "rerank-v3.5"}}},
        }
    )
    assert isinstance(p.reranker, CohereReranker)


# =================================================================== hierarchical
def test_cluster_and_extractive_summary() -> None:
    vectors = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [1, 0.05]], dtype=float)
    groups = cluster(vectors, size=3)
    assert groups == [[0, 1, 4], [2, 3]]
    assert cluster(np.zeros((0, 2)), 3) == []
    assert cluster(np.zeros((3, 2)), 2) == [[0, 1], [2]]  # zero vectors don't crash
    summary = extractive_summary(
        [
            "# Title\n\nThe LM-200 runs on batteries for months.",
            "short",
            "Invoices are emailed after every payment.",
        ]
    )
    assert (
        summary
        == "The LM-200 runs on batteries for months. Invoices are emailed after every payment."
    )


def tree_pipeline(tmp_path: Path, generator: Any = None, **params: Any) -> Pipeline:
    raw = {
        "blueprint": "hierarchical",
        "index_dir": str(tmp_path / "idx"),
        "components": {"retriever": {"params": {"cluster_size": 3, **params}}},
    }
    return Pipeline.from_dict(raw, components={"generator": generator} if generator else None)


def test_tree_is_built_and_retrieved(tmp_path: Path) -> None:
    pipeline = tree_pipeline(tmp_path)
    retriever = pipeline.retriever
    assert isinstance(retriever, HierarchicalRetriever)
    assert retriever.llm is None
    pipeline.ingest(DATA)
    tree = json.loads((tmp_path / "idx" / "hierarchy" / "tree.json").read_text(encoding="utf-8"))
    assert tree["nodes"]
    levels = {int(doc.split("-")[1]) for doc in tree["nodes"]}
    assert 1 in levels
    assert max(levels) >= 2  # summaries of summaries
    contexts = pipeline.retrieve("overview of thermostat setup warranty and billing", top_k=10)
    summaries = [c for c in contexts if c.chunk.metadata.get("synthetic")]
    assert summaries
    node = summaries[0].chunk
    assert node.metadata["children"]
    assert node.metadata["source_docs"]
    assert node.metadata["level"] >= 1
    # summaries are excluded from retrieval metrics
    report = evaluate(pipeline, load_dataset(EXAMPLES / "eval.jsonl"))
    assert all(not d.startswith("summary-") for r in report.results for d in r.retrieved_doc_ids)


def test_summaries_never_mix_access_scopes(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(4):
        (docs / f"public{i}.md").write_text(
            f"Public refund policy number {i} applies to every customer order.", encoding="utf-8"
        )
        (docs / f"hr{i}.md").write_text(
            f"---\nallowed_groups: [hr]\n---\nSecret payroll code ZX-{i}00 for refund approvals.",
            encoding="utf-8",
        )
    pipeline = tree_pipeline(tmp_path)
    pipeline.ingest(docs)
    nodes = [
        c
        for c in pipeline.retrieve("refund policy payroll", groups=["hr"], top_k=30)
        if c.chunk.metadata.get("synthetic")
    ]
    assert nodes
    for n in nodes:
        restricted = "allowed_groups" in n.chunk.metadata
        sources = n.chunk.metadata["source_docs"]
        assert (
            all(s.startswith("hr") for s in sources)
            if restricted
            else all(s.startswith("public") for s in sources)
        )
    public_text = " ".join(
        c.chunk.text for c in pipeline.retrieve("refund policy payroll code", top_k=30)
    )
    assert "ZX-" not in public_text


class Summarizer(LLMGenerator):
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str) -> Completion:
        if system.startswith("You summarise related passages"):
            self.calls += 1
            return Completion(text=f"LLM summary {self.calls} of the cluster.")
        return Completion(text="Answer [1].")


def test_llm_summaries_reused_on_reingest(tmp_path: Path) -> None:
    llm = Summarizer()
    pipeline = tree_pipeline(tmp_path, llm)
    assert pipeline.retriever.llm is llm  # type: ignore[attr-defined]
    pipeline.ingest(DATA)
    first = llm.calls
    assert first > 0
    pipeline.ingest(DATA / "privacy-and-data.md")  # same content → every cluster unchanged
    assert llm.calls == first
    reloaded = tree_pipeline(tmp_path, llm)
    reloaded.ingest(DATA / "privacy-and-data.md")  # summaries persisted with the tree
    assert llm.calls == first
    (tmp_path / "changed.md").write_text(
        "A brand new document about LM-900 heat pumps and their warranty.", encoding="utf-8"
    )
    reloaded.ingest(tmp_path / "changed.md")
    assert llm.calls > first


def test_hierarchical_validation(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="summarizer"):
        tree_pipeline(tmp_path, summarizer="magic")
    with pytest.raises(ConfigError, match="cluster_size"):
        tree_pipeline(tmp_path, cluster_size=1)
    with pytest.raises(ConfigError, match="max_levels"):
        tree_pipeline(tmp_path, max_levels=0)

    class GenerateOnly:
        def generate(self, query: str, contexts: Any) -> Answer:
            return Answer(text="")

    with pytest.raises(ConfigError, match="complete"):
        tree_pipeline(tmp_path, GenerateOnly(), summarizer="llm")
    single = tmp_path / "single"
    single.mkdir()
    (single / "one.md").write_text("Only one chunk of text lives here.", encoding="utf-8")
    p = tree_pipeline(tmp_path / "s")
    p.ingest(single)
    assert p.retriever.node_doc_ids == []  # type: ignore[attr-defined]


def test_selector_summary_questions() -> None:
    r = recommend({"summary_questions": True, "long_documents": True}, installed=lambda m: True)
    assert r.blueprint == "hierarchical"
    assert "graph" in r.alternatives


# =================================================================== feedback
def test_normalize_rating() -> None:
    assert normalize_rating("up") == 1.0
    assert normalize_rating("Thumbs_Down") == 0.0
    assert normalize_rating(True) == 1.0
    assert normalize_rating(False) == 0.0
    assert normalize_rating(5) == 1.0
    assert normalize_rating(3) == 0.5
    assert normalize_rating("4") == 0.75
    for bad in (0, 6, "meh", 2.5e9):
        with pytest.raises(FeedbackError):
            normalize_rating(bad)


def answer(question: str = "How do I reset my password?") -> Answer:
    ctx = [rc("account.md#0", "Tap Forgot password."), rc("graph#facts", "facts")]
    ctx[1].chunk.metadata["synthetic"] = True
    return Answer(
        text="Tap Forgot password [1][2].",
        citations=["account.md#0", "graph#facts"],
        contexts=ctx,
        question=question,
        route="docs",
    )


def test_record_and_store(tmp_path: Path) -> None:
    record = record_from_answer(
        answer(), "up", comment="x" * 3000, owner="portal", groups=["support"]
    )
    assert record.cited_docs == ["account.md"]  # synthetic contexts excluded
    assert len(record.comment or "") == 2000
    store = FeedbackStore(tmp_path / "f.sqlite")
    store.add(record)
    store.add(record_from_answer(answer("Why is it broken?"), "down"))
    store.add(record_from_answer(answer("Why is it broken?"), 2))
    store.add(record_from_answer(answer("Unrated?").model_copy(update={"question": None}), 4))
    stats = store.stats()
    assert stats["total"] == 4
    assert stats["positive"] == 2
    assert stats["negative"] == 2
    assert stats["worst_questions"][0] == {
        "question": "Why is it broken?",
        "average": 0.125,
        "count": 2,
    }
    assert stats["routes"]["docs"]["count"] == 4
    rows = store.export_eval()
    assert rows[0] == {
        "id": "feedback-1",
        "question": "How do I reset my password?",
        "ground_truth": "Tap Forgot password [1][2].",
        "relevant_doc_ids": ["account.md"],
        "groups": ["support"],
    }
    assert rows[1] == {"id": "feedback-review-1", "question": "Why is it broken?"}
    assert FeedbackStore(tmp_path / "empty.sqlite").stats()["average"] is None


def test_answer_log_scopes_to_owner() -> None:
    log = AnswerLog(size=2)
    a, b, c = answer(), answer(), answer()
    log.remember(a, "alice|-", ["support"])
    assert log.get(a.id, "alice|-") == (a, ["support"])
    with pytest.raises(FeedbackError, match="Unknown answer"):
        log.get(a.id, "mallory|-")
    log.remember(b, "alice|-", None)
    log.remember(c, "alice|-", None)
    with pytest.raises(FeedbackError):
        log.get(a.id, "alice|-")  # evicted


def test_answers_have_ids_and_questions(tmp_path: Path) -> None:
    p = Pipeline.from_dict({"blueprint": "naive", "index_dir": str(tmp_path / "i")})
    p.ingest(DATA)
    a, b = p.query("What does error E-1042 mean?"), p.query("What does error E-1042 mean?")
    assert a.id != b.id
    assert a.question == "What does error E-1042 mean?"
    streamed = list(p.stream("What does error E-1042 mean?"))[-1].answer
    assert streamed is not None
    assert streamed.question == "What does error E-1042 mean?"
    fid = p.record_feedback(a, "up", comment="great")
    stored = p.feedback_store.all()
    assert stored[0].id == fid
    assert stored[0].route == "naive"  # blueprint when there's no route
    router = Router([Route("docs", p)], sessions_dir=tmp_path / "r")
    routed = router.query("What does error E-1042 mean?")
    router.record_feedback(routed, 1)
    assert router.feedback_store.all()[0].route == "docs"


def test_feedback_over_http(tmp_path: Path) -> None:
    pipeline = Pipeline.from_dict(
        {"blueprint": "hybrid", "index_dir": str(tmp_path / "i"), "feedback": {"enabled": True}}
    )
    pipeline.ingest(DATA)
    keys = [
        ApiKeySpec(key="alice-key-0123456789", name="alice", groups=["support"]),
        ApiKeySpec(key="mallory-key-012345678", name="mallory"),
    ]
    app = App(pipeline, ServeConfig(api_keys=keys))

    def call(path: str, key: str, body: dict[str, Any]) -> Any:
        return app.handle(
            "POST", path, {"authorization": f"Bearer {key}"}, json.dumps(body).encode()
        )

    answer_body = call(
        "/query", "alice-key-0123456789", {"question": "How do I reset my password?"}
    ).body
    answer_id = answer_body["id"]
    ok = call(
        "/feedback",
        "alice-key-0123456789",
        {"answer_id": answer_id, "rating": "up", "comment": "thanks"},
    )
    assert ok.status == 200
    record = pipeline.feedback_store.all()[0]
    assert (record.id, record.owner, record.groups, record.rating) == (
        ok.body["id"],
        "alice",
        ["support"],
        1.0,
    )
    assert record.question == "How do I reset my password?"
    forged = call("/feedback", "mallory-key-012345678", {"answer_id": answer_id, "rating": "down"})
    assert forged.status == 404  # can't rate someone else's answer
    assert (
        call("/feedback", "alice-key-0123456789", {"answer_id": "nope", "rating": "up"}).status
        == 404
    )
    assert (
        call("/feedback", "alice-key-0123456789", {"answer_id": answer_id, "rating": "meh"}).status
        == 400
    )
    assert call("/feedback", "alice-key-0123456789", {"answer_id": answer_id}).status == 422
    chat = call("/chat", "alice-key-0123456789", {"message": "How much does Lumen Plus cost?"}).body
    assert (
        call("/feedback", "alice-key-0123456789", {"answer_id": chat["id"], "rating": 5}).status
        == 200
    )
    stream = call(
        "/query", "alice-key-0123456789", {"question": "What does E-1042 mean?", "stream": True}
    )
    final = json.loads(
        b"".join(stream.stream).decode().strip().split("\n\n")[-1].split("data: ", 1)[1]
    )
    assert (
        call(
            "/feedback", "alice-key-0123456789", {"answer_id": final["id"], "rating": "down"}
        ).status
        == 200
    )
    assert len(pipeline.feedback_store.all()) == 3
    disabled = App(Pipeline.from_dict({"blueprint": "naive", "index_dir": str(tmp_path / "d")}))
    assert (
        disabled.handle("POST", "/feedback", {}, b'{"answer_id": "x", "rating": "up"}').status
        == 404
    )


def test_tenant_streaming_over_http(tmp_path: Path) -> None:
    """Regression: streamed queries must pass the tenant through (v0.6-v0.8 dropped it)."""
    acme = tmp_path / "acme"
    acme.mkdir()
    (acme / "pricing.md").write_text(
        "Acme pays 100 dollars per seat for the Gold plan.", encoding="utf-8"
    )
    pipeline = Pipeline.from_dict(
        {"blueprint": "hybrid", "index_dir": str(tmp_path / "i"), "tenancy": {"enabled": True}}
    )
    pipeline.ingest(acme, tenant="acme")
    app = App(
        pipeline,
        ServeConfig(api_keys=[ApiKeySpec(key="acme-key-0123456789", name="acme", tenant="acme")]),
    )
    response = app.handle(
        "POST",
        "/query",
        {"authorization": "Bearer acme-key-0123456789"},
        b'{"question": "Gold plan price?", "stream": true}',
    )
    assert response.status == 200
    body = b"".join(response.stream or []).decode()
    assert "100" in body


def test_feedback_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("rag.yaml").write_text("blueprint: hybrid\n", encoding="utf-8")
    runner = CliRunner()
    assert "No feedback yet" in runner.invoke(cli_app, ["feedback", "stats"]).output
    p = Pipeline.from_yaml("rag.yaml")
    p.ingest(DATA)
    p.record_feedback(p.query("How do I reset my password?"), "up")
    p.record_feedback(p.query("What is the capital of France?"), "down")
    stats = runner.invoke(cli_app, ["feedback", "stats"])
    assert stats.exit_code == 0, stats.output
    assert "2 ratings" in stats.output
    assert "capital of France" in stats.output
    exported = runner.invoke(cli_app, ["feedback", "export", "--as-eval", "-o", "fb.jsonl"])
    assert exported.exit_code == 0, exported.output
    rows = [json.loads(line) for line in Path("fb.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["relevant_doc_ids"] == ["account-and-password.md"]
    assert rows[1] == {"id": "feedback-review-1", "question": "What is the capital of France?"}
    assert len(load_dataset("fb.jsonl")) == 2  # valid eval dataset
    everything = runner.invoke(cli_app, ["feedback", "export"])
    assert "Wrote 2 rows" in everything.output
