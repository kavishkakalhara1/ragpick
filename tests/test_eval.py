from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from conftest import EXAMPLES
from ragpick import Pipeline
from ragpick.config import ComponentSpec, PriceSpec
from ragpick.errors import ConfigError, ProviderError
from ragpick.eval import evaluate, evaluate_configs, load_dataset, write_reports
from ragpick.eval.metrics import (
    DEFAULT_PRICES,
    LLMJudge,
    estimate_cost,
    hit_rate_at_k,
    mean,
    parse_judge_response,
    percentile,
    precision_at_k,
    reciprocal_rank,
)
from ragpick.eval.runner import METRICS, build_judge, parse_judge_option
from ragpick.models import Answer, Chunk, Completion, RetrievedChunk, Usage

DATA = EXAMPLES / "data"


# ------------------------------------------------------------------ metric math
def test_hit_rate() -> None:
    assert hit_rate_at_k(["a", "b", "c"], ["c"], 3) == 1.0
    assert hit_rate_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert hit_rate_at_k([], ["c"], 3) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"c", "b"}) == pytest.approx(0.5)
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_precision_at_k() -> None:
    assert precision_at_k(["a", "x", "a", "y"], {"a"}, 4) == pytest.approx(0.5)
    assert precision_at_k(["a"], {"a"}, 4) == pytest.approx(0.25)  # missing slots count
    assert precision_at_k(["a", "b"], {"a", "b"}, 2) == 1.0
    with pytest.raises(ValueError, match="positive"):
        precision_at_k(["a"], {"a"}, 0)


def test_percentile_and_mean() -> None:
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == pytest.approx(50.5)
    assert percentile(values, 95) == pytest.approx(95.05)
    assert percentile([], 50) is None
    assert mean([1.0, None, 3.0]) == 2.0
    assert mean([None]) is None


def test_estimate_cost() -> None:
    usage = [Usage(prompt_tokens=1_000_000, completion_tokens=500_000)]
    assert estimate_cost(usage, "gpt-4o-mini") == pytest.approx(0.15 + 0.30)
    custom = {"my-model": PriceSpec(prompt_per_1m=1.0, completion_per_1m=2.0)}
    assert estimate_cost(Usage(prompt_tokens=10, completion_tokens=5), "my-model", custom) == (
        pytest.approx(20 / 1_000_000)
    )
    assert estimate_cost(usage, "unknown-model") is None
    assert estimate_cost(usage, "fake") == 0.0
    assert "gpt-4o-mini" in DEFAULT_PRICES


def test_parse_judge_response() -> None:
    assert parse_judge_response('{"score": 0.8, "reason": "ok"}') == (0.8, "ok")
    assert parse_judge_response('Sure! ```json\n{"score": 1, "reason": "x"}\n```') == (1.0, "x")
    assert parse_judge_response('{"score": 7}') == (1.0, "")
    assert parse_judge_response('{"score": -2, "reason": "r"}') == (0.0, "r")
    score, reason = parse_judge_response("no json here")
    assert score is None
    assert "unparseable" in reason
    assert parse_judge_response('{"reason": "missing score"}')[0] is None
    assert parse_judge_response('{"score": "high"}')[0] is None


class ScriptedJudge:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        self.prompts.append((system, user))
        return Completion(text=self.replies[len(self.prompts) - 1], usage=Usage(prompt_tokens=10))


def test_llm_judge_prompts() -> None:
    judge_llm = ScriptedJudge(
        ['{"score": 1, "reason": "supported"}', '{"score": 0.5, "reason": "partial"}']
    )
    judge = LLMJudge(judge_llm)
    ctx = [
        RetrievedChunk(
            chunk=Chunk(id="a#0", doc_id="a", text="The sky is blue."), score=1, source="vector"
        )
    ]
    faith = judge.faithfulness("The sky is blue [1].", ctx)
    assert (faith.score, faith.reason) == (1.0, "supported")
    assert faith.usage.prompt_tokens == 10
    rel = judge.answer_relevance("What colour is the sky?", "Blue.")
    assert rel.score == 0.5
    assert "FAITHFULNESS" in judge_llm.prompts[0][1]
    assert "The sky is blue." in judge_llm.prompts[0][1]
    assert "What colour is the sky?" in judge_llm.prompts[1][1]
    assert "JSON" in judge_llm.prompts[0][0]
    empty_llm = ScriptedJudge(['{"score": 1, "reason": "idk"}'])
    LLMJudge(empty_llm).faithfulness("I don't know.", [])
    assert "(no context)" in empty_llm.prompts[0][1]


# ------------------------------------------------------------------ dataset
def test_load_example_dataset() -> None:
    examples = load_dataset(EXAMPLES / "eval.jsonl")
    assert 10 <= len(examples) <= 15
    assert all(e.relevant_doc_ids for e in examples)
    assert any(e.groups == ["support"] for e in examples)


def test_dataset_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_dataset(tmp_path / "missing.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"question": "ok"}\n\n# comment\n{oops\n', encoding="utf-8")
    with pytest.raises(ConfigError, match=r"bad\.jsonl:4: invalid JSON"):
        load_dataset(bad)
    typo = tmp_path / "typo.jsonl"
    typo.write_text('{"question": "q", "relevant_docs": ["a"]}\n', encoding="utf-8")
    with pytest.raises(
        ConfigError, match=r"(?s)dataset row in .*typo\.jsonl:1.*relevant_docs: unknown key"
    ):
        load_dataset(typo)
    missing = tmp_path / "missing_q.jsonl"
    missing.write_text('{"ground_truth": "x"}\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="question: required"):
        load_dataset(missing)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="no examples"):
        load_dataset(empty)


# ------------------------------------------------------------------ runner
def pipeline_for(tmp_path: Path, blueprint: str) -> Pipeline:
    p = Pipeline.from_dict(
        {
            "blueprint": blueprint,
            "name": blueprint,
            "index_dir": str(tmp_path / blueprint),
            "query": {"top_k": 3},
        }
    )
    p.ingest(DATA)
    return p


def test_evaluate_retrieval_metrics(tmp_path: Path) -> None:
    dataset = load_dataset(EXAMPLES / "eval.jsonl")
    progress: list[tuple[int, int]] = []
    report = evaluate(
        pipeline_for(tmp_path, "hybrid"), dataset, progress=lambda d, t: progress.append((d, t))
    )
    assert report.label == "hybrid"
    assert report.k == 3
    assert report.model == "fake"
    assert report.num_examples == len(dataset)
    assert progress[-1] == (len(dataset), len(dataset))
    m = report.metrics
    assert set(m) == set(METRICS)
    assert 0.5 <= (m["hit_rate@k"] or 0) <= 1.0
    assert 0 < (m["mrr"] or 0) <= (m["hit_rate@k"] or 0)
    assert 0 < (m["precision@k"] or 0) <= 1
    assert m["faithfulness"] is None  # no judge
    assert m["latency_p50_ms"] is not None
    assert (m["latency_p95_ms"] or 0) >= (m["latency_p50_ms"] or 0)
    assert (m["total_tokens"] or 0) > 0
    assert m["est_cost_usd"] == 0.0
    assert m["errors"] == 0.0
    # the access-controlled question finds the internal doc because it runs with groups
    acl = next(r for r in report.results if "Tier 2" in r.question)
    assert acl.hit == 1.0


def test_evaluate_matches_manual_computation(tmp_path: Path) -> None:
    from ragpick.eval.dataset import EvalExample

    class StubPipeline:
        """Anything with query() and the engine summary properties can be evaluated."""

        name = "stub"
        blueprint = "naive"
        top_k = 2
        model_name = "fake"

        def query(self, question: str, groups: Any = None) -> Answer:
            docs = {"q1": ["a", "b"], "q2": ["x", "b"]}[question]
            ctx = [
                RetrievedChunk(chunk=Chunk(id=f"{d}#0", doc_id=d, text=d), score=1, source="vector")
                for d in docs
            ]
            return Answer(
                text="t",
                contexts=ctx,
                usage=Usage(prompt_tokens=5, completion_tokens=1, latency_ms=1),
            )

    dataset = [
        EvalExample(question="q1", relevant_doc_ids=["b"]),
        EvalExample(question="q2", relevant_doc_ids=["z"]),
        EvalExample(question="q1"),  # no labels: excluded from retrieval metrics
    ]
    report = evaluate(StubPipeline(), dataset)  # type: ignore[arg-type]
    assert report.metrics["hit_rate@k"] == pytest.approx(0.5)
    assert report.metrics["mrr"] == pytest.approx(0.25)
    assert report.metrics["precision@k"] == pytest.approx(0.25)
    assert report.metrics["total_tokens"] == 18.0
    assert report.results[2].hit is None


def test_evaluate_with_judge_and_errors(tmp_path: Path) -> None:
    from ragpick.eval.dataset import EvalExample

    class FlakyJudge:
        calls = 0

        def complete(self, system: str, user: str) -> Completion:
            FlakyJudge.calls += 1
            if FlakyJudge.calls > 2:
                raise ProviderError("judge down")
            return Completion(text='{"score": 0.75, "reason": "fine"}')

    pipeline = pipeline_for(tmp_path, "naive")
    dataset = [EvalExample(question="How do I reset my password?"), EvalExample(question="E-1042?")]
    report = evaluate(pipeline, dataset, judge=LLMJudge(FlakyJudge()))
    assert report.results[0].faithfulness == 0.75
    assert report.results[0].answer_relevance_reason == "fine"
    assert report.results[1].error == "judge failed: judge down"
    assert report.metrics["faithfulness"] == 0.75
    assert report.metrics["errors"] == 1.0


def test_evaluate_records_query_errors(tmp_path: Path) -> None:
    from ragpick.eval.dataset import EvalExample

    pipeline = Pipeline.from_dict({"blueprint": "naive", "index_dir": str(tmp_path / "none")})
    report = evaluate(pipeline, [EvalExample(question="q")])
    assert report.results[0].error is not None
    assert "ragpick ingest" in report.results[0].error
    assert report.metrics["latency_p50_ms"] is None
    assert report.metrics["errors"] == 1.0


def test_evaluate_configs_compares_blueprints(tmp_path: Path) -> None:
    configs = [EXAMPLES / "naive.yaml", EXAMPLES / "hybrid.yaml"]
    seen: set[str] = set()
    reports = evaluate_configs(
        configs, EXAMPLES / "eval.jsonl", data=DATA, progress=lambda label, d, t: seen.add(label)
    )
    assert [r.label for r in reports] == ["naive", "hybrid"]
    assert seen == {"naive", "hybrid"}
    assert [r.blueprint for r in reports] == ["naive", "hybrid"]
    # the keyword-heavy dataset favours hybrid retrieval
    assert (reports[1].metrics["mrr"] or 0) >= (reports[0].metrics["mrr"] or 0)


def test_evaluate_configs_auto_ingests_and_dedupes_labels(tmp_path: Path) -> None:
    cfg = tmp_path / "a.yaml"
    cfg.write_text(
        f"name: same\nblueprint: naive\ndata: {DATA}\nindex_dir: idx\n"
        "eval:\n  judge: {type: fake}\n",
        encoding="utf-8",
    )
    cfg2 = tmp_path / "b.yaml"
    cfg2.write_text(
        f"name: same\nblueprint: naive\ndata: {DATA}\nindex_dir: idx\n", encoding="utf-8"
    )
    reports = evaluate_configs([cfg, cfg2], EXAMPLES / "eval.jsonl")
    assert (tmp_path / "idx" / "ragpick.json").is_file()  # persisted on first use
    assert [r.label for r in reports] == ["same", "same (b.yaml)"]
    assert reports[0].metrics["faithfulness"] == 1.0  # fake judge's constant score
    assert reports[1].metrics["faithfulness"] is None


def test_evaluate_configs_without_index_or_data(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("blueprint: naive\nindex_dir: idx\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="pass --data"):
        evaluate_configs([cfg], EXAMPLES / "eval.jsonl")


def test_judge_construction() -> None:
    assert build_judge(None) is None
    assert build_judge(ComponentSpec(type="fake")) is not None
    from ragpick.registry import registry

    class NoComplete:
        def generate(self, query: str, contexts: list[RetrievedChunk]) -> Answer:
            return Answer(text="")

    registry.add("generator", "_nocomplete", NoComplete)
    try:
        with pytest.raises(ConfigError, match="cannot be used as a judge"):
            build_judge(ComponentSpec(type="_nocomplete"))
    finally:
        registry.remove("generator", "_nocomplete")
    assert parse_judge_option(None) is None
    assert parse_judge_option("openai:gpt-4o-mini") == ComponentSpec(
        type="openai", params={"model": "gpt-4o-mini"}
    )
    assert parse_judge_option("fake") == ComponentSpec(type="fake")


def test_write_reports(tmp_path: Path) -> None:
    reports = evaluate_configs([EXAMPLES / "naive.yaml"], EXAMPLES / "eval.jsonl", data=DATA)
    json_path = write_reports(reports, tmp_path / "out" / "results.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["reports"][0]["label"] == "naive"
    assert len(payload["reports"][0]["results"]) == reports[0].num_examples
    csv_path = write_reports(reports, tmp_path / "results.csv")
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["label"] == "naive"
    assert rows[0]["faithfulness"] == ""
    assert float(rows[0]["hit_rate@k"]) >= 0
    with pytest.raises(ConfigError, match="Unsupported output format"):
        write_reports(reports, tmp_path / "results.txt")
