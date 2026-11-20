"""ragpick bench: candidate configs, ranking, CLI."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from conftest import EXAMPLES
from ragpick.cli import app
from ragpick.config import build_config
from ragpick.errors import BlueprintError, ConfigError
from ragpick.eval.bench import ARCHITECTURE_COMPONENTS, _rank, candidate_raw_configs, run_bench
from ragpick.eval.runner import EvalReport

DATA = EXAMPLES / "data"
DATASET = EXAMPLES / "eval.jsonl"


def test_candidates_keep_providers_and_drop_architecture() -> None:
    base = {
        "name": "bot",
        "blueprint": "naive",
        "index_dir": "x",
        "requirements": {"keyword_heavy": True},
        "components": {
            "embedder": {"type": "fake", "params": {"dimension": 32}},
            "generator": {"type": "fake", "params": {"max_sentences": 1}},
            "retriever": {"type": "vector"},
            "reranker": {"type": "cross_encoder"},
            "chunker": {"type": "fixed"},
        },
        "query": {"top_k": 4},
    }
    raws = candidate_raw_configs(base, ["naive", "hybrid", "parent_child"])
    assert list(raws) == ["naive", "hybrid", "parent_child"]
    for name, raw in raws.items():
        assert raw["blueprint"] == name
        assert raw["name"] == name
        assert "index_dir" not in raw
        assert raw["query"]["top_k"] == 4  # same k for every candidate
        assert set(raw["components"]) == {"embedder", "generator"}
        assert not set(raw["components"]) & set(ARCHITECTURE_COMPONENTS)
        cfg = build_config(raw)
        assert cfg.components.embedder.params == {"dimension": 32}
        assert cfg.requirements.keyword_heavy
    assert raws["parent_child"]["query"]["top_k"] == 4  # overrides the blueprint's own top_k
    assert candidate_raw_configs(None, ["naive"], top_k=2)["naive"]["query"] == {"top_k": 2}
    assert candidate_raw_configs({}, ["naive"])["naive"]["query"] == {"top_k": 5}


def _report(label: str, mrr: float | None, latency: float) -> EvalReport:
    return EvalReport(
        label=label,
        blueprint=label,
        k=5,
        model="fake",
        num_examples=1,
        metrics={"mrr": mrr, "latency_p50_ms": latency, "faithfulness": None},
        results=[],
    )


def test_rank_by_metric_then_latency() -> None:
    reports = [
        _report("a", 0.5, 1.0),
        _report("b", 0.9, 9.0),
        _report("c", 0.9, 2.0),
        _report("d", None, 0),
    ]
    assert [r.label for r in _rank(reports, "mrr")] == ["c", "b", "a"]
    with pytest.raises(ConfigError, match="--judge"):
        _rank(reports, "faithfulness")


def test_run_bench_offline() -> None:
    result = run_bench(DATASET, DATA)
    assert set(result.ranking) == {
        "naive",
        "hybrid",
        "parent_child",
        "corrective",
        "graph",
        "agentic",
        "multimodal",
        "conversational",
    }  # no sql
    assert result.winner == result.ranking[0]
    assert result.metric == "mrr"
    best = result.report_for(result.winner).metrics["mrr"] or 0
    assert all((r.metrics["mrr"] or 0) <= best for r in result.reports)
    assert len({r.k for r in result.reports}) == 1
    written = yaml.safe_load(result.winner_config)
    assert written["blueprint"] == result.winner
    assert written["version"] == 1
    assert "components" not in written  # nothing to override with an empty base
    build_config(written)  # the written config is valid


def test_run_bench_with_base_config_and_subset(tmp_path: Path) -> None:
    base = tmp_path / "rag.yaml"
    base.write_text(
        "name: shop\nblueprint: naive\nindex_dir: prod-index\n"
        "components:\n  embedder: {type: fake, params: {dimension: 64}}\n"
        "eval:\n  judge: {type: fake}\n",
        encoding="utf-8",
    )
    result = run_bench(
        DATASET,
        DATA,
        base_config=base,
        blueprints=["naive", "hybrid"],
        metric="faithfulness",
        top_k=2,
    )
    assert result.ranking[0] in ("naive", "hybrid")
    assert all(r.k == 2 for r in result.reports)
    assert all(r.metrics["faithfulness"] == 1.0 for r in result.reports)
    written = yaml.safe_load(result.winner_config)
    assert written["name"] == "shop"
    assert written["index_dir"] == "prod-index"
    assert written["components"]["embedder"]["params"]["dimension"] == 64
    assert not (tmp_path / "prod-index").exists()  # bench never touches real indexes


def test_run_bench_validation() -> None:
    with pytest.raises(ConfigError, match="Unknown metric"):
        run_bench(DATASET, DATA, metric="vibes")
    with pytest.raises(BlueprintError, match="planned"):
        run_bench(DATASET, DATA, blueprints=["hierarchical"])


runner = CliRunner()


def test_bench_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shutil.copytree(EXAMPLES, tmp_path / "examples", ignore=shutil.ignore_patterns(".ragpick"))
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "bench",
            "examples/eval.jsonl",
            "--data",
            "examples/data",
            "-o",
            "bench.csv",
            "--write",
            "best.yaml",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "No base config" in result.output
    assert "Winner:" in result.output
    assert "Ranking:" in result.output
    assert (tmp_path / "bench.csv").is_file()
    assert yaml.safe_load((tmp_path / "best.yaml").read_text(encoding="utf-8"))["blueprint"]
    again = runner.invoke(
        app, ["bench", "examples/eval.jsonl", "--data", "examples/data", "--write", "best.yaml"]
    )
    assert again.exit_code == 1
    assert "already exists" in again.output
    # uses rag.yaml as the base config when present
    Path("rag.yaml").write_text("blueprint: naive\n", encoding="utf-8")
    third = runner.invoke(
        app, ["bench", "examples/eval.jsonl", "--data", "examples/data", "-b", "naive"]
    )
    assert third.exit_code == 0, third.output
    assert "No base config" not in third.output
