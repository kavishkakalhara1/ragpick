"""CLI commands via typer's CliRunner (fully offline)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from conftest import EXAMPLES
from ragpick import __version__
from ragpick.cli import _mask_secrets, app

runner = CliRunner()


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temp project dir containing a copy of examples/ as the cwd."""
    shutil.copytree(EXAMPLES, tmp_path / "examples", ignore=shutil.ignore_patterns(".ragpick"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run(*args: str, input: str | None = None) -> tuple[int, str]:
    result = runner.invoke(app, list(args), input=input)
    return result.exit_code, result.output


def test_version_and_help() -> None:
    code, out = run("--version")
    assert code == 0
    assert __version__ in out
    code, out = run("--help")
    assert code == 0
    for cmd in ("init", "recommend", "blueprints", "ingest", "query", "eval"):
        assert cmd in out


def test_acceptance_flow_offline(workdir: Path) -> None:
    """init --blueprint naive --yes → ingest examples/data → query returns citations."""
    code, out = run("init", "--blueprint", "naive", "--yes")
    assert code == 0, out
    config = yaml.safe_load((workdir / "rag.yaml").read_text(encoding="utf-8"))
    assert config["blueprint"] == "naive"
    assert config["components"]["embedder"]["type"] == "fake"
    assert config["components"]["generator"]["type"] == "fake"

    code, out = run("ingest", "examples/data")
    assert code == 0, out
    assert "Ingested 9 documents" in out

    code, out = run("query", "How do I reset my password?", "--show-contexts")
    assert code == 0, out
    assert "Forgot password" in out
    assert "Citations: account-and-password.md#0" in out
    assert "Contexts" in out


def test_init_refuses_overwrite_and_validates(workdir: Path) -> None:
    assert run("init", "--yes")[0] == 0
    code, out = run("init", "--yes")
    assert code == 1
    assert "already exists" in out
    code, out = run("init", "--yes", "--force", "--blueprint", "agentic")
    assert code == 1
    assert "planned" in out
    code, out = run("init", "--yes", "--force", "--provider", "nope")
    assert code == 1
    assert "Unknown provider" in out


@pytest.mark.parametrize(
    ("provider", "embedder", "generator", "hint"),
    [
        ("openai", "openai", "openai", "OPENAI_API_KEY"),
        ("ollama", "sentence_transformers", "ollama", "ollama serve"),
    ],
)
def test_init_providers(
    workdir: Path, provider: str, embedder: str, generator: str, hint: str
) -> None:
    code, out = run(
        "init", "--yes", "--provider", provider, "--blueprint", "hybrid", "-o", "x.yaml"
    )
    assert code == 0, out
    assert hint in out
    text = (workdir / "x.yaml").read_text(encoding="utf-8")
    assert "sk-" not in text  # never writes secrets
    config = yaml.safe_load(text)
    assert config["components"]["embedder"]["type"] == embedder
    assert config["components"]["generator"]["type"] == generator


def test_init_interactive(workdir: Path) -> None:
    answers = "\n".join(
        [
            "markdown,pdf",  # data types
            "medium",  # corpus size
            "y",  # keyword heavy
            "n",  # long documents
            "n",  # multi-hop
            "n",  # strict grounding
            "low",  # latency
            "y",  # citations
            "y",  # access control
            "",  # blueprint: accept recommendation
            "fake",  # provider
            "support-bot",  # name
        ]
    )
    code, out = run("init", input=answers + "\n")
    assert code == 0, out
    assert "ragpick recommends" in out
    config = yaml.safe_load((workdir / "rag.yaml").read_text(encoding="utf-8"))
    assert config["name"] == "support-bot"
    assert config["blueprint"] == "hybrid"
    assert config["requirements"]["corpus_size"] == "medium"
    assert config["requirements"]["access_control"] is True
    assert config["components"]["reranker"] == {"type": "none"}  # low latency override


def test_recommend(workdir: Path) -> None:
    code, out = run("recommend")
    assert code == 0
    assert "No rag.yaml found" in out
    assert "Blueprint: naive" in out
    code, out = run("recommend", "--config", "examples/hybrid.yaml")
    assert code == 0, out
    assert "Blueprint: hybrid" in out
    assert "BM25" in out
    assert "[n] markers" in out  # rich markup is escaped
    assert "ragpick[rerank]" in out


def test_recommend_reads_store_from_config(workdir: Path) -> None:
    Path("rag.yaml").write_text(
        "requirements: {corpus_size: large}\ncomponents: {store: {type: qdrant}}\n",
        encoding="utf-8",
    )
    code, out = run("recommend")
    assert code == 0, out
    assert "in-memory store" not in out
    Path("plain.yaml").write_text("requirements: {corpus_size: large}\n", encoding="utf-8")
    assert "in-memory store" in run("recommend", "-c", "plain.yaml")[1]


def test_blueprints_lists_roadmap() -> None:
    code, out = run("blueprints")
    assert code == 0
    for name in ("naive", "hybrid", "parent_child", "sql", "graph", "agentic", "corrective"):
        assert name in out
    assert "planned" in out
    assert "stable" in out


def test_query_with_groups(workdir: Path) -> None:
    cfg = "examples/hybrid.yaml"
    assert run("ingest", "--config", cfg)[0] == 0  # uses `data` from the config
    question = "When should support escalate an E-3301 ticket to Tier 2?"
    code, out = run("query", question, "--config", cfg, "--top-k", "9")
    assert code == 0
    assert "internal-escalation.md" not in out
    code, out = run("query", question, "--config", cfg, "--groups", "support")
    assert code == 0
    assert "internal-escalation.md" in out


def test_query_without_index_is_friendly(workdir: Path) -> None:
    run("init", "--yes")
    code, out = run("query", "hello")
    assert code == 1
    assert "Run `ragpick ingest PATH` first" in out
    assert "Traceback" not in out


def test_missing_config_is_friendly(workdir: Path) -> None:
    code, out = run("ingest", "examples/data")
    assert code == 1
    assert "Config file not found" in out
    assert "ragpick init" in out


def test_verbose_shows_traceback(workdir: Path) -> None:
    code, out = run("--verbose", "ingest", "examples/data")
    assert code == 1
    assert "Traceback" in out


def test_unexpected_error_is_summarised(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run("init", "--yes")

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr("ragpick.cli.load_engine", boom)
    code, out = run("query", "x")
    assert code == 1
    assert "Unexpected error: RuntimeError: kaboom" in out
    assert "--verbose" in out


def test_invalid_config_shows_key_path(workdir: Path) -> None:
    Path("rag.yaml").write_text("blueprint: naive\nquery: {top_k: zero}\n", encoding="utf-8")
    code, out = run("query", "x")
    assert code == 1
    assert "query.top_k" in out


def test_set_overrides(workdir: Path) -> None:
    run("init", "--yes")
    code, out = run("ingest", "examples/data", "--set", "index_dir=custom-index")
    assert code == 0, out
    assert (workdir / "custom-index" / "ragpick.json").is_file()


def test_eval_compares_configs(workdir: Path) -> None:
    code, out = run(
        "eval",
        "examples/eval.jsonl",
        "--config",
        "examples/naive.yaml",
        "--config",
        "examples/hybrid.yaml",
        "--output",
        "results.json",
    )
    assert code == 0, out
    assert "Evaluation results" in out
    assert "hit_rate@k" in out
    assert "naive" in out
    assert "hybrid" in out
    payload = json.loads((workdir / "results.json").read_text(encoding="utf-8"))
    assert [r["label"] for r in payload["reports"]] == ["naive", "hybrid"]


def test_eval_with_data_judge_and_csv(workdir: Path) -> None:
    run("init", "--yes")
    code, out = run(
        "eval", "examples/eval.jsonl", "--data", "examples/data", "--judge", "fake", "-o", "r.csv"
    )
    assert code == 0, out
    assert "faithfulness" in out
    assert "1.000" in out
    assert (workdir / "r.csv").is_file()


def test_validate_masks_secrets(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_KEY", "sk-super-secret")
    Path("rag.yaml").write_text(
        "blueprint: naive\n"
        "components:\n  generator: {type: fake, params: {api_key: '${MY_KEY}'}}\n",
        encoding="utf-8",
    )
    # FakeGenerator does not accept api_key -> validation passes, creation is not attempted
    code, out = run("validate")
    assert code == 0, out
    assert "sk-super-secret" not in out
    assert "***" in out
    assert _mask_secrets({"token": "", "list": [{"password": "p"}], "n": 1}) == {
        "token": "",
        "list": [{"password": "***"}],
        "n": 1,
    }
