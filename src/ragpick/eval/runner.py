"""Run pipelines over an evaluation dataset and aggregate the results."""

from __future__ import annotations

import csv
import json
import logging
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ragpick.config import (
    ComponentSpec,
    RagConfig,
    is_router_config,
    load_config,
    load_router_config,
    read_yaml,
)
from ragpick.errors import ConfigError, RagpickError
from ragpick.eval.dataset import EvalExample, load_dataset
from ragpick.eval.metrics import (
    LLMJudge,
    estimate_cost,
    hit_rate_at_k,
    mean,
    percentile,
    precision_at_k,
    reciprocal_rank,
)
from ragpick.interfaces import TextCompleter
from ragpick.models import Usage
from ragpick.pipeline import Pipeline
from ragpick.registry import registry
from ragpick.router import Router

logger = logging.getLogger(__name__)

#: Aggregate metric names, in display order.
METRICS = (
    "hit_rate@k",
    "mrr",
    "precision@k",
    "faithfulness",
    "answer_relevance",
    "latency_p50_ms",
    "latency_p95_ms",
    "total_tokens",
    "est_cost_usd",
    "errors",
)


class ExampleResult(BaseModel):
    """Per-question result."""

    question: str
    answer: str = ""
    route: str | None = None
    citations: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    relevant_doc_ids: list[str] | None = None
    ground_truth: str | None = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    hit: float | None = None
    reciprocal_rank: float | None = None
    precision: float | None = None
    faithfulness: float | None = None
    faithfulness_reason: str | None = None
    answer_relevance: float | None = None
    answer_relevance_reason: str | None = None
    error: str | None = None


class EvalReport(BaseModel):
    """Aggregated results of one pipeline over a dataset."""

    label: str
    blueprint: str | None
    config_path: str | None = None
    k: int
    model: str
    num_examples: int
    metrics: dict[str, float | None]
    results: list[ExampleResult]


def build_judge(spec: ComponentSpec | None) -> LLMJudge | None:
    """Create an :class:`LLMJudge` from a generator spec (``None`` → no judge)."""
    if spec is None:
        return None
    completer = registry.create("generator", spec.type, **spec.params)
    if not isinstance(completer, TextCompleter):
        raise ConfigError(
            f"eval.judge: generator '{spec.type}' cannot be used as a judge "
            "(it does not implement `complete`)"
        )
    return LLMJudge(completer)


def parse_judge_option(value: str | None) -> ComponentSpec | None:
    """Parse a CLI ``--judge`` value: ``"openai"`` or ``"openai:gpt-4o-mini"``."""
    if not value:
        return None
    type_, _, model = value.partition(":")
    return ComponentSpec(type=type_, params={"model": model} if model else {})


def _engine_prices(engine: Any) -> dict[str, Any]:
    if isinstance(engine, Pipeline):
        return dict(engine.config.eval.prices)
    eval_config = getattr(engine, "eval_config", None)
    return dict(eval_config.prices) if eval_config is not None else {}


def evaluate(
    pipeline: Any,
    dataset: Sequence[EvalExample],
    *,
    judge: LLMJudge | None = None,
    label: str | None = None,
    config_path: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> EvalReport:
    """Evaluate ``pipeline`` (a :class:`Pipeline` or :class:`~ragpick.router.Router`).

    Retrieval metrics are computed for examples with ``relevant_doc_ids``; LLM
    metrics only when a ``judge`` is given. Errors raised by ragpick components are
    recorded per example and counted in the ``errors`` metric.
    """
    k = pipeline.top_k
    results: list[ExampleResult] = []
    usages: list[Usage] = []
    for i, example in enumerate(dataset):
        result = ExampleResult(
            question=example.question,
            relevant_doc_ids=example.relevant_doc_ids,
            ground_truth=example.ground_truth,
        )
        start = time.perf_counter()
        try:
            answer = pipeline.query(example.question, groups=example.groups)
        except RagpickError as exc:
            result.error = str(exc)
            logger.warning("Query failed for %r: %s", example.question, exc)
        else:
            result.latency_ms = (time.perf_counter() - start) * 1000
            result.answer = answer.text
            result.route = answer.route
            result.citations = answer.citations
            result.retrieved_doc_ids = [
                c.chunk.doc_id for c in answer.contexts if not c.chunk.metadata.get("synthetic")
            ]
            result.prompt_tokens = answer.usage.prompt_tokens
            result.completion_tokens = answer.usage.completion_tokens
            usages.append(answer.usage)
            if example.relevant_doc_ids:
                docs, rel = result.retrieved_doc_ids, example.relevant_doc_ids
                result.hit = hit_rate_at_k(docs, rel, k)
                result.reciprocal_rank = reciprocal_rank(docs, rel)
                result.precision = precision_at_k(docs, rel, k)
            if judge is not None:
                try:
                    faith = judge.faithfulness(answer.text, answer.contexts)
                    relevance = judge.answer_relevance(example.question, answer.text)
                except RagpickError as exc:
                    result.error = f"judge failed: {exc}"
                else:
                    result.faithfulness, result.faithfulness_reason = faith.score, faith.reason
                    result.answer_relevance = relevance.score
                    result.answer_relevance_reason = relevance.reason
        results.append(result)
        if progress:
            progress(i + 1, len(dataset))

    latencies = [r.latency_ms for r in results if r.error is None]
    model = pipeline.model_name
    metrics: dict[str, float | None] = {
        "hit_rate@k": mean([r.hit for r in results]),
        "mrr": mean([r.reciprocal_rank for r in results]),
        "precision@k": mean([r.precision for r in results]),
        "faithfulness": mean([r.faithfulness for r in results]),
        "answer_relevance": mean([r.answer_relevance for r in results]),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "total_tokens": float(sum(u.total_tokens for u in usages)),
        "est_cost_usd": estimate_cost(usages, model, _engine_prices(pipeline)),
        "errors": float(sum(1 for r in results if r.error)),
    }
    return EvalReport(
        label=label or pipeline.name,
        blueprint=pipeline.blueprint,
        config_path=config_path,
        k=k,
        model=model,
        num_examples=len(results),
        metrics=metrics,
        results=results,
    )


def evaluate_configs(
    config_paths: Sequence[str | Path | RagConfig],
    dataset_path: str | Path,
    *,
    data: str | Path | None = None,
    judge: ComponentSpec | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> list[EvalReport]:
    """Evaluate several configs (files or :class:`RagConfig` objects) on one dataset.

    Indexing per config:

    * ``data`` given → ingest it into a temporary index (existing indexes untouched);
    * otherwise use the config's persisted index, ingesting the config's ``data``
      path first if no index exists yet.

    ``judge`` overrides each config's ``eval.judge``.
    """
    dataset = load_dataset(dataset_path)
    reports: list[EvalReport] = []
    labels: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="ragpick-eval-") as tmp:
        for n, path in enumerate(config_paths):
            overrides: dict[str, Any] = {}
            if data is not None:
                overrides["index_dir"] = str(Path(tmp) / f"index-{n}")
            if not isinstance(path, RagConfig) and is_router_config(read_yaml(path)):
                if data is not None:
                    raise ConfigError(
                        f"{path} is a router: it ingests each route's own `data`; omit --data"
                    )
                router = Router.from_config(load_router_config(path))
                if not router.ready():
                    router.ingest()
                label = router.name if router.name not in labels else f"{router.name} #{n + 1}"
                labels.add(label)
                router_judge = build_judge(judge or router.eval_config.judge)

                def router_progress(done: int, total: int, label: str = label) -> None:
                    if progress:
                        progress(label, done, total)

                reports.append(
                    evaluate(
                        router,
                        dataset,
                        judge=router_judge,
                        label=label,
                        config_path=str(path),
                        progress=router_progress,
                    )
                )
                continue
            if isinstance(path, RagConfig):
                config = path.model_copy(deep=True).with_base_dir(path.base_dir)
                if data is not None:
                    config.index_dir = overrides["index_dir"]
                source = config.name
            else:
                config = load_config(path, overrides)
                source = str(path)
            pipeline = Pipeline(config)
            if data is not None:
                pipeline.ingest(data)
            elif not pipeline.ready():
                if config.data is None:
                    raise ConfigError(
                        f"{source}: no index at {pipeline.index_dir} and no `data` path in the "
                        "config. Run `ragpick ingest`, set `data`, or pass --data."
                    )
                pipeline.ingest()
            label = config.name
            if label in labels:
                label = f"{label} ({Path(source).name})" if source != label else f"{label} #{n + 1}"
            labels.add(label)
            judge_ = build_judge(judge or config.eval.judge)

            def report_progress(done: int, total: int, label: str = label) -> None:
                if progress:
                    progress(label, done, total)

            reports.append(
                evaluate(
                    pipeline,
                    dataset,
                    judge=judge_,
                    label=label,
                    config_path=None if isinstance(path, RagConfig) else str(path),
                    progress=report_progress,
                )
            )
    return reports


def parse_thresholds(items: Sequence[str]) -> dict[str, float]:
    """Parse CLI thresholds like ``["hit_rate@k=0.8", "mrr=0.5"]``.

    Raises:
        ConfigError: malformed items or unknown metrics.
    """
    out: dict[str, float] = {}
    for item in items:
        name, sep, raw = item.partition("=")
        name = name.strip()
        if not sep or name not in METRICS:
            raise ConfigError(
                f"Invalid threshold '{item}': use METRIC=VALUE with one of: {', '.join(METRICS)}"
            )
        try:
            out[name] = float(raw)
        except ValueError:
            raise ConfigError(f"Invalid threshold value in '{item}'") from None
    return out


def check_thresholds(
    reports: Sequence[EvalReport],
    fail_under: dict[str, float] | None = None,
    fail_over: dict[str, float] | None = None,
) -> list[str]:
    """Human-readable threshold violations (empty when everything passes).

    A metric that is required but missing (e.g. faithfulness without a judge) counts
    as a violation, so a misconfigured CI job can't pass silently.
    """
    failures: list[str] = []
    for metric in [*(fail_under or {}), *(fail_over or {})]:
        if metric not in METRICS:
            raise ConfigError(f"Unknown metric '{metric}' in thresholds")
    for report in reports:
        for metric, minimum in (fail_under or {}).items():
            value = report.metrics.get(metric)
            if value is None:
                failures.append(
                    f"{report.label}: {metric} is not available (required >= {minimum:g})"
                )
            elif value < minimum:
                failures.append(f"{report.label}: {metric} = {value:.3f} < {minimum:g}")
        for metric, maximum in (fail_over or {}).items():
            value = report.metrics.get(metric)
            if value is None:
                failures.append(
                    f"{report.label}: {metric} is not available (required <= {maximum:g})"
                )
            elif value > maximum:
                failures.append(f"{report.label}: {metric} = {value:.3f} > {maximum:g}")
    return failures


def write_reports(reports: Sequence[EvalReport], path: str | Path) -> Path:
    """Write reports as JSON (full detail) or CSV (one aggregate row per config).

    The format is chosen by the file extension (``.json`` or ``.csv``).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()
    if suffix == ".json":
        payload = {"reports": [r.model_dump() for r in reports]}
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif suffix == ".csv":
        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["label", "blueprint", "model", "k", "num_examples", *METRICS])
            for r in reports:
                row = [r.label, r.blueprint or "", r.model, r.k, r.num_examples]
                row += ["" if r.metrics.get(m) is None else r.metrics[m] for m in METRICS]
                writer.writerow(row)
    else:
        raise ConfigError(f"Unsupported output format '{p.suffix}': use .json or .csv")
    return p
