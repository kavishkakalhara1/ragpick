"""``ragpick bench``: run every stable blueprint on your data and pick the best.

A base config supplies the *provider* parts (loader, embedder, store, generator,
eval settings); each blueprint supplies the *architecture* (chunker, keyword index,
retriever, reranker). Every candidate uses the same ``query.top_k`` so metrics are
comparable.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from ragpick.blueprints import get_blueprint, list_blueprints
from ragpick.config import ComponentSpec, RagConfig, build_config, read_yaml
from ragpick.errors import BlueprintError, ConfigError
from ragpick.eval.runner import EvalReport, evaluate_configs

#: Components that define the architecture; taken from each blueprint, not the base.
ARCHITECTURE_COMPONENTS = ("chunker", "keyword_index", "retriever", "reranker")

#: Metrics bench can rank by (all higher-is-better).
RANKING_METRICS = ("hit_rate@k", "mrr", "precision@k", "faithfulness", "answer_relevance")


class BenchResult(BaseModel):
    """Outcome of a benchmark run."""

    metric: str
    winner: str
    ranking: list[str]
    reports: list[EvalReport]
    winner_config: str

    def report_for(self, blueprint: str) -> EvalReport:
        """The report of ``blueprint``."""
        return next(r for r in self.reports if r.label == blueprint)


def default_bench_blueprints() -> list[str]:
    """Stable blueprints that answer from documents (``sql`` needs tables and an LLM)."""
    return [
        b.name for b in list_blueprints(include_planned=False) if b.metadata.data == "documents"
    ]


def candidate_raw_configs(
    base: Mapping[str, Any] | None, blueprints: Sequence[str], top_k: int | None = None
) -> dict[str, dict[str, Any]]:
    """Raw (uninterpolated) config mappings, one per blueprint, sharing the base's providers."""
    base = copy.deepcopy(dict(base or {}))
    base.pop("blueprint", None)
    base.pop("index_dir", None)
    components = {
        name: spec
        for name, spec in (base.pop("components", None) or {}).items()
        if name not in ARCHITECTURE_COMPONENTS
    }
    query = dict(base.pop("query", None) or {})
    query["top_k"] = top_k or query.get("top_k", 5)
    out: dict[str, dict[str, Any]] = {}
    for name in blueprints:
        out[name] = {
            **copy.deepcopy(base),
            "name": name,
            "blueprint": name,
            "components": copy.deepcopy(components),
            "query": dict(query),
        }
    return out


def _rank(reports: Sequence[EvalReport], metric: str) -> list[EvalReport]:
    scored = [r for r in reports if r.metrics.get(metric) is not None]
    if not scored:
        hint = " (configure eval.judge or pass --judge)" if metric in RANKING_METRICS[3:] else ""
        raise ConfigError(f"No blueprint produced a value for '{metric}'{hint}")
    order = {r.label: i for i, r in enumerate(reports)}

    def key(r: EvalReport) -> tuple[float, float, int]:
        latency = r.metrics.get("latency_p50_ms")
        return (
            -(r.metrics[metric] or 0.0),
            latency if latency is not None else 0.0,
            order[r.label],
        )

    return sorted(scored, key=key)


def run_bench(
    dataset: str | Path,
    data: str | Path,
    *,
    base_config: str | Path | None = None,
    blueprints: Sequence[str] | None = None,
    metric: str = "mrr",
    top_k: int | None = None,
    judge: ComponentSpec | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> BenchResult:
    """Benchmark blueprints on ``dataset`` with documents from ``data``.

    Ties on ``metric`` are broken by lower median latency. Each blueprint gets a
    temporary index, so existing indexes are never touched.
    """
    if metric not in RANKING_METRICS:
        raise ConfigError(f"Unknown metric '{metric}'. Choose one of: {', '.join(RANKING_METRICS)}")
    names = list(blueprints) if blueprints else default_bench_blueprints()
    for name in names:
        if not get_blueprint(name).is_stable:
            raise BlueprintError(f"Blueprint '{name}' is planned and cannot be benchmarked yet")
    base_raw = read_yaml(base_config) if base_config is not None else {}
    base_dir = Path(base_config).resolve().parent if base_config is not None else None
    raws = candidate_raw_configs(base_raw, names, top_k)
    configs: list[RagConfig] = [
        build_config(raw, source=f"bench:{name}", base_dir=base_dir) for name, raw in raws.items()
    ]
    reports = evaluate_configs(configs, dataset, data=data, judge=judge, progress=progress)
    ranked = _rank(reports, metric)
    winner = ranked[0].label
    winner_raw = raws[winner]
    winner_raw.pop("name", None)
    winner_raw = {"version": 1, "name": base_raw.get("name", "ragpick-app"), **winner_raw}
    if not winner_raw.get("components"):
        winner_raw.pop("components", None)
    if "index_dir" in base_raw:
        winner_raw["index_dir"] = base_raw["index_dir"]
    return BenchResult(
        metric=metric,
        winner=winner,
        ranking=[r.label for r in ranked],
        reports=reports,
        winner_config=yaml.safe_dump(winner_raw, sort_keys=False),
    )
