"""Rule-based, explainable blueprint selector.

Rules live in the :data:`RULES` table. Each rule has a condition over a
:class:`SelectionContext` and may:

* vote for a blueprint (the matching rule with the highest ``priority`` wins);
* add a reason, a warning and alternatives;
* override components (e.g. disable the reranker for low latency).

If the winning blueprint is only *planned*, the rule's ``fallback`` is used and a
warning is emitted. To extend the selector, append a :class:`Rule` to
:data:`RULES` (or pass your own table to :func:`recommend`).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ragpick.blueprints import get_blueprint
from ragpick.config import Requirements
from ragpick.config import deep_merge as _deep_merge
from ragpick.registry import Registry, registry

STRUCTURED_TYPES = frozenset({"sql", "csv", "database", "structured", "tables", "spreadsheet"})
DOCUMENT_TYPES = frozenset({"markdown", "md", "text", "txt", "pdf"})
FILTERING_STORES_HINT = "memory, qdrant"


@dataclass(frozen=True)
class SelectionContext:
    """Everything a rule may look at."""

    requirements: Requirements
    store: str = "memory"
    store_supports_filters: bool = True
    installed: Callable[[str], bool] = field(
        default=lambda module: importlib.util.find_spec(module) is not None
    )

    @property
    def data_types(self) -> set[str]:
        """Lowercased data types."""
        return {d.lower() for d in self.requirements.data_types}

    def fmt(self, template: str) -> str:
        """Format ``template`` with requirement fields, ``store`` and helper values."""
        req = self.requirements
        return template.format(
            **req.model_dump(),
            store=self.store,
            structured=", ".join(sorted(self.data_types & STRUCTURED_TYPES)),
            unknown=", ".join(sorted(self.data_types - STRUCTURED_TYPES - DOCUMENT_TYPES)),
            filtering_stores=FILTERING_STORES_HINT,
        )


@dataclass(frozen=True)
class Rule:
    """One row of the selector's rule table."""

    id: str
    when: Callable[[SelectionContext], bool]
    reason: str | None = None
    blueprint: str | None = None
    priority: int = 0
    fallback: str | None = None
    warning: str | None = None
    alternatives: tuple[str, ...] = ()
    overrides: Mapping[str, Any] = field(default_factory=dict)


class Recommendation(BaseModel):
    """The selector's output."""

    blueprint: str
    reasons: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(
        default_factory=dict, description="Component overrides to apply on top of the blueprint."
    )
    matched_rules: list[str] = Field(default_factory=list)


def _r(ctx: SelectionContext) -> Requirements:
    return ctx.requirements


RULES: list[Rule] = [
    # ---- blueprint votes ------------------------------------------------------
    Rule(
        id="structured_data",
        when=lambda c: bool(c.data_types & STRUCTURED_TYPES),
        blueprint="sql",
        priority=100,
        fallback="hybrid",
        reason=(
            "Structured data ({structured}) is best answered with text-to-SQL: aggregations "
            "and filters over rows are not a retrieval problem."
        ),
        alternatives=("hybrid",),
    ),
    Rule(
        id="keyword_heavy",
        when=lambda c: _r(c).keyword_heavy,
        blueprint="hybrid",
        priority=50,
        reason=(
            "Keyword-heavy queries (product codes, legal terms, IDs) need exact matching: "
            "BM25 catches terms that embeddings blur, and Reciprocal Rank Fusion combines "
            "both rankings."
        ),
        alternatives=("naive",),
    ),
    Rule(
        id="medium_or_large_corpus",
        when=lambda c: _r(c).corpus_size in ("medium", "large"),
        blueprint="hybrid",
        priority=40,
        reason=(
            "A {corpus_size} corpus has many near-duplicate passages; hybrid retrieval "
            "(BM25 + vectors) is more robust than vector search alone at this scale."
        ),
        alternatives=("naive",),
    ),
    Rule(
        id="small_simple_corpus",
        when=lambda c: _r(c).corpus_size == "small" and not _r(c).keyword_heavy,
        blueprint="naive",
        priority=30,
        reason=(
            "A small corpus without keyword-heavy queries is served well by plain vector "
            "search, the simplest blueprint to run and debug."
        ),
        alternatives=("hybrid",),
    ),
    # ---- adjustments ------------------------------------------------------------
    Rule(
        id="low_latency",
        when=lambda c: _r(c).latency == "low",
        reason=(
            "Low latency required: the reranker is disabled, because a cross-encoder "
            "typically adds 50-300 ms per query."
        ),
        overrides={"reranker": {"type": "none"}},
    ),
    Rule(
        id="rerank_for_precision",
        when=lambda c: (
            _r(c).latency == "normal" and (_r(c).keyword_heavy or _r(c).corpus_size != "small")
        ),
        alternatives=("hybrid + cross_encoder reranker for higher precision (ragpick[rerank])",),
    ),
    Rule(
        id="large_corpus_memory_store",
        when=lambda c: _r(c).corpus_size == "large" and c.store == "memory",
        warning=(
            "The in-memory store keeps every vector in RAM and searches exhaustively; for "
            ">100k chunks use a vector database, e.g. `store: {{type: qdrant}}` "
            "(pip install 'ragpick[qdrant]')."
        ),
    ),
    Rule(
        id="citations",
        when=lambda c: _r(c).needs_citations,
        reason=(
            "Citations: contexts are numbered in the prompt and [n] markers are mapped back "
            "to chunk IDs in `Answer.citations`."
        ),
    ),
    Rule(
        id="access_control",
        when=lambda c: _r(c).access_control and c.store_supports_filters,
        reason=(
            "Access control: chunks with `allowed_groups` metadata are filtered inside the "
            "'{store}' store and BM25 before ranking, so restricted text never reaches the LLM."
        ),
    ),
    Rule(
        id="access_control_unsupported_store",
        when=lambda c: _r(c).access_control and not c.store_supports_filters,
        warning=(
            "Store '{store}' does not support metadata filters, so `allowed_groups` access "
            "control cannot be enforced. Use a store that does ({filtering_stores})."
        ),
    ),
    Rule(
        id="pdf_extra_missing",
        when=lambda c: "pdf" in c.data_types and not c.installed("pypdf"),
        warning="PDF files need the pdf extra: pip install 'ragpick[pdf]'.",
    ),
    Rule(
        id="long_documents",
        when=lambda c: "pdf" in c.data_types,
        alternatives=("parent_child (planned): small chunks for search, full sections as context",),
    ),
    Rule(
        id="unknown_data_types",
        when=lambda c: bool(c.data_types - STRUCTURED_TYPES - DOCUMENT_TYPES),
        warning=(
            "No built-in loader for: {unknown}. Built-in loaders handle markdown, text and "
            "pdf; convert the files or register a custom loader plugin."
        ),
    ),
]


def _store_supports_filters(store: str, reg: Registry) -> bool:
    if not reg.has("store", store):
        return False
    return bool(getattr(reg.get("store", store), "supports_filters", False))


def recommend(
    requirements: Requirements | Mapping[str, Any],
    *,
    store: str = "memory",
    rules: Sequence[Rule] | None = None,
    component_registry: Registry | None = None,
    installed: Callable[[str], bool] | None = None,
) -> Recommendation:
    """Recommend a blueprint for ``requirements``, with reasons and warnings.

    Args:
        requirements: a :class:`~ragpick.config.Requirements` or a mapping of its fields.
        store: the vector store type that will be used (for filter-support checks).
        rules: rule table to evaluate (defaults to :data:`RULES`).
        component_registry: registry used to look up the store's capabilities.
        installed: ``module -> bool`` predicate (defaults to an import-spec check).
    """
    req = (
        requirements
        if isinstance(requirements, Requirements)
        else Requirements.model_validate(requirements)
    )
    reg = component_registry or registry
    ctx_kwargs: dict[str, Any] = {
        "requirements": req,
        "store": store,
        "store_supports_filters": _store_supports_filters(store, reg),
    }
    if installed is not None:
        ctx_kwargs["installed"] = installed
    ctx = SelectionContext(**ctx_kwargs)
    table = list(RULES if rules is None else rules)
    matched = [rule for rule in table if rule.when(ctx)]

    voters = [rule for rule in matched if rule.blueprint]
    if not voters:
        raise ValueError("No selector rule chose a blueprint; add a default rule to the table")
    winner = max(voters, key=lambda r: r.priority)  # first one wins on ties
    blueprint = winner.blueprint
    assert blueprint is not None
    warnings: list[str] = []
    if not get_blueprint(blueprint).is_stable:
        fallback = winner.fallback or "naive"
        warnings.append(
            f"The ideal blueprint '{blueprint}' is planned for a future release; "
            f"falling back to '{fallback}'."
        )
        blueprint = fallback

    reasons: list[str] = []
    alternatives: list[str] = []
    overrides: dict[str, Any] = {}
    for rule in matched:
        votes_elsewhere = rule.blueprint not in (None, blueprint) and rule is not winner
        if rule.reason and not votes_elsewhere:
            reasons.append(ctx.fmt(rule.reason))
        if rule.warning:
            warnings.append(ctx.fmt(rule.warning))
        for alt in rule.alternatives:
            if alt != blueprint and alt not in alternatives:
                alternatives.append(alt)
        if rule.overrides:
            overrides = _deep_merge(overrides, rule.overrides)
    if winner.blueprint != blueprint and winner.blueprint not in alternatives:
        alternatives.insert(0, f"{winner.blueprint} (planned)")

    return Recommendation(
        blueprint=blueprint,
        reasons=reasons,
        alternatives=alternatives,
        warnings=warnings,
        overrides=overrides,
        matched_rules=[rule.id for rule in matched],
    )
