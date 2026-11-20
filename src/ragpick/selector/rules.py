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

STRUCTURED_TYPES = frozenset(
    {"sql", "sqlite", "csv", "database", "structured", "tables", "spreadsheet"}
)
DOCUMENT_TYPES = frozenset({"markdown", "md", "text", "txt", "pdf"})
IMAGE_TYPES = frozenset(
    {
        "image",
        "images",
        "png",
        "jpg",
        "jpeg",
        "diagrams",
        "screenshots",
        "scanned",
        "charts",
        "slides",
    }
)
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
            unknown=", ".join(
                sorted(self.data_types - STRUCTURED_TYPES - DOCUMENT_TYPES - IMAGE_TYPES)
            ),
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
            "and filters over rows are not a retrieval problem. The query runs read-only, "
            "with a timeout and a row cap."
        ),
        alternatives=("hybrid",),
    ),
    Rule(
        id="images",
        when=lambda c: bool(c.data_types & IMAGE_TYPES),
        blueprint="multimodal",
        priority=80,
        reason=(
            "Images (diagrams, screenshots, scans) hold answers text search can't see: the "
            "multimodal loader describes every image at ingest so it is retrieved with the text."
        ),
        warning=(
            "Image descriptions need a vision-capable generator (openai, or ollama with a "
            "vision model); offline, only file names, alt text and captions are indexed."
        ),
        alternatives=("hybrid",),
    ),
    Rule(
        id="strict_grounding",
        when=lambda c: _r(c).strict_grounding,
        blueprint="corrective",
        priority=70,
        reason=(
            "Strict grounding: every retrieved context is graded for relevance; poor "
            "retrieval triggers a rewritten query, and if that also fails the answer is "
            '"I don\'t know" rather than a guess.'
        ),
        alternatives=("hybrid",),
    ),
    Rule(
        id="multi_hop",
        when=lambda c: _r(c).multi_hop,
        blueprint="graph",
        priority=65,
        reason=(
            "Multi-hop questions: a knowledge graph links entities across documents, so "
            "retrieval can follow relations (product → part → error code) that no single "
            "chunk states, and the LLM receives the connecting facts."
        ),
        alternatives=("hybrid",),
    ),
    Rule(
        id="complex_questions",
        when=lambda c: _r(c).complex_questions,
        blueprint="agentic",
        priority=62,
        reason=(
            "Multi-part and comparison questions: the agentic retriever plans several "
            "searches (one per sub-question or follow-up) and fuses what it finds."
        ),
        alternatives=("hybrid",),
    ),
    Rule(
        id="agentic_low_latency",
        when=lambda c: _r(c).complex_questions and _r(c).latency == "low",
        warning=(
            "agentic runs up to max_steps searches (plus a planning LLM call each); lower "
            "max_steps or measure with `ragpick bench` against your latency budget."
        ),
    ),
    Rule(
        id="multi_tenant",
        when=lambda c: _r(c).multi_tenant,
        reason=(
            "Multi-tenant: enable `tenancy: {{enabled: true}}` so every tenant gets an "
            "isolated index, and bind API keys to tenants in `serve.api_keys`."
        ),
    ),
    Rule(
        id="conversational",
        when=lambda c: _r(c).conversational,
        blueprint="conversational",
        priority=35,
        reason=(
            "Follow-up questions: chat() rewrites each follow-up into a standalone question "
            "using the session history. This works with every blueprint; the conversational "
            "blueprint adds hybrid retrieval and persistent sessions."
        ),
    ),
    Rule(
        id="multi_hop_strict",
        when=lambda c: _r(c).multi_hop and _r(c).strict_grounding,
        warning=(
            "strict_grounding wins over multi_hop (corrective is recommended); compare it "
            "with graph using `ragpick bench -b corrective -b graph`."
        ),
    ),
    Rule(
        id="long_documents",
        when=lambda c: _r(c).long_documents,
        blueprint="parent_child",
        priority=60,
        reason=(
            "Long, structured documents: small child chunks are searched precisely "
            "(hybrid BM25 + vector), and their full parent sections are passed to the "
            "LLM so answers keep the surrounding context."
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
        id="corrective_long_documents",
        when=lambda c: _r(c).strict_grounding and _r(c).long_documents,
        reason=(
            "Long documents: the corrective retriever searches with parent_child "
            "(small chunks for search, whole sections as context)."
        ),
        overrides={
            "chunker": {"type": "parent_child"},
            "retriever": {"type": "corrective", "params": {"base": "parent_child"}},
        },
    ),
    Rule(
        id="corrective_low_latency",
        when=lambda c: _r(c).strict_grounding and _r(c).latency == "low",
        warning=(
            "corrective adds an LLM grading call per query (two more when it retries); "
            "check latency with `ragpick bench` before committing to it."
        ),
    ),
    Rule(
        id="mixed_structured_and_documents",
        when=lambda c: (
            bool(c.data_types & STRUCTURED_TYPES) and bool(c.data_types & DOCUMENT_TYPES)
        ),
        warning=(
            "The sql blueprint answers from tables only. Use a router config with a sql "
            "route and a document route (e.g. hybrid); see docs/routing.md."
        ),
    ),
    Rule(
        id="sql_access_control",
        when=lambda c: bool(c.data_types & STRUCTURED_TYPES) and _r(c).access_control,
        warning=(
            "sql supports table-level access control (retriever param `table_groups`), "
            "not row-level security; split restricted rows into their own tables."
        ),
    ),
    Rule(
        id="long_documents_low_latency",
        when=lambda c: _r(c).long_documents and _r(c).latency == "low",
        warning=(
            "parent_child sends whole parent sections to the LLM; with a low-latency "
            "target, lower `parent_size` or `query.top_k` to keep prompts small."
        ),
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
        id="pdf_parent_child_hint",
        when=lambda c: "pdf" in c.data_types and not _r(c).long_documents,
        alternatives=("parent_child, if the PDFs are long manuals or contracts",),
    ),
    Rule(
        id="unknown_data_types",
        when=lambda c: bool(c.data_types - STRUCTURED_TYPES - DOCUMENT_TYPES - IMAGE_TYPES),
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
