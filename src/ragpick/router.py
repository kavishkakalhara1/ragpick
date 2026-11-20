"""Query routing: send each question to the pipeline best suited to answer it.

A router config lists routes, each pointing at an ordinary ``rag.yaml``::

    version: 1
    name: lumen-assistant
    router:
      strategy: auto              # llm | keywords | auto
      default: docs
      routes:
        - name: docs
          config: docs.yaml
          description: Product manuals, troubleshooting and policies
        - name: analytics
          config: sql.yaml
          description: Orders, sales, prices, counts and totals
          keywords: [how many, total, revenue, average, top]
          groups: [staff]         # only staff may be routed here

``llm`` asks a model to pick a route from the descriptions; ``keywords`` scores
keyword hits and description overlap (offline, deterministic); ``auto`` uses the LLM
when one is available. If the chosen route answers "I don't know" and
``fallback`` is on, the default route is tried.

:func:`load_engine` returns a :class:`Pipeline` or a :class:`Router` for any config
file; both offer ``query``, ``retrieve``, ``ingest`` and ``ready``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragpick import tracing
from ragpick.components.generators import IDK
from ragpick.components.keyword import tokenize
from ragpick.config import (
    ConversationConfig,
    EvalConfig,
    RouterConfig,
    ServeConfig,
    TracingConfig,
    is_router_config,
    load_config,
    load_router_config,
    read_yaml,
)
from ragpick.conversation import ConversationManager, make_store, owner_key
from ragpick.errors import ConfigError
from ragpick.interfaces import TextCompleter
from ragpick.models import Answer, RetrievedChunk, StreamEvent
from ragpick.pipeline import IngestStats, Pipeline, ProgressCallback
from ragpick.registry import registry
from ragpick.tracing import set_attributes, span

logger = logging.getLogger(__name__)

ROUTER_SYSTEM = (
    "You route questions to the data source that can answer them. "
    'Respond with JSON only: {"route": "<name>"}.'
)

ROUTER_PROMPT = """Sources:
{sources}

Question: {question}

Which source should answer the question?"""


@dataclass
class Route:
    """A named destination pipeline."""

    name: str
    pipeline: Pipeline
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    groups: list[str] | None = None

    def allows(self, groups: list[str] | None) -> bool:
        """Whether a caller with ``groups`` may use this route."""
        return self.groups is None or bool(set(groups or []) & set(self.groups))


def _keyword_score(route: Route, question: str, terms: set[str]) -> float:
    lowered = question.lower()
    hits = sum(
        1 for kw in route.keywords if re.search(rf"(?<!\w){re.escape(kw.lower())}(?!\w)", lowered)
    )
    overlap = len(terms & set(tokenize(route.description)))
    return 2.0 * hits + overlap


def _cannot_answer(answer: Answer) -> bool:
    if answer.text.strip() == IDK:
        return True
    return not answer.contexts or all(c.score <= 0 and c.source == "sql" for c in answer.contexts)


class Router:
    """Routes questions between pipelines; see the module docstring."""

    def __init__(
        self,
        routes: Sequence[Route],
        *,
        strategy: str = "auto",
        default: str | None = None,
        fallback: bool = True,
        llm: TextCompleter | None = None,
        name: str = "ragpick-router",
        eval_config: EvalConfig | None = None,
        tracing_config: TracingConfig | None = None,
        serve_config: ServeConfig | None = None,
        exporters: Sequence[Any] = (),
        conversation_config: ConversationConfig | None = None,
        sessions_dir: Path | None = None,
    ) -> None:
        if not routes:
            raise ConfigError("A router needs at least one route")
        if strategy not in ("auto", "llm", "keywords"):
            raise ConfigError(f"router.strategy must be auto, llm or keywords, got {strategy!r}")
        self.routes = list(routes)
        self.by_name = {r.name: r for r in self.routes}
        if default is not None and default not in self.by_name:
            raise ConfigError(f"Unknown default route '{default}'")
        self.default = default or self.routes[0].name
        self.fallback = fallback
        if llm is None and strategy in ("auto", "llm"):
            llm = next(
                (
                    r.pipeline.generator
                    for r in self.routes
                    if isinstance(r.pipeline.generator, TextCompleter)
                    and not getattr(r.pipeline.generator, "is_fake", False)
                ),
                None,
            )
        if strategy == "llm" and llm is None:
            raise ConfigError(
                "router.strategy: llm needs an LLM: set router.llm or give a route a real "
                "generator (openai, ollama, langchain, llamaindex)"
            )
        self.llm = llm if strategy != "keywords" else None
        self.strategy = "llm" if self.llm is not None else "keywords"
        self.name = name
        self.eval_config = eval_config or EvalConfig()
        self.tracing_config = tracing_config or TracingConfig()
        self.serve_config = serve_config or ServeConfig()
        self.exporters = list(exporters)
        self.conversation_config = conversation_config or ConversationConfig()
        self.sessions_dir = sessions_dir or Path(".ragpick/router")
        self._conversations: ConversationManager | None = None

    # ------------------------------------------------------------------ building
    @classmethod
    def from_config(cls, config: RouterConfig) -> Router:
        """Build a router (and its route pipelines) from a :class:`RouterConfig`."""
        routes = []
        for spec in config.router.routes:
            path = config.resolve_path(spec.config)
            raw = read_yaml(path)
            if is_router_config(raw):
                raise ConfigError(f"Route '{spec.name}': nested routers are not supported ({path})")
            routes.append(
                Route(
                    name=spec.name,
                    pipeline=Pipeline(load_config(path)),
                    description=spec.description,
                    keywords=list(spec.keywords),
                    groups=spec.groups,
                )
            )
        llm: TextCompleter | None = None
        if config.router.llm is not None:
            candidate = registry.create(
                "generator", config.router.llm.type, **config.router.llm.params
            )
            if not isinstance(candidate, TextCompleter):
                raise ConfigError("router.llm must be a generator that supports complete()")
            llm = candidate
        exporters = []
        for exporter_spec in config.tracing.exporters:
            params = dict(exporter_spec.params)
            if isinstance(params.get("path"), str):
                params["path"] = str(config.resolve_path(params["path"]))
            exporters.append(registry.create("trace_exporter", exporter_spec.type, **params))
        return cls(
            routes,
            strategy=config.router.strategy,
            default=config.router.default,
            fallback=config.router.fallback,
            llm=llm,
            name=config.name,
            eval_config=config.eval,
            tracing_config=config.tracing,
            serve_config=config.serve,
            exporters=exporters,
            conversation_config=config.conversation,
            sessions_dir=config.resolve_path(config.sessions_dir),
        )

    # ------------------------------------------------------------------ summary (eval/serve)
    @property
    def blueprint(self) -> str:
        """``"router(<route blueprints>)"``."""
        inner = ",".join(f"{r.name}={r.pipeline.blueprint}" for r in self.routes)
        return f"router({inner})"

    @property
    def top_k(self) -> int:
        """The default route's ``top_k``."""
        return self.by_name[self.default].pipeline.top_k

    @property
    def model_name(self) -> str:
        """The default route's model name."""
        return self.by_name[self.default].pipeline.model_name

    def ready(self) -> bool:
        """Whether every route can answer queries."""
        return all(r.pipeline.ready() for r in self.routes)

    # ------------------------------------------------------------------ routing
    def choose(self, question: str, groups: list[str] | None = None) -> tuple[Route | None, str]:
        """Pick a route for ``question``: ``(route, reason)``; ``None`` if none is allowed."""
        allowed = [r for r in self.routes if r.allows(groups)]
        if not allowed:
            return None, "no route is available to the caller's groups"
        if len(allowed) == 1:
            return allowed[0], "only route available"
        if self.llm is not None:
            sources = "\n".join(f"- {r.name}: {r.description or r.name}" for r in allowed)
            reply = self.llm.complete(
                ROUTER_SYSTEM, ROUTER_PROMPT.format(sources=sources, question=question)
            ).text
            match = re.search(r"\{.*\}", reply, re.DOTALL)
            try:
                picked = json.loads(match.group(0))["route"] if match else None
            except (ValueError, KeyError, TypeError):
                picked = None
            chosen = next((r for r in allowed if r.name == picked), None)
            if chosen is not None:
                return chosen, "llm"
            logger.warning("Router LLM reply %r named no allowed route; using keywords", reply[:80])
        terms = set(tokenize(question))
        scores = [(_keyword_score(r, question, terms), -i, r) for i, r in enumerate(allowed)]
        best_score, _, best = max(scores, key=lambda s: (s[0], s[1]))
        if best_score > 0:
            return best, f"keywords (score {best_score:g})"
        default = self.by_name[self.default]
        return (default if default in allowed else allowed[0]), "default"

    def _fallback_route(self, chosen: Route, groups: list[str] | None) -> Route | None:
        default = self.by_name[self.default]
        if self.fallback and default is not chosen and default.allows(groups):
            return default
        return None

    def retrieve(
        self,
        question: str,
        *,
        groups: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        tenant: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve contexts from the chosen route."""
        route, _ = self.choose(question, groups)
        if route is None:
            return []
        return route.pipeline.retrieve(
            question, groups=groups, filters=filters, top_k=top_k, tenant=tenant
        )

    def query(
        self,
        question: str,
        *,
        groups: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        trace: bool | None = None,
        tenant: str | None = None,
    ) -> Answer:
        """Route ``question`` and answer it; ``answer.route`` names the route used.

        ``tenant`` is passed to the route pipelines (which must be multi-tenant).
        """
        if not (self.tracing_config.enabled if trace is None else trace):
            return self._query(question, groups, filters, top_k, tenant)
        with tracing.trace("query", router=self.name, groups=len(groups or [])) as root:
            answer = self._query(question, groups, filters, top_k, tenant)
        answer.trace = root
        for exporter in self.exporters:
            try:
                exporter.export(root)
            except Exception as exc:  # exporters must never break queries
                logger.warning("Trace exporter %s failed: %s", type(exporter).__name__, exc)
        return answer

    def stream(
        self,
        question: str,
        *,
        groups: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        tenant: str | None = None,
    ) -> Iterator[StreamEvent]:
        """Route, then stream from the chosen route (no fallback: text is already sent)."""
        route, _ = self.choose(question, groups)
        if route is None:
            yield StreamEvent(type="contexts")
            yield StreamEvent(type="delta", text=IDK)
            yield StreamEvent(type="answer", answer=Answer(text=IDK))
            return
        events = route.pipeline.stream(
            question, groups=groups, filters=filters, top_k=top_k, tenant=tenant
        )
        for event in events:
            if event.answer is not None:
                event.answer.route = route.name
            yield event

    def _query(
        self,
        question: str,
        groups: list[str] | None,
        filters: dict[str, Any] | None,
        top_k: int | None,
        tenant: str | None = None,
    ) -> Answer:
        with span("route", strategy=self.strategy):
            route, reason = self.choose(question, groups)
            set_attributes(route=route.name if route else None, reason=reason)
        if route is None:
            return Answer(text=IDK)
        answer = route.pipeline.query(
            question, groups=groups, filters=filters, top_k=top_k, tenant=tenant
        )
        answer.route = route.name
        backup = self._fallback_route(route, groups) if _cannot_answer(answer) else None
        if backup is not None:
            with span("fallback", from_route=route.name, to_route=backup.name):
                second = backup.pipeline.query(
                    question, groups=groups, filters=filters, top_k=top_k
                )
                set_attributes(answered=not _cannot_answer(second))
            if not _cannot_answer(second):
                second.route = backup.name
                return second
        return answer

    # ------------------------------------------------------------------ conversations
    @property
    def conversations(self) -> ConversationManager:
        """The router's chat-session manager (routing uses the standalone question)."""
        if self._conversations is None:
            self._conversations = ConversationManager(
                self.conversation_config,
                make_store(self.conversation_config, self.sessions_dir),
                self.llm,
            )
        return self._conversations

    def chat(
        self,
        message: str,
        *,
        session_id: str | None = None,
        owner: str | None = None,
        groups: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        trace: bool | None = None,
        tenant: str | None = None,
    ) -> Answer:
        """Chat turn: rewrite the follow-up, then route and answer it."""
        return self.conversations.chat(
            self.query,
            message,
            session_id,
            owner_key(owner, tenant),
            groups=groups,
            filters=filters,
            top_k=top_k,
            trace=trace,
            tenant=tenant,
        )

    def chat_stream(
        self,
        message: str,
        *,
        session_id: str | None = None,
        owner: str | None = None,
        groups: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        tenant: str | None = None,
    ) -> Iterator[StreamEvent]:
        """Streaming :meth:`chat`."""
        return self.conversations.chat_stream(
            self.stream,
            message,
            session_id,
            owner_key(owner, tenant),
            groups=groups,
            filters=filters,
            top_k=top_k,
            tenant=tenant,
        )

    # ------------------------------------------------------------------ ingest
    def ingest(
        self,
        path: str | Path | None = None,
        *,
        progress: ProgressCallback | None = None,
        tenant: str | None = None,
    ) -> dict[str, IngestStats]:
        """Ingest every route that needs it, from each route's own ``data`` path.

        For multi-tenant routes, ingest each route config per tenant instead.
        """
        if tenant is not None:
            raise ConfigError(
                "Routers can't ingest per tenant; ingest each route's config with --tenant"
            )
        if path is not None:
            raise ConfigError("A router ingests each route's own `data`; run it without a PATH")
        stats: dict[str, IngestStats] = {}
        for route in self.routes:
            if not route.pipeline.requires_ingest:
                continue
            if route.pipeline.config.data is None:
                raise ConfigError(f"Route '{route.name}' needs ingest but its config has no `data`")
            stats[route.name] = route.pipeline.ingest(progress=progress)
        return stats


Engine = Pipeline | Router


def load_engine(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Engine:
    """Load any config file: a :class:`Router` for router configs, else a :class:`Pipeline`."""
    raw = read_yaml(path)
    if is_router_config(raw):
        return Router.from_config(load_router_config(path, overrides))
    return Pipeline.from_yaml(path, overrides)
