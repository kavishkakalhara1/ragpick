"""Agentic retrieval: an LLM plans and runs several searches before answering.

Many real questions need more than one lookup: comparisons ("LM-200 vs LM-300 Pro
warranty"), multi-part questions ("how do I reset my password and enable 2FA?"), or
follow-ups that only make sense after a first result. The agentic retriever runs a short
loop:

1. search for the question itself;
2. show the LLM what has been found so far; it replies with
   ``{"action": "search", "query": "..."}`` or ``{"action": "finish"}``;
3. repeat up to ``max_steps`` searches, stopping early on ``finish`` or when two searches
   in a row add nothing new;
4. fuse all result lists with Reciprocal Rank Fusion and return the top ``k``.

Offline (fake generator), the plan is deterministic: the question is split into parts
("A and B", "A vs B", several sentences) and each part is searched. Every search goes
through the base retriever with the caller's filters, so access control holds.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from ragpick.components.corrective import build_base_retriever
from ragpick.components.retrievers import reciprocal_rank_fusion
from ragpick.errors import ConfigError
from ragpick.interfaces import (
    ChunkIndexer,
    Embedder,
    Generator,
    KeywordIndex,
    TextCompleter,
    VectorStore,
)
from ragpick.models import Chunk, RetrievedChunk
from ragpick.registry import register
from ragpick.tracing import set_attributes, span

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = (
    "You plan document searches to answer a question. Respond with JSON only: either "
    '{"action": "search", "query": "<a new, specific search query>"} or {"action": "finish"}. '
    "Finish as soon as the passages found cover every part of the question."
)

PLANNER_PROMPT = """Question: {question}

Searches so far:
{searches}

Passages found so far:
{passages}

Search again for missing information, or finish."""

_SPLIT_RE = re.compile(
    r"\s*(?:\?|;|\band also\b|\bas well as\b|\bversus\b|\bvs\.?"
    r"|\bcompared (?:to|with)\b|\band\b)\s*",
    re.IGNORECASE,
)
_PASSAGE_CHARS = 300


def decompose(question: str, max_parts: int = 4) -> list[str]:
    """Split a multi-part question into searchable parts (offline planner)."""
    parts = [p.strip(" ,.") for p in _SPLIT_RE.split(question)]
    seen: list[str] = []
    for part in parts:
        searchable = len(part.split()) >= 2 or any(ch.isdigit() for ch in part)  # "LM-200"
        if searchable and part.lower() not in (s.lower() for s in seen):
            seen.append(part)
    return seen[:max_parts]


def parse_action(reply: str) -> tuple[str, str]:
    """``(action, query)`` from a planner reply; unparseable → ``("finish", "")``."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            action = str(data.get("action", "")).lower()
            query = str(data.get("query", "")).strip()
            if action == "search" and query:
                return "search", query
            if action == "finish":
                return "finish", ""
        except (ValueError, AttributeError):
            pass
    return "finish", ""


@register("retriever", "agentic")
class AgenticRetriever:
    """Plan → search → repeat → fuse; see the module docstring.

    Args:
        base: retriever used for each search: ``hybrid`` (default), ``vector``,
            ``parent_child`` or ``graph``.
        max_steps: maximum number of searches (including the first).
        per_search_k: results requested per search.
        planner: ``auto`` (LLM when the generator is a real LLM), ``llm`` or
            ``decompose`` (offline question splitting).
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        keyword_index: KeywordIndex,
        generator: Generator,
        index_dir: Path | None = None,
        k: int = 5,
        base: str = "hybrid",
        max_steps: int = 4,
        per_search_k: int = 4,
        planner: str = "auto",
        rrf_k: int = 60,
    ) -> None:
        if planner not in ("auto", "llm", "decompose"):
            raise ConfigError(f"planner must be auto, llm or decompose, got {planner!r}")
        if max_steps < 1:
            raise ConfigError("max_steps must be >= 1")
        real_llm = isinstance(generator, TextCompleter) and not getattr(generator, "is_fake", False)
        if planner == "llm" and not isinstance(generator, TextCompleter):
            raise ConfigError("planner: llm needs a generator that supports complete()")
        self.mode = "llm" if planner == "llm" or (planner == "auto" and real_llm) else "decompose"
        self.llm: TextCompleter | None = generator if self.mode == "llm" else None  # type: ignore[assignment]
        self.base = build_base_retriever(
            base, embedder, store, keyword_index, per_search_k, rrf_k, None, generator, index_dir
        )
        self.k = k
        self.max_steps = max_steps
        self.per_search_k = per_search_k
        self.rrf_k = rrf_k

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Forward ingest-time chunk indexing to the base retriever (e.g. ``graph``)."""
        if isinstance(self.base, ChunkIndexer):
            self.base.index_chunks(chunks)

    def _search(
        self,
        query: str,
        step: int,
        filters: dict[str, Any] | None,
        found: dict[str, RetrievedChunk],
    ) -> tuple[list[RetrievedChunk], int]:
        with span("agent_search", step=step, query=query):
            results = self.base.retrieve(query, self.per_search_k, filters)
            new = [r for r in results if r.chunk.id not in found]
            for r in new:
                found[r.chunk.id] = r
            set_attributes(results=len(results), new=len(new))
        return results, len(new)

    def _next_query(
        self, question: str, searches: list[str], found: dict[str, RetrievedChunk], step: int
    ) -> str | None:
        assert self.llm is not None
        passages = "\n".join(
            f"[{cid}] {r.chunk.text[:_PASSAGE_CHARS]}" for cid, r in list(found.items())[:12]
        )
        with span("agent_plan", step=step) as s:
            reply = self.llm.complete(
                PLANNER_SYSTEM,
                PLANNER_PROMPT.format(
                    question=question,
                    searches="\n".join(f"- {q}" for q in searches),
                    passages=passages or "(none)",
                ),
            ).text
            action, query = parse_action(reply)
            if s is not None:
                s.attributes.update(action=action, query=query)
        if action != "search" or query.lower() in (q.lower() for q in searches):
            return None
        return query

    def retrieve(
        self, query: str, k: int | None = None, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Run the search loop and return the fused top ``k`` contexts."""
        k = k or self.k
        found: dict[str, RetrievedChunk] = {}
        searches: list[str] = []
        lists: list[list[RetrievedChunk]] = []
        planned = [query] if self.llm is not None else [query, *decompose(query)]
        idle = 0
        step = 0
        while step < self.max_steps:
            if step < len(planned):
                next_query: str | None = planned[step]
            elif self.llm is not None:
                next_query = self._next_query(query, searches, found, step)
            else:
                next_query = None
            if next_query is None:
                break
            if next_query.lower() in (q.lower() for q in searches):
                step += 1
                continue
            results, new = self._search(next_query, step, filters, found)
            searches.append(next_query)
            lists.append(results)
            idle = idle + 1 if new == 0 else 0
            step += 1
            if idle >= 2:
                break
        set_attributes(searches=len(searches), planner=self.mode)
        fused = reciprocal_rank_fusion(lists, self.rrf_k) if lists else []
        return fused[:k]
