"""Corrective RAG: grade retrieved contexts, retry with a better query, or refuse.

Flow per query:

1. retrieve with a base retriever (``hybrid``, ``vector`` or ``parent_child``);
2. **grade** every context for relevance to the question;
3. if fewer than ``min_relevant`` pass, **rewrite** the query (LLM) or widen the
   search (offline), retrieve again and re-grade;
4. return only relevant contexts. If none survive, return nothing
   (``on_failure: empty``), so the generator answers "I don't know" instead of
   guessing from irrelevant text.

Grading uses the pipeline's generator (``complete``) when it is a real LLM
(``grader: llm`` / ``auto``), or a lexical overlap test (``grader: lexical``,
the ``auto`` choice for offline fake models).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from ragpick.components.keyword import tokenize
from ragpick.components.retrievers import HybridRetriever, ParentChildRetriever, VectorRetriever
from ragpick.errors import ConfigError
from ragpick.interfaces import (
    ChunkIndexer,
    Embedder,
    Generator,
    KeywordIndex,
    Retriever,
    TextCompleter,
    VectorStore,
)
from ragpick.models import Chunk, RetrievedChunk
from ragpick.registry import register
from ragpick.tracing import set_attributes, span

logger = logging.getLogger(__name__)

GRADER_SYSTEM = (
    "You judge whether retrieved passages help answer a question. "
    'Respond with JSON only: {"relevant": [<numbers of the helpful passages>]}.'
)

GRADER_PROMPT = """Question: {question}

Passages:
{passages}

Which passages contain information that helps answer the question? A passage that is
only on the same general topic but does not help answer it is NOT relevant."""

REWRITE_SYSTEM = (
    "You rewrite search queries for a document search engine. Reply with the rewritten "
    "query only: no quotes, no explanation."
)

REWRITE_PROMPT = """The search for this question returned no relevant passages:

{question}

Rewrite it as a better search query: keep exact terms (codes, names, IDs), add likely
synonyms or the terms a document would use, and remove filler words."""

_PASSAGE_CHARS = 800
_NUMBERS_RE = re.compile(r"\d+")


def lexical_relevance(question: str, text: str) -> float:
    """Share of the question's content terms that appear in ``text`` (0..1)."""
    terms = set(tokenize(question))
    if not terms:
        return 0.0
    return len(terms & set(tokenize(text))) / len(terms)


def parse_relevant(reply: str, n: int) -> list[int] | None:
    """Parse ``{"relevant": [1, 3]}`` into 0-based indices; ``None`` if unparseable."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            values = data["relevant"]
            if isinstance(values, list):
                picked = [int(v) - 1 for v in values]
                return sorted({i for i in picked if 0 <= i < n})
        except (ValueError, KeyError, TypeError):
            pass
    return None


def build_base_retriever(
    base: str,
    embedder: Embedder,
    store: VectorStore,
    keyword_index: KeywordIndex,
    k: int,
    rrf_k: int = 60,
    child_k: int | None = None,
    generator: Generator | None = None,
    index_dir: Path | None = None,
) -> Retriever:
    """Construct a named base retriever: ``hybrid``, ``vector``, ``parent_child`` or ``graph``."""
    if base == "hybrid":
        return HybridRetriever(embedder, store, keyword_index, k=k, rrf_k=rrf_k)
    if base == "vector":
        return VectorRetriever(embedder, store, k=k)
    if base == "parent_child":
        return ParentChildRetriever(
            embedder, store, keyword_index, k=k, child_k=child_k, rrf_k=rrf_k
        )
    if base == "graph":
        if generator is None or index_dir is None:
            raise ConfigError("base: graph needs the pipeline's generator and index_dir")
        from ragpick.components.graph import GraphRetriever

        return GraphRetriever(
            embedder, store, keyword_index, generator, index_dir, k=k, rrf_k=rrf_k
        )
    raise ConfigError(f"Unknown base retriever '{base}': use hybrid, vector, parent_child or graph")


@register("retriever", "corrective")
class CorrectiveRetriever:
    """Grade, retry, or refuse: see the module docstring.

    Args:
        base: ``hybrid`` (default), ``vector`` or ``parent_child``.
        grader: ``auto`` (LLM when the generator is a real LLM, else lexical),
            ``llm`` or ``lexical``.
        min_relevant: contexts that must pass grading to skip the retry.
        threshold: lexical grader's minimum share of question terms in a context.
        rewrite: allow one corrective retry.
        on_failure: ``empty`` (return nothing, so the answer is "I don't know") or
            ``keep`` (fall back to the ungraded contexts).
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
        grader: str = "auto",
        min_relevant: int = 1,
        threshold: float = 0.3,
        rewrite: bool = True,
        on_failure: str = "empty",
        rrf_k: int = 60,
        child_k: int | None = None,
    ) -> None:
        if grader not in ("auto", "llm", "lexical"):
            raise ConfigError(f"grader must be auto, llm or lexical, got {grader!r}")
        if on_failure not in ("empty", "keep"):
            raise ConfigError(f"on_failure must be empty or keep, got {on_failure!r}")
        llm_ok = isinstance(generator, TextCompleter) and not getattr(generator, "is_fake", False)
        if grader == "llm" and not isinstance(generator, TextCompleter):
            raise ConfigError(
                "grader: llm needs a generator that supports complete() "
                "(openai, ollama, langchain, llamaindex)"
            )
        self.mode = "llm" if grader == "llm" or (grader == "auto" and llm_ok) else "lexical"
        self.llm: TextCompleter | None = generator if self.mode == "llm" else None  # type: ignore[assignment]
        self.base = build_base_retriever(
            base, embedder, store, keyword_index, k, rrf_k, child_k, generator, index_dir
        )
        self.k = k
        self.min_relevant = min_relevant
        self.threshold = threshold
        self.rewrite_enabled = rewrite
        self.on_failure = on_failure

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Forward ingest-time chunk indexing to the base retriever (e.g. ``graph``)."""
        if isinstance(self.base, ChunkIndexer):
            self.base.index_chunks(chunks)

    # -------------------------------------------------------------- grading
    def grade(self, question: str, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Return the relevant subset of ``results`` (order preserved)."""
        if not results:
            return []
        with span("grade", grader=self.mode, contexts=len(results)):
            if self.llm is None:
                keep = [
                    r
                    for r in results
                    if lexical_relevance(question, r.chunk.text) >= self.threshold
                ]
            else:
                passages = "\n\n".join(
                    f"[{i}] {r.chunk.text[:_PASSAGE_CHARS]}" for i, r in enumerate(results, start=1)
                )
                reply = self.llm.complete(
                    GRADER_SYSTEM, GRADER_PROMPT.format(question=question, passages=passages)
                ).text
                indices = parse_relevant(reply, len(results))
                if indices is None:
                    logger.warning(
                        "Unparseable grader reply; keeping all contexts: %r", reply[:120]
                    )
                    set_attributes(unparseable=True)
                    keep = list(results)
                else:
                    keep = [results[i] for i in indices]
            set_attributes(relevant=len(keep))
            return keep

    def rewrite(self, question: str) -> str:
        """Rewrite the question into a better search query (LLM mode only)."""
        if self.llm is None:
            return question
        with span("rewrite"):
            text = self.llm.complete(REWRITE_SYSTEM, REWRITE_PROMPT.format(question=question)).text
            query = text.strip().strip('"').splitlines()[0].strip() if text.strip() else question
            set_attributes(query=query)
            return query or question

    # -------------------------------------------------------------- retrieval
    def retrieve(
        self, query: str, k: int | None = None, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Corrective retrieval; may return fewer than ``k`` (or zero) contexts."""
        k = k or self.k
        first = self.base.retrieve(query, k, filters)
        relevant = self.grade(query, first)
        attempts = 1
        if len(relevant) < self.min_relevant and self.rewrite_enabled:
            attempts = 2
            with span("correct", strategy="rewrite" if self.llm else "widen"):
                if self.llm is not None:
                    second = self.base.retrieve(self.rewrite(query), k, filters)
                else:  # offline: look further down the ranking
                    second = self.base.retrieve(query, 3 * k, filters)
                seen = {r.chunk.id for r in relevant}
                extra = [r for r in second if r.chunk.id not in seen]
                relevant = relevant + self.grade(query, extra)
        set_attributes(attempts=attempts, kept=len(relevant))
        if not relevant and self.on_failure == "keep":
            return first
        return relevant[:k]
