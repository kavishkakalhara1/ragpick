"""Evaluation metrics.

Retrieval metrics work on the ranked list of retrieved chunks, each mapped to its
``doc_id``; a chunk is *relevant* when its document is in ``relevant_doc_ids``:

* ``hit_rate@k``: 1 if any of the first ``k`` chunks is relevant, else 0.
* ``MRR``: ``1 / rank`` of the first relevant chunk (0 if none).
* ``precision@k``: relevant chunks among the first ``k``, divided by ``k``.

LLM-judged metrics (faithfulness, answer relevance) use :class:`LLMJudge`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel

from ragpick.components.generators import format_contexts
from ragpick.config import PriceSpec
from ragpick.eval import prompts
from ragpick.interfaces import TextCompleter
from ragpick.models import RetrievedChunk, Usage


def hit_rate_at_k(retrieved: Sequence[str], relevant: set[str] | Sequence[str], k: int) -> float:
    """1.0 if any of the first ``k`` retrieved doc ids is relevant."""
    rel = set(relevant)
    return 1.0 if any(d in rel for d in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str] | Sequence[str]) -> float:
    """``1 / rank`` of the first relevant doc id (1-based), or 0."""
    rel = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved: Sequence[str], relevant: set[str] | Sequence[str], k: int) -> float:
    """Share of the first ``k`` positions holding a relevant doc id (missing positions count)."""
    if k <= 0:
        raise ValueError("k must be positive")
    rel = set(relevant)
    return sum(1 for d in retrieved[:k] if d in rel) / k


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (``q`` in 0..100); ``None`` for no values."""
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def mean(values: Sequence[float | None]) -> float | None:
    """Mean of the non-``None`` values; ``None`` if there are none."""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


# --------------------------------------------------------------------- cost
#: USD per 1M tokens. Prices change; override them in ``eval.prices``.
DEFAULT_PRICES: dict[str, PriceSpec] = {
    "fake": PriceSpec(prompt_per_1m=0.0, completion_per_1m=0.0),
    "gpt-4o-mini": PriceSpec(prompt_per_1m=0.15, completion_per_1m=0.60),
    "gpt-4o": PriceSpec(prompt_per_1m=2.50, completion_per_1m=10.00),
    "gpt-4.1-mini": PriceSpec(prompt_per_1m=0.40, completion_per_1m=1.60),
    "gpt-4.1": PriceSpec(prompt_per_1m=2.00, completion_per_1m=8.00),
}


def estimate_cost(
    usage: Usage | Sequence[Usage], model: str, prices: Mapping[str, PriceSpec] | None = None
) -> float | None:
    """Estimated USD cost of ``usage`` for ``model``; ``None`` if the price is unknown."""
    table = {**DEFAULT_PRICES, **(prices or {})}
    price = table.get(model)
    if price is None:
        return None
    usages = [usage] if isinstance(usage, Usage) else list(usage)
    prompt = sum(u.prompt_tokens for u in usages)
    completion = sum(u.completion_tokens for u in usages)
    return (prompt * price.prompt_per_1m + completion * price.completion_per_1m) / 1_000_000


# --------------------------------------------------------------------- LLM judge
class JudgeScore(BaseModel):
    """A judge verdict. ``score`` is ``None`` when the judge output was unusable."""

    score: float | None
    reason: str
    usage: Usage = Usage()


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_response(text: str) -> tuple[float | None, str]:
    """Extract ``(score, reason)`` from a judge reply; the score is clamped to 0..1."""
    match = _JSON_RE.search(text)
    if match:
        try:
            data = json.loads(match.group(0))
            score = max(0.0, min(1.0, float(data["score"])))
            return score, str(data.get("reason", "")).strip()
        except (ValueError, KeyError, TypeError):
            pass
    return None, f"unparseable judge output: {text[:120]!r}"


class LLMJudge:
    """Scores answers with an LLM that implements :class:`~ragpick.interfaces.TextCompleter`."""

    def __init__(self, completer: TextCompleter) -> None:
        self.completer = completer

    def _ask(self, prompt: str) -> JudgeScore:
        completion = self.completer.complete(prompts.JUDGE_SYSTEM, prompt)
        score, reason = parse_judge_response(completion.text)
        return JudgeScore(score=score, reason=reason, usage=completion.usage)

    def faithfulness(self, answer: str, contexts: list[RetrievedChunk]) -> JudgeScore:
        """Is the answer supported by the contexts? (0..1)"""
        rendered = format_contexts(contexts) if contexts else "(no context)"
        return self._ask(prompts.FAITHFULNESS_PROMPT.format(contexts=rendered, answer=answer))

    def answer_relevance(self, question: str, answer: str) -> JudgeScore:
        """Does the answer address the question? (0..1)"""
        return self._ask(prompts.ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer))
