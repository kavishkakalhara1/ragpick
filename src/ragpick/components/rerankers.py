"""Rerankers."""

from __future__ import annotations

from typing import Any

from ragpick.components._optional import require
from ragpick.models import RetrievedChunk
from ragpick.registry import register


@register("reranker", "none")
class NoopReranker:
    """Keeps the retriever's order and truncates to ``top_n``."""

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        """Return the first ``top_n`` chunks unchanged."""
        return list(chunks[:top_n])


@register("reranker", "cross_encoder")
class CrossEncoderReranker:
    """Scores (query, chunk) pairs with a cross-encoder (requires ``ragpick[rerank]``).

    The returned chunks carry the cross-encoder score and keep their original
    ``source``.
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 32,
        model_obj: Any | None = None,
    ) -> None:
        self.model_name = model
        self.batch_size = batch_size
        if model_obj is None:
            st = require("sentence_transformers", "rerank", "CrossEncoderReranker")
            model_obj = st.CrossEncoder(model, device=device)
        self._model = model_obj

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        """Return the ``top_n`` chunks with the highest cross-encoder score."""
        if not chunks:
            return []
        pairs = [(query, c.chunk.text) for c in chunks]
        scores = self._model.predict(pairs, batch_size=self.batch_size)
        rescored = [
            c.model_copy(update={"score": float(s)}) for c, s in zip(chunks, scores, strict=True)
        ]
        rescored.sort(key=lambda c: -c.score)
        return rescored[:top_n]
