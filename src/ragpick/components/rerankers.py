"""Rerankers."""

from __future__ import annotations

import os
from typing import Any

import httpx

from ragpick.components._optional import require
from ragpick.errors import MissingCredentialError, ProviderError
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


class HTTPReranker:
    """Base for hosted rerank APIs (``POST <endpoint> {model, query, documents, top_n}``)."""

    endpoint = "/rerank"
    env_key = ""
    provider = ""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        key = api_key or os.environ.get(self.env_key)
        if not key and client is None:
            raise MissingCredentialError(
                f"{self.env_key} not set; export the key (or pass `api_key` in the reranker params)"
            )
        self.model = model
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        """Reorder ``chunks`` by the provider's relevance score; keep ``top_n``."""
        if not chunks:
            return []
        payload = {
            "model": self.model,
            "query": query,
            "documents": [c.chunk.text for c in chunks],
            "top_n": min(top_n, len(chunks)),
        }
        try:
            response = self._client.post(self.endpoint, json=payload)
            response.raise_for_status()
            results = response.json()["results"]
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"{self.provider} rerank returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise ProviderError(f"{self.provider} rerank failed: {exc}") from exc
        out = []
        for item in results:
            index = int(item["index"])
            if 0 <= index < len(chunks):
                out.append(
                    chunks[index].model_copy(update={"score": float(item["relevance_score"])})
                )
        out.sort(key=lambda c: -c.score)
        return out[:top_n]


@register("reranker", "cohere")
class CohereReranker(HTTPReranker):
    """Cohere Rerank (``COHERE_API_KEY``), e.g. ``rerank-v3.5``. No extra dependency."""

    endpoint = "/v2/rerank"
    env_key = "COHERE_API_KEY"
    provider = "Cohere"

    def __init__(
        self,
        model: str = "rerank-v3.5",
        api_key: str | None = None,
        base_url: str = "https://api.cohere.com",
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(model, base_url, api_key, timeout, client)


@register("reranker", "jina")
class JinaReranker(HTTPReranker):
    """Jina Reranker (``JINA_API_KEY``), e.g. ``jina-reranker-v2-base-multilingual``."""

    endpoint = "/v1/rerank"
    env_key = "JINA_API_KEY"
    provider = "Jina"

    def __init__(
        self,
        model: str = "jina-reranker-v2-base-multilingual",
        api_key: str | None = None,
        base_url: str = "https://api.jina.ai",
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(model, base_url, api_key, timeout, client)
