"""Embedders."""

from __future__ import annotations

import hashlib
import itertools
import os
import re
from typing import Any

import numpy as np

from ragpick.components._optional import require
from ragpick.errors import MissingCredentialError
from ragpick.registry import register

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@register("embedder", "fake")
class FakeEmbedder:
    """Deterministic, offline embedder based on the hashing trick.

    Each lowercase token (and token bigram) is hashed with BLAKE2 into one of
    ``dimension`` buckets with a ±1 sign, and the vector is L2-normalised. Texts that
    share words get similar vectors, so retrieval behaves sensibly in tests and
    demos — but this is *not* a semantic model.
    """

    def __init__(self, dimension: int = 256, model: str | None = None) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self.model = model or "fake-hash"

    @property
    def dimension(self) -> int:
        """Vector length."""
        return self._dimension

    def _features(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        return tokens + [f"{a} {b}" for a, b in itertools.pairwise(tokens)]

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text."""
        vec = np.zeros(self._dimension, dtype=np.float64)
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            sign = 1.0 if value & 1 else -1.0
            vec[(value >> 1) % self._dimension] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return [float(x) for x in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts``."""
        return [self.embed_one(t) for t in texts]


OPENAI_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


@register("embedder", "openai")
class OpenAIEmbedder:
    """OpenAI (or OpenAI-compatible) embeddings (requires ``ragpick[openai]``).

    The API key is read from ``api_key`` or the ``OPENAI_API_KEY`` environment
    variable. ``client`` may be injected (useful for tests and custom transports).
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 128,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self._dimensions = dimensions
        if client is None:
            openai = require("openai", "openai", "OpenAIEmbedder")
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key and base_url is None:
                raise MissingCredentialError(
                    "OPENAI_API_KEY not set; install ragpick[openai] and export the key "
                    "(or pass `api_key` in the embedder params)"
                )
            client = openai.OpenAI(api_key=key, base_url=base_url)
        self._client = client

    @property
    def dimension(self) -> int:
        """Vector length (from ``dimensions`` or the model's known default)."""
        if self._dimensions is not None:
            return self._dimensions
        if self.model in OPENAI_EMBEDDING_DIMENSIONS:
            return OPENAI_EMBEDDING_DIMENSIONS[self.model]
        self._dimensions = len(self.embed(["dimension probe"])[0])
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` in batches."""
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            kwargs: dict[str, Any] = {"model": self.model, "input": batch}
            if self._dimensions is not None:
                kwargs["dimensions"] = self._dimensions
            response = self._client.embeddings.create(**kwargs)
            items = sorted(response.data, key=lambda d: d.index)
            out.extend([list(map(float, item.embedding)) for item in items])
        return out


@register("embedder", "sentence_transformers")
class SentenceTransformerEmbedder:
    """Local embeddings via sentence-transformers (requires ``ragpick[local]``)."""

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        normalize: bool = True,
        batch_size: int = 32,
        model_obj: Any | None = None,
    ) -> None:
        self.model_name = model
        self.normalize = normalize
        self.batch_size = batch_size
        if model_obj is None:
            st = require("sentence_transformers", "local", "SentenceTransformerEmbedder")
            model_obj = st.SentenceTransformer(model, device=device)
        self._model = model_obj

    @property
    def dimension(self) -> int:
        """Vector length reported by the model."""
        return int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` locally."""
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return [[float(x) for x in row] for row in vectors]
