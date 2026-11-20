from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ragpick.components.embedders import (
    FakeEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
)
from ragpick.errors import MissingCredentialError, MissingDependencyError
from ragpick.interfaces import Embedder


def cos(a: list[float], b: list[float]) -> float:
    return float(np.dot(a, b))


def test_fake_embedder_is_deterministic_normalized_and_lexical() -> None:
    emb = FakeEmbedder(dimension=128)
    assert isinstance(emb, Embedder)
    assert emb.dimension == 128
    a, a2, b, c = emb.embed(
        ["reset your password", "reset your password", "password reset steps", "invoice tax"]
    )
    assert a == a2
    assert len(a) == 128
    assert abs(np.linalg.norm(a) - 1.0) < 1e-9
    assert cos(a, b) > cos(a, c)
    assert FakeEmbedder(dimension=128).embed_one("x") == emb.embed_one("x")
    assert emb.embed_one("") == [0.0] * 128
    with pytest.raises(ValueError, match="positive"):
        FakeEmbedder(dimension=0)


class FakeOpenAIClient:
    def __init__(self, dim: int = 3) -> None:
        self.calls: list[dict[str, Any]] = []
        self.dim = dim
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        # return out of order to check sorting by index
        data = [
            SimpleNamespace(index=i, embedding=[float(i)] * self.dim)
            for i in range(len(kwargs["input"]))
        ]
        return SimpleNamespace(data=list(reversed(data)))


def test_openai_embedder_batches_and_orders() -> None:
    client = FakeOpenAIClient()
    emb = OpenAIEmbedder(model="text-embedding-3-small", batch_size=2, client=client)
    vectors = emb.embed(["a", "b", "c"])
    assert vectors == [[0.0] * 3, [1.0] * 3, [0.0] * 3]
    assert [len(c["input"]) for c in client.calls] == [2, 1]
    assert emb.dimension == 1536


def test_openai_embedder_dimensions() -> None:
    client = FakeOpenAIClient(dim=5)
    emb = OpenAIEmbedder(model="custom-model", client=client)
    assert emb.dimension == 5  # probed
    emb2 = OpenAIEmbedder(model="text-embedding-3-large", dimensions=256, client=client)
    assert emb2.dimension == 256
    emb2.embed(["x"])
    assert client.calls[-1]["dimensions"] == 256


def test_openai_embedder_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(MissingDependencyError, match=r"ragpick\[openai\]"):
        OpenAIEmbedder()


def test_openai_embedder_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(OpenAI=lambda **kw: SimpleNamespace(kw=kw))
    monkeypatch.setitem(sys.modules, "openai", fake)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError, match="OPENAI_API_KEY not set"):
        OpenAIEmbedder()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    emb = OpenAIEmbedder()
    assert emb._client.kw["api_key"] == "sk-test"


def test_sentence_transformer_embedder_with_mock_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class Model:
        def __init__(self, name: str, device: str | None = None) -> None:
            self.name = name

        def get_sentence_embedding_dimension(self) -> int:
            return 2

        def encode(self, texts: list[str], **kwargs: Any) -> Any:
            return np.ones((len(texts), 2))

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model)
    )
    emb = SentenceTransformerEmbedder(model="m")
    assert emb.dimension == 2
    assert emb.embed(["a", "b"]) == [[1.0, 1.0], [1.0, 1.0]]
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(MissingDependencyError, match=r"ragpick\[local\]"):
        SentenceTransformerEmbedder()
