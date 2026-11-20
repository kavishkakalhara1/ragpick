"""Core data models shared by every component."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ragpick.tracing import Span

RetrievalSource = Literal["vector", "keyword", "fused", "sql"]


class Document(BaseModel):
    """A source document produced by a :class:`~ragpick.interfaces.Loader`.

    ``metadata`` is copied onto every chunk, so document-level fields such as
    ``allowed_groups`` (used for access control) propagate automatically.
    """

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable piece of a document, optionally carrying its embedding."""

    id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = Field(default=None, repr=False)


class RetrievedChunk(BaseModel):
    """A chunk returned by a retriever together with its score and origin."""

    chunk: Chunk
    score: float
    source: RetrievalSource


class Usage(BaseModel):
    """Token usage and latency of a single generation (or a whole query)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens


class Answer(BaseModel):
    """The result of a RAG query.

    ``citations`` holds the IDs of the chunks the model cited, in citation order.
    ``trace`` is set when tracing is enabled (see :mod:`ragpick.tracing`); ``route``
    names the route that answered when a :class:`~ragpick.router.Router` is used.
    """

    text: str
    citations: list[str] = Field(default_factory=list)
    contexts: list[RetrievedChunk] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    route: str | None = None
    trace: Span | None = None


class Completion(BaseModel):
    """Raw text returned by a :class:`~ragpick.interfaces.TextCompleter`."""

    text: str
    usage: Usage = Field(default_factory=Usage)
