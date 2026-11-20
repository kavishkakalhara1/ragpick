"""Component interfaces.

All interfaces are :class:`typing.Protocol` classes, so third-party components do
not need to inherit from anything: matching the method signatures is enough.
They are ``runtime_checkable`` so the pipeline can verify components at build time.

Filters
-------
``filters`` arguments are plain dicts with metadata equality / ``in`` semantics,
see :func:`ragpick.filters.matches_filters`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from ragpick.models import Answer, Chunk, Completion, Document, RetrievedChunk

Filters = dict[str, Any]


@runtime_checkable
class Loader(Protocol):
    """Turns a path (file or directory) into documents."""

    def load(self, path: str) -> Iterable[Document]:
        """Yield documents found at ``path``."""
        ...


@runtime_checkable
class Chunker(Protocol):
    """Splits documents into chunks."""

    def split(self, docs: Iterable[Document]) -> list[Chunk]:
        """Return the chunks for all ``docs``."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to dense vectors."""

    @property
    def dimension(self) -> int:
        """Length of the vectors produced by :meth:`embed`."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts``; the output has the same length and order."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Stores chunk embeddings and supports similarity search."""

    def add(self, chunks: list[Chunk]) -> None:
        """Insert or replace (by chunk id) chunks that carry embeddings."""
        ...

    def search(
        self, vector: list[float], k: int, filters: Filters | None = None
    ) -> list[RetrievedChunk]:
        """Return the ``k`` most similar chunks that match ``filters``."""
        ...

    def delete(self, doc_ids: list[str]) -> None:
        """Remove every chunk belonging to ``doc_ids``."""
        ...

    def persist(self, path: str) -> None:
        """Write the store to ``path`` (a directory)."""
        ...

    def load(self, path: str) -> None:
        """Load a store previously written with :meth:`persist`."""
        ...


@runtime_checkable
class KeywordIndex(Protocol):
    """Sparse/lexical index (e.g. BM25)."""

    def add(self, chunks: list[Chunk]) -> None:
        """Insert or replace (by chunk id) chunks."""
        ...

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[RetrievedChunk]:
        """Return the ``k`` best lexical matches for ``query``."""
        ...

    def persist(self, path: str) -> None:
        """Write the index to ``path`` (a directory)."""
        ...

    def load(self, path: str) -> None:
        """Load an index previously written with :meth:`persist`."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Finds the chunks relevant to a query."""

    def retrieve(self, query: str, k: int, filters: Filters | None = None) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks, best first."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Reorders retrieved chunks, usually with a more expensive model."""

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        """Return the ``top_n`` best chunks, best first."""
        ...


@runtime_checkable
class Generator(Protocol):
    """Produces a grounded answer from retrieved contexts."""

    def generate(self, query: str, contexts: list[RetrievedChunk]) -> Answer:
        """Answer ``query`` using only ``contexts``."""
        ...


@runtime_checkable
class DocumentIndexer(Protocol):
    """Optional: a component that indexes raw documents itself during ingest.

    The pipeline calls :meth:`index_documents` with every loaded document (before
    chunking) for its retriever, reranker and generator. The SQL retriever uses this
    to load CSV files into its database.
    """

    def index_documents(self, docs: list[Document]) -> None:
        """Index ``docs`` (called once per ingest)."""
        ...


@runtime_checkable
class TextCompleter(Protocol):
    """Raw prompt-in, text-out model access (used by LLM judges in evaluation).

    Generators backed by an LLM implement this in addition to :class:`Generator`.
    """

    def complete(self, system: str, user: str) -> Completion:
        """Complete a single system + user prompt."""
        ...
