"""Chunkers: split documents into retrievable chunks.

Chunk ids are ``"<doc_id>#<index>"``. Each chunk inherits its document's metadata
plus ``chunk_index``.
"""

from __future__ import annotations

from collections.abc import Iterable

from ragpick.models import Chunk, Document
from ragpick.registry import register

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


def _validate(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")


def _make_chunks(doc: Document, pieces: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for piece in pieces:
        text = piece.strip()
        if not text:
            continue
        index = len(chunks)
        chunks.append(
            Chunk(
                id=f"{doc.id}#{index}",
                doc_id=doc.id,
                text=text,
                metadata={**doc.metadata, "chunk_index": index},
            )
        )
    return chunks


@register("chunker", "fixed")
class FixedSizeChunker:
    """Splits text into windows of ``chunk_size`` characters with ``overlap``."""

    def __init__(self, chunk_size: int = 800, overlap: int = 100) -> None:
        _validate(chunk_size, overlap)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> list[str]:
        """Split a single string."""
        if not text.strip():
            return []
        step = self.chunk_size - self.overlap
        pieces = []
        for start in range(0, len(text), step):
            pieces.append(text[start : start + self.chunk_size])
            if start + self.chunk_size >= len(text):
                break
        return pieces

    def split(self, docs: Iterable[Document]) -> list[Chunk]:
        """Split every document."""
        return [c for doc in docs for c in _make_chunks(doc, self.split_text(doc.text))]


@register("chunker", "recursive")
class RecursiveChunker:
    """Splits on the coarsest separator that yields pieces under ``chunk_size``.

    Tries paragraphs, then lines, then sentences, then words, then characters,
    and greedily merges adjacent pieces back together up to ``chunk_size``. The
    last ``overlap`` characters (on a separator boundary where possible) of each
    chunk are repeated at the start of the next.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 100,
        separators: list[str] | None = None,
    ) -> None:
        _validate(chunk_size, overlap)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = list(separators) if separators is not None else list(DEFAULT_SEPARATORS)
        if "" not in self.separators:
            self.separators.append("")

    def _atoms(self, text: str, separators: list[str]) -> list[str]:
        """Break ``text`` into pieces no longer than ``chunk_size``, keeping separators."""
        if len(text) <= self.chunk_size:
            return [text]
        sep, rest = separators[0], separators[1:]
        if sep == "":
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        if sep not in text:
            return self._atoms(text, rest)
        parts = text.split(sep)
        atoms: list[str] = []
        for i, part in enumerate(parts):
            piece = part + sep if i < len(parts) - 1 else part
            if not piece:
                continue
            atoms.extend(self._atoms(piece, rest) if len(piece) > self.chunk_size else [piece])
        return atoms

    def _tail(self, text: str) -> str:
        """Last ``overlap`` characters of ``text``, starting on a word boundary if possible."""
        if self.overlap == 0:
            return ""
        tail = text[-self.overlap :]
        cut = tail.find(" ")
        if 0 <= cut < len(tail) - 1 and len(text) > self.overlap:
            tail = tail[cut + 1 :]
        return tail

    def split_text(self, text: str) -> list[str]:
        """Split a single string."""
        if not text.strip():
            return []
        chunks: list[str] = []
        current = ""
        for atom in self._atoms(text, self.separators):
            if current and len(current) + len(atom) > self.chunk_size:
                chunks.append(current)
                tail = self._tail(current)
                current = tail if len(tail) + len(atom) <= self.chunk_size else ""
            current += atom
        if current.strip():
            chunks.append(current)
        return chunks

    def split(self, docs: Iterable[Document]) -> list[Chunk]:
        """Split every document."""
        return [c for doc in docs for c in _make_chunks(doc, self.split_text(doc.text))]


#: Metadata keys written by :class:`ParentChildChunker` on every child chunk.
PARENT_ID_KEY = "parent_id"
PARENT_TEXT_KEY = "parent_text"


@register("chunker", "parent_child")
class ParentChildChunker:
    """Two-level chunking for the ``parent_child`` blueprint.

    Each document is split into large *parent* sections (``parent_size``), and each
    parent into small *child* chunks (``child_size``). Only children are embedded and
    indexed; each child carries ``parent_id`` and ``parent_text`` metadata, so a
    retriever can search precisely on children and hand whole parents to the LLM.

    Ids: parent ``"<doc_id>#<p>"``, child ``"<doc_id>#<p>.<c>"``.
    """

    def __init__(
        self,
        parent_size: int = 2000,
        parent_overlap: int = 0,
        child_size: int = 400,
        child_overlap: int = 50,
    ) -> None:
        if child_size >= parent_size:
            raise ValueError(
                f"child_size ({child_size}) must be smaller than parent_size ({parent_size})"
            )
        self.parents = RecursiveChunker(parent_size, parent_overlap)
        self.children = RecursiveChunker(child_size, child_overlap)

    def split(self, docs: Iterable[Document]) -> list[Chunk]:
        """Return child chunks (with parent text in metadata) for every document."""
        out: list[Chunk] = []
        for doc in docs:
            for pi, parent_text in enumerate(
                t.strip() for t in self.parents.split_text(doc.text) if t.strip()
            ):
                parent_id = f"{doc.id}#{pi}"
                for ci, child_text in enumerate(
                    t.strip() for t in self.children.split_text(parent_text) if t.strip()
                ):
                    out.append(
                        Chunk(
                            id=f"{parent_id}.{ci}",
                            doc_id=doc.id,
                            text=child_text,
                            metadata={
                                **doc.metadata,
                                "chunk_index": len(out),
                                PARENT_ID_KEY: parent_id,
                                PARENT_TEXT_KEY: parent_text,
                            },
                        )
                    )
        return out
