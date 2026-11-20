from __future__ import annotations

import itertools
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from ragpick.components.chunkers import FixedSizeChunker, RecursiveChunker
from ragpick.components.loaders import DirectoryLoader, PdfLoader, TextLoader, split_front_matter
from ragpick.errors import MissingDependencyError
from ragpick.models import Document


# ---------------------------------------------------------------- loaders
def test_front_matter_parsing() -> None:
    meta, body = split_front_matter("---\nallowed_groups: [hr]\ntitle: X\n---\n\nBody text")
    assert meta == {"allowed_groups": ["hr"], "title": "X"}
    assert body == "Body text"
    assert split_front_matter("no front matter") == ({}, "no front matter")
    assert split_front_matter("---\nunterminated")[0] == {}
    assert split_front_matter("---\n: [bad\n---\nx")[0] == {}
    assert split_front_matter("---\n- a list\n---\nx")[0] == {}
    assert split_front_matter("----\nx")[0] == {}


def test_text_loader(tmp_path: Path) -> None:
    f = tmp_path / "note.md"
    f.write_text("---\nallowed_groups: [hr]\n---\nHello", encoding="utf-8")
    [doc] = list(TextLoader().load(str(f)))
    assert doc.id == "note.md"
    assert doc.text == "Hello"
    assert doc.metadata["allowed_groups"] == ["hr"]
    assert doc.metadata["title"] == "note"
    [raw] = list(TextLoader(front_matter=False).load(str(f)))
    assert raw.text.startswith("---")


def test_directory_loader(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "sub" / "b.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "sub" / "skip.bin").write_bytes(b"\x00")
    (tmp_path / "draft.md").write_text("draft", encoding="utf-8")
    docs = list(DirectoryLoader(exclude=["draft.md"]).load(str(tmp_path)))
    assert [d.id for d in docs] == ["a.md", "sub/b.txt"]
    only_md = list(DirectoryLoader(glob="**/*.md").load(str(tmp_path)))
    assert [d.id for d in only_md] == ["a.md", "draft.md"]
    single = list(DirectoryLoader().load(str(tmp_path / "a.md")))
    assert [d.id for d in single] == ["a.md"]
    with pytest.raises(FileNotFoundError):
        list(DirectoryLoader().load(str(tmp_path / "missing")))


def test_pdf_loader_requires_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pypdf", None)
    (tmp_path / "x.pdf").write_bytes(b"%PDF")
    with pytest.raises(MissingDependencyError, match=r"ragpick\[pdf\]"):
        list(PdfLoader().load(str(tmp_path / "x.pdf")))
    # a directory with pdfs but include_pdf=False just skips them
    assert list(DirectoryLoader(include_pdf=False).load(str(tmp_path))) == []


def test_pdf_loader_with_mocked_pypdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Page:
        def __init__(self, text: str | None) -> None:
            self._text = text

        def extract_text(self) -> str | None:
            return self._text

    class Reader:
        def __init__(self, path: str) -> None:
            self.pages = [Page("page one"), Page(None), Page("page three")]

    fake: Any = types.ModuleType("pypdf")
    fake.PdfReader = Reader
    monkeypatch.setitem(sys.modules, "pypdf", fake)
    (tmp_path / "manual.pdf").write_bytes(b"%PDF")
    [doc] = list(DirectoryLoader().load(str(tmp_path)))
    assert doc.id == "manual.pdf"
    assert doc.text == "page one\n\n\n\npage three"
    assert doc.metadata["pages"] == 3


# ---------------------------------------------------------------- chunkers
@pytest.mark.parametrize("cls", [FixedSizeChunker, RecursiveChunker])
def test_chunker_rejects_bad_sizes(cls: type[Any]) -> None:
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        cls(chunk_size=100, overlap=150)
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        cls(chunk_size=100, overlap=100)
    with pytest.raises(ValueError, match="positive"):
        cls(chunk_size=0, overlap=0)
    with pytest.raises(ValueError, match=">= 0"):
        cls(chunk_size=10, overlap=-1)


@pytest.mark.parametrize("cls", [FixedSizeChunker, RecursiveChunker])
def test_chunker_empty_documents(cls: type[Any]) -> None:
    chunker = cls(chunk_size=50, overlap=10)
    assert chunker.split([]) == []
    assert chunker.split([Document(id="e", text=""), Document(id="w", text="   \n\n ")]) == []


def test_fixed_size_chunker_windows_and_overlap() -> None:
    chunker = FixedSizeChunker(chunk_size=10, overlap=3)
    assert chunker.split_text("abcdefghijklmnopqrst") == ["abcdefghij", "hijklmnopq", "opqrst"]
    assert chunker.split_text("short") == ["short"]
    assert chunker.split_text("abcdefghij") == ["abcdefghij"]


def test_chunk_ids_and_metadata_inheritance() -> None:
    doc = Document(id="d.md", text="x" * 25, metadata={"allowed_groups": ["hr"]})
    chunks = FixedSizeChunker(chunk_size=10, overlap=0).split([doc])
    assert [c.id for c in chunks] == ["d.md#0", "d.md#1", "d.md#2"]
    assert all(c.doc_id == "d.md" for c in chunks)
    assert all(c.metadata["allowed_groups"] == ["hr"] for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1, 2]


def test_recursive_chunker_prefers_paragraphs() -> None:
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = RecursiveChunker(chunk_size=30, overlap=0).split_text(text)
    assert [c.strip() for c in chunks] == [
        "First paragraph here.",
        "Second paragraph here.",
        "Third paragraph here.",
    ]


def test_recursive_chunker_respects_size_and_covers_text() -> None:
    words = [f"word{i}" for i in range(300)]
    text = " ".join(words)
    chunker = RecursiveChunker(chunk_size=100, overlap=20)
    chunks = chunker.split_text(text)
    assert all(len(c) <= 100 for c in chunks)
    joined = " ".join(chunks)
    assert all(w in joined for w in words)
    # overlap: each chunk starts with words from the end of the previous one
    for prev, nxt in itertools.pairwise(chunks):
        assert nxt.split()[0] in prev.split()


def test_recursive_chunker_hard_splits_long_tokens() -> None:
    chunks = RecursiveChunker(chunk_size=10, overlap=2).split_text("x" * 35)
    assert all(len(c) <= 10 for c in chunks)
    assert "".join(c[:8] for c in chunks).startswith("x" * 24)


def test_recursive_chunker_custom_separators_and_no_overlap() -> None:
    chunker = RecursiveChunker(chunk_size=12, overlap=0, separators=["|"])
    assert chunker.separators == ["|", ""]
    assert [c.strip("|") for c in chunker.split_text("aaaa|bbbb|cccc|dddd")] == [
        "aaaa|bbbb",
        "cccc|dddd",
    ]


def test_recursive_chunker_merges_small_pieces() -> None:
    text = "a. b. c. d. e."
    assert RecursiveChunker(chunk_size=100, overlap=10).split_text(text) == [text]
