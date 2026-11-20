"""Document loaders."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from ragpick.components._optional import require
from ragpick.models import Document
from ragpick.registry import register

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = frozenset({".txt", ".md", ".markdown"})


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split a leading YAML front-matter block (``---`` … ``---``) from ``text``.

    Returns ``({}, text)`` when there is no valid front matter.
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines(keepends=True)
    if lines[0].strip() != "---":
        return {}, text
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            try:
                meta = yaml.safe_load("".join(lines[1:i])) or {}
            except yaml.YAMLError:
                return {}, text
            if not isinstance(meta, dict):
                return {}, text
            return meta, "".join(lines[i + 1 :]).lstrip("\n")
    return {}, text


def _doc_id(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:  # pragma: no cover - defensive
            pass
    return path.name


@register("loader", "text")
class TextLoader:
    """Loads ``.txt`` and ``.md`` files.

    Markdown YAML front matter becomes document metadata, which is how access
    control is attached to documents::

        ---
        allowed_groups: [hr]
        ---
    """

    def __init__(self, encoding: str = "utf-8", front_matter: bool = True) -> None:
        self.encoding = encoding
        self.front_matter = front_matter

    def load(self, path: str) -> Iterator[Document]:
        """Load the single file at ``path``."""
        yield self.load_file(Path(path))

    def load_file(self, path: Path, root: Path | None = None) -> Document:
        """Load ``path``, using its path relative to ``root`` as the document id."""
        text = path.read_text(encoding=self.encoding, errors="replace")
        meta: dict[str, Any] = {}
        if self.front_matter:
            meta, text = split_front_matter(text)
        meta.setdefault("source", str(path))
        meta.setdefault("title", path.stem)
        return Document(id=_doc_id(path, root), text=text, metadata=meta)


@register("loader", "pdf")
class PdfLoader:
    """Loads PDF files page by page (requires ``ragpick[pdf]``)."""

    def load(self, path: str) -> Iterator[Document]:
        """Load the single PDF at ``path``."""
        yield self.load_file(Path(path))

    def load_file(self, path: Path, root: Path | None = None) -> Document:
        """Load ``path`` into one document; pages are separated by blank lines."""
        pypdf = require("pypdf", "pdf", "PdfLoader")
        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        meta = {"source": str(path), "title": path.stem, "pages": len(pages)}
        return Document(id=_doc_id(path, root), text="\n\n".join(pages), metadata=meta)


@register("loader", "directory")
class DirectoryLoader:
    """Walks a directory and dispatches each file to a loader by extension.

    Files with unsupported extensions are skipped (and logged at debug level).
    ``path`` may also be a single file.
    """

    def __init__(
        self,
        glob: str = "**/*",
        encoding: str = "utf-8",
        include_pdf: bool = True,
        exclude: list[str] | None = None,
    ) -> None:
        self.glob = glob
        self.exclude = list(exclude or [])
        self._text = TextLoader(encoding=encoding)
        self._pdf = PdfLoader() if include_pdf else None

    def _loader_for(self, path: Path) -> TextLoader | PdfLoader | None:
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            return self._text
        if suffix == ".pdf":
            return self._pdf
        return None

    def load(self, path: str) -> Iterator[Document]:
        """Yield documents for all supported files under ``path``."""
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"Path not found: {root}")
        if root.is_file():
            files, base = [root], root.parent
        else:
            files, base = sorted(p for p in root.glob(self.glob) if p.is_file()), root
        for file in files:
            rel = file.relative_to(base)
            if any(rel.match(pattern) for pattern in self.exclude):
                continue
            loader = self._loader_for(file)
            if loader is None:
                logger.debug("Skipping unsupported file %s", file)
                continue
            yield loader.load_file(file, base)
