"""Multimodal ingest: describe images so they can be searched like text.

The ``multimodal`` loader reads a directory of text, Markdown, PDFs **and images**
(``.png``, ``.jpg``, ``.jpeg``, ``.webp``, ``.gif``), plus images embedded in PDFs.
Every image becomes a document whose text is a description:

* **vision** (``describe: auto`` with a vision-capable generator such as OpenAI
  ``gpt-4o-mini`` or an Ollama ``llava`` model): the model describes the image,
  transcribing visible text, numbers and labels;
* **offline** (fake generator, or ``describe: offline``): the description is built
  from the file name, the Markdown alt text of every ``![alt](image.png)`` reference,
  and an optional caption file next to the image (``diagram.png.txt``/``diagram.txt``).

Image documents carry ``modality: image``, ``image_path`` and ``mime`` metadata, so a
UI can show the picture next to the answer. Front matter and ``allowed_groups``
work as for text: a caption file may start with front matter to restrict an image.
"""

from __future__ import annotations

import logging
import mimetypes
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ragpick.components._optional import require
from ragpick.components.loaders import PdfLoader, TextLoader, split_front_matter
from ragpick.errors import ConfigError, ProviderError
from ragpick.interfaces import Generator, VisionCompleter
from ragpick.models import Document
from ragpick.registry import register

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})
TEXT_EXTENSIONS = frozenset({".txt", ".md", ".markdown"})

DESCRIBE_PROMPT = (
    "Describe this image for a search index. State what it shows, transcribe all "
    "visible text verbatim (labels, headings, error messages, table cells), list key "
    "numbers and values, and say what a reader would learn from it. Be factual and "
    "concise; don't speculate."
)

_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _words(stem: str) -> str:
    return " ".join(re.split(r"[-_\s]+", stem)).strip()


def _mime(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


@register("loader", "multimodal")
class MultimodalLoader:
    """Loads text, Markdown, PDFs and images; images become described documents.

    Args:
        describe: ``auto`` (vision when the generator supports it), ``vision`` or
            ``offline``.
        prompt: instructions for the vision model.
        pdf_images: also describe images embedded in PDFs (needs ``ragpick[pdf]``).
        max_image_bytes: larger images are skipped (with a warning).
    """

    def __init__(
        self,
        generator: Generator,
        glob: str = "**/*",
        describe: str = "auto",
        prompt: str = DESCRIBE_PROMPT,
        pdf_images: bool = True,
        max_image_bytes: int = 20_000_000,
        encoding: str = "utf-8",
    ) -> None:
        if describe not in ("auto", "vision", "offline"):
            raise ConfigError(f"describe must be auto, vision or offline, got {describe!r}")
        can_see = isinstance(generator, VisionCompleter) and not getattr(
            generator, "is_fake", False
        )
        if describe == "vision" and not can_see:
            raise ConfigError(
                "describe: vision needs a vision-capable generator (openai or ollama with a "
                "vision model)"
            )
        self.vision: VisionCompleter | None = (
            generator if can_see and describe != "offline" else None  # type: ignore[assignment]
        )
        self.glob = glob
        self.prompt = prompt
        self.pdf_images = pdf_images
        self.max_image_bytes = max_image_bytes
        self._text = TextLoader(encoding=encoding)
        self._pdf = PdfLoader()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def alt_texts(files: list[Path]) -> dict[Path, list[str]]:
        """Alt texts of ``![alt](path)`` references in Markdown files, by resolved image path."""
        found: dict[Path, list[str]] = {}
        for md in files:
            if md.suffix.lower() not in (".md", ".markdown"):
                continue
            for alt, target in _MD_IMAGE_RE.findall(
                md.read_text(encoding="utf-8", errors="replace")
            ):
                if "://" in target:
                    continue
                path = (md.parent / target).resolve()
                if alt.strip() and alt.strip() not in found.setdefault(path, []):
                    found[path].append(alt.strip())
        return found

    @staticmethod
    def caption(image: Path) -> tuple[dict[str, Any], str]:
        """``(front matter, text)`` of a caption file next to ``image``, if any."""
        for candidate in (image.with_name(image.name + ".txt"), image.with_suffix(".txt")):
            if candidate.is_file():
                meta, text = split_front_matter(candidate.read_text(encoding="utf-8"))
                return meta, text.strip()
        return {}, ""

    def describe_bytes(self, data: bytes, mime: str, fallback: str) -> tuple[str, str]:
        """``(description, source)`` using the vision model, or the offline fallback."""
        if self.vision is None:
            return fallback, "offline"
        try:
            text = self.vision.describe_image(data, mime, self.prompt).text.strip()
        except ProviderError as exc:
            logger.warning("Image description failed (%s); using the offline description", exc)
            return fallback, "offline-fallback"
        return (f"{text}\n\n{fallback}" if text else fallback), "vision"

    def _image_doc(self, path: Path, root: Path, alts: dict[Path, list[str]]) -> Document | None:
        size = path.stat().st_size
        if size > self.max_image_bytes:
            logger.warning("Skipping %s (%d bytes > max_image_bytes)", path, size)
            return None
        cap_meta, cap_text = self.caption(path)
        parts = [f"Image: {_words(path.stem)}."]
        alt = alts.get(path.resolve(), [])
        if alt:
            parts.append("Alt text: " + "; ".join(alt) + ".")
        if cap_text:
            parts.append(f"Caption: {cap_text}")
        description, source = self.describe_bytes(path.read_bytes(), _mime(path), " ".join(parts))
        meta = {
            "source": str(path),
            "title": path.stem,
            "modality": "image",
            "image_path": str(path),
            "mime": _mime(path),
            "description_source": source,
            **cap_meta,
        }
        return Document(id=path.relative_to(root).as_posix(), text=description, metadata=meta)

    def _pdf_image_docs(self, path: Path, root: Path, pdf_doc: Document) -> Iterator[Document]:
        pypdf = require("pypdf", "pdf", "MultimodalLoader(pdf_images=True)")
        reader = pypdf.PdfReader(str(path))
        base_id = path.relative_to(root).as_posix()
        for page_no, page in enumerate(reader.pages, start=1):
            for i, image in enumerate(getattr(page, "images", []) or [], start=1):
                data = image.data
                if len(data) > self.max_image_bytes:
                    continue
                mime = mimetypes.guess_type(image.name)[0] or "image/png"
                fallback = f"Image {i} on page {page_no} of {path.name}."
                description, source = self.describe_bytes(data, mime, fallback)
                meta = {
                    **pdf_doc.metadata,
                    "modality": "image",
                    "image_path": f"{path}#page={page_no}",
                    "mime": mime,
                    "page": page_no,
                    "description_source": source,
                }
                yield Document(id=f"{base_id}:image-{page_no}-{i}", text=description, metadata=meta)

    # ------------------------------------------------------------------ loading
    def load(self, path: str) -> Iterator[Document]:
        """Yield text, PDF and image documents found at ``path``."""
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"Path not found: {root}")
        files = [root] if root.is_file() else sorted(p for p in root.glob(self.glob) if p.is_file())
        base = root.parent if root.is_file() else root
        alts = self.alt_texts(files)
        for file in files:
            suffix = file.suffix.lower()
            if file.name.endswith(tuple(f"{ext}.txt" for ext in IMAGE_EXTENSIONS)):
                continue  # caption file, used by its image
            if suffix in TEXT_EXTENSIONS:
                if suffix == ".txt" and any(
                    file.with_suffix(ext).is_file() for ext in IMAGE_EXTENSIONS
                ):
                    continue  # caption file for an image of the same name
                yield self._text.load_file(file, base)
            elif suffix == ".pdf":
                doc = self._pdf.load_file(file, base)
                yield doc
                if self.pdf_images:
                    yield from self._pdf_image_docs(file, base, doc)
            elif suffix in IMAGE_EXTENSIONS:
                image_doc = self._image_doc(file, base, alts)
                if image_doc is not None:
                    yield image_doc
            else:
                logger.debug("Skipping unsupported file %s", file)
