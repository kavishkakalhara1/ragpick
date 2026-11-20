"""Long-context retrieval: give the model whole documents instead of chunks.

Modern models read 100k+ tokens. For small corpora, chopping documents into chunks
throws away context the model could use. The ``long_context`` retriever:

1. keeps every ingested document whole (``<index_dir>/long_context/docs.json``);
2. ranks **documents** with hybrid chunk search: a document's score is the sum of its
   chunks' fused scores;
3. returns whole documents, best first, until ``max_context_tokens`` is reached (a
   document that doesn't fit is truncated, and marked as such).

With ``include_all: true`` every document the caller may see is returned, in a stable
order, whenever they all fit the budget. The prompt prefix is then identical across
questions, which is what provider-side prompt caching (e.g. OpenAI's automatic caching
of long prefixes) needs to make repeated questions cheap.

Access control applies to documents: a document is returned only if its metadata
(including ``allowed_groups``) matches the caller's filters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragpick.components.generators import estimate_tokens
from ragpick.components.retrievers import HybridRetriever
from ragpick.errors import ConfigError
from ragpick.filters import matches_filters
from ragpick.interfaces import Embedder, KeywordIndex, VectorStore
from ragpick.models import Chunk, Document, RetrievedChunk
from ragpick.registry import register
from ragpick.tracing import span

DOCS_FILE = "docs.json"
TRUNCATED = "\n\n[document truncated to fit the context budget]"


@register("retriever", "long_context")
class LongContextRetriever:
    """Whole-document retrieval within a token budget; see the module docstring.

    Args:
        max_docs: most documents to return (ignored when ``include_all`` applies).
        max_context_tokens: budget for all returned documents (≈ 4 characters/token).
        include_all: return every visible document when they all fit the budget.
        candidates: chunks retrieved to rank documents.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        keyword_index: KeywordIndex,
        index_dir: Path,
        k: int = 3,
        max_docs: int = 3,
        max_context_tokens: int = 100_000,
        include_all: bool = True,
        candidates: int = 40,
        rrf_k: int = 60,
    ) -> None:
        if max_docs < 1:
            raise ConfigError("max_docs must be >= 1")
        if max_context_tokens < 100:
            raise ConfigError("max_context_tokens must be >= 100")
        self.base = HybridRetriever(embedder, store, keyword_index, k=candidates, rrf_k=rrf_k)
        self.path = Path(index_dir) / "long_context"
        self.k = k
        self.max_docs = max_docs
        self.max_context_tokens = max_context_tokens
        self.include_all = include_all
        self.candidates = candidates
        self.docs: dict[str, Document] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        file = self.path / DOCS_FILE
        if file.is_file():
            data = json.loads(file.read_text(encoding="utf-8"))
            self.docs = {d["id"]: Document.model_validate(d) for d in data["documents"]}

    def index_documents(self, docs: list[Document]) -> None:
        """Keep whole documents (replacing earlier versions with the same id)."""
        self._load()
        for doc in docs:
            if doc.text.strip():
                self.docs[doc.id] = doc
        self.path.mkdir(parents=True, exist_ok=True)
        payload = {"format": 1, "documents": [d.model_dump() for d in self.docs.values()]}
        (self.path / DOCS_FILE).write_text(json.dumps(payload), encoding="utf-8")

    def _context(self, doc: Document, score: float, budget: int) -> tuple[RetrievedChunk, int]:
        tokens = estimate_tokens(doc.text)
        text = doc.text
        truncated = tokens > budget
        if truncated:
            text = doc.text[: max(0, budget * 4 - len(TRUNCATED))] + TRUNCATED
            tokens = budget
        chunk = Chunk(
            id=f"{doc.id}#full",
            doc_id=doc.id,
            text=text,
            metadata={**doc.metadata, "whole_document": True, "truncated": truncated},
        )
        return RetrievedChunk(chunk=chunk, score=score, source="fused"), tokens

    def retrieve(
        self, query: str, k: int | None = None, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Whole documents for ``query``, best first, within the token budget."""
        self._load()
        with span("long_context", documents=len(self.docs)) as s:
            visible = {i: d for i, d in self.docs.items() if matches_filters(d.metadata, filters)}
            scores: dict[str, float] = {}
            for hit in self.base.retrieve(query, self.candidates, filters):
                if hit.chunk.doc_id in visible:
                    scores[hit.chunk.doc_id] = scores.get(hit.chunk.doc_id, 0.0) + hit.score
            total = sum(estimate_tokens(d.text) for d in visible.values())
            everything = bool(self.include_all and visible and total <= self.max_context_tokens)
            if everything:
                order = sorted(visible)  # stable prefix → provider prompt caching
            else:
                ranked = sorted(scores, key=lambda i: (-scores[i], i))
                order = ranked[: min(self.max_docs, k or self.max_docs)]
            budget = self.max_context_tokens
            results: list[RetrievedChunk] = []
            for doc_id in order:
                if budget <= 0:
                    break
                context, used = self._context(visible[doc_id], scores.get(doc_id, 0.0), budget)
                results.append(context)
                budget -= used
            if s is not None:
                s.attributes.update(
                    visible=len(visible),
                    returned=len(results),
                    mode="all" if everything else "ranked",
                    tokens=self.max_context_tokens - budget,
                )
        return results
