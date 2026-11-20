"""Pure-Python BM25 keyword index (Okapi BM25)."""

from __future__ import annotations

import heapq
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ragpick.filters import matches_filters
from ragpick.models import Chunk, RetrievedChunk
from ragpick.registry import register

# Words joined by -, _, ., / or : (e.g. "ABC-123", "v2.1", "ISO/IEC") form one token.
# Their parts are indexed too, so "ABC-123" also matches a query for "abc".
_TOKEN_RE = re.compile(r"\w+(?:[-_./:]\w+)*", re.UNICODE)
_PART_RE = re.compile(r"[-_./:]")

ENGLISH_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "not",
        "of",
        "on",
        "or",
        "our",
        "so",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)


def tokenize(text: str, stopwords: frozenset[str] = ENGLISH_STOPWORDS) -> list[str]:
    """Lowercase, split into word tokens, keep compound tokens and their parts."""
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        parts = [p for p in _PART_RE.split(token) if p]
        if len(parts) > 1:
            tokens.append(token)
            tokens.extend(p for p in parts if p not in stopwords)
        elif token not in stopwords:
            tokens.append(token)
    return tokens


@register("keyword_index", "bm25")
class BM25Index:
    """Okapi BM25 over chunk texts with metadata filtering.

    ``score(q, d) = sum_t idf(t) * tf(t,d)*(k1+1) / (tf(t,d) + k1*(1 - b + b*|d|/avgdl))``
    with ``idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))`` (always positive).
    """

    supports_filters = True

    def __init__(
        self, k1: float = 1.5, b: float = 0.75, stopwords: list[str] | None = None
    ) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 >= 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self.stopwords = frozenset(stopwords) if stopwords is not None else ENGLISH_STOPWORDS
        self._chunks: dict[str, Chunk] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._len: dict[str, int] = {}
        self._postings: dict[str, set[str]] = {}
        self._total_len = 0

    def __len__(self) -> int:
        return len(self._chunks)

    def _remove(self, chunk_id: str) -> None:
        tf = self._tf.pop(chunk_id)
        for term in tf:
            posting = self._postings[term]
            posting.discard(chunk_id)
            if not posting:
                del self._postings[term]
        self._total_len -= self._len.pop(chunk_id)
        del self._chunks[chunk_id]

    def add(self, chunks: list[Chunk]) -> None:
        """Index chunks (replacing chunks with the same id)."""
        for chunk in chunks:
            if chunk.id in self._chunks:
                self._remove(chunk.id)
            tokens = tokenize(chunk.text, self.stopwords)
            tf = Counter(tokens)
            self._chunks[chunk.id] = chunk.model_copy(update={"embedding": None})
            self._tf[chunk.id] = tf
            self._len[chunk.id] = len(tokens)
            self._total_len += len(tokens)
            for term in tf:
                self._postings.setdefault(term, set()).add(chunk.id)

    def delete(self, doc_ids: list[str]) -> None:
        """Remove all chunks of ``doc_ids``."""
        drop = set(doc_ids)
        for chunk_id in [cid for cid, c in self._chunks.items() if c.doc_id in drop]:
            self._remove(chunk_id)

    def idf(self, term: str) -> float:
        """Inverse document frequency of ``term``."""
        n = len(self._chunks)
        df = len(self._postings.get(term, ()))
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> dict[str, float]:
        """BM25 score for every chunk containing at least one query term."""
        if not self._chunks:
            return {}
        avgdl = self._total_len / len(self._chunks) or 1.0
        result: dict[str, float] = {}
        for term in set(tokenize(query, self.stopwords)):
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = self.idf(term)
            for chunk_id in posting:
                tf = self._tf[chunk_id][term]
                norm = self.k1 * (1 - self.b + self.b * self._len[chunk_id] / avgdl)
                result[chunk_id] = result.get(chunk_id, 0.0) + idf * tf * (self.k1 + 1) / (
                    tf + norm
                )
        return result

    def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Top-``k`` chunks by BM25 score among those matching ``filters``."""
        if k <= 0:
            return []
        candidates = (
            (score, cid)
            for cid, score in self.scores(query).items()
            if matches_filters(self._chunks[cid].metadata, filters)
        )
        # ties broken by chunk id for determinism
        best = heapq.nsmallest(k, candidates, key=lambda sc: (-sc[0], sc[1]))
        return [
            RetrievedChunk(chunk=self._chunks[cid], score=score, source="keyword")
            for score, cid in best
        ]

    def persist(self, path: str) -> None:
        """Write chunks and parameters to ``path/bm25.json`` (postings are rebuilt on load)."""
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        data = {
            "format": 1,
            "k1": self.k1,
            "b": self.b,
            "stopwords": sorted(self.stopwords),
            "chunks": [c.model_dump(exclude={"embedding"}) for c in self._chunks.values()],
        }
        (root / "bm25.json").write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: str) -> None:
        """Replace the index with the one persisted at ``path``."""
        file = Path(path) / "bm25.json"
        if not file.is_file():
            raise FileNotFoundError(f"No BM25 index found at {file.parent}")
        data = json.loads(file.read_text(encoding="utf-8"))
        self.__init__(k1=data["k1"], b=data["b"], stopwords=data["stopwords"])  # type: ignore[misc]
        self.add([Chunk.model_validate(c) for c in data["chunks"]])
