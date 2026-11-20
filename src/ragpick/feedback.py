"""Answer feedback: collect ratings, find weak spots, turn them into evaluation data.

Every :class:`~ragpick.models.Answer` has an ``id`` and records its ``question``.
Ratings are stored in a SQLite file next to the index:

* Python: ``pipeline.record_feedback(answer, "up" | "down" | 1-5, comment=...)``;
* HTTP: ``POST /feedback {"answer_id", "rating", "comment"?}``. The server only
  accepts ratings for answers it gave to the same API key and tenant, so ratings
  can't be forged or attributed to someone else's answer.

``ragpick feedback stats`` shows satisfaction and the worst-rated questions;
``ragpick feedback export --as-eval`` writes liked answers as regression tests (their
cited documents become ``relevant_doc_ids``) and disliked ones as questions to review.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import closing
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ragpick.errors import RagpickError
from ragpick.models import Answer

POSITIVE = 0.75
NEGATIVE = 0.25
_UP = {"up", "+1", "1", "good", "yes", "thumbs_up", "like", "helpful"}
_DOWN = {"down", "-1", "0", "bad", "no", "thumbs_down", "dislike", "unhelpful"}


class FeedbackError(RagpickError):
    """Invalid feedback (unknown answer, bad rating)."""


def normalize_rating(rating: str | int | float | bool) -> float:
    """Map ``"up"/"down"``, ``True/False`` or a 1-5 star rating to a score in 0..1."""
    if isinstance(rating, bool):
        return 1.0 if rating else 0.0
    if isinstance(rating, int | float):
        if 1 <= rating <= 5:
            return (float(rating) - 1.0) / 4.0
        raise FeedbackError(f"Numeric ratings must be 1-5, got {rating}")
    value = str(rating).strip().lower()
    if value in _UP:
        return 1.0
    if value in _DOWN:
        return 0.0
    try:
        return normalize_rating(float(value))
    except ValueError:
        raise FeedbackError(f"Unknown rating {rating!r}: use up/down or 1-5") from None


class FeedbackRecord(BaseModel):
    """One stored rating."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created: float = Field(default_factory=time.time)
    answer_id: str
    rating: float
    raw_rating: str
    comment: str | None = None
    question: str | None = None
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    cited_docs: list[str] = Field(default_factory=list)
    route: str | None = None
    owner: str = "local"
    tenant: str | None = None
    groups: list[str] | None = None


def record_from_answer(
    answer: Answer,
    rating: str | int | float | bool,
    *,
    comment: str | None = None,
    owner: str | None = None,
    tenant: str | None = None,
    groups: list[str] | None = None,
) -> FeedbackRecord:
    """Build a :class:`FeedbackRecord` for ``answer``."""
    cited = set(answer.citations)
    docs = sorted(
        {
            c.chunk.doc_id
            for c in answer.contexts
            if c.chunk.id in cited and not c.chunk.metadata.get("synthetic")
        }
    )
    return FeedbackRecord(
        answer_id=answer.id,
        rating=normalize_rating(rating),
        raw_rating=str(rating),
        comment=(comment or None) and comment[:2000],
        question=answer.question,
        answer=answer.text,
        citations=answer.citations,
        cited_docs=docs,
        route=answer.route,
        owner=owner or "local",
        tenant=tenant,
        groups=groups,
    )


class FeedbackStore:
    """Ratings in a SQLite file (thread-safe)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS feedback "
                "(id TEXT PRIMARY KEY, record TEXT NOT NULL, created REAL NOT NULL)"
            )
            conn.commit()

    def add(self, record: FeedbackRecord) -> str:
        """Store ``record``; returns its id."""
        with self._lock, closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                "INSERT INTO feedback (id, record, created) VALUES (?, ?, ?)",
                (record.id, record.model_dump_json(), record.created),
            )
            conn.commit()
        return record.id

    def all(self) -> list[FeedbackRecord]:
        """Every record, oldest first."""
        with self._lock, closing(sqlite3.connect(self.path)) as conn:
            rows = conn.execute("SELECT record FROM feedback ORDER BY created, id").fetchall()
        return [FeedbackRecord.model_validate_json(r[0]) for r in rows]

    def stats(self, worst: int = 5) -> dict[str, Any]:
        """Totals, average score, per-route counts and the worst-rated questions."""
        records = self.all()
        by_route: dict[str, list[float]] = {}
        by_question: dict[str, list[float]] = {}
        for r in records:
            by_route.setdefault(r.route or "-", []).append(r.rating)
            if r.question:
                by_question.setdefault(r.question, []).append(r.rating)
        ranked = sorted(
            by_question.items(), key=lambda kv: (sum(kv[1]) / len(kv[1]), -len(kv[1]), kv[0])
        )
        return {
            "total": len(records),
            "positive": sum(1 for r in records if r.rating >= POSITIVE),
            "negative": sum(1 for r in records if r.rating <= NEGATIVE),
            "average": (sum(r.rating for r in records) / len(records)) if records else None,
            "routes": {
                k: {"count": len(v), "average": sum(v) / len(v)}
                for k, v in sorted(by_route.items())
            },
            "worst_questions": [
                {"question": q, "average": sum(v) / len(v), "count": len(v)}
                for q, v in ranked[:worst]
                if sum(v) / len(v) < POSITIVE
            ],
        }

    def export_eval(self) -> list[dict[str, Any]]:
        """Evaluation rows: liked answers → regression tests, disliked → to review."""
        rows: list[dict[str, Any]] = []
        positives: dict[str, FeedbackRecord] = {}
        negatives: dict[str, FeedbackRecord] = {}
        for r in self.all():
            if not r.question:
                continue
            if r.rating >= POSITIVE and r.cited_docs:
                positives[r.question] = r
            elif r.rating <= NEGATIVE:
                negatives[r.question] = r
        for n, r in enumerate(positives.values(), start=1):
            row: dict[str, Any] = {
                "id": f"feedback-{n}",
                "question": r.question,
                "ground_truth": r.answer,
                "relevant_doc_ids": r.cited_docs,
            }
            if r.groups:
                row["groups"] = r.groups
            rows.append(row)
        for n, r in enumerate((r for q, r in negatives.items() if q not in positives), start=1):
            row = {"id": f"feedback-review-{n}", "question": r.question}
            if r.groups:
                row["groups"] = r.groups
            rows.append(row)
        return rows


class AnswerLog:
    """Recent answers the server returned, keyed by id, with their owner (LRU)."""

    def __init__(self, size: int = 10_000) -> None:
        self.size = size
        self._data: OrderedDict[str, tuple[str, list[str] | None, Answer]] = OrderedDict()
        self._lock = threading.Lock()

    def remember(self, answer: Answer, owner: str, groups: list[str] | None) -> None:
        """Keep ``answer`` so its owner can rate it later."""
        with self._lock:
            self._data[answer.id] = (owner, groups, answer)
            self._data.move_to_end(answer.id)
            while len(self._data) > self.size:
                self._data.popitem(last=False)

    def get(self, answer_id: str, owner: str) -> tuple[Answer, list[str] | None]:
        """The answer and the groups it was produced for.

        Raises:
            FeedbackError: unknown, expired, or someone else's answer (indistinguishable).
        """
        with self._lock:
            entry = self._data.get(answer_id)
        if entry is None or entry[0] != owner:
            raise FeedbackError("Unknown answer")
        return entry[2], entry[1]


def export_jsonl(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write ``rows`` as JSON Lines."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return p
