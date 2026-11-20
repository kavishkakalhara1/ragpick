"""Conversations: chat history, follow-up rewriting, and session storage.

``engine.chat(message, session_id=...)`` works on every :class:`~ragpick.Pipeline` and
:class:`~ragpick.Router`:

1. the session's recent turns are loaded (sessions belong to an *owner*: the API key
   and tenant, so one client can never read another's history);
2. a follow-up such as "and can I cancel it?" is rewritten into a standalone question
   ("Can I cancel my Lumen Plus subscription?"), by the LLM (``rewrite: llm``/``auto``)
   or offline by attaching the previous question as context (``heuristic``);
3. the standalone question is answered by the normal grounded pipeline;
4. the turn is saved (at most ``max_turns`` are kept).

Sessions live in memory (``store: memory``, LRU-bounded) or in
``<index_dir>/sessions.sqlite`` (``store: sqlite``) so they survive restarts.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ragpick.config import ConversationConfig
from ragpick.errors import RagpickError
from ragpick.interfaces import TextCompleter
from ragpick.models import Answer, StreamEvent
from ragpick.tracing import set_attributes, span

REWRITE_SYSTEM = (
    "You rewrite the user's latest message into a standalone question that can be "
    "understood without the conversation. Keep names, codes and numbers exactly. If the "
    "message is already standalone, return it unchanged. Reply with the question only."
)

REWRITE_PROMPT = """Conversation so far:
{history}

Latest message: {message}

Standalone question:"""

_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_FOLLOWUP_START = re.compile(
    r"^(and|but|so|also|then|what about|how about|and what|what else|why|how come)\b",
    re.IGNORECASE,
)
_PRONOUNS = re.compile(
    r"\b(it|its|they|them|their|that|this|those|these|one|ones)\b", re.IGNORECASE
)


class ConversationError(RagpickError):
    """An invalid or unknown session."""


class ChatTurn(BaseModel):
    """One exchange in a conversation."""

    question: str
    standalone: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)


def new_session_id() -> str:
    """A random session id."""
    return uuid.uuid4().hex


def validate_session_id(session_id: str) -> str:
    """Check a session id (8-64 letters, digits, ``-``, ``_``)."""
    if not isinstance(session_id, str) or not _SESSION_RE.match(session_id):
        raise ConversationError(
            f"Invalid session id {session_id!r}: use 8-64 letters, digits, '-' or '_'"
        )
    return session_id


class MemorySessionStore:
    """In-process session storage (LRU over ``max_sessions``)."""

    def __init__(self, max_sessions: int = 1000, ttl_seconds: float | None = None) -> None:
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._data: OrderedDict[str, tuple[str, list[ChatTurn], float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str, owner: str) -> list[ChatTurn]:
        """Turns of ``session_id`` (``[]`` if new or expired).

        Raises:
            ConversationError: the session belongs to another owner.
        """
        with self._lock:
            entry = self._data.get(session_id)
            if entry is None:
                return []
            stored_owner, turns, updated = entry
            if self.ttl_seconds is not None and time.time() - updated > self.ttl_seconds:
                del self._data[session_id]
                return []
            if stored_owner != owner:
                raise ConversationError("Unknown session")
            self._data.move_to_end(session_id)
            return list(turns)

    def append(self, session_id: str, owner: str, turn: ChatTurn, max_turns: int) -> None:
        """Add ``turn`` (keeping the last ``max_turns``)."""
        with self._lock:
            entry = self._data.get(session_id)
            if entry is not None and entry[0] != owner:
                raise ConversationError("Unknown session")
            turns = (entry[1] if entry else []) + [turn]
            self._data[session_id] = (owner, turns[-max_turns:], time.time())
            self._data.move_to_end(session_id)
            while len(self._data) > self.max_sessions:
                self._data.popitem(last=False)

    def delete(self, session_id: str, owner: str) -> bool:
        """Forget a session; ``False`` if it didn't exist."""
        with self._lock:
            entry = self._data.get(session_id)
            if entry is None:
                return False
            if entry[0] != owner:
                raise ConversationError("Unknown session")
            del self._data[session_id]
            return True


class SqliteSessionStore:
    """Session storage in a SQLite file (survives restarts; safe across threads)."""

    def __init__(self, path: str | Path, ttl_seconds: float | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, owner TEXT NOT NULL, "
                "turns TEXT NOT NULL, updated REAL NOT NULL)"
            )
            conn.commit()

    def _row(self, conn: sqlite3.Connection, session_id: str) -> tuple[str, str, float] | None:
        row = conn.execute(
            "SELECT owner, turns, updated FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return (str(row[0]), str(row[1]), float(row[2])) if row else None

    def get(self, session_id: str, owner: str) -> list[ChatTurn]:
        """Turns of ``session_id`` (``[]`` if new or expired)."""
        with self._lock, closing(sqlite3.connect(self.path)) as conn:
            row = self._row(conn, session_id)
            if row is None:
                return []
            if self.ttl_seconds is not None and time.time() - row[2] > self.ttl_seconds:
                conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                conn.commit()
                return []
            if row[0] != owner:
                raise ConversationError("Unknown session")
            return [ChatTurn.model_validate(t) for t in json.loads(row[1])]

    def append(self, session_id: str, owner: str, turn: ChatTurn, max_turns: int) -> None:
        """Add ``turn`` (keeping the last ``max_turns``)."""
        with self._lock, closing(sqlite3.connect(self.path)) as conn:
            row = self._row(conn, session_id)
            if row is not None and row[0] != owner:
                raise ConversationError("Unknown session")
            turns = [ChatTurn.model_validate(t) for t in json.loads(row[1])] if row else []
            turns = ([*turns, turn])[-max_turns:]
            conn.execute(
                "INSERT OR REPLACE INTO sessions (id, owner, turns, updated) VALUES (?, ?, ?, ?)",
                (session_id, owner, json.dumps([t.model_dump() for t in turns]), time.time()),
            )
            conn.commit()

    def delete(self, session_id: str, owner: str) -> bool:
        """Forget a session; ``False`` if it didn't exist."""
        with self._lock, closing(sqlite3.connect(self.path)) as conn:
            row = self._row(conn, session_id)
            if row is None:
                return False
            if row[0] != owner:
                raise ConversationError("Unknown session")
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            return True


SessionStore = MemorySessionStore | SqliteSessionStore


def make_store(config: ConversationConfig, index_dir: Path) -> SessionStore:
    """Session store for ``config``."""
    if config.store == "sqlite":
        return SqliteSessionStore(index_dir / "sessions.sqlite", config.ttl_seconds)
    return MemorySessionStore(config.max_sessions, config.ttl_seconds)


def is_followup(message: str) -> bool:
    """Heuristic: does ``message`` depend on earlier turns?"""
    words = message.split()
    return (
        len(words) <= 4
        or bool(_FOLLOWUP_START.match(message.strip()))
        or (len(words) <= 10 and bool(_PRONOUNS.search(message)))
    )


class QuestionRewriter:
    """Turns follow-ups into standalone questions (LLM or heuristic)."""

    def __init__(self, llm: TextCompleter | None, mode: str = "auto") -> None:
        self.mode = mode
        self.llm = llm if mode in ("auto", "llm") else None

    def rewrite(self, message: str, history: list[ChatTurn]) -> str:
        """Standalone version of ``message`` given ``history``."""
        if self.mode == "off" or not history:
            return message
        if self.llm is not None:
            transcript = "\n".join(
                f"User: {t.question}\nAssistant: {t.answer[:500]}" for t in history[-6:]
            )
            text = self.llm.complete(
                REWRITE_SYSTEM, REWRITE_PROMPT.format(history=transcript, message=message)
            ).text.strip()
            rewritten = text.splitlines()[0].strip().strip('"') if text else ""
            if rewritten:
                return rewritten
        if is_followup(message):
            previous = history[-1].standalone.rstrip("?. ")
            return f"{message.rstrip('?. ')}? (regarding: {previous})"
        return message


QueryFn = Callable[..., Answer]
StreamFn = Callable[..., Iterator[StreamEvent]]


class ConversationManager:
    """Runs chat turns for an engine; see the module docstring."""

    def __init__(
        self, config: ConversationConfig, store: SessionStore, llm: TextCompleter | None
    ) -> None:
        self.config = config
        self.store = store
        mode = config.rewrite
        if mode == "llm" and llm is None:
            raise ConversationError(
                "conversation.rewrite: llm needs a generator that supports complete()"
            )
        self.rewriter = QuestionRewriter(llm, mode)

    def _prepare(
        self, message: str, session_id: str | None, owner: str
    ) -> tuple[str, list[ChatTurn], str]:
        if not message.strip():
            raise ConversationError("Empty message")
        sid = validate_session_id(session_id) if session_id else new_session_id()
        history = self.store.get(sid, owner)
        with span("rewrite_question", turns=len(history), mode=self.rewriter.mode):
            standalone = self.rewriter.rewrite(message, history)
            set_attributes(standalone=standalone)
        return sid, history, standalone

    def _save(self, sid: str, owner: str, message: str, standalone: str, answer: Answer) -> None:
        turn = ChatTurn(
            question=message, standalone=standalone, answer=answer.text, citations=answer.citations
        )
        self.store.append(sid, owner, turn, self.config.max_turns)
        answer.session_id = sid
        answer.standalone_question = standalone

    def chat(
        self, query: QueryFn, message: str, session_id: str | None, owner: str, **kwargs: Any
    ) -> Answer:
        """Answer one chat turn with ``query`` (the engine's ``query``)."""
        sid, _, standalone = self._prepare(message, session_id, owner)
        answer = query(standalone, **kwargs)
        self._save(sid, owner, message, standalone, answer)
        return answer

    def chat_stream(
        self, stream: StreamFn, message: str, session_id: str | None, owner: str, **kwargs: Any
    ) -> Iterator[StreamEvent]:
        """Stream one chat turn; the final ``answer`` event carries ``session_id``."""
        sid, _, standalone = self._prepare(message, session_id, owner)
        for event in stream(standalone, **kwargs):
            if event.answer is not None:
                self._save(sid, owner, message, standalone, event.answer)
            yield event


def owner_key(owner: str | None, tenant: str | None) -> str:
    """Session owner: the caller identity plus tenant."""
    return f"{owner or 'local'}|{tenant or '-'}"
