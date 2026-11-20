"""Generate an evaluation dataset from your own documents.

Writing 50 good questions by hand is the main reason teams skip evaluation.
:func:`generate_dataset` samples chunks evenly across documents and turns each into a
question with a reference answer:

* **LLM** (``mode: auto`` with a real generator, or ``llm``): the model writes a natural,
  specific question answered by the chunk, plus a short answer;
* **offline** (``heuristic``): questions come from section headings ("How do I reset your
  password?"), codes ("What does E-1042 mean?") or document titles.

Every row records ``relevant_doc_ids`` (the chunk's document), so retrieval metrics work
immediately, and rows from access-restricted documents carry those documents'
``groups``, so evaluation runs with the right permissions and nothing leaks.
Review the generated file before trusting it: it's a starting point, not ground truth.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from pathlib import Path

from ragpick.components.generators import split_sentences
from ragpick.errors import ConfigError
from ragpick.eval.dataset import EvalExample
from ragpick.filters import ACL_KEY
from ragpick.interfaces import TextCompleter
from ragpick.models import Chunk
from ragpick.pipeline import Pipeline

GENERATE_SYSTEM = (
    "You write evaluation questions for a document search system. "
    'Respond with JSON only: {"question": "...", "answer": "..."}'
)

GENERATE_PROMPT = """Write one question that a real user might ask, which this passage answers,
and the answer. Rules:
- The question must be specific and answerable from this passage alone.
- Don't copy sentences verbatim, and don't mention "the passage" or "the document".
- The answer is one or two sentences, using only facts from the passage.

Passage (from {doc_id}):
{text}"""

_CODE_RE = re.compile(r"\b[A-Z]{1,6}-\d{2,6}\b")
_HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)
_IMPERATIVE = frozenset(
    {
        "reset", "enable", "disable", "delete", "install", "set", "change", "add", "remove",
        "create", "update", "connect", "cancel", "export", "import", "share", "configure",
        "upgrade", "download", "restart", "register", "return",
    }
)  # fmt: skip


def _first_sentence(text: str) -> str:
    for sentence in split_sentences(text):
        clean = sentence.replace("**", "").strip().lstrip("-*0123456789. ").strip()
        if len(clean.split()) >= 4:
            return clean
    return ""


def heuristic_question(chunk: Chunk) -> tuple[str, str] | None:
    """An offline ``(question, answer)`` for ``chunk``, or ``None``."""
    text = chunk.text
    for match in _HEADING_RE.finditer(text):
        heading = match.group(2).strip()
        answer = _first_sentence(text[match.end() :])
        if not answer or len(heading.split()) < 2:
            continue
        first = heading.split()[0].lower()
        if first in _IMPERATIVE:
            return f"How do I {heading[0].lower()}{heading[1:]}?", answer
        return f"What should I know about {heading[0].lower()}{heading[1:]}?", answer
    for sentence in split_sentences(text):
        code = _CODE_RE.search(sentence)
        if code:
            return f"What does {code.group(0)} mean?", sentence.replace("**", "").strip(" -*")
    title = str(chunk.metadata.get("title") or "").replace("-", " ").strip()
    answer = _first_sentence(text)
    if title and answer:
        return f"What does the documentation say about {title}?", answer
    return None


def parse_generated(reply: str) -> tuple[str, str] | None:
    """``(question, answer)`` from an LLM reply, or ``None``."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        question = str(data["question"]).strip()
        answer = str(data.get("answer", "")).strip()
    except (ValueError, KeyError, TypeError):
        return None
    return (question, answer) if question.endswith("?") and len(question) > 10 else None


def sample_chunks(chunks: list[Chunk], count: int, seed: int = 0) -> list[Chunk]:
    """Pick up to ``count`` chunks round-robin across documents (deterministic)."""
    rng = random.Random(seed)
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)
    queues = []
    for doc_id in sorted(by_doc):
        items = list(by_doc[doc_id])
        rng.shuffle(items)
        queues.append(items)
    picked: list[Chunk] = []
    while len(picked) < count and any(queues):
        for queue in queues:
            if queue and len(picked) < count:
                picked.append(queue.pop())
    return picked


def generate_dataset(
    pipeline: Pipeline,
    path: str | Path | None = None,
    *,
    count: int = 20,
    seed: int = 0,
    mode: str = "auto",
    min_chars: int = 120,
    progress: Callable[[int, int], None] | None = None,
) -> list[EvalExample]:
    """Generate up to ``count`` evaluation examples from the documents at ``path``.

    Uses the pipeline's loader and chunker (so ids match what ``ingest`` produces) and
    its generator for LLM questions. ``path`` defaults to the config's ``data``.
    """
    if mode not in ("auto", "llm", "heuristic"):
        raise ConfigError(f"mode must be auto, llm or heuristic, got {mode!r}")
    generator = pipeline.generator
    real_llm = isinstance(generator, TextCompleter) and not getattr(generator, "is_fake", False)
    if mode == "llm" and not isinstance(generator, TextCompleter):
        raise ConfigError("mode: llm needs a generator that supports complete()")
    llm: TextCompleter | None = None
    if isinstance(generator, TextCompleter) and (mode == "llm" or (mode == "auto" and real_llm)):
        llm = generator
    if path is None:
        if pipeline.config.data is None:
            raise ConfigError("No path given and no `data` set in the config")
        path = pipeline.config.resolve_path(pipeline.config.data)
    docs = list(pipeline.loader.load(str(path)))
    chunks = [
        c
        for c in pipeline.chunker.split(docs)
        if len(c.text) >= min_chars and not c.metadata.get("synthetic")
    ]
    if not chunks:
        raise ConfigError(
            f"No chunks of at least {min_chars} characters at {path} "
            "(dataset generation needs text documents and a chunker that produces chunks)"
        )
    examples: list[EvalExample] = []
    seen: set[str] = set()
    candidates = sample_chunks(chunks, max(count * 2, count + 5), seed)
    for chunk in candidates:
        if len(examples) >= count:
            break
        if llm is not None:
            reply = llm.complete(
                GENERATE_SYSTEM, GENERATE_PROMPT.format(doc_id=chunk.doc_id, text=chunk.text[:3000])
            ).text
            pair = parse_generated(reply) or heuristic_question(chunk)
        else:
            pair = heuristic_question(chunk)
        if pair is None or pair[0].lower() in seen:
            continue
        seen.add(pair[0].lower())
        groups = chunk.metadata.get(ACL_KEY)
        examples.append(
            EvalExample(
                id=f"gen-{len(examples) + 1}",
                question=pair[0],
                ground_truth=pair[1] or None,
                relevant_doc_ids=[chunk.doc_id],
                groups=list(groups) if isinstance(groups, list) and groups else None,
            )
        )
        if progress:
            progress(len(examples), min(count, len(candidates)))
    return examples


def write_dataset(examples: list[EvalExample], path: str | Path) -> Path:
    """Write examples as JSONL (``None`` fields omitted)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(e.model_dump(exclude_none=True), ensure_ascii=False) for e in examples]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
