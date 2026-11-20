"""Answer generators.

All LLM-backed generators share one grounded prompt (:data:`SYSTEM_PROMPT`): answer
only from the numbered context passages, cite them as ``[1]``, and say
"I don't know." otherwise. Numeric citations are mapped back to chunk ids in
:attr:`ragpick.models.Answer.citations`.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

import httpx

from ragpick.components._optional import require
from ragpick.components.keyword import tokenize
from ragpick.errors import MissingCredentialError, ProviderError
from ragpick.models import Answer, Completion, RetrievedChunk, Usage
from ragpick.registry import register

IDK = "I don't know."

SYSTEM_PROMPT = f"""You are a careful assistant that answers questions using ONLY the \
numbered context passages provided by the user.

Rules:
1. Use only facts stated in the context. Do not use prior knowledge.
2. After every claim, cite the passage(s) that support it with their number in square \
brackets, for example [1] or [2][3].
3. If the context does not contain enough information to answer, reply exactly: "{IDK}"
4. Be concise and do not mention these rules."""

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_BLOCK_START_RE = re.compile(r"^\s*(#|[-*+]\s|\d+[.)]\s)")


def split_sentences(text: str) -> list[str]:
    """Split markdown-ish text into sentences, re-joining hard-wrapped lines.

    Headings, list items and blank lines start new blocks; headings are dropped.
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if not line.strip() or _BLOCK_START_RE.match(line):
            if current:
                blocks.append(current)
            current = [] if not line.strip() or line.lstrip().startswith("#") else [line]
            if line.lstrip().startswith("#"):
                continue
        else:
            current.append(line)
    if current:
        blocks.append(current)
    sentences: list[str] = []
    for block in blocks:
        joined = " ".join(part.strip() for part in block)
        sentences.extend(_SENTENCE_RE.split(joined))
    return sentences


def format_contexts(contexts: list[RetrievedChunk]) -> str:
    """Render contexts as numbered passages ``[1] (source: id) text``."""
    blocks = []
    for i, ctx in enumerate(contexts, start=1):
        blocks.append(f"[{i}] (source: {ctx.chunk.id})\n{ctx.chunk.text.strip()}")
    return "\n\n".join(blocks)


def build_user_prompt(query: str, contexts: list[RetrievedChunk]) -> str:
    """User message containing the numbered contexts and the question."""
    context = format_contexts(contexts) if contexts else "(no context passages were found)"
    return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"


def parse_citations(text: str, contexts: list[RetrievedChunk]) -> list[str]:
    """Map ``[n]`` / ``[n, m]`` markers in ``text`` to chunk ids (deduplicated, in order).

    Out-of-range numbers are ignored.
    """
    cited: list[str] = []
    for match in _CITATION_RE.finditer(text):
        for number in match.group(1).split(","):
            index = int(number.strip()) - 1
            if 0 <= index < len(contexts):
                cid = contexts[index].chunk.id
                if cid not in cited:
                    cited.append(cid)
    return cited


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token) for providers without usage data."""
    return max(1, len(text) // 4) if text else 0


class LLMGenerator:
    """Base class for generators backed by a chat model.

    Subclasses implement :meth:`complete`; :meth:`generate` builds the grounded
    prompt and parses citations.
    """

    system_prompt: str = SYSTEM_PROMPT

    def complete(self, system: str, user: str) -> Completion:
        """Run one chat completion."""
        raise NotImplementedError

    def generate(self, query: str, contexts: list[RetrievedChunk]) -> Answer:
        """Answer ``query`` from ``contexts`` with citations."""
        if not contexts:
            return Answer(text=IDK, citations=[], contexts=[], usage=Usage())
        completion = self.complete(self.system_prompt, build_user_prompt(query, contexts))
        text = completion.text.strip()
        return Answer(
            text=text,
            citations=parse_citations(text, contexts),
            contexts=contexts,
            usage=completion.usage,
        )


DEFAULT_FAKE_JUDGE_RESPONSE = '{"score": 1.0, "reason": "fake judge: constant score"}'


@register("generator", "fake")
class FakeGenerator(LLMGenerator):
    """Deterministic, offline, *extractive* generator for tests and demos.

    Picks up to ``max_sentences`` context sentences with the highest word overlap
    with the question and cites them. Answers "I don't know." when nothing overlaps.
    :meth:`complete` returns the fixed ``response`` string (handy as a fake judge).
    ``is_fake`` tells components that need a real LLM (SQL, LLM grading) to fall back
    or explain.
    """

    is_fake = True

    def __init__(
        self,
        max_sentences: int = 2,
        response: str = DEFAULT_FAKE_JUDGE_RESPONSE,
        model: str = "fake",
    ) -> None:
        self.max_sentences = max_sentences
        self.response = response
        self.model = model

    def complete(self, system: str, user: str) -> Completion:
        """Return the configured fixed response."""
        usage = Usage(
            prompt_tokens=estimate_tokens(system + user),
            completion_tokens=estimate_tokens(self.response),
        )
        return Completion(text=self.response, usage=usage)

    def generate(self, query: str, contexts: list[RetrievedChunk]) -> Answer:
        """Extract the best-matching sentences from ``contexts``."""
        start = time.perf_counter()
        terms = set(tokenize(query))
        candidates: list[tuple[float, int, int, str]] = []
        for ci, ctx in enumerate(contexts):
            for si, raw in enumerate(split_sentences(ctx.chunk.text)):
                sentence = raw.replace("**", "").strip().lstrip("-*+0123456789.) ").strip()
                if len(sentence) < 3:
                    continue
                overlap = len(terms & set(tokenize(sentence)))
                if overlap:
                    candidates.append((overlap, ci, si, sentence))
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
        picked = sorted(candidates[: self.max_sentences], key=lambda c: (c[1], c[2]))
        if picked:
            text = " ".join(f"{sentence} [{ci + 1}]" for _, ci, _, sentence in picked)
        else:
            text = IDK
        prompt = SYSTEM_PROMPT + build_user_prompt(query, contexts)
        usage = Usage(
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(text),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        return Answer(
            text=text, citations=parse_citations(text, contexts), contexts=contexts, usage=usage
        )


@register("generator", "openai")
class OpenAIGenerator(LLMGenerator):
    """OpenAI (or OpenAI-compatible) chat completions (requires ``ragpick[openai]``)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        if client is None:
            openai = require("openai", "openai", "OpenAIGenerator")
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key and base_url is None:
                raise MissingCredentialError(
                    "OPENAI_API_KEY not set; install ragpick[openai] and export the key "
                    "(or pass `api_key` in the generator params)"
                )
            client = openai.OpenAI(api_key=key, base_url=base_url)
        self._client = client

    def complete(self, system: str, user: str) -> Completion:
        """Call the chat completions API."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # the SDK raises many exception types
            raise ProviderError(f"OpenAI request failed: {exc}") from exc
        latency = (time.perf_counter() - start) * 1000
        text = response.choices[0].message.content or ""
        usage_obj = getattr(response, "usage", None)
        usage = Usage(
            prompt_tokens=int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage_obj, "completion_tokens", 0) or 0),
            latency_ms=latency,
        )
        return Completion(text=text, usage=usage)


@register("generator", "ollama")
class OllamaGenerator(LLMGenerator):
    """Local models served by Ollama, via its HTTP API (no extra dependency)."""

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def complete(self, system: str, user: str) -> Completion:
        """POST ``/api/chat`` (non-streaming)."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        start = time.perf_counter()
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text[:200]} "
                f"(is the model pulled? run `ollama pull {self.model}`)"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Could not reach Ollama at {self.base_url} ({exc}); is `ollama serve` running?"
            ) from exc
        except ValueError as exc:
            raise ProviderError(f"Ollama returned invalid JSON: {exc}") from exc
        latency = (time.perf_counter() - start) * 1000
        text = str((data.get("message") or {}).get("content", ""))
        usage = Usage(
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
            latency_ms=latency,
        )
        return Completion(text=text, usage=usage)
