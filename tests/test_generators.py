from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from ragpick.components.generators import (
    IDK,
    SYSTEM_PROMPT,
    FakeGenerator,
    LLMGenerator,
    OllamaGenerator,
    OpenAIGenerator,
    build_user_prompt,
    estimate_tokens,
    parse_citations,
)
from ragpick.errors import MissingCredentialError, MissingDependencyError, ProviderError
from ragpick.interfaces import Generator, TextCompleter
from ragpick.models import Chunk, RetrievedChunk


def ctx(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(id=cid, doc_id=cid, text=text), score=1.0, source="vector")


CONTEXTS = [
    ctx("pw.md#0", "Tap Forgot password on the sign-in screen. The link is valid for 30 minutes."),
    ctx("bill.md#0", "Lumen Plus costs $4.99 per month."),
]


def test_system_prompt_rules() -> None:
    assert "ONLY" in SYSTEM_PROMPT
    assert "[1]" in SYSTEM_PROMPT
    assert IDK in SYSTEM_PROMPT


def test_build_user_prompt_numbers_contexts() -> None:
    prompt = build_user_prompt("How?", CONTEXTS)
    assert "[1] (source: pw.md#0)" in prompt
    assert "[2] (source: bill.md#0)" in prompt
    assert prompt.endswith("Question: How?\nAnswer:")
    assert "no context passages" in build_user_prompt("How?", [])


def test_parse_citations() -> None:
    assert parse_citations("A [2]. B [1][2]. C [1, 2] [7] [0]", CONTEXTS) == [
        "bill.md#0",
        "pw.md#0",
    ]
    assert parse_citations("no citations", CONTEXTS) == []


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abc") == 1
    assert estimate_tokens("a" * 40) == 10


def test_fake_generator_extracts_and_cites() -> None:
    gen = FakeGenerator()
    assert isinstance(gen, Generator)
    assert isinstance(gen, TextCompleter)
    answer = gen.generate("How do I reset my password?", CONTEXTS)
    assert "Forgot password" in answer.text
    assert answer.citations == ["pw.md#0"]
    assert answer.contexts == CONTEXTS
    assert answer.usage.prompt_tokens > 0
    assert answer.usage.completion_tokens > 0


def test_fake_generator_says_idk() -> None:
    gen = FakeGenerator()
    assert gen.generate("quantum chromodynamics", CONTEXTS).text == IDK
    empty = gen.generate("anything", [])
    assert empty.text == IDK
    assert empty.citations == []


def test_fake_generator_complete_returns_fixed_response() -> None:
    completion = FakeGenerator(response='{"score": 0.5}').complete("s", "u")
    assert completion.text == '{"score": 0.5}'


def test_llm_generator_base_requires_complete() -> None:
    with pytest.raises(NotImplementedError):
        LLMGenerator().complete("s", "u")


class FakeChatClient:
    def __init__(
        self, content: str | None = "Use the reset link [1].", error: Exception | None = None
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.content = content
        self.error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=12),
        )


def test_openai_generator_with_mock_client() -> None:
    client = FakeChatClient()
    gen = OpenAIGenerator(model="gpt-4o-mini", max_tokens=200, client=client)
    answer = gen.generate("How do I reset my password?", CONTEXTS)
    assert answer.text == "Use the reset link [1]."
    assert answer.citations == ["pw.md#0"]
    assert answer.usage.prompt_tokens == 100
    assert answer.usage.completion_tokens == 12
    call = client.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 200
    assert call["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "[1] (source: pw.md#0)" in call["messages"][1]["content"]
    # no contexts: no API call
    assert gen.generate("q", []).text == IDK
    assert len(client.calls) == 1


def test_openai_generator_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = OpenAIGenerator(client=FakeChatClient(error=RuntimeError("rate limited")))
    with pytest.raises(ProviderError, match="rate limited"):
        gen.complete("s", "u")
    assert OpenAIGenerator(client=FakeChatClient(content=None)).complete("s", "u").text == ""
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(MissingDependencyError, match=r"ragpick\[openai\]"):
        OpenAIGenerator()


def test_openai_generator_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kw: SimpleNamespace(kw=kw))
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError, match="OPENAI_API_KEY not set"):
        OpenAIGenerator()
    gen = OpenAIGenerator(base_url="http://localhost:8000/v1")  # compatible server, no key
    assert gen._client.kw["base_url"] == "http://localhost:8000/v1"


def ollama_client(handler: Any) -> httpx.Client:
    return httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(handler))


def test_ollama_generator_with_mock_transport() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "message": {"content": "It costs $4.99 [2]."},
                "prompt_eval_count": 50,
                "eval_count": 7,
            },
        )

    gen = OllamaGenerator(model="llama3.1", client=ollama_client(handler))
    answer = gen.generate("How much is Lumen Plus?", CONTEXTS)
    assert answer.citations == ["bill.md#0"]
    assert answer.usage.prompt_tokens == 50
    assert answer.usage.completion_tokens == 7
    assert seen["path"] == "/api/chat"
    assert seen["body"]["stream"] is False
    assert seen["body"]["model"] == "llama3.1"


def test_ollama_generator_errors() -> None:
    def not_found(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model not found")

    with pytest.raises(ProviderError, match="ollama pull"):
        OllamaGenerator(client=ollama_client(not_found)).complete("s", "u")

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ProviderError, match="ollama serve"):
        OllamaGenerator(client=ollama_client(down)).complete("s", "u")

    def garbage(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    with pytest.raises(ProviderError, match="invalid JSON"):
        OllamaGenerator(client=ollama_client(garbage)).complete("s", "u")

    gen = OllamaGenerator(base_url="http://example:11434/")
    assert gen.base_url == "http://example:11434"
