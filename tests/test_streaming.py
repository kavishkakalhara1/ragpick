"""Streaming: generators, pipeline, router, cache, HTTP (SSE), CLI."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

import fake_frameworks as ff
from conftest import EXAMPLES
from ragpick import Pipeline, Route, Router, StreamEvent
from ragpick.adapters import LangChainGenerator, LlamaIndexGenerator
from ragpick.cache import Cache, wrap_generator
from ragpick.cli import app as cli_app
from ragpick.components.generators import (
    IDK,
    FakeGenerator,
    LLMGenerator,
    OllamaGenerator,
    OpenAIGenerator,
)
from ragpick.errors import ProviderError
from ragpick.interfaces import StreamingGenerator
from ragpick.models import Answer, Chunk, Completion, RetrievedChunk, Usage
from ragpick.server import App, asgi_app, make_server

DATA = EXAMPLES / "data"
CTX = [
    RetrievedChunk(
        chunk=Chunk(id="a#0", doc_id="a", text="Reset via the link."), score=1, source="vector"
    )
]


def collect(items: Any) -> tuple[str, Usage | None]:
    text, usage = "", None
    for item in items:
        if isinstance(item, Usage):
            usage = item
        else:
            text += item
    return text, usage


# ------------------------------------------------------------------ generators
def test_fake_and_base_streaming() -> None:
    fake = FakeGenerator()
    assert isinstance(fake, StreamingGenerator)
    pieces = list(fake.stream("How to reset?", CTX))
    assert len([p for p in pieces if isinstance(p, str)]) > 1
    text, usage = collect(pieces)
    assert text == fake.generate("How to reset?", CTX).text
    assert usage is not None
    assert collect(fake.stream("q", []))[0] == IDK

    class OneShot(LLMGenerator):
        def complete(self, system: str, user: str) -> Completion:
            return Completion(text="Answer [1].", usage=Usage(prompt_tokens=3))

    text, usage = collect(OneShot().stream("q", CTX))
    assert text == "Answer [1]."
    assert usage == Usage(prompt_tokens=3)


def test_openai_streaming() -> None:
    def chunk(content: str | None = None, usage: Any = None, choices: bool = True) -> Any:
        delta = SimpleNamespace(content=content)
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=delta)] if choices else [], usage=usage
        )

    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return iter(
            [
                chunk("Use "),
                chunk(None),
                chunk("the link [1]."),
                chunk(choices=False, usage=SimpleNamespace(prompt_tokens=9, completion_tokens=4)),
            ]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    text, usage = collect(OpenAIGenerator(client=client, max_tokens=50).stream("q", CTX))
    assert text == "Use the link [1]."
    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens) == (9, 4)
    assert calls[0]["stream"] is True
    assert calls[0]["stream_options"] == {"include_usage": True}
    assert calls[0]["max_tokens"] == 50

    def broken(**kwargs: Any) -> Any:
        raise RuntimeError("quota")

    bad = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=broken)))
    with pytest.raises(ProviderError, match="quota"):
        list(OpenAIGenerator(client=bad).stream("q", CTX))


def ollama(handler: Any) -> OllamaGenerator:
    client = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return OllamaGenerator(client=client)


def test_ollama_streaming() -> None:
    lines = [
        {"message": {"content": "Use "}},
        {"message": {"content": "the link [1]."}},
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 7, "eval_count": 2},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines) + "\n\n")

    text, usage = collect(ollama(handler).stream("q", CTX))
    assert text == "Use the link [1]."
    assert usage is not None
    assert usage.prompt_tokens == 7
    with pytest.raises(ProviderError, match="ollama pull"):
        list(ollama(lambda r: httpx.Response(404)).stream("q", CTX))

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderError, match="ollama serve"):
        list(ollama(down).stream("q", CTX))
    with pytest.raises(ProviderError, match="invalid JSON"):
        list(ollama(lambda r: httpx.Response(200, content="nope\n")).stream("q", CTX))


class StreamingChat(ff.LCChatModel):
    def stream(self, messages: Any) -> Any:
        yield ff.AIMessage("Hi ")
        yield ff.AIMessage(
            [{"type": "text", "text": "there [1]"}], {"input_tokens": 5, "output_tokens": 2}
        )


class FailingStreamChat(ff.LCChatModel):
    def stream(self, messages: Any) -> Any:
        raise RuntimeError("boom")
        yield  # pragma: no cover


class StreamingLLM(ff.LILLM):
    def stream_complete(self, prompt: str) -> Any:
        yield SimpleNamespace(delta="It ")
        yield SimpleNamespace(delta=None)
        yield SimpleNamespace(delta="is [1].")


class FailingStreamLLM(ff.LILLM):
    def stream_complete(self, prompt: str) -> Any:
        raise RuntimeError("down")
        yield  # pragma: no cover


def test_adapter_streaming() -> None:
    text, usage = collect(LangChainGenerator(chat_model=StreamingChat()).stream("q", CTX))
    assert text == "Hi there [1]"
    assert usage is not None
    assert usage.prompt_tokens == 5
    assert (
        collect(LangChainGenerator(chat_model=ff.LCChatModel()).stream("q", CTX))[0]
        == "Answer [1]."
    )
    with pytest.raises(ProviderError, match="boom"):
        list(LangChainGenerator(chat_model=FailingStreamChat()).stream("q", CTX))
    text, usage = collect(LlamaIndexGenerator(llm=StreamingLLM()).stream("q", CTX))
    assert text == "It is [1]."
    assert usage is not None
    assert collect(LlamaIndexGenerator(llm=ff.LILLM()).stream("q", CTX))[0] == "It is [2]."
    with pytest.raises(ProviderError, match="down"):
        list(LlamaIndexGenerator(llm=FailingStreamLLM()).stream("q", CTX))


# ------------------------------------------------------------------ pipeline / router / cache
@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> Pipeline:
    p = Pipeline.from_dict(
        {"blueprint": "hybrid", "index_dir": str(tmp_path_factory.mktemp("s") / "i")}
    )
    p.ingest(DATA)
    return p


def test_pipeline_stream_matches_query(pipeline: Pipeline) -> None:
    events = list(pipeline.stream("What does error E-1042 mean?"))
    assert events[0].type == "contexts"
    assert events[-1].type == "answer"
    assert all(e.type == "delta" for e in events[1:-1])
    final = events[-1].answer
    assert final is not None
    assert final.text == "".join(e.text for e in events[1:-1]).strip()
    assert final.text == pipeline.query("What does error E-1042 mean?").text
    assert final.citations == pipeline.query("What does error E-1042 mean?").citations
    assert final.usage.latency_ms > 0


def test_pipeline_stream_with_non_streaming_generator(tmp_path: Path) -> None:
    class Plain:
        def generate(self, query: str, contexts: list[RetrievedChunk]) -> Answer:
            return Answer(text="Plain [1].", usage=Usage(prompt_tokens=2))

    p = Pipeline.from_dict(
        {"blueprint": "naive", "index_dir": str(tmp_path / "i")}, components={"generator": Plain()}
    )
    p.ingest(DATA)
    events = list(p.stream("anything"))
    assert [e.type for e in events] == ["contexts", "delta", "answer"]
    assert events[-1].answer.citations  # type: ignore[union-attr]
    assert events[-1].answer.usage.prompt_tokens == 2  # type: ignore[union-attr]


def test_router_stream(pipeline: Pipeline) -> None:
    router = Router([Route("docs", pipeline, groups=None), Route("hr", pipeline, groups=["hr"])])
    events = list(router.stream("What does error E-1042 mean?"))
    assert events[-1].answer.route == "docs"  # type: ignore[union-attr]
    closed = Router([Route("hr", pipeline, groups=["hr"])])
    events = list(closed.stream("x"))
    assert [e.type for e in events] == ["contexts", "delta", "answer"]
    assert events[-1].answer.text == IDK  # type: ignore[union-attr]


def test_cached_stream(tmp_path: Path) -> None:
    inner = FakeGenerator()
    gen = wrap_generator(inner, Cache(tmp_path / "c.sqlite"), {"t": "fake"})
    first, _usage1 = collect(gen.stream("How to reset?", CTX))
    second, usage2 = collect(gen.stream("How to reset?", CTX))
    assert first.strip() == second
    assert usage2 is not None
    assert usage2.cached
    assert gen.generate("How to reset?", CTX).usage.cached  # shared cache entry

    class Plain:
        def generate(self, query: str, contexts: list[RetrievedChunk]) -> Answer:
            return Answer(text="P [1].")

    plain = wrap_generator(Plain(), Cache(tmp_path / "d.sqlite"), {})
    assert collect(plain.stream("q", CTX))[0] == "P [1]."
    assert collect(plain.stream("q", CTX))[0] == "P [1]."


# ------------------------------------------------------------------ HTTP
def parse_sse(body: str) -> list[tuple[str, Any]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_app_stream_response(pipeline: Pipeline) -> None:
    app = App(pipeline)
    response = app.handle(
        "POST", "/query", {}, json.dumps({"question": "E-1042?", "stream": True}).encode()
    )
    assert response.status == 200
    assert response.content_type == "text/event-stream"
    assert response.stream is not None
    events = parse_sse(b"".join(response.stream).decode())
    assert events[0][0] == "contexts"
    assert events[-1][0] == "answer"
    assert events[-1][1]["citations"]
    assert {e for e, _ in events[1:-1]} == {"delta"}
    bad = app.handle(
        "POST", "/query", {}, json.dumps({"question": "q", "stream": True, "trace": True}).encode()
    )
    assert bad.status == 400


def test_app_stream_errors(tmp_path: Path, pipeline: Pipeline) -> None:
    empty = App(Pipeline.from_dict({"blueprint": "naive", "index_dir": str(tmp_path / "none")}))
    assert (
        empty.handle("POST", "/query", {}, b'{"question": "q", "stream": true}').status == 503
    )  # before streaming

    class MidStreamFailure:
        name, blueprint = "x", "naive"
        serve_config = App(pipeline).config

        def __init__(self, exc: Exception) -> None:
            self.exc = exc

        def stream(self, *args: Any, **kwargs: Any) -> Any:
            yield StreamEvent(type="contexts")
            raise self.exc

    for exc, message in [
        (ProviderError("secret detail"), "model provider failed"),
        (RuntimeError("bug"), "Internal"),
    ]:
        response = App(MidStreamFailure(exc)).handle(
            "POST", "/query", {}, b'{"question": "q", "stream": true}'
        )  # type: ignore[arg-type]
        events = parse_sse(b"".join(response.stream or []).decode())
        assert events[-1][0] == "error"
        assert message in events[-1][1]["error"]
        assert "secret" not in events[-1][1]["error"]


def test_asgi_and_stdlib_streaming(pipeline: Pipeline) -> None:
    body = {"question": "What does error E-1042 mean?", "stream": True}

    async def via_asgi() -> httpx.Response:
        transport = httpx.ASGITransport(app=asgi_app(App(pipeline)))
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await client.post("/query", json=body)

    r = asyncio.run(via_asgi())
    assert r.headers["content-type"] == "text/event-stream"
    assert parse_sse(r.text)[-1][0] == "answer"

    server = make_server(App(pipeline), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with (
            httpx.Client(trust_env=False, timeout=10) as client,
            client.stream(
                "POST", f"http://127.0.0.1:{server.server_address[1]}/query", json=body
            ) as response,
        ):
            assert response.headers["content-type"] == "text/event-stream"
            assert "content-length" not in response.headers
            text = "".join(response.iter_text())
        assert parse_sse(text)[-1][1]["citations"]
    finally:
        server.shutdown()
        server.server_close()


def test_cli_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("rag.yaml").write_text("blueprint: hybrid\n", encoding="utf-8")
    runner = CliRunner()
    assert runner.invoke(cli_app, ["ingest", str(DATA)]).exit_code == 0
    result = runner.invoke(cli_app, ["query", "What does error E-1042 mean?", "--stream"])
    assert result.exit_code == 0, result.output
    assert "E-1042" in result.output
    assert "Citations: error-codes.md#0" in result.output
    both = runner.invoke(cli_app, ["query", "x", "--stream", "--trace"])
    assert both.exit_code == 1
    assert "can't be combined" in both.output
