"""HTTP API: request handling, auth/groups, ASGI and stdlib servers, CLI."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from conftest import EXAMPLES
from ragpick import Pipeline, Route, Router
from ragpick.cli import app as cli_app
from ragpick.config import ApiKeySpec, ServeConfig
from ragpick.errors import ConfigError, ProviderError, RagpickError
from ragpick.server import App, asgi_app, is_loopback, make_server

DATA = EXAMPLES / "data"
SUPPORT_KEY = "support-key-0123456789"
ADMIN_KEY = "admin-key-0123456789ab"


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> Pipeline:
    tmp = tmp_path_factory.mktemp("srv")
    p = Pipeline.from_dict({"blueprint": "hybrid", "index_dir": str(tmp / "idx")})
    p.ingest(DATA)
    return p


def call(
    app: App, method: str, path: str, body: Any = None, **headers: str
) -> tuple[int, Any, dict[str, str]]:
    raw = (
        json.dumps(body).encode()
        if body is not None and not isinstance(body, bytes)
        else (body or b"")
    )
    response = app.handle(
        method, path, {k.lower().replace("_", "-"): v for k, v in headers.items()}, raw
    )
    return response.status, response.body, response.headers


def secured(pipeline: Pipeline, **kw: Any) -> App:
    keys = [
        ApiKeySpec(key=SUPPORT_KEY, name="portal", groups=["support"]),
        ApiKeySpec(key=ADMIN_KEY, name="admin"),
    ]
    return App(pipeline, ServeConfig(api_keys=keys, **kw))


QUESTION = "When should support escalate an E-3301 ticket to Tier 2?"


def test_health_info_and_routing(pipeline: Pipeline) -> None:
    app = App(pipeline)
    assert app.config == pipeline.config.serve
    status, body, _ = call(app, "GET", "/health")
    assert (status, body["status"]) == (200, "ok")
    status, body, _ = call(app, "GET", "/info/")
    assert body["blueprint"] == "hybrid"
    assert "routes" not in body
    assert call(app, "GET", "/nope")[0] == 404
    assert call(app, "POST", "/health")[0] == 405
    assert call(app, "POST", "/info")[0] == 405
    assert call(app, "GET", "/query")[0] == 405
    assert call(app, "OPTIONS", "/query")[0] == 204


def test_query_and_retrieve(pipeline: Pipeline) -> None:
    app = App(pipeline)
    status, body, _ = call(
        app, "POST", "/query?x=1", {"question": "What does error E-1042 mean?", "top_k": 2}
    )
    assert status == 200
    assert body["citations"][0].startswith("error-codes.md")
    assert len(body["contexts"]) <= 2
    assert set(body["contexts"][0]) == {"id", "doc_id", "score", "source", "text", "metadata"}
    assert body["usage"]["total_tokens"] > 0
    assert body["route"] is None
    assert "trace" not in body
    status, body, _ = call(app, "POST", "/retrieve", {"question": "E-1042"})
    assert status == 200
    assert body["contexts"]


def test_without_keys_groups_come_from_request(pipeline: Pipeline) -> None:
    app = App(pipeline)
    _, public, _ = call(app, "POST", "/retrieve", {"question": QUESTION, "top_k": 9})
    assert all(c["doc_id"] != "internal-escalation.md" for c in public["contexts"])
    _, support, _ = call(
        app, "POST", "/retrieve", {"question": QUESTION, "top_k": 9, "groups": ["support"]}
    )
    assert any(c["doc_id"] == "internal-escalation.md" for c in support["contexts"])


def test_api_keys_and_group_enforcement(pipeline: Pipeline) -> None:
    app = secured(pipeline)
    assert call(app, "GET", "/health")[0] == 200  # health stays public
    assert call(app, "GET", "/info")[1] == {
        "error": "Missing API key: send 'Authorization: Bearer <key>'"
    }
    assert call(app, "GET", "/info", authorization="Basic abc")[0] == 401
    assert (
        call(app, "GET", "/info", authorization="Bearer wrong-key-000000000")[1]["error"]
        == "Invalid API key"
    )
    auth = {"authorization": f"Bearer {SUPPORT_KEY}"}
    assert call(app, "GET", "/info", **auth)[0] == 200
    # omitting groups → the key's groups
    _, body, _ = call(app, "POST", "/retrieve", {"question": QUESTION, "top_k": 9}, **auth)
    assert any(c["doc_id"] == "internal-escalation.md" for c in body["contexts"])
    # claiming a group the key doesn't have → 403
    status, body, _ = call(app, "POST", "/query", {"question": QUESTION, "groups": ["hr"]}, **auth)
    assert status == 403
    assert "hr" in body["error"]
    # narrowing to a subset is allowed
    status, _, _ = call(app, "POST", "/query", {"question": QUESTION, "groups": []}, **auth)
    assert status == 200
    # an unrestricted key may act as any group
    admin = {"authorization": f"Bearer {ADMIN_KEY}"}
    assert call(app, "POST", "/query", {"question": QUESTION, "groups": ["hr"]}, **admin)[0] == 200


def test_request_validation(pipeline: Pipeline) -> None:
    app = App(pipeline, ServeConfig(max_body_bytes=1024))
    assert call(app, "POST", "/query", b"{not json")[0] == 400
    status, body, _ = call(app, "POST", "/query", {"question": ""})
    assert status == 422
    assert "question" in body["error"]
    assert call(app, "POST", "/query", {"question": "q", "unknown": 1})[0] == 422
    assert call(app, "POST", "/query", {"question": "q", "top_k": 500})[0] == 422
    assert call(app, "POST", "/query", {"question": "x" * 2000})[0] == 413


def test_traces_and_cors(pipeline: Pipeline) -> None:
    app = App(pipeline)
    assert call(app, "POST", "/query", {"question": "q", "trace": True})[0] == 403
    traced = App(pipeline, ServeConfig(allow_traces=True, cors_origins=["https://app.example"]))
    status, body, headers = call(
        traced,
        "POST",
        "/query",
        {"question": "E-1042?", "trace": True},
        origin="https://app.example",
    )
    assert status == 200
    assert body["trace"]["name"] == "query"
    assert headers["Access-Control-Allow-Origin"] == "https://app.example"
    assert call(traced, "GET", "/health", origin="https://evil.example")[2] == {}
    star = App(pipeline, ServeConfig(cors_origins=["*"]))
    assert (
        call(star, "OPTIONS", "/query", origin="https://x")[2]["Access-Control-Allow-Origin"] == "*"
    )


class ExplodingEngine:
    name = "boom"
    blueprint = "naive"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.serve_config = ServeConfig()

    def ready(self) -> bool:
        return False

    def query(self, *args: Any, **kwargs: Any) -> Any:
        raise self.exc

    retrieve = query


@pytest.mark.parametrize(
    ("exc", "status", "message"),
    [
        (ProviderError("rate limited"), 502, "model provider failed"),
        (ConfigError("bad config"), 500, "configuration error"),
        (RagpickError("bad request"), 400, "bad request"),
        (RuntimeError("bug"), 500, "Internal server error"),
    ],
)
def test_error_mapping(exc: Exception, status: int, message: str) -> None:
    app = App(ExplodingEngine(exc))  # type: ignore[arg-type]
    got, body, _ = call(app, "POST", "/query", {"question": "q"})
    assert got == status
    assert message in body["error"]
    assert "rate limited" not in body["error"]  # provider details stay in the logs


def test_not_ready(tmp_path: Path) -> None:
    empty = Pipeline.from_dict({"blueprint": "naive", "index_dir": str(tmp_path / "none")})
    app = App(empty)
    assert call(app, "GET", "/health")[:2] == (
        503,
        {"status": "not_ready", "version": call(app, "GET", "/health")[1]["version"]},
    )
    status, body, _ = call(app, "POST", "/query", {"question": "q"})
    assert status == 503
    assert "ragpick ingest" in body["error"]


def test_router_info(pipeline: Pipeline) -> None:
    router = Router([Route("docs", pipeline, "manuals"), Route("more", pipeline, "other")])
    app = App(router)
    _, body, _ = call(app, "GET", "/info")
    assert [r["name"] for r in body["routes"]] == ["docs", "more"]
    _, answer, _ = call(app, "POST", "/query", {"question": "What does error E-1042 mean?"})
    assert answer["route"] in ("docs", "more")


def test_asgi_app(pipeline: Pipeline) -> None:
    application = asgi_app(secured(pipeline, max_body_bytes=2048))

    async def run() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return [
                await client.get("/health"),
                await client.post("/query", json={"question": "E-1042?"}),
                await client.post(
                    "/query",
                    json={"question": "E-1042?"},
                    headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                ),
                await client.post(
                    "/query", content=b"x" * 5000, headers={"Authorization": f"Bearer {ADMIN_KEY}"}
                ),
                await client.options("/query"),
            ]

    health, unauth, ok, too_big, options = asyncio.run(run())
    assert health.status_code == 200
    assert health.headers["content-type"] == "application/json"
    assert unauth.status_code == 401
    assert ok.status_code == 200
    assert ok.json()["citations"]
    assert too_big.status_code == 413
    assert options.status_code == 204
    assert options.content == b""


def test_asgi_lifespan_and_other_scopes(pipeline: Pipeline) -> None:
    application = asgi_app(App(pipeline))
    sent: list[dict[str, Any]] = []
    messages = iter([{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}])

    async def receive() -> dict[str, Any]:
        return next(messages)

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(application({"type": "lifespan"}, receive, send))
    assert [m["type"] for m in sent] == ["lifespan.startup.complete", "lifespan.shutdown.complete"]
    asyncio.run(application({"type": "websocket"}, receive, send))  # ignored
    assert len(sent) == 2


def test_stdlib_server(pipeline: Pipeline) -> None:
    server = make_server(secured(pipeline, max_body_bytes=2048), "127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = httpx.Client(trust_env=False, timeout=10)  # ignore proxy env vars
    try:
        base = f"http://127.0.0.1:{port}"
        assert client.get(f"{base}/health").json()["status"] == "ok"
        auth = {"Authorization": f"Bearer {SUPPORT_KEY}"}
        r = client.post(f"{base}/query", json={"question": QUESTION}, headers=auth)
        assert r.status_code == 200
        assert any(c.startswith("internal-escalation.md") for c in r.json()["citations"])
        assert (
            client.post(
                f"{base}/query", json={"question": QUESTION, "groups": ["hr"]}, headers=auth
            ).status_code
            == 403
        )
        assert client.post(f"{base}/query", content=b"x" * 5000, headers=auth).status_code == 413
        assert client.delete(f"{base}/query", headers=auth).status_code == 405
        assert client.options(f"{base}/query").status_code == 204
    finally:
        client.close()
        server.shutdown()
        server.server_close()


def test_is_loopback() -> None:
    assert is_loopback("127.0.0.1")
    assert is_loopback("localhost")
    assert not is_loopback("0.0.0.0")


def test_serve_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("rag.yaml").write_text("blueprint: naive\n", encoding="utf-8")
    runner = CliRunner()
    started: dict[str, Any] = {}

    class FakeServer:
        def serve_forever(self) -> None:
            started["served"] = True

        def server_close(self) -> None:
            started["closed"] = True

    def fake_make_server(app: App, host: str, port: int) -> FakeServer:
        started.update(host=host, port=port, app=app)
        return FakeServer()

    monkeypatch.setattr("ragpick.server.make_server", fake_make_server)
    result = runner.invoke(
        cli_app, ["serve", "--server", "stdlib", "--host", "0.0.0.0", "--port", "9001"]
    )
    assert result.exit_code == 0, result.output
    assert started["host"] == "0.0.0.0"
    assert started["port"] == 9001
    assert started["served"]
    assert started["closed"]
    assert "Nothing ingested yet" in result.output
    assert "no serve.api_keys" in result.output
    assert "standard-library server" in result.output

    calls: dict[str, Any] = {}
    fake_uvicorn = SimpleNamespace(run=lambda application, **kw: calls.update(kw, app=application))
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    result = runner.invoke(cli_app, ["serve", "--server", "uvicorn"])
    assert result.exit_code == 0, result.output
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 8000
    assert callable(calls["app"])
    assert "no serve.api_keys" not in result.output  # loopback: no warning

    bad = runner.invoke(cli_app, ["serve", "--server", "gunicorn"])
    assert bad.exit_code == 1
    assert "--server must be" in bad.output
