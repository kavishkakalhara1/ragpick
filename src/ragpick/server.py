"""HTTP API for ragpick (``ragpick serve``). Standard library only.

Endpoints (JSON):

* ``GET /health``: liveness and readiness;
* ``GET /info``: application name, blueprint(s), routes;
* ``POST /query``: ``{"question", "groups"?, "filters"?, "top_k"?, "trace"?}`` → answer;
* ``POST /retrieve``: same body → contexts only.

**Security.** With ``serve.api_keys`` configured, every request except ``/health``
needs ``Authorization: Bearer <key>``. Each key lists the groups its holder may act
as: a request's ``groups`` must be a subset of them, and requests that omit
``groups`` act as all of the key's groups. Access control is therefore enforced at
the API boundary; clients can't simply claim ``"groups": ["hr"]``.

The same :class:`App` is exposed as an ASGI application (:func:`asgi_app`, served by
uvicorn when installed) and through the standard library's threaded HTTP server.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ragpick import __version__
from ragpick.config import ApiKeySpec, ServeConfig
from ragpick.errors import (
    ConfigError,
    IndexNotFoundError,
    MissingCredentialError,
    MissingDependencyError,
    ProviderError,
    RagpickError,
)
from ragpick.models import Answer, RetrievedChunk
from ragpick.pipeline import Pipeline
from ragpick.router import Engine

logger = logging.getLogger(__name__)

_HIDDEN_METADATA = {"parent_text"}


class QueryRequest(BaseModel):
    """Body of ``POST /query`` and ``POST /retrieve``."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    groups: list[str] | None = None
    filters: dict[str, Any] | None = None
    top_k: int | None = Field(None, ge=1, le=50)
    trace: bool = False


@dataclass
class Response:
    """A JSON HTTP response."""

    status: int
    body: Any
    headers: dict[str, str] = field(default_factory=dict)

    def encoded(self) -> bytes:
        """The body as UTF-8 JSON."""
        return json.dumps(self.body, default=str).encode("utf-8")


class HTTPError(Exception):
    """Raised inside handlers to return an error response."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def context_json(ctx: RetrievedChunk) -> dict[str, Any]:
    """Public JSON form of a retrieved context."""
    meta = {k: v for k, v in ctx.chunk.metadata.items() if k not in _HIDDEN_METADATA}
    return {
        "id": ctx.chunk.id,
        "doc_id": ctx.chunk.doc_id,
        "score": ctx.score,
        "source": ctx.source,
        "text": ctx.chunk.text,
        "metadata": meta,
    }


def answer_json(answer: Answer, include_trace: bool) -> dict[str, Any]:
    """Public JSON form of an answer."""
    body: dict[str, Any] = {
        "text": answer.text,
        "citations": answer.citations,
        "route": answer.route,
        "contexts": [context_json(c) for c in answer.contexts],
        "usage": answer.usage.model_dump() | {"total_tokens": answer.usage.total_tokens},
    }
    if include_trace and answer.trace is not None:
        body["trace"] = answer.trace.model_dump()
    return body


class App:
    """Framework-independent request handling for an engine (pipeline or router)."""

    def __init__(self, engine: Engine, config: ServeConfig | None = None) -> None:
        self.engine = engine
        if config is None:
            config = engine.config.serve if isinstance(engine, Pipeline) else engine.serve_config
        self.config = config

    # -------------------------------------------------------------- helpers
    def _authenticate(self, headers: dict[str, str]) -> ApiKeySpec | None:
        if not self.config.api_keys:
            return None
        auth = headers.get("authorization", "")
        scheme, _, token = auth.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPError(401, "Missing API key: send 'Authorization: Bearer <key>'")
        for spec in self.config.api_keys:
            if hmac.compare_digest(token.encode("utf-8"), spec.key.encode("utf-8")):
                return spec
        raise HTTPError(401, "Invalid API key")

    @staticmethod
    def _groups(key: ApiKeySpec | None, requested: list[str] | None) -> list[str] | None:
        if key is None or key.groups is None:
            return requested
        if requested is None:
            return list(key.groups)
        forbidden = sorted(set(requested) - set(key.groups))
        if forbidden:
            raise HTTPError(403, f"This API key may not act as groups: {', '.join(forbidden)}")
        return requested

    def _cors(self, headers: dict[str, str]) -> dict[str, str]:
        origin = headers.get("origin")
        allowed = self.config.cors_origins
        if not origin or not allowed or ("*" not in allowed and origin not in allowed):
            return {}
        return {
            "Access-Control-Allow-Origin": "*" if "*" in allowed else origin,
            "Access-Control-Allow-Headers": "Authorization, Content-Type",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Vary": "Origin",
        }

    def _parse(self, body: bytes) -> QueryRequest:
        try:
            data = json.loads(body or b"{}")
        except ValueError:
            raise HTTPError(400, "Request body must be valid JSON") from None
        try:
            return QueryRequest.model_validate(data)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'body'}: {e['msg']}" for e in exc.errors()
            )
            raise HTTPError(422, problems) from None

    # -------------------------------------------------------------- endpoints
    def _health(self) -> Response:
        ready = self.engine.ready()
        return Response(
            200 if ready else 503,
            {"status": "ok" if ready else "not_ready", "version": __version__},
        )

    def _info(self) -> Response:
        body: dict[str, Any] = {
            "name": self.engine.name,
            "blueprint": self.engine.blueprint,
            "version": __version__,
        }
        routes = getattr(self.engine, "routes", None)
        if routes is not None:
            body["routes"] = [
                {"name": r.name, "description": r.description, "blueprint": r.pipeline.blueprint}
                for r in routes
            ]
        return Response(200, body)

    def _query(self, req: QueryRequest, groups: list[str] | None) -> Response:
        trace = req.trace and self.config.allow_traces
        if req.trace and not self.config.allow_traces:
            raise HTTPError(403, "Traces are disabled on this server (serve.allow_traces)")
        answer = self.engine.query(
            req.question, groups=groups, filters=req.filters, top_k=req.top_k, trace=trace or None
        )
        return Response(200, answer_json(answer, trace))

    def _retrieve(self, req: QueryRequest, groups: list[str] | None) -> Response:
        contexts = self.engine.retrieve(
            req.question, groups=groups, filters=req.filters, top_k=req.top_k
        )
        return Response(200, {"contexts": [context_json(c) for c in contexts]})

    def handle(self, method: str, path: str, headers: dict[str, str], body: bytes) -> Response:
        """Dispatch one request. ``headers`` keys must be lowercase."""
        cors = self._cors(headers)
        try:
            response = self._dispatch(
                method.upper(), path.split("?", 1)[0].rstrip("/") or "/", headers, body
            )
        except HTTPError as exc:
            response = Response(exc.status, {"error": exc.message})
        except IndexNotFoundError as exc:
            response = Response(503, {"error": str(exc)})
        except ProviderError as exc:
            logger.warning("Provider error: %s", exc)
            response = Response(502, {"error": "The model provider failed; see server logs."})
        except (ConfigError, MissingCredentialError, MissingDependencyError) as exc:
            logger.error("Server configuration error: %s", exc)
            response = Response(500, {"error": "Server configuration error; see server logs."})
        except RagpickError as exc:
            response = Response(400, {"error": str(exc)})
        except Exception:
            logger.exception("Unhandled error for %s %s", method, path)
            response = Response(500, {"error": "Internal server error"})
        response.headers.update(cors)
        return response

    def _dispatch(self, method: str, path: str, headers: dict[str, str], body: bytes) -> Response:
        if method == "OPTIONS":
            return Response(204, None)
        if path == "/health":
            if method != "GET":
                raise HTTPError(405, "Use GET")
            return self._health()
        if path not in ("/info", "/query", "/retrieve"):
            raise HTTPError(404, f"Not found: {path}")
        key = self._authenticate(headers)
        if path == "/info":
            if method != "GET":
                raise HTTPError(405, "Use GET")
            return self._info()
        if method != "POST":
            raise HTTPError(405, "Use POST")
        if len(body) > self.config.max_body_bytes:
            raise HTTPError(413, f"Body exceeds {self.config.max_body_bytes} bytes")
        req = self._parse(body)
        groups = self._groups(key, req.groups)
        if key is not None:
            logger.info("%s %s by %s (groups=%s)", method, path, key.name, groups)
        return self._query(req, groups) if path == "/query" else self._retrieve(req, groups)


# ----------------------------------------------------------------------- ASGI
Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]


def asgi_app(app: App) -> Callable[[Scope, Receive, Send], Awaitable[None]]:
    """Wrap ``app`` as an ASGI 3 application (engine calls run in a worker thread)."""

    async def application(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])
        }
        chunks: list[bytes] = []
        size = 0
        more = True
        while more:  # always drain the body, but keep at most max_body_bytes of it
            message = await receive()
            chunk = message.get("body", b"")
            size += len(chunk)
            if size <= app.config.max_body_bytes:
                chunks.append(chunk)
            more = bool(message.get("more_body"))
        if size > app.config.max_body_bytes:
            response = Response(413, {"error": f"Body exceeds {app.config.max_body_bytes} bytes"})
        else:
            response = await asyncio.to_thread(
                app.handle, scope["method"], scope.get("path", "/"), headers, b"".join(chunks)
            )
        payload = b"" if response.status == 204 else response.encoded()
        out_headers = [(b"content-type", b"application/json")] + [
            (k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in response.headers.items()
        ]
        await send(
            {"type": "http.response.start", "status": response.status, "headers": out_headers}
        )
        await send({"type": "http.response.body", "body": payload})

    return application


# ----------------------------------------------------------------------- stdlib server
def make_server(app: App, host: str, port: int) -> ThreadingHTTPServer:
    """A threaded standard-library HTTP server for ``app`` (call ``serve_forever()``)."""

    class Handler(BaseHTTPRequestHandler):
        server_version = f"ragpick/{__version__}"

        def _respond(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length > app.config.max_body_bytes:
                response = Response(
                    413, {"error": f"Body exceeds {app.config.max_body_bytes} bytes"}
                )
            else:
                body = self.rfile.read(length) if length else b""
                headers = {k.lower(): v for k, v in self.headers.items()}
                response = app.handle(self.command, self.path, headers, body)
            payload = b"" if response.status == 204 else response.encoded()
            self.send_response(response.status, HTTPStatus(response.status).phrase)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            for k, v in response.headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = do_OPTIONS = do_PUT = do_DELETE = _respond  # noqa: N815

        def log_message(self, format: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), format % args)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def is_loopback(host: str) -> bool:
    """Whether ``host`` only accepts local connections."""
    return host in ("127.0.0.1", "localhost", "::1")
