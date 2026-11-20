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
from collections.abc import Awaitable, Callable, Iterator, MutableMapping
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ragpick import __version__
from ragpick.config import ApiKeySpec, ServeConfig
from ragpick.conversation import ConversationError, owner_key, validate_session_id
from ragpick.errors import (
    ConfigError,
    IndexNotFoundError,
    MissingCredentialError,
    MissingDependencyError,
    ProviderError,
    RagpickError,
)
from ragpick.feedback import AnswerLog, FeedbackError
from ragpick.models import Answer, RetrievedChunk, StreamEvent
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
    stream: bool = False
    tenant: str | None = Field(None, max_length=64)


class ChatRequest(BaseModel):
    """Body of ``POST /chat``."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(None, max_length=64)
    groups: list[str] | None = None
    filters: dict[str, Any] | None = None
    top_k: int | None = Field(None, ge=1, le=50)
    tenant: str | None = Field(None, max_length=64)
    stream: bool = False


class FeedbackRequest(BaseModel):
    """Body of ``POST /feedback``."""

    model_config = ConfigDict(extra="forbid")

    answer_id: str = Field(min_length=1, max_length=64)
    rating: str | int | float | bool
    comment: str | None = Field(None, max_length=2000)
    tenant: str | None = Field(None, max_length=64)


@dataclass
class Response:
    """A JSON HTTP response, or a server-sent-events stream when ``stream`` is set."""

    status: int
    body: Any
    headers: dict[str, str] = field(default_factory=dict)
    stream: Iterator[bytes] | None = None
    content_type: str = "application/json"

    def encoded(self) -> bytes:
        """The body as UTF-8 JSON."""
        return json.dumps(self.body, default=str).encode("utf-8")


class HTTPError(Exception):
    """Raised inside handlers to return an error response."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def sse(event: str, data: Any) -> bytes:
    """Encode one server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n".encode()


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
        "id": answer.id,
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
        feedback = (
            engine.config.feedback
            if isinstance(engine, Pipeline)
            else getattr(engine, "feedback_config", None)  # duck-typed engines: disabled
        )
        self.answers: AnswerLog | None = None
        if feedback is not None and feedback.enabled:
            self.answers = AnswerLog(feedback.answer_log_size)

    def _remember(
        self, answer: Answer, owner: str, tenant: str | None, groups: list[str] | None
    ) -> None:
        if self.answers is not None:
            self.answers.remember(answer, owner_key(owner, tenant), groups)

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

    @staticmethod
    def _tenant(key: ApiKeySpec | None, requested: str | None) -> str | None:
        if key is None or key.tenant is None:
            return requested
        if requested is not None and requested != key.tenant:
            raise HTTPError(403, "This API key is bound to a different tenant")
        return key.tenant

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

    def _parse(self, body: bytes, model: type[BaseModel] = QueryRequest) -> Any:
        try:
            data = json.loads(body or b"{}")
        except ValueError:
            raise HTTPError(400, "Request body must be valid JSON") from None
        try:
            return model.model_validate(data)
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

    def _stream(
        self, req: QueryRequest, groups: list[str] | None, tenant: str | None, owner: str
    ) -> Response:
        events = self.engine.stream(
            req.question, groups=groups, filters=req.filters, top_k=req.top_k, tenant=tenant
        )
        first = next(events)  # runs retrieval now, so its errors get a proper status code

        def encode(event: StreamEvent) -> bytes:
            if event.type == "contexts":
                return sse("contexts", {"contexts": [context_json(c) for c in event.contexts]})
            if event.type == "delta":
                return sse("delta", {"text": event.text})
            assert event.answer is not None
            self._remember(event.answer, owner, tenant, groups)
            return sse("answer", answer_json(event.answer, include_trace=False))

        def body() -> Iterator[bytes]:
            yield encode(first)
            try:
                for event in events:
                    yield encode(event)
            except ProviderError as exc:
                logger.warning("Provider error while streaming: %s", exc)
                yield sse("error", {"error": "The model provider failed; see server logs."})
            except Exception:
                logger.exception("Error while streaming")
                yield sse("error", {"error": "Internal server error"})

        return Response(
            200,
            None,
            {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            stream=body(),
            content_type="text/event-stream",
        )

    def _query(
        self, req: QueryRequest, groups: list[str] | None, tenant: str | None, owner: str
    ) -> Response:
        if req.stream:
            if req.trace:
                raise HTTPError(400, "Streaming responses cannot include traces")
            return self._stream(req, groups, tenant, owner)
        trace = req.trace and self.config.allow_traces
        if req.trace and not self.config.allow_traces:
            raise HTTPError(403, "Traces are disabled on this server (serve.allow_traces)")
        answer = self.engine.query(
            req.question,
            groups=groups,
            filters=req.filters,
            top_k=req.top_k,
            trace=trace or None,
            tenant=tenant,
        )
        self._remember(answer, owner, tenant, groups)
        return Response(200, answer_json(answer, trace))

    def _retrieve(
        self, req: QueryRequest, groups: list[str] | None, tenant: str | None
    ) -> Response:
        contexts = self.engine.retrieve(
            req.question, groups=groups, filters=req.filters, top_k=req.top_k, tenant=tenant
        )
        return Response(200, {"contexts": [context_json(c) for c in contexts]})

    def _chat(
        self, req: ChatRequest, groups: list[str] | None, tenant: str | None, owner: str
    ) -> Response:
        kwargs: dict[str, Any] = {
            "session_id": req.session_id,
            "owner": owner,
            "groups": groups,
            "filters": req.filters,
            "top_k": req.top_k,
            "tenant": tenant,
        }
        if not req.stream:
            answer = self.engine.chat(req.message, **kwargs)
            self._remember(answer, owner, tenant, groups)
            body = answer_json(answer, include_trace=False)
            body.update(
                session_id=answer.session_id, standalone_question=answer.standalone_question
            )
            return Response(200, body)
        events = self.engine.chat_stream(req.message, **kwargs)
        first = next(events)

        def encode(event: StreamEvent) -> bytes:
            if event.type == "contexts":
                return sse("contexts", {"contexts": [context_json(c) for c in event.contexts]})
            if event.type == "delta":
                return sse("delta", {"text": event.text})
            assert event.answer is not None
            self._remember(event.answer, owner, tenant, groups)
            data = answer_json(event.answer, include_trace=False)
            data.update(
                session_id=event.answer.session_id,
                standalone_question=event.answer.standalone_question,
            )
            return sse("answer", data)

        def stream_body() -> Iterator[bytes]:
            yield encode(first)
            try:
                for event in events:
                    yield encode(event)
            except ProviderError as exc:
                logger.warning("Provider error while streaming: %s", exc)
                yield sse("error", {"error": "The model provider failed; see server logs."})
            except Exception:
                logger.exception("Error while streaming")
                yield sse("error", {"error": "Internal server error"})

        return Response(
            200,
            None,
            {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            stream=stream_body(),
            content_type="text/event-stream",
        )

    def _feedback(self, req: FeedbackRequest, key: ApiKeySpec | None, owner: str) -> Response:
        if self.answers is None:
            raise HTTPError(404, "Feedback is disabled on this server (feedback.enabled)")
        tenant = self._tenant(key, req.tenant)
        answer, groups = self.answers.get(req.answer_id, owner_key(owner, tenant))
        feedback_id = self.engine.record_feedback(
            answer, req.rating, comment=req.comment, owner=owner, tenant=tenant, groups=groups
        )
        return Response(200, {"id": feedback_id})

    def handle(self, method: str, path: str, headers: dict[str, str], body: bytes) -> Response:
        """Dispatch one request. ``headers`` keys must be lowercase."""
        cors = self._cors(headers)
        try:
            response = self._dispatch(
                method.upper(), path.split("?", 1)[0].rstrip("/") or "/", headers, body
            )
        except HTTPError as exc:
            response = Response(exc.status, {"error": exc.message})
        except ConversationError as exc:
            status = 404 if "Unknown session" in str(exc) else 400
            response = Response(status, {"error": str(exc)})
        except FeedbackError as exc:
            status = 404 if "Unknown answer" in str(exc) else 400
            response = Response(status, {"error": str(exc)})
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
        is_session = path.startswith("/sessions/")
        if path not in ("/info", "/query", "/retrieve", "/chat", "/feedback") and not is_session:
            raise HTTPError(404, f"Not found: {path}")
        key = self._authenticate(headers)
        owner = key.name if key is not None else "anonymous"
        if is_session:
            if method != "DELETE":
                raise HTTPError(405, "Use DELETE")
            session_id = path.removeprefix("/sessions/")
            tenant = self._tenant(key, None)
            deleted = self.engine.conversations.store.delete(
                validate_session_id(session_id), owner_key(owner, tenant)
            )
            if not deleted:
                raise ConversationError("Unknown session")
            return Response(200, {"deleted": session_id})
        if path == "/info":
            if method != "GET":
                raise HTTPError(405, "Use GET")
            return self._info()
        if method != "POST":
            raise HTTPError(405, "Use POST")
        if len(body) > self.config.max_body_bytes:
            raise HTTPError(413, f"Body exceeds {self.config.max_body_bytes} bytes")
        if path == "/feedback":
            return self._feedback(self._parse(body, FeedbackRequest), key, owner)
        if path == "/chat":
            chat_req: ChatRequest = self._parse(body, ChatRequest)
            return self._chat(
                chat_req,
                self._groups(key, chat_req.groups),
                self._tenant(key, chat_req.tenant),
                owner,
            )
        req = self._parse(body)
        groups = self._groups(key, req.groups)
        tenant = self._tenant(key, req.tenant)
        if key is not None:
            logger.info("%s %s by %s (groups=%s tenant=%s)", method, path, key.name, groups, tenant)
        if path == "/query":
            return self._query(req, groups, tenant, owner)
        return self._retrieve(req, groups, tenant)


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
        out_headers = [(b"content-type", response.content_type.encode("latin-1"))] + [
            (k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in response.headers.items()
        ]
        await send(
            {"type": "http.response.start", "status": response.status, "headers": out_headers}
        )
        if response.stream is not None:
            stream = response.stream
            while (chunk := await asyncio.to_thread(next, stream, None)) is not None:
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b""})
            return
        payload = b"" if response.status == 204 else response.encoded()
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
            self.send_response(response.status, HTTPStatus(response.status).phrase)
            self.send_header("Content-Type", response.content_type)
            for k, v in response.headers.items():
                self.send_header(k, v)
            if response.stream is not None:  # HTTP/1.0: the connection closes at the end
                self.end_headers()
                for chunk in response.stream:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                return
            payload = b"" if response.status == 204 else response.encoded()
            self.send_header("Content-Length", str(len(payload)))
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
