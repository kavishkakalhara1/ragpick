# Serving an HTTP API

`ragpick serve` exposes any config (a pipeline or a router) as a JSON API.

```bash
ragpick serve -c rag.yaml                         # http://127.0.0.1:8000
ragpick serve -c router.yaml --host 0.0.0.0 --port 8080
```

It runs on **uvicorn** when installed (`pip install uvicorn`, recommended for
production) and falls back to Python's built-in threaded HTTP server otherwise. The
application is a plain ASGI app, so you can also mount it in your own server:

```python
from ragpick import load_engine
from ragpick.server import App, asgi_app

application = asgi_app(App(load_engine("rag.yaml")))  # uvicorn module:application
```

## Endpoints

| Method & path | Body | Returns |
|---|---|---|
| `GET /health` | – | `{"status": "ok" \| "not_ready", "version"}`, 503 until ingested. No auth. |
| `GET /info` | – | name, blueprint, version, routes (for routers) |
| `POST /query` | `{"question", "groups"?, "filters"?, "top_k"?, "trace"?}` | `{"text", "citations", "route", "contexts", "usage", "trace"?}` |
| `POST /retrieve` | same | `{"contexts": [...]}` (no generation) |

```bash
curl -s localhost:8000/query \
  -H "Authorization: Bearer $PORTAL_KEY" \
  -d '{"question": "How do I reset my password?"}'
```

Add `"stream": true` to get **server-sent events** instead of one JSON body: one
`contexts` event, many `delta` events (`{"text": "..."}`), then a final `answer` event
(the same JSON as a normal response). Retrieval errors still return a normal error
status, because retrieval runs before the stream starts. A provider failure mid-stream
ends the stream with an `error` event.

```bash
curl -N localhost:8000/query -d '{"question": "How do I reset my password?", "stream": true}'
```

Each context is `{"id", "doc_id", "score", "source", "text", "metadata"}`. Errors are
`{"error": "..."}` with a meaningful status: 401/403 auth, 404, 405, 413 body too large,
422 invalid body, 502 model provider failed, 503 not ingested. Provider and
configuration details are logged on the server, not returned to clients.

## Authentication and groups

```yaml
serve:
  host: 0.0.0.0
  port: 8000
  api_keys:
    - {key: "${PORTAL_KEY}", name: support-portal, groups: [support]}
    - {key: "${HR_APP_KEY}", name: hr-app, groups: [hr, support]}
    - {key: "${ADMIN_KEY}", name: admin}          # groups: null → may act as any groups
  cors_origins: ["https://portal.example.com"]
  allow_traces: false
  max_body_bytes: 65536
```

- With `api_keys` set, every endpoint except `/health` requires
  `Authorization: Bearer <key>` (compared in constant time).
- **The key decides the groups.** A request that omits `groups` acts as all of its key's
  groups; a request may narrow them (`"groups": []` is public-only) but can't add a
  group its key doesn't have (403). This is what makes document- and table-level access
  control safe over HTTP.
- Without `api_keys`, request `groups` are trusted as sent. That's fine on localhost or
  behind your own authenticated backend; `ragpick serve` warns when binding a public
  interface without keys.
- Keep keys in environment variables (`${PORTAL_KEY}`); they are never written to disk.

## Tenants

With `tenancy` enabled, requests carry `"tenant": "acme"`, or the API key fixes it
(`api_keys[].tenant`); see [Multi-tenancy](tenancy.md).

## Operations

- Traces in responses are off unless `serve.allow_traces: true`. Server-side tracing
  (`tracing.enabled`, JSONL/OpenTelemetry exporters) works independently.
- Turn on `cache` to make repeated questions instant and free.
- One process loads the index into memory; for more throughput run several uvicorn
  workers behind a load balancer, with a shared vector database such as Qdrant.
