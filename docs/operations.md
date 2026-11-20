# Tracing and caching

## Tracing

A trace records every stage of a query with its timing and key facts: what was retrieved,
how many contexts passed grading, the SQL that ran, token counts, and whether the answer
came from the cache.

```bash
ragpick query "What does error E-1042 mean?" --trace
```

```text
query 2.4 ms  groups=0 blueprint=corrective
├── retrieve 0.7 ms  retriever=corrective k=5 attempts=1 kept=2 results=2
│   ├── vector_search 0.5 ms  k=5 results=5
│   ├── keyword_search 0.0 ms  k=5 results=2
│   └── grade 0.2 ms  grader=lexical contexts=5 relevant=2
├── rerank 0.0 ms  reranker=none top_n=3 results=2
└── generate 0.1 ms  generator=fake prompt_tokens=516 completion_tokens=33 cached=False
```

Record and export traces for every query:

```yaml
tracing:
  enabled: true
  exporters:
    - {type: jsonl, params: {path: .ragpick/traces.jsonl}}   # one JSON object per query
    - {type: console}                                        # logs a tree (INFO level)
```

In Python, `Pipeline.query(..., trace=True)` sets `answer.trace`, a tree of `Span`
objects (`name`, `duration_ms`, `attributes`, `children`, `error`). `span.find("grade")`
looks up a stage.

**Instrument your own components** with the same helpers. They do nothing when tracing
is off:

```python
from ragpick.tracing import set_attributes, span


def retrieve(self, query, k, filters=None):
    with span("my_search", k=k):
        results = ...
        set_attributes(results=len(results))
        return results
```

**Custom exporters** (e.g. OpenTelemetry, Langfuse) are plugins of kind
`trace_exporter`: a class with `export(root: Span)`, registered with
`@register("trace_exporter", "otel")` or the `ragpick.components` entry point
`"trace_exporter.otel"`. A failing exporter is logged and never breaks a query.

## Caching

Repeated work is expensive: re-ingesting unchanged documents re-embeds them, and
re-running an evaluation pays for every answer again. Turn on the cache:

```yaml
cache:
  enabled: true
  dir: .ragpick/cache        # a single SQLite file inside
  embeddings: true           # document and query embeddings
  generations: true          # answers and LLM completions (grading, SQL, rewrites)
  ttl_seconds: 604800        # optional: expire after 7 days
```

- **Keys are fingerprints.** Each key hashes the component type, its params and the prompt
  version together with the input. Changing the model, temperature or prompt therefore never
  returns a stale result. Secrets in params only ever enter the cache inside a hash.
- **Cached answers are free:** `answer.usage.cached` is `True` and token counts are 0, so
  evaluation cost estimates reflect what you actually paid.
- Because cached answers return almost instantly, turn the cache off when you want
  `ragpick eval` / `bench` to measure real latency.

```bash
ragpick cache stats      # entries per kind
ragpick cache clear
```
