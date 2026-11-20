# Design decisions

Where the v0.1 brief left a choice open, ragpick picked the simplest option that keeps the
interfaces extensible. Each decision is listed here so it can be revisited.

## Configuration

1. **Component `type` change replaces the component.** When a layer (your YAML or a
   `--set` flag) changes a component's `type`, the blueprint's `params` for that component
   are dropped, because they belong to a different class (e.g. `dimension` for `fake` makes
   no sense for `openai`). If the type stays the same, `params` are merged key by key.
2. **CLI flags are generic `--set key.path=value` overrides**, parsed as YAML scalars,
   instead of one flag per setting. A few common settings also have dedicated flags
   (`query --top-k`, `query --groups`).
3. **Relative paths resolve against the config file's directory**, not the current
   directory, so `ragpick eval --config examples/naive.yaml` works from anywhere.
4. **New `data` key** (optional) holds the default source path for `ingest` and is used by
   `eval` to build an index automatically when none exists.
5. **`${VAR}` without a default fails loudly** with the key path. `${VAR:-default}`
   provides a fallback. Interpolation only happens in memory; `ragpick init` never writes
   secrets, and `ragpick validate` masks secret-looking keys (`*key*`, `*token*`,
   `*secret*`, `*password*`).
6. **`components` is required when no `blueprint` is set.** There is no implicit default
   blueprint; the error message suggests setting one.

## Blueprints and components

7. **Stable blueprints default to `fake` embedder and generator.** A blueprint describes
   the architecture (chunking, retrieval, fusion, reranking); the model provider is a
   separate choice. This makes every blueprint runnable offline. `ragpick init` asks for a
   provider (`fake`, `openai`, `ollama`), and `--yes` defaults to `fake`. Using the fake
   generator is logged at INFO level.
8. **Planned blueprints are YAML files with metadata only.** `ragpick blueprints` shows
   the roadmap from the same source of truth; loading one raises a clear error.
9. **Dependency injection by constructor parameter name.** The pipeline passes
   `embedder`, `store` and `keyword_index` to any component whose constructor declares
   a parameter with that name. Plugins get the same treatment without registration hooks.
10. **A retriever that needs a keyword index gets `bm25` automatically** when
    `components.keyword_index` is not set.
11. **Re-ingest is an upsert per document**: chunks of re-ingested documents are deleted
    first; other documents in the index are kept.
12. **Chunk ids are `"<doc_id>#<n>"`; doc ids are paths relative to the ingested
    directory.** These ids are what `relevant_doc_ids` in eval datasets refer to.
13. **The index records its embedder** in `ragpick.json`; querying with a different
    embedder config fails with a "re-ingest" message instead of returning garbage.
14. **Separate `Registry` instances start empty.** Only the global registry loads
    built-ins, which keeps isolated tests simple. `Registry.remove()` exists for cleanup.
15. **Entry points are named `"<kind>.<name>"`** and point at the factory. An entry
    point without a dot is just imported (side-effect registration). Broken plugins are
    logged and skipped.

## Access control and filters

16. **Filters are a plain dict.** A scalar value means equality, a list means `in`, list-valued
    metadata matches on intersection, and a chunk missing the key does not match.
17. **`allowed_groups` is special.** Chunks without it are public, and chunks with it
    require a group intersection. `allowed_groups: []` means "nobody".
18. **Secure by default.** `Pipeline.query(groups=None)` returns public chunks only. ACL
    filtering happens inside the stores and BM25 *before* ranking, so restricted text
    never reaches fusion, reranking, or the LLM.
19. **Document metadata comes from Markdown front matter.** That's how `allowed_groups`
    is attached to files. Programmatic loaders can set any metadata.
20. **Qdrant ACL**: implemented as `is_empty(allowed_groups) OR match_any(groups)`. Because
    Qdrant's `is_empty` also matches `[]`, an empty list is stored as a private sentinel so
    "nobody" is not treated as "public".

## Generation

21. **The model sees numeric citations `[1]`; `Answer.citations` holds chunk ids** (stable,
    meaningful outside the prompt). Out-of-range numbers are ignored.
22. **No contexts → "I don't know." without calling the LLM.**
23. **`TextCompleter` protocol** (`complete(system, user)`) gives raw LLM access for judges;
    LLM generators implement both it and `Generator`. The fake generator's `complete`
    returns a fixed, configurable string (a constant-score fake judge for plumbing tests).
24. **`Answer.usage.latency_ms` covers the whole query** (retrieval + generation).

## Evaluation

25. **Retrieval metrics are computed over the ranked context list mapped to `doc_id`s.**
    `precision@k` divides by `k` even when fewer than `k` contexts are returned. Rows without
    `relevant_doc_ids` are excluded from retrieval averages.
26. **`k` is the config's `query.top_k`.**
27. **Eval rows may carry `groups`**, so access-controlled questions can be evaluated.
28. **`eval --data PATH` ingests into a temporary index per config**, so comparisons never
    overwrite real indexes. Without `--data`, the config's existing index is used, built from
    `data` if missing.
29. **Cost is estimated for generation tokens only**, from a built-in price table that
    `eval.prices` can extend or override. Unknown models show `n/a`. Judge tokens are not included.
30. **Judge failures and query errors are recorded per row** and counted in `errors`; they
    don't abort the run.

## v0.2

33. **Parent text lives in child metadata** (`parent_id`, `parent_text`) instead of a new
    "document store" interface. It costs storage, but works unchanged with every
    `VectorStore`, BM25 and persistence path, including Qdrant.
34. **`parent_child` retriever wraps a base retriever** (`hybrid` default, or `vector`) and
    fetches `child_k` (default `4 * k`) children to find `k` distinct parents. Parent
    contexts get the best child's score and keep the child's `source`.
35. **Selector: `long_documents` (priority 60) beats `keyword_heavy` (50)**, because
    `parent_child` already uses hybrid search for children. Structured data (100) still wins.
36. **`ragpick bench` uses one `top_k` for every blueprint** (base config's, or 5), otherwise
    metrics are not comparable. The base config supplies providers; architecture keys
    (`chunker`, `keyword_index`, `retriever`, `reranker`) always come from each blueprint.
    Ties on the metric go to lower p50 latency. The written winner config is built from
    the *uninterpolated* base YAML, so `${ENV_VAR}` secrets are never expanded into it.
37. **Adapters depend on nothing.** They call public methods (`embed_documents`, `invoke`,
    `get_text_embedding_batch`, …) and build objects from `class_path` + `kwargs`, so
    ragpick doesn't pin LangChain/LlamaIndex versions. The inner constructor arguments
    are called `kwargs` to avoid confusion with the component's own `params`.
38. **External retrievers are post-filtered** for metadata filters and `allowed_groups`,
    because they can't apply ragpick filters themselves.
39. **`embed_query` is an optional embedder method**, used by retrievers for queries when
    present (asymmetric models), so the `Embedder` protocol stays unchanged.
40. **setup-uv is pinned to an exact version** (`v10.3.3`) because it no longer publishes
    major tags, and sets `activate-environment: true`, which newer versions no longer imply.

## v0.3

41. **Tracing is built in and dependency-free**: a context-variable span stack whose
    `span()`/`set_attributes()` are no-ops without an active trace. OpenTelemetry and
    similar backends are `trace_exporter` plugins rather than a dependency.
42. **Cache = one SQLite file**, keyed by SHA-256 of (namespace, component fingerprint,
    input). Generator fingerprints include the system prompt, so prompt changes invalidate
    answers. Cached answers report `usage.cached = True` and zero tokens.
43. **The generator is built before the retriever** and injected by name, so retrievers
    that need an LLM (`corrective`, `sql`) reuse the pipeline's (cached, traced) model.
44. **`corrective` grades in one LLM call per round** (all passages in one prompt), not one
    call per passage. An unparseable grader reply keeps all contexts and logs a warning
    (fail open on parsing, so a flaky model doesn't silence every answer). `grader: auto`
    uses lexical grading for fake generators (`is_fake = True`).
45. **Corrective refusal returns zero contexts**, and the generator answers "I don't know"
    without an LLM call. `on_failure: keep` opts out.
46. **SQL uses the standard-library `sqlite3`** with defence in depth: read-only URI +
    `query_only`, an authorizer that allows only `SELECT`/`READ`/`FUNCTION`/`RECURSIVE`
    and only reads of visible real tables, single-`SELECT` validation, a progress-handler
    timeout and a row cap. CTE names aren't tables; reads inside CTEs are checked
    individually. Other databases (Postgres, …) can come later behind the same retriever.
47. **SQL access control is table-level** (`table_groups`), enforced in both the schema
    prompt and the authorizer; row-level security is out of scope. Access-denied errors
    are not retried, so the model can't probe.
48. **CSV ingest goes through a `DocumentIndexer` hook**: the `csv` loader yields one
    document per file, the `none` chunker keeps the vector store empty, and the SQL
    retriever builds `<index_dir>/sql/data.sqlite`. With `database:` set, no ingest is
    required (`requires_ingest = False`).
49. **The `sql` blueprint defaults to the fake generator** like every blueprint (so building
    it never needs a key), and the retriever explains at query time that SQL needs a real
    LLM. Blueprints declare `data: documents | structured`; `bench` compares document ones.

## v0.4

50. **Routing composes whole pipelines** (a router config lists route `rag.yaml` files)
    rather than adding a "router retriever", because SQL and document pipelines differ in
    ingest, index and data. Nested routers are rejected to keep traces and evaluation simple.
51. **Router LLM defaults to the first route's real LLM**; `auto` silently uses keywords
    without one, while `strategy: llm` fails loudly. Invalid LLM replies fall back to
    keyword scoring, not an error.
52. **Fallback goes to the `default` route only** (one retry), triggered by "I don't know",
    no contexts, or a failed SQL query. Route errors (e.g. configuration) are raised, not
    swallowed.
53. **Route `groups` gate routing itself**, so a caller can never reach (or learn about by
    being routed to) a route outside their groups.
54. **The HTTP server has no dependencies**: one framework-independent `App.handle`,
    exposed as an ASGI app (uvicorn if installed) and via `ThreadingHTTPServer`. Engine
    calls run in a worker thread under ASGI.
55. **API keys are bound to groups.** Requests may narrow but never widen their key's
    groups; omitted groups mean the key's groups. Without keys, request groups are
    trusted, and `serve` warns on non-loopback hosts. `/health` stays unauthenticated for
    load balancers.
56. **Errors don't leak internals over HTTP**: provider and configuration errors become
    502/500 with a generic message; details go to the server log.
57. **The OpenTelemetry exporter replays finished traces** rather than instrumenting live,
    so ragpick's tracing stays dependency-free and every backend sees identical spans.
    Root spans carry a wall-clock `timestamp` for absolute times.

## v0.5

58. **Streams are events, not raw text**: contexts first (so UIs can show sources while
    the answer is written), then deltas, then the full `Answer` with parsed citations.
    Generators without `stream()` produce one delta. Streams aren't traced (generators
    and context variables don't mix reliably), and routers don't fall back mid-stream.
59. **Streamed HTTP responses retrieve before streaming**, so "not ingested" and
    validation errors keep proper status codes; later failures end the stream with an
    `error` event that hides provider details.
60. **GraphRAG v1 is local search**: entity matching + N-hop expansion fused with hybrid
    search by RRF, plus a facts context. Global "themes" search needs community
    summarisation and is deferred.
61. **Graph extraction runs through a `ChunkIndexer` hook** and is persisted as JSON next to
    the other indexes; the graph stores chunk copies (like BM25) so it can return chunks
    and check their metadata. Re-ingest replaces a document's graph contributions.
62. **The offline extractor is deliberately conservative**: codes, acronyms and
    multi-word capitalised names only (single capitalised words are mostly UI labels and
    sentence starts), with codes kept inside longer names so "LM-300" links to "LM-300 Pro".
63. **Graph facts respect access control** by requiring a supporting chunk the caller can
    see, so relations learned from restricted documents never appear for others.
64. **Synthetic contexts** (`metadata.synthetic`) are excluded from retrieval metrics,
    since they aren't documents.
65. **PostgreSQL table checks use `EXPLAIN (FORMAT JSON)`** relation lists, which see through
    views, CTEs and subqueries, plus read-only sessions, `statement_timeout` and a
    function denylist. A least-privilege database role remains the primary boundary.
    `psycopg` is loaded lazily with a custom install hint (no new extra).

## v0.6

66. **Communities use deterministic weighted label propagation** (stdlib, reproducible,
    near-linear) rather than Louvain/Leiden; it's good enough for summarising themes.
    Communities have at least two entities.
67. **Community summaries are keyed by a signature** of members, relations and supporting
    chunks, and reused while unchanged, so re-ingest only re-summarises what changed.
68. **Global search returns community summaries as synthetic contexts** (the generator does
    the "reduce" step); there is no separate map step per community, which keeps it to
    one LLM call. Questions route to global search by cue phrases (`mode: auto`), and fall
    back to local search if no visible summary matches.
69. **A summary is visible only if all its supporting chunks are visible** (conservative:
    a summary can't be partially redacted, so any restricted source hides it).
70. **Agentic retrieval is a retriever, not a chat agent**: it plans searches and returns
    fused contexts, so the grounded generator, citations, evaluation and access control
    stay unchanged. It always searches the original question first; repeats, two idle
    searches in a row, or `max_steps` end the loop. Unparseable plans count as "finish".
71. **Tenancy creates a child pipeline per tenant** with its own index directory and fresh
    stateful components (store, keyword index, retriever), sharing stateless ones
    (embedder, generator, cache). An LRU bounds memory. Tenant ids are restricted to a
    filesystem-safe pattern before any path is built.
72. **Tenancy is explicit both ways**: multi-tenant pipelines require a tenant, normal
    pipelines reject one (a `TenantError`, HTTP 400). Tenant-bound API keys override and
    restrict the request's tenant (403 on mismatch).

## v0.7

73. **Multimodal = describe-then-index.** Images become text documents at ingest, so
    every retriever, store, reranker, evaluation and access-control feature works
    unchanged. Native image embeddings (CLIP-style) can come later as an embedder.
74. **The generator is built first** and injected into loaders by parameter name, so
    the multimodal loader reuses the pipeline's (cached) vision model.
75. **Vision is an optional protocol** (`describe_image`) on the OpenAI and Ollama
    generators. If a vision call fails, the image keeps its offline description
    (`description_source: offline-fallback`), so one bad image doesn't abort ingest.
76. **Offline image descriptions use what authors already wrote**: file names, Markdown
    alt text and caption files. Caption front matter can restrict images.
77. **pgvector without the pgvector package**: vectors are sent as text and cast to
    `vector`; filters become parameterised SQL over JSONB (values compared as text, list
    metadata matched by array overlap), and the table name must be a plain identifier.
78. **Generated datasets are a starting point.** Chunks are sampled round-robin across
    documents for coverage. LLM output that doesn't parse falls back to the heuristic,
    and restricted documents' groups are copied onto their questions.

## v0.8

79. **Conversations rewrite, then retrieve**: the grounded pipeline always sees one
    standalone question, so retrieval, citations, access control and evaluation are
    unchanged. History isn't injected into the answer prompt.
80. **The offline rewrite is conservative**: only messages that look like follow-ups
    (short, a follow-up opener, or a pronoun) get the previous question attached.
81. **Sessions are owned by (API key, tenant)**, and a foreign session id behaves exactly
    like an unknown one (404), so session ids can't be used to probe or read history.
82. **Thresholds are explicit and strict**: `--fail-under` / `--fail-over` per metric;
    missing metrics fail; CLI values override config values.
83. **Chroma filters use derived scalar fields** (`m:`, `l:`, `acl:`) because Chroma
    metadata can't hold lists and `where` can't test membership; the original metadata
    is stored as JSON. A reference evaluator checks the translation against
    `matches_filters` in tests.
84. **LanceDB filters use a `tags` list column** (`key=value`) and `array_has_any`, with
    every value quoted and escaped, and run as a pre-filter before vector ranking.

## v0.9

85. **RAPTOR uses the "collapsed tree"**: summary nodes live in the same vector store and
    keyword index as chunks, so any retriever, reranker and filter applies unchanged.
86. **Clustering is greedy nearest-neighbour** (deterministic, numpy, no new
    dependency) rather than GMM/UMAP as in the RAPTOR paper.
87. **Trees are built per access scope**, so summaries inherit exactly one
    `allowed_groups` set and access control is enforced by the normal filters.
88. **Summaries are cached by a signature** of their members, persisted with the tree;
    re-ingest rebuilds structure but only re-summarises changed clusters.
89. **Answers carry an `id`**; the server rates only answers it remembers giving to the
    same (API key, tenant), and builds the feedback record from its own copy, so clients
    can't inject questions/answers or rate others' answers.
90. **Feedback becomes evaluation data**: liked answers with cited documents become
    regression tests; disliked ones become unanswered questions to review.
91. **Hosted rerankers use `httpx` directly**, with the same `{index, relevance_score}`
    result shape for Cohere and Jina, and keep each chunk's original `source`.

## Tooling

31. **`types-PyYAML` is not a dev dependency** (it wasn't in the approved list), so mypy
    treats `yaml` as untyped via an override. Adding the stub package would be a one-line
    improvement.
32. **Interactive questions use `rich.prompt`** (`Prompt.ask(choices=...)`, `Confirm.ask`).
    Recent typer releases no longer depend on `click` and `typer.prompt` does not accept
    choice types; `rich` is already a core dependency, so this works on every supported
    typer version.
