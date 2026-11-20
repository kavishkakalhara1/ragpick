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

## Tooling

31. **`types-PyYAML` is not a dev dependency** (it wasn't in the approved list), so mypy
    treats `yaml` as untyped via an override. Adding the stub package would be a one-line
    improvement.
32. **Interactive questions use `rich.prompt`** (`Prompt.ask(choices=...)`, `Confirm.ask`).
    Recent typer releases no longer depend on `click` and `typer.prompt` does not accept
    choice types; `rich` is already a core dependency, so this works on every supported
    typer version.
