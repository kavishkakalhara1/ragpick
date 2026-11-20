# Upgrading

## From 0.x to 1.0

1.0 removes nothing: every 0.9 config and API keeps working. What changes is the promise.
From 1.0 on, the [public surface](stability.md) only changes compatibly within 1.x.

If you're upgrading from an older 0.x release, check these behaviour changes (newest first):

| Since | Change | What to do |
|---|---|---|
| 1.0 | `Router.from_yaml()` / `Router.from_dict()` added; `RagpickDeprecationWarning` exported | nothing, just new options |
| 0.9 | Every `Answer` has `id` and `question`; streamed `/query` now honours `tenant` | nothing; tenant-bound streaming clients now work |
| 0.8 | `Answer.session_id` / `standalone_question`; new `conversation` config | nothing |
| 0.7 | The generator is built **before** the loader (loaders can receive it) | custom loaders declaring a `generator` parameter now get it injected |
| 0.7.1 | Cached vision generators are detected on Python 3.12+ | upgrade to ≥ 0.7.1 (or 0.8+) if you use `multimodal` with the cache |
| 0.6 | `CorrectiveRetriever(index_dir=...)` parameter (optional) | nothing |
| 0.6 | Multi-tenant pipelines **require** `tenant=`; normal pipelines reject it | only affects configs with `tenancy.enabled: true` |
| 0.5 | Contexts with `metadata.synthetic` (graph facts, community/RAPTOR summaries) are excluded from retrieval metrics | retrieval metrics may rise slightly for `graph` |
| 0.5 | SQL backends raise `SQLQueryError` (not `sqlite3.Error`) for retryable errors | update custom code that caught `sqlite3.Error` from `SQLDatabase.run` |
| 0.4 | `evaluate()` accepts any engine; the eval runner records each question's `route` | nothing |
| 0.3 | `ragpick bench` compares document blueprints by default (`-b sql` to include SQL) | pass `-b` explicitly for structured data |
| 0.2 | `Answer.citations` for `parent_child` are parent ids (`doc#3`), not child ids | update code that parses citation ids |

After upgrading, run your evaluation set with thresholds so regressions show up immediately:

```bash
pip install -U ragpick
ragpick validate -c rag.yaml
ragpick eval eval.jsonl -c rag.yaml --fail-under hit_rate@k=0.85
```

Indexes built by any 0.x version are read by 1.0. If you change the **embedder**, ragpick
detects the mismatch and asks you to re-ingest.

## Coming from LangChain or LlamaIndex

You don't have to rewrite your components: ragpick's adapters run them inside a ragpick
pipeline ([guide](adapters.md)).

```yaml
blueprint: hybrid                      # architecture chosen (and benchmarked) by ragpick
components:
  loader:    {type: llamaindex}        # SimpleDirectoryReader
  embedder:  {type: langchain, params: {class_path: langchain_openai.OpenAIEmbeddings}}
  generator: {type: langchain, params: {class_path: langchain_openai.ChatOpenAI, kwargs: {model: gpt-4o-mini}}}
```

A sensible migration path:

1. Generate a starter evaluation set from your documents: `ragpick generate-eval ./docs`.
2. Wrap your existing embedder/LLM with the adapters, then run `ragpick bench` to see
   which blueprint wins on your data.
3. Add access control (`allowed_groups` front matter), `ragpick serve` with API keys,
   and `--fail-under` gates in CI.
