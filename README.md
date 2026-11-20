# ragpick

**Pick the right RAG architecture for your system, define it as code, and prove it works on your data.**

[![CI](https://github.com/kavishkakalhara1/ragpick/actions/workflows/ci.yml/badge.svg)](https://github.com/kavishkakalhara1/ragpick/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ragpick.svg)](https://pypi.org/project/ragpick/)
[![Python](https://img.shields.io/pypi/pyversions/ragpick.svg)](https://pypi.org/project/ragpick/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## Why ragpick

Building a RAG pipeline is easy. **Choosing the right one is the hard part.** Plain vector
search misses the product codes and IDs your customers type. Internal documents leak into
answers for the wrong users. A "better" pipeline ships without evidence that it is better.

ragpick makes the architecture an explicit, testable decision:

- **Recommend.** Describe your requirements and get a blueprint plus the reasons for it,
  the alternatives, and warnings.
- **Build.** The pipeline is a validated `rag.yaml` (architecture as code), with
  document-level **access control** built in.
- **Evaluate.** Compare blueprints on your own questions: hit rate, MRR, faithfulness,
  latency and cost, side by side.

ragpick works alongside LangChain and LlamaIndex rather than replacing them. It owns the
interfaces, config, selector and evaluation; anything else plugs in as a component.

## 60-second quickstart (offline, no API key)

```bash
pip install ragpick
git clone https://github.com/kavishkakalhara1/ragpick && cd ragpick   # for the sample data

ragpick init --blueprint naive --yes          # writes rag.yaml with fake, offline models
ragpick ingest examples/data                  # load → chunk → embed → index
ragpick query "How do I reset my password?"   # answer with citations
```

```text
╭──────────────────────────────── Answer ────────────────────────────────╮
│ On the sign-in screen of the Lumen app, tap "Forgot password" and      │
│ enter the email address on your account. [1] You will receive a reset  │
│ link that is valid for 30 minutes. [1]                                 │
╰────────────────────────────────────────────────────────────────────────╯
Citations: account-and-password.md#0
```

Ask which architecture fits, then compare two blueprints on a question set:

```bash
ragpick recommend --config examples/hybrid.yaml
ragpick eval examples/eval.jsonl --config examples/naive.yaml --config examples/hybrid.yaml
```

No questions yet? Generate a starter set from your documents (then review it):

```bash
ragpick generate-eval examples/data -o my-eval.jsonl --count 20
```

Or let `ragpick bench` try every stable blueprint on your data and pick the winner:

```bash
ragpick bench examples/eval.jsonl --data examples/data --write best.yaml
```

```text
           Evaluation results
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric           ┃   naive ┃   hybrid ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ hit_rate@k       │   0.929 │    1.000 │
│ mrr              │   0.774 │    0.893 │
│ precision@k      │   0.357 │    0.381 │
│ ...              │         │          │
```

The fake components are deterministic stand-ins for tests and demos: the embedder is a
hashing trick and the generator quotes the best-matching sentences.

## Quickstart with OpenAI

```bash
pip install "ragpick[openai]"
export OPENAI_API_KEY=sk-...

ragpick init --blueprint hybrid --provider openai --yes
ragpick ingest ./docs
ragpick query "How do I reset my password?" --groups support --show-contexts
```

Or in Python:

```python
from ragpick import Pipeline

p = Pipeline.from_yaml("rag.yaml")
p.ingest("./docs")
ans = p.query("How do I reset my password?", groups=["support"])
print(ans.text, ans.citations)
```

A `rag.yaml` names a blueprint and overrides only what differs:

```yaml
version: 1
name: my-support-bot
blueprint: hybrid
requirements: {corpus_size: medium, keyword_heavy: true, access_control: true}
components:
  embedder:  {type: openai, params: {model: text-embedding-3-small}}
  generator: {type: openai, params: {model: gpt-4o-mini, temperature: 0}}
query: {top_k: 5}
```

Restrict a document to groups with front matter. Queries without matching `--groups` never see it:

```markdown
---
allowed_groups: [support]
---
```

## Blueprints

| Blueprint | Status | Best for |
|---|---|---|
| `naive` | ✅ stable | prototypes, small corpora |
| `hybrid` | ✅ stable | **default for commercial systems**; keyword-heavy domains (BM25 + vector + RRF, optional cross-encoder) |
| `parent_child` | ✅ stable | long structured documents: search small chunks, answer from whole sections |
| `sql` | ✅ stable | structured data (CSV, SQLite, PostgreSQL): read-only text-to-SQL with table-level access control |
| `graph` | ✅ stable | multi-hop and "main themes" questions: knowledge graph + community summaries (GraphRAG) |
| `agentic` | ✅ stable | multi-part and comparison questions: plans several searches, fuses the results |
| `multimodal` | ✅ stable | images, diagrams and scanned pages: described at ingest (vision LLM or offline), searched with the text |
| `conversational` | ✅ stable | chat assistants: follow-ups rewritten with the session history, persistent sessions |
| `hierarchical` | 🗺 planned | RAPTOR-style summary trees: details and big-picture answers |
| `corrective` | ✅ stable | grades retrieval, retries, and says "I don't know" instead of guessing |

## CLI

| Command | What it does |
|---|---|
| `ragpick init` | interactive questions → `rag.yaml` (`--blueprint NAME --yes` for scripts) |
| `ragpick recommend` | blueprint, reasons, alternatives, warnings |
| `ragpick blueprints` | catalogue and roadmap |
| `ragpick ingest PATH` | load, chunk, embed, index, persist |
| `ragpick query "…"` | answer with citations (`--groups`, `--show-contexts`, `--trace`, `--stream`) |
| `ragpick eval DATASET -c a.yaml -c b.yaml` | comparison table (`--output`, `--fail-under hit_rate@k=0.8` for CI) |
| `ragpick bench DATASET --data DOCS` | try every stable blueprint, pick the winner (`--write rag.yaml`) |
| `ragpick generate-eval DOCS` | generate evaluation questions + answers from your documents |
| `ragpick validate` | print the merged config (secrets masked) |
| `ragpick cache stats` / `clear` | inspect or clear the embedding/LLM cache |
| `ragpick serve` | JSON HTTP API with API keys mapped to groups |
| `ragpick chat` | interactive chat; follow-up questions use the conversation |

## Optional extras

| Extra | Adds |
|---|---|
| `openai` | OpenAI embeddings and chat models (and OpenAI-compatible servers) |
| `local` | sentence-transformers embeddings |
| `rerank` | cross-encoder reranker |
| `qdrant` | Qdrant vector store (pgvector, Chroma and LanceDB work with `psycopg[binary]`, `chromadb`, `lancedb`) |
| `pdf` | PDF loader |
| `all` | everything above |

The Ollama generator needs no extra.

## Routing and serving

Answer from documents *and* databases: a router sends each question to the right
pipeline, and `ragpick serve` puts it behind an HTTP API where **API keys decide which
groups a caller may act as**:

```yaml
# router.yaml
router:
  routes:
    - {name: docs, config: hybrid.yaml, description: Manuals, troubleshooting, policies}
    - {name: analytics, config: sql.yaml, description: Orders, sales and prices,
       keywords: [how many, total, revenue], groups: [staff]}
serve:
  api_keys: [{key: "${PORTAL_KEY}", name: support-portal, groups: [support]}]
```

```bash
ragpick serve -c router.yaml
curl -s localhost:8000/query -H "Authorization: Bearer $PORTAL_KEY" \
  -d '{"question": "What does error E-1042 mean?"}'
```

Answers can stream token by token: `ragpick query "…" --stream`, `pipeline.stream(...)`,
or `POST /query {"stream": true}` (server-sent events).

Building a chat assistant? `pipeline.chat(message, session_id=...)` rewrites follow-ups
("and can I cancel it?") into standalone questions; see [docs/conversations.md](docs/conversations.md).

Serving many customers? `tenancy: {enabled: true}` gives every tenant an isolated
index, and API keys can be bound to a tenant. See [docs/tenancy.md](docs/tenancy.md).

See [docs/routing.md](docs/routing.md) and [docs/serving.md](docs/serving.md).

## Tracing and caching

See where time goes, and stop paying twice for the same work:

```bash
ragpick query "What does error E-1042 mean?" --trace
```

```text
query 2.4 ms
├── retrieve 0.7 ms  retriever=corrective attempts=1 kept=2
│   ├── vector_search 0.5 ms  results=5
│   ├── keyword_search 0.0 ms  results=2
│   └── grade 0.2 ms  grader=lexical contexts=5 relevant=2
└── generate 0.1 ms  prompt_tokens=516 completion_tokens=33 cached=False
```

```yaml
tracing: {enabled: true, exporters: [{type: jsonl, params: {path: .ragpick/traces.jsonl}}]}
cache: {enabled: true}     # embeddings, answers and LLM calls, keyed by model fingerprint
```

Traces can also go to OpenTelemetry (`{type: otel}` exporter). See [docs/operations.md](docs/operations.md).

## LangChain and LlamaIndex

Use their loaders, embedding models, retrievers and LLMs as ragpick components, with no
extra dependency in ragpick:

```yaml
components:
  embedder:
    type: langchain
    params: {class_path: langchain_openai.OpenAIEmbeddings, kwargs: {model: text-embedding-3-small}}
  generator:
    type: llamaindex
    params: {class_path: llama_index.llms.openai.OpenAI, kwargs: {model: gpt-4o-mini}}
```

See [docs/adapters.md](docs/adapters.md).

## Documentation

[Concepts](docs/concepts.md) · [Blueprints](docs/blueprints.md) ·
[Config reference](docs/config-reference.md) · [Writing a plugin](docs/plugins.md) ·
[Evaluation](docs/evaluation.md) · [Routing](docs/routing.md) · [Serving](docs/serving.md) ·
[Multi-tenancy](docs/tenancy.md) · [Conversations](docs/conversations.md) ·
[Tracing & caching](docs/operations.md) ·
[LangChain & LlamaIndex](docs/adapters.md) ·
[Design decisions](docs/decisions.md)

## Roadmap

- **v0.2** ✅: `parent_child` blueprint, `ragpick bench`, LangChain and LlamaIndex adapters
- **v0.3** ✅: `sql` and `corrective` blueprints, tracing, caching
- **v0.4** ✅: query routing between `sql` and document pipelines, `ragpick serve` HTTP API,
  OpenTelemetry trace exporter
- **v0.5** ✅: `graph` blueprint (GraphRAG local search), streaming answers, PostgreSQL
  backend for `sql`
- **v0.6** ✅: GraphRAG global search (community summaries), `agentic` blueprint,
  per-tenant indexes
- **v0.7** ✅: `multimodal` blueprint, pgvector store, evaluation datasets generated from
  your documents
- **v0.8** ✅: `conversational` blueprint (chat history, follow-up rewriting), Chroma and
  LanceDB stores, CI-friendly `ragpick eval --fail-under`
- **v0.9**: `hierarchical` (RAPTOR) blueprint, answer feedback collection, reranker
  plugins (Cohere, Jina)
- **Later**: TypeScript SDK

## Contributing

Contributions are very welcome, especially new loaders, stores and blueprints. See
[CONTRIBUTING.md](CONTRIBUTING.md) to get set up in two minutes with `uv`.

## Author

Created and maintained by **Kavishka Kalhara**: [kavishka-kalhara.me](https://kavishka-kalhara.me) · [GitHub @kavishkakalhara1](https://github.com/kavishkakalhara1)

## License

Copyright 2026 Kavishka Kalhara. Licensed under [Apache-2.0](LICENSE).
