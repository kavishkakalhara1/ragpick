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
| `sql` | 🗺 planned | structured data, analytics questions |
| `graph` | 🗺 planned | multi-hop and global questions (GraphRAG) |
| `agentic` | 🗺 planned | multi-step questions and tool use |
| `corrective` | 🗺 planned | noisy corpora with retrieval grading |

## CLI

| Command | What it does |
|---|---|
| `ragpick init` | interactive questions → `rag.yaml` (`--blueprint NAME --yes` for scripts) |
| `ragpick recommend` | blueprint, reasons, alternatives, warnings |
| `ragpick blueprints` | catalogue and roadmap |
| `ragpick ingest PATH` | load, chunk, embed, index, persist |
| `ragpick query "…"` | answer with citations (`--groups`, `--show-contexts`) |
| `ragpick eval DATASET -c a.yaml -c b.yaml` | comparison table (`--output results.json/.csv`) |
| `ragpick bench DATASET --data DOCS` | try every stable blueprint, pick the winner (`--write rag.yaml`) |
| `ragpick validate` | print the merged config (secrets masked) |

## Optional extras

| Extra | Adds |
|---|---|
| `openai` | OpenAI embeddings and chat models (and OpenAI-compatible servers) |
| `local` | sentence-transformers embeddings |
| `rerank` | cross-encoder reranker |
| `qdrant` | Qdrant vector store |
| `pdf` | PDF loader |
| `all` | everything above |

The Ollama generator needs no extra.

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
[Evaluation](docs/evaluation.md) · [LangChain & LlamaIndex](docs/adapters.md) ·
[Design decisions](docs/decisions.md)

## Roadmap

- **v0.2** ✅: `parent_child` blueprint, `ragpick bench`, LangChain and LlamaIndex adapters
- **v0.3**: `sql` and `corrective` blueprints, tracing, caching
- **Later**: `graph` and `agentic` blueprints, `ragpick serve` HTTP API, multi-tenancy,
  TypeScript SDK

## Contributing

Contributions are very welcome, especially new loaders, stores and blueprints. See
[CONTRIBUTING.md](CONTRIBUTING.md) to get set up in two minutes with `uv`.

## Author

Created and maintained by **Kavishka Kalhara**: [kavishka-kalhara.me](https://kavishka-kalhara.me) · [GitHub @kavishkakalhara1](https://github.com/kavishkakalhara1)

## License

Copyright 2026 Kavishka Kalhara. Licensed under [Apache-2.0](LICENSE).
