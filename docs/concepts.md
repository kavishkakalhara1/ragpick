# Concepts

## Components

A pipeline is made of small, swappable components. Each one is a `typing.Protocol` in
`ragpick.interfaces`, so any object with the right methods works. No base class is needed.

| Kind | Protocol | Built-in types |
|---|---|---|
| `loader` | `load(path) -> Iterable[Document]` | `directory`, `text`, `pdf` |
| `chunker` | `split(docs) -> list[Chunk]` | `recursive`, `fixed` |
| `embedder` | `embed(texts)`, `dimension` | `fake`, `openai`, `sentence_transformers` |
| `store` | `add`, `search`, `delete`, `persist`, `load` | `memory`, `qdrant` |
| `keyword_index` | `add`, `search`, `persist`, `load` | `bm25` |
| `retriever` | `retrieve(query, k, filters)` | `vector`, `keyword`, `hybrid` |
| `reranker` | `rerank(query, chunks, top_n)` | `none`, `cross_encoder` |
| `generator` | `generate(query, contexts) -> Answer` | `fake`, `openai`, `ollama` |

Data flows through these models: `Document` → `Chunk` → `RetrievedChunk` → `Answer`
(text, `citations` as chunk ids, `contexts`, `usage`).

**Ingest:** loader → chunker → embedder → store (+ keyword index) → persisted to `index_dir`.

**Query:** retriever (with ACL filters) → reranker → generator → `Answer`.

## Blueprints

A blueprint is a named, tested architecture: a YAML file with metadata (`description`,
`best_for`, `tradeoffs`, `status`) and default components. Your `rag.yaml` names a
blueprint and overrides only what differs:

```yaml
blueprint: hybrid
components:
  embedder:  {type: openai, params: {model: text-embedding-3-small}}
  generator: {type: openai, params: {model: gpt-4o-mini}}
```

See [Blueprints](blueprints.md) for the catalogue and roadmap.

## Architecture as code

`rag.yaml` is validated by Pydantic. Mistakes are reported with their YAML key path:

```text
Invalid configuration in rag.yaml:
  - query.topk: unknown key (check for typos)
  - components.retriever: required key is missing (set it, or choose a `blueprint` that provides it)
```

Layers merge as **blueprint defaults → `rag.yaml` → `--set` flags**. Secrets come from
the environment with `${OPENAI_API_KEY}` and are never written back to disk. Check the
merged result with `ragpick validate`.

## The selector

`ragpick recommend` runs a table of explainable rules over your `requirements`:

| Situation | Effect |
|---|---|
| small corpus, not keyword-heavy | `naive` |
| keyword-heavy, or medium/large corpus | `hybrid` |
| structured data (sql, csv, …) | `sql` (planned) → warns and falls back to `hybrid` |
| `latency: low` | reranker disabled, with a reason |
| `access_control` with a store that cannot filter | warning |
| large corpus on the in-memory store | warning, suggests Qdrant |
| pdf data without `ragpick[pdf]` | warning |

Every recommendation lists **reasons**, **alternatives** and **warnings**. The rules live
in `ragpick.selector.rules.RULES` and are easy to extend.

## Access control

Document-level access control is built in. Attach groups to a document with Markdown front
matter (or any loader's metadata):

```markdown
---
allowed_groups: [support]
---
# Internal escalation guide
```

Then query as a member of some groups:

```python
p.query("How do I escalate?", groups=["support"])
```

```bash
ragpick query "How do I escalate?" --groups support
```

Rules:

- Chunks **without** `allowed_groups` are public.
- Chunks **with** it are returned only if the caller's groups intersect it.
- `groups=None` (or no `--groups`) means **public chunks only**. This is secure by default.
- Filtering happens inside every store and the BM25 index *before* ranking, so restricted
  text never reaches fusion, reranking, or the LLM.

Other metadata filters use the same mechanism: `p.query(q, filters={"lang": ["en", "de"]})`.
