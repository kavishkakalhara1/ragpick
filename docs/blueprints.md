# Blueprints

Run `ragpick blueprints` to see this list in your terminal.

| Blueprint | Status | Best for |
|---|---|---|
| `naive` | stable | prototypes, small corpora, natural-language questions over prose |
| `hybrid` | stable | the default for commercial systems; keyword-heavy domains; medium and large corpora |
| `parent_child` | planned | long structured documents where answers need surrounding context |
| `sql` | planned | structured data and analytics questions (text-to-SQL) |
| `graph` | planned | multi-hop and "main themes" questions (GraphRAG) |
| `agentic` | planned | complex questions needing several lookups or tools |
| `corrective` | planned | noisy corpora: grade retrieval, rewrite, or fall back |

## naive

```text
directory loader → recursive chunker (800/100) → embedder → memory store
→ vector retriever → generator
```

The simplest architecture that works. Its weak spot is exact terms: an embedding of
"E-4017" is barely different from "E-4071", so error codes, SKUs and legal references are
often missed.

## hybrid

```text
                       ┌→ vector search ─┐
query (+ ACL filters) ─┤                 ├→ RRF fusion → [cross-encoder] → generator
                       └→ BM25 search ───┘
```

BM25 catches exact terms and vectors catch paraphrases. **Reciprocal Rank Fusion**
combines the two rankings without score calibration:

```text
score(chunk) = Σ_lists  weight / (rrf_k + rank)        (rrf_k = 60 by default)
```

Tune it in your config:

```yaml
blueprint: hybrid
components:
  retriever:
    params: {k: 20, rrf_k: 60, candidates: 40, vector_weight: 1.0, keyword_weight: 1.0}
  reranker: {type: cross_encoder}          # pip install "ragpick[rerank]"
```

On the bundled example dataset (keyword-heavy support questions), `hybrid` beats
`naive` on hit rate and MRR even with the offline fake embedder:

```bash
ragpick eval examples/eval.jsonl --config examples/naive.yaml --config examples/hybrid.yaml
```

## Adding a blueprint

A blueprint is one YAML file in `src/ragpick/blueprints/`:

```yaml
metadata:
  name: my_blueprint
  description: One paragraph on what it does.
  best_for: [...]
  tradeoffs: [...]
  status: stable          # or planned
defaults:
  components: {...}       # same schema as rag.yaml `components`
  query: {top_k: 5}
```

Add a test that builds it (see `tests/test_blueprints.py`) and, if it should be
recommended, a selector rule. See [CONTRIBUTING](https://github.com/kavishkakalhara1/ragpick/blob/main/CONTRIBUTING.md).
