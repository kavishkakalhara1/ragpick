# Blueprints

Run `ragpick blueprints` to see this list in your terminal.

| Blueprint | Status | Best for |
|---|---|---|
| `naive` | stable | prototypes, small corpora, natural-language questions over prose |
| `hybrid` | stable | the default for commercial systems; keyword-heavy domains; medium and large corpora |
| `parent_child` | stable | long structured documents where answers need surrounding context |
| `sql` | stable | structured data and analytics questions (text-to-SQL) |
| `graph` | stable | multi-hop questions across documents (GraphRAG local search) |
| `agentic` | planned | complex questions needing several lookups or tools |
| `corrective` | stable | noisy corpora, high-stakes answers: grade retrieval, retry, or refuse |

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

## parent_child

```text
document ─→ parent sections (2000 chars) ─→ child chunks (400 chars) ─→ indexed
query ─→ hybrid search over children ─→ dedupe by parent ─→ parent sections ─→ generator
```

Small chunks are precise to search but often too small to answer from; large chunks give
context but match poorly. `parent_child` does both: it searches children and hands the
LLM each matching child's whole parent section. Each child stores its parent's text in
metadata (`parent_id`, `parent_text`), so it works with every store and needs no extra
document store. Citations and contexts use parent ids (`manual.md#3`).

```yaml
blueprint: parent_child
components:
  chunker:
    params: {parent_size: 2000, parent_overlap: 0, child_size: 400, child_overlap: 50}
  retriever:
    params: {k: 5, base: hybrid, child_k: 20}   # base: hybrid | vector
query:
  top_k: 3        # parents are big: fewer contexts keep prompts reasonable
```

Trade-offs: bigger prompts, and parent text is stored once per child. Set
`requirements.long_documents: true` and `ragpick recommend` will suggest it.

## corrective

```text
query → base retrieval (hybrid) → grade each context ─┬─ enough relevant → generator
                                                      └─ too few → rewrite query → retrieve → grade
                                                                   └─ still none → "I don't know"
```

Plain RAG answers from whatever it retrieved, even when that's irrelevant, and that is
where many hallucinations come from. `corrective` checks first:

- **Grading** asks the LLM which passages actually help answer the question (one call per
  query). With the offline fake generator, a lexical test (share of question terms present)
  is used instead.
- **Correction.** If fewer than `min_relevant` contexts pass, an LLM rewrites the query
  (exact terms kept, synonyms added) and it retrieves again. Offline, it searches deeper in
  the ranking instead.
- **Refusal.** If nothing relevant is found, no contexts are passed on and the answer is
  "I don't know." (`on_failure: keep` falls back to the ungraded contexts.)

```yaml
blueprint: corrective
components:
  retriever:
    params: {base: hybrid, grader: auto, min_relevant: 1, rewrite: true, on_failure: empty}
  generator: {type: openai, params: {model: gpt-4o-mini}}   # enables LLM grading
```

`base: parent_child` combines it with parent-child retrieval (set the chunker to
`parent_child` too). Run `ragpick query --trace` to see the grade/rewrite steps. Set
`requirements.strict_grounding: true` and the selector recommends it.

## graph

```text
ingest:  chunk → extract entities & relations (LLM or heuristic) → knowledge graph
query:   question entities → expand N hops → graph-ranked chunks ─┐
         hybrid BM25 + vector ────────────────────────────────────┴→ RRF → [facts, chunks…]
```

Some questions need two facts that live in different places ("which error code relates
to the battery of the thermostat that runs without a C-wire?"). `graph` links entities
across chunks, so retrieval can follow those connections, and the LLM gets them
spelled out:

```text
Knowledge graph facts:
- LM-200 → related to → LM-300 Pro (source: getting-started.md#0)
- E-1042 → related to → AAA (source: error-codes.md#0)
```

```yaml
blueprint: graph
components:
  retriever:
    params: {hops: 1, extractor: auto, graph_weight: 1.0, include_facts: true, max_facts: 15}
  generator: {type: openai, params: {model: gpt-4o-mini}}   # enables LLM extraction
```

- **Extraction.** With a real LLM (`extractor: auto` or `llm`), each chunk is turned into
  typed entities and labelled relations ("requires", "is covered by") at ingest. That's one
  LLM call per chunk, so enable the `cache`. Offline, a heuristic picks up product codes,
  IDs, acronyms and multi-word names, and relates entities that share a sentence.
- **Access control.** Chunks are filtered by `allowed_groups` as usual, and a fact is
  included only if at least one chunk *visible to the caller* supports it.
- **Evaluation.** The facts context is marked `synthetic` and doesn't count in retrieval
  metrics.
- **Scope.** This is GraphRAG *local* search. Corpus-wide "what are the main themes?"
  questions (community summaries) are on the roadmap.

Set `requirements.multi_hop: true` and the selector recommends it.

## sql

```text
question + schema → LLM writes one SELECT → SQLite (read-only) → result rows → generator
```

For structured data, retrieval is the wrong tool: "how many LM-300s did we sell in
October?" needs `SUM`, not similar text. The `sql` blueprint lets the LLM write the query
and answers from the real result rows, citing the SQL.

```yaml
blueprint: sql
data: tables                      # CSV files: one table per file (loader type: csv)
components:
  retriever:
    params:
      # database: ./shop.sqlite                     # …or an existing SQLite file
      # database: ${SHOP_DATABASE_URL}              # …or postgresql://reader:…@host/db
      table_groups: {salaries: [hr]}
      max_rows: 50
  generator: {type: openai, params: {model: gpt-4o-mini}}   # needs a real LLM
```

**Safety is enforced by SQLite, not by the prompt:**

| Layer | What it stops |
|---|---|
| read-only connection (`mode=ro`, `query_only`) | any write, even if everything else failed |
| authorizer allow-list | anything but `SELECT`/reads/functions (`ATTACH`, `PRAGMA`, writes in CTEs), reads of hidden tables and of `sqlite_master` |
| statement check | multiple statements, non-`SELECT` statements |
| timeout and row cap | runaway queries, huge results |

**PostgreSQL** (`database: postgresql://…`, `pip install "psycopg[binary]"`): sessions
are `READ ONLY` with a `statement_timeout`; administrative, file and network functions
(`pg_*`, `lo_*`, `dblink`, `set_config`, …) are rejected; and before running a query,
`EXPLAIN (FORMAT JSON)` lists every table the plan reads, so a hidden table is
denied even through a view, CTE or subquery. Connect with a role that can only `SELECT`
the tables you expose (that's the real boundary), use `schemas: [...]` to pick
schemas, and list views over restricted tables in `table_groups` too, so they don't
appear in the schema the LLM sees.

A failed query is retried once with the error message. Denied access is never retried,
so a model can't probe for hidden tables.

**Access control is table-level.** `table_groups: {salaries: [hr]}` hides the table from
the schema the LLM sees *and* denies it in the authorizer, unless the caller is in `hr`
(`--groups hr`). There is no row-level security: split restricted rows into their own
table. `ragpick bench` skips `sql` unless you pass `-b sql`.

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
