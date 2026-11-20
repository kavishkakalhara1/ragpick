# Blueprints

Run `ragpick blueprints` to see this list in your terminal.

| Blueprint | Status | Best for |
|---|---|---|
| `naive` | stable | prototypes, small corpora, natural-language questions over prose |
| `hybrid` | stable | the default for commercial systems; keyword-heavy domains; medium and large corpora |
| `parent_child` | stable | long structured documents where answers need surrounding context |
| `sql` | stable | structured data and analytics questions (text-to-SQL) |
| `graph` | stable | multi-hop and "main themes" questions (GraphRAG local + global search) |
| `agentic` | stable | multi-part and comparison questions needing several lookups |
| `multimodal` | stable | images, diagrams, scans and charts inside documents |
| `conversational` | stable | chat assistants with follow-up questions |
| `hierarchical` | stable | RAPTOR-style summary trees: details and overviews |
| `long_context` | planned | small corpora with long-context models |
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
- **Global search.** After ingest, related entities are grouped into communities
  (label propagation), and each community gets a short summary: LLM-written, or a
  template offline. Unchanged communities keep their summaries, so re-ingesting doesn't
  re-pay for them. Questions like "what are the main themes?", "give an overview of …" or
  "summarise …" (`mode: auto`) are answered from the most relevant summaries instead of
  individual chunks. `mode: global` / `local` forces one or the other. A summary is
  shown only if the caller can see **every** chunk it was built from, so summaries never
  mix in restricted content.

Set `requirements.multi_hop: true` and the selector recommends it.

## agentic

```text
question → search ─┐
           plan ───┼→ search (sub-question / follow-up) ─┐
           plan ───┴→ … up to max_steps ──────────────────┴→ RRF over all searches → generator
```

"How do I reset my password **and** how much does Lumen Plus cost?" needs two lookups.
The agentic retriever searches for the question, then lets the LLM decide whether to run
another search (and with what query) or finish, up to `max_steps`. It stops early when
searches stop finding anything new. All result lists are fused with Reciprocal Rank
Fusion.

```yaml
blueprint: agentic
components:
  retriever:
    params: {base: hybrid, max_steps: 4, per_search_k: 4, planner: auto}
  generator: {type: openai, params: {model: gpt-4o-mini}}   # enables the LLM planner
```

Offline (or with `planner: decompose`), multi-part questions are split on "and", "vs",
"compared to", `;` and sentence breaks, and each part is searched. Every search goes
through the base retriever (`hybrid`, `vector`, `parent_child` or `graph`) with the caller's
filters, so access control holds. `ragpick query --trace` shows each `agent_plan` and
`agent_search` step. Set `requirements.complex_questions: true` and the selector
recommends it.

## multimodal

```text
ingest: text / Markdown / PDF ───────────────────────────────┐
        images + images inside PDFs → description ───────────┴→ chunk → hybrid index
        (vision LLM, or file name + alt text + caption offline)
query:  hybrid search over text and image descriptions → answer cites the image
```

Diagrams, screenshots, wiring charts and scanned pages often hold exactly the answer,
and plain text pipelines can't see them. `multimodal` turns every image into a
description at ingest:

- **With a vision model** (`generator: openai` with `gpt-4o-mini`, or `ollama` with a vision
  model like `llava`), each image is described, transcribing visible text, labels and
  numbers. Turn on `cache` so re-ingesting doesn't pay again: descriptions are keyed by
  image hash.
- **Offline**, the description comes from the file name, the alt text of every Markdown
  `![alt](image.png)` reference, and an optional caption file next to the image
  (`diagram.png.txt` or `diagram.txt`). A caption file may start with front matter
  (`allowed_groups: [hr]`) to restrict the image.
- Images inside PDFs are extracted and described too (`pdf_images: true`, needs
  `ragpick[pdf]`).

Image documents keep `modality: image`, `image_path` and `mime` in their metadata
(returned by `ragpick serve`), so your UI can show the picture next to the answer.

```yaml
blueprint: multimodal
components:
  loader: {params: {describe: auto, pdf_images: true, max_image_bytes: 20000000}}
  generator: {type: openai, params: {model: gpt-4o-mini}}
cache: {enabled: true}
```

Set `requirements.data_types` to include `diagrams`, `images`, `screenshots`, `scanned`, …
and the selector recommends it.

## conversational

`hybrid` retrieval plus chat sessions stored in SQLite. Each follow-up is rewritten into a
standalone question before retrieval. `chat()` works with every blueprint; this one just
sets good defaults. See [Conversations](conversations.md).

## hierarchical

```text
ingest: chunks → cluster by embedding → summarise each cluster → cluster the summaries → …
        (summary nodes are indexed next to the chunks)
query:  hybrid search over chunks + summaries ("collapsed tree")
```

Chunk search answers detail questions well and overview questions badly ("what does the
warranty cover overall?" needs several chunks at once). RAPTOR-style trees add summary
nodes at increasing levels of abstraction, and searching them together with the chunks
lets each question find its natural level.

```yaml
blueprint: hierarchical
components:
  retriever:
    params: {cluster_size: 6, max_levels: 3, summarizer: auto}
  generator: {type: openai, params: {model: gpt-4o-mini}}   # LLM-written summaries
cache: {enabled: true}
```

- Clustering is deterministic (each seed takes its nearest unassigned neighbours).
  Offline summaries are extractive (the first informative sentence of each member).
- **Access control:** a tree is built per access scope (public, and each distinct
  `allowed_groups` set), and summary nodes carry that scope's `allowed_groups`. A summary
  never mixes restricted and public text.
- Re-ingest rebuilds the tree; clusters whose members are unchanged reuse their summary.
- Summary nodes are `synthetic` (excluded from retrieval metrics) and record their
  `children` and `source_docs`.

Set `requirements.summary_questions: true` and the selector recommends it.

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
