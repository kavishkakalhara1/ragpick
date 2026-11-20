# Evaluation

Choosing a blueprint should be backed by numbers from **your** data. `ragpick eval` runs
one or more configs over a question set and prints a comparison table.

## Dataset format

JSON Lines, one question per line. Only `question` is required:

```json
{"question": "What does error E-1042 mean?", "ground_truth": "Low battery.", "relevant_doc_ids": ["error-codes.md"]}
{"question": "When do we escalate to Tier 2?", "relevant_doc_ids": ["internal-escalation.md"], "groups": ["support"]}
```

| Field | Purpose |
|---|---|
| `question` | The query. |
| `relevant_doc_ids` | Document ids (paths relative to the ingested directory) that answer it. Enables retrieval metrics. |
| `ground_truth` | Reference answer, stored in the results for review. |
| `groups` | Run the query as a member of these groups (to evaluate access-controlled content). |
| `id` | Optional identifier. |

Blank lines and lines starting with `#` or `//` are ignored. Errors name the file and line.

## Generating a dataset

No questions yet? Generate a starter set from your documents:

```bash
ragpick generate-eval ./docs -o eval.jsonl --count 30 -c rag.yaml
```

- With a real LLM generator, each sampled chunk becomes a natural question and a short
  reference answer. Offline (`--mode heuristic`), questions come from section headings
  ("How do I reset your password?"), codes ("What does E-1042 mean?") and titles.
- Chunks are sampled round-robin across documents (`--seed` makes it reproducible), using
  the config's loader and chunker, so `relevant_doc_ids` match what `ingest` produces.
- Questions from documents with `allowed_groups` get those `groups`, so evaluation runs
  with the right permissions.
- **Review the file.** Delete vague questions, fix answers, and add the hard cases your
  users really ask.

## Running

```bash
# compare two configs; each gets a temporary index built from --data
ragpick eval examples/eval.jsonl \
  --config examples/naive.yaml --config examples/hybrid.yaml \
  --data examples/data --output results.json
```

Without `--data`, each config uses its persisted index and builds it from the config's
`data` path if none exists. `--output` writes `.json` (all rows) or `.csv` (one row per config).

From Python:

```python
from ragpick import Pipeline
from ragpick.eval import evaluate, load_dataset

report = evaluate(Pipeline.from_yaml("rag.yaml"), load_dataset("eval.jsonl"))
print(report.metrics)
```

## Bench: try every blueprint

`ragpick bench` answers "which blueprint should I use?" with data. It runs every stable
blueprint on your documents and questions, using the same models and the same `top_k`,
and ranks them:

```bash
ragpick bench eval.jsonl --data ./docs --config rag.yaml --metric mrr --write best.yaml
```

- **Providers** (loader, embedder, store, generator, `eval` settings) come from the base
  `--config` (default: `./rag.yaml` if present, otherwise offline fake components).
- **Architecture** (chunker, keyword index, retriever, reranker) comes from each blueprint.
- `--metric` picks the winner: `hit_rate@k`, `mrr` (default), `precision@k`, or, with a
  judge, `faithfulness` / `answer_relevance`. Ties go to the lower median latency.
- `-b NAME` (repeatable) limits the candidates; `--top-k` overrides k for all of them.
  By default only document blueprints are compared; `sql` answers from tables and
  needs a real LLM, so include it explicitly (`-b sql`) with a table dataset.
- `--write best.yaml` saves the winner's config: your providers plus `blueprint: <winner>`.
  Secrets stay as `${ENV_VAR}` references.
- Each candidate gets a temporary index; your real indexes are never touched.

```python
from ragpick.eval.bench import run_bench

result = run_bench("eval.jsonl", "./docs", base_config="rag.yaml", metric="mrr")
print(result.winner, result.ranking)
```

## Metrics

**Retrieval** (no LLM needed). These are computed over the ranked contexts mapped to their
`doc_id`, with `k = query.top_k`:

| Metric | Definition |
|---|---|
| `hit_rate@k` | 1 if any of the top-k contexts comes from a relevant document. |
| `mrr` | 1 / rank of the first relevant context (0 if none). |
| `precision@k` | relevant contexts in the top k, divided by k. |

**LLM-judged.** These need a judge (`eval.judge` in the config or `--judge openai:gpt-4o-mini`):

| Metric | Question asked of the judge |
|---|---|
| `faithfulness` | Is every claim in the answer supported by the retrieved contexts? |
| `answer_relevance` | Does the answer address the question directly? |

Each score is 0–1 with a short reason per row (in the JSON output). All judge prompts live
in `ragpick/eval/prompts.py`.

**Operational:**

| Metric | Definition |
|---|---|
| `latency_p50_ms` / `latency_p95_ms` | end-to-end query latency percentiles |
| `total_tokens` | prompt + completion tokens of the generator |
| `est_cost_usd` | from the price table (USD per 1M tokens); `n/a` for unknown models |
| `errors` | rows where the query or judge failed (recorded, not fatal) |

Override or extend prices in the config:

```yaml
eval:
  judge: {type: openai, params: {model: gpt-4o-mini}}
  prices:
    my-finetuned-model: {prompt_per_1m: 0.30, completion_per_1m: 1.20}
```

## Gating CI on quality

Fail the build when retrieval quality drops or answers get slow or expensive:

```bash
ragpick eval eval.jsonl -c rag.yaml \
  --fail-under hit_rate@k=0.85 --fail-under mrr=0.6 \
  --fail-over latency_p95_ms=2000 --fail-over est_cost_usd=0.50
```

or keep the thresholds next to the config:

```yaml
eval:
  fail_under: {hit_rate@k: 0.85, faithfulness: 0.8}
  fail_over: {latency_p95_ms: 2000}
```

Misses are printed and the command exits with status **1**; CLI flags override config
values. A required metric that isn't available (e.g. `faithfulness` without a judge) also
fails, so a misconfigured job can't pass silently. In GitHub Actions:

```yaml
- run: pip install ragpick
- run: ragpick eval eval.jsonl -c rag.yaml --data docs --fail-under hit_rate@k=0.85
```

## Tips

- If `cache` is enabled, repeated runs reuse answers: costs drop to zero and latencies to
  near zero. Disable the cache when measuring latency.

- 30–100 representative questions catch most regressions. Include the exact-term queries
  your users really send (codes, IDs), plus some that should return "I don't know".
- Add eval to CI: run `ragpick eval ... --output results.json` and fail the build if
  `hit_rate@k` drops.
- The `fake` generator and judge are for plumbing tests. Their answer metrics aren't
  meaningful, but retrieval metrics are.
