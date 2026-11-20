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

## Tips

- 30–100 representative questions catch most regressions. Include the exact-term queries
  your users really send (codes, IDs), plus some that should return "I don't know".
- Add eval to CI: run `ragpick eval ... --output results.json` and fail the build if
  `hit_rate@k` drops.
- The `fake` generator and judge are for plumbing tests. Their answer metrics aren't
  meaningful, but retrieval metrics are.
