# Answer feedback

Evaluation datasets go stale; your users' thumbs-up and thumbs-down don't. ragpick records
ratings next to the index and turns them into evaluation data.

## Collecting ratings

Every `Answer` has an `id` and remembers its `question`.

```python
answer = pipeline.query("How do I reset my password?", groups=["support"])
pipeline.record_feedback(answer, "up", comment="Exactly what I needed", groups=["support"])
pipeline.record_feedback(other_answer, 2)  # 1-5 stars, True/False, "up"/"down"
```

Over HTTP, enable it and send the answer's `id` back:

```yaml
feedback:
  enabled: true
  answer_log_size: 10000     # recent answers the server remembers for rating
```

```bash
curl -s localhost:8000/feedback -H "Authorization: Bearer $KEY" \
  -d '{"answer_id": "9b2f...", "rating": "down", "comment": "Outdated price"}'
```

The server only accepts ratings for answers **it gave to the same API key and tenant**:
the question, answer, citations and groups are taken from its own records, not from the
request. Unknown, expired and other clients' answers all return 404, so ratings can't be
forged or attributed to someone else. Answers from `/query`, `/chat` and streamed
responses can all be rated.

Ratings are normalised to 0–1: up = 1, down = 0, stars map 1→0 … 5→1. They're stored in
`<index_dir>/feedback.sqlite` (routers: `sessions_dir`).

## Using them

```bash
ragpick feedback stats                       # totals, per-route averages, worst-rated questions
ragpick feedback export -o feedback.jsonl    # every record
ragpick feedback export --as-eval -o regressions.jsonl
```

`--as-eval` writes a valid [evaluation dataset](evaluation.md):

- **liked answers** (≥ 0.75) become regression tests: the question, the answer as
  `ground_truth`, and the documents it cited as `relevant_doc_ids`, plus the caller's
  `groups`;
- **disliked answers** (≤ 0.25) become questions to review (no reference yet): fix the
  content or pipeline, write the expected answer, and keep them as tests.

Run them in CI next to your other questions:

```bash
ragpick eval regressions.jsonl -c rag.yaml --fail-under hit_rate@k=0.9
```
