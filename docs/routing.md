# Routing

Real assistants answer from more than one kind of source: manuals *and* an orders
database, public FAQs *and* internal runbooks. A **router** sends each question to the
pipeline best suited to answer it. Each route is an ordinary `rag.yaml`, so every
blueprint, evaluation and access-control feature keeps working.

```yaml
# router.yaml
version: 1
name: lumen-assistant
router:
  strategy: auto                # llm | keywords | auto
  default: docs                 # used when nothing matches, and for fallback
  fallback: true                # retry on `default` if the chosen route can't answer
  routes:
    - name: docs
      config: hybrid.yaml
      description: Product manuals, troubleshooting, error codes, warranty and policies
    - name: analytics
      config: sql.yaml
      description: Orders, sales, prices, quantities, counts and totals
      keywords: [how many, total, revenue, average, top, orders, sold]
      groups: [staff]           # only staff are ever routed here
```

Use it like any config:

```bash
ragpick validate -c router.yaml          # checks the router and every route config
ragpick ingest -c router.yaml            # ingests each route from its own `data`
ragpick query "How many LM-300 units did we sell?" -c router.yaml --groups staff --trace
ragpick eval eval.jsonl -c router.yaml   # per-question route is in the JSON output
ragpick serve -c router.yaml
```

```python
from ragpick import load_engine

engine = load_engine("router.yaml")  # a Router (or a Pipeline for plain configs)
answer = engine.query("How many LM-300 units did we sell?", groups=["staff"])
print(answer.route, answer.text)
```

## Strategies

| Strategy | How it decides |
|---|---|
| `llm` | One LLM call with the route names and descriptions; the reply is `{"route": "<name>"}`. Uses `router.llm` if set, otherwise the first route with a real LLM generator. An invalid reply falls back to keywords. |
| `keywords` | Offline and deterministic: each keyword/phrase found in the question scores 2, each question word that appears in the description scores 1. The highest score wins; with no match, the `default` route is used. |
| `auto` | `llm` when an LLM is available, else `keywords`. |

Good descriptions matter more than anything else: say what each source *contains*.

## Fallback

If the chosen route answers "I don't know", returns no contexts, or its SQL failed, and
`fallback` is on, the question is re-asked on the `default` route. If that answers, its
answer is returned (with `answer.route` set to the default). The trace shows a
`fallback` span.

## Access control

- `groups` on a route restricts who can be routed there. Callers outside those groups
  are never sent to it, so the router can't leak a route's existence or data.
- The caller's groups are passed through to the chosen pipeline, so document-level
  (`allowed_groups`) and SQL table-level (`table_groups`) rules still apply.
