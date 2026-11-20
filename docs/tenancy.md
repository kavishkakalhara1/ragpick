# Multi-tenancy

SaaS products must keep each customer's documents apart. With tenancy enabled, one
ragpick config (and one `ragpick serve` process) holds an **isolated index per
tenant**:

```yaml
blueprint: hybrid
tenancy:
  enabled: true
  max_loaded: 32          # tenant indexes kept in memory (least recently used are unloaded)
cache: {enabled: true}    # shared by all tenants
```

```bash
ragpick ingest ./customers/acme   --tenant acme
ragpick ingest ./customers/globex --tenant globex
ragpick query "What is our Gold plan price?" --tenant acme
```

```python
pipeline = Pipeline.from_yaml("rag.yaml")
pipeline.ingest("./customers/acme", tenant="acme")
answer = pipeline.query("What is our Gold plan price?", tenant="acme", groups=["sales"])
```

## How it works

- Each tenant's index lives in `<index_dir>/tenants/<tenant>/`, with its own vector store,
  BM25 index, knowledge graph and SQL database. Nothing is shared between tenants
  except the stateless parts: models and the embedding/answer cache.
- Tenant pipelines are created on first use and kept in an LRU of `max_loaded`.
- Tenant ids must be 1–64 letters, digits, `-` or `_` (starting with a letter or
  digit). Anything else (`../x`, `a/b`) is rejected before touching the filesystem.
- A multi-tenant pipeline refuses calls without a tenant, and a normal pipeline refuses
  calls with one, so a misconfiguration can't quietly fall back to the wrong index.
- Document-level (`allowed_groups`) access control still applies inside each tenant.

## Over HTTP

Bind API keys to tenants so a client can only ever reach its own data:

```yaml
serve:
  api_keys:
    - {key: "${ACME_KEY}", name: acme-portal, tenant: acme, groups: [support]}
    - {key: "${GLOBEX_KEY}", name: globex-portal, tenant: globex}
    - {key: "${ADMIN_KEY}", name: admin}        # no tenant: requests choose one
```

```bash
curl -s localhost:8000/query -H "Authorization: Bearer $ACME_KEY" \
  -d '{"question": "What is our Gold plan price?"}'
```

A tenant-bound key that sends a different `"tenant"` gets **403**. Keys without a tenant
must send `"tenant"` in the body (400 otherwise). Without API keys, the request's
`tenant` is trusted, which is only appropriate behind your own authenticated backend.

Routers pass `tenant` through to their routes (the route configs must enable tenancy);
ingest each route config per tenant.
