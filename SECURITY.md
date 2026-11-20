# Security policy

ragpick enforces security boundaries on behalf of its users: document-level access control,
tenant isolation, the read-only SQL sandbox, API-key authentication and session/feedback
ownership. We take reports about these seriously.

## Supported versions

| Version | Supported |
|---|---|
| 1.x (latest minor) | ✅ security fixes |
| 0.x | ❌ please upgrade ([upgrade guide](docs/upgrading.md)) |

## Reporting a vulnerability

**Please don't open a public issue for security problems.**

Report privately through GitHub:
**[Security → Report a vulnerability](https://github.com/kavishkakalhara1/ragpick/security/advisories/new)**.

If you can't use GitHub's private reporting, contact the maintainer, Kavishka Kalhara, via
[kavishka-kalhara.me](https://kavishka-kalhara.me) and we'll set up a private channel.

Please include:

- the ragpick version, Python version, and the blueprint/components involved;
- a minimal config and steps (or a test) that reproduce the issue;
- what an attacker can do, and what access they need (e.g. an API key bound to another
  group or tenant, control over documents, control over model output).

## What to expect

- Acknowledgement within **3 business days**, and an initial assessment within **7 days**.
- A fix in a patch release as soon as possible (usually within 30 days for confirmed
  issues), with a GitHub security advisory and a CVE where appropriate.
- Credit in the advisory and CHANGELOG, unless you prefer to stay anonymous.

Please give us a reasonable time to release a fix before any public disclosure.

## In scope

Reports are especially welcome for:

- **Access-control bypass:** retrieving, citing, summarising or leaking content from
  documents with `allowed_groups` the caller doesn't belong to, in any store, retriever,
  graph fact, community or RAPTOR summary, or cache.
- **Tenant isolation:** reading or writing another tenant's index, or tenant ids that escape
  the index directory.
- **SQL sandbox escape:** writes, reading hidden tables (`table_groups`), reading system
  catalogs, file/network access, or bypassing the timeout, with SQLite or PostgreSQL.
- **HTTP API:** authentication bypass, an API key acting outside its groups or tenant, reading
  or continuing another key's chat session, rating another key's answers.
- **Secret leakage:** API keys or `${ENV}` secrets written to disk, logs, traces, error
  responses or `ragpick validate` output.
- **Injection** through filters into SQL (pgvector, LanceDB) or store queries.

## Out of scope

- Prompt injection that only changes the wording of an answer **within** the documents the
  caller is allowed to see (LLM behaviour, not a ragpick boundary).
- Loading untrusted config files: `class_path` adapters and plugins import Python code by
  design. Treat `rag.yaml` like code.
- Running `ragpick serve` on a public interface **without** `serve.api_keys` (it warns; the
  request's `groups` and `tenant` are trusted in that mode by design).
- Vulnerabilities in third-party packages (report them upstream; tell us if ragpick should
  pin or work around them).

## Hardening guidance

- Configure `serve.api_keys` with `groups` and `tenant` for every client, and keep keys in
  environment variables.
- Connect the `sql` blueprint with a **read-only database role** limited to the tables you
  expose; ragpick's sandbox is defence in depth, not a replacement.
- Put views over restricted tables in `table_groups` too.
- Enable `--fail-under` evaluation gates with questions that must *not* be answerable for
  a given group, so access-control regressions fail CI.
