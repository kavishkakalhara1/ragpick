# Contributing to ragpick

Thanks for helping! This guide gets you from clone to green tests in a couple of minutes.

## Development setup

You need [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone https://github.com/kavishkakalhara1/ragpick && cd ragpick
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

pytest                          # fully offline, no API keys
ruff check . && ruff format --check .
mypy                            # strict mode on src/
```

Before opening a pull request, run all three and make sure coverage stays at or above 85%:

```bash
pytest --cov=ragpick --cov-fail-under=85
```

If you change `src/ragpick/config.py`, regenerate the config reference:

```bash
python scripts/gen_config_reference.py
```

Preview the docs with `mkdocs serve`.

## Ground rules

- **Tests must run offline.** Use `FakeEmbedder`/`FakeGenerator`, and mock remote clients
  (inject a fake `client=`, or use `httpx.MockTransport`).
- **Full type hints**: mypy strict must pass.
- **Docstrings** on every public class and function.
- **No `print()` in library code**; use `logging.getLogger(__name__)`.
- **No new dependencies** in core without discussion. Provider SDKs go into an extra and
  are imported lazily with `ragpick.components._optional.require(...)`, so users get a
  friendly "install ragpick[extra]" message.
- Record non-obvious choices in `docs/decisions.md`.

## Adding a component

1. Implement the protocol from `ragpick/interfaces.py` in the matching module under
   `src/ragpick/components/` (or in your own package, see [docs/plugins.md](docs/plugins.md)).
2. Register it: `@register("store", "pgvector")`.
3. Stores and keyword indexes **must** support `filters`, including the
   `allowed_groups` access-control rules (reuse `ragpick.filters.matches_filters`), and set
   `supports_filters = True`.
4. Add tests: behaviour, filtering and ACL (add your store to the parametrized ACL tests
   in `tests/test_stores_keyword.py`), persistence round-trip, and a missing-extra error.
5. Document it in `docs/concepts.md`.

## Adding a blueprint

1. Add `src/ragpick/blueprints/<name>.yaml` with a `metadata` block and `defaults`
   (see `hybrid.yaml`). Use `status: planned` for roadmap entries without an implementation.
2. Add it to `_ORDER` in `blueprints/__init__.py` if it should be listed in a specific position.
3. Add an end-to-end test in `tests/test_pipeline.py` and, if the selector should recommend
   it, a `Rule` in `src/ragpick/selector/rules.py` with a test.
4. Document it in `docs/blueprints.md` and the README table.

## Good first issues

- **Loaders**: HTML, DOCX, Notion/Confluence exports, CSV-as-documents.
- **Stores**: pgvector, Chroma, LanceDB, Elasticsearch/OpenSearch (with filter push-down).
- **Blueprints**: `graph` and `agentic` are next on the roadmap; routing between `sql`
  and document pipelines is a good intermediate step.
- **SQL**: Postgres/MySQL backends for the `sql` retriever (read-only roles + the same
  validation).
- **Trace exporters**: OpenTelemetry, Langfuse, Phoenix.
- **Embedders and generators**: Cohere, Voyage, Bedrock, Azure OpenAI (or use them today
  through the LangChain / LlamaIndex adapters).
- **Eval**: more metrics (context recall vs `ground_truth`, citation accuracy).

## Releasing (maintainers)

1. Bump `__version__` in `src/ragpick/__init__.py` and update `CHANGELOG.md`.
2. Tag and push: `git tag v0.1.1 && git push --tags`. The release workflow publishes to
   TestPyPI and then PyPI with trusted publishing.

By contributing you agree that your contributions are licensed under Apache-2.0 and that
you will follow our [Code of Conduct](CODE_OF_CONDUCT.md).
