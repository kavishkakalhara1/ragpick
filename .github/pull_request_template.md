## What does this change?

<!-- What and why. Link the issue it closes, e.g. "Closes #12". -->

## Checklist

- [ ] Tests added or updated (they run offline: fake components, mocked clients)
- [ ] `pytest`, `ruff check .`, `ruff format --check .` and `mypy` pass
- [ ] Docs updated (and `python scripts/gen_config_reference.py` if config changed)
- [ ] `CHANGELOG.md` entry under *Unreleased*
- [ ] **Public API:** no change, or the snapshot is updated (`python scripts/api_snapshot.py`)
      and the change is backward compatible ([stability policy](../docs/stability.md))
- [ ] **Security:** if this touches access control, tenancy, the SQL sandbox, auth or
      sessions, the PR explains how the boundary is preserved and tests cover it
