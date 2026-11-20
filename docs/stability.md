# Stability and versioning

ragpick 1.x follows [semantic versioning](https://semver.org/). Within 1.x, upgrading a
minor or patch release never requires changes to code, configs or integrations that use
the **public surface** described below.

## What is public (covered by the guarantee)

| Area | Covered |
|---|---|
| Python API | everything in `ragpick.__all__` (`Pipeline`, `Router`, `load_config`, `load_engine`, `register`, `registry`, the models, …) and the documented methods of `Pipeline` and `Router` (`query`, `retrieve`, `stream`, `ingest`, `chat`, `chat_stream`, `record_feedback`, `from_yaml`, `from_dict`, `ready`, `for_tenant`) |
| Interfaces | the protocols in `ragpick.interfaces` (`Loader`, `Chunker`, `Embedder`, `VectorStore`, `KeywordIndex`, `Retriever`, `Reranker`, `Generator`, and the optional ones) |
| Data models | fields of `Document`, `Chunk`, `RetrievedChunk`, `Usage`, `Answer`, `Completion`, `StreamEvent` |
| Configuration | every key of `rag.yaml` and router configs ([config reference](config-reference.md)); `version: 1` configs keep loading |
| Components | built-in registry names (`type: hybrid`, `type: pgvector`, …) and their documented params; the `ragpick.components` entry-point group |
| Blueprints | names of stable blueprints |
| CLI | commands and their options |
| HTTP API | endpoints, request fields and response fields of `ragpick serve` |
| Evaluation | dataset format and metric names |

**Additions** (new methods, optional parameters, config keys, components, blueprints,
response fields, metrics) can arrive in any minor release.

## What is not public

- Modules and attributes starting with `_`, and helpers not listed above (e.g. functions
  inside `ragpick.components.*` besides the registered component classes).
- Prompt texts, offline heuristics and ranking details: these improve answers and
  may change in minor releases (changes are noted in the CHANGELOG).
- Log messages, trace span names and attributes (useful for debugging, not an API).
- The on-disk index layout under `index_dir`. Minor releases stay able to read indexes
  from earlier 1.x versions, or tell you to re-ingest.

## How changes are made

1. **Deprecate first.** A public feature that must change keeps working and emits a
   `RagpickDeprecationWarning` (a `DeprecationWarning`) naming the replacement and
   the major version that removes it. The CHANGELOG lists it under *Deprecated*.
2. **Remove only in a major release** (2.0), after at least one minor release with
   the warning.
3. **Security fixes** may change behaviour in a patch release when unavoidable, and are
   always called out in the CHANGELOG.

To surface deprecations in your test suite:

```python
import warnings
from ragpick import RagpickDeprecationWarning

warnings.simplefilter("error", RagpickDeprecationWarning)
```

## How it's enforced

`tests/api_snapshot.json` records the public surface: exports, protocol and engine
signatures, model fields, config keys, CLI options, HTTP endpoints, registry names
and blueprints. `tests/test_api_stability.py` fails CI whenever the surface changes. A
contributor making an *intentional* change regenerates the snapshot
(`python scripts/api_snapshot.py`) and documents it, so every API change is an explicit,
reviewed decision.
