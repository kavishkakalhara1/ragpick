# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-20

### Added

- `parent_child` blueprint (now stable): `ParentChildChunker` indexes small child chunks
  and keeps their parent section in metadata; `ParentChildRetriever` searches children
  (hybrid or vector) and returns deduplicated parent sections.
- `ragpick bench`: runs every stable blueprint on your data with the same models and
  `top_k`, ranks them by a metric (`--metric`), and can write the winner's config
  (`--write rag.yaml`). Python API: `ragpick.eval.bench.run_bench`.
- LangChain and LlamaIndex adapters (`type: langchain` / `type: llamaindex`) for loaders,
  embedders, retrievers and generators, configured by `class_path` + `kwargs` or built
  from Python objects. No new dependencies.
- `requirements.long_documents` and selector rules recommending `parent_child`.
- Optional `embed_query` on embedders, used for searches (asymmetric embedding models).
- Docs: adapters guide, bench section, docs deployment workflow (GitHub Pages).

### Changed

- `evaluate_configs` accepts `RagConfig` objects as well as file paths.
- CI and release workflows use Node 24 action versions; publishing skips files that
  already exist, so a failed release can be re-run.

## [0.1.0] - 2026-11-20

### Added

- Core models (`Document`, `Chunk`, `RetrievedChunk`, `Answer`, `Usage`) and runtime-checkable
  component protocols.
- Component registry with the `@register` decorator and `ragpick.components` entry-point plugins.
- `rag.yaml` schema with blueprint → user → CLI merge order, key-path validation errors and
  `${ENV_VAR}` interpolation.
- Components: directory/text/PDF loaders (Markdown front matter → metadata), fixed and
  recursive chunkers, fake/OpenAI/sentence-transformers embedders, in-memory (numpy) and
  Qdrant stores, pure-Python BM25, vector/keyword/hybrid (RRF) retrievers, no-op and
  cross-encoder rerankers, fake/OpenAI/Ollama generators with citation parsing.
- Document-level access control (`allowed_groups`) in every store and in BM25.
- Blueprints: `naive` and `hybrid` (stable); `parent_child`, `sql`, `graph`, `agentic`,
  `corrective` (planned).
- Rule-based, explainable selector (`ragpick recommend`).
- Evaluation harness: hit rate@k, MRR, precision@k, LLM-judged faithfulness and answer
  relevance, latency percentiles, tokens and estimated cost; JSON/CSV output.
- CLI: `init`, `recommend`, `blueprints`, `ingest`, `query`, `eval`, `validate`.

[Unreleased]: https://github.com/kavishkakalhara1/ragpick/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/kavishkakalhara1/ragpick/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/kavishkakalhara1/ragpick/releases/tag/v0.1.0
