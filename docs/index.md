# ragpick

**Choose, build, evaluate, and run the right RAG architecture.**

Most RAG projects don't fail because of the LLM. They fail because the *architecture* was
picked by guesswork: vector search where exact product codes matter, no access control
for internal documents, no evidence that the "improved" pipeline is actually better.

ragpick turns that choice into an engineering process:

1. **Describe your requirements** (corpus size, keyword-heavy queries, latency, access
   control) and `ragpick recommend` explains which **blueprint** fits and why.
2. **Define the pipeline as YAML**. Architecture as code, validated with readable errors.
3. **Evaluate** competing blueprints on *your* data with `ragpick eval` and pick the
   winner from numbers, not vibes.

ragpick sits next to LangChain and LlamaIndex rather than replacing them. It owns the
interfaces, config, selector and evaluation; other libraries plug in as components.

## Install

```bash
pip install ragpick              # core, offline components
pip install "ragpick[openai]"    # + OpenAI embeddings and chat models
pip install "ragpick[all]"       # + sentence-transformers, Qdrant, PDF
```

## Next steps

- [Concepts](concepts.md): components, blueprints, the selector, access control.
- [Blueprints](blueprints.md): what each architecture is good at.
- [Config reference](config-reference.md): every `rag.yaml` key.
- [Writing a plugin](plugins.md): add your own loader, store, or model.
- [Evaluation](evaluation.md): metrics and how to compare configs.
- [Design decisions](decisions.md): the choices behind v0.1.
