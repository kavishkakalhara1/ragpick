# Writing a plugin

Any class that satisfies a protocol from `ragpick.interfaces` can be a component. Register
it and refer to it by name in `rag.yaml`.

## 1. Implement the protocol

```python
# ragpick_cohere/__init__.py
import cohere


class CohereEmbedder:
    """Cohere embeddings for ragpick."""

    def __init__(self, model: str = "embed-english-v3.0", api_key: str | None = None) -> None:
        self.model = model
        self._client = cohere.Client(api_key)

    @property
    def dimension(self) -> int:
        return 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embed(texts=texts, model=self.model, input_type="search_document")
        return response.embeddings
```

Constructor keyword arguments become the component's `params`. ragpick validates them
against the signature, so a typo in `rag.yaml` fails with the list of accepted params.

## 2. Register it

### In your own code (no packaging needed)

```python
from ragpick import register


@register("embedder", "cohere")
class CohereEmbedder: ...
```

Import the module before building the pipeline.

### As an installable package (entry points)

```toml
# pyproject.toml of your plugin package
[project.entry-points."ragpick.components"]
"embedder.cohere" = "ragpick_cohere:CohereEmbedder"
```

The entry-point name is `<kind>.<name>`. Once the package is installed, `type: cohere` just
works. An entry point whose name has no dot is simply imported, so a module can register
several components with `@register`. Plugins can't shadow built-in names, and a broken
plugin is logged and skipped.

## 3. Use it

```yaml
blueprint: hybrid
components:
  embedder: {type: cohere, params: {model: embed-english-v3.0}}
```

## Dependency injection

Components that need other components declare them as constructor parameters named
`embedder`, `store` or `keyword_index`. The pipeline passes them in:

```python
@register("retriever", "mmr")
class MMRRetriever:
    def __init__(self, embedder, store, k: int = 10, diversity: float = 0.3) -> None: ...

    def retrieve(self, query, k, filters=None): ...
```

## Rules for stores and indexes

- **Honour `filters`**, including the `allowed_groups` access-control semantics. Reuse
  `ragpick.filters.matches_filters`, or translate filters natively as `QdrantStore` does.
- Set `supports_filters = True` on the class so `ragpick recommend` doesn't warn users
  who need access control.
- `add` must upsert by chunk id, and `delete(doc_ids)` must remove every chunk of those documents.

## Testing your plugin

Use the global registry and clean up afterwards, or build an isolated `Registry()`:

```python
from ragpick import Pipeline, registry


def test_my_embedder(tmp_path):
    registry.add("embedder", "mine", MyEmbedder)
    try:
        p = Pipeline.from_dict(
            {
                "blueprint": "naive",
                "index_dir": str(tmp_path),
                "components": {"embedder": {"type": "mine"}},
            }
        )
        p.ingest("tests/data")
        assert p.query("hello").contexts
    finally:
        registry.remove("embedder", "mine")
```
