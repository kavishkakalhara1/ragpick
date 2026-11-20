"""LlamaIndex adapters (``type: llamaindex``).

LlamaIndex objects are used through their public methods only, so ragpick does not
depend on LlamaIndex; install ``llama-index`` packages yourself.

YAML example::

    components:
      loader:
        type: llamaindex          # defaults to llama_index.core.SimpleDirectoryReader
      embedder:
        type: llamaindex
        params:
          class_path: llama_index.embeddings.openai.OpenAIEmbedding
          kwargs: {model: text-embedding-3-small}
      generator:
        type: llamaindex
        params:
          class_path: llama_index.llms.openai.OpenAI
          kwargs: {model: gpt-4o-mini, temperature: 0}
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from ragpick.adapters._common import (
    documents_with_ids,
    import_object,
    require_method,
    resolve,
    to_retrieved,
)
from ragpick.components.generators import LLMGenerator, estimate_tokens
from ragpick.errors import ProviderError
from ragpick.models import Completion, Document, RetrievedChunk, Usage
from ragpick.registry import register

DEFAULT_READER = "llama_index.core.SimpleDirectoryReader"


@register("embedder", "llamaindex")
class LlamaIndexEmbedder:
    """Wraps a LlamaIndex embedding model.

    Uses ``get_text_embedding_batch`` for documents and ``get_query_embedding`` for queries.
    """

    def __init__(
        self,
        class_path: str | None = None,
        kwargs: dict[str, Any] | None = None,
        dimension: int | None = None,
        embed_model: Any | None = None,
    ) -> None:
        self._obj = resolve(embed_model, class_path, kwargs, "LlamaIndexEmbedder")
        require_method(self._obj, "get_text_embedding_batch", "LlamaIndexEmbedder")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Vector length (given, or probed once)."""
        if self._dimension is None:
            self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed documents."""
        return [[float(x) for x in v] for v in self._obj.get_text_embedding_batch(texts)]

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query."""
        method = getattr(self._obj, "get_query_embedding", None)
        vector = method(query) if callable(method) else self.embed([query])[0]
        return [float(x) for x in vector]


@register("loader", "llamaindex")
class LlamaIndexLoader:
    """Wraps a LlamaIndex reader (default: ``SimpleDirectoryReader``).

    The reader is created per path as ``class_path(**{path_arg: path}, **kwargs)`` and
    read with ``load_data()``. With the default ``path_arg="input_dir"``, a single
    file is passed as ``input_files=[path]`` instead.
    """

    def __init__(
        self,
        class_path: str = DEFAULT_READER,
        kwargs: dict[str, Any] | None = None,
        path_arg: str = "input_dir",
        reader_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.kwargs = dict(kwargs or {})
        self.path_arg = path_arg
        if reader_factory is None:
            cls = import_object(class_path)

            def reader_factory(path: str) -> Any:
                if self.path_arg == "input_dir" and Path(path).is_file():
                    return cls(input_files=[path], **self.kwargs)
                return cls(**{self.path_arg: path}, **self.kwargs)

        self._factory = reader_factory

    def load(self, path: str) -> list[Document]:
        """Load documents from ``path``."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Path not found: {path}")
        items = []
        for d in self._factory(path).load_data():
            text = getattr(d, "text", None)
            if text is None:
                text = d.get_content()
            items.append((str(text), dict(getattr(d, "metadata", {}) or {})))
        return documents_with_ids(items, path)


@register("retriever", "llamaindex")
class LlamaIndexRetriever:
    """Wraps a LlamaIndex retriever (``retrieve(query) -> list[NodeWithScore]``).

    ragpick filters and ``allowed_groups`` access control are applied to the returned
    nodes (post-filtering).
    """

    def __init__(
        self,
        class_path: str | None = None,
        kwargs: dict[str, Any] | None = None,
        k: int = 10,
        retriever: Any | None = None,
    ) -> None:
        self._obj = resolve(retriever, class_path, kwargs, "LlamaIndexRetriever")
        require_method(self._obj, "retrieve", "LlamaIndexRetriever")
        self.k = k

    def retrieve(
        self, query: str, k: int | None = None, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Run the LlamaIndex retriever and convert its nodes."""
        hits = []
        for item in self._obj.retrieve(query):
            node = getattr(item, "node", item)
            meta = dict(getattr(node, "metadata", {}) or {})
            # prefer file paths (what eval datasets reference) over LlamaIndex's internal UUIDs
            meta["source"] = (
                meta.get("source")
                or meta.get("file_path")
                or meta.get("file_name")
                or getattr(node, "ref_doc_id", None)
            )
            text = node.get_content() if callable(getattr(node, "get_content", None)) else ""
            hits.append(
                (str(text), meta, getattr(node, "node_id", None), getattr(item, "score", None))
            )
        return to_retrieved(hits, k or self.k, filters)


@register("generator", "llamaindex")
class LlamaIndexGenerator(LLMGenerator):
    """Wraps a LlamaIndex LLM (``complete(prompt) -> CompletionResponse``).

    The system and user prompts are joined into one prompt. Also works as a judge.
    """

    def __init__(
        self,
        class_path: str | None = None,
        kwargs: dict[str, Any] | None = None,
        llm: Any | None = None,
    ) -> None:
        self._obj = resolve(llm, class_path, kwargs, "LlamaIndexGenerator")
        require_method(self._obj, "complete", "LlamaIndexGenerator")
        self.model = str((kwargs or {}).get("model") or getattr(self._obj, "model", "") or "")

    def complete(self, system: str, user: str) -> Completion:
        """Call ``llm.complete`` with the combined prompt."""
        prompt = f"{system}\n\n{user}"
        start = time.perf_counter()
        try:
            response = self._obj.complete(prompt)
        except Exception as exc:
            raise ProviderError(f"LlamaIndex LLM call failed: {exc}") from exc
        text = str(getattr(response, "text", response))
        usage = Usage(
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(text),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        return Completion(text=text, usage=usage)

    def stream_complete(self, system: str, user: str) -> Iterator[str | Usage]:
        """Stream with ``llm.stream_complete`` (falls back to ``complete``)."""
        if not callable(getattr(self._obj, "stream_complete", None)):
            yield from super().stream_complete(system, user)
            return
        prompt = f"{system}\n\n{user}"
        start = time.perf_counter()
        parts: list[str] = []
        try:
            for item in self._obj.stream_complete(prompt):
                piece = getattr(item, "delta", None)
                if piece:
                    parts.append(str(piece))
                    yield str(piece)
        except Exception as exc:
            raise ProviderError(f"LlamaIndex LLM call failed: {exc}") from exc
        yield Usage(
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens("".join(parts)),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
