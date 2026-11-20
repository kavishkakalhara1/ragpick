"""LangChain adapters (``type: langchain``).

LangChain objects are used through their public methods only, so ragpick does not
depend on LangChain; install the LangChain packages you want to use yourself.

YAML example::

    components:
      embedder:
        type: langchain
        params:
          class_path: langchain_openai.OpenAIEmbeddings
          kwargs: {model: text-embedding-3-small}
      generator:
        type: langchain
        params:
          class_path: langchain_openai.ChatOpenAI
          kwargs: {model: gpt-4o-mini, temperature: 0}

From Python, pass ready-made objects instead: ``LangChainEmbedder(embeddings=obj)``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from ragpick.adapters._common import documents_with_ids, require_method, resolve, to_retrieved
from ragpick.components.generators import LLMGenerator, estimate_tokens
from ragpick.errors import ProviderError
from ragpick.models import Completion, Document, RetrievedChunk, Usage
from ragpick.registry import register


@register("embedder", "langchain")
class LangChainEmbedder:
    """Wraps a LangChain ``Embeddings`` (``embed_documents`` / ``embed_query``)."""

    def __init__(
        self,
        class_path: str | None = None,
        kwargs: dict[str, Any] | None = None,
        dimension: int | None = None,
        embeddings: Any | None = None,
    ) -> None:
        self._obj = resolve(embeddings, class_path, kwargs, "LangChainEmbedder")
        require_method(self._obj, "embed_documents", "LangChainEmbedder")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Vector length (given, or probed once with a query embedding)."""
        if self._dimension is None:
            self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed documents with ``embed_documents``."""
        return [[float(x) for x in v] for v in self._obj.embed_documents(texts)]

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query with ``embed_query`` (falls back to ``embed_documents``)."""
        method = getattr(self._obj, "embed_query", None)
        vector = method(query) if callable(method) else self._obj.embed_documents([query])[0]
        return [float(x) for x in vector]


@register("loader", "langchain")
class LangChainLoader:
    """Wraps a LangChain document loader class.

    The loader is created per path as ``class_path(**{path_arg: path}, **kwargs)`` and
    read with ``lazy_load()`` (or ``load()``). Document ids come from the ``source``
    metadata relative to the ingested path.
    """

    def __init__(
        self,
        class_path: str | None = None,
        kwargs: dict[str, Any] | None = None,
        path_arg: str = "file_path",
        loader_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.kwargs = dict(kwargs or {})
        self.path_arg = path_arg
        if loader_factory is None:
            if not class_path:
                resolve(None, None, None, "LangChainLoader")  # raises a helpful error
            from ragpick.adapters._common import import_object

            cls = import_object(str(class_path))

            def loader_factory(path: str) -> Any:
                return cls(**{self.path_arg: path}, **self.kwargs)

        self._factory = loader_factory

    def load(self, path: str) -> list[Document]:
        """Load documents from ``path``."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Path not found: {path}")
        loader = self._factory(path)
        lazy = getattr(loader, "lazy_load", None)
        items = list(lazy()) if callable(lazy) else list(loader.load())
        return documents_with_ids(
            [(str(d.page_content), dict(getattr(d, "metadata", {}) or {})) for d in items], path
        )


@register("retriever", "langchain")
class LangChainRetriever:
    """Wraps a LangChain retriever (anything with ``invoke(query) -> list[Document]``).

    Use this when the index is managed outside ragpick. ragpick's filters and
    ``allowed_groups`` access control are applied to the returned documents
    (post-filtering), so configure the retriever to return enough candidates.
    """

    def __init__(
        self,
        class_path: str | None = None,
        kwargs: dict[str, Any] | None = None,
        k: int = 10,
        retriever: Any | None = None,
    ) -> None:
        self._obj = resolve(retriever, class_path, kwargs, "LangChainRetriever")
        require_method(self._obj, "invoke", "LangChainRetriever")
        self.k = k

    def retrieve(
        self, query: str, k: int | None = None, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Run the LangChain retriever and convert its documents."""
        docs = self._obj.invoke(query)
        hits = []
        for d in docs:
            meta = dict(getattr(d, "metadata", {}) or {})
            score = meta.get("score", meta.get("relevance_score"))
            hits.append((str(d.page_content), meta, getattr(d, "id", None), score))
        return to_retrieved(hits, k or self.k, filters)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content blocks
        parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
        return "".join(parts)
    return str(content)


@register("generator", "langchain")
class LangChainGenerator(LLMGenerator):
    """Wraps a LangChain chat model (``invoke(messages) -> AIMessage``).

    Uses ragpick's grounded prompt and citation parsing, and also works as an
    evaluation judge.
    """

    def __init__(
        self,
        class_path: str | None = None,
        kwargs: dict[str, Any] | None = None,
        chat_model: Any | None = None,
    ) -> None:
        self._obj = resolve(chat_model, class_path, kwargs, "LangChainGenerator")
        require_method(self._obj, "invoke", "LangChainGenerator")
        self.model = str((kwargs or {}).get("model") or getattr(self._obj, "model_name", "") or "")

    def complete(self, system: str, user: str) -> Completion:
        """Invoke the chat model with a system and a human message."""
        start = time.perf_counter()
        try:
            message = self._obj.invoke([("system", system), ("human", user)])
        except Exception as exc:  # provider SDKs raise many types
            raise ProviderError(f"LangChain model call failed: {exc}") from exc
        text = _message_text(getattr(message, "content", message))
        usage_meta = getattr(message, "usage_metadata", None) or {}
        usage = Usage(
            prompt_tokens=int(usage_meta.get("input_tokens") or estimate_tokens(system + user)),
            completion_tokens=int(usage_meta.get("output_tokens") or estimate_tokens(text)),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        return Completion(text=text, usage=usage)

    def stream_complete(self, system: str, user: str) -> Iterator[str | Usage]:
        """Stream with the chat model's ``stream()`` (falls back to ``invoke``)."""
        if not callable(getattr(self._obj, "stream", None)):
            yield from super().stream_complete(system, user)
            return
        start = time.perf_counter()
        usage_meta: dict[str, Any] = {}
        parts: list[str] = []
        try:
            for chunk in self._obj.stream([("system", system), ("human", user)]):
                piece = _message_text(getattr(chunk, "content", chunk))
                if piece:
                    parts.append(piece)
                    yield piece
                usage_meta = getattr(chunk, "usage_metadata", None) or usage_meta
        except Exception as exc:  # provider SDKs raise many types
            raise ProviderError(f"LangChain model call failed: {exc}") from exc
        text = "".join(parts)
        yield Usage(
            prompt_tokens=int(usage_meta.get("input_tokens") or estimate_tokens(system + user)),
            completion_tokens=int(usage_meta.get("output_tokens") or estimate_tokens(text)),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
