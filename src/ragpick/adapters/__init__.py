"""Adapters for LangChain and LlamaIndex objects.

Registered as ``type: langchain`` and ``type: llamaindex`` for the ``loader``,
``embedder``, ``retriever`` and ``generator`` kinds. Neither framework is a ragpick
dependency: objects are used through their public methods only.
"""

from ragpick.adapters.langchain import (
    LangChainEmbedder,
    LangChainGenerator,
    LangChainLoader,
    LangChainRetriever,
)
from ragpick.adapters.llamaindex import (
    LlamaIndexEmbedder,
    LlamaIndexGenerator,
    LlamaIndexLoader,
    LlamaIndexRetriever,
)

__all__ = [
    "LangChainEmbedder",
    "LangChainGenerator",
    "LangChainLoader",
    "LangChainRetriever",
    "LlamaIndexEmbedder",
    "LlamaIndexGenerator",
    "LlamaIndexLoader",
    "LlamaIndexRetriever",
]
