"""ragpick: choose, build, evaluate, and run the right RAG architecture.

Public API::

    from ragpick import Pipeline, load_config, registry

    p = Pipeline.from_yaml("rag.yaml")
    p.ingest("./docs")
    answer = p.query("How do I reset my password?", groups=["support"])
"""

from __future__ import annotations

import logging

from ragpick.config import RagConfig, load_config
from ragpick.errors import RagpickError
from ragpick.models import Answer, Chunk, Document, RetrievedChunk, Usage
from ragpick.pipeline import Pipeline
from ragpick.registry import register, registry

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Chunk",
    "Document",
    "Pipeline",
    "RagConfig",
    "RagpickError",
    "RetrievedChunk",
    "Usage",
    "__version__",
    "load_config",
    "register",
    "registry",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
