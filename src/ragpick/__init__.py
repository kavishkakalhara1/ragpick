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
from ragpick.models import Answer, Chunk, Document, RetrievedChunk, StreamEvent, Usage
from ragpick.pipeline import Pipeline
from ragpick.registry import register, registry
from ragpick.router import Route, Router, load_engine

__version__ = "0.9.0"

__all__ = [
    "Answer",
    "Chunk",
    "Document",
    "Pipeline",
    "RagConfig",
    "RagpickError",
    "RetrievedChunk",
    "Route",
    "Router",
    "StreamEvent",
    "Usage",
    "__version__",
    "load_config",
    "load_engine",
    "register",
    "registry",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
