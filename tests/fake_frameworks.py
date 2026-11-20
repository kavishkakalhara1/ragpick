"""Minimal stand-ins for LangChain / LlamaIndex objects (duck-typed, no imports)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragpick.components.embedders import FakeEmbedder

_EMB = FakeEmbedder(dimension=16)


# ---------------------------------------------------------------- LangChain-like
@dataclass
class LCDocument:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


class LCEmbeddings:
    def __init__(self, model: str = "m") -> None:
        self.model = model
        self.query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _EMB.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return _EMB.embed_one(text)


class LCDirectoryLoader:
    """Like langchain's TextLoader/DirectoryLoader: one doc per .md file, two for big.md."""

    def __init__(self, file_path: str, suffix: str = ".md") -> None:
        self.path = Path(file_path)
        self.suffix = suffix

    def lazy_load(self) -> Any:
        files = [self.path] if self.path.is_file() else sorted(self.path.glob(f"*{self.suffix}"))
        for f in files:
            text = f.read_text(encoding="utf-8")
            if f.name == "big.md":
                for page, part in enumerate(text.split("\n\n")):
                    yield LCDocument(part, {"source": str(f), "page": page})
            else:
                yield LCDocument(text, {"source": str(f)})


class LCLoadOnly:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> list[LCDocument]:
        return [LCDocument("only doc", {})]


class LCRetriever:
    def __init__(self, docs: list[LCDocument] | None = None) -> None:
        self.docs = docs or []

    def invoke(self, query: str) -> list[LCDocument]:
        return list(self.docs)


@dataclass
class AIMessage:
    content: Any
    usage_metadata: dict[str, int] | None = None


class LCChatModel:
    def __init__(
        self, model: str = "chat-m", reply: Any = "Answer [1].", fail: bool = False
    ) -> None:
        self.model_name = model
        self.reply = reply
        self.fail = fail
        self.calls: list[Any] = []

    def invoke(self, messages: list[tuple[str, str]]) -> AIMessage:
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("quota exceeded")
        return AIMessage(self.reply, {"input_tokens": 11, "output_tokens": 3})


# ---------------------------------------------------------------- LlamaIndex-like
class LIEmbedding:
    def __init__(self, model: str = "m") -> None:
        self.model = model

    def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        return _EMB.embed(texts)

    def get_query_embedding(self, query: str) -> list[float]:
        return _EMB.embed_one(query)


@dataclass
class LIDocument:
    text: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_content(self) -> str:
        return "content via get_content"


class LIReader:
    """Like SimpleDirectoryReader(input_dir=...) / (input_files=[...])."""

    def __init__(self, input_dir: str | None = None, input_files: list[str] | None = None) -> None:
        if input_files is not None:
            self.files = [Path(f) for f in input_files]
        else:
            self.files = sorted(Path(str(input_dir)).glob("*.md"))

    def load_data(self) -> list[LIDocument]:
        docs = [
            LIDocument(f.read_text(encoding="utf-8"), {"file_path": str(f)}) for f in self.files
        ]
        docs.append(LIDocument(None, {"file_name": "virtual.txt"}))
        return docs


@dataclass
class LINode:
    text: str
    metadata: dict[str, Any]
    node_id: str
    ref_doc_id: str | None = None

    def get_content(self) -> str:
        return self.text


@dataclass
class NodeWithScore:
    node: LINode
    score: float | None


class LIRetriever:
    def __init__(self, nodes: list[NodeWithScore] | None = None) -> None:
        self.nodes = nodes or []

    def retrieve(self, query: str) -> list[NodeWithScore]:
        return list(self.nodes)


@dataclass
class CompletionResponse:
    text: str


class LILLM:
    def __init__(self, model: str = "li-m", reply: str = "It is [2].", fail: bool = False) -> None:
        self.model = model
        self.reply = reply
        self.fail = fail
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> CompletionResponse:
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("server down")
        return CompletionResponse(self.reply)


class NotAnEmbedder:
    pass
