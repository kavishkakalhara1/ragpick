"""LangChain / LlamaIndex adapters, using duck-typed fakes from tests/fake_frameworks.py."""

from __future__ import annotations

from pathlib import Path

import pytest

import fake_frameworks as ff
from conftest import EXAMPLES
from ragpick import Pipeline
from ragpick.adapters import (
    LangChainEmbedder,
    LangChainGenerator,
    LangChainLoader,
    LangChainRetriever,
    LlamaIndexEmbedder,
    LlamaIndexGenerator,
    LlamaIndexLoader,
    LlamaIndexRetriever,
)
from ragpick.adapters._common import import_object, resolve
from ragpick.errors import ComponentConfigError, ProviderError
from ragpick.eval.metrics import LLMJudge
from ragpick.filters import acl_filter
from ragpick.interfaces import Embedder, Generator, Loader, Retriever, TextCompleter
from ragpick.models import Chunk, RetrievedChunk
from ragpick.registry import registry

CTX = [
    RetrievedChunk(chunk=Chunk(id="a#0", doc_id="a", text="Alpha."), score=1, source="vector"),
    RetrievedChunk(chunk=Chunk(id="b#0", doc_id="b", text="Beta."), score=1, source="vector"),
]


# ---------------------------------------------------------------- common
def test_import_object_forms_and_errors() -> None:
    assert import_object("fake_frameworks:LCEmbeddings") is ff.LCEmbeddings
    assert import_object("fake_frameworks.LCEmbeddings") is ff.LCEmbeddings
    assert import_object("pathlib:Path.cwd") == Path.cwd
    with pytest.raises(ComponentConfigError, match="Invalid class_path"):
        import_object("nodots")
    with pytest.raises(ComponentConfigError, match="pip install"):
        import_object("no_such_pkg_xyz.Thing")
    with pytest.raises(ComponentConfigError, match="has no attribute"):
        import_object("fake_frameworks:Missing")


def test_resolve_errors() -> None:
    with pytest.raises(ComponentConfigError, match="set `class_path`"):
        resolve(None, None, None, "X")
    with pytest.raises(ComponentConfigError, match="could not create"):
        resolve(None, "fake_frameworks.LCEmbeddings", {"bogus": 1}, "X")


def test_registered_under_both_names() -> None:
    for kind in ("loader", "embedder", "retriever", "generator"):
        assert registry.has(kind, "langchain")
        assert registry.has(kind, "llamaindex")


# ---------------------------------------------------------------- LangChain
def test_langchain_embedder() -> None:
    emb = LangChainEmbedder(class_path="fake_frameworks.LCEmbeddings", kwargs={"model": "x"})
    assert isinstance(emb, Embedder)
    assert emb._obj.model == "x"
    assert len(emb.embed(["a", "b"])) == 2
    assert emb.dimension == 16  # probed via embed_query
    assert emb._obj.query_calls == 1
    assert LangChainEmbedder(embeddings=ff.LCEmbeddings(), dimension=3).dimension == 3

    class DocsOnly:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 2.0] for _ in texts]

    assert LangChainEmbedder(embeddings=DocsOnly()).embed_query("q") == [1.0, 2.0]
    with pytest.raises(ComponentConfigError, match="no `embed_documents"):
        LangChainEmbedder(embeddings=ff.NotAnEmbedder())


def test_langchain_loader_ids_and_pages(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "big.md").write_text("page one\n\npage two", encoding="utf-8")
    loader = LangChainLoader(class_path="fake_frameworks:LCDirectoryLoader", path_arg="file_path")
    assert isinstance(loader, Loader)
    docs = loader.load(str(tmp_path))
    assert [d.id for d in docs] == ["a.md", "big.md:0", "big.md:1"]
    assert docs[1].metadata["page"] == 0
    single = loader.load(str(tmp_path / "a.md"))
    assert [d.id for d in single] == ["a.md"]
    load_only = LangChainLoader(class_path="fake_frameworks.LCLoadOnly").load(str(tmp_path))
    assert load_only[0].text == "only doc"
    assert load_only[0].id.endswith("-0")  # no source metadata
    with pytest.raises(FileNotFoundError):
        loader.load(str(tmp_path / "missing"))
    with pytest.raises(ComponentConfigError, match="class_path"):
        LangChainLoader()
    factory_loader = LangChainLoader(loader_factory=lambda p: ff.LCDirectoryLoader(p))
    assert len(factory_loader.load(str(tmp_path))) == 3


def test_langchain_retriever_post_filters_acl() -> None:
    docs = [
        ff.LCDocument(
            "secret", {"source": "hr.md", "allowed_groups": ["hr"], "score": 0.9}, id="hr#0"
        ),
        ff.LCDocument("public one", {"source": "faq.md"}, id="faq#0"),
        ff.LCDocument("public two", {"source": "faq.md", "relevance_score": 0.2}),
    ]
    retriever = LangChainRetriever(retriever=ff.LCRetriever(docs), k=5)
    assert isinstance(retriever, Retriever)
    public = retriever.retrieve("q", filters=acl_filter([]))
    assert [r.chunk.id for r in public] == ["faq#0", "faq.md#3"]
    assert public[0].score == pytest.approx(1 / 2)  # 1/rank when no score
    assert public[1].score == 0.2
    hr = retriever.retrieve("q", 1, acl_filter(["hr"]))
    assert [(r.chunk.id, r.score, r.chunk.doc_id) for r in hr] == [("hr#0", 0.9, "hr.md")]
    assert len(retriever.retrieve("q")) == 3
    with pytest.raises(ComponentConfigError, match="no `invoke"):
        LangChainRetriever(retriever=ff.NotAnEmbedder())


def test_langchain_generator() -> None:
    model = ff.LCChatModel()
    gen = LangChainGenerator(chat_model=model)
    assert isinstance(gen, Generator)
    assert isinstance(gen, TextCompleter)
    answer = gen.generate("q", CTX)
    assert answer.citations == ["a#0"]
    assert answer.usage.prompt_tokens == 11
    assert model.calls[0][0][0] == "system"
    assert model.calls[0][1][0] == "human"
    assert gen.model == "chat-m"
    blocks = LangChainGenerator(
        chat_model=ff.LCChatModel(reply=[{"type": "text", "text": "Hi [2]"}, "!"])
    )
    assert blocks.generate("q", CTX).text == "Hi [2]!"
    other = LangChainGenerator(chat_model=ff.LCChatModel(reply=42))
    assert other.complete("s", "u").text == "42"
    built = LangChainGenerator(class_path="fake_frameworks.LCChatModel", kwargs={"model": "gpt-x"})
    assert built.model == "gpt-x"
    with pytest.raises(ProviderError, match="quota exceeded"):
        LangChainGenerator(chat_model=ff.LCChatModel(fail=True)).complete("s", "u")


# ---------------------------------------------------------------- LlamaIndex
def test_llamaindex_embedder() -> None:
    emb = LlamaIndexEmbedder(class_path="fake_frameworks.LIEmbedding")
    assert isinstance(emb, Embedder)
    assert emb.dimension == 16
    assert len(emb.embed(["x"])[0]) == 16
    assert emb.embed_query("x") == emb.embed(["x"])[0]

    class BatchOnly:
        def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
            return [[0.5] for _ in texts]

    assert LlamaIndexEmbedder(embed_model=BatchOnly(), dimension=1).embed_query("q") == [0.5]
    with pytest.raises(ComponentConfigError, match="get_text_embedding_batch"):
        LlamaIndexEmbedder(embed_model=ff.NotAnEmbedder())


def test_llamaindex_loader(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    loader = LlamaIndexLoader(class_path="fake_frameworks.LIReader")
    assert isinstance(loader, Loader)
    docs = loader.load(str(tmp_path))
    assert [d.id for d in docs] == ["a.md", "b.md", "virtual.txt"]
    assert docs[2].text == "content via get_content"
    single = loader.load(str(tmp_path / "a.md"))  # passed as input_files
    assert [d.id for d in single] == ["a.md", "virtual.txt"]
    custom = LlamaIndexLoader(reader_factory=lambda p: ff.LIReader(input_dir=p))
    assert len(custom.load(str(tmp_path))) == 3
    with pytest.raises(FileNotFoundError):
        loader.load(str(tmp_path / "nope"))
    with pytest.raises(ComponentConfigError, match="pip install"):
        LlamaIndexLoader()  # llama_index is not installed in the test env


def test_llamaindex_retriever() -> None:
    nodes = [
        ff.NodeWithScore(ff.LINode("restricted", {"allowed_groups": ["hr"]}, "n1", "hr.md"), 0.9),
        # like real LlamaIndex: ref_doc_id is a UUID, file_path is the real source
        ff.NodeWithScore(ff.LINode("open", {"file_path": "faq.md"}, "n2", "0b1c-uuid"), None),
    ]
    retriever = LlamaIndexRetriever(retriever=ff.LIRetriever(nodes))
    assert isinstance(retriever, Retriever)
    public = retriever.retrieve("q", filters=acl_filter(None))
    assert [(r.chunk.id, r.chunk.doc_id, r.score) for r in public] == [("n2", "faq.md", 0.5)]
    assert [r.chunk.doc_id for r in retriever.retrieve("q", 5, acl_filter(["hr"]))] == [
        "hr.md",
        "faq.md",
    ]
    with pytest.raises(ComponentConfigError, match="no `retrieve"):
        LlamaIndexRetriever(retriever=ff.NotAnEmbedder())


def test_llamaindex_generator_and_judge() -> None:
    llm = ff.LILLM()
    gen = LlamaIndexGenerator(llm=llm)
    assert isinstance(gen, TextCompleter)
    answer = gen.generate("q", CTX)
    assert answer.citations == ["b#0"]
    assert "Question: q" in llm.prompts[0]
    assert answer.usage.prompt_tokens > 0
    judge = LLMJudge(LlamaIndexGenerator(llm=ff.LILLM(reply='{"score": 0.6, "reason": "ok"}')))
    assert judge.answer_relevance("q", "a").score == 0.6
    assert (
        LlamaIndexGenerator(class_path="fake_frameworks.LILLM", kwargs={"model": "z"}).model == "z"
    )
    with pytest.raises(ProviderError, match="server down"):
        LlamaIndexGenerator(llm=ff.LILLM(fail=True)).complete("s", "u")


# ---------------------------------------------------------------- via config
def test_pipeline_built_from_yaml_class_paths(tmp_path: Path) -> None:
    pipeline = Pipeline.from_dict(
        {
            "blueprint": "hybrid",
            "index_dir": str(tmp_path / "idx"),
            "components": {
                "loader": {
                    "type": "llamaindex",
                    "params": {"class_path": "fake_frameworks.LIReader"},
                },
                "embedder": {
                    "type": "langchain",
                    "params": {
                        "class_path": "fake_frameworks:LCEmbeddings",
                        "kwargs": {"model": "e"},
                    },
                },
                "generator": {
                    "type": "langchain",
                    "params": {
                        "class_path": "fake_frameworks.LCChatModel",
                        "kwargs": {"model": "g"},
                    },
                },
            },
        }
    )
    stats = pipeline.ingest(EXAMPLES / "data")
    assert stats.documents == 10  # 9 example .md files + the fake reader's virtual doc
    answer = pipeline.query("What does error E-1042 mean?")
    assert answer.text == "Answer [1]."
    assert answer.citations
