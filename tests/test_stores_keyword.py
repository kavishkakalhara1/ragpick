"""Vector stores and BM25: correctness, filtering/ACL, persistence."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from conftest import make_chunks
from ragpick.components.embedders import FakeEmbedder
from ragpick.components.keyword import BM25Index, tokenize
from ragpick.components.stores import InMemoryVectorStore, QdrantStore, qdrant_filter
from ragpick.errors import MissingDependencyError
from ragpick.filters import acl_filter
from ragpick.interfaces import KeywordIndex, VectorStore
from ragpick.models import Chunk

EMB = FakeEmbedder(dimension=64)


def ids(results: list[Any]) -> list[str]:
    return [r.chunk.id for r in results]


# ===================================================================== memory store
def test_memory_store_search_ranks_by_similarity() -> None:
    store = InMemoryVectorStore()
    assert isinstance(store, VectorStore)
    assert store.search(EMB.embed_one("x"), 3) == []
    store.add(
        make_chunks(
            {
                "a#0": "how to reset your password",
                "b#0": "invoice and tax questions",
                "c#0": "shipping times and delivery",
            }
        )
    )
    results = store.search(EMB.embed_one("reset password"), 2)
    assert ids(results)[0] == "a#0"
    assert len(results) == 2
    assert results[0].source == "vector"
    assert results[0].score >= results[1].score
    assert results[0].chunk.embedding is None
    assert store.search(EMB.embed_one("x"), 0) == []


def test_memory_store_upsert_delete_and_dimension_checks() -> None:
    store = InMemoryVectorStore()
    assert store.dimension is None
    store.add([])
    store.add(make_chunks({"a#0": "one", "a#1": "two", "b#0": "three"}))
    assert len(store) == 3
    assert store.dimension == 64
    store.add(make_chunks({"a#0": "replaced text"}))
    assert len(store) == 3
    assert ids(store.search(EMB.embed_one("replaced text"), 1)) == ["a#0"]
    store.delete(["a"])
    assert len(store) == 1
    store.delete(["zzz"])
    assert ids(store.search(EMB.embed_one("three"), 5)) == ["b#0"]
    with pytest.raises(ValueError, match="no embedding"):
        store.add([Chunk(id="x", doc_id="x", text="t")])
    with pytest.raises(ValueError, match="dimension"):
        store.add([Chunk(id="y", doc_id="y", text="t", embedding=[1.0, 0.0])])


def test_memory_store_metadata_filters() -> None:
    store = InMemoryVectorStore()
    store.add(
        make_chunks(
            {"a#0": "refund policy", "b#0": "refund policy", "c#0": "refund policy"},
            {"a#0": {"lang": "en"}, "b#0": {"lang": "de"}, "c#0": {}},
        )
    )
    q = EMB.embed_one("refund policy")
    assert ids(store.search(q, 5, {"lang": "en"})) == ["a#0"]
    assert sorted(ids(store.search(q, 5, {"lang": ["en", "de"]}))) == ["a#0", "b#0"]
    assert store.search(q, 5, {"lang": "fr"}) == []


# ===================================================================== BM25
def test_tokenize_keeps_codes_and_parts() -> None:
    assert tokenize("The SKU ABC-123 is v2.1") == [
        "sku",
        "abc-123",
        "abc",
        "123",
        "v2.1",
        "v2",
        "1",
    ]
    assert tokenize("") == []


def test_bm25_matches_reference_formula() -> None:
    index = BM25Index(k1=1.2, b=0.75)
    assert isinstance(index, KeywordIndex)
    docs = {"d1#0": "apple apple banana", "d2#0": "banana cherry", "d3#0": "cherry durian fig"}
    index.add([Chunk(id=k, doc_id=k.split("#")[0], text=v) for k, v in docs.items()])

    # reference computation for query "apple banana"
    n, avgdl = 3, (3 + 2 + 3) / 3

    def idf(df: int) -> float:
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def term(tf: int, dl: int, df: int) -> float:
        return idf(df) * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl))

    expected_d1 = term(2, 3, 1) + term(1, 3, 2)
    expected_d2 = term(1, 2, 2)
    scores = index.scores("apple banana")
    assert scores["d1#0"] == pytest.approx(expected_d1)
    assert scores["d2#0"] == pytest.approx(expected_d2)
    assert "d3#0" not in scores
    results = index.search("apple banana", 5)
    assert ids(results) == ["d1#0", "d2#0"]
    assert results[0].source == "keyword"


def test_bm25_exact_codes_beat_common_words() -> None:
    index = BM25Index()
    index.add(
        [
            Chunk(id="a#0", doc_id="a", text="error code E-4471 means the battery is faulty"),
            Chunk(id="b#0", doc_id="b", text="error messages are shown on the display error error"),
        ]
    )
    assert ids(index.search("what does E-4471 mean", 1)) == ["a#0"]
    assert ids(index.search("4471", 1)) == ["a#0"]


def test_bm25_edge_cases_and_validation() -> None:
    index = BM25Index()
    assert index.search("anything", 3) == []
    index.add([Chunk(id="a#0", doc_id="a", text="hello world")])
    assert index.search("the of and", 3) == []  # only stopwords
    assert index.search("hello", 0) == []
    assert index.idf("missing") > index.idf("hello")
    with pytest.raises(ValueError, match="k1"):
        BM25Index(b=2)


def test_bm25_upsert_delete_and_ties() -> None:
    index = BM25Index()
    index.add(
        [
            Chunk(id="b#0", doc_id="b", text="same text"),
            Chunk(id="a#0", doc_id="a", text="same text"),
        ]
    )
    assert ids(index.search("same", 5)) == ["a#0", "b#0"]  # deterministic tie-break
    index.add([Chunk(id="a#0", doc_id="a", text="different words")])
    assert len(index) == 2
    assert ids(index.search("same", 5)) == ["b#0"]
    index.delete(["b"])
    assert len(index) == 1
    assert index.search("same", 5) == []
    assert ids(index.search("different", 5)) == ["a#0"]


def test_bm25_custom_stopwords() -> None:
    index = BM25Index(stopwords=["foo"])
    index.add([Chunk(id="a#0", doc_id="a", text="foo the bar")])
    assert ids(index.search("the", 1)) == ["a#0"]
    assert index.search("foo", 1) == []


# ===================================================================== ACL in every store
def _acl_store(kind: str, chunks: list[Chunk]) -> Any:
    if kind == "memory":
        store: Any = InMemoryVectorStore()
    elif kind == "bm25":
        store = BM25Index()
    else:
        store = QdrantStore(client=FakeQdrantClient(), models=FAKE_MODELS)
    store.add(chunks)
    return store


def _search(store: Any, filters: dict[str, Any] | None) -> list[str]:
    if isinstance(store, BM25Index):
        return sorted(ids(store.search("refund policy", 10, filters)))
    return sorted(ids(store.search(EMB.embed_one("refund policy"), 10, filters)))


@pytest.mark.parametrize("kind", ["memory", "bm25", "qdrant"])
@pytest.mark.parametrize(
    ("groups", "expected"),
    [
        (None, ["public#0"]),
        ([], ["public#0"]),
        (["hr"], ["hr#0", "public#0"]),
        (["admin"], ["finance#0", "public#0"]),
        (["hr", "finance"], ["finance#0", "hr#0", "public#0"]),
        (["support"], ["public#0"]),
    ],
)
def test_access_control_filtering(
    kind: str, groups: list[str] | None, expected: list[str], acl_chunks: list[Chunk]
) -> None:
    store = _acl_store(kind, acl_chunks)
    assert _search(store, acl_filter(groups)) == expected


@pytest.mark.parametrize("kind", ["memory", "bm25", "qdrant"])
def test_no_filters_returns_everything(kind: str, acl_chunks: list[Chunk]) -> None:
    store = _acl_store(kind, acl_chunks)
    assert len(_search(store, None)) == 4


@pytest.mark.parametrize("kind", ["memory", "bm25", "qdrant"])
def test_acl_combined_with_metadata_filter(kind: str) -> None:
    chunks = make_chunks(
        {"a#0": "refund policy", "b#0": "refund policy", "c#0": "refund policy"},
        {
            "a#0": {"lang": "en", "allowed_groups": ["hr"]},
            "b#0": {"lang": "en"},
            "c#0": {"lang": "de"},
        },
    )
    store = _acl_store(kind, chunks)
    assert _search(store, acl_filter([], {"lang": "en"})) == ["b#0"]
    assert _search(store, acl_filter(["hr"], {"lang": "en"})) == ["a#0", "b#0"]


# ===================================================================== persistence
def test_memory_store_persistence_round_trip(tmp_path: Path, acl_chunks: list[Chunk]) -> None:
    store = InMemoryVectorStore()
    store.add(acl_chunks)
    store.persist(str(tmp_path / "store"))
    loaded = InMemoryVectorStore()
    loaded.load(str(tmp_path / "store"))
    assert len(loaded) == 4
    q = EMB.embed_one("refund policy employee")
    assert ids(loaded.search(q, 4)) == ids(store.search(q, 4))
    assert _search(loaded, acl_filter(["hr"])) == ["hr#0", "public#0"]
    loaded.add(make_chunks({"new#0": "new chunk"}))  # index map rebuilt correctly
    assert len(loaded) == 5
    with pytest.raises(FileNotFoundError):
        InMemoryVectorStore().load(str(tmp_path / "nope"))


def test_bm25_persistence_round_trip(tmp_path: Path, acl_chunks: list[Chunk]) -> None:
    index = BM25Index(k1=1.1, b=0.5)
    index.add(acl_chunks)
    index.persist(str(tmp_path / "kw"))
    loaded = BM25Index()
    loaded.load(str(tmp_path / "kw"))
    assert (loaded.k1, loaded.b) == (1.1, 0.5)
    assert loaded.scores("employee refund") == pytest.approx(index.scores("employee refund"))
    assert _search(loaded, acl_filter(["finance"])) == ["finance#0", "public#0"]
    with pytest.raises(FileNotFoundError):
        BM25Index().load(str(tmp_path / "nope"))


# ===================================================================== qdrant (fake client)
class _Model(dict[str, Any]):
    """Records constructor kwargs; mimics qdrant_client.models classes."""


def _model(name: str) -> Any:
    return lambda **kw: _Model(_t=name, **kw)


FAKE_MODELS = SimpleNamespace(
    Filter=_model("Filter"),
    FieldCondition=_model("FieldCondition"),
    MatchAny=_model("MatchAny"),
    IsEmptyCondition=_model("IsEmptyCondition"),
    PayloadField=_model("PayloadField"),
    PointStruct=_model("PointStruct"),
    VectorParams=_model("VectorParams"),
    FilterSelector=_model("FilterSelector"),
    Distance=SimpleNamespace(COSINE="Cosine", DOT="Dot"),
)


def _get(payload: dict[str, Any], dotted: str) -> Any:
    node: Any = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _eval_condition(cond: _Model, payload: dict[str, Any]) -> bool:
    """Tiny interpreter for the subset of Qdrant filters ragpick produces."""
    t = cond["_t"]
    if t == "Filter":
        must = cond.get("must") or []
        should = cond.get("should") or []
        ok = all(_eval_condition(c, payload) for c in must)
        if should:
            ok = ok and any(_eval_condition(c, payload) for c in should)
        return ok
    if t == "IsEmptyCondition":
        value = _get(payload, cond["is_empty"]["key"])
        return value is None or value == []
    if t == "FieldCondition":
        value = _get(payload, cond["key"])
        values = value if isinstance(value, list) else [value]
        return value is not None and bool(set(values) & set(cond["match"]["any"]))
    raise AssertionError(t)


class FakeQdrantClient:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, Any]] = {}
        self.points: dict[str, dict[str, Any]] = {}

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, collection_name: str, vectors_config: Any) -> None:
        self.collections[collection_name] = vectors_config

    def upsert(self, collection_name: str, points: list[Any]) -> None:
        for p in points:
            self.points[p["id"]] = p

    def query_points(
        self,
        collection_name: str,
        query: list[float],
        limit: int,
        query_filter: Any,
        with_payload: bool,
    ) -> Any:
        scored = []
        for p in self.points.values():
            if query_filter is not None and not _eval_condition(query_filter, p["payload"]):
                continue
            score = sum(a * b for a, b in zip(query, p["vector"], strict=True))
            scored.append(SimpleNamespace(payload=p["payload"], score=score))
        scored.sort(key=lambda s: -s.score)
        return SimpleNamespace(points=scored[:limit])

    def delete(self, collection_name: str, points_selector: Any) -> None:
        cond = points_selector["filter"]
        self.points = {
            k: p for k, p in self.points.items() if not _eval_condition(cond, p["payload"])
        }


def test_qdrant_filter_translation() -> None:
    assert qdrant_filter(None, FAKE_MODELS) is None
    f = qdrant_filter({"lang": "en", "allowed_groups": ["hr"]}, FAKE_MODELS)
    assert f["must"][0] == {
        "_t": "FieldCondition",
        "key": "metadata.lang",
        "match": {"_t": "MatchAny", "any": ["en"]},
    }
    acl = f["must"][1]
    assert acl["should"][0]["_t"] == "IsEmptyCondition"
    assert acl["should"][1]["match"]["any"] == ["hr"]


def test_qdrant_store_crud_with_fake_client() -> None:
    client = FakeQdrantClient()
    store = QdrantStore(collection="c", client=client, models=FAKE_MODELS)
    assert isinstance(store, VectorStore)
    assert store.search(EMB.embed_one("x"), 3) == []
    store.delete(["a"])  # no collection yet: no-op
    store.add([])
    store.add(
        make_chunks(
            {"a#0": "reset your password", "b#0": "tax invoice"}, {"a#0": {"allowed_groups": []}}
        )
    )
    assert client.collections["c"]["size"] == 64
    assert client.collections["c"]["distance"] == "Cosine"
    assert QdrantStore.point_id("a#0") == QdrantStore.point_id("a#0")
    results = store.search(EMB.embed_one("reset password"), 1)
    assert ids(results) == ["a#0"]
    assert results[0].chunk.metadata["allowed_groups"] == []  # sentinel restored
    store.add(make_chunks({"c#0": "more"}))  # collection exists path
    store.delete(["a"])
    assert sorted(ids(store.search(EMB.embed_one("x"), 5))) == ["b#0", "c#0"]
    store.persist("ignored")
    store.load("ignored")
    with pytest.raises(ValueError, match="no embedding"):
        store.add([Chunk(id="x", doc_id="x", text="t")])


def test_qdrant_store_builds_real_client(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def client_factory(**kw: Any) -> Any:
        captured.update(kw)
        return FakeQdrantClient()

    monkeypatch.setitem(sys.modules, "qdrant_client", SimpleNamespace(QdrantClient=client_factory))
    monkeypatch.setitem(sys.modules, "qdrant_client.models", FAKE_MODELS)
    QdrantStore()
    assert captured["location"] == ":memory:"
    QdrantStore(url="http://localhost:6333", api_key="k")
    assert captured["url"] == "http://localhost:6333"
    assert captured["location"] is None


def test_qdrant_store_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", None)
    with pytest.raises(MissingDependencyError, match=r"ragpick\[qdrant\]"):
        QdrantStore()
