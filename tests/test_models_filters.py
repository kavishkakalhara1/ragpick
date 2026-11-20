from __future__ import annotations

import pytest

from ragpick.filters import ACL_KEY, acl_filter, matches_filters
from ragpick.models import Answer, Chunk, Document, RetrievedChunk, Usage


def test_models_defaults_and_usage_total() -> None:
    doc = Document(id="a", text="hello")
    assert doc.metadata == {}
    chunk = Chunk(id="a#0", doc_id="a", text="hello")
    assert chunk.embedding is None
    usage = Usage(prompt_tokens=3, completion_tokens=4, latency_ms=1.5)
    assert usage.total_tokens == 7
    ans = Answer(text="x", contexts=[RetrievedChunk(chunk=chunk, score=1.0, source="vector")])
    assert ans.citations == []
    assert ans.usage.total_tokens == 0


def test_retrieved_chunk_source_is_validated() -> None:
    chunk = Chunk(id="a#0", doc_id="a", text="t")
    with pytest.raises(ValueError, match="source"):
        RetrievedChunk(chunk=chunk, score=1.0, source="magic")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("metadata", "filters", "expected"),
    [
        ({"lang": "en"}, None, True),
        ({"lang": "en"}, {}, True),
        ({"lang": "en"}, {"lang": "en"}, True),
        ({"lang": "en"}, {"lang": "de"}, False),
        ({"lang": "en"}, {"lang": ["de", "en"]}, True),
        ({"tags": ["a", "b"]}, {"tags": "b"}, True),
        ({"tags": ["a", "b"]}, {"tags": ["c"]}, False),
        ({}, {"lang": "en"}, False),
        ({"lang": "en", "year": 2024}, {"lang": "en", "year": 2023}, False),
        ({"x": {"nested": 1}}, {"x": {"nested": 1}}, True),
    ],
)
def test_matches_filters_equality_and_in(
    metadata: dict[str, object], filters: dict[str, object] | None, expected: bool
) -> None:
    assert matches_filters(metadata, filters) is expected


def test_acl_public_chunks_always_visible() -> None:
    assert matches_filters({}, {ACL_KEY: []})
    assert matches_filters({ACL_KEY: None}, {ACL_KEY: ["hr"]})


def test_acl_restricted_chunks_need_intersection() -> None:
    meta = {ACL_KEY: ["hr", "finance"]}
    assert matches_filters(meta, {ACL_KEY: ["finance"]})
    assert not matches_filters(meta, {ACL_KEY: ["support"]})
    assert not matches_filters(meta, {ACL_KEY: []})
    assert matches_filters({ACL_KEY: "hr"}, {ACL_KEY: "hr"})


def test_acl_filter_is_secure_by_default() -> None:
    assert acl_filter(None) == {ACL_KEY: []}
    combined = acl_filter(["a"], {"lang": "en"})
    assert combined == {"lang": "en", ACL_KEY: ["a"]}
