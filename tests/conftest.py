from __future__ import annotations

from pathlib import Path

import pytest

from ragpick.components.embedders import FakeEmbedder
from ragpick.models import Chunk

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


def make_chunks(
    texts: dict[str, str], meta: dict[str, dict[str, object]] | None = None
) -> list[Chunk]:
    """Build embedded chunks: ``{chunk_id: text}``; doc_id is the part before '#'."""
    embedder = FakeEmbedder(dimension=64)
    meta = meta or {}
    chunks = []
    for cid, text in texts.items():
        chunks.append(
            Chunk(
                id=cid,
                doc_id=cid.split("#")[0],
                text=text,
                metadata=dict(meta.get(cid, {})),
                embedding=embedder.embed_one(text),
            )
        )
    return chunks


@pytest.fixture(autouse=True)
def wide_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rich wraps CLI output at the terminal width; pin it so assertions don't depend
    on the length of temporary paths or the CI runner's terminal."""
    monkeypatch.setenv("COLUMNS", "200")


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dimension=64)


@pytest.fixture
def acl_chunks() -> list[Chunk]:
    """Three chunks about refunds with different access levels."""
    return make_chunks(
        {
            "public#0": "refund policy for all customers",
            "hr#0": "refund policy for employee expenses",
            "finance#0": "refund policy for finance approvals",
            "empty#0": "refund policy restricted to nobody",
        },
        {
            "hr#0": {"allowed_groups": ["hr"]},
            "finance#0": {"allowed_groups": ["finance", "admin"]},
            "empty#0": {"allowed_groups": []},
        },
    )
