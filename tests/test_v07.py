"""v0.7: pgvector store, multimodal ingest, generated evaluation datasets."""

from __future__ import annotations

import json
import struct
import sys
import types
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from conftest import EXAMPLES, make_chunks
from ragpick import Pipeline
from ragpick.cache import Cache, wrap_generator
from ragpick.cli import app as cli_app
from ragpick.components.generators import (
    FakeGenerator,
    LLMGenerator,
    OllamaGenerator,
    OpenAIGenerator,
)
from ragpick.components.multimodal import MultimodalLoader
from ragpick.components.pgvector import PgVectorStore, filter_sql
from ragpick.errors import ConfigError, MissingDependencyError, ProviderError
from ragpick.eval import load_dataset
from ragpick.eval.generate import (
    generate_dataset,
    heuristic_question,
    parse_generated,
    sample_chunks,
    write_dataset,
)
from ragpick.filters import acl_filter
from ragpick.interfaces import VisionCompleter
from ragpick.models import Chunk, Completion, Usage
from ragpick.selector import recommend

DATA = EXAMPLES / "data"
IMAGES = EXAMPLES / "images"


def png_bytes(color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    raw = b"".join(b"\x00" + bytes(color) * 2 for _ in range(2))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


# =================================================================== pgvector
class PgCursor:
    def __init__(self, db: FakePg) -> None:
        self.db = db
        self._rows: list[Any] = []

    def execute(self, sql: str, params: Any = None) -> None:
        self.db.log.append((sql, params))
        if sql.startswith("SELECT to_regclass"):
            self._rows = [(self.db.table_exists or None,)]
        elif sql.startswith("SELECT id, doc_id"):
            self._rows = self.db.rows

    def executemany(self, sql: str, rows: list[Any]) -> None:
        self.db.log.append((sql, rows))

    def fetchone(self) -> Any:
        return self._rows[0]

    def fetchall(self) -> list[Any]:
        return self._rows

    def close(self) -> None:
        pass


class FakePg:
    def __init__(self) -> None:
        self.log: list[tuple[str, Any]] = []
        self.rows: list[Any] = []
        self.table_exists = False
        self.kwargs: dict[str, Any] = {}

    def connect(self, dsn: str, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        db = self

        class Conn:
            def cursor(self) -> PgCursor:
                return PgCursor(db)

            def close(self) -> None:
                pass

        return Conn()


def test_filter_sql() -> None:
    assert filter_sql(None) == ("TRUE", [])
    where, params = filter_sql({"lang": "en", "allowed_groups": ["hr"]})
    assert where.count("metadata ? %s") == 2  # existence check for lang, NOT ? for the ACL
    assert "NOT (metadata ? %s)" in where
    assert where.count("%s") == len(params)
    assert ["en"] in params
    assert ["hr"] in params
    _, numeric = filter_sql({"year": 2024, "flags": [True]})
    assert ["2024"] in numeric
    assert ["true"] in numeric


def test_pgvector_store_sql() -> None:
    pg = FakePg()
    store = PgVectorStore("postgresql://x", table="kb", psycopg=pg)
    assert store.search([0.1], 3) == []  # table missing
    assert store.search([0.1], 0) == []
    store.add([])
    chunks = make_chunks({"a#0": "alpha", "b#0": "beta"}, {"a#0": {"allowed_groups": ["hr"]}})
    store.add(chunks)
    statements = [sql for sql, _ in pg.log]
    assert "CREATE EXTENSION IF NOT EXISTS vector" in statements
    assert any("CREATE TABLE IF NOT EXISTS kb" in s and "vector(64)" in s for s in statements)
    assert any("USING hnsw (embedding vector_cosine_ops)" in s for s in statements)
    insert_sql, rows = pg.log[-1]
    assert "ON CONFLICT (id) DO UPDATE" in insert_sql
    assert rows[0][0] == "a#0"
    assert json.loads(rows[0][3]) == {"allowed_groups": ["hr"]}
    assert rows[0][4].startswith("[")
    assert pg.kwargs == {"autocommit": True}
    pg.rows = [
        ("a#0", "a", "alpha", {"allowed_groups": ["hr"]}, 0.9),
        ("b#0", "b", "beta", "{}", 0.5),
    ]
    results = store.search([0.1, 0.2], 2, acl_filter(["hr"]))
    assert [(r.chunk.id, r.score, r.source) for r in results] == [
        ("a#0", 0.9, "vector"),
        ("b#0", 0.5, "vector"),
    ]
    sql, params = pg.log[-1]
    assert "ORDER BY embedding <=> %s::vector LIMIT %s" in sql
    assert params[-1] == 2
    store.delete(["a"])
    assert pg.log[-1] == ("DELETE FROM kb WHERE doc_id = ANY(%s)", (["a"],))
    store.delete([])
    store.persist("x")
    store.load("x")
    with pytest.raises(ValueError, match="no embedding"):
        store.add([Chunk(id="x", doc_id="x", text="t")])


def test_pgvector_options_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    pg = FakePg()
    store = PgVectorStore("postgresql://x", create_extension=False, index="none", psycopg=pg)
    store.add(make_chunks({"a#0": "alpha"}))
    statements = [sql for sql, _ in pg.log]
    assert "CREATE EXTENSION IF NOT EXISTS vector" not in statements
    assert not any("USING hnsw" in s for s in statements)
    pg2 = FakePg()
    pg2.table_exists = True
    PgVectorStore("postgresql://x", psycopg=pg2).delete(["a"])
    assert pg2.log[-1][0].startswith("DELETE FROM ragpick_chunks")
    with pytest.raises(ConfigError, match="table name"):
        PgVectorStore("postgresql://x", table="drop table; --", psycopg=pg)
    with pytest.raises(ConfigError, match="index"):
        PgVectorStore("postgresql://x", index="btree", psycopg=pg)
    monkeypatch.setitem(sys.modules, "psycopg", None)
    with pytest.raises(MissingDependencyError, match="psycopg"):
        PgVectorStore("postgresql://x")


# =================================================================== multimodal
class Vision(LLMGenerator):
    def __init__(
        self, reply: str = "A wiring diagram showing terminals R, C, W1 and G.", fail: bool = False
    ) -> None:
        self.reply = reply
        self.fail = fail
        self.calls: list[tuple[int, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        return Completion(text="Answer [1].")

    def describe_image(self, image: bytes, mime: str, prompt: str) -> Completion:
        self.calls.append((len(image), mime))
        if self.fail:
            raise ProviderError("vision down")
        return Completion(text=self.reply)


def test_offline_multimodal_loader() -> None:
    docs = {d.id: d for d in MultimodalLoader(FakeGenerator()).load(str(IMAGES))}
    assert set(docs) == {"README.md", "lm-300-wiring-diagram.png", "thermostat-error-screen.png"}
    diagram = docs["lm-300-wiring-diagram.png"]
    assert diagram.metadata["modality"] == "image"
    assert diagram.metadata["mime"] == "image/png"
    assert diagram.metadata["description_source"] == "offline"
    assert diagram.metadata["image_path"].endswith("lm-300-wiring-diagram.png")
    assert "Image: lm 300 wiring diagram." in diagram.text
    assert "Alt text: LM-300 Pro wiring diagram with terminal colours." in diagram.text
    assert "G (green) to the fan" in diagram.text  # caption file
    assert "modality" not in docs["README.md"].metadata


def test_vision_descriptions_and_fallback(tmp_path: Path) -> None:
    (tmp_path / "chart.png").write_bytes(png_bytes())
    vision = Vision()
    assert isinstance(vision, VisionCompleter)
    [doc] = list(MultimodalLoader(vision).load(str(tmp_path)))
    assert doc.text.startswith("A wiring diagram showing terminals")
    assert doc.metadata["description_source"] == "vision"
    assert vision.calls[0][1] == "image/png"
    [failed] = list(MultimodalLoader(Vision(fail=True)).load(str(tmp_path)))
    assert failed.metadata["description_source"] == "offline-fallback"
    assert failed.text == "Image: chart."
    [offline] = list(MultimodalLoader(vision, describe="offline").load(str(tmp_path / "chart.png")))
    assert offline.metadata["description_source"] == "offline"
    with pytest.raises(ConfigError, match="vision-capable"):
        MultimodalLoader(FakeGenerator(), describe="vision")
    with pytest.raises(ConfigError, match="describe"):
        MultimodalLoader(FakeGenerator(), describe="ocr")
    with pytest.raises(FileNotFoundError):
        list(MultimodalLoader(FakeGenerator()).load(str(tmp_path / "missing")))


def test_captions_restrict_access_and_big_images_are_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "payroll.png").write_bytes(png_bytes())
    (tmp_path / "payroll.txt").write_text(
        "---\nallowed_groups: [hr]\n---\nPayroll chart Q3.", encoding="utf-8"
    )
    (tmp_path / "huge.png").write_bytes(png_bytes() + b"\x00" * 2000)
    (tmp_path / "notes.bin").write_bytes(b"\x00")
    docs = {
        d.id: d for d in MultimodalLoader(FakeGenerator(), max_image_bytes=500).load(str(tmp_path))
    }
    assert set(docs) == {"payroll.png"}  # caption file isn't a document; huge.png skipped
    assert docs["payroll.png"].metadata["allowed_groups"] == ["hr"]
    assert "Payroll chart Q3." in docs["payroll.png"].text
    assert "max_image_bytes" in caplog.text


def test_pdf_images_with_mocked_pypdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Page:
        def __init__(self, text: str, images: list[Any]) -> None:
            self._text = text
            self.images = images

        def extract_text(self) -> str:
            return self._text

    class Reader:
        def __init__(self, path: str) -> None:
            self.pages = [
                Page("Manual page one", [SimpleNamespace(name="fig1.jpg", data=png_bytes())]),
                Page("Page two", [SimpleNamespace(name="big.png", data=b"x" * 50)]),
            ]

    fake: Any = types.ModuleType("pypdf")
    fake.PdfReader = Reader
    monkeypatch.setitem(sys.modules, "pypdf", fake)
    (tmp_path / "manual.pdf").write_bytes(b"%PDF")
    vision = Vision(reply="Figure: exploded view of the LM-300.")
    docs = {d.id: d for d in MultimodalLoader(vision, max_image_bytes=40).load(str(tmp_path))}
    assert set(docs) == {"manual.pdf"}  # both embedded images exceed max_image_bytes=40
    pdf_docs = {d.id: d for d in MultimodalLoader(vision).load(str(tmp_path))}
    assert set(pdf_docs) == {"manual.pdf", "manual.pdf:image-1-1", "manual.pdf:image-2-1"}
    image = pdf_docs["manual.pdf:image-1-1"]
    assert image.metadata["page"] == 1
    assert image.metadata["mime"] == "image/jpeg"
    assert image.metadata["image_path"].endswith("manual.pdf#page=1")
    assert image.text.startswith("Figure: exploded view")
    assert "Image 1 on page 1 of manual.pdf." in image.text
    no_images = {d.id for d in MultimodalLoader(vision, pdf_images=False).load(str(tmp_path))}
    assert no_images == {"manual.pdf"}


def test_multimodal_blueprint_end_to_end(tmp_path: Path) -> None:
    pipeline = Pipeline.from_dict({"blueprint": "multimodal", "index_dir": str(tmp_path / "i")})
    assert isinstance(pipeline.loader, MultimodalLoader)
    pipeline.ingest(IMAGES)
    answer = pipeline.query("Which wire goes to the fan on the LM-300 Pro?")
    assert answer.citations[0].startswith("lm-300-wiring-diagram.png")
    assert answer.contexts[0].chunk.metadata["modality"] == "image"
    vision = Vision()
    seen = Pipeline.from_dict(
        {"blueprint": "multimodal", "index_dir": str(tmp_path / "v")},
        components={"generator": vision},
    )
    seen.ingest(IMAGES)
    assert len(vision.calls) == 2


def test_openai_and_ollama_describe_image() -> None:
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="A red error screen."))],
            usage=SimpleNamespace(prompt_tokens=800, completion_tokens=6),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = OpenAIGenerator(client=client).describe_image(b"\x89PNG", "image/png", "Describe")
    assert result.text == "A red error screen."
    assert result.usage.prompt_tokens == 800
    content = calls[0]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Describe"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,iVBORw")

    def broken(**kwargs: Any) -> Any:
        raise RuntimeError("model has no vision")

    bad = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=broken)))
    with pytest.raises(ProviderError, match="no vision"):
        OpenAIGenerator(client=bad).describe_image(b"x", "image/png", "p")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["messages"][0]["images"] == ["eA=="]
        return httpx.Response(200, json={"message": {"content": "A chart."}, "eval_count": 3})

    ollama = OllamaGenerator(
        client=httpx.Client(base_url="http://o", transport=httpx.MockTransport(handler))
    )
    assert ollama.describe_image(b"x", "image/png", "p").text == "A chart."
    down = OllamaGenerator(
        client=httpx.Client(
            base_url="http://o", transport=httpx.MockTransport(lambda r: httpx.Response(500))
        )
    )
    with pytest.raises(ProviderError, match="image description failed"):
        down.describe_image(b"x", "image/png", "p")


def test_cached_image_descriptions(tmp_path: Path) -> None:
    vision = Vision()
    cached = wrap_generator(vision, Cache(tmp_path / "c.sqlite"), {"t": "v"})
    assert isinstance(cached, VisionCompleter)
    first = cached.describe_image(b"img", "image/png", "p")  # type: ignore[attr-defined]
    second = cached.describe_image(b"img", "image/png", "p")  # type: ignore[attr-defined]
    assert first.text == second.text
    assert second.usage == Usage(cached=True)
    assert len(vision.calls) == 1
    cached.describe_image(b"other", "image/png", "p")  # type: ignore[attr-defined]
    assert len(vision.calls) == 2
    plain = wrap_generator(FakeGenerator(), Cache(tmp_path / "d.sqlite"), {})
    assert not isinstance(plain, VisionCompleter)  # no describe_image on the inner generator


def test_selector_images() -> None:
    r = recommend({"data_types": ["markdown", "diagrams"]}, installed=lambda m: True)
    assert r.blueprint == "multimodal"
    assert any("vision-capable" in w for w in r.warnings)
    assert not any("No built-in loader" in w for w in r.warnings)


# =================================================================== generated datasets
def chunk(text: str, cid: str = "doc.md#0", **meta: Any) -> Chunk:
    return Chunk(id=cid, doc_id=cid.split("#")[0], text=text, metadata=meta)


def test_heuristic_questions() -> None:
    reset = chunk(
        "# Account\n\n## Reset your password\n\nTap Forgot password on the sign-in screen to start."
    )
    assert heuristic_question(reset) == (
        "How do I reset your password?",
        "Tap Forgot password on the sign-in screen to start.",
    )
    plan = chunk("## Lumen Plus pricing\n\nLumen Plus costs 4.99 dollars per month for everyone.")
    assert heuristic_question(plan)[0] == "What should I know about lumen Plus pricing?"  # type: ignore[index]
    code = chunk("Errors:\n- **E-1042**: Low battery. Replace the batteries.")
    assert heuristic_question(code) == ("What does E-1042 mean?", "E-1042: Low battery.")
    titled = chunk(
        "This page explains how the warranty works for every product.", title="warranty-and-returns"
    )
    assert (
        heuristic_question(titled)[0]
        == "What does the documentation say about warranty and returns?"
    )  # type: ignore[index]
    assert heuristic_question(chunk("tiny")) is None


def test_parse_generated() -> None:
    assert parse_generated('{"question": "How long is the warranty?", "answer": "Two years."}') == (
        "How long is the warranty?",
        "Two years.",
    )
    assert parse_generated('{"question": "no question mark", "answer": "x"}') is None
    assert parse_generated('{"question": "Short?"}') is None
    assert parse_generated("nothing") is None
    assert parse_generated('{"answer": "x"}') is None


def test_sample_chunks_round_robin_and_deterministic() -> None:
    chunks = [chunk("x", f"{d}#{i}") for d in ("a", "b", "c") for i in range(3)]
    picked = sample_chunks(chunks, 4, seed=1)
    assert [c.doc_id for c in picked[:3]] == ["a", "b", "c"]
    assert len(picked) == 4
    assert [c.id for c in sample_chunks(chunks, 4, seed=1)] == [c.id for c in picked]
    assert len(sample_chunks(chunks, 50)) == 9


def test_generate_dataset_offline(tmp_path: Path) -> None:
    pipeline = Pipeline.from_dict(
        {"blueprint": "hybrid", "index_dir": str(tmp_path / "i"), "data": str(DATA)}
    )
    examples = generate_dataset(pipeline, count=8)
    assert 5 <= len(examples) <= 8
    assert len({e.question for e in examples}) == len(examples)
    assert {e.id for e in examples} == {f"gen-{i}" for i in range(1, len(examples) + 1)}
    by_doc = {e.relevant_doc_ids[0]: e for e in examples}  # type: ignore[index]
    assert len(by_doc) >= 5  # spread across documents
    for e in examples:
        restricted = e.relevant_doc_ids[0] in (
            "internal-escalation.md",
            "employee-discount-policy.md",
        )  # type: ignore[index]
        assert (e.groups is not None) == restricted
    path = write_dataset(examples, tmp_path / "out" / "gen.jsonl")
    assert len(load_dataset(path)) == len(examples)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert all(
        ("groups" in row) == (e.groups is not None) for row, e in zip(rows, examples, strict=True)
    )


def test_generate_dataset_with_llm(tmp_path: Path) -> None:
    class Writer(LLMGenerator):
        def __init__(self) -> None:
            self.n = 0

        def complete(self, system: str, user: str) -> Completion:
            self.n += 1
            if self.n == 2:
                return Completion(text="unusable")  # falls back to the heuristic
            return Completion(
                text=json.dumps(
                    {"question": f"Generated question number {self.n}?", "answer": "A."}
                )
            )

    writer = Writer()
    pipeline = Pipeline.from_dict(
        {"blueprint": "naive", "index_dir": str(tmp_path / "i")}, components={"generator": writer}
    )
    examples = generate_dataset(pipeline, DATA, count=4, seed=3)
    assert len(examples) == 4
    assert examples[0].question == "Generated question number 1?"
    assert not examples[1].question.startswith("Generated")
    heuristic = generate_dataset(pipeline, DATA, count=2, mode="heuristic")
    assert all(not e.question.startswith("Generated") for e in heuristic)


def test_generate_dataset_errors(tmp_path: Path) -> None:
    pipeline = Pipeline.from_dict({"blueprint": "naive", "index_dir": str(tmp_path / "i")})
    with pytest.raises(ConfigError, match="No path given"):
        generate_dataset(pipeline)
    with pytest.raises(ConfigError, match="mode"):
        generate_dataset(pipeline, DATA, mode="magic")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ConfigError, match="No chunks"):
        generate_dataset(pipeline, empty)

    class GenerateOnly:
        def generate(self, query: str, contexts: Any) -> Any:
            raise AssertionError

    plain = Pipeline.from_dict({"blueprint": "naive"}, components={"generator": GenerateOnly()})
    with pytest.raises(ConfigError, match="complete"):
        generate_dataset(plain, DATA, mode="llm")


def test_generate_eval_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("rag.yaml").write_text(f"blueprint: hybrid\ndata: {DATA}\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(cli_app, ["generate-eval", "--count", "6"])
    assert result.exit_code == 0, result.output
    assert "Wrote" in result.output
    assert "with access groups" in result.output
    assert len(load_dataset(tmp_path / "eval.jsonl")) >= 5
    again = runner.invoke(cli_app, ["generate-eval"])
    assert again.exit_code == 1
    assert "already exists" in again.output
    assert runner.invoke(cli_app, ["generate-eval", "--force", "-o", "g.jsonl"]).exit_code == 0
    evaluated = runner.invoke(cli_app, ["eval", "g.jsonl", "--data", str(DATA)])
    assert evaluated.exit_code == 0, evaluated.output
    assert "hit_rate@k" in evaluated.output
    Path("router.yaml").write_text(
        "router: {routes: [{name: a, config: rag.yaml}]}\n", encoding="utf-8"
    )
    routed = runner.invoke(cli_app, ["generate-eval", "-c", "router.yaml", "-o", "r.jsonl"])
    assert routed.exit_code == 1
    assert "per route" in routed.output
