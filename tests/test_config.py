from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ragpick.config import (
    build_config,
    deep_merge,
    interpolate_env,
    load_config,
    load_requirements,
    merge_layers,
    parse_overrides,
)
from ragpick.errors import BlueprintError, ConfigError

FULL = {
    "version": 1,
    "name": "bot",
    "components": {
        "loader": {"type": "directory"},
        "chunker": {"type": "recursive", "params": {"chunk_size": 200, "overlap": 20}},
        "embedder": {"type": "fake"},
        "store": {"type": "memory"},
        "retriever": {"type": "vector"},
        "generator": {"type": "fake"},
    },
}


def write(tmp_path: Path, data: object, name: str = "rag.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_full_config_without_blueprint(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, FULL))
    assert cfg.name == "bot"
    assert cfg.components.reranker.type == "none"
    assert cfg.components.keyword_index is None
    assert cfg.query.top_k == 5
    assert cfg.base_dir == tmp_path.resolve()
    assert cfg.index_path == tmp_path.resolve() / ".ragpick/index"


def test_blueprint_defaults_fill_components(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, {"blueprint": "hybrid"}))
    assert cfg.components.retriever.type == "hybrid"
    assert cfg.components.keyword_index is not None
    assert cfg.components.keyword_index.type == "bm25"
    assert cfg.components.retriever.params == {"k": 20, "rrf_k": 60}


def test_merge_order_blueprint_user_overrides(tmp_path: Path) -> None:
    raw = {
        "blueprint": "hybrid",
        "components": {"retriever": {"params": {"k": 50}}},
        "query": {"top_k": 3},
    }
    cfg = load_config(write(tmp_path, raw), overrides=parse_overrides(["query.top_k=7"]))
    # user params merge with blueprint params when the type is unchanged
    assert cfg.components.retriever.params == {"k": 50, "rrf_k": 60}
    assert cfg.components.retriever.type == "hybrid"
    # CLI override wins over user config
    assert cfg.query.top_k == 7


def test_changing_component_type_replaces_params(tmp_path: Path) -> None:
    raw = {
        "blueprint": "naive",
        "components": {"embedder": {"type": "openai", "params": {"model": "m"}}},
    }
    cfg = load_config(write(tmp_path, raw))
    assert cfg.components.embedder.type == "openai"
    assert cfg.components.embedder.params == {"model": "m"}  # no leftover `dimension`


def test_override_can_switch_blueprint(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, {"blueprint": "naive"}), {"blueprint": "hybrid"})
    assert cfg.components.retriever.type == "hybrid"


def test_missing_component_reports_key_path(tmp_path: Path) -> None:
    raw = {"components": {"loader": {"type": "directory"}}}
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, raw))
    message = str(exc.value)
    assert "components.chunker: required key is missing" in message
    assert "blueprint" in message


def test_typo_reports_unknown_key_path(tmp_path: Path) -> None:
    raw = {"blueprint": "naive", "query": {"topk": 3}}
    with pytest.raises(ConfigError, match=r"query\.topk: unknown key"):
        load_config(write(tmp_path, raw))


def test_wrong_type_reports_key_path(tmp_path: Path) -> None:
    raw = {"blueprint": "naive", "query": {"top_k": 0}, "requirements": {"corpus_size": "huge"}}
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, raw))
    assert "query.top_k" in str(exc.value)
    assert "requirements.corpus_size" in str(exc.value)


def test_list_index_in_error_path() -> None:
    raw = {"blueprint": "naive", "requirements": {"data_types": ["md", 3]}}
    with pytest.raises(ConfigError, match=r"requirements\.data_types\[1\]"):
        build_config(raw)


def test_unknown_and_planned_blueprints(tmp_path: Path) -> None:
    with pytest.raises(BlueprintError, match="Unknown blueprint 'nope'"):
        load_config(write(tmp_path, {"blueprint": "nope"}))
    with pytest.raises(BlueprintError, match="planned"):
        load_config(write(tmp_path, {"blueprint": "multimodal"}))
    with pytest.raises(ConfigError, match="blueprint: must be a string"):
        build_config({"blueprint": 3})


def test_unsupported_version(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="version"):
        load_config(write(tmp_path, {**FULL, "version": 2}))


def test_env_interpolation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MY_MODEL", "text-embedding-3-small")
    raw = {
        "blueprint": "naive",
        "name": "${APP_NAME:-default-app}",
        "components": {"embedder": {"type": "fake", "params": {"model": "${MY_MODEL}"}}},
    }
    path = write(tmp_path, raw)
    cfg = load_config(path)
    assert cfg.components.embedder.params["model"] == "text-embedding-3-small"
    assert cfg.name == "default-app"
    # the file on disk is untouched: secrets are never written back
    assert "${MY_MODEL}" in path.read_text(encoding="utf-8")


def test_env_interpolation_missing_variable_reports_path() -> None:
    with pytest.raises(ConfigError, match=r"components\.generator\.params\.api_key.*SECRET_X"):
        interpolate_env(
            {"components": {"generator": {"params": {"api_key": "${SECRET_X}"}}}}, env={}
        )
    assert interpolate_env(["${A}", 1], env={"A": "x"}) == ["x", 1]


def test_file_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(bad)
    lst = tmp_path / "list.yaml"
    lst.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(lst)
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="components"):
        load_config(empty)
    with pytest.raises(ConfigError, match="mapping"):
        build_config([1, 2])  # type: ignore[arg-type]


def test_parse_overrides() -> None:
    assert parse_overrides(["query.top_k=3", "components.reranker.type=none", "name="]) == {
        "query": {"top_k": 3},
        "components": {"reranker": {"type": "none"}},
        "name": "",
    }
    assert parse_overrides(["a.b=[x, y]"]) == {"a": {"b": ["x", "y"]}}
    with pytest.raises(ConfigError, match=r"KEY\.PATH=VALUE"):
        parse_overrides(["oops"])
    with pytest.raises(ConfigError, match="empty key"):
        parse_overrides(["=3"])


def test_deep_merge_and_merge_layers() -> None:
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}
    base = {"components": {"x": {"type": "t", "params": {"p": 1}}}}
    assert merge_layers(base, {"components": {"x": {"params": {"q": 2}}}}) == {
        "components": {"x": {"type": "t", "params": {"p": 1, "q": 2}}}
    }
    assert merge_layers(base, {"components": "oops"}) == {"components": "oops"}


def test_resolve_path_absolute(tmp_path: Path) -> None:
    cfg = build_config({"blueprint": "naive"}, base_dir=tmp_path)
    assert cfg.resolve_path("/abs/x") == Path("/abs/x")
    assert cfg.resolve_path("rel") == tmp_path / "rel"


def test_load_requirements(tmp_path: Path) -> None:
    path = write(tmp_path, {"requirements": {"keyword_heavy": True}})
    req = load_requirements(path)
    assert req.keyword_heavy
    assert req.corpus_size == "small"
    bad = write(tmp_path, {"requirements": {"latency": "fast"}}, "bad.yaml")
    with pytest.raises(ConfigError, match=r"requirements\.latency"):
        load_requirements(bad)
