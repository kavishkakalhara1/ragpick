"""Pipeline configuration: Pydantic schema plus the YAML loader.

Layers are merged in this order (later wins):

1. blueprint defaults (``blueprint: hybrid``)
2. the user's YAML file
3. explicit overrides (CLI ``--set key.path=value`` flags)

Within ``components``, an override that changes a component's ``type`` replaces
that component entirely (the old type's params would not apply); an override with
the same type merges ``params`` key by key.

``${ENV_VAR}`` and ``${ENV_VAR:-default}`` are interpolated in memory only; the
resolved values are never written back to disk.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, model_validator

from ragpick.errors import ConfigError

CorpusSize = Literal["small", "medium", "large"]
Latency = Literal["low", "normal"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentSpec(_Strict):
    """A component reference: registry ``type`` plus constructor ``params``."""

    type: str = Field(description="Registered component name, e.g. `openai` or `bm25`.")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Keyword arguments passed to the component."
    )


class Requirements(_Strict):
    """What the application needs. Used by the selector (`ragpick recommend`)."""

    data_types: list[str] = Field(
        default_factory=lambda: ["markdown"],
        description="Kinds of source data, e.g. markdown, pdf, html, sql, csv.",
    )
    corpus_size: CorpusSize = Field(
        "small", description="small (<1k chunks), medium, or large (>100k chunks)."
    )
    keyword_heavy: bool = Field(
        False, description="Queries rely on exact terms: product codes, legal terms, IDs."
    )
    long_documents: bool = Field(
        False,
        description="Documents are long and structured (manuals, policies, contracts).",
    )
    strict_grounding: bool = Field(
        False,
        description='Wrong answers are costly: prefer "I don\'t know" over guessing.',
    )
    latency: Latency = Field("normal", description="`low` disables expensive stages.")
    needs_citations: bool = Field(True, description="Answers must cite their sources.")
    access_control: bool = Field(
        False, description="Documents are restricted to groups (`allowed_groups` metadata)."
    )


class ComponentsConfig(_Strict):
    """The components that make up a pipeline."""

    loader: ComponentSpec = Field(description="Turns a path into documents.")
    chunker: ComponentSpec = Field(description="Splits documents into chunks.")
    embedder: ComponentSpec = Field(description="Embeds chunks and queries.")
    store: ComponentSpec = Field(description="Vector store.")
    keyword_index: ComponentSpec | None = Field(
        None, description="Keyword index (required by the `hybrid` and `keyword` retrievers)."
    )
    retriever: ComponentSpec = Field(description="Retrieval strategy.")
    reranker: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(type="none"), description="Optional reranker."
    )
    generator: ComponentSpec = Field(description="Answer generator (LLM).")


class QueryConfig(_Strict):
    """Query-time settings."""

    top_k: int = Field(5, ge=1, description="Number of contexts passed to the generator.")


class PriceSpec(_Strict):
    """Price of a model in USD per one million tokens."""

    prompt_per_1m: float = Field(0.0, ge=0)
    completion_per_1m: float = Field(0.0, ge=0)


class EvalConfig(_Strict):
    """Evaluation settings."""

    judge: ComponentSpec | None = Field(
        None, description="Generator used as LLM judge (must support `complete`)."
    )
    prices: dict[str, PriceSpec] = Field(
        default_factory=dict,
        description="Model name → price; merged over ragpick's built-in price table.",
    )


class TracingConfig(_Strict):
    """Query tracing (nested, timed spans per stage)."""

    enabled: bool = Field(False, description="Record a trace for every query.")
    exporters: list[ComponentSpec] = Field(
        default_factory=list,
        description="Where traces go: `jsonl` (params: path) and/or `console`.",
    )


class CacheConfig(_Strict):
    """Persistent cache for embeddings and LLM calls."""

    enabled: bool = Field(False, description="Turn caching on.")
    dir: str = Field(".ragpick/cache", description="Cache directory.")
    embeddings: bool = Field(True, description="Cache embeddings.")
    generations: bool = Field(True, description="Cache answers and LLM completions.")
    ttl_seconds: float | None = Field(None, gt=0, description="Expire entries after this age.")


class ApiKeySpec(_Strict):
    """An API key for `ragpick serve` and the groups its holder may act as."""

    key: str = Field(min_length=16, description="The secret; use `${ENV_VAR}` interpolation.")
    name: str = Field("client", description="Label used in logs.")
    groups: list[str] | None = Field(
        None,
        description="Groups this key may query as (`null` = any groups the request names).",
    )


class ServeConfig(_Strict):
    """HTTP API settings for `ragpick serve`."""

    host: str = Field("127.0.0.1", description="Interface to bind.")
    port: int = Field(8000, ge=1, le=65535, description="Port to bind.")
    api_keys: list[ApiKeySpec] = Field(
        default_factory=list,
        description="If set, requests need `Authorization: Bearer <key>`.",
    )
    max_body_bytes: int = Field(65536, ge=1024, description="Largest accepted request body.")
    cors_origins: list[str] = Field(
        default_factory=list, description="Origins allowed by CORS (`*` for any)."
    )
    allow_traces: bool = Field(False, description="Let clients request traces in responses.")


class RagConfig(_Strict):
    """Top-level schema of ``rag.yaml``."""

    version: Literal[1] = Field(1, description="Config schema version.")
    name: str = Field("ragpick-app", description="Application name.")
    blueprint: str | None = Field(None, description="Blueprint supplying default components.")
    data: str | None = Field(
        None, description="Default source path for `ragpick ingest` / `ragpick eval`."
    )
    requirements: Requirements = Field(
        default_factory=Requirements, description="Inputs for `ragpick recommend`."
    )
    components: ComponentsConfig = Field(
        description="Pipeline components (usually partly supplied by the blueprint)."
    )
    query: QueryConfig = Field(default_factory=QueryConfig, description="Query-time settings.")
    eval: EvalConfig = Field(default_factory=EvalConfig, description="Evaluation settings.")
    tracing: TracingConfig = Field(default_factory=TracingConfig, description="Query tracing.")
    cache: CacheConfig = Field(default_factory=CacheConfig, description="Caching.")
    serve: ServeConfig = Field(default_factory=ServeConfig, description="HTTP API settings.")
    index_dir: str = Field(".ragpick/index", description="Where indexes are persisted.")

    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)

    @property
    def base_dir(self) -> Path:
        """Directory that relative paths (``index_dir``, ``data``) resolve against."""
        return self._base_dir

    def with_base_dir(self, base_dir: Path) -> RagConfig:
        """Set the directory relative paths resolve against and return ``self``."""
        self._base_dir = base_dir
        return self

    def resolve_path(self, path: str) -> Path:
        """Resolve ``path`` relative to :attr:`base_dir` (absolute paths are kept)."""
        p = Path(path).expanduser()
        return p if p.is_absolute() else (self._base_dir / p)

    @property
    def index_path(self) -> Path:
        """Absolute index directory."""
        return self.resolve_path(self.index_dir)


class RouteSpec(_Strict):
    """One destination of a router: another pipeline config."""

    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$", description="Route name.")
    config: str = Field(description="Path to the route's rag.yaml (relative to this file).")
    description: str = Field("", description="What this route answers (shown to the LLM router).")
    keywords: list[str] = Field(
        default_factory=list, description="Words/phrases that point to this route."
    )
    groups: list[str] | None = Field(
        None, description="Only callers in these groups may be routed here."
    )


class RouterSpec(_Strict):
    """How questions are routed."""

    strategy: Literal["auto", "llm", "keywords"] = Field(
        "auto", description="`auto` uses the LLM when one is available, else keywords."
    )
    routes: list[RouteSpec] = Field(min_length=1, description="Possible destinations.")
    default: str | None = Field(None, description="Fallback route (default: the first).")
    fallback: bool = Field(
        True, description="Retry on the default route when the chosen one can't answer."
    )
    llm: ComponentSpec | None = Field(
        None, description="Generator used to classify questions (default: a route's LLM)."
    )


class RouterConfig(_Strict):
    """A config that routes each question to one of several pipelines."""

    version: Literal[1] = Field(1, description="Config schema version.")
    name: str = Field("ragpick-router", description="Application name.")
    router: RouterSpec
    eval: EvalConfig = Field(default_factory=EvalConfig, description="Evaluation settings.")
    tracing: TracingConfig = Field(default_factory=TracingConfig, description="Query tracing.")
    serve: ServeConfig = Field(default_factory=ServeConfig, description="HTTP API settings.")

    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)

    @model_validator(mode="after")
    def _check_routes(self) -> RouterConfig:
        names = [r.name for r in self.router.routes]
        if len(set(names)) != len(names):
            raise ValueError("route names must be unique")
        if self.router.default is not None and self.router.default not in names:
            raise ValueError(f"default route '{self.router.default}' is not one of {names}")
        return self

    @property
    def base_dir(self) -> Path:
        """Directory that route config paths resolve against."""
        return self._base_dir

    def with_base_dir(self, base_dir: Path) -> RouterConfig:
        """Set the base directory and return ``self``."""
        self._base_dir = base_dir
        return self

    def resolve_path(self, path: str) -> Path:
        """Resolve ``path`` relative to :attr:`base_dir`."""
        p = Path(path).expanduser()
        return p if p.is_absolute() else (self._base_dir / p)


# ---------------------------------------------------------------------------
# merging & interpolation
# ---------------------------------------------------------------------------
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def interpolate_env(value: Any, env: Mapping[str, str] | None = None, path: str = "") -> Any:
    """Recursively replace ``${VAR}`` / ``${VAR:-default}`` in string values.

    Raises:
        ConfigError: if a variable without default is unset (the key path is reported).
    """
    env = os.environ if env is None else env
    if isinstance(value, dict):
        return {k: interpolate_env(v, env, _join(path, k)) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env(v, env, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in env:
                return env[name]
            if default is not None:
                return default
            raise ConfigError(
                f"{path or '<root>'}: environment variable '{name}' is not set "
                f"(use ${{{name}:-default}} to provide a fallback)"
            )

        return _ENV_RE.sub(repl, value)
    return value


def _join(path: str, key: Any) -> str:
    return f"{path}.{key}" if path else str(key)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def merge_layers(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge two raw config layers, honouring the component ``type`` replacement rule."""
    override = dict(override)
    base_components = base.get("components") or {}
    over_components = override.pop("components", None)
    merged = deep_merge(base, override)
    if over_components is not None:
        if not isinstance(over_components, Mapping):
            merged["components"] = over_components  # let validation report it
            return merged
        components: dict[str, Any] = copy.deepcopy(dict(base_components))
        for name, spec in over_components.items():
            current = components.get(name)
            if (
                isinstance(spec, Mapping)
                and isinstance(current, Mapping)
                and spec.get("type", current.get("type")) == current.get("type")
            ):
                components[name] = deep_merge(current, spec)
            else:
                components[name] = copy.deepcopy(spec)
        merged["components"] = components
    return merged


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """Turn ``["query.top_k=3", "components.reranker.type=none"]`` into a nested dict.

    Values are parsed as YAML scalars/collections (``3`` → int, ``[a, b]`` → list).
    """
    result: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"Invalid override '{item}': expected KEY.PATH=VALUE")
        key, raw = item.split("=", 1)
        parts = [p for p in key.strip().split(".") if p]
        if not parts:
            raise ConfigError(f"Invalid override '{item}': empty key")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw) if raw.strip() else ""
    return result


def format_validation_error(exc: ValidationError, source: str, what: str = "configuration") -> str:
    """Render a Pydantic error as a list of ``key.path: message`` lines."""
    lines = [f"Invalid {what} in {source}:"]
    for err in exc.errors():
        loc = ""
        for part in err["loc"]:
            loc = f"{loc}[{part}]" if isinstance(part, int) else _join(loc, part)
        msg = err["msg"]
        if err["type"] == "extra_forbidden":
            msg = "unknown key (check for typos)"
        elif err["type"] == "missing":
            msg = "required key is missing"
            if loc.startswith("components."):
                msg += " (set it, or choose a `blueprint` that provides it)"
        lines.append(f"  - {loc or '<root>'}: {msg}")
    return "\n".join(lines)


def build_config(
    raw: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
    *,
    source: str = "<config>",
    base_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> RagConfig:
    """Validate a raw config mapping (blueprint → raw → overrides) into a :class:`RagConfig`."""
    from ragpick.blueprints import blueprint_defaults

    if not isinstance(raw, Mapping):
        raise ConfigError(f"{source}: top level must be a mapping, got {type(raw).__name__}")
    layered = merge_layers(raw, overrides or {})
    blueprint = layered.get("blueprint")
    if blueprint is not None:
        if not isinstance(blueprint, str):
            raise ConfigError(
                f"Invalid configuration in {source}:\n  - blueprint: must be a string"
            )
        layered = merge_layers(blueprint_defaults(blueprint), layered)
    layered = interpolate_env(layered, env)
    try:
        config = RagConfig.model_validate(layered)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc, source)) from None
    return config.with_base_dir(base_dir or Path.cwd())


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping from ``path`` with friendly errors."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p} (create one with `ragpick init`)")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p}: invalid YAML: {exc}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: top level must be a mapping")
    return data


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> RagConfig:
    """Load, merge, interpolate and validate ``rag.yaml``.

    Relative paths inside the config resolve against the config file's directory.
    """
    p = Path(path)
    return build_config(read_yaml(p), overrides, source=str(p), base_dir=p.resolve().parent)


def load_requirements(path: str | Path) -> Requirements:
    """Load only the ``requirements`` section (other sections need not be complete)."""
    raw = read_yaml(path).get("requirements") or {}
    try:
        return Requirements.model_validate(raw)
    except ValidationError as exc:
        message = format_validation_error(exc, str(path))
        raise ConfigError(message.replace("  - ", "  - requirements.")) from None


def is_router_config(raw: Mapping[str, Any]) -> bool:
    """Whether a raw config mapping is a router config (has a ``router`` key)."""
    return isinstance(raw, Mapping) and "router" in raw


def load_router_config(
    path: str | Path, overrides: Mapping[str, Any] | None = None
) -> RouterConfig:
    """Load and validate a router config file."""
    p = Path(path)
    raw = deep_merge(read_yaml(p), overrides or {})
    raw = interpolate_env(raw)
    try:
        config = RouterConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc, str(p))) from None
    return config.with_base_dir(p.resolve().parent)
