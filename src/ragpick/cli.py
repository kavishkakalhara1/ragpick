"""The ``ragpick`` command-line interface (typer + rich)."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, ParamSpec, TypeVar

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from ragpick import __version__
from ragpick.blueprints import blueprint_defaults, list_blueprints
from ragpick.config import Requirements, load_config, load_requirements, parse_overrides, read_yaml
from ragpick.errors import RagpickError
from ragpick.eval.runner import (
    METRICS,
    EvalReport,
    evaluate_configs,
    parse_judge_option,
    write_reports,
)
from ragpick.pipeline import Pipeline
from ragpick.selector import Recommendation, recommend

app = typer.Typer(
    name="ragpick",
    help="Choose, build, evaluate, and run the right RAG architecture.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
)
console = Console()
err_console = Console(stderr=True, soft_wrap=True)
_state: dict[str, bool] = {"verbose": False}

P = ParamSpec("P")
R = TypeVar("R")

DEFAULT_CONFIG = Path("rag.yaml")
ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Pipeline config file.")]
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", help="Override a config value, e.g. --set query.top_k=3 (repeatable)."),
]

PROVIDERS: dict[str, dict[str, Any]] = {
    "fake": {
        "embedder": {"type": "fake", "params": {"dimension": 256}},
        "generator": {"type": "fake"},
    },
    "openai": {
        "embedder": {"type": "openai", "params": {"model": "text-embedding-3-small"}},
        "generator": {"type": "openai", "params": {"model": "gpt-4o-mini", "temperature": 0}},
    },
    "ollama": {
        "embedder": {
            "type": "sentence_transformers",
            "params": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        },
        "generator": {"type": "ollama", "params": {"model": "llama3.1"}},
    },
}


def friendly(func: Callable[P, R]) -> Callable[P, R]:
    """Turn exceptions into one-line errors (full traceback with ``--verbose``)."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except (typer.Exit, typer.Abort):
            raise
        except (RagpickError, FileNotFoundError, ValueError) as exc:
            if _state["verbose"]:
                err_console.print_exception()
            err_console.print(f"[bold red]Error:[/] {escape(str(exc))}", highlight=False)
            raise typer.Exit(1) from None
        except Exception as exc:
            if _state["verbose"]:
                err_console.print_exception()
            else:
                err_console.print(
                    f"[bold red]Unexpected error:[/] {type(exc).__name__}: {escape(str(exc))}\n"
                    "Run again with --verbose for details.",
                    highlight=False,
                )
            raise typer.Exit(1) from None

    return wrapper


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"ragpick {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Debug logging and full tracebacks.")
    ] = False,
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = None,
) -> None:
    """ragpick: choose, build, evaluate, and run the right RAG architecture."""
    _state["verbose"] = verbose
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=err_console, show_path=verbose, show_time=verbose)],
        force=True,
    )


def _overrides(items: list[str] | None) -> dict[str, Any]:
    return parse_overrides(items or [])


def _print_recommendation(rec: Recommendation) -> None:
    body = [f"[bold]Blueprint:[/] [cyan]{rec.blueprint}[/]"]
    if rec.reasons:
        body.append("\n[bold]Why[/]")
        body += [f"  • {escape(r)}" for r in rec.reasons]
    if rec.overrides:
        body.append("\n[bold]Component overrides[/]")
        body += [f"  • {escape(k)}: {escape(str(v))}" for k, v in rec.overrides.items()]
    if rec.alternatives:
        body.append("\n[bold]Alternatives[/]")
        body += [f"  • {escape(a)}" for a in rec.alternatives]
    if rec.warnings:
        body.append("\n[bold yellow]Warnings[/]")
        body += [f"  [yellow]![/] {escape(w)}" for w in rec.warnings]
    console.print(Panel("\n".join(body), title="ragpick recommends", expand=False))


# --------------------------------------------------------------------------- init
def _ask(question: str, default: str, choices: list[str] | None = None) -> str:
    return Prompt.ask(question, default=default, choices=choices, console=console)


def _confirm(question: str, default: bool) -> bool:
    return Confirm.ask(question, default=default, console=console)


def _ask_requirements() -> Requirements:
    data_types = _ask("Data types (comma separated)", "markdown")
    corpus = _ask(
        "Corpus size: small (<1k chunks), medium, large (>100k)",
        "small",
        ["small", "medium", "large"],
    )
    keyword = _confirm("Do queries use exact terms (product codes, IDs, legal terms)?", False)
    latency = _ask("Latency requirement", "normal", ["low", "normal"])
    citations = _confirm("Must answers cite their sources?", True)
    acl = _confirm("Are some documents restricted to certain user groups?", False)
    return Requirements.model_validate(
        {
            "data_types": [d.strip() for d in data_types.split(",") if d.strip()],
            "corpus_size": corpus,
            "keyword_heavy": keyword,
            "latency": latency,
            "needs_citations": citations,
            "access_control": acl,
        }
    )


def _render_config(
    name: str, blueprint: str, requirements: Requirements, provider: str, overrides: dict[str, Any]
) -> str:
    components: dict[str, Any] = {**PROVIDERS[provider], **overrides}
    data: dict[str, Any] = {
        "version": 1,
        "name": name,
        "blueprint": blueprint,
        "requirements": requirements.model_dump(),
        "components": components,
        "query": {"top_k": 5},
        "index_dir": ".ragpick/index",
    }
    header = (
        "# ragpick pipeline config. Docs: https://kavishkakalhara1.github.io/ragpick/config-reference/\n"
        f"# Components not listed here come from the '{blueprint}' blueprint.\n"
        "# Secrets: use ${ENV_VAR} interpolation; never put API keys in this file.\n"
    )
    return header + str(yaml.safe_dump(data, sort_keys=False))


@app.command()
@friendly
def init(
    blueprint: Annotated[
        str | None, typer.Option("--blueprint", "-b", help="Blueprint to use (skip the selector).")
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(help="Model provider: fake (offline), openai, or ollama."),
    ] = None,
    name: Annotated[str | None, typer.Option(help="Application name.")] = None,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to write.")
    ] = DEFAULT_CONFIG,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Non-interactive; accept defaults.")
    ] = False,
    force: Annotated[bool, typer.Option(help="Overwrite an existing file.")] = False,
) -> None:
    """Answer a few questions and write a rag.yaml."""
    if output.exists() and not force:
        raise RagpickError(f"{output} already exists (use --force to overwrite)")
    if provider is not None and provider not in PROVIDERS:
        raise RagpickError(f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}")
    requirements = Requirements() if yes else _ask_requirements()
    rec = recommend(requirements)
    if blueprint is None:
        _print_recommendation(rec)
        blueprint = rec.blueprint if yes else _ask("Blueprint", rec.blueprint)
    blueprint_defaults(blueprint)  # validates the name (and rejects planned blueprints)
    if provider is None:
        provider = "fake" if yes else _ask("Provider", "fake", list(PROVIDERS))
    if name is None:
        name = "ragpick-app" if yes else _ask("Application name", "ragpick-app")
    overrides = rec.overrides if blueprint == rec.blueprint else {}
    output.write_text(
        _render_config(name, blueprint, requirements, provider, overrides), encoding="utf-8"
    )
    console.print(
        f"[green]✓[/] Wrote {output} (blueprint [cyan]{blueprint}[/], provider {provider})"
    )
    if provider == "openai":
        console.print("  Next: pip install 'ragpick[openai]' and export OPENAI_API_KEY")
    elif provider == "ollama":
        console.print("  Next: pip install 'ragpick[local]' and run `ollama serve`")
    console.print('  Then: ragpick ingest ./docs && ragpick query "your question"')


# --------------------------------------------------------------------------- recommend
def _store_type(path: Path) -> str:
    raw = read_yaml(path)
    store = ((raw.get("components") or {}).get("store") or {}).get("type")
    if store:
        return str(store)
    blueprint = raw.get("blueprint")
    if isinstance(blueprint, str):
        return str(blueprint_defaults(blueprint)["components"]["store"]["type"])
    return "memory"


@app.command(name="recommend")
@friendly
def recommend_cmd(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Read `requirements` from this file.")
    ] = None,
) -> None:
    """Recommend a blueprint from the config's requirements, with reasons."""
    if config is None and DEFAULT_CONFIG.exists():
        config = DEFAULT_CONFIG
    if config is None:
        console.print("[dim]No rag.yaml found; using default requirements.[/]")
        requirements, store = Requirements(), "memory"
    else:
        requirements, store = load_requirements(config), _store_type(config)
    _print_recommendation(recommend(requirements, store=store))


# --------------------------------------------------------------------------- blueprints
@app.command()
@friendly
def blueprints() -> None:
    """List all blueprints, including the roadmap."""
    table = Table(title="ragpick blueprints", show_lines=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Description")
    table.add_column("Best for")
    for bp in list_blueprints():
        status = "[green]stable[/]" if bp.is_stable else "[yellow]planned[/]"
        table.add_row(
            bp.name,
            status,
            bp.metadata.description,
            "\n".join(f"• {b}" for b in bp.metadata.best_for),
        )
    console.print(table)


# --------------------------------------------------------------------------- ingest
@app.command()
@friendly
def ingest(
    path: Annotated[
        Path | None, typer.Argument(help="File or directory to ingest (default: config `data`).")
    ] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
) -> None:
    """Load, chunk, embed, index and persist documents."""
    pipeline = Pipeline.from_yaml(config, _overrides(set_))
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Embedding chunks", total=None)

        def on_progress(stage: str, done: int, total: int) -> None:
            progress.update(task, completed=done, total=total)

        stats = pipeline.ingest(path, progress=on_progress)
    console.print(
        f"[green]✓[/] Ingested {stats.documents} documents → {stats.chunks} chunks "
        f"in {stats.seconds:.1f}s (index: {stats.index_dir})"
    )


# --------------------------------------------------------------------------- query
def _split_groups(groups: str | None) -> list[str] | None:
    if groups is None:
        return None
    return [g.strip() for g in groups.split(",") if g.strip()]


@app.command()
@friendly
def query(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    config: ConfigOpt = DEFAULT_CONFIG,
    groups: Annotated[
        str | None, typer.Option(help="Caller's groups for access control, e.g. support,hr.")
    ] = None,
    show_contexts: Annotated[
        bool, typer.Option("--show-contexts", help="Show retrieved contexts.")
    ] = False,
    top_k: Annotated[int | None, typer.Option(help="Number of contexts.")] = None,
    set_: SetOpt = None,
) -> None:
    """Answer a question from the index."""
    pipeline = Pipeline.from_yaml(config, _overrides(set_))
    answer = pipeline.query(question, groups=_split_groups(groups), top_k=top_k)
    console.print(Panel(Text(answer.text), title="Answer", expand=False))
    if answer.citations:
        console.print("[bold]Citations:[/] " + ", ".join(answer.citations), highlight=False)
    if show_contexts:
        table = Table(title="Contexts")
        table.add_column("#", justify="right")
        table.add_column("Chunk")
        table.add_column("Score", justify="right")
        table.add_column("Source")
        table.add_column("Text")
        for i, ctx in enumerate(answer.contexts, start=1):
            snippet = " ".join(ctx.chunk.text.split())
            snippet = snippet[:160] + ("…" if len(snippet) > 160 else "")
            table.add_row(str(i), Text(ctx.chunk.id), f"{ctx.score:.4f}", ctx.source, Text(snippet))
        console.print(table)
    usage = answer.usage
    console.print(
        f"[dim]{usage.latency_ms:.0f} ms · {usage.prompt_tokens} prompt + "
        f"{usage.completion_tokens} completion tokens[/]"
    )


# --------------------------------------------------------------------------- eval
_HIGHER_IS_BETTER = {"hit_rate@k", "mrr", "precision@k", "faithfulness", "answer_relevance"}
_LOWER_IS_BETTER = {"latency_p50_ms", "latency_p95_ms", "total_tokens", "est_cost_usd", "errors"}


def _fmt_metric(name: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    if name == "est_cost_usd":
        return f"${value:.4f}"
    if name in ("total_tokens", "errors"):
        return f"{value:.0f}"
    if name.startswith("latency"):
        return f"{value:.1f}"
    return f"{value:.3f}"


def _results_table(reports: list[EvalReport]) -> Table:
    """Metrics as rows, configs as columns (fits narrow terminals); best value in green."""
    table = Table(title="Evaluation results")
    table.add_column("Metric", style="bold", no_wrap=True)
    for r in reports:
        table.add_column(Text(f"{r.label}\n({r.blueprint or '-'})"), justify="right")
    for metric in METRICS:
        values = [r.metrics.get(metric) for r in reports]
        present = [v for v in values if v is not None]
        best: float | None = None
        if len(reports) > 1 and len(set(present)) > 1:
            best = max(present) if metric in _HIGHER_IS_BETTER else min(present)
        cells = []
        for value in values:
            text = _fmt_metric(metric, value)
            cells.append(f"[bold green]{text}[/]" if best is not None and value == best else text)
        table.add_row(metric, *cells)
    return table


@app.command(name="eval")
@friendly
def eval_cmd(
    dataset: Annotated[Path, typer.Argument(help="JSONL dataset of questions.")],
    config: Annotated[
        list[Path] | None,
        typer.Option("--config", "-c", help="Config to evaluate (repeat to compare)."),
    ] = None,
    data: Annotated[
        Path | None,
        typer.Option(help="Ingest this path into a temporary index for each config."),
    ] = None,
    judge: Annotated[
        str | None,
        typer.Option(help="LLM judge, e.g. openai:gpt-4o-mini (overrides eval.judge)."),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write results (.json or .csv).")
    ] = None,
) -> None:
    """Evaluate one or more configs on a dataset and compare them."""
    configs = config or [DEFAULT_CONFIG]
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        tasks: dict[str, Any] = {}

        def on_progress(label: str, done: int, total: int) -> None:
            if label not in tasks:
                tasks[label] = progress.add_task(f"Evaluating {label}", total=total)
            progress.update(tasks[label], completed=done)

        reports = evaluate_configs(
            configs, dataset, data=data, judge=parse_judge_option(judge), progress=on_progress
        )
    console.print(_results_table(reports))
    k_values = sorted({r.k for r in reports})
    console.print(
        f"[dim]{reports[0].num_examples} questions · k = {', '.join(map(str, k_values))} · "
        "best value per metric in green[/]"
    )
    if output is not None:
        written = write_reports(reports, output)
        console.print(f"[green]✓[/] Wrote {written}")


# --------------------------------------------------------------------------- config check
_SECRET_HINTS = ("key", "token", "secret", "password")


def _mask_secrets(value: Any, key: str = "") -> Any:
    """Replace values of secret-looking keys (api_key, token, ...) with ``***``."""
    if isinstance(value, dict):
        return {k: _mask_secrets(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask_secrets(v, key) for v in value]
    if value and isinstance(value, str) and any(h in key.lower() for h in _SECRET_HINTS):
        return "***"
    return value


@app.command(name="validate")
@friendly
def validate_cmd(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Validate a config and print the fully merged result."""
    cfg = load_config(config, _overrides(set_))
    console.print(f"[green]✓[/] {config} is valid")
    merged = _mask_secrets(cfg.model_dump(mode="json"))
    console.print(yaml.safe_dump(merged, sort_keys=False), highlight=False, markup=False)


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
