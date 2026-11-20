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
from rich.tree import Tree

from ragpick import __version__
from ragpick.blueprints import blueprint_defaults, list_blueprints
from ragpick.config import (
    Requirements,
    is_router_config,
    load_config,
    load_requirements,
    load_router_config,
    parse_overrides,
    read_yaml,
)
from ragpick.errors import RagpickError
from ragpick.eval.runner import (
    METRICS,
    EvalReport,
    evaluate_configs,
    parse_judge_option,
    write_reports,
)
from ragpick.router import Router, load_engine
from ragpick.selector import Recommendation, recommend
from ragpick.tracing import Span

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
    long_docs = _confirm("Are documents long and structured (manuals, policies, contracts)?", False)
    multi_hop = _confirm("Do questions connect facts across several documents?", False)
    complex_q = _confirm("Are questions often multi-part or comparisons (A vs B)?", False)
    tenants = _confirm("Must several customers' data be kept separate (multi-tenant)?", False)
    strict = _confirm('Is a wrong answer worse than "I don\'t know"?', False)
    latency = _ask("Latency requirement", "normal", ["low", "normal"])
    citations = _confirm("Must answers cite their sources?", True)
    acl = _confirm("Are some documents restricted to certain user groups?", False)
    return Requirements.model_validate(
        {
            "data_types": [d.strip() for d in data_types.split(",") if d.strip()],
            "corpus_size": corpus,
            "keyword_heavy": keyword,
            "long_documents": long_docs,
            "multi_hop": multi_hop,
            "complex_questions": complex_q,
            "multi_tenant": tenants,
            "strict_grounding": strict,
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
    if requirements.multi_tenant:
        data["tenancy"] = {"enabled": True}
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
    if blueprint == "sql" and provider == "fake":
        console.print(
            "  [yellow]![/] The sql blueprint needs a real LLM to write queries: "
            "re-run with --provider openai or ollama."
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
    tenant: Annotated[str | None, typer.Option(help="Tenant to ingest for (multi-tenant).")] = None,
    set_: SetOpt = None,
) -> None:
    """Load, chunk, embed, index and persist documents (every route, for a router)."""
    engine = load_engine(config, _overrides(set_))
    if isinstance(engine, Router):
        if tenant is not None:
            raise RagpickError("Routers can't ingest per tenant; ingest each route config")
        if path is not None:
            raise RagpickError("A router ingests each route's own `data`; run without PATH")
        for route_name, route_stats in engine.ingest().items():
            console.print(
                f"[green]✓[/] {route_name}: {route_stats.documents} documents → "
                f"{route_stats.chunks} chunks (index: {route_stats.index_dir})"
            )
        return
    pipeline = engine
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

        stats = pipeline.ingest(path, progress=on_progress, tenant=tenant)
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
    trace: Annotated[bool, typer.Option("--trace", help="Show timings of every stage.")] = False,
    stream: Annotated[
        bool, typer.Option("--stream", help="Print the answer as it is generated.")
    ] = False,
    tenant: Annotated[str | None, typer.Option(help="Tenant to query (multi-tenant).")] = None,
    set_: SetOpt = None,
) -> None:
    """Answer a question from the index (or the routes of a router config)."""
    engine = load_engine(config, _overrides(set_))
    if stream:
        if trace:
            raise RagpickError("--stream and --trace can't be combined")
        final = None
        events = engine.stream(question, groups=_split_groups(groups), top_k=top_k, tenant=tenant)
        for event in events:
            if event.type == "delta":
                console.print(event.text, end="", markup=False, highlight=False, soft_wrap=True)
            elif event.type == "answer":
                final = event.answer
        console.print()
        if final is not None:
            if final.route:
                console.print(f"[dim]route: {escape(final.route)}[/]")
            if final.citations:
                console.print("[bold]Citations:[/] " + ", ".join(final.citations), highlight=False)
        return
    answer = engine.query(
        question,
        groups=_split_groups(groups),
        top_k=top_k,
        trace=True if trace else None,
        tenant=tenant,
    )
    title = f"Answer · route: {answer.route}" if answer.route else "Answer"
    console.print(Panel(Text(answer.text), title=title, expand=False))
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
    if trace and answer.trace is not None:
        console.print(_trace_tree(answer.trace))
    usage = answer.usage
    cached = " · cached" if usage.cached else ""
    console.print(
        f"[dim]{usage.latency_ms:.0f} ms · {usage.prompt_tokens} prompt + "
        f"{usage.completion_tokens} completion tokens{cached}[/]"
    )


def _trace_tree(root: Span) -> Tree:
    def label(s: Span) -> Text:
        text = Text(f"{s.name} ", style="bold")
        text.append(f"{s.duration_ms:.1f} ms", style="cyan")
        attrs = " ".join(f"{k}={v}" for k, v in s.attributes.items() if k != "sql")
        if attrs:
            text.append(f"  {attrs}", style="dim")
        if "sql" in s.attributes:
            text.append(f"\n{s.attributes['sql']}", style="green")
        if s.error:
            text.append(f"  {s.error}", style="red")
        return text

    def add(node: Tree, s: Span) -> None:
        for child in s.children:
            add(node.add(label(child)), child)

    tree = Tree(label(root))
    add(tree, root)
    return tree


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


# --------------------------------------------------------------------------- generate-eval
@app.command(name="generate-eval")
@friendly
def generate_eval(
    path: Annotated[
        Path | None, typer.Argument(help="Documents to generate from (default: config `data`).")
    ] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
    output: Annotated[Path, typer.Option("--output", "-o", help="JSONL file to write.")] = Path(
        "eval.jsonl"
    ),
    count: Annotated[int, typer.Option(min=1, help="Number of questions.")] = 20,
    seed: Annotated[int, typer.Option(help="Sampling seed (same seed → same chunks).")] = 0,
    mode: Annotated[str, typer.Option(help="auto, llm or heuristic.")] = "auto",
    force: Annotated[bool, typer.Option(help="Overwrite the output file.")] = False,
    set_: SetOpt = None,
) -> None:
    """Generate an evaluation dataset (questions + reference answers) from your documents."""
    from ragpick.eval.generate import generate_dataset, write_dataset

    if output.exists() and not force:
        raise RagpickError(f"{output} already exists (use --force to overwrite)")
    engine = load_engine(config, _overrides(set_))
    if isinstance(engine, Router):
        raise RagpickError("Generate a dataset per route config, not for a router")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Generating questions", total=count)
        examples = generate_dataset(
            engine,
            path,
            count=count,
            seed=seed,
            mode=mode,
            progress=lambda done, total: progress.update(task, completed=done),
        )
    written = write_dataset(examples, output)
    restricted = sum(1 for e in examples if e.groups)
    console.print(
        f"[green]✓[/] Wrote {len(examples)} questions to {written}"
        + (f" ({restricted} with access groups)" if restricted else "")
    )
    console.print(
        "  Review them, then: ragpick eval " + escape(str(written)) + " -c " + escape(str(config))
    )


# --------------------------------------------------------------------------- bench
@app.command()
@friendly
def bench(
    dataset: Annotated[Path, typer.Argument(help="JSONL dataset of questions.")],
    data: Annotated[Path, typer.Option(help="Documents to index for every blueprint.")],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", "-c", help="Base config for providers (default: rag.yaml if present)."
        ),
    ] = None,
    blueprint: Annotated[
        list[str] | None,
        typer.Option("--blueprint", "-b", help="Blueprint to include (repeatable; default: all)."),
    ] = None,
    metric: Annotated[str, typer.Option(help="Metric that picks the winner.")] = "mrr",
    top_k: Annotated[
        int | None, typer.Option(help="Contexts per query for every blueprint.")
    ] = None,
    judge: Annotated[
        str | None, typer.Option(help="LLM judge, e.g. openai:gpt-4o-mini (overrides eval.judge).")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write results (.json or .csv).")
    ] = None,
    write: Annotated[
        Path | None, typer.Option(help="Write the winning blueprint's config to this file.")
    ] = None,
) -> None:
    """Run every stable blueprint on your data and recommend the best one."""
    from ragpick.eval.bench import run_bench

    if config is None and DEFAULT_CONFIG.exists():
        config = DEFAULT_CONFIG
    if config is None:
        console.print("[dim]No base config; using fake offline components.[/]")
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
                tasks[label] = progress.add_task(f"Benchmarking {label}", total=total)
            progress.update(tasks[label], completed=done)

        result = run_bench(
            dataset,
            data,
            base_config=config,
            blueprints=blueprint,
            metric=metric,
            top_k=top_k,
            judge=parse_judge_option(judge),
            progress=on_progress,
        )
    console.print(_results_table(result.reports))
    best = result.report_for(result.winner)
    score = _fmt_metric(result.metric, best.metrics[result.metric])
    ranking = " > ".join(result.ranking)
    console.print(
        Panel(
            f"[bold]Winner:[/] [cyan]{escape(result.winner)}[/] "
            f"({escape(result.metric)} = {score})\n"
            f"[bold]Ranking:[/] {escape(ranking)}",
            title="ragpick bench",
            expand=False,
        )
    )
    if output is not None:
        console.print(f"[green]✓[/] Wrote {write_reports(result.reports, output)}")
    if write is not None:
        if write.exists():
            raise RagpickError(f"{write} already exists; choose another --write path")
        write.write_text(result.winner_config, encoding="utf-8")
        console.print(f"[green]✓[/] Wrote {write} (blueprint {result.winner})")


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
    """Validate a config (pipeline or router) and print the fully merged result."""
    if is_router_config(read_yaml(config)):
        router_cfg = load_router_config(config, _overrides(set_))
        for route in router_cfg.router.routes:  # each route must be valid too
            load_config(router_cfg.resolve_path(route.config))
        console.print(
            f"[green]✓[/] {config} is a valid router ({len(router_cfg.router.routes)} routes)"
        )
        merged = _mask_secrets(router_cfg.model_dump(mode="json"))
    else:
        cfg = load_config(config, _overrides(set_))
        console.print(f"[green]✓[/] {config} is valid")
        merged = _mask_secrets(cfg.model_dump(mode="json"))
    console.print(yaml.safe_dump(merged, sort_keys=False), highlight=False, markup=False)


# --------------------------------------------------------------------------- serve
@app.command()
@friendly
def serve(
    config: ConfigOpt = DEFAULT_CONFIG,
    host: Annotated[str | None, typer.Option(help="Interface (default: serve.host).")] = None,
    port: Annotated[int | None, typer.Option(help="Port (default: serve.port).")] = None,
    server: Annotated[
        str, typer.Option(help="auto (uvicorn if installed), uvicorn, or stdlib.")
    ] = "auto",
    set_: SetOpt = None,
) -> None:
    """Serve the pipeline (or router) as a JSON HTTP API."""
    import importlib.util

    from ragpick.server import App, asgi_app, is_loopback, make_server

    if server not in ("auto", "uvicorn", "stdlib"):
        raise RagpickError(f"--server must be auto, uvicorn or stdlib, got {server!r}")
    engine = load_engine(config, _overrides(set_))
    application = App(engine)
    bind_host = host or application.config.host
    bind_port = port or application.config.port
    if not engine.ready():
        console.print("[yellow]![/] Nothing ingested yet: /query will return 503 until you ingest.")
    if not application.config.api_keys and not is_loopback(bind_host):
        console.print(
            "[bold yellow]Warning:[/] no serve.api_keys configured and the server is reachable "
            f"on {bind_host}. Anyone who can connect may query as any group."
        )
    use_uvicorn = server == "uvicorn" or (
        server == "auto" and importlib.util.find_spec("uvicorn") is not None
    )
    url = f"http://{bind_host}:{bind_port}"
    console.print(f"[green]✓[/] Serving [cyan]{escape(engine.name)}[/] on {url}  (POST /query)")
    if use_uvicorn:
        from ragpick.components._optional import require

        uvicorn = require(
            "uvicorn", "serve", "ragpick serve --server uvicorn", "pip install uvicorn"
        )
        uvicorn.run(asgi_app(application), host=bind_host, port=bind_port, log_level="info")
        return
    console.print("[dim]Using the standard-library server (pip install uvicorn for production).[/]")
    httpd = make_server(application, bind_host, bind_port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    finally:
        httpd.server_close()


# --------------------------------------------------------------------------- cache
cache_app = typer.Typer(help="Inspect or clear the embedding/LLM cache.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")


def _open_cache(config: Path) -> Any:
    from ragpick.cache import Cache

    cfg = load_config(config)
    path = cfg.resolve_path(cfg.cache.dir) / "cache.sqlite"
    if not path.exists():
        raise RagpickError(
            f"No cache at {path}. Enable it with `cache: {{enabled: true}}` in {config}."
        )
    return Cache(path)


@cache_app.command("stats")
@friendly
def cache_stats(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Show how many entries the cache holds."""
    cache = _open_cache(config)
    try:
        stats = cache.stats()
    finally:
        cache.close()
    table = Table(title=f"Cache ({cache.path})")
    table.add_column("Kind")
    table.add_column("Entries", justify="right")
    for kind, count in stats.items():
        table.add_row(kind, str(count), style="bold" if kind == "total" else None)
    console.print(table)


@cache_app.command("clear")
@friendly
def cache_clear(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Delete every cache entry."""
    cache = _open_cache(config)
    try:
        removed = cache.clear()
    finally:
        cache.close()
    console.print(f"[green]✓[/] Removed {removed} cache entries")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
