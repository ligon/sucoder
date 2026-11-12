"""Local dream entrypoint, independent of interactive agent startup."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Optional

import typer

from . import dream
from . import dream_run
from . import dream_wiki
from . import dream_budget
from . import dream_reader
from . import dream_reader_agent
from .config import ConfigError, load_config


app = typer.Typer(invoke_without_command=True,
    help="Explore the current project and prepare a reviewed companion-wiki candidate, or use explicit evidence commands.")


@app.callback()
def explore(
    ctx: typer.Context,
    harness: Optional[str] = typer.Option(None, "--harness", help="Native codex/claude; otherwise use configured defaults."),
    model: Optional[str] = typer.Option(None, "--model", help="Native model; otherwise use configured defaults."),
    purpose: Optional[str] = typer.Option(None, "--purpose", help="What the dream should help readers understand; topics remain agent-selected."),
    job_path: Optional[Path] = typer.Option(None, "--job", help="Resume this private exploratory job with its frozen settings and remaining allowance."),
    budget_path: Optional[Path] = typer.Option(None, "--budget", help="Explicit shared private budget directory for a new job."),
    attempt: int = typer.Option(1, "--attempt", min=1, max=1000000, help="Explicit native attempt within the same persisted job budget."),
    max_total_seconds: Optional[int] = typer.Option(None, "--max-total-seconds", min=1, help="Whole-job elapsed ceiling, including discovery and review."),
    max_total_calls: Optional[int] = typer.Option(None, "--max-total-calls", min=1, help="Whole-job native invocation ceiling, including discovery and review."),
    reserve_seconds: Optional[int] = typer.Option(None, "--reserve-seconds", min=0, help="Time reserved for review, repair and finalization."),
    reserve_calls: Optional[int] = typer.Option(None, "--reserve-calls", min=0, help="Invocations reserved for review, repair and finalization."),
    discovery_calls: Optional[int] = typer.Option(None, "--discovery-calls", min=1, help="Maximum native investigation turns before authorship."),
    discovery_bytes: Optional[int] = typer.Option(None, "--discovery-bytes", min=1, help="Maximum project-navigation bytes delivered during discovery."),
    max_total_tokens: Optional[int] = typer.Option(None, "--max-total-tokens", min=1, help="Requested hard token ceiling; current native adapters reject it before provider work."),
    max_cost_usd: Optional[str] = typer.Option(None, "--max-cost-usd", help="Requested hard dollar ceiling; current native adapters reject it before provider work."),
    detach: bool = typer.Option(False, "--detach", help="Launch the durable job in the background and print its location and process ID."),
    explore_project: bool = typer.Option(False, "--explore", help="Explore the committed project with open topic choice instead of dreaming over this mirror's sessions and notes."),
):
    """Without a subcommand, dream over this mirror's finished sessions and agent notes (or, with --explore, the project)."""
    if ctx.invoked_subcommand is not None:
        supplied = [parameter.name for parameter in ctx.command.params
                    if parameter.name is not None and
                    getattr(ctx.get_parameter_source(parameter.name), "name", None) == "COMMANDLINE"]
        if supplied:
            raise typer.BadParameter("Put explicit-command options after the subcommand; unused dream options: " + ", ".join(supplied))
        return
    from . import dream_explore
    overrides = dict(harness=harness, model=model, purpose=purpose,
        max_total_seconds=max_total_seconds, max_total_calls=max_total_calls,
        reserve_seconds=reserve_seconds, reserve_calls=reserve_calls,
        max_discovery_calls=discovery_calls, max_discovery_bytes=discovery_bytes,
        max_total_tokens=max_total_tokens, max_cost_usd=max_cost_usd,
        mode="project" if explore_project else None)
    overrides = {key: value for key, value in overrides.items() if value is not None}
    if detach:
        _data_command(ctx, "dream_job", _detach, ctx.obj["config_path"], job_path,
            overrides=overrides, budget_path=budget_path, attempt=attempt)
    else:
        _data_command(ctx, "dream_job", dream_explore.start, job_path,
            overrides=overrides, budget_path=budget_path, attempt=attempt)


def _detach(config, config_path, job_path=None, *, overrides=None, budget_path=None, attempt=1):
    """Start the same persisted job without a terminal or parent-process stdin."""
    from . import dream_explore
    job = dream_explore.prepare(config, job_path, overrides=overrides, budget_path=budget_path, attempt=attempt)
    directory = Path(job["job_path"])
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    handles = []
    try:
        info = os.fstat(fd)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise dream.DreamError("Detached job directory must be private and account-owned", 3)
        for name in ("worker.stdout.log", "worker.stderr.log"):
            handle = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600, dir_fd=fd)
            try:
                info = os.fstat(handle)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
                    raise dream.DreamError("Detached logs must be private regular files owned by this account", 3)
                handles.append(os.fdopen(handle, "ab"))
            except BaseException:
                os.close(handle)
                raise
        argv = [sys.executable, "-c", "from sucoder.cli import app; app()",
            "--config", str(Path(config_path).expanduser().absolute()), "dream", "--job", str(directory),
            "--attempt", str(attempt)]
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=handles[0], stderr=handles[1],
            start_new_session=True, close_fds=True)
        return dict(job, schema_version=1, operation="dream_job", state="detached", pid=process.pid,
            stdout_path=str(directory / "worker.stdout.log"), stderr_path=str(directory / "worker.stderr.log"))
    finally:
        for stream in handles:
            stream.close()
        os.close(fd)


@app.command("plan")
def plan(
    ctx: typer.Context,
    mirror: str = typer.Argument(..., help="Configured mirror available on this host."),
    sources: Path = typer.Option(..., "--sources", help="Explicit version-1 JSON source catalog; git, claude-jsonl or codex-jsonl units."),
    base: str = typer.Option(..., "--base", help="Full integration ref matching the catalog, e.g. refs/heads/main."),
    maintenance: Optional[str] = typer.Option(None, "--maintenance", help="Explicit maintenance reason; requires --topic."),
    topic: Optional[str] = typer.Option(None, "--topic", help="Topic name for an explicit maintenance pass."),
):
    """Write a private plan and print JSON. Exit 2: invalid input; 3: unavailable evidence/storage."""
    try:
        config = load_config(ctx.obj["config_path"])
        value = dream.create_plan(config, mirror, sources, base, maintenance=maintenance, topic=topic)
    except (ConfigError, dream.DreamError, OSError, UnicodeError) as exc:
        typer.echo(json.dumps(dict(schema_version=1, operation="dream_plan", state="rejected", error=str(exc))))
        code = 3 if isinstance(exc, OSError) else getattr(exc, "exit_code", 2)
        raise typer.Exit(code) from exc
    typer.echo(json.dumps(value, sort_keys=True))


@app.command("snapshot")
def snapshot(
    ctx: typer.Context,
    mirror: str = typer.Argument(..., help="Configured local mirror."),
    plan_path: Path = typer.Option(..., "--plan", help="Stored private plan.json."),
):
    """Revalidate a frozen plan and privately copy selected evidence; no git writes."""
    _data_command(ctx, "dream_snapshot", dream_run.snapshot, mirror, plan_path)


@app.command("run")
def run(
    ctx: typer.Context,
    mirror: str = typer.Argument(..., help="Configured local mirror."),
    plan_path: Path = typer.Option(..., "--plan", help="Stored private plan.json."),
    harness: str = typer.Option(..., "--harness", help="fixture (exact quotations), codex or claude (native structured roles)."),
    model: Optional[str] = typer.Option(None, "--model", help="Required for native roles; sends admitted evidence and prior wiki to this model's provider."),
    attempt: int = typer.Option(1, "--attempt", min=1, help="Explicit native attempt number; every attempt shares the persisted job budget."),
    mode: str = typer.Option("wiki", "--mode", help="Native output: wiki author/editor (default) or legacy notes."),
    purpose: str = typer.Option(dream_wiki.PURPOSE, "--purpose", show_default=False, help="What the wiki should help readers understand; wiki mode only."),
    budget_path: Optional[Path] = typer.Option(None, "--budget", help="Private persistent job-budget directory; defaults to this plan's job-budget. Share explicitly across plans."),
    max_total_seconds: Optional[int] = typer.Option(None, "--max-total-seconds", min=1, help="Immutable whole-job elapsed ceiling; defaults to the plan's elapsed limit."),
    max_total_calls: Optional[int] = typer.Option(None, "--max-total-calls", min=1, help="Immutable whole-job native invocation ceiling across all attempts."),
    reserve_seconds: Optional[int] = typer.Option(None, "--reserve-seconds", min=0, help="Time kept for review/repair/finalization; ordinary author work stops earlier."),
    reserve_calls: Optional[int] = typer.Option(None, "--reserve-calls", min=0, help="Native invocations kept for review/repair/consolidation."),
    max_total_tokens: Optional[int] = typer.Option(None, "--max-total-tokens", min=1, help="Requested hard token ceiling; rejected by current native adapters before provider startup."),
    max_cost_usd: Optional[str] = typer.Option(None, "--max-cost-usd", help="Requested hard dollar ceiling; rejected when the native adapter cannot enforce it."),
):
    """Prepare a candidate on a new dream branch, preserving worktree and index."""
    policy = dict(max_elapsed_seconds=max_total_seconds, max_native_calls=max_total_calls,
        reserve_seconds=reserve_seconds, reserve_calls=reserve_calls, max_total_tokens=max_total_tokens, max_cost_usd=max_cost_usd)
    if harness == "fixture" and (budget_path is not None or any(value is not None for value in policy.values())):
        raise typer.BadParameter("Job budgets apply to native codex/claude execution")
    if harness == "fixture" and model is None and attempt == 1 and mode in ("wiki", "notes") and purpose == dream_wiki.PURPOSE:
        _data_command(ctx, "dream_run", dream_run.run_fixture, mirror, plan_path, harness)
    else:
        _data_command(ctx, "dream_run", dream_wiki.run, mirror, plan_path, harness, model, attempt, mode, purpose,
            budget_path=budget_path, budget_limits=policy)


@app.command("budget")
def budget_status(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Configured mirror for an explicit plan; omit for the current project's latest job."),
    plan_path: Optional[Path] = typer.Option(None, "--plan", help="Stored private plan.json for explicit evidence execution."),
    budget_path: Optional[Path] = typer.Option(None, "--budget", help="Explicit shared budget directory; omit for the plan's default."),
    job_path: Optional[Path] = typer.Option(None, "--job", help="Specific exploratory job; omit to inspect the current project's latest job."),
):
    """Inspect the persisted allowance and observed usage without inference."""
    if mirror is not None and plan_path is not None and job_path is None:
        _data_command(ctx, "dream_budget", dream_budget.inspect, mirror, plan_path, budget_path)
    elif mirror is None and plan_path is None and budget_path is None:
        from . import dream_explore
        _data_command(ctx, "dream_budget", dream_explore.budget_status, job_path)
    else:
        raise typer.BadParameter("Use MIRROR --plan PLAN [--budget PATH], or omit them and optionally select --job PATH")


@app.command("status")
def status(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Configured mirror for an explicit plan; omit for the current project's latest job."),
    plan_path: Optional[Path] = typer.Option(None, "--plan", help="Stored private plan.json for an explicit run."),
    run_id: Optional[str] = typer.Option(None, "--run", help="Select a run ID when this plan has multiple native runs."),
    job_path: Optional[Path] = typer.Option(None, "--job", help="Specific exploratory job; omit to inspect the current project's latest job."),
):
    """Print JSON separating candidate preparation from validated integration."""
    if mirror is not None and plan_path is not None and job_path is None:
        _data_command(ctx, "dream_status", dream_run.status, mirror, plan_path, run_id)
    elif mirror is None and plan_path is None and run_id is None:
        from . import dream_explore
        _data_command(ctx, "dream_status", dream_explore.status, job_path)
    else:
        raise typer.BadParameter("Use MIRROR --plan PLAN [--run RUN_ID], or omit them and optionally select --job PATH")


@app.command("read")
def read(
    ctx: typer.Context,
    mirror: str = typer.Argument(..., help="Configured local mirror."),
    topic: Optional[str] = typer.Option(None, "--topic", help="Semantic topic name; omit for bounded session orientation."),
    archives: bool = typer.Option(False, "--archives", help="Read one archive entry; requires topic and target."),
    target: Optional[str] = typer.Option(None, "--target", help="Specific CUSTOM_ID to retrieve."),
    max_bytes: Optional[int] = typer.Option(None, "--max-bytes", min=1, help="Explicit UTF-8 allowance; defaults to 2 KiB orientation or 12 KiB topic."),
    base: Optional[str] = typer.Option(None, "--base", help="Revision/ref to read without checkout; default: the memory branch if it exists, else HEAD."),
):
    """Read a pinned wiki view; ordinary reads omit LOGBOOK and archives."""
    _data_command(ctx, "dream_read", dream_reader.read, mirror, topic, archives, target, max_bytes, base)


@app.command("source")
def source(
    ctx: typer.Context,
    mirror: str = typer.Argument(..., help="Configured local mirror."),
    plan_path: Path = typer.Option(..., "--plan", help="Explicit admitted plan with local frozen snapshots."),
    unit: Optional[str] = typer.Option(None, "--unit", help="Admitted unit ID; omit to search/read all selected units."),
    search: Optional[str] = typer.Option(None, "--search", help="Literal search with neighboring lines."),
    start: Optional[int] = typer.Option(None, "--start", min=0, help="Absolute UTF-8 byte start in the admitted unit."),
    end: Optional[int] = typer.Option(None, "--end", min=1, help="Exclusive absolute UTF-8 byte end."),
    max_bytes: Optional[int] = typer.Option(None, "--max-bytes", min=1, help="Explicit evidence allowance; default 32 KiB."),
):
    """Retrieve exact source snapshots; never discover additional evidence."""
    _data_command(ctx, "dream_source", dream_reader.source, mirror, plan_path, unit, search, start, end, max_bytes)


@app.command("ask")
def ask(
    ctx: typer.Context,
    mirror: str = typer.Argument(..., help="Configured local mirror."),
    question: str = typer.Argument(..., help="Question for a fresh native reader."),
    plan_path: Path = typer.Option(..., "--plan", help="Explicit admitted plan with frozen source snapshots."),
    harness: str = typer.Option(..., "--harness", help="Native codex or claude."),
    model: str = typer.Option(..., "--model", help="Explicit model; transmits admitted reader context to its provider."),
    base: Optional[str] = typer.Option(None, "--base", help="Pinned wiki revision or integration ref; default: the memory branch if it exists, else HEAD."),
    source_only: bool = typer.Option(False, "--source-only", help="Use the identical evidence interface without wiki access."),
    wiki_only: bool = typer.Option(False, "--wiki-only", help="Read only the companion and specific archives; disable original-source tools."),
    attempt: int = typer.Option(1, "--attempt", min=1, help="Explicit attempt; uncertain native outcomes are never silently retried."),
):
    """Answer through bounded, traced retrieval without construction context."""
    if source_only and wiki_only:
        raise typer.BadParameter("--source-only and --wiki-only cannot be combined")
    if wiki_only:
        from . import dream_reader_eval
        _data_command(ctx, "dream_ask", dream_reader_eval.ask, mirror, plan_path, question, harness, model, base, "wiki-only", attempt)
        return
    _data_command(ctx, "dream_ask", dream_reader_agent.ask, mirror, plan_path, question, harness, model, base, not source_only, attempt)


def _data_command(ctx, operation, function, *args, **kwargs):
    try:
        config = load_config(ctx.obj["config_path"])
        value = function(config, *args, **kwargs)
    except (ConfigError, dream.DreamError, OSError, UnicodeError) as exc:
        value = dict(schema_version=1, operation=operation, state="rejected", error=str(exc))
        if isinstance(exc, dream_budget.Exhausted):
            value.update(state="budget_exhausted", budget=exc.budget)
        typer.echo(json.dumps(value))
        code = 3 if isinstance(exc, OSError) else getattr(exc, "exit_code", 2)
        raise typer.Exit(code) from exc
    typer.echo(json.dumps(value, sort_keys=True))
