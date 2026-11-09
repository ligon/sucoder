"""Local dream entrypoint, independent of interactive agent startup."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import dream
from . import dream_run
from . import dream_wiki
from .config import ConfigError, load_config


app = typer.Typer(help="Plan, snapshot and publish companion-wiki candidates from explicit git or transcript evidence.")


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
    attempt: int = typer.Option(1, "--attempt", min=1, help="Explicit native attempt number; a new number starts a new run and allowance."),
    mode: str = typer.Option("wiki", "--mode", help="Native output: wiki author/editor (default) or legacy notes."),
    purpose: str = typer.Option(dream_wiki.PURPOSE, "--purpose", show_default=False, help="What the wiki should help readers understand; wiki mode only."),
):
    """Prepare a candidate on a new dream branch, preserving worktree and index."""
    if harness == "fixture" and model is None and attempt == 1 and mode in ("wiki", "notes") and purpose == dream_wiki.PURPOSE:
        _data_command(ctx, "dream_run", dream_run.run_fixture, mirror, plan_path, harness)
    else:
        _data_command(ctx, "dream_run", dream_wiki.run, mirror, plan_path, harness, model, attempt, mode, purpose)


@app.command("status")
def status(
    ctx: typer.Context,
    mirror: str = typer.Argument(..., help="Configured local mirror."),
    plan_path: Path = typer.Option(..., "--plan", help="Stored private plan.json for this run."),
    run_id: Optional[str] = typer.Option(None, "--run", help="Select a run ID when this plan has multiple native runs."),
):
    """Print JSON separating candidate preparation from validated integration."""
    _data_command(ctx, "dream_status", dream_run.status, mirror, plan_path, run_id)


def _data_command(ctx, operation, function, *args):
    try:
        config = load_config(ctx.obj["config_path"])
        value = function(config, *args)
    except (ConfigError, dream.DreamError, OSError, UnicodeError) as exc:
        typer.echo(json.dumps(dict(schema_version=1, operation=operation, state="rejected", error=str(exc))))
        code = 3 if isinstance(exc, OSError) else getattr(exc, "exit_code", 2)
        raise typer.Exit(code) from exc
    typer.echo(json.dumps(value, sort_keys=True))
