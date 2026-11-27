"""``sucoder dream organize``: organizer jobs (docs/dream/organizer.org)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer


app = typer.Typer(help="Organize one feature area of the memory wiki into a hub; the organizer is the memory branch's only writer.")


def _data(ctx, operation, function, *args, **kwargs):
    from .dream_cli import _data_command  # dream_cli imports this module at its end.
    _data_command(ctx, operation, function, *args, **kwargs)


@app.command("plan")
def plan(
    ctx: typer.Context,
    mirror: str = typer.Argument(..., help="Configured mirror whose memory branch to organize."),
    area: str = typer.Option(..., "--area", help="Area identifier, e.g. dream-memory; the hub is .sucoder/wiki/hubs/AREA.org."),
    title: Optional[str] = typer.Option(None, "--title", help="Hub and index title; required the first time an area is organized."),
    group: Optional[str] = typer.Option(None, "--group", help="Index group heading for the area; required the first time."),
    summary: Optional[str] = typer.Option(None, "--summary", help="One-line index summary; required the first time."),
    project: str = typer.Option("HEAD", "--project", help="Project revision whose README.org and docs/ headings are admitted."),
    no_notes: bool = typer.Option(False, "--no-notes", help="Do not admit accepted session notes from the private store."),
):
    """Freeze an organizer job without model calls or git writes, and print it as JSON."""
    from . import dream_organizer
    _data(ctx, "organizer_plan", dream_organizer.plan, mirror, area, title=title, group=group, summary=summary,
          project=project, notes=not no_notes)


@app.command("run")
def run(
    ctx: typer.Context,
    mirror: str = typer.Argument(..., help="Configured mirror."),
    plan_path: Path = typer.Option(..., "--plan", help="Stored organizer plan.json."),
    harness: str = typer.Option(..., "--harness", help="Organizer and first reviewer: claude or codex."),
    model: str = typer.Option(..., "--model", help="Model for the organizer and first reviewer; sends the frozen input to its provider."),
    cross_harness: str = typer.Option(..., "--cross-harness", help="Second reviewer's harness; must differ from --harness."),
    cross_model: str = typer.Option(..., "--cross-model", help="Second reviewer's model, from that other family."),
    attempt: int = typer.Option(1, "--attempt", min=1, max=1000000, help="Explicit attempt; a started call is never silently repeated."),
):
    """Organize the area and publish it in place; a rerun replays saved results without model calls."""
    from . import dream_organizer
    _data(ctx, "organizer_run", dream_organizer.run, mirror, plan_path, harness, model, cross_harness, cross_model,
          attempt=attempt)


@app.command("status")
def status(
    ctx: typer.Context,
    mirror: str = typer.Argument(..., help="Configured mirror."),
    plan_path: Path = typer.Option(..., "--plan", help="Stored organizer plan.json."),
    run_id: Optional[str] = typer.Option(None, "--run", help="Select a run when the plan has several."),
):
    """Replay a run without inference and report whether its generation is published."""
    from . import dream_organizer
    _data(ctx, "organizer_status", dream_organizer.status, mirror, plan_path, run_id)
