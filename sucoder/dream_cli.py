"""Local dream entrypoint, independent of interactive agent startup."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import dream
from .config import ConfigError, load_config


app = typer.Typer(help="Plan a bounded companion-wiki revision. Git evidence planning only; no model calls yet.")


@app.command("plan")
def plan(
    ctx: typer.Context,
    mirror: str = typer.Argument(..., help="Configured mirror available on this host."),
    sources: Path = typer.Option(..., "--sources", help="Explicit version-1 JSON source catalog; git units only."),
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
