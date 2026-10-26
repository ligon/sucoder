"""Noninteractive compute-side CLI for independent headless workers."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional
import uuid

import typer

from . import dispatch as control
from . import dispatch_verify


app = typer.Typer(help="Dispatch independent headless agent jobs on the current Slurm cluster.",
                  no_args_is_help=True)


def _emit(value):
    typer.echo(json.dumps(value, sort_keys=True))


def _error(exc):
    _emit({"schema_version": 1, "state": "error", "reason": str(exc)})
    raise typer.Exit(2)


from .dispatch_pool_cli import app as pool_app
app.add_typer(pool_app, name="pool")


@app.callback()
def main(ctx: typer.Context, root: Path = typer.Option(
        control.DEFAULT_DIRECTORY_ROOT, "--root", help="Private dispatch records on shared storage.")):
    ctx.obj = {"root": root.expanduser()}


def _directory(ctx, dispatch_id):
    if not re.fullmatch(r"[0-9a-f]{32}", dispatch_id):
        raise ValueError("Expected the 32-character dispatch ID, not a scheduler job number.")
    return ctx.obj["root"] / dispatch_id


@app.command()
def submit(
    ctx: typer.Context,
    repository: Path = typer.Argument(..., help="Full Git repository on shared storage; never modified."),
    ref: str = typer.Option(..., "--ref", help="Committed source revision; resolved and pinned before submission."),
    task_file: Path = typer.Option(..., "--task-file", help="UTF-8 worker task file."),
    agent: str = typer.Option(..., "--agent", help="Headless adapter: codex or claude."),
    account: str = typer.Option(..., "--account"),
    partition: str = typer.Option(..., "--partition"),
    qos: Optional[str] = typer.Option(None, "--qos"),
    exclude: Optional[str] = typer.Option(None, "--exclude", help="Slurm nodes to exclude, separated by commas."),
    cpus: int = typer.Option(1, "--cpus", min=1),
    mem: str = typer.Option("8G", "--mem"),
    time: str = typer.Option("00:30:00", "--time"),
    model: Optional[str] = typer.Option(None, "--model"),
    executable: Optional[Path] = typer.Option(None, "--executable", help="Agent executable available on receiving nodes."),
    agent_home: Optional[Path] = typer.Option(None, "--agent-home", help="Shared native configuration/authentication source; copied privately."),
    claude_user_config: Optional[Path] = typer.Option(None, "--claude-user-config"),
    require_evidence: List[str] = typer.Option(None, "--require-evidence", help="Required evidence filename, relative to the private evidence directory; repeatable."),
    snapshot_minutes: float = typer.Option(10, "--snapshot-minutes", min=0.1, help="Periodic Git/WIP publication cadence; final publication also runs on orderly exit."),
    project_profile: Optional[Path] = typer.Option(None, "--project-profile", help="Explicit project JSON profile; bootstrap and configuration are frozen before submission."),
    local_disk_root: str = typer.Option("/local", "--local-disk-root"),
    timeout: float = typer.Option(30, "--timeout", min=0.1, help="Submission subprocess deadline in seconds."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate and describe; write no files and submit no job."),
):
    """Submit one worker and return its dispatch ID and Slurm receipt as JSON."""
    try:
        value = control.submit(
            repository, ref, task_file, agent,
            str(executable) if executable else (shutil.which(agent) or agent),
            {"account": account, "partition": partition, "qos": qos, "exclude": exclude,
             "cpus": cpus, "mem": mem, "time": time},
            model=model, directory_root=ctx.obj["root"], local_disk_root=local_disk_root,
            source_agent_home=agent_home, source_claude_user_config=claude_user_config,
            required_evidence=require_evidence or [],
            snapshot_minutes=snapshot_minutes,
            project_profile=project_profile,
            timeout=timeout, dry_run=dry_run,
        )
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        _error(exc)
    _emit(value)
    if value.get("state") in ("submission_unknown", "submission_failed"):
        raise typer.Exit(3 if value["state"] == "submission_unknown" else 1)


@app.command()
def verify_stage(
    ctx: typer.Context,
    repository: Path = typer.Argument(..., help="Shared source or private results.git; never modified."),
    ref: str = typer.Option(..., "--ref", help="Resolved and frozen before the verification job is submitted."),
    project_profile: Path = typer.Option(..., "--project-profile"),
    directory: Optional[Path] = typer.Option(None, "--directory", help="New private shared directory for one verification job."),
    command_file: Optional[Path] = typer.Option(None, "--command-file", help="JSON argv array; supports {python}, {worktree}, {evidence}."),
    script_file: Optional[Path] = typer.Option(None, "--script-file", help="Existing deterministic burst RUN script; its bytes are frozen."),
    require_evidence: List[str] = typer.Option(None, "--require-evidence", help="Required selected proof filename; repeatable."),
    cpus: int = typer.Option(1, "--cpus", min=1),
    timeout: int = typer.Option(600, "--timeout", min=1),
    local_disk_root: str = typer.Option("/local", "--local-disk-root"),
):
    """Freeze a deterministic request for the existing burst job; submit no job."""
    try:
        destination = directory or (ctx.obj["root"] / ("verify-" + uuid.uuid4().hex))
        value = dispatch_verify.freeze(repository, ref, project_profile, command_file, destination,
                                       control._shared_path, control._git, script_file=script_file,
                                       timeout=timeout, cpus=cpus, local_disk_root=local_disk_root,
                                       required_evidence=require_evidence or [])
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        _error(exc)
    _emit(value)


@app.command()
def status(ctx: typer.Context, dispatch_id: str, timeout: float = typer.Option(10, min=0.1)):
    """Read worker evidence and a bounded scheduler observation."""
    try:
        value = control.status(_directory(ctx, dispatch_id), timeout=timeout)
    except (ValueError, OSError) as exc:
        _error(exc)
    _emit(value)


@app.command()
def wait(
    ctx: typer.Context, dispatch_id: str,
    timeout: float = typer.Option(60, min=0.1),
    poll: float = typer.Option(2, min=0.1),
    require_task_complete: bool = typer.Option(False, "--require-task-complete", help="Also require the worker to report task completion; this is not scientific acceptance."),
):
    """Wait within a deadline; return the worker exit or unavailable status."""
    try:
        value = control.wait(_directory(ctx, dispatch_id), timeout=timeout, poll_seconds=poll)
    except (ValueError, OSError) as exc:
        _error(exc)
    _emit(value)
    result = value.get("result") or {}
    if (result.get("state") == "succeeded" and result.get("exit_code") == 0
            and result.get("supervisor_exit_code", 0) == 0):
        if require_task_complete and result.get("task", {}).get("status") != "completed":
            raise typer.Exit(4)
        return
    code = result.get("supervisor_exit_code", result.get("exit_code"))
    raise typer.Exit(code if isinstance(code, int) and 0 < code < 256 else 3)


@app.command()
def result(ctx: typer.Context, dispatch_id: str):
    """Return the explicit final worker result, never inferred scheduler success."""
    try:
        value = control.read_result(_directory(ctx, dispatch_id))
    except (ValueError, OSError) as exc:
        _error(exc)
    _emit(value)


@app.command()
def logs(
    ctx: typer.Context, dispatch_id: str,
    stream: str = typer.Option("stderr", help="stdout, stderr, scheduler, or scheduler-stderr."),
    tail: int = typer.Option(80, min=1, max=10000),
):
    """Print a bounded tail from this dispatch's private log."""
    try:
        directory = _directory(ctx, dispatch_id)
        # Validate the record and its uid before opening any artifact.
        control.read_result(directory)
        names = {"stdout": "stdout.log", "stderr": "stderr.log", "scheduler": "slurm.log",
                 "scheduler-stderr": "slurm.stderr.log"}
        if stream not in names:
            raise ValueError("stream must be stdout, stderr, scheduler, or scheduler-stderr")
        path = directory / names[stream]
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 1024 * 1024))
            text = handle.read().decode("utf-8", errors="replace")
        typer.echo("\n".join(text.splitlines()[-tail:]))
    except (ValueError, OSError) as exc:
        _error(exc)
