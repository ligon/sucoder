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


# Installed releases ship no docs/, so --help carries the operating contract
# (exit codes, defaults, units) and points at the full reference online.
DOCS = "Full reference: docs/dispatch.org in https://github.com/ligon/sucoder"
DISPATCH_ID_HELP = "32-character dispatch ID printed by submit, not the Slurm job number."

app = typer.Typer(help="Dispatch independent headless agent jobs on the current Slurm cluster.",
                  epilog=("Run it on the cluster as the coordinating account; it never routes through --target or --config.  "
                          "submit prints a dispatch ID that status, wait, result and logs take.  "
                          "Give --root before the subcommand: sucoder dispatch --root DIR status ID.\n\n" + DOCS),
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


LOCAL_DISK_ROOT_HELP = "Receiving node's local-disk parent; the receiving job uses ROOT/job$SLURM_JOB_ID."


@app.command(epilog=(
    "Exit status: 0 submitted, or a valid --dry-run.  1 sbatch could not be run, so no job exists.  "
    "2 request refused before submission (JSON state error) or a usage error.  "
    "3 submission_unknown: Slurm may already have accepted the job.\n\n"
    "After exit 3 never resubmit blindly.  Look for the job comment sucoder-dispatch:DISPATCH_ID "
    "(squeue --me -o '%i %k' lists it while the job is queued or running) and reconcile first.\n\n" + DOCS))
def submit(
    ctx: typer.Context,
    repository: Path = typer.Argument(..., help="Full Git repository on shared storage; never modified."),
    ref: str = typer.Option(..., "--ref", help="Committed source revision; resolved and pinned before submission."),
    task_file: Path = typer.Option(..., "--task-file", help="UTF-8 file holding the worker's complete assignment; its content is frozen at submission."),
    agent: str = typer.Option(..., "--agent", help="Headless adapter: codex or claude."),
    account: str = typer.Option(..., "--account", help="Slurm account to charge."),
    partition: str = typer.Option(..., "--partition", help="Slurm partition."),
    qos: Optional[str] = typer.Option(None, "--qos", help="Slurm QOS; Slurm's default when omitted."),
    exclude: Optional[str] = typer.Option(None, "--exclude", help="Slurm nodes to exclude, separated by commas, such as the coordinator's own node."),
    cpus: int = typer.Option(1, "--cpus", min=1, help="CPUs for the worker's single task."),
    mem: str = typer.Option("8G", "--mem", help="Memory per node: an integer with an optional K, M, G or T suffix."),
    time: str = typer.Option("00:30:00", "--time", help="Slurm time limit: minutes, MM:SS, HH:MM:SS, D-HH, D-HH:MM or D-HH:MM:SS.  "
                             "At least two minutes, because the worker reserves a 120-second TERM window."),
    model: Optional[str] = typer.Option(None, "--model", help="Native model name passed to the agent as --model; the agent's default when omitted."),
    executable: Optional[Path] = typer.Option(None, "--executable", help="Agent executable; must exist at this path on receiving nodes.  "
                                              "Default: the --agent name found on this host's PATH."),
    agent_home: Optional[Path] = typer.Option(None, "--agent-home", help="Shared native configuration/authentication source; copied privately.  "
                                              "Default: ~/.codex or ~/.claude."),
    claude_user_config: Optional[Path] = typer.Option(None, "--claude-user-config", help="Shared Claude user configuration seed for --agent claude.  "
                                                      "Default: ~/.claude.json."),
    require_evidence: List[str] = typer.Option(None, "--require-evidence", help="Required evidence filename, relative to the private evidence directory; repeatable."),
    snapshot_minutes: float = typer.Option(10, "--snapshot-minutes", min=0.1, help="Periodic Git/WIP publication cadence; final publication also runs on orderly exit."),
    publication_timeout: float = typer.Option(30, "--publication-timeout", help="Seconds per workspace publication (positive, at most 300); extends the early shutdown window."),
    project_profile: Optional[Path] = typer.Option(None, "--project-profile", help="Explicit project JSON profile; bootstrap and configuration are frozen before submission."),
    local_disk_root: str = typer.Option("/local", "--local-disk-root", help=LOCAL_DISK_ROOT_HELP),
    timeout: float = typer.Option(30, "--timeout", min=0.1, help="Deadline in seconds for the sbatch call; expiry is submission_unknown."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate and print the planned sbatch command; write no files and submit no job."),
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
            publication_timeout=publication_timeout,
            project_profile=project_profile,
            timeout=timeout, dry_run=dry_run,
        )
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        _error(exc)
    _emit(value)
    if value.get("state") in ("submission_unknown", "submission_failed"):
        raise typer.Exit(3 if value["state"] == "submission_unknown" else 1)


@app.command(epilog=(
    "Exit status: 0 the stage is frozen and its JSON printed.  2 request refused (JSON state error) or a usage error.\n\n"
    "Full reference: docs/dispatch-projects.org in https://github.com/ligon/sucoder"))
def verify_stage(
    ctx: typer.Context,
    repository: Path = typer.Argument(..., help="Shared source or private results.git; never modified."),
    ref: str = typer.Option(..., "--ref", help="Resolved and frozen before the verification job is submitted."),
    project_profile: Path = typer.Option(..., "--project-profile", help="Project JSON profile; its configuration and bootstrap bytes are frozen."),
    directory: Optional[Path] = typer.Option(None, "--directory", help="New private shared directory for one verification job.  "
                                             "Default: a fresh verify-* directory under --root."),
    command_file: Optional[Path] = typer.Option(None, "--command-file", help="JSON argv array; supports {python}, {worktree}, {evidence}.  "
                                                "Give exactly one of --command-file or --script-file."),
    script_file: Optional[Path] = typer.Option(None, "--script-file", help="Existing deterministic burst RUN script; its bytes are frozen."),
    require_evidence: List[str] = typer.Option(None, "--require-evidence", help="Required selected proof filename; repeatable."),
    cpus: int = typer.Option(1, "--cpus", min=1, help="CPU budget for the verification command."),
    timeout: int = typer.Option(600, "--timeout", min=1, help="Deadline in seconds for each verification step: the clone, then the command."),
    local_disk_root: str = typer.Option("/local", "--local-disk-root", help=LOCAL_DISK_ROOT_HELP),
):
    """Freeze a deterministic verification request for a separate burst job; submit no job.

    The stage pins the exact commit, project profile, helpers and either a JSON
    command or an existing RUN script.  It allocates no resources and starts no
    model; run it from a new burst allocation.  Each job needs a fresh stage.
    """
    try:
        destination = directory or (ctx.obj["root"] / ("verify-" + uuid.uuid4().hex))
        value = dispatch_verify.freeze(repository, ref, project_profile, command_file, destination,
                                       control._shared_path, control._git, script_file=script_file,
                                       timeout=timeout, cpus=cpus, local_disk_root=local_disk_root,
                                       required_evidence=require_evidence or [])
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        _error(exc)
    _emit(value)


@app.command(epilog=(
    "Status reports liveness and publication evidence, not success: a fresh heartbeat shows the supervisor is alive, "
    "not that the task is progressing.  Use wait or result for the outcome.\n\n"
    "Exit status: 0 the observation is printed.  2 invalid dispatch ID or record."))
def status(ctx: typer.Context, dispatch_id: str = typer.Argument(..., help=DISPATCH_ID_HELP),
           timeout: float = typer.Option(10, min=0.1, help="Deadline in seconds for the scheduler query.")):
    """Read worker evidence and a bounded scheduler observation."""
    try:
        value = control.status(_directory(ctx, dispatch_id), timeout=timeout)
    except (ValueError, OSError) as exc:
        _error(exc)
    _emit(value)


@app.command(epilog=(
    "Exit status: 0 the final result succeeded with zero native and supervisor exits "
    "(and, with --require-task-complete, the worker reported its task completed).  "
    "4 succeeded, but --require-task-complete was given and the task was not reported completed.  "
    "3 no final result before --timeout, or a failed result without an exit code.  "
    "2 invalid dispatch ID or record.  Any other nonzero status is a failed result's exit code.\n\n"
    "A worker's own exit code can also be 2, 3 or 4; the printed JSON is authoritative."))
def wait(
    ctx: typer.Context, dispatch_id: str = typer.Argument(..., help=DISPATCH_ID_HELP),
    timeout: float = typer.Option(60, min=0.1, help="Seconds to wait for the final result."),
    poll: float = typer.Option(2, min=0.1, help="Seconds between checks."),
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


@app.command(epilog=(
    "Exit status: 0 whenever the record is readable, including while the worker is still running; "
    "check result_state (present, missing or unreadable).  2 invalid dispatch ID or record."))
def result(ctx: typer.Context, dispatch_id: str = typer.Argument(..., help=DISPATCH_ID_HELP)):
    """Return the explicit final worker result, never inferred scheduler success."""
    try:
        value = control.read_result(_directory(ctx, dispatch_id))
    except (ValueError, OSError) as exc:
        _error(exc)
    _emit(value)


@app.command(epilog="Exit status: 0 the tail is printed.  2 invalid dispatch ID or record, unknown stream, or a log not yet written.")
def logs(
    ctx: typer.Context, dispatch_id: str = typer.Argument(..., help=DISPATCH_ID_HELP),
    stream: str = typer.Option("stderr", help="Worker stdout or stderr, or Slurm's own log: scheduler or scheduler-stderr."),
    tail: int = typer.Option(80, min=1, max=10000, help="Lines to print, taken from the last 1 MiB of the log."),
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
