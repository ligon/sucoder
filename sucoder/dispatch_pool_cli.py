"""Existing-allocation pool commands; no scheduler allocation creation."""
from pathlib import Path
from typing import List, Optional
import json
import subprocess
import typer
from . import dispatch_pool as control

app = typer.Typer(help='Attach, reserve and submit isolated steps in an existing allocation.', no_args_is_help=True)


def _emit(value):
    typer.echo(json.dumps(value, sort_keys=True))


def _error(exc):
    _emit(dict(schema_version=1, state='error', reason=str(exc)))
    raise typer.Exit(2)


@app.command()
def attach(directory: Path, job_id: int = typer.Option(..., '--job-id'),
           node: str = typer.Option(..., '--node'), cpus: int = typer.Option(..., '--cpus'),
           mem: str = typer.Option(..., '--mem'), submit_before: float = typer.Option(..., '--submit-before'),
           worker_deadline: float = typer.Option(..., '--worker-deadline'), expires: float = typer.Option(..., '--expires'),
           source: str = typer.Option(..., '--source'), image: str = typer.Option(..., '--image'),
           image_sha256: str = typer.Option(..., '--image-sha256')):
    """Bind a new private pool record to this uid's running single-node job.

    Cutoffs are Unix seconds. --cpus/--mem are the total worker budget, with
    coordinator/keeper/verifier resources already subtracted by the caller.
    """
    try:
        value = control.attach(directory, job_id, node=node, cpus=cpus, mem=mem,
                               submit_before=submit_before, worker_deadline=worker_deadline,
                               expires=expires, source=source, image=image, image_sha256=image_sha256)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        _error(exc)
    _emit(value)


@app.command()
def submit(ctx: typer.Context, pool: Path, repository: Path,
           ref: str = typer.Option(..., '--ref'), task_file: Path = typer.Option(..., '--task-file'),
           agent: str = typer.Option(..., '--agent'), executable: Path = typer.Option(..., '--executable'),
           agent_home: Optional[Path] = typer.Option(None, '--agent-home'),
           project_profile: Optional[Path] = typer.Option(None, '--project-profile'),
           model: Optional[str] = typer.Option(None, '--model'), cpus: int = typer.Option(1, '--cpus'),
           mem: str = typer.Option('8G', '--mem'), time: str = typer.Option('00:30:00', '--time'),
           require_evidence: List[str] = typer.Option(None, '--require-evidence'),
           snapshot_minutes: float = typer.Option(10, '--snapshot-minutes', min=0.1),
           publication_timeout: float = typer.Option(30, '--publication-timeout', help='Seconds per workspace publication (positive, at most 300).'),
           timeout: float = typer.Option(30, '--timeout')):
    """Reserve budget and start exactly one existing-allocation step.

    A missing receiving receipt is ambiguous and is never retried automatically.
    Use ordinary dispatch status/result/logs for a returned dispatch ID.
    """
    try:
        value = control.submit(pool, repository, ref, task_file, agent, executable,
                               cpus=cpus, mem=mem, time_limit=time, timeout=timeout,
                               directory_root=ctx.obj['root'], source_agent_home=agent_home,
                               project_profile=project_profile, model=model,
                               required_evidence=require_evidence or [], snapshot_minutes=snapshot_minutes,
                               publication_timeout=publication_timeout)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        _error(exc)
    _emit(value)
    if value['state'] != 'submitted':
        raise typer.Exit(3)


@app.command()
def status(pool: Path, timeout: float = typer.Option(10, '--timeout')):
    """Reconcile exact step terminal proof; unknown steps retain their budget."""
    try:
        _emit(control.status(pool, timeout=timeout))
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        _error(exc)


@app.command()
def stop_worker(pool: Path, dispatch_id: str, timeout: float = typer.Option(10, '--timeout')):
    """Send TERM only to this pool worker's proven numeric step, never its job."""
    try:
        _emit(control.stop_worker(pool, dispatch_id, timeout=timeout))
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        _error(exc)
