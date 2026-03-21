"""Typer-powered CLI entry point."""

from __future__ import annotations

import os
import shlex
import pwd
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import click

try:
    from click.shell_completion import CompletionItem as ClickCompletionItem
except (ImportError, AttributeError):  # pragma: no cover - defensive
    ClickCompletionItem = None  # type: ignore[assignment]

import typer

from . import __version__
from .config import (
    AgentLauncher,
    BranchPrefixes,
    Config,
    ConfigError,
    MirrorSettings,
    _detect_git_toplevel,
    build_default_config,
    load_config,
)
from .executor import CommandExecutor
from .logging_utils import setup_logger
from .mirror import MirrorError, MirrorManager
from .startup_checks import StartupError, run_startup_checks

app = typer.Typer(help="sucoder – Unix-sandboxed agent collaboration toolkit for managing agent mirrors.")


def _default_config_path() -> Path:
    return Path("~/.sucoder/config.yaml").expanduser()


def _load_config_path(config_path: Optional[Path]) -> Path:
    if config_path:
        return Path(config_path).expanduser()
    return _default_config_path()


def _get_config(ctx: typer.Context) -> Config:
    obj = ctx.obj or {}
    config = obj.get("config")
    if config is None:
        raise typer.Exit(code=2)
    return config


def _get_use_sudo_for_agent(ctx: Optional[click.Context], config: Optional[Config] = None) -> bool:
    obj = (ctx.obj if ctx and ctx.obj else {}) or {}
    use_sudo = obj.get("use_sudo_for_agent")
    if use_sudo is None:
        use_sudo = True

    # If we're already running as the agent user, skip sudo to avoid failures in restricted environments.
    if config and config.agent_user == pwd.getpwuid(os.getuid()).pw_name:
        return False

    return bool(use_sudo)


def _get_config_for_completion(ctx: typer.Context) -> Optional[Config]:
    """Best-effort resolution of configuration during shell completion."""
    obj = ctx.obj or {}
    config = obj.get("config")
    if isinstance(config, Config):
        return config

    config_param = ctx.params.get("config") if ctx.params else None
    try:
        config_path = _load_config_path(config_param)
    except Exception:
        return None

    try:
        return load_config(config_path)
    except ConfigError:
        pass

    try:
        return build_default_config()
    except ConfigError:
        return None


def _mirror_completion(
    ctx: typer.Context, param, incomplete: str
) -> List[Any]:
    config = _get_config_for_completion(ctx)
    if not config:
        return []
    items: List[Any] = []
    for name, settings in sorted(config.mirrors.items()):
        if incomplete and not name.startswith(incomplete):
            continue
        help_text = str(settings.canonical_repo)
        if ClickCompletionItem is not None:
            items.append(ClickCompletionItem(name, help=help_text))
        else:
            items.append(name)
    return items


def _build_executor(
    config: Config,
    logger,
    dry_run: bool,
    *,
    use_sudo_for_agent: bool = True,
    mirror_settings: Optional[MirrorSettings] = None,
) -> CommandExecutor:
    if mirror_settings and mirror_settings.remote:
        from .executor import RemoteExecutor
        from .session import RemoteSession
        from .tunnel import SshControl, TunnelError

        remote = mirror_settings.remote
        session = RemoteSession.load(mirror_settings.name)

        # 1. Establish ControlMaster to the gateway (authenticates
        #    once; may prompt for pin + OTP).
        gw_control = SshControl(
            gateway=remote.gateway,
            control_persist=remote.control_persist,
        )
        try:
            gw_control.ensure(logger)
        except TunnelError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc

        # 2. Pin a login node through the authenticated connection.
        if not session.login_node:
            import subprocess as _sp
            try:
                result = _sp.run(
                    ["ssh", *gw_control.ssh_options(), remote.gateway, "hostname"],
                    capture_output=True, text=True, check=True,
                )
                session.login_node = result.stdout.strip()
                session.save()
                logger.info("Pinned login node: %s", session.login_node)
            except _sp.CalledProcessError as exc:
                typer.echo(
                    f"Failed to reach remote gateway {remote.gateway}: "
                    f"{exc.stderr.strip()}",
                    err=True,
                )
                raise typer.Exit(code=1) from exc

        # 3. Establish ControlMaster to the login node (goes through
        #    the gateway ControlMaster — no re-auth needed).
        ln_control = SshControl(
            gateway=session.login_node,
            control_persist=remote.control_persist,
            jump_host=remote.gateway,
            jump_control=gw_control,
        )
        try:
            ln_control.ensure(logger)
        except TunnelError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc

        # 4. If SLURM is configured, allocate a compute node and
        #    establish a ControlMaster through the login node to it.
        #    The login node becomes a pure TCP proxy — no shell, no load.
        target_node = session.login_node
        target_control = ln_control
        if remote.slurm is not None:
            target_node, target_control = _ensure_slurm_node(
                remote, session, ln_control, gw_control, logger,
            )

        # The executor uses the target node ControlMaster directly —
        # no -J needed since the socket routes through the gateway.
        return RemoteExecutor(
            human_user=config.human_user,
            agent_user=config.human_user,  # Same user on remote
            agent_group=config.human_user,
            logger=logger,
            dry_run=dry_run,
            use_sudo_for_agent=False,
            gateway=remote.gateway,
            login_node=target_node,
            remote_mirror_root=str(remote.mirror_root),
            local_mirror_root=str(config.mirror_root),
            ssh_options=remote.ssh_options,
            control_socket_path=str(target_control.socket_path),
        )

    return CommandExecutor(
        human_user=config.human_user,
        agent_user=config.agent_user,
        agent_group=config.agent_group,
        logger=logger,
        dry_run=dry_run,
        use_sudo_for_agent=use_sudo_for_agent,
    )


def _ensure_slurm_node(
    remote,
    session,
    ln_control,
    gw_control,
    logger,
):
    """Allocate a SLURM compute node and establish SSH through the login node.

    Re-uses an existing allocation if the session already has a live
    SLURM job.  Returns ``(compute_node, SshControl)`` for the compute
    node.
    """
    import subprocess as _sp
    from .tunnel import SshControl, TunnelError

    slurm = remote.slurm

    # Check whether a previous allocation is still running.
    if session.slurm_job_id and session.compute_node:
        check_cmd = [
            "ssh", *ln_control.ssh_options(), session.login_node,
            f"squeue --job {session.slurm_job_id} --noheader -o %T",
        ]
        result = _sp.run(check_cmd, capture_output=True, text=True, check=False)
        state = result.stdout.strip()
        if state in ("RUNNING", "PENDING"):
            logger.info(
                "Reusing SLURM job %d on %s (state: %s)",
                session.slurm_job_id, session.compute_node, state,
            )
        else:
            logger.info(
                "Previous SLURM job %d is %s; allocating a new node",
                session.slurm_job_id, state or "gone",
            )
            session.slurm_job_id = None
            session.compute_node = None

    # Allocate a new compute node if needed.
    if not session.slurm_job_id:
        salloc_parts = [
            "salloc", "--no-shell",
            f"--partition={slurm.partition}",
            f"--account={slurm.account}",
            f"--time={slurm.time}",
        ]
        if slurm.qos:
            salloc_parts.append(f"--qos={slurm.qos}")

        salloc_cmd_str = " ".join(salloc_parts)
        ssh_cmd = [
            "ssh", *ln_control.ssh_options(), session.login_node,
            salloc_cmd_str,
        ]
        typer.echo(f"Requesting SLURM allocation ({slurm.partition}, {slurm.time})...")
        logger.debug("salloc command: %s", ssh_cmd)

        try:
            result = _sp.run(ssh_cmd, capture_output=True, text=True, check=True)
        except _sp.CalledProcessError as exc:
            typer.echo(
                f"Failed to allocate SLURM node: {exc.stderr.strip()}",
                err=True,
            )
            raise typer.Exit(code=1) from exc

        # Parse job ID from salloc output.  Typical output:
        #   "salloc: Granted job allocation 12345678"
        combined = result.stdout + result.stderr
        job_id = None
        for line in combined.splitlines():
            if "Granted job allocation" in line:
                for token in line.split():
                    if token.isdigit():
                        job_id = int(token)
                        break
            if job_id:
                break

        if not job_id:
            typer.echo(
                f"Could not parse SLURM job ID from salloc output:\n{combined}",
                err=True,
            )
            raise typer.Exit(code=1)

        # Query squeue for the node name.
        squeue_cmd = [
            "ssh", *ln_control.ssh_options(), session.login_node,
            f"squeue --job {job_id} --noheader -o %N",
        ]
        try:
            result = _sp.run(squeue_cmd, capture_output=True, text=True, check=True)
        except _sp.CalledProcessError as exc:
            typer.echo(
                f"Failed to query node for job {job_id}: {exc.stderr.strip()}",
                err=True,
            )
            raise typer.Exit(code=1) from exc

        compute_node = result.stdout.strip()
        if not compute_node:
            typer.echo(f"squeue returned empty node name for job {job_id}.", err=True)
            raise typer.Exit(code=1)

        session.slurm_job_id = job_id
        session.compute_node = compute_node
        session.save()
        typer.echo(f"Allocated compute node {compute_node} (job {job_id})")
        logger.info("SLURM job %d allocated node %s", job_id, compute_node)

    # Establish ControlMaster to the compute node via the login node.
    # Compute nodes are ephemeral with rotating host keys; skip
    # strict checking to avoid interactive prompts that break the
    # ControlMaster handshake.
    cn_control = SshControl(
        gateway=session.compute_node,
        control_persist=remote.control_persist,
        jump_host=session.login_node,
        jump_control=ln_control,
        extra_options=[
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
        ],
    )
    try:
        cn_control.ensure(logger)
    except TunnelError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    return session.compute_node, cn_control


def _prompt_yes_no(message: str) -> bool:
    return typer.confirm(message, default=True)


def _get_active_target(ctx: Optional[click.Context]) -> Optional["RemoteConfig"]:
    """Return the resolved --target from the Typer context, if any."""
    obj = (ctx.obj if ctx and ctx.obj else {}) or {}
    return obj.get("target")


def _build_manager_for_mirror(
    config: Config, logger, dry_run: bool, mirror_name: str,
) -> MirrorManager:
    """Build a MirrorManager with the correct executor for the given mirror.

    When ``--target`` was passed on the CLI, its :class:`RemoteConfig`
    is applied to the mirror settings (overriding any per-mirror
    ``remote`` block).  For local execution the standard
    :class:`CommandExecutor` is used.
    """
    settings = config.mirrors.get(mirror_name)

    # Overlay the CLI target onto the mirror settings if provided.
    try:
        click_ctx = click.get_current_context()
    except RuntimeError:
        click_ctx = None
    target = _get_active_target(click_ctx)
    if target is not None and settings is not None:
        # Apply target's remote config to a copy of the settings and
        # store it back so that context_for() also sees it.
        from dataclasses import replace
        settings = replace(settings, remote=target)
        config.mirrors[mirror_name] = settings  # type: ignore[index]

    return _build_manager(config, logger, dry_run, mirror_settings=settings)


def _build_manager(
    config: Config, logger, dry_run: bool, *, mirror_settings: Optional[MirrorSettings] = None,
) -> MirrorManager:
    ctx = None
    try:
        ctx = click.get_current_context()
    except RuntimeError:
        ctx = None

    executor = _build_executor(
        config,
        logger,
        dry_run=dry_run,
        use_sudo_for_agent=_get_use_sudo_for_agent(ctx, config),
        mirror_settings=mirror_settings,
    )
    return MirrorManager(config, executor, logger, prompt_handler=_prompt_yes_no)


def _resolve_mirror_name(ctx: typer.Context, mirror: Optional[str]) -> str:
    """Return the mirror name, defaulting to the sole mirror when omitted.

    Resolution order when *mirror* is ``None``:

    1. If the config contains exactly one mirror, use it.
    2. Detect the git root of the current working directory and match it
       against the ``canonical_repo`` of every configured mirror.
    3. If no configured mirror matches, create an ephemeral
       :class:`MirrorSettings` from the git root (analogous to
       :func:`build_default_config`) and inject it into the in-memory
       config so downstream code can use it normally.
    4. If we're not inside a git repo at all, fall through to the
       original "Multiple mirrors configured" error.
    """
    if mirror is not None:
        return mirror
    config = _get_config(ctx)
    names = list(config.mirrors.keys())
    if len(names) == 1:
        return names[0]

    # Step 1 & 2: try git-based detection
    try:
        git_toplevel = _detect_git_toplevel()
    except ConfigError:
        # Not inside a git repo – fall through to the error.
        raise typer.BadParameter(
            f"Multiple mirrors configured; specify one of: {', '.join(sorted(names))}",
            param_hint="MIRROR",
        )

    # Step 1: match cwd's repo root against configured mirrors.
    resolved_toplevel = git_toplevel.resolve()
    for name, settings in config.mirrors.items():
        if settings.canonical_repo.resolve() == resolved_toplevel:
            return name

    # Step 2: create an ephemeral mirror for an unconfigured repo.
    mirror_name = git_toplevel.name
    if mirror_name not in config.mirrors:
        prefixes = BranchPrefixes(human=config.human_user, agent=config.agent_user)
        launcher = config.agent_launcher or AgentLauncher()
        ephemeral = MirrorSettings(
            name=mirror_name,
            canonical_repo=git_toplevel,
            mirror_name=mirror_name,
            branch_prefixes=prefixes,
            agent_launcher=launcher,
            skills=list(config.skills),
        )
        # config.mirrors is typed as Mapping but is a plain dict at runtime.
        config.mirrors[mirror_name] = ephemeral  # type: ignore[index]
        return mirror_name

    # Name collision with an existing mirror – require explicit selection.
    raise typer.BadParameter(
        f"Multiple mirrors configured; specify one of: {', '.join(sorted(names))}",
        param_hint="MIRROR",
    )


def _agent_shorthand(name: str) -> List[str]:
    """Turn a short agent name into a command list."""
    return [name]


def _parse_agent_command(command: Optional[str]) -> Optional[List[str]]:
    if command is None:
        return None
    parts = shlex.split(command)
    return parts or None


def _parse_agent_env(entries: Optional[List[str]]) -> Optional[Dict[str, str]]:
    if not entries:
        return None
    env: Dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise typer.BadParameter(
                f"Environment override must be KEY=VALUE, received `{entry}`."
            )
        key, value = entry.split("=", 1)
        if not key:
            raise typer.BadParameter("Environment variable name cannot be empty.")
        env[key] = value
    return env


@app.callback()
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration YAML (defaults to ~/.sucoder/config.yaml).",
    ),
    use_sudo_for_agent: bool = typer.Option(
        True,
        "--agent-sudo/--no-agent-sudo",
        help="Use sudo to impersonate the agent user when running agent commands (default: enabled).",
    ),
    target: Optional[str] = typer.Option(
        None,
        "--target",
        "-T",
        help="Named execution target (e.g. 'savio'). Omit for local execution.",
    ),
) -> None:
    """Load configuration once and store it on the Typer context."""
    config_explicitly_set = config is not None
    default_path = _default_config_path()
    is_default_config = False
    config_path: Optional[Path] = None

    if config_explicitly_set:
        # User passed --config explicitly; always load from file.
        config_path = Path(config).expanduser()  # type: ignore[arg-type]
        try:
            loaded_config = load_config(config_path)
        except ConfigError as exc:
            typer.echo(f"Configuration error: {exc}", err=True)
            raise typer.Exit(code=2) from exc
    elif default_path.exists():
        # Default config file exists; load it.
        config_path = default_path
        try:
            loaded_config = load_config(config_path)
        except ConfigError as exc:
            typer.echo(f"Configuration error: {exc}", err=True)
            raise typer.Exit(code=2) from exc
    else:
        # Zero-config mode: derive configuration from the environment.
        try:
            loaded_config = build_default_config()
        except ConfigError as exc:
            typer.echo(f"Configuration error: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        is_default_config = True

    try:
        run_startup_checks(
            loaded_config,
            config_path,
            use_sudo=use_sudo_for_agent,
        )
    except StartupError as exc:
        if is_default_config:
            typer.echo(f"Warning: {exc}", err=True)
        else:
            typer.echo(f"Startup validation failed: {exc}", err=True)
            raise typer.Exit(code=2) from exc

    # Resolve --target to a RemoteConfig (or None for local).
    resolved_target: Optional["RemoteConfig"] = None
    if target is not None:
        try:
            resolved_target = loaded_config.resolve_target(target)
        except ConfigError as exc:
            typer.echo(f"Target error: {exc}", err=True)
            raise typer.Exit(code=2) from exc

    ctx.obj = {
        "config": loaded_config,
        "config_path": config_path,
        "use_sudo_for_agent": use_sudo_for_agent,
        "is_default_config": is_default_config,
        "target": resolved_target,
        "target_name": target,
    }


@app.command("version")
def version() -> None:
    """Display version information."""
    typer.echo(__version__)


@app.command("agents-clone")
def agents_clone(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Mirror name defined in configuration.", shell_complete=_mirror_completion),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without executing."),
    lfs: bool = typer.Option(
        False,
        "--lfs/--no-lfs",
        help="Download Git LFS objects during clone (default: skip LFS to avoid failures).",
    ),
) -> None:
    """Clone the canonical repository into an agent-controlled mirror."""
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)
    manager = _build_manager_for_mirror(config, logger, dry_run, mirror)
    mirror_ctx = manager.context_for(mirror)
    try:
        if mirror_ctx.is_remote:
            manager.ensure_remote_clone(mirror_ctx)
        else:
            manager.ensure_clone(mirror_ctx, skip_lfs=not lfs)
    except MirrorError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command("prepare-canonical")
def prepare_canonical(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Mirror name defined in configuration.", shell_complete=_mirror_completion),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without executing."),
    use_sudo: bool = typer.Option(
        False,
        "--sudo/--no-sudo",
        help="Prefix corrective commands with sudo (default: --no-sudo).",
    ),
    agent_remote: bool = typer.Option(
        True,
        "--agent-remote/--no-agent-remote",
        help="Configure a remote named after the agent prefix pointing to the mirror (default: enabled).",
    ),
) -> None:
    """Fix ownership and permissions on the canonical repository."""
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)
    manager = _build_manager(config, logger, dry_run=dry_run)
    try:
        manager.prepare_canonical(
            manager.context_for(mirror),
            use_sudo=use_sudo,
            setup_agent_remote=agent_remote,
        )
    except MirrorError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command("sync")
def sync(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Mirror name defined in configuration.", shell_complete=_mirror_completion),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without executing."),
) -> None:
    """Fetch updates from the canonical repository."""
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)
    manager = _build_manager_for_mirror(config, logger, dry_run, mirror)
    try:
        manager.sync(manager.context_for(mirror))
    except MirrorError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command("start-task")
def start_task(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Mirror name defined in configuration.", shell_complete=_mirror_completion),
    task: str = typer.Argument(..., help="Task identifier used to name the branch."),
    base: Optional[str] = typer.Option(
        None,
        "--base",
        "-b",
        help="Base branch name (falls back to mirror default).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without executing."),
) -> None:
    """Create and check out a task branch for the agent."""
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)
    manager = _build_manager(config, logger, dry_run=dry_run)
    try:
        branch = manager.start_task(
            manager.context_for(mirror),
            task_name=task,
            base_branch=base,
        )
    except MirrorError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(branch)


@app.command("status")
def status(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Mirror name defined in configuration.", shell_complete=_mirror_completion),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
) -> None:
    """Display git status for the mirror."""
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)
    manager = _build_manager_for_mirror(config, logger, False, mirror)
    try:
        output = manager.status(manager.context_for(mirror))
    except MirrorError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(output)


@app.command("worktrees")
def worktrees(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Mirror name defined in configuration.", shell_complete=_mirror_completion),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
    diff: bool = typer.Option(False, "--diff", help="Include diff --stat for each worktree."),
    base: Optional[str] = typer.Option(
        None,
        "--base",
        "-b",
        help="Base branch for ahead count (defaults to mirror setting).",
    ),
    show_main: bool = typer.Option(False, "--main", help="Include the main worktree in the listing."),
    watch: Optional[int] = typer.Option(
        None,
        "--watch",
        "-w",
        help="Refresh every N seconds.",
    ),
) -> None:
    """List active worktrees in the mirror with status details."""
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)
    manager = _build_manager_for_mirror(config, logger, False, mirror)
    mirror_ctx = manager.context_for(mirror)

    def _display() -> None:
        try:
            output = manager.worktrees_summary(
                mirror_ctx,
                include_diff=diff,
                base_branch=base,
                include_main=show_main,
            )
        except MirrorError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(output)

    if watch is not None:
        import time
        try:
            while True:
                typer.clear()
                _display()
                time.sleep(watch)
        except KeyboardInterrupt:
            pass
    else:
        _display()


@app.command("agents-run")
def agents_run(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Mirror name defined in configuration.", shell_complete=_mirror_completion),
    task: Optional[str] = typer.Option(
        None,
        "--task",
        "-t",
        help="Optional task identifier for creating a fresh branch before launch.",
    ),
    base: Optional[str] = typer.Option(
        None,
        "--base",
        "-b",
        help="Base branch to use when creating a task branch.",
    ),
    sync: bool = typer.Option(
        True,
        "--sync/--no-sync",
        help="Fetch latest canonical changes before launching (ignored when --task is used).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without executing."),
    agent: Optional[str] = typer.Option(
        None,
        "--agent",
        "-a",
        help="Agent to use (e.g. claude, codex, gemini).",
    ),
    agent_command: Optional[str] = typer.Option(
        None,
        "--agent-command",
        help="Override the full agent command (example: --agent-command 'foo --flag').",
    ),
    agent_env: Optional[List[str]] = typer.Option(
        None,
        "--agent-env",
        help="Override or add agent environment variables (repeat as KEY=VALUE).",
        metavar="KEY=VALUE",
    ),
    inline_prompt: Optional[bool] = typer.Option(
        None,
        "--inline-prompt",
        is_flag=False,
        help="Force whether context prelude text is appended to the agent command (true/false).",
    ),
    lfs: bool = typer.Option(
        False,
        "--lfs/--no-lfs",
        help="Download Git LFS objects during clone (default: skip LFS to avoid failures).",
    ),
    extra_args: Optional[List[str]] = typer.Argument(
        None,
        help="Additional arguments appended to the agent launch command.",
        metavar="ARGS...",
    ),
) -> None:
    """Launch the configured agent inside the mirror working tree."""
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)
    manager = _build_manager_for_mirror(config, logger, dry_run, mirror)
    command_override = _parse_agent_command(agent_command) or (_agent_shorthand(agent) if agent else None)
    env_override = _parse_agent_env(agent_env)
    try:
        manager.launch_agent(
            manager.context_for(mirror),
            sync=sync,
            task_name=task,
            base_branch=base,
            extra_args=extra_args,
            command_override=command_override,
            env_override=env_override,
            supports_inline_prompt=inline_prompt,
        )
    except MirrorError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command("collaborate")
def collaborate(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Mirror name defined in configuration.", shell_complete=_mirror_completion),
    task: Optional[str] = typer.Option(
        None,
        "--task",
        "-t",
        help="Optional task identifier used to create a task branch before launch.",
    ),
    base: Optional[str] = typer.Option(
        None,
        "--base",
        "-b",
        help="Base branch to use when creating a task branch (defaults to mirror setting).",
    ),
    sync: bool = typer.Option(
        True,
        "--sync/--no-sync",
        help="Fetch latest canonical changes before launching (ignored when --task is used).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without executing."),
    use_sudo: bool = typer.Option(
        False,
        "--sudo/--no-sudo",
        help="Prefix canonical permission commands with sudo (default: --no-sudo).",
    ),
    agent_remote: bool = typer.Option(
        True,
        "--agent-remote/--no-agent-remote",
        help="Configure a remote pointing at the agent mirror during canonical prep (default: enabled).",
    ),
    agent: Optional[str] = typer.Option(
        None,
        "--agent",
        "-a",
        help="Agent to use (e.g. claude, codex, gemini).",
    ),
    agent_command: Optional[str] = typer.Option(
        None,
        "--agent-command",
        help="Override the full agent command (example: --agent-command 'foo --flag').",
    ),
    agent_env: Optional[List[str]] = typer.Option(
        None,
        "--agent-env",
        help="Override or add agent environment variables (repeat as KEY=VALUE).",
        metavar="KEY=VALUE",
    ),
    inline_prompt: Optional[bool] = typer.Option(
        None,
        "--inline-prompt",
        is_flag=False,
        help="Force whether context prelude text is appended to the agent command (true/false).",
    ),
    lfs: bool = typer.Option(
        False,
        "--lfs/--no-lfs",
        help="Download Git LFS objects during clone (default: skip LFS to avoid failures).",
    ),
    extra_args: Optional[List[str]] = typer.Argument(
        None,
        help="Additional arguments appended to the agent launch command.",
        metavar="ARGS...",
    ),
) -> None:
    """Prepare canonical, ensure the mirror exists, and launch the agent in one step."""
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)
    manager = _build_manager_for_mirror(config, logger, dry_run, mirror)
    command_override = _parse_agent_command(agent_command) or (_agent_shorthand(agent) if agent else None)
    env_override = _parse_agent_env(agent_env)
    try:
        manager.bootstrap(
            manager.context_for(mirror),
            use_sudo=use_sudo,
            setup_agent_remote=agent_remote,
            sync=sync,
            task_name=task,
            base_branch=base,
            extra_args=extra_args,
            command_override=command_override,
            env_override=env_override,
            supports_inline_prompt=inline_prompt,
            skip_lfs=not lfs,
        )
    except MirrorError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command("attach")
def attach(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Mirror name defined in configuration.", shell_complete=_mirror_completion),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
) -> None:
    """Reconnect to an existing remote agent session via tmux."""
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    settings = config.mirrors.get(mirror)

    # Apply --target overlay so `-T savio attach` works the same as
    # `-T savio collaborate`.
    try:
        click_ctx = click.get_current_context()
    except RuntimeError:
        click_ctx = None
    target = _get_active_target(click_ctx)
    if target is not None and settings is not None:
        from dataclasses import replace
        settings = replace(settings, remote=target)

    if not settings or not settings.remote:
        typer.echo("Mirror is not configured for remote execution.", err=True)
        raise typer.Exit(code=1)

    from .session import RemoteSession
    from .tunnel import SshControl

    session = RemoteSession.load(mirror)
    if not session.login_node:
        typer.echo(
            "No active session found. Run 'sucoder collaborate' or "
            "'sucoder agents-run' first to establish a session.",
            err=True,
        )
        raise typer.Exit(code=1)

    remote = settings.remote
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)

    # Reuse ControlMaster if active; re-establish if expired.
    control = SshControl(
        gateway=remote.gateway,
        control_persist=remote.control_persist,
    )
    try:
        control.ensure(logger)
    except Exception:
        pass  # Best-effort; ssh will prompt directly if needed
    control_opts = control.ssh_options() if control.is_active() else []

    # For SLURM targets, attach to the compute node (via login node).
    if session.compute_node and remote.slurm is not None:
        attach_target = session.compute_node
        jump_chain = f"{remote.gateway},{session.login_node}"
    else:
        attach_target = session.login_node
        jump_chain = remote.gateway

    tmux_name = f"sucoder-{mirror}"
    attach_cmd = (
        f"tmux attach-session -t {shlex.quote(tmux_name)} "
        f"|| tmux new-session -s {shlex.quote(tmux_name)}"
    )
    os.execvp("ssh", [
        "ssh", "-t",
        *control_opts,
        "-J", jump_chain,
        attach_target,
        attach_cmd,
    ])


@app.command("mirrors-list")
def mirrors_list(ctx: typer.Context) -> None:
    """Display configured mirrors with their canonical repositories."""
    config = _get_config(ctx)
    entries = sorted(config.mirrors.items())
    if not entries:
        typer.echo("No mirrors configured.")
        return

    name_width = max(len("Mirror"), *(len(name) for name, _ in entries))
    branch_width = max(len("Base"), *(len(settings.default_base_branch) for _, settings in entries))

    header = f"{'Mirror':<{name_width}}  {'Base':<{branch_width}}  Canonical Repo  Mirror Path"
    typer.echo(header)
    typer.echo("-" * len(header))

    for name, settings in entries:
        canonical = str(settings.canonical_repo)
        mirror_path = str(config.mirror_root / settings.mirror_dirname)
        base = settings.default_base_branch
        typer.echo(f"{name:<{name_width}}  {base:<{branch_width}}  {canonical}  {mirror_path}")


@app.command("skills-list")
def skills_list(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
) -> None:
    """List configured skill paths and highlight accessibility issues."""
    config = _get_config(ctx)
    logger = setup_logger("sucoder.skills", config.log_dir, verbose)
    executor = _build_executor(
        config,
        logger,
        dry_run=False,
        use_sudo_for_agent=_get_use_sudo_for_agent(ctx, config),
    )

    path_usage: Dict[Path, Set[str]] = defaultdict(set)
    for mirror_name, settings in config.mirrors.items():
        for entry in settings.skills:
            path_usage[entry].add(mirror_name)

    default_skills_dir = Path("~/.sucoder/skills").expanduser()
    path_usage.setdefault(default_skills_dir, set()).add("default")

    catalog_path = MirrorManager._default_skills_catalog_path()
    if catalog_path:
        path_usage[catalog_path].add("catalog")
        path_usage.setdefault(catalog_path.parent, set()).add("catalog")

    if not path_usage:
        typer.echo("No skill paths configured.")
        raise typer.Exit(code=1)

    exit_code = 0
    for raw_path in sorted(path_usage.keys(), key=lambda p: str(p)):
        contexts = ", ".join(sorted(path_usage[raw_path]))
        try:
            path = raw_path
            exists = path.exists()
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Failed to stat skill path %s: %s", raw_path, exc)
            typer.secho(f"[ERROR] {raw_path} (contexts: {contexts}) – {exc}", fg="red")
            exit_code = 1
            continue

        if exists:
            if path.is_dir():
                readable = path.is_dir() and _agent_can_access_path(path, executor)
                status = "OK" if readable else "UNREADABLE"
                color = "green" if readable else "yellow"
                typer.secho(
                    f"[{status}] directory {path} (contexts: {contexts})",
                    fg=color,
                )
                if readable:
                    entries = _collect_directory_preview(path, executor)
                    if entries:
                        typer.echo(f"  sample: {', '.join(entries)}")
                else:
                    exit_code = 1
            else:
                readable = _agent_can_read_file(path, executor)
                status = "OK" if readable else "UNREADABLE"
                color = "green" if readable else "yellow"
                typer.secho(
                    f"[{status}] file {path} (contexts: {contexts})",
                    fg=color,
                )
                if not readable:
                    exit_code = 1
        else:
            typer.secho(
                f"[MISSING] {path} (contexts: {contexts})",
                fg="red",
            )
            exit_code = 1

    raise typer.Exit(code=exit_code)


def _agent_can_read_file(path: Path, executor: CommandExecutor) -> bool:
    """Return True when the agent user can read the file."""
    if not path.exists() or path.is_dir():
        return False
    result = executor.run_agent(
        ["test", "-r", str(path)],
        check=False,
    )
    return result.returncode == 0


def _agent_can_access_path(path: Path, executor: CommandExecutor) -> bool:
    """Return True when the agent user can read and execute the directory."""
    if not path.exists() or not path.is_dir():
        return False
    result = executor.run_agent(
        ["test", "-r", str(path), "-a", "-x", str(path)],
        check=False,
    )
    return result.returncode == 0


def _collect_directory_preview(path: Path, executor: CommandExecutor, limit: int = 6) -> List[str]:
    """Return a small, sorted sample of entries from a directory."""
    try:
        result = executor.run_agent(
            ["ls", "-1", str(path)],
            check=False,
        )
        if result.returncode != 0:
            typer.secho(f"  failed to list {path}: {result.stderr.strip()}", fg="yellow")
            return []
        entries = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
    except OSError as exc:  # pragma: no cover - defensive
        typer.secho(f"  failed to list {path}: {exc}", fg="yellow")
        return []
    if len(entries) > limit:
        extras = len(entries) - limit
        trimmed = entries[:limit]
        trimmed.append(f"...(+{extras})")
        return trimmed
    return entries
