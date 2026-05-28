"""Typer-powered CLI entry point."""

from __future__ import annotations

import os
import shlex
import pwd
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
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


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


app = typer.Typer(help="sucoder – Unix-sandboxed agent collaboration toolkit for managing agent mirrors.")

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@contextmanager
def _spinner(message: str):
    """Show an animated spinner with *message* while the body executes.

    The spinner writes to stderr so it doesn't pollute captured stdout.
    When the body finishes the spinner line is replaced with a final
    status (done or error).

    DO NOT use this around code that may trigger an interactive SSH
    auth prompt: the 100 ms refresh overwrites the password/OTP prompt
    on the terminal, producing what looks like a silent hang.  Use
    :func:`_ensure_ssh_visible` for SSH ControlMaster setup instead.
    """
    is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    if not is_tty:
        typer.echo(message + " ...", err=True)
        yield
        return

    stop = threading.Event()
    error = [False]

    def _spin():
        i = 0
        while not stop.is_set():
            frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
            sys.stderr.write(f"\r{frame} {message} ...")
            sys.stderr.flush()
            i += 1
            stop.wait(0.1)

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    try:
        yield
    except BaseException:
        error[0] = True
        raise
    finally:
        stop.set()
        t.join()
        if error[0]:
            sys.stderr.write(f"\r✗ {message}\n")
        else:
            sys.stderr.write(f"\r✓ {message}\n")
        sys.stderr.flush()


def _ensure_ssh_visible(control, label: str, logger) -> None:
    """Bring a ControlMaster (and any jump-host parents) online with
    auth prompts visible to the user.

    Replaces ``with _spinner(...): control.ensure(logger)`` for SSH
    chain setup.  ``ensure()`` may recurse into ``jump_control.ensure``
    when a parent socket has expired, which triggers an interactive
    auth prompt (password / OTP) written to ``/dev/tty``.  Inside a
    spinner block, the 100 ms refresh thread overwrites that prompt
    and the user sees only the spinner — indistinguishable from a
    network hang.

    By echoing a plain ``"Connecting to X ..."`` line and letting
    ``ensure()`` run without animation, the prompt stays on screen.
    Raises :class:`TunnelError` unchanged so callers can decide
    whether to exit or fall back.
    """
    typer.echo(f"Connecting to {label} ...", err=True)
    control.ensure(logger)
    typer.echo(f"✓ Connected to {label}", err=True)


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
    debug_ssh: bool = False,
    local_disk_override: Optional[bool] = None,
    cli_ctx: Optional[click.Context] = None,
) -> CommandExecutor:
    if mirror_settings and mirror_settings.remote:
        from .executor import RemoteExecutor
        from .session import RemoteSession
        from .tunnel import SshControl, TunnelError

        remote = mirror_settings.remote

        # Resolve the target name for session scoping.  Prefer the
        # explicit ``cli_ctx``; fall back to ``click.get_current_context``
        # as a best-effort -- typer >=0.21 (with typer 0.26 in particular)
        # no longer pushes its Context onto Click's global stack, so the
        # fallback raises RuntimeError in normal CLI invocations and we
        # end up with ``_target_name = None``.  Callers in subcommands
        # have the typer.Context in hand and MUST pass it as ``cli_ctx``.
        if cli_ctx is None:
            try:
                cli_ctx = click.get_current_context()
            except RuntimeError:
                cli_ctx = None
        _target_name = ((cli_ctx.obj or {}).get("target_name") if cli_ctx else None)
        session = RemoteSession.load(mirror_settings.name, target_name=_target_name)

        # 1. Establish ControlMaster to the gateway (authenticates
        #    once; may prompt for pin + OTP).
        gw_control = SshControl(
            gateway=remote.gateway,
            control_persist=remote.control_persist,
            debug=debug_ssh,
        )
        try:
            gw_control.ensure(logger)
        except TunnelError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc

        # 2. Pin a login node through the authenticated connection.
        if not session.login_node:
            import subprocess as _sp
            pin_cmd = ["ssh"]
            if debug_ssh:
                pin_cmd.append("-vvv")
            pin_cmd.extend([*gw_control.ssh_options(), remote.gateway, "hostname"])
            with _spinner("Resolving login node"):
                try:
                    result = _sp.run(
                        pin_cmd,
                        capture_output=True, text=True, check=True,
                    )
                    if debug_ssh and result.stderr:
                        logger.debug("SSH debug (pin login node):\n%s", result.stderr.rstrip())
                    session.login_node = result.stdout.strip()
                    session.save()
                    logger.info("Pinned login node: %s", session.login_node)
                except _sp.CalledProcessError as exc:
                    if debug_ssh and exc.stderr:
                        logger.error("SSH debug (pin login node FAILED):\n%s", exc.stderr.rstrip())
                    typer.echo(
                        f"Failed to reach remote gateway {remote.gateway}: "
                        f"{exc.stderr.strip()}",
                        err=True,
                    )
                    raise typer.Exit(code=1) from exc

        # 3. Establish ControlMaster to the login node (goes through
        #    the gateway ControlMaster — no re-auth needed if gw is
        #    still fresh; if it's not, ensure() will recurse into
        #    jump_control.ensure() and prompt for the gateway password
        #    again.  That prompt must be visible (no spinner overlay),
        #    so we use _ensure_ssh_visible instead of _spinner.
        ln_control = SshControl(
            gateway=session.login_node,
            control_persist=remote.control_persist,
            jump_host=remote.gateway,
            jump_control=gw_control,
            debug=debug_ssh,
        )
        try:
            _ensure_ssh_visible(ln_control, session.login_node, logger)
        except TunnelError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc

        # 3b. Establish ControlMaster to the data transfer node (DTN).
        #     The DTN has fat pipes and spare CPU compared with the
        #     hammered login nodes, so filesystem scaffolding and git
        #     transport route through it.
        dtn_control = SshControl(
            gateway=remote.transfer_host,
            control_persist=remote.control_persist,
            jump_host=remote.gateway,
            jump_control=gw_control,
            debug=debug_ssh,
        )
        try:
            _ensure_ssh_visible(dtn_control, remote.transfer_host, logger)
        except TunnelError as exc:
            # DTN is optional — fall back to the login node if
            # the DTN is unreachable.
            logger.warning(
                "DTN %s unreachable, falling back to login node: %s",
                remote.transfer_host, exc,
            )
            dtn_control = ln_control

        # 4. If SLURM is configured, allocate a compute node and
        #    establish a ControlMaster through the login node to it.
        #    The login node becomes a pure TCP proxy — no shell, no load.
        target_node = session.login_node
        target_control = ln_control
        prev_compute_node = session.compute_node
        if remote.slurm is not None:
            target_node, target_control = _ensure_slurm_node(
                remote, session, ln_control, gw_control, logger,
                debug_ssh=debug_ssh,
            )

        # Resolve local-disk setting: CLI flag overrides config.
        use_local_disk = False
        local_disk_root = ""
        cfg_local_disk = remote.slurm.local_disk if remote.slurm else None
        if remote.slurm is not None:
            if local_disk_override is True:
                use_local_disk = True
                local_disk_root = cfg_local_disk or "/local"
            elif local_disk_override is False:
                use_local_disk = False
            elif cfg_local_disk:
                use_local_disk = True
                local_disk_root = cfg_local_disk
            if use_local_disk:
                logger.info(
                    "Using local disk %s on compute node (bypassing shared FS)",
                    local_disk_root,
                )

        # Compute the remote mirror root.  With local disk, the mirror
        # lives on the compute node's local storage; otherwise on the
        # shared filesystem (Lustre).
        #
        # When the user hasn't expressed a preference (no --local-disk
        # flag AND no config setting), fall back to the session's saved
        # value.  This lets `sucoder pull` find the mirror without
        # re-specifying --local-disk.
        #
        # However, if the compute node changed (SLURM gave us a
        # different node) and the saved root is a node-local path,
        # that data is unreachable — fall back to shared FS instead.
        node_changed = (
            prev_compute_node is not None
            and session.compute_node is not None
            and prev_compute_node != session.compute_node
        )
        if use_local_disk:
            remote_mirror_root = f"{local_disk_root.rstrip('/')}/mirrors"
        elif local_disk_override is None and not cfg_local_disk and session.remote_mirror_root:
            saved_root = session.remote_mirror_root
            if node_changed and saved_root != str(remote.mirror_root):
                # The saved mirror root was on a different node's local
                # disk; that storage is unreachable from the new node.
                remote_mirror_root = str(remote.mirror_root)
                logger.warning(
                    "Compute node changed (%s -> %s); discarding stale "
                    "local-disk mirror root %s — using shared FS: %s",
                    prev_compute_node, session.compute_node,
                    saved_root, remote_mirror_root,
                )
            else:
                remote_mirror_root = saved_root
                if remote_mirror_root != str(remote.mirror_root):
                    logger.info("Using saved mirror root from session: %s", remote_mirror_root)
        else:
            remote_mirror_root = str(remote.mirror_root)

        # The executor uses the target node ControlMaster directly —
        # no -J needed since the socket routes through the gateway.
        # For compute-node targets, also pass the login node info so
        # that _build_ssh_command can include a ProxyCommand fallback
        # through the login node if the compute-node socket is stale.
        executor_kwargs: Dict[str, Any] = dict(
            human_user=config.human_user,
            agent_user=config.human_user,  # Same user on remote
            agent_group=config.human_user,
            logger=logger,
            dry_run=dry_run,
            use_sudo_for_agent=False,
            gateway=remote.gateway,
            login_node=target_node,
            remote_mirror_root=remote_mirror_root,
            local_mirror_root=str(config.mirror_root),
            ssh_options=remote.ssh_options,
            control_socket_path=str(target_control.socket_path),
            is_compute_node=(remote.slurm is not None),
            slurm_job_id=session.slurm_job_id,
            debug_ssh=debug_ssh,
        )
        # Detect whether the resolved mirror root is on local disk
        # (either from --local-disk flag, config, or session fallback).
        is_local_disk = (
            use_local_disk
            or remote_mirror_root != str(remote.mirror_root)
        )
        if is_local_disk:
            # Local disk is only on the compute node — scaffolding
            # and git transport must go through the compute node,
            # not the DTN.  Don't set scaffolding_node so that
            # run_on_login_node falls through to run_agent.
            pass
        else:
            # Route filesystem scaffolding and git transport through
            # the DTN (or login node as fallback).
            executor_kwargs["scaffolding_node"] = str(dtn_control.gateway)
            executor_kwargs["scaffolding_socket_path"] = str(dtn_control.socket_path)
        # For compute-node targets, the proxy fields are still needed
        # for the SSH ProxyCommand fallback to the login node.
        if remote.slurm is not None:
            executor_kwargs["proxy_node"] = session.login_node
            executor_kwargs["proxy_socket_path"] = str(ln_control.socket_path)

        # Persist the effective mirror root so that later commands
        # (e.g. `sucoder pull`) know where the mirror lives without
        # needing --local-disk.
        session.remote_mirror_root = remote_mirror_root
        session.save()

        return RemoteExecutor(**executor_kwargs)

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
    *,
    debug_ssh: bool = False,
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
            # Keep compute_node for --nodelist affinity (local-disk data
            # may still be on that node).
            preferred_node = session.compute_node
            session.compute_node = None

    # Allocate a new compute node if needed.
    if not session.slurm_job_id:
        preferred_node = locals().get("preferred_node")
        salloc_parts = [
            "salloc", "--no-shell",
            f"--partition={slurm.partition}",
            f"--account={slurm.account}",
            f"--time={slurm.time}",
        ]
        if slurm.qos:
            salloc_parts.append(f"--qos={slurm.qos}")
        if slurm.cpus_per_task:
            salloc_parts.append(f"--cpus-per-task={slurm.cpus_per_task}")
        if slurm.mem:
            salloc_parts.append(f"--mem={slurm.mem}")
        if preferred_node:
            salloc_parts.extend([
                f"--nodelist={preferred_node}",
                "--immediate=30",
            ])
            logger.info(
                "Requesting node %s (30s timeout before fallback)",
                preferred_node,
            )

        salloc_cmd_str = " ".join(salloc_parts)
        ssh_cmd = [
            "ssh", *ln_control.ssh_options(), session.login_node,
            salloc_cmd_str,
        ]
        logger.debug("salloc command: %s", ssh_cmd)

        with _spinner(f"Requesting SLURM allocation ({slurm.partition}, {slurm.time})"):
            try:
                result = _sp.run(ssh_cmd, capture_output=True, text=True, check=True)
            except _sp.CalledProcessError as exc:
                if preferred_node:
                    # Preferred node busy — retry without --nodelist.
                    typer.echo(
                        f"⚠  Node {preferred_node} unavailable; "
                        "allocating any node.  Unpulled agent work on "
                        f"{preferred_node}:/local/ may be orphaned — "
                        "run `sucoder pull` when that node is reachable.",
                        err=True,
                    )
                    fallback_parts = [
                        p for p in salloc_parts
                        if not p.startswith("--nodelist=")
                        and not p.startswith("--immediate=")
                    ]
                    fallback_cmd = [
                        "ssh", *ln_control.ssh_options(), session.login_node,
                        " ".join(fallback_parts),
                    ]
                    try:
                        result = _sp.run(
                            fallback_cmd, capture_output=True, text=True, check=True,
                        )
                    except _sp.CalledProcessError as exc2:
                        typer.echo(
                            f"Failed to allocate SLURM node: {exc2.stderr.strip()}",
                            err=True,
                        )
                        raise typer.Exit(code=1) from exc2
                else:
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
        debug=debug_ssh,
    )
    # No spinner here: ensure() may recurse through ln_control and
    # gw_control, either of which can trigger a re-auth prompt.  A
    # spinner would obscure the prompt and look like a hang.
    try:
        _ensure_ssh_visible(
            cn_control, f"compute node {session.compute_node}", logger,
        )
    except TunnelError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    # Start a deadline timer on the compute node so both the human
    # (via tmux status message) and the agent (via a sentinel file)
    # get warnings before the SLURM allocation expires.
    _start_slurm_timer(
        session, ln_control, cn_control, logger,
    )

    return session.compute_node, cn_control


def _start_slurm_timer(
    session,
    ln_control,
    cn_control,
    logger,
):
    """Launch a background timer on the compute node that warns before
    the SLURM allocation expires.

    Writes warnings to ``$HOME/.cache/sucoder/slurm-deadline.warn`` (for
    the agent to check) and flashes ``tmux display-message`` (for the
    human).  Warnings fire at 30, 15, and 5 minutes remaining.

    NOTE: previously these paths lived under ``/tmp/`` which is
    world-writable on shared compute nodes.  Per-session prompts that
    reference the old location (e.g. ``~/.sucoder/prompts/savio-node.org``)
    should be updated to the new path.

    As a backstop, if the sucoder tmux session exits without cancelling
    the job (crash, network loss, etc.), the timer cancels it to avoid
    burning idle allocation time.
    """
    import shlex
    import subprocess as _sp
    import textwrap

    job_id = session.slurm_job_id
    if not job_id:
        return

    tmux_session = f"sucoder-{session.mirror_name}"

    # Defensive shell-quoting.  ``mirror_name`` and therefore
    # ``tmux_session`` come from configuration the user controls; if a
    # mirror were ever named with shell metacharacters the unquoted
    # interpolation below would be a command-injection vector.
    # ``job_id`` is an int from ``int(token)`` so it's already safe, but
    # we quote it for symmetry and to insulate against future changes.
    q_tmux = shlex.quote(tmux_session)
    q_job = shlex.quote(str(job_id))

    # Use a per-user runtime directory rather than world-writable /tmp.
    # On a shared HPC compute node, predictable /tmp/slurm-*.warn paths
    # are subject to symlink races: a co-resident user can pre-create
    # the path as a symlink to a sensitive file and have the timer
    # overwrite it.  ``$HOME/.cache/sucoder/`` is per-user (NFS-shared
    # across nodes, owned by the same uid) and not writable by other
    # local users on the compute node, which closes that vector.
    #
    # The agent reads ``slurm-deadline.warn`` from the same location
    # (the agent runs as the same user inside tmux on the compute
    # node), so the path remains discoverable to consumers.

    # The script runs on the compute node, querying squeue via the
    # login node is unnecessary — SLURM_JOB_ID is in the environment
    # and squeue works locally on compute nodes too.
    timer_script = textwrap.dedent(f"""\
        #!/bin/bash
        set -u
        STATE_DIR="${{HOME}}/.cache/sucoder"
        mkdir -p "$STATE_DIR"
        chmod 700 "$STATE_DIR" 2>/dev/null || true
        WARN_FILE="$STATE_DIR/slurm-deadline.warn"
        WARN5="$STATE_DIR/.slurm-warn-5"
        WARN15="$STATE_DIR/.slurm-warn-15"
        WARN30="$STATE_DIR/.slurm-warn-30"
        rm -f "$WARN5" "$WARN15" "$WARN30" "$WARN_FILE"

        # Wait for the agent tmux session to appear before monitoring.
        # The timer starts before the session is created, so we must
        # not treat its absence as "agent exited".
        TMUX_READY=0
        for i in $(seq 1 120); do
            if tmux has-session -t {q_tmux} 2>/dev/null; then
                TMUX_READY=1
                break
            fi
            sleep 5
        done
        if [ "$TMUX_READY" -eq 0 ]; then
            # User owns SLURM lifecycle (see `sucoder release`); leave
            # the allocation alone even if the agent's tmux session
            # never appeared, since the user may want to debug or
            # reuse the compute node manually.
            echo "Timed out waiting for tmux session {q_tmux}; SLURM job {q_job} kept alive. Run 'sucoder release' or 'scancel {q_job}' to free the allocation." > "$WARN_FILE"
            exit 1
        fi

        while true; do
            left=$(squeue --job {q_job} --noheader -o "%L" 2>/dev/null)
            if [ -z "$left" ]; then
                msg="SLURM job {q_job} is no longer queued — allocation may have ended."
                echo "$msg" > "$WARN_FILE"
                tmux display-message "$msg" 2>/dev/null
                break
            fi

            # If the agent tmux session is gone, write a warning but
            # do NOT auto-cancel the SLURM allocation.  Users own the
            # SLURM lifecycle (use `sucoder release <mirror>` for
            # explicit cancel); an automatic scancel here would tear
            # down the allocation on transient agent failures and
            # destroy any chance of reattaching.
            if ! tmux has-session -t {q_tmux} 2>/dev/null; then
                echo "Agent tmux session is gone; SLURM job {q_job} kept alive. Run 'sucoder release' or 'scancel {q_job}' to free the allocation." > "$WARN_FILE"
                break
            fi

            IFS=: read -ra parts <<< "$left"
            if [ ${{#parts[@]}} -eq 3 ]; then
                mins=$(( ${{parts[0]#0}}*60 + ${{parts[1]#0}} ))
            elif [ ${{#parts[@]}} -eq 2 ]; then
                mins=${{parts[0]#0}}
            else
                mins=999
            fi
            if [ "$mins" -le 5 ] && [ ! -f "$WARN5" ]; then
                msg="SLURM: ~${{mins}} min left (job {q_job}). Commit and save NOW."
                echo "$msg" > "$WARN_FILE"
                tmux display-message -t {q_tmux} "$msg" 2>/dev/null
                touch "$WARN5"
            elif [ "$mins" -le 15 ] && [ ! -f "$WARN15" ]; then
                msg="SLURM: ~${{mins}} min left (job {q_job}). Start wrapping up."
                echo "$msg" > "$WARN_FILE"
                tmux display-message -t {q_tmux} "$msg" 2>/dev/null
                touch "$WARN15"
            elif [ "$mins" -le 30 ] && [ ! -f "$WARN30" ]; then
                msg="SLURM: ~${{mins}} min left (job {q_job})."
                echo "$msg" > "$WARN_FILE"
                tmux display-message -t {q_tmux} "$msg" 2>/dev/null
                touch "$WARN30"
            fi
            sleep 60
        done
    """)

    # Write the script to the compute node via stdin, then run it.
    # The script lives in the user's runtime cache rather than /tmp for
    # the same symlink-race reasons; the path is computed remotely
    # (rather than passed in argv) so a stale local copy can't trip up
    # the SSH command-line.
    ssh_opts = cn_control.ssh_options()
    node = session.compute_node

    write_result = _sp.run(
        ["ssh", *ssh_opts, node,
         'mkdir -p "$HOME/.cache/sucoder" && '
         'chmod 700 "$HOME/.cache/sucoder" 2>/dev/null || true; '
         'cat > "$HOME/.cache/sucoder/slurm-timer.sh" && '
         'chmod 700 "$HOME/.cache/sucoder/slurm-timer.sh"'],
        input=timer_script, capture_output=True, text=True, check=False,
    )
    if write_result.returncode != 0:
        logger.warning("Failed to write SLURM timer script: %s",
                        write_result.stderr.strip())
        return

    run_result = _sp.run(
        ["ssh", *ssh_opts, node,
         'nohup "$HOME/.cache/sucoder/slurm-timer.sh" > /dev/null 2>&1 &'],
        capture_output=True, text=True, check=False,
    )
    if run_result.returncode == 0:
        logger.info("SLURM deadline timer started on %s for job %d",
                     node, job_id)
    else:
        logger.warning("Failed to start SLURM timer: %s",
                        run_result.stderr.strip())


def _prompt_yes_no(message: str) -> bool:
    return typer.confirm(message, default=True)


def _get_active_target(ctx: Optional[click.Context]) -> Optional["RemoteConfig"]:
    """Return the resolved --target from the Typer context, if any."""
    obj = (ctx.obj if ctx and ctx.obj else {}) or {}
    return obj.get("target")


def _get_debug_ssh(ctx: Optional[click.Context]) -> bool:
    """Return whether --debug-ssh was set."""
    obj = (ctx.obj if ctx and ctx.obj else {}) or {}
    return bool(obj.get("debug_ssh", False))


def _get_local_disk_override(ctx: Optional[click.Context]) -> Optional[bool]:
    """Return the --local-disk CLI override, or None to use config default."""
    obj = (ctx.obj if ctx and ctx.obj else {}) or {}
    return obj.get("local_disk")


def _build_manager_for_mirror(
    config: Config, logger, dry_run: bool, mirror_name: str,
    *,
    cli_ctx: Optional[click.Context] = None,
) -> MirrorManager:
    """Build a MirrorManager with the correct executor for the given mirror.

    When ``--target`` was passed on the CLI, its :class:`RemoteConfig`
    is applied to the mirror settings (overriding any per-mirror
    ``remote`` block).  For local execution the standard
    :class:`CommandExecutor` is used.

    Callers in subcommands must pass ``cli_ctx`` (their typer.Context).
    The fallback to ``click.get_current_context`` exists only for
    library-style callers; in typer >=0.21 it raises RuntimeError
    during normal CLI invocations because typer no longer pushes its
    Context onto Click's global stack.
    """
    settings = config.mirrors.get(mirror_name)

    # Overlay the CLI target onto the mirror settings if provided.
    if cli_ctx is None:
        try:
            cli_ctx = click.get_current_context()
        except RuntimeError:
            cli_ctx = None
    target = _get_active_target(cli_ctx)
    if target is not None and settings is not None:
        # Apply target's remote config to a copy of the settings and
        # store it back so that context_for() also sees it.
        from dataclasses import replace
        settings = replace(settings, remote=target)
        config.mirrors[mirror_name] = settings  # type: ignore[index]

    return _build_manager(
        config, logger, dry_run, mirror_settings=settings, cli_ctx=cli_ctx,
    )


def _build_manager(
    config: Config, logger, dry_run: bool, *, mirror_settings: Optional[MirrorSettings] = None,
    cli_ctx: Optional[click.Context] = None,
) -> MirrorManager:
    if cli_ctx is None:
        try:
            cli_ctx = click.get_current_context()
        except RuntimeError:
            cli_ctx = None

    executor = _build_executor(
        config,
        logger,
        dry_run=dry_run,
        use_sudo_for_agent=_get_use_sudo_for_agent(cli_ctx, config),
        mirror_settings=mirror_settings,
        debug_ssh=_get_debug_ssh(cli_ctx),
        local_disk_override=_get_local_disk_override(cli_ctx),
        cli_ctx=cli_ctx,
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


def _parse_optional_bool(value: Optional[str], *, option_name: str) -> Optional[bool]:
    """Parse a tri-state CLI string (``true``/``false``/unset) into ``Optional[bool]``.

    Used for options like ``--inline-prompt true`` that need to distinguish
    "unset" (auto-detect) from "explicitly true" and "explicitly false".
    Modelled as a string rather than ``Optional[bool]`` with ``is_flag=False``
    because the latter relies on a typer/click handshake that is deprecated in
    typer >=0.21 and broken across typer 0.12 + click >=8.2.
    """
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ("true", "yes", "1", "on"):
        return True
    if normalized in ("false", "no", "0", "off"):
        return False
    raise typer.BadParameter(
        f"{option_name} must be one of true/false (got `{value}`).",
    )


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
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
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
    debug_ssh: bool = typer.Option(
        False,
        "--debug-ssh",
        help="Enable verbose SSH tracing (-vvv) for all remote connections.",
    ),
    local_disk: Optional[bool] = typer.Option(
        None,
        "--local-disk/--no-local-disk",
        help="Use compute-node local disk instead of shared filesystem. "
             "Overrides the slurm.local_disk config setting.",
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
        "debug_ssh": debug_ssh,
        "local_disk": local_disk,
    }


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
    manager = _build_manager_for_mirror(config, logger, dry_run, mirror, cli_ctx=ctx)
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
    manager = _build_manager(config, logger, dry_run=dry_run, cli_ctx=ctx)
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
    manager = _build_manager_for_mirror(config, logger, dry_run, mirror, cli_ctx=ctx)
    try:
        manager.sync(manager.context_for(mirror))
    except MirrorError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command("pull")
def pull(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Mirror name defined in configuration.", shell_complete=_mirror_completion),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
    node: Optional[str] = typer.Option(
        None,
        "--node",
        help="Pull from a specific compute node (e.g. --node n0047.savio3).",
    ),
) -> None:
    """Fetch agent commits from the mirror into canonical.

    Works for both local mirrors (filesystem path on the same host)
    and remote mirrors (over SSH, possibly via a SLURM allocation).
    For remote mirrors, reconnects to the active SLURM allocation
    (if any) and uses the configured tunnel. For local mirrors, just
    fetches from the mirror's filesystem path — no SLURM/SSH needed.
    """
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)

    # Check session health before reconnecting.
    from .session import RemoteSession
    settings = config.mirrors.get(mirror)
    if settings is None:
        typer.echo(f"Unknown mirror: {mirror}", err=True)
        raise typer.Exit(code=1)

    # Apply target overlay.  ``ctx`` is the typer.Context for this
    # subcommand; it is a click.Context subclass and carries the
    # callback's obj.  Don't use ``click.get_current_context()`` here --
    # typer >=0.21 does not push its Context onto Click's global
    # stack, so that call raises RuntimeError under normal CLI usage.
    target = _get_active_target(ctx)
    if target is not None:
        from dataclasses import replace
        settings = replace(settings, remote=target)
        config.mirrors[mirror] = settings  # type: ignore[index]

    manager = _build_manager_for_mirror(config, logger, False, mirror, cli_ctx=ctx)
    mirror_ctx = manager.context_for(mirror)

    if not settings.is_remote:
        # Local mirrors: just fetch agent commits from the mirror's
        # filesystem path. No SLURM/SSH plumbing needed.
        if node:
            typer.echo(
                "--node is only meaningful for remote mirrors; ignoring.",
                err=True,
            )
        try:
            manager._pull_from_local(mirror_ctx)
        except MirrorError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        typer.echo("Pull complete.")
        return

    _obj = (ctx.obj if ctx.obj else {}) or {}
    _tgt = _obj.get("target_name")
    session = RemoteSession.load(settings.name, target_name=_tgt)

    # If --node is specified, inject it so _build_executor allocates
    # on that node (needed to SSH and pull from its local disk).
    if node:
        session.slurm_job_id = None
        session.compute_node = node
        session.save()
        logger.info("Targeting node %s for pull", node)

    if settings.remote and settings.remote.slurm and not session.slurm_job_id and not node:
        typer.echo(
            "No active SLURM allocation in the session.  "
            "Run `sucoder collaborate` first to establish one.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        manager._pull_from_remote(mirror_ctx)
    except MirrorError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("Pull complete.")


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
    manager = _build_manager(config, logger, dry_run=dry_run, cli_ctx=ctx)
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
    manager = _build_manager_for_mirror(config, logger, False, mirror, cli_ctx=ctx)
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
    manager = _build_manager_for_mirror(config, logger, False, mirror, cli_ctx=ctx)
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
    inline_prompt: Optional[str] = typer.Option(
        None,
        "--inline-prompt",
        help="Force whether context prelude text is appended to the agent command "
             "(true/false; omit for auto-detect).",
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
    manager = _build_manager_for_mirror(config, logger, dry_run, mirror, cli_ctx=ctx)
    command_override = _parse_agent_command(agent_command) or (_agent_shorthand(agent) if agent else None)
    env_override = _parse_agent_env(agent_env)
    inline_prompt_flag = _parse_optional_bool(inline_prompt, option_name="--inline-prompt")
    try:
        manager.launch_agent(
            manager.context_for(mirror),
            sync=sync,
            task_name=task,
            base_branch=base,
            extra_args=extra_args,
            command_override=command_override,
            env_override=env_override,
            supports_inline_prompt=inline_prompt_flag,
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
    inline_prompt: Optional[str] = typer.Option(
        None,
        "--inline-prompt",
        help="Force whether context prelude text is appended to the agent command "
             "(true/false; omit for auto-detect).",
    ),
    lfs: bool = typer.Option(
        False,
        "--lfs/--no-lfs",
        help="Download Git LFS objects during clone (default: skip LFS to avoid failures).",
    ),
    node: Optional[str] = typer.Option(
        None,
        "--node",
        help="Request a specific compute node (e.g. --node n0047.savio3). "
             "Useful to recover work on local disk from a previous session.",
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

    # If --node is specified, inject it into the session so that
    # _ensure_slurm_node uses --nodelist to request that node.
    if node:
        from .session import RemoteSession
        # ``ctx`` is the typer.Context for this subcommand and carries
        # the callback's obj; don't go through click.get_current_context
        # (typer >=0.21 doesn't push onto Click's global stack).
        _tgt = ((ctx.obj or {}).get("target_name") if ctx else None)
        _session = RemoteSession.load(mirror, target_name=_tgt)
        # Clear the old job so _ensure_slurm_node allocates a new one,
        # but set compute_node so it becomes the preferred_node.
        _session.slurm_job_id = None
        _session.compute_node = node
        _session.save()
        logger.info("Requesting specific node %s", node)

    manager = _build_manager_for_mirror(config, logger, dry_run, mirror, cli_ctx=ctx)
    command_override = _parse_agent_command(agent_command) or (_agent_shorthand(agent) if agent else None)
    env_override = _parse_agent_env(agent_env)
    inline_prompt_flag = _parse_optional_bool(inline_prompt, option_name="--inline-prompt")
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
            supports_inline_prompt=inline_prompt_flag,
            skip_lfs=not lfs,
        )
    except MirrorError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command("audit")
def audit(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(
        None,
        help="Mirror name (required for --target code/all).",
        shell_complete=_mirror_completion,
    ),
    scope: str = typer.Option(
        "skills",
        "--scope",
        "-s",
        help="What to audit: 'skills', 'code', or 'all'.",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Review from scratch instead of only changes since last audit.",
    ),
    approve: bool = typer.Option(
        False,
        "--approve",
        help="Advance the audited baseline to the current state after review.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
) -> None:
    """Run a compliance audit on agent-written skills and/or mirror code changes."""
    valid_scopes = ("skills", "code", "all")
    if scope not in valid_scopes:
        typer.echo(f"--scope must be one of: {', '.join(valid_scopes)}", err=True)
        raise typer.Exit(code=1)

    config = _get_config(ctx)
    logger = setup_logger("sucoder.audit", config.log_dir, verbose)

    # Resolve mirror name when code audit is requested.
    mirror_name: Optional[str] = None
    if scope in ("code", "all"):
        try:
            mirror_name = _resolve_mirror_name(ctx, mirror)
        except (typer.BadParameter, Exception) as exc:
            typer.echo(
                f"Mirror name required for code audit: {exc}\n"
                f"Usage: sucoder audit MIRROR --scope {scope}",
                err=True,
            )
            raise typer.Exit(code=1)

    # Check auditor user exists.
    auditor_user = os.environ.get("SUCODER_AUDITOR_USER", "auditor")
    try:
        import pwd as _pwd
        _pwd.getpwnam(auditor_user)
    except KeyError:
        typer.echo(
            f"Auditor user '{auditor_user}' does not exist.\n"
            f"Create it with:  make create-auditor-user\n"
            f"Or set SUCODER_AUDITOR_USER to use a different user.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Build an executor that runs commands as the auditor user.
    from .executor import CommandExecutor
    auditor_executor = CommandExecutor(
        human_user=config.human_user,
        agent_user=auditor_user,
        agent_group=auditor_user,
        logger=logger,
        use_sudo_for_agent=ctx.obj.get("use_sudo_for_agent", True),
    )

    # We need a MirrorManager for the audit methods.
    manager = MirrorManager(
        config=config,
        executor=CommandExecutor(
            human_user=config.human_user,
            agent_user=config.agent_user,
            agent_group=config.agent_group,
            logger=logger,
            use_sudo_for_agent=ctx.obj.get("use_sudo_for_agent", True),
        ),
        logger=logger,
    )

    multi = scope == "all"
    any_output = False

    # --- Skills audit ---
    if scope in ("skills", "all"):
        if multi:
            typer.echo("=== Skills Audit ===\n")
        skills_report = manager.audit_agent_skills(
            full=full,
            auditor_executor=auditor_executor,
        )
        if skills_report is None:
            typer.echo("Nothing to audit (skills).")
        else:
            typer.echo(skills_report)
            any_output = True
        if approve and skills_report is not None:
            manager.advance_audited_ref(auditor_executor)
            typer.echo("\nSkills audited baseline advanced to current state.")
        if multi:
            typer.echo("")  # blank separator

    # --- Code audit ---
    if scope in ("code", "all"):
        assert mirror_name is not None
        if multi:
            typer.echo(f"=== Code Audit (mirror: {mirror_name}) ===\n")
        code_report = manager.audit_code_changes(
            mirror_name,
            full=full,
            auditor_executor=auditor_executor,
        )
        if code_report is None:
            typer.echo("Nothing to audit (code).")
        else:
            typer.echo(code_report)
            any_output = True
        if approve and code_report is not None:
            manager.advance_audited_code_ref(mirror_name, auditor_executor)
            typer.echo("\nCode audited baseline advanced to current state.")

    if not any_output and not multi:
        raise typer.Exit(0)


@app.command("attach")
def attach(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Mirror name defined in configuration.", shell_complete=_mirror_completion),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
    node: Optional[str] = typer.Option(
        None,
        "--node",
        help="Attach to a specific compute node (e.g. --node n0047.savio3).",
    ),
    via_srun: bool = typer.Option(
        False,
        "--via-srun",
        help=(
            "Reconnect by running 'srun --jobid=<JOB> --overlap --pty' on "
            "the login node instead of SSHing directly to the compute "
            "node.  Useful when (a) direct SSH login->compute is blocked "
            "by site policy, (b) the session's compute_node was not "
            "recorded, or (c) you need to be inside the job's cgroup "
            "for diagnostics.  SLURM targets only."
        ),
    ),
) -> None:
    """Reconnect to an existing remote agent session via tmux."""
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    settings = config.mirrors.get(mirror)

    # Apply --target overlay so `-T savio attach` works the same as
    # `-T savio collaborate`.  ``ctx`` is the typer.Context for this
    # subcommand (a click.Context subclass); use it directly rather
    # than ``click.get_current_context``, which raises under typer
    # >=0.21 (it no longer pushes onto Click's global stack).
    target = _get_active_target(ctx)
    if target is not None and settings is not None:
        from dataclasses import replace
        settings = replace(settings, remote=target)

    if not settings or not settings.remote:
        typer.echo("Mirror is not configured for remote execution.", err=True)
        raise typer.Exit(code=1)

    # Refuse incoherent --via-srun usage early, before any SSH / squeue
    # round-trips.  --via-srun is only meaningful for SLURM allocations
    # (it joins the job via srun --overlap), and it ignores --node
    # because srun routes via the jobid, not a hostname — passing both
    # is almost always a mistake we should surface rather than paper
    # over.
    if via_srun and settings.remote.slurm is None:
        typer.echo(
            "--via-srun requires a SLURM target.",
            err=True,
        )
        raise typer.Exit(code=1)
    if via_srun and node:
        typer.echo(
            "--via-srun ignores --node (srun routes via the jobid). "
            "Drop one of the two.",
            err=True,
        )
        raise typer.Exit(code=1)

    from .session import RemoteSession
    from .tunnel import SshControl

    _tgt_name = ((ctx.obj or {}).get("target_name") if ctx else None)
    session = RemoteSession.load(mirror, target_name=_tgt_name)
    if not session.login_node:
        typer.echo(
            "No active session found. Run 'sucoder collaborate' or "
            "'sucoder agents-run' first to establish a session.",
            err=True,
        )
        raise typer.Exit(code=1)

    remote = settings.remote
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)
    debug_ssh = _get_debug_ssh(ctx)

    # Reuse ControlMaster if active; re-establish if expired.
    control = SshControl(
        gateway=remote.gateway,
        control_persist=remote.control_persist,
        debug=debug_ssh,
    )
    try:
        control.ensure(logger)
    except Exception:
        pass  # Best-effort; ssh will prompt directly if needed
    control_opts = control.ssh_options() if control.is_active() else []

    # For SLURM targets, attach to the compute node (via login node).
    # --node overrides the session's compute_node.  --via-srun stays
    # on the login node and steps into the allocation with `srun
    # --overlap`, so a missing compute_node is acceptable in that mode
    # (srun finds the node from the jobid).
    compute = node or session.compute_node
    srun_prefix = ""
    if remote.slurm is not None and session.slurm_job_id and (compute or via_srun):
        # Verify the SLURM allocation is still ours before routing to
        # the compute node.  Without this check we'd silently fall
        # through to a login-node shell (or worse, attach to a node
        # that now belongs to someone else's job).
        import subprocess as _sp
        check_cmd = [
            "ssh", *control_opts, "-J", remote.gateway, session.login_node,
            f"squeue --job {shlex.quote(str(session.slurm_job_id))} "
            "--noheader -o %T 2>/dev/null",
        ]
        try:
            check = _sp.run(
                check_cmd, capture_output=True, text=True, check=False,
                timeout=20,
            )
            state = check.stdout.strip()
        except _sp.TimeoutExpired:
            state = ""
        if state in ("RUNNING", "PENDING"):
            if via_srun:
                # Land on the login node; `srun --overlap` joins the
                # existing allocation and drops us on the compute node
                # *inside the job's cgroup*.  This is the recovery
                # path for orphaned sessions and for clusters that
                # block direct SSH to compute nodes.
                attach_target = session.login_node
                jump_chain = remote.gateway
                srun_prefix = (
                    f"srun --jobid={shlex.quote(str(session.slurm_job_id))} "
                    "--overlap --pty "
                )
            else:
                attach_target = compute
                jump_chain = f"{remote.gateway},{session.login_node}"
        else:
            where = compute or f"jobid {session.slurm_job_id}"
            typer.echo(
                f"SLURM job {session.slurm_job_id} on {where} is "
                f"{state or 'gone'} — the allocation has ended.\n"
                "Run `sucoder collaborate` to start a new one, or "
                "`sucoder release` to clear the stale session record.",
                err=True,
            )
            raise typer.Exit(code=1)
    elif remote.slurm is not None and not session.slurm_job_id:
        # SLURM target but no recorded job — likely a stale session
        # from before the SLURM allocation succeeded.  Refuse to put
        # the user on the login node silently.
        typer.echo(
            "No SLURM job recorded for this session — nothing to "
            "attach to.  Run `sucoder collaborate` to start one.",
            err=True,
        )
        raise typer.Exit(code=1)
    else:
        # Non-SLURM remote: attach on the login node.
        attach_target = session.login_node
        jump_chain = remote.gateway

    tmux_name = f"sucoder-{mirror}"
    # When via_srun is set, `srun_prefix` is applied to both branches
    # so the fallback `tmux new-session` also runs inside the
    # allocation, not on the login node.
    attach_cmd = (
        f"{srun_prefix}tmux attach-session -t {shlex.quote(tmux_name)} "
        f"|| {srun_prefix}tmux new-session -s {shlex.quote(tmux_name)}"
    )
    os.execvp("ssh", [
        "ssh", "-t",
        *control_opts,
        "-J", jump_chain,
        attach_target,
        attach_cmd,
    ])


@app.command("release")
def release(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Mirror name defined in configuration.", shell_complete=_mirror_completion),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip the confirmation prompt.",
    ),
) -> None:
    """Cancel the SLURM allocation for this mirror and clear session state.

    Use after ``sucoder collaborate`` on a SLURM-backed target when
    you're done with the compute node.  Auto-cancel was deliberately
    removed from the agent wrapper and the backstop timer (so a
    transient agent failure can't tear down your allocation); this
    command is the explicit way to free the resources.
    """
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    settings = config.mirrors.get(mirror)

    # ``ctx`` is the typer.Context for this subcommand and carries the
    # callback's obj; don't go through ``click.get_current_context``
    # (typer >=0.21 doesn't push onto Click's global stack).
    target = _get_active_target(ctx)
    if target is not None and settings is not None:
        from dataclasses import replace
        settings = replace(settings, remote=target)

    if not settings or not settings.remote:
        typer.echo("Mirror is not configured for remote execution.", err=True)
        raise typer.Exit(code=1)
    if settings.remote.slurm is None:
        typer.echo(
            f"Target for mirror {mirror} has no SLURM config; "
            "nothing to release.",
            err=True,
        )
        raise typer.Exit(code=1)

    from .session import RemoteSession
    from .tunnel import SshControl

    _tgt_name = ((ctx.obj or {}).get("target_name") if ctx else None)
    session = RemoteSession.load(mirror, target_name=_tgt_name)
    if not session.slurm_job_id:
        typer.echo(
            f"No SLURM allocation recorded for mirror {mirror}"
            f"{f' (target {_tgt_name})' if _tgt_name else ''}. "
            "Nothing to release.",
        )
        raise typer.Exit(code=0)

    job_id = session.slurm_job_id
    compute_node = session.compute_node or "<unknown>"
    if not force:
        if not _prompt_yes_no(
            f"Cancel SLURM job {job_id} on {compute_node} for mirror {mirror}? [y/N] ",
        ):
            typer.echo("Aborted.")
            raise typer.Exit(code=0)

    remote = settings.remote
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)
    debug_ssh = _get_debug_ssh(ctx)

    # Reuse / re-establish the gateway and login-node ControlMasters
    # so scancel reaches the cluster.
    gw_control = SshControl(
        gateway=remote.gateway,
        control_persist=remote.control_persist,
        debug=debug_ssh,
    )
    try:
        _ensure_ssh_visible(gw_control, remote.gateway, logger)
    except Exception as exc:  # noqa: BLE001  -- want broad fallback here
        typer.echo(f"Failed to reach gateway: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not session.login_node:
        typer.echo(
            "No login node recorded in session; cannot reach SLURM. "
            f"Run `scancel {job_id}` manually from a login node.",
            err=True,
        )
        raise typer.Exit(code=1)

    ln_control = SshControl(
        gateway=session.login_node,
        control_persist=remote.control_persist,
        jump_host=remote.gateway,
        jump_control=gw_control,
        debug=debug_ssh,
    )
    try:
        _ensure_ssh_visible(ln_control, session.login_node, logger)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Failed to reach login node: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    import subprocess as _sp
    scancel_cmd = [
        "ssh", *ln_control.ssh_options(), session.login_node,
        f"scancel {shlex.quote(str(job_id))}",
    ]
    logger.debug("scancel command: %s", scancel_cmd)
    result = _sp.run(scancel_cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        # scancel exits 0 even for nonexistent jobs (just prints a
        # warning to stderr), so a real failure usually means an SSH
        # or auth problem.
        typer.echo(
            f"scancel returned {result.returncode}: {result.stderr.strip()}",
            err=True,
        )
        raise typer.Exit(code=1)

    # Clear the SLURM-specific session fields so future commands don't
    # think a stale allocation is still ours.  Keep login_node so
    # subsequent attaches can still resolve the gateway path.
    session.slurm_job_id = None
    session.compute_node = None
    # The mirror root may have been on the now-released local disk;
    # forget it so the next collaborate picks a fresh root.
    if session.remote_mirror_root and session.remote_mirror_root.startswith("/local"):
        session.remote_mirror_root = None
    session.save()

    typer.echo(f"Released SLURM job {job_id} on {compute_node}.")
    if result.stderr.strip():
        typer.echo(result.stderr.strip(), err=True)


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
        cli_ctx=ctx,
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


@app.command("mcp-suggest")
def mcp_suggest(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(
        None, help="Mirror name (auto-detected when omitted).",
        shell_complete=_mirror_completion,
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
    apply: bool = typer.Option(False, "--apply", help="Save accepted servers to mirror prefs."),
) -> None:
    """Scan mirror for tech stack indicators and suggest MCP servers."""
    import json as _json

    from .config import McpServerConfig
    from .mcp_discovery import detect_suggestions
    from .workspace_prefs import WorkspacePrefs

    mirror_name = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    logger = setup_logger(f"sucoder.{mirror_name}", config.log_dir, verbose)
    manager = _build_manager_for_mirror(config, logger, False, mirror_name, cli_ctx=ctx)
    mirror_ctx = manager.context_for(mirror_name)
    mirror_path = mirror_ctx.mirror_path

    if not mirror_path.exists():
        typer.secho(f"Mirror path does not exist: {mirror_path}", fg="red")
        raise typer.Exit(code=1)

    existing: Dict[str, McpServerConfig] = dict(mirror_ctx.settings.mcp_servers)
    repo_mcp = mirror_path / ".mcp.json"
    if repo_mcp.exists():
        try:
            repo_data = _json.loads(repo_mcp.read_text(encoding="utf-8"))
            for name in repo_data.get("mcpServers", {}):
                existing[name] = McpServerConfig(command="")
        except (_json.JSONDecodeError, OSError):
            pass

    suggestions = detect_suggestions(mirror_path, existing)

    if not suggestions:
        typer.echo("No additional MCP servers suggested for this repo.")
        raise typer.Exit()

    typer.echo(f"Suggested MCP servers for {mirror_name}:\n")
    for s in suggestions:
        env_note = f"  (requires: {', '.join(s.required_env)})" if s.required_env else ""
        if s.required_env and not all(os.environ.get(v) for v in s.required_env):
            status = typer.style("missing env", fg="yellow")
        else:
            status = typer.style("ready", fg="green")
        typer.echo(f"  {s.name}: {s.description}{env_note} [{status}]")

    if apply:
        prefs = WorkspacePrefs.load(mirror_path)
        decisions = {s.name: True for s in suggestions}
        prefs.set_mcp_discovery(decisions)
        prefs.save()
        typer.echo(f"\nSaved {len(suggestions)} server(s) to mirror prefs.")
