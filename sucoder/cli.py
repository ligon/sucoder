"""Typer-powered CLI entry point."""

from __future__ import annotations

import getpass
import os
import re
import shlex
import pwd
import subprocess
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
from .executor import CommandError, CommandExecutor
from .logging_utils import setup_logger
from .mirror import (
    MirrorError,
    MirrorManager,
    # Re-exported for backward-compatible imports (e.g. tests) and reuse;
    # the implementations live in mirror.py so the confined launch path can
    # build them without importing cli (which would be a circular import).
    _build_sbatch_command,
    confined_attach_command,
    confined_tmux_target,
)
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


def _resolve_cert_username(
    username: Optional[str], config: Optional[Config]
) -> str:
    """Resolve the BRC username for an auto-mint, never raising.

    Precedence: an explicit *username* (the gateway control's SSH
    ``User``, i.e. a target's ``remote_user``) -> ``$BRC_USER`` -> the
    configured ``human_user`` -> the local OS user.  *config* may be
    ``None`` at call sites that cannot thread it -- typer >=0.21 no longer
    exposes the Click context via ``click.get_current_context`` during a
    command (verified: it raises ``RuntimeError``), so there is no global
    fallback -- in which case we drop to ``getpass.getuser()`` rather than
    crash.  The gateway hop is interactive-only, so the local user is a
    sane last resort.
    """
    if username:
        return username
    env_user = os.environ.get("BRC_USER")
    if env_user:
        return env_user
    if config is not None and getattr(config, "human_user", None):
        return config.human_user
    return getpass.getuser()


def _maybe_offer_cert_mint(
    control,
    logger,
    *,
    username: Optional[str] = None,
    config: Optional[Config] = None,
) -> None:
    """Offer to mint a fresh gateway cert before a cold connect, if it's stale.

    A missing/expired ``cert_file`` otherwise degrades to a per-connection
    PIN+OTP prompt from ssh; minting once buys a ~12h OTP-free window instead.
    Strictly opt-in and interactive -- it fires only on the *gateway* hop (the
    only one that presents the cert, tunnel.py:348), when the mux is cold, at a
    real TTY, and when the cert actually reads stale -- so agents, cron, and
    warm reuse are untouched.  Any failure is non-fatal: we fall through to
    ssh's own prompt.  ``getattr`` guards keep it inert for the fake controls
    used in tests.  *config* is threaded from the calling command so the
    default mint username can be the configured ``human_user`` (see
    :func:`_resolve_cert_username`); it is optional and degrades gracefully
    when a call site cannot supply it.
    """
    cert_file = getattr(control, "cert_file", None)
    if getattr(control, "jump_host", None) is not None or not cert_file:
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return
    try:
        if control.is_active():
            return
    except Exception:  # noqa: BLE001 -- a probe failure shouldn't block connect
        pass
    glyph, msg = _cert_status(str(cert_file))
    if glyph == "✓":  # valid -- nothing to do
        return
    if not typer.confirm(f"Gateway cert: {msg}. Mint a fresh one now?", default=True):
        return
    from . import cert as cert_mod

    username = _resolve_cert_username(username, config)
    pin = typer.prompt("BRC PIN", hide_input=True)
    otp = typer.prompt("BRC OTP")
    try:
        cert_mod.mint(
            cert_file, cert_mod.DEFAULT_CA_URL, username, pin, otp,
            cert_mod.DEFAULT_LIFETIME,
        )
    except cert_mod.CertError as exc:
        typer.echo(
            f"Cert minting failed ({exc}); continuing -- ssh will prompt.", err=True
        )
        return
    g2, m2 = _cert_status(str(cert_file))
    typer.echo(f"  {g2} {m2}")


def _connect_with_retry(
    control,
    label: str,
    logger,
    *,
    max_wait: int = 60,
    initial_delay: int = 3,
    sleep=time.sleep,
    config: Optional[Config] = None,
) -> None:
    """Bring up an SSH ControlMaster, retrying *transient* SSH failures.

    Some SSH closures clear on their own within seconds:

    * a just-allocated SLURM compute node refuses SSH while ``sshd`` /
      ``pam_slurm_adopt`` register the job, and
    * a busy HPC login node sheds connections during the protocol-banner
      exchange (``MaxStartups`` / fail2ban),

    both surfacing as ``kex_exchange_identification: Connection closed by
    remote host``.  Retry with capped exponential backoff for up to
    *max_wait* seconds -- but *only* for failures that look transient
    (see :func:`sucoder.tunnel.is_transient_ssh_error`).  A genuine auth
    or host error is re-raised immediately rather than making the user
    wait out the full backoff.  ``sleep`` is injectable for tests.
    """
    from .tunnel import TunnelError, is_transient_ssh_error

    # Before a cold gateway connect, offer to mint a fresh cert if the
    # configured one is stale -- so one OTP buys a 12h window instead of an
    # OTP per connection (no-op unless interactive + gateway hop + stale cert).
    _maybe_offer_cert_mint(
        control, logger, username=getattr(control, "user", None), config=config,
    )

    waited = 0
    delay = initial_delay
    attempt = 0
    while True:
        attempt += 1
        try:
            _ensure_ssh_visible(control, label, logger)
            return
        except TunnelError as exc:
            # Classify against both the message and the captured ssh
            # stderr: real establish() failures carry the reason on
            # ``exc.stderr``; the message is a generic wrapper.
            reason = f"{exc}\n{getattr(exc, 'stderr', '') or ''}"
            if not is_transient_ssh_error(reason) or waited >= max_wait:
                raise
            logger.info(
                "%s not reachable yet (attempt %d): transient SSH closure "
                "-- retrying in %ds",
                label, attempt, delay,
            )
            sleep(delay)
            waited += delay
            delay = min(delay * 2, 15)


def _ssh_debug_hint(debug_ssh: bool) -> str:
    """A suffix nudging toward ``--debug-ssh``, unless it is already on.

    Appended to user-facing SSH failure messages so a transient
    ``kex_exchange_identification`` closure (which prints no remote
    reason) has an obvious next diagnostic step.
    """
    if debug_ssh:
        return ""
    return "  (re-run with --debug-ssh for a full SSH trace)"


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


def _reconcile_login_node(remote, session, target_name, logger) -> bool:
    """Adopt the warm tunnel session's login node into a SLURM mirror session.

    sucoder keeps two independent login-node pins: the *mirror* session
    (dialed by collaborate/attach/release/renew) and the *tunnel* session
    (kept warm by ``tunnel up``, and what the ssh_config ``-ln`` block points
    at).  They are set by separate round-robin ``ssh <gateway> hostname``
    calls and can drift, leaving a mirror command dialing a stale, wedged
    node while a live tunnel to a different node sits warm and neglected.

    For a SLURM-backed session the login node is only a routing hop -- the
    allocation lives on ``compute_node`` + job id, so any healthy login node
    works -- so prefer the node the tunnel infra keeps warm.  No-op for a
    non-SLURM session, where the agent tmux lives ON the login node and the
    pin is not a swappable hop.  Returns True if the pin changed.
    """
    if remote.slurm is None or not target_name:
        return False
    from .session import RemoteSession
    warm = RemoteSession.load(_tunnel_session_name(target_name)).login_node
    if not warm or warm == session.login_node:
        return False
    old = session.login_node
    session.login_node = warm
    session.save()
    logger.info(
        "Using warm tunnel login node %s (mirror session had %s) for %s.",
        warm, old or "no pin", session.mirror_name,
    )
    return True


def _login_node_via_gateway(gw_control, gateway, *, debug_ssh: bool = False) -> str:
    """Ask the warm gateway ControlMaster which login node it landed on.

    The gateway mux is, by construction, connected to a *reachable* login
    node (it authenticated), so ``hostname`` over it names a node we can
    actually reach right now -- unlike a possibly-stale saved pin.  Returns
    the empty string if the probe fails.
    """
    cmd = ["ssh", *gw_control.ssh_options(), "-o", "BatchMode=yes", gateway, "hostname"]
    if debug_ssh:
        cmd.insert(1, "-v")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


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

        # 0. Reconcile the mirror pin with the warm tunnel node so we ride a
        #    live login node instead of a stale one that drifted apart from
        #    the tunnel infra (no-op for non-SLURM targets).
        _reconcile_login_node(remote, session, _target_name, logger)

        # 1. Establish ControlMaster to the gateway (authenticates
        #    once; may prompt for pin + OTP).
        gw_control = SshControl(
            gateway=remote.gateway,
            **remote.ssh_control_kwargs(),
            debug=debug_ssh,
        )
        try:
            # Retry transient banner-phase closures (a busy login/gateway
            # can shed connections under MaxStartups); a real auth failure
            # still surfaces immediately.
            _connect_with_retry(gw_control, remote.gateway, logger, config=config)
        except TunnelError as exc:
            typer.echo(str(exc) + _ssh_debug_hint(debug_ssh), err=True)
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
            **remote.ssh_control_kwargs(),
            jump_host=remote.gateway,
            jump_control=gw_control,
            debug=debug_ssh,
        )
        try:
            # Retry transient closures: a hammered login node frequently
            # drops us during the banner exchange
            # (``kex_exchange_identification``).  This is the failure that
            # used to abort the whole launch on the first attempt.
            _connect_with_retry(ln_control, session.login_node, logger)
        except TunnelError as exc:
            # The pinned login node is wedged (not merely transient -- retries
            # are exhausted).  The warm gateway mux is on a *healthy* login
            # node (it authenticated above), so re-pin to whatever it landed
            # on and retry once, rather than dying on a dead pin.  Guarded to
            # SLURM targets, where the login node is just a routing hop.
            fresh = (
                _login_node_via_gateway(gw_control, remote.gateway, debug_ssh=debug_ssh)
                if remote.slurm is not None else ""
            )
            if not fresh or fresh == session.login_node:
                typer.echo(str(exc) + _ssh_debug_hint(debug_ssh), err=True)
                raise typer.Exit(code=1) from exc
            logger.info(
                "Login node %s unreachable; re-pinning to %s (off the warm "
                "gateway mux) and retrying.", session.login_node, fresh,
            )
            session.login_node = fresh
            session.save()
            ln_control = SshControl(
                gateway=session.login_node,
                **remote.ssh_control_kwargs(),
                jump_host=remote.gateway,
                jump_control=gw_control,
                debug=debug_ssh,
            )
            try:
                _connect_with_retry(ln_control, session.login_node, logger)
            except TunnelError as exc2:
                typer.echo(str(exc2) + _ssh_debug_hint(debug_ssh), err=True)
                raise typer.Exit(code=1) from exc2

        # 3b. Establish ControlMaster to the data transfer node (DTN).
        #     The DTN has fat pipes and spare CPU compared with the
        #     hammered login nodes, so filesystem scaffolding and git
        #     transport route through it.
        dtn_control = SshControl(
            gateway=remote.transfer_host,
            **remote.ssh_control_kwargs(),
            jump_host=remote.gateway,
            jump_control=gw_control,
            debug=debug_ssh,
        )
        try:
            # DTN is optional and already falls back to the login node, so
            # ride out only a *brief* transient blip rather than the full
            # 60s window before falling back.
            _connect_with_retry(
                dtn_control, remote.transfer_host, logger, max_wait=6,
            )
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
        #
        #    Exception: a ``confined`` target fuses allocate+launch into a
        #    single ``sbatch`` submitted later (its batch body runs in the
        #    job cgroup).  No compute node exists at build time, so we skip
        #    salloc and return a *login-node* executor; the confined
        #    collaborate flow submits the job and resolves the node itself.
        confined = remote.slurm is not None and remote.slurm.confined
        target_node = session.login_node
        target_control = ln_control
        prev_compute_node = session.compute_node
        if remote.slurm is not None and not confined:
            target_node, target_control = _ensure_slurm_node(
                remote, session, ln_control, gw_control, logger,
                debug_ssh=debug_ssh,
            )

        # Resolve local-disk setting: CLI flag overrides config.
        # Confined targets stage the prelude + batch script to NFS and have
        # no compute-node-local disk at build time, so local disk never
        # applies to them (the saved compute-node root, if any, is also
        # unreachable from the login node).
        use_local_disk = False
        local_disk_root = ""
        cfg_local_disk = remote.slurm.local_disk if (remote.slurm and not confined) else None
        if remote.slurm is not None and not confined:
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
        if confined:
            # NFS only: the batch script and prelude are staged to the
            # shared FS; there is no compute-node local disk at build time.
            remote_mirror_root = str(remote.mirror_root)
        elif use_local_disk:
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
            is_compute_node=(remote.slurm is not None and not confined),
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
        # for the SSH ProxyCommand fallback to the login node.  A confined
        # target's executor already targets the login node, so there is no
        # compute-node socket to fall back from.
        if remote.slurm is not None and not confined:
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


def _resolve_recorded_node(
    job_id, ln_control, login_node, logger,
    *, attempts: int = 5, delay: int = 3, sleep=time.sleep,
):
    """Resolve the node of an already-recorded SLURM *job_id*.

    Returns ``(state, node)``:

    * ``("RUNNING", "n0123.savio3")`` -- the job holds a node; adopt it.
    * ``(state, None)`` -- the job is alive but not yet placed
      (PENDING/CONFIGURING), or the probe failed.  Caller must NOT
      reallocate: the id is live and reallocating leaks it.
    * ``(None, None)`` -- the job has LEFT the queue (terminal/invalid id).
      Safe to clear the id and allocate fresh.

    This is the read-side counterpart to the persist-before-query write in
    :func:`_ensure_slurm_node`: ``salloc`` bills from the moment it grants,
    so the job id is written to the session *before* the node is queried.
    An interrupt or a squeue failure in that window leaves ``slurm_job_id``
    set with ``compute_node`` still None -- a state the allocation state
    machine could not previously act on (it fell through reuse, adopt and
    salloc alike, straight into ``SshControl(gateway=None)``).

    Distinguishing "not placed yet" from "gone" is the whole job here, and
    both mistakes are expensive: calling a live job dead leaks a 24h
    allocation, calling a dead job live wedges the session forever.  So an
    empty ``%N`` with a state word means *wait*, never *gone* -- and the
    state word is never itself persisted as a node name.  (Same three-way
    contract as ``MirrorManager._poll_confined_node``, which does this over
    the executor for confined sbatch jobs; here we only have an ssh hop.)
    """
    import subprocess as _sp

    if not login_node:
        return None, None
    cmd = [
        "ssh", *ln_control.ssh_options(with_fallback=True), login_node,
        f"squeue --job {int(job_id)} --noheader -o '%T %N'",
    ]
    last_state = None
    for attempt in range(1, max(1, attempts) + 1):
        result = _sp.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            if "invalid job id" in stderr.lower():
                return None, None  # squeue is certain: the job is gone.
            # An ssh/squeue failure is NOT evidence the job is dead.  Say
            # "unknown, still alive" so the caller refuses to reallocate.
            logger.debug(
                "node-query for job %s failed (%s); not assuming it is gone",
                job_id, stderr,
            )
            return "UNKNOWN", None
        parts = result.stdout.split()
        if not parts:
            # Empty output from a successful squeue == the job left the
            # queue.  Terminal, and the only signal we accept as "gone".
            return None, None
        last_state = parts[0]
        if len(parts) >= 2 and parts[1]:
            return last_state, parts[1]
        # State word but no node: PENDING / CONFIGURING.  Brief wait --
        # salloc has already granted, so placement lands in seconds.
        if attempt < attempts:
            logger.debug(
                "job %s is %s with no node yet; re-querying in %ds",
                job_id, last_state, delay,
            )
            sleep(delay)
    return last_state, None


def _adopt_existing_allocation(node, ln_control, login_node, logger):
    """Find a live SLURM job the current user already holds on *node*.

    Returns ``(job_id, node_name)`` for the first RUNNING job the user
    owns on *node*, or ``None`` when there is none (or the probe fails).

    This lets a second ``collaborate <mirror> --node <held-node>`` attach
    to a node the user already reserved.  A whole-node allocation is
    exclusive, so a fresh ``salloc --nodelist`` on it would just time out
    and fall back to a different node.  Adopting the existing job id means
    we SSH straight into the node we already own and share it.
    """
    import subprocess as _sp

    if not login_node:
        return None
    probe = (
        f"squeue --me --nodelist={shlex.quote(node)} "
        "--states=RUNNING --noheader -o '%i %N'"
    )
    cmd = [
        "ssh", *ln_control.ssh_options(with_fallback=True), login_node, probe,
    ]
    result = _sp.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.debug(
            "adopt probe for %s failed (%s); will allocate instead",
            node, result.stderr.strip(),
        )
        return None
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        token = parts[0]
        # salloc --no-shell jobs have plain-integer ids; skip array
        # elements like "12345_1" which aren't a whole-node reservation
        # we'd want to share this way.
        if token.isdigit():
            return int(token), parts[1]
    return None


# ``_build_sbatch_command`` moved to mirror.py (and is re-exported via the
# ``from .mirror import ...`` block at the top of this module) so the confined
# launch path in mirror.py can build the sbatch argv without importing cli.


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

    # A job id recorded WITHOUT a node is the persist-before-query window
    # below (salloc grants -> id saved -> squeue resolves the node): an
    # interrupt or a failed node-query between those two saves leaves
    # exactly this state on disk, on purpose, so the allocation stays
    # recoverable instead of leaking.  Nothing ever taught the read side to
    # act on it, so all three gates below skipped it (reuse wants a node;
    # adopt and salloc both want NO job id) and the fall-through handed
    # ``SshControl(gateway=None)`` a None host -- a TypeError from inside
    # Popen, on every subsequent run.  Sticky, because nothing cleared it.
    #
    # Resolve it before the gates: adopt the node if the job still holds
    # one, clear the id only when squeue positively says the job is gone,
    # and refuse to do either while its fate is unknown -- reallocating on
    # a live id leaks a 24h job that `release` can no longer find.
    if session.slurm_job_id and not session.compute_node:
        state, node = _resolve_recorded_node(
            session.slurm_job_id, ln_control, session.login_node, logger,
        )
        if node:
            session.compute_node = node
            session.save()
            typer.echo(
                f"Recovered compute node {node} for recorded SLURM job "
                f"{session.slurm_job_id}."
            )
            logger.info(
                "Recovered node %s for job %s (session had no node)",
                node, session.slurm_job_id,
            )
        elif state is None:
            logger.info(
                "Recorded SLURM job %s has left the queue; allocating fresh",
                session.slurm_job_id,
            )
            session.slurm_job_id = None
            session.save()
        else:
            # Alive but unplaced (PENDING/CONFIGURING) or unreachable.
            # Bail rather than guess; the id stays on disk, so the operator
            # can retry or `release` it.
            typer.echo(
                f"SLURM job {session.slurm_job_id} is recorded for this "
                f"mirror but holds no node yet (state: {state}).  It is "
                "still queued -- retry shortly, or run `sucoder release` to "
                "give it up.",
                err=True,
            )
            raise typer.Exit(code=1)

    # Check whether a previous allocation is still running.
    if session.slurm_job_id and session.compute_node:
        check_cmd = [
            "ssh", *ln_control.ssh_options(with_fallback=True), session.login_node,
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

    # Resolve the preferred node: either carried over from a dead job
    # (local-disk affinity, set above) or explicitly requested via
    # --node (collaborate stores it in session.compute_node with no job
    # id).  Then, before reserving anything, see whether we already hold
    # a live allocation on that node and adopt it instead.
    if not session.slurm_job_id:
        preferred_node = locals().get("preferred_node") or session.compute_node
        if preferred_node:
            adopted = _adopt_existing_allocation(
                preferred_node, ln_control, session.login_node, logger,
            )
            if adopted is not None:
                job_id, node_name = adopted
                session.slurm_job_id = job_id
                session.compute_node = node_name
                session.save()
                typer.echo(
                    f"Adopting existing SLURM job {job_id} on {node_name} "
                    "(sharing the reserved node)."
                )
                logger.info(
                    "Adopted SLURM job %d on %s for mirror %s",
                    job_id, node_name, session.mirror_name,
                )

    # Allocate a new compute node if we still don't have one.
    if not session.slurm_job_id:
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
            "ssh", *ln_control.ssh_options(with_fallback=True), session.login_node,
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
                        "ssh", *ln_control.ssh_options(with_fallback=True),
                        session.login_node,
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

        # Persist the job id NOW, before the node-query below.  salloc has
        # already granted the allocation (it bills from this point), so if
        # anything downstream fails -- the squeue node-query, the SSH to
        # the compute node, agent launch -- the job id must be on disk so
        # `sucoder release` / `scancel` can reclaim it.  Recording it only
        # after the node-query (the historical behaviour) leaked the
        # allocation on any failure in between: a granted-but-unrecorded
        # 24h job that nothing could find.
        session.slurm_job_id = job_id
        session.save()
        logger.info("Recorded SLURM job %d (node pending)", job_id)

        # Query squeue for the node name.
        squeue_cmd = [
            "ssh", *ln_control.ssh_options(with_fallback=True), session.login_node,
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

        # job id already persisted above; now record the resolved node.
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
        **remote.ssh_control_kwargs(),
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
        # Retry: a node can refuse SSH for a few seconds right after
        # allocation (sshd / pam_slurm_adopt still registering the job).
        _connect_with_retry(
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


# Bash helper embedded verbatim into the on-node deadline timer (see
# ``_start_slurm_timer``).  Converts SLURM ``squeue -o %L`` time-left
# (TIME_LEFT) into whole minutes remaining.  ``%L`` renders as
# ``D-HH:MM:SS`` once a day or more remains, ``HH:MM:SS`` under a day,
# and ``MM:SS`` under an hour; a job with no time limit prints
# ``UNLIMITED``.  Splitting on ``:`` alone mis-handles the ``D-HH``
# field (bash reads ``D-HH`` as the arithmetic ``D - HH``), so the day
# component is split off on ``-`` first.  Leading zeros are forced to
# base 10 to avoid octal errors (``08``/``09``).  Non-numeric values
# (``UNLIMITED``/``INVALID``/empty) return a large sentinel so no
# deadline warning ever fires.  Kept as a module constant (not inlined
# in the f-string) so it is unit-testable under bash and free of
# brace-escaping noise.
_SLURM_TIME_LEFT_TO_MINS_SH = r'''
left_to_mins() {
    local s="$1" days=0 rest a b c
    if [ -z "$s" ]; then echo 999999; return; fi
    case "$s" in
        *-*) days="${s%%-*}"; rest="${s#*-}" ;;
        *)   rest="$s" ;;
    esac
    IFS=: read -r a b c <<< "$rest"
    if [ -z "$c" ]; then b="$a"; a=0; fi
    case "${days}${a}${b}" in
        *[!0-9]*) echo 999999; return ;;
    esac
    echo $(( 10#${days:-0}*1440 + 10#${a:-0}*60 + 10#${b:-0} ))
}
'''.strip("\n")


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

    If the sucoder tmux session disappears (crash, network loss, etc.),
    the timer records a warning and exits but deliberately does NOT
    ``scancel`` the job -- the user owns the SLURM lifecycle (see
    ``sucoder release``), and auto-cancelling on a transient agent
    hiccup would destroy any chance of reattaching.

    Time-left parsing is delegated to the ``left_to_mins`` bash helper
    (module constant ``_SLURM_TIME_LEFT_TO_MINS_SH``) so the
    ``D-HH:MM:SS`` day format is handled correctly and is unit-testable.
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

        # Make each deadline warning linger on the status line so a
        # full-screen agent TUI doesn't redraw over it before the human
        # notices (scoped to our session via -t, not the global -g).
        tmux set-option -t {q_tmux} display-time 15000 2>/dev/null || true

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

            mins=$(left_to_mins "$left")
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

    # Inject the time-left parser (kept as a module constant so it can
    # be unit-tested under bash) ahead of the monitoring loop.  Done
    # post-dedent so the helper's column-0 body doesn't flatten the
    # common-indent prefix and push the ``#!`` off byte 0.
    timer_script = timer_script.replace(
        'rm -f "$WARN5" "$WARN15" "$WARN30" "$WARN_FILE"\n',
        'rm -f "$WARN5" "$WARN15" "$WARN30" "$WARN_FILE"\n\n'
        + _SLURM_TIME_LEFT_TO_MINS_SH + "\n",
        1,
    )

    # Write the script to the compute node via stdin, then run it.
    # The script lives in the user's runtime cache rather than /tmp for
    # the same symlink-race reasons; the path is computed remotely
    # (rather than passed in argv) so a stale local copy can't trip up
    # the SSH command-line.
    ssh_opts = cn_control.ssh_options(with_fallback=True)
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
    # Derive target_name with the SAME expression _build_executor uses (the
    # bare obj lookup, NOT the gateway-split fallback), so the confined launch
    # persists its job id to the same session file attach/release/renew read.
    target_name = (cli_ctx.obj or {}).get("target_name") if cli_ctx else None
    return MirrorManager(
        config, executor, logger,
        prompt_handler=_prompt_yes_no, target_name=target_name,
    )


def _create_ephemeral_mirror(config: Config, git_toplevel: Path) -> str:
    """Build a :class:`MirrorSettings` from a git root, inject it into the
    in-memory config, and return its name.

    Mirrors :func:`build_default_config`'s derivation so that an
    unconfigured-but-cwd-resident repo can be operated on transparently.
    """
    mirror_name = git_toplevel.name
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


def _resolve_mirror_name(ctx: typer.Context, mirror: Optional[str]) -> str:
    """Return the mirror name, creating an ephemeral entry when needed.

    When *mirror* is given explicitly but is not configured, and we're
    inside a git repo whose root name matches it, synthesise an ephemeral
    mirror just like the no-arg path does.  Without this, ``attach Foo`` /
    ``release Foo`` rejected the very mirror that a no-arg ``collaborate``
    had auto-created from the same directory ("Mirror is not configured
    for remote execution").

    When *mirror* is ``None``:

    1. If the config contains exactly one mirror, use it.
    2. Match the cwd's git root against each mirror's ``canonical_repo``.
    3. Otherwise create an ephemeral :class:`MirrorSettings` from the git
       root and inject it into the in-memory config.
    4. If we're not inside a git repo at all, raise "Multiple mirrors
       configured".
    """
    config = _get_config(ctx)

    if mirror is not None:
        if mirror in config.mirrors:
            return mirror
        # Explicit name that isn't configured: accept it only if it names
        # the git repo we're standing in (don't fabricate a mismatched
        # canonical_repo for an arbitrary name — let downstream report
        # "not configured" in that case).
        try:
            git_toplevel = _detect_git_toplevel()
        except ConfigError:
            return mirror
        if git_toplevel.name == mirror:
            return _create_ephemeral_mirror(config, git_toplevel)
        return mirror

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
    if git_toplevel.name not in config.mirrors:
        return _create_ephemeral_mirror(config, git_toplevel)

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
             "Useful to recover work on local disk from a previous session. "
             "If you already hold a live allocation on that node, this "
             "session adopts (shares) it instead of reserving a new one.",
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
    except CommandError as exc:
        # Belt and braces: a subprocess failure that escaped bootstrap
        # without being converted to a MirrorError should still print as
        # a clean error (plus the tail of the failing command's stderr),
        # not a Python traceback.
        typer.echo(str(exc), err=True)
        detail = (exc.result.stderr or "").strip()
        if detail:
            typer.echo("\n".join(detail.splitlines()[-5:]), err=True)
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

    # Ride the warm tunnel node rather than a stale mirror pin (SLURM only).
    _reconcile_login_node(remote, session, _tgt_name, logger)

    # Reuse ControlMaster if active; re-establish if expired.
    control = SshControl(
        gateway=remote.gateway,
        **remote.ssh_control_kwargs(),
        debug=debug_ssh,
    )
    try:
        # Retry transient banner-phase closures; a real failure is
        # swallowed (best-effort) and ssh will prompt directly if needed.
        _connect_with_retry(control, remote.gateway, logger, config=config)
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
    confined = remote.slurm is not None and remote.slurm.confined
    if confined:
        # A confined agent runs INSIDE its job cgroup; a direct SSH to the
        # compute node would land OUTSIDE it (the very escape confinement
        # exists to prevent).  So always join via `srun --overlap`, even
        # though compute_node is recorded.
        if node:
            # `--node` is meaningless once we route by jobid; the early
            # incoherence guard didn't fire (the user didn't pass
            # --via-srun), so say so rather than silently dropping it.
            typer.echo(
                "note: --node is ignored for a confined target "
                "(attach joins the job by id via srun --overlap).",
                err=True,
            )
        via_srun = True
    if remote.slurm is not None:
        # SLURM target: NEVER silently fall through to a login-node shell.
        # A bare login-node tmux masquerades as the agent session and
        # hides the fact that there's nothing to attach to (observed in
        # the field: `attach` spawned an empty shell on the login node
        # while the real allocation was elsewhere / gone).
        if not session.slurm_job_id:
            # No recorded job — likely a stale session from before the
            # allocation succeeded (or a crash that didn't persist it).
            typer.echo(
                "No SLURM job recorded for this session — nothing to "
                "attach to.  Run `sucoder collaborate` to start one.",
                err=True,
            )
            raise typer.Exit(code=1)
        if not (compute or via_srun):
            # We have a job id but no idea which node it landed on, and
            # the caller didn't ask to join via srun.  Refuse rather than
            # dropping onto the login node.
            typer.echo(
                f"SLURM job {session.slurm_job_id} is recorded but its "
                "compute node is unknown.  Re-run with `--via-srun` to "
                "join the allocation by jobid, or pass `--node <node>`.",
                err=True,
            )
            raise typer.Exit(code=1)

        # Verify the SLURM allocation is still ours before routing to
        # the compute node, so we don't attach to a node that now
        # belongs to someone else's job.
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
    else:
        # Genuine non-SLURM remote: attach on the login node.
        attach_target = session.login_node
        jump_chain = remote.gateway

    if confined:
        # Confined attach: dedicated `-L` socket + sanitized session name,
        # and NO `|| tmux new-session` fallback (which would spawn an
        # unconfined orphan on the login node).  `confined_attach_command`
        # already carries its own `srun --jobid --overlap --pty` prefix, so
        # `srun_prefix` is unused here.
        session_name, socket = confined_tmux_target(mirror)
        attach_cmd = shlex.join(
            confined_attach_command(session.slurm_job_id, session_name, socket)
        )
    else:
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
    # Other sessions co-resident on this node share the same job id.
    # Releasing must not scancel the allocation out from under them.
    siblings = RemoteSession.holders_of_job(
        job_id, exclude_key=session._session_key,
    )
    if not force:
        if siblings:
            prompt = (
                f"SLURM job {job_id} on {compute_node} is shared with "
                f"{', '.join(siblings)}. Detach mirror {mirror} (kill its "
                "agent) but keep the job alive for the others? [y/N] "
            )
        else:
            prompt = (
                f"Cancel SLURM job {job_id} on {compute_node} for mirror "
                f"{mirror}? [y/N] "
            )
        if not _prompt_yes_no(prompt):
            typer.echo("Aborted.")
            raise typer.Exit(code=0)

    remote = settings.remote
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)
    debug_ssh = _get_debug_ssh(ctx)

    # Reuse / re-establish the gateway ControlMaster so scancel reaches the
    # cluster.  The round-robin gateway lands on a healthy login node, which
    # is all a cluster-wide scancel needs.
    gw_control = SshControl(
        gateway=remote.gateway,
        **remote.ssh_control_kwargs(),
        debug=debug_ssh,
    )
    try:
        _connect_with_retry(gw_control, remote.gateway, logger, config=config)
    except Exception as exc:  # noqa: BLE001  -- want broad fallback here
        typer.echo(f"Failed to reach gateway: {exc}" + _ssh_debug_hint(debug_ssh), err=True)
        raise typer.Exit(code=1) from exc

    # `scancel` is cluster-wide, so releasing does NOT need the mirror's
    # pinned login node (which may be wedged) -- we run it over the gateway
    # ControlMaster below.  A login-node hop is only needed to reach a
    # *compute* node's tmux in the sibling-detach branch, where it is
    # established best-effort.
    import subprocess as _sp

    def _forget_allocation():
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

    if siblings:
        # Another session still holds this job; detach this mirror rather
        # than cancelling the shared allocation.  Kill this mirror's tmux
        # session on the node (best effort) so its agent stops, but leave
        # the SLURM job running for the co-resident sessions.
        if session.compute_node:
            # Prefer the pinned login node as the jump to the compute node,
            # but fall back to the gateway (itself a login node) if that pin
            # is unreachable -- a wedged login node must not strand the
            # tmux-kill.  Best-effort throughout.
            jump_host, jump_control = remote.gateway, gw_control
            if session.login_node:
                ln_control = SshControl(
                    gateway=session.login_node,
                    **remote.ssh_control_kwargs(),
                    jump_host=remote.gateway,
                    jump_control=gw_control,
                    debug=debug_ssh,
                )
                try:
                    _connect_with_retry(ln_control, session.login_node, logger)
                    jump_host, jump_control = session.login_node, ln_control
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "login node %s unreachable (%s); routing the tmux-kill "
                        "through the gateway instead.", session.login_node, exc,
                    )
            cn_control = SshControl(
                gateway=session.compute_node,
                **remote.ssh_control_kwargs(),
                jump_host=jump_host,
                jump_control=jump_control,
                extra_options=[
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                ],
                debug=debug_ssh,
            )
            # Defensive: this sibling-detach branch is unreachable for a
            # confined target (each sbatch is its own job, so `siblings` is
            # always empty and the scancel branch runs instead).  But if it
            # were ever reached (shared-confined mode, hand-edited session),
            # a confined session lives on a dedicated `-L` socket under a
            # sanitized name -- the unconfined `sucoder-<mirror>` on the
            # default socket would target the wrong server and no-op.
            if remote.slurm is not None and remote.slurm.confined:
                session_name, socket = confined_tmux_target(mirror)
                tmux_kill = (
                    f"tmux -L {shlex.quote(socket)} kill-session "
                    f"-t {shlex.quote(session_name)}"
                )
            else:
                tmux_name = f"sucoder-{mirror}"
                tmux_kill = f"tmux kill-session -t {shlex.quote(tmux_name)}"
            kill_cmd = [
                "ssh", *cn_control.ssh_options(with_fallback=True),
                session.compute_node,
                f"{tmux_kill} 2>/dev/null || true",
            ]
            logger.debug("detach tmux command: %s", kill_cmd)
            _sp.run(kill_cmd, capture_output=True, text=True, check=False)
        _forget_allocation()
        typer.echo(
            f"Detached mirror {mirror} from SLURM job {job_id}; the "
            f"allocation stays alive for: {', '.join(siblings)}."
        )
        return

    # No siblings: cancel the job.  Run `scancel` over the gateway
    # ControlMaster (round-robin -> a *healthy* login node) rather than the
    # mirror's pinned login node, which may be wedged.  This mirrors the
    # `nodes` command, which runs `sinfo` on the gateway the same way;
    # `_run_remote_capture` adds BatchMode + a wall-clock timeout so a dead
    # mux fails fast instead of hanging.
    result = _run_remote_capture(
        gw_control, remote.gateway,
        f"scancel {shlex.quote(str(job_id))}", debug=debug_ssh,
    )
    if result.returncode != 0:
        # scancel exits 0 even for a nonexistent job (just warns to stderr),
        # so a real non-zero here is an SSH/auth/timeout problem, not "no
        # such job".
        typer.echo(
            f"scancel returned {result.returncode}: {result.stderr.strip()}",
            err=True,
        )
        raise typer.Exit(code=1)

    _forget_allocation()
    typer.echo(f"Released SLURM job {job_id} on {compute_node}.")
    if result.stderr.strip():
        typer.echo(result.stderr.strip(), err=True)


def _run_remote_capture(
    control, host: str, command: str, *, debug: bool = False, timeout: int = 30
) -> subprocess.CompletedProcess:
    """Run *command* on *host* over an established ControlMaster socket.

    Reuses the warm mux (``ControlMaster=auto`` via
    :meth:`SshControl.ssh_options`) so no re-auth is needed when the
    tunnel is already up.  ``BatchMode=yes`` is a post-auth safety belt:
    if the mux has died since :func:`_ensure_ssh_visible` brought it up,
    ssh fails fast instead of silently dropping to an interactive
    ``/dev/tty`` prompt (which a captured-output run would hang on).

    ``BatchMode`` blocks an interactive *prompt* but not a wedged TCP
    path (a zombie mux whose daemon answers ``-O check`` but whose
    connection is dead), so a wall-clock ``timeout`` bounds the run the
    way the probes in :mod:`sucoder.tunnel` do.  A timeout is returned as
    a synthetic non-zero ``CompletedProcess`` (exit 124) so the caller's
    return-code handling covers it.

    Returns the :class:`subprocess.CompletedProcess` (``check=False``);
    the caller decides what a non-zero exit means.
    """
    ssh_cmd = ["ssh"]
    if debug:
        ssh_cmd.append("-v")
    ssh_cmd += [*control.ssh_options(), "-o", "BatchMode=yes", host, command]
    try:
        return subprocess.run(
            ssh_cmd, capture_output=True, text=True, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        return subprocess.CompletedProcess(
            ssh_cmd,
            124,
            stdout=stdout,
            stderr=f"timed out after {timeout}s (wedged tunnel?)",
        )


def _relaunch_session(mirror, config, ctx, logger, dry_run):
    """Re-allocate a fresh SLURM node and relaunch the agent detached.

    Clears the recorded job/node so the executor allocates anew (rather
    than adopting the slice we're replacing), rebuilds the manager (which
    re-allocates and re-arms the on-node deadline timer), and launches
    the agent in a *detached* tmux on the new node.  Returns the new job
    id, or ``None`` if anything failed.

    On the NFS-backed condo target the mirror persists across nodes, so
    no re-clone is needed; ``sync=False`` leaves the working tree exactly
    as the previous agent left it (it rehydrates from its handoff note).

    NB: drives the real allocation + launch path; cluster-validate.  The
    relaunch uses the mirror's *configured* agent (any per-invocation
    ``--agent``/``--agent-command`` override from the original
    ``collaborate`` is not persisted, so configure the agent in
    config.yaml for a renewable target).
    """
    from .session import RemoteSession

    _tgt = ((ctx.obj or {}).get("target_name") if ctx else None)
    session = RemoteSession.load(mirror, target_name=_tgt)
    # Force a fresh allocation: clear job AND node so _ensure_slurm_node
    # does not adopt the slice we're about to replace.
    session.slurm_job_id = None
    session.compute_node = None
    session.save()
    try:
        manager = _build_manager_for_mirror(
            config, logger, dry_run, mirror, cli_ctx=ctx,
        )
        manager.launch_agent(
            manager.context_for(mirror),
            sync=False,
            detached=True,
        )
    except Exception as exc:  # noqa: BLE001 - relaunch must not crash the loop
        logger.warning("relaunch failed: %s", exc)
        return None
    return RemoteSession.load(mirror, target_name=_tgt).slurm_job_id


@app.command("renew")
def renew(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(
        None, help="Mirror name defined in configuration.",
        shell_complete=_mirror_completion,
    ),
    drain_minutes: int = typer.Option(
        20, "--drain-minutes",
        help="Start a proactive re-allocation this many minutes before --time.",
    ),
    poll_interval: int = typer.Option(
        60, "--poll-interval",
        help="Seconds between SLURM job-state probes.",
    ),
    checkpoint_grace: int = typer.Option(
        60, "--checkpoint-grace",
        help="Seconds to let the agent commit/push after the drain nudge.",
    ),
    once: bool = typer.Option(
        False, "--once",
        help="Run a single probe/act cycle and exit (for cron-driven renewal "
             "or smoke checks); pair with --poll-interval 0.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without executing."),
) -> None:
    """Keep a persistent SLURM session alive across allocation turnover.

    Monitors the current allocation and, as it nears its courtesy
    ``--time`` (or if it vanishes at a maintenance reboot), re-allocates
    a fresh node and relaunches the agent in a *detached* tmux.  Reattach
    any time with ``sucoder attach``; stop with Ctrl-C (the allocation is
    left running) or ``sucoder release`` (frees it).

    Requires an existing ``sucoder collaborate`` session on a SLURM
    target.  A transient probe failure (SSH blip) never triggers a
    relaunch -- only a successful probe reporting the job gone/terminal
    does.
    """
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    settings = config.mirrors.get(mirror)
    target = _get_active_target(ctx)
    if target is not None and settings is not None:
        from dataclasses import replace
        settings = replace(settings, remote=target)
    if not settings or not settings.remote or settings.remote.slurm is None:
        typer.echo("renew requires a SLURM-backed remote target.", err=True)
        raise typer.Exit(code=1)

    from .session import RemoteSession
    from .tunnel import SshControl, TunnelError
    from .renew import JobStatus, RenewSettings, parse_time_left, run_renew_loop

    _tgt_name = ((ctx.obj or {}).get("target_name") if ctx else None)
    session = RemoteSession.load(mirror, target_name=_tgt_name)
    if not session.slurm_job_id or not session.login_node:
        typer.echo(
            f"No live SLURM session for mirror {mirror}. "
            "Run `sucoder collaborate` first.",
            err=True,
        )
        raise typer.Exit(code=1)

    remote = settings.remote
    confined = remote.slurm is not None and remote.slurm.confined
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)
    debug_ssh = _get_debug_ssh(ctx)
    # Ride the warm tunnel node rather than a stale mirror pin.
    _reconcile_login_node(remote, session, _tgt_name, logger)
    login_node = session.login_node

    # Establish (or reuse) the gateway + login-node ControlMasters once;
    # probe/scancel multiplex through them for the life of the loop.
    gw_control = SshControl(
        gateway=remote.gateway, **remote.ssh_control_kwargs(), debug=debug_ssh,
    )
    ln_control = SshControl(
        gateway=login_node, **remote.ssh_control_kwargs(),
        jump_host=remote.gateway, jump_control=gw_control, debug=debug_ssh,
    )
    try:
        _connect_with_retry(gw_control, remote.gateway, logger, config=config)
        _connect_with_retry(ln_control, login_node, logger)
    except TunnelError as exc:
        typer.echo(f"Failed to reach the cluster: {exc}" + _ssh_debug_hint(debug_ssh), err=True)
        raise typer.Exit(code=1) from exc

    def probe(job_id):
        cmd = f"squeue --job {shlex.quote(str(job_id))} -h -o '%T|%L'"
        result = _run_remote_capture(ln_control, login_node, cmd, debug=debug_ssh)
        if result.returncode != 0:
            logger.debug("probe failed rc=%s: %s", result.returncode, result.stderr.strip())
            return JobStatus(ok=False)
        line = result.stdout.strip()
        if not line:
            return JobStatus(ok=True, state=None)  # job no longer queued
        state, _, left = line.partition("|")
        return JobStatus(
            ok=True, state=state.strip() or None, mins_left=parse_time_left(left),
        )

    def scancel(job_id):
        _run_remote_capture(
            ln_control, login_node, f"scancel {shlex.quote(str(job_id))}", debug=debug_ssh,
        )

    def request_checkpoint():
        msg = "renew: re-allocation imminent -- commit, push, write handoff now"
        write = (
            'mkdir -p "$HOME/.cache/sucoder" && '
            f'printf %s {shlex.quote(msg)} > "$HOME/.cache/sucoder/renew-requested"'
        )
        if confined:
            # The sentinel lives on NFS $HOME (visible from the login node)
            # and the confined agent polls it from inside its cgroup -- no
            # compute-node SSH (which would land OUTSIDE the cgroup) needed.
            _run_remote_capture(ln_control, login_node, write, debug=debug_ssh)
            return
        cur = RemoteSession.load(mirror, target_name=_tgt_name)
        node = cur.compute_node
        if not node:
            return
        cn = SshControl(
            gateway=node, **remote.ssh_control_kwargs(),
            jump_host=login_node, jump_control=ln_control,
            extra_options=[
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
            ],
            debug=debug_ssh,
        )
        _run_remote_capture(cn, node, write, debug=debug_ssh)

    def _log(message):
        logger.info(message)
        typer.echo(message)

    rs = RenewSettings(
        poll_interval=poll_interval,
        drain_minutes=drain_minutes,
        checkpoint_grace=checkpoint_grace,
    )
    typer.echo(
        f"Renewing mirror {mirror}: watching job {session.slurm_job_id} on "
        f"{session.compute_node or '?'} (drain {drain_minutes}m, poll "
        f"{poll_interval}s). Ctrl-C to stop."
    )
    try:
        run_renew_loop(
            session.slurm_job_id, rs,
            probe=probe,
            relaunch=lambda: _relaunch_session(mirror, config, ctx, logger, dry_run),
            scancel=scancel,
            request_checkpoint=request_checkpoint,
            log=_log,
            max_iterations=1 if once else None,
        )
    except KeyboardInterrupt:
        typer.echo(
            "\nrenew stopped; the allocation is left running "
            "(`sucoder release` to free it)."
        )


# SLURM partition names are alphanumeric plus `_`, `-`, `.`; a comma lets
# `sinfo -p` take a list (e.g. `savio3,savio4_htc`).  Restricting to this
# set (and rejecting a leading `-`) keeps shell metacharacters out and
# stops a value being mistaken for an `sinfo` option.
# `\Z` (not `$`) so a trailing newline can't sneak through.
_PARTITION_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_,.-]*\Z")


@app.command("nodes")
def nodes(
    ctx: typer.Context,
    partition: Optional[str] = typer.Argument(
        None,
        help="SLURM partition to query (defaults to the target's slurm.partition).",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Increase console logging."
    ),
) -> None:
    """Show compute-node availability for a SLURM partition (read-only).

    Reuses the target's warm gateway ControlMaster, so it costs no OTP
    when a tunnel is already up (and falls back to a fresh,
    OTP-prompting connection otherwise).  Runs ``sinfo`` on the login
    node and prints one row per node --- state, CPUs
    (Allocated/Idle/Other/Total) and load --- followed by the
    drained/down nodes and their reasons.

    The partition defaults to the target's ``slurm.partition``; an
    optional positional argument overrides it (e.g. ``savio3_gpu``).
    Composes with ``collaborate --node`` / ``--local-disk``: see which
    nodes are free, then aim at one.

    Caveat: ``sinfo`` reports SLURM state, not Lustre health.  A node can
    show ``idle`` while its filesystem mount is wedged, so weigh the
    drain reasons and any anomalous load on an otherwise-idle node --- the
    query cannot promise a node's filesystem is healthy.
    """
    config = _get_config(ctx)
    remote = _get_active_target(ctx)
    if remote is None:
        typer.echo(
            "`nodes` needs a remote target; pass one with -T, "
            "e.g. `sucoder -T savio-node nodes`.",
            err=True,
        )
        raise typer.Exit(code=2)

    part = partition or (remote.slurm.partition if remote.slurm else None)
    if not part:
        typer.echo(
            "No partition given and the target has no `slurm.partition`. "
            "Pass one explicitly, e.g. `sucoder -T <target> nodes savio3`.",
            err=True,
        )
        raise typer.Exit(code=2)

    if not _PARTITION_RE.match(part):
        typer.echo(
            f"Invalid partition name {part!r}; expected letters, digits, and "
            "'_', '-', '.', ',' (e.g. savio3 or savio3,savio4_htc).",
            err=True,
        )
        raise typer.Exit(code=2)

    logger = setup_logger("sucoder.nodes", config.log_dir, verbose)
    debug_ssh = _get_debug_ssh(ctx)

    from .tunnel import SshControl

    # A jump-less control to the gateway.  On BRC `ssh <gateway>` lands
    # on a login node (the same host `_build_executor` pins via `ssh
    # <gateway> hostname`), where the SLURM client commands live, so we
    # can run sinfo directly over this socket without pinning a node.
    gw_control = SshControl(
        gateway=remote.gateway,
        **remote.ssh_control_kwargs(),
        debug=debug_ssh,
    )
    try:
        _connect_with_retry(gw_control, remote.gateway, logger, config=config)
    except Exception as exc:  # noqa: BLE001 -- surface any setup failure
        typer.echo(f"Failed to reach gateway {remote.gateway}: {exc}" + _ssh_debug_hint(debug_ssh), err=True)
        raise typer.Exit(code=1) from exc

    q = shlex.quote(part)
    avail = _run_remote_capture(
        gw_control,
        remote.gateway,
        f'sinfo -p {q} -N -o "%N %6t %.15C %.6O"',
        debug=debug_ssh,
    )
    if avail.returncode != 0:
        detail = avail.stderr.strip() or avail.stdout.strip() or "(no output)"
        typer.echo(
            f"`sinfo -p {part}` failed on {remote.gateway} "
            f"(exit {avail.returncode}): {detail}",
            err=True,
        )
        raise typer.Exit(code=1)

    drain = _run_remote_capture(
        gw_control, remote.gateway, f"sinfo -p {q} -R", debug=debug_ssh
    )

    typer.echo(f"Partition {part}:")
    typer.echo(avail.stdout.rstrip("\n"))
    typer.echo("")
    typer.echo("Drained/down nodes (admins often drain sick nodes):")
    if drain.returncode != 0:
        # Don't masquerade a failed query as a healthy partition.
        typer.echo("  (could not query drain reasons)")
        detail = drain.stderr.strip()
        if detail:
            typer.echo(f"  sinfo -R failed: {detail}", err=True)
    else:
        # `sinfo -R` prints a header row even when nothing is drained, so
        # a header-only result (<= 1 line) means "none".
        drain_lines = drain.stdout.strip().splitlines()
        if len(drain_lines) <= 1:
            typer.echo("  (none reported)")
        else:
            typer.echo(drain.stdout.rstrip("\n"))

    typer.echo(
        "\nNote: sinfo reports SLURM state, not Lustre health --- a node can "
        "show 'idle' while its filesystem mount is wedged. Weigh drain "
        "reasons and anomalous load on idle nodes accordingly.",
        err=True,
    )


# ----------------------------------------------------------------------
# Persistent (warm) tunnels — the cheap hops cost no compute money, so
# keeping the gateway / login-node / DTN ControlMasters alive removes the
# OTP friction from every `collaborate` and lets plain ssh / Emacs TRAMP
# ride the same mux.  This is the `up`/`status` spike; `down` is included
# so the warm sockets can be torn down and re-tested.
# ----------------------------------------------------------------------

tunnel_app = typer.Typer(
    help="Keep the cheap SSH tunnels (gateway/login/DTN) to a target warm.",
)
app.add_typer(tunnel_app, name="tunnel")


def _resolve_tunnel_target(ctx: typer.Context):
    """Return ``(RemoteConfig, target_name)`` for a `tunnel` subcommand.

    The tunnel commands are target-scoped, not mirror-scoped: they bring
    up the hops shared by every mirror on that cluster.  A ``-T <target>``
    is therefore required.
    """
    remote = _get_active_target(ctx)
    target_name = (ctx.obj or {}).get("target_name") if ctx.obj else None
    if remote is None or not target_name:
        typer.echo(
            "`tunnel` requires a target: `sucoder -T <target> tunnel ...`",
            err=True,
        )
        raise typer.Exit(code=2)
    return remote, target_name


@app.command("cert")
def cert(
    ctx: typer.Context,
    user: Optional[str] = typer.Option(
        None, "--user",
        help="BRC username (default: $BRC_USER or your local login name).",
    ),
    lifetime: str = typer.Option(
        "12h", "--lifetime",
        help="Requested cert lifetime (the MSM CA caps this at 12h).",
    ),
) -> None:
    """Mint a short-lived BRC SSH certificate for this target's gateway.

    Prompts for your BRC PIN + one-time code, POSTs them to the MSM CA, and
    writes the cert to the target's ``cert_file`` so subsequent connections
    (and `tunnel up`) are OTP-free until it expires.  Requires ``-T <target>``.
    """
    from . import cert as cert_mod

    remote, target_name = _resolve_tunnel_target(ctx)
    if not remote.cert_file:
        typer.echo(
            f"Target {target_name} has no `cert_file`; nothing to mint into. "
            "Add e.g. `cert_file: ~/.ssh/ssh_certs/brc_cert` under the target.",
            err=True,
        )
        raise typer.Exit(code=2)
    config = _get_config(ctx)
    username = user or os.environ.get("BRC_USER") or remote.remote_user or config.human_user
    typer.echo(f"Minting a {lifetime} BRC cert for {username} @ {remote.gateway} ...")
    pin = typer.prompt("BRC PIN", hide_input=True)
    otp = typer.prompt("BRC OTP")
    try:
        data = cert_mod.mint(
            remote.cert_file, cert_mod.DEFAULT_CA_URL, username, pin, otp, lifetime,
        )
    except cert_mod.CertError as exc:
        typer.echo(f"Cert request failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    glyph, msg = _cert_status(str(remote.cert_file))
    typer.echo(f"  {glyph} {msg}")
    key_id = data.get("key_id")
    typer.echo(
        f"Minted for {username}{f' (key id {key_id})' if key_id else ''}. "
        f"`sucoder -T {target_name} tunnel up` is OTP-free until it expires."
    )


def _tunnel_session_name(target_name: str) -> str:
    """Session key for a target's warm-tunnel state (login-node pin)."""
    return f"tunnel-{target_name}"


def _warm_free_tunnels(remote, session, logger, *, debug_ssh: bool, config=None):
    """Bring the gateway, login-node, and DTN ControlMasters online.

    Returns ``(gw_control, ln_control, dtn_control)``.  ``ln_control``
    may be ``None`` if the login node could not be pinned; ``dtn_control``
    may be ``None`` if the DTN is unreachable (it is optional).  Pins and
    persists the login node into *session* on first run.
    """
    import subprocess as _sp
    from .tunnel import SshControl, TunnelError

    gw_control = SshControl(
        gateway=remote.gateway,
        **remote.ssh_control_kwargs(),
        debug=debug_ssh,
    )
    _connect_with_retry(gw_control, remote.gateway, logger, config=config)

    # Pin a login node through the (now warm) gateway if we don't have one.
    if not session.login_node:
        pin_cmd = ["ssh", *gw_control.ssh_options(), remote.gateway, "hostname"]
        try:
            result = _sp.run(pin_cmd, capture_output=True, text=True, check=True)
            session.login_node = result.stdout.strip()
            session.save()
            logger.info("Pinned login node: %s", session.login_node)
        except _sp.CalledProcessError as exc:
            logger.warning(
                "Could not pin login node: %s", (exc.stderr or "").strip(),
            )

    ln_control = None
    if session.login_node:
        ln_control = SshControl(
            gateway=session.login_node,
            **remote.ssh_control_kwargs(),
            jump_host=remote.gateway,
            jump_control=gw_control,
            debug=debug_ssh,
        )
        try:
            _connect_with_retry(ln_control, session.login_node, logger)
        except TunnelError as exc:
            logger.warning("Login node %s unreachable: %s", session.login_node, exc)
            ln_control = None

    dtn_control = SshControl(
        gateway=remote.transfer_host,
        **remote.ssh_control_kwargs(),
        jump_host=remote.gateway,
        jump_control=gw_control,
        debug=debug_ssh,
    )
    try:
        # DTN is optional; ride out only a brief transient blip.
        _connect_with_retry(dtn_control, remote.transfer_host, logger, max_wait=6)
    except TunnelError as exc:
        logger.warning("DTN %s unreachable: %s", remote.transfer_host, exc)
        dtn_control = None

    return gw_control, ln_control, dtn_control


@tunnel_app.command("up")
def tunnel_up(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
    no_config_edit: bool = typer.Option(
        False,
        "--no-config-edit",
        help="Warm the sockets but do not touch ~/.ssh/config.",
    ),
) -> None:
    """Authenticate once and warm the gateway/login/DTN ControlMasters.

    Subsequent `sucoder -T <target> collaborate`, plain `ssh`, and Emacs
    TRAMP reuse these warm sockets without a fresh OTP prompt.  Unless
    `--no-config-edit` is given, writes a managed `~/.ssh/config` block of
    `<target>-gw`/`-ln`/`-dtn` aliases pointing at the same sockets.
    """
    import getpass
    from . import sshconfig
    from .session import RemoteSession
    from .tunnel import TunnelError

    config = _get_config(ctx)
    remote, target_name = _resolve_tunnel_target(ctx)
    logger = setup_logger("sucoder.tunnel", config.log_dir, verbose)
    debug_ssh = _get_debug_ssh(ctx)
    session = RemoteSession.load(_tunnel_session_name(target_name))

    try:
        gw, ln, dtn = _warm_free_tunnels(
            remote, session, logger, debug_ssh=debug_ssh, config=config,
        )
    except TunnelError as exc:
        typer.echo(str(exc) + _ssh_debug_hint(debug_ssh), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Warm tunnels for target {target_name} (ControlPersist {remote.control_persist}):")
    typer.echo(f"  {'✓' if gw and gw.is_active() else '✗'} gateway  {remote.gateway}")
    if session.login_node:
        typer.echo(f"  {'✓' if ln and ln.is_active() else '✗'} login    {session.login_node}")
    else:
        typer.echo("  ✗ login    (not pinned)")
    typer.echo(f"  {'✓' if dtn and dtn.is_active() else '✗'} DTN      {remote.transfer_host}")

    if no_config_edit:
        return

    block = sshconfig.render_block(
        target_name,
        remote.gateway,
        remote.transfer_host,
        login_node=session.login_node,
        user=remote.remote_user or config.human_user,
        **remote.ssh_control_kwargs(),
    )
    path = sshconfig.write_block(block, target_name)
    aliases = sshconfig.alias_names(target_name)
    typer.echo(
        f"Wrote ~/.ssh/config block: {aliases['gw']}, {aliases['ln']}, {aliases['dtn']}  ({path})"
    )


@tunnel_app.command("status")
def tunnel_status(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Report whether each warm tunnel is alive (no auth, no network cost)."""
    from . import sshconfig
    from .session import RemoteSession
    from .tunnel import SshControl

    remote, target_name = _resolve_tunnel_target(ctx)
    session = RemoteSession.load(_tunnel_session_name(target_name))

    gw = SshControl(gateway=remote.gateway, **remote.ssh_control_kwargs())
    hops = [("gateway", remote.gateway, gw)]
    if session.login_node:
        hops.append((
            "login", session.login_node,
            SshControl(
                gateway=session.login_node,
                **remote.ssh_control_kwargs(),
                jump_host=remote.gateway,
                jump_control=gw,
            ),
        ))
    hops.append((
        "dtn", remote.transfer_host,
        SshControl(
            gateway=remote.transfer_host,
            **remote.ssh_control_kwargs(),
            jump_host=remote.gateway,
            jump_control=gw,
        ),
    ))

    results = [
        {"hop": name, "host": host, "active": ctrl.is_active()}
        for name, host, ctrl in hops
    ]
    cfg_present = sshconfig.block_present(target_name)

    if json_output:
        import json
        typer.echo(json.dumps(
            {"target": target_name, "ssh_config": cfg_present, "hops": results},
            indent=2,
        ))
        return

    typer.echo(f"Warm tunnels for target {target_name}:")
    for r in results:
        mark = "✓ ACTIVE" if r["active"] else "✗ DEAD  "
        typer.echo(f"  {mark}  {r['hop']:<8} {r['host']}")
    typer.echo(f"  ssh_config block: {'present' if cfg_present else 'absent'}")


def _parse_cert_time(raw: str):
    """Parse an ssh-keygen ``Valid: ... to <ts>`` timestamp; None if unknown.

    ssh-keygen prints local-time stamps whose exact format varies by
    OpenSSH version, so try the common ones and give up gracefully.
    """
    import datetime

    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _cert_status(cert_file: str):
    """Best-effort status of a gateway SSH cert for ``tunnel doctor``.

    Returns ``(glyph, message)`` and never raises.  ``glyph`` is ``✓``
    (present/valid), ``•`` (configured but not minted / unreadable), or
    ``⚠`` (expired or expiring soon).
    """
    import datetime
    import os
    import subprocess as _sp

    key = os.path.expanduser(cert_file)
    pub = key + "-cert.pub"
    if not os.path.exists(pub):
        return ("•", f"cert configured but not minted ({pub} absent) — "
                     "mint one with `sucoder -T <target> cert`")
    try:
        out = _sp.run(
            ["ssh-keygen", "-L", "-f", pub],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout
    except (OSError, _sp.SubprocessError):
        return ("•", f"cert present ({pub}); ssh-keygen unavailable to read validity")
    valid_line = next(
        (ln.strip() for ln in out.splitlines() if "Valid:" in ln), ""
    )
    to_raw = valid_line.split(" to ", 1)[-1].strip() if " to " in valid_line else ""
    expires = _parse_cert_time(to_raw)
    if expires is None:
        return ("✓", f"cert present ({valid_line or pub})")
    now = datetime.datetime.now()
    if expires <= now:
        return ("⚠", f"cert EXPIRED ({to_raw}) — re-mint: `sucoder -T <target> cert`")
    if expires - now <= datetime.timedelta(hours=2):
        return ("⚠", f"cert expires soon ({to_raw}) — re-mint before it lapses")
    return ("✓", f"cert valid to {to_raw}")


@tunnel_app.command("doctor")
def tunnel_doctor(ctx: typer.Context) -> None:
    """Diagnose ssh_config / session issues that silently break tunnel reuse.

    The headline check is *shadowing*: a ``Host *`` (or other matching)
    block earlier in ``~/.ssh/config`` that sets ``ControlPath``/
    ``ControlMaster`` overrides the managed aliases (ssh uses the first
    value per keyword), so ssh looks for the wrong socket and
    re-authenticates.  Also flags a missing block and login-node pin
    drift.  Exits non-zero if any problem is found.
    """
    from . import sshconfig
    from .session import RemoteSession

    remote, target_name = _resolve_tunnel_target(ctx)
    session = RemoteSession.load(_tunnel_session_name(target_name))
    aliases = sshconfig.alias_names(target_name)
    problems = 0

    typer.echo(f"tunnel doctor — target {target_name}:")

    # 1. Managed block present?
    if sshconfig.block_present(target_name):
        typer.echo(
            f"  ✓ ssh_config block present "
            f"({aliases['gw']}, {aliases['ln']}, {aliases['dtn']})"
        )
    else:
        problems += 1
        typer.echo(
            "  ✗ ssh_config block missing — run "
            f"`sucoder -T {target_name} tunnel up`"
        )

    # 2. Shadowing Host/Match blocks before ours (the ControlPath trap).
    shadow = sshconfig.find_shadowing_hosts(target_name)
    if shadow:
        problems += 1
        typer.echo(
            "  ✗ a block BEFORE the managed block shadows its connection "
            "sharing (ssh uses the first value per keyword):"
        )
        for label, key in shadow:
            typer.echo(f"      `{label}` sets {key} — the alias's {key} is ignored.")
        typer.echo(
            f"    Fix: re-run `sucoder -T {target_name} tunnel up` (writes the "
            "block at the top), or move the block above that stanza."
        )
    else:
        typer.echo("  ✓ no preceding block shadows the managed aliases")

    # 3. Login-node pin drift: alias HostName vs the pinned node.
    if session.login_node:
        configured = sshconfig.managed_hostnames(target_name).get(aliases["ln"])
        if configured and configured != session.login_node:
            problems += 1
            typer.echo(
                f"  ✗ login alias HostName ({configured}) != pinned login "
                f"node ({session.login_node}) — re-run "
                f"`sucoder -T {target_name} tunnel up` to re-pin."
            )
        else:
            typer.echo(f"  ✓ login node pinned: {session.login_node}")
    else:
        typer.echo(
            f"  • login node not pinned yet — run "
            f"`sucoder -T {target_name} tunnel up`"
        )

    # 3b. Mirror-vs-tunnel login-node drift.  collaborate/attach/renew dial the
    #     *mirror* session's login node, pinned independently of this tunnel
    #     session, so it can drift onto a stale node.  Informational only: it
    #     self-heals (each mirror command adopts the warm tunnel node), so it
    #     doesn't fail the exit code -- but surfacing it explains a transient
    #     wedge and the auto-repin.
    if session.login_node:
        mirror_pins = RemoteSession.login_nodes_for_target(target_name)
        drifted = {k: n for k, n in mirror_pins.items() if n != session.login_node}
        if drifted:
            typer.echo(
                f"  • mirror session pin(s) differ from the warm login node "
                f"({session.login_node}) — they auto-reconcile on the next "
                "collaborate/attach/renew:"
            )
            for key, node in sorted(drifted.items()):
                typer.echo(f"      {key}: {node}")
        elif mirror_pins:
            typer.echo("  ✓ mirror sessions agree with the warm login node")

    # 4. Gateway SSH certificate (optional; enables passwordless gateway auth).
    #    Informational only — a missing/expired cert just means a password
    #    prompt, it doesn't break tunnel reuse, so it never fails the exit code.
    if remote.cert_file:
        glyph, msg = _cert_status(str(remote.cert_file))
        typer.echo(f"  {glyph} {msg}")

    typer.echo("  (run `tunnel status` to check whether the sockets are live)")
    if problems:
        typer.echo(f"\n{problems} problem(s) found.", err=True)
        raise typer.Exit(code=1)
    typer.echo("\nAll checks passed.")


@tunnel_app.command("down")
def tunnel_down(
    ctx: typer.Context,
    prune: bool = typer.Option(
        False, "--prune", help="Also remove the ~/.ssh/config block.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
) -> None:
    """Close the warm tunnels for a target (`ssh -O exit` on each socket)."""
    from . import sshconfig
    from .session import RemoteSession
    from .tunnel import SshControl

    config = _get_config(ctx)
    remote, target_name = _resolve_tunnel_target(ctx)
    logger = setup_logger("sucoder.tunnel", config.log_dir, verbose)
    session = RemoteSession.load(_tunnel_session_name(target_name))

    # Close children before the gateway they ride on.
    hosts = [h for h in (session.login_node, remote.transfer_host, remote.gateway) if h]
    for host in hosts:
        SshControl(gateway=host, **remote.ssh_control_kwargs()).close(logger)
        typer.echo(f"  closed {host}")

    if prune and sshconfig.remove_block(target_name):
        typer.echo(f"Removed ~/.ssh/config block for {target_name}.")


def _forward_spec(local_port: int, remote_port: int) -> str:
    """The ssh ``-L`` spec used for compute-node service forwards.

    The connection terminates *on the node* (``localhost`` is resolved
    there), so services bound to 127.0.0.1 — the common default for
    Jupyter-style token URLs — are reachable.
    """
    return f"{local_port}:localhost:{remote_port}"


def _mux_forward(action: str, spec: str, socket_path: str, node: str):
    """Run ``ssh -O forward|cancel -L <spec>`` against *node*'s mux socket.

    Returns ``(returncode, stderr)``.  ``-O`` requests are mux *control*
    operations: they open no remote session, so they succeed even when
    the master is session-saturated (sshd ``MaxSessions`` — the
    "Session open refused by peer" state), unlike spawning a shell.
    """
    import subprocess as _sp

    try:
        result = _sp.run(
            ["ssh", "-O", action, "-L", spec,
             "-o", f"ControlPath={socket_path}", node],
            capture_output=True,
            stdin=_sp.DEVNULL,
            text=True,
            check=False,
            timeout=15,
        )
    except _sp.TimeoutExpired:
        return -1, "mux request timed out"
    return result.returncode, (result.stderr or "").strip()


def _resolve_forward_node(target_name: str, explicit: Optional[str]):
    """Pick the compute node for a forward; raise typer.Exit if ambiguous."""
    from .session import RemoteSession

    if explicit:
        return explicit
    nodes = RemoteSession.compute_nodes_for_target(target_name)
    distinct = sorted({n for n in nodes.values() if n})
    if len(distinct) == 1:
        return distinct[0]
    if not distinct:
        typer.echo(
            f"No collaborate session records a compute node for target "
            f"{target_name} — pass --node (e.g. from `squeue --me`).",
            err=True,
        )
        raise typer.Exit(code=2)
    typer.echo("Multiple compute nodes in play — pick one with --node:", err=True)
    for key, n in sorted(nodes.items()):
        typer.echo(f"  {n}  (session {key})", err=True)
    raise typer.Exit(code=2)


@tunnel_app.command("forward")
def tunnel_forward(
    ctx: typer.Context,
    port: int = typer.Argument(
        ..., min=1, max=65535,
        help="Port the service listens on, on the compute node.",
    ),
    node: Optional[str] = typer.Option(
        None, "--node",
        help="Compute node running the service (default: the node recorded "
             "by this target's collaborate session).",
    ),
    local_port: Optional[int] = typer.Option(
        None, "--local-port", min=1, max=65535,
        help="Local listen port (default: same as PORT).",
    ),
    cancel: bool = typer.Option(
        False, "--cancel", help="Tear down a previously created forward.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
) -> None:
    """Forward localhost:PORT to a compute node's localhost:PORT.

    Rides the warm ControlMasters: if a collaborate session already
    holds a mux to the node, the forward is added to it with **zero new
    authentication**; otherwise a node master is established through the
    gateway/login sockets (no OTP).  The forward terminates on the node,
    so loopback-bound services (Jupyter, claude-science, …) work.  After
    this, open the service's printed URL with the host replaced by
    ``localhost``.
    """
    from .session import RemoteSession
    from .tunnel import SshControl, TunnelError

    config = _get_config(ctx)
    remote, target_name = _resolve_tunnel_target(ctx)
    logger = setup_logger("sucoder.tunnel", config.log_dir, verbose)
    debug_ssh = _get_debug_ssh(ctx)
    session = RemoteSession.load(_tunnel_session_name(target_name))
    local = local_port or port

    if cancel:
        record = next(
            (f for f in session.forwards
             if f.get("local_port") == local
             and (node is None or f.get("node") == node)),
            None,
        )
        if record is None:
            typer.echo(
                f"No recorded forward on localhost:{local} for target "
                f"{target_name} (see `tunnel forwards`).",
                err=True,
            )
            raise typer.Exit(code=1)
        spec = _forward_spec(record["local_port"], record["remote_port"])
        sock = SshControl(
            gateway=record["node"], **remote.ssh_control_kwargs(),
        ).socket_path
        rc, err = _mux_forward("cancel", spec, str(sock), record["node"])
        session.forwards = [f for f in session.forwards if f is not record]
        session.save()
        if rc == 0:
            typer.echo(
                f"✓ cancelled localhost:{record['local_port']} → "
                f"{record['node']}:{record['remote_port']}"
            )
        else:
            # A dead master already dropped its listeners; removing the
            # record is all that's left to do.
            typer.echo(
                f"Removed the record; mux cancel failed ({err or f'rc={rc}'}) "
                "— the node master is probably gone, so the listener is too."
            )
        return

    node = _resolve_forward_node(target_name, node)

    existing = next(
        (f for f in session.forwards if f.get("local_port") == local), None,
    )
    if existing:
        typer.echo(
            f"localhost:{local} already forwards to {existing.get('node')}:"
            f"{existing.get('remote_port')} — cancel it first:\n"
            f"  sucoder -T {target_name} tunnel forward {local} --cancel",
            err=True,
        )
        raise typer.Exit(code=1)

    # Reuse a live node master (e.g. a running collaborate session's)
    # outright; only when there is none do we walk the gw -> ln -> node
    # chain, and the warm gw/ln sockets make that OTP-free.
    gw_control = SshControl(
        gateway=remote.gateway, **remote.ssh_control_kwargs(), debug=debug_ssh,
    )
    ln_control = SshControl(
        gateway=session.login_node, **remote.ssh_control_kwargs(),
        jump_host=remote.gateway, jump_control=gw_control, debug=debug_ssh,
    ) if session.login_node else None
    node_control = SshControl(
        gateway=node, **remote.ssh_control_kwargs(),
        jump_host=session.login_node,
        jump_control=ln_control, debug=debug_ssh,
    )
    try:
        if not node_control.is_active():
            if not session.login_node:
                typer.echo(
                    f"Login node not pinned — run `sucoder -T {target_name} "
                    "tunnel up` first.",
                    err=True,
                )
                raise typer.Exit(code=1)
            _connect_with_retry(gw_control, remote.gateway, logger, config=config)
            _connect_with_retry(ln_control, session.login_node, logger)
            _connect_with_retry(node_control, node, logger)
    except TunnelError as exc:
        typer.echo(str(exc) + _ssh_debug_hint(debug_ssh), err=True)
        raise typer.Exit(code=1) from exc

    spec = _forward_spec(local, port)
    rc, err = _mux_forward("forward", spec, str(node_control.socket_path), node)
    if rc != 0:
        hint = ""
        if "address already in use" in err.lower():
            hint = (
                f"\nlocalhost:{local} is taken on this machine — pick "
                "another with --local-port."
            )
        typer.echo(f"Could not add forward ({err or f'rc={rc}'}).{hint}", err=True)
        raise typer.Exit(code=1)

    session.forwards.append(
        {"local_port": local, "node": node, "remote_port": port},
    )
    session.save()
    typer.echo(f"✓ forwarding localhost:{local} → {node}:{port}")
    typer.echo(f"  open:    http://localhost:{local}/")
    typer.echo(
        f"  cancel:  sucoder -T {target_name} tunnel forward {local} --cancel"
    )


@tunnel_app.command("forwards")
def tunnel_forwards(ctx: typer.Context) -> None:
    """List this target's recorded port forwards and probe their masters."""
    from .session import RemoteSession
    from .tunnel import SshControl

    remote, target_name = _resolve_tunnel_target(ctx)
    session = RemoteSession.load(_tunnel_session_name(target_name))
    if not session.forwards:
        typer.echo(f"No forwards recorded for target {target_name}.")
        return
    typer.echo(f"Forwards for target {target_name}:")
    for f in session.forwards:
        node_name = f.get("node", "?")
        live = SshControl(
            gateway=node_name, **remote.ssh_control_kwargs(),
        ).is_active(deep=False)
        glyph = "✓" if live else "✗"
        status = (
            "mux live" if live
            else "mux DOWN — re-run `tunnel forward` to re-establish"
        )
        typer.echo(
            f"  {glyph} localhost:{f.get('local_port')} → "
            f"{node_name}:{f.get('remote_port')}  ({status})"
        )


@app.command("mirrors-list")
def mirrors_list(ctx: typer.Context) -> None:
    """Display configured mirrors with their canonical repositories."""
    config = _get_config(ctx)
    entries = sorted(config.mirrors.items())
    if not entries:
        typer.echo("No mirrors configured.")
        return

    name_width = max(len("Mirror"), *(len(name) for name, _ in entries))
    branch_width = max(
        len("Base"),
        *(len(settings.default_base_branch or "(auto)") for _, settings in entries),
    )

    header = f"{'Mirror':<{name_width}}  {'Base':<{branch_width}}  Canonical Repo  Mirror Path"
    typer.echo(header)
    typer.echo("-" * len(header))

    for name, settings in entries:
        canonical = str(settings.canonical_repo)
        mirror_path = str(config.mirror_root / settings.mirror_dirname)
        base = settings.default_base_branch or "(auto)"
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
