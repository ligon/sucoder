"""Command execution helpers shared across the toolkit."""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .logging_utils import describe_remote, progress, summarize_command


class CommandError(RuntimeError):
    """Raised when a subprocess finishes with a non-zero exit code."""

    def __init__(self, message: str, result: "CommandResult"):
        super().__init__(message)
        self.result = result


# Lines OpenSSH prints when a ControlMaster refuses a session and the
# client dials afresh.  Clusters that allow one session per connection
# print them routinely; logged first, they hid the error that mattered.
_SSH_MUX_NOISE_PREFIXES = (
    "mux_client_request_session:",
    "controlsocket ",
)


def _without_ssh_mux_noise(stderr: str) -> str:
    """``stderr`` stripped, minus OpenSSH's multiplexing chatter."""
    lines = [
        line for line in (stderr or "").splitlines()
        if not line.strip().lower().startswith(_SSH_MUX_NOISE_PREFIXES)
    ]
    return "\n".join(lines).strip()


@dataclass
class CommandResult:
    """Representation of a completed subprocess."""

    requested_args: List[str]
    executed_args: List[str]
    stdout: str
    stderr: str
    returncode: int

    def check_returncode(self) -> None:
        if self.returncode != 0:
            raise CommandError(
                f"Command {self.requested_args} failed with code {self.returncode}",
                self,
            )


def _format_display(args: Sequence[str]) -> str:
    return shlex.join(list(args))


@dataclass
class CommandExecutor:
    """Run subprocesses as either the human user or the agent user."""

    human_user: str
    agent_user: str
    agent_group: str
    logger: logging.Logger
    dry_run: bool = False
    use_sudo_for_agent: bool = True
    default_umask: int = 0o007

    def run_human(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        capture_output: bool = True,
        timeout: Optional[int] = None,
    ) -> CommandResult:
        return self._run(
            list(args),
            check=check,
            cwd=cwd,
            env=env,
            as_agent=False,
            capture_output=capture_output,
            timeout=timeout,
        )

    def run_agent(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        umask: Optional[int] = None,
        capture_output: bool = True,
        timeout: Optional[int] = None,
        input: Optional[str] = None,
        show_progress: bool = True,
    ) -> CommandResult:
        return self._run(
            list(args),
            check=check,
            cwd=cwd,
            env=env,
            as_agent=True,
            umask=umask or self.default_umask,
            capture_output=capture_output,
            timeout=timeout,
            input=input,
            show_progress=show_progress,
        )

    def _run(
        self,
        args: List[str],
        *,
        check: bool,
        cwd: Optional[str],
        env: Optional[Mapping[str, str]],
        as_agent: bool,
        umask: Optional[int] = None,
        capture_output: bool = True,
        timeout: Optional[int] = None,
        input: Optional[str] = None,
        show_progress: bool = True,
        progress_command: Optional[str] = None,
    ) -> CommandResult:
        requested_args = list(args)
        executed_args = (
            self._wrap_agent_command(args, env=env, umask=umask, cwd=cwd)
            if as_agent
            else list(args)
        )

        display_label = "agent" if as_agent else "human"
        self.logger.debug(
            "[%s] %s",
            display_label,
            _format_display(requested_args),
        )

        if self.dry_run:
            self.logger.info("DRY-RUN: %s", _format_display(requested_args))
            return CommandResult(
                requested_args=requested_args,
                executed_args=executed_args,
                stdout="",
                stderr="",
                returncode=0,
            )

        run_kwargs: Dict[str, Any] = {
            "cwd": cwd,
            # When running as agent via sudo, env is intentionally set to None here
            # because _wrap_agent_command prepends 'env K=V' to the sudo command,
            # which is the correct way to pass env vars through sudo.
            "env": None if as_agent and self.use_sudo_for_agent else env,
            "text": True,
        }
        if capture_output:
            run_kwargs["capture_output"] = True
            if input is None:
                # Prevent subprocesses (especially SSH) from inheriting and
                # consuming the parent's stdin.  Without this, SSH may read
                # from stdin for password/host-key prompts, corrupting it
                # before interactive helpers like _prompt_choice() call
                # input().  Commands that genuinely need stdin use
                # capture_output=False, or pipe content via ``input``.
                run_kwargs["stdin"] = subprocess.DEVNULL
        if input is not None:
            # Feed content over stdin (e.g. writing a remote file via
            # ``cat >``) without it ever appearing on a command line.
            run_kwargs["input"] = input
        if timeout is not None:
            run_kwargs["timeout"] = timeout
        # Say what we are about to wait on.  Only for commands that leave
        # this machine: those are the ones that cost seconds each (a session
        # open on a BRC login node is ~8s before the command even starts),
        # and a launch makes a dozen of them back to back.
        #
        # ``show_progress=False`` is for a caller that repeats one identical
        # round trip -- a bounded poll -- where a line per attempt reads as
        # a hang rather than as a wait.  Such a caller announces the wait
        # once itself; the suppressed trips stay visible in the debug log
        # below, so -v loses nothing.
        #
        # ``progress_command`` is what a caller that wrapped the command
        # wants named instead of the wrapper.  The ssh argv carries the
        # whole remote script -- ``export``, ``cd``, the ``timeout``
        # guard -- and the line keeps 60 characters of it, so a
        # scaffolding command in a mirror read ``cd /global/home/users/
        # ligon/mirrors/Emu-GMM && if command...`` and lost the one word
        # that says what the round trip is for.  Display only: the
        # ``[human] ssh ...`` debug record above and the error lines
        # keep the exact argv.
        remote = describe_remote(executed_args)
        if remote is not None and progress_command is not None:
            remote = (remote[0], progress_command)
        if remote is not None and show_progress:
            progress(*remote)
        started = time.monotonic()
        try:
            result_proc = subprocess.run(
                executed_args,
                check=False,
                **run_kwargs,
            )
        except subprocess.TimeoutExpired:
            message = (
                f"Command timed out after {timeout}s: "
                f"{_format_display(requested_args)}"
            )
            self.logger.error(message)
            raise CommandError(
                message,
                CommandResult(
                    requested_args=requested_args,
                    executed_args=executed_args,
                    stdout="",
                    stderr="(timed out)",
                    returncode=-1,
                ),
            )

        if remote is not None:
            self.logger.debug(
                "%s: %s -> exit %s in %.1fs",
                remote[0], summarize_command(remote[1]),
                result_proc.returncode, time.monotonic() - started,
            )

        result = CommandResult(
            requested_args=requested_args,
            executed_args=executed_args,
            stdout=(result_proc.stdout or "") if capture_output else "",
            stderr=(result_proc.stderr or "") if capture_output else "",
            returncode=result_proc.returncode,
        )

        if check and result_proc.returncode != 0:
            message = (
                f"Command failed with exit code {result_proc.returncode}: "
                f"{_format_display(requested_args)}"
            )
            stderr = _without_ssh_mux_noise(result.stderr)
            if stderr:
                self.logger.error(stderr)
            raise CommandError(message, result)

        return result

    def _wrap_agent_command(
        self,
        args: Sequence[str],
        *,
        env: Optional[Mapping[str, str]],
        umask: Optional[int],
        cwd: Optional[str] = None,
    ) -> List[str]:
        if not self.use_sudo_for_agent:
            return list(args)

        command_str = _format_display(args)
        umask_value = f"{umask:04o}" if umask is not None else f"{self.default_umask:04o}"

        quoted_user = shlex.quote(self.agent_user)
        check = (
            f'if [ "$(whoami)" != {quoted_user} ]; then '
            f'echo "Error: running as $(whoami), expected {quoted_user}" >&2; '
            f"exit 1; "
            f"fi"
        )
        # Use explicit profile sourcing instead of ``bash -l`` so that
        # stderr noise from login scripts (e.g. D-Bus errors from
        # /etc/profile.d/sommelier.sh on headless systems) is suppressed.
        profile_source = (
            "{ . /etc/profile 2>/dev/null; "
            "[ -f ~/.bash_profile ] && . ~/.bash_profile 2>/dev/null "
            "|| { [ -f ~/.bash_login ] && . ~/.bash_login 2>/dev/null "
            "|| { [ -f ~/.profile ] && . ~/.profile 2>/dev/null; }; }; } 2>/dev/null; "
        )
        # `sudo` resets the working directory to the target user's home
        # (unless sudoers `cwd=*` is configured), so passing `cwd=` to
        # subprocess.run() only affects the outer sudo process, not the
        # bash that ultimately runs the command.  Embed `cd` in the
        # wrapped script so callers' `cwd` argument is honoured for the
        # actual command.  Using `cd ... || exit 1` rather than `&&` so
        # the failure mode is loud (non-zero exit) instead of silent
        # (running the command in the wrong directory).
        cd_clause = f"cd {shlex.quote(cwd)} || exit 1; " if cwd else ""
        script = f"{profile_source}{check}; {cd_clause}umask {umask_value}; {command_str}"
        command: List[str] = ["bash", "-c", script]

        if env:
            env_args = ["env"] + [f"{key}={value}" for key, value in env.items()]
            command = env_args + command

        return ["sudo", "-u", self.agent_user] + command


@dataclass
class RemoteExecutor(CommandExecutor):
    """Execute agent commands on a remote host over SSH.

    ``run_agent`` wraps commands in ``ssh -J <gateway> <login_node>``.
    ``run_human`` is inherited unchanged and runs locally.

    No sudo is needed on the remote side — the SSH connection
    authenticates as the same user.
    """

    gateway: str = ""
    login_node: str = ""
    remote_mirror_root: str = "~/mirrors"
    local_mirror_root: str = ""
    ssh_options: Dict[str, str] = field(default_factory=dict)
    control_socket_path: Optional[str] = None  # Path to ControlMaster socket
    is_compute_node: bool = False                 # Target is a SLURM compute node
    slurm_job_id: Optional[int] = None          # Active SLURM allocation, if any
    local_disk_root: Optional[str] = None       # Local-disk tiering root (e.g. /local);
                                                # None = agent works on the shared mirror
    proxy_node: str = ""                         # Login node hostname (compute-node proxy)
    proxy_socket_path: Optional[str] = None      # Login node ControlMaster socket
    scaffolding_node: str = ""                   # Login node for filesystem ops
    scaffolding_socket_path: Optional[str] = None  # Its ControlMaster socket
    debug_ssh: bool = False                      # Emit -vvv SSH tracing
    forward_x11: bool = False                    # Request X11 forwarding on
                                                 # interactive (TTY) hops only
    forward_x11_explicit: bool = False           # X11 was explicitly requested
                                                 # (config/CLI) rather than the
                                                 # DISPLAY-present default; gates
                                                 # the speculative srun --x11
    _scaffolding_unreachable: bool = field(default=False, init=False, repr=False)

    # Default timeout (seconds) for remote SSH commands.  Long enough
    # for normal Lustre latency, short enough to surface hangs.
    DEFAULT_SSH_TIMEOUT: int = 120

    # Fail-fast guards for one-shot (non-interactive) SSH commands.
    # These are deliberately *tighter* than the ControlMaster's own
    # keepalive (which wants to linger for days, see RemoteConfig):
    # a one-shot command is already bounded by DEFAULT_SSH_TIMEOUT, so
    # when the master has died and ssh falls back to a fresh dial we want
    # a wedged *transport* (dead node, half-open connection) surfaced as
    # a clean SSH error (exit 255) well inside that ceiling instead of
    # burning the full 120s.  ServerAlive* only tears down a genuinely
    # dead link; a slow-but-alive remote command (e.g. a Lustre stat that
    # blocks but whose sshd still answers keepalive probes) still rides to
    # the subprocess timeout, which is the correct outcome.
    # CONNECT_TIMEOUT bounds the initial dial; INTERVAL x COUNT_MAX (45s)
    # bounds an established-but-silent link.  Both sit under 120s.
    CONNECT_TIMEOUT: int = 10
    KEEPALIVE_INTERVAL: int = 15
    KEEPALIVE_COUNT_MAX: int = 3

    def run_agent(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        umask: Optional[int] = None,
        capture_output: bool = True,
        timeout: Optional[int] = None,
        input: Optional[str] = None,
        show_progress: bool = True,
    ) -> CommandResult:
        """Run a command on the remote login node via SSH.

        ``input``, when set, is piped to the remote command over stdin
        (e.g. ``cat > file``) so large content never appears on a command
        line.  ``show_progress=False`` drops the per-round-trip progress
        line for a caller that repeats one identical probe.
        """
        remote_cwd = self._translate_path(cwd) if cwd else None
        # Allocate a TTY when output isn't captured (interactive agents).
        needs_tty = not capture_output
        ssh_args = self._build_ssh_command(
            args, cwd=remote_cwd, env=env, allocate_tty=needs_tty,
        )
        # Apply a default timeout for captured (non-interactive) SSH
        # commands so Lustre hangs don't block forever.
        effective_timeout = timeout
        if effective_timeout is None and capture_output:
            effective_timeout = self.DEFAULT_SSH_TIMEOUT
        try:
            result = self._run(
                ssh_args,
                check=check,
                cwd=None,           # SSH itself runs locally
                env=None,           # env is embedded in the remote command
                as_agent=False,     # no sudo wrapping
                capture_output=capture_output,
                timeout=effective_timeout,
                input=input,
                show_progress=show_progress,
                progress_command=_format_display(args),
            )
            if self.debug_ssh and result.stderr:
                self.logger.debug(
                    "SSH debug (%s):\n%s",
                    _format_display(args),
                    result.stderr.rstrip(),
                )
            return result
        except CommandError as exc:
            if self.debug_ssh and exc.result.stderr:
                self.logger.error(
                    "SSH debug (FAILED %s):\n%s",
                    _format_display(args),
                    exc.result.stderr.rstrip(),
                )
            if exc.result.returncode == 255:
                self._enrich_ssh_error(exc)
            raise

    def _enrich_ssh_error(self, exc: CommandError) -> None:
        """Add diagnostic hints to SSH connection failures (exit 255)."""
        import os as _os
        hints: List[str] = []
        sock = _os.environ.get("SSH_AUTH_SOCK")
        if not sock:
            hints.append(
                "SSH_AUTH_SOCK is not set --- ssh-agent may not be running.  "
                "Try: eval $(ssh-agent) && ssh-add"
            )
        elif not _os.path.exists(sock):
            hints.append(
                f"SSH_AUTH_SOCK points to {sock} which does not exist.  "
                "The ssh-agent may have died.  Try: eval $(ssh-agent) && ssh-add"
            )
        if self.is_compute_node:
            hints.append(
                f"Target is compute node {self.login_node} --- "
                "verify the SLURM allocation is still RUNNING "
                f"(job {self.slurm_job_id})."
            )
        if self.control_socket_path:
            from pathlib import Path as _Path
            if not _Path(self.control_socket_path).exists():
                hints.append(
                    f"ControlMaster socket {self.control_socket_path} "
                    "does not exist --- the connection has expired."
                )
        if hints:
            detail = "\n  - ".join([""] + hints)
            self.logger.error("SSH connection diagnostic:%s", detail)

    # Remote exit status of coreutils ``timeout`` when it had to stop the
    # command (see ``_build_login_node_command``).
    _REMOTE_TIMEOUT_RETURNCODE = 124

    def _remote_time_limit(self, timeout: int) -> int:
        """Seconds the remote command may run before it is told to stop.

        Short of our own ``timeout`` so the remote side stops the command
        -- and git, on SIGTERM, removes its ``.git/*.lock`` -- before we
        stop waiting for it.
        """
        return max(timeout - 15, timeout // 2, 1)

    def run_on_login_node(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> CommandResult:
        """Run a command on the scaffolding node (the login node).

        Filesystem scaffolding (mkdir, git init, git reset, etc.) on the
        shared mirror runs on the login node, which sees the same shared
        filesystem as the compute node and spares a compute-node target
        the three-hop chain.  Without a separate scaffolding node this is
        ``run_agent``.

        Only an SSH connection failure (exit 255: the command most likely
        never started) falls back to ``run_agent``, and only when that is
        a different host.  A timeout does not: the command may still be
        running remotely, holding e.g. ``.git/index.lock``, and running it
        again elsewhere would collide with it.  The remote command is
        stopped shortly before our own timeout (exit 124), so a timeout
        normally ends it cleanly.
        """
        scaffolding = (
            self.scaffolding_node and self.scaffolding_socket_path
            and not self._scaffolding_unreachable
        )
        if not scaffolding:
            return self.run_agent(
                args, check=check, cwd=cwd, env=env, timeout=timeout,
            )
        remote_cwd = self._translate_path(cwd) if cwd else None
        effective_timeout = timeout if timeout is not None else self.DEFAULT_SSH_TIMEOUT
        time_limit = self._remote_time_limit(effective_timeout)
        ssh_args = self._build_login_node_command(
            args, cwd=remote_cwd, env=env, time_limit=time_limit,
        )
        try:
            result = self._run(
                ssh_args,
                check=check,
                cwd=None,
                env=None,
                as_agent=False,
                capture_output=True,
                timeout=effective_timeout,
                progress_command=_format_display(args),
            )
            if self.debug_ssh and result.stderr:
                self.logger.debug(
                    "SSH debug (login-node %s):\n%s",
                    _format_display(args),
                    result.stderr.rstrip(),
                )
            if result.returncode == self._REMOTE_TIMEOUT_RETURNCODE:
                self._log_remote_timeout(args, time_limit)
            return result
        except CommandError as exc:
            if self.debug_ssh and exc.result.stderr:
                self.logger.error(
                    "SSH debug (login-node FAILED %s):\n%s",
                    _format_display(args),
                    exc.result.stderr.rstrip(),
                )
            rc = exc.result.returncode
            if rc == self._REMOTE_TIMEOUT_RETURNCODE:
                self._log_remote_timeout(args, time_limit)
            elif rc == -1:
                self.logger.error(
                    "%s did not finish %s within %ss and could not be "
                    "confirmed stopped; it may still be running there (a git "
                    "command holds .git/index.lock until it exits).  Not "
                    "retrying it on another node.",
                    self.scaffolding_node, summarize_command(_format_display(args)),
                    effective_timeout,
                )
            elif rc == 255 and self.scaffolding_node != self.login_node:
                # The connection failed, so the command most likely never
                # started.  Stop trying this node for the rest of the run
                # rather than paying for the failure on every command.
                self._scaffolding_unreachable = True
                self.logger.warning(
                    "Scaffolding node %s unreachable; using %s for the rest "
                    "of this run",
                    self.scaffolding_node, self.login_node,
                )
                return self.run_agent(
                    args, check=check, cwd=cwd, env=env, timeout=timeout,
                )
            raise

    def _log_remote_timeout(self, args: Sequence[str], time_limit: int) -> None:
        self.logger.error(
            "%s: %s was stopped after %ss (remote timeout).  The shared "
            "filesystem may be slow or wedged there.",
            self.scaffolding_node, summarize_command(_format_display(args)),
            time_limit,
        )

    def _build_login_node_command(
        self,
        args: Sequence[str],
        *,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        time_limit: Optional[int] = None,
    ) -> List[str]:
        """Build an SSH command targeting the scaffolding node.

        Uses ``BatchMode=yes`` so that if the ControlMaster socket is
        dead and the ProxyCommand falls through to a fresh connection,
        SSH fails immediately instead of hanging for a password prompt
        that will never be answered (we're running non-interactively).

        With ``time_limit``, the command runs under coreutils ``timeout``
        where the remote has it: killing our local ssh does not stop the
        remote process, so the remote side has to bound itself.
        """
        from .tunnel import _control_socket_path as _gw_sock

        ssh_cmd: List[str] = ["ssh"]
        if self.debug_ssh:
            ssh_cmd.append("-vvv")
        ssh_cmd.extend([
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={self.scaffolding_socket_path}",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10",
        ])
        # ProxyCommand fallback through gateway in case the socket
        # is stale.
        if self.gateway:
            gw_socket = _gw_sock(self.gateway)
            ssh_cmd.extend([
                "-o",
                f"ProxyCommand=ssh -o ControlMaster=auto "
                f"-o ControlPath={gw_socket} "
                f"-o BatchMode=yes -o ConnectTimeout=10 "
                f"-W %h:%p {self.gateway}",
            ])
        ssh_cmd.append(self.scaffolding_node)

        parts: List[str] = []
        if env:
            for k, v in env.items():
                parts.append(f"export {shlex.quote(k)}={shlex.quote(v)};")
        if cwd:
            if cwd.startswith("~/"):
                cwd_expr = '"$HOME"/' + shlex.quote(cwd[2:])
            elif cwd == "~":
                cwd_expr = '"$HOME"'
            else:
                cwd_expr = shlex.quote(cwd)
            parts.append(f"cd {cwd_expr} &&")
        command = _format_display(args)
        if time_limit is not None:
            # SIGTERM only, never SIGKILL: git removes its lock files on
            # TERM, and a KILL would leave them behind.  The ``if`` (not a
            # ``$(...)`` prefix) keeps this working in shells that do not
            # word-split, and runs the command unbounded where there is no
            # ``timeout``.
            command = (
                f"if command -v timeout >/dev/null 2>&1; "
                f"then timeout -s TERM {int(time_limit)} {command}; "
                f"else {command}; fi"
            )
        parts.append(command)
        ssh_cmd.append(" ".join(parts))
        return ssh_cmd

    def run_agent_interactive(
        self,
        args: Sequence[str],
        *,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> CommandResult:
        """Launch an interactive command on the remote node with TTY."""
        remote_cwd = self._translate_path(cwd) if cwd else None
        ssh_args = self._build_ssh_command(
            args, cwd=remote_cwd, env=env, allocate_tty=True,
        )
        return self._run(
            ssh_args,
            check=False,
            cwd=None,
            env=None,
            as_agent=False,
            capture_output=False,
            progress_command=_format_display(args),
        )

    def _build_ssh_command(
        self,
        args: Sequence[str],
        *,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        allocate_tty: bool = False,
    ) -> List[str]:
        ssh_cmd: List[str] = ["ssh"]
        if self.debug_ssh:
            ssh_cmd.append("-vvv")
        if allocate_tty:
            ssh_cmd.append("-t")
            if self.forward_x11:
                # Interactive hops only: one-shot commands never open X
                # clients, and requesting a forward would just make sshd
                # mint an xauth cookie per command.  Trusted forwarding
                # (-Y semantics) is deliberate -- untrusted X11 breaks
                # many clients and expires after ssh's 20-minute window.
                ssh_cmd.extend([
                    "-o", "ForwardX11=yes",
                    "-o", "ForwardX11Trusted=yes",
                ])
        else:
            # Non-interactive: fail immediately instead of opening
            # /dev/tty for a password prompt that would look like a hang.
            ssh_cmd.extend(["-o", "BatchMode=yes"])
            # Fail-fast guards (one-shot commands only; the interactive
            # path keeps the ControlMaster's longer tolerance so a live
            # tmux session is never torn down).  A dead/wedged transport
            # now surfaces as exit 255 in ~45s instead of eating the full
            # DEFAULT_SSH_TIMEOUT.
            ssh_cmd.extend([
                "-o", f"ConnectTimeout={self.CONNECT_TIMEOUT}",
                "-o", f"ServerAliveInterval={self.KEEPALIVE_INTERVAL}",
                "-o", f"ServerAliveCountMax={self.KEEPALIVE_COUNT_MAX}",
            ])
        # Reuse ControlMaster connection if available (avoids re-auth).
        # Include a ProxyCommand fallback so that a stale socket doesn't
        # leave SSH with no route to an unresolvable internal hostname.
        #
        # Login-node targets: proxy through the gateway.
        # Compute-node targets: proxy through the login node (the
        #   gateway cannot reach compute nodes, but the login node can).
        if self.control_socket_path:
            ssh_cmd.extend([
                "-o", "ControlMaster=auto",
                "-o", f"ControlPath={self.control_socket_path}",
            ])
            if self.is_compute_node and self.proxy_node and self.proxy_socket_path:
                # Compute nodes are ephemeral with rotating host keys.
                ssh_cmd.extend([
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                ])
                # Route through the login node's ControlMaster.
                # One-shot hops also bound the inner dial so a wedged
                # proxy fails fast rather than hanging the whole command.
                proxy_batch = (
                    "" if allocate_tty
                    else f" -o BatchMode=yes -o ConnectTimeout={self.CONNECT_TIMEOUT}"
                )
                ssh_cmd.extend([
                    "-o",
                    f"ProxyCommand=ssh -o ControlMaster=auto "
                    f"-o ControlPath={self.proxy_socket_path}"
                    f"{proxy_batch} "
                    f"-W %h:%p {self.proxy_node}",
                ])
            elif self.gateway and not self.is_compute_node:
                from .tunnel import _control_socket_path as _gw_sock
                gw_socket = _gw_sock(self.gateway)
                # One-shot hops also bound the inner dial so a wedged
                # proxy fails fast rather than hanging the whole command.
                proxy_batch = (
                    "" if allocate_tty
                    else f" -o BatchMode=yes -o ConnectTimeout={self.CONNECT_TIMEOUT}"
                )
                ssh_cmd.extend([
                    "-o",
                    f"ProxyCommand=ssh -o ControlMaster=auto "
                    f"-o ControlPath={gw_socket}"
                    f"{proxy_batch} "
                    f"-W %h:%p {self.gateway}",
                ])
        for key, val in self.ssh_options.items():
            ssh_cmd.extend(["-o", f"{key}={val}"])
        if self.control_socket_path or not self.gateway:
            ssh_cmd.append(self.login_node)
        else:
            ssh_cmd.extend(["-J", self.gateway, self.login_node])

        # Build the remote shell command as a single string.
        parts: List[str] = []
        if env:
            for k, v in env.items():
                parts.append(f"export {shlex.quote(k)}={shlex.quote(v)};")
        if cwd:
            # Replace leading ~ with $HOME so bash expands it on the remote.
            # shlex.quote would wrap ~ in single quotes, preventing expansion.
            if cwd.startswith("~/"):
                cwd_expr = '"$HOME"/' + shlex.quote(cwd[2:])
            elif cwd == "~":
                cwd_expr = '"$HOME"'
            else:
                cwd_expr = shlex.quote(cwd)
            parts.append(f"cd {cwd_expr} &&")
        parts.append(_format_display(args))

        ssh_cmd.append(" ".join(parts))
        return ssh_cmd

    def _translate_path(self, local_path: str) -> str:
        """Rewrite a local mirror path to its remote equivalent.

        If the path starts with the local mirror root, the prefix is
        replaced with the remote mirror root.  Otherwise the path is
        returned as-is.
        """
        if self.local_mirror_root and local_path.startswith(self.local_mirror_root):
            suffix = local_path[len(self.local_mirror_root):]
            return self.remote_mirror_root.rstrip("/") + suffix
        return local_path
