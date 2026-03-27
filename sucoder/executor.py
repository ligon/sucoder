"""Command execution helpers shared across the toolkit."""

from __future__ import annotations

import logging
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


class CommandError(RuntimeError):
    """Raised when a subprocess finishes with a non-zero exit code."""

    def __init__(self, message: str, result: "CommandResult"):
        super().__init__(message)
        self.result = result


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
    ) -> CommandResult:
        requested_args = list(args)
        executed_args = (
            self._wrap_agent_command(args, env=env, umask=umask)
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
        if timeout is not None:
            run_kwargs["timeout"] = timeout
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
            if result.stderr:
                self.logger.error(result.stderr.strip())
            raise CommandError(message, result)

        return result

    def _wrap_agent_command(
        self,
        args: Sequence[str],
        *,
        env: Optional[Mapping[str, str]],
        umask: Optional[int],
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
        script = f"{check}; umask {umask_value}; {command_str}"
        command: List[str] = ["bash", "-lc", script]

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
    proxy_node: str = ""                         # Login node hostname (compute-node proxy)
    proxy_socket_path: Optional[str] = None      # Login node ControlMaster socket
    scaffolding_node: str = ""                   # DTN (or login node) for filesystem ops
    scaffolding_socket_path: Optional[str] = None  # DTN ControlMaster socket
    debug_ssh: bool = False                      # Emit -vvv SSH tracing

    # Default timeout (seconds) for remote SSH commands.  Long enough
    # for normal Lustre latency, short enough to surface hangs.
    DEFAULT_SSH_TIMEOUT: int = 120

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
    ) -> CommandResult:
        """Run a command on the remote login node via SSH."""
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

    # Return codes that indicate a DTN-level failure (not a command
    # failure) and warrant falling back to the compute node.
    _DTN_FALLBACK_RETURNCODES = frozenset({
        -1,   # our timeout sentinel
        255,  # SSH connection error
    })

    def run_on_login_node(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> CommandResult:
        """Run a command on the scaffolding node (DTN or login node).

        Filesystem scaffolding (mkdir, git init, etc.) runs on the DTN
        (data transfer node) when available, since it has fat pipes and
        spare CPU.  Falls back to the login node, or to ``run_agent``
        for non-compute targets without a separate scaffolding node.

        If the DTN times out or the SSH connection fails, falls back to
        ``run_agent`` (the compute node) since shared FS is visible
        from all cluster nodes.
        """
        scaffolding = self.scaffolding_node and self.scaffolding_socket_path
        if not scaffolding:
            return self.run_agent(
                args, check=check, cwd=cwd, env=env, timeout=timeout,
            )
        remote_cwd = self._translate_path(cwd) if cwd else None
        ssh_args = self._build_login_node_command(
            args, cwd=remote_cwd, env=env,
        )
        effective_timeout = timeout if timeout is not None else self.DEFAULT_SSH_TIMEOUT
        try:
            result = self._run(
                ssh_args,
                check=check,
                cwd=None,
                env=None,
                as_agent=False,
                capture_output=True,
                timeout=effective_timeout,
            )
            if self.debug_ssh and result.stderr:
                self.logger.debug(
                    "SSH debug (login-node %s):\n%s",
                    _format_display(args),
                    result.stderr.rstrip(),
                )
            return result
        except CommandError as exc:
            if self.debug_ssh and exc.result.stderr:
                self.logger.error(
                    "SSH debug (login-node FAILED %s):\n%s",
                    _format_display(args),
                    exc.result.stderr.rstrip(),
                )
            # Timeout or SSH connection failure — the DTN itself is
            # unreachable, not a problem with the command.  Fall back
            # to the compute node (shared FS is visible from there).
            if exc.result.returncode in self._DTN_FALLBACK_RETURNCODES:
                self.logger.warning(
                    "Scaffolding node %s unavailable (rc=%d); "
                    "falling back to compute node",
                    self.scaffolding_node, exc.result.returncode,
                )
                return self.run_agent(
                    args, check=check, cwd=cwd, env=env, timeout=timeout,
                )
            raise

    def _build_login_node_command(
        self,
        args: Sequence[str],
        *,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> List[str]:
        """Build an SSH command targeting the scaffolding node (DTN).

        Uses ``BatchMode=yes`` so that if the ControlMaster socket is
        dead and the ProxyCommand falls through to a fresh connection,
        SSH fails immediately instead of hanging for a password prompt
        that will never be answered (we're running non-interactively).
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
        parts.append(_format_display(args))
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
                ssh_cmd.extend([
                    "-o",
                    f"ProxyCommand=ssh -o ControlMaster=auto "
                    f"-o ControlPath={self.proxy_socket_path} "
                    f"-W %h:%p {self.proxy_node}",
                ])
            elif self.gateway and not self.is_compute_node:
                from .tunnel import _control_socket_path as _gw_sock
                gw_socket = _gw_sock(self.gateway)
                ssh_cmd.extend([
                    "-o",
                    f"ProxyCommand=ssh -o ControlMaster=auto "
                    f"-o ControlPath={gw_socket} "
                    f"-W %h:%p {self.gateway}",
                ])
        for key, val in self.ssh_options.items():
            ssh_cmd.extend(["-o", f"{key}={val}"])
        if self.control_socket_path:
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
