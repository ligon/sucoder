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
    ) -> CommandResult:
        return self._run(
            list(args),
            check=check,
            cwd=cwd,
            env=env,
            as_agent=False,
            capture_output=capture_output,
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
    ) -> CommandResult:
        return self._run(
            list(args),
            check=check,
            cwd=cwd,
            env=env,
            as_agent=True,
            umask=umask or self.default_umask,
            capture_output=capture_output,
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
        result_proc = subprocess.run(
            executed_args,
            check=False,
            **run_kwargs,
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

    def run_agent(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        umask: Optional[int] = None,
        capture_output: bool = True,
    ) -> CommandResult:
        """Run a command on the remote login node via SSH."""
        remote_cwd = self._translate_path(cwd) if cwd else None
        # Allocate a TTY when output isn't captured (interactive agents).
        needs_tty = not capture_output
        ssh_args = self._build_ssh_command(
            args, cwd=remote_cwd, env=env, allocate_tty=needs_tty,
        )
        return self._run(
            ssh_args,
            check=check,
            cwd=None,           # SSH itself runs locally
            env=None,           # env is embedded in the remote command
            as_agent=False,     # no sudo wrapping
            capture_output=capture_output,
        )

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
        if allocate_tty:
            ssh_cmd.append("-t")
        # Reuse ControlMaster connection if available (avoids re-auth).
        # When a ControlMaster socket exists for the login node, it
        # already routes through the gateway — no -J needed.
        if self.control_socket_path:
            ssh_cmd.extend([
                "-o", "ControlMaster=auto",
                "-o", f"ControlPath={self.control_socket_path}",
            ])
            ssh_cmd.append(self.login_node)
        else:
            ssh_cmd.extend(["-J", self.gateway, self.login_node])
        for key, val in self.ssh_options.items():
            ssh_cmd.extend(["-o", f"{key}={val}"])

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
