"""Command execution helpers shared across the toolkit."""

from __future__ import annotations

import logging
import shlex
import subprocess
from dataclasses import dataclass
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

        check = (
            f'if [ "$(whoami)" != "{self.agent_user}" ]; then '
            f'echo "Error: running as $(whoami), expected {self.agent_user}" >&2; '
            f"exit 1; "
            f"fi"
        )
        script = f"{check}; umask {umask_value}; {command_str}"
        command: List[str] = ["bash", "-lc", script]

        if env:
            env_args = ["env"] + [f"{key}={value}" for key, value in env.items()]
            command = env_args + command

        return ["sudo", "-u", self.agent_user] + command
