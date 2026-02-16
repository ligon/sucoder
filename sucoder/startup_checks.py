"""Startup validation for sucoder."""

from __future__ import annotations

import logging
import pwd
from pathlib import Path
from typing import Optional

from .config import Config
from .executor import CommandExecutor


class StartupError(RuntimeError):
    """Raised when startup validation fails."""


def run_startup_checks(
    config: Config,
    config_path: Path,
    *,
    logger: Optional[logging.Logger] = None,
    use_sudo: bool = True,
) -> None:
    """Validate runtime assumptions before executing commands."""
    logger = logger or logging.getLogger("sucoder.startup")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    _ensure_agent_account(config, logger)
    _ensure_config_access(config, config_path, logger, use_sudo=use_sudo)


def _ensure_agent_account(config: Config, logger: logging.Logger) -> None:
    """Ensure the configured agent account exists and has a home directory."""
    try:
        info = pwd.getpwnam(config.agent_user)
    except KeyError as exc:
        raise StartupError(f"Agent user {config.agent_user!r} not found.") from exc

    home = Path(info.pw_dir)
    if not home.exists():
        raise StartupError(f"Home directory {home} for agent {config.agent_user!r} does not exist.")

    logger.debug("Agent user %s found with home %s", config.agent_user, home)


def _ensure_config_access(
    config: Config,
    config_path: Path,
    logger: logging.Logger,
    *,
    use_sudo: bool,
) -> None:
    """Ensure the agent can read but not write the configuration file."""
    if not config_path.exists():
        raise StartupError(f"Configuration file {config_path} does not exist.")

    executor = CommandExecutor(
        human_user=config.human_user,
        agent_user=config.agent_user,
        agent_group=config.agent_group,
        logger=logger,
        dry_run=False,
        use_sudo_for_agent=use_sudo,
    )

    read_result = executor.run_agent(
        ["test", "-r", str(config_path)],
        check=False,
    )
    if read_result.returncode != 0:
        raise StartupError(
            f"Agent user {config.agent_user!r} cannot read configuration {config_path}."
        )

    write_result = executor.run_agent(
        ["test", "-w", str(config_path)],
        check=False,
    )
    if write_result.returncode == 0:
        raise StartupError(
            f"Configuration file {config_path} is writable by agent user {config.agent_user!r}."
        )

    logger.debug(
        "Configuration %s is readable but not writable by agent %s",
        config_path,
        config.agent_user,
    )
