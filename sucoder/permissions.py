"""Filesystem permission utilities."""

from __future__ import annotations

import logging
from pathlib import Path

from .executor import CommandExecutor

logger = logging.getLogger(__name__)


def ensure_directory(path: Path) -> None:
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def apply_agent_repo_permissions(
    executor: CommandExecutor,
    repo_path: Path,
    *,
    agent_group: str,
) -> None:
    """Grant group write access and setgid on directories for agent-owned repositories."""
    repo = str(repo_path)
    agent_user = executor.agent_user

    # Detect foreign-owned entries so we can warn instead of failing outright.
    foreign = executor.run_agent(
        [
            "find",
            repo,
            "-not",
            "-user",
            agent_user,
            "-print",
            "-quit",
        ],
        check=False,
    )
    skipped_example = foreign.stdout.strip()
    if skipped_example:
        executor.logger.warning(
            "Skipping permission adjustments for entries not owned by %s (example: %s).",
            agent_user,
            skipped_example,
        )

    result = executor.run_agent(
        [
            "find",
            repo,
            "-user",
            agent_user,
            "-exec",
            "chgrp",
            "-h",  # Don't follow symlinks
            agent_group,
            "{}",
            "+",
        ],
        check=False,  # Don't fail on permission denied errors
    )
    if result.returncode != 0:
        logger.warning(
            "Permission adjustment (chgrp) failed (exit %d): %s",
            result.returncode,
            result.stderr.strip() if result.stderr else "(no stderr)",
        )

    result = executor.run_agent(
        [
            "find",
            repo,
            "-user",
            agent_user,
            "-exec",
            "chmod",
            "g+rwX",
            "{}",
            "+",
        ],
        check=False,  # Don't fail on permission denied errors
    )
    if result.returncode != 0:
        logger.warning(
            "Permission adjustment (chmod g+rwX) failed (exit %d): %s",
            result.returncode,
            result.stderr.strip() if result.stderr else "(no stderr)",
        )

    result = executor.run_agent(
        [
            "find",
            repo,
            "-type",
            "d",
            "-user",
            agent_user,
            "-exec",
            "chmod",
            "g+s",
            "{}",
            "+",
        ],
        check=False,  # Don't fail on permission denied errors
    )
    if result.returncode != 0:
        logger.warning(
            "Permission adjustment (chmod g+s) failed (exit %d): %s",
            result.returncode,
            result.stderr.strip() if result.stderr else "(no stderr)",
        )


def ensure_directory_mode(
    executor: CommandExecutor,
    path: Path,
    mode: str,
    *,
    as_agent: bool = False,
) -> None:
    """Ensure a directory has the desired mode (e.g., 2770)."""
    runner = executor.run_agent if as_agent else executor.run_human
    runner(
        ["chmod", mode, str(path)],
        check=True,
    )
