"""Filesystem permission utilities."""

from __future__ import annotations

import grp
import logging
import os
import pwd
from pathlib import Path
from typing import List, Optional

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


def check_parent_traversable(
    path: Path,
    agent_user: Optional[str] = None,
    agent_group: Optional[str] = None,
) -> List[Path]:
    """Return parent directories of *path* that the agent cannot traverse.

    Walks upward from *path*.parent and checks each directory for execute
    permission accessible to the agent.  A directory is considered
    traversable if:

    * the agent owns it and ``u+x`` is set, **or**
    * the directory's group matches *agent_group* and ``g+x`` is set, **or**
    * ``o+x`` (world-execute) is set.

    Stops at the first ancestor with ``o+x``, since everything above it is
    assumed accessible (``/``, ``/home``, etc. are normally world-executable).
    """
    # Resolve numeric IDs for the agent so we can compare against stat results.
    agent_uid: Optional[int] = None
    agent_gid: Optional[int] = None
    if agent_user:
        try:
            agent_uid = pwd.getpwnam(agent_user).pw_uid
        except KeyError:
            pass
    if agent_group:
        try:
            agent_gid = grp.getgrnam(agent_group).gr_gid
        except KeyError:
            pass

    blocking: List[Path] = []
    for parent in path.parents:
        try:
            st = parent.stat()
        except PermissionError:
            blocking.append(parent)
            continue

        mode = st.st_mode

        # World-executable — this dir and everything above is fine.
        if mode & 0o001:
            break

        # Check whether the agent can traverse via owner or group bits.
        can_traverse = False
        if agent_uid is not None and st.st_uid == agent_uid and mode & 0o100:
            can_traverse = True
        elif agent_gid is not None and st.st_gid == agent_gid and mode & 0o010:
            can_traverse = True

        if not can_traverse:
            blocking.append(parent)

    return blocking


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
