"""Launcher-side helpers for managed Codex state (docs/codex-local-state.org).

The runner that owns the Codex process on the node is
``codex_state_runner.py``, staged whole like the native context installer.
This module prepares what the launcher must decide before anything runs: the
stable scope digest, the ``sqlite_home`` placeholder inserted at the argv
stage (before any nvm or installer wrapping, so it survives shell quoting),
the durable root and its refusals, the history selector, and the outermost
runner command.
"""
from __future__ import annotations

import hashlib
import json
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from .agent_mode import config_override

PLACEHOLDER = "__SUCODER_CODEX_STATE_DB__"
RUNNER_SOURCE = Path(__file__).with_name("codex_state_runner.py")
DURABLE_ROOT_NAME = ".sucoder-state"
# ``RUN/GENERATION``: a run is a 32-hex UUID, a generation a six-digit
# sequence plus an eight-hex random suffix.  The selector names history and
# nothing else; it never carries a path or shell text.
RESTORE_SELECTOR = re.compile(r"[0-9a-f]{32}/[0-9]{6}-[0-9a-f]{8}")


@dataclass(frozen=True)
class CodexStateLaunch:
    """What the launcher decides once per managed launch and hands the runner."""

    scope: str
    durable_root: str
    checkpoint_minutes: int
    checkpoint_timeout_seconds: int
    restore: Optional[str] = None   # ``RUN/GENERATION`` to continue from
    fresh: bool = False             # start a new independent history


def scope_id(cluster: str, user: str, mirror_path: str, slot: str = "default") -> str:
    """SHA-256 of the canonical, unsanitized identity tuple (design: Identities)."""
    canonical = json.dumps([cluster, user, mirror_path, slot], ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def with_sqlite_home_placeholder(command: Sequence[str]) -> List[str]:
    """``codex -c sqlite_home="<placeholder>" ...``, refusing an explicit sqlite_home."""
    if not command:
        raise ValueError("empty command")
    for arg in command:
        if "sqlite_home=" in arg:
            raise ValueError(
                "The Codex command already sets sqlite_home; managed state placement "
                "refuses to override an explicit choice. Remove it or disable codex_state."
            )
    return [command[0], "-c", config_override("sqlite_home", PLACEHOLDER), *command[1:]]


def validate_restore_selector(value: str) -> str:
    """``RUN/GENERATION`` exactly, or ValueError; the runner resolves it."""
    if not isinstance(value, str) or not RESTORE_SELECTOR.fullmatch(value):
        raise ValueError(
            f"Invalid Codex state selector {value!r}: expected RUN/GENERATION, where RUN "
            "is 32 hex characters and GENERATION is six digits, a dash and 8 hex "
            "characters (as listed under the durable root's runs/ directory)."
        )
    return value


def _normalize(path: str) -> str:
    """``normpath`` with any run of leading slashes collapsed to one.

    POSIX lets ``//x`` differ from ``/x``, so ``normpath`` preserves exactly
    two leading slashes; without this ``//g/mirrors/sample`` compares unequal
    to ``/g/mirrors/sample`` and slips past every refusal below.  Every path
    the refusals compare goes through here.  No realpath: symlinks are the
    operator's choice (``~/mirrors`` onto Lustre).
    """
    return posixpath.normpath(re.sub(r"^/+", "/", path))


def durable_root_for(mirror_path: str, configured: Optional[str]) -> str:
    """The durable root: configured, else ``<dir containing the mirrors>/.sucoder-state``.

    ``mirror_path`` is the resolved shared mirror path (``~`` expanded on the
    target, not realpath'd: on Savio ``~/mirrors`` is a symlink onto Lustre,
    and the sibling must live there too).  The runner appends
    ``codex/<scope>`` itself.
    """
    if configured:
        return _normalize(configured)
    return posixpath.join(posixpath.dirname(_normalize(mirror_path)), DURABLE_ROOT_NAME)


def _within(path: str, parent: str) -> bool:
    parent = _normalize(parent)
    return path == parent or path.startswith(parent.rstrip("/") + "/")


def check_durable_root(durable_root: str, mirror_path: str, local_disk_root: Optional[str]) -> str:
    """Refuse a root that would nest with the mirror repository or sit on local disk.

    The state must not live inside the shared mirror repository (it would be
    part of the working tree), the repository must not live inside the state
    root (publication and retention walk that tree), and allocation-local
    storage is not durable.  Returns the normalized root.
    """
    root = _normalize(durable_root)
    mirror = _normalize(mirror_path)
    if root == mirror:
        raise ValueError(
            f"slurm.codex_state.durable_root {durable_root!r} is the shared mirror "
            f"repository itself; the state root must be outside every mirror."
        )
    if _within(root, mirror):
        raise ValueError(
            f"slurm.codex_state.durable_root {durable_root!r} is inside the shared mirror "
            f"repository {mirror}; the state root must be outside every mirror."
        )
    if _within(mirror, root):
        raise ValueError(
            f"slurm.codex_state.durable_root {durable_root!r} contains the shared mirror "
            f"repository {mirror}; the state root must not enclose a repository."
        )
    if local_disk_root and _within(root, local_disk_root):
        raise ValueError(
            f"slurm.codex_state.durable_root {durable_root!r} is under the node-local "
            f"root {local_disk_root}; allocation-local storage is not durable."
        )
    return root


def runner_command(
    runner_path: str,
    scope: str,
    command: Sequence[str],
    *,
    durable_root: str,
    checkpoint_minutes: int,
    checkpoint_timeout_seconds: int,
    restore: Optional[str] = None,
    fresh: bool = False,
    python: str = "python3",
) -> List[str]:
    """The outermost command: the staged runner owning everything after ``--``.

    ``restore`` (``RUN/GENERATION``) and ``fresh`` are mutually exclusive;
    without either the runner reuses an intact run already on this
    allocation's local disk, else starts a fresh history; nothing is
    selected across allocations automatically (that is Slice 3).
    """
    if restore is not None and fresh:
        raise ValueError("restore and fresh are mutually exclusive")
    argv = [
        python, runner_path,
        "--scope", scope,
        "--placeholder", PLACEHOLDER,
        "--durable-root", durable_root,
        "--checkpoint-minutes", str(int(checkpoint_minutes)),
        "--checkpoint-timeout", str(int(checkpoint_timeout_seconds)),
    ]
    if restore is not None:
        argv += ["--restore", validate_restore_selector(restore)]
    elif fresh:
        argv.append("--fresh")
    return [*argv, "--", *command]


def runner_source() -> str:
    """The runner script to stage, byte for byte."""
    return RUNNER_SOURCE.read_text(encoding="utf-8")
