"""Launcher-side helpers for managed Codex state (docs/codex-local-state.org).

The runner that owns the Codex process on the node is
``codex_state_runner.py``, staged whole like the native context installer.
This module prepares what the launcher must decide before anything runs: the
stable scope digest, the ``sqlite_home`` placeholder inserted at the argv
stage (before any nvm or installer wrapping, so it survives shell quoting),
and the outermost runner command.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Sequence

from .agent_mode import config_override

PLACEHOLDER = "__SUCODER_CODEX_STATE_DB__"
RUNNER_SOURCE = Path(__file__).with_name("codex_state_runner.py")


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


def runner_command(runner_path: str, scope: str, command: Sequence[str], python: str = "python3") -> List[str]:
    """The outermost command: the staged runner owning everything after ``--``."""
    return [python, runner_path, "--scope", scope, "--placeholder", PLACEHOLDER, "--", *command]


def runner_source() -> str:
    """The runner script to stage, byte for byte."""
    return RUNNER_SOURCE.read_text(encoding="utf-8")
