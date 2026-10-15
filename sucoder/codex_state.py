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
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .agent_mode import config_override

PLACEHOLDER = "__SUCODER_CODEX_STATE_DB__"
RUNNER_SOURCE = Path(__file__).with_name("codex_state_runner.py")
DURABLE_ROOT_NAME = ".sucoder-state"
# ``RUN/GENERATION``: a run is a 32-hex UUID, a generation a six-digit
# sequence plus an eight-hex random suffix.  The selector names history and
# nothing else; it never carries a path or shell text.
RESTORE_SELECTOR = re.compile(r"[0-9a-f]{32}/[0-9]{6}-[0-9a-f]{8}")
RUN_ID = re.compile(r"[0-9a-f]{32}")


@dataclass(frozen=True)
class CodexStateLaunch:
    """What the launcher decides once per managed launch and hands the runner."""

    scope: str
    durable_root: str
    checkpoint_minutes: int
    checkpoint_timeout_seconds: int
    restore: Optional[str] = None   # ``RUN/GENERATION`` to continue from
    fresh: bool = False             # start a new independent history

    @property
    def selection(self) -> Optional[str]:
        """The history selection as the operator spelled it, or None.

        A selection rides in the argv of the command that starts the runner,
        so a launch that would attach to a running session instead refuses
        and names it with this text.
        """
        if self.restore is not None:
            return f"--codex-state-from {self.restore}"
        return "--codex-state-fresh" if self.fresh else None


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


# ----------------------------------------------------------------- operator status

STATUS_FILES = ("status.json", "owner.json")


def local_scope_dir(local_disk_root: str, job_id: int, scope: str) -> str:
    """``<root>/job<ID>/state/codex/<scope>``: the runner's scope directory on the node.

    The allocation-local root is the local-tier layout's ``<root>/job<ID>``
    (``local_tier.local_root_shell`` spells the same path as a shell word);
    the runner keeps ``latest-run`` and one directory per run under
    ``state/codex/<scope>`` below it.
    """
    root = local_disk_root.rstrip("/") or "/"
    return posixpath.join(root, f"job{job_id}", "state", "codex", scope)


@dataclass(frozen=True)
class CodexStateStatus:
    """What ``sucoder codex-state-status`` found for one allocation.

    ``run`` is None when ``latest-run`` is absent: a refused or never-placed
    launch does not write it, so there is no managed run to report.
    ``commands`` lists every read sent, or on a dry run the reads that would
    be sent (the run id is unknown there, so ``<run>`` stands in for it).
    """

    job_id: int
    scope: str
    scope_dir: str
    commands: List[List[str]]
    run: Optional[str] = None
    status: Optional[Dict[str, Any]] = None
    owner: Optional[Dict[str, Any]] = None
    dry_run: bool = False

    @property
    def run_dir(self) -> Optional[str]:
        return None if self.run is None else posixpath.join(self.scope_dir, self.run)


def _seconds(value: Any) -> str:
    """``12.3s`` for a number; ``unknown`` for None (never measured)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "unknown"
    return f"{value:g}s"


def _name_of(record: Any) -> Optional[str]:
    """The ``name`` of a published-generation record, or None."""
    if isinstance(record, Mapping) and isinstance(record.get("name"), str):
        return str(record["name"])
    return None


def describe_pending_work(pending: Any) -> str:
    """Active goals and queued items as the runner reports them, or ``none``.

    Explicit thread ids and counts: an active goal drives unprompted model
    requests the moment its thread resumes and a queued item executes once,
    so the operator decides per thread (pause the goal in Codex) and must
    never read a summary that hides one.  Mirrors the runner's own wording
    without importing it (the staged runner is Python 3.6 code).
    """
    if not isinstance(pending, Mapping):
        return "not reported"
    parts: List[str] = []
    goals = pending.get("active_goals") or []
    if goals:
        parts.append(f"active goal on thread(s) {', '.join(str(t) for t in goals)}")
    queued = pending.get("queued") or {}
    if isinstance(queued, Mapping):
        for tid, count in sorted(queued.items()):
            parts.append(f"{count} queued item(s) on thread {tid}")
    for note in pending.get("unreadable") or []:
        parts.append(f"a store the runner cannot read ({note})")
    return "; ".join(parts) if parts else "none"


def format_status(status: Mapping[str, Any], *, run: str, owner: Optional[Mapping[str, Any]] = None) -> str:
    """The operator's one-line-per-fact summary of a run's ``status.json``.

    Every key is optional: a launch refused or blocked before its
    ``starting`` transition has written only ``state``, ``reason`` and the
    transitions.  ``run`` comes from ``latest-run``, which the caller has
    already read, so the first line always names it.  The numbers an
    operator acts on (durable age, the unsynced generation, replica lag,
    pending work) are printed as the runner recorded them, never rolled up.
    """
    owner = owner or {}
    lines: List[str] = []
    mode = status.get("mode") or ("restore" if status.get("restore") else None) or (
        "reuse" if owner.get("reused_run") else None) or "unknown"
    lines.append(f"mode: {mode}  run: {run}")

    state = status.get("state") or "unknown"
    detail = ""
    if state == "exited":
        detail = f" (exit {status.get('exit')}, {status.get('outcome') or 'unknown'})"
    elif state in ("misplaced", "stopped_misplaced"):
        detail = f" (misplaced: {', '.join(status.get('misplaced') or []) or 'unknown'})"
    elif state == "unverified" and status.get("note"):
        detail = f" ({status['note']})"
    lines.append(f"state: {state}{detail}")

    replica = status.get("replica")
    if isinstance(replica, Mapping):
        threads = replica.get("threads") or {}
        text = (f"replica: age {_seconds(replica.get('age_seconds'))}, lag {replica.get('lag_bytes', 'unknown')} bytes, "
                f"{len(threads)} thread(s)")
        for key in ("anomalies", "errors"):
            found = replica.get(key)
            if isinstance(found, Mapping) and found:
                text += f"; {key}: " + ", ".join(f"{tid}: {why}" for tid, why in sorted(found.items()))
        lines.append(text)
    else:
        lines.append("replica: not reported yet")

    last = _name_of(status.get("last_generation"))
    if last is not None:
        record = status["last_generation"]
        text = f"last durable generation: {last} ({record.get('class') or 'unknown'}, age {_seconds(status.get('durable_age_seconds'))})"
    else:
        text = "last durable generation: none"
    unsynced = _name_of(status.get("unsynced_generation"))
    if unsynced is not None:
        text += f"; unsynced: {unsynced}"
    lines.append(text)

    checkpoint = status.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        text = (f"checkpoint: {checkpoint.get('state') or 'unknown'}, last outcome {checkpoint.get('last_outcome') or 'none'}, "
                f"{checkpoint.get('dropped', 0)} dropped")
        flight = checkpoint.get("in_flight")
        if isinstance(flight, Mapping):
            text += f"; in flight: {flight.get('generation')} ({flight.get('trigger')})"
        if checkpoint.get("reason"):
            text += f"; reason: {checkpoint['reason']}"
        lines.append(text)
    else:
        lines.append("checkpoint: not reported yet")

    pending = status.get("pending_work")
    if not isinstance(pending, Mapping):
        pending = owner.get("pending_work")
    lines.append(f"pending work: {describe_pending_work(pending)}")

    restored = status.get("restored_from")
    if not isinstance(restored, Mapping):
        restored = owner.get("restored_from")
    if isinstance(restored, Mapping):
        text = f"restored from: {restored.get('run')}/{restored.get('generation')} ({restored.get('class') or 'unknown'}"
        if restored.get("captured_at_iso"):
            text += f", captured {restored['captured_at_iso']}"
        lines.append(text + ")")
    if status.get("restore_ignored"):
        lines.append(f"restore ignored: {status['restore_ignored']} (this allocation's run was reused instead)")

    durable = status.get("durable") or owner.get("durable")
    if durable:
        lines.append(f"durable run: {durable}")
    if status.get("reason"):
        lines.append(f"reason: {status['reason']}")
    return "\n".join(lines)
