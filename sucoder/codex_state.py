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
import secrets
import shlex
from dataclasses import dataclass, field
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
    unattended: bool = False        # no client attached: sealed-only restore, pending work refuses
    temporary_local_root: bool = False  # remote control without a local-tier clone
    remote_control: bool = False        # private installation identity as well as SQLite

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


def app_server_command(command: Sequence[str]) -> List[str]:
    """The bare ``codex app-server`` the runner runs before a restored launch.

    ``command`` is the launch's Codex argv right after
    :func:`with_sqlite_home_placeholder` (``codex -c sqlite_home=... ...``).
    The result keeps the executable and the placeholder and carries the
    launch's global config overrides (``-c``/``--config``, ``--enable``,
    ``--disable``), because ``thread/resume`` loads the configuration the
    thread was recorded with (its model provider, for one) and fails when an
    override that defined it is missing; subcommands, prompts and other flags
    are not app-server options and stay out.
    """
    if len(command) < 3 or PLACEHOLDER not in command[2]:
        raise ValueError("the command does not start with the sqlite_home placeholder")
    carried: List[str] = []
    rest = list(command[3:])
    i = 0
    while i < len(rest):
        arg = rest[i]
        if arg in ("-c", "--config", "--enable", "--disable") and i + 1 < len(rest):
            carried += [arg, rest[i + 1]]
            i += 2
            continue
        if arg.startswith(("--config=", "--enable=", "--disable=")):
            carried.append(arg)
        i += 1
    return [*command[:3], *carried, "app-server"]


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
    unattended: bool = False,
    app_server: Optional[Sequence[str]] = None,
    temporary_local_root: bool = False,
    remote_helper: Optional[str] = None,
    publish_run: bool = False,
    python: str = "python3",
) -> List[str]:
    """The outermost command: the staged runner owning everything after ``--``.

    ``restore`` (``RUN/GENERATION``) and ``fresh`` are mutually exclusive;
    without either the runner reuses an intact run already on this
    allocation's local disk, else starts a fresh history.  ``unattended``
    tells the runner nobody is watching (a renewal's detached replacement):
    a restore then accepts sealed generations only and pending work in the
    stores refuses rather than resumes.  Nothing is selected across
    allocations automatically; a renewal passes the generation its
    predecessor's drain acknowledgement named.

    ``app_server`` is the Codex app-server command (wrapped the way the
    launch's own command is, carrying the sqlite_home placeholder once).  On a
    restore or reuse the runner runs it once, before Codex starts, to move
    threads recorded in another allocation's clone of this mirror into this
    launch's clone through ``thread/resume`` (issue 46).
    """
    if restore is not None and fresh:
        raise ValueError("restore and fresh are mutually exclusive")
    if app_server is not None and sum(PLACEHOLDER in arg for arg in app_server) != 1:
        raise ValueError("the app-server command must carry the sqlite_home placeholder exactly once")
    argv = [
        python, runner_path,
        "--scope", scope,
        "--placeholder", PLACEHOLDER,
        "--durable-root", durable_root,
        "--checkpoint-minutes", str(int(checkpoint_minutes)),
        "--checkpoint-timeout", str(int(checkpoint_timeout_seconds)),
    ]
    if app_server is not None:
        argv += ["--app-server", json.dumps(list(app_server), separators=(",", ":"))]
    if restore is not None:
        argv += ["--restore", validate_restore_selector(restore)]
    elif fresh:
        argv.append("--fresh")
    if unattended:
        argv.append("--unattended")
    if temporary_local_root:
        argv.append("--temporary-local-root")
    if remote_helper is not None:
        argv += ["--remote-helper", remote_helper]
    if publish_run:
        argv.append("--publish-run")
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


def describe_thread_cwd(record: Mapping[str, Any]) -> str:
    """One line for the runner's ``thread_cwd`` record (issue 46): how many
    restored threads were moved into this allocation's clone, how many could
    not be and why, and what was left as recorded."""
    clone = record.get("clone") or "this clone"
    moved = record.get("moved_count")
    if not isinstance(moved, int):
        moved = len(record.get("moved") or [])
    failed = record.get("failed") or []
    parts = []
    if moved:
        parts.append(f"{moved} moved to {clone}")
    if failed:
        first = failed[0].get("error") if isinstance(failed[0], Mapping) else None
        why = record.get("skipped") or (record.get("helper") or {}).get("error") or first or "see thread-cwd.log"
        parts.append(f"{len(failed)} NOT moved ({why}); resume with \"Use current directory\" or codex resume -C {clone}")
    elif record.get("skipped"):
        parts.append(f"unchanged ({record['skipped']})")
    if record.get("kept_counts"):
        parts.append(f"left as recorded: {record['kept_counts']}")
    return "thread cwd: " + ("; ".join(parts) if parts else "nothing to move")


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
    cwd = status.get("thread_cwd")
    if isinstance(cwd, Mapping):
        lines.append(describe_thread_cwd(cwd))

    durable = status.get("durable") or owner.get("durable")
    if durable:
        lines.append(f"durable run: {durable}")
    if status.get("reason"):
        lines.append(f"reason: {status['reason']}")
    return "\n".join(lines)


# ----------------------------------------------------------------- drain request and acknowledgement

DRAIN_MODES = ("release", "renew")
DRAIN_REQUEST = "drain.request"
DRAIN_ACK = "drain.ack"
DRAIN_POLL_SECONDS = 2.0
# States in which no runner is left to read a request: a refused or blocked
# launch never started Codex, and ``released`` is the end of a renew hold.
# ``exited`` is among them only once the final generation has been
# published (a sealed or crash ``last_generation``): the exit path writes
# ``exited``, takes the final capture, then reads a late request once, so a
# run that has exited with only a warm generation may still be mid-capture
# and is asked like any other.
DRAIN_NO_OWNER_STATES = ("refused", "blocked", "released")
DRAIN_EXITED_STATES = ("exited", "stopped_misplaced")
DRAIN_FINAL_CLASSES = ("sealed", "crash")
DRAIN_TERMINAL_OUTCOMES = ("sealed", "crash", "warm_only", "failed")


def new_drain_nonce() -> str:
    """Sixteen lowercase hex characters, within the runner's 8-64 grammar."""
    return secrets.token_hex(8)


def drain_request_text(nonce: str, mode: str, requested_at: float, requester: str) -> str:
    """The request as the runner parses it (``parse_drain_request``): one JSON object."""
    return json.dumps({"nonce": nonce, "mode": mode, "requested_at": requested_at, "requester": requester},
                      sort_keys=True, separators=(",", ":"))


def drain_request_script(run_dir: str, nonce: str, text: str) -> str:
    """The one shell command that places the request atomically.

    The runner consumes ``<run>/drain.request`` whole and rejects a torn one,
    so the text goes to ``drain.request.tmp-<nonce>`` first and a ``mv``
    inside the run directory (a rename on one filesystem) publishes it.
    ``printf '%s\\n'`` takes the text as an argument, so no character in it
    is interpreted; every word is shell-quoted, and the executor's own
    ``shlex.join`` of the ``sh -c`` argv re-quotes the whole script for the
    remote shell, so a quote inside the requester text survives both.
    """
    tmp = f"{run_dir}/{DRAIN_REQUEST}.tmp-{nonce}"
    final = f"{run_dir}/{DRAIN_REQUEST}"
    return (f"umask 077 && printf '%s\\n' {shlex.quote(text)} > {shlex.quote(tmp)} "
            f"&& mv -f {shlex.quote(tmp)} {shlex.quote(final)}")


def _generation_label(record: Any) -> str:
    """``<name> (<manifest sha prefix>)`` for an acknowledgement's generation record."""
    name = _name_of(record) or "unknown"
    sha = record.get("manifest_sha256") if isinstance(record, Mapping) else None
    return f"{name} ({str(sha)[:12]})" if isinstance(sha, str) and sha else name


def describe_drain_ack(ack: Mapping[str, Any], last_durable: Any = None) -> str:
    """The one-line outcome of an acknowledgement the runner wrote.

    ``warm_only`` is final for the requester: a person holds the terminal,
    Codex was not signalled, and the runner updates the acknowledgement under
    the same nonce when they exit.  ``failed`` names the runner's reason and
    the last durable generation it knew (``last_durable`` in the
    acknowledgement, else the caller's ``status.json`` reading).
    """
    outcome = ack.get("outcome")
    generation = ack.get("generation")
    if outcome == "sealed":
        return f"drained: sealed generation {_generation_label(generation)}"
    if outcome == "warm_only":
        return (f"drained: warm generation {_name_of(generation) or 'unknown'} only "
                "(a person holds the terminal; it seals when they exit)")
    if outcome == "crash":
        return (f"drained: crash generation {_generation_label(generation)}; Codex exited {ack.get('exit')}, "
                "kept for recovery, not sealed")
    last = _name_of(ack.get("last_durable")) or _name_of(last_durable) or "none"
    if outcome == "failed":
        return f"drain failed: {ack.get('reason') or 'no reason recorded'}; last durable generation {last}"
    return f"drain acknowledged with an unknown outcome {outcome!r}; last durable generation {last}"


@dataclass(frozen=True)
class CodexStateDrain:
    """What one drain request came to.

    ``run`` is None when ``latest-run`` is absent (no managed run, nothing
    written).  ``requested`` says whether ``drain.request`` was placed;
    ``ack`` is the acknowledgement that matched this request's nonce and
    run, or None (a stale acknowledgement from an earlier request never
    counts).  ``outcome`` is the one line the operator reads.
    ``last_durable`` is the last durable generation known at the end, from
    the acknowledgement or from ``status.json``.  ``state`` is the run's
    recorded state when no request was written because no owner was left
    to answer one (refused, blocked, released, or exited with its final
    generation published), else None.  ``exit`` and ``exited_at`` carry
    the recorded exit code and newest exit transition's time in those
    no-owner answers, so renewal can reject a seal from an earlier launch
    of the same run.  ``commands`` lists every command sent, or on a dry
    run the ones that would be.
    """

    job_id: int
    scope: str
    mode: str
    nonce: str
    commands: List[List[str]]
    outcome: str
    run: Optional[str] = None
    requested: bool = False
    ack: Optional[Dict[str, Any]] = None
    last_durable: Optional[Dict[str, Any]] = None
    waited_seconds: float = 0.0
    timeout_seconds: float = 0.0
    dry_run: bool = False
    state: Optional[str] = None
    exit: Optional[int] = None
    exited_at: Optional[float] = None

    @property
    def acknowledged(self) -> bool:
        return self.ack is not None

    @property
    def settled(self) -> bool:
        """Codex stopped and its state published: sealed or crash, with the process gone."""
        return self.ack is not None and self.ack.get("outcome") in ("sealed", "crash") and bool(self.ack.get("stopped"))

    def as_json(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id, "scope": self.scope, "mode": self.mode, "nonce": self.nonce, "run": self.run,
            "requested": self.requested, "outcome": self.outcome, "ack": self.ack, "last_durable": self.last_durable,
            "state": self.state, "exit": self.exit, "exited_at": self.exited_at,
            "waited_seconds": self.waited_seconds, "timeout_seconds": self.timeout_seconds,
            "commands": self.commands,
        }


# ----------------------------------------------------------------- renewal: transfer token, attempt, readiness

GENERATION_NAME = re.compile(r"[0-9]{6}-[0-9a-f]{8}")
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
NONCE_HEX = re.compile(r"[0-9a-f]{8,64}")
# ``status.json`` states of a replacement run, as the renewal reads them
# (the runner's ``Status.update`` vocabulary).  ``placed`` is the only
# ready state; the pending ones precede it on a healthy launch; everything
# else means this launch will not become the managed owner.
READINESS_READY_STATE = "placed"
READINESS_PENDING_STATES = ("starting", "restoring", "launched")


@dataclass(frozen=True)
class TransferToken:
    """What a drain acknowledgement proves, as the replacement launch needs it.

    A token exists only for a *sealed* generation whose writer is proven
    stopped: the acknowledgement's outcome is ``sealed``, the generation's
    class is ``sealed`` and ``stopped`` is true (the child's pid with its
    ``/proc`` start time is gone; a dead pid alone is not that, under pid
    reuse).  ``selector`` is the ``RUN/GENERATION`` the replacement passes
    as ``--restore``; ``manifest_sha256`` is what readiness checks the
    restored run's origin against, so a run restored from any other
    generation (or reused in place) is never mistaken for the successor.
    ``nonce`` names the drain request that produced it.
    """

    nonce: str
    run: str
    generation: str
    manifest_sha256: str
    exit: Optional[int] = None
    child: Optional[Dict[str, Any]] = None

    @property
    def selector(self) -> str:
        return f"{self.run}/{self.generation}"

    def as_json(self) -> Dict[str, Any]:
        return {"nonce": self.nonce, "run": self.run, "generation": self.generation,
                "manifest_sha256": self.manifest_sha256, "exit": self.exit, "child": self.child}

    @classmethod
    def from_json(cls, data: Any) -> "TransferToken":
        """A token as :meth:`as_json` wrote it, validated as strictly as one from an acknowledgement."""
        if not isinstance(data, Mapping):
            raise ValueError("a transfer token must be a mapping")
        return _transfer_token(data.get("nonce"), data.get("run"), data.get("generation"), data.get("manifest_sha256"),
                               exit_code=data.get("exit"), child=data.get("child"))


def _transfer_token(nonce: Any, run: Any, generation: Any, manifest: Any, *, exit_code: Any, child: Any) -> TransferToken:
    if not isinstance(nonce, str) or not NONCE_HEX.fullmatch(nonce):
        raise ValueError(f"nonce {nonce!r} is not 8-64 lowercase hex")
    if not isinstance(run, str) or not RUN_ID.fullmatch(run):
        raise ValueError(f"run {run!r} is not a 32-hex run id")
    if not isinstance(generation, str) or not GENERATION_NAME.fullmatch(generation):
        raise ValueError(f"generation {generation!r} is not NNNNNN-hhhhhhhh")
    if not isinstance(manifest, str) or not SHA256_HEX.fullmatch(manifest):
        raise ValueError(f"manifest sha256 {manifest!r} is not 64 hex characters")
    if exit_code is not None and (isinstance(exit_code, bool) or not isinstance(exit_code, int)):
        raise ValueError(f"exit {exit_code!r} is not an integer")
    if child is not None and not isinstance(child, Mapping):
        raise ValueError("child must be a mapping of pid and starttime")
    return TransferToken(nonce=nonce, run=run, generation=generation, manifest_sha256=manifest,
                         exit=exit_code, child=dict(child) if child is not None else None)


def transfer_token_from_ack(ack: Mapping[str, Any], *, mode: str = "renew") -> TransferToken:
    """The token a drain acknowledgement carries, or ValueError saying why it carries none.

    Only a ``sealed`` outcome with the child proven ``stopped`` is
    transferable: ``warm_only`` means a person still holds the terminal
    (defer), ``crash`` and ``failed`` leave no consistent sealed state
    (recovery is explicit, never unattended).  ``mode`` must match the
    request's: a release-mode acknowledgement ends the runner, which then
    holds nothing against a restart in its allocation.
    """
    if ack.get("mode") != mode:
        raise ValueError(f"acknowledgement is for mode {ack.get('mode')!r}, not {mode}")
    outcome = ack.get("outcome")
    if outcome != "sealed":
        raise ValueError(f"outcome is {outcome!r}, not sealed")
    if ack.get("stopped") is not True:
        raise ValueError("the child process is not proven stopped")
    generation = ack.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("the acknowledgement names no generation")
    if generation.get("class") != "sealed":
        raise ValueError(f"the generation's class is {generation.get('class')!r}, not sealed")
    return _transfer_token(ack.get("nonce"), ack.get("run"), generation.get("name"), generation.get("manifest_sha256"),
                           exit_code=ack.get("exit"), child=ack.get("child"))


def transfer_token_from_exit(status: Mapping[str, Any], *, nonce: str) -> TransferToken:
    """The token of a run that already exited cleanly with a sealed final generation.

    A drain finds no owner to ask there (``codex_state_drain`` reports it
    without writing a request): Codex exited 0, the final capture sealed,
    and the runner is gone, so nothing can write to that history.  The
    generation must have been published at or after ``exited_at``, the
    newest exit transition's time; a reused run can retain an older seal
    when its later captures fail.  ``nonce`` is the request's, labelling
    the attempt.
    """
    if status.get("state") not in DRAIN_EXITED_STATES:
        raise ValueError(f"state is {status.get('state')!r}, not exited")
    last = status.get("last_generation")
    if not isinstance(last, Mapping) or last.get("class") != "sealed":
        raise ValueError("the final generation is not sealed")
    exit_code = status.get("exit")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code != 0:
        raise ValueError("the run did not record a clean exit (exit 0)")
    published_at, exited_at = last.get("published_at"), status.get("exited_at")
    if (isinstance(published_at, bool) or not isinstance(published_at, (int, float))
            or isinstance(exited_at, bool) or not isinstance(exited_at, (int, float))):
        raise ValueError("the final generation's publication or the last exit time is missing or invalid")
    if not published_at >= exited_at:
        raise ValueError("the sealed generation was not published after the last exit")
    return _transfer_token(nonce, status.get("run"), last.get("name"), last.get("manifest_sha256"),
                           exit_code=exit_code, child=None)


def transfer_token_from_drain(report: CodexStateDrain) -> TransferToken:
    """The token a renew-mode drain came to, or ValueError saying why it came to none.

    An acknowledgement must be sealed and stopped
    (:func:`transfer_token_from_ack`).  Without one, a run that had no owner
    left to ask because it had exited with a sealed final generation is as
    good: Codex exited 0, the final capture sealed and the runner is gone,
    so nothing can write to that history.  Everything else -- no answer in
    time, warm only, crash, failed, refused, blocked, released, no managed
    run -- is the reason to defer, in the report's own words.
    """
    if report.ack is not None:
        return transfer_token_from_ack(report.ack, mode=report.mode)
    if report.run is not None and not report.requested and report.state in DRAIN_EXITED_STATES:
        return transfer_token_from_exit(
            {"state": report.state, "run": report.run, "last_generation": report.last_durable,
             "exit": report.exit, "exited_at": report.exited_at}, nonce=report.nonce,
        )
    raise ValueError(report.outcome)


@dataclass
class ReplacementAttempt:
    """The renewal transaction, as the session record carries it between loop turns.

    Retains the old allocation (job, node, managed run) and the pending new
    one (attempt id, job, node) with the transfer token for the whole
    transaction, so a failure after submission leaves the new job findable
    (never submitted twice) and a failure before it leaves the old
    allocation reachable; neither replaces the old provenance early.
    ``state`` is ``drained`` (token held, nothing submitted), ``submitted``
    (new job recorded), ``ready`` (the successor run is placed on the
    token's generation; the predecessor is retained until cancellation is
    confirmed).  ``failures`` keeps the reasons of earlier tries of this
    attempt, newest last; ``cancel:`` entries record cleanup retries and
    do not count as further replacement submissions.
    """

    attempt: str
    old_job: int
    old_node: Optional[str]
    old_run: Optional[str]
    token: TransferToken
    state: str = "drained"
    new_job: Optional[int] = None
    new_node: Optional[str] = None
    submitted_at: Optional[float] = None    # epoch seconds when sbatch answered
    running_since: Optional[float] = None   # first RUNNING observation: bounds "no status yet"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    failures: List[str] = field(default_factory=list)

    def as_json(self) -> Dict[str, Any]:
        return {"attempt": self.attempt, "old_job": self.old_job, "old_node": self.old_node, "old_run": self.old_run,
                "token": self.token.as_json(), "state": self.state, "new_job": self.new_job, "new_node": self.new_node,
                "submitted_at": self.submitted_at, "running_since": self.running_since,
                "created_at": self.created_at, "updated_at": self.updated_at,
                "failures": list(self.failures)}

    @classmethod
    def from_json(cls, data: Any) -> "ReplacementAttempt":
        """A record as :meth:`as_json` wrote it; ValueError for anything else."""
        if not isinstance(data, Mapping):
            raise ValueError("a replacement attempt must be a mapping")
        attempt = data.get("attempt")
        if not isinstance(attempt, str) or not NONCE_HEX.fullmatch(attempt):
            raise ValueError(f"attempt {attempt!r} is not 8-64 lowercase hex")
        old_job, new_job = data.get("old_job"), data.get("new_job")
        for label, value in (("old_job", old_job), ("new_job", new_job)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{label} {value!r} is not a job id")
        if old_job is None:
            raise ValueError("old_job is missing")
        state = data.get("state")
        if state not in ("drained", "submitted", "ready"):
            raise ValueError(f"state {state!r} is not drained, submitted or ready")
        failures = data.get("failures") or []
        if not isinstance(failures, list) or not all(isinstance(item, str) for item in failures):
            raise ValueError("failures must be a list of strings")
        submitted_at = data.get("submitted_at")
        if submitted_at is not None and (isinstance(submitted_at, bool) or not isinstance(submitted_at, (int, float))):
            raise ValueError(f"submitted_at {submitted_at!r} is not a time")
        running_since = data.get("running_since")
        if running_since is not None and (isinstance(running_since, bool) or not isinstance(running_since, (int, float))):
            raise ValueError(f"running_since {running_since!r} is not a time")
        return cls(
            attempt=attempt, old_job=old_job, old_node=_optional_str(data.get("old_node"), "old_node"),
            old_run=_optional_str(data.get("old_run"), "old_run"), token=TransferToken.from_json(data.get("token")),
            state=state, new_job=new_job, new_node=_optional_str(data.get("new_node"), "new_node"),
            submitted_at=None if submitted_at is None else float(submitted_at),
            running_since=None if running_since is None else float(running_since),
            created_at=_optional_str(data.get("created_at"), "created_at"),
            updated_at=_optional_str(data.get("updated_at"), "updated_at"), failures=list(failures),
        )


def _optional_str(value: Any, label: str) -> Optional[str]:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{label} {value!r} is not text")
    return value


def new_attempt_id() -> str:
    """Sixteen lowercase hex characters naming one replacement attempt."""
    return secrets.token_hex(8)


@dataclass(frozen=True)
class Readiness:
    """Whether a replacement run has become the managed owner: ``ready``, ``pending`` or ``failed``, with why.

    ``no_status`` marks the pending answer given because the job has no
    ``status.json`` to read at all: a launch still cloning the mirror, or
    one whose runner refused before placing a run, which writes no
    ``latest-run`` and so never shows a state.  The caller bounds that
    case by the time since submission, which this function cannot know.
    """

    outcome: str
    reason: str
    no_status: bool = False

    @property
    def ready(self) -> bool:
        return self.outcome == "ready"


def classify_readiness(status: Optional[Mapping[str, Any]], token: TransferToken) -> Readiness:
    """Read a replacement run's ``status.json`` against the token it was launched with.

    Ready means ``placed`` -- Codex's stores verified under the managed
    directory -- *and* ``restored_from`` naming exactly the token's run,
    generation and manifest hash: ``tmux has-session`` says a shell exists,
    this says the successor owns the history the predecessor sealed.  A run
    reused in place (``reused_run`` or ``restore_ignored``) or restored from
    anything else is a failure, not a successor.  No status yet, or a state before
    ``placed``, is pending; every other state is a failure with the
    runner's reason where it recorded one.
    """
    if status is None:
        return Readiness("pending", "no managed run recorded yet (the job is cloning the mirror, then starting the runner)",
                         no_status=True)
    state = status.get("state")
    if not isinstance(state, str):
        return Readiness("failed", "status.json records no state")
    if status.get("restore_ignored") or status.get("reused_run"):
        return Readiness("failed", "the allocation already held a run for this scope and reused it in place; "
                                   "the token's generation was not restored")
    if state in READINESS_PENDING_STATES:
        return Readiness("pending", f"state {state}")
    if state != READINESS_READY_STATE:
        reason = status.get("reason")
        return Readiness("failed", f"state {state}" + (f": {reason}" if isinstance(reason, str) and reason else ""))
    origin = status.get("restored_from")
    if not isinstance(origin, Mapping):
        return Readiness("failed", f"placed, but not restored from {token.selector}: mode {status.get('mode')!r}")
    if (origin.get("run"), origin.get("generation"), origin.get("manifest_sha256")) != (
            token.run, token.generation, token.manifest_sha256):
        return Readiness("failed", f"placed, but restored from {origin.get('run')}/{origin.get('generation')} "
                                   f"(manifest {str(origin.get('manifest_sha256'))[:12]}), not {token.selector} "
                                   f"(manifest {token.manifest_sha256[:12]})")
    return Readiness("ready", f"placed on {token.selector}; run {status.get('run')}")
