"""Which agent sessions need the human: status records written by the agent.

Design: docs/attention.org.

A SuCoder-launched Claude session gets hook settings (``--settings``) whose
command writes one small *record* per session to
``$HOME/.sucoder/attention/<key>.rec`` on whichever host runs the agent.  The
record is the hook's own JSON payload behind a one-line header, replaced
atomically on every event, so the newest event is the session's state:

- ``UserPromptSubmit`` -- working;
- ``PostToolUse`` -- working again (clears a granted permission request);
  recorded without its payload, which carries the tool's output;
- ``Stop`` -- the turn finished (idle);
- ``Notification`` -- Claude asks for a permission or an answer, or has sat
  idle for a minute waiting for input: the human is needed.  Only those
  notification types fire the hook (a matcher on ``notification_type``);
  ``auth_success``, ``agent_completed`` and the like leave the state alone;
- ``SessionStart`` / ``SessionEnd`` -- started (awaiting a first prompt) /
  ended.

A Codex session gets the equivalent hooks as ``-c hooks.<Event>=...``
overrides (:func:`codex_overrides`).  Codex runs a ``-c`` hook only once it
has been reviewed and trusted, and records that trust against the hook's
exact definition, so the Codex hook command is the same for every launch:
the record's key arrives in ``$SUCODER_ATTENTION_KEY`` instead.  Codex
events: ``PermissionRequest`` (waiting, permission), ``PreToolUse`` for its
``request_user_input`` tool (waiting, question), ``Stop`` and ``Interrupt``
(idle), and the four shared with Claude.  Codex has no idle notification;
:func:`synthesize_idle_waits` turns a Codex turn idle for
:data:`IDLE_WAIT_SECONDS` into a waiting request of kind ``idle``, as
Claude's own ``idle_prompt`` does.

The hook is plain POSIX shell with no dependency on SuCoder being installed
where the agent runs, so it works on a compute node as well as locally, and
it always exits 0: a status write must never block or fail the agent.

Everything else here is pure: :func:`parse_record` and the renderers take
text in, and the caller (``sucoder attention``) does the reading and SSH.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

RECORD_DIR = ".sucoder/attention"  # relative to the agent's $HOME
RECORD_SUFFIX = ".rec"
RECORD_HEADER = "sucoder-attention-v1"
DIR_ENV = "SUCODER_ATTENTION_DIR"
KEY_ENV = "SUCODER_ATTENTION_KEY"   # the record key, for a hook whose command cannot carry it
MAX_RECORD_BYTES = 65536
LOCAL_TARGET = "local"
CLAUDE = "claude"
CODEX = "codex"

# The hook events SuCoder installs (either harness), and the state each one
# leaves behind.  A Codex ``PreToolUse`` is hooked only for its question tool.
EVENT_STATES: Dict[str, str] = {
    "SessionStart": "idle",
    "UserPromptSubmit": "working",
    "PreToolUse": "working",
    "PostToolUse": "working",
    "Notification": "waiting",
    "PermissionRequest": "waiting",
    "Stop": "idle",
    "Interrupt": "idle",
    "SessionEnd": "ended",
}
CLAUDE_EVENTS = ("SessionStart", "UserPromptSubmit", "PostToolUse", "Notification", "Stop", "SessionEnd")
CODEX_EVENTS = ("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PermissionRequest",
                "Stop", "Interrupt", "SessionEnd")

# Codex matchers (regexes).  ``SessionStart`` also fires after a compaction,
# mid-turn, and that must not read as idle.  ``request_user_input`` is the
# tool through which Codex asks the human a question (``_async`` variant
# included): the one ``PreToolUse`` that means waiting.
CODEX_MATCHERS: Dict[str, str] = {
    "SessionStart": "^(startup|resume|clear|fork)$",
    "PreToolUse": "^request_user_input",
}
CODEX_QUESTION_TOOL = "request_user_input"
# Codex caps SessionEnd and Interrupt hooks at 3 s; the rest get SuCoder's 10.
CODEX_SHORT_TIMEOUT_EVENTS = frozenset({"SessionEnd", "Interrupt"})
# Claude's idle_prompt fires after a minute; a Codex turn idle that long is
# treated the same way.
IDLE_WAIT_SECONDS = 60

# Claude's ``notification_type`` values that mean the agent is blocked on the
# human, each with the *kind* the push filter selects on.  Read from Claude
# Code 2.1.316; other types (auth_success, agent_completed, computer_use_*,
# push_notification, ...) are informational and are not hooked.
WAITING_TYPES: Dict[str, str] = {
    "permission_prompt": "permission",
    "worker_permission_prompt": "permission",
    "elicitation_dialog": "question",
    "elicitation_url_dialog": "question",
    "agent_needs_input": "question",
    "idle_prompt": "idle",
}
NOTIFY_KINDS = ("permission", "question", "idle", "other")

# Display/sort order: what most needs the human first.  ``untracked`` is a
# live session with no record (launched without hooks), so it could be
# waiting; it sorts just after the sessions known to need the human.
STATE_ORDER = ("waiting", "idle", "untracked", "working", "ended", "unknown")

_KEY_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _token(name: str) -> str:
    return _UNSAFE_RE.sub("_", name)


def attention_key(mirror: str, target: Optional[str]) -> str:
    """``<mirror>--<target>``, the same pairing as the session records.

    Sanitized to ``[A-Za-z0-9._-]`` so it can sit unquoted inside the hook's
    shell command and in a file name.
    """
    key = f"{_token(mirror)}--{_token(target) if target else LOCAL_TARGET}"
    if not _KEY_RE.match(key):  # e.g. a mirror name starting with '.'
        key = "_" + key
    return key


def split_key(key: str) -> Tuple[str, str]:
    mirror, sep, target = key.rpartition("--")
    return (mirror, target) if sep else (key, "")


# Events recorded without their stdin payload, which carries tool input or
# output: only the event name (and, for Codex's question tool, its name).
NAME_ONLY_EVENTS = frozenset({"PostToolUse", "PreToolUse"})


def hook_shell_command(key: Optional[str], *, event: Optional[str] = None,
                       harness: str = CLAUDE) -> str:
    """The shell command a hook runs: payload on stdin -> record.

    Writes a header line (format, epoch seconds, host, harness) and then the
    payload to a temporary file beside the record and renames it into place,
    so a reader never sees a torn record.  ``$SUCODER_ATTENTION_DIR``
    overrides the directory (tests; unusual homes).

    With *key* ``None`` the key is read from ``$SUCODER_ATTENTION_KEY`` at
    run time and checked there, so one command serves every launch (Codex's
    trust is per command).  For an *event* in :data:`NAME_ONLY_EVENTS` the
    payload is drained and replaced by ``{"hook_event_name": event}``.
    """
    if harness not in (CLAUDE, CODEX):
        raise ValueError(f"unknown harness {harness!r}")
    if key is None:
        guard = (f'k="${{{KEY_ENV}:-}}"; '
                 'case "$k" in ""|.*|*[!A-Za-z0-9._-]*) cat >/dev/null; exit 0;; esac; ')
        name = "$k"
    elif _KEY_RE.match(key):
        guard, name = "", key
    else:
        raise ValueError(f"attention key {key!r} is not shell- and path-safe")
    if event in NAME_ONLY_EVENTS:
        if not re.match(r"^[A-Za-z]+$", event):
            raise ValueError(f"unsafe hook event name {event!r}")
        extra = f',\"tool_name\":\"{CODEX_QUESTION_TOOL}\"' if event == "PreToolUse" else ""
        body = f"cat >/dev/null; printf '{{\"hook_event_name\":\"{event}\"{extra}}}'"
    else:
        body = f"head -c {MAX_RECORD_BYTES}"
    stamp = f'"%s\\t%s\\t%s\\t%s\\n" {RECORD_HEADER} "$(date +%s)" "$(hostname)" {harness}'
    return (
        guard
        + f'd="${{{DIR_ENV}:-$HOME/{RECORD_DIR}}}"; '
        'mkdir -p "$d" 2>/dev/null || exit 0; '
        f't="$d/.{name}.$$"; '
        f'{{ printf {stamp}; '
        f'{body}; }} > "$t" 2>/dev/null '
        f'&& mv -f "$t" "$d/{name}{RECORD_SUFFIX}" 2>/dev/null; '
        'rm -f "$t" 2>/dev/null; exit 0'
    )


def claude_settings(key: str) -> Dict[str, object]:
    """Settings for ``claude --settings``: one command hook per event.

    Hooks from ``--settings`` are added to the user's and project's own;
    they do not replace them.
    """
    def entry(event: str) -> List[Dict[str, object]]:
        command = {"type": "command", "command": hook_shell_command(key, event=event), "timeout": 10}
        group: Dict[str, object] = {"hooks": [command]}
        if event == "Notification":
            group["matcher"] = "|".join(WAITING_TYPES)
        return [group]
    return {"hooks": {event: entry(event) for event in CLAUDE_EVENTS}}


def _toml_string(value: str) -> str:
    """A TOML basic string (JSON's escapes are TOML's, for this ASCII text)."""
    if not value.isascii():
        raise ValueError("Codex hook text must be ASCII")
    return json.dumps(value)


def codex_overrides() -> List[str]:
    """``-c hooks.<Event>=[...]`` arguments for a Codex launch.

    Identical for every launch -- the key comes from ``$SUCODER_ATTENTION_KEY``
    -- because Codex trusts a ``-c`` hook by a hash of its definition: one
    review per account, not one per mirror.  Changing anything here (the
    command, a matcher, a timeout) asks every account to review again.
    """
    args: List[str] = []
    for event in CODEX_EVENTS:
        timeout = 3 if event in CODEX_SHORT_TIMEOUT_EVENTS else 10
        command = hook_shell_command(None, event=event, harness=CODEX)
        handler = f"{{type={_toml_string('command')},command={_toml_string(command)},timeout={timeout}}}"
        matcher = CODEX_MATCHERS.get(event)
        group = (f"matcher={_toml_string(matcher)}," if matcher else "") + f"hooks=[{handler}]"
        args.extend(["-c", f"hooks.{event}=[{{{group}}}]"])
    return args


def claude_settings_json(key: str) -> str:
    return json.dumps(claude_settings(key), separators=(",", ":"))


@dataclass
class Status:
    key: str
    mirror: str
    target: str
    state: str
    event: Optional[str] = None
    at: Optional[float] = None
    host: Optional[str] = None
    message: Optional[str] = None
    kind: Optional[str] = None
    session_id: Optional[str] = None
    cwd: Optional[str] = None
    transcript_path: Optional[str] = None
    source: str = "local"
    harness: Optional[str] = None

    def age(self, now: float) -> Optional[float]:
        return None if self.at is None else max(0.0, now - self.at)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def parse_record(key: str, text: str, *, source: str = "local") -> Status:
    """One record file's text -> :class:`Status`; malformed reads as unknown."""
    mirror, target = split_key(key)
    status = Status(key=key, mirror=mirror, target=target, state="unknown", source=source)
    header, _, body = text.partition("\n")
    fields = header.split("\t")
    if len(fields) < 3 or fields[0] != RECORD_HEADER:
        return status
    try:
        status.at = float(fields[1])
    except ValueError:
        pass
    status.host = fields[2].strip() or None
    harness = fields[3].strip() if len(fields) > 3 else ""
    status.harness = harness if harness in (CLAUDE, CODEX) else CLAUDE
    try:
        payload = json.loads(body) if body.strip() else {}
    except ValueError:
        # Cut off at MAX_RECORD_BYTES (a Codex permission request carries the
        # tool's whole input): the leading fields still say what happened.
        payload = _salvage_payload(body)
    if not isinstance(payload, dict):
        return status
    event = payload.get("hook_event_name")
    status.event = event if isinstance(event, str) else None
    status.state = EVENT_STATES.get(status.event or "", "unknown")
    for name in ("message", "session_id", "cwd", "transcript_path"):
        value = payload.get(name)
        if isinstance(value, str):
            setattr(status, name, value)
    if status.event == "Notification":
        status.kind = notification_kind(payload.get("notification_type"), status.message)
        if status.kind is None:  # an informational type the matcher let through
            status.state = "unknown"
    elif status.event == "PermissionRequest":
        status.kind = "permission"
        status.message = permission_message(payload)
    elif status.event == "PreToolUse":
        tool = payload.get("tool_name")
        if isinstance(tool, str) and tool.startswith(CODEX_QUESTION_TOOL):
            status.state, status.kind = "waiting", "question"
            status.message = "Codex is asking you a question"
    return status


_STRING_FIELD_RE = r'"{name}"\s*:\s*"((?:[^"\\]|\\.)*)"'


def _salvage_payload(body: str) -> Dict[str, object]:
    """Top-level string fields recoverable from a truncated JSON payload."""
    out: Dict[str, object] = {}
    for name in ("hook_event_name", "session_id", "cwd", "transcript_path", "tool_name", "message"):
        match = re.search(_STRING_FIELD_RE.format(name=name), body)
        if match:
            try:
                out[name] = json.loads(f'"{match.group(1)}"')
            except ValueError:
                continue
    return out if isinstance(out.get("hook_event_name"), str) else {}


def permission_message(payload: Dict[str, object]) -> str:
    """One line for a Codex permission request: the tool and what it wants."""
    tool = payload.get("tool_name") if isinstance(payload.get("tool_name"), str) else "a tool"
    what = None
    tool_input = payload.get("tool_input")
    if isinstance(tool_input, dict):
        for field in ("description", "command", "cmd"):
            value = tool_input.get(field)
            if isinstance(value, list):
                value = " ".join(str(v) for v in value)
            if isinstance(value, str) and value.strip():
                what = value.strip()
                break
    return f"Codex needs your approval for {tool}" + (f": {what}" if what else "")


def notification_kind(notification_type: object, message: Optional[str]) -> Optional[str]:
    """``permission``/``question``/``idle``/``other`` for a blocking
    notification, ``None`` for an informational one.

    Without a ``notification_type`` (an older Claude) the message decides.
    """
    if isinstance(notification_type, str) and notification_type:
        return WAITING_TYPES.get(notification_type)
    text = (message or "").lower()
    if "permission" in text:
        return "permission"
    if "waiting for your input" in text:
        return "idle"
    return "other"


# Dump every record in the agent's attention directory on one host: each
# record preceded by an RS (0x1e) line naming it.  One round trip per host.
LIST_RECORDS_SH = (
    f'd="${{{DIR_ENV}:-$HOME/{RECORD_DIR}}}"; '
    f'for f in "$d"/*{RECORD_SUFFIX}; do '
    '[ -f "$f" ] || continue; '
    'n="${f##*/}"; '
    "printf '\\036%s\\n' \"$n\"; "
    f'head -c {MAX_RECORD_BYTES + 512} "$f"; '
    "printf '\\n'; "
    'done; exit 0'
)


def parse_listing(text: str, *, source: str) -> List[Status]:
    """Split :data:`LIST_RECORDS_SH` output into statuses."""
    statuses = []
    for chunk in text.split("\x1e")[1:]:
        name, _, body = chunk.partition("\n")
        name = name.strip()
        if not name.endswith(RECORD_SUFFIX):
            continue
        statuses.append(parse_record(name[: -len(RECORD_SUFFIX)], body, source=source))
    return statuses


def sort_statuses(statuses: Iterable[Status], now: float) -> List[Status]:
    """Most in need of the human first; within a state, longest waiting first."""
    def rank(s: Status):
        order = STATE_ORDER.index(s.state) if s.state in STATE_ORDER else len(STATE_ORDER)
        age = s.age(now)
        return (order, -(age if age is not None else -1.0), s.key)
    return sorted(statuses, key=rank)


def merge_statuses(statuses: Iterable[Status]) -> List[Status]:
    """One status per key: the newest record wins (a key seen on two hosts)."""
    best: Dict[str, Status] = {}
    for s in statuses:
        old = best.get(s.key)
        if old is None or (s.at or 0) > (old.at or 0):
            best[s.key] = s
    return list(best.values())


def format_age(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}"
    return f"{seconds // 86400}d"


def _one_line(text: Optional[str], width: int) -> str:
    if not text:
        return ""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def render_table(statuses: Sequence[Status], now: float) -> str:
    if not statuses:
        return "No attention records. Sessions report once launched with attention.hooks enabled."
    rows = [("STATE", "AGE", "MIRROR", "TARGET", "HOST", "DETAIL")]
    for s in sort_statuses(statuses, now):
        detail = _one_line(s.message, 60) if s.state in ("waiting", "untracked") else ""
        if s.state == "waiting" and s.kind:
            detail = f"[{s.kind}] {detail}".rstrip()
        rows.append((s.state, format_age(s.age(now)), s.mirror, s.target,
                     (s.host or "?").split(".")[0], detail))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]) - 1)]
    lines = []
    for r in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(r[:-1], widths)) + "  " + r[-1])
    return "\n".join(line.rstrip() for line in lines)


def render_tmux(statuses: Sequence[Status], now: float, *, max_names: int = 4) -> str:
    """One short line for a tmux ``status-right``: who is waiting, and counts."""
    ordered = sort_statuses(statuses, now)
    waiting = [s for s in ordered if s.state == "waiting"]
    working = sum(1 for s in ordered if s.state == "working")
    idle = sum(1 for s in ordered if s.state == "idle")
    untracked = sum(1 for s in ordered if s.state == "untracked")
    parts = []
    if waiting:
        names = [f"{s.mirror}({format_age(s.age(now))})" for s in waiting[:max_names]]
        if len(waiting) > max_names:
            names.append(f"+{len(waiting) - max_names}")
        parts.append("WAIT " + " ".join(names))
    parts.append(f"run {working}")
    if idle:
        parts.append(f"idle {idle}")
    if untracked:
        parts.append(f"untracked {untracked}")
    return " | ".join(parts)


def synthesize_idle_waits(statuses: Iterable[Status], now: float,
                          *, after: float = IDLE_WAIT_SECONDS) -> List[Status]:
    """A Codex session idle for *after* seconds is waiting on the human.

    Claude says so itself (``idle_prompt``, a ``Notification`` a minute
    after its turn ends); Codex has no such event, so its ``Stop`` or
    ``Interrupt`` record is promoted here to a waiting request of kind
    ``idle``, dated when the minute ran out -- the same request a Claude
    session would have raised, so ``notify_kinds`` and
    ``notify_after_minutes`` treat both alike.  Its ``(key, at)`` is stable
    across passes, so it notifies once.
    """
    out = []
    for s in statuses:
        if (s.harness == CODEX and s.state == "idle" and s.event in ("Stop", "Interrupt")
                and s.at is not None and now - s.at >= after):
            s = Status(**{**s.as_dict(), "state": "waiting", "kind": "idle",
                          "at": s.at + after, "message": "Codex is waiting for your input"})
        out.append(s)
    return out


def due_notifications(statuses: Sequence[Status], notified: Set[Tuple[str, Optional[float]]], now: float,
                      *, kinds: Sequence[str] = NOTIFY_KINDS, after_seconds: float = 0.0) -> List[Status]:
    """Waiting sessions to notify about now; adds them to *notified*.

    A request is one record, ``(key, event time)``: it notifies once, when it
    has waited *after_seconds* and its kind is in *kinds*.  A new event on a
    session still waiting (a second permission prompt) is a new request.
    """
    due = []
    for s in statuses:
        if s.state != "waiting" or (s.key, s.at) in notified:
            continue
        if (s.kind or "other") not in kinds:
            continue
        age = s.age(now)
        if after_seconds > 0 and (age is None or age < after_seconds):
            continue
        notified.add((s.key, s.at))
        due.append(s)
    return due


def waiting_requests(statuses: Iterable[Status]) -> Set[Tuple[str, Optional[float]]]:
    """The requests already pending, for a watcher's baseline."""
    return {(s.key, s.at) for s in statuses if s.state == "waiting"}


def cache_payload(statuses: Sequence[Status], *, errors: Sequence[str], now: Optional[float] = None) -> Dict[str, object]:
    return {
        "version": 1,
        "collected_at": time.time() if now is None else now,
        "errors": list(errors),
        "sessions": [s.as_dict() for s in statuses],
    }


def statuses_from_cache(payload: Dict[str, object]) -> List[Status]:
    out = []
    for row in payload.get("sessions", []) or []:
        if isinstance(row, dict) and isinstance(row.get("key"), str):
            known = {k: row.get(k) for k in Status.__dataclass_fields__ if k in row}
            out.append(Status(**known))
    return out


# ---------------------------------------------------------------------------
# Live sessions without records ("untracked").
#
# Hooks are attached at launch, so a session started before attention.hooks
# was enabled (or by an older SuCoder, or a harness without hooks) writes no
# record and would otherwise be invisible.  The listing therefore also asks
# what is running -- locally from the process table, on a cluster from
# ``squeue`` (or tmux, on a host without a scheduler) in the same round trip
# as the records -- and shows a live session with no record, or only an
# ``ended`` one, as ``untracked``: alive, state unknown.  It never notifies.
# ---------------------------------------------------------------------------

UNTRACKED = "untracked"

# Harness executables a launch can start (mirror the harness profiles).
AGENT_COMMANDS = frozenset({"claude", "codex", "gemini", "aider", "opencode", "goose", "kimi"})


@dataclass(frozen=True)
class LiveSession:
    """A running agent session, keyed like the records it may lack.

    *targets* are the target names it could belong to: ``(None,)`` for this
    host, one name when known, several when a cluster job matched no single
    target.  *label* names its source (``local``, a cluster, a target).
    """

    mirror: str
    targets: Tuple[Optional[str], ...]
    label: str
    host: Optional[str] = None
    detail: str = ""
    session_id: Optional[str] = None
    cwd: Optional[str] = None
    transcript_path: Optional[str] = None


# The wrapped script of ``LocalExecutor.run_agent``:
#   ... ; cd <quoted dir> || exit 1; umask 0007; <agent> <args>
_LAUNCH_RE = re.compile(
    r"cd ('(?:[^']|'\"'\"')*'|[^\s;']+) \|\| exit 1; umask [0-7]{3,4}; \s*(?:env\s+)?(?:\S+=\S*\s+)*([^\s;'\"]+)"
)
_SESSION_ID_RE = re.compile(r"--session-id[ =]'?([0-9A-Fa-f-]{36})")


def parse_launch_argv(argv: Sequence[str], *, agent_user: str,
                      agents: Iterable[str] = AGENT_COMMANDS) -> Optional[Tuple[str, str, Optional[str]]]:
    """``(cwd, agent, session_id)`` when *argv* is a local agent launch.

    Matches ``sudo -u <agent_user> [env K=V ...] bash -c <script>`` (the
    sudo process; with sudoers ``use_pty`` it stays as the parent) and the
    bare ``bash -c <script>`` it runs, whose script ends in an agent
    command.  Other ``run_agent`` commands (git in the mirror) are not
    launches and return ``None``.
    """
    args = list(argv)
    if args and os.path.basename(args[0]) == "sudo":
        try:
            i = args.index("-u")
        except ValueError:
            return None
        if i + 1 >= len(args) or args[i + 1] != agent_user:
            return None
        args = args[i + 2:]
        if args and args[0] == "env":
            args = args[1:]
            while args and "=" in args[0] and not args[0].startswith("-"):
                args = args[1:]
    if len(args) < 3 or os.path.basename(args[0]) != "bash" or args[1] != "-c":
        return None
    match = _LAUNCH_RE.search(args[2])
    if not match:
        return None
    agent = os.path.basename(match.group(2))
    if agent not in set(agents):
        return None
    try:
        cwd = shlex.split(match.group(1))[0]
    except (ValueError, IndexError):
        return None
    sid = _SESSION_ID_RE.search(args[2])
    return cwd, agent, sid.group(1) if sid else None


def claude_transcript_path(home: str, cwd: str, session_id: str) -> Optional[str]:
    """Where Claude keeps a session's transcript: its project dir is the cwd
    with every non-alphanumeric character replaced by ``-``.  ``None`` for a
    path long enough that Claude would shorten it differently."""
    slug = re.sub(r"[^A-Za-z0-9]", "-", cwd)
    if len(slug) > 200:
        return None
    return f"{home.rstrip('/')}/.claude/projects/{slug}/{session_id}.jsonl"


def scan_local_launches(*, agent_user: str, agent_uid: Optional[int], proc_root: str = "/proc",
                        agents: Iterable[str] = AGENT_COMMANDS) -> List[Tuple[str, str, Optional[str]]]:
    """Local agent launches visible in the process table, one per directory.

    Reads only ``cmdline``, which every user may read.  A bare ``bash -c``
    must belong to *agent_uid*, so another account's look-alike is ignored.
    """
    agents = frozenset(agents)
    found: Dict[str, Tuple[str, str, Optional[str]]] = {}
    try:
        pids = [p for p in os.listdir(proc_root) if p.isdigit()]
    except OSError:
        return []
    for pid in sorted(pids, key=int):
        base = os.path.join(proc_root, pid)
        try:
            with open(os.path.join(base, "cmdline"), "rb") as handle:
                raw = handle.read(1 << 20)
            uid = os.stat(base).st_uid
        except OSError:
            continue
        argv = [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]
        if not argv:
            continue
        if os.path.basename(argv[0]) != "sudo" and agent_uid is not None and uid != agent_uid:
            continue
        launch = parse_launch_argv(argv, agent_user=agent_user, agents=agents)
        if launch is None:
            continue
        cwd, agent, sid = launch
        if cwd not in found or (sid and not found[cwd][2]):
            found[cwd] = (cwd, agent, sid)
    return list(found.values())


def local_live_sessions(launches: Iterable[Tuple[str, str, Optional[str]]], *,
                        mirror_paths: Mapping[str, str], mirror_root: str,
                        agent_home: str, host: Optional[str] = None) -> List[LiveSession]:
    """Map scanned launches to mirrors (``mirror_paths``: dir -> name)."""
    out = []
    root = mirror_root.rstrip("/") + "/"
    for cwd, agent, sid in launches:
        mirror = mirror_paths.get(cwd.rstrip("/"))
        if mirror is None:
            mirror = cwd[len(root):] if cwd.startswith(root) else os.path.basename(cwd.rstrip("/"))
        transcript = claude_transcript_path(agent_home, cwd, sid) if (agent == "claude" and sid) else None
        out.append(LiveSession(
            mirror=mirror, targets=(None,), label=LOCAL_TARGET, host=host,
            detail=f"{agent} launched without status hooks", session_id=sid, cwd=cwd,
            transcript_path=transcript,
        ))
    return out


LIVE_MARKER = "\x1fsucoder-live"
LIVE_RC_MARKER = "\x1fsucoder-live-rc "


def records_and_live_sh(live_command: str) -> str:
    """:data:`LIST_RECORDS_SH`, then *live_command*, in one round trip.

    The records half runs in a subshell so its ``exit 0`` returns to the
    script.  Unit-separator markers (0x1f, never raw in a record, whose
    payload is JSON) delimit the live half and carry its exit status.
    """
    return (
        f"( {LIST_RECORDS_SH} ); "
        "printf '\\037sucoder-live\\n'; "
        f"{live_command}; "
        "printf '\\n\\037sucoder-live-rc %s\\n' \"$?\"; exit 0"
    )


def split_records_and_live(text: str) -> Tuple[str, Optional[str], Optional[int]]:
    """``(records text, live text, live exit status)``; the live parts are
    ``None`` when the script never reached them (unanswered, not empty)."""
    records, sep, rest = text.partition(LIVE_MARKER + "\n")
    if not sep:
        return text, None, None
    live, sep, tail = rest.rpartition(LIVE_RC_MARKER)
    if not sep:
        return records, None, None
    try:
        rc = int(tail.strip().split()[0])
    except (ValueError, IndexError):
        rc = None
    return records, live, rc


def live_from_squeue(text: str, *, label: str, targets: Mapping[str, object],
                     candidates: Sequence[str], mirror_tokens: Mapping[str, str]) -> Tuple[List[LiveSession], List[str]]:
    """Running SuCoder jobs on one cluster, as live sessions."""
    from .sessions_report import match_target, parse_squeue, token_to_mirror

    jobs, bad = parse_squeue(text)
    out = []
    for job in jobs:
        if job.state != "RUNNING":
            continue
        mirror = token_to_mirror(job.token, mirror_tokens) or job.token
        target = match_target(job, targets, candidates)
        names: Tuple[Optional[str], ...] = (target,) if target else tuple(candidates)
        out.append(LiveSession(
            mirror=mirror, targets=names, label=label, host=job.node or None,
            detail=f"no status hooks; job {job.job_id}, {job.time_left} left",
        ))
    return out, bad


def live_from_tmux(text: str, *, target: str, host: str,
                   mirror_tokens: Mapping[str, str]) -> List[LiveSession]:
    """SuCoder tmux sessions on a host without a scheduler."""
    from .sessions_report import JOB_NAME_PREFIX, parse_tmux_sessions, token_to_mirror

    out = []
    for name in parse_tmux_sessions(text):
        token = name[len(JOB_NAME_PREFIX):]
        out.append(LiveSession(
            mirror=token_to_mirror(token, mirror_tokens) or token, targets=(target,),
            label=target, host=host, detail=f"no status hooks; tmux {name}",
        ))
    return out


def mark_untracked(statuses: Sequence[Status], live: Iterable[LiveSession]) -> List[Status]:
    """*statuses* plus an ``untracked`` status for each live session that
    has no record, or only an ``ended`` one (which it then replaces)."""
    by_key = {s.key: s for s in statuses}
    drop: Set[str] = set()
    added: Dict[str, Status] = {}
    for ls in live:
        keys = [attention_key(ls.mirror, t) for t in ls.targets] or [attention_key(ls.mirror, ls.label)]
        if any(k in by_key and by_key[k].state != "ended" for k in keys):
            continue
        drop.update(k for k in keys if k in by_key)
        if len(ls.targets) == 1:
            key, target = keys[0], ls.targets[0] or LOCAL_TARGET
        else:  # a job no single target claims: file it under its cluster
            key, target = attention_key(ls.mirror, ls.label), ls.label
        if key in added:
            continue
        added[key] = Status(
            key=key, mirror=ls.mirror, target=target, state=UNTRACKED, host=ls.host,
            message=ls.detail, session_id=ls.session_id, cwd=ls.cwd,
            transcript_path=ls.transcript_path, source=ls.label,
        )
    return [s for s in statuses if s.key not in drop] + list(added.values())
