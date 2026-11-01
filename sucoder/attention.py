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

The hook is plain POSIX shell with no dependency on SuCoder being installed
where the agent runs, so it works on a compute node as well as locally, and
it always exits 0: a status write must never block or fail the agent.

Everything else here is pure: :func:`parse_record` and the renderers take
text in, and the caller (``sucoder attention``) does the reading and SSH.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

RECORD_DIR = ".sucoder/attention"  # relative to the agent's $HOME
RECORD_SUFFIX = ".rec"
RECORD_HEADER = "sucoder-attention-v1"
DIR_ENV = "SUCODER_ATTENTION_DIR"
MAX_RECORD_BYTES = 65536
LOCAL_TARGET = "local"

# The hook events SuCoder installs, and the state each one leaves behind.
EVENT_STATES: Dict[str, str] = {
    "SessionStart": "idle",
    "UserPromptSubmit": "working",
    "PostToolUse": "working",
    "Notification": "waiting",
    "Stop": "idle",
    "SessionEnd": "ended",
}

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

# Display/sort order: what most needs the human first.
STATE_ORDER = ("waiting", "idle", "working", "ended", "unknown")

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


# Events recorded without their stdin payload (only the event name).
NAME_ONLY_EVENTS = frozenset({"PostToolUse"})


def hook_shell_command(key: str, *, event: Optional[str] = None) -> str:
    """The shell command a Claude hook runs: payload on stdin -> record.

    Writes a header line (format, epoch seconds, host) and then the payload
    to a temporary file beside the record and renames it into place, so a
    reader never sees a torn record.  ``$SUCODER_ATTENTION_DIR`` overrides
    the directory (tests; unusual homes).

    For an *event* in :data:`NAME_ONLY_EVENTS` the payload is drained and
    replaced by ``{"hook_event_name": event}``.
    """
    if not _KEY_RE.match(key):
        raise ValueError(f"attention key {key!r} is not shell- and path-safe")
    if event in NAME_ONLY_EVENTS:
        if not re.match(r"^[A-Za-z]+$", event):
            raise ValueError(f"unsafe hook event name {event!r}")
        body = f"cat >/dev/null; printf '{{\"hook_event_name\":\"{event}\"}}'"
    else:
        body = f"head -c {MAX_RECORD_BYTES}"
    return (
        f'd="${{{DIR_ENV}:-$HOME/{RECORD_DIR}}}"; '
        'mkdir -p "$d" 2>/dev/null || exit 0; '
        f't="$d/.{key}.$$"; '
        f'{{ printf "%s\\t%s\\t%s\\n" {RECORD_HEADER} "$(date +%s)" "$(hostname)"; '
        f'{body}; }} > "$t" 2>/dev/null '
        f'&& mv -f "$t" "$d/{key}{RECORD_SUFFIX}" 2>/dev/null; '
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
    return {"hooks": {event: entry(event) for event in EVENT_STATES}}


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
    try:
        payload = json.loads(body) if body.strip() else {}
    except ValueError:
        return status
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
    return status


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
        detail = _one_line(s.message, 60) if s.state == "waiting" else ""
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
    parts = []
    if waiting:
        names = [f"{s.mirror}({format_age(s.age(now))})" for s in waiting[:max_names]]
        if len(waiting) > max_names:
            names.append(f"+{len(waiting) - max_names}")
        parts.append("WAIT " + " ".join(names))
    parts.append(f"run {working}")
    if idle:
        parts.append(f"idle {idle}")
    return " | ".join(parts)


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
