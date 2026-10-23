"""Message replies read from the agent's transcript: the laptop half.

docs/reply-from-transcript.org.  ``message`` records every delivered frame
in a local sent-message log; ``message --wait`` and ``peek --reply`` then
run the standalone probe (``reply_probe.py``) over the recipient's existing
route and map its answer to a status and an exit code.  Pure apart from the
sent log's file I/O; the CLI does the SSH.
"""

from __future__ import annotations

import base64
import datetime as _dt
import json
import os
import re
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROBE_SCHEMA = 1
PROBE_BUDGET_SECONDS = 60      # the probe's own alarm; the transport allows 90
SKEW_SECONDS = 30.0

# Exit codes.  0 and 3 keep their meanings ("replied"; "could not look").
EXIT_NO_ANSWER = 4             # delivered (or queued, or absent), but no answer
EXIT_AMBIGUOUS = 1
EXIT_UNAVAILABLE = 3

STATUS_EXIT: Dict[str, Optional[int]] = {
    "replied": 0,
    "seen": EXIT_NO_ANSWER,
    "queued": EXIT_NO_ANSWER,
    "absent": EXIT_NO_ANSWER,
    "ambiguous": EXIT_AMBIGUOUS,
    "unreadable": EXIT_UNAVAILABLE,
    "unrecognized": EXIT_UNAVAILABLE,
    "transport": EXIT_UNAVAILABLE,  # every poll failed before the probe answered
    "no-adapter": None,             # the screen path decides, with today's exit codes
}

# Worst first: could not look, then ambiguous (or a failed delivery), then
# no answer, then success.  With several recipients the exit is the worst.
_EXIT_ORDER = {EXIT_UNAVAILABLE: 3, EXIT_AMBIGUOUS: 2, EXIT_NO_ANSWER: 1, 0: 0}

_MSG_ID = re.compile(r"\A[0-9a-f]{8}\Z")


def valid_message_id(msg_id: str) -> bool:
    return bool(_MSG_ID.match(msg_id or ""))


def worst_exit(codes: Iterable[int]) -> int:
    """The worst of *codes* in the order 3 > 1 > 4 > 0 (an unlisted code,
    such as 2, ranks above them all)."""
    worst = 0
    for code in codes:
        if _EXIT_ORDER.get(code, 9) > _EXIT_ORDER.get(worst, 9):
            worst = code
    return worst


# ----------------------------------------------------------------- sent log

def default_sent_log_path() -> Path:
    return Path("~/.sucoder/sent.jsonl").expanduser()


# Indirection the tests replace, so a run never writes the real log.
sent_log_path = default_sent_log_path


def utc_iso(epoch: float) -> str:
    return _dt.datetime.fromtimestamp(epoch, _dt.timezone.utc).isoformat().replace("+00:00", "Z")


def append_sent(entry: Dict[str, Any]) -> Path:
    """Append one line to the sent log, creating it private (0600)."""
    path = sent_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return path


def lookup_sent(msg_id: str) -> List[Dict[str, Any]]:
    """Every logged delivery of *msg_id* (one per recipient), oldest first."""
    path = sent_log_path()
    found: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                try:
                    entry = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(entry, dict) and entry.get("id") == msg_id:
                    found.append(entry)
    except FileNotFoundError:
        return []
    return found


# ----------------------------------------------------------------- identity

def transcript_identity(record, recipient, *, mirror_path: Optional[str],
                        local_root: Optional[str], token: str) -> Optional[Dict[str, Any]]:
    """What the probe needs to find *recipient*'s transcript, from its
    session *record*; None without one, which means the screen path.

    The harness comes from what the launch recorded, never from the pane
    name (managed Codex runs under ``python3``; Claude's executable is a
    versioned file).  A record only counts for the session it was written
    for: the same job, or no job for a login session.
    """
    if record is None:
        return None
    job = getattr(record, "slurm_job_id", None)
    if recipient.job_id is not None and job != recipient.job_id:
        return None
    if recipient.job_id is None and job is not None:
        return None
    owner = getattr(record, "managed_owner", None)
    harness = getattr(record, "harness", None)
    claude_id = getattr(record, "claude_session_id", None)
    identity: Dict[str, Any] = {}
    if claude_id or harness == "claude":
        identity["harness"] = "claude"
        identity["claude_session_id"] = claude_id
    elif isinstance(owner, dict) and owner.get("scope") and owner.get("durable_root"):
        identity["harness"] = "codex"
        identity["codex"] = {
            "durable_root": owner["durable_root"], "scope": owner["scope"],
            "job": None if recipient.job_id is None else str(recipient.job_id),
            "local_root": local_root,
        }
    elif harness == "codex":
        identity["harness"] = "codex"
    else:
        return None
    cwds = [p for p in (mirror_path,) if p]
    if local_root:
        cwds.append(f"{local_root.rstrip('/')}/mirrors/{token}")
    identity["cwds"] = cwds
    identity["start_date"] = (getattr(record, "created", None) or "")[:10] or None
    identity["mailboxes"] = ["~/.sucoder/mail"] + [f"{p.rstrip('/')}/.sucoder/mail" for p in cwds]
    return identity


def probe_spec(identity: Dict[str, Any], *, msg_id: str, header: str, sent_at: float) -> Dict[str, Any]:
    spec = dict(identity)
    spec.update(schema=PROBE_SCHEMA, id=msg_id, header=header, sent_at=sent_at,
                skew=SKEW_SECONDS, budget=PROBE_BUDGET_SECONDS)
    return spec


def probe_source() -> str:
    return Path(__file__).with_name("reply_probe.py").read_text(encoding="utf-8")


def probe_command(spec: Dict[str, Any], source: Optional[str] = None) -> str:
    """The shell that runs the probe on *spec* and prints its one JSON line.

    The source travels zlib-compressed and base64-encoded inside the ``-c``
    argument and the spec as one base64 JSON word, so nothing in either
    needs quoting through ``job_tmux_command``'s layers or the ``ssh -n``
    route (no stdin).  ``-I`` keeps the remote cwd and environment out of
    the import path; stderr is discarded.
    """
    code = base64.b64encode(zlib.compress((source or probe_source()).encode("utf-8"), 9)).decode("ascii")
    args = base64.b64encode(json.dumps(spec, sort_keys=True).encode("utf-8")).decode("ascii")
    return (f'python3 -I -c "import base64,zlib;exec(zlib.decompress(base64.b64decode(\'{code}\')))" '
            f"{args} 2>/dev/null")


def parse_probe_output(stdout: str) -> Optional[Dict[str, Any]]:
    """The probe's answer from its stdout (the last JSON line with a
    ``schema`` and a known ``status``), or None when it never answered."""
    for raw in reversed((stdout or "").splitlines()):
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            answer = json.loads(raw)
        except ValueError:
            continue
        if isinstance(answer, dict) and "schema" in answer and answer.get("status") in STATUS_EXIT:
            return answer
    return None
