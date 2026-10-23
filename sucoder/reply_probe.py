"""Find a ``sucoder message`` delivery and its reply in an agent's transcript.

docs/reply-from-transcript.org.  The screen reader (``messaging.extract_reply``)
misses replies that scrolled off a full-screen TUI, and cannot tell a message
still queued from one taken in and ignored.  Both harnesses record what the
model took in and said, in order, in a JSONL transcript; this reads it.

Standalone and stdlib-only, kept Python 3.6 compatible: the laptop ships this
file to the cluster as ``python3 -I -c "import base64,zlib;exec(...)"`` with
its arguments as one base64 JSON word (``replies.probe_command``), through the
same routes ``message`` types into.  It prints exactly one JSON line with a
schema version, and any exception becomes ``{"status": "unrecognized"}``,
never a traceback.  It reads only the transcript and mailbox files named
below and, as a last resort for finding a configuration directory, the
``CLAUDE_CONFIG_DIR`` and ``CODEX_HOME`` entries of this uid's
``/proc/*/environ`` and nothing else from those files.

The pure half (the adapters, the shared reader, identification over a given
directory tree) is what the tests import; ``main`` is the remote entry point.

Statuses: ``replied`` (a tagged reply in assistant text, or a mailbox file),
``seen`` (delivered, no reply yet; ``turn_ended`` when the agent's turn has
ended since), ``queued`` (Claude queue entry only), ``absent`` (read past the
send, no delivery), ``ambiguous`` (several transcripts hold a delivery),
``unreadable`` (another uid's, or permission denied), ``unrecognized``
(frame bytes in no shape an adapter names, or nothing proves the read went
past the send), ``no-adapter`` (no transcript identified).
"""

import base64
import calendar
import errno
import glob
import json
import os
import re
import signal
import stat
import sys
import time

SCHEMA = 1
DEFAULT_SKEW = 30.0                 # seconds of clock skew allowed against the send time
MAILBOX_MAX = 64 * 1024             # a mailbox file larger than this is ignored
TAIL_BYTES = 256 * 1024             # without the frame in a file, how much of its end proves freshness
REPLY_MAX = 16 * 1024               # probe-side bound on returned reply text (the laptop caps far lower)
ENV_KEYS = ("CLAUDE_CONFIG_DIR", "CODEX_HOME")   # all that is ever kept from /proc/*/environ
_ID = re.compile(r"[0-9a-f]{8}")
WHY_SHAPE = "frame in an entry no adapter rule names"

# What an adapter says a transcript entry is.
INPUT = "input"          # an input delivered to the model (framed or not, human or sucoder)
ENQUEUE = "enqueue"      # Claude: typed while busy, not yet taken up
TEXT = "text"            # assistant text the model wrote
TURN_END = "turn_end"    # the agent's turn ended


class ProbeTimeout(Exception):
    pass


# ----------------------------------------------------------------- matching

_PASTE_OPEN = re.compile(r"<pasted_content\b[^>]*>")


def anchored(text, header):
    """Does *text* begin with *header* (the whole ``[message <id> from ...]``)?

    Leading whitespace and one ``<pasted_content ...>`` opening tag (Claude
    delivers a long ``send-keys`` burst as a paste) are allowed before it,
    nothing else: a body that quotes someone else's frame after other text
    is not that frame's delivery (review S2).
    """
    if not isinstance(text, str) or not header:
        return False
    s = text.lstrip()
    m = _PASTE_OPEN.match(s)
    if m:
        s = s[m.end():].lstrip()
    return s.startswith(header)


def tagged_reply(text, msg_id):
    """The reply in assistant *text*: from a line ``^\\W*REPLY <id>:`` to the
    end of the block, without the tag; ``None`` without such a line."""
    m = re.search(r"^\W*REPLY %s:" % re.escape(msg_id), text, re.M)
    if not m:
        return None
    return text[m.end():].strip()


_BARE = re.compile(r"^\W*REPLY\s*:", re.M)


def bare_reply(text):
    m = _BARE.search(text)
    if not m:
        return None
    return text[m.end():].strip()


# ----------------------------------------------------------------- adapters

# Claude Code entry types that carry no delivery and no reply, though some
# copy the frame or the reply text (last-prompt, api-request-blob).
_CLAUDE_KNOWN = frozenset([
    "last-prompt", "api-request-blob", "api-request", "api-request-shape", "system", "summary",
    "file-history-snapshot", "file-history-delta", "ai-title", "agent-name", "atis-latch", "mode",
    "permission-mode", "pr-link", "custom-title", "tag",
])


def claude_events(entry):
    """What one Claude Code transcript entry is, as ``[(kind, text)]``; ``[]``
    for a known entry that matters to nobody, ``None`` for an unknown one.

    Deliveries: a ``type: user`` entry with string content (or text items)
    that is no compaction summary, and an ``attachment`` of type
    ``queued_command`` (input absorbed mid-turn).  A turn-end dequeue arrives
    as the former.  ``tool_result`` entries, sidechains and every other copy
    of the frame are not deliveries.  Assistant output is one entry per
    block; only ``text`` blocks can hold a reply, never ``tool_use`` inputs.
    """
    if not isinstance(entry, dict):
        return None
    kind = entry.get("type")
    if entry.get("isSidechain"):
        return []
    if kind == "user":
        if entry.get("isCompactSummary") or entry.get("isMeta"):
            return []
        content = (entry.get("message") or {}).get("content")
        if isinstance(content, str):
            return [(INPUT, content)]
        if isinstance(content, list):
            items = [i for i in content if isinstance(i, dict)]
            if any(i.get("type") == "tool_result" for i in items):
                return []
            texts = [i.get("text") for i in items if i.get("type") == "text" and isinstance(i.get("text"), str)]
            return [(INPUT, texts[0])] if texts else [(INPUT, "")]
        return None
    if kind == "attachment":
        attachment = entry.get("attachment") or {}
        if attachment.get("type") == "queued_command":
            prompt = attachment.get("prompt")
            if isinstance(prompt, list):
                texts = [i.get("text") for i in prompt if isinstance(i, dict) and isinstance(i.get("text"), str)]
                prompt = texts[0] if texts else ""
            return [(INPUT, prompt if isinstance(prompt, str) else "")]
        return []
    if kind == "queue-operation":
        if entry.get("operation") == "enqueue" and isinstance(entry.get("content"), str):
            return [(ENQUEUE, entry["content"])]
        return []
    if kind == "assistant":
        message = entry.get("message") or {}
        out = []
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                out.append((TEXT, block["text"]))
        if message.get("stop_reason") == "end_turn":
            out.append((TURN_END, None))
        return out
    if kind in _CLAUDE_KNOWN:
        return []
    return None


_CODEX_KNOWN = frozenset(["session_meta", "turn_context", "compacted", "token_usage_record", "world_state"])


def codex_events(entry):
    """What one Codex rollout entry is (see :func:`claude_events`).

    Delivery: ``response_item`` / ``message`` / ``role: user``.  Reply:
    ``response_item`` / ``message`` / ``role: assistant`` ``output_text`` of
    any phase (a ``commentary`` reply mid-turn counts).  ``compacted``
    replays, ``event_msg`` twins and ``agent_message`` items from
    sub-agents are neither.  ``task_complete`` ends the turn.
    """
    if not isinstance(entry, dict):
        return None
    kind = entry.get("type")
    payload = entry.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    if kind == "response_item":
        if payload.get("type") != "message":
            return []
        items = [i for i in payload.get("content") or [] if isinstance(i, dict) and isinstance(i.get("text"), str)]
        role = payload.get("role")
        if role == "user":
            texts = [i["text"] for i in items if i.get("type") in ("input_text", "text")]
            return [(INPUT, texts[0] if texts else "")]
        if role == "assistant":
            return [(TEXT, i["text"]) for i in items if i.get("type") == "output_text"]
        return []
    if kind == "event_msg":
        if payload.get("type") == "task_complete":
            return [(TURN_END, None)]
        return []
    if kind in _CODEX_KNOWN:
        return []
    return None


ADAPTERS = {"claude": claude_events, "codex": codex_events}


# ----------------------------------------------------------------- the shared reader

_STAMP = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?(Z|[+-]\d\d:?\d\d)?$")


def entry_time(entry):
    """An entry's ``timestamp`` as epoch seconds (UTC), or ``None``."""
    value = entry.get("timestamp") if isinstance(entry, dict) else None
    if not isinstance(value, str):
        return None
    m = _STAMP.match(value.strip())
    if not m:
        return None
    seconds = calendar.timegm(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S"))
    if m.group(2):
        seconds += float(m.group(2))
    zone = m.group(3)
    if zone and zone != "Z":
        sign = -1 if zone[0] == "+" else 1
        digits = zone[1:].replace(":", "")
        seconds += sign * (int(digits[:2]) * 3600 + int(digits[2:]) * 60)
    return float(seconds)


def read_transcript(data, harness, msg_id, header, sent_at, skew=DEFAULT_SKEW):
    """Read one transcript's bytes for message *msg_id*.

    Only complete, newline-terminated lines are parsed: a replica grows by
    appending, so a final line without a newline is skipped, not an error.
    The frame is found with a byte search first and JSON is parsed only
    from the line holding its first occurrence (or, without one, the file's
    tail, which can still prove the read went past the send).  File
    position is the order: Claude writes a queued attachment with its
    enqueue time after later entries.

    From the delivery on, assistant text is scanned for a tagged reply; a
    bare ``REPLY:`` counts too, since nothing else was delivered before it;
    the window closes at the next delivered input of any kind.
    """
    classify = ADAPTERS[harness]
    end = data.rfind(b"\n") + 1
    needle = ("[message %s " % msg_id).encode("ascii")
    first = data.find(needle, 0, end)
    if first < 0:
        start = max(0, end - TAIL_BYTES)
        if start:
            start = data.rfind(b"\n", 0, start) + 1
    else:
        start = data.rfind(b"\n", 0, first) + 1
    floor = sent_at - skew
    fresh = unknown = delivered = enqueued = turn_ended = False
    reply = bare = None
    for raw in data[start:end].split(b"\n"):
        if not raw.strip():
            continue
        hit = needle in raw
        try:
            entry = json.loads(raw.decode("utf-8"))
        except ValueError:
            unknown = unknown or hit
            continue
        stamp = entry_time(entry)
        if stamp is not None and stamp >= floor:
            fresh = True
        events = classify(entry)
        if events is None:
            unknown = unknown or hit
            continue
        closed = False
        for kind, text in events:
            if not delivered:
                if kind == INPUT and anchored(text, header):
                    delivered = True
                elif kind == ENQUEUE and anchored(text, header):
                    enqueued = True
                continue
            if kind == INPUT:
                if anchored(text, header):
                    continue            # the same delivery again (a twin), not a new input
                closed = True
                break
            if kind == TEXT:
                found = tagged_reply(text, msg_id)
                if found is not None:
                    reply = found
                    closed = True
                    break
                if bare is None:
                    bare = bare_reply(text)
            elif kind == TURN_END:
                turn_ended = True
        if closed:
            break
    if not fresh and not delivered and start:
        # Entries before the frame's first copy can prove the read went past the send too.
        lo = max(0, start - TAIL_BYTES)
        if lo:
            lo = data.rfind(b"\n", 0, lo) + 1
        fresh = _any_after(data[lo:start], floor)
    if reply is None and bare is not None:
        reply = bare
    if reply is not None:
        return {"status": "replied", "reply": reply[:REPLY_MAX], "source": "transcript"}
    if delivered:
        return {"status": "seen", "turn_ended": turn_ended}
    if enqueued:
        return {"status": "queued"}
    if unknown:
        return {"status": "unrecognized", "why": WHY_SHAPE}
    if fresh:
        return {"status": "absent"}
    return {"status": "unrecognized", "why": "no entry stamped after the send"}


def _any_after(chunk, floor):
    """Does any JSON line in *chunk* (whole lines) carry a timestamp at or after *floor*?"""
    for raw in chunk.split(b"\n"):
        try:
            stamp = entry_time(json.loads(raw.decode("utf-8")))
        except ValueError:
            continue
        if stamp is not None and stamp >= floor:
            return True
    return False


# ----------------------------------------------------------------- files

class Unreadable(Exception):
    pass


def read_file(path):
    """A transcript's bytes; :class:`Unreadable` for another uid's file or a
    denied read, never an empty answer that would read as ``absent``."""
    try:
        st = os.stat(path)
        if st.st_uid != os.getuid():
            raise Unreadable("%s belongs to uid %d" % (path, st.st_uid))
        with open(path, "rb") as handle:
            return handle.read()
    except (IOError, OSError) as exc:
        if getattr(exc, "errno", None) in (errno.EACCES, errno.EPERM):
            raise Unreadable("%s: permission denied" % path)
        raise


def first_entry(path):
    """The first line of a rollout, parsed (Codex ``session_meta``), or None."""
    try:
        with open(path, "rb") as handle:
            line = handle.readline(1024 * 1024)
        return json.loads(line.decode("utf-8")) if line.endswith(b"\n") else None
    except (IOError, OSError, ValueError):
        return None


def thread_source(path):
    meta = first_entry(path)
    if not isinstance(meta, dict) or meta.get("type") != "session_meta":
        return None, None
    payload = meta.get("payload") or {}
    return payload.get("thread_source"), payload.get("cwd")


def environ_dirs(proc_root="/proc"):
    """``{name: [values]}`` for :data:`ENV_KEYS` across this uid's processes.

    Only those two variables are kept from each ``environ``; everything
    else in the file (tokens, keys) is discarded as it is read (review S5).
    """
    found = dict((key, []) for key in ENV_KEYS)
    uid = os.getuid()
    try:
        names = os.listdir(proc_root)
    except OSError:
        return found
    for name in names:
        if not name.isdigit():
            continue
        try:
            if os.stat(os.path.join(proc_root, name)).st_uid != uid:
                continue
            with open(os.path.join(proc_root, name, "environ"), "rb") as handle:
                data = handle.read(1024 * 1024)
        except (IOError, OSError):
            continue
        for item in data.split(b"\0"):
            key, sep, value = item.partition(b"=")
            if not sep:
                continue
            key_text = key.decode("ascii", "replace")
            if key_text in ENV_KEYS:
                text = value.decode("utf-8", "replace")
                if text and text not in found[key_text]:
                    found[key_text].append(text)
    return found


def encode_project(cwd):
    """Claude's project directory name for *cwd*: every non-alphanumeric
    character becomes ``-``."""
    return re.sub(r"[^A-Za-z0-9]", "-", cwd)


def _paths(path):
    """*path* with ``~`` expanded, and its realpath when that differs."""
    expanded = os.path.expanduser(path)
    out = [expanded]
    real = os.path.realpath(expanded)
    if real != expanded:
        out.append(real)
    return out


def _uniq(items):
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


# ----------------------------------------------------------------- identification

def _evaluate(paths, harness, spec):
    """``[(path, result)]`` for each transcript in *paths*."""
    results = []
    for path in paths:
        try:
            data = read_file(path)
        except Unreadable as exc:
            results.append((path, {"status": "unreadable", "why": str(exc)}))
            continue
        except (IOError, OSError):
            continue
        result = read_transcript(data, harness, spec["id"], spec["header"], spec["sent_at"],
                                 spec.get("skew", DEFAULT_SKEW))
        results.append((path, result))
    return results


_RANK = {"unrecognized-frame": 4, "unreadable": 3, "absent": 2, "unrecognized": 1}


def combine(results):
    """One answer from several candidate transcripts, or None without any.

    Exactly one with a delivery wins; more than one is ``ambiguous``.
    Without a delivery the strongest of the rest: queued, then an
    unrecognized frame, unreadable, absent, and last a transcript nothing
    proves was read past the send.
    """
    if not results:
        return None
    delivered = [(p, r) for p, r in results if r["status"] in ("replied", "seen")]
    if len(delivered) == 1:
        path, result = delivered[0]
        result = dict(result)
        result["transcript"] = path
        return result
    if delivered:
        return {"status": "ambiguous", "candidates": [p for p, _ in delivered]}
    queued = [(p, r) for p, r in results if r["status"] == "queued"]
    if len(queued) > 1:
        return {"status": "ambiguous", "candidates": [p for p, _ in queued]}
    if queued:
        result = dict(queued[0][1])
        result["transcript"] = queued[0][0]
        return result

    path, result = max(results, key=lambda item: _rank(item[1]))
    result = dict(result)
    result["transcript"] = path
    return result


def _rank(result):
    key = result["status"]
    if key == "unrecognized" and result.get("why") == WHY_SHAPE:
        key = "unrecognized-frame"
    return _RANK.get(key, 0)


def strongest(answers):
    """The most informative of several non-delivery answers, or None."""
    return max(answers, key=_rank) if answers else None


def _recent(paths, floor):
    """*paths* modified at or after *floor*, newest first."""
    stamped = []
    for path in paths:
        try:
            mtime = os.stat(path).st_mtime
        except OSError:
            continue
        if mtime >= floor:
            stamped.append((mtime, path))
    return [path for _, path in sorted(stamped, reverse=True)]


def claude_config_dirs(spec, proc_root="/proc", with_environ=False):
    dirs = []
    if spec.get("claude_config_dir"):
        dirs.extend(_paths(spec["claude_config_dir"]))
    dirs.extend(_paths("~/.claude"))
    if with_environ:
        for value in environ_dirs(proc_root)["CLAUDE_CONFIG_DIR"]:
            dirs.extend(_paths(value))
    return _uniq([d for d in dirs if os.path.isdir(d)])


_DECISIVE = ("replied", "seen", "queued", "ambiguous", "unreadable")


def find_claude(spec, proc_root="/proc"):
    """Claude: the transcript named by the ``--session-id`` SuCoder chose at
    launch; else -- no id recorded, or a ``--continue`` or ``/clear`` in the
    leftover shell started a new file -- the files in the mirror's project
    directories written since the send.  Only when neither gives anything
    are the configuration directories of this uid's processes consulted.
    None when no transcript is identified."""
    floor = spec["sent_at"] - spec.get("skew", DEFAULT_SKEW)
    session_id = spec.get("claude_session_id")
    answers = []
    searched = None
    for with_environ in (False, True):
        dirs = claude_config_dirs(spec, proc_root, with_environ)
        if dirs == searched:
            break
        searched = dirs
        if isinstance(session_id, str) and re.match(r"^[0-9a-fA-F-]{36}$", session_id):
            named = []
            for config in dirs:
                named.extend(glob.glob(os.path.join(config, "projects", "*", session_id + ".jsonl")))
            answer = combine(_evaluate(_uniq(named), "claude", spec))
            if answer is not None and answer["status"] in _DECISIVE:
                return answer
            if answer is not None:
                answers.append(answer)
        projects = []
        for config in dirs:
            for cwd in spec.get("cwds") or []:
                for path in _paths(cwd):
                    projects.append(os.path.join(config, "projects", encode_project(path)))
        files = []
        for project in _uniq(projects):
            files.extend(glob.glob(os.path.join(project, "*.jsonl")))
        answer = combine(_evaluate(_recent(_uniq(files), floor), "claude", spec))
        if answer is not None and answer["status"] in _DECISIVE:
            return answer
        if answer is not None:
            answers.append(answer)
        if answers:
            break
    return strongest(answers)


def _job_matches(meta, job):
    allocation = meta.get("allocation") if isinstance(meta, dict) else None
    return isinstance(allocation, dict) and job is not None and str(allocation.get("job")) == str(job)


def managed_codex_threads(codex, floor):
    """Managed Codex (``codex_state_runner``): the runs under
    ``<durable_root>/codex/<scope>/runs/`` whose ``run.json`` names this
    allocation's job, and in them the rollouts whose ``session_meta`` says
    ``thread_source: user`` -- forked sub-agents copy the parent's
    deliveries and are never chosen.  The durable replica is read unless
    the node-local run's ``status.json`` shows it lagging and the
    node-local rollout is readable here.  ``[path]``."""
    durable = os.path.join(os.path.expanduser(codex["durable_root"]), "codex", codex["scope"])
    runs = []
    for meta_path in glob.glob(os.path.join(durable, "runs", "*", "run.json")):
        try:
            with open(meta_path, "r") as handle:
                meta = json.load(handle)
        except (IOError, OSError, ValueError):
            continue
        if _job_matches(meta, codex.get("job")):
            runs.append(os.path.dirname(meta_path))
    if not runs:
        # No run names the job (an older run.json, or the job id unknown):
        # every run of the scope, limited to rollouts written since the send.
        runs = [os.path.dirname(p) for p in glob.glob(os.path.join(durable, "runs", "*", "run.json"))]
        recent_only = True
    else:
        recent_only = False
    paths = []
    for run in runs:
        run_id = os.path.basename(run)
        status = {}
        local_run = None
        if codex.get("local_root"):
            local_run = os.path.join(codex["local_root"], "state", "codex", codex["scope"], run_id)
            try:
                with open(os.path.join(local_run, "status.json"), "r") as handle:
                    status = json.load(handle)
            except (IOError, OSError, ValueError):
                status = {}
        replica = status.get("replica") if isinstance(status, dict) else None
        threads = (replica or {}).get("threads") if isinstance(replica, dict) else None
        listed = sorted(threads) if isinstance(threads, dict) else []
        found = [os.path.basename(p)[:-len(".jsonl")]
                 for p in glob.glob(os.path.join(run, "rollouts", "*.jsonl"))]
        for tid in _uniq(listed + sorted(found)):
            durable_path = os.path.join(run, "rollouts", tid + ".jsonl")
            path = durable_path
            info = threads.get(tid) if isinstance(threads, dict) else None
            if isinstance(info, dict) and local_run:
                lag = (info.get("local_bytes") or 0) - (info.get("replicated_bytes") or 0)
                local_path = os.path.join(local_run, "home", "sessions", info.get("relative_path") or "")
                if (lag > 0 or not os.path.isfile(durable_path)) and os.path.isfile(local_path):
                    path = local_path
            if not os.path.isfile(path):
                continue
            source, _ = thread_source(path)
            if source != "user":
                continue
            paths.append(path)
    return _recent(paths, floor) if recent_only else _uniq(paths)


def _day_dirs(home, start_date, now):
    """``sessions/YYYY/MM/DD`` from the day before *start_date* (rollout
    names are local time) to tomorrow, at most 62 of them: a rollout stays
    in its creation-day directory, so "today or yesterday" is not enough."""
    try:
        start = calendar.timegm(time.strptime(start_date[:10], "%Y-%m-%d")) - 86400
    except (TypeError, ValueError):
        start = now - 2 * 86400
    start = max(start, now - 61 * 86400)
    out = []
    day = start
    while day <= now + 86400:
        out.append(os.path.join(home, "sessions", time.strftime("%Y/%m/%d", time.gmtime(day))))
        day += 86400
    return out


def unmanaged_codex_threads(spec, proc_root="/proc", now=None):
    """Unmanaged Codex: ``${CODEX_HOME:-~/.codex}/sessions/...`` rollouts
    whose ``session_meta.cwd`` is the mirror and ``thread_source`` is
    ``user`` (or absent, in versions that predate it), written since the
    send.  ``[path]``."""
    now = time.time() if now is None else now
    floor = spec["sent_at"] - spec.get("skew", DEFAULT_SKEW)
    homes = []
    if spec.get("codex_home"):
        homes.extend(_paths(spec["codex_home"]))
    homes.extend(_paths("~/.codex"))
    for value in environ_dirs(proc_root)["CODEX_HOME"]:
        homes.extend(_paths(value))
    cwds = []
    for cwd in spec.get("cwds") or []:
        cwds.extend(_paths(cwd))
    paths = []
    for home in _uniq(homes):
        for day in _day_dirs(home, spec.get("start_date"), now):
            for path in _recent(glob.glob(os.path.join(day, "rollout-*.jsonl")), floor):
                source, cwd = thread_source(path)
                if source not in ("user", None):
                    continue
                if cwd is not None and (cwd in cwds or os.path.realpath(cwd) in cwds):
                    paths.append(path)
    return _uniq(paths)


def find_codex(spec, proc_root="/proc"):
    floor = spec["sent_at"] - spec.get("skew", DEFAULT_SKEW)
    if spec.get("codex"):
        paths = managed_codex_threads(spec["codex"], floor)
    else:
        paths = unmanaged_codex_threads(spec, proc_root)
    return combine(_evaluate(paths, "codex", spec))


# ----------------------------------------------------------------- mailbox

def read_mailbox(dirs, msg_id, sent_at, skew=DEFAULT_SKEW):
    """``<dir>/<id>.reply``: a regular file (not a symlink) owned by this
    uid, modified after the send, at most :data:`MAILBOX_MAX` bytes; its
    text, or None.  Anything else is ignored, never followed."""
    if not _ID.match(msg_id or "") or len(msg_id) != 8:
        return None
    for directory in dirs:
        path = os.path.join(os.path.expanduser(directory), msg_id + ".reply")
        try:
            st = os.lstat(path)
        except OSError:
            continue
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid():
            continue
        if st.st_mtime < sent_at - skew or st.st_size > MAILBOX_MAX:
            continue
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "rb") as handle:
                data = handle.read(MAILBOX_MAX + 1)
        except (IOError, OSError):
            continue
        if len(data) > MAILBOX_MAX:
            continue
        return data.decode("utf-8", "replace").strip()
    return None


# ----------------------------------------------------------------- entry point

def probe(spec, proc_root="/proc"):
    """The answer for *spec* (see ``replies.probe_spec``), before the schema."""
    msg_id = spec.get("id")
    if not isinstance(msg_id, str) or not _ID.match(msg_id) or len(msg_id) != 8:
        raise ValueError("bad message id")
    if not isinstance(spec.get("header"), str) or not spec["header"].startswith("[message %s " % msg_id):
        raise ValueError("bad frame header")
    spec["sent_at"] = float(spec["sent_at"])
    harness = spec.get("harness")
    if harness == "claude":
        answer = find_claude(spec, proc_root)
    elif harness == "codex":
        answer = find_codex(spec, proc_root)
    else:
        answer = None
    if answer is None:
        answer = {"status": "no-adapter", "why": "no transcript identified"}
    answer = dict(answer)
    answer["harness"] = harness
    mail = read_mailbox(spec.get("mailboxes") or [], msg_id, spec["sent_at"], spec.get("skew", DEFAULT_SKEW))
    replies = []
    if answer["status"] == "replied":
        replies.append({"source": "transcript", "text": answer.get("reply") or ""})
    if mail is not None:
        replies.append({"source": "mailbox", "text": mail[:REPLY_MAX]})
        if answer["status"] != "replied":
            answer["status"] = "replied"
            answer["reply"] = mail[:REPLY_MAX]
            answer["source"] = "mailbox"
    answer["replies"] = replies
    return answer


def _alarm(signum, frame):
    raise ProbeTimeout()


def main(argv):
    out = {"schema": SCHEMA, "status": "unrecognized", "why": "no arguments"}
    try:
        if not argv:
            raise ValueError("no arguments")
        spec = json.loads(base64.b64decode(argv[0]).decode("utf-8"))
        budget = int(spec.get("budget") or 60)
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(max(1, budget))
        out = probe(spec)
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        out["schema"] = SCHEMA
    except BaseException as exc:   # noqa: B902 -- one JSON line, whatever happened
        out = {"schema": SCHEMA, "status": "unrecognized", "why": type(exc).__name__}
    sys.stdout.write(json.dumps(out, sort_keys=True) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
