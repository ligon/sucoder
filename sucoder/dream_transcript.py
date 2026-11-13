"""Explicit, byte-bound transcript evidence, independent of reply classification.

See implementation ledger sections 3-6. Unknown native records survive intact;
normalization never upgrades summaries, replay copies or tool text into truth.
"""
from __future__ import annotations

import os
import io
from pathlib import Path
import stat
import time

from .dream import DreamError, _chunks, _json, _object, _text, canonical, digest


KINDS = ("claude-jsonl", "codex-jsonl")
CONTRACT = "native-jsonl-evidence-v1"


def _open_source(path):
    """Pin every directory component, refusing symlink races during traversal."""
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        return os.open(path.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory)
    finally:
        os.close(directory)


def read_prefix(locator, limit, deadline):
    """Read only an explicit regular file; admit a complete-line frozen prefix."""
    _object(locator, ("path", "thread_id"), ("byte_end",))
    _text(locator["thread_id"], "thread_id")
    path = Path(_text(locator["path"], "transcript path"))
    if not path.is_absolute() or path != path.resolve():
        raise DreamError("Transcript path must be absolute and canonical, without symlinks")
    cut = locator.get("byte_end")
    if "byte_end" in locator and (type(cut) is not int or not 1 <= cut <= limit):
        raise DreamError("Transcript byte_end must be positive and fit max_source_bytes")
    if time.monotonic() >= deadline:
        raise DreamError("Transcript inspection deadline exhausted", 3)
    try:
        fd = _open_source(path)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise DreamError("Transcript must be a regular file")
            raw = stream.read(cut if cut is not None else limit + 1)
            if cut is not None:
                if len(raw) != cut or not raw.endswith(b"\n"):
                    raise DreamError("Transcript prefix shrank or byte_end is not a complete-line cut", 3)
            else:
                if len(raw) > limit:
                    raise DreamError("Transcript exceeds max_source_bytes; select an explicit byte_end")
                raw = raw[:raw.rfind(b"\n") + 1]
            if not raw:
                raise DreamError("Transcript has no complete JSONL records", 3)
            stream.seek(0)
            after = path.stat()
            if (stream.read(len(raw)) != raw or path.resolve() != path or
                    (after.st_dev, after.st_ino) != (info.st_dev, info.st_ino)):
                raise DreamError("Transcript changed while copying its prefix", 3)
    except OSError as exc:
        raise DreamError("Explicit transcript is unavailable: " + str(exc), 3) from None
    if time.monotonic() >= deadline:
        raise DreamError("Transcript inspection deadline exhausted", 3)
    return raw


def _claude(entry):
    kind = entry.get("type")
    provenance = dict(thread_id=entry.get("sessionId"), parent=entry.get("parentUuid"),
                      native_id=entry.get("uuid"), sidechain=entry.get("isSidechain", False),
                      summary=entry.get("isCompactSummary", False))
    events = []
    if kind in ("assistant", "user"):
        message = entry.get("message")
        if not isinstance(message, dict):
            return [("unknown", entry)], provenance
        content = message.get("content")
        if isinstance(content, str):
            events.append(("summary" if provenance["summary"] else "message", dict(role=kind, text=content)))
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    events.append(("unknown", block))
                    continue
                tag = block.get("type")
                event = ("tool_call" if tag == "tool_use" else "tool_result" if tag == "tool_result"
                         else "summary" if provenance["summary"] else "message" if tag == "text" else "unknown")
                events.append((event, dict(block, role=kind)))
        else:
            events.append(("unknown", entry))
    elif kind == "summary" or (kind == "system" and entry.get("subtype") == "compact_boundary"):
        events.append(("summary", entry))
    else:
        events.append(("unknown", entry))
    return events, provenance


def _codex(entry):
    kind = entry.get("type")
    payload = entry.get("payload")
    if not isinstance(payload, dict):
        return [("unknown", entry)], {}
    provenance = dict(thread_id=payload.get("id") if kind == "session_meta" else None,
                      lineage=payload.get("source"), forked_from=payload.get("forked_from_id"))
    if kind == "session_meta":
        return [("identity", payload)], provenance
    if kind == "compacted":
        return [("summary", payload)], provenance
    if kind == "response_item":
        tag = payload.get("type")
        event = ("message" if tag == "message" else "tool_call" if tag in ("function_call", "custom_tool_call")
                 else "tool_result" if tag in ("function_call_output", "custom_tool_call_output") else "unknown")
        return [(event, payload)], provenance
    # event_msg mirrors are retained as references, never independent messages.
    if kind == "event_msg":
        return [("native_observation", payload)], provenance
    return [("unknown", entry)], provenance


def normalize(raw, kind, thread_id):
    """Preserve native order, exact line ranges, raw records and pending calls."""
    if kind not in KINDS:
        raise DreamError("Unsupported transcript evidence kind")
    if not raw or not raw.endswith(b"\n"):
        raise DreamError("Normalization requires a nonempty complete-line prefix")
    records, identities, calls, seen_calls, orphans, ambiguous = [], set(), {}, set(), set(), set()
    offset = 0
    for number, line in enumerate(io.BytesIO(raw), 1):
        # JSONL boundaries are LF only, not Unicode or ASCII control separators.
        entry = _json(line)
        if not isinstance(entry, dict):
            raise DreamError("Transcript record must be a JSON object at line " + str(number))
        events, provenance = (_claude(entry) if kind == "claude-jsonl" else _codex(entry))
        identity = provenance.get("thread_id")
        if identity is not None:
            identities.add(_text(identity, "native thread identity"))
        normalized = []
        for event, payload in events:
            call_id = payload.get("call_id", payload.get("tool_use_id", payload.get("id"))) if isinstance(payload, dict) else None
            if event == "tool_call" and isinstance(call_id, str):
                if call_id in seen_calls:
                    ambiguous.add(call_id)
                seen_calls.add(call_id)
                calls.setdefault(call_id, []).append(dict(call_id=call_id, line=number))
            if event == "tool_result" and isinstance(call_id, str):
                if call_id not in calls:
                    orphans.add(call_id)
                elif call_id not in ambiguous and len(calls[call_id]) == 1:
                    del calls[call_id]
            normalized.append(dict(kind=event, payload=payload))
        records.append(dict(line=number, byte_start=offset, byte_end=offset + len(line),
                            sha256=digest(line), timestamp=entry.get("timestamp"),
                            provenance=provenance, events=normalized, native=entry))
        offset += len(line)
    if identities != {thread_id}:
        raise DreamError("Transcript identity is missing, ambiguous or differs from thread_id", 3)
    return dict(schema_version=1, contract=CONTRACT, kind=kind, thread_id=thread_id,
                sha256=digest(raw), byte_start=0, byte_end=len(raw), records=records,
                pending_calls=[item for key in sorted(calls) for item in calls[key]],
                orphan_results=sorted(orphans), ambiguous_call_ids=sorted(ambiguous), session_completion="not_established",
                independence="not_established")


def admit(unit, repository, identity, limits, deadline):
    raw = read_prefix(unit["locator"], limits.max_source_bytes, deadline)
    normalized = normalize(raw, unit["kind"], unit["locator"]["thread_id"])
    return dict(unit_id=unit["unit_id"], kind=unit["kind"], shareable=unit["shareable"],
                source_id=digest(canonical([identity["repository_uuid"], unit["kind"], normalized["thread_id"]])),
                repository=repository, repository_uuid=identity["repository_uuid"],
                path=unit["locator"]["path"], thread_id=normalized["thread_id"],
                byte_start=0, byte_end=len(raw), bytes=len(raw), sha256=digest(raw),
                normalization=dict(contract=CONTRACT, sha256=digest(canonical(normalized)),
                                   records=len(normalized["records"]),
                                   unknown_events=sum(e["kind"] == "unknown" for r in normalized["records"] for e in r["events"])),
                chunks=_chunks(raw, 0, limits.chunk_bytes,
                               (limits.max_role_invocations - 1 - limits.max_repairs) // 2))


def request_unit(unit):
    """Reconstruct only the admitted frozen prefix, never a later append."""
    return dict(unit_id=unit["unit_id"], kind=unit["kind"], shareable=unit["shareable"],
                locator=dict(path=unit["path"], thread_id=unit["thread_id"], byte_end=unit["byte_end"]))
