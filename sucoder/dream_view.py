"""Rendered Claude session windows as admitted evidence (essay-alignment S3).

A raw Claude transcript is mostly tool output, thinking and metadata: this
mirror's largest session holds about 92 KB of user and assistant text in
6.97 MB. Authors receive admitted bytes and quotes must match them exactly
(dream_wiki.source_bundle, validate_proposal), so the compact view is itself
the evidence: ``claude-view`` admits the deterministic rendering of a
complete-line raw range and records the raw digest it came from. A render map
replaces normalized records for the editor's human-attribution check. The
``claude-jsonl`` and ``codex-jsonl`` contracts are unchanged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import time

from .dream import DreamError, _chunks, _json, _object, _text, canonical, digest
from . import dream_transcript

KIND = "claude-view"
CONTRACT = "claude-view-v1"
MAX_RAW_BYTES = 16 * 1024 * 1024
TOOL_HEAD, TOOL_TAIL, CALL_LIMIT = 240, 120, 300
REDACTED = "[REDACTED]"
# High-confidence credential shapes only; the briefs' rule against copying
# secrets still applies to anything these miss (charter, disposition 2).
PATTERNS = [re.compile(p) for p in (
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
    r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}",
    r"\bgh[pousr]_[A-Za-z0-9]{30,}",
    r"\bgithub_pat_[A-Za-z0-9_]{30,}",
    r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
    r"\bAKIA[0-9A-Z]{16}\b",
)]
LABELLED = re.compile(r"(?i)\b((?:authorization:\s*bearer|api[_-]?key|access[_-]?token|client[_-]?secret|"
                      r"password|passwd|secret)[\"']?\s*[=:]?\s*[\"']?)([^\s\"']{12,})")
# Email addresses are personal data: Claude Code puts the user's address into
# session context, and this mirror's live session renders it (E1 run record).
# Every address is redacted except SSH remote specs (git@host:...). An address
# may follow a literal \n, \r or \t escape in rendered tool input, but not any
# other backslash, so "\n@pytest.fixture" and "60-@pytest.fixture" (decorators
# in escaped commands and grep output) are not mistaken for addresses.
EMAIL = re.compile(r"(?:(?<![A-Za-z0-9._%+\\-])|(?<=\\[nrt]))(?!git@)[A-Za-z0-9._%+-]*[A-Za-z0-9_%+]"
                   r"@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")


def redact(text):
    count = 0
    for pattern in PATTERNS:
        text, n = pattern.subn(REDACTED, text)
        count += n
    text, n = LABELLED.subn(lambda m: m.group(1) + REDACTED, text)
    count += n
    text, n = EMAIL.subn(REDACTED, text)
    return text, count + n


def read_range(locator, deadline):
    """Copy one complete-line raw range of an explicit regular file."""
    _object(locator, ("path", "thread_id", "byte_start", "byte_end"))
    _text(locator["thread_id"], "thread_id")
    path = Path(_text(locator["path"], "transcript path"))
    if not path.is_absolute() or path != path.resolve():
        raise DreamError("Transcript path must be absolute and canonical, without symlinks")
    start, end = locator["byte_start"], locator["byte_end"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end or end - start > MAX_RAW_BYTES:
        raise DreamError("A session window needs 0 <= byte_start < byte_end within 16 MiB")
    if time.monotonic() >= deadline:
        raise DreamError("Transcript inspection deadline exhausted", 3)
    try:
        fd = dream_transcript._open_source(path)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise DreamError("Transcript must be a regular file")
            if start:
                stream.seek(start - 1)
                if stream.read(1) != b"\n":
                    raise DreamError("A session window must start at a complete line", 3)
            stream.seek(start)
            raw = stream.read(end - start)
            after = path.stat()
            if (len(raw) != end - start or not raw.endswith(b"\n") or path.resolve() != path or
                    (after.st_dev, after.st_ino) != (info.st_dev, info.st_ino)):
                raise DreamError("Session window shrank, is not a complete-line cut, or changed", 3)
    except OSError as exc:
        raise DreamError("Session transcript is unavailable: " + str(exc), 3) from None
    return raw


def _content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and isinstance(b.get("text"), str))
    return json.dumps(content, sort_keys=True, ensure_ascii=False)


def _clip(text, head, tail):
    data = text.encode()
    if len(data) <= head + tail + 64:
        return text
    first = data[:head].decode("utf-8", errors="ignore")
    last = data[-tail:].decode("utf-8", errors="ignore")
    return first + "\n[... " + str(len(data) - len(first.encode()) - len(last.encode())) + " bytes omitted ...]\n" + last


def _blocks(entry, offset):
    events, provenance = dream_transcript._claude(entry)
    side = "SIDECHAIN " if provenance.get("sidechain") else ""
    for event, payload in events:
        if event == "message" and isinstance(payload, dict):
            role = payload.get("role")
            body = payload.get("text") if isinstance(payload.get("text"), str) else ""
            label = "USER" if role == "user" else "AGENT"
        elif event == "summary":
            role, label = "summary", "SUMMARY (compaction; not evidence of delivery)"
            body = _content_text(payload.get("text") or payload.get("summary") or payload.get("content") or "")
            body = _clip(body, TOOL_HEAD * 4, TOOL_TAIL * 4)
        elif event == "tool_call":
            role, label = "tool", "TOOL CALL " + str(payload.get("name", "?"))
            body = _clip(json.dumps(payload.get("input"), sort_keys=True, ensure_ascii=False), CALL_LIMIT, 0)
        elif event == "tool_result":
            role = "tool"
            label = "TOOL RESULT" + (" (error)" if payload.get("is_error") else "")
            body = _clip(_content_text(payload.get("content")), TOOL_HEAD, TOOL_TAIL)
        else:
            continue  # Thinking, metadata and unknown records carry no conversation text.
        if body.strip():
            yield dict(offset=offset, label=side + label, role=role, sidechain=bool(provenance.get("sidechain")),
                       summary=bool(provenance.get("summary")) or event == "summary", body=body.strip("\n"))


def render(raw, thread_id, byte_start):
    """Deterministic text and map for a complete-line raw window."""
    if not raw or not raw.endswith(b"\n"):
        raise DreamError("Rendering requires a nonempty complete-line window")
    blocks, identities, dropped, offset = [], set(), 0, byte_start
    for line in raw.splitlines(keepends=True):
        entry = _json(line)
        if not isinstance(entry, dict):
            raise DreamError("Transcript record must be a JSON object")
        found = list(_blocks(entry, offset))
        if entry.get("sessionId") is not None:
            identities.add(entry["sessionId"])
        dropped += not found
        blocks.extend(found)
        offset += len(line)
    if identities != {thread_id}:
        raise DreamError("Session window identity is missing, ambiguous or differs from thread_id", 3)
    head = ("SESSION " + thread_id + " raw bytes " + str(byte_start) + "-" + str(byte_start + len(raw)) +
            "; " + str(dropped) + " records without conversation text omitted\n\n")
    parts, mapping, redactions, position = [head], [], 0, len(head.encode())
    for block in blocks:
        body, n = redact(block["body"])
        redactions += n
        text = "[@" + str(block["offset"]) + "] " + block["label"] + "\n" + body + "\n\n"
        size = len(text.encode())
        mapping.append(dict(start=position, end=position + size, offset=block["offset"], role=block["role"],
                            sidechain=block["sidechain"], summary=block["summary"]))
        parts.append(text)
        position += size
    return "".join(parts), dict(contract=CONTRACT, blocks=mapping, dropped_records=dropped, redactions=redactions)


def admit(unit, repository, identity, limits, deadline):
    locator = unit["locator"]
    raw = read_range(locator, deadline)
    text, mapping = render(raw, locator["thread_id"], locator["byte_start"])
    data = text.encode()
    if len(data) > limits.max_source_bytes:
        raise DreamError("Rendered session window exceeds max_source_bytes; select a narrower window")
    return dict(unit_id=unit["unit_id"], kind=KIND, shareable=unit["shareable"],
                source_id=digest(canonical([identity["repository_uuid"], KIND, locator["thread_id"]])),
                repository=repository, repository_uuid=identity["repository_uuid"],
                path=locator["path"], thread_id=locator["thread_id"],
                raw=dict(byte_start=locator["byte_start"], byte_end=locator["byte_end"], sha256=digest(raw)),
                render=dict(contract=CONTRACT, sha256=digest(canonical(mapping)), blocks=len(mapping["blocks"]),
                            dropped_records=mapping["dropped_records"], redactions=mapping["redactions"]),
                byte_start=0, byte_end=len(data), bytes=len(data), sha256=digest(data),
                chunks=_chunks(data, 0, limits.chunk_bytes,
                               (limits.max_role_invocations - 1 - limits.max_repairs) // 2))


def request_unit(unit):
    return dict(unit_id=unit["unit_id"], kind=KIND, shareable=unit["shareable"],
                locator=dict(path=unit["path"], thread_id=unit["thread_id"],
                             byte_start=unit["raw"]["byte_start"], byte_end=unit["raw"]["byte_end"]))


def snapshot(unit, deadline):
    """Re-read and re-render a frozen window; both digests must reproduce."""
    raw = read_range(request_unit(unit)["locator"], deadline)
    if digest(raw) != unit["raw"]["sha256"]:
        raise DreamError("Session window bytes differ from the frozen selection", 3)
    text, mapping = render(raw, unit["thread_id"], unit["raw"]["byte_start"])
    if digest(canonical(mapping)) != unit["render"]["sha256"]:
        raise DreamError("Session window rendering differs from the frozen plan", 3)
    return text.encode(), mapping


def shape(unit, limits):
    """Persisted-plan schema check for a view unit (dream_run._plan_shape)."""
    _object(unit, ("unit_id", "kind", "shareable", "source_id", "repository", "repository_uuid", "path",
                   "thread_id", "raw", "render", "byte_start", "byte_end", "bytes", "sha256", "chunks"))
    _object(unit["raw"], ("byte_start", "byte_end", "sha256"))
    _object(unit["render"], ("contract", "sha256", "blocks", "dropped_records", "redactions"))
    if (type(unit["shareable"]) is not bool or unit["byte_start"] != 0 or unit["render"]["contract"] != CONTRACT or
            type(unit["bytes"]) is not int or not 0 < unit["byte_end"] == unit["bytes"] <= limits.max_source_bytes or
            not isinstance(unit["chunks"], list) or not unit["chunks"]):
        raise DreamError("Invalid frozen session window")


def user_quote(mapping, citation):
    """True when a citation lies inside one main-thread user block (editor attribution)."""
    for block in mapping["blocks"]:
        if (block["start"] <= citation["byte_start"] < citation["byte_end"] <= block["end"] and
                block["role"] == "user" and not block["sidechain"] and not block["summary"]):
            return True
    return False
