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
from .messaging import begins_with_frame

KIND = "claude-view"
# Each version is frozen as issued so saved plans, receipts and evidence handles
# replay byte for byte. Version 2 differs from 1 only for a zero tail (issue 119):
# version 1's data[-0:] kept a tool call's whole input after its clipped head.
# Version 3 records who wrote each block (origin) and redacts a block's whole body,
# with the PATTERNS_V3 family below, before clipping it.
CONTRACT_V1 = "claude-view-v1"
CONTRACT_V2 = "claude-view-v2"
CONTRACT_V3 = "claude-view-v3"
CONTRACT_V4 = "claude-view-v4"  # Typed peer messages have agent origin.
CONTRACT = CONTRACT_V4
CONTRACTS = (CONTRACT_V1, CONTRACT_V2, CONTRACT_V3, CONTRACT_V4)
LEGACY = (CONTRACT_V1, CONTRACT_V2)
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


# Version 3 patterns. An alphanumeric lookbehind replaces \b, so a label or token
# after an underscore is caught (GEMINI_API_KEY=...). A private key whose END line
# lies outside the text is redacted to its end. A labelled value needs an explicit
# = or :, and one that starts like a call or subscript ("secrets.token_hex(6)",
# "os.environ[...]") is code, not a secret.
_EDGE = r"(?<![A-Za-z0-9])"
PATTERNS_V3 = [re.compile(p) for p in (
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)",
    _EDGE + r"sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}",
    _EDGE + r"[rsp]k_(?:live|test)_[A-Za-z0-9]{16,}",
    _EDGE + r"gh[pousr]_[A-Za-z0-9]{30,}",
    _EDGE + r"github_pat_[A-Za-z0-9_]{30,}",
    _EDGE + r"glpat-[A-Za-z0-9_-]{20,}",
    _EDGE + r"xox[abprs]-[A-Za-z0-9-]{10,}",
    _EDGE + r"(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Za-z0-9])",
    _EDGE + r"AIza[0-9A-Za-z_-]{35}",
    _EDGE + r"ya29\.[0-9A-Za-z_-]{20,}",
    _EDGE + r"(?:hf|npm)_[A-Za-z0-9]{30,}",
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
    r"(?i)" + _EDGE + r"bearer\s+[A-Za-z0-9._~+/-]{16,}=*",
)]
_LABEL = (r"(?:api[_-]?key|access[_-]?key|access[_-]?token|auth[_-]?token|refresh[_-]?token|client[_-]?secret|"
          r"secret[_-]?key|token|password|passwd|secret)")
LABELLED_V3 = [re.compile(p) for p in (
    r"(?i)" + _EDGE + r"((?:[A-Za-z0-9]+[_-])*" + _LABEL + r"(?:\\?[\"'])?\s*[=:]\s*(?:\\?[\"'])?)(?![A-Za-z_][\w.]*[(\[])([^\s\"'\\]{8,})",
    r"(?i)(--(?:api[_-]?key|access[_-]?token|auth[_-]?token|token|password|passwd|secret)(?:=|\s+))([^\s\"']{8,})",
    r"(?i)([?&](?:key|api[_-]?key|token|access[_-]?token|password|sig|signature)=)([^&\s\"'#]{8,})",
)]
USERINFO_V3 = re.compile(r"(?<=://)[^\s/:@]+:[^\s/@]+(?=@)")


def redact(text, contract=CONTRACT_V1):
    count = 0
    if contract in LEGACY:
        for pattern in PATTERNS:
            text, n = pattern.subn(REDACTED, text)
            count += n
        text, n = LABELLED.subn(lambda m: m.group(1) + REDACTED, text)
        count += n
    else:
        for pattern in PATTERNS_V3:
            text, n = pattern.subn(REDACTED, text)
            count += n
        for pattern in LABELLED_V3:
            text, n = pattern.subn(lambda m: m.group(1) + REDACTED, text)
            count += n
        text, n = USERINFO_V3.subn(REDACTED, text)
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


def _clip(text, head, tail, contract=CONTRACT_V1):
    data = text.encode()
    if len(data) <= head + tail + 64:
        return text
    first = data[:head].decode("utf-8", errors="ignore")
    # Version 1 keeps data[-0:], the whole body, for a zero tail; later versions keep no tail.
    last = data[-tail:].decode("utf-8", errors="ignore") if tail or contract == CONTRACT_V1 else ""
    return first + "\n[... " + str(len(data) - len(first.encode()) - len(last.encode())) + " bytes omitted ...]\n" + last


def _contract(contract):
    if contract not in CONTRACTS:
        raise DreamError("Unknown session rendering contract", 3)
    return contract


def _blocks(entry, offset, contract=CONTRACT_V1):
    return _blocks_v1(entry, offset, contract) if contract in LEGACY else _blocks_v3(entry, offset, contract)


# Claude Code appends these to user turns; their text is the harness's, not the human's.
REMINDER = re.compile(r"<system-reminder>[\s\S]*?(?:</system-reminder>|\Z)")
COMMAND_PREFIXES = ("<command-name>", "<command-message>", "<command-args>", "<local-command-stdout>",
                    "<local-command-stderr>", "<local-command-caveat>", "<bash-input>", "<bash-stdout>",
                    "<bash-stderr>", "<user-prompt-submit-hook>")


def _user_origin(entry, text):
    """Who wrote a user-turn text in version 3, from the transcript's own record fields.

    Claude Code marks typed messages origin={"kind": "human"} and its own injections
    with another kind; meta records and command echoes carry none. A record without
    an origin is unattributed, never human.
    """
    if entry.get("isMeta"):
        return "harness", "meta"
    origin = entry.get("origin")
    if isinstance(origin, dict) and isinstance(origin.get("kind"), str):
        if origin["kind"] == "human":
            return "human", ""
        return "harness", re.sub(r"[^a-z0-9-]", "", origin["kind"].lower())[:40] or "other"
    if text.lstrip().startswith(COMMAND_PREFIXES):
        return "harness", "command"
    return "unattributed", ""


def _blocks_v3(entry, offset, contract=CONTRACT_V3):
    """Origin and redaction for v3 and v4, including relay attribution in v4."""
    events, provenance = dream_transcript._claude(entry)
    sidechain = bool(provenance.get("sidechain"))
    side = "SIDECHAIN " if sidechain else ""
    for event, payload in events:
        pieces = []  # (label, role, origin, text, clip)
        if event == "message" and isinstance(payload, dict):
            role = payload.get("role")
            text = payload.get("text") if isinstance(payload.get("text"), str) else ""
            if role != "user":
                pieces.append(("AGENT", role, "agent", text, None))
            else:
                origin, kind = _user_origin(entry, text)
                if origin == "harness":
                    pieces.append(("HARNESS " + kind, role, "harness", text, None))
                else:
                    label = "USER" if origin == "human" else "USER (origin not recorded)"
                    if contract == CONTRACT_V4 and begins_with_frame(text):
                        origin, label = "agent", "AGENT (sucoder relay)"
                    at = 0
                    for match in REMINDER.finditer(text):
                        pieces.append((label, role, origin, text[at:match.start()], None))
                        pieces.append(("HARNESS system-reminder", role, "harness", match.group(0),
                                       (TOOL_HEAD * 4, TOOL_TAIL * 4)))
                        at = match.end()
                    pieces.append((label, role, origin, text[at:], None))
        elif event == "summary":
            text = _content_text(payload.get("text") or payload.get("summary") or payload.get("content") or "")
            pieces.append(("SUMMARY (compaction; not evidence of delivery)", "summary", "summary", text,
                           (TOOL_HEAD * 4, TOOL_TAIL * 4)))
        elif event == "tool_call":
            pieces.append(("TOOL CALL " + str(payload.get("name", "?")), "tool", "tool",
                           json.dumps(payload.get("input"), sort_keys=True, ensure_ascii=False), (CALL_LIMIT, 0)))
        elif event == "tool_result":
            pieces.append(("TOOL RESULT" + (" (error)" if payload.get("is_error") else ""), "tool", "tool",
                           _content_text(payload.get("content")), (TOOL_HEAD, TOOL_TAIL)))
        else:
            continue  # Thinking, metadata and unknown records carry no conversation text.
        for label, role, origin, text, clip in pieces:
            body, redactions = redact(text, contract)
            if clip is not None:
                body = _clip(body, clip[0], clip[1], contract)
            if body.strip():
                yield dict(offset=offset, label=side + label, role=role, sidechain=sidechain,
                           summary=bool(provenance.get("summary")) or event == "summary", origin=origin,
                           body=body.strip("\n"), redactions=redactions)


def _blocks_v1(entry, offset, contract):
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
            body = _clip(body, TOOL_HEAD * 4, TOOL_TAIL * 4, contract)
        elif event == "tool_call":
            role, label = "tool", "TOOL CALL " + str(payload.get("name", "?"))
            body = _clip(json.dumps(payload.get("input"), sort_keys=True, ensure_ascii=False), CALL_LIMIT, 0, contract)
        elif event == "tool_result":
            role = "tool"
            label = "TOOL RESULT" + (" (error)" if payload.get("is_error") else "")
            body = _clip(_content_text(payload.get("content")), TOOL_HEAD, TOOL_TAIL, contract)
        else:
            continue  # Thinking, metadata and unknown records carry no conversation text.
        if body.strip():
            yield dict(offset=offset, label=side + label, role=role, sidechain=bool(provenance.get("sidechain")),
                       summary=bool(provenance.get("summary")) or event == "summary", body=body.strip("\n"))


def render(raw, thread_id, byte_start, contract=CONTRACT_V1, *, binding=None):
    """Deterministic text and map for a complete-line raw window under one rendering contract."""
    from . import dream_codex_view
    if contract == dream_codex_view.CONTRACT:
        return dream_codex_view.render(raw, thread_id, byte_start, binding)
    _contract(contract)
    if not raw or not raw.endswith(b"\n"):
        raise DreamError("Rendering requires a nonempty complete-line window")
    blocks, identities, dropped, offset = [], set(), 0, byte_start
    for line in raw.splitlines(keepends=True):
        entry = _json(line)
        if not isinstance(entry, dict):
            raise DreamError("Transcript record must be a JSON object")
        found = list(_blocks(entry, offset, contract))
        if entry.get("sessionId") is not None:
            identities.add(entry["sessionId"])
        dropped += not found
        blocks.extend(found)
        offset += len(line)
    if identities != {thread_id}:
        raise DreamError("Session window identity is missing, ambiguous or differs from thread_id", 3)
    return _format_window(blocks, dropped, thread_id, byte_start, offset, contract)


def _format_window(blocks, dropped, thread_id, byte_start, byte_end, contract):
    head = ("SESSION " + thread_id + " raw bytes " + str(byte_start) + "-" + str(byte_end) +
            "; " + str(dropped) + " records without conversation text omitted\n\n")
    parts, mapping, redactions, position = [head], [], 0, len(head.encode())
    for block in blocks:
        # Versions 1 and 2 redact the clipped body; version 3 blocks arrive redacted before clipping.
        body, n = redact(block["body"]) if contract in LEGACY else (block["body"], block["redactions"])
        redactions += n
        text = "[@" + str(block["offset"]) + "] " + block["label"] + "\n" + body + "\n\n"
        size = len(text.encode())
        mapping.append(dict(start=position, end=position + size, offset=block["offset"], role=block["role"],
                            sidechain=block["sidechain"], summary=block["summary"]))
        if contract not in LEGACY:
            mapping[-1]["origin"] = block["origin"]
        parts.append(text)
        position += size
    return "".join(parts), dict(contract=contract, blocks=mapping, dropped_records=dropped, redactions=redactions)


def admit(unit, repository, identity, limits, deadline):
    locator = unit["locator"]
    # A request without a rendering contract is version 1, so saved requests rebuild unchanged.
    contract = _contract(unit.get("render_contract", CONTRACT_V1))
    if "processing" in unit:
        from . import dream_isolate
        value = dream_isolate.run(lambda: _processed(locator, deadline, contract), deadline, unit["processing"])
        text, mapping, raw_digest = value["text"], value["mapping"], value["raw_sha256"]
    else:
        raw = read_range(locator, deadline)
        text, mapping = render(raw, locator["thread_id"], locator["byte_start"], contract)
        raw_digest = digest(raw)
    if "expected_raw_sha256" in unit and unit["expected_raw_sha256"] != raw_digest:
        raise DreamError("Session window bytes differ from the frozen discovery selection", 3)
    data = text.encode()
    if len(data) > limits.max_source_bytes:
        raise DreamError("Rendered session window exceeds max_source_bytes; select a narrower window")
    result = dict(unit_id=unit["unit_id"], kind=KIND, shareable=unit["shareable"],
                source_id=digest(canonical([identity["repository_uuid"], KIND, locator["thread_id"]])),
                repository=repository, repository_uuid=identity["repository_uuid"],
                path=locator["path"], thread_id=locator["thread_id"],
                raw=dict(byte_start=locator["byte_start"], byte_end=locator["byte_end"], sha256=raw_digest),
                render=dict(contract=contract, sha256=digest(canonical(mapping)), blocks=len(mapping["blocks"]),
                            dropped_records=mapping["dropped_records"], redactions=mapping["redactions"]),
                byte_start=0, byte_end=len(data), bytes=len(data), sha256=digest(data),
                chunks=_chunks(data, 0, limits.chunk_bytes,
                               (limits.max_role_invocations - 1 - limits.max_repairs) // 2))
    if "processing" in unit:
        result["processing"] = unit["processing"]
    return result


def _processed(locator, deadline, contract=CONTRACT_V1):
    """Run only inside a bounded worker for explicitly versioned units."""
    raw = read_range(locator, deadline)
    text, mapping = render(raw, locator["thread_id"], locator["byte_start"], contract)
    return dict(text=text, mapping=mapping, raw_sha256=digest(raw))


def request_unit(unit):
    result = dict(unit_id=unit["unit_id"], kind=KIND, shareable=unit["shareable"],
                locator=dict(path=unit["path"], thread_id=unit["thread_id"],
                             byte_start=unit["raw"]["byte_start"], byte_end=unit["raw"]["byte_end"]))
    if "processing" in unit:
        result["processing"] = unit["processing"]
        result["expected_raw_sha256"] = unit["raw"]["sha256"]
    if unit["render"]["contract"] != CONTRACT_V1:
        result["render_contract"] = unit["render"]["contract"]
    return result


def snapshot(unit, deadline):
    """Re-read and re-render a frozen window under its own contract; both digests must reproduce."""
    contract = _contract(unit["render"]["contract"])
    if "processing" in unit:
        from . import dream_isolate
        value = dream_isolate.run(lambda: _processed(request_unit(unit)["locator"], deadline, contract),
                                  deadline, unit["processing"])
        if value["raw_sha256"] != unit["raw"]["sha256"]:
            raise DreamError("Session window bytes differ from the frozen selection", 3)
        if digest(canonical(value["mapping"])) != unit["render"]["sha256"]:
            raise DreamError("Session window rendering differs from the frozen plan", 3)
        return value["text"].encode(), value["mapping"]
    raw = read_range(request_unit(unit)["locator"], deadline)
    if digest(raw) != unit["raw"]["sha256"]:
        raise DreamError("Session window bytes differ from the frozen selection", 3)
    text, mapping = render(raw, unit["thread_id"], unit["raw"]["byte_start"], contract)
    if digest(canonical(mapping)) != unit["render"]["sha256"]:
        raise DreamError("Session window rendering differs from the frozen plan", 3)
    return text.encode(), mapping


def shape(unit, limits):
    """Persisted-plan schema check for a view unit (dream_run._plan_shape)."""
    _object(unit, ("unit_id", "kind", "shareable", "source_id", "repository", "repository_uuid", "path",
                   "thread_id", "raw", "render", "byte_start", "byte_end", "bytes", "sha256", "chunks"), ("processing",))
    if "processing" in unit:
        from . import dream_isolate
        dream_isolate.validate(unit["processing"])
    _object(unit["raw"], ("byte_start", "byte_end", "sha256"))
    _object(unit["render"], ("contract", "sha256", "blocks", "dropped_records", "redactions"))
    if (type(unit["shareable"]) is not bool or unit["byte_start"] != 0 or unit["render"]["contract"] not in CONTRACTS or
            type(unit["bytes"]) is not int or not 0 < unit["byte_end"] == unit["bytes"] <= limits.max_source_bytes or
            not isinstance(unit["chunks"], list) or not unit["chunks"]):
        raise DreamError("Invalid frozen session window")


def user_quote(mapping, citation):
    """True when a citation lies inside one main-thread user block (editor attribution)."""
    for block in mapping["blocks"]:
        # Version 3 maps record an origin; only human text counts. Earlier maps have none.
        if (block["start"] <= citation["byte_start"] < citation["byte_end"] <= block["end"] and
                block["role"] == "user" and block.get("origin", "human") == "human" and
                not block["sidechain"] and not block["summary"]):
            return True
    return False
