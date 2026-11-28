"""codex-view-v1. Only conversation items in the original rollout are evidence."""
from __future__ import annotations

import io
import re

from . import dream_transcript, dream_view
from .dream import DreamError, _json, digest
from .messaging import begins_with_frame

CONTRACT = "codex-view-v1"
CONTEXT_CLIP = (dream_view.TOOL_HEAD * 4, dream_view.TOOL_TAIL * 4)


def meta(line):
    entry = _json(line)
    if not isinstance(entry, dict) or entry.get("type") != "session_meta" or not isinstance(entry.get("payload"), dict):
        raise DreamError("Codex rollout must begin with session_meta", 3)
    return entry["payload"]


def exclusion(payload):
    if payload.get("forked_from_id") is not None:
        return "fork"
    if payload.get("thread_source") == "subagent" or (
            isinstance(payload.get("source"), dict) and "subagent" in payload["source"]):
        return "subagent"
    if payload.get("thread_source") != "user":
        return "unknown thread source"
    return None


def identity(line, thread_id):
    payload = meta(line)
    if payload.get("id") != thread_id or exclusion(payload):
        raise DreamError("Codex rollout identity differs from the requested user thread", 3)
    return dict(thread_id=thread_id, bytes=len(line), sha256=digest(line),
                interactive=payload.get("source") != "exec" and payload.get("originator") != "codex_exec")


def _name(value):
    text, _ = dream_view.redact(value if isinstance(value, str) else "?", CONTRACT)
    return re.sub(r"[^\w. @:/-]", "_", text)[:120] or "?"


def _at(values, index):
    return values[index] if isinstance(values, list) and index < len(values) else None


def _messages(payload, interactive):
    role = payload.get("role")
    metadata = payload.get("internal_chat_message_metadata_passthrough")
    metadata = metadata if isinstance(metadata, dict) else {}
    content = payload.get("content")
    for i, item in enumerate(content if isinstance(content, list) else []):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            continue
        text = item["text"]
        kind = _at(metadata.get("content_item_kinds"), i)
        if role == "assistant" and item.get("type") == "output_text":
            phase = payload.get("phase")
            label = "AGENT " + phase if phase in ("commentary", "final_answer") else "AGENT"
            yield label, role, "agent", text, None
        elif role == "developer" and item.get("type") == "input_text":
            yield "HARNESS " + _name(kind or role), role, "harness", text, CONTEXT_CLIP
        elif role == "user" and item.get("type") == "input_text":
            details = _at(metadata.get("content_item_metadata"), i)
            provenance = details.get("provenance") if isinstance(details, dict) else None
            author = provenance.get("type") if isinstance(provenance, dict) else None
            if begins_with_frame(text):
                yield "AGENT (sucoder relay)", role, "agent", text, None
            elif author == "user" and interactive:
                yield "USER", role, "human", text, None
            elif author and author != "user":
                yield "HARNESS " + _name(kind or author), role, "harness", text, CONTEXT_CLIP
            else:
                yield "USER (origin not recorded)", role, "unattributed", text, None


def pieces(entry, interactive):
    metadata = entry.get("metadata")
    if isinstance(metadata, dict) and metadata.get("inherited_user_message") is True:
        return
    events, _ = dream_transcript._codex(entry)
    for event, payload in events:
        if event == "message":
            yield from _messages(payload, interactive)
        elif event == "tool_call":
            key = "arguments" if payload["type"] == "function_call" else "input"
            yield ("TOOL CALL " + _name(payload.get("name")), "tool", "tool",
                   dream_view._content_text(payload.get(key, "")), (dream_view.CALL_LIMIT, 0))
        elif event == "tool_result":
            yield ("TOOL RESULT", "tool", "tool", dream_view._content_text(payload.get("output", "")),
                   (dream_view.TOOL_HEAD, dream_view.TOOL_TAIL))
        elif event == "summary" and isinstance(payload.get("message"), str):
            yield "SUMMARY (compaction; not evidence of delivery)", "summary", "summary", payload["message"], CONTEXT_CLIP
        elif (entry.get("type") == "response_item" and isinstance(payload, dict) and
              payload.get("type") == "agent_message"):
            content = payload.get("content")
            if isinstance(content, list):
                text = "\n".join(part if isinstance(part, str) else part["text"] for part in content
                                 if isinstance(part, str) or isinstance(part, dict) and isinstance(part.get("text"), str))
                yield ("AGENT " + _name(payload.get("author")) + " to " + _name(payload.get("recipient")),
                       "assistant", "agent", text, None)


def blocks(entry, offset, binding):
    for label, role, origin, text, clip in pieces(entry, binding["interactive"]):
        body, count = dream_view.redact(text, CONTRACT)
        if clip:
            body = dream_view._clip(body, *clip, CONTRACT)
        if body.strip():
            yield dict(offset=offset, label=label, role=role, origin=origin, body=body.strip("\n"),
                       redactions=count, sidechain=False, summary=origin == "summary")


def render(raw, thread_id, byte_start, binding):
    if not raw or not raw.endswith(b"\n"):
        raise DreamError("Rendering requires a nonempty complete-line window")
    stream = io.BytesIO(raw)
    if byte_start == 0:
        found = identity(stream.readline(), thread_id)
        if binding is not None and binding != found:
            raise DreamError("Frozen Codex session_meta changed", 3)
        binding = found
        stream.seek(0)
    if binding is None or binding["thread_id"] != thread_id:
        raise DreamError("Codex window requires the frozen file's session_meta identity", 3)
    rendered, dropped, offset = [], 0, byte_start
    for line in stream:
        entry = _json(line)
        if not isinstance(entry, dict):
            raise DreamError("Transcript record must be a JSON object")
        group = list(blocks(entry, offset, binding))
        rendered.extend(group)
        dropped += not group
        offset += len(line)
    return dream_view._format_window(rendered, dropped, thread_id, byte_start, offset, CONTRACT)
