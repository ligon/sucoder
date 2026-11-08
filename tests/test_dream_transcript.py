"""Full evidence normalization must retain what reply classification omits."""
import json
from pathlib import Path
import time

import pytest

from sucoder import dream, dream_run, dream_transcript as transcript
from tests.test_dream import invoke, workspace  # noqa: F401


def encoded(*records):
    return b"".join(json.dumps(record).encode() + b"\n" for record in records)


def native(kind, text="Source qualification"):
    if kind == "codex-jsonl":
        return encoded(dict(type="session_meta", payload=dict(id="thread")),
                       dict(type="response_item", payload=dict(type="message", role="user",
                            content=[dict(type="input_text", text=text)])))
    return encoded(dict(type="user", sessionId="thread", uuid="message", message=dict(content=text)))


def catalog(workspace, kind, raw=None, **locator):
    path = workspace[1].parent / "transcript.jsonl"
    path.write_bytes(native(kind) if raw is None else raw)
    request = dict(workspace[3], units=[dict(unit_id="conversation", kind=kind, shareable=True,
                   locator=dict(path=str(path), thread_id="thread", **locator))])
    return request, path


@pytest.mark.parametrize("kind", transcript.KINDS)
def test_plan_snapshot_and_appended_prefix(workspace, kind):
    request, path = catalog(workspace, kind)
    raw = path.read_bytes()
    result = invoke(workspace, request)
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    unit = plan["units"][0]
    assert unit["byte_end"] == len(raw)
    assert unit["normalization"]["contract"] == transcript.CONTRACT
    path.write_bytes(raw + b'{"later":"append"}\npartial')
    result = dream_run.snapshot(workspace[2], "sample", Path(plan["plan_path"]))
    directory = Path(plan["plan_path"]).parent
    assert (directory / result["coverage"][0]["file"]).read_bytes() == raw
    normalized = json.loads((directory / ("normalized-" + unit["sha256"] + ".json")).read_bytes())
    assert normalized["session_completion"] == "not_established"
    assert normalized["independence"] == "not_established"
    assert normalized["sha256"] == unit["sha256"]
    assert dream.digest(dream.canonical(normalized)) == unit["normalization"]["sha256"]
    with pytest.raises(dream.DreamError, match="git evidence only"):
        dream_run.run_fixture(workspace[2], "sample", Path(plan["plan_path"]), "fixture")


@pytest.mark.parametrize("kind", transcript.KINDS)
def test_changed_shrunk_or_identity_mismatched_transcript_is_not_empty(workspace, kind):
    request, path = catalog(workspace, kind)
    result = invoke(workspace, request)
    plan = json.loads(result.output)
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b"qualification", b"contradiction"))
    with pytest.raises(dream.DreamError):
        dream_run.snapshot(workspace[2], "sample", Path(plan["plan_path"]))
    path.write_bytes(raw[:10])
    with pytest.raises(dream.DreamError, match="shrank"):
        dream_run.snapshot(workspace[2], "sample", Path(plan["plan_path"]))
    path.write_bytes(raw)
    request["units"][0]["locator"]["thread_id"] = "wrong"
    assert invoke(workspace, request).exit_code == 3


@pytest.mark.parametrize("kind", transcript.KINDS)
def test_partial_tail_is_excluded_but_complete_unknown_record_is_retained(workspace, kind):
    raw = native(kind) + b'{"type":"new-kind","payload":{"x":1}}\n'
    request, path = catalog(workspace, kind, raw + b'{"partial":')
    value = json.loads(invoke(workspace, request).output)
    assert value["units"][0]["byte_end"] == len(raw)
    assert value["units"][0]["normalization"]["unknown_events"] == 1
    request["units"][0]["locator"]["byte_end"] = len(raw) - 1
    assert invoke(workspace, request).exit_code == 3


def test_codex_calls_results_compaction_and_mirrors_keep_native_provenance():
    raw = native("codex-jsonl") + encoded(
        dict(type="response_item", payload=dict(type="function_call", name="exec", call_id="c1", arguments='{"cmd":"false"}')),
        dict(type="response_item", payload=dict(type="function_call_output", call_id="c1", output="exit 1")),
        dict(type="compacted", payload=dict(message="I succeeded", replacement_history=[dict(role="assistant")])),
        dict(type="event_msg", payload=dict(type="agent_message", message="mirror")),
        dict(type="response_item", payload=dict(type="custom_tool_call", call_id="pending", input="unfinished")),
        dict(type="response_item", payload=dict(type="function_call_output", call_id="unseen", output="unknown")))
    value = transcript.normalize(raw, "codex-jsonl", "thread")
    kinds = [e["kind"] for r in value["records"] for e in r["events"]]
    assert kinds == ["identity", "message", "tool_call", "tool_result", "summary", "native_observation", "tool_call", "tool_result"]
    assert value["pending_calls"] == [dict(call_id="pending", line=7)]
    assert value["orphan_results"] == ["unseen"]
    assert value["records"][3]["events"][0]["payload"]["output"] == "exit 1"
    for record in value["records"]:
        line = raw[record["byte_start"]:record["byte_end"]]
        assert json.loads(line) == record["native"]
        assert dream.digest(line) == record["sha256"]


def test_claude_sidechains_error_results_and_summaries_are_preserved():
    raw = encoded(
        dict(type="assistant", sessionId="thread", uuid="a", parentUuid="p", isSidechain=True,
             message=dict(content=[dict(type="tool_use", id="t1", name="Bash", input=dict(command="false"))])),
        dict(type="user", sessionId="thread", parentUuid="a", message=dict(content=[
             dict(type="tool_result", tool_use_id="t1", is_error=True, content="failed")])),
        dict(type="user", sessionId="thread", isCompactSummary=True, message=dict(content="Claimed success")))
    value = transcript.normalize(raw, "claude-jsonl", "thread")
    assert value["records"][0]["provenance"]["sidechain"] is True
    assert value["records"][1]["events"][0]["payload"]["is_error"] is True
    assert value["records"][2]["events"][0]["kind"] == "summary"
    assert value["pending_calls"] == []


def test_reused_call_id_at_the_cut_is_not_completed_by_an_earlier_result():
    raw = native("codex-jsonl") + encoded(
        dict(type="response_item", payload=dict(type="function_call", call_id="same", name="exec", arguments="first")),
        dict(type="response_item", payload=dict(type="function_call_output", call_id="same", output="first result")),
        dict(type="response_item", payload=dict(type="function_call", call_id="same", name="exec", arguments="replayed or reused")))
    result = transcript.normalize(raw, "codex-jsonl", "thread")
    assert result["pending_calls"] == [dict(call_id="same", line=5)]
    assert result["ambiguous_call_ids"] == ["same"]


@pytest.mark.parametrize("raw", [b"null\n", b"{}\n", b'{"sessionId":"thread","sessionId":"other"}\n', b"\xff\n", b"not JSON\n"])
def test_invalid_or_unidentified_input_is_rejected(raw):
    with pytest.raises(dream.DreamError):
        transcript.normalize(raw, "claude-jsonl", "thread")


def test_explicit_cut_bounds_reads_and_symlinks_fifo_are_rejected(workspace, tmp_path):
    request, path = catalog(workspace, "codex-jsonl")
    locator = request["units"][0]["locator"]
    raw = path.read_bytes()
    path.write_bytes(raw + b"x" * 100000)
    assert transcript.read_prefix(dict(locator, byte_end=len(raw)), len(raw), time.monotonic() + 5) == raw
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(dream.DreamError, match="symlinks"):
        transcript.read_prefix(dict(locator, path=str(link)), len(raw), time.monotonic() + 5)
    import os
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(dream.DreamError, match="regular"):
        transcript.read_prefix(dict(locator, path=str(fifo)), len(raw), time.monotonic() + 5)


def test_copied_path_keeps_logical_source_identity(workspace):
    request, path = catalog(workspace, "codex-jsonl")
    # Fix the repository UUID so independently built plans share an identity.
    from tests.test_dream import add_registry
    add_registry(workspace[0], "1f1d8769-8c8d-48d1-bd45-694a490bc124")
    first = json.loads(invoke(workspace, request).output)["units"][0]
    copy = path.with_name("copied.jsonl")
    copy.write_bytes(path.read_bytes())
    request["units"][0]["locator"]["path"] = str(copy)
    second = json.loads(invoke(workspace, request).output)["units"][0]
    assert first["source_id"] == second["source_id"]
    assert first["sha256"] == second["sha256"]
