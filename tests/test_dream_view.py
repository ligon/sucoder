"""Rendered session windows and the session inventory (essay-alignment S3)."""
import json
import os
from pathlib import Path
import time

import pytest

from sucoder import dream, dream_experience, dream_memory, dream_run, dream_view, dream_wiki as wiki
from tests.test_dream import git, workspace  # noqa: F401
from tests.test_dream_wiki import EditorTransport, planned

SID = "0b7c1d2e-3f40-4a5b-8c6d-7e8f90a1b2c3"
SECRET = "sk-ant-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"


def record(kind, content, **extra):
    return dict(type=kind, sessionId=SID, uuid=os.urandom(4).hex(), message=dict(role=kind, content=content), **extra)


def session(path, *records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in records))
    return path


def sample(tmp_path):
    long_output = "line of output\n" * 200 + "API_KEY=" + "Z" * 24 + "\n"
    return session(tmp_path / "s" / (SID + ".jsonl"),
        dict(type="file-history-snapshot", snapshot={}),
        record("user", "Use --session-id <uuid> when you launch; that is the maintainer's decision."),
        record("assistant", [dict(type="thinking", thinking="private reasoning"),
                             dict(type="text", text="I will check the launch flags."),
                             dict(type="tool_use", id="t1", name="Bash", input=dict(command="echo " + SECRET))]),
        record("user", [dict(type="tool_result", tool_use_id="t1", content=long_output)]),
        record("assistant", [dict(type="text", text="A forked helper thought otherwise.")], isSidechain=True),
        dict(type="summary", sessionId=SID, summary="Earlier the agent claimed REPLY 1733 was sent."))


def test_rendering_keeps_conversation_clips_tools_drops_thinking_and_redacts(tmp_path):
    raw = sample(tmp_path).read_bytes()
    text, mapping = dream_view.render(raw, SID, 0)
    assert text.startswith("SESSION " + SID + " raw bytes 0-" + str(len(raw)) + "; 1 records without")
    assert "USER\nUse --session-id <uuid> when you launch" in text
    assert "AGENT\nI will check the launch flags." in text and "private reasoning" not in text
    assert "TOOL CALL Bash" in text and SECRET not in text and "[REDACTED]" in text
    assert "bytes omitted ..." in text and "Z" * 24 not in text
    assert "SIDECHAIN AGENT\nA forked helper" in text
    assert "SUMMARY (compaction; not evidence of delivery)" in text
    assert mapping["redactions"] >= 2 and mapping["dropped_records"] == 1
    roles = [(b["role"], b["sidechain"], b["summary"]) for b in mapping["blocks"]]
    assert ("user", False, False) in roles and ("assistant", True, False) in roles and ("summary", False, True) in roles
    data = text.encode()
    for block in mapping["blocks"]:
        assert data[block["start"]:block["end"]].decode().startswith("[@" + str(block["offset"]) + "] ")
        assert raw[block["offset"]:].startswith(b"{")  # Offsets name the raw record's first byte.
    assert dream_view.render(raw, SID, 0) == (text, mapping)


def test_user_attribution_only_for_main_thread_user_blocks(tmp_path):
    raw = sample(tmp_path).read_bytes()
    text, mapping = dream_view.render(raw, SID, 0)
    data = text.encode()
    def cite(quote):
        start = data.find(quote.encode())
        return dict(byte_start=start, byte_end=start + len(quote.encode()))
    assert dream_view.user_quote(mapping, cite("that is the maintainer's decision"))
    assert not dream_view.user_quote(mapping, cite("I will check the launch flags"))
    assert not dream_view.user_quote(mapping, cite("A forked helper thought otherwise"))
    assert not dream_view.user_quote(mapping, cite("REPLY 1733 was sent"))


def test_windows_must_be_complete_lines_of_one_session(tmp_path):
    path = sample(tmp_path)
    raw = path.read_bytes()
    second = raw.index(b"\n") + 1
    third = raw.index(b"\n", second) + 1
    window = dict(path=str(path), thread_id=SID, byte_start=second, byte_end=third)
    assert dream_view.read_range(window, time.monotonic() + 5) == raw[second:third]
    for bad in (dict(window, byte_start=second + 1), dict(window, byte_end=third - 1),
                dict(window, byte_start=third, byte_end=third)):
        with pytest.raises(dream.DreamError):
            dream_view.read_range(bad, time.monotonic() + 5)
    with pytest.raises(dream.DreamError, match="identity"):
        dream_view.render(raw[second:third], "another-session", second)
    link = tmp_path / "link.jsonl"
    link.symlink_to(path)
    with pytest.raises(dream.DreamError, match="canonical"):
        dream_view.read_range(dict(window, path=str(link)), time.monotonic() + 5)


def view_plan(workspace, path, *, memory=False):
    """Two committed passages plus one rendered session window."""
    planned(workspace)
    repo, config_path, config, request = workspace
    base = "refs/heads/main"
    if memory:
        base = "refs/heads/sucoder/memory"
        dream_memory.ensure(dream.GitSource(config, "sample", time.monotonic() + 30), base,
                            git(repo, "rev-parse", "HEAD"), repo.parent, time.monotonic() + 30)
    raw = path.read_bytes()
    units = [dict(unit_id=name, kind="git", shareable=True,
                  locator=dict(repository="sample", commit="HEAD", path=name + ".org")) for name in ("evidence-a", "evidence-b")]
    units.append(dict(unit_id="session", kind="claude-view", shareable=True,
                      locator=dict(path=str(path), thread_id=SID, byte_start=0, byte_end=len(raw))))
    catalog = config_path.parent / "view-sources.json"
    catalog.write_text(json.dumps(dict(request, integration_ref=base, units=units)))
    return dream.create_plan(config, "sample", catalog, base)


def test_a_session_window_is_planned_snapshotted_and_published_with_its_raw_range(workspace):
    path = sample(workspace[0].parent)
    plan = view_plan(workspace, path, memory=True)
    unit = next(u for u in plan["units"] if u["kind"] == "claude-view")
    raw = path.read_bytes()
    assert unit["raw"] == dict(byte_start=0, byte_end=len(raw), sha256=dream.digest(raw))
    assert unit["render"]["contract"] == "claude-view-v1" and unit["bytes"] < len(raw)
    result = wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "explicit-model", transport=EditorTransport())
    assert result["state"] == "published"
    receipt = json.loads(git(workspace[0], "show", result["candidate_commit"] + ":.sucoder/wiki/receipts/" + result["run_id"] + ".json"))
    view = next(u for u in receipt["units"] if u["kind"] == "claude-view")
    assert view["raw"] == unit["raw"] and "path" not in view and str(path) not in json.dumps(receipt)
    assert SECRET not in json.dumps(receipt)
    source = dream.GitSource(workspace[2], "sample", time.monotonic() + 30)
    listed = dream_experience.consumed(source, result["candidate_commit"])
    assert listed == {SID: [[0, len(raw)]]}


def test_a_changed_window_is_refused_before_any_role_call(workspace):
    path = sample(workspace[0].parent)
    plan = view_plan(workspace, path)
    data = path.read_bytes().replace(b"launch flags", b"launch FLAGS")
    path.write_bytes(data)
    transport = EditorTransport()
    with pytest.raises(dream.DreamError, match="no longer matches"):  # Revalidation re-reads the window.
        wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "explicit-model", transport=transport)
    assert transport.calls == []


def test_notes_mode_refuses_session_windows(workspace):
    plan = view_plan(workspace, sample(workspace[0].parent))
    with pytest.raises(dream.DreamError, match="wiki mode"):
        wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "explicit-model", mode="notes",
                 transport=EditorTransport())


def test_inventory_lists_only_this_mirrors_sessions_and_marks_live_and_recent(workspace, tmp_path):
    repo, _, config, _ = workspace
    home = tmp_path / "home"
    projects = dream_experience.project_dir(home, repo)
    old = session(projects / (SID + ".jsonl"), record("user", "old work"))
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    recent = session(projects / "11111111-2222-4333-8444-555555555555.jsonl", record("user", "now"))
    session(projects / "notes.jsonl", record("user", "not a session file"))
    session(dream_experience.project_dir(home, tmp_path / "elsewhere") / "99999999-2222-4333-8444-555555555555.jsonl",
            record("user", "another project"))
    source = dream.GitSource(config, "sample", time.monotonic() + 30)
    found = dream_experience.sessions(source, home=home, live={"11111111-2222-4333-8444-555555555555"})
    assert [(s["thread_id"], s["state"]) for s in found] == [("11111111-2222-4333-8444-555555555555", "live"), (SID, "finished")]
    assert dream_experience.sessions(source, home=home)[0]["state"] == "recent"
    assert all(s["workspace"] == str(source.path) for s in found)
