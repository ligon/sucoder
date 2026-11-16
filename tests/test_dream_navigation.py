"""Adversarial version 3 source, cursor and processing cases; no inference."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from sucoder import dream, dream_discover as discover, dream_inventory, dream_isolate
from sucoder import dream_navigation, dream_session_scan as scan, dream_view
from tests.test_dream_discover import Source, request


SID = "6d1f0c2a-9b3e-4c5d-8e7f-0a1b2c3d4e5f"
OTHER = "7d1f0c2a-9b3e-4c5d-8e7f-0a1b2c3d4e5f"


def line(body, sid=SID, **extra):
    return (json.dumps(dict(type="user", sessionId=sid, message=dict(role="user", content=body), **extra)) + "\n").encode()


def inventory(tmp_path, raw, *, second=None, limits=None, processing=None):
    path = tmp_path / (SID + ".jsonl")
    path.write_bytes(raw)
    rows = [dict(path=str(path), thread_id=SID, bytes=len(raw), state="finished", modified=0, consumed=[])]
    if second is not None:
        other = tmp_path / (OTHER + ".jsonl")
        other.write_bytes(second)
        rows.append(dict(path=str(other), thread_id=OTHER, bytes=len(second), state="finished", modified=0, consumed=[]))
    source = Source({"note.txt": b"committed\n"})
    source.deadline = time.monotonic() + 30
    return discover.Inventory(source, "pinned", sessions=rows, navigation=3, limits=limits, processing=processing), path


def continue_request(value, action="search", path="@sessions", query="needle"):
    return request(action, path, query=query, cursor=value["next_cursor"])


def test_search_beyond_the_old_whole_session_cutoff_and_raw_read_bounds(tmp_path, monkeypatch):
    filler = (json.dumps(dict(type="attachment", sessionId=SID, attachment=dict(content="x" * (1024 * 1024)))) + "\n").encode()
    raw = filler * 20 + line("needle: the later human decision")
    inv, path = inventory(tmp_path, raw)
    calls = []
    original = os.fdopen

    class Stream:
        def __init__(self, stream): self.stream = stream
        def read(self, size=-1):
            assert 0 < size <= scan.LIMITS["chunk_bytes"]
            calls.append(size)
            return self.stream.read(size)
        def __getattr__(self, name): return getattr(self.stream, name)

    monkeypatch.setattr(scan.os, "fdopen", lambda *a, **kw: Stream(original(*a, **kw)))
    state = dict(session=0, position=0, range_end=0, block=0, record_sha256="", align=False, skip_start=-1, had_gaps=False)
    page = scan.page(list(inv.sessions.values()), "search", "needle", state, 16384,
                     time.monotonic() + 10, scan.LIMITS)
    assert not page["observation"]["matches"]
    assert page["observation"]["state"] == "incomplete"
    assert sum(calls) <= scan.LIMITS["scan_bytes"]
    assert page["continuation"]["position"] > 0
    later = scan.page(list(inv.sessions.values()), "search", "needle", page["continuation"], 16384,
                      time.monotonic() + 10, scan.LIMITS)
    assert later["observation"]["state"] == "complete_unfrozen"
    hit = later["observation"]["matches"][0]
    assert hit["offset"] > 20_550_000 and "later human decision" in hit["text"]
    assert later["receipts"][0]["byte_end"] <= path.stat().st_size


def test_equal_length_revision_swap_never_claims_coherent_absence(tmp_path):
    a, b = line("absent") + line("needle"), line("needle") + line("absent")
    assert len(a) == len(b)
    inv, path = inventory(tmp_path, a, limits=dict(scan.LIMITS, records=1))
    first = inv.lookup(request("search", "@sessions", query="needle", cursor=""), 16384)
    assert first["state"] == "incomplete" and not first["matches"]
    first_receipts = copy.deepcopy(inv.navigator.last["receipts"])
    path.write_bytes(b)
    second = inv.lookup(continue_request(first), 16384)
    # A record-count stop can require a final empty end-of-scope page.
    while second["next_cursor"]:
        assert not second["matches"]
        second = inv.lookup(continue_request(second), 16384)
    assert second["state"] == "complete_unfrozen"
    assert "does not establish query absence" in second["source_consistency"]
    assert first_receipts[0]["sha256"] == dream.digest(a)


def test_cursor_resumes_blocks_and_rejects_a_raw_only_change(tmp_path):
    raw = (json.dumps(dict(type="assistant", sessionId=SID, ignored="old",
        message=dict(role="assistant", content=[dict(type="text", text="needle " + "x" * 500) for _ in range(12)]))) + "\n").encode()
    inv, path = inventory(tmp_path, raw)
    first = inv.lookup(request("search", "@sessions", query="needle", cursor=""), 2048)
    assert first["state"] == "incomplete" and first["matches"] and first["next_cursor"]
    body = inv.navigator.cursors[first["next_cursor"]]
    assert body["state"]["block"] > 0 and body["state"]["record_sha256"] == dream.digest(raw)
    path.write_bytes(raw.replace(b'"old"', b'"NEW"'))
    assert dream_view.render(raw, SID, 0) == dream_view.render(path.read_bytes(), SID, 0)
    changed = inv.lookup(continue_request(first), 2048)
    assert changed["state"] == "continued_record_changed" and not changed["next_cursor"]


def test_oversized_records_advance_across_pages_and_leave_gaps(tmp_path):
    limits = dict(scan.LIMITS, scan_bytes=131072, record_bytes=65536)
    inv, _ = inventory(tmp_path, line("x" * 400000) + line("needle after oversized record"), limits=limits)
    page = inv.lookup(request("search", "@sessions", query="needle", cursor=""), 16384)
    positions = []
    found = []
    for _ in range(10):
        found.extend(page.get("matches", []))
        assert page["raw_scanned_bytes"] <= 131072
        if not page["next_cursor"]:
            break
        positions.append(inv.navigator.cursors[page["next_cursor"]]["state"]["position"])
        page = inv.lookup(continue_request(page), 16384)
    else:
        pytest.fail("oversized-record cursor did not finish")
    assert positions == sorted(set(positions)) and positions[0] > 0
    assert page["state"] == "complete_with_gaps" and "after oversized" in found[0]["text"]


def test_wrong_cursor_binding_is_rejected_before_opening_a_source(tmp_path, monkeypatch):
    inv, _ = inventory(tmp_path, line("one") + line("two"), limits=dict(scan.LIMITS, records=1))
    first = inv.lookup(request("search", "@sessions", query="needle", cursor=""), 16384)
    def forbidden(*args): raise AssertionError("invalid cursor opened a source")
    monkeypatch.setattr(scan.dream_transcript, "_open_source", forbidden)
    assert inv.lookup(continue_request(first, query="other"), 16384)["state"] == "invalid_request"
    assert inv.lookup(request("search", "@sessions", query="needle", start=1, cursor=""), 16384)["state"] == "invalid_request"


def test_reads_and_selection_align_without_reading_the_prefix(tmp_path):
    attachment = (json.dumps(dict(type="attachment", sessionId=SID, attachment=dict(content="x" * 143000))) + "\n").encode()
    before = line("before")
    inv, path = inventory(tmp_path, before + attachment + line("later needle"))
    lo, hi = len(before), len(before) + len(attachment)
    value = inv.lookup(request("read", "@sessions/" + SID, start=lo, end=lo + 1000, cursor=""), 16384)
    assert (value["byte_start"], value["byte_end"]) == (lo, hi)
    assert "1 records without conversation text omitted" in value["text"]
    with path.open("ab") as stream:
        stream.write(line("appended outside pinned prefix"))
    after = inv.lookup(request("read", "@sessions/" + SID, start=hi, cursor=""), 16384)
    assert "later needle" in after["text"] and "appended" not in after["text"]
    selected = inv.selected([dict(path="@sessions/" + SID, start=lo, end=lo + 1000)])
    assert selected[0]["locator"]["byte_end"] == hi
    assert selected[0]["processing"]["contract"] == dream_isolate.POLICY["contract"]


def test_timeout_terminates_a_multi_session_lookup_with_unknown_reads(tmp_path):
    policy = dict(dream_isolate.POLICY, max_seconds=0.08)
    inv, _ = inventory(tmp_path, line("-----BEGIN A PRIVATE KEY-----\n" * 4000),
                       second=line("needle in later session", OTHER), processing=policy)
    started = time.monotonic()
    value = inv.lookup(request("search", "@sessions", query="needle", cursor=""), 16384)
    assert time.monotonic() - started < 2
    assert value["state"] == "processing_deadline" and value["unsearched_scope"] == "@sessions"
    assert value["raw_scanned_bytes"] is None and value["actual_reads"] == "unknown" and not value["next_cursor"]
    later = inv.lookup(request("search", "@sessions/" + OTHER, query="needle", cursor=""), 16384)
    assert later["state"] == "complete_unfrozen" and later["matches"]


def test_worker_watchdog_reaps_hung_child_under_an_external_timeout():
    script = '''import os, time
from sucoder import dream_isolate as isolate
policy = dict(isolate.POLICY, max_seconds=0.05)
try:
    isolate.run(lambda: time.sleep(10), time.monotonic()+10, policy)
except isolate.ProcessingError as exc:
    assert exc.reason == 'processing_deadline'
else:
    raise AssertionError('worker escaped deadline')
try:
    os.waitpid(-1, os.WNOHANG)
except ChildProcessError:
    pass
else:
    raise AssertionError('worker was not reaped')
'''
    subprocess.run([sys.executable, "-c", script], check=True, timeout=3)


def test_worker_caps_output_and_additional_memory():
    with pytest.raises(dream_isolate.ProcessingError, match="processing_output_limit"):
        dream_isolate.run(lambda: "x" * 1000, time.monotonic() + 2,
                          dict(dream_isolate.POLICY, max_output_bytes=128))
    with pytest.raises(dream_isolate.ProcessingError, match="processing_memory_limit"):
        dream_isolate.run(lambda: bytearray(64 * 1024 * 1024), time.monotonic() + 2,
                          dict(dream_isolate.POLICY, memory_headroom=8 * 1024 * 1024))


def test_new_admission_and_snapshot_are_interruptible_and_old_bytes_are_preserved(tmp_path):
    inv, path = inventory(tmp_path, line("ordinary human evidence"))
    unit = dict(unit_id="session", kind=dream_view.KIND, shareable=False,
        locator=dict(path=str(path), thread_id=SID, byte_start=0, byte_end=path.stat().st_size))
    limits = dream.Limits.parse({})
    identity = dict(repository_uuid="a1a745a8-8239-4f92-ab1d-bcd22ee9ed07")
    legacy = dream_view.admit(unit, "sample", identity, limits, time.monotonic() + 3)
    bounded = dream_view.admit(dict(unit, processing=dream_isolate.POLICY), "sample", identity, limits, time.monotonic() + 3)
    assert {k: v for k, v in bounded.items() if k != "processing"} == legacy
    assert dream_view.snapshot(bounded, time.monotonic() + 3) == dream_view.snapshot(legacy, time.monotonic() + 3)
    path.write_bytes(line("-----BEGIN A PRIVATE KEY-----\n" * 4000))
    unit["locator"]["byte_end"] = path.stat().st_size
    policy = dict(dream_isolate.POLICY, max_seconds=0.08)
    for function in (
        lambda: dream_view.admit(dict(unit, processing=policy), "sample", identity, limits, time.monotonic() + 2),
        lambda: dream_view.snapshot(dict(bounded, processing=policy,
            raw=dict(byte_start=0, byte_end=path.stat().st_size, sha256=dream.digest(path.read_bytes()))), time.monotonic() + 2),
        lambda: inv.navigator.window(dict(inv.sessions[SID], bytes=path.stat().st_size), 0, path.stat().st_size, 65536),
    ):
        inv.navigator.processing = policy
        with pytest.raises(dream_isolate.ProcessingError, match="processing_deadline"):
            function()


def test_expired_inventory_does_not_enumerate_and_invalid_entries_spend_a_cap(tmp_path, monkeypatch):
    from sucoder import dream_experience as experience, dream_sessions
    source = type("Source", (), {"path": tmp_path, "deadline": time.monotonic() - 1})()
    def forbidden(*args): raise AssertionError("expired inventory enumerated")
    monkeypatch.setattr(dream_inventory.os, "scandir", forbidden)
    args = dict(home=tmp_path, quiet_minutes=30, live=(), memory_commit=None, config=None, limits=scan.LIMITS)
    value = dream_inventory.collect(source, deadline=source.deadline, **args)
    assert value["coverage"]["entries"] == 0 and value["coverage"]["state"] == "incomplete"
    monkeypatch.undo()
    directory = experience.project_dir(tmp_path, tmp_path)
    directory.mkdir(parents=True)
    for n in range(20):
        (directory / (str(n) + ".txt")).write_text("invalid candidate")
    monkeypatch.setattr(experience, "workspaces", lambda source: [str(tmp_path)])
    monkeypatch.setattr(dream_sessions, "workspaces", lambda *args: [])
    args["limits"] = dict(scan.LIMITS, inventory_entries=5)
    value = dream_inventory.collect(source, deadline=time.monotonic() + 2, **args)
    assert not value["sessions"] and value["coverage"]["entries"] == 5
    assert value["coverage"]["undiscovered_sources"] is None
    assert "inventory_entries_limit" in value["coverage"]["reasons"]


def test_tail_probe_limit_is_unavailable_rather_than_empty(tmp_path, monkeypatch):
    from sucoder import dream_experience as experience, dream_sessions
    directory = experience.project_dir(tmp_path, tmp_path)
    directory.mkdir(parents=True)
    (directory / (SID + ".jsonl")).write_bytes(b"x" * (2 * 1024 * 1024))
    source = type("Source", (), {"path": tmp_path})()
    monkeypatch.setattr(experience, "workspaces", lambda source: [str(tmp_path)])
    monkeypatch.setattr(dream_sessions, "workspaces", lambda *args: [])
    value = dream_inventory.collect(source, home=tmp_path, quiet_minutes=30, live=(), memory_commit=None,
        config=None, limits=scan.LIMITS, deadline=time.monotonic() + 3)
    assert value["sessions"][0]["state"] == "unavailable"
    assert value["sessions"][0]["reason"] == "complete_prefix_probe_limit"
    assert value["coverage"]["tail_bytes"] == 1024 * 1024


def test_json_escaped_read_size_retains_a_skip_cursor(tmp_path):
    first = line("\u679c" * 4000)
    inv, _ = inventory(tmp_path, first + line("needle after an undeliverable record"))
    value = inv.lookup(request("read", "@sessions/" + SID, cursor=""), 16384)
    assert value["state"] == "range_too_large" and value["next_cursor"]
    assert (value["byte_start"], value["byte_end"]) == (0, len(first))
    assert len(dream.canonical(value)) <= 16384
    later = inv.lookup(continue_request(value, action="read", path="@sessions/" + SID, query=""), 16384)
    assert later["state"] == "complete_with_gaps" and later["earlier_gaps"] and "needle after" in later["text"]


def test_unterminated_rewritten_prefix_is_an_explicit_gap(tmp_path):
    raw = line("needle")
    inv, path = inventory(tmp_path, raw)
    path.write_bytes(raw[:-1] + b"x")
    value = inv.lookup(request("search", "@sessions", query="needle", cursor=""), 16384)
    assert value["state"] == "complete_with_gaps" and not value["matches"]
    assert value["gaps"][0]["reason"] == "unterminated_pinned_record"


def test_inventory_candidate_metadata_and_aggregate_probe_limits(tmp_path, monkeypatch):
    from sucoder import dream_experience as experience, dream_sessions
    directory = experience.project_dir(tmp_path, tmp_path)
    directory.mkdir(parents=True)
    for n in range(3):
        sid = SID[:-1] + str(n)
        (directory / (sid + ".jsonl")).write_bytes(b"x" * (2 * 1024 * 1024))
    source = type("Source", (), {"path": tmp_path})()
    monkeypatch.setattr(experience, "workspaces", lambda source: [str(tmp_path)])
    monkeypatch.setattr(dream_sessions, "workspaces", lambda *args: [])
    args = dict(home=tmp_path, quiet_minutes=30, live=(), memory_commit=None, config=None, deadline=time.monotonic() + 3)
    candidates = dream_inventory.collect(source, limits=dict(scan.LIMITS, inventory_candidates=1), **args)
    assert candidates["coverage"]["candidates"] == 1 and len(candidates["sessions"]) == 1
    assert candidates["coverage"]["undiscovered_sources"] is None
    metadata = dream_inventory.collect(source, limits=dict(scan.LIMITS, inventory_metadata=1), **args)
    assert metadata["coverage"]["metadata_operations"] == 1 and not metadata["sessions"]
    total = dream_inventory.collect(source, limits=dict(scan.LIMITS, tail_total_bytes=2 * 1024 * 1024), **args)
    assert total["coverage"]["tail_bytes"] == 2 * 1024 * 1024
    assert total["coverage"]["unprobed_candidates_lower_bound"] == 1


def test_skipping_or_alignment_cannot_deliver_a_record_after_the_requested_end(tmp_path):
    raw = line("x" * 1000) + line("OUTSIDE REQUESTED END")
    limits = dict(scan.LIMITS, chunk_bytes=256, record_bytes=256, scan_bytes=2048)
    inv, _ = inventory(tmp_path, raw, limits=limits)
    value = inv.lookup(request("read", "@sessions/" + SID, start=0, end=100, cursor=""), 4096)
    assert value["state"] == "complete_with_gaps" and not value["next_cursor"]
    assert not value["text"] and "OUTSIDE" not in json.dumps(value)
    aligned = inv.lookup(request("read", "@sessions/" + SID, start=10, end=100, cursor=""), 4096)
    assert aligned["state"] == "no_complete_record" and not aligned["next_cursor"] and not aligned["text"]
    with pytest.raises(dream_isolate.ProcessingError, match="No complete record starts"):
        inv.selected([dict(path="@sessions/" + SID, start=10, end=100)])
    only, _ = inventory(tmp_path, line("x" * 1000), limits=limits)
    finished = only.lookup(request("read", "@sessions/" + SID, cursor=""), 4096)
    assert finished["state"] == "complete_with_gaps" and not finished["next_cursor"] and not finished["text"]


@pytest.mark.parametrize("timeout", [False, True])
def test_worker_cleans_up_descendants_on_return_and_timeout(tmp_path, timeout):
    script = '''import ctypes, os, signal, subprocess, sys, time
from pathlib import Path
from sucoder import dream_isolate as isolate
assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0  # Test-local child subreaper.
path = Path(sys.argv[1])
timeout = sys.argv[2] == 'True'
def work():
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
    path.write_text(str(process.pid))
    if timeout:
        time.sleep(10)
    return process.pid
try:
    value = isolate.run(work, time.monotonic()+3, dict(isolate.POLICY, max_seconds=0.3))
except isolate.ProcessingError as exc:
    assert timeout and exc.reason == 'processing_deadline'
else:
    assert not timeout and value == int(path.read_text())
pid = int(path.read_text())
try:
    end = time.monotonic()+0.5
    while True:
        found, status = os.waitpid(pid, os.WNOHANG)
        if found or time.monotonic() >= end:
            break
        time.sleep(0.01)
    assert found == pid and os.WIFSIGNALED(status), 'worker left a live descendant'
finally:
    try:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    except (ChildProcessError, ProcessLookupError):
        pass
'''
    subprocess.run([sys.executable, "-c", script, str(tmp_path / "descendant.pid"), str(timeout)], check=True, timeout=4)


def test_gap_details_and_final_cursor_bytes_cannot_discard_a_valid_page(tmp_path):
    limits = dict(scan.LIMITS, record_bytes=4096, chunk_bytes=1024, scan_bytes=40000)
    raw = line("x" * 5000) * 4 + line("a" * 1113) + line("last " + "b" * 200)
    inv, _ = inventory(tmp_path, raw, limits=limits)
    value = inv.lookup(request("read", "@sessions/" + SID, cursor=""), 2048)
    shown = []
    for _ in range(5):
        assert value["state"] != "response_metadata_limit" and len(dream.canonical(value)) <= 2048
        shown.append(value.get("text", ""))
        if not value["next_cursor"]:
            break
        value = inv.lookup(continue_request(value, action="read", path="@sessions/" + SID, query=""), 2048)
    else:
        pytest.fail("gap-plus-cursor read failed to finish")
    assert value["state"] == "complete_with_gaps" and any("last b" in text for text in shown)


def test_malformed_record_errors_are_bounded_and_keep_unknown_read_accounting(tmp_path):
    key = json.dumps("\u754c" * 4000, ensure_ascii=False).encode()
    raw = b'{"sessionId":"' + SID.encode() + b'",' + key + b':1,' + key + b':2}\n'
    inv, _ = inventory(tmp_path, raw)
    value = inv.lookup(request("search", "@sessions", query="needle", cursor=""), 1024)
    assert value["state"] == "source_processing_error"
    assert value["actual_reads"] == "unknown" and value["raw_scanned_bytes"] is None
    assert value["raw_scan_upper_bound"] == scan.LIMITS["scan_bytes"] + scan.LIMITS["alignment_bytes"]
    assert len(dream.canonical(value)) <= 1024 and "\u754c" not in str(value)
    assert inv.navigator.last["receipts"] is None


def test_terminal_metadata_fallback_retains_unknown_read_bounds(tmp_path, monkeypatch):
    inv, _ = inventory(tmp_path, line("ordinary"))
    monkeypatch.setattr(scan, "page", lambda *args: dict(observation=dict(state="unavailable", detail="x" * 20000,
        raw_scanned_bytes=None, actual_reads="unknown"), continuation=None, receipts=None))
    value = inv.lookup(request("search", "@sessions", query="needle", cursor=""), 1024)
    assert value["state"] == "response_metadata_limit" and value["actual_reads"] == "unknown"
    assert value["raw_scanned_bytes"] is None and value["raw_scan_upper_bound"] > 0
