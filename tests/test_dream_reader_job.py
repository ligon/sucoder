"""Session reader jobs, milestone R1 (docs/dream/session-reader.org), with fake native transports only."""
import copy
import json
import os
from pathlib import Path
import time

import pytest
import yaml

from sucoder import cli, dream, dream_budget, dream_experience, dream_explore, dream_native, dream_notes
from sucoder import dream_reader_job as reader
from sucoder.config import ConfigError, _parse_dream, load_config
from sucoder.dream import DreamError
from tests.test_dream import add_registry, git, runner, workspace  # noqa: F401
from tests.test_dream_explore import clock  # noqa: F401
from tests.dream_submission_fixture import native_reply


IDENTITY = "3c2b1a09-8f7e-4d6c-9b5a-4e3d2c1b0a99"
SID = "5e55a0de-1111-4222-8333-444455556666"
OPTIONS = dict(harness="codex", model="fixture-model", reader_chunk_bytes=1024)
REMINDER = "<system-reminder>Always run the full suite before committing.</system-reminder>"


def record(kind, content, stamp, number, **extra):
    return dict(type=kind, sessionId=SID, uuid="u-" + str(number), timestamp=stamp,
                message=dict(role=kind, content=content), **extra)


def turn_one():
    day = "2026-11-20T10:00:0"
    return [dict(type="file-history-snapshot", snapshot={}),
            record("user", "Please keep the timeout at 30 seconds; that is my decision.\n" + REMINDER, day + "0Z", 1,
                   origin=dict(kind="human")),
            record("assistant", [dict(type="text", text="I will run the dream notes tests in parallel now."),
                                 dict(type="tool_use", id="t1", name="Bash",
                                      input=dict(command="pytest -n 2 tests/test_dream_notes.py"))], day + "1Z", 2),
            record("user", [dict(type="tool_result", tool_use_id="t1", content="56 passed in 3.1s")], day + "2Z", 3),
            record("assistant", [dict(type="text", text="All dream tests passed; the timeout stays at 30 seconds.")],
                   day + "3Z", 4)]


def turn_two():
    day = "2026-11-21T09:00:0"
    return [record("user", "Now check whether the slow tests ran, and tell me what you find.", day + "0Z", 5,
                   origin=dict(kind="human")),
            record("assistant", [dict(type="tool_use", id="t2", name="Bash", input=dict(command="pytest -m slow"))],
                   day + "1Z", 6),
            record("user", [dict(type="tool_result", tool_use_id="t2", is_error=True,
                                 content="ERROR: 2 failed, 1 passed in 40.2s")], day + "2Z", 7),
            record("assistant", [dict(type="text", text="Two slow tests failed; the cause is not yet known. " +
                                      "The output of the slow run does not name the failing tests. " * 5)],
                   day + "3Z", 8)]


def turn_three():
    day = "2026-11-22T08:00:0"
    return [record("user", "Mark the slow failures as a known issue for later.", day + "0Z", 9,
                   origin=dict(kind="human")),
            record("assistant", [dict(type="text", text="Recorded the slow failures as a known issue.")], day + "1Z", 10)]


def lines(records):
    return b"".join(json.dumps(r).encode() + b"\n" for r in records)


def age(path, seconds=3600):
    os.utime(path, (time.time() - seconds, time.time() - seconds))


def write_session(home, repo, *, seconds=3600):
    path = dream_experience.project_dir(home, repo) / (SID + ".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(lines(turn_one() + turn_two()))
    age(path, seconds)
    return path


def at(path, needle):
    """The raw offset of the record holding ``needle``."""
    offset = 0
    for line in path.read_bytes().splitlines(keepends=True):
        if needle.encode() in line:
            return offset
        offset += len(line)
    raise AssertionError(needle)


def item(kind, text, citations, basis, origin, when="", **extra):
    value = dict(kind=kind, text=text, when=when, basis=basis, origin=origin, by="", command="", outcome="", target="",
                 citations=[dict(record=record, quote=quote) for record, quote in citations])
    value.update(extra)
    return value


def script(path):
    """A faithful reading of both turns: one chunk each."""
    user, call, output = at(path, "keep the timeout"), at(path, "pytest -n 2"), at(path, "56 passed")
    agent = at(path, "All dream tests passed")
    slow, failure, cause = at(path, "pytest -m slow"), at(path, "2 failed"), at(path, "cause is not yet known")
    return {0: [item("decision", "The human decided to keep the timeout at 30 seconds.",
                     [(user, "keep the timeout at 30 seconds")], "human said", "human", "2026-11-20", by="human"),
                item("command", "The dream notes tests ran in parallel and passed.",
                     [(call, "pytest -n 2"), (output, "56 passed")], "output shows", "tool",
                     command="pytest -n 2 tests/test_dream_notes.py", outcome="worked"),
                item("fact", "The agent reported that all dream tests passed.",
                     [(agent, "All dream tests passed")], "agent reported", "agent")],
            1: [item("command", "The slow tests failed: two of three.", [(slow, "pytest -m slow"), (failure, "2 failed")],
                     "output shows", "tool", "2026-11-21", command="pytest -m slow", outcome="failed"),
                item("open_question", "Why two slow tests failed is not yet known.",
                     [(cause, "the cause is not yet known")], "agent reported", "agent")]}


class Native:
    """A fake native transport: scripted reader items per chunk index and a fixed review."""
    version = "reader-fixture-1"

    def __init__(self, items=None, verdict="accept", findings=()):
        self.items, self.verdict, self.findings, self.calls = items or {}, verdict, list(findings), []

    def __call__(self, harness, model, task, schema, deadline):
        self.calls.append(copy.deepcopy(task))
        result = {key: task[key] for key in ("assignment_id", "input_sha256", "role")}
        if task["role"] == "reader":
            result.update(failures=[], items=copy.deepcopy(self.items.get(task["input"]["chunk"]["index"], [])))
        else:
            result.update(failures=[], verdict=self.verdict, findings=copy.deepcopy(self.findings))
        events = [dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(result))),
                  dict(type="turn.completed", usage={})]
        trace = dict(harness=harness, model=model, version=self.version, returncode=0,
                     input_sha256=dream.digest(dream.canonical(task)),
                     stdout="\n".join(json.dumps(e) for e in events) + "\n", stderr="", boundary="linux-landlock-inline-v1")
        if "effort" in task:
            trace["effort"] = task["effort"]
        return native_reply(task, result, dict(native={}, currency_usd=None), trace)


@pytest.fixture(autouse=True)
def no_real_native(monkeypatch):
    """Every test here uses Native; none may start, or even probe, an installed harness."""
    def forbidden(*args, **kwargs):
        raise AssertionError("A reader test attempted to construct a real native transport")
    monkeypatch.setattr(dream_native, "Transport", forbidden)


@pytest.fixture(autouse=True)
def v1_jobs(monkeypatch):
    """Tests prepare session-reader-v1 jobs, as every job was before v2 existed, unless they request v2."""
    monkeypatch.setattr(reader, "READER", reader.READER_V1)


@pytest.fixture
def v2(v1_jobs, monkeypatch):
    """New jobs freeze session-reader-v2, the module's default."""
    monkeypatch.setattr(reader, "READER", reader.READER_V2)


@pytest.fixture
def no_live(monkeypatch):
    monkeypatch.setattr(dream_explore, "_live_sessions", lambda config: set())


@pytest.fixture
def mirror(workspace, clock, no_live):  # noqa: F811
    repo, config_path, config, _ = workspace
    add_registry(repo, IDENTITY)
    return repo, config_path, config, write_session(Path(os.environ["HOME"]), repo)


def source_of(result):
    return dream_notes.Store(Path(result["store"])).source("claude", SID)


def jobs(config):
    directory = Path(config.log_dir) / "dream-reader-jobs"
    return sorted(p.name for p in directory.iterdir() if p.is_dir()) if directory.exists() else []


def forbid_native(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A completed reader job attempted native startup")
    monkeypatch.setattr(dream_native, "Transport", forbidden)


# End to end

def test_a_finished_session_is_read_into_an_accepted_cited_note(mirror):
    repo, _, config, path = mirror
    native = Native(script(path))
    result = reader.start(config, cwd=repo, overrides=OPTIONS, transport=native)
    assert result["state"] == "accepted" and result["calls"] == dict(reader=2, reviewer=1)
    assert [task["role"] for task in native.calls] == ["reader", "reader", "reviewer"]
    first, second, review = (task["input"] for task in native.calls)
    raw = path.read_bytes()
    # Turn-aligned chunks: the second starts at the second human turn and the pair covers the session.
    assert first["chunk"]["byte_start"] == 0 and second["chunk"]["byte_start"] == at(path, "slow tests ran")
    assert second["chunk"]["byte_end"] == len(raw) and first["chunk"]["count"] == 2
    assert first["chunk"]["text"].startswith("SESSION " + SID) and "[@" + str(at(path, "keep")) + "] USER\n" in first["chunk"]["text"]
    assert "HARNESS system-reminder" in first["chunk"]["text"]
    assert first["chunk"]["dates"][0]["date"] == "2026-11-20" and second["chunk"]["dates"][0]["date"] == "2026-11-21"
    # Notes are carried forward between chunks, append-only.
    assert first["carried"] == [] and [c["text"] for c in second["carried"]] == [i["text"] for i in script(path)[0]]
    assert second["allowance"]["remaining_items"] == 197
    assert native.calls[0]["contract"] == reader.ROLES and native.calls[0]["submission_contract"]
    # The reviewer sees the note and the cited blocks, not the whole session.
    assert len(review["items"]) == 5 and any("56 passed" in e["text"] for e in review["excerpts"])
    assert not any("Now check whether" in e["text"] for e in review["excerpts"])
    session = source_of(result)
    rows = session.notes(accepted_only=True)
    assert len(rows) == 1 and rows[0]["note_id"] == result["note_id"] and rows[0]["review"]["verdict"] == "accept"
    note = rows[0]["note"]
    assert note["render_contract"] == "claude-view-v3" and note["reader"] == dict(harness="codex", model="fixture-model")
    assert [i["kind"] for i in note["items"]] == ["decision", "command", "fact", "command", "open_question"]
    decision = note["items"][0]
    assert decision["by"] == "human" and decision["when"] == "2026-11-20" and "outcome" not in decision
    cite = decision["citations"][0]
    assert raw[cite["byte_start"]:cite["byte_end"]].endswith(b"\n") and b"keep the timeout" in raw[cite["byte_start"]:cite["byte_end"]]
    assert note["items"][2]["when"] is None
    # Coverage is recorded under the source lease, chunk by chunk, and is the whole session.
    assert session.coverage() == result["ranges"] == note["ranges"] and session.covered() == [[0, len(raw)]]
    assert git(repo, "for-each-ref", "refs/heads/sucoder/") == ""  # The memory branch is never written.
    assert result["estimate"]["calls"] == dict(reader=2, reviewer=1) and result["estimate"]["ceiling"] is False


def test_a_saved_job_replays_from_cache_with_native_calls_forbidden(mirror, monkeypatch, clock):  # noqa: F811
    repo, _, config, path = mirror
    native = Native(script(path))
    result = reader.start(config, cwd=repo, overrides=OPTIONS, transport=native)
    job = Path(result["job_path"])
    before = sorted(p.name for p in job.iterdir())
    clock.advance(10 ** 6)  # The wallet's time has passed.
    forbid_native(monkeypatch)
    with path.open("ab") as stream:  # The session was resumed and grew.
        stream.write(lines(turn_three()))
    age(path)
    assert reader.run(config, job) == result
    assert reader.start(config, job) == result
    assert reader.start(config, result["job_id"]) == result
    assert len(native.calls) == 3 and sorted(p.name for p in job.iterdir()) == before
    with pytest.raises(DreamError, match="immutable"):
        reader.start(config, job, overrides=dict(model="other-model"))


def test_a_second_read_skips_covered_ranges_and_reads_only_the_new_tail(mirror):
    repo, _, config, path = mirror
    first = reader.start(config, cwd=repo, overrides=OPTIONS, transport=Native(script(path)))
    size = path.stat().st_size
    with path.open("ab") as stream:
        stream.write(lines(turn_three()))
    age(path)
    mark = at(path, "Mark the slow failures")
    native = Native({0: [item("decision", "The human decided to track the slow failures as a known issue.",
                              [(mark, "Mark the slow failures as a known issue")], "human said", "human",
                              "2026-11-22", by="human")]})
    second = reader.start(config, cwd=repo, overrides=OPTIONS, transport=native)
    assert second["state"] == "accepted" and second["job_id"] != first["job_id"]
    reads = [task["input"]["chunk"] for task in native.calls if task["role"] == "reader"]
    assert [(c["byte_start"], c["byte_end"]) for c in reads] == [(size, path.stat().st_size)]
    assert "keep the timeout" not in reads[0]["text"]
    assert second["ranges"] == [[size, path.stat().st_size, second["ranges"][0][2]]]
    session = source_of(second)
    assert session.covered() == [[0, path.stat().st_size]] and len(session.coverage()) == 3
    assert len(session.notes(accepted_only=True)) == 2
    # Nothing is left: no job and no call, explicitly or by default.
    count, idle = len(jobs(config)), Native()
    assert reader.start(config, cwd=repo, session=SID, overrides=OPTIONS, transport=idle)["state"] == "covered"
    assert reader.start(config, cwd=repo, overrides=OPTIONS, transport=idle)["state"] == "no_session"
    assert idle.calls == [] and len(jobs(config)) == count


def test_ranges_admitted_by_explorer_windows_count_as_covered():
    assert reader._gaps(100, [[0, 10], [40, 60]]) == [[10, 40], [60, 100]]
    assert reader._gaps(100, [[50, 70], [0, 30], [20, 40], [None, 90], [80, 200]]) == [[40, 50], [70, 80]]
    assert reader._gaps(100, [[0, 100]]) == [] and reader._gaps(100, []) == [[0, 100]]


def test_a_rejected_review_is_stored_with_its_coverage_but_never_accepted(mirror):
    repo, _, config, path = mirror
    native = Native(script(path), verdict="reject",
                    findings=[dict(item=2, finding="The agent's claim is not checked against the tool output.")])
    result = reader.start(config, cwd=repo, overrides=OPTIONS, transport=native)
    assert result["state"] == "rejected"
    assert result["review"]["findings"] == ["item 2: The agent's claim is not checked against the tool output."]
    session = source_of(result)
    assert session.notes(accepted_only=True) == [] and session.notes()[0]["review"]["verdict"] == "reject"
    assert session.covered() == [[0, path.stat().st_size]]


def test_a_reading_with_nothing_to_record_covers_the_session_without_a_review(mirror):
    repo, _, config, path = mirror
    native = Native()
    result = reader.start(config, cwd=repo, overrides=OPTIONS, transport=native)
    assert result["state"] == "empty" and result["review"] is None and result["calls"] == dict(reader=2, reviewer=0)
    session = source_of(result)
    assert session.notes()[0]["note"]["items"] == [] and session.notes()[0]["review"] is None
    assert session.covered() == [[0, path.stat().st_size]]


# Mechanical rejections: nothing reaches the store

def bad_reading(name, path):
    user, call, agent = at(path, "keep the timeout"), at(path, "pytest -n 2"), at(path, "All dream tests passed")
    return {
        "quote": item("fact", "The agent said the tests failed.", [(agent, "All dream tests failed")],
                      "agent reported", "agent"),
        "origin": item("fact", "Tool output showed the tests passed.", [(agent, "All dream tests passed")],
                       "agent reported", "tool"),
        "harness": item("decision", "The human decided to always run the full suite.",
                        [(user, "Always run the full suite before committing.")], "human said", "human", by="human"),
        "outcome": item("command", "The parallel tests worked.", [(call, "pytest -n 2")], "output shows", "tool",
                        command="pytest -n 2", outcome="worked"),
        "date": item("fact", "The agent reported the tests passed.", [(agent, "All dream tests passed")],
                     "agent reported", "agent", "2026-11-25"),
        "record": item("fact", "A record outside the chunk.", [(at(path, "cause is not yet known"), "cause")],
                       "agent reported", "agent"),
        "field": item("fact", "A fact with an outcome.", [(agent, "All dream tests passed")], "agent reported", "agent",
                      outcome="worked"),
    }[name]


@pytest.mark.parametrize("name, message", [
    ("quote", "quote does not match the rendering"),
    ("origin", "has origin agent, not the stated tool"),
    ("harness", "has origin harness, not the stated human"),
    ("outcome", "needs a cited TOOL RESULT"),
    ("date", "neither a cited record's date nor quoted"),
    ("record", "not a rendered record of the delivered chunk"),
    ("field", "outcome applies only"),
])
def test_a_mechanically_invalid_item_stops_the_job_before_any_store_write(mirror, name, message):
    repo, _, config, path = mirror
    native = Native({0: [script(path)[0][0], bad_reading(name, path)]})
    with pytest.raises(reader.Rejected, match=message):
        reader.start(config, cwd=repo, overrides=OPTIONS, transport=native)
    assert [task["role"] for task in native.calls] == ["reader"]
    job = Path(config.log_dir) / "dream-reader-jobs" / jobs(config)[0]
    failures = [json.loads(p.read_bytes()) for p in job.glob("failure-*.json")]
    assert [f["state"] for f in failures] == ["rejected"]
    session = dream_notes.Store(Path(config.log_dir) / "dream-notes" / IDENTITY).source("claude", SID)
    assert session.coverage() == [] and session.notes() == [] and session.record() is None
    # The same attempt fails the same way from its saved result, without another call.
    with pytest.raises(reader.Rejected, match=message):
        reader.run(config, job, transport=native)
    assert len(native.calls) == 1


# session-reader-v2: unused kind fields are blanked, failing items are dropped, unreliable chunks reject the note

def pilot_fields(path):
    """Valid items that fill kind fields their kinds do not use, as the first pilot's reader did."""
    user, call, output = at(path, "keep the timeout"), at(path, "pytest -n 2"), at(path, "56 passed")
    return {0: [item("fact", "The human asked to keep the timeout at 30 seconds.",
                     [(user, "keep the timeout at 30 seconds")], "human said", "human", by="human"),
                item("command", "The dream notes tests ran in parallel and passed.",
                     [(call, "pytest -n 2"), (output, "56 passed")], "output shows", "tool", by="agent",
                     command="pytest -n 2 tests/test_dream_notes.py", outcome="worked")],
            1: [item("open_question", "Why two slow tests failed is not yet known.",
                     [(at(path, "cause is not yet known"), "the cause is not yet known")], "agent reported", "agent",
                     outcome="unknown")]}


def saved(job, name):
    return json.loads((job / name).read_bytes())


def test_unused_kind_fields_are_blanked_under_v2_and_still_reject_a_v1_job(mirror, monkeypatch):
    repo, _, config, path = mirror
    output = pilot_fields(path)
    # A job prepared before v2 rejects the whole note at the first unused field, as the pilot's job did.
    native = Native(output)
    with pytest.raises(reader.Rejected, match="Chunk 0 item 0: by applies only to a kind that uses it, not to fact"):
        reader.start(config, cwd=repo, overrides=OPTIONS, transport=native)
    old = Path(config.log_dir) / "dream-reader-jobs" / jobs(config)[0]
    assert "reader_contract" not in saved(old, "job.json") and not list(old.glob("dropped-*.json"))
    assert native.calls[0]["contract"] == reader.ROLES and native.calls[0]["brief"] == reader.BRIEFS["reader"]
    session = dream_notes.Store(Path(config.log_dir) / "dream-notes" / IDENTITY).source("claude", SID)
    assert session.coverage() == [] and session.notes() == []
    monkeypatch.setattr(reader, "READER", reader.READER_V2)
    native = Native(output)
    result = reader.start(config, cwd=repo, overrides=OPTIONS, transport=native)
    assert result["state"] == "accepted" and result["reader_contract"] == reader.READER_V2
    assert result["chunk_items"] == [dict(chunk=0, proposed=2, accepted=2, dropped=0),
                                     dict(chunk=1, proposed=1, accepted=1, dropped=0)]
    assert result["dropped_items"] == 0 and result["items"] == 3
    job = Path(result["job_path"])
    assert saved(job, "job.json")["reader_contract"] == reader.READER_V2
    assert saved(job, "spec-000001.json")["contract"] == reader.ROLES_V2
    assert saved(job, "dropped-000001-000001.json") == dict(contract=reader.READER_V2, attempt=1, chunk=1, proposed=1,
                                                            accepted=1, dropped=[])
    # The v2 role contract's reader brief states both rules plainly; the reviewer brief is unchanged.
    brief = native.calls[0]["brief"]
    assert native.calls[0]["contract"] == reader.ROLES_V2 and brief == reader.BRIEFS_V2["reader"]
    assert "stay empty strings unless the item's kind uses them" in brief and "copied verbatim" in brief
    assert native.calls[-1]["role"] == "reviewer" and native.calls[-1]["brief"] == reader.BRIEFS["reviewer"]
    note = source_of(result).notes(accepted_only=True)[0]["note"]
    fact, command, question = note["items"]
    assert "by" not in fact and "by" not in command and command["outcome"] == "worked" and "outcome" not in question
    assert [c["text"] for c in native.calls[1]["input"]["carried"]] == [i["text"] for i in output[0]]
    assert native.calls[1]["input"]["carried"][0]["by"] == ""


def test_items_that_fail_a_check_are_dropped_recorded_and_never_carried_or_reviewed(mirror, v2, monkeypatch,
                                                                                    clock):  # noqa: F811
    repo, _, config, path = mirror
    good = script(path)
    agent, slow, failure = at(path, "All dream tests passed"), at(path, "pytest -m slow"), at(path, "2 failed")
    # The pilot's genuine failures: an altered quote, a quote from another record and tool text labelled agent.
    altered = item("fact", "The agent said the tests failed.", [(agent, "All dream tests failed")],
                   "agent reported", "agent")
    elsewhere = item("fact", "Two slow tests failed.", [(slow, "2 failed")], "agent reported", "tool")
    labelled = item("fact", "Two of three slow tests failed.", [(failure, "2 failed")], "agent reported", "agent")
    native = Native({0: [good[0][0], altered, good[0][1], good[0][2]],
                     1: [elsewhere, good[1][0], labelled, good[1][1]]})  # Exactly half dropped: the chunk survives.
    result = reader.start(config, cwd=repo, overrides=OPTIONS, transport=native)
    assert result["state"] == "accepted" and result["items"] == 5 and result["dropped_items"] == 3
    assert result["chunk_items"] == [dict(chunk=0, proposed=4, accepted=3, dropped=1),
                                     dict(chunk=1, proposed=4, accepted=2, dropped=2)]
    job = Path(result["job_path"])
    assert saved(job, "dropped-000001-000000.json") == dict(
        contract=reader.READER_V2, attempt=1, chunk=0, proposed=4, accepted=3,
        dropped=[dict(chunk=0, item=1, reason="the quote does not match the rendering of record " + str(agent),
                      proposed=altered)])
    later = saved(job, "dropped-000001-000001.json")
    assert [(d["chunk"], d["item"], d["proposed"]) for d in later["dropped"]] == [(1, 0, elsewhere), (1, 2, labelled)]
    assert later["dropped"][0]["reason"] == "the quote does not match the rendering of record " + str(slow)
    assert later["dropped"][1]["reason"] == ("the quoted text of record " + str(failure) +
                                             " has origin tool, not the stated agent")
    # Only accepted items are carried forward, reviewed and stored.
    accepted = [i["text"] for i in good[0] + good[1]]
    reads = [task["input"] for task in native.calls if task["role"] == "reader"]
    assert [c["text"] for c in reads[1]["carried"]] == accepted[:3]
    review = [task["input"] for task in native.calls if task["role"] == "reviewer"]
    assert len(review) == 1 and [i["text"] for i in review[0]["items"]] == accepted
    note = source_of(result).notes(accepted_only=True)[0]["note"]
    assert [i["text"] for i in note["items"]] == accepted
    # The completed job replays, dropped records included, without a call or a write.
    files = {p.name: p.read_bytes() for p in job.iterdir() if p.is_file()}
    monkeypatch.setattr(reader, "READER", reader.READER_V4)
    clock.advance(10 ** 6)
    forbid_native(monkeypatch)
    assert reader.run(config, job) == result
    assert {p.name: p.read_bytes() for p in job.iterdir() if p.is_file()} == files
    (job / "dropped-000001-000001.json").write_bytes(dream.canonical(dict(later, dropped=[])) + b"\n")
    with pytest.raises(DreamError, match="dropped-000001-000001.json differs from its role results"):
        reader.run(config, job)


@pytest.mark.parametrize("case, message", [
    ("most", "Chunk 0: 2 of its 3 proposed items failed .* more than half of a chunk's items are dropped"),
    ("all", "Chunk 0: none of its 2 proposed items passed .* no item of a chunk survives"),
])
def test_a_chunk_that_loses_more_than_half_or_all_of_its_items_rejects_the_note(mirror, v2, case, message):
    repo, _, config, path = mirror
    bad = [bad_reading("quote", path), bad_reading("origin", path)]
    proposed = dict(most=[script(path)[0][0]] + bad, all=bad)[case]
    native = Native({0: proposed, 1: script(path)[1]})
    with pytest.raises(reader.Rejected, match=message):
        reader.start(config, cwd=repo, overrides=OPTIONS, transport=native)
    assert [task["role"] for task in native.calls] == ["reader"]
    job = Path(config.log_dir) / "dream-reader-jobs" / jobs(config)[0]
    assert [json.loads(p.read_bytes())["state"] for p in job.glob("failure-*.json")] == ["rejected"]
    record = saved(job, "dropped-000001-000000.json")
    assert record["proposed"] == len(proposed) and record["accepted"] == len(proposed) - 2
    assert [d["proposed"] for d in record["dropped"]] == bad and all(d["reason"] for d in record["dropped"])
    session = dream_notes.Store(Path(config.log_dir) / "dream-notes" / IDENTITY).source("claude", SID)
    assert session.coverage() == [] and session.notes() == [] and session.record() is None
    with pytest.raises(reader.Rejected, match=message):
        reader.run(config, job, transport=native)
    assert len(native.calls) == 1


def test_a_saved_v1_job_replays_to_its_stored_outcome_when_new_jobs_freeze_v2(mirror, monkeypatch,
                                                                             clock):  # noqa: F811
    repo, _, config, path = mirror
    result = reader.start(config, cwd=repo, overrides=OPTIONS, transport=Native(script(path)))
    job = Path(result["job_path"])
    assert "reader_contract" not in saved(job, "job.json") and "chunk_items" not in result
    assert saved(job, "spec-000001.json")["contract"] == reader.ROLES and not list(job.glob("dropped-*.json"))
    files = {p.name: p.read_bytes() for p in job.iterdir() if p.is_file()}
    monkeypatch.setattr(reader, "READER", reader.READER_V2)
    monkeypatch.setattr(reader, "READER", reader.READER_V4)
    clock.advance(10 ** 6)
    forbid_native(monkeypatch)
    assert reader.run(config, job) == result and reader.start(config, result["job_id"]) == result
    assert {p.name: p.read_bytes() for p in job.iterdir() if p.is_file()} == files


@pytest.mark.parametrize("named", [reader.READER_V1, "session-reader-v9"])
def test_a_header_naming_v1_keeps_v1_and_an_unknown_reader_contract_is_refused(mirror, v2, named):
    repo, _, config, path = mirror
    prepared = reader.prepare(config, cwd=repo, overrides=OPTIONS)
    assert prepared["reader_contract"] == reader.READER_V2
    job = Path(prepared["job_path"])
    header = dict(saved(job, "job.json"), reader_contract=named)
    for name, value in (("job.json", header), ("identity.json", dict(sha256=dream.digest(dream.canonical(header))))):
        (job / name).write_bytes(dream.canonical(value) + b"\n")
    native = Native(pilot_fields(path))
    if named == reader.READER_V1:
        with pytest.raises(reader.Rejected, match="by applies only to a kind that uses it, not to fact"):
            reader.run(config, job, transport=native)
        assert native.calls[0]["contract"] == reader.ROLES and not list(job.glob("dropped-*.json"))
    else:
        with pytest.raises(DreamError, match="Invalid frozen reader contract"):
            reader.run(config, job, transport=native)
        assert native.calls == []


# Sessions that must not be read

def write_config(config_path, **dream_settings):
    raw = yaml.safe_load(config_path.read_text())
    raw["dream"] = dream_settings
    config_path.write_text(yaml.safe_dump(raw))
    return load_config(config_path)


def test_reader_effort_is_frozen_for_both_roles_and_cached_replay(mirror, monkeypatch):
    repo, config_path, _, path = mirror
    config = write_config(config_path, reader_effort="medium")
    assert reader.prepare(config, cwd=repo, overrides=OPTIONS)["settings"]["effort"] == "medium"
    with pytest.raises(DreamError, match="only for the codex"):
        reader.prepare(config, cwd=repo, overrides=dict(OPTIONS, harness="claude"))
    monkeypatch.chdir(repo)
    outcome = runner.invoke(cli.app, ["--config", str(config_path), "dream", "read-session", "--prepare-only",
                                      "--harness", "codex", "--model", "fixture-model", "--chunk-bytes", "1024",
                                      "--effort", "ultra"])
    assert outcome.exit_code == 0, outcome.output
    job = Path(json.loads(outcome.output)["job_path"])
    assert saved(job, "job.json")["settings"]["effort"] == "ultra"
    config = write_config(config_path, reader_effort="low")
    native = Native(script(path))
    result = reader.start(config, job, transport=native)
    assert [(t["role"], t["effort"]) for t in native.calls] == [
        ("reader", "ultra"), ("reader", "ultra"), ("reviewer", "ultra")]
    assert saved(job, "spec-000001.json")["effort"] == "ultra"
    forbid_native(monkeypatch)
    assert reader.start(config, job) == result
    with pytest.raises(DreamError, match="immutable"):
        reader.start(config, job, overrides=dict(effort="low"))


def test_legacy_reader_identity_and_assignment():
    settings = dict(harness="codex", model="reader", max_total_calls=3, reader_chunk_bytes=1024)
    header = dict(settings=settings, job_id="f" * 32, chunks=[])
    job = reader.Reader(header, None, 60, None, native_version="fixture")
    # Run and assignment digests from c0fa15c for a header predating reader_contract.
    assert job.run_id == "67310846e6eb4435ca16aa2f5bba1bace823b16aa5645690e03be9d24eb46068"
    assert job.make_assignment("reader", {})["assignment_id"] == (
        "4dfcec531f4d8698f26bb493510f420d67a19f414cebee0abdab1360de985be4")


@pytest.mark.parametrize("state", ["live", "recent", "excluded"])
def test_live_recent_and_excluded_sessions_are_never_read(workspace, clock, monkeypatch, state):  # noqa: F811
    repo, config_path, config, _ = workspace
    add_registry(repo, IDENTITY)
    write_session(Path(os.environ["HOME"]), repo, seconds=0 if state == "recent" else 3600)
    monkeypatch.setattr(dream_explore, "_live_sessions", lambda config: {SID} if state == "live" else set())
    if state == "excluded":
        config = write_config(config_path, reader_exclude_sessions=[SID[:8]])
    native = Native()
    with pytest.raises(DreamError, match=state):
        reader.start(config, cwd=repo, session=SID, overrides=OPTIONS, transport=native)
    assert reader.start(config, cwd=repo, overrides=OPTIONS, transport=native)["state"] == "no_session"
    assert native.calls == [] and jobs(config) == []


def test_an_exclusion_added_after_preparation_still_stops_the_read(mirror):
    repo, config_path, config, path = mirror
    prepared = reader.start(config, cwd=repo, overrides=OPTIONS, prepare_only=True)
    assert prepared["state"] == "prepared" and prepared["chunks"] == 2
    excluded = write_config(config_path, reader_exclude_sessions=[SID])
    native = Native(script(path))
    with pytest.raises(DreamError, match="excluded"):
        reader.run(excluded, Path(prepared["job_path"]), transport=native)
    assert native.calls == []


def test_a_busy_source_lease_refuses_before_any_native_call(mirror):
    repo, _, config, path = mirror
    prepared = reader.prepare(config, cwd=repo, overrides=OPTIONS)
    native = Native(script(path))
    with dream_notes.Store(Path(prepared["store"])).source("claude", SID).lease():
        with pytest.raises(DreamError, match="busy"):
            reader.run(config, Path(prepared["job_path"]), transport=native)
    assert native.calls == []
    assert reader.run(config, Path(prepared["job_path"]), transport=native)["state"] == "accepted"


# Budgets

def test_an_exhausted_wallet_stops_before_any_call_and_writes_nothing(mirror, clock):  # noqa: F811
    repo, _, config, path = mirror
    prepared = reader.prepare(config, cwd=repo, overrides=OPTIONS)
    clock.advance(901)
    native = Native(script(path))
    with pytest.raises(dream_budget.Exhausted):
        reader.run(config, Path(prepared["job_path"]), transport=native)
    assert native.calls == []
    session = dream_notes.Store(Path(prepared["store"])).source("claude", SID)
    assert session.coverage() == [] and session.notes() == []


def test_the_preflight_refuses_a_wallet_too_small_for_the_session(mirror):
    repo, _, config, path = mirror
    with pytest.raises(DreamError, match="needs at least 2 reader and 1 review calls"):
        reader.start(config, cwd=repo, overrides=dict(OPTIONS, max_total_calls=2, reserve_calls=1),
                     transport=Native(script(path)))
    assert jobs(config) == []
    estimate = reader.start(config, cwd=repo, overrides=OPTIONS, prepare_only=True)["estimate"]
    assert estimate["contract"] == reader.ESTIMATE and estimate["calibrated"] is False
    assert estimate["input_tokens"] == estimate["delivered_bytes"] // 4 and estimate["chunks"] == 2


# Configuration and command

def test_reader_settings_are_validated_and_documented_defaults_hold():
    settings = _parse_dream(dict(reader_exclude_sessions=["dcdc7362", SID], reader_chunk_bytes=4096))
    assert settings.reader_exclude_sessions == ["dcdc7362", SID] and settings.reader_chunk_bytes == 4096
    assert _parse_dream(None).reader_exclude_sessions == [] and _parse_dream(None).reader_chunk_bytes == 131072
    for bad in (dict(reader_exclude_sessions="dcdc7362"), dict(reader_exclude_sessions=["DCDC7362"]),
                dict(reader_exclude_sessions=["dcdc"]), dict(reader_exclude_sessions=[12345678]),
                dict(reader_chunk_bytes=512), dict(reader_chunk_bytes=True)):
        with pytest.raises(ConfigError, match="reader_"):
            _parse_dream(bad)
    root = Path(__file__).resolve().parents[1]
    example = [line[1:].split("#")[0] for line in (root / "config.example.yaml").read_text().splitlines()
               if line.startswith("#   reader_")]
    assert _parse_dream(yaml.safe_load("\n".join(example))) == _parse_dream(dict(reader_exclude_sessions=["dcdc7362"]))
    assert "dream.reader_exclude_sessions" in (root / "README.org").read_text()


def test_the_command_prepares_without_a_native_call(mirror, monkeypatch):
    repo, config_path, config, _ = mirror
    monkeypatch.chdir(repo)
    forbid_native(monkeypatch)
    outcome = runner.invoke(cli.app, ["--config", str(config_path), "dream", "read-session", "--prepare-only",
                                      "--harness", "codex", "--model", "fixture-model", "--chunk-bytes", "1024"])
    assert outcome.exit_code == 0, outcome.output
    value = json.loads(outcome.output)
    assert value["operation"] == "dream_session_read" and value["state"] == "prepared" and value["chunks"] == 2
    refused = runner.invoke(cli.app, ["--config", str(config_path), "dream", "read-session", "--session", "nope",
                                      "--harness", "codex", "--model", "fixture-model"])
    assert refused.exit_code == 2 and json.loads(refused.output)["state"] == "rejected"


class NativeV3(Native):
    def __init__(self, items, judgments=None, verdict="accept", malformed=None):
        super().__init__(items, verdict)
        self.judgments = judgments or {}
        self.malformed = malformed

    def __call__(self, harness, model, task, schema, deadline):
        if task["role"] != "reviewer":
            return super().__call__(harness, model, task, schema, deadline)
        self.calls.append(copy.deepcopy(task))
        result = {key: task[key] for key in ("assignment_id", "input_sha256", "role")}
        rows = [dict(item=i["index"], verdict=self.judgments.get(i["index"], "supported"),
                     finding="The citation does not establish this claim." if i["index"] in self.judgments else "")
                for i in task["input"]["items"]]
        if self.malformed == "missing":
            rows.pop()
        if self.malformed == "duplicate":
            rows.append(rows[0])
        result.update(failures=[], verdict=self.verdict, overall_findings=[], item_verdicts=rows)
        events = [dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(result))),
                  dict(type="turn.completed", usage={})]
        trace = dict(harness=harness, model=model, version=self.version, returncode=0,
                     input_sha256=dream.digest(dream.canonical(task)),
                     stdout="\n".join(json.dumps(e) for e in events) + "\n", stderr="", boundary="linux-landlock-inline-v1")
        return native_reply(task, result, dict(native={}, currency_usd=None), trace)


@pytest.fixture
def v3(v1_jobs, monkeypatch):
    monkeypatch.setattr(reader, "READER", reader.READER_V3)


def test_v3_filters_and_records_review_removals_and_replays(mirror, v3, monkeypatch, clock):
    repo, _, config, path = mirror
    native = NativeV3(script(path), {1: "overstated", 3: "unsupported"})
    result = reader.start(config, cwd=repo, overrides=OPTIONS, transport=native)
    assert (result["reviewed_items"], result["kept_items"], result["removed_items"]) == (5, 3, 2)
    job = Path(result["job_path"])
    record = saved(job, "review-items-000001.json")
    assert [(r["item"], r["stage"], r["verdict"]) for r in record["removed"]] == [
        (1, "review", "overstated"), (3, "review", "unsupported")]
    assert all(r["finding"] and r["proposed"] for r in record["removed"])
    note = source_of(result).notes(accepted_only=True)[0]["note"]
    original = script(path)[0] + script(path)[1]
    assert [i["text"] for i in note["items"]] == [original[i]["text"] for i in (0, 2, 4)]
    assert len(result["review"]["findings"]) == 2
    assert native.calls[-1]["contract"] == reader.ROLES_V3
    assert native.calls[-1]["brief"] == reader.BRIEFS_V3["reviewer"]
    files = {p.name: p.read_bytes() for p in job.iterdir() if p.is_file()}
    monkeypatch.setattr(reader, "READER", reader.READER_V4)
    clock.advance(10 ** 6)
    forbid_native(monkeypatch)
    assert reader.run(config, job) == result
    assert {p.name: p.read_bytes() for p in job.iterdir() if p.is_file()} == files


@pytest.mark.parametrize("judgments, verdict, malformed", [
    ({0: "overstated", 1: "unsupported", 2: "overstated"}, "accept", None),
    ({i: "unsupported" for i in range(5)}, "accept", None),
    ({}, "reject", None), ({}, "accept", "missing"), ({}, "accept", "duplicate")])
def test_v3_review_fails_closed(mirror, v3, judgments, verdict, malformed):
    repo, _, config, path = mirror
    native = NativeV3(script(path), judgments, verdict, malformed)
    with pytest.raises(reader.Rejected):
        reader.start(config, cwd=repo, overrides=OPTIONS, transport=native)
    job = Path(config.log_dir) / "dream-reader-jobs" / jobs(config)[0]
    assert not list(job.glob("note-*.json")) and not list(job.glob("intent-*.json"))
    session = dream_notes.Store(Path(config.log_dir) / "dream-notes" / IDENTITY).source("claude", SID)
    assert session.notes() == [] and session.coverage() == []
    if malformed is None:
        assert len(saved(job, "review-items-000001.json")["removed"]) == len(judgments)


def test_v3_estimate_is_calibrated_and_frozen(mirror, v3):
    repo, _, config, _ = mirror
    prepared = reader.prepare(config, cwd=repo, overrides=OPTIONS)
    header = saved(Path(prepared["job_path"]), "job.json")
    old = reader._estimate(header["chunks"], header["settings"]["reader_chunk_bytes"])
    estimate = header["estimate"]
    assert header["reader_contract"] == reader.READER_V3
    assert estimate["contract"] == reader.ESTIMATE_V2
    assert estimate["input_tokens"] == (old["input_tokens"] * 18 + 9) // 10
    assert estimate["output_tokens"] == 52000
    assert estimate["calibrated"] and not estimate["ceiling"]
    assert "one pilot: claude-sonnet-4-6, 2026-11-27, 3 reader and 1 reviewer call" in estimate["basis"]
    assert "Other models may differ" in estimate["basis"]


def test_v3_exactly_half_removed_is_accepted(mirror, v3):
    repo, _, config, path = mirror
    proposed = script(path)
    proposed[1] = proposed[1][:1]
    result = reader.start(config, cwd=repo, overrides=OPTIONS,
                          transport=NativeV3(proposed, {0: "overstated", 3: "unsupported"}))
    assert result["state"] == "accepted"
    assert (result["reviewed_items"], result["kept_items"], result["removed_items"]) == (4, 2, 2)


@pytest.fixture
def v4(v1_jobs, monkeypatch):
    monkeypatch.setattr(reader, "READER", reader.READER_V4)


def v4_script(path):
    proposed = script(path)
    for rows in proposed.values():
        for row in rows:
            for citation in row["citations"]:
                citation["role"] = "support"
    proposed[0][0]["citations"].append(dict(record=at(path, "All dream tests passed"),
        quote="the timeout stays at 30 seconds", role="context"))
    return proposed


def test_v4_stores_context_and_replays(mirror, v4, monkeypatch, clock):
    repo, _, config, path = mirror
    native = NativeV3(v4_script(path))
    result = reader.start(config, cwd=repo, overrides=OPTIONS, transport=native)
    note = source_of(result).notes(accepted_only=True)[0]["note"]
    assert note["contract"] == dream_notes.NOTE_V2
    assert [c["role"] for c in note["items"][0]["citations"]] == ["support", "context"]
    assert native.calls[0]["contract"] == reader.ROLES_V4
    assert any(e["role"] == "context" for e in native.calls[-1]["input"]["excerpts"])
    assert result["estimate"]["contract"] == reader.ESTIMATE_V3
    job = Path(result["job_path"])
    files = {p.name: p.read_bytes() for p in job.iterdir() if p.is_file()}
    clock.advance(10 ** 6)
    forbid_native(monkeypatch)
    assert reader.run(config, job) == result
    assert {p.name: p.read_bytes() for p in job.iterdir() if p.is_file()} == files


@pytest.mark.parametrize("case", ["two", "fact", "agent", "human", "sidechain", "only", "quote"])
def test_v4_refuses_invalid_context(mirror, v4, case):
    repo, _, config, path = mirror
    prepared = reader.prepare(config, cwd=repo, overrides=OPTIONS)
    header = saved(Path(prepared["job_path"]), "job.json")
    view = reader._views(header, time.monotonic() + 30)[0]
    value = v4_script(path)[0][0]
    context = value["citations"][-1]
    if case == "two":
        value["citations"].append(copy.deepcopy(context))
    elif case == "fact":
        value.update(kind="fact", by="")
    elif case == "agent":
        value["by"] = "agent"
    elif case == "human":
        context.update(value["citations"][0], role="context")
    elif case == "sidechain":
        for block in view["records"][context["record"]]:
            block["sidechain"] = True
    elif case == "only":
        value["citations"] = [context]
    else:
        context["quote"] = "This is absent."
    with pytest.raises(reader.Rejected):
        reader._resolve(value, view, header["chunks"][0], "item", reader.READER_V4)


@pytest.mark.parametrize("harness,model,output,calibrated", [
    ("claude", "claude-sonnet-4-6", 78760, True),
    ("codex", "gpt-6.1-sol", 10025, True),
    ("codex", "unknown", 78760, False),
    ("claude", "gpt-6.1-sol", 78760, False)])
def test_v4_model_calibration(harness, model, output, calibrated):
    old = dict(calls=dict(reader=3, reviewer=1), input_tokens=101, ceiling=False)
    estimate = reader._estimate_v3(old, harness, model)
    assert estimate["input_tokens"] == 182
    assert estimate["output_tokens"] == output
    assert estimate["calibrated"] is calibrated and not estimate["ceiling"]
    assert all(b["date"] == "2026-11-27" and b["model"] and b["calls"] for b in estimate["pilot_basis"])
