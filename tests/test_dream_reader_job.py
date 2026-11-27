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
        return native_reply(task, result, dict(native={}, currency_usd=None), trace)


@pytest.fixture(autouse=True)
def no_real_native(monkeypatch):
    """Every test here uses Native; none may start, or even probe, an installed harness."""
    def forbidden(*args, **kwargs):
        raise AssertionError("A reader test attempted to construct a real native transport")
    monkeypatch.setattr(dream_native, "Transport", forbidden)


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


# Sessions that must not be read

def write_config(config_path, **dream_settings):
    raw = yaml.safe_load(config_path.read_text())
    raw["dream"] = dream_settings
    config_path.write_text(yaml.safe_dump(raw))
    return load_config(config_path)


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
