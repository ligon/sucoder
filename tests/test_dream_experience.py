"""Experience dreams over this mirror's sessions and notes (essay-alignment S4)."""
import copy
import json
import os
from pathlib import Path
import time

import pytest

from sucoder import dream, dream_discover, dream_experience, dream_explore as explore, dream_view
from tests.test_dream import git, workspace  # noqa: F401
from tests.test_dream_explore import ExploringTransport, clock, options  # noqa: F401
from tests.test_dream_explore_cli import exploratory_runtime, invoke  # noqa: F401
from tests.test_dream_wiki import TOPIC, planned

SID = "6d1f0c2a-9b3e-4c5d-8e7f-0a1b2c3d4e5f"
MEMORY = "refs/heads/sucoder/memory"
TRUTH = "The compaction summary says REPLY 1733 was sent, but no assistant entry holds it: a summary is not proof."


def line(kind, content, **extra):
    return json.dumps(dict(type=kind, sessionId=SID, message=dict(role=kind, content=content), **extra)).encode() + b"\n"


def write_session(home, repo, *, age=3600):
    """A finished session in the Claude project directory derived from the mirror path."""
    path = dream_experience.project_dir(home, repo) / (SID + ".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(line("user", "Did my reply 1733 arrive?") + line("assistant", [dict(type="text", text=TRUTH)]) +
                     line("assistant", [dict(type="tool_use", id="t", name="Read", input=dict(file_path="/x"))]))
    os.utime(path, (time.time() - age, time.time() - age))
    return path


class ExperienceTransport(ExploringTransport):
    """Lists @sessions, reads the session, then selects it with the two committed sources."""

    def __call__(self, harness, model, task, schema, deadline):
        if task["role"] != "explorer":
            return super().__call__(harness, model, task, schema, deadline)
        self.calls.append(copy.deepcopy(task))
        step = len(task["input"]["history"])
        result = {key: task[key] for key in ("assignment_id", "input_sha256", "role")}
        result.update(failures=[], action="finish", path="", query="", offset=0, start=0, end=0, selections=[],
                      topic="reply-evidence", rationale="Check the reply note against what the session saw.", questions=[])
        if step == 0:
            result.update(action="list", path="@sessions")
        elif step == 1:
            result.update(action="read", path="@sessions/" + SID)
        else:
            size = task["input"]["history"][0]["observation"]["entries"][0]["bytes"]
            result["selections"] = [dict(path="@sessions/" + SID, start=0, end=size),
                                    dict(path="evidence-a.org", start=0, end=0), dict(path="evidence-b.org", start=0, end=0)]
        events = [dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(result))),
                  dict(type="turn.completed", usage={})]
        trace = dict(harness=harness, model=model, version=self.version, returncode=0,
                     input_sha256=dream.digest(dream.canonical(task)),
                     stdout="\n".join(json.dumps(e) for e in events) + "\n", stderr="", boundary="linux-landlock-inline-v1")
        return result, dict(native={}, currency_usd=None), trace


class ReadsUntilToldTransport(ExperienceTransport):
    """Keeps reading until the navigation leaves only finish, as the first live explorer did."""

    def __call__(self, harness, model, task, schema, deadline):
        if task["role"] == "explorer" and task["input"]["navigation"]["actions"] == ["finish"]:
            if "sessions_root" not in task["input"]["navigation"]:  # Project mode: the evidence-a/b selection.
                return ExploringTransport.__call__(self, harness, model, task, schema, deadline)
            return super().__call__(harness, model, task, schema, deadline)
        if task["role"] != "explorer":
            return super().__call__(harness, model, task, schema, deadline)
        self.calls.append(copy.deepcopy(task))
        step = len(task["input"]["history"])
        experience = "sessions_root" in task["input"]["navigation"]
        path = ("@sessions" + ("/" + SID if step else "")) if experience else "evidence-a.org"
        result = {key: task[key] for key in ("assignment_id", "input_sha256", "role")}
        result.update(failures=[], action="list" if experience and step == 0 else "read", path=path,
                      query="", offset=0, start=0, end=0, selections=[], topic="reply-evidence",
                      rationale="Keep checking the evidence.", questions=[])
        events = [dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(result))),
                  dict(type="turn.completed", usage={})]
        trace = dict(harness=harness, model=model, version=self.version, returncode=0,
                     input_sha256=dream.digest(dream.canonical(task)),
                     stdout="\n".join(json.dumps(e) for e in events) + "\n", stderr="", boundary="linux-landlock-inline-v1")
        return result, dict(native={}, currency_usd=None), trace


@pytest.fixture
def no_live(monkeypatch):
    monkeypatch.setattr(explore, "_live_sessions", lambda config: set())


def inventory(workspace, home, **kwargs):
    repo, _, config, _ = workspace
    source = dream.GitSource(config, "sample", time.monotonic() + 30)
    rows = dream_experience.sessions(source, home=home, **kwargs)
    return dream_discover.Inventory(source, git(repo, "rev-parse", "HEAD"), sessions=rows), rows


def test_a_dream_corrects_a_wrong_note_from_what_the_session_saw(workspace, clock, no_live):
    planned(workspace)  # The companion holds the wrong note: "a compaction summary proves that a reply was sent".
    repo, _, config, _ = workspace
    path = write_session(Path(os.environ["HOME"]), repo)
    native = ExperienceTransport()
    result = explore.start(config, cwd=repo, transport=native, overrides=dict(harness="codex", model="explicit-model"))
    assert result["state"] == "published"
    job = json.loads((Path(result["job_path"]) / "job.json").read_bytes())
    assert job["mode"] == "experience" and job["purpose"].startswith(explore.EXPERIENCE_PURPOSE)
    explorer = [t for t in native.calls if t["role"] == "explorer"]
    assert explorer[0]["brief"] == dream_discover.EXPERIENCE_BRIEF
    assert explorer[0]["input"]["navigation"]["sessions"] == 1
    listing = explorer[1]["input"]["history"][0]["observation"]
    assert listing["entries"][0]["state"] == "finished" and listing["entries"][0]["consumed"] == []
    assert TRUTH in explorer[2]["input"]["history"][1]["observation"]["text"]
    corrected = git(repo, "show", MEMORY + ":" + TOPIC)
    assert "is not proof" in corrected and "proves that a reply was sent" not in corrected
    plan = json.loads(Path(result["plan_path"]).read_bytes())
    view = next(u for u in plan["units"] if u["kind"] == dream_view.KIND)
    assert view["raw"]["byte_start"] == 0 and view["raw"]["byte_end"] == path.stat().st_size
    with path.open("ab") as stream:  # A resumed session grows after the dream.
        stream.write(line("user", "Thanks."))
    assert explore.status(config, Path(result["job_path"]))["state"] == "published"
    _, rows = inventory(workspace, Path(os.environ["HOME"]), memory_commit=git(repo, "rev-parse", MEMORY))
    assert rows[0]["consumed"] == [[0, view["raw"]["byte_end"]]]


@pytest.mark.parametrize("mode", ["experience", "project"])
def test_the_last_discovery_call_offers_only_finish(workspace, clock, no_live, mode):
    planned(workspace)
    repo, _, config, _ = workspace
    write_session(Path(os.environ["HOME"]), repo)
    native = ReadsUntilToldTransport()
    result = explore.start(config, cwd=repo, transport=native, overrides=options(mode=mode))
    assert result["state"] == "published"
    navigations = [t["input"]["navigation"] for t in native.calls if t["role"] == "explorer"]
    assert len(navigations) == config.dream.max_discovery_calls
    assert navigations[-1]["actions"] == ["finish"] and navigations[-1]["remaining_discovery_calls"] == 1
    assert all(n["actions"] == ["list", "read", "search", "finish"] for n in navigations[:-1])
    brief = native.calls[0]["brief"]
    assert brief == (dream_discover.EXPERIENCE_BRIEF if mode == "experience" else dream_discover.BRIEF)
    assert dream_discover.TURN_RULES in brief


def test_session_windows_snap_to_records_and_respect_the_pinned_size(workspace, tmp_path):
    path = write_session(tmp_path, workspace[0])
    inv, rows = inventory(workspace, tmp_path)
    size = rows[0]["bytes"]
    with path.open("ab") as stream:
        stream.write(line("user", "written after discovery"))
    root = inv.lookup(dict(action="list", path="", offset=0, query="", start=0, end=0), 16384)
    assert root["entries"][0] == dict(path="@sessions", kind="directory", sessions=1)
    first = path.read_bytes().index(b"\n") + 1
    window = inv.lookup(dict(action="read", path="@sessions/" + SID, offset=0, query="", start=3, end=0), 16384)
    assert window["byte_start"] == first and window["byte_end"] == size and "written after" not in window["text"]
    assert window["text"].startswith("SESSION " + SID) and "[@" + str(first) + "] AGENT" in window["text"]
    hits = inv.lookup(dict(action="search", path="@sessions", offset=0, query="not proof", start=0, end=0), 16384)
    assert hits["matches"][0]["offset"] == first and "not proof" in hits["matches"][0]["text"]
    units = inv.selected([dict(path="@sessions/" + SID, start=1, end=size - 1)])
    second = path.read_bytes().index(b"\n", first) + 1
    assert units[0]["kind"] == dream_view.KIND
    assert units[0]["locator"] == dict(path=str(path), thread_id=SID, byte_start=first, byte_end=second)
    with pytest.raises(dream.DreamError, match="nonzero raw end"):
        inv.selected([dict(path="@sessions/" + SID, start=0, end=0)])


def test_a_record_longer_than_the_requested_window_is_read_whole(workspace, tmp_path):
    # The first live runs met a 143 KB attachment record: every end short of its own end gave an empty window.
    attachment = json.dumps(dict(type="attachment", sessionId=SID, attachment=dict(content="x" * 60000))).encode() + b"\n"
    path = dream_experience.project_dir(tmp_path, workspace[0]) / (SID + ".jsonl")
    path.parent.mkdir(parents=True)
    path.write_bytes(line("user", "before") + attachment + line("assistant", [dict(type="text", text=TRUTH + " detail" * 300)]))
    os.utime(path, (time.time() - 3600, time.time() - 3600))
    inv, rows = inventory(workspace, tmp_path)
    lo = len(line("user", "before"))
    hi = lo + len(attachment)
    window = inv.lookup(dict(action="read", path="@sessions/" + SID, offset=0, query="", start=lo, end=lo + 1000), 16384)
    assert (window["byte_start"], window["byte_end"], window["next_offset"]) == (lo, hi, hi)
    assert "1 records without conversation text omitted" in window["text"]
    after = inv.lookup(dict(action="read", path="@sessions/" + SID, offset=0, query="", start=window["next_offset"],
                            end=rows[0]["bytes"]), 16384)
    assert TRUTH in after["text"]
    assert inv.selected([dict(path="@sessions/" + SID, start=lo, end=lo + 1000)])[0]["locator"]["byte_end"] == hi
    big = inv.lookup(dict(action="read", path="@sessions/" + SID, offset=0, query="", start=0, end=rows[0]["bytes"]), 1024)
    assert big["state"] == "range_too_large" and big["next_offset"] == rows[0]["bytes"]


@pytest.mark.parametrize("state, kwargs", [("live", dict(live={SID})), ("recent", dict(quiet_minutes=120))])
def test_live_or_recent_sessions_are_listed_but_never_read_or_selected(workspace, tmp_path, state, kwargs):
    write_session(tmp_path, workspace[0])
    inv, rows = inventory(workspace, tmp_path, **kwargs)
    assert rows[0]["state"] == state
    observation = inv.lookup(dict(action="list", path="@sessions", offset=0, query="", start=0, end=0), 16384)
    assert observation["entries"][0]["state"] == state
    with pytest.raises(dream.DreamError, match="cannot be read or selected"):
        inv.selected([dict(path="@sessions/" + SID, start=0, end=rows[0]["bytes"])])


def test_project_mode_keeps_open_exploration_without_sessions(workspace, clock, no_live):
    repo, _, config, _ = workspace
    write_session(Path(os.environ["HOME"]), repo)
    native = ExploringTransport(no_selection=True)
    result = explore.start(config, cwd=repo, overrides=options(mode="project"), transport=native)
    job = Path(result["job_path"])
    header = json.loads((job / "job.json").read_bytes())
    assert header["mode"] == "project" and header["purpose"] == explore.dream_wiki.PURPOSE
    assert native.calls[0]["brief"] == dream_discover.BRIEF
    assert "sessions" not in json.loads((job / "inventory.json").read_bytes())
    with pytest.raises(dream.DreamError, match="immutable"):
        explore.prepare(config, job, overrides=dict(mode="experience"))


def test_cli_explore_flag_selects_project_mode(workspace, exploratory_runtime):
    assert invoke(workspace, "--explore").exit_code == 0
    assert invoke(workspace).exit_code == 0
    calls = exploratory_runtime[1]
    assert calls[0][3]["overrides"]["mode"] == "project" and "mode" not in calls[1][3]["overrides"]


def test_a_job_header_frozen_before_these_settings_still_loads(workspace, clock, no_live):
    repo, _, config, _ = workspace
    # Before S4 every bare job explored the project; its stored inventory has no sessions.
    result = explore.start(config, cwd=repo, overrides=options(mode="project"), transport=ExploringTransport(no_selection=True))
    job = Path(result["job_path"])
    header = json.loads((job / "job.json").read_bytes())
    for key in ("memory_branch", "inject", "inject_bytes", "session_quiet_minutes"):
        header["settings"].pop(key)
    header.pop("mode")
    header["project"].pop("memory_ref")
    for name, value in (("job.json", header), ("identity.json", dict(sha256=dream.digest(dream.canonical(header))))):
        (job / name).chmod(0o600)
        (job / name).write_bytes(dream.canonical(value) + b"\n")
    # Header surgery changes the job identity that replayed requests are keyed on, so only the
    # validation of an older header is checked here: it loads and resumes as project mode.
    assert explore.prepare(config, job)["job_id"] == header["job_id"]
    with explore._job(job) as (_, _, frozen):
        assert "mode" not in frozen and "memory_branch" not in frozen["settings"]
