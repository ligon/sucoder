"""Version 3 wallet, journal and end-to-end recovery with fake native roles."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import pytest

from sucoder import dream, dream_budget, dream_explore as explore, dream_native, dream_run
from sucoder import dream_navigation, dream_isolate
from sucoder.config import ConfigError, _parse_dream
from tests.test_dream import git, workspace  # noqa: F401
from tests.test_dream_explore import ExploringTransport, clock, options  # noqa: F401
from tests.test_dream_experience import SID, TRUTH, line, no_live, write_session  # noqa: F401
from tests.test_dream_wiki import TOPIC, planned


class Scripted(ExploringTransport):
    def __init__(self, choose):
        super().__init__()
        self.choose = choose

    def __call__(self, harness, model, task, schema, deadline):
        if task["role"] != "explorer":
            return super().__call__(harness, model, task, schema, deadline)
        self.calls.append(copy.deepcopy(task))
        value = {k: task[k] for k in ("assignment_id", "input_sha256", "role")}
        value.update(failures=[], action="finish", path="", query="", offset=0, start=0, end=0, cursor="",
                     selections=[], topic="reply-evidence", rationale="Verify the selected evidence.", questions=[])
        value.update(self.choose(task))
        if harness == "codex":
            events = [dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(value))),
                      dict(type="turn.completed", usage={})]
        else:
            events = [dict(type="system", subtype="init", tools=["StructuredOutput"], mcp_servers=[]),
                dict(type="result", subtype="success", is_error=False, structured_output=value, usage={}, total_cost_usd=None)]
        trace = dict(harness=harness, model=model, version=self.version, returncode=0,
            input_sha256=dream.digest(dream.canonical(task)), stdout="\n".join(json.dumps(e) for e in events) + "\n",
            stderr="", boundary="linux-landlock-inline-v1")
        return value, dict(native={}, currency_usd=None), trace


def read_then_finish(task):
    return dict(action="read", path="note.org") if not task["input"]["history"] else {}


def load(path):
    return json.loads(path.read_bytes())


@pytest.mark.parametrize("total,requested,expected", [(16, None, 8), (32, None, 16), (40, None, 20), (32, 8, 8)])
def test_allocation_freezes_actual_wallet_basis_and_explicit_limits(workspace, total, requested, expected):
    values = options(max_total_calls=total)
    if requested is not None:
        values["max_discovery_calls"] = requested
    value = explore.prepare(workspace[2], cwd=workspace[0], overrides=values)
    header = load(Path(value["job_path"]) / "job.json")
    allocation = header["discovery_allocation"]
    assert allocation["resolved"] == header["settings"]["max_discovery_calls"] == expected
    assert allocation["basis_remaining_calls"] == total and allocation["requested"] == requested
    assert allocation["mode"] == ("auto" if requested is None else "fixed")
    assert header["processing"] == dream_isolate.POLICY


def test_null_can_restore_auto_after_an_inherited_fixed_ceiling():
    fixed = _parse_dream(dict(max_discovery_calls=8))
    assert _parse_dream({}, fixed).max_discovery_calls == 8
    assert _parse_dream(dict(max_discovery_calls=None), fixed).max_discovery_calls is None
    for invalid in (0, False, 1.5, "auto"):
        with pytest.raises(ConfigError, match="max_discovery_calls"):
            _parse_dream(dict(max_discovery_calls=invalid))


def test_shared_wallet_allocation_uses_remaining_calls_and_rejects_zero_before_prepare(workspace, clock):
    repo, _, config, _ = workspace
    shared = repo.parent / "shared"
    native = Scripted(lambda task: {})
    first = explore.start(config, cwd=repo, overrides=options(max_total_calls=8), budget_path=shared, transport=native)
    second = explore.prepare(config, cwd=repo, overrides=options(max_total_calls=8), budget_path=shared)
    assert load(Path(second["job_path"]) / "job.json")["discovery_allocation"]["resolved"] == 3  # (8 - 1) // 2
    for _ in range(3):
        explore.start(config, cwd=repo, overrides=options(max_total_calls=8), budget_path=shared, transport=native)
    before = len(native.calls)
    with pytest.raises(dream_budget.Exhausted, match="no unreserved discovery and author capacity"):
        explore.prepare(config, cwd=repo, overrides=options(max_total_calls=8), budget_path=shared)
    assert len(native.calls) == before == 4
    assert explore.budget_status(config, Path(first["job_path"]))["remaining_calls"] == 4


def test_tiny_delivery_and_one_call_capacity_narrow_the_native_schema(workspace, clock):
    for limits in (dict(max_total_calls=5), dict(max_discovery_bytes=1024)):
        native = Scripted(lambda task: {})
        result = explore.start(workspace[2], cwd=workspace[0], overrides=options(**limits), transport=native)
        task = native.calls[0]
        assert task["input"]["navigation"]["actions"] == ["finish"]
        assert task["result_schema"]["properties"]["action"]["enum"] == ["finish"]
        assert result["state"] == "no_selection" and len(native.calls) == 1


@pytest.mark.parametrize("artifact,mutation", [("advice", "delete"), ("advice", "corrupt"),
    ("lookup", "delete"), ("lookup", "corrupt"), ("terminal", "corrupt")])
def test_missing_or_changed_journal_artifacts_fail_without_provider_startup(workspace, clock, monkeypatch, artifact, mutation):
    native = Scripted(read_then_finish)
    result = explore.start(workspace[2], cwd=workspace[0], overrides=options(), transport=native)
    job = Path(result["job_path"])
    if artifact == "advice":
        path = sorted(p for p in job.glob("navigation-*.json") if "marker" not in p.name)[0]
    else:
        path = next(p for p in job.glob("lookup-*.json") if
            (load(p)["selection"] is not None) == (artifact == "terminal"))
    if mutation == "delete":
        path.unlink()  # Deliberate corruption of this test's disposable private job.
    else:
        value = load(path)
        if artifact == "advice":
            value["navigation"]["remaining_discovery_calls"] += 1
        else:
            value["observation"]["text"] = "SUBSTITUTED OBSERVATION"
        path.write_bytes(dream.canonical(value) + b"\n")
    clock.advance(901)
    def forbidden(*args, **kwargs): raise AssertionError("corrupt replay started a provider")
    monkeypatch.setattr(dream_native, "Transport", forbidden)
    with pytest.raises(dream.DreamError):
        explore.start(workspace[2], job)
    assert len(native.calls) == 2


@pytest.mark.parametrize("after_commit", [False, True])
def test_interrupted_envelope_publication_recovers_without_repeating_native_call(workspace, clock, monkeypatch, after_commit):
    native = Scripted(read_then_finish)
    prepared = explore.prepare(workspace[2], cwd=workspace[0], overrides=options())
    job = Path(prepared["job_path"])
    original = dream_run._store
    interrupted = False
    def store(fd, name, data):
        nonlocal interrupted
        if name.startswith("lookup-") and not interrupted:
            interrupted = True
            if after_commit:
                original(fd, name, data)
            raise OSError("injected lookup publication interruption")
        return original(fd, name, data)
    monkeypatch.setattr(dream_run, "_store", store)
    with pytest.raises(OSError, match="publication interruption"):
        explore.run(workspace[2], job, transport=native)
    assert len(native.calls) == 1
    result = explore.run(workspace[2], job, transport=native)
    assert result["state"] == "no_selection" and len(native.calls) == 2
    envelopes = [load(p) for p in job.glob("lookup-*.json")]
    assert len(envelopes) == 2 and all(v["sha256"] == dream_navigation.checksum(v) for v in envelopes)


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_late_session_decision_reaches_reviewed_receipt_and_replays_saved_pages(workspace, clock, no_live, monkeypatch, harness):
    planned(workspace)
    repo, _, config, _ = workspace
    path = write_session(Path(os.environ["HOME"]), repo)
    filler = (json.dumps(dict(type="attachment", sessionId=SID, attachment=dict(content="x" * 1024 * 1024))) + "\n").encode()
    prefix = filler * 20
    path.write_bytes(prefix + line("user", TRUTH))
    os.utime(path, (time.time() - 3600, time.time() - 3600))

    def choose(task):
        history = task["input"]["history"]
        if len(history) < 2:
            return dict(action="search", path="@sessions", query="REPLY 1733",
                        cursor=history[-1]["observation"]["next_cursor"] if history else "")
        if len(history) == 2:
            hit = history[-1]["observation"]["matches"][0]
            return dict(action="read", path=hit["path"], start=hit["offset"], end=hit["end"])
        shown = history[-1]["observation"]
        return dict(selections=[dict(path="@sessions/" + SID, start=shown["byte_start"], end=shown["byte_end"]),
            dict(path="evidence-a.org", start=0, end=0), dict(path="evidence-b.org", start=0, end=0)])

    native = Scripted(choose)
    result = explore.start(config, cwd=repo, overrides=options(harness, max_total_calls=32), transport=native)
    assert result["state"] == "published"
    job = Path(result["job_path"])
    history = native.calls[3]["input"]["history"]
    assert history[0]["observation"]["state"] == "incomplete" and not history[0]["observation"]["matches"]
    assert history[1]["observation"]["state"] == "complete_unfrozen"
    assert TRUTH in history[2]["observation"]["text"]
    author = next(task for task in native.calls if task["role"] == "author")
    assert TRUTH in json.dumps(author["input"])
    assert "mutable sources" in json.dumps(author["input"])
    plan = load(Path(result["plan_path"]))
    view = next(u for u in plan["units"] if u["kind"] == "claude-view")
    assert view["raw"]["byte_start"] == len(prefix) > 20_550_000
    assert view["raw"]["sha256"] == dream.digest(line("user", TRUTH)) and "processing" in view
    receipt = json.loads(git(repo, "show", result["candidate_commit"] + ":.sucoder/wiki/receipts/" + result["run_id"] + ".json"))
    assert any(u.get("raw") == view["raw"] for u in receipt["units"])
    assert "--session-id <uuid>" in git(repo, "show", result["candidate_commit"] + ":" + TOPIC)
    # Unselected raw metadata may change; v3 replay validates committed observations.
    with path.open("r+b") as stream:
        stream.seek(150)
        stream.write(b"Y")
    clock.advance(901)
    def forbidden(*args, **kwargs): raise AssertionError("saved navigation was recomputed or provider restarted")
    monkeypatch.setattr(dream_native, "Transport", forbidden)
    monkeypatch.setattr(dream_navigation.scan, "page", forbidden)
    assert explore.start(config, job) == result
    assert len(native.calls) == 6


@pytest.mark.parametrize("reserved", [False, True])
def test_lost_marker_and_advice_cannot_replace_an_existing_assignment(workspace, clock, monkeypatch, reserved):
    native = Scripted(read_then_finish)
    prepared = explore.prepare(workspace[2], cwd=workspace[0], overrides=options())
    job = Path(prepared["job_path"])
    original = dream_run._store
    interrupted = False
    def store(fd, name, data):
        nonlocal interrupted
        if reserved and name.startswith("lookup-") and not interrupted:
            interrupted = True
            raise OSError("injected interruption after reservation")
        return original(fd, name, data)
    def reserve(*args):
        raise OSError("injected interruption after assignment")
    monkeypatch.setattr(dream_run, "_store", store)
    if not reserved:
        monkeypatch.setattr(dream_budget.Job, "reserve", reserve)
    with pytest.raises(OSError, match="injected interruption"):
        explore.run(workspace[2], job, transport=native)
    before = len(native.calls)
    for path in job.glob("navigation-*.json"):
        path.unlink()  # Deliberately lose both artifacts in this disposable test job.
    with pytest.raises(dream.DreamError, match="missing after (a native reservation|assignment publication)"):
        explore.run(workspace[2], job, transport=native)
    assert len(native.calls) == before == int(reserved)


@pytest.mark.parametrize("replacement", ["metadata", "conversation"])
def test_selected_raw_digest_survives_the_finish_to_admission_boundary(workspace, clock, no_live, monkeypatch, replacement):
    planned(workspace)
    repo, _, config, _ = workspace
    path = write_session(Path(os.environ["HOME"]), repo)
    raw = (json.dumps(dict(type="attachment", sessionId=SID, attachment=dict(content="x" * 100000))) + "\n").encode()
    path.write_bytes(raw)
    os.utime(path, (time.time() - 3600, time.time() - 3600))
    changed = raw.replace(b"xxx", b"YYY", 1) if replacement == "metadata" else line("user", "x" * (len(raw) - len(line("user", ""))))
    assert len(changed) == len(raw)
    native = Scripted(lambda task: dict(selections=[dict(path="@sessions/" + SID, start=0, end=len(raw))]))
    original = dream_run._store
    def store(fd, name, data):
        result = original(fd, name, data)
        if name.startswith("lookup-") and json.loads(data).get("selection") is not None:
            path.write_bytes(changed)
        return result
    monkeypatch.setattr(dream_run, "_store", store)
    with pytest.raises(dream.DreamError, match="frozen discovery selection"):
        explore.start(config, cwd=repo, overrides=options(), transport=native)
    assert len(native.calls) == 1 and native.calls[0]["role"] == "explorer"


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_invalid_fields_provide_a_bounded_example_and_consume_their_turn(workspace, clock, no_live, harness):
    repo, _, config, _ = workspace
    write_session(Path(os.environ["HOME"]), repo)
    def choose(task):
        history = task["input"]["history"]
        if not history:
            return dict(action="search", path="@sessions", query="not proof", start=10)
        if len(history) == 1:
            return dict(action="search", path="@sessions", query="not proof")
        return {}
    native = Scripted(choose)
    result = explore.start(config, cwd=repo, overrides=options(harness), transport=native)
    history = native.calls[2]["input"]["history"]
    assert history[0]["observation"]["state"] == "invalid_request"
    assert history[0]["observation"]["example"]["start"] == 0
    assert history[1]["observation"]["state"] == "complete_unfrozen" and history[1]["observation"]["matches"]
    assert result["state"] == "no_selection" and len(native.calls) == 3
    assert explore.budget_status(config, Path(result["job_path"]))["reserved_calls"] == 3


def test_malformed_source_is_never_counted_as_known_zero_reads(workspace, clock, no_live):
    repo, _, config, _ = workspace
    path = write_session(Path(os.environ["HOME"]), repo)
    key = json.dumps("\u754c" * 4000, ensure_ascii=False).encode()
    path.write_bytes(b'{"sessionId":"' + SID.encode() + b'",' + key + b':1,' + key + b':2}\n')
    os.utime(path, (time.time() - 3600, time.time() - 3600))
    native = Scripted(lambda task: dict(action="search", path="@sessions", query="needle")
                      if not task["input"]["history"] else {})
    result = explore.start(config, cwd=repo, overrides=options(), transport=native)
    job = Path(result["job_path"])
    outcome = load(job / "chosen.json")["outcome"]
    coverage = outcome["navigation_coverage"]
    assert coverage["raw_scanned_bytes"] is None and coverage["unknown_actual_reads"]
    assert "\u754c" not in json.dumps(native.calls[1]["input"], ensure_ascii=False)
    assert explore.status(config, job)["state"] == "no_selection"
