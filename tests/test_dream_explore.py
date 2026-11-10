"""Project exploration, durable editorial handoff and inference-free recovery."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest

from sucoder import dream, dream_budget, dream_explore as explore, dream_native
from sucoder.config import DreamSettings
from tests.test_dream import git, workspace  # noqa: F401
from tests.test_dream_budget import Clock
from tests.test_dream_curate import NoopTransport
from tests.test_dream_wiki import EditorTransport, TOPIC, planned


QUESTION = "Should the maintainer change the timeout?"


class ExploringTransport:
    version = EditorTransport.version

    def __init__(self, *, no_selection=False, editor=None, fail=False, on_finish=None):
        self.calls = []
        self.no_selection, self.fail, self.on_finish = no_selection, fail, on_finish
        self.editor = editor or EditorTransport()

    def __call__(self, harness, model, task, schema, deadline):
        self.calls.append(copy.deepcopy(task))
        if task["role"] != "explorer":
            return self.editor(harness, model, task, schema, deadline)
        result = {key: task[key] for key in ("assignment_id", "input_sha256", "role")}
        result.update(failures=[], action="finish", path="", query="", offset=0, start=0, end=0,
            selections=[], topic="reply-evidence", rationale="Connect native delivery evidence with the summary limitation.",
            questions=[QUESTION])
        if not self.no_selection:
            step = len(task["input"]["history"])
            if step == 0:
                result.update(action="list")
            elif step in (1, 2):
                result.update(action="read", path="evidence-" + ("a" if step == 1 else "b") + ".org")
            else:
                result["selections"] = [dict(path=path, start=0, end=0)
                    for path in ("evidence-a.org", "evidence-b.org")]
        if result["action"] == "finish" and self.on_finish:
            self.on_finish()
        if harness == "codex":
            events = [dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(result))),
                dict(type="turn.completed", usage={})]
        else:
            events = [dict(type="system", subtype="init", tools=["StructuredOutput"], mcp_servers=[]),
                dict(type="result", subtype="success", is_error=False, structured_output=result, usage={}, total_cost_usd=None)]
        trace = dict(harness=harness, model=model, version=self.version, returncode=1 if self.fail else 0,
            input_sha256=dream.digest(dream.canonical(task)), stdout="" if self.fail else "\n".join(json.dumps(e) for e in events) + "\n",
            stderr="fixture native failure" if self.fail else "", boundary="linux-landlock-inline-v1")
        return (None, None, trace) if self.fail else (result, dict(native={}, currency_usd=None), trace)


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(dream_budget, "_clock", value)
    return value


def options(harness="codex", **limits):
    return dict(harness=harness, model="explicit-model", **limits)


def header(path):
    return json.loads((Path(path) / "job.json").read_bytes())


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_canonical_subdirectory_uses_exact_head_preserves_dirty_state_and_replays_expired_job(workspace, monkeypatch, clock, harness):
    repo, _, config, _ = workspace
    stale = repo.parent.parent / "stale-mirrors" / "sample"
    stale.parent.mkdir()
    subprocess.check_call(["git", "clone", "--quiet", "--no-hardlinks", str(repo), str(stale)])
    planned(workspace)
    config = replace(config, mirror_root=stale.parent,
        mirrors={"sample": replace(config.mirrors["sample"], canonical_repo=repo)})
    commit = git(repo, "rev-parse", "HEAD")
    assert git(stale, "rev-parse", "HEAD") != commit
    (repo / "evidence-a.org").write_text("Human staged edits that are not source evidence.\n")
    git(repo, "add", "evidence-a.org")
    (repo / "evidence-a.org").write_text("Human unstaged edits that must survive.\n")
    subdir = repo / "untracked-subdir"
    subdir.mkdir()
    before = (git(repo, "status", "--porcelain"), (repo / ".git/index").read_bytes(),
        git(repo, "symbolic-ref", "HEAD"), git(repo, "rev-parse", "HEAD"))
    native = ExploringTransport()
    result = explore.start(config, cwd=subdir, overrides=options(harness), transport=native)
    assert result["state"] == "candidate_ready"
    assert result["project"]["path"] == str(repo) and result["project"]["base_commit"] == commit
    assert [task["role"] for task in native.calls] == ["explorer"] * 4 + ["author", "reviewer"]
    assert before == (git(repo, "status", "--porcelain"), (repo / ".git/index").read_bytes(),
        git(repo, "symbolic-ref", "HEAD"), git(repo, "rev-parse", "HEAD"))
    reads = [task["input"]["history"] for task in native.calls if task["role"] == "explorer"][-1]
    assert "compaction summary reports" in reads[1]["observation"]["text"]
    assert "Human unstaged" not in json.dumps(reads)
    plan = json.loads(Path(result["plan_path"]).read_bytes())
    assert {u["path"] for u in plan["units"]} == {"evidence-a.org", "evidence-b.org"}
    assert all(u["commit"] == commit for u in plan["units"])
    assert plan["maintenance"].startswith("Exploratory job ")
    assert result["questions"] == [QUESTION]
    assert "--session-id <uuid>" in git(repo, "show", result["candidate_commit"] + ":" + TOPIC)
    job = Path(result["job_path"])
    summary = explore.budget_status(config, job)
    assert summary["reserved_calls"] == summary["call_states"]["completed"] == 6
    assert sum(len(task["input"]["sources"]) for task in native.calls if task["role"] == "author") == 2
    clock.advance(901)
    def forbidden(*args, **kwargs):
        raise AssertionError("Cached exploratory replay attempted native startup")
    monkeypatch.setattr(dream_native, "Transport", forbidden)
    changed = replace(config, log_dir=repo.parent / "different-logs", dream=DreamSettings(model="wrong-model"))
    assert explore.start(changed, job) == result
    checked = explore.status(changed, job)
    assert checked["state"] == "candidate_ready" and checked["budget"]["state"] == "budget_exhausted"
    assert checked["budget"]["reserved_calls"] == 6 and len(native.calls) == 6


def test_unconfigured_project_from_subdirectory_can_record_no_selection_without_candidate(workspace):
    repo, _, config, _ = workspace
    subdir = repo / "untracked"
    subdir.mkdir()
    config = replace(config, mirrors={})
    native = ExploringTransport(no_selection=True)
    result = explore.start(config, cwd=subdir, overrides=options(), transport=native)
    assert result["state"] == "no_selection"
    assert result["project"]["path"] == str(repo) and result["project"]["name"] == repo.name
    assert result["questions"] == [QUESTION] and not list(Path(result["job_path"]).glob("dream/*/plan.json"))
    assert explore.status(config, cwd=subdir)["job_id"] == result["job_id"]
    assert [task["role"] for task in native.calls] == ["explorer"]


def test_latest_lookup_survives_another_job_interrupted_during_initialization(workspace):
    repo, _, config, _ = workspace
    native = ExploringTransport(no_selection=True)
    result = explore.start(config, cwd=repo, overrides=options(), transport=native)
    incomplete = Path(result["job_path"]).parent / ("0" * 32)
    incomplete.mkdir(mode=0o700)
    assert explore.status(config, cwd=repo)["job_id"] == result["job_id"]
    assert len(native.calls) == 1


def test_new_job_reexplores_accepted_unchanged_sources_and_still_reviews_noop(workspace):
    planned(workspace)
    repo, _, config, _ = workspace
    first = explore.start(config, cwd=repo, overrides=options(), transport=ExploringTransport())
    git(repo, "merge", "--ff-only", first["candidate_commit"])
    native = ExploringTransport(editor=NoopTransport())
    second = explore.start(config, cwd=repo, overrides=options(), transport=native)
    assert second["job_id"] != first["job_id"]
    assert second["state"] == "unchanged"
    assert [task["role"] for task in native.calls] == ["explorer"] * 4 + ["author", "reviewer"]
    first_plan, second_plan = [json.loads(Path(value["plan_path"]).read_bytes()) for value in (first, second)]
    assert [u["sha256"] for u in first_plan["units"]] == [u["sha256"] for u in second_plan["units"]]
    assert first_plan["maintenance"] != second_plan["maintenance"]
    assert explore.status(config, cwd=repo)["job_id"] == second["job_id"]


def test_shared_budget_leaves_closeout_reserved_and_never_resets_between_jobs(workspace, clock):
    repo, _, config, _ = workspace
    shared = repo.parent.parent / "shared-budget"
    native = ExploringTransport(no_selection=True)
    jobs = [explore.start(config, cwd=repo, overrides=options(max_total_calls=8), budget_path=shared,
        transport=native) for _ in range(4)]
    with pytest.raises(dream_budget.Exhausted, match="reserved for authorship, review"):
        explore.start(config, cwd=repo, overrides=options(max_total_calls=8), budget_path=shared, transport=native)
    summary = explore.budget_status(config, Path(jobs[-1]["job_path"]))
    assert summary["reserved_calls"] == 4 and summary["remaining_calls"] == 4
    assert len(native.calls) == 4 and all(task["role"] == "explorer" for task in native.calls)
    assert {header(job["job_path"])["budget_path"] for job in jobs} == {str(shared)}


def test_immediate_selection_cannot_spend_the_last_author_invocation(workspace, clock):
    planned(workspace)
    repo, _, config, _ = workspace
    shared = repo.parent.parent / "shared-budget"
    spent = ExploringTransport(no_selection=True)
    for _ in range(4):
        explore.start(config, cwd=repo, overrides=options(max_total_calls=8), budget_path=shared, transport=spent)
    prepared = explore.prepare(config, cwd=repo, overrides=options(max_total_calls=8), budget_path=shared)
    class FinishingTransport(ExploringTransport):
        def __call__(self, harness, model, task, schema, deadline):
            result, usage, trace = super().__call__(harness, model, task, schema, deadline)
            result["selections"] = [dict(path=path, start=0, end=0)
                for path in ("evidence-a.org", "evidence-b.org")]
            trace["stdout"] = "\n".join(json.dumps(event) for event in (
                dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(result))),
                dict(type="turn.completed", usage={}))) + "\n"
            return result, usage, trace
    native = FinishingTransport(no_selection=True)
    job = Path(prepared["job_path"])
    with pytest.raises(dream_budget.Exhausted, match="reserved for authorship, review"):
        explore.run(config, job, transport=native)
    assert native.calls == []
    report = explore.budget_status(config, job)
    assert report["reserved_calls"] == 4 and report["remaining_calls"] == 4
    assert not list(job.glob("dream/*/plan.json"))


def test_failed_discovery_is_not_silently_repeated(workspace):
    repo, _, config, _ = workspace
    prepared = explore.prepare(config, cwd=repo, overrides=options())
    native = ExploringTransport(fail=True)
    job = Path(prepared["job_path"])
    with pytest.raises(dream.DreamError, match="Native role failed"):
        explore.run(config, job, transport=native)
    with pytest.raises(dream.DreamError, match="earlier role attempt has no validated result"):
        explore.run(config, job, transport=native)
    assert len(native.calls) == 1
    summary = explore.status(config, job)
    assert summary["state"] == "incomplete" and summary["budget"]["reserved_calls"] == 1


def test_unknown_provider_outcome_keeps_its_reservation_without_automatic_retry(workspace):
    repo, _, config, _ = workspace
    prepared = explore.prepare(config, cwd=repo, overrides=options())
    class InterruptedTransport(ExploringTransport):
        def __call__(self, harness, model, task, schema, deadline):
            self.calls.append(copy.deepcopy(task))
            raise OSError("Provider outcome was not returned")
    native = InterruptedTransport()
    job = Path(prepared["job_path"])
    with pytest.raises(OSError, match="outcome was not returned"):
        explore.run(config, job, transport=native)
    with pytest.raises(dream.DreamError, match="earlier role attempt has no validated result"):
        explore.run(config, job, transport=native)
    report = explore.budget_status(config, job)
    assert report["reserved_calls"] == report["call_states"]["uncertain"] == 1
    assert report["usage"]["calls_without_token_usage"] == 1
    assert report["usage"]["reported_cost_usd"] is None
    assert len(native.calls) == 1


def test_moved_publication_base_during_discovery_stops_before_editor(workspace):
    planned(workspace)
    repo, _, config, _ = workspace
    prepared = explore.prepare(config, cwd=repo, overrides=options())
    native = ExploringTransport(on_finish=lambda: git(repo, "commit", "--allow-empty", "-qm", "Concurrent human commit"))
    job = Path(prepared["job_path"])
    with pytest.raises(dream.DreamError, match="publication base changed"):
        explore.run(config, job, transport=native)
    assert [task["role"] for task in native.calls] == ["explorer"] * 4
    assert not list(job.glob("dream/*/plan.json"))
    assert explore.status(config, job)["budget"]["reserved_calls"] == 4


@pytest.mark.parametrize("stage", ["plan-pointer.json", "result-000001.json"])
def test_handoff_interruption_recovers_the_exact_plan_and_published_result(workspace, monkeypatch, clock, stage):
    planned(workspace)
    repo, _, config, _ = workspace
    prepared = explore.prepare(config, cwd=repo, overrides=options())
    job = Path(prepared["job_path"])
    native = ExploringTransport()
    original = explore._store
    raised = []
    def interrupted(fd, name, value):
        if name == stage and not raised:
            raised.append(name)
            raise OSError("simulated handoff interruption")
        return original(fd, name, value)
    monkeypatch.setattr(explore, "_store", interrupted)
    with pytest.raises(OSError, match="simulated handoff"):
        explore.run(config, job, transport=native)
    paths = list(job.glob("dream/*/plan.json"))
    assert len(paths) == 1
    frozen = paths[0].read_bytes()
    calls_before = len(native.calls)
    assert calls_before == (4 if stage == "plan-pointer.json" else 6)
    monkeypatch.setattr(explore, "_store", original)
    if stage == "result-000001.json":
        clock.advance(901)
        def forbidden(*args, **kwargs):
            raise AssertionError("Published handoff recovery attempted native startup")
        monkeypatch.setattr(dream_native, "Transport", forbidden)
        result = explore.run(config, job)
        assert len(native.calls) == calls_before
    else:
        result = explore.run(config, job, transport=native)
        assert len(native.calls) == 6
    assert result["state"] == "candidate_ready" and result["plan_path"] == str(paths[0])
    assert list(job.glob("dream/*/plan.json")) == paths and paths[0].read_bytes() == frozen
    assert explore.status(config, job)["budget"]["reserved_calls"] == 6
