"""Project exploration, durable editorial handoff and inference-free recovery."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import pytest

from sucoder import dream, dream_budget, dream_discover, dream_explore as explore, dream_native, dream_run
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
    assert result["state"] == "published" and result["candidate_ref"] is None
    assert git(repo, "rev-parse", "refs/heads/sucoder/memory") == result["candidate_commit"]
    assert not [r for r in git(repo, "for-each-ref", "--format=%(refname)").split() if r.startswith("refs/heads/dream/")]
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
    assert checked["state"] == "published" and checked["budget"]["state"] == "budget_exhausted"
    assert checked["budget"]["reserved_calls"] == 6 and len(native.calls) == 6


def commit_bulk_directory(repo, count, width=240):
    """Commit ``bulk/`` holding ``count`` long-named copies of one blob, by plumbing."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}

    def plumb(*args, data=b""):
        return subprocess.run(["git", "-C", str(repo), *args], input=data, env=env,
                              check=True, capture_output=True).stdout.decode().strip()
    blob = plumb("hash-object", "-w", "--stdin", data=b"bulk\n")
    rows = "".join(f"100644 blob {blob}\t{i:05d}{'x' * (width - 5)}\n" for i in range(count))
    bulk = plumb("mktree", data=rows.encode())
    tree = plumb("mktree", data=(plumb("ls-tree", "HEAD") + f"\n040000 tree {bulk}\tbulk\n").encode())
    commit = plumb("commit-tree", tree, "-p", "HEAD", "-m", "Bulk directory")
    plumb("update-ref", "refs/heads/main", commit)
    plumb("read-tree", "-u", "--reset", "HEAD")
    return commit


def test_a_project_listing_past_the_plan_bound_is_explored_and_replayed(workspace, monkeypatch, clock):
    # LSMS_Library (19,466 files; a 2.3 MB listing) was refused on 2026-11-13 with
    # "Git inspection exceeds its byte allowance": the inventory shared the 2 MiB
    # plan bound.  7,200 paths of 245 bytes list past 2 MiB with or without sizes.
    planned(workspace)
    repo, _, config, _ = workspace
    commit = commit_bulk_directory(repo, 7200)
    assert len(subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", "-l", "-z", commit])) > dream_run.MAX_PLAN_BYTES
    files = len(git(repo, "ls-tree", "-r", "--name-only", commit).splitlines())
    native = ExploringTransport()
    result = explore.start(config, cwd=repo, overrides=options(), transport=native)
    assert result["state"] == "published" and result["project"]["base_commit"] == commit
    job = Path(result["job_path"])
    stored = json.loads((job / "inventory.json").read_bytes())
    assert stored["commit"] == commit and len(stored["files"]) == files
    listed = [task["input"]["history"] for task in native.calls if task["role"] == "explorer"][-1][0]["observation"]
    assert listed["inventory_files"] == files and listed["state"] == "complete"
    # Replay re-derives the inventory and compares it with the stored copy.
    clock.advance(901)

    def forbidden(*args, **kwargs):
        raise AssertionError("Cached exploratory replay attempted native startup")
    monkeypatch.setattr(dream_native, "Transport", forbidden)
    assert explore.start(config, job) == result
    assert explore.status(config, job)["state"] == "published"


class SearchingTransport(ExploringTransport):
    """Searches one file path, then finishes with no selection."""

    def __init__(self, path, query):
        super().__init__(no_selection=True)
        self.path, self.query = path, query

    def __call__(self, harness, model, task, schema, deadline):
        result, usage, trace = super().__call__(harness, model, task, schema, deadline)
        if task["role"] == "explorer" and not task["input"]["history"]:
            result.update(action="search", path=self.path, query=self.query)
            trace["stdout"] = "\n".join(json.dumps(event) for event in (
                dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(result))),
                dict(type="turn.completed", usage={}))) + "\n"
        return result, usage, trace


def test_navigation_is_frozen_per_job_so_earlier_jobs_still_replay(workspace, monkeypatch, clock):
    repo, _, config, _ = workspace
    (repo / "deep").mkdir()
    (repo / "deep" / "long.txt").write_text("".join(
        "def needle_gate(self):\n" if n == 4941 else f"line {n}\n" for n in range(1, 10635)))
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "deep file")
    searches = {}
    for version in (1, 2):
        # A version-1 job is one prepared before navigation versions existed: its header has no key.
        monkeypatch.setattr(dream_discover, "NAVIGATION", version)
        native = SearchingTransport("deep/long.txt", "def needle_gate")
        result = explore.start(config, cwd=repo, overrides=options(), transport=native)
        assert result["state"] == "no_selection"
        assert header(result["job_path"]).get("navigation_version") == (None if version == 1 else 2)
        searches[version] = (result, native.calls[1]["input"]["history"][0]["observation"])
    assert searches[1][1] == dict(state="complete", matches=[], next_offset=0, searched_bytes=0, searched_files=0, omitted=[])
    assert searches[2][1]["matches"] == [dict(path="deep/long.txt", line=4941, text="def needle_gate(self):")]
    clock.advance(901)

    def forbidden(*args, **kwargs):
        raise AssertionError("Cached exploratory replay attempted native startup")
    monkeypatch.setattr(dream_native, "Transport", forbidden)
    for result, _ in searches.values():
        assert explore.start(config, Path(result["job_path"])) == result
        assert explore.status(config, Path(result["job_path"]))["state"] == "no_selection"


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
    assert first["state"] == "published"  # In place: no human merge stands between the two jobs.
    native = ExploringTransport(editor=NoopTransport())
    second = explore.start(config, cwd=repo, overrides=options(), transport=native)
    assert second["job_id"] != first["job_id"]
    assert second["state"] == "unchanged"
    assert [task["role"] for task in native.calls] == ["explorer"] * 4 + ["author", "reviewer"]
    first_plan, second_plan = [json.loads(Path(value["plan_path"]).read_bytes()) for value in (first, second)]
    assert [u["sha256"] for u in first_plan["units"]] == [u["sha256"] for u in second_plan["units"]]
    assert second_plan["base_commit"] == first["candidate_commit"]
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
    # Legacy candidate mode: the launch branch is the publication base.
    prepared = explore.prepare(config, cwd=repo, overrides=options(memory_branch=""))
    native = ExploringTransport(on_finish=lambda: git(repo, "commit", "--allow-empty", "-qm", "Concurrent human commit"))
    job = Path(prepared["job_path"])
    with pytest.raises(dream.DreamError, match="publication base changed"):
        explore.run(config, job, transport=native)
    assert [task["role"] for task in native.calls] == ["explorer"] * 4
    assert not list(job.glob("dream/*/plan.json"))
    assert explore.status(config, job)["budget"]["reserved_calls"] == 4
    assert "refs/heads/sucoder/memory" not in git(repo, "for-each-ref", "--format=%(refname)").split()


def test_a_launch_branch_commit_during_discovery_does_not_stop_a_memory_dream(workspace):
    planned(workspace)
    repo, _, config, _ = workspace
    launch = git(repo, "rev-parse", "HEAD")
    native = ExploringTransport(on_finish=lambda: git(repo, "commit", "--allow-empty", "-qm", "Concurrent human commit"))
    result = explore.start(config, cwd=repo, overrides=options(), transport=native)
    assert result["state"] == "published" and result["project"]["base_commit"] == launch
    plan = json.loads(Path(result["plan_path"]).read_bytes())
    assert all(u["commit"] == launch for u in plan["units"])  # Evidence stays pinned to the launch commit.
    assert git(repo, "rev-parse", "HEAD") != launch


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
    assert result["state"] == "published" and result["plan_path"] == str(paths[0])
    assert list(job.glob("dream/*/plan.json")) == paths and paths[0].read_bytes() == frozen
    assert explore.status(config, job)["budget"]["reserved_calls"] == 6
