"""Local-tier memory survives the clone and rejects durable publication races."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import time

import pytest

from sucoder import dream, dream_explore as explore, dream_local, dream_memory, dream_run
from sucoder.config import DreamSettings
from sucoder.local_tier import build_prepare_script, work_path
from tests.test_dream import git, workspace  # noqa: F401
from tests.test_dream_explore import ExploringTransport, options
from tests.test_dream_memory import MEMORY, seed, source
from tests.test_dream_wiki import planned
from tests.test_observations import remote
from tests.test_mirror import build_manager


@pytest.fixture
def local(workspace):
    planned(workspace)
    repo, _, config, _ = workspace
    target = remote()
    target.mirror_root = config.mirror_root
    target.slurm.local_disk = str(repo.parent.parent / "local")
    config = replace(config, targets={"cluster": target})
    clone = Path(work_path(target.slurm.local_disk, "sample", 42))
    clone.parent.mkdir(parents=True)
    git(repo, "clone", "-q", "--no-hardlinks", str(repo), str(clone))
    git(clone, "config", "user.name", "Fixture")
    git(clone, "config", "user.email", "fixture@example.test")
    return clone, config


def test_dream_publishes_only_to_configured_durable_mirror(workspace, local):
    clone, config = local
    public = workspace[0].parent.parent / "public.git"
    git(workspace[0], "init", "--bare", "-q", str(public))
    git(workspace[0], "remote", "add", "origin", str(public))
    native = ExploringTransport()
    result = explore.start(config, cwd=clone, overrides=options(mode="project"), transport=native)
    assert result["state"] == "published"
    assert git(clone, "rev-parse", MEMORY) == git(workspace[0], "rev-parse", MEMORY) == result["candidate_commit"]
    assert git(public, "for-each-ref", "--format=%(refname)") == ""
    assert explore.status(config, result["job_id"])["validation"]["state"] == "published"
    calls = len(native.calls)
    assert explore.start(config, result["job_id"], transport=native) == result
    assert len(native.calls) == calls


def test_durable_race_retains_reviewed_work_and_reports_conflict(workspace, local):
    clone, config = local
    observed = []
    def move_mirror():
        # Another node publishes a valid independent seed after this job froze
        # the absence of the durable branch, before this job's local publication.
        observed.append(seed(workspace))
    native = ExploringTransport(on_finish=move_mirror)
    prepared = explore.prepare(config, cwd=clone, overrides=options(mode="project"))
    job = Path(prepared["job_path"])
    with pytest.raises(dream_local.Conflict, match="moved since planning"):
        explore.run(config, job, transport=native)
    assert git(workspace[0], "rev-parse", MEMORY) == observed[0]
    retained = git(clone, "rev-parse", MEMORY)
    assert retained != observed[0]
    assert list(job.glob("dream/*/candidate-*.txt")) and list(job.glob("dream/*/native-*-receipt.json"))
    checked = explore.status(config, job)
    assert checked["validation"]["state"] == "conflict"


def test_prepare_imports_durable_tip_before_freezing_navigation(workspace, local):
    commit = seed(workspace)
    clone, config = local
    job = explore.prepare(config, cwd=clone, overrides=options(mode="project"))
    header = json.loads((Path(job["job_path"]) / "job.json").read_bytes())
    assert git(clone, "rev-parse", MEMORY) == commit
    assert header["memory_origin"]["expected_tip"] == commit
    assert header["leads"]["commit"] == commit


def test_an_ordinary_checkout_and_origin_url_do_not_authorize_publication(workspace, monkeypatch):
    repo, _, config, _ = workspace
    public = repo.parent.parent / "public.git"
    git(repo, "init", "--bare", "-q", str(public))
    git(repo, "remote", "add", "origin", str(public))
    planned(workspace)
    monkeypatch.setenv("SUCODER_LOCAL_ROOT", str(repo.parent))
    result = explore.start(config, cwd=repo, overrides=options(mode="project"), transport=ExploringTransport())
    assert result["state"] == "published" and git(public, "for-each-ref", "--format=%(refname)") == ""
    header = json.loads((Path(result["job_path"]) / "job.json").read_bytes())
    assert "memory_origin" not in header


def test_resume_rechecks_local_tier_configuration_before_provider_work(workspace, local):
    clone, config = local
    job = explore.prepare(config, cwd=clone, overrides=options(mode="project"))
    native = ExploringTransport()
    with pytest.raises(dream.DreamError, match="binding changed"):
        explore.start(replace(config, targets={}), job["job_id"], transport=native)
    assert not native.calls


def test_interrupted_durable_publication_replays_without_more_provider_calls(workspace, local, monkeypatch):
    clone, config = local
    prepared = explore.prepare(config, cwd=clone, overrides=options(mode="project"))
    native = ExploringTransport()
    original = dream_run._store
    interrupted = []
    def store(fd, name, data):
        if name.startswith("native-") and name.endswith("-run.json") and not interrupted:
            interrupted.append(name)
            raise OSError("after durable publication")
        return original(fd, name, data)
    monkeypatch.setattr(dream_run, "_store", store)
    with pytest.raises(OSError, match="after durable publication"):
        explore.run(config, prepared["job_path"], transport=native)
    tip = git(workspace[0], "rev-parse", MEMORY)
    calls = len(native.calls)
    result = explore.start(config, prepared["job_id"], transport=native)
    assert result["state"] == "published" and result["candidate_commit"] == tip
    assert git(clone, "rev-parse", MEMORY) == tip and len(native.calls) == calls


@pytest.mark.parametrize("branch", ["sucoder/memory", "private/dream-memory"])
def test_fresh_prepared_clone_has_local_memory_branch(workspace, branch):
    repo = workspace[0]
    index = repo / ".sucoder/wiki/index.org"
    index.parent.mkdir(parents=True)
    index.write_text("#+TITLE: Memory\n* Project lessons\nA maintained project lesson.\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "seed memory index")
    commit = seed(workspace)
    if branch != "sucoder/memory":
        git(repo, "branch", branch, MEMORY)
    local_root = repo.parent.parent / "local"
    script = build_prepare_script(mirror_path=str(repo), mirror_token="sample", local_disk_root=str(local_root),
        job_id=99, memory_branch=branch)
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    clone = Path(work_path(str(local_root), "sample", 99))
    assert git(clone, "rev-parse", "refs/heads/" + branch) == commit
    launcher = repo.parent.parent / "launcher"
    launcher.mkdir()
    manager = build_manager(launcher)
    ctx = manager.context_for("sample")
    ctx = replace(ctx, config=replace(ctx.config, mirror_root=clone.parent),
        settings=replace(ctx.settings, dream=DreamSettings(memory_branch=branch, checkout=False)))
    assert "PROJECT MEMORY" in manager._memory_block(ctx)
    assert "A maintained project lesson" in manager._memory_block(ctx)
