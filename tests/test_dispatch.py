"""Headless control plane: disposable repositories and fake scheduler only."""
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from sucoder import dispatch
from sucoder.local_tier import build_worker_prepare_script, build_worker_snapshot_script


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "cluster")
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.com")
    (repo / "tracked").write_text("base\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    executable = tmp_path / "agent"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setattr(dispatch, "_shared_path", lambda path, label: path.expanduser().resolve())
    monkeypatch.setattr(dispatch, "DEFAULT_DIRECTORY_ROOT", tmp_path / "results")
    return dict(repo=repo, ref="HEAD", task_file=repo / "tracked", agent="codex",
                executable=executable, resources=dict(account="a", partition="p", qos="q", cpus=1, mem="1G", time="00:05:00"))


def scheduler(monkeypatch, response="123;cluster", error=None, returncode=0):
    real = dispatch._run
    calls = []
    def run(argv, **kwargs):
        if argv[0] == "sbatch":
            calls.append(argv)
            if error:
                raise error
            return SimpleNamespace(stdout=response, stderr="", returncode=returncode)
        return real(argv, **kwargs)
    monkeypatch.setattr(dispatch, "_run", run)
    return calls


def worker_record(report, **updates):
    manifest = json.loads((Path(report["directory"]) / "manifest.json").read_text())
    record = {key: manifest[key] for key in ("schema_version", "dispatch_id", "uid", "commit", "branch", "agent")}
    record.update(job_id=str(report["job_id"]), cluster=report["cluster"], state="running")
    record.update(updates)
    return record


def test_dry_run_has_no_mutation_or_scheduler(source, monkeypatch):
    calls = scheduler(monkeypatch)
    before = git(source["repo"], "show-ref")
    report = dispatch.submit(**source, dry_run=True)
    assert report["state"] == "dry_run"
    assert not calls
    assert not Path(report["directory"]).exists()
    assert git(source["repo"], "show-ref") == before
    assert "--export=NONE" in report["command"]
    assert "--no-requeue" in report["command"]
    assert "--signal=B:TERM@120" in report["command"]
    assert "--comment=sucoder-dispatch:" + report["dispatch_id"] in report["command"]


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan"), 301])
def test_publication_budget_refuses_before_staging(source, value):
    with pytest.raises(ValueError, match="publication timeout"):
        dispatch.submit(**source, publication_timeout=value)
    assert not dispatch.DEFAULT_DIRECTORY_ROOT.exists()


def test_publication_budget_is_frozen_and_extends_scheduler_notice(source, monkeypatch):
    scheduler(monkeypatch)
    report = dispatch.submit(**source, publication_timeout=120.5)
    manifest = json.loads((Path(report['directory']) / 'manifest.json').read_text())
    assert manifest['publication_timeout'] == 120.5
    assert manifest['checkpoint_timeout'] == 30
    assert '--signal=B:TERM@181' in report['command']
    source['resources']['time'] = '00:02:00'
    with pytest.raises(ValueError, match='181-second'):
        dispatch.submit(**source, publication_timeout=120.5)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -float("inf"), True, "5", None, 0, -1])
def test_submit_rejects_invalid_timeout_before_source_or_scheduler(source, monkeypatch, timeout):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid timeout reached source inspection or scheduler")
    monkeypatch.setattr(dispatch, "_git", forbidden)
    monkeypatch.setattr(dispatch, "_run", forbidden)
    with pytest.raises(ValueError, match="finite and positive"):
        dispatch.submit(**source, timeout=timeout)
    assert not dispatch.DEFAULT_DIRECTORY_ROOT.exists()


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -float("inf"), True, "5", None, 0, -1])
def test_status_rejects_invalid_timeout_before_record_or_scheduler(monkeypatch, timeout):
    monkeypatch.setattr(dispatch, "_directory", lambda *args: pytest.fail("invalid timeout opened records"))
    with pytest.raises(ValueError, match="finite and positive"):
        dispatch.status("unused", timeout=timeout)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -float("inf"), True, "5", None, -1])
def test_wait_rejects_invalid_timeout_before_polling(monkeypatch, timeout):
    monkeypatch.setattr(dispatch, "status", lambda *args, **kwargs: pytest.fail("invalid timeout polled scheduler"))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        dispatch.wait("unused", timeout=timeout)


@pytest.mark.parametrize("poll_seconds", [float("nan"), float("inf"), -float("inf"), True, "5", None, 0, -1])
def test_wait_rejects_invalid_poll_before_polling(monkeypatch, poll_seconds):
    monkeypatch.setattr(dispatch, "status", lambda *args, **kwargs: pytest.fail("invalid poll interval polled scheduler"))
    with pytest.raises(ValueError, match="finite and positive"):
        dispatch.wait("unused", timeout=0, poll_seconds=poll_seconds)


def test_zero_wait_timeout_keeps_one_bounded_observation(monkeypatch):
    calls = []
    def observe(directory, *, timeout):
        calls.append(timeout)
        return {"result_state": "missing", "state": "running"}
    monkeypatch.setattr(dispatch, "status", observe)
    report = dispatch.wait("unused", timeout=0)
    assert report["wait_timed_out"]
    assert calls == [0.001]


def test_scheduler_rejects_invalid_timeout_before_contact(monkeypatch):
    monkeypatch.setattr(dispatch, "_run", lambda *args, **kwargs: pytest.fail("invalid timeout contacted scheduler"))
    with pytest.raises(ValueError, match="finite and positive"):
        dispatch._scheduler({"job_id": 123, "cluster": "cluster"}, float("nan"))


def test_submit_pins_private_source_and_allowlisted_secrets(source, monkeypatch):
    calls = scheduler(monkeypatch)
    for name in dispatch.ENVIRONMENT_ALLOWLIST:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("SUCODER_CODEX_STATE_RUN", "/local/old/run")
    monkeypatch.setenv("CODEX_HOME", "/local/old/home")
    (source["repo"] / "tracked").write_text("dirty")
    git(source["repo"], "add", "tracked")
    before = (git(source["repo"], "show-ref"), git(source["repo"], "status", "--porcelain"), (source["repo"] / ".git/index").read_bytes(), (source["repo"] / ".git/config").read_bytes())
    report = dispatch.submit(**source)
    directory = Path(report["directory"])
    assert report["job_id"] == 123 and report["cluster"] == "cluster"
    assert len(calls) == 1
    assert git(directory / "results.git", "rev-parse", report["commit"]) == report["commit"]
    assert git(directory / "results.git", "rev-parse", "dispatch/" + report["dispatch_id"]) == report["commit"]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["source_agent_home"] != "/local/old/home"
    assert json.loads((directory / "environment.json").read_text()) == {"OPENAI_API_KEY": "test-secret"}
    for name in ("manifest.json", "environment.json", "task.txt", "worker.py", "runner.py", "dispatch_protocol.py", "dispatch_artifacts.py", "submission.json"):
        assert (directory / name).stat().st_mode & 0o077 == 0
    assert directory.stat().st_mode & 0o077 == 0
    assert (directory / "evidence").is_dir() and (directory / "evidence").stat().st_mode & 0o077 == 0
    assert (git(source["repo"], "show-ref"), git(source["repo"], "status", "--porcelain"), (source["repo"] / ".git/index").read_bytes(), (source["repo"] / ".git/config").read_bytes()) == before


def test_submission_timeout_is_unknown_and_never_retries(source, monkeypatch):
    calls = scheduler(monkeypatch, error=subprocess.TimeoutExpired("sbatch", 1))
    report = dispatch.submit(**source, timeout=1)
    assert report["state"] == "submission_unknown"
    assert len(calls) == 1
    assert json.loads((Path(report["directory"]) / "submission.json").read_text())["state"] == "submission_unknown"


def test_partial_source_refuses_before_staging(source, monkeypatch):
    calls = scheduler(monkeypatch)
    git(source["repo"], "config", "remote.gate.promisor", "true")
    with pytest.raises(ValueError, match="partial/promisor"):
        dispatch.submit(**source)
    assert not calls and not dispatch.DEFAULT_DIRECTORY_ROOT.exists()


@pytest.mark.parametrize("path", ["/local/job123/repo", "/tmp/a", "/dev/shm/a"])
def test_node_local_paths_refuse(path):
    with pytest.raises(ValueError, match="shared storage"):
        dispatch._shared_path(Path(path), "source")


def test_status_empty_scheduler_is_not_success(source, monkeypatch):
    scheduler(monkeypatch)
    report = dispatch.submit(**source)
    monkeypatch.setattr(dispatch, "_run", lambda argv, **kw: SimpleNamespace(stdout="", stderr="", returncode=0))
    status = dispatch.status(report["directory"])
    assert status["result_state"] == "missing"
    assert status["scheduler"]["state"] == "unknown"
    assert status["state"] == "submitted"
    assert dispatch.wait(report["directory"], timeout=0)["wait_timed_out"]
    (Path(report["directory"]) / "result.json").write_text("bad JSON")
    assert dispatch.read_result(report["directory"])["result_state"] == "unreadable"


def test_result_requires_identity(source, monkeypatch):
    scheduler(monkeypatch)
    report = dispatch.submit(**source)
    directory = Path(report["directory"])
    result = worker_record(report, dispatch_id="0" * 32, state="succeeded", exit_code=0)
    (directory / "result.json").write_text(json.dumps(result))
    assert dispatch.read_result(directory)["result_state"] == "unreadable"
    result["dispatch_id"] = report["dispatch_id"]
    (directory / "result.json").write_text(json.dumps(result))
    monkeypatch.setattr(dispatch, "_run", lambda *args, **kw: (_ for _ in ()).throw(OSError()))
    answer = dispatch.wait(directory, timeout=0)
    assert answer["state"] == "succeeded" and not answer["wait_timed_out"]
    assert not answer["scheduler"]["available"]


def test_worker_scripts_pin_publish_and_snapshot_without_touching_index(source, tmp_path):
    identifier = "a" * 32
    commit = git(source["repo"], "rev-parse", "HEAD")
    results = tmp_path / "private.git"
    subprocess.check_call(["git", "clone", "-q", "--bare", str(source["repo"]), str(results)])
    root = tmp_path / "local"
    prepare = tmp_path / "prepare.sh"
    snapshot = tmp_path / "snapshot.sh"
    prepare.write_text(build_worker_prepare_script(results_path=str(results), dispatch_id=identifier, commit=commit, local_disk_root=str(root)))
    snapshot.write_text(build_worker_snapshot_script(results_path=str(results), dispatch_id=identifier, local_disk_root=str(root)))
    env = dict(os.environ, SLURM_JOB_ID="123")
    subprocess.check_call(["bash", str(prepare)], env=env)
    work = root / "job123/mirrors" / ("dispatch_" + identifier)
    assert git(work, "rev-parse", "HEAD") == commit
    assert git(work, "symbolic-ref", "--short", "HEAD") == "dispatch/" + identifier
    (work / "tracked").write_text("staged\n")
    git(work, "add", "tracked")
    (work / "tracked").write_text("unstaged\n")
    (work / "untracked").write_text("new\n")
    index_before = (work / ".git/index").read_bytes()
    first = json.loads(subprocess.check_output(["bash", str(snapshot)], env=env, text=True))
    second = json.loads(subprocess.check_output(["bash", str(snapshot)], env=env, text=True))
    assert first["ref"] != second["ref"]
    assert git(results, "show", first["ref"] + ":tracked") == "unstaged"
    assert git(results, "show", first["ref"] + ":untracked") == "new"
    assert (work / ".git/index").read_bytes() == index_before
    assert git(work, "rev-parse", "HEAD") == commit
    assert git(source["repo"], "rev-parse", "HEAD") == commit
    git(work, "commit", "-qm", "worker commit")
    assert git(results, "rev-parse", "dispatch/" + identifier) == git(work, "rev-parse", "HEAD")
    assert "--force" not in snapshot.read_text()


def test_optional_qos_and_excluded_node_list(source, monkeypatch):
    scheduler(monkeypatch)
    source["resources"].update(qos=None, exclude="node1,node2")
    report = dispatch.submit(**source, dry_run=True)
    assert "--exclude=node1,node2" in report["command"]
    assert not any(arg.startswith("--qos=") for arg in report["command"])
    source["resources"]["exclude"] = "node1;bad"
    with pytest.raises(ValueError, match="excluded node"):
        dispatch.submit(**source, dry_run=True)


def test_scheduler_shares_deadline_across_queue_and_accounting(monkeypatch):
    times = iter([0, 1, 3])
    monkeypatch.setattr(dispatch.time, "monotonic", lambda: next(times))
    calls = []
    def query(argv, **kw):
        calls.append(kw["timeout"])
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(dispatch, "_run", query)
    result = dispatch._scheduler(dict(job_id=123, uid=os.getuid(), cluster="cluster"), 2)
    assert calls == [1]
    assert not result["available"]


def test_shared_protocol_staged_and_digest_matches(source, monkeypatch):
    import hashlib
    from sucoder import dispatch_protocol
    scheduler(monkeypatch)
    report = dispatch.submit(**source)
    directory = Path(report["directory"])
    assert dispatch.ENVIRONMENT_ALLOWLIST is dispatch_protocol.ENVIRONMENT_ALLOWLIST
    assert (directory / "dispatch_protocol.py").read_bytes() == Path(dispatch_protocol.__file__).read_bytes()
    digests = json.loads((directory / "digests.json").read_text())
    assert digests["dispatch_protocol.py"] == hashlib.sha256((directory / "dispatch_protocol.py").read_bytes()).hexdigest()


def test_runtime_path_is_narrow_and_supports_shared_node(source, monkeypatch, tmp_path):
    agent = source["executable"]
    agent.write_text("#!/usr/bin/env node\n")
    node = tmp_path / "nodebin/node"
    node.parent.mkdir()
    node.write_text("node")
    monkeypatch.setattr(dispatch.shutil, "which", lambda name: str(node))
    assert dispatch._runtime_path(agent) == [str(agent.parent), str(node.parent), "/usr/local/bin", "/usr/bin", "/bin"]
    monkeypatch.setattr(dispatch.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="no shared Node"):
        dispatch._runtime_path(agent)


@pytest.mark.parametrize("answer", ["", "123;cluster"])
def test_nonzero_sbatch_postexecution_remains_unknown(source, monkeypatch, answer):
    calls = scheduler(monkeypatch, response=answer, returncode=1)
    report = dispatch.submit(**source)
    assert report["state"] == "submission_unknown"
    assert report["returncode"] == 1 and len(calls) == 1
    if answer:
        assert report["job_id"] == 123 and report["cluster"] == "cluster"


@pytest.mark.parametrize("key,value", [("dispatch_id", "f" * 32), ("uid", -1), ("cluster", "other")])
def test_foreign_receipt_never_queries_scheduler(source, monkeypatch, key, value):
    scheduler(monkeypatch)
    report = dispatch.submit(**source)
    directory = Path(report["directory"])
    receipt = json.loads((directory / "submission.json").read_text())
    receipt[key] = value
    (directory / "submission.json").write_text(json.dumps(receipt))
    calls = []
    monkeypatch.setattr(dispatch, "_run", lambda *args, **kwargs: calls.append(args))
    observed = dispatch.status(directory)
    assert observed["submission_state"] == "unreadable" and not calls


def test_missing_cluster_does_not_query_arbitrary_current_cluster(source, monkeypatch):
    monkeypatch.delenv("SLURM_CLUSTER_NAME")
    scheduler(monkeypatch, response="123")
    report = dispatch.submit(**source)
    calls = []
    monkeypatch.setattr(dispatch, "_run", lambda *args, **kwargs: calls.append(args))
    observed = dispatch.status(report["directory"])
    assert observed["scheduler"]["state"] == "unknown_source" and not calls


@pytest.mark.parametrize("field,value", [("job_id", "999"), ("cluster", "other"),
                                               ("commit", "f" * 40), ("branch", "other"),
                                               ("agent", "claude"), ("schema_version", 2)])
def test_worker_result_source_must_match_receipt(source, monkeypatch, field, value):
    scheduler(monkeypatch)
    report = dispatch.submit(**source)
    directory = Path(report["directory"])
    result = worker_record(report, state="succeeded", exit_code=0)
    result[field] = value
    (directory / "result.json").write_text(json.dumps(result))
    (directory / "status.json").write_text(json.dumps(result))
    monkeypatch.setattr(dispatch, "_run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))
    observed = dispatch.status(directory)
    assert observed["result_state"] == "unreadable" and observed["worker_state"] == "unreadable"
    assert observed["state"] != "succeeded"


def test_automated_publication_ignores_mutated_origin(source, tmp_path):
    from sucoder.local_tier import POST_COMMIT_HOOK, build_post_commit_hook
    assert build_post_commit_hook() == POST_COMMIT_HOOK
    identifier = "c" * 32
    commit = git(source["repo"], "rev-parse", "HEAD")
    before = git(source["repo"], "show-ref")
    results = tmp_path / "private.git"
    subprocess.check_call(["git", "clone", "-q", "--bare", str(source["repo"]), str(results)])
    root = tmp_path / "disk"
    prepare = tmp_path / "prepare.sh"
    snapshot = tmp_path / "snapshot.sh"
    prepare.write_text(build_worker_prepare_script(results_path=str(results), dispatch_id=identifier, commit=commit, local_disk_root=str(root)))
    snapshot.write_text(build_worker_snapshot_script(results_path=str(results), dispatch_id=identifier, local_disk_root=str(root)))
    env = dict(os.environ, SLURM_JOB_ID="123")
    subprocess.check_call(["bash", str(prepare)], env=env)
    work = root / "job123/mirrors" / ("dispatch_" + identifier)
    git(work, "remote", "set-url", "origin", str(source["repo"]))
    (work / "tracked").write_text("committed worker change")
    git(work, "commit", "-qam", "private commit")
    assert git(results, "rev-parse", "dispatch/" + identifier) == git(work, "rev-parse", "HEAD")
    (work / "tracked").write_text("dirty worker change")
    saved = json.loads(subprocess.check_output(["bash", str(snapshot)], env=env, text=True))
    assert git(results, "show", saved["ref"] + ":tracked") == "dirty worker change"
    assert git(source["repo"], "show-ref") == before


@pytest.mark.parametrize("redirection", ["gitfile", "gitdir_symlink", "work_symlink",
                                         "core_worktree", "common_dir", "objects_symlink"])
def test_automatic_snapshot_refuses_redirected_clone_metadata(source, tmp_path, redirection):
    identifier = "d" * 32
    commit = git(source["repo"], "rev-parse", "HEAD")
    results = tmp_path / "private.git"
    subprocess.check_call(["git", "clone", "-q", "--bare", str(source["repo"]), str(results)])
    root = tmp_path / "disk"
    prepare, snapshot = tmp_path / "prepare.sh", tmp_path / "snapshot.sh"
    prepare.write_text(build_worker_prepare_script(results_path=str(results), dispatch_id=identifier,
                                                  commit=commit, local_disk_root=str(root)))
    snapshot.write_text(build_worker_snapshot_script(results_path=str(results), dispatch_id=identifier,
                                                    local_disk_root=str(root)))
    env = dict(os.environ, SLURM_JOB_ID="123")
    subprocess.check_call(["bash", str(prepare)], env=env)
    work = root / "job123/mirrors" / ("dispatch_" + identifier)
    source_git = source["repo"] / ".git"
    before = ((source_git / "index").read_bytes(), (source_git / "config").read_bytes(),
              git(source["repo"], "show-ref"), git(results, "show-ref"),
              sorted(str(path.relative_to(source_git / "objects")) for path in (source_git / "objects").rglob("*") if path.is_file()))
    if redirection in ("gitfile", "gitdir_symlink"):
        (work / ".git").rename(work / "saved-git")
        if redirection == "gitfile":
            (work / ".git").write_text("gitdir: %s\n" % source_git)
        else:
            (work / ".git").symlink_to(source_git, target_is_directory=True)
    elif redirection == "work_symlink":
        saved_work = root / "saved-work"
        work.rename(saved_work)
        work.symlink_to(source["repo"], target_is_directory=True)
    elif redirection == "core_worktree":
        git(work, "config", "core.worktree", str(source["repo"]))
    elif redirection == "common_dir":
        (work / ".git/commondir").write_text(str(source_git) + "\n")
    else:
        (work / ".git/objects").rename(work / "saved-objects")
        (work / ".git/objects").symlink_to(source_git / "objects", target_is_directory=True)
    outcome = subprocess.run(["bash", str(snapshot)], env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, timeout=5)
    assert outcome.returncode != 0
    assert "dispatch snapshot refused" in outcome.stderr
    after = ((source_git / "index").read_bytes(), (source_git / "config").read_bytes(),
             git(source["repo"], "show-ref"), git(results, "show-ref"),
             sorted(str(path.relative_to(source_git / "objects")) for path in (source_git / "objects").rglob("*") if path.is_file()))
    assert after == before


@pytest.mark.parametrize("limit", ["1", "01:59", "00:01:59"])
def test_short_jobs_refuse_insufficient_shutdown_signal_window(source, limit):
    source["resources"]["time"] = limit
    with pytest.raises(ValueError, match="120-second"):
        dispatch.submit(**source, dry_run=True)


def test_manifest_freezes_dispatch_adapter_and_requested_model(source, monkeypatch):
    from sucoder import __version__
    scheduler(monkeypatch)
    report = dispatch.submit(**source, model="test-model")
    directory = Path(report["directory"])
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["dispatcher_version"] == __version__
    assert manifest["adapter_version"] == 1
    assert manifest["requested_model"] == manifest["model"] == "test-model"
    assert manifest["created_at"] <= report["submission_started_at"] <= report["receipt_observed_at"]


@pytest.mark.parametrize("heartbeat,age,stale", [(990, 10, False), (900, 100, True), (None, None, None), ("bad", None, None)])
def test_heartbeat_freshness_is_an_observation_not_task_progress(source, monkeypatch, heartbeat, age, stale):
    scheduler(monkeypatch)
    report = dispatch.submit(**source)
    directory = Path(report["directory"])
    record = worker_record(report, heartbeat_at=heartbeat)
    (directory / "status.json").write_text(json.dumps(record))
    monkeypatch.setattr(dispatch, "_scheduler", lambda *args: dict(available=False, state="unavailable"))
    monkeypatch.setattr(dispatch.time, "time", lambda: 1000)
    observed = dispatch.status(directory)
    assert observed["observation_at"] == 1000
    assert observed["heartbeat_age_seconds"] == age and observed["stale"] is stale
    assert observed["state"] == "running"
    assert "progress" not in observed


def test_record_directory_must_not_be_a_symlink(source, monkeypatch, tmp_path):
    scheduler(monkeypatch)
    report = dispatch.submit(**source)
    alias = tmp_path / ("e" * 32)
    alias.symlink_to(report["directory"], target_is_directory=True)
    with pytest.raises(ValueError, match="redirect"):
        dispatch.read_result(alias)


def test_record_directory_name_must_match_its_dispatch_identity(source, monkeypatch):
    scheduler(monkeypatch)
    report = dispatch.submit(**source)
    directory = Path(report["directory"])
    renamed = directory.with_name("e" * 32)
    directory.rename(renamed)
    manifest = json.loads((renamed / "manifest.json").read_text())
    manifest["directory"] = str(renamed)
    (renamed / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        dispatch.read_result(renamed)


def test_required_evidence_is_frozen_and_helper_is_staged(source, monkeypatch):
    import hashlib
    scheduler(monkeypatch)
    report = dispatch.submit(**source, required_evidence=["report.json", "plots/result.svg"])
    directory = Path(report["directory"])
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["required_evidence"] == ["report.json", "plots/result.svg"]
    assert (directory / "dispatch_artifacts.py").is_file()
    digests = json.loads((directory / "digests.json").read_text())
    assert digests["dispatch_artifacts.py"] == hashlib.sha256((directory / "dispatch_artifacts.py").read_bytes()).hexdigest()


@pytest.mark.parametrize("name", ["", ".", "/absolute", "../parent", "a/../parent", "a//b", "a/./b", "nul\x00name", "windows\\path"])
def test_required_evidence_names_refuse_unsafe_paths_without_staging(source, name):
    with pytest.raises(ValueError, match="relative POSIX"):
        dispatch.submit(**source, required_evidence=[name], dry_run=True)
    assert not dispatch.DEFAULT_DIRECTORY_ROOT.exists()


def test_manifest_schema_must_be_supported(source, monkeypatch):
    scheduler(monkeypatch)
    report = dispatch.submit(**source)
    directory = Path(report["directory"])
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["schema_version"] = 2
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        dispatch.read_result(directory)


def test_shared_steering_channels_are_private_mutable_and_not_frozen(source, monkeypatch):
    scheduler(monkeypatch)
    report = dispatch.submit(**source)
    directory = Path(report["directory"])
    digests = json.loads((directory / "digests.json").read_text())
    for name in ("MESSAGES_TO_AGENT.txt", "QUESTIONS_FROM_AGENT.txt"):
        channel = directory / name
        assert channel.is_file() and channel.stat().st_mode & 0o077 == 0
        assert name not in digests
        with channel.open("a") as stream:
            stream.write("new steering\n")
        assert channel.read_text().endswith("new steering\n")
    assert "task.txt" in digests
