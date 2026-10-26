"""Project integration: real Git, fake agents, no model or survey data access."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

import pytest

from sucoder import dispatch, dispatch_project as project, dispatch_worker as worker
from tests.test_dispatch import source, scheduler
from tests.test_dispatch_worker import staged, save, fake_run, git, worker_subprocess, wait_for


BOOTSTRAP = r'''set -eu
test -z "${OPENAI_API_KEY+x}"
python3 - <<'PY'
import json, os
p = os.environ['SUCODER_PROJECT_RESULT']
with open(p, 'w') as f:
    json.dump({'schema_version': 1, 'state': 'ready',
               'environment': {'PROJECT_READY': 'yes'},
               'runtime': {'cwd': os.getcwd()}}, f)
PY
'''


def profile_file(tmp_path, script=BOOTSTRAP, **overrides):
    path = tmp_path / "profile.json"
    bootstrap = tmp_path / "bootstrap.sh"
    bootstrap.write_text(script)
    profile = {"schema_version": 1, "bootstrap": "bootstrap.sh", "timeout_seconds": 5,
               "environment": {"PROJECT_SOURCE": "{worktree}"},
               "local_directories": {"PROJECT_DATA": "data", "PROJECT_CONFIG": "config"},
               "output_environment": ["PROJECT_READY"],
               "thread_limits": {"OMP_NUM_THREADS": 1, "LSMS_BUILD_WORKERS": 1}}
    profile.update(overrides)
    path.write_text(json.dumps(profile))
    return path


def stage_worker(staged, path):
    prepared = project.read_profile(path)
    names = project.stage_profile(prepared, staged.control)
    shutil.copyfile(Path(project.__file__), staged.control / "dispatch_project.py")
    names.append("dispatch_project.py")
    staged.manifest["project"] = {"profile": "project.json", "sha256": prepared["sha256"], "helper": "dispatch_project.py"}
    for name in names:
        (staged.control / name).chmod(0o600)
    save(staged)
    digests = json.loads((staged.control / "digests.json").read_text())
    digests["environment.json"] = hashlib.sha256((staged.control / "environment.json").read_bytes()).hexdigest()
    digests.update({name: hashlib.sha256((staged.control / name).read_bytes()).hexdigest() for name in names})
    (staged.control / "digests.json").write_text(json.dumps(digests))
    return prepared


def run_worker(staged):
    return fake_run(staged, {"SLURM_JOB_ID": "234", "SLURM_CPUS_PER_TASK": "2", "PATH": os.environ["PATH"]})


@pytest.mark.parametrize("key", ["HOME", "PATH", "CODEX_HOME", "SUCODER_LOCAL_ROOT", "SLURM_JOB_ID",
                                 "GIT_CONFIG_GLOBAL", "XDG_CONFIG_HOME", "OPENAI_API_KEY", "BASH_ENV",
                                 "NODE_OPTIONS", "LD_AUDIT", "ANTHROPIC_CUSTOM_HEADERS"])
def test_profile_cannot_replace_agent_or_scheduler_context(tmp_path, key):
    path = profile_file(tmp_path, environment={key: "wrong"})
    with pytest.raises(ValueError, match="protected"):
        project.read_profile(path)


def test_profile_dry_run_does_not_stage_or_submit(source, monkeypatch, tmp_path):
    calls = scheduler(monkeypatch)
    path = profile_file(tmp_path)
    before = list(tmp_path.iterdir())
    result = dispatch.submit(**source, project_profile=path, dry_run=True)
    assert result["project"]["sha256"]
    assert not calls
    assert not Path(result["directory"]).exists()
    assert list(tmp_path.iterdir()) == before


def test_controller_freezes_helper_and_private_reader_bytes(source, monkeypatch, tmp_path):
    scheduler(monkeypatch)
    reader = tmp_path / "reader"
    reader.write_text("private-reader-token")
    reader.chmod(0o600)
    path = profile_file(tmp_path, credentials={"reader": str(reader)},
                        credential_environment={"LSMS_S3_CREDS": "reader"})
    report = dispatch.submit(**source, project_profile=path)
    directory = Path(report["directory"])
    reader.write_text("rotated")
    (tmp_path / "bootstrap.sh").write_text("exit 99")
    path.write_text("{}")
    assert (directory / "project-secret-reader").read_text() == "private-reader-token"
    assert (directory / "project-bootstrap.sh").read_text() == BOOTSTRAP
    assert "private-reader-token" not in (directory / "project.json").read_text()
    assert "private-reader-token" not in (directory / "manifest.json").read_text()
    manifest = worker.load_manifest(str(directory / "manifest.json"))
    assert manifest["project"]["sha256"] == report["project"]["sha256"]


def test_profile_environment_reaches_agent_after_isolated_setup(staged, tmp_path):
    path = profile_file(tmp_path)
    stage_worker(staged, path)
    executable = Path(staged.manifest["executable"])
    executable.write_text(executable.read_text() + "\nassert os.environ['PROJECT_READY'] == 'yes'\n"
                          "assert os.environ['PROJECT_SOURCE'] == os.getcwd()\n"
                          "assert os.environ['PROJECT_DATA'].startswith(os.environ['SUCODER_LOCAL_ROOT'] + '/')\n")
    before = git(staged.source, "show-ref"), git(staged.source, "status", "--porcelain")
    assert run_worker(staged) == 0
    result = json.loads((staged.control / "result.json").read_text())
    assert result["project"]["state"] == "ready"
    assert result["project"]["runtime"]["cwd"] == result["work_path"]
    assert (staged.control / "project-runtime.json").exists()
    assert before == (git(staged.source, "show-ref"), git(staged.source, "status", "--porcelain"))


def test_bad_bootstrap_stops_before_native_version_or_agent(staged, tmp_path):
    stage_worker(staged, profile_file(tmp_path, script="echo private-diagnostic >&2\nexit 17\n"))
    assert run_worker(staged) != 0
    result = json.loads((staged.control / "result.json").read_text())
    assert result["project"]["state"] == "failed"
    assert "private-diagnostic" not in json.dumps(result)
    assert not (staged.control / "native-version.log").exists()
    assert result["exit_code"] is None
    assert result["coordinator_acceptance"] == "pending"


def test_receiving_asset_change_refuses_before_native_launch(staged, tmp_path):
    asset = tmp_path / "image"
    asset.write_bytes(b"approved image")
    path = profile_file(tmp_path, assets={"runtime/image": {"path": str(asset), "sha256": project.digest_file(asset)}})
    stage_worker(staged, path)
    asset.write_bytes(b"different queued image")
    assert run_worker(staged) != 0
    result = json.loads((staged.control / "result.json").read_text())
    assert result["project"]["state"] == "failed"
    assert not (staged.control / "native-version.log").exists()


def test_bootstrap_cannot_change_tracked_source_before_agent(staged, tmp_path):
    stage_worker(staged, profile_file(tmp_path, script='echo changed > tracked.txt\n' + BOOTSTRAP))
    assert run_worker(staged) != 0
    result = json.loads((staged.control / "result.json").read_text())
    assert "changed tracked source" in result["reason"]
    assert not (staged.control / "native-version.log").exists()


def test_reader_is_local_private_and_model_secret_is_excluded_from_setup(staged, tmp_path):
    reader = tmp_path / "reader"
    reader.write_text("approved-reader")
    reader.chmod(0o600)
    script = 'test "$(cat "$LSMS_S3_CREDS")" = approved-reader\n' + BOOTSTRAP
    path = profile_file(tmp_path, script=script, credentials={"reader": str(reader)},
                        credential_environment={"LSMS_S3_CREDS": "reader"})
    (staged.control / "environment.json").write_text(json.dumps({"OPENAI_API_KEY": "private-model-token"}))
    stage_worker(staged, path)
    executable = Path(staged.manifest["executable"])
    executable.write_text(executable.read_text() + "\nassert os.environ['OPENAI_API_KEY'] == 'private-model-token'\n"
                          "assert open(os.environ['LSMS_S3_CREDS']).read() == 'approved-reader'\n")
    assert run_worker(staged) == 0
    result = json.loads((staged.control / "result.json").read_text())
    assert "approved-reader" not in json.dumps(result)
    assert "private-model-token" not in json.dumps(result)
    assert not (staged.control / "project-secret-reader").exists()
    assert reader.read_text() == "approved-reader"


def test_project_cannot_raise_requested_cpu_budget(source, monkeypatch, tmp_path):
    calls = scheduler(monkeypatch)
    path = profile_file(tmp_path, thread_limits={"LSMS_BUILD_WORKERS": 8})
    with pytest.raises(ValueError, match="CPU budget"):
        dispatch.submit(**source, project_profile=path)
    assert not calls


def test_setup_sigterm_stops_before_any_native_process(staged, tmp_path):
    ready = staged.control / "setup-ready"
    script = 'touch ' + str(ready) + '\nsleep 30\n'
    stage_worker(staged, profile_file(tmp_path, script=script, timeout_seconds=60))
    process = worker_subprocess(staged)
    try:
        wait_for(ready.exists, process)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=8)
        result = json.loads((staged.control / "result.json").read_text())
        assert process.returncode == 143, (stdout, stderr, result)
        assert result["state"] == "interrupted"
        assert result["project"]["state"] == "failed"
        assert result["exit_code"] is None
        assert not (staged.control / "native-version.log").exists()
    finally:
        if process.poll() is None:
            worker.stop_owned(process, 1)


def test_setup_failure_cleans_detached_owned_helper_before_native_start(staged, tmp_path):
    pid_file = staged.control / "mount-helper.pid"
    script = ('setsid sh -c \'echo $$ > ' + str(pid_file) + '; exec sleep 30\' >/dev/null 2>&1 &\nsleep 30\n')
    stage_worker(staged, profile_file(tmp_path, script=script, timeout_seconds=1))
    start = time.monotonic()
    assert fake_run(staged, {"SLURM_JOB_ID": "234"}, pid_file=pid_file) != 0
    assert time.monotonic() - start < 8
    pid = int(pid_file.read_text())
    try:
        state = Path('/proc/%s/stat' % pid).read_text().rsplit(')', 1)[1].split()[0]
    except FileNotFoundError:
        state = None
    assert state in (None, 'Z', 'X'), 'detached helper remains able to write'
    assert not (staged.control / "native-version.log").exists()


def test_synthetic_profile_refuses_ambient_provider_data_credentials(staged, tmp_path):
    path = profile_file(tmp_path, require_absent_environment=["AWS_ACCESS_KEY_ID"])
    (staged.control / "environment.json").write_text(json.dumps({"AWS_ACCESS_KEY_ID": "unapproved"}))
    stage_worker(staged, path)
    assert run_worker(staged) != 0
    result = json.loads((staged.control / "result.json").read_text())
    assert result["project"]["state"] == "failed"
    assert "unapproved" not in json.dumps(result)
    assert not (staged.control / "native-version.log").exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "a\\b"])
def test_local_directory_escape_refuses(tmp_path, name):
    with pytest.raises(ValueError, match="relative POSIX"):
        project.read_profile(profile_file(tmp_path, local_directories={"PROJECT_DATA": name}))
