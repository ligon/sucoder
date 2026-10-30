"""Portable worker tests: fake native agent, no real SQLite or model calls."""
import json
import errno
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace

import pytest

from sucoder import dispatch_worker as worker


class FakeJobOwnership:
    """Explicit fake allocation: only this test's children and registered PID.

    Never enumerate or signal other processes in pytest's real Slurm job.
    The registered PID models a Slurm member reparented outside the tree.
    """
    def __init__(self, job_id, pid_file=None):
        self.job_id = str(job_id)
        self.supervisor_pid = os.getpid()
        self.pid_file = pid_file
        self.runner = worker.load_runner(str(Path(worker.__file__).with_name("codex_state_runner.py")))
        self.known = {}

    def members(self):
        candidates = self.runner.descendants(self.supervisor_pid)
        if self.pid_file is not None and Path(self.pid_file).exists():
            candidates.append(int(Path(self.pid_file).read_text()))
        for pid in candidates:
            started = self.runner.read_starttime(pid)
            if started is not None:
                self.known.setdefault(pid, started)
        return [{"pid": pid, "starttime": started, "cgroups": [("fake", "", "/fake/job_" + self.job_id)]}
                for pid, started in self.known.items() if self.still_owned({"pid": pid, "starttime": started})]

    def still_owned(self, record):
        pid = record["pid"]
        if pid == self.supervisor_pid or self.runner.read_starttime(pid) != record["starttime"]:
            return False
        try:
            fields = Path("/proc/%s/stat" % pid).read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            return False
        return fields[0] not in ("Z", "X")

    def describe(self):
        return {"job_id": self.job_id, "prefixes": ["/fake/job_" + self.job_id], "fake": True}


def fake_run(staged, inherited, pid_file=None):
    previous_umask = os.umask(0o077)
    try:
        return worker.run(str(staged.path), inherited, str(staged.mounts),
                          ownership_factory=lambda job_id: FakeJobOwnership(job_id, pid_file))
    finally:
        # The standalone supervisor owns its process umask; this in-process
        # fake must not leak that state into unrelated filesystem tests.
        os.umask(previous_umask)


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], stderr=subprocess.DEVNULL).decode().strip()


@pytest.fixture
def staged(tmp_path):
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    disk = tmp_path / "disk"
    disk.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-q")
    git(source, "config", "user.name", "Test")
    git(source, "config", "user.email", "test@example.invalid")
    (source / "tracked.txt").write_text("source\n")
    git(source, "add", "tracked.txt")
    git(source, "commit", "-qm", "fixture")
    commit = git(source, "rev-parse", "HEAD")
    subprocess.check_call(["git", "clone", "--bare", "-q", str(source), str(control / "results.git")])
    for name in ("dispatch_worker.py", "codex_state_runner.py", "dispatch_protocol.py", "dispatch_artifacts.py"):
        shutil.copyfile(Path(worker.__file__).with_name(name), control / name)
    manifest = {"schema_version": 1, "dispatch_id": "a" * 32, "uid": os.getuid(),
                "directory": str(control), "repo": str(source), "commit": commit,
                "branch": "dispatch/" + "a" * 32, "agent": "claude", "model": None,
                "local_disk_root": str(disk), "scope": "b" * 32,
                "runtime_path": ["/usr/bin", "/bin"],
                "task_file": str(control / "task.txt"), "prepare_script": str(control / "prepare.sh"),
                "snapshot_script": str(control / "snapshot.sh"),
                "runner_path": str(control / "codex_state_runner.py"), "worker_path": str(control / "dispatch_worker.py"),
                "environment_file": str(control / "environment.json"), "source_agent_home": None,
                "source_claude_user_config": None, "resources": {},
                "checkpoint_minutes": 10, "checkpoint_timeout": 2}
    executable = control / "fake-claude"
    executable.write_text("#!" + sys.executable + "\n" + """
import json, os, sys, time
if '--version' in sys.argv:
    print('2.1.0 (Claude Code)')
    sys.exit(0)
assert '--print' in sys.argv
assert '--dangerously-skip-permissions' in sys.argv
assert 'TMUX' not in os.environ
assert 'CODEX_THREAD_ID' not in os.environ
assert 'SUCODER_CODEX_REMOTE_HELPER' not in os.environ
assert 'PRIVATE_PARENT_SECRET' not in os.environ
task = sys.stdin.read()
assert 'Initial commit:' in task
assert 'do the task' in task
assert os.path.isfile('MESSAGES_TO_AGENT.txt')
path = os.path.join(os.environ['CLAUDE_CONFIG_DIR'], 'session.jsonl')
with open(path, 'w') as stream:
    stream.write(json.dumps({'task': 'accepted'}) + '\\n')
    stream.flush()
    time.sleep(.45)
print('worker final')
""")
    executable.chmod(0o700)
    manifest["executable"] = str(executable)
    (control / "task.txt").write_text("do the task\n")
    (control / "environment.json").write_text("{}")
    # These helpers exercise the immutable checkout contract; the controller
    # tests cover production helper generation and append-only publication.
    (control / "prepare.sh").write_text("set -eu\n" + """
work="$SUCODER_LOCAL_ROOT/mirrors/dispatch_$SUCODER_DISPATCH_ID"
mkdir -p "$(dirname "$work")"
git clone -q "$RESULTS" "$work"
git -C "$work" checkout -q -b "dispatch/$SUCODER_DISPATCH_ID" "$PIN"
""".replace('"$RESULTS"', repr(str(control / "results.git"))).replace('"$PIN"', repr(commit)))
    (control / "snapshot.sh").write_text("printf '%s\\n' '{\"clean\":true}'\n")
    path = control / "manifest.json"
    path.write_text(json.dumps(manifest))
    for item in control.iterdir():
        if item.is_file():
            item.chmod(0o600 if item != executable else 0o700)
    (control / "digests.json").write_text(json.dumps({item.name: hashlib.sha256(item.read_bytes()).hexdigest()
                                                    for item in control.iterdir() if item.is_file() and item != executable}))
    (control / "digests.json").chmod(0o600)
    mounts = tmp_path / "mountinfo"
    mounts.write_text("1 0 8:1 / / rw - ext4 /dev/test rw\n")
    return SimpleNamespace(control=control, source=source, disk=disk, path=path,
                           manifest=manifest, mounts=mounts)


def save(staged):
    staged.path.write_text(json.dumps(staged.manifest))
    digests = json.loads((staged.control / "digests.json").read_text())
    digests[staged.path.name] = hashlib.sha256(staged.path.read_bytes()).hexdigest()
    (staged.control / "digests.json").write_text(json.dumps(digests))


def test_environment_rebinds_node_and_narrowly_inherits_credentials(staged):
    (staged.control / "environment.json").write_text(json.dumps({"CLAUDE_CODE_OAUTH_TOKEN": "private-token"}))
    inherited = {"SLURM_JOB_ID": "234", "SLURM_CPUS_ON_NODE": "2", "PATH": "/local/job123/bin:/usr/bin",
                 "SUCODER_LOCAL_ROOT": "/local/job123", "SUCODER_CODEX_REMOTE_HELPER": "coordinator",
                 "CODEX_SQLITE_HOME": "/shared/sqlite", "CODEX_THREAD_ID": "parent", "TMUX": "parent",
                 "OPENAI_API_KEY": "not-requested", "PRIVATE_PARENT_SECRET": "never-export",
                 "GIT_CONFIG_COUNT": "1", "PYTHONPATH": "/local/job123/library"}
    root, job_id, mount = worker.receiving_root(staged.manifest, inherited, str(staged.mounts))
    environment = worker.environment_for(staged.manifest, inherited, root)
    assert root == str(staged.disk / "job234")
    assert environment["SLURM_CPUS_ON_NODE"] == "2"
    assert environment["SUCODER_LOCAL_ROOT"] == root
    assert environment["CLAUDE_CODE_OAUTH_TOKEN"] == "private-token"
    assert environment["PATH"] == "/usr/bin:/bin"
    assert not (staged.control / "environment.json").exists()
    assert not set(("TMUX", "CODEX_THREAD_ID", "CODEX_SQLITE_HOME", "SUCODER_CODEX_REMOTE_HELPER",
                    "OPENAI_API_KEY", "PRIVATE_PARENT_SECRET", "GIT_CONFIG_COUNT", "PYTHONPATH")) & set(environment)


def test_thread_budgets_come_from_receiving_allocation(staged):
    staged.manifest["resources"] = {"cpus": 2}
    environment = worker.environment_for(staged.manifest, {"SLURM_CPUS_PER_TASK": "3", "OMP_NUM_THREADS": "56"},
                                         str(staged.disk / "job234"))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert environment[key] == "3"


@pytest.mark.parametrize("job_id", ["", "123_4", "../123", "0", "-1"])
def test_refuses_unverified_receiving_job(staged, job_id):
    with pytest.raises(worker.Refused, match="SLURM_JOB_ID"):
        worker.receiving_root(staged.manifest, {"SLURM_JOB_ID": job_id}, str(staged.mounts))


def test_longest_actual_mount_refuses_shared_filesystem(staged):
    staged.mounts.write_text("1 0 8:1 / / rw - ext4 /dev/test rw\n"
                            "2 1 0:1 / %s rw - lustre shared rw\n" % staged.disk)
    with pytest.raises(worker.Refused, match="lustre"):
        worker.receiving_root(staged.manifest, {"SLURM_JOB_ID": "234"}, str(staged.mounts))


def test_job_root_symlink_is_not_accepted(staged):
    (staged.disk / "job234").symlink_to(staged.source, target_is_directory=True)
    with pytest.raises(worker.Refused, match="symlink"):
        worker.receiving_root(staged.manifest, {"SLURM_JOB_ID": "234"}, str(staged.mounts))


@pytest.mark.parametrize("mutation", ["permissions", "uid", "branch", "task_escape"])
def test_refuses_untrusted_manifest(staged, mutation):
    if mutation == "permissions":
        staged.path.chmod(0o644)
    elif mutation == "uid":
        staged.manifest["uid"] += 1
    elif mutation == "branch":
        staged.manifest["branch"] = "main"
    else:
        staged.manifest["task_file"] = str(staged.source / "tracked.txt")
    if mutation != "permissions":
        save(staged)
    with pytest.raises(worker.Refused):
        worker.load_manifest(str(staged.path))


def test_claude_headless_runs_in_private_clone_and_preserves_text(staged):
    before = git(staged.source, "status", "--porcelain")
    code = fake_run(staged, {"SLURM_JOB_ID": "234", "PATH": os.environ["PATH"],
                                      "SUCODER_CODEX_REMOTE_HELPER": "never", "TMUX": "never",
                                      "PRIVATE_PARENT_SECRET": "never"})
    result = json.loads((staged.control / "result.json").read_text())
    diagnostics = tuple(result.get(key) for key in ("state", "reason", "placement", "preservation", "workspace_snapshot", "cleanup_error"))
    assert code == 0, diagnostics
    assert result["state"] == "succeeded"
    assert result["exit_code"] == 0
    assert result["supervisor_exit_code"] == 0
    assert result["final_head"] == staged.manifest["commit"]
    assert result["job_id"] == "234"
    assert result["placement"]["state"] in ("verified_observations", "verified_state_files")
    assert result["placement"]["sqlite"] == "not_observed"
    assert result["preservation"]["state"] == "text_preserved"
    assert result["coordinator_acceptance"] == "pending"
    assert result["task"]["status"] == "not_reported"
    assert result["publication"]["state"] == "complete"
    assert result["publication"]["evidence"]["state"] == "complete"
    assert result["versions"]["native"]["version"] == "2.1.0"
    assert (staged.control / "final.txt").read_text() == "worker final\n"
    assert (staged.control / "state/claude/.claude/session.jsonl").exists()
    assert git(staged.source, "status", "--porcelain") == before
    assert git(staged.source, "rev-parse", "HEAD") == staged.manifest["commit"]
    assert (staged.control / "result.json").stat().st_mode & 0o077 == 0


def test_process_exit_is_propagated_without_fabricating_placement(staged):
    executable = Path(staged.manifest["executable"])
    executable.write_text("#!" + sys.executable + "\nimport sys\nsys.exit(17)\n")
    assert fake_run(staged, {"SLURM_JOB_ID": "234", "PATH": os.environ["PATH"]}) == 17
    result = json.loads((staged.control / "result.json").read_text())
    assert result["exit_code"] == 17
    assert result["state"] == "failed"
    assert result["placement"]["state"] == "unverified"


def test_clean_exit_without_native_state_observations_is_unverified(staged):
    Path(staged.manifest["executable"]).write_text("#!" + sys.executable + "\nimport time\ntime.sleep(.2)\n")
    code = fake_run(staged, {"SLURM_JOB_ID": "234", "PATH": os.environ["PATH"]})
    result = json.loads((staged.control / "result.json").read_text())
    assert code == worker.EXIT_PRESERVATION, result
    assert result["exit_code"] == 0
    assert result["state"] == "failed"
    assert result["placement"]["state"] == "unverified"
    assert result["placement"]["observed"] == []


def test_workspace_pin_mismatch_refuses_before_agent(staged):
    staged.manifest["commit"] = "f" * 40
    save(staged)
    assert fake_run(staged, {"SLURM_JOB_ID": "234", "PATH": os.environ["PATH"]}) == worker.EXIT_REFUSED
    result = json.loads((staged.control / "result.json").read_text())
    assert result["state"] == "refused"
    assert "immutable source commit" in result["reason"]
    assert not (staged.control / "stdout.log").exists()


def test_codex_command_always_uses_managed_fresh_unattended_runner(staged):
    staged.manifest["agent"] = "codex"
    root = str(staged.disk / "job234")
    environment = worker.environment_for(staged.manifest, {"PATH": "/usr/bin"}, root)
    worker.seed_home(staged.manifest, environment)
    command = worker.command_for(staged.manifest, environment)
    assert command[:2] == [sys.executable, staged.manifest["runner_path"]]
    assert "--fresh" in command and "--unattended" in command
    assert "sqlite_home=\"__SUCODER_CODEX_STATE_DB__\"" in command
    assert "exec" in command and "--output-last-message" in command
    assert command[-1] == "-"
    assert not any("app-server" in arg for arg in command)
    assert environment["CODEX_HOME"] == str(staged.control / "codex-home")
    assert command[command.index("--codex-home") + 1] == environment["CODEX_HOME"]


def test_codex_canonical_home_and_recovery_identity_survive_new_job_roots(staged):
    staged.manifest["agent"] = "codex"
    source = staged.control / "native-seed"
    source.mkdir(mode=0o700)
    auth = source / "auth.json"
    auth.write_text('{"test":"private-auth"}')
    source_identity = "ffffffff-ffff-4fff-8fff-ffffffffffff"
    (source / "installation_id").write_text(source_identity)
    staged.manifest["source_agent_home"] = str(source)
    first = worker.environment_for(staged.manifest, {}, str(staged.disk / "job234"))
    worker.seed_home(staged.manifest, first)
    canonical = Path(first["CODEX_HOME"])
    assert canonical == staged.control / "codex-home"
    assert not str(canonical).startswith(str(staged.disk))
    assert (canonical / "auth.json").read_bytes() == auth.read_bytes()
    assert (canonical / "auth.json").stat().st_mode & 0o077 == 0
    installation = (canonical / "installation_id").read_text()
    assert str(uuid.UUID(installation)) == installation
    assert installation != source_identity
    assert (canonical / "installation_id").stat().st_mode & 0o077 == 0
    assert (source / "installation_id").read_text() == source_identity
    runner = worker.load_runner(staged.manifest["runner_path"])
    assert all((canonical / name).is_dir() for name in runner.BIND_DIRS)
    second = dict(first, HOME=str(staged.disk / "job999/home"), SUCODER_LOCAL_ROOT=str(staged.disk / "job999"))
    first_command = worker.command_for(staged.manifest, first)
    second_command = worker.command_for(staged.manifest, second)
    assert first_command[first_command.index("--codex-home") + 1] == second_command[second_command.index("--codex-home") + 1]
    assert first_command[first_command.index("--durable-root") + 1] == second_command[second_command.index("--durable-root") + 1]
    assert auth.read_text() == '{"test":"private-auth"}'


def test_source_auth_config_is_copied_privately_and_unchanged(staged):
    source = staged.control / "seed"
    source.mkdir(mode=0o700)
    original = source / ".credentials.json"
    original.write_text('{"credential":"private"}')
    staged.manifest["source_agent_home"] = str(source)
    root = str(staged.disk / "job234")
    environment = worker.environment_for(staged.manifest, {}, root)
    config = worker.seed_home(staged.manifest, environment)
    copied = Path(config) / ".credentials.json"
    assert copied.read_bytes() == original.read_bytes()
    assert copied.stat().st_mode & 0o077 == 0
    copied.write_text("local changes")
    assert original.read_text() == '{"credential":"private"}'


def test_native_claude_global_configuration_follows_config_dir(staged):
    original = staged.control / "source-user-config.json"
    original.write_text('{"account":"private"}')
    staged.manifest["source_claude_user_config"] = str(original)
    environment = worker.environment_for(staged.manifest, {}, str(staged.disk / "job234"))
    config = worker.seed_home(staged.manifest, environment)
    assert (Path(config) / ".claude.json").read_bytes() == original.read_bytes()
    assert not (Path(environment["HOME"]) / ".claude.json").exists()


def test_claude_preservation_excludes_databases_sidecars_and_symlinks(staged):
    home = staged.control / "local-home"
    home.mkdir()
    for name in ("memory.sqlite", "memory.sqlite-wal", "memory.sqlite-shm", "memory.db-journal", "transcript.jsonl"):
        (home / name).write_text("test text, never a SQLite database")
    (home / "link.json").symlink_to(staged.source / "tracked.txt")
    report = worker.preserve_claude(staged.manifest, str(home))
    assert report["files"] == ["transcript.jsonl"]
    assert len(report["excluded"]) == 5
    assert not (staged.control / "state/claude/memory.sqlite").exists()


def test_claude_preservation_does_not_archive_auth_or_configuration_seeds(staged):
    home = staged.control / "local-home"
    (home / ".claude").mkdir(parents=True)
    for relative in (".claude/.credentials.json", ".claude/settings.json", ".claude.json"):
        (home / relative).write_text('{"secret":"never archive"}')
    (home / ".claude/transcript.jsonl").write_text("task transcript\n")
    report = worker.preserve_claude(staged.manifest, str(home))
    assert report["files"] == [".claude/transcript.jsonl"]
    assert len(report["excluded"]) == 3


def test_unexpected_claude_sqlite_state_refuses_complete_preservation(staged):
    home = staged.control / "local-home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude/native-state.sqlite").write_text("fake filename, no SQLite database")
    report = worker.preserve_claude(staged.manifest, str(home))
    assert report["state"] == "unsupported"
    assert report["unsupported_stores"] == [".claude/native-state.sqlite"]


def test_unexpected_claude_xdg_sqlite_is_also_unsupported(staged):
    home = staged.control / "local-home"
    (home / "state").mkdir(parents=True)
    (home / "state/native-state.sqlite").write_text("fake filename, no SQLite database")
    report = worker.preserve_claude(staged.manifest, str(home))
    assert report["state"] == "unsupported"
    assert report["unsupported_stores"] == ["state/native-state.sqlite"]


def test_cannot_inspect_owned_live_process_is_a_refusal(staged, monkeypatch):
    proc = staged.control / "proc"
    (proc / "123").mkdir(parents=True)
    runner = SimpleNamespace(descendants=lambda pid: [], inspect_file_mounts=lambda pids, **kwargs: [])
    original_listdir = worker.os.listdir
    def denied(path):
        if str(path) == str(proc / "123/fd"):
            raise PermissionError(errno.EACCES, "permission denied")
        return original_listdir(path)
    monkeypatch.setattr(worker.os, "listdir", denied)
    with pytest.raises(worker.Refused, match="pid=123 errno=%s" % errno.EACCES):
        worker.inspect_placement(SimpleNamespace(pid=123), runner, str(staged.disk), str(staged.disk), str(proc))


def test_misplaced_descriptor_uses_managed_runner_evidence(staged):
    runner = SimpleNamespace(descendants=lambda pid: [], inspect_file_mounts=lambda pids, **kwargs: [
        {"pid": os.getpid(), "path": "/shared/state.sqlite-wal", "filesystem": "lustre"}])
    report = worker.inspect_placement(SimpleNamespace(pid=os.getpid()), runner, str(staged.disk), str(staged.disk))
    assert report["misplaced"] == ["/shared/state.sqlite-wal"]
    assert report["databases"] == []


def test_injected_input_and_local_tools_are_not_native_state_evidence(staged):
    home = str(staged.disk / "home")
    runner = SimpleNamespace(descendants=lambda pid: [], inspect_file_mounts=lambda pids, **kwargs: [
        {"pid": os.getpid(), "path": home + "/tmp/dispatch-input.txt", "filesystem": "ext4"},
        {"pid": os.getpid(), "path": home + "/cache/tool.json", "filesystem": "ext4"}])
    report = worker.inspect_placement(SimpleNamespace(pid=os.getpid()), runner, str(staged.disk), home)
    assert report["observed"] == []


def test_missing_codex_generation_is_independent_of_process_exit(staged):
    assert worker.codex_preservation(staged.manifest, str(staged.disk))["state"] == "unverified"


def test_codex_crash_generation_is_preserved_without_calling_it_sealed(staged):
    scope = staged.disk / "state/codex" / staged.manifest["scope"] / ("c" * 32)
    scope.mkdir(parents=True)
    generation = staged.control / "state/codex/generation"
    generation.mkdir(parents=True)
    content = json.dumps({"class": "crash", "scope": staged.manifest["scope"]})
    (generation / "manifest.json").write_text(content)
    (scope / "status.json").write_text(json.dumps({"state": "exited", "last_generation": {
        "path": str(generation), "class": "crash", "manifest_sha256": hashlib.sha256(content.encode()).hexdigest()}}))
    report = worker.codex_preservation(staged.manifest, str(staged.disk))
    assert report["state"] == "crash"
    assert report["class"] == "crash"


def test_tampered_staged_code_is_refused_before_execution(staged):
    (staged.control / "prepare.sh").write_text("exit 99\n")
    with pytest.raises(worker.Refused, match="integrity record"):
        worker.load_manifest(str(staged.path))


def test_missing_staged_protocol_is_refused(staged):
    (staged.control / "dispatch_protocol.py").unlink()
    with pytest.raises((worker.Refused, OSError)):
        worker.load_manifest(str(staged.path))


def test_helper_timeout_keeps_private_diagnostics(staged):
    helper = staged.control / "slow-helper.sh"
    helper.write_text("printf 'private-helper-diagnostic' >&2\nsleep 20\n")
    start = time.monotonic()
    with pytest.raises(worker.Refused, match="timed out") as error:
        worker.run_script(str(helper), {"PATH": "/usr/bin:/bin"}, .2)
    assert time.monotonic() - start < 5
    assert "private-helper-diagnostic" not in str(error.value)
    logs = list(staged.control.glob("slow-helper.sh.*.stderr.log"))
    assert len(logs) == 1
    assert logs[0].read_text() == "private-helper-diagnostic"
    assert logs[0].stat().st_mode & 0o077 == 0


def test_helper_escaped_pipe_writer_cannot_block_timeout(staged):
    pid_path = staged.control / "escaped.pid"
    helper = staged.control / "escaped-helper.sh"
    code = ("import os,time; pid=os.fork(); "
            "os.setsid() if pid==0 else None; "
            "open(%r,'w').write(str(os.getpid())) if pid==0 else None; time.sleep(20)" % str(pid_path))
    import shlex
    helper.write_text(shlex.quote(sys.executable) + " -c " + shlex.quote(code) + "\n")
    start = time.monotonic()
    try:
        with pytest.raises(worker.Refused, match="timed out"):
            worker.run_script(str(helper), {"PATH": "/usr/bin:/bin"}, .5)
        assert time.monotonic() - start < 5
    finally:
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def worker_subprocess(staged):
    command = [sys.executable, "-c",
               "import sys,importlib.util; from sucoder.dispatch_worker import run; "
               "s=importlib.util.spec_from_file_location('fake_dispatch_test',sys.argv[3]); "
               "m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
               "sys.exit(run(sys.argv[1],mountinfo_path=sys.argv[2],ownership_factory=m.FakeJobOwnership))",
               str(staged.path), str(staged.mounts), __file__]
    environment = dict(os.environ, SLURM_JOB_ID="234")
    return subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, start_new_session=True)


def wait_for(predicate, process, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            pytest.fail("fake worker exited before expected checkpoint: %s %r %r" % (process.returncode, stdout, stderr))
        time.sleep(.02)
    pytest.fail("fake worker did not reach expected checkpoint")


def test_sigterm_running_worker_publishes_interrupted_result(staged):
    executable = Path(staged.manifest["executable"])
    executable.write_text(executable.read_text().replace("time.sleep(.45)", "time.sleep(20)"))
    process = worker_subprocess(staged)
    status_path = staged.control / "status.json"
    try:
        wait_for(lambda: status_path.exists() and json.loads(status_path.read_text()).get("placement", {}).get("state") == "verified_observations", process)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=8)
        result = json.loads((staged.control / "result.json").read_text())
        assert process.returncode == 143, (result, stdout, stderr)
        assert result["state"] == "interrupted"
        assert result["supervisor_exit_code"] == 143
        assert result["exit_code"] == 143
        assert result["workspace_snapshot"] == {"clean": True}
        assert result["final_head"] == staged.manifest["commit"]
    finally:
        if process.poll() is None:
            worker.stop_owned(process, 1)


def test_sigterm_during_prepare_publishes_bounded_interruption(staged):
    ready = staged.control / "prepare-ready"
    (staged.control / "prepare.sh").write_text("touch %s\nsleep 20\n" % ready)
    digests = json.loads((staged.control / "digests.json").read_text())
    digests["prepare.sh"] = hashlib.sha256((staged.control / "prepare.sh").read_bytes()).hexdigest()
    (staged.control / "digests.json").write_text(json.dumps(digests))
    process = worker_subprocess(staged)
    try:
        wait_for(ready.exists, process)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=8)
        result = json.loads((staged.control / "result.json").read_text())
        assert process.returncode == 143, (result, stdout, stderr)
        assert result["state"] == "interrupted"
        assert result["exit_code"] is None
        assert "helper interrupted" in result["reason"]
    finally:
        if process.poll() is None:
            worker.stop_owned(process, 1)


def test_secondary_cleanup_error_does_not_hide_final_record(staged, monkeypatch):
    Path(staged.manifest["executable"]).write_text("#!" + sys.executable + "\nimport sys\nsys.exit(17)\n")
    def unavailable(child, timeout):
        raise worker.Refused("cleanup unavailable")
    monkeypatch.setattr(worker, "stop_owned", unavailable)
    code = fake_run(staged, {"SLURM_JOB_ID": "234", "PATH": os.environ["PATH"]})
    result = json.loads((staged.control / "result.json").read_text())
    assert code == 17, result
    assert result["exit_code"] == 17
    assert result["state"] == "failed"
    assert "cleanup unavailable" in result["cleanup_error"]


def proc_member(root, pid, cgroup, *, state="S", started=42, uid=None):
    directory = root / str(pid)
    directory.mkdir()
    fields = [state] + ["0"] * 19
    fields[19] = str(started)
    (directory / "stat").write_text("%s (fake agent) %s\n" % (pid, " ".join(fields)))
    (directory / "cgroup").write_text(cgroup)
    uid = os.getuid() if uid is None else uid
    (directory / "status").write_text("Uid:\t%s\t%s\t%s\t%s\n" % (uid, uid, uid, uid))
    return directory


@pytest.mark.parametrize("hierarchy", ["0::", "7:cpu,memory:"])
def test_cgroup_fence_includes_reparented_sessions_and_other_steps(tmp_path, hierarchy):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    prefix = "/slurm/uid_%s/job_234" % os.getuid()
    (proc / "self/cgroup").write_text(hierarchy + prefix + "/step_batch\n")
    proc_member(proc, 11, hierarchy + prefix + "/step_batch\n")
    proc_member(proc, 12, hierarchy + prefix + "/step_9/new_session\n")
    proc_member(proc, 13, hierarchy + prefix.replace("job_234", "job_999") + "/step_batch\n")
    proc_member(proc, 14, hierarchy + prefix + "/step_9\n", state="Z")
    owner = worker.JobOwnership("234", str(proc), supervisor_pid=999)
    members = owner.members()
    assert [record["pid"] for record in sorted(members, key=lambda item: item["pid"])] == [11, 12]
    assert owner.describe()["prefixes"] == [prefix]
    assert owner.still_owned(members[0])
    directory = proc / str(members[0]["pid"])
    (directory / "cgroup").write_text(hierarchy + prefix.replace("job_234", "job_999") + "/step_9\n")
    assert not owner.still_owned(members[0])


def test_cgroup_fence_refuses_coordinator_job_id(tmp_path):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/slurm/job_999/step_batch\n")
    with pytest.raises(worker.Refused, match="actual cgroup"):
        worker.JobOwnership("234", str(proc))


def test_cgroup_unknown_live_membership_is_a_refusal(tmp_path):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/slurm/job_234/step_batch\n")
    directory = proc_member(proc, 11, "0::/slurm/job_234/step_8\n")
    (directory / "cgroup").unlink()
    owner = worker.JobOwnership("234", str(proc), supervisor_pid=999)
    with pytest.raises(worker.Refused, match="membership"):
        owner.members()


def test_stop_allocation_rechecks_incarnation_and_membership_before_signals(monkeypatch):
    record = {"pid": 123, "starttime": 42}
    class Moved:
        calls = 0
        def members(self):
            self.calls += 1
            return [record] if self.calls == 1 else []
        def still_owned(self, candidate):
            return False
    signals = []
    monkeypatch.setattr(worker.os, "kill", lambda pid, signum: signals.append((pid, signum)))
    worker.stop_allocation(Moved(), .1)
    assert signals == []


def test_daemon_detached_before_leader_exit_is_stopped_and_preserved(staged):
    pid_file = staged.control / "native-daemon.pid"
    executable = Path(staged.manifest["executable"])
    executable.write_text("#!" + sys.executable + "\n" + """
import os,sys,time
sys.stdin.read()
pid=os.fork()
if pid == 0:
    os.setsid()
    path=os.path.join(os.environ['CLAUDE_CONFIG_DIR'],'daemon.json')
    with open(path,'w') as stream:
        stream.write('{"state":"owned daemon"}')
        stream.flush()
        with open(%r,'w') as marker:
            marker.write(str(os.getpid()))
        time.sleep(20)
    os._exit(0)
while not os.path.exists(%r):
    time.sleep(.01)
time.sleep(.3)
print('leader complete')
""" % (str(pid_file), str(pid_file)))
    try:
        code = fake_run(staged, {"SLURM_JOB_ID": "234", "PATH": os.environ["PATH"]}, pid_file=pid_file)
        result = json.loads((staged.control / "result.json").read_text())
        assert code == 0, tuple(result.get(key) for key in ("state", "reason", "placement", "preservation", "cleanup_error"))
        daemon = int(pid_file.read_text())
        if Path("/proc/%s/stat" % daemon).exists():
            assert Path("/proc/%s/stat" % daemon).read_text().rsplit(")", 1)[1].split()[0] == "Z"
        assert (staged.control / "state/claude/.claude/daemon.json").exists()
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_durable_copy_exception_is_only_for_the_managed_supervisor(staged):
    staged.manifest["agent"] = "codex"
    pid = os.getpid()
    path = str(staged.control / "state/codex" / staged.manifest["scope"] / "runs" / ("c" * 32) /
               (".incoming-" + "d" * 32) / "db/state_5.sqlite")
    runner = SimpleNamespace(descendants=lambda current: [], REQUIRED_STORES=("state_5.sqlite",), OPTIONAL_STORES=(),
                             inspect_file_mounts=lambda pids, **kwargs: [{"pid": pid, "path": path, "filesystem": "lustre"}])
    report = worker.inspect_placement(SimpleNamespace(pid=pid), runner, str(staged.disk), str(staged.disk), manifest=staged.manifest)
    assert report["misplaced"] == []
    assert report["durable_checkpoint_copies"] == [path]
    runner.inspect_file_mounts = lambda pids, **kwargs: [{"pid": pid + 1, "path": path, "filesystem": "lustre"}]
    report = worker.inspect_placement(SimpleNamespace(pid=pid), runner, str(staged.disk), str(staged.disk), manifest=staged.manifest)
    assert report["misplaced"] == [path]


@pytest.mark.parametrize('budget', [None, 120])
def test_workspace_budget_reaches_periodic_and_final_publication(staged, monkeypatch, budget):
    staged.manifest['snapshot_minutes'] = .001
    if budget is not None:
        staged.manifest['publication_timeout'] = budget
    save(staged)
    original = worker.run_script
    calls = []
    def observed(path, environment, timeout, *args, **kwargs):
        if path == staged.manifest['snapshot_script']:
            calls.append((timeout, bool(args)))
        return original(path, environment, timeout, *args, **kwargs)
    monkeypatch.setattr(worker, 'run_script', observed)
    assert fake_run(staged, {'SLURM_JOB_ID': '234', 'PATH': os.environ['PATH']}) == 0
    assert any(periodic for _, periodic in calls) and any(not periodic for _, periodic in calls)
    assert all(value == (budget if budget is not None else 2) for value, _ in calls)
    assert staged.manifest['checkpoint_timeout'] == 2


@pytest.mark.parametrize('budget', [0, -1, True, float('nan'), float('inf'), 301])
def test_receiving_manifest_refuses_invalid_publication_budget(staged, budget):
    staged.manifest['publication_timeout'] = budget
    save(staged)
    with pytest.raises(worker.Refused, match='publication timeout'):
        worker.load_manifest(str(staged.control / 'manifest.json'))


def test_heartbeat_tracks_phases_without_fsync_every_descriptor_poll(staged, monkeypatch):
    writes = []
    original = worker.atomic_json
    def observed(path, value):
        if str(path).endswith("/status.json"):
            writes.append((value["phase"], value["heartbeat_at"], value["observed_at"], value["state"],
                           value["placement"]["state"], value["preservation"]["state"], value["publication"]["state"]))
        return original(path, value)
    monkeypatch.setattr(worker, "atomic_json", observed)
    code = fake_run(staged, {"SLURM_JOB_ID": "234", "PATH": os.environ["PATH"]})
    result = json.loads((staged.control / "result.json").read_text())
    assert code == 0, result
    phases = [entry[0] for entry in writes]
    assert phases[0] == "cloning" and phases[-1] == "finished"
    assert all(phase in phases for phase in ("agent_setup", "agent_execution", "final_publication", "preservation"))
    for previous, current in zip(writes, writes[1:]):
        if (previous[0], previous[3:]) == (current[0], current[3:]):
            assert current[1] - previous[1] >= 1.8, writes
    assert all(entry[1] == entry[2] and entry[1] > 0 for entry in writes)


def test_dirty_result_publication_precedes_conversation_preservation_failure(staged, monkeypatch):
    marker = staged.control / "workspace-published"
    script = staged.control / "snapshot.sh"
    script.write_text("touch %s\nprintf '%s\\n' '{\"ref\":\"refs/test/dirty\",\"commit\":\"abc\"}'\n" % (marker, "%s"))
    digests = json.loads((staged.control / "digests.json").read_text())
    digests["snapshot.sh"] = hashlib.sha256(script.read_bytes()).hexdigest()
    (staged.control / "digests.json").write_text(json.dumps(digests))
    def failed_preservation(manifest, home, heartbeat=None):
        assert marker.exists(), "conversation preservation ran before final workspace publication"
        raise RuntimeError("injected conversation copy failure")
    monkeypatch.setattr(worker, "preserve_claude", failed_preservation)
    code = fake_run(staged, {"SLURM_JOB_ID": "234", "PATH": os.environ["PATH"]})
    result = json.loads((staged.control / "result.json").read_text())
    assert code == worker.EXIT_PRESERVATION, result
    assert result["exit_code"] == 0
    assert result["publication"]["state"] == "complete"
    assert result["publication"]["snapshot_ref"] == "refs/test/dirty"
    assert result["publication"]["published_at"] > 0
    assert "conversation copy failure" in result["reason"]


def test_blocked_task_outcome_is_separate_from_infrastructure_success(staged):
    executable = Path(staged.manifest["executable"])
    text = executable.read_text().replace("print('worker final')", """
with open(os.environ['SUCODER_DISPATCH_TASK_RESULT'],'w') as stream:
    json.dump({'status':'blocked','summary':'scientific input missing','unexecuted':['analysis']},stream)
with open(os.path.join(os.environ['SUCODER_DISPATCH_EVIDENCE_DIR'],'check.txt'),'w') as stream:
    stream.write('reviewable evidence')
print('worker final')
""")
    executable.write_text(text)
    staged.manifest["required_evidence"] = ["check.txt"]
    save(staged)
    code = fake_run(staged, {"SLURM_JOB_ID": "234", "PATH": os.environ["PATH"]})
    result = json.loads((staged.control / "result.json").read_text())
    assert code == 0, result
    assert result["state"] == "succeeded"
    assert result["task"]["status"] == "blocked"
    assert result["task"]["unexecuted"] == ["analysis"]
    assert result["coordinator_acceptance"] == "pending"
    assert result["publication"]["evidence"]["state"] == "complete"
    assert result["publication"]["evidence"]["count"] == 1


def test_missing_required_evidence_is_an_infrastructure_failure(staged):
    staged.manifest["required_evidence"] = ["missing-check.txt"]
    save(staged)
    code = fake_run(staged, {"SLURM_JOB_ID": "234", "PATH": os.environ["PATH"]})
    result = json.loads((staged.control / "result.json").read_text())
    assert code == worker.EXIT_PRESERVATION, result
    assert result["exit_code"] == 0
    assert result["state"] == "failed"
    assert result["publication"]["state"] == "incomplete"
    assert result["publication"]["incomplete"] is True
    assert result["publication"]["incomplete_components"] == ["evidence"]
    assert result["publication"]["workspace"]["state"] == "complete"
    assert result["publication"]["workspace"]["incomplete"] is False
    assert result["publication"]["workspace"]["head"] == staged.manifest["commit"]
    assert result["publication"]["workspace"]["published_at"] > 0
    assert result["publication"]["evidence"]["state"] == "incomplete"
    assert "missing-check.txt" in result["publication"]["evidence"]["missing"]


def test_shared_steering_channels_remain_visible_through_local_links(staged):
    work = staged.disk / "work"
    work.mkdir()
    prompt = worker.prompt_for(staged.manifest, str(work)).decode()
    messages = staged.control / "MESSAGES_TO_AGENT.txt"
    questions = staged.control / "QUESTIONS_FROM_AGENT.txt"
    assert str(messages) in prompt and str(questions) in prompt
    assert messages.stat().st_mode & 0o077 == 0
    assert questions.stat().st_mode & 0o077 == 0
    assert (work / messages.name).is_symlink()
    messages.write_text("new coordinator steering\n")
    assert (work / messages.name).read_text() == "new coordinator steering\n"
    (work / questions.name).write_text("worker question\n")
    assert questions.read_text() == "worker question\n"
    # Mutable channels are intentionally outside the immutable staging digests.
    assert worker.load_manifest(str(staged.path))["dispatch_id"] == staged.manifest["dispatch_id"]


def test_tracked_source_channel_is_preserved_and_absolute_channel_is_primary(staged):
    work = staged.disk / "work"
    work.mkdir()
    git(work, "init", "-q")
    git(work, "config", "user.name", "Test")
    git(work, "config", "user.email", "test@example.invalid")
    tracked = work / "MESSAGES_TO_AGENT.txt"
    tracked.write_text("tracked project guidance\n")
    git(work, "add", tracked.name)
    git(work, "commit", "-qm", "tracked fixture channel")
    prompt = worker.prompt_for(staged.manifest, str(work)).decode()
    shared = staged.control / tracked.name
    shared.write_text("live coordinator update\n")
    assert not tracked.is_symlink()
    assert tracked.read_text() == "tracked project guidance\n"
    assert git(work, "diff", "--", tracked.name) == ""
    assert str(shared) in prompt
    assert "Use these absolute channels" in prompt


def test_nondumpable_same_uid_process_is_owned_despite_root_proc_directory(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/slurm/job_234/step_batch\n")
    directory = proc_member(proc, 11, "0::/slurm/job_234/step_9/new_session\n", uid=1234)
    proc_member(proc, 12, "0::/slurm/job_234/step_batch\n", uid=0)
    original_stat = worker.os.stat
    def root_owned_proc(path, *args, **kwargs):
        if str(path) == str(directory):
            return SimpleNamespace(st_uid=0)
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(worker.os, "stat", root_owned_proc)
    owner = worker.JobOwnership("234", str(proc), uid=1234, supervisor_pid=999)
    members = owner.members()
    assert [record["pid"] for record in members] == [11]
    assert owner.still_owned(members[0])


@pytest.mark.parametrize("status", [None, "Name: hidden\n", "Uid:\tinvalid\n"])
def test_unknown_live_process_uid_refuses_instead_of_skipping(tmp_path, status):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/slurm/job_234/step_batch\n")
    directory = proc_member(proc, 11, "0::/slurm/job_234/step_9\n")
    if status is None:
        (directory / "status").unlink()
    else:
        (directory / "status").write_text(status)
    owner = worker.JobOwnership("234", str(proc), supervisor_pid=999)
    with pytest.raises(worker.Refused, match="membership|uid credentials"):
        owner.members()


@pytest.mark.parametrize("disappearance", [errno.ENOENT, errno.ESRCH])
def test_fd_directory_teardown_is_not_live_inspection_denial(staged, monkeypatch, disappearance):
    proc = staged.control / "proc"
    (proc / "123").mkdir(parents=True)
    runner = SimpleNamespace(descendants=lambda pid: [], inspect_file_mounts=lambda pids, **kwargs: [])
    original_listdir = worker.os.listdir
    def disappeared(path):
        if str(path) == str(proc / "123/fd"):
            raise OSError(disappearance, "process teardown")
        return original_listdir(path)
    monkeypatch.setattr(worker.os, "listdir", disappeared)
    observed = worker.inspect_placement(SimpleNamespace(pid=123), runner, str(staged.disk), str(staged.disk), str(proc))
    assert observed["inspected_pids"] == []
    assert observed["observed"] == observed["misplaced"] == []


def test_claude_preservation_skips_install_inventory_but_keeps_project_memories(staged):
    home = staged.control / "local-home"
    paths = (".claude/plugins/marketplaces/example/README.md", ".claude/skills/example/SKILL.md",
             ".claude/projects/project/.git/config.json",
             ".claude/projects/project/memory/skills/note.md", ".claude/projects/project/session.jsonl")
    for relative in paths:
        path = home / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private conversation or install inventory\n")
    report = worker.preserve_claude(staged.manifest, str(home))
    assert set(report["files"]) == {".claude/projects/project/memory/skills/note.md", ".claude/projects/project/session.jsonl"}
    assert set(report["excluded_subtrees"]) == {".claude/plugins", ".claude/skills", ".claude/projects/project/.git"}
