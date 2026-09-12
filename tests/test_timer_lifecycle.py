"""Run the generated watchdog and its starter under real bash/flock."""
import os
import shutil
import signal
import shlex
import subprocess
import time
from types import SimpleNamespace

import pytest

from sucoder.slurm_timer import build_timer_script, timer_identity


@pytest.fixture
def timers(tmp_path, monkeypatch):
    from sucoder import slurm_timer
    monkeypatch.setattr(slurm_timer, "TIMER_LIFECYCLE_SH",
                        slurm_timer.TIMER_LIFECYCLE_SH.replace(
                            'RUNTIME_DIR="/tmp/sucoder-$UID"',
                            "RUNTIME_DIR=" + shlex.quote(str(tmp_path / "runtime"))))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("bash", "flock", "hostname", "mkdir", "chmod", "cat", "rm",
                 "nohup", "sleep", "tail", "seq", "touch", "git", "mktemp", "date", "stat"):
        executable = shutil.which(name)
        if not executable:
            pytest.skip(f"requires {name}")
        (bin_dir / name).symlink_to(executable)
    for name, body in (("tmux", "exit 0"), ("squeue", "echo 02:00:00")):
        path = bin_dir / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o700)
    env = dict(os.environ, HOME=str(tmp_path), PATH=str(bin_dir))
    clients = []

    def start(target="savio", job=12, confined=False, snapshot_dir=None):
        scope = timer_identity("example", target)
        script = tmp_path / f"timer-{scope}-{job}.sh"
        if not script.exists():
            script.write_text(build_timer_script(
                mirror_token="example", tmux_session="sucoder-example",
                job_id=None if confined else job,
                timer_scope=scope, tmux_socket="test" if confined else None,
                snapshot_dir=str(snapshot_dir) if snapshot_dir else None,
            ))
        child_env = dict(env, SLURM_JOB_ID=str(job))
        child = subprocess.Popen(["bash", str(script), "--ensure"], env=child_env,
                                 start_new_session=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
        clients.append(child)
        return child

    yield SimpleNamespace(start=start, root=tmp_path, bin=bin_dir, env=env)
    # Only groups created by these tests; never pkill by name or a saved PID.
    for child in clients:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)


def finish(child):
    out, err = child.communicate(timeout=15)
    assert child.returncode == 0, err
    return out


@pytest.mark.parametrize("confined", [False, True])
def test_timer_survives_starter_and_reuses_owner(timers, confined):
    assert "STARTED" in finish(timers.start(confined=confined))
    owner = next(timers.root.glob("runtime/timers/*/*/owner"))
    identity = owner.read_text()
    assert "REUSED" in finish(timers.start(confined=confined))
    assert owner.read_text() == identity
    os.kill(int(identity.split()[0]), 0)


def test_concurrent_starters_create_one_timer(timers):
    a, b = timers.start(), timers.start()
    outputs = [finish(a), finish(b)]
    assert sum("STARTED" in out for out in outputs) == 1
    assert sum("REUSED" in out for out in outputs) == 1
    assert len(list(timers.root.glob("runtime/timers/*/*/owner"))) == 1


def test_target_and_allocation_have_distinct_timers(timers):
    for target, job in [("savio", 12), ("savio-htc", 12), ("savio", 13)]:
        assert "STARTED" in finish(timers.start(target, job))
    owners = list(timers.root.glob("runtime/timers/*/*/owner"))
    assert len({p.read_text() for p in owners}) == 3
    assert timer_identity("a/b", None) != timer_identity("a_b", None)


def test_dead_owner_metadata_does_not_signal_unrelated_process(timers):
    node = subprocess.check_output(["hostname"], text=True).strip()
    (timers.root / "runtime").mkdir(mode=0o700)
    state = timers.root / "runtime/timers" / timer_identity("example", "savio") / f"{node}-12"
    state.mkdir(parents=True)
    # The current test runner PID is intentionally paired with a wrong birth
    # time. No lock exists, so a fresh timer must replace this stale record.
    (state / "owner").write_text(f"{os.getpid()} 0\n")
    (state / "status").write_text("monitoring\n")
    assert "STARTED" in finish(timers.start())
    assert (state / "owner").read_text().split()[0] != str(os.getpid())


def test_missing_executable_reports_failure(timers):
    (timers.bin / "squeue").unlink()
    child = timers.start()
    out, err = child.communicate(timeout=15)
    assert child.returncode != 0
    assert "squeue unavailable" in err
    assert "STARTED" not in out


def test_watchdog_warns_and_snapshots_without_changing_index(timers):
    origin, work = timers.root / "origin", timers.root / "work"

    def git(*args):
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()

    git("init", "--bare", str(origin))
    git("init", "-b", "main", str(work))
    git("-C", str(work), "config", "user.name", "Test")
    git("-C", str(work), "config", "user.email", "test@example.com")
    (work / "tracked").write_text("base")
    git("-C", str(work), "add", ".")
    git("-C", str(work), "commit", "-m", "base")
    git("-C", str(work), "remote", "add", "origin", str(origin))
    git("-C", str(work), "push", "origin", "main")
    (work / "tracked").write_text("staged")
    git("-C", str(work), "add", ".")
    (work / "tracked").write_text("working")
    (work / "untracked").write_text("new")
    before = git("-C", str(work), "write-tree")
    (timers.bin / "squeue").write_text("#!/bin/sh\necho 00:04:00\n")
    assert "STARTED" in finish(timers.start(snapshot_dir=work))
    deadline = time.monotonic() + 5
    ref = "refs/sucoder/wip/example"
    while time.monotonic() < deadline:
        check = subprocess.run(["git", "-C", str(origin), "rev-parse", "--verify", ref],
                               capture_output=True)
        if check.returncode == 0:
            break
        time.sleep(0.02)
    assert check.returncode == 0
    assert git("-C", str(origin), "show", f"{ref}:tracked") == "working"
    assert git("-C", str(origin), "show", f"{ref}:untracked") == "new"
    assert git("-C", str(work), "write-tree") == before
    warning = next(timers.root.glob("runtime/timers/*/*/slurm-deadline-*.warn"))
    assert "Commit and save NOW" in warning.read_text()


def test_timer_ssh_timeout_is_advisory(monkeypatch, caplog):
    import logging
    from sucoder import cli

    def timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 30)

    monkeypatch.setattr(subprocess, "run", timeout)
    session = SimpleNamespace(mirror_name="test", slurm_job_id=1, compute_node="node")
    control = SimpleNamespace(ssh_options=lambda **kw: [])
    cli._start_slurm_timer(session, control, control, logging.getLogger("timer-test"))
    assert "exit -1" in caplog.text
    assert "snapshots are unavailable" in caplog.text


def test_locks_work_when_home_does_not_support_flock(timers):
    """Model the cluster: flock on HOME fails, node-local locks work."""
    real_flock = shutil.which("flock")
    (timers.bin / "flock").unlink()
    (timers.bin / "flock").write_text(
        '#!/bin/bash\n'
        'for fd in 8 9; do\n'
        '  path=$(/usr/bin/readlink /proc/$$/fd/$fd 2>/dev/null)\n'
        '  case "$path" in "$HOME"/.cache/*)\n'
        '    echo "No locks available" >&2; exit 71 ;; esac\n'
        'done\n'
        f'exec {shlex.quote(real_flock)} "$@"\n'
    )
    (timers.bin / "flock").chmod(0o700)
    assert "STARTED" in finish(timers.start())
    assert "REUSED" in finish(timers.start())
    runtime = timers.root / "runtime"
    assert runtime.stat().st_mode & 0o777 == 0o700
    assert list(runtime.glob("timers/*/*/run.lock"))
    assert not list((timers.root / ".cache").rglob("*.lock"))


@pytest.mark.parametrize("kind", ["symlink", "public", "file"])
def test_unsafe_runtime_directory_is_rejected(timers, kind):
    runtime = timers.root / "runtime"
    if kind == "symlink":
        destination = timers.root / "destination"
        destination.mkdir(mode=0o755)
        runtime.symlink_to(destination, target_is_directory=True)
    elif kind == "public":
        runtime.mkdir(mode=0o755)
    else:
        runtime.write_text("keep")
    child = timers.start()
    out, err = child.communicate(timeout=15)
    assert child.returncode != 0
    assert "unsafe runtime directory" in err
    assert "STARTED" not in out
    if kind == "symlink":
        assert destination.stat().st_mode & 0o777 == 0o755
        assert not list(destination.iterdir())
    elif kind == "public":
        assert runtime.stat().st_mode & 0o777 == 0o755
    else:
        assert runtime.read_text() == "keep"
