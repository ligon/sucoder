"""Exited Linux children may lose proc access before their PID disappears."""
import builtins
import errno
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import pytest

from sucoder import dispatch_worker as worker


@pytest.mark.parametrize("boundary", ["directory", "link", "environment", "mounts"])
@pytest.mark.parametrize("state", ["Z", "R", "unknown"])
def test_permission_race_requires_fresh_exit_proof(tmp_path, monkeypatch, boundary, state):
    process = tmp_path / "123"
    (process / "fd").mkdir(parents=True)
    status = process / "stat"
    status.write_text("123 (native child) R 1 2\n")
    original_listdir, original_readlink, original_open = os.listdir, os.readlink, builtins.open

    def deny_after_transition():
        if state == "unknown":
            status.unlink()
        else:
            status.write_text("123 (native child) " + state + " 1 2\n")
        raise PermissionError(errno.EACCES, "process changed during inspection")

    def listdir(path):
        if str(path) == str(process / "fd"):
            if boundary == "directory":
                deny_after_transition()
            return ["7"]
        return original_listdir(path)

    def readlink(path):
        if str(path) == str(process / "fd/7"):
            if boundary == "link":
                deny_after_transition()
            return str(tmp_path / "local-state")
        return original_readlink(path)

    def open_file(path, *args, **kwargs):
        if str(path) == str(process / "environ"):
            deny_after_transition()
        return original_open(path, *args, **kwargs)

    def mounts(pids, **kwargs):
        if boundary == "mounts":
            deny_after_transition()
        return []

    monkeypatch.setattr(worker.os, "listdir", listdir)
    monkeypatch.setattr(worker.os, "readlink", readlink)
    monkeypatch.setattr(builtins, "open", open_file)
    runner = SimpleNamespace(descendants=lambda pid: [], inspect_file_mounts=mounts)
    arguments = dict(proc_root=str(tmp_path), expected_environment={} if boundary == "environment" else None)
    if state == "Z":
        result = worker.inspect_placement(SimpleNamespace(pid=123), runner, str(tmp_path), str(tmp_path), **arguments)
        assert result["databases"] == []
        assert result["environment_verified"] is False
    else:
        with pytest.raises(worker.Refused, match="cannot inspect owned child"):
            worker.inspect_placement(SimpleNamespace(pid=123), runner, str(tmp_path), str(tmp_path), **arguments)


def test_real_zombie_after_ownership_scan_is_no_longer_a_writer(tmp_path, monkeypatch):
    child = subprocess.Popen(["/bin/true"])
    try:
        deadline = time.monotonic() + 5
        while Path("/proc/%s/stat" % child.pid).read_text().rsplit(")", 1)[1].split()[0] != "Z":
            assert time.monotonic() < deadline
            time.sleep(.005)  # do not poll/reap the zombie before inspecting it
        original = worker.os.listdir

        def denied(path):
            if str(path) == "/proc/%s/fd" % child.pid:
                raise PermissionError(errno.EACCES, "Linux zombie fd access")
            return original(path)

        monkeypatch.setattr(worker.os, "listdir", denied)
        ownership = SimpleNamespace(members=lambda: [{"pid": child.pid}])
        runner = SimpleNamespace(inspect_file_mounts=lambda pids, **kwargs: [])
        result = worker.inspect_placement(child, runner, str(tmp_path), str(tmp_path), ownership=ownership)
        assert result["inspected_pids"] == []
    finally:
        child.wait(timeout=5)


@pytest.mark.parametrize("state", ["Z", "X", "R", "S", "D"])
def test_only_terminal_kernel_states_prove_exit(tmp_path, state):
    process = tmp_path / "123"
    process.mkdir()
    (process / "stat").write_text("123 (child name) " + state + " 1 2\n")
    assert worker.process_has_exited(123, str(tmp_path)) == (state in ("Z", "X"))


def test_missing_or_malformed_metadata_inside_existing_pid_is_not_exit(tmp_path):
    assert worker.process_has_exited(123, str(tmp_path))
    process = tmp_path / "123"
    process.mkdir()
    assert not worker.process_has_exited(123, str(tmp_path))
    (process / "stat").write_text("unparseable")
    assert not worker.process_has_exited(123, str(tmp_path))


def test_unreadable_stat_is_not_exit(tmp_path, monkeypatch):
    process = tmp_path / "123"
    process.mkdir()
    (process / "stat").write_text("123 (child) Z 1 2\n")
    original = builtins.open

    def denied(path, *args, **kwargs):
        if str(path) == str(process / "stat"):
            raise PermissionError(errno.EACCES, "cannot prove process state")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", denied)
    assert not worker.process_has_exited(123, str(tmp_path))
