"""Descriptor placement inspection never stats a shared descriptor target."""
import errno
import os

import pytest

from sucoder import codex_state_runner as runner


def _proc(tmp_path):
    process = tmp_path / "123"
    (process / "fd").mkdir(parents=True)
    (process / "fdinfo").mkdir()
    (process / "mountinfo").write_text(
        "10 1 0:1 / /shared rw - nfs4 server:/shared rw\n"
        "20 1 8:1 / /local/job123 rw - ext4 /dev/sda rw\n")
    return process


def test_mount_evidence_does_not_follow_inaccessible_database(tmp_path, monkeypatch):
    process = _proc(tmp_path)
    (process / "fd" / "7").symlink_to("/shared/stalled/state.sqlite-wal")
    (process / "fdinfo" / "7").write_text("pos:\t0\nflags:\t0100002\nmnt_id:\t10\n")
    original_stat = runner.os.stat
    def refuse_stat(*args, **kwargs):
        if str(args[0]).startswith(str(process / "fd")):
            raise AssertionError("descriptor target stat would block on shared I/O")
        return original_stat(*args, **kwargs)
    monkeypatch.setattr(runner.os, "stat", refuse_stat)
    records = runner.inspect_file_mounts([123], str(tmp_path))
    assert records == [{"pid": 123, "fd": "7", "path": "/shared/stalled/state.sqlite-wal",
                        "mount_id": "10", "flags": 0o100002,
                        "filesystem": "nfs4", "mountpoint": "/shared"}]


def test_disappearing_descriptor_and_process_are_races(tmp_path):
    process = _proc(tmp_path)
    (process / "fd" / "7").symlink_to("/local/job123/state.sqlite")
    assert runner.inspect_file_mounts([123, 999], str(tmp_path)) == []


def test_unknown_mount_is_explicit_not_assumed_local(tmp_path):
    process = _proc(tmp_path)
    (process / "fd" / "7").symlink_to("/mystery/state.sqlite")
    (process / "fdinfo" / "7").write_text("flags:\t02\nmnt_id:\t99\n")
    assert runner.inspect_file_mounts([123], str(tmp_path))[0]["filesystem"] is None


def test_unreadable_owned_metadata_remains_failure(tmp_path, monkeypatch):
    _proc(tmp_path)
    original = runner.os.listdir
    def deny(path):
        if str(path).endswith("123/fd"):
            raise PermissionError(errno.EACCES, "denied")
        return original(path)
    monkeypatch.setattr(runner.os, "listdir", deny)
    with pytest.raises(PermissionError):
        runner.inspect_file_mounts([123], str(tmp_path))
