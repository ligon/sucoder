"""Placement belongs to the launched Codex, not databases used by its tools."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from sucoder import codex_state_runner as runner


# A copied Python executable named codex provides the same /proc/exe boundary
# as the native binary, without model requests, credentials or bubblewrap.
_DATABASE = r'''
import pathlib, sqlite3, sys, time
connection = sqlite3.connect(sys.argv[1])
connection.execute("create table t(x)")
connection.commit()
pathlib.Path(sys.argv[2]).touch()
while True:
    time.sleep(0.1)
'''

_LAUNCH = r'''
import json, pathlib, sqlite3, subprocess, sys, time
options = json.loads(sys.argv[1])
connection = sqlite3.connect(options["owner_db"])
connection.execute("create table t(x)")
connection.commit()
# A native app server started by the native frontend still owns state.
subprocess.Popen([sys.executable, "-c", options["database"],
                  options["server_db"], options["server_ready"]])
# An ordinary tool can use both project databases and a nested test Codex.
subprocess.Popen([options["python"], "-c", options["database"],
                  options["tool_db"], options["tool_ready"]])
subprocess.Popen([options["python"], "-c",
    "import subprocess, sys; p = subprocess.Popen(sys.argv[1:]); p.wait()",
    sys.executable, "-c", options["database"],
    options["nested_db"], options["nested_ready"]])
while not all(pathlib.Path(options[name]).exists()
              for name in ("server_ready", "tool_ready", "nested_ready")):
    time.sleep(0.01)
pathlib.Path(options["ready"]).touch()
while True:
    time.sleep(0.1)
'''


@pytest.mark.skipif(not os.path.exists("/proc/self/exe"), reason="placement inspection requires Linux /proc")
@pytest.mark.parametrize("misplaced", [None, "owner", "server"])
def test_monitor_distinguishes_native_state_from_tool_databases(tmp_path, monkeypatch, misplaced):
    native = tmp_path / "codex"
    shutil.copy2(sys.executable, native)
    db = tmp_path / "managed"
    db.mkdir()
    options = {"python": sys.executable, "database": _DATABASE}
    for name in ("owner", "server", "tool", "nested"):
        parent = db if name in ("owner", "server") and name != misplaced else tmp_path
        directory = parent / name
        directory.mkdir()
        options[name + "_db"] = str(directory / "state_5.sqlite")
        options[name + "_ready"] = str(tmp_path / (name + ".ready"))
    options["ready"] = str(tmp_path / "launch.ready")
    monkeypatch.setitem(runner.LOG_FILE, "path", str(tmp_path / "runner.log"))
    monkeypatch.setitem(runner.LOG_FILE, "quiet", True)
    child = subprocess.Popen([str(native), "-c", _LAUNCH, json.dumps(options)], start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not os.path.exists(options["ready"]):
            assert child.poll() is None, "synthetic native Codex exited during startup"
            assert time.monotonic() < deadline, "synthetic processes did not open their databases"
            time.sleep(0.01)
        updates = []
        status = SimpleNamespace(update=lambda state, **kw: updates.append(dict(state=state, **kw)))
        result = {}

        def end_after_inspection(timeout):
            child.terminate()
            child.wait(timeout=5)

        runner.monitor(child, str(db), str(tmp_path), status, 10, result,
                       exit_event=SimpleNamespace(wait=end_after_inspection))
        assert result["outcome"] == ("misplaced" if misplaced else "placed")
        if misplaced:
            assert result["misplaced"] == [options[misplaced + "_db"]]
            assert child.wait(timeout=5) != 0
        else:
            assert set(updates[0]["placed"]) == {options["owner_db"], options["server_db"]}
    finally:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)


@pytest.mark.parametrize("unreadable", [False, True])
def test_placement_inspection_keeps_wrappers_and_uncertain_processes(monkeypatch, unreadable):
    # bwrap -> bash -> npm shim -> native -> tool -> nested native.
    parents = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    executables = {1: "/bin/bwrap", 2: "/bin/bash", 3: "/bin/node",
                   4: "/bin/codex (deleted)", 5: "/bin/python3", 6: "/bin/codex"}
    monkeypatch.setattr(runner, "descendants", lambda pid: [2, 3, 4, 5, 6])
    monkeypatch.setattr(runner, "parent_of", parents.get)

    def executable(path):
        pid = int(path.split("/")[2])
        if unreadable and pid == 5:
            raise PermissionError(path)
        return executables[pid]

    monkeypatch.setattr(runner.os, "readlink", executable)
    assert runner.placement_pids(1) == ([1, 2, 3, 4, 5, 6] if unreadable else [1, 2, 3, 4])
