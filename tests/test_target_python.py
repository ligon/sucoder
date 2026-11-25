"""The target's ``python:`` setting reaches every staged helper on that target.

A bare ``python3`` is whatever the target's PATH finds first; on Savio a
clean environment finds the image's 3.6.8, below SuCoder's floor.
"""
from __future__ import annotations

import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from sucoder import PYTHON_FLOOR
from sucoder.codex_remote import build_remote_command
from sucoder.config import (AgentLauncher, ConfigError, ConfigWarning, ToolPreflightConfig,
                            _parse_remote_config, _parse_slurm_config)
from sucoder.replies import probe_command
from sucoder.tool_preflight import BELOW, HOST_TAG, LINE_TAG
from tests.test_codex_remote import fake_codex  # noqa: F401 - fixture
from tests.test_codex_state import _state_manager

SAVIO = "/usr/bin/python3.12"


def target(**extra):
    return {"gateway": "gw", "transfer_host": "dtn", **extra}


def recorder(path, *, exit_code=0):
    """A stand-in interpreter that records its argv, one JSON line per call."""
    log = path.with_suffix(".calls")
    path.write_text("#!/bin/sh\nprintf '%%s\\n' \"$*\" >> %s\nexit %d\n" % (log, exit_code))
    path.chmod(0o755)
    return log


# ------------------------------------------------------------------ config

def test_python_defaults_to_python3_and_takes_a_name_or_an_absolute_path():
    assert _parse_remote_config(target()).python == "python3"
    assert _parse_remote_config(target(python=SAVIO)).python == SAVIO
    assert _parse_remote_config(target(python="python3.12")).python == "python3.12"


@pytest.mark.parametrize("value", ["bin/python3", "~/bin/python3", "/usr/bin/python3 -I", "py;id",
                                   "$(id)", "-c", "", None, 3.12])
def test_python_must_be_one_shell_word_needing_no_quoting(value):
    with pytest.raises(ConfigError, match="remote.python"):
        _parse_remote_config(target(python=value))


def test_python_nested_under_slurm_is_flagged_as_target_level():
    with pytest.warns(ConfigWarning, match="`python` is nested under `slurm:`"):
        _parse_slurm_config({"partition": "p", "account": "a", "python": SAVIO})


# ------------------------------------------------------------------ launch

def test_launch_runs_the_runner_and_context_installer_under_the_target_python(tmp_path, monkeypatch):
    manager, ctx, confined = _state_manager(tmp_path, monkeypatch)
    ctx.settings.remote.python = SAVIO
    ctx.settings.agent_launcher = AgentLauncher(command=["claude"])
    manager.launch_agent(ctx, sync=False, command_override=["codex", "remote-control"],
                         supports_inline_prompt=True, detached=True)
    launched = confined.call_args.args[1]
    assert launched[0] == SAVIO
    inner = launched[launched.index("--") + 1:]
    assert inner[0] == SAVIO and inner[1].endswith(".txt")


def test_remote_control_status_runs_the_helper_under_the_target_python(tmp_path, monkeypatch):
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    ctx.settings.remote.python = SAVIO
    monkeypatch.setattr(manager, "_confined_job_state", lambda job: "RUNNING")
    scope, _ = manager._codex_scope(ctx)
    run = f"/tmp/sucoder-codex-state-test/state/codex/{scope}/" + "a" * 32
    helper = "/home/coder/.cache/sucoder/helper.py"
    helpers = []

    def reply(args, **kw):
        command = list(args[3:])
        if command == ["cat", run + "/status.json"]:
            text = json.dumps({"remote_control": {"helper": helper}})
        else:
            helpers.append(command)
            text = json.dumps({"status": "connected", "run": run})
        return SimpleNamespace(returncode=0, stdout=text, stderr="")

    manager.executor.run_agent.side_effect = reply
    manager.codex_remote_request(ctx, 4242, "status", run_dir=run)
    assert helpers == [[SAVIO, helper, "status", "--run", run]]


def test_preflight_grades_the_target_python_against_the_floor(tmp_path, monkeypatch):
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    ctx.settings.remote.python = SAVIO
    stdout = "\n".join([f"{HOST_TAG}\tn0034.savio4", "\t".join([LINE_TAG, SAVIO, SAVIO, "Python 3.6.8"])])
    manager.executor.run_agent.return_value = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
    _, reports = manager.tool_preflight(ctx)
    assert SAVIO in manager.executor.run_agent.call_args.kwargs["input"]
    report = next(r for r in reports if r.name == SAVIO)
    assert report.status == BELOW and report.floor == "%d.%d" % PYTHON_FLOOR


def test_a_configured_floor_for_the_target_python_is_kept(tmp_path, monkeypatch):
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    manager.config.tool_preflight = ToolPreflightConfig(floors={"python3": None})
    stdout = "\n".join([f"{HOST_TAG}\th", "\t".join([LINE_TAG, "python3", "/usr/bin/python3", "Python 3.6.8"])])
    manager.executor.run_agent.return_value = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
    _, reports = manager.tool_preflight(ctx)
    assert next(r for r in reports if r.name == "python3").floor is None


# ------------------------------------------------------------------ remote helpers

def test_reply_probe_runs_under_the_given_python_and_defaults_to_python3(tmp_path):
    python = tmp_path / "py"
    log = recorder(python)
    command = probe_command({"spec": 1}, source="pass", python=str(python))
    subprocess.run(["bash", "-c", command], env={"PATH": "/usr/bin:/bin"}, timeout=30, check=True)
    assert log.read_text().startswith("-I -c ")
    assert probe_command({"spec": 1}, source="pass").count("python3 -I -c ") == 2


def test_remote_supervisor_registers_through_the_runner_interpreter(fake_codex, tmp_path):  # noqa: F811
    python = tmp_path / "runner-python"
    log = recorder(python)
    image = tmp_path / "image-bin"
    image.mkdir()
    stale = recorder(image / "python3", exit_code=1)
    command = build_remote_command([str(fake_codex)], "server")
    env = dict(os.environ, FAKE_RECORD=str(tmp_path), PATH=f"{image}:{os.environ['PATH']}",
               SUCODER_CODEX_REMOTE_HELPER=str(tmp_path / "helper.py"), SUCODER_CODEX_REMOTE_PYTHON=str(python))
    proc = subprocess.Popen(command, env=env, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(100):
            if log.exists() and len(log.read_text().splitlines()) >= 2:
                break
            subprocess.run(["sleep", "0.1"])
    finally:
        proc.terminate()
        proc.communicate(timeout=15)
    calls = log.read_text().splitlines()
    assert [c.split()[1] for c in calls] == ["register", "wait"]
    assert all(c.split()[0] == str(tmp_path / "helper.py") for c in calls)
    assert not stale.exists()
