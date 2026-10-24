"""Exercise the pilot boundary without authenticating or starting real agents."""

import json
import os
from pathlib import Path
import subprocess

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "happier-pilot"


@pytest.fixture
def pilot(tmp_path):
    fake = tmp_path / "provider"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "print(json.dumps({'args': sys.argv[1:], 'env': {k:v for k,v in os.environ.items() "
        "if k.startswith('HAPPIER_') or k == 'CLAUDE_CONFIG_DIR'}}))\n"
    )
    fake.chmod(0o700)
    local = tmp_path / "local"
    local.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HAPPIER_", "SLURM_", "SUCODER_"))}
    env.update(
        SLURM_JOB_ID="1234", SUCODER_LOCAL_ROOT=str(local),
        SUCODER_HAPPIER_PILOT_CLI=str(fake), HAPPIER_CLAUDE_PATH=str(fake),
        CLAUDE_CONFIG_DIR=str(tmp_path / "claude-config"),
    )
    return env, local


def invoke(name, args, env):
    return subprocess.run(["bash", str(SCRIPTS / name), *args], env=env, text=True, capture_output=True)


def test_adapter_preserves_provider_arguments_and_config(pilot):
    env, local = pilot
    args = ["--session-id", "a-session", "--system-prompt", "context with spaces\nand a line", "--model", "opus"]
    result = invoke("claude", args, env)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["args"] == ["claude", "--happy-starting-mode", "remote", "--permission-mode", "default", *args]
    assert data["env"]["CLAUDE_CONFIG_DIR"] == env["CLAUDE_CONFIG_DIR"]
    assert data["env"]["HAPPIER_HOME_DIR"] == str(local / "happier-pilot-1234")
    assert data["env"]["HAPPIER_SESSION_AUTOSTART_DAEMON"] == "0"
    assert data["env"]["HAPPIER_TRANSCRIPT_STORAGE"] == "persisted"
    assert (local / "happier-pilot-1234").stat().st_mode & 0o777 == 0o700


def test_control_uses_remote_start_and_default_permissions_for_claude(pilot):
    env, _ = pilot
    result = invoke("control", ["claude", "--model", "sonnet"], env)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["args"] == [
        "claude", "--happy-starting-mode", "remote", "--permission-mode", "default",
        "--model", "sonnet",
    ]
    assert data["env"]["HAPPIER_TRANSCRIPT_STORAGE"] == "persisted"


def test_explicit_runtime_options_can_override_pilot_defaults(pilot):
    env, _ = pilot
    args = ["--happy-starting-mode", "local", "--permission-mode", "plan"]
    result = invoke("claude", args, env)
    assert result.returncode == 0, result.stderr
    forwarded = json.loads(result.stdout)["args"]
    assert forwarded[-len(args):] == args


def test_sucoder_records_the_adapter_as_claude_with_its_provider_uuid(tmp_path, monkeypatch):
    from tests.test_reply_launch import _claude_launch
    from sucoder.config import AgentType

    manager, argv = _claude_launch(tmp_path, monkeypatch, [str(SCRIPTS / "claude")])
    assert manager._detected_agent_type == AgentType.CLAUDE
    assert argv[argv.index("--session-id") + 1] == manager._claude_session_id
    assert manager._launch_identity() == ("claude", manager._claude_session_id)


@pytest.mark.parametrize("mode", ["direct", "persisted"])
def test_explicit_transcript_mode_reaches_happier(pilot, mode):
    env, _ = pilot
    result = invoke("control", ["status"], dict(env, HAPPIER_TRANSCRIPT_STORAGE=mode))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["env"]["HAPPIER_TRANSCRIPT_STORAGE"] == mode


def test_invalid_transcript_mode_stops_before_starting_happier(pilot):
    env, local = pilot
    result = invoke("control", ["status"], dict(env, HAPPIER_TRANSCRIPT_STORAGE="driect"))
    assert result.returncode == 2
    assert "must be direct or persisted" in result.stderr
    assert result.stdout == ""
    assert list(local.iterdir()) == []


def test_two_allocations_get_different_homes_even_with_the_same_local_root(pilot):
    env, _ = pilot
    first = json.loads(invoke("control", ["status"], env).stdout)
    second = json.loads(invoke("control", ["status"], dict(env, SLURM_JOB_ID="5678")).stdout)
    assert first["env"]["HAPPIER_HOME_DIR"] != second["env"]["HAPPIER_HOME_DIR"]


def test_probe_needs_no_allocation_and_creates_no_happier_home(pilot):
    env, local = pilot
    del env["SLURM_JOB_ID"]
    result = invoke("claude", ["--version"], env)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["args"] == ["--version"]
    assert list(local.iterdir()) == []


@pytest.mark.parametrize("name,args", [("claude", []), ("control", ["auth", "login"])])
def test_runtime_refuses_outside_an_allocation(pilot, name, args):
    env, _ = pilot
    del env["SLURM_JOB_ID"]
    result = invoke(name, args, env)
    assert result.returncode == 2
    assert "Slurm allocation" in result.stderr


def test_adapter_refuses_recursion(pilot):
    env, _ = pilot
    result = invoke("claude", [], dict(env, HAPPIER_CLAUDE_PATH=str(SCRIPTS / "claude")))
    assert result.returncode == 2
    assert "points back" in result.stderr


def test_control_refuses_another_instances_home(pilot):
    env, local = pilot
    result = invoke("control", ["daemon", "stop", "--kill-sessions"], dict(env, HAPPIER_HOME_DIR=str(local)))
    assert result.returncode == 2
    assert "conflicts" in result.stderr


def test_control_refuses_a_symlinked_home(pilot, tmp_path):
    env, local = pilot
    (local / "happier-pilot-1234").symlink_to(tmp_path, target_is_directory=True)
    result = invoke("control", ["status"], env)
    assert result.returncode == 2
    assert "symlink" in result.stderr


@pytest.mark.parametrize("args", [["setup", "--cloud", "--yes"], ["service", "install"], ["daemon", "start"], ["daemon", "stop", "--all"]])
def test_control_refuses_service_install_and_detached_daemon(pilot, args):
    env, _ = pilot
    assert invoke("control", args, env).returncode == 2


def test_install_rejects_a_corrupt_release_before_unpacking(pilot, tmp_path):
    env, _ = pilot
    archive = tmp_path / "release.tar.gz"
    archive.write_bytes(b"not the pinned release")
    destination = tmp_path / "installed"
    result = invoke("install", [str(destination), str(archive)], env)
    assert result.returncode == 1
    assert "SHA256 mismatch" in result.stderr
    assert not destination.exists()
    assert not list(tmp_path.glob("installed.install.*"))


def test_install_keeps_an_existing_destination(pilot, tmp_path):
    env, _ = pilot
    destination = tmp_path / "installed"
    destination.mkdir()
    marker = destination / "marker"
    marker.write_text("keep this")
    result = invoke("install", [str(destination)], env)
    assert result.returncode == 2
    assert marker.read_text() == "keep this"
