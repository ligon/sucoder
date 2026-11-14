"""Dream startup uses derived defaults and diagnoses unavailable log locks."""
import errno
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from sucoder import cli, config as settings, dream, dream_explore, dream_native
from tests.test_dream import workspace  # noqa: F401
from tests.test_dream_explore import ExploringTransport, options


def test_missing_default_config_launches_without_agent_detection_or_writing_config(workspace, monkeypatch):
    repo, _, _, _ = workspace
    monkeypatch.chdir(repo)
    native = ExploringTransport(no_selection=True)
    monkeypatch.setattr(dream_native, "Transport", lambda *a, **kw: native)
    def forbidden(*a, **kw):
        raise AssertionError("Dream tried interactive agent detection")
    monkeypatch.setattr(settings, "detect_agent_command", forbidden)
    result = CliRunner().invoke(cli.app, ["dream", "--explore", "--harness", "codex", "--model", "fixture"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["state"] == "no_selection"
    assert len(native.calls) == 1
    assert not Path("~/.sucoder/config.yaml").expanduser().exists()
    for command in ("status", "budget"):
        result = CliRunner().invoke(cli.app, ["dream", command])
        assert result.exit_code == 0, result.output
    assert len(native.calls) == 1


def test_missing_default_config_without_model_keeps_noninteractive_diagnostic(workspace, monkeypatch):
    monkeypatch.chdir(workspace[0])
    result = CliRunner().invoke(cli.app, ["dream", "--harness", "codex"])
    assert result.exit_code == 2, result.output
    assert "supply --model; dreaming never prompts" in json.loads(result.output)["error"]


def test_detached_derived_defaults_do_not_pass_a_missing_config_to_child(workspace, monkeypatch):
    from sucoder import dream_cli
    monkeypatch.chdir(workspace[0])
    calls = []
    original = dream_cli.subprocess.Popen
    def spawn(argv, **kwargs):
        if "from sucoder.cli import app; app()" not in argv:
            return original(argv, **kwargs)
        calls.append(argv)
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(dream_cli.subprocess, "Popen", spawn)
    result = CliRunner().invoke(cli.app, ["dream", "--explore", "--detach", "--harness", "codex", "--model", "fixture"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["state"] == "detached"
    assert "--config" not in calls[0]
    assert "--job" in calls[0]


def test_invalid_implicit_config_never_falls_back(workspace, monkeypatch):
    path = Path("~/.sucoder/config.yaml").expanduser()
    path.parent.mkdir()
    path.write_text("invalid: [\n")
    monkeypatch.chdir(workspace[0])
    result = CliRunner().invoke(cli.app, ["dream", "--harness", "codex", "--model", "fixture"])
    assert result.exit_code == 2, result.output
    assert "Failed to parse YAML" in json.loads(result.output)["error"]


@pytest.mark.parametrize("existing", [False, True])
def test_explicit_missing_or_invalid_config_never_falls_back(workspace, monkeypatch, existing):
    path = workspace[1].parent / "bad.yaml"
    if existing:
        path.write_text("invalid: [\n")
    monkeypatch.chdir(workspace[0])
    result = CliRunner().invoke(cli.app, ["--config", str(path), "dream", "--harness", "codex", "--model", "fixture"])
    assert result.exit_code == 2, result.output
    assert json.loads(result.output)["state"] == "rejected"


def test_unavailable_log_locks_reject_before_job_creation_or_provider_work(workspace, monkeypatch):
    repo, _, config, _ = workspace
    native = ExploringTransport(no_selection=True)
    def unavailable(*args):
        raise OSError(errno.ENOLCK, "No locks available")
    monkeypatch.setattr(fcntl, "flock", unavailable)
    with pytest.raises(dream.DreamError) as failure:
        dream_explore.start(config, cwd=repo, overrides=options(), transport=native)
    assert str(config.log_dir / "dream-jobs") in str(failure.value)
    assert "log_dir" in str(failure.value) and "lock support" in str(failure.value)
    assert not native.calls
    assert not list((config.log_dir / "dream-jobs").glob("[0-9a-f]" * 32))


def test_other_log_errors_keep_their_original_cause(workspace, monkeypatch):
    def unavailable(*args):
        raise OSError(errno.EIO, "fixture I/O failure")
    monkeypatch.setattr(fcntl, "flock", unavailable)
    with pytest.raises(OSError, match="fixture I/O failure"):
        dream_explore.prepare(workspace[2], cwd=workspace[0], overrides=options())
