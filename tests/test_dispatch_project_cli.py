"""CLI composition, including real frozen verification requests."""
import json
from pathlib import Path

from typer.testing import CliRunner

from sucoder import cli, dispatch
from tests.test_dispatch import source, scheduler
from tests.test_dispatch_cli import _no_interactive
from tests.test_dispatch_project import profile_file


def test_submit_forwards_explicit_project_profile(monkeypatch, tmp_path):
    _no_interactive(monkeypatch)
    observed = {}
    def submit(*args, **kwargs):
        observed.update(kwargs)
        return {"state": "dry_run"}
    monkeypatch.setattr(dispatch, "submit", submit)
    result = CliRunner().invoke(cli.app, ["dispatch", "submit", "/shared/repo", "--ref", "HEAD",
        "--task-file", "/shared/task", "--agent", "codex", "--account", "a", "--partition", "p",
        "--project-profile", str(tmp_path / "profile.json"), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert observed["project_profile"] == tmp_path / "profile.json"


def test_verify_stage_freezes_real_request_without_interactive_startup_or_submission(source, monkeypatch, tmp_path):
    _no_interactive(monkeypatch)
    calls = scheduler(monkeypatch)
    profile = profile_file(tmp_path)
    command = tmp_path / "command.json"
    command.write_text(json.dumps(["{python}", "-c", "pass"]))
    stage = tmp_path / "verify"
    result = CliRunner().invoke(cli.app, ["dispatch", "verify-stage", str(source["repo"]), "--ref", "HEAD",
        "--project-profile", str(profile), "--command-file", str(command), "--directory", str(stage)])
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["state"] == "prepared"
    assert Path(value["directory"]) == stage
    assert (stage / "verification-request.json").is_file()
    assert not calls
