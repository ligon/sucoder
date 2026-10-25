"""Headless dispatch bypasses interactive launch and propagates job evidence."""
import json

import pytest

from typer.testing import CliRunner

from sucoder import cli, dispatch


runner = CliRunner()
ID = "a" * 32


def _no_interactive(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("headless dispatch entered interactive startup")
    monkeypatch.setattr(cli, "run_startup_checks", refuse)
    monkeypatch.setattr(cli, "build_default_config", refuse)
    monkeypatch.setattr(cli, "load_config", refuse)


def test_help_does_not_detect_agents_or_check_privileges(monkeypatch):
    _no_interactive(monkeypatch)
    result = runner.invoke(cli.app, ["dispatch", "--help"])
    assert result.exit_code == 0, result.output
    assert "submit" in result.output


def test_submit_uses_headless_controller_and_explicit_resources(monkeypatch, tmp_path):
    _no_interactive(monkeypatch)
    seen = {}
    def submit(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return {"dispatch_id": ID, "state": "submitted", "job_id": 51}
    monkeypatch.setattr(dispatch, "submit", submit)
    result = runner.invoke(cli.app, ["dispatch", "--root", str(tmp_path), "submit", "/shared/repo",
        "--ref", "abc123", "--task-file", "/shared/task", "--agent", "claude",
        "--account", "free", "--partition", "htc", "--cpus", "2", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["job_id"] == 51
    assert seen["args"][0].as_posix() == "/shared/repo"
    assert seen["args"][3] == "claude"
    assert seen["args"][5]["cpus"] == 2
    assert seen["kwargs"]["dry_run"] is True
    assert seen["kwargs"]["directory_root"] == tmp_path


def test_ambiguous_submission_is_unavailable_not_success(monkeypatch):
    _no_interactive(monkeypatch)
    monkeypatch.setattr(dispatch, "submit", lambda *a, **k: {"state": "submission_unknown"})
    result = runner.invoke(cli.app, ["dispatch", "submit", "/shared/repo", "--ref", "main",
        "--task-file", "/shared/task", "--agent", "codex", "--account", "a", "--partition", "p"])
    assert result.exit_code == 3
    assert json.loads(result.output)["state"] == "submission_unknown"


def test_wait_propagates_worker_failure(monkeypatch):
    _no_interactive(monkeypatch)
    monkeypatch.setattr(dispatch, "wait", lambda *a, **k: {
        "state": "failed", "result": {"state": "failed", "exit_code": 17}})
    result = runner.invoke(cli.app, ["dispatch", "wait", ID])
    assert result.exit_code == 17


def test_wait_timeout_does_not_claim_success(monkeypatch):
    _no_interactive(monkeypatch)
    monkeypatch.setattr(dispatch, "wait", lambda *a, **k: {
        "state": "submitted", "wait_timed_out": True, "result": None})
    result = runner.invoke(cli.app, ["dispatch", "wait", ID])
    assert result.exit_code == 3


def test_wait_preserves_validation_failure_after_native_success(monkeypatch):
    _no_interactive(monkeypatch)
    monkeypatch.setattr(dispatch, "wait", lambda *a, **k: {
        "state": "failed", "result": {"state": "failed", "exit_code": 0,
                                         "supervisor_exit_code": 74}})
    result = runner.invoke(cli.app, ["dispatch", "wait", ID])
    assert result.exit_code == 74


def test_wait_returns_zero_for_explicit_worker_success(monkeypatch):
    _no_interactive(monkeypatch)
    monkeypatch.setattr(dispatch, "wait", lambda *a, **k: {
        "state": "succeeded", "result": {"state": "succeeded", "exit_code": 0}})
    assert runner.invoke(cli.app, ["dispatch", "wait", ID]).exit_code == 0


def test_wait_honors_supervisor_failure_even_with_stale_success_label(monkeypatch):
    _no_interactive(monkeypatch)
    monkeypatch.setattr(dispatch, "wait", lambda *a, **k: {
        "state": "succeeded", "result": {"state": "succeeded", "exit_code": 0,
                                          "supervisor_exit_code": 74}})
    assert runner.invoke(cli.app, ["dispatch", "wait", ID]).exit_code == 74


def test_wait_task_requirement_is_distinct_from_infrastructure(monkeypatch):
    _no_interactive(monkeypatch)
    monkeypatch.setattr(dispatch, "wait", lambda *a, **k: {
        "state": "succeeded", "result": {"state": "succeeded", "exit_code": 0,
                                           "task": {"status": "blocked"}}})
    assert runner.invoke(cli.app, ["dispatch", "wait", ID]).exit_code == 0
    assert runner.invoke(cli.app, ["dispatch", "wait", ID, "--require-task-complete"]).exit_code == 4


def test_dispatch_refuses_interactive_remote_routing(monkeypatch):
    _no_interactive(monkeypatch)
    result = runner.invoke(cli.app, ["--target", "savio", "dispatch", "--help"])
    assert result.exit_code == 2
    assert "current Slurm cluster" in result.output


def test_invalid_id_cannot_escape_result_root(monkeypatch):
    _no_interactive(monkeypatch)
    result = runner.invoke(cli.app, ["dispatch", "status", "../other"])
    assert result.exit_code == 2
    assert json.loads(result.output)["state"] == "error"


@pytest.mark.parametrize("global_options", [
    ["--time", "00:10:00"], ["--local-disk-root", "/different"],
    ["--local-disk"], ["--no-local-disk"], ["--login-node", "ln001"],
    ["--debug-ssh"], ["--agent-sudo"], ["--no-agent-sudo"], ["--x11"], ["--no-x11"],
])
def test_dispatch_refuses_explicit_global_startup_options(monkeypatch, global_options):
    _no_interactive(monkeypatch)
    result = runner.invoke(cli.app, [*global_options, "dispatch", "--help"])
    assert result.exit_code == 2, result.output
    normalized = " ".join(result.output.split())
    assert "global startup options" in normalized
    assert "dispatch submit" in normalized


def test_dispatch_keeps_clear_config_routing_refusal(monkeypatch):
    _no_interactive(monkeypatch)
    result = runner.invoke(cli.app, ["--config", "/shared/config.yaml", "dispatch", "--help"])
    assert result.exit_code == 2, result.output
    assert "current Slurm cluster" in result.output
