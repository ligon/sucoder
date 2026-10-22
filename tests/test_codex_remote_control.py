"""Control commands inspect existing owners, including placement overrides."""
import json
import logging
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from sucoder import cli
from sucoder.config import Config
from sucoder.mirror import MirrorError
from tests.test_cli import (_confined_mirror_settings, _install_build_executor_fakes,
                            _slurm_config, _write_config)
from tests.test_codex_state import _state_manager


def test_observer_uses_login_route_without_allocating(tmp_path, monkeypatch):
    captured, allocations = _install_build_executor_fakes(monkeypatch, tmp_path)
    cli._build_executor(Config(human_user="coder", mirror_root=tmp_path / "mirrors"),
                        logging.getLogger("test"), dry_run=False,
                        mirror_settings=_confined_mirror_settings(tmp_path, confined=False),
                        cli_ctx=SimpleNamespace(obj={"codex_observe_only": True}))
    assert allocations == []
    assert captured["kwargs"]["is_compute_node"] is False


@pytest.mark.parametrize("explicit", [False, True])
def test_status_targets_actual_run_despite_changed_placement(tmp_path, monkeypatch, explicit):
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(manager, "_confined_job_state", lambda job: "RUNNING")
    scope, _ = manager._codex_scope(ctx)
    run = f"/tmp/sucoder-codex-state-test/state/codex/{scope}/" + "a" * 32
    helper = "/home/coder/.cache/sucoder/helper.py"
    calls = []

    def reply(args, **kw):
        calls.append(list(args))
        assert args[:3] == ["srun", "--jobid=4242", "--overlap"]
        command = args[3:]
        if command[0] == "tmux":
            text = run + "\n"
        elif command == ["cat", run + "/status.json"]:
            text = json.dumps({"remote_control": {"helper": helper}})
        else:
            assert command == ["python3", helper, "status", "--run", run]
            text = json.dumps({"status": "connected", "run": run})
        return SimpleNamespace(returncode=0, stdout=text, stderr="")

    # A later config change cannot redirect diagnostics into /local/job4242.
    ctx.settings.remote.slurm.codex_state.enabled = False
    manager.executor.run_agent.side_effect = reply
    assert manager.codex_remote_request(ctx, 4242, "status", run_dir=run if explicit else None)["run"] == run
    assert any("tmux" in c for c in calls) == (not explicit)
    assert all("/local/job" not in " ".join(c) for c in calls)


def test_status_rejects_another_scope_before_reading(tmp_path, monkeypatch):
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(manager, "_confined_job_state", lambda job: "RUNNING")
    with pytest.raises(MirrorError, match="scope"):
        manager.codex_remote_request(ctx, 4242, "status", run_dir="/tmp/state/codex/wrong/" + "a" * 32)
    manager.executor.run_agent.assert_not_called()


def _invoke(tmp_path, monkeypatch, command, *, job=4242, local=False, extra=()):
    from sucoder import session
    config = _write_config(tmp_path, skills_entry=tmp_path / "skills") if local else _slurm_config(tmp_path)
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)
    directory = tmp_path / "sessions"
    directory.mkdir(exist_ok=True)
    monkeypatch.setattr(session, "_session_dir", lambda: directory)
    if job:
        (directory / "sample--fake-slurm.yaml").write_text(f"slurm_job_id: {job}\nlogin_node: ln001\n")
    calls = []

    def request(ctx, job, action, **kw):
        calls.append((ctx, job, action, kw))
        return {"status": "connected", "node": "actual", "pairingCode": "123", "manualPairingCode": "abc"}

    def build(*args, **kw):
        assert kw["cli_ctx"].obj["codex_observe_only"] is True
        return SimpleNamespace(context_for=lambda mirror: mirror, codex_remote_request=request)

    monkeypatch.setattr(cli, "_build_manager_for_mirror", build)
    argv = ["--config", str(config)] + ([] if local else ["-T", "fake-slurm"])
    result = CliRunner().invoke(cli.app, [*argv, command, "sample", *extra])
    return result, calls


@pytest.mark.parametrize("command, action", [("codex-remote-status", "status"), ("codex-remote-pair", "pair")])
def test_cli_routes_exact_recorded_job(tmp_path, monkeypatch, command, action):
    result, calls = _invoke(tmp_path, monkeypatch, command, extra=["--json"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == [("sample", 4242, action, {"run_dir": None})]
    assert json.loads(result.output)["node"] == "actual"


def test_cli_missing_allocation_never_builds_or_requests(tmp_path, monkeypatch):
    result, calls = _invoke(tmp_path, monkeypatch, "codex-remote-status", job=None)
    assert result.exit_code == 1 and "No allocation" in result.output
    assert calls == []


def test_cli_local_explicit_run(tmp_path, monkeypatch):
    result, calls = _invoke(tmp_path, monkeypatch, "codex-remote-pair", local=True,
                            extra=["--run", "/tmp/exact", "--json"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == [("sample", 0, "pair", {"run_dir": "/tmp/exact"})]


def test_cli_qr_json_conflict_never_requests_code(tmp_path, monkeypatch):
    result, calls = _invoke(tmp_path, monkeypatch, "codex-remote-pair", extra=["--qr", "--json"])
    assert result.exit_code != 0 and "cannot be combined" in result.output
    assert calls == []


def test_lifecycle_status_uses_published_run_not_latest_or_current_placement(tmp_path, monkeypatch):
    from sucoder.session import RemoteSession
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(manager, "_confined_job_state", lambda job: "RUNNING")
    scope, _ = manager._codex_scope(ctx)
    run = f"/tmp/sucoder-codex-state-live/state/codex/{scope}/" + "b" * 32
    session = RemoteSession.load(ctx.settings.name, target_name=manager.target_name)
    session.managed_owner = {"remote_isolated": True}
    session.save()
    ctx.settings.remote.slurm.codex_state.enabled = False
    calls = []

    def reply(args, **kw):
        calls.append(args)
        if "tmux" in args:
            value = run + "\n"
        else:
            assert args[-1] in (run + "/status.json", run + "/owner.json")
            value = json.dumps({"run": "b" * 32})
        return SimpleNamespace(returncode=0, stdout=value, stderr="")

    manager.executor.run_agent.side_effect = reply
    report = manager.codex_state_status(ctx, 4242)
    assert report.run_dir == run
    assert len(calls) == 3
    assert not any("latest-run" in " ".join(c) for c in calls)


def test_unpublished_successor_keeps_missing_status_watchdog_active(tmp_path, monkeypatch):
    from sucoder.session import RemoteSession
    from tests.test_codex_state_renew import _token
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(manager, "_confined_job_state", lambda job: "RUNNING")
    session = RemoteSession.load(ctx.settings.name, target_name=manager.target_name)
    session.managed_owner = {"remote_isolated": True}
    session.save()
    monkeypatch.setattr(manager, "_confined_job_state", lambda job: "RUNNING")
    manager.executor.run_agent.return_value = SimpleNamespace(returncode=0, stdout="\n", stderr="")
    result = manager.codex_state_readiness(ctx, 4242, _token(), budget_seconds=0)
    assert result.outcome == "pending" and result.no_status is True


@pytest.mark.parametrize("state", [None, "PENDING", "COMPLETING", "SUSPENDED"])
def test_control_does_not_enter_an_inactive_allocation(tmp_path, monkeypatch, state):
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(manager, "_confined_job_state", lambda job: state)
    with pytest.raises(MirrorError, match="running owner"):
        manager.codex_remote_request(ctx, 4242, "status")
    manager.executor.run_agent.assert_not_called()


def test_release_drains_recorded_remote_owner_after_config_optout(tmp_path, monkeypatch):
    from tests.test_codex_state_release import _Cluster
    cluster = _Cluster(tmp_path, monkeypatch, codex_state=False, harness="claude", session_yaml=(
        "login_node: ln002\nslurm_job_id: 7654321\ncompute_node: n0032\n"
        "managed_owner:\n  command: [codex]\n  remote_isolated: true\n"))
    result = cluster.release("-f")
    assert result.exit_code == 0, (result.output, result.exception)
    assert cluster.events == ["drain", "scancel"]
