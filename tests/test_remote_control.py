"""Remote-control argv, real shell transport, and service lifecycle (ledger 4).

No Codex service, SSH connection, or allocation is started by these tests.
Private tmux servers run only a local fake agent or cat.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sucoder import cli
from sucoder.agent_mode import (
    CODEX_REMOTE_CONTROL, MODE_PROBE_SH, SERVICE_LAUNCH_PROBE_SH, TERMINAL,
    codex_remote_command, config_override, service_launch, service_marker,
)
from sucoder.config import AgentLauncher, BranchPrefixes, Config, MirrorSettings, RemoteConfig, SlurmConfig
from sucoder.messaging import Recipient, plan_recipients, send_keys_command
from sucoder.mirror import MirrorError, MirrorManager
from sucoder.session import RemoteSession
from sucoder.sessions_report import (
    SESSION_PANE_PROBE_SH, parse_login_session_status, parse_pane_status, render_report,
)
from tests.test_cli import _SnapshotsHarness, _message_report, _run_message, _run_peek
from tests.test_messaging import PLAN, _job, _report
from tests.test_mirror import _confined_manager, _confined_responder


LAUNCH = {"command": ["codex", "remote-control"], "model": "test-model"}
PRELUDE = ('SYSTEM PROMPT\nquotes: " and \'\n$HOME $(touch INJECTED) `false` \\\n'
           'TARGET INSTRUCTIONS\nSKILL CATALOG\nUnicode: \u03bb \U0001f642\x7f\n')


@pytest.mark.parametrize("command, expected", [
    (["codex", "remote-control"], ["codex", "remote-control"]),
    (["/opt/bin/codex", "remote-control", "start"], ["/opt/bin/codex", "remote-control"]),
    (["codex", "-c", 'model="remote-control"', "remote-control", "start", "--enable", "x"],
     ["codex", "-c", 'model="remote-control"', "remote-control", "--enable", "x"]),
    (["codex", "remote-control", "--config", 'model="start"', "start"],
     ["codex", "remote-control", "--config", 'model="start"']),
    (["codex", "--model", "remote-control"], None),
    (["codex", "--profile", "remote-control"], None),
    (["codex", "--", "remote-control"], None),
    (["codex", "exec", "remote-control"], None),
    (["codex", "please inspect remote-control"], None),
    (["claude", "remote-control"], None),
])
def test_remote_control_detection_respects_subcommands_and_option_values(command, expected):
    assert codex_remote_command(command) == expected


@pytest.mark.parametrize("action", ["stop", "status", "pair", "list", "unpair", "arbitrary prompt"])
def test_service_management_is_not_an_agent_launch(action):
    with pytest.raises(ValueError, match="management commands separately"):
        codex_remote_command(["codex", "remote-control", action])


def test_config_override_preserves_full_unicode_and_controls_as_toml():
    tomllib = pytest.importorskip("tomllib")
    text = PRELUDE + "\x00\b\t\r\f"
    assert tomllib.loads(config_override("developer_instructions", text)) == {
        "developer_instructions": text,
    }


@pytest.fixture
def session_dir(tmp_path, monkeypatch):
    path = tmp_path / "sessions"
    monkeypatch.setattr("sucoder.session._session_dir", lambda: path)
    return path


def _manager(tmp_path, monkeypatch, *, remote=False):
    settings = MirrorSettings(name="sample", canonical_repo=tmp_path,
                              mirror_name="sample", branch_prefixes=BranchPrefixes())
    if remote:
        settings.remote = RemoteConfig(gateway="cluster", slurm=SlurmConfig("p", "a", confined=True))
    config = Config(human_user="human", mirror_root=tmp_path, mirrors={"sample": settings})
    executor = SimpleNamespace(dry_run=False, run_agent=Mock(return_value=SimpleNamespace(returncode=0)))
    manager = MirrorManager(config, executor, logging.getLogger("remote-control-test"))
    monkeypatch.setattr(manager, "_ensure_mirror_exists", lambda ctx: tmp_path)
    monkeypatch.setattr(manager, "_compose_context_prelude", lambda ctx: PRELUDE)
    for method in ("_maybe_run_tool_preflight", "_maybe_run_poetry_auto_install",
                   "_maybe_suggest_mcp_servers", "_report_agent_binary",
                   "_auto_commit_agent_skills", "_maybe_run_audit"):
        monkeypatch.setattr(manager, method, lambda *a, **kw: None)
    return manager, manager.context_for("sample")


def _overrides(command):
    return dict(arg.split("=", 1) for i, arg in enumerate(command)
                if i and command[i - 1] == "-c" and "=" in arg)


@pytest.mark.parametrize("start", [False, True])
@pytest.mark.parametrize("remote", [False, True])
def test_launch_adapts_prelude_model_and_permissions(tmp_path, monkeypatch, start, remote):
    manager, ctx = _manager(tmp_path, monkeypatch, remote=remote)
    ctx.settings.agent_launcher = AgentLauncher(command=["claude"], model="configured-model")
    confined = Mock(return_value=0)
    monkeypatch.setattr(manager, "_launch_confined", confined)
    command = ["codex", "remote-control"] + (["start"] if start else [])
    manager.launch_agent(ctx, sync=False, command_override=command, model_override="chosen-model",
                         supports_inline_prompt=False, detached=True)
    if remote:
        args, kwargs = confined.call_args
        launched = args[1]
        encoded = kwargs["remote_prelude_text"]
        assert kwargs["prelude_sentinel"] in launched
        assert "developer_instructions=" not in " ".join(launched)
    else:
        launched = manager.executor.run_agent.call_args.args[0]
        encoded = "developer_instructions=" + _overrides(launched)["developer_instructions"]
    values = _overrides(launched)
    assert json.loads(encoded.split("=", 1)[1]) == PRELUDE
    assert json.loads(values["model"]) == "chosen-model"
    assert json.loads(values["sandbox_mode"]) == "danger-full-access"
    assert json.loads(values["approval_policy"]) == "never"
    assert "start" not in launched and "--model" not in launched and "--sandbox" not in launched
    assert PRELUDE not in launched
    assert manager._service_launch == {"command": ["codex", "remote-control"], "model": "chosen-model"}


@pytest.mark.parametrize("needs_yolo", [False, True])
def test_service_flags_preserve_explicit_restrictions_and_full_prelude(tmp_path, monkeypatch, needs_yolo):
    manager, ctx = _manager(tmp_path, monkeypatch)
    ctx.settings.agent_launcher = AgentLauncher(
        command=["codex", "remote-control", "-c", 'sandbox_mode="read-only"',
                 "-c", 'approval_policy="on-request"', "-c", 'developer_instructions="earlier"'],
        needs_yolo=needs_yolo,
    )
    manager.launch_agent(ctx, sync=False, detached=True)
    values = _overrides(manager.executor.run_agent.call_args.args[0])
    assert json.loads(values["sandbox_mode"]) == "read-only"
    assert json.loads(values["approval_policy"]) == "on-request"
    assert json.loads(values["developer_instructions"]) == PRELUDE


def test_invalid_service_command_fails_before_mirror_setup(tmp_path, monkeypatch):
    manager, ctx = _manager(tmp_path, monkeypatch)
    setup = Mock()
    monkeypatch.setattr(manager, "_ensure_mirror_exists", setup)
    with pytest.raises(MirrorError, match="management commands separately"):
        manager.launch_agent(ctx, command_override=["codex", "remote-control", "stop"])
    setup.assert_not_called()


@pytest.mark.skipif(not shutil.which("tmux"), reason="tmux not installed")
def test_staged_prelude_survives_real_tmux_and_shell_quoting(tmp_path, monkeypatch, session_dir):
    """Exercise the same staging, nested shells, and pane marker used remotely."""
    manager, ctx = _manager(tmp_path, monkeypatch)
    fake = tmp_path / "codex"
    output = tmp_path / "argv.json"
    fake.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"open({str(output)!r}, 'w').write(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    fake.chmod(0o700)
    # HOME is scoped to these test subprocesses, not changed on the agent host.
    env = dict(os.environ, HOME=str(tmp_path), TMUX="")

    def run_agent(args, **kwargs):
        return subprocess.run(args, input=kwargs.get("input"), capture_output=True, text=True, env=env)

    manager.executor.run_agent = run_agent
    manager._service_launch = {"command": [str(fake), "remote-control"], "model": None}
    command = manager._build_remote_agent_cmd_str(
        ctx, [str(fake), "remote-control", "-c", "__PRELUDE__"],
        remote_prelude_text=config_override("developer_instructions", PRELUDE),
        prelude_sentinel="__PRELUDE__",
    )
    # A second launch must not overwrite instructions a queued launch will read.
    manager._build_remote_agent_cmd_str(
        ctx, [str(fake), "remote-control", "-c", "__PRELUDE__"],
        remote_prelude_text=config_override("developer_instructions", "other launch"),
        prelude_sentinel="__PRELUDE__",
    )
    assert len(list((tmp_path / ".cache/sucoder").glob("prelude-*.txt"))) == 2
    sock = "sucoder-rc-" + tmp_path.name
    base = ["tmux", "-L", sock]
    try:
        subprocess.run(base + ["new-session", "-d", "-s", "sucoder-test", "-c", str(tmp_path),
                               "bash -c " + shlex.quote(command)], check=True, env=env)
        for _ in range(100):
            if output.exists() and output.stat().st_size:
                break
            time.sleep(0.02)
        args = json.loads(output.read_text())
        assert args[:2] == ["remote-control", "-c"]
        assert json.loads(args[2].split("=", 1)[1]) == PRELUDE
        assert not (tmp_path / "INJECTED").exists()
        mode = subprocess.run(["bash", "-c", MODE_PROBE_SH, "_", "sucoder-test", sock],
                              capture_output=True, text=True, check=True)
        assert mode.stdout.strip() == CODEX_REMOTE_CONTROL
        launch = subprocess.run(["bash", "-c", SERVICE_LAUNCH_PROBE_SH, "_", "sucoder-test", sock],
                                capture_output=True, text=True, check=True)
        assert service_launch(launch.stdout) == manager._service_launch
        status = subprocess.run(["bash", "-c", SESSION_PANE_PROBE_SH, "_", "sucoder-test", sock],
                                capture_output=True, text=True, check=True)
        assert parse_pane_status(status.stdout).agent_mode == CODEX_REMOTE_CONTROL
    finally:
        subprocess.run(base + ["kill-server"], capture_output=True)


@pytest.mark.parametrize("prior_launch", [None, LAUNCH])
def test_confined_reuse_keeps_existing_mode(tmp_path, monkeypatch, prior_launch):
    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    session = RemoteSession("sample", slurm_job_id=42, service_launch=prior_launch)
    session.save()
    manager._service_launch = LAUNCH if prior_launch is None else None
    calls = []
    manager.executor.run_agent = _confined_responder(calls, live_state="RUNNING")
    manager._launch_confined(ctx, ["codex", "remote-control"], remote_prelude_text=None,
                             prelude_sentinel="__X__", env=None, detached=True)
    assert RemoteSession.load("sample").service_launch == prior_launch
    assert not any(c["args"][0] == "sbatch" for c in calls)


def test_confined_submission_records_service_before_polling(tmp_path, monkeypatch):
    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    manager._service_launch = LAUNCH
    calls = []
    manager.executor.run_agent = _confined_responder(calls)

    def poll(job_id):
        session = RemoteSession.load("sample")
        assert session.slurm_job_id == job_id and session.service_launch == LAUNCH
        raise MirrorError("still pending")

    monkeypatch.setattr(manager, "_poll_confined_node", poll)
    with pytest.raises(MirrorError, match="still pending"):
        manager._launch_confined(ctx, ["codex", "remote-control"], remote_prelude_text=None,
                                 prelude_sentinel="__X__", env=None, detached=True)
    staged = [c["input"] for c in calls if c["input"]]
    assert any("@sucoder-agent-mode" in body for body in staged)


@pytest.mark.parametrize("created, existing", [(True, None), (False, None), (False, LAUNCH)])
def test_remote_launch_records_creation_or_actual_existing_mode(
    tmp_path, monkeypatch, session_dir, created, existing,
):
    manager, ctx = _manager(tmp_path, monkeypatch)
    manager._service_launch = LAUNCH
    manager.executor.run_agent.side_effect = [
        SimpleNamespace(returncode=0, stdout="SUCODER_CREATED\n" if created else ""),
        SimpleNamespace(returncode=0, stdout=json.dumps(existing) if existing else ""),
    ]
    assert manager._launch_remote_service(ctx, "sucoder-sample", "cat", tmp_path, detached=True) == 0
    assert RemoteSession.load("sample").service_launch == (LAUNCH if created else existing)
    assert manager.executor.run_agent.call_count == (1 if created else 2)


@pytest.mark.skipif(not shutil.which("tmux"), reason="tmux not installed")
def test_detached_service_reuse_works_without_a_tty(tmp_path, monkeypatch, session_dir):
    manager, ctx = _manager(tmp_path, monkeypatch)
    sock = "sucoder-reuse-" + tmp_path.name
    real_tmux = shutil.which("tmux")
    wrapper = tmp_path / "tmux"
    wrapper.write_text(f'#!/bin/sh\nexec {shlex.quote(real_tmux)} -L {shlex.quote(sock)} "$@"\n')
    wrapper.chmod(0o700)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", TMUX="")

    def run_agent(args, **kwargs):
        return subprocess.run(args, cwd=kwargs.get("cwd"), env=env, text=True, capture_output=True)

    manager.executor.run_agent = run_agent
    manager._service_launch = LAUNCH
    command = service_marker(LAUNCH) + "cat"
    try:
        assert manager._launch_remote_service(ctx, "sucoder-sample", command, tmp_path, detached=True) == 0
        assert RemoteSession.load("sample").service_launch == LAUNCH
        for _ in range(100):
            probe = run_agent(["bash", "-c", SERVICE_LAUNCH_PROBE_SH, "_", "sucoder-sample", ""])
            if service_launch(probe.stdout) == LAUNCH:
                break
            time.sleep(0.02)
        assert service_launch(probe.stdout) == LAUNCH
        manager._service_launch = {**LAUNCH, "model": "different-request"}
        assert manager._launch_remote_service(ctx, "sucoder-sample", "false", tmp_path, detached=True) == 0
        assert RemoteSession.load("sample").service_launch == LAUNCH
    finally:
        subprocess.run([real_tmux, "-L", sock, "kill-server"], capture_output=True)


def test_service_session_roundtrip_and_legacy_defaults(session_dir):
    RemoteSession("sample", target_name="savio", slurm_job_id=42, service_launch=LAUNCH).save()
    assert RemoteSession.load("sample", "savio").service_launch == LAUNCH
    (session_dir / "old.yaml").write_text("slurm_job_id: 12\n")
    assert RemoteSession.load("old").service_launch is None
    assert RemoteSession.load("missing").service_launch is None


@pytest.mark.parametrize("value", [[], {}, {"command": ["sh", "-c", "bad"]},
                                  {"command": ["codex", "remote-control", "stop"]},
                                  {"command": ["codex", 42]}, {"command": LAUNCH["command"], "model": []}])
def test_invalid_service_record_cannot_be_relaunched(value):
    with pytest.raises(ValueError):
        service_launch(value)


@pytest.mark.parametrize("force", [False, True])
def test_message_refuses_service_and_peek_describes_log(monkeypatch, capsys, force):
    report = _message_report(pane="codex")
    report.groups[0].entries[0].agent_mode = CODEX_REMOTE_CONTROL
    code, trips = _run_message(monkeypatch, report, recipient="M", text="do work", force=force)
    assert code == 1 and trips == []
    assert "connected Codex client" in capsys.readouterr().out
    code, trips = _run_peek(monkeypatch, report, mirror="M", pane="server ready\n")
    assert code == 0 and len(trips) == 1
    out = capsys.readouterr().out
    assert "Remote-control service log" in out and "server ready" in out


def test_broadcast_skips_service_jobs_and_login_panes():
    from sucoder.sessions_report import LoginSession
    job = _job(1, "M", pane="codex")
    job.agent_mode = CODEX_REMOTE_CONTROL
    login = LoginSession("ln003.brc", "sucoder-L", "savio-node", pane="codex",
                         agent_mode=CODEX_REMOTE_CONTROL)
    chosen, skipped = plan_recipients(_report(job, _job(2, "N"), logins=[login]),
                                      everyone=True, force=True, **PLAN)
    assert [r.session for r in chosen] == ["sucoder-N"]
    assert len(skipped) == 2 and all("remote-control service" in s for s in skipped)
    with pytest.raises(ValueError, match="Remote-control"):
        send_keys_command(Recipient("service", "host", "sess", agent_mode=CODEX_REMOTE_CONTROL), "hello")


def test_live_report_metadata_without_launcher_records():
    names, statuses = parse_login_session_status(
        "sucoder-M\tcodex\tcodex-remote-control\nsucoder-old\tclaude\nsucoder-unknown\t\tunknown\n"
    )
    assert set(names) == {"sucoder-M", "sucoder-old", "sucoder-unknown"}
    assert statuses["sucoder-M"].agent_mode == CODEX_REMOTE_CONTROL
    assert statuses["sucoder-old"].agent_mode == TERMINAL
    assert statuses["sucoder-unknown"].command is None
    entry = _job(1, "M", pane="codex")
    entry.agent_mode = CODEX_REMOTE_CONTROL
    report = _report(entry)
    assert "remote-control service" in render_report(report)
    entry.pane = "bash"
    assert "service exited" in render_report(report)


def test_batched_job_probe_preserves_service_mode_and_snapshot_age(monkeypatch):
    report = _message_report(pane=None)
    capture = Mock(return_value=SimpleNamespace(
        returncode=0, stdout="41\tcodex\tcodex-remote-control\nwip41\t2 minutes ago\n",
    ))
    monkeypatch.setattr(cli, "_capture_over_tunnel", capture)
    cli._probe_session_panes(
        report, _SnapshotsHarness(times=("12-00:00:00",)).config,
        {"hpc.brc": ["t0"]}, logging.getLogger("test"), False,
        hosts={"hpc.brc": ("ln001.brc", object())},
    )
    entry = report.groups[0].entries[0]
    assert entry.pane == "codex" and entry.agent_mode == CODEX_REMOTE_CONTROL
    assert entry.wip == "2 minutes ago"
    assert capture.call_count == 1


@pytest.mark.parametrize("launch", [None, LAUNCH])
def test_relaunch_preserves_service_command_and_model(session_dir, monkeypatch, launch):
    RemoteSession("sample", slurm_job_id=42, compute_node="old", service_launch=launch).save()
    manager = Mock()

    def start(ctx, **kwargs):
        current = RemoteSession.load("sample")
        assert current.slurm_job_id is None and current.compute_node is None
        assert kwargs == ({"sync": False, "detached": True, "command_override": LAUNCH["command"],
                           "model_override": LAUNCH["model"]} if launch else {"sync": False, "detached": True})
        current.slurm_job_id = 43
        current.save()
        return 0

    manager.launch_agent.side_effect = start
    monkeypatch.setattr(cli, "_build_manager_for_mirror", lambda *a, **kw: manager)
    ctx = SimpleNamespace(obj={})
    assert cli._relaunch_session("sample", object(), ctx, logging.getLogger("test"), False) == 43


def test_invalid_renewal_record_keeps_original_job(session_dir, monkeypatch):
    RemoteSession("sample", slurm_job_id=42, service_launch={"command": ["sh"]}).save()
    build = Mock()
    monkeypatch.setattr(cli, "_build_manager_for_mirror", build)
    assert cli._relaunch_session("sample", object(), None, logging.getLogger("test"), False) is None
    assert RemoteSession.load("sample").slurm_job_id == 42
    build.assert_not_called()


@pytest.mark.parametrize("confined", [False, True])
@pytest.mark.parametrize("metadata_ok", [False, True])
def test_renew_heals_live_metadata_and_keeps_checkpoint_file_only(
    tmp_path, monkeypatch, session_dir, capsys, confined, metadata_ok,
):
    import sucoder.renew as renew_module
    manager, context = _manager(tmp_path, monkeypatch, remote=True)
    context.settings.remote.slurm.confined = confined
    RemoteSession("sample", slurm_job_id=42, compute_node="node1", login_node="login1").save()
    monkeypatch.setattr(cli, "_connect_with_retry", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_reconcile_login_node", lambda *a, **kw: None)
    calls = []

    def capture(control, host, command, **kw):
        calls.append((host, command))
        if command.startswith("squeue"):
            return SimpleNamespace(returncode=0, stdout="RUNNING|01:00:00\n", stderr="")
        if "@sucoder-service-launch" in command:
            return SimpleNamespace(returncode=0 if metadata_ok else 255, stdout=json.dumps(LAUNCH), stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cli, "_run_remote_capture", capture)

    def loop(job_id, settings, *, probe, request_checkpoint, **kwargs):
        assert probe(job_id).ok is metadata_ok
        if metadata_ok:
            assert RemoteSession.load("sample").service_launch == LAUNCH
            assert probe(job_id).ok
            request_checkpoint()

    monkeypatch.setattr(renew_module, "run_renew_loop", loop)
    ctx = SimpleNamespace(obj={"config": manager.config}, params={})
    cli.renew(ctx, mirror="sample", drain_minutes=20, poll_interval=0,
              checkpoint_grace=0, once=True, verbose=False, dry_run=False)
    mode_calls = [(host, cmd) for host, cmd in calls if "@sucoder-service-launch" in cmd]
    assert len(mode_calls) == 1
    assert mode_calls[0][0] == ("login1" if confined else "node1")
    assert ("srun --jobid=42 --overlap" in mode_calls[0][1]) is confined
    assert not any("send-keys" in cmd for _, cmd in calls)
    if metadata_ok:
        assert "renew-requested" in calls[-1][1]
        assert "active chats are not automatically resumed" in capsys.readouterr().out
