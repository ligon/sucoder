"""Claude launched with ``--session-id``, recorded for the reply reader
(docs/reply-from-transcript.org, identification step 1)."""

import re
import uuid
from types import SimpleNamespace

import pytest
import yaml

from sucoder.config import AgentLauncher
from sucoder.mirror import _claude_session_chosen
from sucoder.session import RemoteSession

from tests.test_codex_state import _state_manager
from tests.test_mirror import _confined_manager, _confined_responder


def _session_id_in(argv):
    at = argv.index("--session-id")
    value = argv[at + 1]
    assert uuid.UUID(value).version == 4
    return value


def _claude_launch(tmp_path, monkeypatch, command):
    manager, ctx, confined = _state_manager(tmp_path, monkeypatch, enabled=False)
    ctx.settings.agent_launcher = AgentLauncher(command=["claude"])
    manager.launch_agent(ctx, sync=False, command_override=list(command), detached=True)
    return manager, confined.call_args.args[1]


def test_a_sucoder_launched_claude_gets_a_session_id(tmp_path, monkeypatch):
    manager, argv = _claude_launch(tmp_path, monkeypatch, ["claude"])
    assert argv[0] == "claude" and argv.count("--session-id") == 1
    assert _session_id_in(argv) == manager._claude_session_id


@pytest.mark.parametrize("flags", [
    ["--session-id", "11111111-2222-4333-8444-555555555555"], ["--session-id=11111111-2222-4333-8444-555555555555"],
    ["--resume"], ["--resume=abc"], ["-r", "abc"], ["--continue"], ["-c"],
])
def test_a_command_that_already_picks_a_session_gets_none(tmp_path, monkeypatch, flags):
    manager, argv = _claude_launch(tmp_path, monkeypatch, ["claude", *flags])
    assert manager._claude_session_id is None
    assert argv.count("--session-id") == (1 if flags[0] == "--session-id" else 0)
    assert _claude_session_chosen(["claude", *flags])


def test_other_harnesses_get_no_session_id(tmp_path, monkeypatch):
    manager, argv = _claude_launch(tmp_path, monkeypatch, ["gemini"])
    assert "--session-id" not in argv and manager._claude_session_id is None
    assert not _claude_session_chosen(["claude", "--model", "opus"])


def test_a_confined_launch_records_the_session_id_beside_the_job(tmp_path, monkeypatch):
    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    ctx.settings.agent_launcher = AgentLauncher(command=["claude"])
    ctx.settings.remote.slurm.wip_snapshot_minutes = 0
    calls = []
    manager.executor.run_agent = _confined_responder(calls, sbatch_out="9")
    manager.launch_agent(ctx, sync=False, detached=True)
    chosen = manager._claude_session_id
    assert re.fullmatch(r"[0-9a-f-]{36}", chosen)
    staged = " ".join(str(c["input"] or "") + " ".join(c["args"]) for c in calls)
    assert f"--session-id {chosen}" in staged
    record = RemoteSession.load("sample")
    assert record.slurm_job_id == 9 and record.claude_session_id == chosen and record.harness == "claude"


@pytest.mark.parametrize("exists_rc, recorded", [(1, True), (0, False)])
def test_an_unconfined_launch_records_only_a_session_it_creates(tmp_path, monkeypatch, exists_rc, recorded):
    manager, ctx, confined = _state_manager(tmp_path, monkeypatch, enabled=False)
    ctx.settings.remote.slurm.confined = False
    ctx.settings.agent_launcher = AgentLauncher(command=["claude"])
    monkeypatch.setattr(manager, "_prepare_local_tier", lambda ctx, root: (tmp_path, "export SUCODER_LOCAL_ROOT=/local"))
    manager.executor.run_agent.side_effect = lambda args, **kw: SimpleNamespace(
        returncode=exists_rc if list(args)[:2] == ["tmux", "has-session"] else 0, stdout="", stderr="")
    RemoteSession("sample", slurm_job_id=41, claude_session_id="old-one").save()
    manager.launch_agent(ctx, sync=False, detached=True)
    launched = manager.executor.run_agent.call_args.args[0]
    assert launched[:2] == ["tmux", "new-session"] and f"--session-id {manager._claude_session_id}" in launched[-1]
    record = RemoteSession.load("sample")
    assert record.slurm_job_id == 41
    assert record.claude_session_id == (manager._claude_session_id if recorded else "old-one")


def test_the_new_fields_round_trip_and_old_records_still_load(tmp_path, monkeypatch):
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path)
    RemoteSession("M", target_name="t0", slurm_job_id=41, harness="claude",
                  claude_session_id="83cd9e50-d565-4635-95a2-da82d4b334cd").save()
    back = RemoteSession.load("M", target_name="t0")
    assert back.harness == "claude" and back.claude_session_id == "83cd9e50-d565-4635-95a2-da82d4b334cd"
    # A record written before the fields existed.
    (tmp_path / "Old--t0.yaml").write_text(yaml.safe_dump({"slurm_job_id": 7, "login_node": "ln002"}))
    old = RemoteSession.load("Old", target_name="t0")
    assert old.slurm_job_id == 7 and old.harness is None and old.claude_session_id is None


@pytest.mark.parametrize("command, harness", [(["claude", "--resume"], "claude"), (["codex"], "codex")])
def test_every_unconfined_launch_that_creates_the_session_rewrites_the_identity(
        tmp_path, monkeypatch, command, harness):
    # A record left by an earlier Claude launch: a resumed Claude (no id of
    # ours) or an unmanaged Codex launch must not inherit its session id.
    manager, ctx, confined = _state_manager(tmp_path, monkeypatch, enabled=False)
    ctx.settings.remote.slurm.confined = False
    ctx.settings.agent_launcher = AgentLauncher(command=["claude"])
    monkeypatch.setattr(manager, "_prepare_local_tier", lambda ctx, root: (tmp_path, "export SUCODER_LOCAL_ROOT=/local"))
    manager.executor.run_agent.side_effect = lambda args, **kw: SimpleNamespace(
        returncode=1 if list(args)[:2] == ["tmux", "has-session"] else 0, stdout="", stderr="")
    RemoteSession("sample", slurm_job_id=41, harness="claude", claude_session_id="old-one").save()
    manager.launch_agent(ctx, sync=False, command_override=list(command), detached=True)
    record = RemoteSession.load("sample")
    assert record.harness == harness and record.claude_session_id is None
