"""Managed Codex state, Slice 3d: renewal with acknowledged handoff.

The renewal loop replaces a managed Codex owner with a successor that
continues the exact generation the predecessor's drain acknowledgement
sealed.  Three layers, as for the earlier slices: the pure pieces (the
transfer token a renew-mode acknowledgement carries, the replacement
attempt the session record retains across loop turns, readiness read from
the successor's ``status.json``); the launcher's replacement authorization
(the drained predecessor, and only it, excluded from the confined reuse
and adoption probes; the new job recorded in the attempt, never over the
old provenance); and the relaunch transaction ``sucoder renew`` drives.
"""

from __future__ import annotations

import pytest

from sucoder.codex_state import (
    ReplacementAttempt, TransferToken, classify_readiness, runner_command, transfer_token_from_ack,
    transfer_token_from_exit,
)
from sucoder.session import RemoteSession

RUN = "0123456789abcdef0123456789abcdef"
NEW_RUN = "fedcba9876543210fedcba9876543210"
NONCE = "0123456789abcdef"
GEN = "000013-cafebabe"
SHA = "ab" * 32


def _ack(outcome="sealed", *, mode="renew", cls="sealed", stopped=True, generation=GEN, **extra):
    record = None if generation is None else {"name": generation, "class": cls, "manifest_sha256": SHA,
                                              "path": f"/durable/generations/{generation}"}
    ack = {"nonce": NONCE, "mode": mode, "outcome": outcome, "generation": record, "exit": 0,
           "child": {"pid": 4321, "starttime": 123456}, "stopped": stopped, "terminal": "none",
           "scope": "ab" * 32, "run": RUN}
    ack.update(extra)
    return ack


def _token():
    return TransferToken(nonce=NONCE, run=RUN, generation=GEN, manifest_sha256=SHA, exit=0,
                         child={"pid": 4321, "starttime": 123456})


# ----------------------------------------------------------------- the runner command

def test_runner_command_passes_unattended_after_the_selection():
    kwargs = dict(durable_root="/g/.sucoder-state", checkpoint_minutes=10, checkpoint_timeout_seconds=30)
    argv = runner_command("/h/r.py", "ab" * 32, ["codex"], restore=f"{RUN}/{GEN}", unattended=True, **kwargs)
    assert argv[-5:] == ["--restore", f"{RUN}/{GEN}", "--unattended", "--", "codex"]
    assert "--unattended" not in runner_command("/h/r.py", "ab" * 32, ["codex"], **kwargs)


# ----------------------------------------------------------------- the transfer token

def test_token_from_a_sealed_stopped_renew_acknowledgement():
    token = transfer_token_from_ack(_ack())
    assert token == _token()
    assert token.selector == f"{RUN}/{GEN}"
    assert TransferToken.from_json(token.as_json()) == token


@pytest.mark.parametrize("ack, why", [
    (_ack("warm_only", cls="warm"), "not sealed"),
    (_ack("crash", cls="crash"), "not sealed"),
    (_ack("failed", generation=None, reason="replica t1 is shrunk"), "not sealed"),
    (_ack(stopped=False), "not proven stopped"),
    (_ack(mode="release"), "mode 'release'"),
    (_ack(cls="warm"), "class is 'warm'"),
    (_ack(generation="latest"), "not NNNNNN-hhhhhhhh"),
    (_ack(run="../etc"), "not a 32-hex run id"),
    (_ack(nonce="XYZ"), "not 8-64 lowercase hex"),
])
def test_token_refuses_everything_but_a_sealed_stopped_acknowledgement(ack, why):
    with pytest.raises(ValueError, match=why):
        transfer_token_from_ack(ack)


def test_token_refuses_a_bad_manifest_hash():
    ack = _ack()
    ack["generation"]["manifest_sha256"] = "ab" * 31
    with pytest.raises(ValueError, match="manifest sha256"):
        transfer_token_from_ack(ack)


def test_token_from_a_clean_exit_with_a_sealed_final_generation():
    status = {"state": "exited", "exit": 0, "exited_at": 20.0, "run": RUN,
              "last_generation": {"name": GEN, "class": "sealed", "manifest_sha256": SHA, "published_at": 20.0}}
    token = transfer_token_from_exit(status, nonce=NONCE)
    assert (token.run, token.generation, token.manifest_sha256, token.exit, token.child) == (RUN, GEN, SHA, 0, None)
    with pytest.raises(ValueError, match="not exited"):
        transfer_token_from_exit(dict(status, state="placed"), nonce=NONCE)
    with pytest.raises(ValueError, match="not sealed"):
        transfer_token_from_exit(dict(status, last_generation=dict(status["last_generation"], **{"class": "crash"})),
                                 nonce=NONCE)


@pytest.mark.parametrize("data", [None, "token", {"nonce": NONCE}, {**_token().as_json(), "exit": "0"},
                                  {**_token().as_json(), "child": "4321"}])
def test_token_from_json_validates(data):
    with pytest.raises(ValueError):
        TransferToken.from_json(data)


# ----------------------------------------------------------------- the replacement attempt

def test_attempt_round_trips_and_validates():
    attempt = ReplacementAttempt(attempt="00ff" * 4, old_job=4242, old_node="n0033.savio4", old_run=RUN, token=_token(),
                                 created_at="2026-10-16T15:00:00+00:00")
    data = attempt.as_json()
    assert data["state"] == "drained" and data["new_job"] is None and data["failures"] == []
    assert ReplacementAttempt.from_json(data) == attempt
    attempt.state, attempt.new_job, attempt.new_node = "submitted", 4343, "n0034.savio4"
    attempt.failures.append("tmux did not come up")
    assert ReplacementAttempt.from_json(attempt.as_json()) == attempt
    for bad in ({**data, "state": "done"}, {**data, "old_job": None}, {**data, "new_job": "4343"},
                {**data, "attempt": "x"}, {**data, "token": None}, {**data, "failures": "none"}, "attempt", None):
        with pytest.raises(ValueError):
            ReplacementAttempt.from_json(bad)


def test_session_carries_the_managed_owner_and_the_attempt_and_loads_legacy_files_blank(tmp_path, monkeypatch):
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path / "sessions")
    session = RemoteSession(mirror_name="sample", target_name="savio", slurm_job_id=4242, compute_node="n0033.savio4")
    session.managed_owner = {"command": ["codex", "--model", "gpt-6-astra"], "model": None, "scope": "ab" * 32,
                             "durable_root": "/g/.sucoder-state"}
    attempt = ReplacementAttempt(attempt="00ff" * 4, old_job=4242, old_node="n0033.savio4", old_run=RUN, token=_token())
    session.replacement = attempt.as_json()
    session.save()
    loaded = RemoteSession.load("sample", target_name="savio")
    assert loaded.managed_owner == session.managed_owner
    assert ReplacementAttempt.from_json(loaded.replacement) == attempt
    # A file written before these fields existed loads them as absent.
    (tmp_path / "sessions" / "old--savio.yaml").write_text("slurm_job_id: 7\ncompute_node: n1\nforwards: []\n")
    legacy = RemoteSession.load("old", target_name="savio")
    assert legacy.slurm_job_id == 7 and legacy.managed_owner is None and legacy.replacement is None


# ----------------------------------------------------------------- readiness

def _placed(origin_run=RUN, origin_gen=GEN, origin_sha=SHA, **extra):
    status = {"state": "placed", "mode": "restore", "run": NEW_RUN, "reused_run": False,
              "child_pid": 4321, "child_starttime": 123456,
              "restored_from": {"run": origin_run, "generation": origin_gen, "class": "sealed",
                                "manifest_sha256": origin_sha, "threads": 2}}
    status.update(extra)
    return status


def test_readiness_is_placed_on_exactly_the_token_generation():
    ready = classify_readiness(_placed(), _token())
    assert ready.ready and ready.outcome == "ready" and NEW_RUN in ready.reason


@pytest.mark.parametrize("status, outcome, why", [
    (None, "pending", "no managed run recorded yet"),
    ({"state": "starting"}, "pending", "state starting"),
    ({"state": "restoring", "restore": f"{RUN}/{GEN}"}, "pending", "state restoring"),
    ({"state": "launched"}, "pending", "state launched"),
    ({"state": "refused", "reason": "bwrap is not installed"}, "failed", "state refused: bwrap is not installed"),
    ({"state": "blocked", "reason": "the durable side did not answer"}, "failed", "state blocked"),
    ({"state": "misplaced"}, "failed", "state misplaced"),
    ({"state": "exited", "exit": 1}, "failed", "state exited"),
    ({"state": "drained"}, "failed", "state drained"),
    ({"mode": "fresh"}, "failed", "records no state"),
    (_placed(restore_ignored=f"{RUN}/{GEN}", reused_run=True), "failed", "reused it in place"),
    (_placed(reused_run=True), "failed", "reused it in place"),
    (_placed(state="launched", reused_run=True), "failed", "reused it in place"),
    ({"state": "placed", "mode": "fresh", "restored_from": None}, "failed", "not restored from"),
    (_placed(origin_gen="000012-00000000"), "failed", "restored from"),
    (_placed(origin_sha="cd" * 32), "failed", "restored from"),
    (_placed(origin_run=NEW_RUN), "failed", "restored from"),
])
def test_readiness_pending_and_failed_states(status, outcome, why):
    result = classify_readiness(status, _token())
    assert result.outcome == outcome and why in result.reason and not result.ready


# ----------------------------------------------------------------- the token a drain report yields

def _report(**extra):
    from sucoder.codex_state import CodexStateDrain
    fields = dict(job_id=4242, scope="ab" * 32, mode="renew", nonce=NONCE, commands=[], outcome="drained: sealed",
                  run=RUN, requested=True, ack=_ack())
    fields.update(extra)
    return CodexStateDrain(**fields)


def test_token_from_a_drain_report():
    from sucoder.codex_state import transfer_token_from_drain
    assert transfer_token_from_drain(_report()) == _token()
    exited = _report(outcome=f"no managed owner to drain: run {RUN} exited (exit 0) and its final generation {GEN} is sealed",
                     requested=False, ack=None, state="exited", exit=0, exited_at=20.0,
                     last_durable={"name": GEN, "class": "sealed", "manifest_sha256": SHA, "published_at": 21.0})
    assert transfer_token_from_drain(exited).selector == f"{RUN}/{GEN}"
    for report, why in [
        (_report(ack=None, outcome="drain unacknowledged after 60 s; last durable generation 000012-0a1b2c3d"), "unacknowledged"),
        (_report(ack=_ack("warm_only", cls="warm")), "not sealed"),
        (_report(ack=None, requested=False, state="released", outcome="no managed owner to drain: run x is released"), "released"),
        (_report(ack=None, requested=False, state="exited", outcome="exited; crash",
                 last_durable={"name": GEN, "class": "crash", "manifest_sha256": SHA}), "not sealed"),
        (_report(ack=None, run=None, requested=False, outcome="no managed run recorded"), "no managed run"),
    ]:
        with pytest.raises(ValueError, match=why):
            transfer_token_from_drain(report)


# ----------------------------------------------------------------- the launcher's replacement authorization

def _replacing_session(tmp_path, *, job=4242, attempt_state="drained"):
    """The session a renewal has prepared: the predecessor as the job, the attempt holding the token."""
    from tests.test_mirror import _confined_manager  # noqa: F401  (the session dir is patched by the manager builder)
    session = RemoteSession(mirror_name="sample", target_name=None, slurm_job_id=job, compute_node="n0033.savio4")
    session.managed_owner = {"command": ["codex"], "model": None, "scope": "ab" * 32, "durable_root": "/g/.sucoder-state"}
    session.replacement = ReplacementAttempt(attempt="00ff" * 4, old_job=job, old_node="n0033.savio4", old_run=RUN,
                                             token=_token(), state=attempt_state).as_json()
    session.save()
    return session


def test_launch_confined_replaces_only_the_drained_predecessor(tmp_path, monkeypatch):
    """The review's P1: the predecessor must not be rediscovered and adopted
    as "the" session.  It is excluded from both probes, the new job must
    differ, and it is recorded in the attempt, not over the old allocation."""
    from tests.test_mirror import _THIS_TARGET, _confined_manager, _confined_responder
    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    _replacing_session(tmp_path)
    calls = []
    manager.executor.run_agent = _confined_responder(calls, live_state="RUNNING", by_name=f"4242|{_THIS_TARGET}")
    rc = manager._launch_confined(ctx, ["codex"], remote_prelude_text=None, prelude_sentinel="__X__", env=None,
                                  detached=True, replace_job=4242)
    assert rc == 0
    assert any(c["args"][0] == "sbatch" for c in calls), "the predecessor is not a live session to reuse"
    assert not any("attach-session" in c["args"] for c in calls)
    assert not any(c["args"][:2] == ["squeue", "--job"] and "4242" in c["args"] for c in calls), (
        "the id-keyed probe is skipped for the predecessor, not answered")
    session = RemoteSession.load("sample", target_name=None)
    assert session.slurm_job_id == 4242 and session.compute_node == "n0033.savio4", "old provenance retained"
    attempt = ReplacementAttempt.from_json(session.replacement)
    assert (attempt.state, attempt.new_job, attempt.new_node) == ("submitted", 12345, "n0001.savio4")
    assert attempt.old_job == 4242 and attempt.token == _token()


def test_launch_confined_refuses_an_unrelated_live_job_beside_the_predecessor(tmp_path, monkeypatch):
    from sucoder.mirror import MirrorError
    from tests.test_mirror import _THIS_TARGET, _confined_manager, _confined_responder
    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    _replacing_session(tmp_path)
    calls = []
    manager.executor.run_agent = _confined_responder(
        calls, live_state="RUNNING", by_name=f"4300|{_THIS_TARGET}\n4242|{_THIS_TARGET}")
    with pytest.raises(MirrorError, match="another live SLURM job 4300 .* beside the drained job 4242"):
        manager._launch_confined(ctx, ["codex"], remote_prelude_text=None, prelude_sentinel="__X__", env=None,
                                 detached=True, replace_job=4242)
    assert not any(c["args"][0] == "sbatch" for c in calls)
    session = RemoteSession.load("sample", target_name=None)
    assert session.slurm_job_id == 4242 and ReplacementAttempt.from_json(session.replacement).new_job is None


def test_launch_confined_still_allocates_when_the_other_named_job_is_on_another_target(tmp_path, monkeypatch):
    from tests.test_mirror import _THIS_TARGET, _confined_manager, _confined_responder
    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    _replacing_session(tmp_path)
    calls = []
    manager.executor.run_agent = _confined_responder(
        calls, live_state="RUNNING", by_name=f"4300|RUNNING|savio3|fc_jevons|normal\n4242|{_THIS_TARGET}")
    assert manager._launch_confined(ctx, ["codex"], remote_prelude_text=None, prelude_sentinel="__X__", env=None,
                                    detached=True, replace_job=4242) == 0
    assert any(c["args"][0] == "sbatch" for c in calls)


def test_launch_confined_refuses_a_submission_answered_with_the_predecessor_id(tmp_path, monkeypatch):
    from sucoder.mirror import MirrorError
    from tests.test_mirror import _confined_manager, _confined_responder
    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    _replacing_session(tmp_path)
    calls = []
    manager.executor.run_agent = _confined_responder(calls, sbatch_out="4242", by_name="")
    with pytest.raises(MirrorError, match="being replaced"):
        manager._launch_confined(ctx, ["codex"], remote_prelude_text=None, prelude_sentinel="__X__", env=None,
                                 detached=True, replace_job=4242)
    assert ReplacementAttempt.from_json(RemoteSession.load("sample", target_name=None).replacement).new_job is None


def test_launch_confined_without_a_record_records_the_job_as_any_launch(tmp_path, monkeypatch):
    """``replace_job`` without an attempt record (API misuse, or a record
    removed by hand) still excludes the predecessor but records the job the
    ordinary way, so nothing is ever submitted and forgotten."""
    from tests.test_mirror import _THIS_TARGET, _confined_manager, _confined_responder
    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    session = RemoteSession(mirror_name="sample", target_name=None, slurm_job_id=4242)
    session.save()
    calls = []
    manager.executor.run_agent = _confined_responder(calls, live_state="RUNNING", by_name=f"4242|{_THIS_TARGET}")
    manager._managed_owner = {"command": ["codex"], "model": None, "scope": "ab" * 32, "durable_root": "/g"}
    assert manager._launch_confined(ctx, ["codex"], remote_prelude_text=None, prelude_sentinel="__X__", env=None,
                                    detached=True, replace_job=4242) == 0
    loaded = RemoteSession.load("sample", target_name=None)
    assert loaded.slurm_job_id == 12345 and loaded.managed_owner == manager._managed_owner


def test_selection_probe_skips_the_predecessor_and_refuses_any_other_live_job(tmp_path, monkeypatch):
    from sucoder.mirror import MirrorError
    from tests.test_mirror import _THIS_TARGET, _confined_manager, _confined_responder
    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    _replacing_session(tmp_path)
    manager.executor.run_agent = _confined_responder([], live_state="RUNNING", by_name=f"4242|{_THIS_TARGET}")
    manager._refuse_selection_over_live_session(ctx, f"--codex-state-from {RUN}/{GEN}", replace_job=4242)
    with pytest.raises(MirrorError, match="already has a running session \\(job 4242\\)"):
        manager._refuse_selection_over_live_session(ctx, f"--codex-state-from {RUN}/{GEN}")
    manager.executor.run_agent = _confined_responder([], live_state="RUNNING", by_name=f"4300|{_THIS_TARGET}")
    with pytest.raises(MirrorError, match="job 4300"):
        manager._refuse_selection_over_live_session(ctx, f"--codex-state-from {RUN}/{GEN}", replace_job=4242)


def test_launch_agent_hands_the_runner_unattended_and_the_authorization_to_the_confined_launch(tmp_path, monkeypatch):
    from sucoder.config import AgentLauncher
    from tests.test_codex_state import _state_manager
    manager, ctx, confined = _state_manager(tmp_path, monkeypatch)
    ctx.settings.agent_launcher = AgentLauncher(command=["claude"])
    manager.launch_agent(ctx, sync=False, command_override=["codex", "--model", "gpt-6-astra"], supports_inline_prompt=True,
                         detached=True, codex_state_from=f"{RUN}/{GEN}", codex_state_unattended=True, replace_job=4242)
    launched = confined.call_args.args[1]
    sep = launched.index("--")
    assert launched[sep - 3:sep] == ["--restore", f"{RUN}/{GEN}", "--unattended"]
    assert confined.call_args.kwargs["replace_job"] == 4242
    assert manager._managed_owner == {
        "command": ["codex", "--model", "gpt-6-astra"], "model": None, "remote_control": None,
        "scope": manager._codex_state_launch(ctx).scope, "durable_root": "/global/scratch/u/mirrors/.sucoder-state"}
    manager.launch_agent(ctx, sync=False, command_override=["claude"], supports_inline_prompt=True, detached=True)
    assert manager._managed_owner is None


# ----------------------------------------------------------------- readiness through the manager

def _proc_stat(pid, starttime, state="S"):
    # Field 2 has spaces and a ')' inside it; field 22 follows fields 3-21.
    return f"{pid} (codex (x)) {state} " + " ".join(str(i) for i in range(4, 22)) + f" {starttime} 23\n"


def test_readiness_polls_status_until_placed_on_the_token(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tests.test_codex_state import _state_manager
    from tests.test_codex_state_release import _Node
    manager, ctx, _confined = _state_manager(tmp_path, monkeypatch)
    answers = [None, {"state": "restoring"}, {"state": "launched"}, _placed()]
    monkeypatch.setattr(manager, "_confined_job_state", lambda job: "RUNNING")
    monkeypatch.setattr(manager, "codex_state_status",
                        lambda c, job: SimpleNamespace(status=answers.pop(0), owner={"runner_pid": 1234}))
    node = _Node({"/proc/4321/stat": _proc_stat(4321, 123456), "/proc/1234/stat": _proc_stat(1234, 654321)})
    manager.executor.run_agent.side_effect = node.run_agent
    slept, clock = [], [0.0]

    def sleep(seconds):
        slept.append(seconds)
        clock[0] += seconds
    result = manager.codex_state_readiness(ctx, 4242, _token(), budget_seconds=100.0, poll_seconds=10.0,
                                           sleep=sleep, clock=lambda: clock[0])
    assert result.ready and slept == [10.0, 10.0, 10.0] and answers == []
    assert [argv for argv, _ in node.calls] == [
        ["srun", "--jobid=4242", "--overlap", "cat", "/proc/4321/stat"],
        ["srun", "--jobid=4242", "--overlap", "cat", "/proc/1234/stat"]]


def test_readiness_answers_a_failure_at_once_and_pending_at_the_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from sucoder.mirror import MirrorError
    from tests.test_codex_state import _state_manager
    manager, ctx, _confined = _state_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(manager, "_confined_job_state", lambda job: "RUNNING")
    monkeypatch.setattr(manager, "codex_state_status",
                        lambda c, job: SimpleNamespace(status={"state": "refused", "reason": "bwrap is not installed"}))
    slept = []
    result = manager.codex_state_readiness(ctx, 12345, _token(), budget_seconds=100.0, sleep=slept.append, clock=lambda: 0.0)
    assert (result.outcome, slept) == ("failed", []) and "bwrap" in result.reason

    def unreadable(c, job):
        raise MirrorError("status.json for job 12345 is not readable JSON")
    monkeypatch.setattr(manager, "codex_state_status", unreadable)
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds
    result = manager.codex_state_readiness(ctx, 12345, _token(), budget_seconds=25.0, poll_seconds=10.0,
                                           sleep=sleep, clock=lambda: clock[0])
    assert result.outcome == "pending" and "budget elapsed" in result.reason and clock[0] == 25.0

    monkeypatch.setattr(manager, "_confined_job_state", lambda job: "PENDING")
    result = manager.codex_state_readiness(ctx, 12345, _token(), budget_seconds=0.0, sleep=sleep, clock=lambda: clock[0])
    assert result.outcome == "pending" and "job 12345 is PENDING" in result.reason
    monkeypatch.setattr(manager, "_confined_job_state", lambda job: None)
    result = manager.codex_state_readiness(ctx, 12345, _token(), budget_seconds=100.0, sleep=sleep, clock=lambda: clock[0])
    assert result.outcome == "failed" and "left the queue" in result.reason


# ----------------------------------------------------------------- the relaunch transaction

class _FakeManager:
    """The manager as ``_relaunch_session`` drives it, with each answer scripted.

    ``launch`` simulates what the real launch does to the session record:
    ``"record"`` writes the new job into the attempt (a confined replacement),
    ``"plain"`` records it as the session's job (an unmanaged launch), an
    exception is raised, and ``"record+raise"`` records then raises (the
    tmux gate failing after sbatch).
    """

    def __init__(self, *, old_state="RUNNING", drain=None, readiness=(), launch="record", cancel=(),
                 confined=True, target_name=None):
        self.calls = []
        self.old_state = old_state
        self.drain_report = drain
        self.readiness_results = list(readiness)
        self.launch = launch
        self.cancel_results = list(cancel)
        self.cancel_sessions = []
        self.confined = confined
        self.target_name = target_name
        self.drain_kwargs = []
        self.executor = type("E", (), {})()
        self.executor.run_agent = self._run_agent
        self.executor.local_disk_root = "/local"

    def _run_agent(self, args, **kwargs):
        from types import SimpleNamespace
        self.calls.append(("run_agent", list(args)))
        self.cancel_sessions.append(RemoteSession.load("sample", target_name=self.target_name))
        result = self.cancel_results.pop(0) if self.cancel_results else 0
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(returncode=result, stdout="", stderr="scheduler unavailable" if result else "")

    def context_for(self, mirror):
        from types import SimpleNamespace
        return SimpleNamespace(settings=SimpleNamespace(name=mirror), confined=self.confined)

    def _confined_job_state(self, job_id):
        self.calls.append(("job_state", job_id))
        if isinstance(self.old_state, Exception):
            raise self.old_state
        return self.old_state

    def codex_state_drain(self, ctx, job_id, **kwargs):
        self.calls.append(("drain", job_id, kwargs["mode"]))
        self.drain_kwargs.append(kwargs)
        return self.drain_report

    def launch_agent(self, ctx, **kwargs):
        self.calls.append(("launch", kwargs))
        session = RemoteSession.load("sample", target_name=self.target_name)
        if self.launch in ("record", "record+raise"):
            session.replacement = dict(session.replacement, state="submitted", new_job=12345, new_node="n0001.savio4")
            session.save()
        elif self.launch == "plain":
            session.slurm_job_id, session.compute_node = 777, "n0002.savio4"
            session.save()
        if isinstance(self.launch, Exception):
            raise self.launch
        if self.launch == "record+raise":
            raise RuntimeError("tmux session did not come up")
        return 0

    def codex_state_readiness(self, ctx, job_id, token, **kwargs):
        self.calls.append(("readiness", job_id, token.selector))
        return self.readiness_results.pop(0)


def _config(*, confined=True, enabled=True):
    from pathlib import Path
    from types import SimpleNamespace
    from sucoder.config import BranchPrefixes, CodexStateConfig, MirrorSettings, RemoteConfig, SlurmConfig
    slurm = SlurmConfig(partition="savio4_htc", account="fc_jevons", confined=confined,
                        codex_state=CodexStateConfig(enabled=enabled))
    return SimpleNamespace(mirrors={"sample": MirrorSettings(
        name="sample", canonical_repo=Path("/repo"), mirror_name="sample", branch_prefixes=BranchPrefixes(),
        remote=RemoteConfig(slurm=slurm))})


def _managed_session(tmp_path, monkeypatch, *, replacement=None, remote_control=None, target_name=None):
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path / "sessions")
    session = RemoteSession(mirror_name="sample", target_name=target_name, slurm_job_id=4242, compute_node="n0033.savio4")
    session.managed_owner = {"command": ["codex", "--model", "gpt-6-astra"], "model": None, "scope": "ab" * 32,
                             "durable_root": "/g/.sucoder-state", "run": RUN, "remote_control": remote_control}
    session.replacement = replacement
    session.save()
    return session


def _relaunch(monkeypatch, fake, config=None, ctx=None):
    import logging
    from sucoder import cli
    monkeypatch.setattr(cli, "_build_manager_for_mirror", lambda *a, **k: fake)
    return cli._relaunch_session("sample", config or _config(), ctx, logging.getLogger("test-renew"), False)


def _ready():
    from sucoder.codex_state import Readiness
    return Readiness("ready", f"placed on {RUN}/{GEN}; run {NEW_RUN}")


def _pending(why="state restoring"):
    from sucoder.codex_state import Readiness
    return Readiness("pending", why)


def _failed(why="state refused: bwrap is not installed"):
    from sucoder.codex_state import Readiness
    return Readiness("failed", why)


@pytest.mark.parametrize("remote_control", [None, "server", "off"])
def test_relaunch_drains_launches_on_the_token_and_promotes_the_ready_successor(tmp_path, monkeypatch, remote_control):
    _managed_session(tmp_path, monkeypatch, remote_control=remote_control)
    fake = _FakeManager(drain=_report(), readiness=[_ready()])
    assert _relaunch(monkeypatch, fake) == 12345
    kinds = [c[0] for c in fake.calls]
    assert kinds == ["job_state", "drain", "launch", "job_state", "readiness", "run_agent"]
    launch = fake.calls[2][1]
    assert launch == {"sync": False, "detached": True, "codex_state_from": f"{RUN}/{GEN}", "codex_state_unattended": True,
                      "replace_job": 4242, "command_override": ["codex", "--model", "gpt-6-astra"], "model_override": None,
                      "codex_remote_control": remote_control}
    assert fake.calls[-1] == ("run_agent", ["scancel", "4242"])
    cancelling = fake.cancel_sessions[0]
    assert cancelling.slurm_job_id == 12345 and cancelling.replacement["state"] == "ready"
    session = RemoteSession.load("sample", target_name=None)
    assert (session.slurm_job_id, session.compute_node, session.replacement) == (12345, "n0001.savio4", None)
    assert session.managed_owner["command"] == ["codex", "--model", "gpt-6-astra"]


def test_relaunch_defers_without_a_transferable_acknowledgement(tmp_path, monkeypatch, caplog):
    _managed_session(tmp_path, monkeypatch)
    fake = _FakeManager(drain=_report(ack=_ack("warm_only", cls="warm"), outcome="drained: warm only"))
    with caplog.at_level("INFO"):
        assert _relaunch(monkeypatch, fake) is None
    assert [c[0] for c in fake.calls] == ["job_state", "drain"]
    assert RemoteSession.load("sample", target_name=None).replacement is None
    assert "deferring the replacement" in caplog.text and "not sealed" in caplog.text


def test_relaunch_keeps_a_pending_replacement_and_only_probes_it_next_turn(tmp_path, monkeypatch):
    _managed_session(tmp_path, monkeypatch)
    fake = _FakeManager(drain=_report(), readiness=[_pending(), _ready()])
    assert _relaunch(monkeypatch, fake) is None
    session = RemoteSession.load("sample", target_name=None)
    attempt = ReplacementAttempt.from_json(session.replacement)
    assert (session.slurm_job_id, attempt.state, attempt.new_job) == (4242, "submitted", 12345)
    assert _relaunch(monkeypatch, fake) == 12345
    assert [c[0] for c in fake.calls] == ["job_state", "drain", "launch", "job_state", "readiness", "job_state", "readiness", "run_agent"], (
        "the second turn neither drains nor launches: it asks about the recorded job")
    assert RemoteSession.load("sample", target_name=None).slurm_job_id == 12345


def test_relaunch_cancels_a_failed_replacement_keeps_the_token_and_gives_up_after_the_bound(tmp_path, monkeypatch, caplog):
    _managed_session(tmp_path, monkeypatch)
    fake = _FakeManager(drain=_report(), readiness=[_failed(), _failed("job 12345 left the queue before its run was placed"),
                                                      _failed()])
    for turn in range(3):
        assert _relaunch(monkeypatch, fake) is None
        attempt = ReplacementAttempt.from_json(RemoteSession.load("sample", target_name=None).replacement)
        assert (attempt.state, attempt.new_job, len(attempt.failures)) == ("drained", None, turn + 1)
    cancels = [c for c in fake.calls if c[0] == "run_agent" and c[1][0] == "scancel"]
    assert [c[1] for c in cancels] == [["scancel", "12345"], ["scancel", "12345"]], "a job that left the queue is not cancelled"
    assert fake.calls.count(("drain", 4242, "renew")) == 1, "the token is reused, the predecessor is not drained again"
    with caplog.at_level("ERROR"):
        assert _relaunch(monkeypatch, fake) is None
    assert "3 replacement attempts failed" in caplog.text
    assert [c[0] for c in fake.calls].count("launch") == 3
    assert RemoteSession.load("sample", target_name=None).slurm_job_id == 4242


def test_relaunch_probes_a_job_recorded_before_the_launch_raised(tmp_path, monkeypatch):
    _managed_session(tmp_path, monkeypatch)
    fake = _FakeManager(drain=_report(), readiness=[_ready()], launch="record+raise")
    assert _relaunch(monkeypatch, fake) == 12345
    assert [c[0] for c in fake.calls] == ["job_state", "drain", "launch", "job_state", "readiness", "run_agent"]


def test_relaunch_records_a_launch_that_submitted_nothing(tmp_path, monkeypatch):
    _managed_session(tmp_path, monkeypatch)
    fake = _FakeManager(drain=_report(), launch=RuntimeError("sbatch failed (exit 1)"))
    assert _relaunch(monkeypatch, fake) is None
    attempt = ReplacementAttempt.from_json(RemoteSession.load("sample", target_name=None).replacement)
    assert attempt.new_job is None and attempt.failures == ["launch: sbatch failed (exit 1)"]
    assert RemoteSession.load("sample", target_name=None).slurm_job_id == 4242


@pytest.mark.parametrize("remote_control", [None, "server", "off"])
def test_relaunch_of_a_lost_predecessor_without_a_token_is_unmanaged_with_the_owners_command(
        tmp_path, monkeypatch, caplog, remote_control):
    _managed_session(tmp_path, monkeypatch, remote_control=remote_control)
    fake = _FakeManager(old_state=None, launch="plain")
    with caplog.at_level("WARNING"):
        assert _relaunch(monkeypatch, fake) == 777
    assert [c[0] for c in fake.calls] == ["job_state", "launch"]
    launch = fake.calls[1][1]
    assert launch == {"sync": False, "detached": True, "command_override": ["codex", "--model", "gpt-6-astra"],
                      "model_override": None, "codex_remote_control": remote_control, "replace_job": 4242}
    assert f"/g/.sucoder-state/codex/{'ab' * 32}/runs/{RUN}" in caplog.text


def test_relaunch_of_a_lost_predecessor_with_a_token_continues_on_it(tmp_path, monkeypatch):
    attempt = ReplacementAttempt(attempt="00ff" * 4, old_job=4242, old_node="n0033.savio4", old_run=RUN, token=_token())
    _managed_session(tmp_path, monkeypatch, replacement=attempt.as_json())
    fake = _FakeManager(old_state=None, readiness=[_ready()])
    assert _relaunch(monkeypatch, fake) == 12345
    assert [c[0] for c in fake.calls] == ["launch", "job_state", "readiness", "run_agent"], "no predecessor probe or drain: the token is held"
    assert fake.calls[0][1]["codex_state_from"] == f"{RUN}/{GEN}"


def test_relaunch_without_a_managed_run_is_unmanaged(tmp_path, monkeypatch):
    _managed_session(tmp_path, monkeypatch)
    fake = _FakeManager(drain=_report(run=None, requested=False, ack=None, outcome="no managed run recorded"), launch="plain")
    assert _relaunch(monkeypatch, fake) == 777
    assert [c[0] for c in fake.calls] == ["job_state", "drain", "launch"] and "codex_state_from" not in fake.calls[2][1]
    assert fake.calls[2][1]["replace_job"] == 4242


def test_relaunch_continues_a_cleanly_exited_sealed_run(tmp_path, monkeypatch):
    _managed_session(tmp_path, monkeypatch)
    report = _report(requested=False, ack=None, state="exited", exit=0, exited_at=20.0,
                     last_durable={"name": GEN, "class": "sealed", "manifest_sha256": SHA, "published_at": 21.0},
                     outcome=f"no managed owner to drain: run {RUN} exited (exit 0) and its final generation {GEN} is sealed")
    fake = _FakeManager(drain=report, readiness=[_ready()])
    assert _relaunch(monkeypatch, fake) == 12345
    assert fake.calls[2][1]["codex_state_from"] == f"{RUN}/{GEN}"


def test_relaunch_refuses_an_unreadable_or_mismatched_record(tmp_path, monkeypatch, caplog):
    _managed_session(tmp_path, monkeypatch, replacement={"attempt": "zz"})
    fake = _FakeManager(drain=_report())
    with caplog.at_level("ERROR"):
        assert _relaunch(monkeypatch, fake) is None
    assert fake.calls == [] and "unreadable replacement record" in caplog.text
    attempt = ReplacementAttempt(attempt="00ff" * 4, old_job=4141, old_node=None, old_run=RUN, token=_token(), new_job=12345,
                                 state="submitted")
    _managed_session(tmp_path, monkeypatch, replacement=attempt.as_json())
    with caplog.at_level("ERROR"):
        assert _relaunch(monkeypatch, fake) is None
    assert fake.calls == []
    assert "record is reconciled" in caplog.text


def test_relaunch_asks_again_when_the_scheduler_cannot_answer(tmp_path, monkeypatch):
    from sucoder.mirror import MirrorError
    _managed_session(tmp_path, monkeypatch)
    fake = _FakeManager(old_state=MirrorError("Could not verify SLURM job 4242"))
    assert _relaunch(monkeypatch, fake) is None
    assert [c[0] for c in fake.calls] == ["job_state"]
    assert RemoteSession.load("sample", target_name=None).replacement is None


def test_relaunch_of_a_managed_owner_on_an_unconfined_target_is_unmanaged_with_its_command(tmp_path, monkeypatch, caplog):
    _managed_session(tmp_path, monkeypatch)
    fake = _FakeManager(launch="plain", confined=False)
    with caplog.at_level("WARNING"):
        assert _relaunch(monkeypatch, fake, config=_config(confined=False)) == 777
    assert fake.calls[0][1]["command_override"] == ["codex", "--model", "gpt-6-astra"]
    assert "replace_job" not in fake.calls[0][1]
    assert "confined targets only" in caplog.text


def test_unmanaged_relaunch_restores_the_old_allocation_when_the_launch_fails_before_recording_one(tmp_path, monkeypatch):
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path / "sessions")
    session = RemoteSession(mirror_name="sample", target_name=None, slurm_job_id=4242, compute_node="n0033.savio4")
    session.save()
    fake = _FakeManager(launch=RuntimeError("sbatch failed"))
    assert _relaunch(monkeypatch, fake) is None
    loaded = RemoteSession.load("sample", target_name=None)
    assert (loaded.slurm_job_id, loaded.compute_node) == (4242, "n0033.savio4")


def test_relaunch_fails_a_replacement_that_never_records_a_run_after_the_bound(tmp_path, monkeypatch):
    """A runner that refuses before placing writes no latest-run, so its
    status is never readable: bounded by the time since first RUNNING,
    not pending forever with the allocation leaking."""
    from sucoder import cli
    from sucoder.codex_state import Readiness
    import time as _time
    submitted = _time.time() - 2000.0
    attempt = ReplacementAttempt(attempt="00ff" * 4, old_job=4242, old_node="n0033.savio4", old_run=RUN, token=_token(),
                                 state="submitted", new_job=12345, new_node="n0001.savio4", submitted_at=submitted,
                                 running_since=submitted)
    _managed_session(tmp_path, monkeypatch, replacement=attempt.as_json())
    nothing = Readiness("pending", "no managed run recorded yet; 360 s budget elapsed", no_status=True)
    fake = _FakeManager(readiness=[nothing, nothing])
    # Recently running: still pending, nothing cancelled.
    monkeypatch.setattr(cli, "_MANAGED_NO_STATUS_BOUND_SECONDS", 3000.0)
    assert _relaunch(monkeypatch, fake) is None
    assert not any(c[0] == "run_agent" for c in fake.calls)
    assert ReplacementAttempt.from_json(RemoteSession.load("sample", target_name=None).replacement).new_job == 12345
    # Past the bound: failed, cancelled, the token kept for a fresh submission.
    monkeypatch.setattr(cli, "_MANAGED_NO_STATUS_BOUND_SECONDS", 900.0)
    assert _relaunch(monkeypatch, fake) is None
    assert [c[1] for c in fake.calls if c[0] == "run_agent"] == [["scancel", "12345"]]
    after = ReplacementAttempt.from_json(RemoteSession.load("sample", target_name=None).replacement)
    assert after.new_job is None and after.state == "drained" and "never placed one" in after.failures[-1]
    assert after.running_since is None, "the next submission gets its own startup bound"


def test_relaunch_after_the_failure_bound_falls_back_only_once_the_predecessor_is_gone(tmp_path, monkeypatch, caplog):
    failures = ["job 1: state refused", "job 2: state refused", "job 3: state refused"]
    attempt = ReplacementAttempt(attempt="00ff" * 4, old_job=4242, old_node="n0033.savio4", old_run=RUN, token=_token(),
                                 failures=failures)
    _managed_session(tmp_path, monkeypatch, replacement=attempt.as_json())
    fake = _FakeManager(old_state="RUNNING")
    with caplog.at_level("ERROR"):
        assert _relaunch(monkeypatch, fake) is None
    assert [c[0] for c in fake.calls] == ["job_state"] and "3 replacement attempts failed" in caplog.text
    assert RemoteSession.load("sample", target_name=None).replacement is not None
    fake = _FakeManager(old_state=None, launch="plain")
    with caplog.at_level("WARNING"):
        assert _relaunch(monkeypatch, fake) == 777
    assert [c[0] for c in fake.calls] == ["job_state", "launch"]
    assert fake.calls[1][1]["command_override"] == ["codex", "--model", "gpt-6-astra"] and "codex_state_from" not in fake.calls[1][1]
    assert fake.calls[1][1]["replace_job"] == 4242
    loaded = RemoteSession.load("sample", target_name=None)
    assert loaded.slurm_job_id == 777 and loaded.replacement is None
    assert f"{RUN}/{GEN}" in caplog.text and "gone after 3 failed replacements" in caplog.text


def test_launch_confined_records_when_the_replacement_was_submitted(tmp_path, monkeypatch):
    from tests.test_mirror import _THIS_TARGET, _confined_manager, _confined_responder
    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    _replacing_session(tmp_path)
    manager.executor.run_agent = _confined_responder([], live_state="RUNNING", by_name=f"4242|{_THIS_TARGET}")
    import time as _time
    before = _time.time()
    manager._launch_confined(ctx, ["codex"], remote_prelude_text=None, prelude_sentinel="__X__", env=None,
                             detached=True, replace_job=4242)
    attempt = ReplacementAttempt.from_json(RemoteSession.load("sample", target_name=None).replacement)
    assert attempt.submitted_at is not None and before <= attempt.submitted_at <= _time.time()
    assert ReplacementAttempt.from_json(attempt.as_json()) == attempt


# ----------------------------------------------------------------- the listings see a pending replacement

def test_session_listings_name_a_pending_replacement_job(tmp_path, monkeypatch):
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path / "sessions")
    attempt = ReplacementAttempt(attempt="00ff" * 4, old_job=4242, old_node="n0033.savio4", old_run=RUN, token=_token(),
                                 state="submitted", new_job=12345, new_node="n0001.savio4")
    session = RemoteSession(mirror_name="sample", target_name="savio", slurm_job_id=4242)
    session.replacement = attempt.as_json()
    session.save()
    RemoteSession(mirror_name="other", target_name="savio", slurm_job_id=4300).save()
    assert RemoteSession.pending_replacement_job(session) == 12345
    assert RemoteSession.pending_replacement_job(RemoteSession.load("other", target_name="savio")) is None
    assert RemoteSession.pending_replacement_job({"replacement": {"new_job": True}}) is None
    assert RemoteSession.holders_of_job(12345) == ["sample--savio"]
    assert RemoteSession.holders_of_job(4242) == ["sample--savio"]
    assert RemoteSession.holders_of_job(4242, exclude_key="sample--savio") == []
    assert RemoteSession.recorded_jobs() == {"other--savio": 4300, "sample--savio": 4242, "sample--savio (replacement)": 12345}
    attempt.state = "ready"
    session.slurm_job_id = 12345
    session.replacement = attempt.as_json()
    session.save()
    assert RemoteSession.pending_replacement_job(session) == 4242
    assert RemoteSession.holders_of_job(4242) == ["sample--savio"]
    assert RemoteSession.holders_of_job(12345) == ["sample--savio"]
    assert RemoteSession.recorded_jobs() == {"other--savio": 4300, "sample--savio": 12345, "sample--savio (replacement)": 4242}


# ----------------------------------------------------------------- review regressions

def test_relaunch_resolves_the_named_target_for_managed_routing_and_drain_timeout(tmp_path, monkeypatch):
    from types import SimpleNamespace
    _managed_session(tmp_path, monkeypatch, target_name="savio")
    config = _config()
    target = config.mirrors["sample"].remote
    target.slurm.codex_state.checkpoint_timeout_seconds = 75
    config.mirrors["sample"].remote = None
    ctx = SimpleNamespace(obj={"target": target, "target_name": "savio"})
    fake = _FakeManager(drain=_report(), readiness=[_ready()], target_name="savio")
    assert _relaunch(monkeypatch, fake, config=config, ctx=ctx) == 12345
    assert fake.calls[1] == ("drain", 4242, "renew")
    assert fake.drain_kwargs[0]["timeout_seconds"] == 105.0
    assert fake.calls[2][1]["codex_state_from"] == f"{RUN}/{GEN}"
    assert config.mirrors["sample"].remote is None, "target resolution does not mutate the mirror settings"


@pytest.mark.parametrize("confined", [True, False])
def test_unmanaged_relaunch_excludes_the_predecessor_only_on_a_confined_target(tmp_path, monkeypatch, confined):
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path / "sessions")
    RemoteSession(mirror_name="sample", slurm_job_id=4242, compute_node="n0033.savio4").save()
    fake = _FakeManager(launch="plain", confined=confined)
    assert _relaunch(monkeypatch, fake, config=_config(confined=confined)) == 777
    assert fake.calls == [("launch", {"sync": False, "detached": True, **({"replace_job": 4242} if confined else {})})]


@pytest.mark.parametrize("exit_code, published_at, exited_at, why", [
    (0, 19.0, 20.0, "not published after the last exit"),
    (1, 21.0, 20.0, "clean exit"),
    (None, 21.0, 20.0, "clean exit"),
    (False, 21.0, 20.0, "clean exit"),
    (0, None, 20.0, "time is missing or invalid"),
    (0, 21.0, None, "time is missing or invalid"),
    (0, "21", 20.0, "time is missing or invalid"),
])
def test_exit_token_refuses_an_obsolete_seal_or_missing_clean_exit_proof(exit_code, published_at, exited_at, why):
    from sucoder.codex_state import transfer_token_from_drain
    status = {"state": "exited", "exit": exit_code, "run": RUN, "exited_at": exited_at,
              "last_generation": {"name": GEN, "class": "sealed", "manifest_sha256": SHA, "published_at": published_at}}
    with pytest.raises(ValueError, match=why):
        transfer_token_from_exit(status, nonce=NONCE)
    report = _report(requested=False, ack=None, state="exited", exit=exit_code, exited_at=exited_at,
                     last_durable=status["last_generation"])
    with pytest.raises(ValueError, match=why):
        transfer_token_from_drain(report)


@pytest.mark.parametrize("state", ["exited", "stopped_misplaced", "released"])
@pytest.mark.parametrize("exit_code, published_at, transferable", [(0, 10.0, False), (1, 21.0, False), (0, 20.0, True)])
def test_drain_reports_the_newest_exit_for_transfer_validation(tmp_path, monkeypatch, state, exit_code, published_at, transferable):
    from sucoder.codex_state import transfer_token_from_drain
    from tests.test_codex_state_release import _Node, _drain
    from tests.test_codex_state_status import _recorded_run
    status = {"state": state, "exit": exit_code, "run": RUN,
              "transitions": [{"state": "exited", "at": 5.0}, {"state": "placed", "at": 15.0},
                              {"state": "stopped_misplaced", "at": 20.0}, {"state": "released", "at": 22.0}],
              "last_generation": {"name": GEN, "class": "sealed", "manifest_sha256": SHA, "published_at": published_at}}
    node = _Node(_recorded_run(status=status))
    report, _ = _drain(tmp_path, monkeypatch, node, mode="renew")
    assert not report.requested and node.scripts == []
    assert (report.exit, report.exited_at) == (exit_code, 20.0)
    if transferable and state != "released":
        assert transfer_token_from_drain(report).selector == f"{RUN}/{GEN}"
    else:
        with pytest.raises(ValueError):
            transfer_token_from_drain(report)


@pytest.mark.parametrize("confined", [True, False])
@pytest.mark.parametrize("child_stat, runner_stat, why", [
    (_proc_stat(4321, 123456), _proc_stat(1234, 654321), None),
    (None, _proc_stat(1234, 654321), "Codex process is gone"),
    (_proc_stat(4321, 123457), _proc_stat(1234, 654321), "Codex process is gone"),
    ("malformed stat", _proc_stat(1234, 654321), "Codex process is gone"),
    (_proc_stat(4321, 123456), None, "runner is gone"),
    (_proc_stat(4321, 123456, "Z"), _proc_stat(1234, 654321), "Codex process is gone"),
    (_proc_stat(4321, 123456, "X"), _proc_stat(1234, 654321), "Codex process is gone"),
    (_proc_stat(4321, 123456), _proc_stat(1234, 654321, "Z"), "runner is gone"),
    (_proc_stat(4321, 123456), _proc_stat(1234, 654321, "X"), "runner is gone"),
])
def test_readiness_checks_codex_incarnation_and_runner_through_the_status_route(
        tmp_path, monkeypatch, confined, child_stat, runner_stat, why):
    from types import SimpleNamespace
    from tests.test_codex_state import _state_manager
    from tests.test_codex_state_release import _Node
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    ctx.settings.remote.slurm.confined = confined
    monkeypatch.setattr(manager, "_confined_job_state", lambda job: "RUNNING")
    monkeypatch.setattr(manager, "codex_state_status", lambda c, job: SimpleNamespace(status=_placed(), owner={"runner_pid": 1234}))
    files = {path: stat for path, stat in [("/proc/4321/stat", child_stat), ("/proc/1234/stat", runner_stat)] if stat is not None}
    node = _Node(files)
    manager.executor.run_agent.side_effect = node.run_agent
    result = manager.codex_state_readiness(ctx, 4242, _token(), budget_seconds=0.0)
    assert result.outcome == ("ready" if why is None else "failed")
    if why is not None:
        assert why in result.reason
    prefix = ["srun", "--jobid=4242", "--overlap"] if confined else []
    paths = ["/proc/4321/stat"] + (["/proc/1234/stat"] if child_stat == _proc_stat(4321, 123456) else [])
    assert [argv for argv, _ in node.calls] == [prefix + ["cat", path] for path in paths]


@pytest.mark.parametrize("cancel", [1, OSError("scheduler unreachable")])
def test_failed_successor_stays_submitted_until_cancellation_succeeds(tmp_path, monkeypatch, caplog, cancel):
    _managed_session(tmp_path, monkeypatch)
    fake = _FakeManager(drain=_report(), readiness=[_failed()] * 5, cancel=[cancel] * 4 + [0])
    for turn in range(4):
        assert _relaunch(monkeypatch, fake) is None
        session = RemoteSession.load("sample")
        attempt = ReplacementAttempt.from_json(session.replacement)
        assert (attempt.state, attempt.new_job, attempt.new_node) == ("submitted", 12345, "n0001.savio4")
        assert session.slurm_job_id == 4242 and len(attempt.failures) == turn + 1
        assert attempt.failures[-1].startswith("cancel:")
    assert "retrying cancellation next turn" in caplog.text
    assert _relaunch(monkeypatch, fake) is None
    attempt = ReplacementAttempt.from_json(RemoteSession.load("sample").replacement)
    assert (attempt.state, attempt.new_job, attempt.new_node) == ("drained", None, None)
    assert [c[0] for c in fake.calls].count("launch") == 1
    assert [c[1] for c in fake.calls if c[0] == "run_agent"] == [["scancel", "12345"]] * 5
    # Cancel retries do not consume the bound on replacement submissions.
    fake.readiness_results.append(_pending())
    assert _relaunch(monkeypatch, fake) is None
    assert [c[0] for c in fake.calls].count("launch") == 2


@pytest.mark.parametrize("cancel", [1, OSError("scheduler unreachable")])
def test_promotion_keeps_the_ready_record_until_predecessor_cancellation_is_confirmed(tmp_path, monkeypatch, caplog, cancel):
    _managed_session(tmp_path, monkeypatch)
    fake = _FakeManager(drain=_report(), readiness=[_ready()], cancel=[cancel])
    assert _relaunch(monkeypatch, fake) == 12345
    session = RemoteSession.load("sample")
    attempt = ReplacementAttempt.from_json(session.replacement)
    assert (session.slurm_job_id, session.compute_node, attempt.state) == (12345, "n0001.savio4", "ready")
    assert RemoteSession.pending_replacement_job(session) == 4242
    assert attempt.failures[-1].startswith("cancel: predecessor job 4242:")
    assert "predecessor job 4242 is still allocated" in caplog.text
    assert fake.cancel_sessions[0].replacement["state"] == "ready", "persist before issuing scancel"


@pytest.mark.parametrize("old_state, cancel", [(None, ()), ("RUNNING", (0,)), ("RUNNING", (1,))])
def test_next_renewal_finishes_the_ready_record_before_draining_the_current_job(tmp_path, monkeypatch, old_state, cancel):
    attempt = ReplacementAttempt(attempt="00ff" * 4, old_job=4242, old_node="n0033.savio4", old_run=RUN, token=_token(),
                                 state="ready", new_job=12345, new_node="n0001.savio4")
    session = _managed_session(tmp_path, monkeypatch, replacement=attempt.as_json())
    session.slurm_job_id, session.compute_node = 12345, "n0001.savio4"
    session.save()
    fake = _FakeManager(cancel=cancel, drain=_report(ack=_ack("warm_only", cls="warm")))

    def job_state(job):
        fake.calls.append(("job_state", job))
        return old_state if job == 4242 else "RUNNING"

    fake._confined_job_state = job_state
    assert _relaunch(monkeypatch, fake) is None
    loaded = RemoteSession.load("sample")
    assert loaded.slurm_job_id == 12345
    if cancel == (1,):
        assert loaded.replacement["state"] == "ready"
        assert fake.calls == [("job_state", 4242), ("run_agent", ["scancel", "4242"])]
    else:
        assert loaded.replacement is None
        assert fake.calls == ([("job_state", 4242)] + ([("run_agent", ["scancel", "4242"])] if cancel else [])
                              + [("job_state", 12345), ("drain", 12345, "renew")])


def test_release_with_a_ready_record_cancels_the_predecessor(tmp_path, monkeypatch):
    from tests.test_codex_state_release import _Cluster
    yaml_text = (
        "login_node: ln002\nslurm_job_id: 7654400\ncompute_node: n0032\n"
        "replacement:\n  attempt: 00ff00ff00ff00ff\n  old_job: 7654321\n  state: ready\n  new_job: 7654400\n"
    )
    cluster = _Cluster(tmp_path, monkeypatch, session_yaml=yaml_text)
    result = cluster.release("-f", "--no-drain")
    assert result.exit_code == 0, (result.output, result.exception)
    commands = [command for _host, command in cluster.captures]
    assert commands[0].startswith("scancel 7654400") and commands[-1] == "scancel 7654321"
    loaded = RemoteSession.load("sample", target_name="fake-slurm")
    assert loaded.slurm_job_id is None and loaded.replacement is None


@pytest.mark.parametrize("pid", [4321, 1234])
@pytest.mark.parametrize("budget", [0.0, 10.0])
def test_readiness_retries_a_liveness_probe_error_without_cancelling(tmp_path, monkeypatch, pid, budget):
    from types import SimpleNamespace
    from sucoder.mirror import MirrorError, _CodexStateRoute
    from tests.test_codex_state import _state_manager
    from tests.test_codex_state_release import _Node
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(manager, "_confined_job_state", lambda job: "RUNNING")
    monkeypatch.setattr(manager, "codex_state_status",
                        lambda c, job: SimpleNamespace(status=_placed(), owner={"runner_pid": 1234}))
    node = _Node({"/proc/4321/stat": _proc_stat(4321, 123456), "/proc/1234/stat": _proc_stat(1234, 654321)})
    manager.executor.run_agent.side_effect = node.run_agent
    read, errors = _CodexStateRoute.read, []

    def transient(route, path):
        if path == f"/proc/{pid}/stat" and not errors:
            errors.append(path)
            raise MirrorError("srun temporarily unavailable")
        return read(route, path)

    monkeypatch.setattr(_CodexStateRoute, "read", transient)
    slept = []
    result = manager.codex_state_readiness(ctx, 4242, _token(), budget_seconds=budget,
                                           sleep=slept.append, clock=lambda: sum(slept))
    if budget == 0:
        assert result.outcome == "pending" and "liveness not readable yet (srun temporarily unavailable)" in result.reason
        result = manager.codex_state_readiness(ctx, 4242, _token(), budget_seconds=0.0)
    else:
        assert slept == [10.0]
    assert result.ready and errors == [f"/proc/{pid}/stat"]
    assert all("scancel" not in argv for argv, _ in node.calls)


@pytest.mark.parametrize("recorded, ready", [(None, True), (654321, True), (654322, False)])
def test_readiness_checks_runner_starttime_when_recorded(tmp_path, monkeypatch, recorded, ready):
    from types import SimpleNamespace
    from tests.test_codex_state import _state_manager
    from tests.test_codex_state_release import _Node
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    owner = {"runner_pid": 1234}
    if recorded is not None:
        owner["runner_starttime"] = recorded
    monkeypatch.setattr(manager, "_confined_job_state", lambda job: "RUNNING")
    monkeypatch.setattr(manager, "codex_state_status", lambda c, job: SimpleNamespace(status=_placed(), owner=owner))
    node = _Node({"/proc/4321/stat": _proc_stat(4321, 123456), "/proc/1234/stat": _proc_stat(1234, 654321)})
    manager.executor.run_agent.side_effect = node.run_agent
    result = manager.codex_state_readiness(ctx, 4242, _token(), budget_seconds=0.0)
    assert result.outcome == ("ready" if ready else "failed")
    if not ready:
        assert "runner is gone" in result.reason


@pytest.mark.parametrize("failure", [1, OSError("login node unavailable")])
@pytest.mark.parametrize("dry_run", [False, True])
def test_renew_hold_reconciles_ready_record_on_every_probe(tmp_path, monkeypatch, caplog, failure, dry_run):
    from types import SimpleNamespace
    from sucoder import cli, tunnel
    import sucoder.renew as renew_module
    from tests.test_codex_state import _state_manager
    from tests.test_codex_state_release import _FakeSshControl
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    attempt = ReplacementAttempt(attempt="00ff" * 4, old_job=4242, old_node="n0033.savio4", old_run=RUN, token=_token(),
                                 state="ready", new_job=12345, new_node="n0001.savio4")
    session = _managed_session(tmp_path, monkeypatch, replacement=attempt.as_json())
    session.slurm_job_id, session.compute_node, session.login_node = 12345, "n0001.savio4", "login1"
    session.save()
    monkeypatch.setattr(cli, "_connect_with_retry", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_reconcile_login_node", lambda *a, **kw: None)
    monkeypatch.setattr(tunnel, "SshControl", _FakeSshControl)
    calls, cancels = [], [failure, 0]

    def capture(control, host, command, **kw):
        calls.append((host, command))
        if command == "scancel 4242":
            assert RemoteSession.load("sample").replacement == attempt.as_json()
            rc = cancels.pop(0)
            if isinstance(rc, Exception):
                raise rc
            return SimpleNamespace(returncode=rc, stdout="", stderr="scheduler unavailable" if rc else "")
        return SimpleNamespace(returncode=0, stdout="RUNNING|1-00:00:00\n" if command.startswith("squeue") else "", stderr="")

    def loop(job_id, settings, *, probe, **kwargs):
        assert probe(job_id).mins_left == 1440  # HOLD: no relaunch is called.
        assert RemoteSession.load("sample").replacement == attempt.as_json()
        assert probe(job_id).ok
        assert RemoteSession.load("sample").replacement == (attempt.as_json() if dry_run else None)
        assert probe(job_id).ok  # Nothing more to cancel after retirement.

    monkeypatch.setattr(cli, "_run_remote_capture", capture)
    monkeypatch.setattr(renew_module, "run_renew_loop", loop)
    with caplog.at_level("INFO"):
        cli.renew(SimpleNamespace(obj={"config": manager.config}, params={}), mirror="sample", drain_minutes=20,
                  poll_interval=0, checkpoint_grace=0, once=True, verbose=False, dry_run=dry_run)
    assert [(host, command) for host, command in calls if command.startswith("scancel")] == ([] if dry_run else [("login1", "scancel 4242")] * 2)
    if not dry_run:
        assert "next poll" in caplog.text and "retired the ready record" in caplog.text
    assert len([command for _, command in calls if "@sucoder-service-launch" in command]) == 1


@pytest.mark.parametrize("state, current_job", [("submitted", 12345), ("ready", 4242)])
def test_renew_reconciliation_ignores_an_unpromoted_or_mismatched_record(tmp_path, monkeypatch, state, current_job):
    import logging
    from sucoder.cli import _reconcile_ready_record
    attempt = ReplacementAttempt(attempt="00ff" * 4, old_job=4242, old_node=None, old_run=RUN, token=_token(),
                                 state=state, new_job=12345)
    session = _managed_session(tmp_path, monkeypatch, replacement=attempt.as_json())
    session.slurm_job_id = current_job
    _reconcile_ready_record(session, lambda job: pytest.fail("must not cancel"), logging.getLogger("test-renew"))
    assert session.replacement == attempt.as_json()


@pytest.mark.parametrize("change", ["release", "release-retained", "attempt", "new_job"])
def test_renew_reconciliation_reloads_after_cancellation(tmp_path, monkeypatch, change):
    import logging
    from types import SimpleNamespace
    from sucoder.cli import _reconcile_ready_record
    attempt = ReplacementAttempt(attempt="00ff" * 4, old_job=4242, old_node=None, old_run=RUN, token=_token(),
                                 state="ready", new_job=12345)
    session = _managed_session(tmp_path, monkeypatch, replacement=attempt.as_json(), target_name="savio")
    session.slurm_job_id, session.compute_node = 12345, "n0001.savio4"
    session.save()
    changed = []

    def scancel(job):
        assert job == 4242
        current = RemoteSession.load("sample", target_name="savio")
        if change.startswith("release"):
            current.slurm_job_id, current.compute_node = None, None
            if change == "release":
                current.replacement = None
        elif change == "attempt":
            current.replacement["attempt"] = "11ee" * 4
        else:
            current.replacement["new_job"] = 23456
        current.save()
        changed.append(current)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    _reconcile_ready_record(session, scancel, logging.getLogger("test-renew"))
    expected = changed[0]
    if change == "release-retained":
        expected.replacement = None  # Retire only the record, preserving the released allocation.
    assert RemoteSession.load("sample", target_name="savio") == expected


@pytest.mark.parametrize("state", ["PENDING", "RUNNING"])
def test_relaunch_no_status_bound_excludes_queue_time(tmp_path, monkeypatch, state):
    import logging
    from sucoder import cli
    from sucoder.codex_state import Readiness
    attempt = ReplacementAttempt(attempt="00ff" * 4, old_job=4242, old_node=None, old_run=RUN, token=_token(),
                                 state="submitted", new_job=12345, submitted_at=1.0)
    session = _managed_session(tmp_path, monkeypatch, replacement=attempt.as_json())
    nothing = Readiness("pending", "no managed run recorded yet", no_status=True)
    fake = _FakeManager(old_state=state, readiness=[nothing, nothing, nothing])
    logger = logging.getLogger("test-renew")
    assert cli._managed_replacement_turn(fake, None, session, attempt, logger, clock=lambda: 2000.0) is None
    recorded = ReplacementAttempt.from_json(RemoteSession.load("sample").replacement)
    assert recorded.new_job == 12345 and recorded.running_since == (2000.0 if state == "RUNNING" else None)
    assert not any(c[0] == "run_agent" for c in fake.calls)
    assert cli._managed_replacement_turn(fake, None, session, attempt, logger, clock=lambda: 2899.0) is None
    assert not any(c[0] == "run_agent" for c in fake.calls)
    assert cli._managed_replacement_turn(fake, None, session, attempt, logger, clock=lambda: 2900.0) is None
    assert [c[1] for c in fake.calls if c[0] == "run_agent"] == ([["scancel", "12345"]] if state == "RUNNING" else [])


def test_relaunch_no_status_bound_restarts_after_requeue(tmp_path, monkeypatch):
    import logging
    from sucoder import cli
    from sucoder.codex_state import Readiness
    attempt = ReplacementAttempt(attempt="00ff" * 4, old_job=4242, old_node=None, old_run=RUN, token=_token(),
                                 state="submitted", new_job=12345, submitted_at=1.0)
    _managed_session(tmp_path, monkeypatch, replacement=attempt.as_json())
    nothing = Readiness("pending", "no managed run recorded yet", no_status=True)
    fake = _FakeManager(readiness=[nothing] * 4)
    for state, now, running_since in [("RUNNING", 1000.0, 1000.0), ("PENDING", 2000.0, None),
                                      ("RUNNING", 3000.0, 3000.0), ("RUNNING", 3899.0, 3000.0)]:
        session = RemoteSession.load("sample")
        attempt = ReplacementAttempt.from_json(session.replacement)
        fake.old_state = state
        assert cli._managed_replacement_turn(fake, None, session, attempt, logging.getLogger("test-renew"),
                                             clock=lambda: now) is None
        recorded = ReplacementAttempt.from_json(RemoteSession.load("sample").replacement)
        assert recorded.new_job == 12345 and recorded.running_since == running_since
        assert recorded.state == "submitted" and recorded.failures == []
        assert not any(c[0] == "run_agent" for c in fake.calls)


def test_relaunch_leaves_running_since_unset_when_the_scheduler_probe_fails(tmp_path, monkeypatch):
    import logging
    from sucoder import cli
    from sucoder.mirror import MirrorError
    attempt = ReplacementAttempt(attempt="00ff" * 4, old_job=4242, old_node=None, old_run=RUN, token=_token(),
                                 state="submitted", new_job=12345, submitted_at=1.0)
    session = _managed_session(tmp_path, monkeypatch, replacement=attempt.as_json())
    fake = _FakeManager(old_state=MirrorError("squeue unavailable"))
    assert cli._managed_replacement_turn(fake, None, session, attempt, logging.getLogger("test-renew"), clock=lambda: 2000.0) is None
    assert fake.calls == [("job_state", 12345)]
    assert RemoteSession.load("sample").replacement == attempt.as_json()


def test_renew_attempt_running_since_round_trips_and_loads_legacy_records():
    attempt = ReplacementAttempt(attempt="00ff" * 4, old_job=4242, old_node=None, old_run=RUN, token=_token(), running_since=10.0)
    assert ReplacementAttempt.from_json(attempt.as_json()) == attempt
    data = attempt.as_json()
    del data["running_since"]
    assert ReplacementAttempt.from_json(data).running_since is None
    for bad in (True, "10", []):
        with pytest.raises(ValueError, match="running_since"):
            ReplacementAttempt.from_json(dict(data, running_since=bad))


@pytest.mark.parametrize("probe_rc", [0, 1, 255])
def test_unconfined_tui_attach_keeps_the_existing_managed_owner(tmp_path, monkeypatch, caplog, probe_rc):
    from types import SimpleNamespace
    from tests.test_codex_state import _state_manager
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    ctx.settings.remote.slurm.confined = False
    session = _managed_session(tmp_path, monkeypatch)
    original = session.managed_owner
    monkeypatch.setattr(manager, "_prepare_local_tier", lambda ctx, root: (tmp_path, "export SUCODER_LOCAL_ROOT=/local"))
    manager.executor.run_agent.side_effect = lambda args, **kw: SimpleNamespace(
        returncode=probe_rc if list(args)[:2] == ["tmux", "has-session"] else 0, stdout="", stderr="")
    with caplog.at_level("INFO"):
        assert manager.launch_agent(ctx, sync=False, command_override=["codex"], model_override="another-model",
                                    supports_inline_prompt=True, detached=True) == 0
    recorded = RemoteSession.load("sample").managed_owner
    assert recorded == (manager._managed_owner if probe_rc == 1 else original)
    assert manager._managed_owner["model"] == "another-model"
    if probe_rc == 0:
        assert "Keeping the existing managed owner record for tmux session sucoder-sample" in caplog.text
    elif probe_rc == 255:
        assert "tmux session sucoder-sample probe returned 255" in caplog.text
        assert "skipping managed owner recording" in caplog.text
    calls = manager.executor.run_agent.call_args_list
    assert calls[-2].args[0] == ["tmux", "has-session", "-t", "=sucoder-sample"]
    assert calls[-1].args[0][:2] == ["tmux", "new-session"] and "-A" in calls[-1].args[0]
