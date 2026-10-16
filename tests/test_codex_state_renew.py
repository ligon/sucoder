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
    status = {"state": "exited", "exit": 0, "run": RUN,
              "last_generation": {"name": GEN, "class": "sealed", "manifest_sha256": SHA}}
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
                     requested=False, ack=None, state="exited",
                     last_durable={"name": GEN, "class": "sealed", "manifest_sha256": SHA})
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

def test_readiness_polls_status_until_placed_on_the_token(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tests.test_codex_state import _state_manager
    manager, ctx, _confined = _state_manager(tmp_path, monkeypatch)
    answers = [None, {"state": "restoring"}, {"state": "launched"}, _placed()]
    monkeypatch.setattr(manager, "_confined_job_state", lambda job: "RUNNING")
    monkeypatch.setattr(manager, "codex_state_status", lambda c, job: SimpleNamespace(status=answers.pop(0)))
    slept, clock = [], [0.0]

    def sleep(seconds):
        slept.append(seconds)
        clock[0] += seconds
    result = manager.codex_state_readiness(ctx, 12345, _token(), budget_seconds=100.0, poll_seconds=10.0,
                                           sleep=sleep, clock=lambda: clock[0])
    assert result.ready and slept == [10.0, 10.0, 10.0] and answers == []


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

    def __init__(self, *, old_state="RUNNING", drain=None, readiness=(), launch="record"):
        self.calls = []
        self.old_state = old_state
        self.drain_report = drain
        self.readiness_results = list(readiness)
        self.launch = launch
        self.executor = type("E", (), {})()
        self.executor.run_agent = self._run_agent
        self.executor.local_disk_root = "/local"

    def _run_agent(self, args, **kwargs):
        from types import SimpleNamespace
        self.calls.append(("run_agent", list(args)))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def context_for(self, mirror):
        from types import SimpleNamespace
        return SimpleNamespace(settings=SimpleNamespace(name=mirror), confined=True)

    def _confined_job_state(self, job_id):
        self.calls.append(("job_state", job_id))
        if isinstance(self.old_state, Exception):
            raise self.old_state
        return self.old_state

    def codex_state_drain(self, ctx, job_id, **kwargs):
        self.calls.append(("drain", job_id, kwargs["mode"]))
        return self.drain_report

    def launch_agent(self, ctx, **kwargs):
        self.calls.append(("launch", kwargs))
        session = RemoteSession.load("sample", target_name=None)
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
    from types import SimpleNamespace
    from sucoder.config import CodexStateConfig
    slurm = SimpleNamespace(confined=confined, codex_state=CodexStateConfig(enabled=enabled))
    return SimpleNamespace(mirrors={"sample": SimpleNamespace(remote=SimpleNamespace(slurm=slurm))})


def _managed_session(tmp_path, monkeypatch, *, replacement=None):
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path / "sessions")
    session = RemoteSession(mirror_name="sample", target_name=None, slurm_job_id=4242, compute_node="n0033.savio4")
    session.managed_owner = {"command": ["codex", "--model", "gpt-6-astra"], "model": None, "scope": "ab" * 32,
                             "durable_root": "/g/.sucoder-state", "run": RUN}
    session.replacement = replacement
    session.save()
    return session


def _relaunch(monkeypatch, fake, config=None):
    import logging
    from sucoder import cli
    monkeypatch.setattr(cli, "_build_manager_for_mirror", lambda *a, **k: fake)
    return cli._relaunch_session("sample", config or _config(), None, logging.getLogger("test-renew"), False)


def _ready():
    from sucoder.codex_state import Readiness
    return Readiness("ready", f"placed on {RUN}/{GEN}; run {NEW_RUN}")


def _pending(why="state restoring"):
    from sucoder.codex_state import Readiness
    return Readiness("pending", why)


def _failed(why="state refused: bwrap is not installed"):
    from sucoder.codex_state import Readiness
    return Readiness("failed", why)


def test_relaunch_drains_launches_on_the_token_and_promotes_the_ready_successor(tmp_path, monkeypatch):
    _managed_session(tmp_path, monkeypatch)
    fake = _FakeManager(drain=_report(), readiness=[_ready()])
    assert _relaunch(monkeypatch, fake) == 12345
    kinds = [c[0] for c in fake.calls]
    assert kinds == ["job_state", "drain", "launch", "readiness"]
    launch = fake.calls[2][1]
    assert launch == {"sync": False, "detached": True, "codex_state_from": f"{RUN}/{GEN}", "codex_state_unattended": True,
                      "replace_job": 4242, "command_override": ["codex", "--model", "gpt-6-astra"], "model_override": None}
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
    assert [c[0] for c in fake.calls] == ["job_state", "drain", "launch", "readiness", "readiness"], (
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
    assert [c[0] for c in fake.calls] == ["job_state", "drain", "launch", "readiness"]


def test_relaunch_records_a_launch_that_submitted_nothing(tmp_path, monkeypatch):
    _managed_session(tmp_path, monkeypatch)
    fake = _FakeManager(drain=_report(), launch=RuntimeError("sbatch failed (exit 1)"))
    assert _relaunch(monkeypatch, fake) is None
    attempt = ReplacementAttempt.from_json(RemoteSession.load("sample", target_name=None).replacement)
    assert attempt.new_job is None and attempt.failures == ["launch: sbatch failed (exit 1)"]
    assert RemoteSession.load("sample", target_name=None).slurm_job_id == 4242


def test_relaunch_of_a_lost_predecessor_without_a_token_is_unmanaged_with_the_owners_command(tmp_path, monkeypatch, caplog):
    _managed_session(tmp_path, monkeypatch)
    fake = _FakeManager(old_state=None, launch="plain")
    with caplog.at_level("WARNING"):
        assert _relaunch(monkeypatch, fake) == 777
    assert [c[0] for c in fake.calls] == ["job_state", "launch"]
    launch = fake.calls[1][1]
    assert launch == {"sync": False, "detached": True, "command_override": ["codex", "--model", "gpt-6-astra"],
                      "model_override": None}
    assert f"/g/.sucoder-state/codex/{'ab' * 32}/runs/{RUN}" in caplog.text


def test_relaunch_of_a_lost_predecessor_with_a_token_continues_on_it(tmp_path, monkeypatch):
    attempt = ReplacementAttempt(attempt="00ff" * 4, old_job=4242, old_node="n0033.savio4", old_run=RUN, token=_token())
    _managed_session(tmp_path, monkeypatch, replacement=attempt.as_json())
    fake = _FakeManager(old_state=None, readiness=[_ready()])
    assert _relaunch(monkeypatch, fake) == 12345
    assert [c[0] for c in fake.calls] == ["launch", "readiness"], "no probe, no drain: the token is held"
    assert fake.calls[0][1]["codex_state_from"] == f"{RUN}/{GEN}"


def test_relaunch_without_a_managed_run_is_unmanaged(tmp_path, monkeypatch):
    _managed_session(tmp_path, monkeypatch)
    fake = _FakeManager(drain=_report(run=None, requested=False, ack=None, outcome="no managed run recorded"), launch="plain")
    assert _relaunch(monkeypatch, fake) == 777
    assert [c[0] for c in fake.calls] == ["job_state", "drain", "launch"] and "codex_state_from" not in fake.calls[2][1]


def test_relaunch_continues_a_cleanly_exited_sealed_run(tmp_path, monkeypatch):
    _managed_session(tmp_path, monkeypatch)
    report = _report(requested=False, ack=None, state="exited", last_durable={"name": GEN, "class": "sealed", "manifest_sha256": SHA},
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
    fake = _FakeManager(launch="plain")
    with caplog.at_level("WARNING"):
        assert _relaunch(monkeypatch, fake, config=_config(confined=False)) == 777
    assert fake.calls[0][1]["command_override"] == ["codex", "--model", "gpt-6-astra"]
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
    status is never readable: bounded by the time since sbatch answered,
    not pending forever with the allocation leaking."""
    from sucoder import cli
    from sucoder.codex_state import Readiness
    import time as _time
    submitted = _time.time() - 2000.0
    attempt = ReplacementAttempt(attempt="00ff" * 4, old_job=4242, old_node="n0033.savio4", old_run=RUN, token=_token(),
                                 state="submitted", new_job=12345, new_node="n0001.savio4", submitted_at=submitted)
    _managed_session(tmp_path, monkeypatch, replacement=attempt.as_json())
    nothing = Readiness("pending", "no managed run recorded yet; 360 s budget elapsed", no_status=True)
    fake = _FakeManager(readiness=[nothing, nothing])
    # Recently submitted: still pending, nothing cancelled.
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
