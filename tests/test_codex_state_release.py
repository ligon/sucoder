"""Managed Codex state, Slice 3c, launcher half: the drain request.

``sucoder release`` asks the allocation's managed Codex state owner to
drain before it cancels anything, and ``sucoder codex-state-drain`` exposes
the same request to operators and to the renewal of Slice 3d.  Three
layers: the manager's request against a mocked executor (the exact atomic
write, the polls through ``srun --overlap``, the nonce and run matching, the
bound), the ``release`` wiring through the CLI (the drain precedes the
cancel, every outcome is printed, the release proceeds whatever the drain
came to), and the drain command itself.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from sucoder import cli
from sucoder import mirror as mirror_mod
from sucoder.codex_state import (
    CodexStateDrain, describe_drain_ack, drain_request_script, drain_request_text,
)
from sucoder.mirror import MirrorError
from tests.test_cli import _FakeSshControl
from tests.test_codex_state import _state_manager
from tests.test_codex_state_status import RUN, SCOPE, SCOPE_DIR, _fresh_placed, _recorded_run, _refused, _srun

NONCE = "0123456789abcdef"
STALE_NONCE = "fedcba9876543210"
RUN_DIR = f"{SCOPE_DIR}/{RUN}"
ACK_PATH = f"{RUN_DIR}/drain.ack"
STATUS_READS = [f"{SCOPE_DIR}/latest-run", f"{RUN_DIR}/status.json", f"{RUN_DIR}/owner.json"]


def _ack(outcome="sealed", *, nonce=NONCE, run=RUN, generation="000013-cafebabe", cls="sealed", exit_code=0, **extra):
    record = None if generation is None else {
        "name": generation, "class": cls, "manifest_sha256": "ab" * 32, "path": f"/durable/generations/{generation}"}
    ack = {"nonce": nonce, "mode": "release", "outcome": outcome, "generation": record, "exit": exit_code,
           "child": {"pid": 4321, "starttime": 123456}, "stopped": outcome in ("sealed", "crash"),
           "acknowledged_at": 1760000100.0, "acknowledged_at_iso": "2026-10-15T10:01:40Z", "requested_at": 1760000090.0,
           "requester": "sucoder release on laptop", "terminal": "none", "scope": SCOPE, "run": run}
    ack.update(extra)
    return ack


class _Node:
    """The run directory as the fake executor sees it.

    Reads are answered from ``files`` (a path not there fails the way
    ``cat`` does); the request write is recorded, not applied; ``drain.ack``
    takes each text in ``acks`` once that many polls of it have happened,
    so a stale acknowledgement can sit there first and the right one appear
    later.
    """

    def __init__(self, files, *, acks=(), write_rc=0):
        self.files = dict(files)
        self.acks = list(acks)
        self.write_rc = write_rc
        self.calls = []
        self.scripts = []
        self.polls = 0

    def run_agent(self, args, **kwargs):
        args = list(args)
        self.calls.append((args, kwargs))
        inner = args[3:] if args[:2] == ["srun", "--jobid=4242"] else args
        if inner[:2] == ["sh", "-c"]:
            self.scripts.append(inner[2])
            return SimpleNamespace(returncode=self.write_rc, stdout="", stderr="sh: write failed\n" if self.write_rc else "")
        assert inner[0] == "cat" and len(inner) == 2, inner
        path = inner[1]
        if path == ACK_PATH:
            self.polls += 1
            for after, text in self.acks:
                if self.polls > after:
                    self.files[path] = text
        if path in self.files:
            return SimpleNamespace(returncode=0, stdout=self.files[path], stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr=f"cat: {path}: No such file or directory\n")


class _Clock:
    """A monotonic clock that advances by what was slept: polls are free, the bound is the sleeps."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def __call__(self):
        return self.now


def _drain(tmp_path, monkeypatch, node, *, confined=True, mode="release", timeout=60.0, dry_run=False,
           requester="sucoder release on laptop"):
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    ctx.settings.remote.slurm.confined = confined
    manager.executor.run_agent.side_effect = node.run_agent
    manager.executor.dry_run = dry_run
    monkeypatch.setattr(mirror_mod, "_state_new_drain_nonce", lambda: NONCE)
    clock = _Clock()
    report = manager.codex_state_drain(ctx, 4242, mode=mode, timeout_seconds=timeout, requester=requester,
                                       sleep=clock.sleep, clock=clock)
    return report, clock


def _script_parts(script):
    """The request script taken apart: (json, tmp path, final path)."""
    words = shlex.split(script)
    assert words[:5] == ["umask", "077", "&&", "printf", "%s\\n"], words     # printf sees the two characters \ and n
    assert words[6] == ">" and words[8:11] == ["&&", "mv", "-f"], words
    return json.loads(words[5]), words[7], words[12]


# ----------------------------------------------------------------- the request, through the manager

def test_confined_drain_writes_the_request_atomically_and_polls_until_its_own_ack_appears(tmp_path, monkeypatch):
    stale = json.dumps(_ack(nonce=STALE_NONCE, generation="000009-00000009"))
    node = _Node(dict(_recorded_run(), **{ACK_PATH: stale}), acks=[(2, json.dumps(_ack()))])
    report, clock = _drain(tmp_path, monkeypatch, node)

    argvs = [argv for argv, _ in node.calls]
    assert argvs[:3] == [_srun(p) for p in STATUS_READS]
    assert argvs[3][:5] == ["srun", "--jobid=4242", "--overlap", "sh", "-c"] and len(argvs[3]) == 6
    assert argvs[4:] == [_srun(ACK_PATH)] * 3          # the stale ack twice, then this request's
    assert all(kw == {"check": False, "capture_output": True} for _, kw in node.calls)
    assert report.commands == argvs

    request, tmp, final = _script_parts(node.scripts[0])
    assert request["nonce"] == NONCE and request["mode"] == "release" and request["requester"] == "sucoder release on laptop"
    assert isinstance(request["requested_at"], float)
    assert tmp == f"{RUN_DIR}/drain.request.tmp-{NONCE}" and final == f"{RUN_DIR}/drain.request"
    assert node.scripts[0] == drain_request_script(
        RUN_DIR, NONCE, drain_request_text(NONCE, "release", request["requested_at"], "sucoder release on laptop"))

    assert (report.job_id, report.scope, report.run, report.mode, report.nonce) == (4242, SCOPE, RUN, "release", NONCE)
    assert report.requested and report.acknowledged and report.settled and not report.dry_run
    assert report.ack == _ack()
    assert report.outcome == "drained: sealed generation 000013-cafebabe (abababababab)"
    assert report.last_durable == _fresh_placed()["last_generation"]
    assert clock.sleeps == [2.0, 2.0, 2.0] and report.waited_seconds == 6.0 and report.timeout_seconds == 60.0


def test_unconfined_drain_takes_the_executor_route_without_srun(tmp_path, monkeypatch):
    node = _Node(_recorded_run(), acks=[(0, json.dumps(_ack()))])
    report, _ = _drain(tmp_path, monkeypatch, node, confined=False)
    argvs = [argv for argv, _ in node.calls]
    assert argvs[:3] == [["cat", p] for p in STATUS_READS]
    assert argvs[3] == ["sh", "-c", node.scripts[0]] and argvs[4:] == [["cat", ACK_PATH]]
    assert not any(argv[0] == "srun" for argv in argvs)
    assert report.outcome.startswith("drained: sealed generation")


def test_an_acknowledgement_of_another_request_or_run_never_counts(tmp_path, monkeypatch):
    other_run = json.dumps(_ack(run="f" * 32))
    node = _Node(dict(_recorded_run(), **{ACK_PATH: json.dumps(_ack(nonce=STALE_NONCE))}), acks=[(10, other_run)])
    report, clock = _drain(tmp_path, monkeypatch, node, timeout=60.0)
    assert node.polls == 30 and clock.sleeps == [2.0] * 30
    assert report.requested and report.ack is None and not report.acknowledged
    assert report.outcome == "drain unacknowledged after 60 s; last durable generation 000012-0a1b2c3d"
    assert report.last_durable == _fresh_placed()["last_generation"] and report.waited_seconds == 60.0


def test_the_bound_is_the_clock_not_a_poll_count(tmp_path, monkeypatch):
    node = _Node(_recorded_run())
    report, clock = _drain(tmp_path, monkeypatch, node, timeout=5.0)
    assert clock.sleeps == [2.0, 2.0, 1.0] and node.polls == 3     # the last sleep is what remains of the bound
    assert report.outcome == "drain unacknowledged after 5 s; last durable generation 000012-0a1b2c3d"
    node = _Node(_recorded_run(), acks=[(0, json.dumps(_ack()))])
    report, clock = _drain(tmp_path, monkeypatch, node, timeout=0.0)
    assert clock.sleeps == [0.0] and node.polls == 1 and report.acknowledged      # one read even at a zero bound


def test_no_managed_run_writes_nothing(tmp_path, monkeypatch):
    node = _Node({})
    report, clock = _drain(tmp_path, monkeypatch, node)
    assert [argv for argv, _ in node.calls] == [_srun(f"{SCOPE_DIR}/latest-run")]
    assert node.scripts == [] and clock.sleeps == []
    assert report.run is None and not report.requested and report.ack is None
    assert report.outcome == "no managed run recorded"
    assert report.nonce == NONCE and report.scope == SCOPE


@pytest.mark.parametrize("state", ["refused", "blocked", "released"])
def test_a_run_with_no_runner_left_to_answer_gets_no_request(tmp_path, monkeypatch, state):
    status = _refused()
    status["state"] = state
    node = _Node(_recorded_run(status=status))
    report, clock = _drain(tmp_path, monkeypatch, node)
    assert node.scripts == [] and node.polls == 0 and clock.sleeps == []
    assert report.run == RUN and not report.requested
    assert report.outcome == f"no managed owner to drain: run {RUN} is {state}; last durable generation none"


def test_an_exited_run_with_only_a_warm_generation_is_still_asked(tmp_path, monkeypatch):
    # The exit path writes ``exited`` before the final capture and reads a
    # late request after it, so a run exited with a warm last generation may
    # still answer; one whose final generation is published cannot.
    status = _fresh_placed()
    status.update({"state": "exited", "exit": 0, "outcome": "placed"})
    node = _Node(_recorded_run(status=status), acks=[(0, json.dumps(_ack()))])
    report, _ = _drain(tmp_path, monkeypatch, node)
    assert report.requested and report.acknowledged


@pytest.mark.parametrize("state, cls, exit_code", [("exited", "sealed", 0), ("exited", "crash", 137), ("stopped_misplaced", "crash", 71)])
def test_a_run_exited_with_its_final_generation_published_is_not_asked(tmp_path, monkeypatch, state, cls, exit_code):
    status = _fresh_placed()
    status.update({"state": state, "exit": exit_code, "outcome": "placed"})
    status["last_generation"] = dict(status["last_generation"], name="000015-f1f1f1f1")
    status["last_generation"]["class"] = cls
    node = _Node(_recorded_run(status=status))
    report, clock = _drain(tmp_path, monkeypatch, node)
    assert node.scripts == [] and node.polls == 0 and clock.sleeps == []
    assert report.run == RUN and not report.requested and report.last_durable == status["last_generation"]
    assert report.outcome == (f"no managed owner to drain: run {RUN} exited (exit {exit_code}) and its final "
                              f"generation 000015-f1f1f1f1 is {cls}")


def test_dry_run_lists_the_commands_with_a_placeholder_and_does_not_wait(tmp_path, monkeypatch):
    node = _Node({})
    report, clock = _drain(tmp_path, monkeypatch, node, dry_run=True)
    assert report.dry_run and not report.requested and report.run is None and clock.sleeps == []
    assert [argv for argv, _ in node.calls] == [_srun(f"{SCOPE_DIR}/latest-run")]
    placeholder = f"{SCOPE_DIR}/<run>"
    assert report.commands[:3] == [_srun(f"{SCOPE_DIR}/latest-run"), _srun(f"{placeholder}/status.json"), _srun(f"{placeholder}/owner.json")]
    write, poll = report.commands[3:]
    assert write[:5] == ["srun", "--jobid=4242", "--overlap", "sh", "-c"]
    request, tmp, final = _script_parts(write[5])
    assert request["nonce"] == NONCE and tmp == f"{placeholder}/drain.request.tmp-{NONCE}" and final == f"{placeholder}/drain.request"
    assert poll == _srun(f"{placeholder}/drain.ack")
    assert report.outcome == "dry run: nothing requested"


def test_renew_mode_rides_in_the_request(tmp_path, monkeypatch):
    node = _Node(_recorded_run(), acks=[(0, json.dumps(_ack(mode="renew")))])
    report, _ = _drain(tmp_path, monkeypatch, node, mode="renew", requester="sucoder codex-state-drain on laptop")
    request, _, _ = _script_parts(node.scripts[0])
    assert request["mode"] == "renew" and request["requester"] == "sucoder codex-state-drain on laptop"
    assert report.mode == "renew" and report.acknowledged
    token = report.as_json()["ack"]
    assert (token["nonce"], token["generation"]["name"], token["generation"]["manifest_sha256"]) == (NONCE, "000013-cafebabe", "ab" * 32)
    assert token["child"] == {"pid": 4321, "starttime": 123456} and token["stopped"] is True


@pytest.mark.parametrize("ack, line", [
    (_ack("warm_only", cls="warm", exit_code=None, generation="000014-0000beef"),
     "drained: warm generation 000014-0000beef only (a person holds the terminal; it seals when they exit)"),
    (_ack("crash", cls="crash", exit_code=137),
     "drained: crash generation 000013-cafebabe (abababababab); Codex exited 137, kept for recovery, not sealed"),
    (_ack("failed", generation=None, reason="replica t9 is shrunk",
          last_durable={"name": "000011-0000aaaa", "class": "warm", "manifest_sha256": "cd" * 32, "path": "/d/000011-0000aaaa"}),
     "drain failed: replica t9 is shrunk; last durable generation 000011-0000aaaa"),
    (_ack("failed", generation=None, reason="no final capture ran"),
     "drain failed: no final capture ran; last durable generation 000012-0a1b2c3d"),
    (_ack("unheard_of", generation=None),
     "drain acknowledged with an unknown outcome 'unheard_of'; last durable generation 000012-0a1b2c3d"),
])
def test_each_acknowledged_outcome_has_its_line(tmp_path, monkeypatch, ack, line):
    node = _Node(_recorded_run(), acks=[(0, json.dumps(ack))])
    report, _ = _drain(tmp_path, monkeypatch, node)
    assert report.acknowledged and report.outcome == line
    assert describe_drain_ack(ack, _fresh_placed()["last_generation"]) == line
    expected_last = ack.get("last_durable") or _fresh_placed()["last_generation"]
    assert report.last_durable == expected_last
    assert report.settled is (ack["outcome"] in ("sealed", "crash"))


def test_an_unreadable_ack_is_reported_with_the_timeout(tmp_path, monkeypatch):
    node = _Node(dict(_recorded_run(), **{ACK_PATH: "{torn"}))
    report, _ = _drain(tmp_path, monkeypatch, node, timeout=2.0)
    assert report.ack is None
    assert report.outcome.startswith("drain unacknowledged after 2 s; last durable generation 000012-0a1b2c3d (drain.ack is not readable JSON:")


def test_a_write_that_fails_is_a_mirror_error_naming_the_request(tmp_path, monkeypatch):
    node = _Node(_recorded_run(), write_rc=1)
    with pytest.raises(MirrorError, match=f"Could not write {RUN_DIR}/drain.request for job 4242 \\(exit 1\\): sh: write failed"):
        _drain(tmp_path, monkeypatch, node)
    assert node.polls == 0


def test_a_bad_mode_or_bound_is_refused_before_anything_is_read(tmp_path, monkeypatch):
    node = _Node(_recorded_run())
    with pytest.raises(MirrorError, match="Drain mode must be one of release, renew, not 'pause'"):
        _drain(tmp_path, monkeypatch, node, mode="pause")
    with pytest.raises(MirrorError, match="cannot be negative"):
        _drain(tmp_path, monkeypatch, node, timeout=-1.0)
    assert node.calls == []


def test_the_record_serializes_for_json_output():
    report = CodexStateDrain(job_id=4242, scope=SCOPE, mode="renew", nonce=NONCE, commands=[["cat", "x"]], outcome="o",
                             run=RUN, requested=True, ack=_ack(), waited_seconds=4.0, timeout_seconds=60.0)
    assert report.as_json() == {
        "job_id": 4242, "scope": SCOPE, "mode": "renew", "nonce": NONCE, "run": RUN, "requested": True, "outcome": "o",
        "ack": _ack(), "last_durable": None, "state": None, "waited_seconds": 4.0, "timeout_seconds": 60.0,
        "commands": [["cat", "x"]]}
    # The run's state rides along only when no owner was left to ask (Slice 3d reads it for a clean sealed exit).
    unasked = CodexStateDrain(job_id=4242, scope=SCOPE, mode="renew", nonce=NONCE, commands=[], outcome="o", run=RUN,
                              state="exited")
    assert unasked.as_json()["state"] == "exited" and not unasked.requested


# ----------------------------------------------------------------- ``release``

def _release_config(tmp_path, *, confined=True, harness="codex", codex_state=True):
    import os
    human = os.environ.get("USER", "coder")
    (tmp_path / "mirrors").mkdir(exist_ok=True)
    (tmp_path / "canonical").mkdir(exist_ok=True)
    state = "      codex_state:\n        enabled: true\n        checkpoint_timeout_seconds: 30\n" if codex_state else ""
    path = tmp_path / "config.yaml"
    path.write_text(f"""
human_user: {human}
agent_user: {human}
agent_group: {human}
mirror_root: {tmp_path / 'mirrors'}
mirrors:
  sample:
    canonical_repo: {tmp_path / 'canonical'}
    mirror_name: sample
    branch_prefixes:
      human: {human}
      agent: {human}
    agent_launcher:
      command: [{harness}]
targets:
  fake-slurm:
    gateway: gw.example.org
    transfer_host: dtn.example.org
    slurm:
      partition: test
      account: test_acct
      confined: {'true' if confined else 'false'}
      local_disk: /local
{state}""", encoding="utf-8")
    return path


class _Cluster:
    """What ``release`` reaches for, stubbed: the gateway capture, the manager, the sibling tmux kill."""

    def __init__(self, tmp_path, monkeypatch, *, drain=None, squeue="RUNNING", siblings=(), session_yaml=None, **config):
        from sucoder import session as session_mod
        from sucoder import tunnel as tunnel_mod
        from sucoder.session import RemoteSession

        self.events = []
        self.drain_calls = []
        self.builder_calls = []
        self.captures = []
        self.drain = drain if drain is not None else CodexStateDrain(
            job_id=7654321, scope=SCOPE, mode="release", nonce=NONCE, commands=[], run=RUN, requested=True,
            ack=_ack(), outcome="drained: sealed generation 000013-cafebabe (abababababab)")
        self.squeue = squeue
        self.config_path = _release_config(tmp_path, **config)
        home = tmp_path / "home"
        home.mkdir(exist_ok=True)           # a test may stand up two clusters in one tmp_path
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)
        self.sessions_dir = home / ".sucoder" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        (self.sessions_dir / "sample--fake-slurm.yaml").write_text(
            session_yaml or "login_node: ln002\nslurm_job_id: 7654321\ncompute_node: n0032\n", encoding="utf-8")
        monkeypatch.setattr(session_mod, "_session_dir", lambda: self.sessions_dir)
        monkeypatch.setattr(cli, "_connect_with_retry", lambda *a, **kw: None)
        monkeypatch.setattr(tunnel_mod, "SshControl", _FakeSshControl)
        monkeypatch.setattr(RemoteSession, "holders_of_job", staticmethod(lambda job_id, **kw: list(siblings)))
        monkeypatch.setattr(cli, "_run_remote_capture", self.capture)
        monkeypatch.setattr(cli, "_build_manager_for_mirror", self.build)
        monkeypatch.setattr(subprocess, "run", self.subprocess_run)

    def capture(self, control, host, command, **kw):
        self.captures.append((host, command))
        if command.startswith("squeue"):
            self.events.append("squeue")
            return SimpleNamespace(returncode=0, stdout=f"{self.squeue}\n", stderr="")
        self.events.append("scancel")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def build(self, config, logger, dry_run, mirror, **kwargs):
        self.builder_calls.append((dry_run, mirror))

        def codex_state_drain(mirror_ctx, job_id, **kw):
            self.events.append("drain")
            self.drain_calls.append((mirror_ctx, job_id, kw))
            if isinstance(self.drain, BaseException):
                raise self.drain
            return self.drain

        return SimpleNamespace(context_for=lambda name: f"ctx:{name}", codex_state_drain=codex_state_drain)

    def subprocess_run(self, argv, **kw):
        self.events.append("tmux-kill" if "kill-session" in " ".join(argv) else "subprocess")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def release(self, *args, input=None):
        argv = ["--config", str(self.config_path), "-T", "fake-slurm", "release", "sample", *args]
        return CliRunner().invoke(cli.app, argv, input=input)

    def job_id(self):
        from sucoder.session import RemoteSession
        return RemoteSession.load("sample", target_name="fake-slurm").slurm_job_id


@pytest.mark.parametrize("outcome", [
    "drained: sealed generation 000013-cafebabe (abababababab)",
    "drained: warm generation 000014-0000beef only (a person holds the terminal; it seals when they exit)",
    "drained: crash generation 000013-cafebabe (abababababab); Codex exited 137, kept for recovery, not sealed",
    "drain failed: replica t9 is shrunk; last durable generation 000011-0000aaaa",
    "drain unacknowledged after 60 s; last durable generation 000012-0a1b2c3d",
    "no managed run recorded",
    f"no managed owner to drain: run {RUN} is refused; last durable generation none",
])
def test_release_drains_before_the_cancel_and_prints_the_outcome(tmp_path, monkeypatch, outcome):
    drain = CodexStateDrain(job_id=7654321, scope=SCOPE, mode="release", nonce=NONCE, commands=[], outcome=outcome)
    cluster = _Cluster(tmp_path, monkeypatch, drain=drain)
    result = cluster.release("-f")
    assert result.exit_code == 0, (result.output, result.exception)
    assert cluster.events == ["drain", "scancel"]
    assert cluster.builder_calls == [(False, "sample")]
    (mirror_ctx, job_id, kw), = cluster.drain_calls
    assert (mirror_ctx, job_id) == ("ctx:sample", 7654321)
    assert kw["mode"] == "release" and kw["timeout_seconds"] == 60.0 and kw["requester"].startswith("sucoder release on ")
    (host, command), = cluster.captures
    assert host == "gw.example.org" and command.startswith("scancel 7654321")
    lines = result.output.splitlines()
    assert lines.index(outcome) < lines.index("Released SLURM job 7654321 on n0032.")
    assert cluster.job_id() is None


def test_release_drain_timeout_can_be_overridden(tmp_path, monkeypatch):
    cluster = _Cluster(tmp_path, monkeypatch)
    result = cluster.release("-f", "--drain-timeout", "5")
    assert result.exit_code == 0, (result.output, result.exception)
    assert cluster.drain_calls[0][2]["timeout_seconds"] == 5.0


def test_release_no_drain_cancels_without_asking(tmp_path, monkeypatch):
    cluster = _Cluster(tmp_path, monkeypatch)
    result = cluster.release("-f", "--no-drain")
    assert result.exit_code == 0, (result.output, result.exception)
    assert cluster.events == ["scancel"] and cluster.builder_calls == []
    assert "drain" not in result.output and "Released SLURM job 7654321" in result.output


def test_release_without_codex_state_never_builds_a_manager(tmp_path, monkeypatch):
    cluster = _Cluster(tmp_path, monkeypatch, codex_state=False)
    result = cluster.release("-f")
    assert result.exit_code == 0, (result.output, result.exception)
    assert cluster.events == ["scancel"] and cluster.builder_calls == []


def test_release_skips_the_drain_for_a_non_codex_harness_unless_told_otherwise(tmp_path, monkeypatch):
    cluster = _Cluster(tmp_path, monkeypatch, harness="claude")
    result = cluster.release("-f")
    assert result.exit_code == 0, (result.output, result.exception)
    assert cluster.events == ["scancel"] and cluster.builder_calls == [] and "drain" not in result.output

    cluster = _Cluster(tmp_path, monkeypatch, harness="claude")
    result = cluster.release("-f", "--harness", "codex")
    assert result.exit_code == 0, (result.output, result.exception)
    assert cluster.events == ["drain", "scancel"]


def test_release_trusts_the_recorded_service_launch_over_the_configured_harness(tmp_path, monkeypatch):
    recorded = ("login_node: ln002\nslurm_job_id: 7654321\ncompute_node: n0032\n"
                "service_launch:\n  command: [codex, app-server]\n  model: null\n")
    cluster = _Cluster(tmp_path, monkeypatch, harness="claude", session_yaml=recorded)
    result = cluster.release("-f")
    assert result.exit_code == 0, (result.output, result.exception)
    assert cluster.events == ["drain", "scancel"]


@pytest.mark.parametrize("state, line", [
    ("COMPLETED", "no managed run recorded: job 7654321 is COMPLETED, so no owner can answer a drain"),
    ("", "no managed run recorded: job 7654321 is gone, so no owner can answer a drain"),
])
def test_release_on_an_unconfined_target_probes_the_job_before_building_a_manager(tmp_path, monkeypatch, state, line):
    # Building a manager there re-allocates a node when the job has ended;
    # a release must never do that, so a job that is not RUNNING is not asked.
    cluster = _Cluster(tmp_path, monkeypatch, confined=False, squeue=state)
    result = cluster.release("-f")
    assert result.exit_code == 0, (result.output, result.exception)
    assert cluster.events == ["squeue", "scancel"] and cluster.builder_calls == []
    assert cluster.captures[0] == ("gw.example.org", "squeue --job 7654321 --noheader -o %T")
    assert line in result.output.splitlines()
    assert "Released SLURM job 7654321" in result.output


def test_release_on_an_unconfined_target_drains_a_running_job(tmp_path, monkeypatch):
    cluster = _Cluster(tmp_path, monkeypatch, confined=False, squeue="RUNNING")
    result = cluster.release("-f")
    assert result.exit_code == 0, (result.output, result.exception)
    assert cluster.events == ["squeue", "drain", "scancel"]
    assert "drained: sealed generation 000013-cafebabe (abababababab)" in result.output


@pytest.mark.parametrize("failure, line", [
    (MirrorError("Could not read latest-run for job 7654321 (exit 1): srun: error: Invalid job id specified"),
     "drain failed: Could not read latest-run for job 7654321 (exit 1): srun: error: Invalid job id specified; "
     "last durable generation unknown"),
    (typer.Exit(code=1), "drain failed: the managed owner could not be reached (exit 1); last durable generation unknown"),
    (RuntimeError("socket gone"), "drain failed: socket gone; last durable generation unknown"),
])
def test_release_proceeds_when_the_drain_itself_fails(tmp_path, monkeypatch, failure, line):
    cluster = _Cluster(tmp_path, monkeypatch, drain=failure)
    result = cluster.release("-f")
    assert result.exit_code == 0, (result.output, result.exception)
    assert cluster.events == ["drain", "scancel"]
    lines = result.output.splitlines()
    assert lines.index(line) < lines.index("Released SLURM job 7654321 on n0032.")
    assert cluster.job_id() is None


def test_release_sibling_detach_drains_the_departing_owner_before_its_tmux_kill(tmp_path, monkeypatch):
    cluster = _Cluster(tmp_path, monkeypatch, siblings=["other--fake-slurm"])
    result = cluster.release("-f")
    assert result.exit_code == 0, (result.output, result.exception)
    assert cluster.events == ["drain", "tmux-kill"] and cluster.captures == []
    assert "drained: sealed generation 000013-cafebabe (abababababab)" in result.output
    assert "Detached mirror sample from SLURM job 7654321" in result.output


def test_release_sibling_detach_that_stops_nothing_requests_nothing(tmp_path, monkeypatch):
    # No compute node recorded: the detach only forgets the allocation and
    # leaves the owner running, so it must not request a drain either.
    cluster = _Cluster(tmp_path, monkeypatch, siblings=["other--fake-slurm"],
                       session_yaml="login_node: ln002\nslurm_job_id: 7654321\n")
    result = cluster.release("-f")
    assert result.exit_code == 0, (result.output, result.exception)
    assert cluster.events == [] and cluster.builder_calls == []
    assert "Detached mirror sample from SLURM job 7654321" in result.output


def test_release_declined_at_the_prompt_requests_nothing(tmp_path, monkeypatch):
    cluster = _Cluster(tmp_path, monkeypatch)
    result = cluster.release(input="\n")
    assert result.exit_code == 0, (result.output, result.exception)
    assert "Aborted." in result.output and cluster.events == [] and cluster.builder_calls == []


# ----------------------------------------------------------------- ``codex-state-drain``

def _drain_cli(tmp_path, monkeypatch, args, *, report=None, job=4242):
    from sucoder import session as session_mod
    from tests.test_cli import _slurm_config

    path = _slurm_config(tmp_path)
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **k: None)
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    record = sessions_dir / "sample--fake-slurm.yaml"
    if job is not None:
        record.write_text(f"login_node: ln001\nslurm_job_id: {job}\ncompute_node: n0001\n", encoding="utf-8")
    elif record.exists():
        record.unlink()
    monkeypatch.setattr(session_mod, "_session_dir", lambda: sessions_dir)
    builder_calls, drain_calls = [], []
    if report is None:
        report = CodexStateDrain(job_id=4242, scope=SCOPE, mode="release", nonce=NONCE, commands=[], run=RUN, requested=True,
                                 ack=_ack(), outcome="drained: sealed generation 000013-cafebabe (abababababab)",
                                 last_durable=_fresh_placed()["last_generation"], waited_seconds=4.0, timeout_seconds=60.0)

    def codex_state_drain(mirror_ctx, job_id, **kw):
        drain_calls.append((mirror_ctx, job_id, kw))
        if isinstance(report, Exception):
            raise report
        return report

    def build(config, logger, dry_run, mirror, **kwargs):
        builder_calls.append((dry_run, mirror))
        return SimpleNamespace(context_for=lambda name: f"ctx:{name}", codex_state_drain=codex_state_drain)

    monkeypatch.setattr(cli, "_build_manager_for_mirror", build)
    argv = ["--config", str(path), "-T", "fake-slurm", "codex-state-drain", "sample", *args]
    return CliRunner().invoke(cli.app, argv), builder_calls, drain_calls


def test_drain_command_prints_the_outcome_line(tmp_path, monkeypatch):
    result, builder_calls, drain_calls = _drain_cli(tmp_path, monkeypatch, [])
    assert result.exit_code == 0, (result.output, result.exception)
    assert builder_calls == [(False, "sample")]
    (mirror_ctx, job_id, kw), = drain_calls
    assert (mirror_ctx, job_id, kw["mode"], kw["timeout_seconds"]) == ("ctx:sample", 4242, "release", 60.0)
    assert kw["requester"].startswith("sucoder codex-state-drain on ")
    assert result.output.strip() == f"drained: sealed generation 000013-cafebabe (abababababab) (job 4242, nonce {NONCE})"


def test_drain_command_json_is_the_whole_record_with_the_acknowledgement(tmp_path, monkeypatch):
    result, _, _ = _drain_cli(tmp_path, monkeypatch, ["--json", "--mode", "renew", "--timeout", "5"])
    assert result.exit_code == 0, (result.output, result.exception)
    data = json.loads(result.output)
    assert data["ack"] == _ack() and data["nonce"] == NONCE and data["run"] == RUN and data["requested"] is True
    assert data["outcome"].startswith("drained: sealed") and data["last_durable"] == _fresh_placed()["last_generation"]


def test_drain_command_passes_mode_and_timeout_through_and_refuses_a_bad_mode(tmp_path, monkeypatch):
    result, _, drain_calls = _drain_cli(tmp_path, monkeypatch, ["--mode", "renew", "--timeout", "5"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert drain_calls[0][2]["mode"] == "renew" and drain_calls[0][2]["timeout_seconds"] == 5.0
    result, builder_calls, _ = _drain_cli(tmp_path, monkeypatch, ["--mode", "pause"])
    assert result.exit_code == 2 and builder_calls == []
    assert "--mode must be one of release, renew" in result.output
    result, builder_calls, _ = _drain_cli(tmp_path, monkeypatch, ["--timeout", "-1"])
    assert result.exit_code == 2 and builder_calls == []


@pytest.mark.parametrize("report, code", [
    (CodexStateDrain(job_id=4242, scope=SCOPE, mode="release", nonce=NONCE, commands=[], run=RUN, requested=True,
                     outcome="drain unacknowledged after 60 s; last durable generation 000012-0a1b2c3d"), 1),
    (CodexStateDrain(job_id=4242, scope=SCOPE, mode="release", nonce=NONCE, commands=[], run=RUN, requested=True,
                     ack=_ack("failed", generation=None, reason="no final capture ran"),
                     outcome="drain failed: no final capture ran; last durable generation 000012-0a1b2c3d"), 1),
    (CodexStateDrain(job_id=4242, scope=SCOPE, mode="release", nonce=NONCE, commands=[], run=RUN, requested=True,
                     ack=_ack("warm_only", cls="warm", exit_code=None),
                     outcome="drained: warm generation 000013-cafebabe only (a person holds the terminal; it seals when they exit)"), 0),
    (CodexStateDrain(job_id=4242, scope=SCOPE, mode="release", nonce=NONCE, commands=[], outcome="no managed run recorded"), 0),
    (CodexStateDrain(job_id=4242, scope=SCOPE, mode="release", nonce=NONCE, commands=[], run=RUN,
                     outcome=f"no managed owner to drain: run {RUN} is refused; last durable generation none"), 0),
])
def test_drain_command_exit_status_follows_the_acknowledgement(tmp_path, monkeypatch, report, code):
    result, _, _ = _drain_cli(tmp_path, monkeypatch, [], report=report)
    assert result.exit_code == code, (result.output, result.exception)
    assert result.output.splitlines()[0].startswith(report.outcome)


def test_drain_command_dry_run_prints_the_commands(tmp_path, monkeypatch):
    report = CodexStateDrain(job_id=4242, scope=SCOPE, mode="release", nonce=NONCE, dry_run=True, outcome="dry run: nothing requested",
                             commands=[_srun(f"{SCOPE_DIR}/latest-run"), _srun(f"{SCOPE_DIR}/<run>/drain.ack")])
    result, builder_calls, _ = _drain_cli(tmp_path, monkeypatch, ["--dry-run"], report=report)
    assert result.exit_code == 0, (result.output, result.exception)
    assert builder_calls == [(True, "sample")]
    assert result.output.splitlines() == [
        f"DRY-RUN: a release-mode drain of job 4242 (scope {SCOPE}, nonce {NONCE}) would be requested and polled with:",
        f"  srun --jobid=4242 --overlap cat {SCOPE_DIR}/latest-run",
        f"  srun --jobid=4242 --overlap cat '{SCOPE_DIR}/<run>/drain.ack'",
    ]


def test_drain_command_reports_a_lookup_failure_and_a_missing_job(tmp_path, monkeypatch):
    failure = MirrorError("slurm.codex_state is not enabled for mirror sample; there is no managed Codex state to report.")
    result, _, _ = _drain_cli(tmp_path, monkeypatch, [], report=failure)
    assert result.exit_code == 1 and "not enabled for mirror sample" in result.output and "Traceback" not in result.output
    result, builder_calls, _ = _drain_cli(tmp_path, monkeypatch, [], job=None)
    assert result.exit_code == 0 and builder_calls == []
    assert "No SLURM allocation recorded for mirror sample (target fake-slurm); no managed Codex run to drain." in result.output
