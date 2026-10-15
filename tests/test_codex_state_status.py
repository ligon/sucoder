"""Managed Codex state, Slice 3b: the operator status surface.

``sucoder codex-state-status`` reads what the runner recorded for the
mirror's allocation -- ``latest-run``, then the run's ``status.json`` and
``owner.json`` -- through the existing in-allocation route and prints one
line per fact.  Three layers: the pure formatter against representative
``status.json`` dicts, the manager's lookup against a mocked executor (the
exact files read and the route taken), and the CLI pass-through.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from sucoder import cli
from sucoder.codex_state import (
    CodexStateStatus, describe_pending_work, format_status, local_scope_dir, scope_id,
)
from sucoder.executor import CommandError, CommandResult
from sucoder.mirror import MirrorError
from tests.test_cli import _slurm_config, _write_config
from tests.test_codex_state import _state_manager

RUN = "0123456789abcdef0123456789abcdef"
PARENT_RUN = "fedcba9876543210fedcba9876543210"
SCOPE = scope_id("cluster", "coder", "/global/scratch/u/mirrors/sample")
SCOPE_DIR = f"/local/job4242/state/codex/{SCOPE}"
DURABLE = f"/global/scratch/u/mirrors/.sucoder-state/codex/{SCOPE}/runs/{RUN}"


# ----------------------------------------------------------------- representative status.json dicts

def _generation(name, cls):
    return {"name": name, "class": cls, "published_at": 1760000000.0, "manifest_sha256": "ab" * 32,
            "path": f"{DURABLE}/generations/{name}", "trigger": "interval"}


def _fresh_placed():
    """A fresh run, placed, replicating, with one warm generation published."""
    return {
        "scope": SCOPE, "run": RUN, "db": f"{SCOPE_DIR}/{RUN}/db", "durable": DURABLE, "mode": "fresh",
        "reused_run": False, "restored_from": None, "pending_work": {"active_goals": [], "queued": {}},
        "state": "placed", "child_pid": 4321, "placed": [f"{SCOPE_DIR}/{RUN}/db/state_5.sqlite"],
        "replica": {"age_seconds": 3.2, "lag_bytes": 0,
                    "threads": {"t1": {"local_bytes": 42408, "replicated_bytes": 42408, "relative_path": "2026/r1.jsonl"},
                                "t2": {"local_bytes": 100, "replicated_bytes": 100, "relative_path": "2026/r2.jsonl"}},
                    "anomalies": {}, "errors": {}, "unrecognized": []},
        "checkpoint": {"state": "idle", "last_outcome": "published", "count": 12, "dropped": 0,
                       "interval_seconds": 600.0, "timeout_seconds": 30.0},
        "last_generation": _generation("000012-0a1b2c3d", "warm"), "unsynced_generation": None,
        "durable_age_seconds": 118.4,
        "transitions": [{"state": "starting", "at": 1.0}, {"state": "launched", "at": 2.0}, {"state": "placed", "at": 3.0}],
    }


def _restored_with_pending_work():
    """A restored run whose stores hold pending work, with a lagging, anomalous replica and an unsynced generation."""
    status = _fresh_placed()
    status.update({
        "mode": "restore", "state": "launched",
        "restored_from": {"run": PARENT_RUN, "generation": "000007-deadbeef", "class": "sealed", "manifest_sha256": "cd" * 32,
                          "threads": 2, "bytes": 1200000, "database_bytes": 1100000, "rollout_bytes": 100000,
                          "captured_at_iso": "2026-10-15T09:00:00Z", "codex_version": "0.12.0"},
        "pending_work": {"active_goals": ["t9"], "queued": {"t9": 2, "t3": 1}},
        "unsynced_generation": _generation("000003-77777777", "warm"),
        "durable_age_seconds": 1830.5,
    })
    status["replica"].update({"age_seconds": 45.0, "lag_bytes": 512, "anomalies": {"t9": "shrunk"}, "errors": {"t3": "EIO"}})
    status["checkpoint"].update({"state": "failed", "last_outcome": "not_durable", "dropped": 3, "reason": "replica t9 is shrunk"})
    return status


def _refused():
    """A refusal before the ``starting`` transition: only the state, the reason and the transitions."""
    return {"state": "refused", "reason": "bwrap is not installed on this node; managed placement is unavailable",
            "durable_age_seconds": None, "transitions": [{"state": "refused", "at": 1.0}]}


def _blocked():
    """A restore the durable side never answered: blocked after ``restoring``."""
    return {"state": "blocked", "restore": f"{PARENT_RUN}/000007-deadbeef", "source": f"{DURABLE}/generations/000007-deadbeef",
            "reason": "the durable side did not answer within 120 s", "durable_age_seconds": None,
            "transitions": [{"state": "restoring", "at": 1.0}, {"state": "blocked", "at": 2.0}]}


# ----------------------------------------------------------------- the formatter

def test_format_fresh_placed_run_one_line_per_fact_in_order():
    assert format_status(_fresh_placed(), run=RUN).splitlines() == [
        f"mode: fresh  run: {RUN}",
        "state: placed",
        "replica: age 3.2s, lag 0 bytes, 2 thread(s)",
        "last durable generation: 000012-0a1b2c3d (warm, age 118.4s)",
        "checkpoint: idle, last outcome published, 0 dropped",
        "pending work: none",
        f"durable run: {DURABLE}",
    ]


def test_format_restored_run_keeps_every_number_the_operator_acts_on():
    lines = format_status(_restored_with_pending_work(), run=RUN).splitlines()
    assert lines[0] == f"mode: restore  run: {RUN}"
    assert lines[1] == "state: launched"
    assert lines[2] == "replica: age 45s, lag 512 bytes, 2 thread(s); anomalies: t9: shrunk; errors: t3: EIO"
    assert lines[3] == "last durable generation: 000012-0a1b2c3d (warm, age 1830.5s); unsynced: 000003-77777777"
    assert lines[4] == "checkpoint: failed, last outcome not_durable, 3 dropped; reason: replica t9 is shrunk"
    assert lines[5] == "pending work: active goal on thread(s) t9; 1 queued item(s) on thread t3; 2 queued item(s) on thread t9"
    assert lines[6] == f"restored from: {PARENT_RUN}/000007-deadbeef (sealed, captured 2026-10-15T09:00:00Z)"
    assert lines[7] == f"durable run: {DURABLE}"
    assert len(lines) == 8


def test_format_refused_run_has_the_reason_and_no_invented_facts():
    assert format_status(_refused(), run=RUN).splitlines() == [
        f"mode: unknown  run: {RUN}",
        "state: refused",
        "replica: not reported yet",
        "last durable generation: none",
        "checkpoint: not reported yet",
        "pending work: not reported",
        "reason: bwrap is not installed on this node; managed placement is unavailable",
    ]


def test_format_blocked_restore_names_the_mode_from_the_restoring_transition():
    lines = format_status(_blocked(), run=RUN).splitlines()
    assert lines[0] == f"mode: unknown  run: {RUN}".replace("unknown", "restore")
    assert lines[1] == "state: blocked"
    assert lines[-1] == "reason: the durable side did not answer within 120 s"


def test_format_exited_and_misplaced_states_carry_their_detail():
    status = _fresh_placed()
    status.update({"state": "exited", "exit": 0, "outcome": "placed"})
    assert format_status(status, run=RUN).splitlines()[1] == "state: exited (exit 0, placed)"
    status.update({"state": "stopped_misplaced", "misplaced": ["/home/u/.codex/state_5.sqlite"]})
    assert format_status(status, run=RUN).splitlines()[1] == "state: stopped_misplaced (misplaced: /home/u/.codex/state_5.sqlite)"


def test_format_falls_back_to_owner_json_for_reuse_pending_work_and_restore_origin():
    status = _refused()      # a reuse refused after the earlier launch wrote owner.json
    owner = {"reused_run": True, "pending_work": {"active_goals": ["t1"], "queued": {}},
             "restored_from": {"run": PARENT_RUN, "generation": "000001-00000001", "class": "crash"}}
    lines = format_status(status, run=RUN, owner=owner).splitlines()
    assert lines[0] == f"mode: reuse  run: {RUN}"
    assert "pending work: active goal on thread(s) t1" in lines
    assert f"restored from: {PARENT_RUN}/000001-00000001 (crash)" in lines


def test_format_notes_an_ignored_restore_and_a_capture_in_flight():
    status = _fresh_placed()
    status["restore_ignored"] = f"{PARENT_RUN}/000007-deadbeef"
    status["checkpoint"]["in_flight"] = {"generation": "000013-0b0b0b0b", "trigger": "request", "class": "warm", "started": 5.0}
    lines = format_status(status, run=RUN).splitlines()
    assert lines[4] == "checkpoint: idle, last outcome published, 0 dropped; in flight: 000013-0b0b0b0b (request)"
    assert f"restore ignored: {PARENT_RUN}/000007-deadbeef (this allocation's run was reused instead)" in lines


@pytest.mark.parametrize("pending, text", [
    (None, "not reported"),
    ({"active_goals": [], "queued": {}}, "none"),
    ({"active_goals": ["a", "b"], "queued": {}}, "active goal on thread(s) a, b"),
    ({"active_goals": [], "queued": {"t": 3}}, "3 queued item(s) on thread t"),
    ({"active_goals": [], "queued": {}, "unreadable": ["goals_1.sqlite: locked"]},
     "a store the runner cannot read (goals_1.sqlite: locked)"),
])
def test_describe_pending_work(pending, text):
    assert describe_pending_work(pending) == text


def test_local_scope_dir_follows_the_local_tier_layout():
    assert local_scope_dir("/local", 4242, SCOPE) == SCOPE_DIR
    assert local_scope_dir("/local/", 4242, SCOPE) == SCOPE_DIR
    assert local_scope_dir("/", 7, "ab") == "/job7/state/codex/ab"


# ----------------------------------------------------------------- the lookup, through the manager

def _answers(files):
    """``run_agent`` answering ``cat <path>`` from *files*; a path not there fails the way cat does."""
    calls = []

    def run_agent(args, **kwargs):
        calls.append((list(args), kwargs))
        path = list(args)[-1]
        if path in files:
            return SimpleNamespace(returncode=0, stdout=files[path], stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr=f"cat: {path}: No such file or directory\n")

    return run_agent, calls


def _recorded_run(status=None, owner=None):
    run_dir = f"{SCOPE_DIR}/{RUN}"
    return {
        f"{SCOPE_DIR}/latest-run": RUN + "\n",
        f"{run_dir}/status.json": json.dumps(status if status is not None else _fresh_placed()),
        f"{run_dir}/owner.json": json.dumps(owner if owner is not None else {"run": RUN, "reused_run": False}),
    }


def _srun(path):
    return ["srun", "--jobid=4242", "--overlap", "cat", path]


def test_confined_lookup_reads_the_three_files_inside_the_allocation(tmp_path, monkeypatch):
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    manager.executor.run_agent.side_effect, calls = _answers(_recorded_run())
    # The manager has launched nothing: the agent type comes from the call,
    # not from a detection that never ran.
    assert not hasattr(manager, "_detected_agent_type")
    report = manager.codex_state_status(ctx, 4242)
    assert [argv for argv, _ in calls] == [
        _srun(f"{SCOPE_DIR}/latest-run"), _srun(f"{SCOPE_DIR}/{RUN}/status.json"), _srun(f"{SCOPE_DIR}/{RUN}/owner.json")]
    assert all(kw == {"check": False, "capture_output": True} for _, kw in calls)
    assert (report.job_id, report.scope, report.scope_dir, report.run, report.dry_run) == (4242, SCOPE, SCOPE_DIR, RUN, False)
    assert report.run_dir == f"{SCOPE_DIR}/{RUN}"
    assert report.status == _fresh_placed() and report.owner == {"run": RUN, "reused_run": False}
    assert report.commands == [argv for argv, _ in calls]
    assert not any("ssh" in argv[0] for argv, _ in calls)


def test_lookup_without_latest_run_reports_no_managed_run_after_one_read(tmp_path, monkeypatch):
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    manager.executor.run_agent.side_effect, calls = _answers({})
    report = manager.codex_state_status(ctx, 4242)
    assert report.run is None and report.status is None and report.owner is None and report.run_dir is None
    assert [argv for argv, _ in calls] == [_srun(f"{SCOPE_DIR}/latest-run")]


def test_unconfined_lookup_reads_through_the_executor_on_the_compute_node(tmp_path, monkeypatch):
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    ctx.settings.remote.slurm.confined = False
    manager.executor.run_agent.side_effect, calls = _answers(_recorded_run())
    report = manager.codex_state_status(ctx, 4242)
    assert [argv for argv, _ in calls] == [
        ["cat", f"{SCOPE_DIR}/latest-run"], ["cat", f"{SCOPE_DIR}/{RUN}/status.json"], ["cat", f"{SCOPE_DIR}/{RUN}/owner.json"]]
    assert report.run == RUN and report.status == _fresh_placed()


def test_dry_run_sends_the_first_read_and_lists_the_rest_with_a_placeholder(tmp_path, monkeypatch):
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    manager.executor.dry_run = True       # the fixture's run_agent answers rc 0 and empty stdout, as the executor does
    report = manager.codex_state_status(ctx, 4242)
    assert report.dry_run and report.run is None and report.status is None
    assert [c.args[0] for c in manager.executor.run_agent.call_args_list] == [_srun(f"{SCOPE_DIR}/latest-run")]
    assert report.commands == [
        _srun(f"{SCOPE_DIR}/latest-run"), _srun(f"{SCOPE_DIR}/<run>/status.json"), _srun(f"{SCOPE_DIR}/<run>/owner.json")]


def test_owner_json_is_optional_but_status_json_is_not(tmp_path, monkeypatch):
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    files = _recorded_run()
    del files[f"{SCOPE_DIR}/{RUN}/owner.json"]
    manager.executor.run_agent.side_effect, _ = _answers(files)
    assert manager.codex_state_status(ctx, 4242).owner is None
    del files[f"{SCOPE_DIR}/{RUN}/status.json"]
    with pytest.raises(MirrorError, match=f"{SCOPE_DIR}/{RUN}/status.json is missing although .*latest-run names run {RUN}"):
        manager.codex_state_status(ctx, 4242)


@pytest.mark.parametrize("name, text, message", [
    ("status.json", "{not json", "status.json for job 4242 is not readable JSON"),
    ("status.json", "[1, 2]", "status.json for job 4242 does not hold a JSON object"),
    ("owner.json", "{not json", "owner.json for job 4242 is not readable JSON"),
    ("latest-run", "../../etc\n", "latest-run for job 4242 names no run id"),
    ("latest-run", "", "latest-run for job 4242 names no run id"),
])
def test_lookup_reports_unreadable_records_with_their_path(tmp_path, monkeypatch, name, text, message):
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    files = _recorded_run()
    files[f"{SCOPE_DIR}/latest-run" if name == "latest-run" else f"{SCOPE_DIR}/{RUN}/{name}"] = text
    manager.executor.run_agent.side_effect, _ = _answers(files)
    with pytest.raises(MirrorError, match=message):
        manager.codex_state_status(ctx, 4242)


def test_lookup_reports_a_failed_read_rather_than_treating_it_as_absent(tmp_path, monkeypatch):
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    manager.executor.run_agent.return_value = SimpleNamespace(
        returncode=1, stdout="", stderr="srun: error: Unable to allocate resources: Invalid job id specified\n")
    with pytest.raises(MirrorError, match=f"Could not read {SCOPE_DIR}/latest-run for job 4242 \\(exit 1\\): srun: error: .*Invalid job id"):
        manager.codex_state_status(ctx, 4242)
    result = CommandResult(requested_args=["ssh"], executed_args=["ssh"], stdout="", stderr="ssh: connect: timed out", returncode=255)
    manager.executor.run_agent.side_effect = CommandError("Command ['ssh'] failed with code 255", result)
    with pytest.raises(MirrorError, match=f"Could not read {SCOPE_DIR}/latest-run for job 4242: Command .* failed with code 255"):
        manager.codex_state_status(ctx, 4242)


def test_lookup_refuses_a_target_the_launch_would_refuse(tmp_path, monkeypatch):
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch, enabled=False)
    with pytest.raises(MirrorError, match="slurm.codex_state is not enabled for mirror sample"):
        manager.codex_state_status(ctx, 4242)
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch, local_disk_root=None)
    with pytest.raises(MirrorError, match="local-disk tiering is off"):
        manager.codex_state_status(ctx, 4242)
    assert not manager.executor.run_agent.called


# ----------------------------------------------------------------- the CLI

def _cli(tmp_path, monkeypatch, args, *, job=4242, report=None, target=True):
    """Invoke ``codex-state-status`` with a stubbed manager; returns (result, builder calls, status calls)."""
    from sucoder import session as session_mod

    path = _slurm_config(tmp_path) if target else _write_config(tmp_path, skills_entry=tmp_path / "skills")
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **k: None)
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    if job is not None:
        (sessions_dir / "sample--fake-slurm.yaml").write_text(
            f"login_node: ln001\nslurm_job_id: {job}\ncompute_node: n0001\n", encoding="utf-8")
    monkeypatch.setattr(session_mod, "_session_dir", lambda: sessions_dir)
    builder_calls, status_calls = [], []
    if report is None:
        report = CodexStateStatus(job_id=4242, scope=SCOPE, scope_dir=SCOPE_DIR, commands=[], run=RUN,
                                  status=_fresh_placed(), owner={"run": RUN})

    def codex_state_status(mirror_ctx, job_id):
        status_calls.append((mirror_ctx, job_id))
        if isinstance(report, Exception):
            raise report
        return report

    manager = SimpleNamespace(context_for=lambda name: f"ctx:{name}", codex_state_status=codex_state_status)

    def build(config, logger, dry_run, mirror, **kwargs):
        builder_calls.append((dry_run, mirror))
        return manager

    monkeypatch.setattr(cli, "_build_manager_for_mirror", build)
    argv = ["--config", str(path)] + (["-T", "fake-slurm"] if target else []) + ["codex-state-status", "sample", *args]
    return CliRunner().invoke(cli.app, argv), builder_calls, status_calls


def test_cli_prints_the_summary_for_the_recorded_job(tmp_path, monkeypatch):
    result, builder_calls, status_calls = _cli(tmp_path, monkeypatch, [])
    assert result.exit_code == 0, (result.output, result.exception)
    assert builder_calls == [(False, "sample")] and status_calls == [("ctx:sample", 4242)]
    lines = result.output.splitlines()
    assert lines[0] == f"managed Codex state for mirror sample: job 4242, scope {SCOPE}"
    assert lines[1] == f"path: {SCOPE_DIR}/{RUN}"
    assert lines[2:] == format_status(_fresh_placed(), run=RUN, owner={"run": RUN}).splitlines()


def test_cli_json_prints_status_json_as_recorded(tmp_path, monkeypatch):
    result, _, _ = _cli(tmp_path, monkeypatch, ["--json"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(result.output) == _fresh_placed()


def test_cli_dry_run_builds_a_dry_manager_and_prints_the_reads(tmp_path, monkeypatch):
    report = CodexStateStatus(job_id=4242, scope=SCOPE, scope_dir=SCOPE_DIR, dry_run=True,
                              commands=[_srun(f"{SCOPE_DIR}/latest-run"), _srun(f"{SCOPE_DIR}/<run>/status.json")])
    result, builder_calls, _ = _cli(tmp_path, monkeypatch, ["--dry-run"], report=report)
    assert result.exit_code == 0, (result.output, result.exception)
    assert builder_calls == [(True, "sample")]
    # shlex.join quotes the placeholder (``<`` is shell syntax), so the line
    # reads back as the one command it stands for.
    assert result.output.splitlines() == [
        f"DRY-RUN: managed Codex state for job 4242 (scope {SCOPE}) would be read with:",
        f"  srun --jobid=4242 --overlap cat {SCOPE_DIR}/latest-run",
        f"  srun --jobid=4242 --overlap cat '{SCOPE_DIR}/<run>/status.json'",
    ]


def test_cli_without_a_recorded_job_says_so_before_building_anything(tmp_path, monkeypatch):
    result, builder_calls, status_calls = _cli(tmp_path, monkeypatch, [], job=None)
    assert result.exit_code == 0, (result.output, result.exception)
    assert "No SLURM allocation recorded for mirror sample (target fake-slurm); no managed Codex run to report." in result.output
    assert builder_calls == [] and status_calls == []


def test_cli_without_a_managed_run_is_a_clean_message_not_an_error(tmp_path, monkeypatch):
    report = CodexStateStatus(job_id=4242, scope=SCOPE, scope_dir=SCOPE_DIR, commands=[_srun(f"{SCOPE_DIR}/latest-run")])
    result, _, _ = _cli(tmp_path, monkeypatch, [], report=report)
    assert result.exit_code == 0, (result.output, result.exception)
    assert result.output.startswith(f"no managed run recorded for job 4242 (scope {SCOPE}): {SCOPE_DIR}/latest-run is absent")


def test_cli_reports_a_lookup_failure_as_exit_1_without_a_traceback(tmp_path, monkeypatch):
    failure = MirrorError(f"Could not read {SCOPE_DIR}/latest-run for job 4242 (exit 1): srun: error: Invalid job id specified")
    result, _, _ = _cli(tmp_path, monkeypatch, [], report=failure)
    assert result.exit_code == 1, (result.output, result.exception)
    assert "Could not read" in result.output and "Invalid job id specified" in result.output
    assert "Traceback" not in result.output


def test_cli_refuses_a_mirror_without_a_slurm_target(tmp_path, monkeypatch):
    result, builder_calls, _ = _cli(tmp_path, monkeypatch, [], target=False)
    assert result.exit_code == 1, (result.output, result.exception)
    assert "Mirror sample has no SLURM target" in result.output and builder_calls == []
