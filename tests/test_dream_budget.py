"""Whole-job budgets persist costs, advice and deadlines across native attempts."""
import copy
import json
from pathlib import Path
import time

import pytest
from typer.testing import CliRunner

from sucoder import cli, dream, dream_budget as budget, dream_native, dream_run, dream_wiki as wiki
from tests.test_dream import workspace  # noqa: F401
from tests.test_dream_wiki import EditorTransport, planned


class Clock:
    def __init__(self):
        self.wall, self.monotonic, self.boot_id = 10000.0, 2000.0, "fixture-boot"

    def __call__(self):
        return dict(wall=self.wall, monotonic=self.monotonic, boot_id=self.boot_id)

    def advance(self, seconds):
        self.wall += seconds
        self.monotonic += seconds


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(budget, "_clock", value)
    return value


def plan():
    return dict(limits=dict(max_elapsed_seconds=100, max_role_invocations=4), repositories={})


def task(label, role="author"):
    value = dict(run_id=dream.digest(label.encode()), role=role, input=dict(label=label), brief="Fixture",
        input_sha256=dream.digest(label.encode()), result_schema={})
    value["assignment_id"] = dream.digest(dream.canonical(value))
    return value


class Native:
    version = "budget-fixture-1"

    def __init__(self, action=None, failed=False):
        self.calls, self.action, self.failed = [], action, failed

    def __call__(self, harness, model, assignment, schema, deadline):
        self.calls.append(copy.deepcopy(assignment))
        if self.action:
            self.action()
        result = {k: assignment[k] for k in ("assignment_id", "input_sha256", "role")}
        result["failures"] = []
        usage = dict(native=dict(input_tokens=11, output_tokens=3), currency_usd=None)
        events = [dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(result))),
            dict(type="turn.completed", usage=usage["native"])]
        trace = dict(harness=harness, model=model, version=self.version, returncode=1 if self.failed else 0,
            input_sha256=dream.digest(dream.canonical(assignment)), stdout="\n".join(json.dumps(e) for e in events),
            stderr="", boundary="linux-landlock-inline-v1")
        return (None, None, trace) if self.failed else (result, usage, trace)


def invoke(job, native, value):
    value = job.assignment(value)
    job.reserve(value)
    return job.transport(native)("codex", "fixture-model", value, {}, time.monotonic() + 100)


def test_reservations_and_reported_usage_survive_restart(tmp_path, clock):
    native = Native()
    with budget.open_job(plan(), tmp_path) as job:
        first = job.descriptor()
        invoke(job, native, task("first"))
        frozen = job.assignment(task("first"))
    clock.advance(4)
    with budget.open_job(plan(), tmp_path) as job:
        assert job.descriptor() == first
        assert job.assignment(task("first")) == frozen
        assert job.summary()["reserved_calls"] == 1
        assert job.summary()["usage"]["reported_input_tokens"] == 11
        assert job.summary()["usage"]["reported_output_tokens"] == 3
        assert job.summary()["usage"]["reported_cost_usd"] is None
        with pytest.raises(dream.DreamError, match="already reserved"):
            job.reserve(frozen)
    assert len(native.calls) == 1


def test_unknown_reservation_stays_charged_and_cannot_be_reissued(tmp_path, clock):
    with budget.open_job(plan(), tmp_path) as job:
        frozen = job.assignment(task("interrupted"))
        job.reserve(frozen)
    native = Native()
    with budget.open_job(plan(), tmp_path) as job:
        value = job.assignment(task("interrupted"))
        with pytest.raises(dream.DreamError, match="silently repeated"):
            job.transport(native)("codex", "fixture-model", value, {}, time.monotonic() + 100)
        assert job.summary()["call_states"]["uncertain"] == 1
        assert job.summary()["usage"]["calls_without_token_usage"] == 1
        invoke(job, native, task("explicit-new-attempt"))
        assert job.summary()["reserved_calls"] == 2
    assert len(native.calls) == 1


def test_closeout_reserve_and_total_cap_are_enforced(tmp_path, clock):
    native = Native()
    with budget.open_job(plan(), tmp_path, supplied=dict(max_native_calls=3, reserve_calls=1)) as job:
        invoke(job, native, task("author-1"))
        invoke(job, native, task("author-2"))
        with pytest.raises(budget.Exhausted, match="reserved for review"):
            invoke(job, native, task("extra-author"))
        invoke(job, native, task("independent-check", "reviewer"))
        with pytest.raises(budget.Exhausted, match="invocation allowance"):
            invoke(job, native, task("extra-review", "reviewer"))
        assert job.summary()["reserved_calls"] == 3
    assert len(native.calls) == 3


def test_failed_and_uncertain_native_calls_remain_in_usage(tmp_path, clock):
    def interrupted():
        raise RuntimeError("unknown native outcome")
    with budget.open_job(plan(), tmp_path) as job:
        invoke(job, Native(failed=True), task("failed"))
        with pytest.raises(RuntimeError):
            invoke(job, Native(action=interrupted), task("uncertain"))
        summary = job.summary()
        assert summary["reserved_calls"] == 2
        assert summary["call_states"]["failed"] == summary["call_states"]["uncertain"] == 1
        assert summary["usage"]["calls_without_token_usage"] == 2
        assert summary["usage"]["reported_cost_usd"] is None


def test_expired_deadline_cannot_be_reset_or_revived_by_clock_rollback(tmp_path, clock):
    with budget.open_job(plan(), tmp_path) as job:
        descriptor = job.descriptor()
    clock.advance(101)
    clock.wall -= 50  # Same boot: monotonic time still proves expiry.
    with budget.open_job(plan(), tmp_path) as job:
        with pytest.raises(budget.Exhausted):
            invoke(job, Native(), task("expired"))
        assert job.header["deadline"] == 10100
    with budget.restore(descriptor) as job:
        assert job.summary()["state"] == "budget_exhausted"
    with pytest.raises(dream.DreamError, match="immutable"):
        with budget.open_job(plan(), tmp_path, supplied=dict(max_elapsed_seconds=200)):
            pass


def test_clock_domain_change_refuses_paid_work_but_allows_inspection(tmp_path, clock):
    with budget.open_job(plan(), tmp_path) as job:
        descriptor = job.descriptor()
    clock.boot_id = "another-boot"
    with budget.open_job(plan(), tmp_path) as job:
        with pytest.raises(budget.Exhausted):
            invoke(job, Native(), task("rebooted"))
    with budget.restore(descriptor) as job:
        assert job.summary()["reserved_calls"] == 0


def test_work_deadline_timeout_is_budget_exhausted_with_closeout_time_left(tmp_path, clock):
    with budget.open_job(plan(), tmp_path, supplied=dict(reserve_seconds=20)) as job:
        native = Native(action=lambda: clock.advance(80), failed=True)
        with pytest.raises(budget.Exhausted, match="closeout time remains"):
            invoke(job, native, task("work-cutoff"))
        assert job.remaining() == 20
        assert job.summary()["reserved_calls"] == 1
        assert list(job.path.glob("stop-*.json"))


def test_time_exhaustion_outside_transport_preserves_original_diagnostic(tmp_path, clock):
    with pytest.raises(budget.Exhausted, match="snapshot diagnostic"):
        with budget.open_job(plan(), tmp_path):
            clock.advance(101)
            raise dream.DreamError("snapshot diagnostic")


def test_budget_lock_prevents_concurrent_coordinators(tmp_path, clock):
    with budget.open_job(plan(), tmp_path):
        with pytest.raises(dream.DreamError, match="Another coordinator"):
            with budget.open_job(plan(), tmp_path):
                pass


@pytest.mark.parametrize("defect", ["policy", "advice", "usage", "gap"])
def test_corrupt_budget_artifacts_cannot_start_another_request(tmp_path, clock, defect):
    native = Native()
    with budget.open_job(plan(), tmp_path) as job:
        invoke(job, native, task("first"))
        root = job.path
    if defect == "gap":
        (root / "reservation-000000.json").rename(root / "reservation-000001.json")
    else:
        name = "policy.json" if defect == "policy" else "returned-000000.json" if defect == "usage" else next(root.glob("advice-*.json")).name
        path = root / name
        value = json.loads(path.read_bytes())
        if defect == "policy":
            value["limits"]["max_native_calls"] += 10
        elif defect == "usage":
            value["usage"]["native"]["input_tokens"] = 0
        else:
            value["advice"]["instructions"] = "Unbound new instructions"
        path.write_bytes(dream.canonical(value))
    with pytest.raises(dream.DreamError):
        with budget.open_job(plan(), tmp_path) as job:
            invoke(job, native, task("first"))
    assert len(native.calls) == 1


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_shared_wallet_and_expired_cached_replay_keep_completed_receipts_stable(workspace, monkeypatch, clock, harness):
    first = planned(workspace)
    shared = workspace[1].parent / "shared-budget"
    transport = EditorTransport()
    policy = dict(max_native_calls=4, reserve_calls=1, max_elapsed_seconds=100)
    a = wiki.run(workspace[2], "sample", Path(first["plan_path"]), harness, "model", transport=transport,
        budget_path=shared, budget_limits=policy)
    second = dream.create_plan(workspace[2], "sample", workspace[1].parent / "sources.json", "refs/heads/main",
        maintenance="Independent explicit pass", topic="reply-evidence")
    b = wiki.run(workspace[2], "sample", Path(second["plan_path"]), harness, "model", transport=transport, budget_path=shared)
    assert a["run_id"] != b["run_id"] and len(transport.calls) == 4
    assert wiki.run(workspace[2], "sample", Path(first["plan_path"]), harness, "model", transport=transport, budget_path=shared) == a
    clock.advance(101)
    assert wiki.run(workspace[2], "sample", Path(first["plan_path"]), harness, "model", transport=transport, budget_path=shared) == a
    assert dream_run.status(workspace[2], "sample", Path(first["plan_path"]), a["run_id"])["state"] == "candidate_ready"
    assert budget.inspect(workspace[2], "sample", Path(first["plan_path"]), shared)["reserved_calls"] == 4
    assert len(transport.calls) == 4


@pytest.mark.parametrize("harness", ["codex", "claude"])
@pytest.mark.parametrize("option,value", [("--max-total-tokens", "1000"), ("--max-cost-usd", "1.00")])
def test_unsupported_hard_limits_fail_before_native_startup(workspace, monkeypatch, harness, option, value):
    frozen = planned(workspace)
    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported hard cap reached native startup")
    monkeypatch.setattr(dream_native, "Transport", forbidden)
    result = CliRunner().invoke(cli.app, ["--config", str(workspace[1]), "dream", "run", "sample", "--plan", frozen["plan_path"],
        "--harness", harness, "--model", "model", option, value], input="")
    assert result.exit_code == 3, result.output
    assert "cannot enforce a hard token or currency" in json.loads(result.output)["error"]
    assert not (Path(frozen["plan_path"]).parent / "job-budget").exists()


def test_cli_exhaustion_is_structured_and_does_not_create_a_started_native_call(workspace, monkeypatch, clock):
    frozen = planned(workspace)
    native = EditorTransport()
    monkeypatch.setattr(dream_native, "Transport", lambda *args: native)
    result = CliRunner().invoke(cli.app, ["--config", str(workspace[1]), "dream", "run", "sample", "--plan", frozen["plan_path"],
        "--harness", "codex", "--model", "model", "--max-total-calls", "1"], input="")
    assert result.exit_code == 3, result.output
    value = json.loads(result.output)
    assert value["state"] == "budget_exhausted" and value["budget"]["reserved_calls"] == 0
    assert not native.calls
    assert not list(Path(frozen["plan_path"]).parent.glob("*-started.json"))


def test_time_slipping_after_reservation_is_denied_without_unknown_provider_outcome(tmp_path, clock):
    native = Native()
    with budget.open_job(plan(), tmp_path) as job:
        value = job.assignment(task("denied"))
        job.reserve(value)
        clock.advance(101)
        with pytest.raises(budget.Exhausted, match="before provider startup"):
            job.transport(native)("codex", "model", value, {}, time.monotonic() + 100)
    with budget.open_job(plan(), tmp_path) as job:
        assert job.summary()["call_states"]["denied_before_provider"] == 1
        assert job.summary()["call_states"]["uncertain"] == 0
        with pytest.raises(budget.Exhausted, match="denied before provider"):
            job.denied(value)
    assert not native.calls


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_notes_consolidation_can_spend_last_reserved_call_and_expired_status_replays(workspace, clock, harness):
    from sucoder import dream_roles
    from tests.test_dream_roles import Transport, plan as notes_plan
    frozen = notes_plan(workspace)
    native = Transport()
    result = wiki.run(workspace[2], "sample", Path(frozen["plan_path"]), harness, "model", mode="notes",
        transport=native, budget_limits=dict(max_native_calls=3))
    assert [a["role"] for a in native.calls] == ["extractor", "verifier", "consolidator"]
    assert all("budget" in a for a in native.calls)
    clock.advance(901)
    assert dream_roles.run(workspace[2], "sample", Path(frozen["plan_path"]), harness, "model", transport=native) == result
    assert dream_run.status(workspace[2], "sample", Path(frozen["plan_path"]))["state"] == "candidate_ready"
    assert len(native.calls) == 3


@pytest.mark.parametrize("phase", ["snapshot", "publish"])
def test_phase_deadline_stops_without_candidate_and_preserves_diagnostic(workspace, monkeypatch, clock, phase):
    frozen = planned(workspace)
    native = EditorTransport()
    def expired(*args, **kwargs):
        clock.advance(101)
        raise dream.DreamError(phase + " deadline detail", 3)
    monkeypatch.setattr(dream_run, "_snapshot" if phase == "snapshot" else "publish", expired)
    with pytest.raises(budget.Exhausted, match=phase + " deadline detail"):
        wiki.run(workspace[2], "sample", Path(frozen["plan_path"]), "codex", "model", transport=native,
            budget_limits=dict(max_elapsed_seconds=100))
    directory = Path(frozen["plan_path"]).parent
    assert not list(directory.glob("candidate-*.txt"))
    summary = budget.inspect(workspace[2], "sample", Path(frozen["plan_path"]))
    assert summary["state"] == "budget_exhausted"
    assert len(native.calls) == (0 if phase == "snapshot" else 2)


def test_expired_already_integrated_result_revalidates_without_native_startup(workspace, monkeypatch, clock):
    from tests.test_dream_progress import accepted, NoCalls
    frozen, _ = accepted(workspace)
    path = Path(frozen["plan_path"])
    result = wiki.run(workspace[2], "sample", path, "codex", "explicit-model", transport=NoCalls())
    assert result["state"] == "already_integrated"
    clock.advance(901)
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed reuse started a native harness after expiry")
    monkeypatch.setattr(dream_native, "Transport", forbidden)
    assert wiki.run(workspace[2], "sample", path, "codex", "explicit-model") == result
    assert budget.inspect(workspace[2], "sample", path)["reserved_calls"] == 0
