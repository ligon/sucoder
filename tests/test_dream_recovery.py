"""Explicit recovery preserves identities, original call caps and replay."""
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sucoder import cli, dream, dream_budget, dream_explore as explore, dream_roles
from tests.test_dream import workspace  # noqa: F401
from tests.test_dream_explore import ExploringTransport, clock, options  # noqa: F401
from tests.test_dream_budget import Native, invoke, plan, task


@pytest.mark.parametrize("field", ["role", "assignment_id", "input_sha256"])
def test_identity_error_and_failure_record_name_role_field_trace_and_prefixes(workspace, field):
    repo, _, config, _ = workspace
    native = ExploringTransport(no_selection=True)
    original = native.__call__
    class Mismatch:
        version = native.version
        def __call__(self, *args):
            result, usage, trace = original(*args)
            result[field] = "incorrect-value"
            return result, usage, trace
    job = explore.prepare(config, cwd=repo, overrides=options())
    with pytest.raises(dream.DreamError) as error:
        explore.run(config, job["job_path"], transport=Mismatch())
    message = str(error.value)
    assert "role=explorer" in message and f"field={field}" in message
    assert "expected=" in message and "received=incorrect-va" in message and "-trace.json" in message
    trace = message.split("private trace: ")[1]
    assert (Path(job["job_path"]) / trace).is_file()
    failure = json.loads(next(Path(job["job_path"]).glob("failure-*.json")).read_text())
    assert failure["error"] == message


def test_job_id_resolves_for_resume_status_and_budget(workspace, monkeypatch):
    repo, path, config, _ = workspace
    native = ExploringTransport(no_selection=True)
    job = explore.start(config, cwd=repo, overrides=options(), transport=native)
    monkeypatch.chdir(repo)
    assert explore.start(config, job["job_id"]) == job
    assert explore.status(config, job["job_id"])["state"] == "no_selection"
    assert explore.budget_status(config, job["job_id"])["reserved_calls"] == 1
    for command in (["--job", job["job_id"]], ["status", "--job", job["job_id"]], ["budget", "--job", job["job_id"]]):
        result = CliRunner().invoke(cli.app, ["--config", str(path), "dream", *command])
        assert result.exit_code == 0, result.output
    assert len(native.calls) == 1


def test_expired_job_requires_explicit_bounded_retry_window(workspace, clock):
    repo, _, config, _ = workspace
    native = ExploringTransport(fail=True)
    job = explore.prepare(config, cwd=repo, overrides=options())
    directory = Path(job["job_path"])
    policy = (directory / "budget/policy.json").read_bytes()
    with pytest.raises(dream.DreamError):
        explore.run(config, directory, transport=native)
    assert len(native.calls) == 1
    clock.advance(901)
    recovered = ExploringTransport(no_selection=True)
    with pytest.raises(dream_budget.Exhausted):
        explore.start(config, job["job_id"], attempt=2, transport=recovered)
    assert not recovered.calls
    result = explore.start(config, job["job_id"], attempt=2, retry_seconds=120, transport=recovered)
    assert result["state"] == "no_selection" and len(recovered.calls) == 1
    assert (directory / "budget/policy.json").read_bytes() == policy
    balance = explore.budget_status(config, job["job_id"])
    assert balance["reserved_calls"] == 2 and balance["remaining_calls"] == 38
    assert balance["retry_windows"][0]["seconds"] == 120
    clock.advance(121)
    assert explore.start(config, job["job_id"], attempt=2) == result
    assert explore.status(config, job["job_id"])["state"] == "no_selection"


def test_retry_window_never_replenishes_call_cap_or_its_own_deadline(tmp_path, clock):
    native = Native()
    with dream_budget.open_job(plan(), tmp_path, supplied=dict(max_native_calls=1, reserve_calls=0)) as budget:
        invoke(budget, native, task("first"))
        original = budget.descriptor()
    clock.advance(101)
    with dream_budget.open_job(plan(), tmp_path) as budget:
        budget.renew(2, 60)
        deadline = budget.summary()["deadline"]
        assert budget.descriptor() == original
        with pytest.raises(dream_budget.Exhausted, match="invocation allowance"):
            invoke(budget, native, task("second"))
    clock.advance(61)
    with dream_budget.open_job(plan(), tmp_path) as budget:
        budget.renew(2, 60)
        assert budget.summary()["deadline"] == deadline and budget.remaining() == 0
        with pytest.raises(dream.DreamError, match="immutable"):
            budget.renew(2, 90)
    assert len(native.calls) == 1


@pytest.mark.parametrize("job,attempt,seconds", [(None, 2, 60), ("job", 1, 60), ("job", 2, 0), ("job", 2, 86401)])
def test_invalid_retry_request_rejects_before_job_creation(workspace, job, attempt, seconds):
    with pytest.raises(dream.DreamError, match="retry-seconds"):
        explore.prepare(workspace[2], job, cwd=workspace[0], attempt=attempt, retry_seconds=seconds)
    assert not (workspace[2].log_dir / "dream-jobs").exists()
