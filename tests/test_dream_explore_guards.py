"""Fast lifecycle invariants for exploratory jobs and their shared budgets."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from sucoder import dream, dream_budget, dream_explore as explore, dream_native
from sucoder.config import DreamSettings
from tests.test_dream import git, workspace  # noqa: F401
from tests.test_dream_explore import ExploringTransport, options


def test_shared_wallet_policy_is_inherited_without_repeating_its_limits(workspace):
    repo, _, config, _ = workspace
    shared = repo.parent.parent / "shared-wallet"
    first = explore.prepare(config, cwd=repo, overrides=options(max_total_calls=12), budget_path=shared)
    second = explore.prepare(config, cwd=repo, overrides=options(), budget_path=shared)
    a, b = [json.loads((Path(job["job_path"]) / "job.json").read_bytes()) for job in (first, second)]
    assert a["settings"] == b["settings"] and b["settings"]["max_total_calls"] == 12
    with pytest.raises(dream.DreamError, match="immutable"):
        explore.prepare(config, cwd=repo, overrides=options(max_total_calls=13), budget_path=shared)


def test_original_wallet_cannot_be_recreated_under_the_same_job(workspace):
    repo, _, config, _ = workspace
    job = Path(explore.prepare(config, cwd=repo, overrides=options())["job_path"])
    wallet = job / "budget"
    wallet.rename(job / "retained-original-wallet")
    frozen = json.loads((job / "job.json").read_bytes())
    with dream_budget.open_job(explore._budget_plan(frozen), job, wallet, explore._limits(frozen["settings"])):
        pass
    with pytest.raises(dream.DreamError, match="original budget identity"):
        explore.run(config, job, transport=ExploringTransport(no_selection=True))


def test_cached_result_questions_cannot_be_substituted_for_native_evidence(workspace):
    repo, _, config, _ = workspace
    native = ExploringTransport(no_selection=True)
    result = explore.start(config, cwd=repo, overrides=options(), transport=native)
    path = Path(result["job_path"]) / "result-000001.json"
    altered = json.loads(path.read_bytes())
    altered["questions"] = ["A fabricated private question"]
    path.write_bytes(dream.canonical(altered))
    with pytest.raises(dream.DreamError, match="differs from its checked native evidence"):
        explore.status(config, Path(result["job_path"]))
    assert len(native.calls) == 1


@pytest.mark.parametrize("kind", ["detached", "unborn", "missing-model", "hard-money", "hard-tokens"])
def test_unusable_prerequisites_reject_before_native_startup(workspace, monkeypatch, kind):
    repo, _, config, _ = workspace
    request = options()
    if kind == "detached":
        git(repo, "checkout", "--detach", "-q")
    elif kind == "unborn":
        git(repo, "checkout", "--orphan", "unborn", "-q")
    elif kind == "missing-model":
        request.pop("model")
    elif kind == "hard-money":
        request["max_cost_usd"] = "1.00"
    else:
        request["max_total_tokens"] = 1000
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid prerequisite started a native harness")
    monkeypatch.setattr(dream_native, "Transport", forbidden)
    with pytest.raises(dream.DreamError):
        explore.start(config, cwd=repo, overrides=request)


def test_budget_too_small_for_discovery_and_author_makes_no_provider_call(workspace):
    repo, _, config, _ = workspace
    native = ExploringTransport(no_selection=True)
    with pytest.raises(dream_budget.Exhausted, match="exploration allowance"):
        explore.start(config, cwd=repo, overrides=options(max_total_calls=4), transport=native)
    assert not native.calls


@pytest.mark.parametrize("model", ["model\nwith-control", "x" * 4097])
def test_invalid_model_is_rejected_before_job_creation_or_native_startup(workspace, monkeypatch, model):
    repo, _, config, _ = workspace
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid model started native transport")
    monkeypatch.setattr(dream_native, "Transport", forbidden)
    with pytest.raises(dream.DreamError):
        explore.start(config, cwd=repo, overrides=dict(harness="codex", model=model))
    assert not (config.log_dir / "dream-jobs").exists()


def test_resumed_job_rejects_a_checkout_replaced_by_a_symlink_before_provider_work(workspace):
    repo, _, config, _ = workspace
    job = Path(explore.prepare(config, cwd=repo, overrides=options())["job_path"])
    retained = repo.with_name("retained-project")
    repo.rename(retained)
    repo.symlink_to(retained, target_is_directory=True)
    native = ExploringTransport(no_selection=True)
    with pytest.raises(dream.DreamError, match="frozen project path"):
        explore.run(config, job, transport=native)
    assert not native.calls
