"""Long valid reasons no longer need native schema retries; overall bounds stay."""
import json
from pathlib import Path

import pytest

from sucoder import dream, dream_concise as concise, dream_progress, dream_review, dream_roles, dream_run, dream_wiki
from tests.dream_submission_fixture import handle_reply
from tests.test_dream import git, workspace  # noqa: F401
from tests.test_dream_review import accepted, gate
from tests.test_dream_wiki import EditorTransport, planned


LONG_REASON = ("The source retains its project, date and uncertainty; no later outcome is established. " * 12)[:936]


def test_v9_accepts_a_long_reason_while_the_v8_schema_stays_frozen():
    value = dict(assignment_id="assignment", input_sha256="input", role="reviewer", failures=[],
        verdict="accept", findings=[], requests=[], task_checks=[], **accepted(gate()))
    value["page_checks"][0]["support"]["reason"] = LONG_REASON
    dream_roles.validate_schema(value, concise.schema("reviewer"))
    with pytest.raises(dream.DreamError, match="Role text exceeds its allowance"):
        dream_roles.validate_schema(value, dream_review.schema("reviewer"))
    assert "at most 40 words" in concise.BRIEFS["reviewer"]
    assert "at most 40 words" not in dream_review.BRIEFS["reviewer"]
    assert concise.schema("author") == dream_review.schema("author")
    assert concise.schema("repair") == dream_review.schema("repair")


def test_long_reason_survives_receipt_native_trace_and_read_only_replay(workspace):
    plan = planned(workspace)
    assert plan["authoring_version"] == 9
    def mutate(task, result):
        if task["role"] == "reviewer":
            handle_reply(task, result)
            result["page_checks"][0]["support"]["reason"] = LONG_REASON
    native = EditorTransport(mutate=mutate)
    result = dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "claude", "fixture", transport=native)
    assert result["state"] == "candidate_ready"
    receipt = json.loads(git(workspace[0], "show", result["candidate_commit"] + ":.sucoder/wiki/receipts/" + result["run_id"] + ".json"))
    assert receipt["contract"] == concise.CONTRACT
    assert receipt["review"]["page_checks"][0]["support"]["reason"] == LONG_REASON
    concise.check_receipt(receipt)
    with pytest.raises(dream.DreamError, match="Role text exceeds its allowance"):
        dream_review.check_receipt(receipt)
    assert dream_run.status(workspace[2], "sample", Path(plan["plan_path"]))["state"] == "candidate_ready"
    assert len(native.calls) == 2
    rows = [json.loads(p.read_bytes()) for p in Path(plan["plan_path"]).parent.glob("*-result.json")]
    saved = next(row for row in rows if row.get("result", {}).get("role") == "reviewer")
    assert saved["result"]["page_checks"][0]["support"]["reason"] == LONG_REASON


def test_relaxed_field_limits_keep_the_existing_cumulative_reply_byte_ceiling(workspace):
    plan = planned(workspace)
    def mutate(task, result):
        if task["role"] == "reviewer":
            handle_reply(task, result)
            for row in result["page_checks"]:
                for name in dream_review.CATEGORIES:
                    row[name]["reason"] = "x" * dream_roles.TEXT["maxLength"]
    with pytest.raises(dream.DreamError, match="Cumulative role results exceed max_proposal_bytes"):
        dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "claude", "fixture",
            transport=EditorTransport(mutate=mutate))
    assert not git(workspace[0], "for-each-ref", "refs/heads/dream/")


@pytest.mark.parametrize("version,contract", [(9, dream_review.CONTRACT), (8, concise.CONTRACT), (7, concise.CONTRACT)])
def test_new_and_frozen_plans_keep_their_own_acceptance_contract(version, contract):
    assert not dream_progress.envelope(dict(authoring_version=version), dict(contract=contract),
        "claude", "fixture", "version", "purpose")


def test_v8_receipt_callback_uses_its_original_validator(monkeypatch):
    seen = []
    monkeypatch.setattr(dream_review, "check_receipt", lambda receipt: seen.append(receipt))
    value = dict(frozen="v8")
    dream_review.Editor.receipt_check(None, value)
    assert seen == [value]
