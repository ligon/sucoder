"""Semantic review coverage and fail-closed decisions, separate from model quality."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sucoder import dream, dream_handles, dream_progress, dream_review as review, dream_run, dream_wiki as wiki
from tests.dream_submission_fixture import handle_reply
from tests.test_dream import git, workspace  # noqa: F401
from tests.test_dream_curate import NoopTransport, noop_plan
from tests.test_dream_wiki import EditorTransport, TOPIC, planned as current_plan


def planned(workspace):
    return current_plan(workspace, authoring_version=8)


def gate():
    # Synthetic public fixture: the failure types observed in E9, not its private text.
    prior = {wiki.INDEX: "One dated probe; other nodes remain unverified.\n", TOPIC:
        "On May 2 an agent reported a host failure. The publication outcome is unknown.\n"}
    pages = {wiki.INDEX: "Every attempt failed because of the operating system.\n", TOPIC:
        "All nodes are affected. Publication has not occurred.\n"}
    return review.manifest("run", dict(decision="change", task_actions=[]), pages, prior)


def accepted(value):
    return dict(page_checks=[dict(unit_id=unit["unit_id"], **{name: dict(status="pass",
        reason="A concrete fixture judgment about the supplied evidence.", evidence=[])
        for name in review.CATEGORIES}) for unit in value["units"]])


def test_diff_includes_index_deleted_caveats_and_final_newline_changes():
    value = gate()
    assert value["units"][0]["kind"] == "index"
    assert "-One dated probe; other nodes remain unverified." in value["units"][0]["diff"]
    assert "+Every attempt failed because" in value["units"][0]["diff"]
    assert "unknown" in value["units"][1]["diff"]
    other = review.manifest("run", {}, {TOPIC: "after"}, {TOPIC: "before"})
    assert "-before\n\\ No newline at end of file\n+after\n" in other["units"][0]["diff"]


@pytest.mark.parametrize("category", review.CATEGORIES)
@pytest.mark.parametrize("status", ["fail", "uncertain"])
def test_any_negative_check_blocks_even_an_accepting_verdict(category, status):
    value = gate()
    result = dict(accepted(value), verdict="accept", findings=[])
    result["page_checks"][0][category].update(status=status, reason="The dated report cannot establish this claim.")
    found = review.findings(result, value, set())
    assert len(found) == 1 and found[0]["path"] == wiki.INDEX
    assert found[0]["severity"] == "blocking" and category in found[0]["message"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "stale", "empty_reason", "unknown_handle"])
def test_matrix_must_cover_each_current_unit_with_valid_judgments(defect):
    value = gate()
    result = accepted(value)
    rows = result["page_checks"]
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif defect == "stale":
        rows[0]["unit_id"] = "change-from-another-candidate"
    elif defect == "empty_reason":
        rows[0]["scope"]["reason"] = " "
    else:
        rows[0]["support"]["evidence"] = [dict(handle="ev-undelivered")]
    assert review.findings(result, value, set())


def test_manifest_identity_binds_candidate_actions_prior_run_and_all_output_bytes():
    args = ["run", dict(decision="change", task_actions=[]), {TOPIC: "after\n"}, {TOPIC: "before\n"}]
    baseline = review.manifest(*args)
    for i, replacement in enumerate(["different-run", dict(decision="change", task_actions=["different"]),
                                    {TOPIC: "different\n"}, {TOPIC: "different\n"}]):
        changed = copy.deepcopy(args)
        changed[i] = replacement
        assert review.manifest(*changed)["units"][0]["unit_id"] != baseline["units"][0]["unit_id"]


@pytest.mark.parametrize("bound", ["MAX_UNITS", "DIFF_BYTES", "GATE_BYTES"])
def test_manifest_overflow_fails_without_omitting_pages(monkeypatch, bound):
    monkeypatch.setattr(review, bound, 1)
    with pytest.raises(dream.DreamError, match="allowance exhausted") as caught:
        gate()
    assert caught.value.exit_code == 3


def test_accepting_reviewer_cannot_override_bad_index_then_repair_is_reviewed(workspace):
    plan = planned(workspace)
    seen = []
    def mutate(task, result):
        if task["role"] == "author":
            next(page for page in result["pages"] if page["path"] == wiki.INDEX)["content"] += "All projects guarantee delivery.\n"
        if task["role"] == "reviewer":
            handle_reply(task, result)
            manifest = task["input"]["candidate"]["review_gate"]
            seen.append(manifest)
            for unit in manifest["units"]:
                if "All projects guarantee" in unit["diff"]:
                    row = next(row for row in result["page_checks"] if row["unit_id"] == unit["unit_id"])
                    row["scope"].update(status="fail", reason="A delivery rule for this project does not establish a guarantee for all projects.")
            assert result["verdict"] == "accept"  # The matrix still blocks the first candidate.
        elif task["role"] == "repair":
            assert any(f["path"] == wiki.INDEX and "Semantic scope fail" in f["message"] for f in task["input"]["findings"])
    native = EditorTransport(mutate=mutate)
    result = wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "claude", "fixture", transport=native)
    assert [task["role"] for task in native.calls] == ["author", "reviewer", "repair", "reviewer"]
    assert seen[0]["candidate_sha256"] != seen[1]["candidate_sha256"]
    receipt = json.loads(git(workspace[0], "show", result["candidate_commit"] + ":.sucoder/wiki/receipts/" + result["run_id"] + ".json"))
    assert receipt["semantic_review"] == seen[1]
    review.check_receipt(receipt)
    assert dream_run.status(workspace[2], "sample", Path(plan["plan_path"]))["state"] == "candidate_ready"
    assert len(native.calls) == 4  # Status is replay only.
    for defect in ("missing", "binding", "matrix", "hash", "malformed"):
        broken = copy.deepcopy(receipt)
        proof = broken["semantic_review"]
        if defect == "missing":
            proof["units"].pop()
        elif defect == "binding":
            proof["units"][0]["after_sha256"] = "0" * 64
        elif defect == "matrix":
            broken["review"]["page_checks"][0]["outcomes"]["status"] = "uncertain"
        elif defect == "malformed":
            proof["units"] = None
        if defect != "hash":
            proof["sha256"] = dream.digest(dream.canonical({k: v for k, v in proof.items() if k != "sha256"}))
        else:
            proof["sha256"] = "0" * 64
        with pytest.raises(dream.DreamError) as caught:
            review.check_receipt(broken)
        assert caught.value.exit_code == 2  # Corrupt shared metadata does not kill discovery.


def test_unchanged_has_review_unit_and_replays_without_inference(workspace):
    plan = noop_plan(workspace)
    native = NoopTransport()
    result = wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "claude", "fixture", transport=native)
    assert result["state"] == "unchanged"
    reviewer = native.calls[-1]
    units = reviewer["input"]["candidate"]["review_gate"]["units"]
    assert len(units) == 1 and units[0]["kind"] == "unchanged"
    assert dream_run.status(workspace[2], "sample", Path(plan["plan_path"]))["state"] == "unchanged"
    assert len(native.calls) == 2


def test_fresh_v8_does_not_reuse_earlier_review_contract():
    assert not dream_progress.envelope(dict(authoring_version=8), dict(contract=dream_handles.CONTRACT),
                                       "claude", "fixture", "version", "purpose")


def test_lookup_cannot_submit_premature_semantic_judgments(monkeypatch):
    monkeypatch.setattr(dream_handles.Editor, "check_result", lambda *args: None)
    value = dict(verdict="lookup", **accepted(gate()))
    with pytest.raises(dream.DreamError, match="premature semantic checks"):
        review.Editor.check_result(review.Editor.__new__(review.Editor), dict(role="reviewer"), value)


def test_v7_policy_and_schema_remain_frozen():
    assert "page_checks" not in dream_handles.schema("reviewer")["properties"]
    assert "candidate.review_gate" not in dream_handles.BRIEFS["reviewer"]
    assert "page_checks" in review.schema("reviewer")["properties"]


def test_review_diff_uses_full_frozen_pages_admitted_by_later_lookup():
    editor = review.Editor.__new__(review.Editor)
    editor.run_id, editor.materialized, editor.moves, editor.extra_prior = "run", {}, [], {}
    editor.context = SimpleNamespace(prior={TOPIC: "Later delivered frozen qualification.\n"})
    delivered = editor.review_candidate({}, {TOPIC: "Revised explanation.\n"})
    unit = delivered["review_gate"]["units"][0]
    assert unit["before_sha256"] == dream.digest(editor.context.prior[TOPIC].encode())
    assert "-Later delivered frozen qualification." in unit["diff"]
    delivered["review_gate"]["units"].clear()
    assert editor.review_gate["units"]  # Delivery cannot mutate the retained proof.


def test_unchanged_proof_allows_additional_frozen_reviewer_lookup_dependencies(monkeypatch):
    monkeypatch.setattr(dream_handles, "check_receipt", lambda receipt: None)
    prior = {wiki.INDEX: "Frozen index.\n"}
    proof = review.manifest("run", dict(decision="unchanged"), {}, prior)
    result = dict(assignment_id="assignment", input_sha256="input", role="reviewer", failures=[],
        verdict="accept", findings=[], requests=[], task_checks=[], **accepted(proof))
    receipt = dict(decision="unchanged", changes=[], semantic_review=proof, review=result,
        context=dict(evidence=dict(entries=[])), outputs=[dict(path=path,sha256=dream.digest(text.encode()))
            for path,text in dict(prior, **{TOPIC: "Full frozen page delivered during review.\n"}).items()])
    review.check_receipt(receipt)
    receipt["outputs"][0]["sha256"] = "0" * 64
    with pytest.raises(dream.DreamError, match="prior binding"):
        review.check_receipt(receipt)
