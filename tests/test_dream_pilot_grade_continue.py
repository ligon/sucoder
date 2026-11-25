"""Grading-only amendments retain the original reader/runtime and allowances."""
import copy
import json

import pytest

from scripts import dream_pilot_grade_continue as amended, dream_pilot_run as pilot
from sucoder import dream
from tests.test_dream_pilot_run import Transport, fixture_proposal, limits, questions


class NegativeGrade(Transport):
    def __call__(self, *args):
        result, usage, trace = super().__call__(*args)
        task = args[2]
        if task["role"] == "grade":
            for item in result["items"]:
                item.update(met="no", reason="The answer omits this required qualification.")
            if task["contract"] == pilot.CONTRACT:
                result["failures"] = ["The answer omits this required qualification."]
            events = [json.loads(line) for line in trace["stdout"].splitlines()]
            events[0]["item"]["text"] = json.dumps(result)
            trace["stdout"] = "\n".join(json.dumps(event) for event in events) + "\n"
        return result, usage, trace


def grade_inputs():
    sources = [dict(unit_id="unit-" + str(i), text="Fixture source " + str(i)) for i in range(2)]
    return dict(sources=sources, question=questions(sources)[0], answer="Incomplete static answer")


def test_complete_negative_grade_and_critical_result_remain_replayable(tmp_path):
    budget = pilot.Budget(tmp_path, dict(limits=limits()))
    native = NegativeGrade(critical=True)
    scoped = budget.transport(native, "grade-v2:0")
    result = amended.grade(tmp_path / "grading", "model", scoped, budget, grade_inputs(), 1)
    assert result["failures"] == []
    assert all(item["met"] == "no" for item in result["items"])
    assert result["critical_errors"][0]["occurred"] == "yes"
    assert amended.grade(tmp_path / "grading", "model", scoped, budget, grade_inputs(), 2) == result
    assert len(native.calls) == 1
    task = native.calls[0]
    assert task["contract"] == amended.CONTRACT and "failures: []" in task["brief"]
    assert "Only inability" in task["result_schema"]["properties"]["failures"]["description"]
    changed = dict(grade_inputs(), answer="Changed answer")
    with pytest.raises(dream.DreamError, match="Frozen amended grade"):
        amended.grade(tmp_path / "grading", "model", scoped, budget, changed, 2)
    assert len(native.calls) == 1


def test_genuine_grader_inability_is_not_accepted(tmp_path):
    class Unable(NegativeGrade):
        def __call__(self, *args):
            result, usage, trace = super().__call__(*args)
            result["failures"] = ["Required evidence is unavailable"]
            return result, usage, trace
    budget = pilot.Budget(tmp_path, dict(limits=limits()))
    native = Unable()
    for _ in range(2):
        with pytest.raises(dream.DreamError):
            amended.grade(tmp_path / "grading", "model", budget.transport(native, "grade"), budget, grade_inputs(), 1)
        assert not (tmp_path / "grading/validated.json").exists()
    assert len(native.calls) == 1


def test_amendment_guard_keeps_parent_guard_and_rejects_drift(tmp_path, monkeypatch):
    proposal = dict(limits=limits(), sources=[])
    budget = pilot.Budget(tmp_path, proposal)
    for name in ("final-wiki.json", "native-version.json"):
        pilot.store(tmp_path / name, {})
    calls = []
    budget.guard = lambda: calls.append("parent checked")
    frozen = amended.freeze_amendment(tmp_path, budget, dict(runtime="parent"), proposal)
    assert calls == ["parent checked"]
    assert frozen["parent_call_ids"] == []
    monkeypatch.setattr(amended, "GRADE_BRIEF", amended.GRADE_BRIEF + " changed")
    with pytest.raises(dream.DreamError, match="changed before transport"):
        budget.remaining()
    assert calls == ["parent checked", "parent checked"]


def test_existing_critical_grade_blocks_amendment(tmp_path):
    for directory in ("evaluator-only", "grading-v2"):
        path = tmp_path / directory / "SuCoder/grade-0/validated.json"
        pilot.store(path, dict(result=dict(critical_errors=[dict(occurred="yes")])))
        with pytest.raises(dream.DreamError, match="previously frozen critical"):
            amended.reject_prior_critical(tmp_path)


def test_negative_full_comparison_preserves_parent_answers_costs_and_replays(tmp_path, monkeypatch):
    proposal, blobs = fixture_proposal(tmp_path, monkeypatch)
    native = NegativeGrade()
    with pytest.raises(dream.DreamError, match="Role reported incomplete work"):
        pilot.run(proposal, blobs, execute=True, native=native)
    root = proposal.parent / "execution"
    parent_budget = pilot.read(root / "budget.json")
    parent_execution = (root / "execution-contract.json").read_bytes()
    parent_answer = (root / "answers/0.json").read_bytes()
    prior = copy.deepcopy(native.calls)
    result = amended.run(proposal, blobs, execute=True, native=native)
    assert result["state"] == "completed" and len(result["answers"]) == 36
    assert result["budget"]["deadline"] == parent_budget["deadline"]
    assert result["budget"]["started_at"] == parent_budget["started_at"]
    assert result["grading_amendment"]["parent_call_ids"] == [task["assignment_id"] for task in prior]
    assert len(result["budget"]["calls"]) == len(native.calls) <= 180
    assert (root / "execution-contract.json").read_bytes() == parent_execution
    assert (root / "answers/0.json").read_bytes() == parent_answer
    assert all(row["score"] == 0 for row in result["answers"])
    assert all(row["paired_record"].startswith("paired-v2-") for row in result["answers"])
    assert all(task["role"] in ("reader", "grade") for task in native.calls[len(prior):])
    count = len(native.calls)
    assert amended.run(proposal, blobs, execute=True, attempt=2, native=native) == result
    assert len(native.calls) == count
    monkeypatch.setattr(pilot.dream_reader_eval, "BRIEF", "Different reader instructions")
    with pytest.raises(dream.DreamError, match="runtime/role contract changed"):
        amended.run(proposal, blobs, execute=True, attempt=2, native=native)
    assert len(native.calls) == count
