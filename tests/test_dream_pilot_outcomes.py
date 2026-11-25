"""A negative terminal reader outcome is retained, never retried for quality."""
import json

import pytest

from scripts import dream_pilot_outcomes as outcomes, dream_pilot_run as pilot
from sucoder import dream
from tests.test_dream_pilot_run import Transport, fixture_proposal, limits


class UnavailableReader(Transport):
    failed_assignment = None

    def __call__(self, *args):
        result, usage, trace = super().__call__(*args)
        task = args[2]
        if (task["role"] == "reader" and result["action"] == "answer" and
                not task["input"]["sources_available"] and self.failed_assignment is None):
            self.failed_assignment = task["assignment_id"]
            result["failures"] = ["The wiki lacks the needed qualification; original sources are unavailable."]
            result["answer"] = "The supplied wiki does not establish this answer."
            events = [json.loads(line) for line in trace["stdout"].splitlines()]
            events[0]["item"]["text"] = json.dumps(result)
            trace["stdout"] = "\n".join(json.dumps(event) for event in events) + "\n"
        return result, usage, trace


def test_outcome_guard_retains_parent_and_rejects_source_drift(tmp_path, monkeypatch):
    budget = pilot.Budget(tmp_path, dict(limits=limits()))
    pilot.store(tmp_path / "grading-v2-contract.json", dict(contract="frozen fixture"))
    checked = []
    budget.guard = lambda: checked.append(True)
    first = outcomes.freeze_outcomes(tmp_path, budget)
    assert checked == [True]
    assert first["runtime"]["contract"] == outcomes.CONTRACT
    monkeypatch.setattr(outcomes, "CONTRACT", "changed")
    with pytest.raises(dream.DreamError, match="changed before transport"):
        budget.remaining()
    assert checked == [True, True]


def test_failed_attempt_is_collected_without_inference_and_complete_comparison_replays(tmp_path, monkeypatch):
    proposal, blobs = fixture_proposal(tmp_path, monkeypatch)
    native = UnavailableReader()
    with pytest.raises(dream.DreamError, match="Role reported incomplete work"):
        pilot.run(proposal, blobs, execute=True, native=native)
    root = proposal.parent / "execution"
    before = pilot.read(root / "budget.json")
    failed_id = native.failed_assignment
    assert failed_id is not None
    result = outcomes.run(proposal, blobs, execute=True, native=native)
    assert result["state"] == "completed" and len(result["answers"]) == 36
    assert result["reader_failures"] == [2]
    assert result["interpretation"].startswith("inconclusive: retained failed")
    assert result["budget"]["deadline"] == before["deadline"]
    assert sum(task["assignment_id"] == failed_id for task in native.calls) == 1
    record = pilot.read(root / "failed-readers/2.json")
    assert record["result"]["state"] == "reader_failed"
    assert record["result"]["failures"] and record["result"]["answer"] == "The supplied wiki does not establish this answer."
    assert not (root / "answers/2.json").exists()
    assert result["answers"][2]["evidence_trace"] == "failed-readers/2.json"
    count = len(native.calls)
    assert outcomes.run(proposal, blobs, execute=True, attempt=2, native=native) == result
    assert len(native.calls) == count
    path = next((root / "logs/dream").rglob("*" + failed_id + "-trace.json"))
    trace = pilot.read(path)
    trace["stdout"] += "\n"
    path.write_bytes(dream.canonical(trace) + b"\n")
    with pytest.raises(dream.DreamError, match="differs from the original journal"):
        outcomes.run(proposal, blobs, execute=True, attempt=2, native=native)
    assert len(native.calls) == count
