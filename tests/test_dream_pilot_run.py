"""Bounded experiment orchestration; provider calls here are protocol fixtures."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import dream_pilot_prepare as preparation, dream_pilot_run as pilot
from sucoder import dream, dream_wiki


def limits():
    return dict(aggregate_elapsed_seconds=7200, aggregate_native_calls=180, reader_elapsed_seconds=120,
        reader_role_calls=8, construction_role_calls=14, construction_elapsed_seconds=900, currency_ceiling=None)


def questions(sources):
    return [dict(id="q" + str(index + 1), category=kind, question="Fixture question " + str(index + 1),
        rubric=[dict(id="criterion-1", criterion="Retains the fixture qualification"), dict(id="criterion-2", criterion="Distinguishes static evidence")],
        critical_errors=["Invented executed result"],
        support=[dict(unit_id=source["unit_id"], quote=source["text"].splitlines()[0]) for source in sources[:2]])
        for index, kind in enumerate(("lookup", "lookup", "synthesis", "synthesis", "caveat", "caveat"))]


class Transport:
    version = "pilot-protocol-fixture-1"

    def __init__(self, critical=False):
        self.calls = []
        self.critical = critical

    def __call__(self, harness, model, task, result_schema, deadline):
        self.calls.append(copy.deepcopy(task))
        inputs, role = task["input"], task["role"]
        result = {key: task[key] for key in ("assignment_id", "input_sha256", "role")}
        result["failures"] = []
        if role == "questions":
            result["questions"] = questions(inputs["sources"])
        elif role == "grade":
            result.update(items=[dict(id=item["id"], met="yes", reason="Fixture assessment") for item in inputs["question"]["rubric"]],
                critical_errors=[dict(criterion=item, occurred="yes" if self.critical else "no", reason="Fixture assessment") for item in inputs["question"]["critical_errors"]])
        elif role == "reviewer":
            result.update(verdict="accept", findings=[], requests=[], task_checks=[])
        elif role in ("author", "repair"):
            path = ".sucoder/wiki/topics/companion.org"
            old = {item["path"]: item["content"] for item in inputs["prior"]}
            content = "#+title: Fixture companion\n* Knowledge\n:PROPERTIES:\n:CUSTOM_ID: knowledge\n:END:\n"
            content += "\n".join(source["text"].splitlines()[0] for source in inputs["sources"]) + "\n"
            targets = ["knowledge"]
            if "seeded-evaluation-error" in old.get(path, ""):
                content += "* Corrected working rule\n:PROPERTIES:\n:CUSTOM_ID: seeded-evaluation-error\n:END:\nPreserve the static fixture qualification.\n"
                targets.append("seeded-evaluation-error")
            pages = {path: content, dream_wiki.INDEX: "#+title: Fixture\n[[file:topics/companion.org::#knowledge][Knowledge]]\n"}
            result.update(decision="change", rationale="Fixture source-grounded revision", pages=[dict(path=key, content=value) for key, value in pages.items()],
                changes=[dict(path=key, before_sha256=dream.digest(old[key].encode()) if key in old else "", reason="Fixture revision") for key in pages],
                references=[dict(path=path, target=target, evidence_status="inference", citations=[dict(unit_id=source["unit_id"], quote=source["text"].splitlines()[0]) for source in inputs["sources"]]) for target in targets],
                omissions=[], requests=[], task_actions=[])
        else:
            kinds = {entry["request"]["kind"] for entry in inputs["retrievals"]}
            result.update(answer="", evidence=[], requests=[])
            if inputs["wiki_available"] and "topic" not in kinds:
                result.update(action="lookup", requests=[dict(kind="topic", key="companion", target="knowledge", query="", start=0, end=0)])
            elif inputs["sources_available"] and "source" not in kinds:
                result.update(action="lookup", requests=[dict(kind="source", key=inputs["admitted_sources"][0]["unit_id"], target="", query="", start=0, end=0)])
            else:
                result.update(action="answer", answer="Static fixture qualification; no executed result.",
                    evidence=[entry["response"]["read_id"] for entry in inputs["retrievals"] if "read_id" in entry["response"]])
        events = [dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(result))),
            dict(type="turn.completed", usage=dict(input_tokens=100, output_tokens=10))]
        trace = dict(harness=harness, model=model, version=self.version, returncode=0,
            input_sha256=dream.digest(dream.canonical(task)), stdout="\n".join(json.dumps(event) for event in events) + "\n",
            stderr="", boundary="linux-landlock-inline-v1")
        usage = dict(native=dict(input_tokens=100, output_tokens=10), currency_usd=None)
        return result, usage, trace


def test_budget_reserves_before_failure_and_never_repeats(tmp_path):
    proposal = dict(limits=limits())
    budget = pilot.Budget(tmp_path, proposal)
    class Uncertain:
        version = "fixture"
        def __call__(self, *args):
            assert pilot.read(tmp_path / "reservation-000.json")["state"] == "started"
            raise RuntimeError("uncertain provider outcome")
    transport = budget.transport(Uncertain(), "case")
    with pytest.raises(RuntimeError):
        transport("codex", "model", dict(assignment_id="call", role="reader"), {}, 999999999)
    assert pilot.read(tmp_path / "returned-000.json")["state"] == "uncertain"
    resumed = pilot.Budget(tmp_path, proposal)
    with pytest.raises(dream.DreamError, match="reservation already exists"):
        resumed.transport(Uncertain(), "case")("codex", "model", dict(assignment_id="call", role="reader"), {}, 999999999)


def test_budget_limits_and_scope_cannot_reset(tmp_path, monkeypatch):
    proposal = dict(limits=limits())
    budget = pilot.Budget(tmp_path, proposal)
    altered = copy.deepcopy(proposal)
    altered["limits"]["aggregate_native_calls"] += 1
    with pytest.raises(dream.DreamError, match="proposal changed"):
        pilot.Budget(tmp_path, altered)
    monkeypatch.setattr(pilot.time, "time", lambda: budget.state["deadline"])
    with pytest.raises(dream.DreamError, match="time allowance"):
        pilot.Budget(tmp_path, proposal)


def test_question_support_and_grade_are_bound_to_rubric():
    sources = [dict(unit_id="unit-" + str(index), text="Fixture source " + str(index)) for index in range(2)]
    result = dict(questions=questions(sources))
    pilot.check_questions(result, sources)
    bad = copy.deepcopy(result)
    bad["questions"][2]["support"] = bad["questions"][2]["support"][:1]
    with pytest.raises(dream.DreamError, match="cross-source"):
        pilot.check_questions(bad, sources)
    with pytest.raises(dream.DreamError, match="frozen rubric"):
        pilot.check_grade(dict(items=[], critical_errors=[]), result["questions"][0])


def test_aggregate_call_cap_survives_resume(tmp_path):
    proposal = dict(limits=dict(limits(), aggregate_native_calls=1))
    budget = pilot.Budget(tmp_path, proposal)
    class Failed:
        version = "fixture"
        def __call__(self, *args):
            return None, None, dict(returncode=1)
    budget.transport(Failed(), "case")("codex", "model", dict(assignment_id="one", role="reader"), {}, 999999999)
    resumed = pilot.Budget(tmp_path, proposal)
    with pytest.raises(dream.DreamError, match="allowance exhausted"):
        resumed.transport(Failed(), "case")("codex", "model", dict(assignment_id="two", role="reader"), {}, 999999999)
    assert len(resumed.state["calls"]) == 1


def test_evaluator_replays_frozen_attempt_and_rejects_corrupt_context(tmp_path):
    budget = pilot.Budget(tmp_path, dict(limits=limits()))
    transport = Transport()
    directory = tmp_path / "evaluator"
    inputs = dict(sources=[dict(unit_id="unit-" + str(index), text="Fixture source " + str(index)) for index in range(2)])
    first = pilot.evaluate(directory, "model", budget.transport(transport, "questions"), budget, "questions", inputs, 1)
    assert pilot.evaluate(directory, "model", budget.transport(transport, "questions"), budget, "questions", inputs, 2) == first
    assert len(transport.calls) == 1
    changed = dict(inputs, added="changed frozen context")
    with pytest.raises(dream.DreamError, match="Frozen evaluation input/result changed"):
        pilot.evaluate(directory, "model", budget.transport(transport, "questions"), budget, "questions", changed, 2)
    assert len(transport.calls) == 1


def test_pilot_decision_requires_quality_before_byte_savings():
    proposal = dict(sources=[dict(repository=name) for name in ("SuCoder", "LSMS_Library")])
    answers = [dict(repository=corpus["repository"], question_id="q" + str(index), condition=condition, category="synthesis",
        grade=dict(items=[dict(id="criterion", met="yes")], critical_errors=[dict(occurred="no")]),
        answer=dict(run_id="fixture", accounting=dict(assignment_bytes=1000 if condition == "source-only" else 700), latency=dict(seconds=1)))
        for corpus in proposal["sources"] for index in range(6) for condition in ("source-only", "wiki-search", "wiki-only")]
    class Finished:
        state = dict(calls=[])
    assert pilot.report(proposal, answers, Finished())["interpretation"].startswith("exploratory criteria met")
    answers[1]["grade"]["items"][0]["met"] = "no"
    assert pilot.report(proposal, answers, Finished())["interpretation"] == "inconclusive or criteria not met"
    answers[1]["grade"]["items"][0]["met"] = "yes"
    answers[1]["grade"]["critical_errors"][0]["occurred"] = "yes"
    assert pilot.report(proposal, answers, Finished())["critical_errors"] == 1
    assert pilot.report(proposal, answers, Finished())["interpretation"] == "inconclusive or criteria not met"


def fixture_proposal(tmp_path, monkeypatch):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    corpora = []
    for name, commit in (("SuCoder", "a" * 40), ("LSMS_Library", "b" * 40)):
        units = []
        for index in range(6):
            raw = ("Fixture source " + str(index) + ": qualification retained.\nStatic fixture; no execution.\n").encode()
            blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            (blobs / blob).write_bytes(raw)
            units.append(dict(unit_id="unit-" + str(index), path="source-" + str(index) + ".org", git_blob=blob,
                byte_start=0, byte_end=len(raw), bytes=len(raw), lines=[1, 2], sha256=dream.digest(raw)))
        corpora.append(dict(repository=name, commit=commit, sources=units, selected_bytes=sum(unit["bytes"] for unit in units)))
    manifest_path = tmp_path / "corpus.json"
    pilot.store(manifest_path, dict(corpora=corpora))
    monkeypatch.setattr(pilot, "MANIFEST", manifest_path)
    verified = preparation.verify(pilot.read(manifest_path), blobs)
    proposal = dict(state="prepared_not_executed", harness="codex", model="fixture-model", sources=verified,
        corpus_manifest_sha256=dream.digest(manifest_path.read_bytes()), limits=limits(),
        reader_schedule=[dict(repository=corpus["repository"], question_id="q" + str(index), condition=condition)
            for corpus in verified for index in range(1, 7) for condition in ("source-only", "wiki-search", "wiki-only")])
    proposal_path = tmp_path / "review" / "pilot-proposal.json"
    pilot.store(proposal_path, proposal)
    return proposal_path, blobs


def test_execute_is_opt_in_and_caps_reject_before_transport(tmp_path, monkeypatch):
    proposal, blobs = fixture_proposal(tmp_path, monkeypatch)
    with pytest.raises(dream.DreamError, match="explicit --execute"):
        pilot.run(proposal, blobs)
    data = pilot.read(proposal)
    data["limits"]["aggregate_native_calls"] = 181
    proposal.write_bytes(dream.canonical(data))
    with pytest.raises(dream.DreamError, match="scoped execution limits"):
        pilot.run(proposal, blobs, execute=True, native=Transport())


def test_complete_disposable_protocol_experiment_and_zero_call_resume(tmp_path, monkeypatch):
    proposal, blobs = fixture_proposal(tmp_path, monkeypatch)
    transport = Transport()
    result = pilot.run(proposal, blobs, execute=True, native=transport)
    assert result["state"] == "completed" and len(result["answers"]) == 36
    assert len(transport.calls) == len(result["budget"]["calls"]) <= 180
    assert all(task["role"] == "questions" for task in transport.calls[:2])
    for task in transport.calls:
        if task["role"] in ("author", "repair", "reviewer", "reader"):
            encoded = json.dumps(task)
            assert "Fixture question" not in encoded or task["role"] == "reader"
            assert "criterion-1" not in encoded and "Invented executed result" not in encoded
        if task["role"] == "grade":
            assert set(task["input"]) == {"sources", "question", "answer"}
    count = len(transport.calls)
    replay = pilot.run(proposal, blobs, execute=True, attempt=2, native=transport)
    assert replay == result and len(transport.calls) == count


def test_graded_critical_error_stops_and_cannot_be_skipped_on_resume(tmp_path, monkeypatch):
    proposal, blobs = fixture_proposal(tmp_path, monkeypatch)
    transport = Transport(critical=True)
    with pytest.raises(dream.DreamError, match="stopped on a graded critical error"):
        pilot.run(proposal, blobs, execute=True, native=transport)
    root = proposal.parent / "execution"
    assert (root / "paired-00.json").exists() and not (root / "report.json").exists()
    assert sum(task["role"] == "grade" for task in transport.calls) == 1
    count = len(transport.calls)
    with pytest.raises(dream.DreamError, match="stopped on a graded critical error"):
        pilot.run(proposal, blobs, execute=True, attempt=2, native=transport)
    assert len(transport.calls) == count
