"""Continue the same bounded pilot, retaining terminal reader failures as data.

Original reader/model inputs and graders are unchanged. Failed answers cannot
establish a positive result. See the failed-reader grounding in the ledger.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from scripts import dream_pilot_grade_continue as grading, dream_pilot_run as pilot
from sucoder import dream, dream_native, dream_reader, dream_reader_eval, dream_roles, dream_run
from sucoder.config import load_config


CONTRACT = "isolated-dream-pilot-failed-outcomes-v1"


class FailedReplay(dream_reader_eval.Coordinator):
    """Replay only retained, completed native events; never issue a model call."""
    journal: list[dict]
    case_index: int
    failures: list[str]

    def call(self, role, inputs):
        inputs = dict(inputs, sources_available=self.session.with_sources,
            admitted_sources=inputs["admitted_sources"] if self.session.with_sources else [],
            allowances=dict(inputs["allowances"], source=dream_reader.SOURCE_BYTES if self.session.with_sources else 0))
        task = self.make_assignment(role, inputs)
        key = self.run_id + "-" + task["assignment_id"]
        saved_task = dream._json(dream_run._read(self.fd, key + "-assignment.json", dream_run.MAX_PLAN_BYTES))
        if saved_task != task:
            raise dream.DreamError("Failed reader assignment changed", 3)
        trace = dream._json(dream_run._read(self.fd, key + "-trace.json", dream_native.MAX_TRACE * 2))
        if (trace["input_sha256"] != dream.digest(dream.canonical(task)) or trace["harness"] != self.harness or
                trace["model"] != self.model or trace["version"] != self.expected_version or trace["returncode"] != 0):
            raise dream.DreamError("Failed reader trace binding changed", 3)
        call = next((row for row in self.journal if row["assignment_id"] == task["assignment_id"]), None)
        if (call is None or call["state"] != "completed" or call["phase"] != "reader:" + str(self.case_index) or
                call["trace_sha256"] != dream.digest(dream.canonical(trace)) or
                call["assignment_bytes"] != len(dream.canonical(task))):
            raise dream.DreamError("Failed reader trace differs from the original journal", 3)
        result, usage = dream_native.parse_trace(trace["stdout"].encode(), self.harness)
        if result is None:
            raise dream.DreamError("Failed reader lacks a completed native result", 3)
        dream_roles.validate_schema(result, self.result_schema(role))
        if any(result[k] != task[k] for k in ("assignment_id", "input_sha256", "role")) or usage != call["usage"]:
            raise dream.DreamError("Failed reader result or usage changed", 3)
        if result["failures"]:
            if result["action"] != "answer":
                raise dream.DreamError("Only a completed terminal reader failure can be collected", 3)
            self.failures.extend(result["failures"])
        self.invocations += 1
        self.version = self.expected_version
        self.usage.append(usage)
        self.traces.append(dict(assignment_id=task["assignment_id"], role=role, input_sha256=task["input_sha256"],
            submitted_sha256=dream.digest(dream.canonical(task)), trace_sha256=dream.digest(dream.canonical(trace)),
            result_sha256=dream.digest(dream.canonical(result))))
        self.deliveries.append(dict(assignment_bytes=len(dream.canonical(task)), common_brief_bytes=len(dream_reader_eval.BRIEF.encode()),
            retrieval_bytes=sum(e["response"]["delivered_bytes"] for e in inputs["retrievals"]),
            orientation_instruction_bytes=sum(len(dream_reader.INSTRUCTIONS.encode()) for e in inputs["retrievals"]
                if e["request"]["kind"] == "orientation" and e["response"]["delivered_bytes"])))
        return result


def failed_reader(root, index, config, name, final, question, condition, model, native, budget, execution, attempt, elapsed):
    path = root / "failed-readers" / (str(index) + ".json")
    binding = dict(question=question, condition=condition, wiki_commit=final["commit"], plan_path=final["plan_path"],
        model=model, native_version=native.version, execution_sha256=dream.digest(dream.canonical(execution)))
    saved = pilot.read(path) if path.exists() else None
    if saved is not None:
        if saved["binding"] != binding:
            raise dream.DreamError("Failed reader case binding changed", 3)
        attempt = saved["attempt"]
    with dream_run._directory(config, Path(final["plan_path"])) as (fd, directory):
        plan, sources = dream_reader.frozen_sources(config, name, Path(final["plan_path"]))
        deadline = time.monotonic() + elapsed
        view = dream_reader.View(config, name, final["commit"], deadline)
        session = dream_reader_eval.Session(config, name, Path(final["plan_path"]), view, condition=condition)
        engine = FailedReplay(plan, fd, "codex", model, deadline, native, session, question, attempt=attempt, elapsed_limit=elapsed)
        engine.journal, engine.case_index, engine.failures = budget.state["calls"], index, []
        with dream_roles.run_lock(fd, "reader-" + engine.run_id):
            result = engine.prepare()
        if not engine.failures:
            raise dream.DreamError("Failure collector requires a retained terminal role failure", 3)
        result.update(state="reader_failed", failures=engine.failures,
            latency=dict(seconds=sum(call.get("elapsed_seconds", 0) for call in budget.state["calls"] if call["phase"] == "reader:" + str(index)),
                scope="All journaled native reader calls for this failed case; no inference during collection."))
        record = dict(binding=binding, attempt=attempt, result=result)
        if saved is not None and saved != record:
            raise dream.DreamError("Failed reader outcome changed during replay", 3)
        pilot.store(path, record)
        return result


def reader_case(root, index, config, name, final, question, condition, model, native, budget, execution, attempt, elapsed):
    args = (root, index, config, name, final, question, condition, model, native, budget, execution, attempt, elapsed)
    if (root / "failed-readers" / (str(index) + ".json")).exists():
        return failed_reader(*args)
    try:
        return pilot.reader_case(*args)
    except dream.DreamError as exc:
        if not str(exc).startswith(("Role reported incomplete work:", "An earlier role attempt has no validated result;")):
            raise
        return failed_reader(*args)


def freeze_outcomes(root, budget):
    parent_guard = budget.guard
    def current():
        return dict(contract=CONTRACT, source_sha256=dream.digest(Path(__file__).read_bytes()),
            grading_contract_sha256=dream.digest((root / "grading-v2-contract.json").read_bytes()))
    path = root / "outcome-contract.json"
    value = current()
    if path.exists():
        frozen = pilot.read(path)
        if frozen["runtime"] != value:
            raise dream.DreamError("Frozen outcome collector changed", 3)
    else:
        frozen = dict(runtime=value, engine_commit=pilot.git(Path(__file__).resolve().parent.parent, "rev-parse", "HEAD"))
        pilot.store(path, frozen)
    def check():
        if parent_guard is None:
            raise dream.DreamError("Outcome collection requires the parent runtime guard", 3)
        parent_guard()
        if current() != frozen["runtime"]:
            raise dream.DreamError("Frozen outcome collector changed before transport", 3)
    budget.guard = check
    check()
    return frozen


def run(proposal_path, blobs, *, execute=False, attempt=1, native=None):
    if not execute:
        raise dream.DreamError("Outcome continuation requires explicit --execute", 3)
    root = proposal_path.parent / "execution"
    if not (root / "final-wiki.json").is_file() or not (root / "budget.json").is_file():
        raise dream.DreamError("Outcome continuation requires the original constructed pilot", 3)
    proposal = pilot.read(proposal_path)
    verified = pilot.verify(pilot.read(pilot.MANIFEST), blobs)
    if verified != proposal["sources"] or dream.digest(pilot.MANIFEST.read_bytes()) != proposal["corpus_manifest_sha256"]:
        raise dream.DreamError("Original pilot source scope changed", 3)
    fd = dream._open_private_directory(root)
    try:
        with dream_roles.run_lock(fd, "pilot"):
            grading.reject_prior_critical(root)
            budget = pilot.Budget(root, proposal)
            native = native or dream_native.Transport("codex", time.monotonic() + budget.remaining())
            model = proposal["model"]
            execution = pilot.freeze_execution(root, budget, native, model)
            amendment = grading.freeze_amendment(root, budget, execution, proposal)
            outcome_contract = freeze_outcomes(root, budget)
            config, final = load_config(root / "config.json"), pilot.read(root / "final-wiki.json")
            sources, questions, origins = {}, {}, {}
            for corpus in verified:
                name = corpus["repository"]
                repo = root / "repos" / name
                origins[name] = pilot.read(root / (name + "-origin.json"))
                if pilot.git(repo, "rev-parse", "HEAD") != final[name]["commit"] or pilot.git(repo, "status", "--porcelain"):
                    raise dream.DreamError("Frozen comparison wiki changed", 3)
                sources[name] = [dict(unit_id=u["unit_id"], path=u["path"], original_commit=corpus["commit"],
                    text=(blobs / u["git_blob"]).read_bytes()[u["byte_start"]:u["byte_end"]].decode()) for u in corpus["units"]]
                directory = root / "evaluator-only" / name / "questions"
                if not (directory / "validated.json").exists():
                    raise dream.DreamError("Original held-out questions are missing", 3)
                deck = pilot.evaluate(directory, model, budget.transport(native, name + ":questions"), budget,
                    "questions", dict(sources=sources[name]), attempt)
                if deck != pilot.read(root / "evaluator-only" / (name + "-questions.json")):
                    raise dream.DreamError("Original held-out questions changed", 3)
                questions[name] = {q["id"]: q for q in deck["questions"]}
            answers = []
            for index, scheduled in enumerate(proposal["reader_schedule"]):
                budget.remaining()
                name, qid, condition = (scheduled[k] for k in ("repository", "question_id", "condition"))
                question = questions[name][qid]
                answer = reader_case(root, index, config, name, final[name], question["question"], condition,
                    model, native, budget, execution, attempt, proposal["limits"]["reader_elapsed_seconds"])
                result = grading.grade(root / "grading-v2" / name / ("grade-" + str(index)), model,
                    budget.transport(native, "grade-v2:" + str(index)), budget,
                    dict(sources=sources[name], question=question, answer=answer["answer"]), attempt)
                paired = dict(scheduled, category=question["category"], answer=answer, grade=result)
                answers.append(paired)
                pilot.store(root / ("paired-outcomes-" + str(index).zfill(2) + ".json"), paired)
                if any(c["occurred"] == "yes" for c in result["critical_errors"]):
                    raise dream.DreamError("Outcome comparison stopped on a graded critical error", 3)
            result = pilot.report(proposal, answers, budget)
            failures = []
            for index, (row, summary) in enumerate(zip(answers, result["answers"])):
                summary["paired_record"] = "paired-outcomes-" + str(index).zfill(2) + ".json"
                summary["reader_state"] = row["answer"]["state"]
                if row["answer"]["state"] == "reader_failed":
                    failures.append(index)
                    summary["evidence_trace"] = "failed-readers/" + str(index) + ".json"
            result.update(reader_failures=failures, origins=origins, final=final,
                engine_commit=execution["engine_commit"], execution_sha256=dream.digest(dream.canonical(execution)),
                grading_amendment=amendment, outcome_contract=outcome_contract)
            if failures:
                result["interpretation"] = "inconclusive: retained failed reader attempts; no positive usefulness claim"
            pilot.store(root / "report-outcomes.json", result)
            return result
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--blobs", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    try:
        result = run(args.proposal, args.blobs, execute=args.execute, attempt=args.attempt)
    except (dream.DreamError, OSError, ValueError, KeyError) as exc:
        print(json.dumps(dict(state="inconclusive", error=str(exc), private_artifacts=str(args.proposal.parent / "execution"))))
        raise SystemExit(3) from exc
    print(json.dumps({k: result[k] for k in ("state", "interpretation", "reader_failures", "critical_errors", "rubric_totals")}))


if __name__ == "__main__":
    main()
