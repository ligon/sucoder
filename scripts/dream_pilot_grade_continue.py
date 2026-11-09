"""Explicit grading-only amendment of a stopped, already constructed pilot.

Reuses the original journal/deadline and unchanged reader runtime. Original
grades remain retained; all comparison grades use this separate contract.
See the corrected-grading grounding in the dream implementation ledger.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from scripts import dream_pilot_run as pilot
from sucoder import dream, dream_native, dream_roles
from sucoder.config import load_config


CONTRACT = "isolated-dream-pilot-grading-v2"
GRADE_BRIEF = pilot.GRADE_BRIEF + (
    " The failures array reports ONLY inability to execute the grading task, such as unavailable evidence or an unusable rubric. "
    "A complete grade must return failures: [], even if every rubric item is unmet or a critical error occurred. "
    "Report deficiencies in the ANSWER only as items with met=no and their reasons, or critical_errors with occurred=yes. "
    "Do not list missing answer details in failures. Echo assignment_id and input_sha256 exactly.")


def grade_schema():
    result = pilot.schema("grade")
    result["properties"]["failures"]["description"] = (
        "Only inability to perform grading. Empty for every complete grade, including incorrect answers. "
        "Answer deficiencies belong in items or critical_errors.")
    return result


class Grader(pilot.Evaluator):
    def __init__(self, *args):
        super().__init__(*args)
        self.brief = GRADE_BRIEF
        self.run_id = dream.digest(dream.canonical([CONTRACT, self.run_id, self.brief, self.result_schema("grade")]))

    def result_schema(self, role):
        return grade_schema()

    def make_assignment(self, role, inputs):
        task = super().make_assignment(role, inputs)
        task.pop("assignment_id")
        task["contract"] = CONTRACT
        task["result_schema"] = self.result_schema(role)
        task["assignment_id"] = dream.digest(dream.canonical(task))
        return task


def grade(directory, model, transport, budget, inputs, attempt):
    """Keep the existing Coordinator's trace checks and semantic-before-save order."""
    pilot.private_directory(directory)
    fd = dream._open_private_directory(directory)
    try:
        path = directory / "validated.json"
        saved = pilot.read(path) if path.exists() else None
        binding = dream.digest(dream.canonical(inputs))
        if saved is not None:
            if saved["input_sha256"] != binding or saved["result_sha256"] != dream.digest(dream.canonical(saved["result"])):
                raise dream.DreamError("Frozen amended grade input/result changed", 3)
            attempt = saved["attempt"]
        plan = dict(plan_id=binding, limits=asdict(dream.Limits(max_role_invocations=1)))
        engine = Grader(plan, fd, model, time.monotonic() + min(120, budget.remaining()), transport,
            "grade", inputs, attempt, dream.digest(str(directory.resolve()).encode()))
        with dream_roles.run_lock(fd, "evaluation-" + engine.run_id):
            result = engine.call("grade", inputs)
            pilot.check_grade(result, inputs["question"])
            if saved is not None and saved["result"] != result:
                raise dream.DreamError("Frozen amended grade differs from its trace", 3)
            pilot.store(path, dict(attempt=attempt, input_sha256=binding,
                result_sha256=dream.digest(dream.canonical(result)), result=result))
            return result
    finally:
        os.close(fd)


def reject_prior_critical(root):
    for path in sorted(root.glob("evaluator-only/*/grade-*/validated.json")) + sorted(root.glob("grading-v2/*/grade-*/validated.json")):
        result = pilot.read(path)["result"]
        if any(item["occurred"] == "yes" for item in result["critical_errors"]):
            raise dream.DreamError("A previously frozen critical grade stops the pilot; no regrading is permitted", 3)


def freeze_amendment(root, budget, execution, proposal):
    """Seal the new grader separately; the parent runtime guard remains required."""
    parent_guard = budget.guard
    artifacts = ["final-wiki.json", "native-version.json"]
    for corpus in proposal["sources"]:
        name = corpus["repository"]
        artifacts.extend([name + "-origin.json", "evaluator-only/" + name + "-questions.json"])
        for stage in ("initial", "incremental", "repair"):
            artifacts.extend([name + "-" + stage + "-plan.json", name + "-" + stage + "-result.json"])
    def current():
        return dict(contract=CONTRACT, source_sha256=dream.digest(Path(__file__).read_bytes()),
            grade_brief=GRADE_BRIEF, grade_schema=grade_schema(),
            parent_execution_sha256=dream.digest(dream.canonical(execution)),
            artifacts={name: dream.digest((root / name).read_bytes()) for name in artifacts})
    path = root / "grading-v2-contract.json"
    value = current()
    if path.exists():
        frozen = pilot.read(path)
        if frozen["runtime"] != value:
            raise dream.DreamError("Amended grader or frozen parent artifacts changed", 3)
    else:
        frozen = dict(runtime=value, parent_call_ids=[call["assignment_id"] for call in budget.state["calls"]],
            engine_commit=pilot.git(Path(__file__).resolve().parent.parent, "rev-parse", "HEAD"))
        pilot.store(path, frozen)
    ids = frozen["parent_call_ids"]
    if [call["assignment_id"] for call in budget.state["calls"][:len(ids)]] != ids:
        raise dream.DreamError("Amended pilot lost its original native-call journal", 3)
    def check():
        if parent_guard is None:
            raise dream.DreamError("Amended grading requires the original runtime guard", 3)
        parent_guard()
        if current() != frozen["runtime"]:
            raise dream.DreamError("Amended grader or frozen parent artifacts changed before transport", 3)
    budget.guard = check
    check()
    return frozen


def run(proposal_path, blobs, *, execute=False, attempt=1, native=None):
    if not execute:
        raise dream.DreamError("Grading amendment requires explicit --execute", 3)
    if type(attempt) is not int or not 1 <= attempt <= 1000000:
        raise dream.DreamError("Invalid explicit attempt", 3)
    root = proposal_path.parent / "execution"
    if not all((root / name).is_file() for name in ("budget.json", "execution-contract.json", "final-wiki.json", "native-version.json")):
        raise dream.DreamError("Grading amendment requires an existing frozen, fully constructed pilot", 3)
    proposal = pilot.read(proposal_path)
    verified = pilot.verify(pilot.read(pilot.MANIFEST), blobs)
    if (verified != proposal["sources"] or proposal["harness"] != "codex" or
            dream.digest(pilot.MANIFEST.read_bytes()) != proposal["corpus_manifest_sha256"] or
            proposal["limits"]["aggregate_native_calls"] > 180 or proposal["limits"]["aggregate_elapsed_seconds"] > 7200 or
            proposal["limits"].get("currency_ceiling") is not None):
        raise dream.DreamError("Amendment differs from the approved pilot scope", 3)
    expected = {(c["repository"], "q" + str(i), condition) for c in verified for i in range(1, 7)
        for condition in pilot.dream_reader_eval.CONDITIONS}
    scheduled = [(r["repository"], r["question_id"], r["condition"]) for r in proposal["reader_schedule"]]
    if len(scheduled) != 36 or set(scheduled) != expected:
        raise dream.DreamError("Amendment must retain the original complete reader schedule", 3)
    fd = dream._open_private_directory(root)
    try:
        with dream_roles.run_lock(fd, "pilot"):
            reject_prior_critical(root)
            budget = pilot.Budget(root, proposal)
            native = native or dream_native.Transport("codex", time.monotonic() + budget.remaining())
            model = proposal["model"]
            if pilot.read(root / "native-version.json") != dict(version=native.version, model=model):
                raise dream.DreamError("Amended pilot native model/build changed", 3)
            execution = pilot.freeze_execution(root, budget, native, model)
            amendment = freeze_amendment(root, budget, execution, proposal)
            config = load_config(root / "config.json")
            final = pilot.read(root / "final-wiki.json")
            sources, questions, origins = {}, {}, {}
            for corpus in verified:
                name = corpus["repository"]
                repo = root / "repos" / name
                origins[name] = pilot.read(root / (name + "-origin.json"))
                if (pilot.git(repo, "rev-parse", "HEAD") != final[name]["commit"] or
                        pilot.git(repo, "status", "--porcelain")):
                    raise dream.DreamError("Frozen comparison wiki changed", 3)
                if any(pilot.git(repo, "rev-parse", origins[name]["imported_commit"] + ":" + u["path"]) != u["git_blob"] for u in corpus["units"]):
                    raise dream.DreamError("Frozen imported sources changed", 3)
                sources[name] = [dict(unit_id=u["unit_id"], path=u["path"], original_commit=corpus["commit"],
                    text=(blobs / u["git_blob"]).read_bytes()[u["byte_start"]:u["byte_end"]].decode()) for u in corpus["units"]]
                directory = root / "evaluator-only" / name / "questions"
                if not (directory / "validated.json").is_file():
                    raise dream.DreamError("Original held-out question validation is missing", 3)
                deck = pilot.evaluate(directory, model, budget.transport(native, name + ":questions"), budget,
                    "questions", dict(sources=sources[name]), attempt)
                if deck != pilot.read(root / "evaluator-only" / (name + "-questions.json")):
                    raise dream.DreamError("Original held-out questions changed", 3)
                questions[name] = {q["id"]: q for q in deck["questions"]}
            answers = []
            for index, row in enumerate(proposal["reader_schedule"]):
                budget.remaining()
                name, qid, condition = (row[k] for k in ("repository", "question_id", "condition"))
                question = questions[name][qid]
                answer = pilot.reader_case(root, index, config, name, final[name], question["question"], condition,
                    model, native, budget, execution, attempt, proposal["limits"]["reader_elapsed_seconds"])
                result = grade(root / "grading-v2" / name / ("grade-" + str(index)), model,
                    budget.transport(native, "grade-v2:" + str(index)), budget,
                    dict(sources=sources[name], question=question, answer=answer["answer"]), attempt)
                paired = dict(row, category=question["category"], answer=answer, grade=result)
                answers.append(paired)
                pilot.store(root / ("paired-v2-" + str(index).zfill(2) + ".json"), paired)
                if any(c["occurred"] == "yes" for c in result["critical_errors"]):
                    raise dream.DreamError("Amended pilot stopped on a graded critical error", 3)
            result = pilot.report(proposal, answers, budget)
            for index, row in enumerate(result["answers"]):
                row["paired_record"] = "paired-v2-" + str(index).zfill(2) + ".json"
            result.update(origins=origins, final=final, engine_commit=execution["engine_commit"],
                execution_sha256=dream.digest(dream.canonical(execution)), grading_amendment=amendment,
                note=result["note"] + " All grades use the separately frozen v2 grader; original grades are retained. Original readers, calls and deadline are unchanged.")
            pilot.store(root / "report-grading-v2.json", result)
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
    print(json.dumps({key: result[key] for key in ("state", "interpretation", "critical_errors", "rubric_totals")}))


if __name__ == "__main__":
    main()
