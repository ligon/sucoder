"""Public editorial lifecycle, materialized review and archive retrieval."""
import copy
import json
from pathlib import Path

import pytest

from sucoder import dream, dream_editor, dream_editorial as org, dream_reader, dream_reader_agent, dream_run, dream_wiki
from tests.test_dream import git, invoke, workspace  # noqa: F401
from tests.test_dream_context import native_result


TOPIC = ".sucoder/wiki/topics/replies.org"
INCOMING = ".sucoder/wiki/topics/navigation.org"
ARCHIVE = ".sucoder/wiki/archive/replies.org"
TASK = "corrected-reply-rule"
HUMAN = "maintainer-timeout"
EVIDENCE = ("A compaction summary does not establish a native reply. Native delivery and subsequent native assistant text are required. "
    "An absorbed queued input may establish delivery through the queued_command attachment. This is documented/static evidence, not a new executed launch.\n")
PAGE = ("#+title: Reply evidence\n#+TODO: TODO NEXT ACTIVE WAITING REVIEW | DONE MOOT\n"
    "* Reply rule\n:PROPERTIES:\n:CUSTOM_ID: reply-proof\n:END:\nA summary alone is not proof of a native reply.\n"
    "** Talk :editorial:\n*** TODO Correct the reply rule\n:PROPERTIES:\n:CUSTOM_ID: corrected-reply-rule\n:OWNER: Sue\n:END:\n"
    "Retain the native delivery and queued-input qualifications before completing this correction.\n"
    "*** TODO Infer a reply from a summary\n:PROPERTIES:\n:CUSTOM_ID: infer-from-summary\n:OWNER: Sue\n:END:\nA proposal awaiting evidence review.\n"
    "*** WAITING Maintainer timeout decision\n:PROPERTIES:\n:CUSTOM_ID: maintainer-timeout\n:OWNER: User\n:REVIEWER: User\n"
    ":REQUIRES_HUMAN: yes\n:BLOCKER: Await recorded maintainer decision\n:RESUME_WHEN: Maintainer decision is supplied\n:END:\n"
    "Keep the ten-second timeout until the human decision is recorded.\n")


def prepare(workspace):
    repo = workspace[0]
    (repo / TOPIC).parent.mkdir(parents=True)
    (repo / TOPIC).write_text(PAGE)
    (repo / INCOMING).write_text("* Navigation\n:PROPERTIES:\n:CUSTOM_ID: navigation\n:END:\n[[file:replies.org::#corrected-reply-rule][Correction history]].\n")
    (repo / dream_wiki.INDEX).write_text("#+title: Companion\n[[file:topics/replies.org::#reply-proof][Reply evidence]]\n[[file:topics/navigation.org::#navigation][Navigation]]\n")
    (repo / dream_wiki.SETTINGS_PATH).write_text(dream_wiki.SETTINGS)
    (repo / "evidence.org").write_text(EVIDENCE)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Seed editorial lifecycle and factual evidence")


def planning(workspace, reason, request=None):
    request = request or dict(workspace[3], units=[dict(unit_id="evidence", kind="git", shareable=True,
        locator=dict(repository="sample", commit="HEAD", path="evidence.org"))])
    value = invoke(workspace, request, "--maintenance", reason, "--topic", "replies")
    assert value.exit_code == 0, value.output
    return json.loads(value.output)


def action(inputs, state="NEXT", target=TASK, *, kind="transition", authority="agent", quote=None):
    page = next(page for page in inputs["prior"] if page["path"] == TOPIC)
    source = inputs["sources"][0]
    return dict(kind=kind, path=TOPIC, target=target, before_sha256=page["inventory"]["targets"][target],
        state=state if kind == "transition" else "", note="Sue: HISTORY_CANARY recorded editorial work; retain the source qualifications.",
        owner="User" if target == HUMAN else "Sue", reviewer=("User" if target == HUMAN else "Independent native reviewer") if state in ("REVIEW", "DONE") else "",
        acceptance=("Recorded maintainer timeout decision." if target == HUMAN else "The rule retains native delivery, queued input and reported/static scope.") if state in ("REVIEW", "DONE") else "",
        blocker="Need the queued-input context" if state == "WAITING" else "",
        resume_when="Inspect the admitted queued-input evidence" if state == "WAITING" else "",
        destination=ARCHIVE if kind == "archive" else "", authority=authority,
        evidence=[] if kind == "archive" else [dict(unit_id=source["unit_id"], quote=quote or source["text"])])


class Lifecycle:
    version = "editorial-fixture-1"

    def __init__(self, state="NEXT", *, target=TASK, archive=False, returned=False, clarify=False,
                 authority="agent", quote=None, reviewer_human=False, authority_verified=True, drop_after_reject=False):
        self.state, self.target, self.archive, self.returned, self.clarify = state, target, archive, returned, clarify
        self.authority, self.quote, self.reviewer_human = authority, quote, reviewer_human
        self.authority_verified, self.drop_after_reject = authority_verified, drop_after_reject
        self.calls = []

    def __call__(self, harness, model, task, contract, deadline):
        self.calls.append(copy.deepcopy(task))
        inputs = task["input"]
        result = {key: task[key] for key in ("assignment_id", "input_sha256", "role")}
        result.update(failures=[], requests=[])
        if task["role"] == "reviewer":
            candidate = inputs["candidate"]
            actions = candidate["task_actions"]
            files = {page["path"]: page["content"] for page in candidate["pages"]}
            materialized = org.inspect(files) if files else {}
            if actions and actions[0]["kind"] == "transition":
                actual = next(h for h in materialized[TOPIC]["headings"] if h["target"] == self.target)
                assert actual["state"] == actions[0]["state"]
                assert "HISTORY_CANARY" in actual["logbook"]
            if actions and actions[0]["kind"] == "archive":
                assert ARCHIVE in files and ":ARCHIVE_FILE:" in files[ARCHIVE]
                assert materialized[TOPIC]["headings"]
                assert "file:../archive/replies.org::#corrected-reply-rule" in files[INCOMING]
            reject = self.returned and actions and actions[0]["state"] == "DONE"
            result.update(verdict="revise" if reject else "accept", findings=[dict(severity="blocking", path=TOPIC,
                message="The completion check omitted queued-input delivery context; return this task for revision.")] if reject else [],
                task_checks=[dict(kind=a["kind"], path=a["path"], target=a["target"], requires_human="yes" if self.reviewer_human else "no",
                    authority_verified="yes" if self.authority_verified else "no", acceptance_satisfied="no" if reject else "yes",
                    reason="Checked recorded authority, retained qualifications and the materialized task/history.") for a in actions])
        else:
            state = "ACTIVE" if task["role"] == "repair" and self.returned else self.state
            actions = ([action(inputs, target=TASK, kind="archive"), action(inputs, target="infer-from-summary", kind="archive")]
                       if self.archive else [action(inputs, state, self.target, authority=self.authority, quote=self.quote)])
            pages, changes, references = [], [], []
            if self.clarify:
                old = next(page for page in inputs["prior"] if page["path"] == TOPIC)
                text = old["content"].replace("A summary alone is not proof of a native reply.",
                    "A summary alone is not proof of a native reply. Native delivery and subsequent assistant text are required. "
                    "The queued_command attachment may establish absorbed input delivery; this records documented/static scope, not a new launch.")
                pages = [dict(path=TOPIC, content=text)]
                changes = [dict(path=TOPIC, before_sha256=old["sha256"], reason="Retain the missing queued-input qualification.")]
                references = [dict(path=TOPIC, target="reply-proof", evidence_status="inference", citations=[
                    dict(unit_id=inputs["sources"][0]["unit_id"], quote=inputs["sources"][0]["text"])])]
            unchanged = task["role"] == "repair" and self.drop_after_reject
            result.update(decision="unchanged" if unchanged else "change", rationale="Record supported editorial work and preserve source scope.",
                pages=pages, changes=changes, references=references, omissions=[], task_actions=[] if unchanged else actions)
        return native_result(harness, model, task, result, self.version)


def execute(workspace, reason, transport, harness="codex", *, integrate=True, request=None, replay=True):
    plan = planning(workspace, reason, request)
    path = Path(plan["plan_path"])
    before = (git(workspace[0], "status", "--porcelain"), (workspace[0] / ".git/index").read_bytes(), git(workspace[0], "symbolic-ref", "HEAD"))
    result = dream_wiki.run(workspace[2], "sample", path, harness, "model", transport=transport)
    assert before == (git(workspace[0], "status", "--porcelain"), (workspace[0] / ".git/index").read_bytes(), git(workspace[0], "symbolic-ref", "HEAD"))
    if replay:
        assert dream_run.status(workspace[2], "sample", path)["state"] == result["state"]
        count = len(transport.calls)
        assert dream_wiki.run(workspace[2], "sample", path, harness, "model", transport=transport) == result
        assert len(transport.calls) == count
    if integrate and result["candidate_commit"]:
        git(workspace[0], "merge", "--ff-only", result["candidate_commit"])
    return plan, result


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_public_transition_is_materialized_independently_reviewed_and_replayed(workspace, harness):
    prepare(workspace)
    transport = Lifecycle("NEXT")
    plan, result = execute(workspace, "Select the correction for work", transport, harness)
    assert result["fidelity"] == dream_editor.CONTRACT
    assert [call["role"] for call in transport.calls] == ["author", "reviewer"]
    assert 'State "NEXT" from "TODO"' in (workspace[0] / TOPIC).read_text()
    receipt = json.loads(git(workspace[0], "show", result["candidate_commit"] + ":.sucoder/wiki/receipts/" + result["run_id"] + ".json"))
    frozen = next(page for page in transport.calls[0]["input"]["prior"] if page["path"] == TOPIC)
    assert receipt["editorial"]["task_actions"][0]["before_sha256"] == frozen["inventory"]["targets"][TASK]
    assert receipt["editorial"]["proposed_at"]
    assert receipt["review"]["task_checks"][0]["acceptance_satisfied"] == "yes"


def test_declared_human_task_cannot_be_completed_from_an_agent_claim(workspace):
    prepare(workspace)
    plan = planning(workspace, "Check the still-pending maintainer decision")
    with pytest.raises(dream.DreamError, match="declared human decision"):
        dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "model", transport=Lifecycle("DONE", target=HUMAN))
    assert not git(workspace[0], "for-each-ref", "refs/heads/dream/")


class ArchiveReader:
    version = "fresh-editorial-reader-fixture-1"

    def __init__(self):
        self.calls = []

    def __call__(self, harness, model, task, schema, deadline):
        self.calls.append(copy.deepcopy(task))
        reads = task["input"]["retrievals"]
        result = {key: task[key] for key in ("assignment_id", "input_sha256", "role")}
        result.update(failures=[], requests=[], answer="", evidence=[])
        kinds = [entry["request"]["kind"] for entry in reads]
        if "topic" not in kinds:
            result.update(action="lookup", requests=[dict(kind="topic", key="replies", target=TASK, query="", start=0, end=0)])
        elif "archive" not in kinds:
            assert "Archived editorial decision" in next(entry["response"]["text"] for entry in reads if entry["request"]["kind"] == "topic")
            result.update(action="lookup", requests=[dict(kind="archive", key="replies", target=TASK, query="", start=0, end=0)])
        else:
            history = next(entry["response"] for entry in reads if entry["request"]["kind"] == "archive")
            assert 'State "ACTIVE" from "REVIEW"' in history["text"]
            assert "queued-input" in history["text"]
            result.update(action="answer", answer="The correction returned for queued-input delivery context, retained that qualification and was independently checked before archival.", evidence=[history["read_id"]])
        return native_result(harness, model, task, result, self.version)


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_public_lifecycle_returned_review_supersession_archive_and_fresh_reader(workspace, harness):
    prepare(workspace)
    for state in ["NEXT", "ACTIVE", "WAITING", "NEXT", "ACTIVE", "REVIEW"]:
        execute(workspace, "Exercise " + state + " in the correction workflow", Lifecycle(state), harness, replay=False)
    active = (workspace[0] / TOPIC).read_text()
    assert ":BLOCKER: Need the queued-input context" not in active
    assert "BLOCKER: Need the queued-input context -> cleared" in active
    returned = Lifecycle("DONE", returned=True, authority="independent_review")
    execute(workspace, "Check completion; retain an explicit returned review", returned, harness, replay=False)
    assert [call["role"] for call in returned.calls] == ["author", "reviewer", "repair", "reviewer"]
    assert 'State "ACTIVE" from "REVIEW"' in (workspace[0] / TOPIC).read_text()
    execute(workspace, "Restore queued-input qualification and request review", Lifecycle("REVIEW", clarify=True), harness, replay=False)
    execute(workspace, "Complete the independently checked correction", Lifecycle("DONE", authority="independent_review"), harness, replay=False)
    execute(workspace, "Retire the superseded summary heuristic", Lifecycle("MOOT", target="infer-from-summary"), harness, replay=False)
    archive = Lifecycle(archive=True)
    plan, result = execute(workspace, "Archive eligible completed editorial tasks", archive, harness)
    active = (workspace[0] / TOPIC).read_text()
    historical = (workspace[0] / ARCHIVE).read_text()
    assert "queued_command" in active and "HISTORY_CANARY" not in active
    assert "WAITING Maintainer timeout decision" in active and ":REQUIRES_HUMAN: yes" in active
    assert "* DONE Correct the reply rule" in historical and "* MOOT Infer a reply from a summary" in historical
    assert 'State "ACTIVE" from "REVIEW"' in historical
    assert ":ARCHIVE_ITAGS: editorial" in historical and ":ARCHIVE_FILE: " + TOPIC in historical
    assert "file:../archive/replies.org::#corrected-reply-rule" in (workspace[0] / INCOMING).read_text()
    ordinary = dream_reader.read(workspace[2], "sample", "replies")
    assert "HISTORY_CANARY" not in ordinary["text"] and "queued_command" in ordinary["text"]
    specific = dream_reader.read(workspace[2], "sample", "replies", archives=True, target=TASK)
    assert "HISTORY_CANARY" in specific["text"] and 'State "DONE" from "REVIEW"' in specific["text"]
    transport = ArchiveReader()
    answer = dream_reader_agent.ask(workspace[2], "sample", Path(plan["plan_path"]),
        "Which missing qualification returned the correction, and how was it checked?", harness, "model", transport=transport)
    assert answer["state"] == "answered" and answer["topic_delivered"]
    assert [entry["request"]["kind"] for entry in answer["retrievals"]] == ["orientation", "topic", "archive"]
    assert len(transport.calls) == 3
    assert dream_reader_agent.ask(workspace[2], "sample", Path(plan["plan_path"]),
        "Which missing qualification returned the correction, and how was it checked?", harness, "model", transport=transport) == answer
    assert len(transport.calls) == 3


def human_request(workspace, kind, actor="user", *, summary=False):
    text = "Maintainer decision: keep the ten-second timeout; the required human review is complete."
    if kind == "codex-jsonl":
        rows = [dict(type="session_meta", payload=dict(id="human-decision")),
            dict(type="compacted", payload=dict(message=text)) if summary else
            dict(type="response_item", payload=dict(type="message", role=actor, content=[dict(type="input_text" if actor == "user" else "output_text", text=text)]))]
    else:
        rows = [dict(type=actor, sessionId="human-decision", isCompactSummary=summary, message=dict(content=text))]
    path = workspace[1].parent / "explicit-human-decision.jsonl"
    path.write_bytes(b"".join(dream.canonical(row) + b"\n" for row in rows))
    request = dict(workspace[3], units=[dict(unit_id="decision", kind=kind, shareable=True,
        locator=dict(path=str(path), thread_id="human-decision"))])
    return request, text


@pytest.mark.parametrize("kind", ["codex-jsonl", "claude-jsonl"])
def test_recorded_native_human_decision_can_complete_a_waiting_task(workspace, kind):
    prepare(workspace)
    request, quote = human_request(workspace, kind)
    transport = Lifecycle("DONE", target=HUMAN, authority="recorded_human", quote=quote, reviewer_human=True)
    execute(workspace, "Record the supplied maintainer decision", transport, request=request)
    text = (workspace[0] / TOPIC).read_text()
    assert "DONE Maintainer timeout decision" in text and ":REQUIRES_HUMAN: yes" in text
    assert "BLOCKER: Await recorded maintainer decision -> cleared" in text


@pytest.mark.parametrize("actor,summary", [("assistant", False), ("user", True)])
def test_assistant_or_summary_claims_do_not_become_human_authority(workspace, actor, summary):
    prepare(workspace)
    request, quote = human_request(workspace, "claude-jsonl", actor, summary=summary)
    plan = planning(workspace, "Check claimed human authority", request)
    with pytest.raises(dream.DreamError, match="native non-summary user evidence"):
        dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "model",
            transport=Lifecycle("DONE", target=HUMAN, authority="recorded_human", quote=quote, reviewer_human=True))
    assert not git(workspace[0], "for-each-ref", "refs/heads/dream/")


def test_accept_verdict_with_failed_authority_check_cannot_publish_completion(workspace):
    prepare(workspace)
    request, quote = human_request(workspace, "codex-jsonl")
    transport = Lifecycle("DONE", target=HUMAN, authority="recorded_human", quote=quote,
        reviewer_human=True, authority_verified=False, drop_after_reject=True)
    _, result = execute(workspace, "Keep unresolved authority pending", transport, request=request)
    assert result["state"] == "unchanged" and result["candidate_commit"] is None
    assert "WAITING Maintainer timeout decision" in (workspace[0] / TOPIC).read_text()


@pytest.mark.parametrize("defect,expected", [("hash", "existing frozen task"), ("state", "preserve existing task"),
    ("human_flag", "preserve existing task"), ("unanchored", "stable CUSTOM_ID"), ("checks", "exactly one check"),
    ("references", "Duplicate authored")])
def test_invalid_editorial_changes_never_publish(workspace, defect, expected):
    prepare(workspace)
    class Defective(Lifecycle):
        def __call__(self, harness, model, task, schema, deadline):
            result, _, _ = super().__call__(harness, model, task, schema, deadline)
            if task["role"] == "reviewer":
                if defect == "checks":
                    result["task_checks"] = []
            elif defect == "hash":
                result["task_actions"][0]["before_sha256"] = "0" * 64
            elif defect == "references":
                reference = dict(path=TOPIC, target=TASK, evidence_status="inference", citations=[dict(
                    unit_id=task["input"]["sources"][0]["unit_id"], quote=task["input"]["sources"][0]["text"])])
                result["references"] = [reference, copy.deepcopy(reference)]
            elif defect != "checks":
                old = next(page for page in task["input"]["prior"] if page["path"] == TOPIC)
                text = old["content"]
                if defect == "state":
                    text = text.replace("TODO Correct the reply rule", "DONE Correct the reply rule")
                elif defect == "human_flag":
                    text = text.replace(":REQUIRES_HUMAN: yes", ":REQUIRES_HUMAN: no")
                else:
                    text += "*** DONE Invented completed decision\nA fabricated result.\n"
                result.update(pages=[dict(path=TOPIC, content=text)], changes=[dict(path=TOPIC,
                    before_sha256=old["sha256"], reason="Invalid fixture change")])
            return native_result(harness, model, task, result, self.version)
    plan = planning(workspace, "Check invalid editorial changes")
    with pytest.raises(dream.DreamError, match=expected):
        dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "model", transport=Defective())
    assert not git(workspace[0], "for-each-ref", "refs/heads/dream/")


def test_named_review_cannot_be_replaced_with_the_authors_assertion(workspace):
    prepare(workspace)
    topic = workspace[0] / TOPIC
    topic.write_text(topic.read_text().replace(":CUSTOM_ID: corrected-reply-rule\n:OWNER: Sue\n",
        ":CUSTOM_ID: corrected-reply-rule\n:OWNER: Sue\n:REVIEWER: Alice\n"))
    git(workspace[0], "add", TOPIC)
    git(workspace[0], "commit", "-qm", "Require a named review for completion")
    class OwnAssertion(Lifecycle):
        def __call__(self, harness, model, task, schema, deadline):
            result, _, _ = super().__call__(harness, model, task, schema, deadline)
            if task["role"] != "reviewer":
                result["task_actions"][0]["reviewer"] = ""
            return native_result(harness, model, task, result, self.version)
    plan = planning(workspace, "Check the required named review")
    with pytest.raises(dream.DreamError, match="requires its named review"):
        dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "model", transport=OwnAssertion("DONE"))


def test_interrupted_editorial_publication_replays_frozen_time_without_inference(workspace, monkeypatch):
    prepare(workspace)
    plan = planning(workspace, "Recover an interrupted task transition")
    transport = Lifecycle("NEXT")
    store = dream_run._store
    def interrupted(fd, name, data):
        if name.endswith("-run.json"):
            raise KeyboardInterrupt("after atomic editorial publication")
        return store(fd, name, data)
    with monkeypatch.context() as patch:
        patch.setattr(dream_run, "_store", interrupted)
        with pytest.raises(KeyboardInterrupt):
            dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "model", transport=transport)
    assert dream_run.status(workspace[2], "sample", Path(plan["plan_path"]))["state"] == "candidate_ready"
    result = dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "model", transport=transport)
    assert result["state"] == "candidate_ready" and len(transport.calls) == 2


def test_public_reopening_preserves_the_prior_resolution(workspace):
    prepare(workspace)
    topic = workspace[0] / TOPIC
    topic.write_text(topic.read_text().replace("*** TODO Correct the reply rule\n", "*** DONE Correct the reply rule\nCLOSED: [2026-11-08 Sun 08:10]\n").replace(
        "Retain the native delivery and queued-input qualifications before completing this correction.",
        ':LOGBOOK:\n- State "DONE" from "REVIEW" [2026-11-08 Sun 08:10]\n  Earlier checked resolution.\n:END:\nRetain the native qualifications.'))
    git(workspace[0], "add", TOPIC)
    git(workspace[0], "commit", "-qm", "Seed an earlier completed resolution")
    execute(workspace, "Reopen after changed evidence without erasing the earlier result", Lifecycle("ACTIVE"), replay=False)
    text = topic.read_text()
    assert 'State "ACTIVE" from "DONE"' in text and "Earlier checked resolution." in text
    assert "ACTIVE Correct the reply rule" in text
    assert "CLOSED:" not in text.split("*** TODO Infer")[0]


def test_changed_private_execution_time_cannot_start_a_new_paid_assignment(workspace):
    prepare(workspace)
    transport = Lifecycle("NEXT")
    plan, result = execute(workspace, "Check the frozen editorial clock", transport, integrate=False)
    clock = Path(plan["plan_path"]).parent / ("editorial-time-" + result["run_id"] + ".json")
    clock.write_bytes(dream.canonical(dict(timestamp="2000-01-01T00:00:00Z")) + b"\n")
    with pytest.raises(dream.DreamError, match="Existing dream artifact differs"):
        dream_run.status(workspace[2], "sample", Path(plan["plan_path"]))
    with pytest.raises(dream.DreamError, match="Existing dream artifact differs"):
        dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "model", transport=transport)
    assert len(transport.calls) == 2
