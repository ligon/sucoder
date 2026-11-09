"""Curation may stop without edits, or repair an independently rejected no-op."""
import copy
import json
from pathlib import Path

import pytest

from sucoder import dream, dream_curate, dream_run, dream_wiki
from tests.test_dream import git, invoke, workspace  # noqa: F401
from tests.test_dream_wiki import EditorTransport, TOPIC, planned


def noop_plan(workspace, *, empty=False):
    repo = workspace[0]
    if not empty:
        path = repo / ".sucoder/wiki/topics/qualification.org"
        path.parent.mkdir(parents=True)
        path.write_text("#+title: Qualifications\n* Recorded qualifications\n:PROPERTIES:\n:CUSTOM_ID: qualifications\n:END:\nFirst qualification.\nSecond: caf\u00e9.\n")
        (repo / ".sucoder/wiki/index.org").write_text("#+title: Companion\n[[file:topics/qualification.org::#qualifications][Qualifications]]\n")
        git(repo, "add", ".sucoder/wiki")
        git(repo, "commit", "-qm", "Existing complete knowledge")
    result = invoke(workspace, None, "--maintenance", "Check whether useful changes are supported", "--topic", "qualification")
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


class NoopTransport(EditorTransport):
    def __init__(self, *, reject=False, repair_changes=False, defect=None):
        super().__init__()
        self.reject_noop, self.repair_changes, self.defect = reject, repair_changes, defect

    def __call__(self, harness, model, task, contract, deadline):
        if task["role"] == "repair" and self.repair_changes:
            return super().__call__(harness, model, task, contract, deadline)
        if task["role"] == "reviewer" and task["input"]["candidate"]["pages"]:
            return super().__call__(harness, model, task, contract, deadline)
        self.calls.append(copy.deepcopy(task))
        result = {key: task[key] for key in ("assignment_id", "input_sha256", "role")}
        result["failures"] = []
        if task["role"] == "reviewer":
            result.update(verdict="revise" if self.reject_noop else "accept", findings=[dict(severity="blocking", path=TOPIC,
                message="The existing summary-proof rule is wrong; correct it using the admitted source and retain the command qualification.")] if self.reject_noop else [])
        else:
            result.update(decision="unchanged", rationale="Useful selected knowledge is already represented; avoid duplicate editorial activity.",
                pages=[], changes=[], references=[], omissions=[])
            if self.defect == "reason": result["rationale"] = ""
            if self.defect == "edits": result["pages"] = [dict(path=dream_wiki.INDEX, content="A stealth edit")]
        if task["contract"] in ("source-grounded-wiki-v4", "source-grounded-wiki-v5"):
            result["requests"] = []
        if task["contract"] == "source-grounded-wiki-v5":
            result["task_checks" if task["role"] == "reviewer" else "task_actions"] = []
        if harness == "codex":
            events = [dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(result))), dict(type="turn.completed", usage={})]
        else:
            events = [dict(type="system", subtype="init", tools=["StructuredOutput"], mcp_servers=[]),
                dict(type="result", subtype="success", is_error=False, structured_output=result, usage={}, total_cost_usd=None)]
        trace = dict(harness=harness, model=model, version=self.version, returncode=0,
            input_sha256=dream.digest(dream.canonical(task)), stdout="\n".join(json.dumps(e) for e in events) + "\n",
            stderr="", boundary="linux-landlock-inline-v1")
        return result, dict(native={}, currency_usd=None), trace


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_reviewed_noop_has_private_coverage_no_candidate_and_cached_replay(workspace, harness):
    plan = noop_plan(workspace)
    repo, path = workspace[0], Path(plan["plan_path"])
    before = (git(repo, "show-ref"), git(repo, "status", "--porcelain"), (repo / ".git/index").read_bytes())
    transport = NoopTransport()
    result = dream_wiki.run(workspace[2], "sample", path, harness, "model", transport=transport)
    assert result["state"] == "unchanged"
    assert result["candidate_commit"] is result["candidate_ref"] is None
    assert result["integration"]["state"] == "no_changes_proposed"
    assert [t["role"] for t in transport.calls] == ["author", "reviewer"]
    assert all(t["input"]["sources"][0]["text"] == "First qualification.\r\nSecond: caf\u00e9.\n" for t in transport.calls)
    assert len(result["coverage"]) == 1
    assert before == (git(repo, "show-ref"), git(repo, "status", "--porcelain"), (repo / ".git/index").read_bytes())
    assert dream_run.status(workspace[2], "sample", path)["state"] == "unchanged"
    assert dream_wiki.run(workspace[2], "sample", path, harness, "model", transport=transport) == result
    assert len(transport.calls) == 2


def test_independent_review_can_reject_noop_and_obtain_correction(workspace):
    plan = planned(workspace)
    transport = NoopTransport(reject=True, repair_changes=True)
    result = dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "model", transport=transport)
    assert result["state"] == "candidate_ready"
    assert [t["role"] for t in transport.calls] == ["author", "reviewer", "repair", "reviewer"]
    assert "assertion is not proof" in git(workspace[0], "show", result["candidate_commit"] + ":" + TOPIC)
    receipt = json.loads(git(workspace[0], "show", result["candidate_commit"] + ":.sucoder/wiki/receipts/" + result["run_id"] + ".json"))
    assert receipt["decision"] == "change"
    assert receipt["contract"] == "source-grounded-wiki-v5"


@pytest.mark.parametrize("defect", ["reason", "edits"])
def test_noop_cannot_hide_edits_or_omit_its_reason(workspace, defect):
    plan = noop_plan(workspace)
    with pytest.raises(dream.DreamError, match="did not pass"):
        dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "model", transport=NoopTransport(defect=defect))
    assert not git(workspace[0], "for-each-ref", "refs/heads/dream/")


def test_changed_wiki_context_invalidates_private_noop(workspace):
    plan = noop_plan(workspace)
    path = Path(plan["plan_path"])
    dream_wiki.run(workspace[2], "sample", path, "codex", "model", transport=NoopTransport())
    repo = workspace[0]
    (repo / ".sucoder/wiki/new-policy.org").write_text("A new human decision changes the context.\n")
    git(repo, "add", ".sucoder/wiki/new-policy.org")
    git(repo, "commit", "-qm", "Record a new independent contribution")
    assert dream_run.status(workspace[2], "sample", path)["state"] == "needs_reconciliation"


def test_empty_initial_wiki_noop_is_not_shared_accepted_progress(workspace):
    plan = noop_plan(workspace, empty=True)
    path = Path(plan["plan_path"])
    result = dream_wiki.run(workspace[2], "sample", path, "claude", "model", transport=NoopTransport())
    assert result["state"] == "unchanged"
    assert dream_run.status(workspace[2], "sample", path)["integration"]["state"] == "no_changes_proposed"
    assert not (workspace[0] / ".sucoder/wiki/sources.json").exists()


def test_interrupted_noop_result_publication_is_recoverable_without_inference(workspace, monkeypatch):
    plan = noop_plan(workspace)
    path = Path(plan["plan_path"])
    transport = NoopTransport()
    store = dream_run._store
    def interrupted(fd, name, data):
        if name.endswith("-run.json"):
            raise KeyboardInterrupt("after complete no-op marker")
        return store(fd, name, data)
    with monkeypatch.context() as patch:
        patch.setattr(dream_run, "_store", interrupted)
        with pytest.raises(KeyboardInterrupt):
            dream_wiki.run(workspace[2], "sample", path, "codex", "model", transport=transport)
    assert dream_run.status(workspace[2], "sample", path)["state"] == "unchanged"
    assert dream_wiki.run(workspace[2], "sample", path, "codex", "model", transport=transport)["state"] == "unchanged"
    assert len(transport.calls) == 2


@pytest.mark.parametrize("kind", ["codex-jsonl", "claude-jsonl"])
def test_transcript_curation_receipt_pins_normalized_interpretation(workspace, kind):
    identity = "synthetic-noop-thread"
    if kind == "codex-jsonl":
        rows = [dict(type="session_meta", payload=dict(id=identity)),
            dict(type="response_item", payload=dict(type="message", role="user",
                content=[dict(type="input_text", text="First qualification.")]))]
    else:
        rows = [dict(type="user", sessionId=identity, uuid="synthetic-user", message=dict(content="First qualification."))]
    transcript = workspace[1].parent / "explicit-transcript.jsonl"
    transcript.write_bytes(b"".join(dream.canonical(row) + b"\n" for row in rows))
    request = dict(workspace[3], units=[dict(unit_id="transcript", kind=kind, shareable=True,
        locator=dict(path=str(transcript), thread_id=identity))])
    envelope = invoke(workspace, request)
    assert envelope.exit_code == 0, envelope.output
    plan = json.loads(envelope.output)
    result = dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "model", transport=NoopTransport())
    receipt = json.loads((Path(plan["plan_path"]).parent / ("native-" + result["run_id"] + "-receipt.json")).read_text())
    assert receipt["units"][0]["normalization"] == plan["units"][0]["normalization"]
    assert dream_run.status(workspace[2], "sample", Path(plan["plan_path"]))["state"] == "unchanged"
