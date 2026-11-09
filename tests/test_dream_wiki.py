"""Behavioral editor cases: multi-source explanation, correction and review."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from sucoder import dream, dream_run, dream_wiki as wiki
from tests.test_dream import git, invoke, workspace  # noqa: F401


TOPIC = ".sucoder/wiki/topics/reply-evidence.org"
POLICY = ("* Maintainer constraint\n:PROPERTIES:\n:CUSTOM_ID: timeout-policy\n:END:\n"
          "Keep the ten-second timeout pending the maintainer's decision.\n"
          "** Talk :editorial:\n*** WAITING Maintainer review\n:PROPERTIES:\n"
          ":CUSTOM_ID: timeout-review\n:END:\nDo not change the timeout without the recorded decision.\n")
OLD = ("#+title: Reply evidence\n#+TODO: TODO NEXT ACTIVE WAITING REVIEW | DONE MOOT\n\n"
       "* Reply proof\n:PROPERTIES:\n:CUSTOM_ID: reply-proof\n:END:\n"
       "Any reply tag in a compaction summary proves that a reply was sent.\n\n" + POLICY)


def planned(workspace):
    repo = workspace[0]
    (repo / "evidence-a.org").write_text("A compaction summary reports earlier events; it does not establish a native assistant reply.\n")
    (repo / "evidence-b.org").write_text("Reply evidence is native assistant text after the matching delivery and before the next input. Use --session-id <uuid> at launch; this is documented capability, not a freshly tested launch.\n")
    (repo / TOPIC).parent.mkdir(parents=True)
    (repo / TOPIC).write_text(OLD)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Seed existing project understanding and two sources")
    request = dict(workspace[3], units=[dict(unit_id=name, kind="git", shareable=True,
        locator=dict(repository="sample", commit="HEAD", path=name + ".org")) for name in ("evidence-a", "evidence-b")])
    result = invoke(workspace, request)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


class EditorTransport:
    version = "editor-fixture-1"

    def __init__(self, *, omit_flag=False, mutate=None, reject=False):
        self.calls = []
        self.omit_flag, self.mutate, self.reject = omit_flag, mutate, reject

    def __call__(self, harness, model, task, contract, deadline):
        self.calls.append(copy.deepcopy(task))
        result = {key: task[key] for key in ("assignment_id", "input_sha256", "role")}
        result["failures"] = []
        inputs = task["input"]
        if task["role"] == "reviewer":
            content = next(page["content"] for page in inputs["candidate"]["pages"] if page["path"] == TOPIC)
            bad = self.reject or "--session-id <uuid>" not in content
            result.update(verdict="revise" if bad else "accept", findings=[dict(severity="blocking", path=TOPIC,
                message="The readable explanation omits the essential --session-id <uuid> qualification.")] if bad else [])
        else:
            flag = "" if self.omit_flag and task["role"] == "author" else "Use =--session-id <uuid>= at launch; this is documented capability, not a newly tested launch.\n"
            content = ("#+title: Reply evidence\n#+TODO: TODO NEXT ACTIVE WAITING REVIEW | DONE MOOT\n\n"
                       "* Reply proof\n:PROPERTIES:\n:CUSTOM_ID: reply-proof\n:END:\n"
                       "A compaction summary's assertion is not proof of a delivered reply.\n"
                       "Native assistant text must follow the matching delivery and precede the next input.\n" + flag +
                       "See [[file:../../../evidence-a.org][summary evidence]] and [[file:../../../evidence-b.org][delivery evidence]].\n\n" + POLICY)
            index = ("#+title: Companion wiki\n* Orientation\n:PROPERTIES:\n:CUSTOM_ID: orientation\n:END:\n"
                     "[[file:topics/reply-evidence.org::#reply-proof][How to recognize reply evidence]]\n")
            before = next(item["sha256"] for item in inputs["prior"] if item["path"] == TOPIC)
            result.update(pages=[dict(path=TOPIC, content=content), dict(path=wiki.INDEX, content=index)],
                changes=[dict(path=TOPIC, before_sha256=before, reason="Correct a mistaken rule using both admitted sources; preserve the maintainer's unresolved constraint."),
                         dict(path=wiki.INDEX, before_sha256="", reason="Provide a short topic-oriented entry point.")],
                references=[dict(path=TOPIC, target="reply-proof", evidence_status="inference", citations=[
                    dict(unit_id=s["unit_id"], quote=s["text"]) for s in inputs["sources"]])],
                omissions=[])
        if task["contract"] in ("source-grounded-wiki-v3", "source-grounded-wiki-v4", "source-grounded-wiki-v5") and task["role"] != "reviewer":
            result.update(decision="change", rationale="Correct unsupported memory and connect the two necessary sources.")
        if task["contract"] in ("source-grounded-wiki-v4", "source-grounded-wiki-v5"):
            result["requests"] = []
        if task["contract"] == "source-grounded-wiki-v5":
            result["task_checks" if task["role"] == "reviewer" else "task_actions"] = []
        if self.mutate:
            self.mutate(task, result)
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
def test_corrects_existing_topic_and_combines_sources_preserving_human_task(workspace, harness):
    plan = planned(workspace)
    repo = workspace[0]
    before = ((repo / ".git/index").read_bytes(), git(repo, "status", "--porcelain"), git(repo, "symbolic-ref", "HEAD"))
    transport = EditorTransport()
    result = wiki.run(workspace[2], "sample", Path(plan["plan_path"]), harness, "explicit-model", transport=transport)
    assert result["state"] == "candidate_ready"
    assert [t["role"] for t in transport.calls] == ["author", "reviewer"]
    for task in transport.calls:
        assert {s["unit_id"] for s in task["input"]["sources"]} == {"evidence-a", "evidence-b"}
        assert task["input"]["editing"]["index_path"] == ".sucoder/wiki/index.org"
        assert task["input"]["editing"]["topic_path_prefix"] == ".sucoder/wiki/topics/"
    content = git(repo, "show", result["candidate_commit"] + ":" + TOPIC)
    assert "is not proof" in content
    assert "precede the next input" in content
    assert "--session-id <uuid>" in content
    assert POLICY.strip() in content
    assert "Claim:" not in content
    assert before == ((repo / ".git/index").read_bytes(), git(repo, "status", "--porcelain"), git(repo, "symbolic-ref", "HEAD"))
    assert wiki.run(workspace[2], "sample", Path(plan["plan_path"]), harness, "explicit-model", transport=transport) == result
    assert len(transport.calls) == 2
    assert dream_run.status(workspace[2], "sample", Path(plan["plan_path"]))["state"] == "candidate_ready"
    receipt = json.loads(git(repo, "show", result["candidate_commit"] + ":.sucoder/wiki/receipts/" + result["run_id"] + ".json"))
    assert receipt["contract"] == "source-grounded-wiki-v5"
    assert len(receipt["references"][0]["citations"]) == 2
    assert all(b["rule"] == "file-utf8-v1" for b in receipt["outputs"])


def test_independent_review_catches_omission_and_repairs_once(workspace):
    plan = planned(workspace)
    transport = EditorTransport(omit_flag=True)
    result = wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "model", transport=transport)
    assert [t["role"] for t in transport.calls] == ["author", "reviewer", "repair", "reviewer"]
    assert result["usage"]["role_invocations"] == 4
    assert "--session-id <uuid>" in git(workspace[0], "show", result["candidate_commit"] + ":" + TOPIC)


@pytest.mark.parametrize("defect", ["path", "base", "anchor", "task", "quote", "link", "directive", "orphan", "home-link"])
def test_mechanical_rejections_never_publish(workspace, defect):
    plan = planned(workspace)
    def mutate(task, result):
        if task["role"] == "reviewer":
            return
        if defect == "path": result["pages"][0]["path"] = "AGENTS.md"
        if defect == "base": result["changes"][0]["before_sha256"] = "f" * 64
        if defect == "anchor": result["pages"][0]["content"] = result["pages"][0]["content"].replace(":CUSTOM_ID: reply-proof", ":CUSTOM_ID: replacement")
        if defect == "task": result["pages"][0]["content"] = result["pages"][0]["content"].replace("WAITING Maintainer", "DONE Maintainer")
        if defect == "quote": result["references"][0]["citations"][0]["quote"] = "Not the supplied evidence"
        if defect == "link": result["pages"][0]["content"] += "[[file:missing.org::#absent][Missing]]\n"
        if defect == "directive": result["pages"][0]["content"] += "#+INCLUDE: /private\n"
        if defect == "orphan": result["pages"][1]["content"] = "#+title: Empty orientation\n"
        if defect == "home-link": result["pages"][0]["content"] += "[[file:~/outside.org][Private]]\n"
    with pytest.raises(dream.DreamError):
        wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "model", transport=EditorTransport(mutate=mutate))
    assert not git(workspace[0], "for-each-ref", "refs/heads/dream/")


def test_inventory_does_not_expand_home_or_remote_paths():
    import time
    data = wiki.inventory({wiki.INDEX: "#+title: Index\n[[file:~/outside.org][Home]]\n[[file:/ssh:host:/secret][Remote]]\n"}, time.monotonic() + 5)
    assert [link["path"] for link in data[wiki.INDEX]["links"]] == ["~/outside.org", "/ssh:host:/secret"]


def test_literal_directives_in_an_example_remain_inert(workspace):
    plan = planned(workspace)
    def quoted(task, result):
        if task["role"] != "reviewer":
            result["pages"][0]["content"] = result["pages"][0]["content"].replace(
                "See [[file:", "#+begin_example\n#+INCLUDE: /private\n#+end_example\nSee [[file:")
    result = wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "model", transport=EditorTransport(mutate=quoted))
    assert result["state"] == "candidate_ready"


def test_cli_defaults_to_editor_and_binds_operator_purpose(workspace, monkeypatch):
    from tests.test_dream_run import command
    plan = planned(workspace)
    transport = EditorTransport()
    monkeypatch.setattr(wiki.dream_native, "Transport", lambda harness, deadline: transport)
    result = command(workspace, "run", plan, "--harness", "codex", "--model", "explicit-model",
                     "--purpose", "Explain how reply evidence differs from historical summaries.")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["fidelity"] == "source-grounded-wiki-v5"
    assert transport.calls[0]["input"]["purpose"] == "Explain how reply evidence differs from historical summaries."
    status = command(workspace, "status", plan)
    assert status.exit_code == 0, status.output
    assert len(transport.calls) == 2


def test_unresolved_semantic_findings_stop_after_single_repair(workspace):
    plan = planned(workspace)
    transport = EditorTransport(reject=True)
    with pytest.raises(dream.DreamError, match="independent review"):
        wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "claude", "model", transport=transport)
    assert [t["role"] for t in transport.calls] == ["author", "reviewer", "repair", "reviewer"]
    assert not git(workspace[0], "for-each-ref", "refs/heads/dream/")


def test_reviewer_must_explain_a_revision_before_spending_repair(workspace):
    plan = planned(workspace)
    def unexplained(task, result):
        if task["role"] == "reviewer":
            result.update(verdict="revise", findings=[])
    transport = EditorTransport(mutate=unexplained)
    with pytest.raises(dream.DreamError, match="actionable blocking finding"):
        wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "model", transport=transport)
    assert [t["role"] for t in transport.calls] == ["author", "reviewer"]


def test_status_checks_whole_files_including_unanchored_intro_after_squash(workspace):
    plan = planned(workspace)
    repo = workspace[0]
    result = wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "model", transport=EditorTransport())
    tree = git(repo, "rev-parse", result["candidate_commit"] + "^{tree}")
    squash = subprocess.check_output(["git", "-C", str(repo), "commit-tree", tree, "-p", plan["base_commit"], "-m", "Human squash"]).decode().strip()
    git(repo, "update-ref", "refs/heads/main", squash, plan["base_commit"])
    assert dream_run.status(workspace[2], "sample", Path(plan["plan_path"]))["integration"]["state"] == "integrated_candidate"
    git(repo, "read-tree", "--reset", "-u", squash)
    page = repo / TOPIC
    page.write_text(page.read_text().replace("#+title: Reply evidence", "#+title: Human-edited orientation"))
    git(repo, "add", TOPIC)
    git(repo, "commit", "-qm", "Change only text outside anchored subtrees")
    assert dream_run.status(workspace[2], "sample", Path(plan["plan_path"]))["integration"]["state"] == "needs_reconciliation"


@pytest.mark.parametrize("quote,accepted", [("\u03bb\nunique", True), ("alpha", False), ("missing", False), ("", False)])
def test_quote_resolution_is_unique_exact_utf8_and_absolute(quote, accepted):
    import time
    text = "alpha \u03bb\nunique evidence\nalpha \u03bb\n"
    source = dict(unit_id="selected", byte_start=101, byte_end=101 + len(text.encode()), text=text)
    proposal = dict(pages=[dict(path=wiki.INDEX, content="[[file:topics/rule.org::#rule][Rule]]\n"),
                          dict(path=".sucoder/wiki/topics/rule.org", content="* Rule\n:PROPERTIES:\n:CUSTOM_ID: rule\n:END:\nExplanation.\n")],
                    changes=[dict(path=path, before_sha256="", reason="Explain evidence") for path in
                             (wiki.INDEX, ".sucoder/wiki/topics/rule.org")],
                    references=[dict(path=".sucoder/wiki/topics/rule.org", target="rule", evidence_status="inference",
                                     citations=[dict(unit_id="selected", quote=quote)])], omissions=[])
    plan = dict(repository="sample", limits=dict(max_consolidation_bytes=65536))
    if not accepted:
        with pytest.raises(dream.DreamError, match="occurring once"):
            wiki.validate_proposal(plan, proposal, {}, [source], time.monotonic() + 5)
    else:
        references = wiki.validate_proposal(plan, proposal, {}, [source], time.monotonic() + 5)[2]
        citation = references[0]["citations"][0]
        assert citation["byte_start"] == 107
        assert citation["byte_end"] == 107 + len(quote.encode())
        assert "byte_start" not in proposal["references"][0]["citations"][0]


def test_legacy_wiki_v1_receipt_replays_with_original_assignments(workspace):
    import time
    plan = planned(workspace)
    class LegacyTransport(EditorTransport):
        def __call__(self, harness, model, task, contract, deadline):
            result, usage, trace = super().__call__(harness, model, task, contract, deadline)
            for reference in result.get("references", []):
                for citation in reference["citations"]:
                    unit = next(s for s in task["input"]["sources"] if s["unit_id"] == citation["unit_id"])
                    citation.update(byte_start=unit["byte_start"], byte_end=unit["byte_end"])
            trace["stdout"] = json.dumps(dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(result)))) + "\n" + json.dumps(dict(type="turn.completed", usage={})) + "\n"
            return result, usage, trace
    deadline = time.monotonic() + 30
    with dream_run._directory(workspace[2], Path(plan["plan_path"])) as (fd, directory):
        plan = dream_run._load(fd, directory.name, "sample")
        _, sources = dream_run._snapshot(workspace[2], "sample", plan, fd, deadline)
        coordinator = wiki.WikiCoordinator(plan, fd, "codex", "model", deadline, LegacyTransport(), contract=wiki.LEGACY_CONTRACT)
        prefix = "native-" + coordinator.run_id
        dream_run._store(fd, prefix + "-spec.json", dream.canonical(dict(contract=wiki.LEGACY_CONTRACT,
            run_id=coordinator.run_id, harness="codex", model="model", native_version=coordinator.expected_version,
            attempt=1, purpose=wiki.PURPOSE)) + b"\n")
        pages, receipt = coordinator.prepare()
        result = dream_run.publish(plan, fd, directory, sources, deadline, wiki.INDEX, pages[wiki.INDEX], receipt, "",
            receipt_name=prefix + "-receipt.json", result_name=prefix + "-run.json", wiki_files=pages)
    status = dream_run.status(workspace[2], "sample", directory / "plan.json")
    assert status["run_id"] == result["run_id"]
    assert status["state"] == "candidate_ready"
