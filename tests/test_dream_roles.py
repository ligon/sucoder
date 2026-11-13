"""Schema-bound semantic roles against disposable repositories and native traces."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from sucoder import dream, dream_native, dream_roles as roles, dream_run
from tests.test_dream import git, invoke, workspace  # noqa: F401
from tests.test_dream_transcript import catalog


class Transport:
    version = "fixture-native-1"
    def __init__(self, change=None):
        self.calls = []
        self.change = change

    def __call__(self, harness, model, task, schema, deadline):
        self.calls.append(copy.deepcopy(task))
        result = {k: task[k] for k in ("assignment_id", "input_sha256", "role")}
        result["failures"] = []
        inputs, role = task["input"], task["role"]
        if role in ("extractor", "verifier"):
            source = inputs["evidence"]
            citation = dict(unit_id=source["unit_id"], byte_start=source["byte_start"],
                            byte_end=source["byte_end"], quote=source["text"])
        if role == "extractor":
            result["claims"] = [dict(claim_id="local", statement="A qualification is recorded in the admitted source.",
                qualification="This is evidence of the recorded text only.", evidence_status="observation", citations=[citation])]
        elif role == "verifier":
            result["verdicts"] = [dict(claim_id=claim["claim_id"], status="supported", reason="The literal evidence records this qualification.",
                                      citations=[citation]) for claim in inputs["claims"]]
        else:
            result["sections"] = [dict(title="Recorded qualifications", claim_ids=[c["claim_id"] for c in inputs["claims"]])]
        if self.change:
            self.change(task, result)
        if harness == "codex":
            events = [dict(type="thread.started", thread_id="fixture"),
                      dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(result))),
                      dict(type="turn.completed", usage={})]
        else:
            events = [dict(type="system", subtype="init", tools=["StructuredOutput"], mcp_servers=[]),
                      dict(type="assistant", message=dict(content=[dict(type="tool_use", name="StructuredOutput", input=result)])),
                      dict(type="result", subtype="success", is_error=False, structured_output=result, usage={}, total_cost_usd=None)]
        trace = dict(harness=harness, model=model, version="fixture-native-1", boundary="linux-landlock-inline-v1",
                     input_sha256=dream.digest(dream.canonical(task)),
                     stdout="\n".join(json.dumps(e) for e in events) + "\n", stderr="", returncode=0)
        return result, dict(native={}, currency_usd=None), trace


def plan(workspace, request=None):
    result = invoke(workspace, request)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


@pytest.mark.parametrize("harness", ["codex", "claude"])
@pytest.mark.parametrize("kind", ["git", "codex-jsonl", "claude-jsonl"])
def test_native_roles_publish_and_restore_without_inference(workspace, harness, kind):
    request = None if kind == "git" else catalog(workspace, kind)[0]
    frozen = plan(workspace, request)
    repo = workspace[0]
    (repo / "note.org").write_text("human staged")
    git(repo, "add", "note.org")
    (repo / "note.org").write_text("human unstaged")
    before = (git(repo, "status", "--porcelain"), (repo / ".git/index").read_bytes(), git(repo, "symbolic-ref", "HEAD"))
    transport = Transport()
    result = roles.run(workspace[2], "sample", Path(frozen["plan_path"]), harness, "explicit-model", transport=transport)
    assert result["state"] == "candidate_ready"
    assert result["harness"] == harness
    assert result["fidelity"] == roles.CONTRACT
    assert [task["role"] for task in transport.calls] == ["extractor", "verifier", "consolidator"]
    assert before == (git(repo, "status", "--porcelain"), (repo / ".git/index").read_bytes(), git(repo, "symbolic-ref", "HEAD"))
    receipt_path = ".sucoder/wiki/receipts/" + result["run_id"] + ".json"
    receipt = json.loads(git(repo, "show", result["candidate_commit"] + ":" + receipt_path))
    assert receipt["usage"]["role_invocations"] == 3
    assert str(repo) not in json.dumps(receipt)
    assert all(c["verification"] == "model_checked" for c in receipt["coverage"])
    # Retry reuses bound traces/results and spends no further provider calls.
    assert roles.run(workspace[2], "sample", Path(frozen["plan_path"]), harness, "explicit-model", transport=transport) == result
    assert len(transport.calls) == 3
    status = dream_run.status(workspace[2], "sample", Path(frozen["plan_path"]))
    assert status["integration"]["state"] == "not_integrated"
    tree = git(repo, "rev-parse", result["candidate_commit"] + "^{tree}")
    commit = subprocess.check_output(["git", "-C", str(repo), "commit-tree", tree, "-p", frozen["base_commit"], "-m", "human squash"]).decode().strip()
    git(repo, "update-ref", "refs/heads/main", commit, frozen["base_commit"])
    assert dream_run.status(workspace[2], "sample", Path(frozen["plan_path"]))["integration"]["state"] == "integrated_candidate"


@pytest.mark.parametrize("defect", ["citation", "range", "assignment", "schema", "failure", "duplicate", "missing-verdict", "invented-claim", "active-org"])
def test_role_defects_prevent_publication(workspace, defect):
    frozen = plan(workspace)
    def change(task, result):
        if task["role"] == "extractor":
            if defect == "citation": result["claims"][0]["citations"][0]["quote"] = "invented"
            if defect == "range": result["claims"][0]["citations"][0]["byte_end"] += 1
            if defect == "assignment": result["assignment_id"] = "wrong"
            if defect == "schema": result["unknown"] = True
            if defect == "failure": result["failures"] = ["unread input"]
            if defect == "duplicate": result["claims"] *= 2
            if defect == "active-org": result["claims"][0]["statement"] = "[[file:/private][hidden link]]"
        if task["role"] == "verifier" and defect == "missing-verdict": result["verdicts"] = []
        if task["role"] == "consolidator" and defect == "invented-claim": result["sections"][0]["claim_ids"].append("invented")
    with pytest.raises(dream.DreamError):
        roles.run(workspace[2], "sample", Path(frozen["plan_path"]), "codex", "model", transport=Transport(change))
    assert not git(workspace[0], "for-each-ref", "refs/heads/dream/")


def test_unresolved_claim_is_qualified_and_contradicted_claim_never_published(workspace):
    frozen = plan(workspace)
    def unresolved(task, result):
        if task["role"] == "verifier": result["verdicts"][0]["status"] = "unresolved"
    result = roles.run(workspace[2], "sample", Path(frozen["plan_path"]), "claude", "model", transport=Transport(unresolved))
    page = git(workspace[0], "show", result["candidate_commit"] + ":.sucoder/wiki/topics/dream-" + result["run_id"] + ".org")
    assert "model verification: unresolved" in page
    def contradicted(task, result):
        if task["role"] == "verifier": result["verdicts"][0]["status"] = "contradicted"
    with pytest.raises(dream.DreamError, match="No publishable claims"):
        roles.run(workspace[2], "sample", Path(frozen["plan_path"]), "codex", "model", transport=Transport(contradicted))


def test_failed_native_request_is_not_automatically_repeated(workspace):
    frozen = plan(workspace)
    calls = []
    def failed(*args):
        calls.append(args)
        return None, None, dict(returncode=1, stderr="rate limit", stdout="", version="fixture-native-1",
            harness=args[0], model=args[1], input_sha256=dream.digest(dream.canonical(args[2])),
            boundary="linux-landlock-inline-v1")
    for expected in ("Native role failed", "earlier role attempt has no validated result"):
        with pytest.raises(dream.DreamError, match=expected):
            roles.run(workspace[2], "sample", Path(frozen["plan_path"]), "claude", "model", transport=failed)
    assert len(calls) == 1
    assert list(Path(frozen["plan_path"]).parent.glob("*-trace.json"))


def test_mutated_saved_result_cannot_substitute_for_native_trace(workspace):
    frozen = plan(workspace)
    roles.run(workspace[2], "sample", Path(frozen["plan_path"]), "codex", "model", transport=Transport())
    directory = Path(frozen["plan_path"]).parent
    for path in directory.glob("*-result.json"):
        result = json.loads(path.read_bytes())
        if result["result"]["role"] == "extractor":
            result["result"]["claims"][0]["statement"] = "A modified assertion."
            path.write_bytes(dream.canonical(result))
            break
    with pytest.raises(dream.DreamError, match="differs from the completed native trace"):
        dream_run.status(workspace[2], "sample", Path(frozen["plan_path"]))


def test_multiple_harnesses_require_explicit_status_selection(workspace):
    frozen = plan(workspace)
    results = [roles.run(workspace[2], "sample", Path(frozen["plan_path"]), harness, "model", transport=Transport())
               for harness in ("codex", "claude")]
    with pytest.raises(dream.DreamError, match="Multiple native runs"):
        dream_run.status(workspace[2], "sample", Path(frozen["plan_path"]))
    for result in results:
        assert dream_run.status(workspace[2], "sample", Path(frozen["plan_path"]), result["run_id"])["run_id"] == result["run_id"]


def test_every_chunk_has_separate_extraction_and_verification(workspace):
    request = dict(workspace[3], limits=dict(chunk_bytes=12))
    frozen = plan(workspace, request)
    transport = Transport()
    result = roles.run(workspace[2], "sample", Path(frozen["plan_path"]), "codex", "model", transport=transport)
    assert len(result["coverage"]) == frozen["work"]["chunks"]
    assert len(transport.calls) == frozen["work"]["required_role_invocations"]
    assert sum(c["byte_end"] - c["byte_start"] for c in result["coverage"]) == frozen["work"]["selected_bytes"]


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_cli_native_pipeline_uses_isolated_subprocesses(workspace, monkeypatch, harness):
    from tests.test_dream_run import command
    scratch = workspace[1].parent
    executable = scratch / harness
    # This executable fixture exercises the real launcher, stdin, schema,
    # authentication staging, Landlock boundary, traces and publisher.
    executable.write_text('''#!/usr/bin/python3 -S
import json,sys
if '--version' in sys.argv:
    print('native-fixture 1.0'); raise SystemExit
if '--help' in sys.argv:
    print('--ignore-user-config --output-schema --ephemeral --ignore-rules --restricted --safe-mode --json-schema --tools --permission-prompts')
    raise SystemExit
task=json.load(sys.stdin)
result={k:task[k] for k in ('assignment_id','input_sha256','role')}
result['failures']=[]
role=task['role']; inputs=task['input']
if role in ('extractor','verifier'):
    source=inputs['evidence']
    citation={k:source[k] for k in ('unit_id','byte_start','byte_end')}
    citation['quote']=source['text']
if role=='extractor':
    result['claims']=[dict(claim_id='fixture',statement='The source records a qualification.',qualification='Recorded text only.',evidence_status='observation',citations=[citation])]
elif role=='verifier':
    result['verdicts']=[dict(claim_id=c['claim_id'],status='supported',reason='The source contains this qualification.',citations=[citation]) for c in inputs['claims']]
else:
    result['sections']=[dict(title='Recorded qualifications',claim_ids=[c['claim_id'] for c in inputs['claims']])]
if '--json-schema' in sys.argv:
    print(json.dumps(dict(type='system',subtype='init',tools=['StructuredOutput'],mcp_servers=[])))
    print(json.dumps(dict(type='result',subtype='success',is_error=False,structured_output=result,usage={})))
else:
    print(json.dumps(dict(type='item.completed',item=dict(type='agent_message',text=json.dumps(result)))))
    print(json.dumps(dict(type='turn.completed',usage={})))
''')
    executable.chmod(0o700)
    import os
    monkeypatch.setenv("PATH", str(scratch) + os.pathsep + os.environ["PATH"])
    frozen = plan(workspace)
    result = command(workspace, "run", frozen, "--harness", harness, "--model", "fixture-model", "--mode", "notes")
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["harness"] == harness
    status = command(workspace, "status", frozen)
    assert status.exit_code == 0, status.output
    assert json.loads(status.output)["run_id"] == value["run_id"]


def test_concurrent_native_coordinator_cannot_repeat_provider_work(workspace):
    frozen = plan(workspace)
    directory = Path(frozen["plan_path"]).parent
    import os
    import time
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        from sucoder import dream_budget
        transport = Transport()
        with dream_budget.open_job(frozen, directory) as job:
            coordinator = roles.Coordinator(frozen, fd, "codex", "model", time.monotonic() + 5, Transport())
            job.attach(coordinator)
            # A different attempt must also respect the whole-job lock.
            with pytest.raises(dream.DreamError, match="Another coordinator"):
                roles.run(workspace[2], "sample", Path(frozen["plan_path"]), "codex", "model", attempt=2, transport=transport)
        assert transport.calls == []
    finally:
        os.close(fd)


def test_explicit_attempt_retries_without_overwriting_failed_attempt(workspace):
    frozen = plan(workspace)
    def failed(*args):
        return None, None, dict(returncode=1, stderr="quota", stdout="", version="fixture-native-1",
            harness=args[0], model=args[1], input_sha256=dream.digest(dream.canonical(args[2])),
            boundary="linux-landlock-inline-v1")
    failed.version = Transport.version
    with pytest.raises(dream.DreamError, match="Native role failed"):
        roles.run(workspace[2], "sample", Path(frozen["plan_path"]), "claude", "model", transport=failed)
    assert dream_run.status(workspace[2], "sample", Path(frozen["plan_path"]))["state"] == "incomplete"
    result = roles.run(workspace[2], "sample", Path(frozen["plan_path"]), "claude", "model", attempt=2, transport=Transport())
    assert dream_run.status(workspace[2], "sample", Path(frozen["plan_path"]), result["run_id"])["state"] == "candidate_ready"
    assert len(list(Path(frozen["plan_path"]).parent.glob("native-*-spec.json"))) == 2


def test_native_status_detects_output_edit_even_when_receipt_survives(workspace):
    frozen = plan(workspace)
    repo = workspace[0]
    result = roles.run(workspace[2], "sample", Path(frozen["plan_path"]), "codex", "model", transport=Transport())
    git(repo, "read-tree", "--reset", "-u", result["candidate_commit"])
    page = repo / (".sucoder/wiki/topics/dream-" + result["run_id"] + ".org")
    page.write_text(page.read_text().replace("qualification is recorded", "unsupported outcome is claimed"))
    git(repo, "add", ".sucoder/wiki")
    git(repo, "commit", "-qm", "Human change retains the original receipt")
    assert dream_run.status(workspace[2], "sample", Path(frozen["plan_path"]))["integration"]["state"] == "needs_reconciliation"


def test_plain_role_text_preserves_significant_internal_spaces():
    assert roles._plain('Use --label="two  spaces".') == 'Use --label="two  spaces".'
    with pytest.raises(dream.DreamError):
        roles._plain("First line\n#+setupfile: /private")
