"""Accepted appends revise established topics with exact bounded delivery."""
import copy
import json
from pathlib import Path

import pytest

from sucoder import dream, dream_prefix, dream_run, dream_transcript, dream_wiki
from tests.test_dream import git, invoke, workspace  # noqa: F401
from tests.test_dream_context import native_result, request
from tests.test_dream_curate import NoopTransport


TOPIC = ".sucoder/wiki/topics/deployment.org"


def encoded(*rows):
    return b"".join(dream.canonical(row) + b"\n" for row in rows)


def evidence(kind):
    first = "Documented deployment requires --dry-run; execution has not been observed. PREFIX_RAW_CANARY"
    last = "Revised documentation requires --dry-run --scope repo; deployment success has not been observed."
    if kind == "codex-jsonl":
        prefix = encoded(dict(type="session_meta", payload=dict(id="append-thread")),
            dict(type="response_item", payload=dict(type="message", role="user", content=[dict(type="input_text", text=first)])),
            dict(type="response_item", payload=dict(type="function_call", call_id="boundary", name="exec",
                arguments='{"command":"RAW_CALL_ARGUMENT_CANARY --dry-run"}')))
        tail = encoded(dict(type="response_item", payload=dict(type="function_call_output", call_id="boundary", output="exit 1; missing scope")),
            dict(type="response_item", payload=dict(type="message", role="user", content=[dict(type="input_text", text=last)])),
            dict(type="future_native", payload=dict(unknown="UNKNOWN_APPEND_CANARY")))
    else:
        prefix = encoded(dict(type="user", sessionId="append-thread", uuid="u1", message=dict(content=first)),
            dict(type="assistant", sessionId="append-thread", uuid="a1", message=dict(content=[dict(type="tool_use", id="boundary", name="Bash",
                input=dict(command="RAW_CALL_ARGUMENT_CANARY --dry-run"))])))
        tail = encoded(dict(type="user", sessionId="append-thread", uuid="r1", parentUuid="a1", message=dict(content=[dict(type="tool_result",
                tool_use_id="boundary", is_error=True, content="exit 1; missing scope")])),
            dict(type="user", sessionId="append-thread", uuid="u2", message=dict(content=last)),
            dict(type="future_native", sessionId="append-thread", unknown="UNKNOWN_APPEND_CANARY"))
    return prefix, tail


def plan(workspace, path, kind, alias="conversation", *, maintenance=False):
    catalog = dict(workspace[3], units=[dict(unit_id=alias, kind=kind, shareable=True,
        locator=dict(path=str(path), thread_id="append-thread"))])
    args = ("--maintenance", "Reconcile source changes", "--topic", "deployment") if maintenance else ()
    result = invoke(workspace, catalog, *args)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


class AppendEditor:
    version = "append-editor-fixture-1"

    def __init__(self, *, lookback=False, forbidden=False):
        self.calls, self.lookback, self.forbidden = [], lookback, forbidden

    def __call__(self, harness, model, task, contract, deadline):
        if self.forbidden:
            raise AssertionError("Accepted unchanged work must not invoke a provider")
        self.calls.append(copy.deepcopy(task))
        result = {key: task[key] for key in ("assignment_id", "input_sha256", "role")}
        result.update(failures=[], requests=[])
        inputs, role = task["input"], task["role"]
        if role == "reviewer":
            result.update(verdict="accept", findings=[])
        else:
            source = inputs["sources"][0]
            if self.lookback and source.get("boundary_pending_calls") and not inputs["context"]["requests"]:
                boundary = source["boundary_pending_calls"][0]
                result.update(decision="lookup", rationale="Inspect the original pending call before interpreting its later error result.",
                    pages=[], changes=[], references=[], omissions=[], requests=[request("source", source["unit_id"],
                        start=boundary["byte_start"], end=boundary["byte_end"])])
            else:
                scope = " --scope repo" if "--scope repo" in source["text"] else ""
                content = ("#+title: Deployment\n* Required command\n:PROPERTIES:\n:CUSTOM_ID: required-command\n:END:\n"
                    "The documented command requires =--dry-run" + scope + "=.\n"
                    "These are documented requirements; a successful deployment has not been observed.\n")
                before = next((item["sha256"] for item in inputs["prior"] if item["path"] == TOPIC), "")
                pages = [dict(path=TOPIC, content=content)]
                changes = [dict(path=TOPIC, before_sha256=before, reason="Retain the original qualification and incorporate the revised documented scope.")]
                if not any(item["path"] == dream_wiki.INDEX for item in inputs["prior"]):
                    pages.append(dict(path=dream_wiki.INDEX, content="#+title: Companion\n[[file:topics/deployment.org::#required-command][Deployment requirements]]\n"))
                    changes.append(dict(path=dream_wiki.INDEX, before_sha256="", reason="Orient readers to the maintained topic."))
                result.update(decision="change", rationale="Update the same semantic topic without upgrading tool reports to successful execution.",
                    pages=pages, changes=changes, omissions=[], references=[dict(path=TOPIC, target="required-command", evidence_status="inference",
                        citations=[dict(unit_id=source["unit_id"], quote=source["text"])])])
        return native_result(harness, model, task, result, self.version)


def accepted(workspace, kind="codex-jsonl", harness="codex"):
    prefix, tail = evidence(kind)
    path = workspace[1].parent / "opt-in-append.jsonl"
    path.write_bytes(prefix)
    first = plan(workspace, path, kind)
    initial = dream_wiki.run(workspace[2], "sample", Path(first["plan_path"]), harness, "model", transport=AppendEditor())
    git(workspace[0], "merge", "--ff-only", initial["candidate_commit"])
    return path, prefix, tail, initial


@pytest.mark.parametrize("harness", ["codex", "claude"])
@pytest.mark.parametrize("kind", dream_transcript.KINDS)
def test_appended_evidence_updates_established_topic_and_accepted_rerun_needs_no_inference(workspace, kind, harness):
    path, prefix, tail, initial = accepted(workspace, kind, harness)
    path.write_bytes(prefix + tail)
    current = plan(workspace, path, kind, "renamed-conversation")
    transport = AppendEditor()
    repo = workspace[0]
    before = (git(repo, "status", "--porcelain"), (repo / ".git/index").read_bytes(), git(repo, "symbolic-ref", "HEAD"))
    result = dream_wiki.run(workspace[2], "sample", Path(current["plan_path"]), harness, "model", transport=transport)
    assert result["state"] == "candidate_ready"
    assert before == (git(repo, "status", "--porcelain"), (repo / ".git/index").read_bytes(), git(repo, "symbolic-ref", "HEAD"))
    assert len(transport.calls) == 2
    for task in transport.calls:
        source = task["input"]["sources"][0]
        assert source["text"].encode() == tail
        assert source["byte_start"] == len(prefix) and source["byte_end"] == len(prefix + tail)
        assert source["boundary_pending_calls"][0]["call_id"] == "boundary"
        assert source["current_pending_calls"] == [] and source["current_orphan_results"] == []
        assert source["native_records"][-1]["event_kinds"] == ["unknown"]
        assert "PREFIX_RAW_CANARY" not in json.dumps(task)
        assert "RAW_CALL_ARGUMENT_CANARY" not in json.dumps(task)
    content = git(repo, "show", result["candidate_commit"] + ":" + TOPIC)
    assert "--dry-run --scope repo" in content and "has not been observed" in content
    receipt = json.loads(git(repo, "show", result["candidate_commit"] + ":.sucoder/wiki/receipts/" + result["run_id"] + ".json"))
    assert receipt["source_delivery"] == dict(initial_bytes=len(tail), accepted_prefix_bytes=len(prefix), requested_bytes=0)
    assert receipt["reused_prefixes"][0]["receipt"]["run_id"] == initial["run_id"]
    assert receipt["units"][0]["normalization"] == current["units"][0]["normalization"]
    assert {item["path"] for item in receipt["outputs"]} >= {TOPIC, dream_wiki.INDEX, dream_wiki.SETTINGS_PATH}
    assert [item["extraction"] for item in receipt["coverage"]] == ["accepted_receipt_reused", "author_inline_delivered"]
    assert receipt["coverage"][0]["byte_end"] == receipt["coverage"][1]["byte_start"] == len(prefix)
    assert dream_run.status(workspace[2], "sample", Path(current["plan_path"]))["state"] == "candidate_ready"
    assert dream_wiki.run(workspace[2], "sample", Path(current["plan_path"]), harness, "model", transport=transport) == result
    assert len(transport.calls) == 2
    git(repo, "merge", "--ff-only", result["candidate_commit"])
    fresh = plan(workspace, path, kind, "another-local-alias")
    reused = dream_wiki.run(workspace[2], "sample", Path(fresh["plan_path"]), harness, "model", transport=AppendEditor(forbidden=True))
    assert reused["state"] == "already_integrated" and reused["usage"]["role_invocations"] == 0
    assert reused["receipts"][0]["run_id"] == result["run_id"]
    assert dream_run.status(workspace[2], "sample", Path(fresh["plan_path"]))["state"] == "already_integrated"
    assert not list(Path(fresh["plan_path"]).parent.glob("*-assignment.json"))
    assert list((repo / ".sucoder/wiki/topics").glob("*.org")) == [repo / TOPIC]


@pytest.mark.parametrize("kind", dream_transcript.KINDS)
def test_pending_call_can_request_exact_older_context_with_separate_accounting(workspace, kind):
    path, prefix, tail, _ = accepted(workspace, kind)
    path.write_bytes(prefix + tail)
    current = plan(workspace, path, kind)
    transport = AppendEditor(lookback=True)
    result = dream_wiki.run(workspace[2], "sample", Path(current["plan_path"]), "codex", "model", transport=transport)
    assert [task["role"] for task in transport.calls] == ["author", "author", "reviewer"]
    assert "RAW_CALL_ARGUMENT_CANARY" not in json.dumps(transport.calls[0])
    assert "RAW_CALL_ARGUMENT_CANARY" in json.dumps(transport.calls[1])
    receipt = json.loads(git(workspace[0], "show", result["candidate_commit"] + ":.sucoder/wiki/receipts/" + result["run_id"] + ".json"))
    assert 0 < receipt["source_delivery"]["requested_bytes"] < len(prefix)
    assert receipt["source_delivery"]["initial_bytes"] == len(tail)
    assert dream_run.status(workspace[2], "sample", Path(current["plan_path"]))["state"] == "candidate_ready"


@pytest.mark.parametrize("defect", ["prefix", "shrink", "outputs", "normalization"])
def test_changed_prefix_or_required_output_requires_explicit_reconciliation(workspace, defect):
    path, prefix, tail, initial = accepted(workspace)
    if defect == "prefix":
        path.write_bytes(prefix.replace(b"--dry-run", b"--unsafe!") + tail)
    elif defect == "shrink":
        path.write_bytes(encoded(dict(type="session_meta", payload=dict(id="append-thread"))))
    else:
        path.write_bytes(prefix + tail)
        if defect == "outputs":
            target = workspace[0] / TOPIC
            target.write_text(target.read_text() + "A new independent human contribution.\n")
        else:
            target = workspace[0] / (".sucoder/wiki/receipts/" + initial["run_id"] + ".json")
            receipt = json.loads(target.read_bytes())
            receipt["units"][0]["normalization"]["contract"] = "unknown-normalization-v99"
            target.write_bytes(dream.canonical(receipt) + b"\n")
        git(workspace[0], "add", str(target.relative_to(workspace[0])))
        git(workspace[0], "commit", "-qm", "Change required accepted state")
    current = plan(workspace, path, "codex-jsonl")
    with pytest.raises(dream.DreamError, match="need reconciliation"):
        dream_wiki.run(workspace[2], "sample", Path(current["plan_path"]), "codex", "model", transport=AppendEditor(forbidden=True))


def test_prefix_proof_tampering_fails_status_without_fetching_context(workspace):
    path, prefix, tail, _ = accepted(workspace)
    path.write_bytes(prefix + tail)
    current = plan(workspace, path, "codex-jsonl")
    result = dream_wiki.run(workspace[2], "sample", Path(current["plan_path"]), "codex", "model", transport=AppendEditor())
    parent = next(Path(current["plan_path"]).parent.glob("prefix-parent-*.json"))
    parent.write_bytes(b"{}\n")
    with pytest.raises(dream.DreamError, match="frozen Git object"):
        dream_run.status(workspace[2], "sample", Path(current["plan_path"]), result["run_id"])


@pytest.mark.parametrize("kind", dream_transcript.KINDS)
def test_repeated_boundary_call_id_preserves_ambiguity_after_append(workspace, kind):
    prefix, tail = evidence(kind)
    if kind == "codex-jsonl":
        repeat = dict(type="response_item", payload=dict(type="function_call", call_id="boundary", name="exec", arguments="second pending call"))
    else:
        repeat = dict(type="assistant", sessionId="append-thread", message=dict(content=[dict(type="tool_use", id="boundary", name="Bash", input={})]))
    prefix += encoded(repeat)
    path = workspace[1].parent / "ambiguous-append.jsonl"
    path.write_bytes(prefix)
    first = plan(workspace, path, kind)
    initial = dream_wiki.run(workspace[2], "sample", Path(first["plan_path"]), "codex", "model", transport=AppendEditor())
    git(workspace[0], "merge", "--ff-only", initial["candidate_commit"])
    path.write_bytes(prefix + tail)
    current = plan(workspace, path, kind)
    transport = AppendEditor()
    dream_wiki.run(workspace[2], "sample", Path(current["plan_path"]), "codex", "model", transport=transport)
    source = transport.calls[0]["input"]["sources"][0]
    assert source["prefix_ambiguous_call_ids"] == source["current_ambiguous_call_ids"] == ["boundary"]
    assert len(source["boundary_pending_calls"]) == len(source["current_pending_calls"]) == 2


def test_interrupted_append_publication_replays_proof_and_roles_without_inference(workspace, monkeypatch):
    path, prefix, tail, _ = accepted(workspace)
    path.write_bytes(prefix + tail)
    current = plan(workspace, path, "codex-jsonl")
    store = dream_run._store
    def interrupted(fd, name, data):
        if name.endswith("-run.json"):
            raise KeyboardInterrupt("after atomic append candidate publication")
        return store(fd, name, data)
    with monkeypatch.context() as patch:
        patch.setattr(dream_run, "_store", interrupted)
        with pytest.raises(KeyboardInterrupt):
            dream_wiki.run(workspace[2], "sample", Path(current["plan_path"]), "codex", "model", transport=AppendEditor())
    assert dream_run.status(workspace[2], "sample", Path(current["plan_path"]))["state"] == "candidate_ready"
    replay = dream_wiki.run(workspace[2], "sample", Path(current["plan_path"]), "codex", "model", transport=AppendEditor(forbidden=True))
    assert replay["state"] == "candidate_ready"


def test_targeted_repair_uses_existing_transcript_without_any_new_session(workspace):
    path, prefix, _, _ = accepted(workspace)
    topic = workspace[0] / TOPIC
    topic.write_text(topic.read_text().replace("requires =--dry-run=", "requires no flags"))
    git(workspace[0], "add", TOPIC)
    git(workspace[0], "commit", "-qm", "Seed a wrong existing summary for targeted repair")
    current = plan(workspace, path, "codex-jsonl", maintenance=True)
    transport = AppendEditor()
    result = dream_wiki.run(workspace[2], "sample", Path(current["plan_path"]), "codex", "model", transport=transport)
    source = transport.calls[0]["input"]["sources"][0]
    assert source["text"].encode() == prefix and source["byte_start"] == 0
    assert "accepted_prefix" not in source
    assert "requires =--dry-run=" in git(workspace[0], "show", result["candidate_commit"] + ":" + TOPIC)


def test_private_append_noop_never_establishes_shared_prefix_coverage(workspace):
    path, prefix, tail, initial = accepted(workspace)
    path.write_bytes(prefix + tail)
    first = plan(workspace, path, "codex-jsonl")
    class NoopAppend(NoopTransport):
        version = AppendEditor.version
    transport = NoopAppend()
    result = dream_wiki.run(workspace[2], "sample", Path(first["plan_path"]), "codex", "model", transport=transport)
    assert result["state"] == "unchanged" and result["candidate_commit"] is None
    (workspace[0] / "unrelated.txt").write_text("Change the planning base without changing accepted wiki evidence.\n")
    git(workspace[0], "add", "unrelated.txt")
    git(workspace[0], "commit", "-qm", "Independent source-tree change")
    fresh = plan(workspace, path, "codex-jsonl")
    transport = NoopAppend()
    repeated = dream_wiki.run(workspace[2], "sample", Path(fresh["plan_path"]), "codex", "model", transport=transport)
    assert repeated["state"] == "unchanged" and len(transport.calls) == 2
    assert transport.calls[0]["input"]["sources"][0]["text"].encode() == tail
    proof = transport.calls[0]["input"]["sources"][0]["accepted_prefix"]
    assert proof["receipt"]["run_id"] == initial["run_id"] and proof["byte_end"] == len(prefix)


def test_squash_integrated_append_is_shared_progress_without_original_role_cache(workspace):
    path, prefix, tail, _ = accepted(workspace)
    path.write_bytes(prefix + tail)
    current = plan(workspace, path, "codex-jsonl")
    result = dream_wiki.run(workspace[2], "sample", Path(current["plan_path"]), "codex", "model", transport=AppendEditor())
    tree = git(workspace[0], "rev-parse", result["candidate_commit"] + "^{tree}")
    squash = git(workspace[0], "commit-tree", tree, "-p", current["base_commit"], "-m", "Squash accepted append fixture")
    git(workspace[0], "update-ref", "refs/heads/main", squash, current["base_commit"])
    assert dream_run.status(workspace[2], "sample", Path(current["plan_path"]))["integration"]["state"] == "integrated_candidate"
    fresh = plan(workspace, path, "codex-jsonl", "independent-local-name")
    reused = dream_wiki.run(workspace[2], "sample", Path(fresh["plan_path"]), "codex", "model", transport=AppendEditor(forbidden=True))
    assert reused["state"] == "already_integrated" and reused["usage"]["role_invocations"] == 0
    assert not list(Path(fresh["plan_path"]).parent.glob("*-assignment.json"))


def test_peer_advance_during_append_cannot_publish_against_stale_base(workspace):
    path, prefix, tail, _ = accepted(workspace)
    path.write_bytes(prefix + tail)
    current = plan(workspace, path, "codex-jsonl")
    before = git(workspace[0], "for-each-ref", "refs/heads/dream/")
    class PeerAdvance(AppendEditor):
        def __call__(self, harness, model, task, contract, deadline):
            result = super().__call__(harness, model, task, contract, deadline)
            if task["role"] == "author":
                git(workspace[0], "commit", "--allow-empty", "-qm", "Peer advances the integration branch")
            return result
    with pytest.raises(dream.DreamError, match="ref transaction failed"):
        dream_wiki.run(workspace[2], "sample", Path(current["plan_path"]), "codex", "model", transport=PeerAdvance())
    assert git(workspace[0], "for-each-ref", "refs/heads/dream/") == before
