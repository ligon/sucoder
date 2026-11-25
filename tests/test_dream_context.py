"""Agents retrieve frozen omitted context and preserve unseen contributions."""
import copy
import json
from pathlib import Path
import time

import pytest

from sucoder import dream, dream_context, dream_curate, dream_native, dream_run, dream_wiki
from tests.test_dream import git, invoke, workspace  # noqa: F401
from tests.test_dream_curate import NoopTransport, noop_plan
from tests.test_dream_wiki import OLD, POLICY, TOPIC, planned


def request(kind, key="", target="", query="", start=0, end=0):
    return dict(kind=kind, key=key, target=target, query=query, start=start, end=end)


def native_result(harness, model, task, result, version):
    if task["contract"] == "source-grounded-wiki-v5":
        result.setdefault("task_checks" if task["role"] == "reviewer" else "task_actions", [])
    if harness == "codex":
        events = [dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(result))),
                  dict(type="turn.completed", usage={})]
    else:
        events = [dict(type="system", subtype="init", tools=["StructuredOutput"], mcp_servers=[]),
                  dict(type="result", subtype="success", is_error=False, structured_output=result,
                       usage={}, total_cost_usd=None)]
    trace = dict(harness=harness, model=model, version=version, returncode=0,
        input_sha256=dream.digest(dream.canonical(task)), stdout="\n".join(json.dumps(e) for e in events) + "\n",
        stderr="", boundary="linux-landlock-inline-v1")
    return result, dict(native={}, currency_usd=None), trace


class ContextEditor:
    version = "context-editor-fixture-1"

    def __init__(self, *, loop=False, edit_partial=False):
        self.calls, self.loop, self.edit_partial = [], loop, edit_partial

    def __call__(self, harness, model, task, contract, deadline):
        self.calls.append(copy.deepcopy(task))
        result = {key: task[key] for key in ("assignment_id", "input_sha256", "role")}
        result.update(failures=[], requests=[])
        inputs = task["input"]
        reads = inputs["context"]["requests"]
        if task["role"] == "reviewer":
            if not any(item["request"]["kind"] == "archive" for item in reads):
                result.update(verdict="lookup", findings=[], requests=[request("archive", "reply-evidence", "retired-rule")])
            else:
                result.update(verdict="accept", findings=[])
        else:
            result.update(decision="lookup", rationale="Inspect relevant frozen knowledge before correcting it.",
                pages=[], changes=[], references=[], omissions=[])
            if not reads or self.loop:
                result["requests"] = [request("topic-search", query="compaction"), request("search", "evidence-a", query="summary")]
            elif not any(item["response"].get("editable") for item in reads) and not self.edit_partial:
                result["requests"] = [request("topic", "reply-evidence")]
            else:
                prior = next((item["response"] for item in reads if item["response"].get("editable")), None)
                content = OLD.replace("Any reply tag in a compaction summary proves that a reply was sent.",
                    "A compaction summary does not prove delivery. Native assistant text must follow the matching delivery and precede the next input.\n"
                    "Use =--session-id <uuid>= at launch; this is documented, not a freshly tested launch.")
                index = next(page["content"] for page in inputs["prior"] if page["path"] == dream_wiki.INDEX)
                index += "\n[[file:topics/reply-evidence.org::#reply-proof][Reply proof]]\n"
                result.update(decision="change", rationale="Correct the false rule and preserve the maintainer constraint.",
                    pages=[dict(path=TOPIC, content=content), dict(path=dream_wiki.INDEX, content=index)],
                    changes=[dict(path=TOPIC, before_sha256=prior["page_sha256"] if prior else dream.digest(OLD.encode()), reason="Correct false memory"),
                        dict(path=dream_wiki.INDEX, before_sha256=dream.digest(next(page["content"] for page in inputs["prior"] if page["path"] == dream_wiki.INDEX).encode()), reason="Add topic")],
                    references=[dict(path=TOPIC, target="reply-proof", evidence_status="inference",
                        citations=[dict(unit_id=unit["unit_id"], quote=unit["text"]) for unit in inputs["sources"]])])
        return native_result(harness, model, task, result, self.version)


def context_plan(workspace):
    planned(workspace)
    root = workspace[0] / ".sucoder/wiki"
    (root / "topics/unrelated.org").write_text("* Other\n:PROPERTIES:\n:CUSTOM_ID: other\n:END:\n" + "UNRELATED_CANARY\n" * 5000)
    (root / "index.org").write_text("#+title: Companion\n[[file:topics/unrelated.org::#other][Unrelated]]\n")
    (root / "settings.org").write_text(dream_wiki.SETTINGS)
    archive = workspace[0] / ".sucoder/wiki/archive/reply-evidence.org"
    archive.parent.mkdir()
    archive.write_text("* DONE Retired rule\n:PROPERTIES:\n:CUSTOM_ID: retired-rule\n:END:\n:LOGBOOK:\nMaintainer rejected summary proof.\n:END:\nHistorical decision, not current delivery evidence.\n")
    git(workspace[0], "add", ".sucoder/wiki")
    git(workspace[0], "commit", "-qm", "Grow wiki beyond prior allowance with an archived decision")
    envelope = invoke(workspace, dict(workspace[3], units=[dict(unit_id=name, kind="git", shareable=True,
        locator=dict(repository="sample", commit="HEAD", path=name + ".org")) for name in ("evidence-a", "evidence-b")]))
    assert envelope.exit_code == 0, envelope.output
    return json.loads(envelope.output)


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_native_editor_finds_omitted_topic_requests_full_page_and_reviewer_history(workspace, harness):
    plan = context_plan(workspace)
    assert plan["prior_wiki"]["selection"]["strategy"] == "orientation"
    assert TOPIC not in {page["path"] for page in plan["prior_wiki"]["selected"]}
    transport = ContextEditor()
    result = dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), harness, "model", transport=transport)
    assert result["state"] == "candidate_ready"
    assert [call["role"] for call in transport.calls] == ["author", "author", "author", "reviewer", "reviewer"]
    assert "UNRELATED_CANARY" not in json.dumps(transport.calls)
    content = git(workspace[0], "show", result["candidate_commit"] + ":" + TOPIC)
    assert "does not prove delivery" in content and POLICY.strip() in content
    receipt = json.loads(git(workspace[0], "show", result["candidate_commit"] + ":.sucoder/wiki/receipts/" + result["run_id"] + ".json"))
    assert receipt["context"]["used"]["prior"] < plan["limits"]["max_prior_bytes"]
    assert receipt["context"]["used"]["source"] > 0
    assert receipt["context"]["requests"][-1]["response"]["editable"] is False
    assert "Maintainer rejected" in receipt["context"]["requests"][-1]["response"]["text"]
    assert dream_run.status(workspace[2], "sample", Path(plan["plan_path"]))["state"] == "candidate_ready"
    assert dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), harness, "model", transport=transport) == result
    assert len(transport.calls) == 5


def test_search_snippet_does_not_authorize_editing_unseen_full_page(workspace):
    plan = context_plan(workspace)
    with pytest.raises(dream.DreamError, match="outside the delivered prior context"):
        dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "model", transport=ContextEditor(edit_partial=True))
    assert not git(workspace[0], "for-each-ref", "refs/heads/dream/")


def test_repeated_lookups_stop_at_shared_request_allowance(workspace):
    plan = context_plan(workspace)
    with pytest.raises(dream.DreamError, match="request-count allowance"):
        dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "model", transport=ContextEditor(loop=True))
    assert not git(workspace[0], "for-each-ref", "refs/heads/dream/")


def test_context_cache_corruption_fails_status_without_fetch_or_inference(workspace, monkeypatch):
    plan = context_plan(workspace)
    result = dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "model", transport=ContextEditor())
    directory = Path(plan["plan_path"]).parent
    cache = directory / ("context-blob-" + git(workspace[0], "rev-parse", "HEAD:" + TOPIC).strip() + ".org")
    cache.write_bytes(b"Changed cached human decision\n")
    def forbidden(*args, **kwargs):
        raise AssertionError("Status fetched context or invoked a provider")
    monkeypatch.setattr(dream_native.Transport, "__call__", forbidden)
    with pytest.raises(dream.DreamError, match="Cached context differs"):
        dream_run.status(workspace[2], "sample", directory / "plan.json", result["run_id"])


def test_original_v3_noop_replays_original_schema_and_identity(workspace):
    envelope = noop_plan(workspace)
    deadline = time.monotonic() + 30
    with dream_run._directory(workspace[2], Path(envelope["plan_path"])) as (fd, directory):
        plan = dream_run._load(fd, directory.name, "sample")
        _, sources = dream_run._snapshot(workspace[2], "sample", plan, fd, deadline)
        coordinator = dream_curate.Curator(plan, fd, "codex", "model", deadline, NoopTransport())
        prefix = "native-" + coordinator.run_id
        dream_run._store(fd, prefix + "-spec.json", dream.canonical(dict(contract=dream_curate.CONTRACT,
            run_id=coordinator.run_id, harness="codex", model="model", native_version=coordinator.expected_version,
            attempt=1, purpose=dream_wiki.PURPOSE)) + b"\n")
        _, receipt = coordinator.prepare()
        result = dream_curate.record_unchanged(plan, fd, sources, receipt, deadline)
    assert receipt["contract"] == dream_curate.CONTRACT
    assert "requests" not in receipt["review"]
    assert dream_run.status(workspace[2], "sample", directory / "plan.json")["run_id"] == result["run_id"]


def test_changed_unavailable_context_does_not_create_new_paid_resume_assignment(workspace):
    plan = noop_plan(workspace)
    class UnavailableNoop:
        version = "unavailable-context-fixture-1"
        calls = 0
        def __call__(self, harness, model, task, contract, deadline):
            self.calls += 1
            result = {key: task[key] for key in ("assignment_id", "input_sha256", "role")}
            result.update(failures=[], requests=[])
            if task["role"] == "reviewer":
                result.update(verdict="accept", findings=[])
            else:
                lookup = not task["input"]["context"]["requests"]
                result.update(decision="lookup" if lookup else "unchanged", rationale="No supported useful change.",
                    pages=[], changes=[], references=[], omissions=[],
                    requests=[request("topic", "not-present")] if lookup else [])
            return native_result(harness, model, task, result, self.version)
    transport = UnavailableNoop()
    path = Path(plan["plan_path"])
    dream_wiki.run(workspace[2], "sample", path, "codex", "model", transport=transport)
    cached = next(p for p in path.parent.glob("context-*.json") if "step-" not in p.name)
    response = json.loads(cached.read_bytes())
    assert response["state"] == "unavailable"
    response["error"] = "A different unavailable result would change the next assignment."
    response["read_id"] = dream.digest(dream.canonical([request("topic", "not-present"),
        {key: value for key, value in response.items() if key != "read_id"}]))
    cached.write_bytes(dream.canonical(response) + b"\n")
    before = transport.calls
    with pytest.raises(dream.DreamError, match="Existing dream artifact differs"):
        dream_wiki.run(workspace[2], "sample", path, "codex", "model", transport=transport)
    assert transport.calls == before


def test_source_lookup_uses_exact_frozen_bytes_and_replays_unavailable_utf8_range(workspace):
    envelope = json.loads(invoke(workspace).output)
    deadline = time.monotonic() + 30
    with dream_run._directory(workspace[2], Path(envelope["plan_path"])) as (fd, directory):
        plan = dream_run._load(fd, directory.name, "sample")
        _, readers = dream_run._snapshot(workspace[2], "sample", plan, fd, deadline)
        sources = dream_wiki.source_bundle(plan, fd)
        unit = sources[0]
        session = dream_context.Session(plan, fd, "test-source", sources, {}, deadline, readers["sample"])
        full = request("source", unit["unit_id"])
        delivered = session.lookup(full)
        assert delivered["excerpts"][0]["text"] == "First qualification.\r\nSecond: caf\u00e9.\n"
        split = unit["byte_start"] + unit["text"].encode().index(b"\xc3") + 1
        invalid = request("source", unit["unit_id"], start=split, end=unit["byte_end"])
        unavailable = session.lookup(invalid)
        assert unavailable["state"] == "unavailable" and "UTF-8" in unavailable["error"]
        replay = dream_context.Session(plan, fd, "test-source", sources, {}, deadline)
        assert replay.lookup(full) == delivered
        assert replay.lookup(invalid) == unavailable
