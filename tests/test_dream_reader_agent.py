"""Fresh native readers select retrievals without construction context."""
import copy
import json
from pathlib import Path
import time

import pytest

from sucoder import dream, dream_reader, dream_reader_agent as agent
from tests.test_dream import workspace  # noqa: F401
from tests.test_dream_reader import prepared


class ReaderTransport:
    version = "fresh-reader-fixture-1"

    def __init__(self):
        self.calls = []

    def __call__(self, harness, model, task, contract, deadline):
        self.calls.append(copy.deepcopy(task))
        inputs = task["input"]
        kinds = {entry["request"]["kind"] for entry in inputs["retrievals"]}
        result = {key: task[key] for key in ("assignment_id", "input_sha256", "role")}
        result.update(failures=[], requests=[], answer="", evidence=[])
        if inputs["wiki_available"] and "topic" not in kinds:
            result.update(action="lookup", requests=[dict(kind="topic", key="replies", target="reply-proof", query="", start=0, end=0)])
        elif "search" not in kinds:
            result.update(action="lookup", requests=[dict(kind="search", key="note", target="", query="qualification", start=0, end=0)])
        else:
            result.update(action="answer", answer="A summary is not proof of a native reply; the topic retains the unresolved maintainer caveat.",
                evidence=[e["response"]["read_id"] for e in inputs["retrievals"] if "read_id" in e["response"]])
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
def test_fresh_reader_retrieval_trace_and_inference_free_resume(workspace, harness):
    plan = prepared(workspace)
    path = Path(plan["plan_path"])
    (path.parent / "construction-conversation.json").write_text("BUILDER_ONLY_CANARY AND PRIVATE GRADING RUBRIC")
    transport = ReaderTransport()
    result = agent.ask(workspace[2], "sample", path, "Does a summary prove a reply?", harness, "explicit-model", transport=transport)
    assert result["topic_delivered"] is True
    assert result["delivered_bytes"]["orientation"] <= 2048
    assert result["delivered_bytes"]["topic"] <= 12288
    assert result["delivered_bytes"]["source"] <= 32768
    assert [e["request"]["kind"] for e in result["retrievals"]] == ["orientation", "topic", "search"]
    assert result["evidence"][1] == result["retrievals"][1]["response"]["read_id"]
    assert len(transport.calls) == 3
    assert "BUILDER_ONLY_CANARY" not in json.dumps(transport.calls)
    assert "HISTORICAL_CANARY" not in json.dumps(transport.calls)
    assert "unresolved" in json.dumps(transport.calls)
    resumed = agent.ask(workspace[2], "sample", path, "Does a summary prove a reply?", harness, "explicit-model", transport=transport)
    assert resumed == result
    assert len(transport.calls) == 3


def test_source_only_reader_uses_same_source_interface_without_wiki(workspace):
    plan = prepared(workspace)
    transport = ReaderTransport()
    result = agent.ask(workspace[2], "sample", Path(plan["plan_path"]), "Read original qualification.", "codex", "model",
                       with_wiki=False, transport=transport)
    assert result["topic_delivered"] is False
    assert result["delivered_bytes"]["orientation"] == result["delivered_bytes"]["topic"] == 0
    assert len(transport.calls) == 2
    assert [e["request"]["kind"] for e in result["retrievals"]] == ["search"]
    direct = dream_reader.source(workspace[2], "sample", Path(plan["plan_path"]), "note", "qualification")
    assert result["retrievals"][0]["response"] == direct


def test_reader_charges_repeated_delivery_and_stops_at_cumulative_limit(workspace, monkeypatch):
    plan = prepared(workspace)
    monkeypatch.setattr(dream_reader, "SOURCE_BYTES", 5)
    view = dream_reader.View(workspace[2], "sample", deadline=time.monotonic() + 10)
    session = agent.ReaderSession(workspace[2], "sample", Path(plan["plan_path"]), view, with_wiki=False)
    request = dict(kind="source", key="note", query="", target="", start=0, end=0)
    first = session.lookup(request)
    assert first["delivered_bytes"] == 5 and first["state"] == "incomplete"
    second = session.lookup(request)
    assert second["state"] == "unavailable" and second["delivered_bytes"] == 0
    assert session.used["source"] == 5
    assert "Cumulative" in second["error"]
