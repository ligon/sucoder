"""Comparison conditions enforce access and preserve native accounting/replay."""
import copy
import json
from pathlib import Path
import time

import pytest
from typer.testing import CliRunner

from sucoder import cli, dream, dream_reader, dream_reader_eval as evaluation
from tests.test_dream import workspace  # noqa: F401
from tests.test_dream_reader import prepared


class Transport:
    version = "comparison-reader-fixture-1"

    def __init__(self):
        self.calls = []

    def __call__(self, harness, model, task, schema, deadline):
        self.calls.append(copy.deepcopy(task))
        inputs = task["input"]
        kinds = {entry["request"]["kind"] for entry in inputs["retrievals"]}
        result = {key: task[key] for key in ("assignment_id", "input_sha256", "role")}
        result.update(failures=[], requests=[], answer="", evidence=[])
        wanted = ("topic" if inputs["wiki_available"] and "topic" not in kinds else
                  "archive" if inputs["wiki_available"] and "archive" not in kinds else
                  "search" if inputs["sources_available"] and "search" not in kinds else None)
        if wanted:
            result.update(action="lookup", requests=[dict(kind=wanted, key="note" if wanted == "search" else "replies",
                target="reply-proof" if wanted == "topic" else "decision" if wanted == "archive" else "",
                query="qualification" if wanted == "search" else "", start=0, end=0)])
        else:
            result.update(action="answer", answer="Preserve the qualification; unavailable original evidence was not inspected.",
                evidence=[entry["response"]["read_id"] for entry in inputs["retrievals"] if "read_id" in entry["response"]])
        if harness == "codex":
            events = [dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(result))), dict(type="turn.completed", usage={})]
        else:
            events = [dict(type="system", subtype="init", tools=["StructuredOutput"], mcp_servers=[]),
                dict(type="result", subtype="success", is_error=False, structured_output=result, usage={}, total_cost_usd=None)]
        trace = dict(harness=harness, model=model, version=self.version, returncode=0,
            input_sha256=dream.digest(dream.canonical(task)), stdout="\n".join(json.dumps(event) for event in events) + "\n",
            stderr="", boundary="linux-landlock-inline-v1")
        return result, dict(native={}, currency_usd=None), trace


@pytest.mark.parametrize("harness", ["codex", "claude"])
@pytest.mark.parametrize("condition", evaluation.CONDITIONS)
def test_conditions_account_for_delivery_and_replay(workspace, harness, condition):
    plan = prepared(workspace)
    path = Path(plan["plan_path"])
    transport = Transport()
    result = evaluation.ask(workspace[2], "sample", path, "What is qualified?", harness, "model", condition=condition, transport=transport)
    wiki, sources = evaluation.CONDITIONS[condition]
    assert result["contract"] == evaluation.CONTRACT and result["condition"] == condition
    assert result["topic_delivered"] is wiki and result["sources_available"] is sources
    assert all(task["brief"] == evaluation.BRIEF for task in transport.calls)
    assert bool(result["accounting"]["delivery_bytes"]["archive"]) is wiki
    assert bool(result["delivered_bytes"]["source"]) is sources
    assert result["accounting"]["retrieval_bytes"] == sum(entry["response"]["delivered_bytes"]
        for task in transport.calls for entry in task["input"]["retrievals"])
    assert result["accounting"]["assignment_bytes"] == sum(len(dream.canonical(task)) for task in transport.calls)
    assert result["accounting"]["common_brief_bytes"] == len(evaluation.BRIEF.encode()) * len(transport.calls)
    if not sources:
        assert all(task["input"]["admitted_sources"] == [] and task["input"]["allowances"]["source"] == 0 for task in transport.calls)
    count = len(transport.calls)
    assert evaluation.ask(workspace[2], "sample", path, "What is qualified?", harness, "model", condition=condition, transport=transport) == result
    assert len(transport.calls) == count


def test_wiki_only_denies_sources_before_retrieval(workspace, monkeypatch):
    plan = prepared(workspace)
    def forbidden(*args, **kwargs):
        raise AssertionError("Wiki-only called the source interface")
    monkeypatch.setattr(dream_reader, "source", forbidden)
    view = dream_reader.View(workspace[2], "sample", deadline=time.monotonic() + 15)
    session = evaluation.Session(workspace[2], "sample", Path(plan["plan_path"]), view, condition="wiki-only")
    for kind in ("source", "search"):
        result = session.lookup(dict(kind=kind, key="note", query="qualification", target="", start=0, end=0))
        assert result["state"] == "unavailable" and result["delivered_bytes"] == 0
    assert session.used["source"] == 0


def test_comparison_rejects_unenforceable_currency_and_bad_options_before_transport(workspace):
    for options in (dict(max_currency_usd=1), dict(condition="unknown"), dict(max_elapsed_seconds=121), dict(attempt=0)):
        with pytest.raises(dream.DreamError):
            evaluation.ask(workspace[2], "sample", Path("absent"), "Question", "codex", "model", **options)


def test_cli_wiki_only_selects_new_contract_and_rejects_conflicting_modes(workspace, monkeypatch):
    calls = []
    def answer(*args):
        calls.append(args)
        return dict(state="answered", condition=args[-2])
    monkeypatch.setattr(evaluation, "ask", answer)
    arguments = ["--config", str(workspace[1]), "dream", "ask", "sample", "Question", "--plan", "absent", "--harness", "codex", "--model", "model", "--wiki-only"]
    runner = CliRunner()
    result = runner.invoke(cli.app, arguments)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["condition"] == "wiki-only"
    assert len(calls) == 1
    result = runner.invoke(cli.app, [*arguments, "--source-only"])
    assert result.exit_code == 2 and len(calls) == 1
