"""Handles preserve exact delivery, authority, replay and complete diagnostics."""
import copy
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from sucoder import dream, dream_context, dream_diagnostics as diag, dream_editor, dream_handles as handles, dream_progress, dream_run, dream_transcript, dream_wiki as wiki
from tests.test_dream import git, workspace  # noqa: F401
from tests.test_dream_context import request
from tests.test_dream_wiki import EditorTransport, POLICY, TOPIC, planned as current_plan


def planned(workspace):
    return current_plan(workspace, authoring_version=7)


def unit(text="same\n\nsame\n\n", origin=17):
    return dict(unit_id="source", kind="git", source_id="frozen-source", sha256=dream.digest(text.encode()),
        byte_start=origin, byte_end=origin + len(text.encode()), text=text)


def registry(source=None, run_id="frozen-run"):
    return handles.Registry(run_id, [source or unit()], time.monotonic() + 60)


def test_identical_passages_select_exact_positions_without_changing_v6_quote_policy():
    source = unit()
    index = registry(source)
    passages = index.deliver("source", source["byte_start"], source["byte_end"])
    assert len(passages) == 2 and passages[0]["text"] == passages[1]["text"]
    assert passages[0]["handle"] != passages[1]["handle"]
    chosen = index.resolve(dict(handle=passages[1]["handle"]))
    assert (chosen["byte_start"], chosen["byte_end"], chosen["quote"]) == (23, 29, "same\n\n")
    from tests.test_dream_diagnostics import case
    args = list(case())
    args[4] = [source]
    args[6] = {"source": [(17, 29)]}
    args[0]["references"][0]["citations"] = [dict(handle=passages[1]["handle"])]
    args[2][TOPIC]["targets"].pop("talk")
    assert diag.references(*args, time.monotonic() + 5, citation_resolver=index.resolve)[0]["citations"] == [chosen]
    args[0]["references"][0]["citations"] = [dict(unit_id="source", quote="same\n\n")]
    with pytest.raises(diag.InvalidProposal, match="two positions"):
        diag.references(*args, time.monotonic() + 5)


def test_utf8_long_paragraph_and_crlf_are_lossless_and_bounded():
    source = unit("caf\u00e9\r\n\r\n" + "\u03bb" * 3000 + "\n")
    index = registry(source)
    passages = index.deliver("source", source["byte_start"], source["byte_end"])
    assert "".join(p["text"] for p in passages) == source["text"]
    assert all(len(p["text"].encode()) <= handles.SPAN_BYTES for p in passages)
    assert all(index.resolve(dict(handle=p["handle"]))["quote"] == p["text"] for p in passages)
    before = index.snapshot()
    index.deliver("source", source["byte_start"], source["byte_end"])
    assert index.snapshot() == before


def test_namespace_binds_run_source_identity_and_provenance():
    source = unit()
    one = registry(source)
    issued = one.deliver("source", 17, 23)[0]["handle"]
    for changed, run in [(dict(source, source_id="other"), "frozen-run"), (source, "other-run"),
                         (dict(source, provenance=dict(raw_sha256="other")), "frozen-run")]:
        other = registry(changed, run)
        other.deliver("source", 17, 23)
        with pytest.raises(dream.DreamError, match="Unknown, stale or undelivered"):
            other.resolve(dict(handle=issued))


@pytest.mark.parametrize("bound", ["MAX_ENTRIES", "REGISTRY_BYTES"])
def test_registry_limits_fail_without_silent_coarsening(monkeypatch, bound):
    monkeypatch.setattr(handles, bound, 1)
    with pytest.raises(dream.DreamError, match="allowance exhausted") as caught:
        registry().deliver("source", 17, 29)
    assert caught.value.exit_code == 3


def session(fd, source, live):
    plan = dict(prior_wiki=dict(files=[]), limits=dict(max_prior_bytes=8192))
    ctx = handles.Session(plan, fd, "scope", [source], {}, time.monotonic() + 60, object() if live else None)
    ctx.windows["source"] = [(23, 29)]
    ctx.evidence = registry(source)
    ctx.evidence.deliver("source", 23, 29)
    return ctx


def test_prefix_lookback_is_explicit_and_replays_without_new_allowances(tmp_path):
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        source = unit()
        ctx = session(fd, source, True)
        initial = ctx.evidence.snapshot()
        assert {e["byte_start"] for e in initial["entries"]} == {23}
        lookup = request("source", "source", start=17, end=23)
        ctx.lookup(lookup)
        first = copy.deepcopy(ctx.payload())
        assert len(first["evidence"]["entries"]) == 2 and ctx.used["source"] == 6
        assert first["evidence"]["entries"][0] == initial["entries"][0]
        for _ in range(dream_context.MAX_REQUESTS - 1):
            ctx.lookup(lookup)
        with pytest.raises(dream.DreamError, match="request-count allowance"):
            ctx.lookup(lookup)
        replay = session(fd, source, False)
        # A later cached lookup exists on disk but is not yet delivered here.
        with pytest.raises(dream.DreamError, match="undelivered evidence handle"):
            replay.evidence.resolve(dict(handle=first["evidence"]["entries"][1]["handle"]))
        replay.lookup(lookup)
        assert replay.payload() == first
        for _ in range(dream_context.MAX_REQUESTS - 1):
            replay.lookup(lookup)
        assert replay.payload() == ctx.payload()
        with pytest.raises(dream.DreamError, match="request-count allowance"):
            replay.lookup(lookup)
    finally:
        os.close(fd)


@pytest.mark.parametrize("tamper", ["excerpts", "negative", "boolean"])
def test_unavailable_cached_response_cannot_reach_context_or_mint_handles(tmp_path, tamper):
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        ctx = session(fd, unit(), False)
        lookup = request("source", "source", start=17, end=23)
        response = dict(state="unavailable", error="not delivered", delivered_bytes=0, searched_bytes=0)
        if tamper == "excerpts":
            response["excerpts"] = [dict(unit_id="source", byte_start=17, byte_end=23, text="same\n\n", sha256=dream.digest(b"same\n\n"))]
        else:
            response["delivered_bytes"] = -6 if tamper == "negative" else False
        response["read_id"] = dream.digest(dream.canonical([lookup, response]))
        key = "context-" + dream.digest(dream.canonical([ctx.run_id, lookup, ctx.used])) + ".json"
        dream_run._store(fd, key, dream.canonical(response) + b"\n")
        before = ctx.evidence.snapshot()
        with pytest.raises(dream.DreamError, match="Malformed unavailable context response") as caught:
            ctx.lookup(lookup)
        assert caught.value.exit_code == 3
        assert ctx.trace == [] and ctx.used["source"] == 0
        assert ctx.evidence.snapshot() == before and ctx.windows == {"source": [(23, 29)]}
    finally:
        os.close(fd)


def test_replay_does_not_replenish_extra_source_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(dream_context, "SOURCE_BYTES", 6)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        ctx = session(fd, unit(), True)
        lookup = request("source", "source", start=17, end=23)
        assert ctx.lookup(lookup)["state"] == "complete"
        before = ctx.evidence.snapshot()
        assert ctx.lookup(lookup)["state"] == "unavailable"
        assert ctx.used["source"] == 6 and ctx.evidence.snapshot() == before
        replay = session(fd, unit(), False)
        replay.lookup(lookup)
        assert replay.lookup(lookup)["state"] == "unavailable"
        assert replay.payload() == ctx.payload()
    finally:
        os.close(fd)


def test_native_human_record_is_separate_from_adjacent_agent_and_summary_records():
    rows = [dict(type="assistant", message=dict(role="assistant", content="Reported approval")),
            dict(type="user", message=dict(role="user", content="Approved by me")),
            dict(type="summary", summary="Compaction approval")]
    for row in rows:
        row["sessionId"] = "fixture"
    raw = b"".join(dream.canonical(row) + b"\n" for row in rows)
    source = dict(unit(raw.decode(), 0), kind="claude-jsonl", provenance=dream_transcript.normalize(raw, "claude-jsonl", "fixture"))
    index = registry(source)
    passages = index.deliver("source", 0, len(raw))
    editor = object.__new__(dream_editor.Editor)
    decisions = [p for p in passages if editor._human_source(index.resolve(dict(handle=p["handle"])), [source])]
    assert [p["text"] for p in decisions] == ["Approved by me"]
    assert decisions[0]["provenance"]["recorded_roles"] == ["user"]
    current = object.__new__(handles.Editor)
    current.run_id, current.deadline, current.prefixes, current.context = "fixture", time.monotonic() + 60, [], SimpleNamespace()
    delivered = current.input_sources([source])[0]
    assert "text" not in delivered and "provenance" not in delivered
    assert "".join(p["text"] for p in delivered["passages"]) == raw.decode()
    assert "records" not in delivered["native_context"]
    assert delivered["native_context"]["pending_calls"] == source["provenance"]["pending_calls"]


def test_long_native_user_message_retains_citable_body_fragments():
    body = "x" * 1800 + "Approve option one. " + "y" * 1800
    raw = dream.canonical(dict(sessionId="fixture", type="user", message=dict(role="user", content=body))) + b"\n"
    source = dict(unit(raw.decode(), 0), kind="claude-jsonl", provenance=dream_transcript.normalize(raw, "claude-jsonl", "fixture"))
    index = registry(source)
    passages = index.deliver("source", 0, len(raw))
    editor = object.__new__(dream_editor.Editor)
    approval = next(p for p in passages if "Approve option one." in p["text"])
    assert editor._human_source(index.resolve(dict(handle=approval["handle"])), [source])


def test_bad_handle_and_uncited_talk_are_repaired_together_then_replayed(workspace):
    plan = planned(workspace)
    assert plan["authoring_version"] == 7
    def mutate(task, result):
        if task["role"] == "reviewer":
            return
        page = next(p for p in result["pages"] if p["path"] == TOPIC)
        page["content"] = page["content"].replace(POLICY, "* Talk\n:PROPERTIES:\n:CUSTOM_ID: talk\n:END:\nWhat supports the rule?\n\n" + POLICY)
        if task["role"] == "author":
            result["references"][0]["citations"] = [dict(handle="ev-undelivered")]
        else:
            assert {f["code"] for f in task["input"]["findings"]} == {"citation_handle", "reference_missing"}
            result["references"].append(dict(result["references"][0], target="talk"))
    transport = EditorTransport(mutate=mutate)
    result = wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "claude", "fixture", transport=transport)
    assert [call["role"] for call in transport.calls] == ["author", "repair", "reviewer"]
    assert dream_run.status(workspace[2], "sample", Path(plan["plan_path"]))["state"] == "candidate_ready"
    assert len(transport.calls) == 3
    for call in transport.calls:
        for source in call["input"]["sources"]:
            assert "text" not in source and source["passages"]
    receipt = json.loads(git(workspace[0], "show", result["candidate_commit"] + ":.sucoder/wiki/receipts/" + result["run_id"] + ".json"))
    assert receipt["evidence_registry"]["sha256"] == receipt["context"]["evidence"]["sha256"]
    assert all(set(c) == {"handle", "unit_id", "quote", "byte_start", "byte_end"} for r in receipt["references"] for c in r["citations"])


def test_expansion_limit_counts_repeated_handle_uses(workspace, monkeypatch):
    plan = planned(workspace)
    monkeypatch.setattr(handles, "EXPANSION_BYTES", 512)
    def repeated(task, result):
        if task["role"] != "reviewer":
            handle = task["input"]["sources"][0]["passages"][0]["handle"]
            result["references"][0]["citations"] = [dict(handle=handle)] * 16
    transport = EditorTransport(mutate=repeated)
    with pytest.raises(dream.DreamError, match="Expanded evidence citation byte allowance"):
        wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "claude", "fixture", transport=transport)
    assert [call["role"] for call in transport.calls] == ["author"]


def test_editorial_receipt_can_be_reused_with_canonical_action_evidence(workspace):
    from tests.test_dream_editor import Lifecycle, planning, prepare
    prepare(workspace)
    plan = planning(workspace, "Advance the checked correction")
    transport = Lifecycle("NEXT")
    result = wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "claude", "fixture", transport=transport)
    receipt = json.loads(git(workspace[0], "show", result["candidate_commit"] + ":.sucoder/wiki/receipts/" + result["run_id"] + ".json"))
    assert receipt["editorial"]["task_actions"][0]["evidence"][0]["handle"]
    assert dream_progress.envelope(plan, receipt, "claude", "fixture", transport.version, wiki.PURPOSE)
    assert dream_run.status(workspace[2], "sample", Path(plan["plan_path"]))["state"] == "candidate_ready"
    for mutation in ("proof", "citation", "length", "container", "quote_type"):
        changed = copy.deepcopy(receipt)
        if mutation == "proof":
            changed["evidence_registry"]["sha256"] = "0" * 64
        elif mutation == "citation":
            changed["editorial"]["task_actions"][0]["evidence"][0]["handle"] = "ev-wrong"
        elif mutation == "container":
            changed["context"]["evidence"] = []
        elif mutation == "quote_type":
            changed["references"][0]["citations"][0]["quote"] = 1
        else:
            snapshot = changed["context"]["evidence"]
            entry = snapshot["entries"][0]
            entry["byte_end"] -= 1
            for owner in [*changed["references"], *changed["editorial"]["task_actions"]]:
                for citation in owner["citations" if "citations" in owner else "evidence"]:
                    citation["byte_end"] = entry["byte_end"]
            snapshot["sha256"] = dream.digest(dream.canonical({k: v for k, v in snapshot.items() if k != "sha256"}))
            changed["evidence_registry"]["sha256"] = snapshot["sha256"]
        with pytest.raises(dream.DreamError, match="evidence (registry proof|handle)"):
            dream_progress.envelope(plan, changed, "claude", "fixture", transport.version, wiki.PURPOSE)
