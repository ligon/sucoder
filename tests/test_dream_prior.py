"""Bounded prior selection keeps frozen plans and unseen contributions safe."""
import json
import os
from pathlib import Path
import time

import pytest

from sucoder import dream, dream_prior, dream_run, dream_wiki
from tests.test_dream import git, invoke, workspace  # noqa: F401
from tests.test_dream_wiki import EditorTransport, TOPIC, planned


def growing_wiki(workspace, *, related=False):
    repo = workspace[0]
    root = repo / ".sucoder/wiki"
    (root / "topics").mkdir(parents=True)
    index = "#+title: Companion\n[[file:topics/replies.org::#rule][Replies]]\n[[file:topics/unrelated.org::#other][Unrelated]]\n"
    (root / "index.org").write_text(index)
    (root / "settings.org").write_text(dream_wiki.SETTINGS)
    page = "* Rule\n:PROPERTIES:\n:CUSTOM_ID: rule\n:END:\nFirst qualification.\n"
    if related:
        page += "[[file:neighbor.org::#related][Necessary related decision]]\n"
        (root / "topics/neighbor.org").write_text("* Related\n:PROPERTIES:\n:CUSTOM_ID: related\n:END:\nA related decision.\n")
    (root / "topics/replies.org").write_text(page)
    (root / "topics/unrelated.org").write_text("* Other\n:PROPERTIES:\n:CUSTOM_ID: other\n:END:\n" + "UNRELATED_CANARY\n" * 5000)
    git(repo, "add", ".sucoder/wiki")
    git(repo, "commit", "-qm", "Growing wiki with unrelated context")


def test_explicit_topic_selects_bounded_neighbor_context_not_whole_wiki(workspace):
    growing_wiki(workspace, related=True)
    result = invoke(workspace, None, "--maintenance", "Review a rule", "--topic", "replies")
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    prior = plan["prior_wiki"]
    assert prior["selection"]["strategy"] == "topic"
    assert {p["path"] for p in prior["selected"]} == {".sucoder/wiki/index.org", ".sucoder/wiki/settings.org",
        ".sucoder/wiki/topics/replies.org", ".sucoder/wiki/topics/neighbor.org"}
    assert prior["selected_bytes"] < 2048
    omitted = next(p for p in prior["omitted"] if p["path"].endswith("unrelated.org"))
    assert omitted["reason"] == "unselected_active_topic" and omitted["byte_end"] > 65536
    dream_run.snapshot(workspace[2], "sample", Path(plan["plan_path"]))
    assert all("UNRELATED_CANARY" not in p.read_text() for p in Path(plan["plan_path"]).parent.glob("prior-*.org"))


def test_unknown_source_in_large_wiki_gets_orientation_and_explicit_catalog(workspace):
    growing_wiki(workspace)
    result = invoke(workspace)
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["prior_wiki"]["selection"]["strategy"] == "orientation"
    assert {p["path"] for p in plan["prior_wiki"]["selected"]} == {".sucoder/wiki/index.org", ".sucoder/wiki/settings.org"}
    context = dream_prior.context(plan)
    assert len(context["omitted_active"]) == 2 and context["catalog_entries_omitted"] == 0
    assert "unavailable" in context["rules"]


def test_oversized_or_malformed_association_metadata_does_not_block_planning(workspace):
    growing_wiki(workspace)
    receipts = workspace[0] / ".sucoder/wiki/receipts"
    receipts.mkdir()
    (receipts / ("a" * 64 + ".json")).write_text(" " * (dream_prior.RECEIPT_BYTES + 1))
    (receipts / ("b" * 64 + ".json")).write_text('{"units": [null]}')
    git(workspace[0], "add", ".sucoder/wiki/receipts")
    git(workspace[0], "commit", "-qm", "Unusable optional association metadata")
    result = invoke(workspace)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["prior_wiki"]["selection"]["strategy"] == "orientation"


def test_source_association_selects_current_human_edited_topic(workspace):
    repo = workspace[0]
    first = planned(workspace)
    run = dream_wiki.run(workspace[2], "sample", Path(first["plan_path"]), "codex", "model", transport=EditorTransport())
    git(repo, "merge", "--ff-only", run["candidate_commit"])
    (repo / TOPIC).write_text((repo / TOPIC).read_text() + "\nA human caveat added after the receipt.\n")
    other = repo / ".sucoder/wiki/topics/unrelated.org"
    other.write_text("* Other\n" + "UNRELATED_CANARY\n" * 5000)
    git(repo, "add", TOPIC, str(other.relative_to(repo)))
    git(repo, "commit", "-qm", "Independent human contribution and unrelated growth")
    request = dict(workspace[3], units=[dict(unit_id=name, kind="git", shareable=True,
        locator=dict(repository="sample", commit="HEAD", path=name + ".org")) for name in ("evidence-a", "evidence-b")])
    result = invoke(workspace, request)
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["prior_wiki"]["selection"]["strategy"] == "source_associations"
    assert TOPIC in {p["path"] for p in plan["prior_wiki"]["selected"]}
    assert str(other.relative_to(repo)) not in {p["path"] for p in plan["prior_wiki"]["selected"]}
    dream_run.snapshot(workspace[2], "sample", Path(plan["plan_path"]))
    assert any("A human caveat" in p.read_text() for p in Path(plan["plan_path"]).parent.glob("prior-*.org"))


def test_legacy_frozen_plan_reconstructs_all_active_policy_without_new_fields(workspace):
    growing_wiki(workspace)
    # Use a small old wiki so the original policy's declared allowance fits.
    path = workspace[0] / ".sucoder/wiki/topics/unrelated.org"
    path.write_text("* Other\nSmall original wiki.\n")
    git(workspace[0], "add", str(path.relative_to(workspace[0])))
    git(workspace[0], "commit", "-qm", "Small legacy context")
    current = invoke(workspace)
    assert current.exit_code == 0
    legacy = dream.build_plan(workspace[2], "sample", workspace[3], "refs/heads/main", prior_policy=dream_prior.LEGACY)
    assert "selection" not in legacy["prior_wiki"]
    legacy["plan_id"] = dream.digest(dream.canonical(legacy))
    root = Path(json.loads(current.output)["plan_path"]).parent.parent / legacy["plan_id"]
    root.mkdir(mode=0o700)
    fd = dream._open_private_directory(root)
    try:
        dream_run._store(fd, "plan.json", dream.canonical(legacy) + b"\n")
    finally:
        os.close(fd)
    result = dream_run.snapshot(workspace[2], "sample", root / "plan.json")
    assert result["state"] == "snapshotted"


def test_unselected_existing_page_cannot_be_overwritten_as_new(workspace):
    growing_wiki(workspace)
    envelope = invoke(workspace)
    plan = json.loads(envelope.output)
    proposal = dict(pages=[dict(path=".sucoder/wiki/topics/unrelated.org", content="* Replace unseen human contribution\n")])
    with pytest.raises(dream.DreamError, match="outside the delivered"):
        dream_wiki.validate_proposal(plan, proposal, {}, [], time.monotonic() + 5)


def test_existing_unselected_links_survive_but_new_unverified_links_do_not(workspace):
    growing_wiki(workspace)
    envelope = json.loads(invoke(workspace).output)
    path = ".sucoder/wiki/index.org"
    old = (workspace[0] / path).read_text()
    proposal = dict(pages=[dict(path=path, content=old + "\nA clarified orientation.\n")],
        changes=[dict(path=path, before_sha256=dream.digest(old.encode()), reason="Clarify orientation")], references=[], omissions=[])
    dream_wiki.validate_proposal(envelope, proposal, {path: old, dream_wiki.SETTINGS_PATH: dream_wiki.SETTINGS}, [], time.monotonic() + 5)
    proposal["pages"][0]["content"] += "[[file:topics/replies.org::#unverified][New unverified target]]\n"
    with pytest.raises(dream.DreamError, match="unavailable"):
        dream_wiki.validate_proposal(envelope, proposal, {path: old, dream_wiki.SETTINGS_PATH: dream_wiki.SETTINGS}, [], time.monotonic() + 5)
