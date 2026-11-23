"""Receipt-backed claims survive narrower selections; old plans retain briefs."""
import json
from pathlib import Path

import pytest

from sucoder import dream, dream_handles, dream_editor, dream_run, dream_support, dream_wiki
from tests.dream_submission_fixture import source_text
from tests.test_dream import git, workspace  # noqa: F401
from tests.test_dream_memory import MEMORY, memory_plan
from tests.test_dream_wiki import EditorTransport, TOPIC


def transport(mutate=None):
    def current_hashes(task, result):
        if task["role"] != "reviewer":
            prior = {page["path"]: page["sha256"] for page in task["input"]["prior"]}
            for change in result["changes"]:
                change["before_sha256"] = prior.get(change["path"], "")
        if mutate:
            mutate(task, result)
    return EditorTransport(mutate=current_hashes)


def next_plan(workspace, *, narrow=False, maintenance="Inspect the narrower evidence without erasing supported memory"):
    catalog = workspace[1].parent / "next.json"
    units = [dict(unit_id=name, kind="git", shareable=True,
        locator=dict(repository="sample", commit="HEAD", path=name + ".org",
            **(dict(lines=[1, 1]) if narrow and name == "evidence-b" else {})))
        for name in ("evidence-a", "evidence-b")]
    catalog.write_text(json.dumps(dict(schema_version=1, repository="sample", integration_ref=MEMORY, units=units)))
    return dream.create_plan(workspace[2], "sample", catalog, MEMORY,
        maintenance=maintenance, topic="reply-evidence" if maintenance else None)


def split_source(workspace):
    path = workspace[0] / "evidence-b.org"
    path.write_text(path.read_text().replace(" Use --session-id", "\nUse --session-id"))
    git(workspace[0], "add", "evidence-b.org")
    git(workspace[0], "commit", "-qm", "separate evidence lines")


def test_narrower_generation_receives_and_keeps_previous_support(workspace):
    memory_plan(workspace)
    split_source(workspace)
    first = next_plan(workspace)
    first_result = dream_wiki.run(workspace[2], "sample", Path(first["plan_path"]), "codex", "fixture", transport=EditorTransport())
    second = next_plan(workspace, narrow=True)
    assert second["editorial_version"] == 2
    assert any(row["path"] == TOPIC and row["target"] == "reply-proof" for row in second["prior_support"])

    def preserve(task, result):
        if task["role"] == "author":
            inputs = task["input"]
            assert "--session-id" not in next(source_text(u) for u in inputs["sources"] if u["unit_id"] == "evidence-b")
            assert any(row["target"] == "reply-proof" for row in inputs["prior_support"])
            assert "a narrower current evidence selection is not a reason to erase" in task["brief"]
            assert "previously_admitted_not_reread" in json.dumps(inputs["prior_support"])
            for page in result["pages"]:
                if page["path"] == TOPIC:
                    page["content"] = page["content"].replace("Native assistant text must follow", "For the documented protocol, native assistant text must follow")
        if task["role"] == "reviewer":
            assert task["input"]["prior_support"] and "preservation as well as correction" in task["brief"]
    native = transport(preserve)
    result = dream_wiki.run(workspace[2], "sample", Path(second["plan_path"]), "codex", "fixture", transport=native)
    assert result["state"] == "published"
    assert "--session-id <uuid>" in git(workspace[0], "show", MEMORY + ":" + TOPIC)
    receipt = json.loads(git(workspace[0], "show", MEMORY + ":.sucoder/wiki/receipts/" + result["run_id"] + ".json"))
    assert receipt["inherited_support"]
    # Support survives another generation, rather than disappearing after one handoff.
    third = next_plan(workspace, narrow=True)
    assert any(row["receipt"].endswith(first_result["run_id"] + ".json") for row in third["prior_support"])
    assert dream_run.status(workspace[2], "sample", Path(second["plan_path"]))["state"] == "published"


def test_contradicting_current_evidence_still_allows_a_correction(workspace):
    plan = memory_plan(workspace)
    dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "fixture", transport=EditorTransport())
    path = workspace[0] / "evidence-b.org"
    path.write_text("Reply evidence requires native assistant text. The launch flag is now --thread <uuid>; --session-id is obsolete.\n")
    git(workspace[0], "add", "evidence-b.org")
    git(workspace[0], "commit", "-qm", "current contradictory flag evidence")
    current = next_plan(workspace)
    def correct(task, result):
        assert task["input"]["prior_support"]
        if task["role"] != "reviewer":
            for page in result["pages"]:
                page["content"] = page["content"].replace("Use =--session-id <uuid>=", "Use =--thread <uuid>=")
            result["changes"][0]["reason"] = "Current committed evidence contradicts the prior launch flag."
        else:
            result.update(verdict="accept", findings=[])
    result = dream_wiki.run(workspace[2], "sample", Path(current["plan_path"]), "codex", "fixture", transport=transport(correct))
    assert result["state"] == "published"
    text = git(workspace[0], "show", MEMORY + ":" + TOPIC)
    assert "Use =--thread <uuid>=" in text and "Use =--session-id <uuid>=" not in text


def test_plan_prepared_before_editorial_amendment_replays_unchanged(workspace, monkeypatch):
    monkeypatch.setattr(dream_support, "VERSION", 1)
    plan = memory_plan(workspace)
    assert "editorial_version" not in plan and "prior_support" not in plan
    native = EditorTransport()
    result = dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "fixture", transport=native)
    assert all(task["brief"] == dream_handles.BRIEFS[task["role"]] for task in native.calls)
    monkeypatch.setattr(dream_support, "VERSION", 2)
    assert dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "fixture", transport=native) == result
    assert dream_run.status(workspace[2], "sample", Path(plan["plan_path"]))["state"] == "published"
    assert len(native.calls) == 2


def test_new_policy_does_not_skip_review_based_on_an_older_policy_receipt(workspace, monkeypatch):
    from sucoder import dream_author
    from tests.test_dream_curate import NoopTransport
    with monkeypatch.context() as previous:
        previous.setattr(dream_support, "VERSION", 1)
        previous.setattr(dream_editor, "Editor", dream_author.Author)
        memory_plan(workspace)
        before = dream.create_plan(workspace[2], "sample", workspace[1].parent / "memory-sources.json", MEMORY, authoring_version=5)
        dream_wiki.run(workspace[2], "sample", Path(before["plan_path"]), "codex", "fixture", transport=EditorTransport())
    after = next_plan(workspace, maintenance=None)
    native = NoopTransport()
    result = dream_wiki.run(workspace[2], "sample", Path(after["plan_path"]), "codex", "fixture", transport=native)
    assert result["state"] == "unchanged"
    assert [call["role"] for call in native.calls] == ["author", "reviewer"]


@pytest.mark.parametrize("field,bad", [("units", None), ("outputs", [None]), ("references", [dict(path="x", target="y", citations=None)]), ("review", [])])
def test_malformed_support_provenance_is_a_diagnostic_not_a_traceback(field, bad):
    with pytest.raises(dream.DreamError, match="receipt"):
        dream_support.check_receipt({field: bad})
