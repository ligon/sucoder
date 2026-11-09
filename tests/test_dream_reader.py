"""Pinned, bounded native-Org projections and original-source retrieval."""
import json
from pathlib import Path
import time

import pytest
from typer.testing import CliRunner

from sucoder import cli, dream, dream_reader as reader, dream_run
from tests.test_dream import git, invoke, workspace  # noqa: F401


TOPIC = ".sucoder/wiki/topics/replies.org"
PAGE = ("#+title: Replies\n#+SETUPFILE: /unavailable/private/settings.org\n"
        "* Reply evidence\n:PROPERTIES:\n:CUSTOM_ID: reply-proof\n:END:\n"
        ":LOGBOOK:\nHISTORICAL_CANARY\n:END:\n"
        "A compaction summary is not proof of a native assistant reply. Caf\u00e9.\n"
        "** Talk :editorial:\nThe maintainer's timeout decision remains unresolved.\n"
        "* Other topic\n:PROPERTIES:\n:CUSTOM_ID: other\n:END:\nUnrelated material.\n")


def prepared(workspace, *, snapshot=True):
    repo = workspace[0]
    (repo / TOPIC).parent.mkdir(parents=True)
    (repo / TOPIC).write_text(PAGE)
    (repo / ".sucoder/wiki/index.org").write_text("#+title: Memory\n[[file:topics/replies.org::#reply-proof][Reply evidence]]\n")
    archive = repo / ".sucoder/wiki/archive/replies.org"
    archive.parent.mkdir()
    archive.write_text("* DONE Decision\n:PROPERTIES:\n:CUSTOM_ID: decision\n:ARCHIVE_FILE: .sucoder/wiki/topics/replies.org\n:END:\n:LOGBOOK:\nRecorded human review.\n:END:\nRetain the caveat.\n* Other entry\n:PROPERTIES:\n:CUSTOM_ID: unrelated\n:END:\nARCHIVE_CANARY\n")
    git(repo, "add", ".sucoder/wiki")
    git(repo, "commit", "-qm", "Seed bounded reader material")
    result = invoke(workspace)
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    if snapshot:
        dream_run.snapshot(workspace[2], "sample", Path(plan["plan_path"]))
    return plan


def test_orientation_then_topic_exclude_history_and_preserve_caveat(workspace):
    prepared(workspace)
    repo = workspace[0]
    before = (git(repo, "status", "--porcelain"), (repo / ".git/index").read_bytes(), git(repo, "show-ref"))
    orientation = reader.read(workspace[2], "sample")
    assert orientation["delivered_bytes"] <= 2048
    assert "dream source" in orientation["text"]
    assert "Reply evidence" in orientation["text"]
    result = reader.read(workspace[2], "sample", "replies", target="reply-proof")
    assert "not proof" in result["text"]
    assert "unresolved" in result["text"]
    assert "HISTORICAL_CANARY" not in result["text"]
    assert "Unrelated material" not in result["text"]
    assert "ARCHIVE_CANARY" not in result["text"]
    assert result["logbook_bytes_omitted"] > 0
    assert result["commit"] == git(repo, "rev-parse", "HEAD")
    assert before == (git(repo, "status", "--porcelain"), (repo / ".git/index").read_bytes(), git(repo, "show-ref"))


def test_specific_archive_entry_retains_decision_history_and_provenance(workspace):
    prepared(workspace)
    with pytest.raises(dream.DreamError, match="specific"):
        reader.read(workspace[2], "sample", "replies", archives=True)
    result = reader.read(workspace[2], "sample", "replies", archives=True, target="decision")
    assert "Recorded human review" in result["text"]
    assert ":ARCHIVE_FILE: .sucoder/wiki/topics/replies.org" in result["text"]
    assert "ARCHIVE_CANARY" not in result["text"]


def test_native_projection_preserves_literal_drawer_example():
    content = "* Topic\n#+begin_example\n:LOGBOOK:\nLiteral\n:END:\n#+end_example\n:LOGBOOK:\nHistorical\n:END:\nCaveat.\n"
    result = reader.projection(TOPIC, content, time.monotonic() + 5)
    assert "Literal" in result["text"]
    assert "Historical" not in result["text"]
    assert "Caveat" in result["text"]


def test_explicit_utf8_budget_reports_truncation(workspace):
    prepared(workspace)
    result = reader.read(workspace[2], "sample", "replies", max_bytes=100)
    assert result["state"] == "incomplete"
    assert result["delivered_bytes"] <= 100
    assert result["budget_bytes_omitted"] > 0
    assert result["text"].encode().decode() == result["text"]
    text, omitted = reader.bounded("caf\u00e9", 4)
    assert (text, omitted) == ("caf", 2)


@pytest.mark.parametrize("topic", ["../private", "/private", "a/../private", "a\\private"])
def test_topic_scope_cannot_escape_wiki(workspace, topic):
    with pytest.raises(dream.DreamError):
        reader.read(workspace[2], "sample", topic)


def test_source_search_and_ranges_use_same_exact_snapshot(workspace):
    plan = prepared(workspace)
    result = reader.source(workspace[2], "sample", Path(plan["plan_path"]), query="qualification")
    excerpt = result["excerpts"][0]
    assert excerpt["text"] == "First qualification.\r\nSecond: caf\u00e9.\n"
    assert excerpt["byte_start"] == plan["units"][0]["byte_start"]
    direct = reader.source(workspace[2], "sample", Path(plan["plan_path"]), unit_id="note",
        start=excerpt["byte_start"], end=excerpt["byte_end"])
    assert direct["excerpts"] == result["excerpts"]
    assert direct["snapshot_bytes_checked"] == len(excerpt["text"].encode())


def test_source_budget_and_unavailable_or_changed_snapshot_are_explicit(workspace):
    plan = prepared(workspace, snapshot=False)
    path = Path(plan["plan_path"])
    with pytest.raises(dream.DreamError, match="unavailable locally"):
        reader.source(workspace[2], "sample", path)
    dream_run.snapshot(workspace[2], "sample", path)
    result = reader.source(workspace[2], "sample", path, max_bytes=5)
    assert result["state"] == "incomplete" and result["delivered_bytes"] == 5
    assert result["omitted"][0]["reason"] == "source_allowance"
    snapshot = path.parent / ("evidence-" + plan["units"][0]["sha256"] + ".txt")
    snapshot.write_bytes(b"changed")
    with pytest.raises(dream.DreamError, match="digest"):
        reader.source(workspace[2], "sample", path)


def test_cli_read_and_source_bypass_interactive_startup(workspace):
    plan = prepared(workspace)
    runner = CliRunner()
    prefix = ["--config", str(workspace[1]), "dream"]
    result = runner.invoke(cli.app, prefix + ["read", "sample", "--topic", "replies", "--target", "reply-proof"])
    assert result.exit_code == 0, result.output
    assert "not proof" in json.loads(result.output)["text"]
    result = runner.invoke(cli.app, prefix + ["source", "sample", "--plan", plan["plan_path"], "--search", "qualification"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["excerpts"]
