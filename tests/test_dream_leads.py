"""New discovery sees frozen prior memory; earlier jobs replay unchanged."""
import json
from pathlib import Path

import pytest

from sucoder import dream, dream_explore as explore, dream_leads
from tests.test_dream import git, workspace  # noqa: F401
from tests.test_dream_explore import ExploringTransport, options
from tests.test_dream_memory import seed


def memory(workspace):
    repo = workspace[0]
    wiki = repo / ".sucoder/wiki"
    (wiki / "topics").mkdir(parents=True)
    (wiki / "index.org").write_text("#+TITLE: Memory\n* Cache behavior\n[[file:topics/cache.org::#cache][Cache behavior]]\n")
    (wiki / "topics/cache.org").write_text("#+TITLE: Cache\n* Cache\n:PROPERTIES:\n:CUSTOM_ID: cache\n:END:\nKnown behavior.\n"
        "** Talk\nDoes the wave reader restamp hashless caches?\n"
        "*** TODO Check the wave reader\n:PROPERTIES:\n:CUSTOM_ID: wave-reader\n:END:\nRead wave.py.\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "prior memory")
    return seed(workspace)


def test_first_navigation_contains_memory_index_open_questions_and_seed(workspace):
    commit = memory(workspace)
    native = ExploringTransport(no_selection=True)
    result = explore.start(workspace[2], cwd=workspace[0], overrides=options(), transport=native)
    navigation = native.calls[0]["input"]["navigation"]
    leads = navigation["memory"]
    assert leads["commit"] == commit and "Cache behavior" in leads["index"]
    assert any("hashless caches" in item["text"] for item in leads["open_items"])
    assert leads["seed"]["kind"] == "open_item"
    assert "state why" in native.calls[0]["brief"]
    assert len(dream.canonical(leads)) <= dream_leads.MAX_BYTES
    assert navigation["remaining_delivery_bytes"] == 131072 - len(dream.canonical(leads))
    frozen = json.loads((Path(result["job_path"]) / "job.json").read_bytes())
    assert frozen["leads"] == leads
    # A later memory ref cannot change this job's advisory navigation or replay.
    git(workspace[0], "update-ref", "refs/heads/sucoder/memory", git(workspace[0], "commit-tree", commit + "^{tree}", "-p", commit, "-m", "later memory"))
    assert explore.start(workspace[2], result["job_id"]) == result
    assert len(native.calls) == 1


def test_pre_lead_job_keeps_its_old_brief_and_assignments(workspace, monkeypatch):
    memory(workspace)
    monkeypatch.setattr(dream_leads, "VERSION", 0)
    native = ExploringTransport(no_selection=True)
    result = explore.start(workspace[2], cwd=workspace[0], overrides=options(), transport=native)
    before = native.calls[0]
    assert "memory" not in before["input"]["navigation"]
    monkeypatch.setattr(dream_leads, "VERSION", 1)
    assert explore.start(workspace[2], result["job_id"]) == result
    assert len(native.calls) == 1
    assert "frozen navigation.memory" not in before["brief"]


def test_no_prior_memory_selects_and_freezes_a_committed_file(workspace):
    job = explore.prepare(workspace[2], cwd=workspace[0], overrides=options(mode="project"))
    header = json.loads((Path(job["job_path"]) / "job.json").read_bytes())
    assert header["leads"]["seed"] == dict(kind="committed_file", path="note.org", start=0)
    assert explore.prepare(workspace[2], job["job_id"]) == job


def test_memory_delivery_is_bounded_and_does_not_read_the_working_copy(workspace):
    memory(workspace)
    (workspace[0] / ".sucoder/wiki/index.org").write_text("PRIVATE UNCOMMITTED DECOY\n")
    job = explore.prepare(workspace[2], cwd=workspace[0], overrides=options(max_discovery_bytes=1024))
    header = json.loads((Path(job["job_path"]) / "job.json").read_bytes())
    assert len(dream.canonical(header["leads"])) <= 512
    assert "PRIVATE UNCOMMITTED DECOY" not in json.dumps(header["leads"])


def test_changed_file_is_preferred_when_memory_has_no_open_item(workspace):
    repo = workspace[0]
    previous = git(repo, "rev-parse", "HEAD")
    receipts = repo / ".sucoder/wiki/receipts"
    receipts.mkdir(parents=True)
    (receipts / ("a" * 64 + ".json")).write_text(json.dumps(dict(units=[dict(kind="git", commit=previous)])))
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "memory receipt")
    seed(workspace)
    (repo / "note.org").write_text("Changed project evidence.\n")
    git(repo, "add", "note.org")
    git(repo, "commit", "-qm", "changed note")
    job = explore.prepare(workspace[2], cwd=repo, overrides=options(mode="project"))
    header = json.loads((Path(job["job_path"]) / "job.json").read_bytes())
    assert header["leads"]["seed"] == dict(kind="changed_file", path="note.org", previous_commit=previous, start=0)


def test_terminal_tasks_are_not_open_leads(workspace):
    repo = workspace[0]
    wiki = repo / ".sucoder/wiki/topics"
    wiki.mkdir(parents=True)
    (wiki / "done.org").write_text("#+TODO: TODO | DONE MOOT\n* Talk\n** DONE Old correction\n"
        ":PROPERTIES:\n:CUSTOM_ID: finished\n:END:\nAlready resolved.\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "finished memory task")
    seed(workspace)
    job = explore.prepare(workspace[2], cwd=repo, overrides=options(mode="project"))
    header = json.loads((Path(job["job_path"]) / "job.json").read_bytes())
    assert header["leads"]["open_items"] == []
    assert header["leads"]["seed"]["kind"] == "committed_file"


def test_session_seed_is_a_complete_record_even_when_random_byte_is_in_final_record(workspace, monkeypatch):
    from types import SimpleNamespace
    from tests.test_dream_experience import write_session
    transcript = write_session(Path.home(), workspace[0])
    monkeypatch.setattr(explore, "_live_sessions", lambda config: set())
    monkeypatch.setattr(dream_leads.random, "Random", lambda seed: SimpleNamespace(
        choice=lambda rows: rows[0], randrange=lambda size: size - 1))
    job = explore.prepare(workspace[2], cwd=workspace[0], overrides=options())
    header = json.loads((Path(job["job_path"]) / "job.json").read_bytes())
    seed = header["leads"]["seed"]
    raw = transcript.read_bytes()
    starts = {0} | {i + 1 for i, char in enumerate(raw) if char == 10 and i + 1 < len(raw)}
    assert seed["kind"] == "session_window" and seed["start"] in starts
