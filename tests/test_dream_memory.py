"""In-place memory-branch generations (docs/dream/essay-alignment.org, S1)."""
from dataclasses import replace
import json
from pathlib import Path
import time

import pytest

from sucoder import dream, dream_memory, dream_reader, dream_run, dream_wiki as wiki
from sucoder.config import ConfigError, DreamSettings, _parse_dream
from tests.test_dream import git, workspace  # noqa: F401
from tests.test_dream_wiki import OLD, TOPIC, EditorTransport, planned

MEMORY = "refs/heads/sucoder/memory"


def source(workspace):
    return dream.GitSource(workspace[2], "sample", time.monotonic() + 30)


def seed(workspace):
    return dream_memory.ensure(source(workspace), MEMORY, git(workspace[0], "rev-parse", "HEAD"),
                               workspace[0].parent, time.monotonic() + 30)


def memory_plan(workspace):
    """An explicit plan against the configured memory branch, as `dream plan --base` would make."""
    planned(workspace)
    seed(workspace)
    repo, config_path, config, request = workspace
    catalog = config_path.parent / "memory-sources.json"
    catalog.write_text(json.dumps(dict(request, integration_ref=MEMORY, units=[
        dict(unit_id=name, kind="git", shareable=True,
             locator=dict(repository="sample", commit="HEAD", path=name + ".org"))
        for name in ("evidence-a", "evidence-b")])))
    return dream.create_plan(config, "sample", catalog, MEMORY)


def refs(repo):
    return git(repo, "for-each-ref", "--format=%(refname)").split()


def test_seed_copies_the_companion_into_an_orphan_wiki_only_branch(workspace):
    planned(workspace)
    repo = workspace[0]
    commit = seed(workspace)
    assert git(repo, "rev-parse", MEMORY) == commit
    assert git(repo, "show", "-s", "--format=%P", commit) == ""
    assert set(git(repo, "ls-tree", "-r", "--name-only", commit).split()) == {TOPIC}
    assert git(repo, "rev-parse", commit + ":" + TOPIC) == git(repo, "rev-parse", "HEAD:" + TOPIC)
    assert seed(workspace) == commit  # Existing branch is reused, never reseeded.


def test_seed_without_a_companion_is_an_empty_root(workspace):
    commit = seed(workspace)
    assert git(workspace[0], "ls-tree", "-r", commit) == ""


def test_a_memory_branch_holding_code_is_refused(workspace):
    repo = workspace[0]
    git(repo, "branch", "sucoder/memory", "main")
    with pytest.raises(dream.DreamError, match="outside .sucoder/wiki/"):
        seed(workspace)


def test_explicit_plan_on_the_memory_branch_publishes_in_place(workspace):
    plan = memory_plan(workspace)
    repo, _, config, _ = workspace
    assert plan["publication"] == "in_place" and plan["integration_ref"] == MEMORY
    base = plan["base_commit"]
    head = git(repo, "rev-parse", "HEAD")
    result = wiki.run(config, "sample", Path(plan["plan_path"]), "codex", "explicit-model", transport=EditorTransport())
    assert result["state"] == "published" and result["candidate_ref"] is None
    assert git(repo, "rev-parse", MEMORY) == result["candidate_commit"]
    assert git(repo, "show", "-s", "--format=%P", result["candidate_commit"]) == base
    assert "is not proof" in git(repo, "show", MEMORY + ":" + TOPIC)
    assert git(repo, "rev-parse", "HEAD") == head and git(repo, "show", "HEAD:" + TOPIC) == OLD.strip()
    assert not [r for r in refs(repo) if r.startswith("refs/heads/dream/")]
    names = set(git(repo, "ls-tree", "-r", "--name-only", MEMORY).split())
    assert all(n.startswith(".sucoder/wiki/") for n in names)
    status = dream_run.status(config, "sample", Path(plan["plan_path"]))
    assert status["state"] == "published" and status["integration"]["state"] == "integrated"


def test_a_later_generation_keeps_an_earlier_one_published_and_a_reset_detaches_it(workspace):
    plan = memory_plan(workspace)
    repo, _, config, _ = workspace
    result = wiki.run(config, "sample", Path(plan["plan_path"]), "codex", "explicit-model", transport=EditorTransport())
    later = git(repo, "commit-tree", result["candidate_commit"] + "^{tree}", "-p", result["candidate_commit"], "-m", "Later generation")
    git(repo, "update-ref", MEMORY, later)
    assert dream_run.status(config, "sample", Path(plan["plan_path"]))["state"] == "published"
    git(repo, "update-ref", MEMORY, plan["base_commit"])
    detached = dream_run.status(config, "sample", Path(plan["plan_path"]))
    assert detached["state"] == "detached" and detached["integration"]["state"] == "not_integrated"


def test_a_concurrent_generation_wins_and_this_one_publishes_nothing(workspace, monkeypatch):
    plan = memory_plan(workspace)
    repo, _, config, _ = workspace
    concurrent = git(repo, "commit-tree", plan["base_commit"] + "^{tree}", "-p", plan["base_commit"], "-m", "Concurrent dream")
    original = dream_run._validate_org
    def advance_then_validate(files, deadline):
        # The other dream lands after this plan's snapshot validation, before publication.
        git(repo, "update-ref", MEMORY, concurrent)
        return original(files, deadline)
    monkeypatch.setattr(dream_run, "_validate_org", advance_then_validate)
    with pytest.raises(dream.DreamError, match="Memory branch moved since planning"):
        wiki.run(config, "sample", Path(plan["plan_path"]), "codex", "explicit-model", transport=EditorTransport())
    assert git(repo, "rev-parse", MEMORY) == concurrent
    assert not [r for r in refs(repo) if r.startswith("refs/heads/dream/")]


def test_an_interrupted_record_after_the_ref_update_replays_without_a_second_update(workspace, monkeypatch):
    plan = memory_plan(workspace)
    repo, _, config, _ = workspace
    original = dream_run._store
    raised = []
    def interrupted(fd, name, data):
        if name.endswith("-run.json") and not raised:
            raised.append(name)
            raise OSError("simulated interruption after the ref transaction")
        return original(fd, name, data)
    monkeypatch.setattr(dream_run, "_store", interrupted)
    transport = EditorTransport()
    with pytest.raises(OSError, match="simulated interruption"):
        wiki.run(config, "sample", Path(plan["plan_path"]), "codex", "explicit-model", transport=transport)
    published = git(repo, "rev-parse", MEMORY)
    assert git(repo, "show", "-s", "--format=%P", published) == plan["base_commit"]
    monkeypatch.setattr(dream_run, "_store", original)
    result = wiki.run(config, "sample", Path(plan["plan_path"]), "codex", "explicit-model", transport=transport)
    assert result["state"] == "published" and result["candidate_commit"] == published
    assert git(repo, "rev-parse", MEMORY) == published and len(transport.calls) == 2


def test_plans_against_code_branches_keep_candidate_publication(workspace):
    plan = planned(workspace)
    repo, _, config, _ = workspace
    assert "publication" not in plan
    result = wiki.run(config, "sample", Path(plan["plan_path"]), "codex", "explicit-model", transport=EditorTransport())
    assert result["state"] == "candidate_ready" and result["candidate_ref"] == "refs/heads/dream/" + result["run_id"]
    assert MEMORY not in refs(repo)


def test_reader_defaults_to_the_memory_branch_once_it_exists(workspace):
    planned(workspace)
    repo, _, config, _ = workspace
    assert dream_reader.View(config, "sample", None).commit == git(repo, "rev-parse", "HEAD")
    commit = seed(workspace)
    assert dream_reader.View(config, "sample", None).commit == commit
    assert dream_reader.read(config, "sample", "reply-evidence", base=None)["commit"] == commit
    off = DreamSettings(memory_branch=None)
    disabled = replace(config, dream=off, mirrors={"sample": replace(config.mirrors["sample"], dream=off)})
    assert dream_reader.View(disabled, "sample", None).commit == git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize("value, expected", [("", None), (None, None), ("  team/memory ", "team/memory")])
def test_memory_branch_setting_accepts_names_or_disabling(value, expected):
    assert _parse_dream(dict(memory_branch=value)).memory_branch == expected
    assert DreamSettings().memory_branch == "sucoder/memory"


@pytest.mark.parametrize("value", ["dream/memory", "../escape", "a//b", "memory.lock", "-flag", 7])
def test_memory_branch_setting_rejects_unsafe_names(value):
    with pytest.raises(ConfigError, match="memory_branch"):
        _parse_dream(dict(memory_branch=value))
