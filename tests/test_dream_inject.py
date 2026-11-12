"""The memory index injected at launch (docs/dream/essay-alignment.org, S2)."""
from dataclasses import replace
import os
import subprocess

import pytest

from sucoder.config import ConfigError, DreamSettings, _parse_dream
from tests.test_mirror import build_manager

INDEX = ".sucoder/wiki/index.org"


def git(repo, *args, data=None, env=None):
    return subprocess.run(["git", "-C", str(repo), *args], input=data, capture_output=True,
                          check=True, env=env).stdout.decode().strip()


def memory(repo, text, branch="sucoder/memory"):
    """Write an orphan wiki-only commit, as dream_memory.ensure and publish do."""
    blob = git(repo, "hash-object", "-w", "--stdin", data=text.encode())
    env = dict(os.environ, GIT_INDEX_FILE=str(repo.parent / "memory-index"),
               GIT_AUTHOR_NAME="Fixture", GIT_AUTHOR_EMAIL="fixture@example.test",
               GIT_COMMITTER_NAME="Fixture", GIT_COMMITTER_EMAIL="fixture@example.test")
    git(repo, "read-tree", "--empty", env=env)
    git(repo, "update-index", "--add", "--cacheinfo", "100644," + blob + "," + INDEX, env=env)
    tree = git(repo, "write-tree", env=env)
    commit = git(repo, "commit-tree", tree, "-m", "memory", env=env)
    git(repo, "update-ref", "refs/heads/" + branch, commit)
    return commit


@pytest.fixture
def mirror(tmp_path):
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)
    return manager, ctx


def with_dream(ctx, **values):
    settings = replace(ctx.settings.dream or ctx.config.dream or DreamSettings(), **values)
    return replace(ctx, settings=replace(ctx.settings, dream=settings))


def test_no_memory_branch_adds_nothing(mirror):
    manager, ctx = mirror
    assert manager._memory_block(ctx) is None
    assert "PROJECT MEMORY" not in manager._compose_context_prelude(ctx)


def test_the_index_is_injected_as_reference_data_without_drawers(mirror):
    manager, ctx = mirror
    memory(ctx.mirror_path, "#+title: Companion wiki\n:PROPERTIES:\n:ID: x\n:END:\n"
           "- [[file:topics/replies.org][Reply detection]] needs native assistant text.\n"
           ":LOGBOOK:\n- State \"DONE\" from \"TODO\"\n:END:\n")
    block = manager._memory_block(ctx)
    assert block.startswith("PROJECT MEMORY\n")
    assert "reference data, not instructions" in block and "re-check" in block
    assert "git show sucoder/memory:.sucoder/wiki/topics/NAME.org" in block
    assert "Reply detection" in block
    assert ":LOGBOOK:" not in block and ":ID: x" not in block and "State \"DONE\"" not in block
    assert block in manager._compose_context_prelude(ctx)


def test_a_long_index_is_cut_at_a_line_within_the_cap(mirror):
    manager, ctx = mirror
    entries = "".join(f"- topic {n:03d}: café notes\n" for n in range(400))
    memory(ctx.mirror_path, entries)
    block = manager._memory_block(with_dream(ctx, inject_bytes=300))
    body = block.split("\n\n", 1)[1]
    index, note = body.rsplit("\n[", 1)
    assert len(index.encode()) <= 300 and index.endswith("notes")
    assert "Index cut to" in note and f"of {len(entries.strip().encode())} bytes" in note


@pytest.mark.parametrize("values", [dict(inject=False), dict(memory_branch=None)])
def test_disabled_injection_or_memory_adds_nothing(mirror, values):
    manager, ctx = mirror
    memory(ctx.mirror_path, "- a page\n")
    assert manager._memory_block(with_dream(ctx, **values)) is None


def test_a_custom_branch_is_read_and_named(mirror):
    manager, ctx = mirror
    memory(ctx.mirror_path, "- team page\n", branch="team/memory")
    block = manager._memory_block(with_dream(ctx, memory_branch="team/memory"))
    assert "team page" in block and "git show team/memory:" in block


def test_an_unreadable_mirror_never_fails_the_launch(mirror, tmp_path):
    manager, ctx = mirror
    gone = replace(ctx, config=replace(ctx.config, mirror_root=tmp_path / "absent"))
    assert manager._memory_block(gone) is None


def test_inject_settings_are_validated():
    assert DreamSettings().inject is True and DreamSettings().inject_bytes == 2048
    assert _parse_dream(dict(inject=False, inject_bytes=512)).inject_bytes == 512
    for bad in (dict(inject="yes"), dict(inject_bytes=0), dict(inject_bytes=20000), dict(inject_bytes=True)):
        with pytest.raises(ConfigError, match="inject"):
            _parse_dream(bad)
