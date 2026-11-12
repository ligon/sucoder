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
    ctx = with_dream(ctx, checkout=False)  # S2's form; the read-only copy (S5) is tested below.
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
    block = manager._memory_block(with_dream(ctx, memory_branch="team/memory", checkout=False))
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


def pages(repo, files, branch="sucoder/memory"):
    """A wiki-only memory commit holding ``files`` (path under .sucoder/wiki/ -> text)."""
    env = dict(os.environ, GIT_INDEX_FILE=str(repo.parent / "pages-index"),
               GIT_AUTHOR_NAME="Fixture", GIT_AUTHOR_EMAIL="fixture@example.test",
               GIT_COMMITTER_NAME="Fixture", GIT_COMMITTER_EMAIL="fixture@example.test")
    git(repo, "read-tree", "--empty", env=env)
    for name, text in files.items():
        blob = git(repo, "hash-object", "-w", "--stdin", data=text.encode())
        git(repo, "update-index", "--add", "--cacheinfo", "100644," + blob + ",.sucoder/wiki/" + name, env=env)
    commit = git(repo, "commit-tree", git(repo, "write-tree", env=env), "-m", "memory", env=env)
    git(repo, "update-ref", "refs/heads/" + branch, commit)
    return commit


LINKED = {"index.org": "- [[file:topics/readme.org][About the readme]]\n",
          "topics/readme.org": "* Readme\nSee [[file:../../../README.md][the project readme]].\n"}


def test_a_local_launch_writes_the_read_only_copy_and_points_into_it(mirror):
    manager, ctx = mirror
    commit = pages(ctx.mirror_path, LINKED)
    block = manager._memory_block(ctx)
    assert "Read a page at .sucoder/wiki/topics/NAME.org: a read-only copy of sucoder/memory at " + commit[:12] in block
    page = ctx.mirror_path / ".sucoder/wiki/topics/readme.org"
    assert "the project readme" in page.read_text() and not page.stat().st_mode & 0o222
    assert (page.parent / "../../../README.md").resolve() == (ctx.mirror_path / "README.md").resolve()
    assert git(ctx.mirror_path, "status", "--porcelain", "--untracked-files=all") == ""
    assert manager._memory_block(ctx) == block  # Current: nothing rewritten.


def test_the_copy_follows_checkout_not_inject(mirror):
    manager, ctx = mirror
    pages(ctx.mirror_path, LINKED)
    assert manager._memory_block(with_dream(ctx, inject=False)) is None
    assert (ctx.mirror_path / ".sucoder/wiki/index.org").exists()


def test_checkout_false_or_a_remote_launch_writes_no_copy(mirror, monkeypatch):
    manager, ctx = mirror
    pages(ctx.mirror_path, LINKED)
    assert "git show sucoder/memory:" in manager._memory_block(with_dream(ctx, checkout=False))
    monkeypatch.setattr(type(ctx), "is_remote", property(lambda self: True))
    manager._memory_block(ctx)
    assert not (ctx.mirror_path / ".sucoder/wiki").exists()


def test_a_failing_copy_never_fails_the_launch(mirror, monkeypatch):
    manager, ctx = mirror
    pages(ctx.mirror_path, LINKED)
    def broken(*args, **kwargs):
        raise OSError("sudo unavailable")
    monkeypatch.setattr(manager.executor, "run_agent", broken)
    assert "git show sucoder/memory:" in manager._memory_block(ctx)


def test_checkout_settings_are_validated():
    assert DreamSettings().checkout is True and _parse_dream(dict(checkout=False)).checkout is False
    with pytest.raises(ConfigError, match="checkout"):
        _parse_dream(dict(checkout="yes"))


def test_a_mirror_without_memory_spawns_no_copy(mirror, monkeypatch):
    # Launch tests read the agent command from run_agent's calls; mirrors without memory add none.
    manager, ctx = mirror
    calls = []
    monkeypatch.setattr(manager.executor, "run_agent", lambda *args, **kwargs: calls.append(args))
    assert manager._memory_block(ctx) is None and calls == []
