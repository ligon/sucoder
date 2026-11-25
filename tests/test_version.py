"""An editable install reports its checkout's version, not the install-time one."""
from __future__ import annotations

import subprocess

import pytest

import sucoder


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "src"
    (root / "sucoder").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[tool.poetry]\nname = 'sucoder'\n")
    git(root, "init", "-q")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.com")
    git(root, "add", ".")
    git(root, "commit", "-qm", "base")
    git(root, "tag", "v2026.7.1")
    return root


def commit(root, name):
    (root / name).write_text(name)
    git(root, "add", name)
    git(root, "commit", "-qm", name)
    return git(root, "rev-parse", "HEAD")


def test_on_a_tag_the_version_is_the_tag(checkout):
    assert sucoder._checkout_version(checkout) == "2026.7.1"


def test_past_a_tag_it_matches_the_build_format(checkout):
    commit(checkout, "a")
    head = commit(checkout, "b")
    short = git(checkout, "rev-parse", "--short", "HEAD")
    assert sucoder._checkout_version(checkout) == f"2026.7.1.dev2+g{short}"
    assert (checkout / ".git" / "sucoder-version").read_text().split()[0] == head


def test_moving_the_checkout_moves_the_version_and_a_repeat_uses_the_cache(checkout, monkeypatch):
    first = sucoder._checkout_version(checkout)
    commit(checkout, "a")
    second = sucoder._checkout_version(checkout)
    assert first == "2026.7.1" and second.startswith("2026.7.1.dev1+g")

    def no_git(*args, **kwargs):
        raise AssertionError("git ran although HEAD matched the cache")

    monkeypatch.setattr(sucoder.subprocess, "run", no_git)
    assert sucoder._checkout_version(checkout) == second


@pytest.mark.parametrize("shape", ["detached", "packed"])
def test_head_resolves_detached_and_packed_refs(checkout, shape):
    head = commit(checkout, "a")
    if shape == "detached":
        git(checkout, "checkout", "-q", "--detach", "HEAD")
    else:
        git(checkout, "pack-refs", "--all")
        assert not (checkout / ".git" / "refs" / "heads" / git(checkout, "branch", "--show-current")).exists()
    assert sucoder._head_commit(checkout / ".git") == head
    assert sucoder._checkout_version(checkout).startswith("2026.7.1.dev1+g")


def test_a_linked_worktree_reads_its_own_head(checkout, tmp_path):
    base = git(checkout, "rev-parse", "HEAD")
    other = tmp_path / "other"
    git(checkout, "worktree", "add", "-q", "--detach", str(other))
    moved = commit(other, "a")
    assert sucoder._head_commit(sucoder._git_dir(other)) == moved
    assert sucoder._head_commit(sucoder._git_dir(checkout)) == base
    assert sucoder._checkout_version(other).startswith("2026.7.1.dev1+g")


def test_without_a_reachable_tag_the_commit_is_still_named(checkout):
    git(checkout, "tag", "-d", "v2026.7.1")
    head = git(checkout, "rev-parse", "HEAD")
    assert sucoder._checkout_version(checkout) == "0+g" + head[:7]


def test_an_installed_package_keeps_its_metadata_version(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    (site / "sucoder").mkdir(parents=True)
    assert sucoder._checkout_version(site) is None
    monkeypatch.setattr(sucoder, "_checkout_version", lambda root: None)
    monkeypatch.setattr(sucoder, "_metadata_version", lambda: "2026.7.1.dev317+gc9dafdb")
    assert sucoder._version() == "2026.7.1.dev317+gc9dafdb"


def test_a_failure_while_reading_the_checkout_never_breaks_the_version(monkeypatch):
    def broken(root):
        raise RuntimeError("unreadable checkout")

    monkeypatch.setattr(sucoder, "_checkout_version", broken)
    monkeypatch.setattr(sucoder, "_metadata_version", lambda: "9.9.9")
    assert sucoder._version() == "9.9.9"


def test_the_version_is_computed_once_on_first_use(monkeypatch):
    real = sucoder.__version__   # present now, so teardown restores it
    monkeypatch.delitem(sucoder.__dict__, "__version__")
    calls = []
    monkeypatch.setattr(sucoder, "_version", lambda: calls.append(1) or "1.2.3")
    assert sucoder.__version__ == "1.2.3" and sucoder.__version__ == "1.2.3"
    assert calls == [1]
    assert real
