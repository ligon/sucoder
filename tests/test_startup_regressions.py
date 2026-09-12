"""Exercise startup against real files/processes, not success-only mocks."""
import subprocess
from pathlib import Path

import pytest

from sucoder.executor import CommandResult
from sucoder.mirror import MirrorError, MirrorManager
from sucoder.remote_bootstrap import INITIALIZE_MIRROR_SH
from tests.test_remote import _build_remote_manager


def git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def remote(tmp_path, monkeypatch):
    manager = _build_remote_manager(tmp_path)
    path = tmp_path / "remote"
    monkeypatch.setattr(manager, "_resolve_remote_path", lambda ctx: str(path))
    monkeypatch.setattr(manager, "_git_transports", lambda ctx: [("test", str(path), None)])
    return manager, manager.context_for("rproj"), path


def test_first_publication_has_no_fetch_or_force(remote, monkeypatch):
    manager, ctx, path = remote
    calls = []
    original = manager.executor.run_human

    def record(args, **kwargs):
        calls.append(args)
        return original(args, **kwargs)

    monkeypatch.setattr(manager.executor, "run_human", record)
    assert manager.ensure_remote_clone(ctx)
    assert (path / "README.md").read_text() == "hi\n"
    assert not any(c[:2] == ["git", "fetch"] for c in calls)
    assert not any("--force" in c or c[:2] == ["git", "reset"] for c in calls)


@pytest.mark.parametrize("rc,stdout", [(255, ""), (128, ""), (0, "unexpected"),
                                      (1, "SUCODER_MIRROR_CREATED")])
def test_failed_initialization_cannot_authorize_mutation(remote, monkeypatch, rc, stdout):
    manager, ctx, path = remote
    path.mkdir()
    (path / "precious").write_text("keep")
    calls = []

    def failed(args, **kwargs):
        calls.append(args)
        return CommandResult(args, args, stdout, "probe failed", rc)

    monkeypatch.setattr(manager.executor, "run_agent", failed)
    with pytest.raises(MirrorError, match="safely initialize"):
        manager.ensure_remote_clone(ctx, allow_unverified_mirror=True)
    assert len(calls) == 1
    assert (path / "precious").read_text() == "keep"


@pytest.mark.parametrize("kind", ["files", "broken", "ancestor", "symlink", "bare"])
def test_existing_unknown_directories_survive(remote, kind):
    manager, ctx, path = remote
    path.mkdir()
    if kind == "broken":
        (path / ".git").mkdir()
    elif kind == "ancestor":
        git(path.parent, "init", "-b", "main")
    elif kind == "symlink":
        target = path.with_name("target")
        path.rename(target)
        path.symlink_to(target, target_is_directory=True)
    elif kind == "bare":
        git(path, "init", "--bare")
    (path / "precious").write_text("keep")
    with pytest.raises(MirrorError):
        manager.ensure_remote_clone(ctx)
    assert (path / "precious").read_text() == "keep"


def test_empty_repository_is_recovered_without_deletion(remote):
    manager, ctx, path = remote
    path.mkdir()
    git(path, "init", "-b", "main")
    git(path, "config", "review.preserve", "yes")
    assert manager.ensure_remote_clone(ctx)
    assert git(path, "config", "review.preserve") == "yes"
    assert (path / "README.md").exists()


def test_unborn_head_with_feature_branch_is_not_empty(remote):
    manager, ctx, path = remote
    subprocess.run(["git", "clone", str(ctx.canonical_path), str(path)], check=True,
                   capture_output=True)
    git(path, "branch", "-m", "feature")
    git(path, "symbolic-ref", "HEAD", "refs/heads/unborn")
    before = git(path, "rev-parse", "feature")
    with pytest.raises(MirrorError, match="Refusing to push"):
        manager.ensure_remote_clone(ctx)
    assert git(path, "rev-parse", "feature") == before
    assert (path / "README.md").exists()


def test_missing_ref_with_disconnect_tail_is_not_transport_failure():
    result = CommandResult([], [], "", "shell startup noise\n"
                           "fatal: couldn't find remote ref main\n"
                           "fatal: the remote end hung up unexpectedly\n", 128)
    assert not MirrorManager._is_transport_failure(result)
    assert MirrorManager._short_git_error(result) == "fatal: couldn't find remote ref main"


def test_failed_status_cannot_authorize_push(remote, monkeypatch):
    manager, ctx, path = remote
    path.mkdir()
    git(path, "init", "-b", "main")
    (path / "precious").write_text("keep")
    original = manager.executor.run_agent

    def fail_status(args, **kwargs):
        if args[:2] == ["git", "status"]:
            return CommandResult(args, args, "", "connection closed", 255)
        return original(args, **kwargs)

    monkeypatch.setattr(manager.executor, "run_agent", fail_status)
    monkeypatch.setattr(manager, "_sync_remote", lambda *a, **kw: pytest.fail("unsafe push"))
    with pytest.raises(MirrorError, match="Could not inspect remote working tree"):
        manager.ensure_remote_clone(ctx, allow_unverified_mirror=True)
    assert (path / "precious").read_text() == "keep"


def test_timeout_cannot_authorize_initialization(remote, monkeypatch):
    from sucoder.executor import CommandError
    manager, ctx, path = remote

    def timeout(args, **kwargs):
        raise CommandError("timeout", CommandResult(args, args, "", "timed out", -1))

    monkeypatch.setattr(manager.executor, "run_agent", timeout)
    with pytest.raises(CommandError):
        manager.ensure_remote_clone(ctx)
    assert not path.exists()


def test_fresh_push_cannot_overwrite_concurrent_commit(remote, monkeypatch):
    manager, ctx, path = remote
    original = manager._sync_remote
    remote_commit = []

    def concurrent_writer(ctx, **kwargs):
        git(path, "config", "user.name", "Concurrent writer")
        git(path, "config", "user.email", "test@example.com")
        (path / "precious").write_text("keep")
        git(path, "add", "precious")
        git(path, "commit", "-m", "concurrent work")
        remote_commit.append(git(path, "rev-parse", "HEAD"))
        return original(ctx, **kwargs)

    monkeypatch.setattr(manager, "_sync_remote", concurrent_writer)
    with pytest.raises(MirrorError, match="Failed to push"):
        manager.ensure_remote_clone(ctx)
    assert git(path, "rev-parse", "HEAD") == remote_commit[0]
    assert (path / "precious").read_text() == "keep"
