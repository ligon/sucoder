import grp
import logging
import os
import pwd
import subprocess
import types
from pathlib import Path
from typing import Callable, Dict, Optional

import pytest

import sucoder.mirror as mirror
from sucoder.config import AgentLauncher, AuditConfig, BranchPrefixes, Config, McpServerConfig, MirrorSettings, NvmConfig
from sucoder.executor import CommandError, CommandExecutor, CommandResult
from sucoder.mirror import (
    MirrorError,
    MirrorManager,
    WorktreeInfo,
    _detect_agent_type,
    _merge_flag_templates,
    _parse_version,
    _parse_worktree_porcelain,
    _probe_binary_version,
    _sanitize_task_name,
)
from sucoder.permissions import check_parent_traversable
from sucoder.workspace_prefs import WorkspacePrefs


def _extract_prelude(args):
    """Extract the system prompt prelude from a launched agent's args.

    Claude uses --system-prompt <content>; other agents use trailing text.
    """
    if "--system-prompt" in args:
        idx = args.index("--system-prompt")
        return args[idx + 1]
    return args[-1]


def run_git(args, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def _git_is(args, *want: str) -> bool:
    """Match a git invocation, skipping leading ``-c <value>`` flags.

    ``MirrorManager._safe_directory_args`` prepends one or more
    ``-c safe.directory=<path>`` pairs to git invocations, so naive
    prefix matches like ``_git_is(args, "rev-parse", "--verify")``
    no longer hold.  This helper walks past any ``-c <value>`` pairs
    after the leading ``git`` and then checks the remaining prefix.
    """
    if not args or args[0] != "git":
        return False
    i = 1
    while i + 1 < len(args) and args[i] == "-c":
        i += 2
    return list(args[i:i + len(want)]) == list(want)


def create_canonical_repo(path: Path) -> None:
    run_git(["init", "-b", "main"], path)
    run_git(["config", "user.email", "test@example.com"], path)
    run_git(["config", "user.name", "Test User"], path)
    (path / "README.md").write_text("hello\n", encoding="utf-8")
    run_git(["add", "README.md"], path)
    run_git(["commit", "-m", "initial"], path)


def build_manager(
    tmp_path: Path,
    *,
    prompt_handler: Optional[Callable[[str], bool]] = None,
    executor: Optional[CommandExecutor] = None,
    report_agent_binary: bool = False,
) -> MirrorManager:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    create_canonical_repo(canonical)

    # In tests agent_user == current user (the dir owner), so `test -w` would
    # always succeed.  Strip owner-write on the canonical directory so the
    # _validate_canonical write-check passes, matching production behaviour
    # where the agent is a different (non-owner) user.
    canonical.chmod(canonical.stat().st_mode & ~0o200)

    os.environ["GIT_CONFIG_GLOBAL"] = str(tmp_path / "gitconfig")

    mirror_root = tmp_path / "mirrors"
    mirror_root.mkdir()

    user = pwd.getpwuid(os.getuid()).pw_name
    group = grp.getgrgid(os.getgid()).gr_name
    if executor is not None:
        user = executor.human_user
        group = executor.agent_group

    settings = MirrorSettings(
        name="sample",
        canonical_repo=canonical,
        mirror_name="sample",
        branch_prefixes=BranchPrefixes(human="ligon", agent="coder"),
        default_base_branch="main",
        task_branch_prefix="task",
    )

    config = Config(
        human_user=user,
        agent_user=user,
        agent_group=group,
        mirror_root=mirror_root,
        log_dir=None,
        mirrors={"sample": settings},
    )

    logger = logging.getLogger("sucoder.test")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    if executor is None:
        executor = CommandExecutor(
            human_user=config.human_user,
            agent_user=config.agent_user,
            agent_group=config.agent_group,
            logger=logger,
            dry_run=False,
            use_sudo_for_agent=False,
        )

    manager = MirrorManager(config, executor, logger, prompt_handler=prompt_handler)
    # Point skills dir at a nonexistent path so auto-commit doesn't
    # interfere with tests that don't care about skills tracking.
    type(manager)._agent_skills_dir = property(lambda self, p=tmp_path / "_no_skills": p)
    # The binary-resolution diagnostic spawns `--version` subprocesses against
    # whatever real agent binaries sit on the developer's PATH and walks the
    # real filesystem.  Left live it would make every launch_agent test
    # host-dependent (and slow, by the probe timeout, whenever a local agent
    # binary is sluggish).  Tests that exercise it opt back in explicitly.
    if not report_agent_binary:
        manager._report_agent_binary = lambda *a, **kw: None  # type: ignore[method-assign]
    return manager


def test_clone_sync_and_start_task(tmp_path: Path) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    manager.ensure_clone(ctx)
    mirror_path = ctx.mirror_path
    assert (mirror_path / ".git").exists()

    push_url = (
        run_git(["remote", "get-url", "--push", ctx.remote_name], mirror_path)
        .stdout.strip()
    )
    assert push_url == "no_push"

    manager.sync(ctx)
    branch = manager.start_task(ctx, task_name="Demo Task", base_branch="main")
    assert branch.startswith("coder/demo-task-")

    head = run_git(["rev-parse", "--abbrev-ref", "HEAD"], mirror_path).stdout.strip()
    assert head == branch

    run_git(["show-ref", "--verify", f"refs/heads/{ctx.remote_name}/main"], mirror_path)

    status = manager.status(ctx)
    assert branch.split("/")[-1] in status
    assert "Remote" in status
    assert "Agent access:" in status


def test_clone_allows_direnv_for_envrc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    canonical = ctx.canonical_path
    # Temporarily restore write so we can commit a new file.
    canonical.chmod(canonical.stat().st_mode | 0o200)
    envrc = canonical / ".envrc"
    envrc.write_text("layout poetry\n", encoding="utf-8")
    run_git(["add", ".envrc"], canonical)
    run_git(["commit", "-m", "add envrc"], canonical)
    canonical.chmod(canonical.stat().st_mode & ~0o200)

    original_run_agent = manager.executor.run_agent
    direnv_calls = []

    def fake_run_agent(args, **kwargs):
        args_list = list(args)
        if args_list[:2] == ["direnv", "allow"]:
            direnv_calls.append(args_list)
            return CommandResult(
                requested_args=args_list,
                executed_args=args_list,
                stdout="",
                stderr="",
                returncode=0,
            )
        return original_run_agent(args, **kwargs)

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        mirror.shutil,
        "which",
        lambda cmd: "/usr/bin/direnv" if cmd == "direnv" else None,
    )

    manager.ensure_clone(ctx)

    assert direnv_calls, "direnv allow should be invoked when .envrc is present."
    assert (ctx.mirror_path / ".envrc").exists()


def test_unlock_git_crypt_no_op_when_already_unlocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_unlock_git_crypt`` short-circuits when the mirror is already unlocked.

    Regression test: a previous version of the function looked for
    ``"encrypted:" not in `git-crypt status` stdout`` to detect the
    already-unlocked case.  ``git-crypt status`` prints
    ``encrypted: <path>`` for every file *configured* to be encrypted
    regardless of lock state, so the early-exit never fired.  That made
    every session start delete the live mirror key and then claw its way
    back out via the chicken-and-egg filter-neuter workaround -- emitting
    alarming ``git-crypt: Error: Unable to open key file`` /
    ``fatal: clean filter 'git-crypt' failed`` messages each time.

    The new check is: mirror key file present AND ``git status`` clean ->
    we're already unlocked, do nothing.  This test asserts no
    ``git-crypt`` invocation and no removal of the mirror key.
    """
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)
    mirror_path = ctx.mirror_path

    # Stage "git-crypt unlocked": both canonical and mirror have a key
    # file at .git/git-crypt/keys/default.  The bytes don't have to be
    # a real git-crypt key -- nothing in this test actually invokes
    # git-crypt on them.
    fake_key = b"\x00GITCRYPTKEY\x00\x00\x00\x02\x00\x00\x00\x00"
    for repo in (ctx.canonical_path, mirror_path):
        keys_dir = repo / ".git" / "git-crypt" / "keys"
        keys_dir.mkdir(parents=True, exist_ok=True)
        (keys_dir / "default").write_bytes(fake_key)
    mirror_key = mirror_path / ".git" / "git-crypt" / "keys" / "default"

    original_run_agent = manager.executor.run_agent
    git_crypt_calls: list[list[str]] = []
    rm_key_calls: list[list[str]] = []

    def spy_run_agent(args, **kwargs):
        args_list = list(args)
        if args_list[:1] == ["git-crypt"]:
            git_crypt_calls.append(args_list)
        if args_list[:2] == ["rm", "-f"] and str(mirror_key) in args_list:
            rm_key_calls.append(args_list)
        return original_run_agent(args, **kwargs)

    monkeypatch.setattr(manager.executor, "run_agent", spy_run_agent)

    manager._unlock_git_crypt(ctx, mirror_path)

    assert mirror_key.is_file(), "Mirror key should not have been deleted."
    assert git_crypt_calls == [], (
        "_unlock_git_crypt should be a no-op when the mirror is already "
        f"unlocked; got git-crypt calls: {git_crypt_calls}"
    )
    assert rm_key_calls == [], (
        "_unlock_git_crypt should not delete the mirror key when it is "
        f"already unlocked; got rm calls: {rm_key_calls}"
    )


def test_unlock_git_crypt_skipped_when_canonical_unlocked_but_no_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No canonical key => nothing to do, regardless of mirror state."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)
    mirror_path = ctx.mirror_path

    # No .git/git-crypt/keys/default in the canonical -> bail immediately.
    original_run_agent = manager.executor.run_agent
    calls: list[list[str]] = []

    def spy_run_agent(args, **kwargs):
        calls.append(list(args))
        return original_run_agent(args, **kwargs)

    monkeypatch.setattr(manager.executor, "run_agent", spy_run_agent)

    manager._unlock_git_crypt(ctx, mirror_path)

    assert not any(c[:1] == ["git-crypt"] for c in calls), (
        f"Should not invoke git-crypt when canonical has no key; got: {calls}"
    )
    assert not any(c[:1] == ["git"] and "status" in c for c in calls), (
        f"Should not even probe `git status` when canonical has no key; got: {calls}"
    )


def test_ensure_clone_skips_lfs_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ensure_clone sets GIT_LFS_SKIP_SMUDGE=1 when skip_lfs is True (default)."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    original_run_agent = manager.executor.run_agent
    clone_envs = []

    def spy_run_agent(args, **kwargs):
        args_list = list(args)
        if "clone" in args_list:
            clone_envs.append(kwargs.get("env"))
        return original_run_agent(args, **kwargs)

    monkeypatch.setattr(manager.executor, "run_agent", spy_run_agent)
    manager.ensure_clone(ctx)  # skip_lfs=True by default

    assert clone_envs, "Expected a clone call"
    assert clone_envs[0] == {"GIT_LFS_SKIP_SMUDGE": "1"}


def test_ensure_clone_allows_lfs_when_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ensure_clone does NOT set GIT_LFS_SKIP_SMUDGE when skip_lfs=False."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    original_run_agent = manager.executor.run_agent
    clone_envs = []

    def spy_run_agent(args, **kwargs):
        args_list = list(args)
        if "clone" in args_list:
            clone_envs.append(kwargs.get("env"))
        return original_run_agent(args, **kwargs)

    monkeypatch.setattr(manager.executor, "run_agent", spy_run_agent)
    manager.ensure_clone(ctx, skip_lfs=False)

    assert clone_envs, "Expected a clone call"
    assert clone_envs[0] is None


def test_context_for_unknown_mirror(tmp_path: Path) -> None:
    manager = build_manager(tmp_path)
    with pytest.raises(MirrorError):
        manager.context_for("missing")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Test Task", "test-task"),
        ("UPPER_case-123", "upper-case-123"),
        ("weird@@@name", "weird-name"),
    ],
)
def test_sanitize_task_name(raw: str, expected: str) -> None:
    assert _sanitize_task_name(raw) == expected


def test_sanitize_task_name_rejects_empty() -> None:
    with pytest.raises(MirrorError):
        _sanitize_task_name("$$$")


def test_prepare_canonical_adjusts_permissions(tmp_path: Path) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    canonical = ctx.canonical_path
    # Restore owner-write so prepare_canonical can create scripts/ etc.
    canonical.chmod(canonical.stat().st_mode | 0o200)
    git_dir = canonical / ".git"
    # Make directories and files group-writable first.
    git_dir.chmod(0o770)
    (git_dir / "HEAD").chmod(0o660)
    (canonical / "README.md").chmod(0o660)

    manager.prepare_canonical(ctx, use_sudo=False)

    git_mode = git_dir.stat().st_mode & 0o777
    git_head_mode = (git_dir / "HEAD").stat().st_mode & 0o777
    readme_mode = (canonical / "README.md").stat().st_mode & 0o777

    assert git_mode & 0o20 == 0  # no group write on .git directory
    assert git_mode & 0o10  # execute retained for directories
    assert git_head_mode & 0o20 == 0  # no group write on git metadata
    assert git_head_mode & 0o40  # group read retained
    # Canonical (upstream) should be read-only for agent (g-w)
    # Mirror (downstream) is where agent writes (g+w, handled by ensure_clone)
    assert readme_mode & 0o20 == 0  # no group write on working tree files
    assert readme_mode & 0o40  # group read retained

    remote_url = (
        run_git(["remote", "get-url", ctx.agent_prefix], canonical).stdout.strip()
    )
    assert remote_url == str(ctx.mirror_path)

    fetch_specs = (
        run_git(
            ["config", "--get-all", f"remote.{ctx.agent_prefix}.fetch"], canonical
        ).stdout.splitlines()
    )
    expected_spec = (
        f"+refs/heads/{ctx.agent_prefix}/*:refs/remotes/{ctx.agent_prefix}/{ctx.agent_prefix}/*"
    )
    assert expected_spec in fetch_specs

    helper_script = canonical / "scripts" / "fetch-agent-branches.sh"
    assert helper_script.exists()
    assert os.access(helper_script, os.X_OK)


def _build_nested_manager(tmp_path: Path, parent_mode: int) -> tuple:
    """Build a MirrorManager whose canonical repo sits under a restrictive parent."""
    deep = tmp_path / "deep"
    deep.mkdir()
    canonical = deep / "canonical"
    canonical.mkdir()
    create_canonical_repo(canonical)

    # Strip owner-write to satisfy _validate_canonical (same as build_manager)
    canonical.chmod(canonical.stat().st_mode & ~0o200)

    # Make the parent directory restrictive *after* repo creation
    deep.chmod(parent_mode)

    os.environ["GIT_CONFIG_GLOBAL"] = str(tmp_path / "gitconfig")
    mirror_root = tmp_path / "mirrors"
    mirror_root.mkdir()

    user = pwd.getpwuid(os.getuid()).pw_name
    group = grp.getgrgid(os.getgid()).gr_name
    settings = MirrorSettings(
        name="nested",
        canonical_repo=canonical,
        mirror_name="nested",
        branch_prefixes=BranchPrefixes(human="ligon", agent="coder"),
        default_base_branch="main",
        task_branch_prefix="task",
    )
    config = Config(
        human_user=user,
        agent_user=user,
        agent_group=group,
        mirror_root=mirror_root,
        log_dir=None,
        mirrors={"nested": settings},
    )
    log = logging.getLogger("sucoder.test_nested")
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        log.addHandler(logging.NullHandler())
    executor = CommandExecutor(
        human_user=config.human_user,
        agent_user=config.agent_user,
        agent_group=config.agent_group,
        logger=log,
        dry_run=False,
        use_sudo_for_agent=False,
    )
    manager = MirrorManager(config, executor, log)
    ctx = manager.context_for("nested")
    return manager, ctx, deep


def test_check_parent_traversable_detects_blocking(tmp_path: Path) -> None:
    """check_parent_traversable flags dirs the agent cannot traverse."""
    deep = tmp_path / "deep" / "repo"
    deep.mkdir(parents=True)
    (tmp_path / "deep").chmod(0o700)  # owner-only, no o+x

    # Use a non-matching user/group so owner/group bits don't help.
    blocking = check_parent_traversable(deep, agent_user="nobody", agent_group="nogroup")
    blocked_names = [p.name for p in blocking]
    assert "deep" in blocked_names

    # Cleanup so pytest can remove tmp_path.
    (tmp_path / "deep").chmod(0o755)


def test_check_parent_traversable_owner_access(tmp_path: Path) -> None:
    """check_parent_traversable allows dirs where agent is the owner with u+x."""
    deep = tmp_path / "deep" / "repo"
    deep.mkdir(parents=True)
    (tmp_path / "deep").chmod(0o700)  # owner-only

    user = pwd.getpwuid(os.getuid()).pw_name
    blocking = check_parent_traversable(deep, agent_user=user, agent_group="nogroup")
    blocked_names = [p.name for p in blocking]
    assert "deep" not in blocked_names

    (tmp_path / "deep").chmod(0o755)


def test_check_parent_traversable_group_access(tmp_path: Path) -> None:
    """check_parent_traversable allows dirs where agent group has g+x."""
    deep = tmp_path / "deep" / "repo"
    deep.mkdir(parents=True)
    (tmp_path / "deep").chmod(0o710)  # owner + group-execute

    group = grp.getgrgid(os.getgid()).gr_name
    blocking = check_parent_traversable(deep, agent_user="nobody", agent_group=group)
    blocked_names = [p.name for p in blocking]
    assert "deep" not in blocked_names

    (tmp_path / "deep").chmod(0o755)


def test_prepare_canonical_passes_with_accessible_parents(tmp_path: Path) -> None:
    """prepare_canonical succeeds when parent dirs have o+x."""
    manager, ctx, deep = _build_nested_manager(tmp_path, parent_mode=0o711)
    canonical = ctx.canonical_path
    canonical.chmod(canonical.stat().st_mode | 0o200)

    # Should not raise
    manager.prepare_canonical(ctx, use_sudo=False)


def test_prepare_canonical_skips_untracked_files(tmp_path: Path) -> None:
    """Untracked files (and dirs only containing untracked files) are not chgrp'd.

    Regression test for the case where ``sucoder collaborate`` runs against a
    working tree that holds untracked clutter the human user can't chgrp
    (caches owned by other users, lockfiles, etc.). The recursive chgrp used
    to abort the whole bootstrap; now we only touch git-tracked paths.
    """
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    canonical = ctx.canonical_path

    canonical.chmod(canonical.stat().st_mode | 0o200)

    # Sentinel untracked tree: a directory full of files that should be
    # ignored by prepare_canonical entirely.
    untracked_dir = canonical / ".aws" / "cli" / "cache"
    untracked_dir.mkdir(parents=True)
    untracked_file = untracked_dir / "session.db-wal"
    untracked_file.write_text("not git's business", encoding="utf-8")

    captured: list[list[str]] = []
    original_run_human = manager.executor.run_human

    def spy(args, **kwargs):
        captured.append(list(args))
        return original_run_human(args, **kwargs)

    manager.executor.run_human = spy  # type: ignore[assignment]
    try:
        manager.prepare_canonical(ctx, use_sudo=False)
    finally:
        manager.executor.run_human = original_run_human  # type: ignore[assignment]

    # Reconstruct the list of paths handed to chgrp/chmod against the working
    # tree (any call where one of the args lives under canonical).
    touched_paths: set[str] = set()
    for args in captured:
        if not args:
            continue
        cmd = args[0]
        if cmd not in {"chgrp", "chmod", "find"}:
            continue
        for token in args[1:]:
            if token.startswith(str(canonical)):
                touched_paths.add(token)

    untracked_str = str(untracked_file)
    untracked_dir_str = str(untracked_dir)
    aws_dir_str = str(canonical / ".aws")

    # The untracked file and the dirs that exist only to hold it must not
    # appear as arguments to chgrp/chmod.
    assert untracked_str not in touched_paths
    assert untracked_dir_str not in touched_paths
    assert aws_dir_str not in touched_paths

    # Sanity: the tracked README and canonical itself *are* touched.
    assert str(canonical) in touched_paths
    assert str(canonical / "README.md") in touched_paths


def test_prepare_canonical_skips_tracked_symlinks(tmp_path: Path) -> None:
    """Tracked symlinks (including dangling ones) are not passed to chgrp/chmod.

    Regression test for the case where ``sucoder collaborate`` runs against
    a mirror like ``~/.Misc`` that tracks symlinks pointing at files which
    don't exist on this host (e.g. ``Organization/are212.org``). Passing
    them directly to ``chgrp`` makes the system call dereference the link
    and abort with "cannot dereference: No such file or directory".
    """
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    canonical = ctx.canonical_path
    canonical.chmod(canonical.stat().st_mode | 0o200)

    # Two tracked symlinks: one well-formed, one dangling. Both must be
    # excluded from chgrp/chmod args regardless of target validity.
    target = canonical / "real_target.txt"
    target.write_text("contents\n", encoding="utf-8")
    good_link = canonical / "live_link.org"
    good_link.symlink_to(target)
    bad_link = canonical / "dangling_link.org"
    bad_link.symlink_to(canonical / "does_not_exist.org")
    run_git(["add", "real_target.txt", "live_link.org", "dangling_link.org"], canonical)
    run_git(["commit", "-m", "add symlinks"], canonical)

    captured: list[list[str]] = []
    original_run_human = manager.executor.run_human

    def spy(args, **kwargs):
        captured.append(list(args))
        return original_run_human(args, **kwargs)

    manager.executor.run_human = spy  # type: ignore[assignment]
    try:
        manager.prepare_canonical(ctx, use_sudo=False)
    finally:
        manager.executor.run_human = original_run_human  # type: ignore[assignment]

    touched_paths: set[str] = set()
    for args in captured:
        if not args or args[0] not in {"chgrp", "chmod", "find"}:
            continue
        for token in args[1:]:
            if token.startswith(str(canonical)):
                touched_paths.add(token)

    # Neither symlink may appear as an arg to chgrp/chmod.
    assert str(good_link) not in touched_paths
    assert str(bad_link) not in touched_paths
    # But the symlink target (a regular tracked file) is touched.
    assert str(target) in touched_paths


def test_prepare_canonical_skips_unowned_tracked_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Tracked files owned by another uid are skipped (warn, don't crash) when --no-sudo.

    Regression test for the case where ``sucoder collaborate`` hits a tracked
    file installed by another user (e.g. a packaged shim like
    ``~/.Misc/bin/evince``). Without ``--sudo``, ``chgrp`` returns EPERM on
    those entries and the single non-zero exit aborts the whole batch.
    """
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    canonical = ctx.canonical_path
    canonical.chmod(canonical.stat().st_mode | 0o200)

    foreign = canonical / "bin" / "evince"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("#!/bin/sh\n", encoding="utf-8")
    run_git(["add", "bin/evince"], canonical)
    run_git(["commit", "-m", "add foreign tool"], canonical)

    # Pretend `bin/evince` is owned by a different uid. lstat() returns
    # an os.stat_result; we wrap it to override st_uid for this one path.
    real_lstat = os.lstat
    foreign_resolved = str(foreign)

    class _FakeStat:
        def __init__(self, base, fake_uid):
            self._base = base
            self.st_uid = fake_uid
        def __getattr__(self, name):
            return getattr(self._base, name)

    def fake_lstat(p):
        st = real_lstat(p)
        if str(p) == foreign_resolved:
            return _FakeStat(st, os.geteuid() + 9999)
        return st

    monkeypatch.setattr(mirror.os, "lstat", fake_lstat)

    captured: list[list[str]] = []
    original_run_human = manager.executor.run_human

    def spy(args, **kwargs):
        captured.append(list(args))
        return original_run_human(args, **kwargs)

    manager.executor.run_human = spy  # type: ignore[assignment]
    try:
        with caplog.at_level(logging.WARNING, logger=manager.logger.name):
            manager.prepare_canonical(ctx, use_sudo=False)
    finally:
        manager.executor.run_human = original_run_human  # type: ignore[assignment]

    touched_paths: set[str] = set()
    for args in captured:
        if not args or args[0] not in {"chgrp", "chmod", "find"}:
            continue
        for token in args[1:]:
            if token.startswith(str(canonical)):
                touched_paths.add(token)

    # Foreign file must not be chgrp/chmod'd.
    assert foreign_resolved not in touched_paths
    # README is owned by us → still touched.
    assert str(canonical / "README.md") in touched_paths
    # User got a warning naming the skipped path.
    warning_text = " ".join(rec.getMessage() for rec in caplog.records)
    assert "evince" in warning_text
    assert "--sudo" in warning_text or "chown" in warning_text


def test_validate_canonical_rejects_writable(tmp_path: Path) -> None:
    """_validate_canonical raises MirrorError when the agent can write to canonical."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    # Restore owner-write so `test -w` succeeds (in tests agent == owner).
    ctx.canonical_path.chmod(ctx.canonical_path.stat().st_mode | 0o200)

    with pytest.raises(MirrorError, match="writable by agent user"):
        manager._validate_canonical(ctx)


def test_validate_canonical_passes_when_not_writable(tmp_path: Path) -> None:
    """_validate_canonical succeeds when canonical is not writable by agent."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    # In tests agent_user == current user (owner), so strip all write bits
    # to make `test -w` fail.
    canonical = ctx.canonical_path
    canonical.chmod(canonical.stat().st_mode & ~0o222)
    try:
        # Should not raise.
        manager._validate_canonical(ctx)
    finally:
        # Restore write so tmp_path cleanup succeeds.
        canonical.chmod(canonical.stat().st_mode | 0o700)


def test_clone_succeeds_when_global_safe_directory_update_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    original_run_agent = manager.executor.run_agent

    def run_agent_with_failure(args, **kwargs):
        if list(args)[:4] == ["git", "config", "--global", "--add"]:
            result = CommandResult(
                requested_args=list(args),
                executed_args=list(args),
                stdout="",
                stderr="lock failed",
                returncode=1,
            )
            raise CommandError("git config --global failed", result)
        return original_run_agent(args, **kwargs)

    monkeypatch.setattr(manager.executor, "run_agent", run_agent_with_failure)

    manager.ensure_clone(ctx)
    assert (ctx.mirror_path / ".git").exists()


def test_launch_agent_uses_configured_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    launcher = manager.config.mirrors["sample"].agent_launcher
    launcher.command = ["echo", "hello"]
    launcher.env = {"FOO": "BAR"}

    recorded = {}

    def fake_run_agent(args, **kwargs):
        recorded["args"] = list(args)
        recorded["kwargs"] = kwargs
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        MirrorManager,
        "_default_system_prompt_path",
        staticmethod(lambda: Path("/nonexistent-system-prompt")),
    )
    manager.config.system_prompt = None

    manager.launch_agent(ctx, sync=False, extra_args=["--flag"])

    assert recorded["args"] == ["echo", "hello", "--flag"]
    assert recorded["kwargs"]["cwd"] == str(ctx.mirror_path)
    assert recorded["kwargs"]["env"] == {"FOO": "BAR"}
    assert recorded["kwargs"]["capture_output"] is False
    assert recorded["kwargs"]["check"] is False


def test_launch_agent_supports_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    launcher = manager.config.mirrors["sample"].agent_launcher
    launcher.env = {"FROM_CONFIG": "1"}

    recorded = {}

    def fake_run_agent(args, **kwargs):
        recorded["args"] = list(args)
        recorded["kwargs"] = kwargs
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        MirrorManager,
        "_compose_context_prelude",
        lambda self, ctx: "INLINE CONTEXT",
    )

    manager.launch_agent(
        ctx,
        sync=False,
        command_override=["foo", "--flag"],
        env_override={"EXTRA": "yes"},
        supports_inline_prompt=False,
    )

    assert recorded["args"] == ["foo", "--flag"]
    assert recorded["kwargs"]["env"] == {"FROM_CONFIG": "1", "EXTRA": "yes"}

    # Precede ensures inline prompt not appended when explicitly disabled.
    assert "INLINE CONTEXT" not in recorded["args"]


def test_launch_agent_remote_wraps_without_scancel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the remote tmux wrapper must NOT auto-cancel SLURM.

    Previously ``; scancel $JOB_ID`` was appended to the wrapped agent
    command.  That turned any agent exit (including a transient `!ls`
    shell-out that crashes claude) into a catastrophic teardown: the
    SLURM allocation was released, the tmux session died, and the user
    was thrown back to their laptop with no way to reattach.

    The wrapper must end in ``; exec bash -l`` instead, so the tmux
    window survives the agent exit and the user can reattach via
    ``sucoder attach`` and inspect / restart claude.
    """
    from sucoder.config import RemoteConfig, SlurmConfig

    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    # Promote the mirror to remote (so the tmux-wrapping branch fires).
    ctx.settings.remote = RemoteConfig(
        gateway="brc.berkeley.edu",
        transfer_host="dtn.brc.berkeley.edu",
        slurm=SlurmConfig(partition="savio3", account="fc_jevons"),
    )

    # Pretend the executor has a SLURM job id assigned (this is what
    # _ensure_slurm_node sets on RemoteExecutor at runtime).  Previously
    # the wrapper used this to emit `; scancel <job_id>` — that's the
    # regression we're guarding.
    manager.executor.slurm_job_id = 1234567  # type: ignore[attr-defined]

    recorded = {}

    def fake_run_agent(args, **kwargs):
        recorded["args"] = list(args)
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        MirrorManager,
        "_default_system_prompt_path",
        staticmethod(lambda: Path("/nonexistent-system-prompt")),
    )
    manager.config.system_prompt = None

    manager.launch_agent(ctx, sync=False)

    args = recorded["args"]
    # Must be wrapped in tmux new-session -A.
    assert args[0] == "tmux"
    assert "new-session" in args
    assert "-A" in args
    # Last arg is the in-tmux command string.
    cmd_str = args[-1]
    # MUST NOT auto-cancel SLURM (the regression we're preventing).
    assert "scancel" not in cmd_str, (
        "Agent wrapper still contains scancel — that auto-cancel "
        "destroys the SLURM allocation on any agent exit (including "
        f"crash or shell-out). Full wrapper: {cmd_str!r}"
    )
    # MUST end with `exec bash -l` so the tmux window survives.
    assert cmd_str.rstrip().endswith("exec bash -l"), (
        "Agent wrapper must append `; exec bash -l` so the tmux "
        "window stays alive after claude exits and the user can "
        f"reattach.  Full wrapper: {cmd_str!r}"
    )


def test_build_remote_agent_cmd_str_joins_and_appends_exec_bash(tmp_path: Path) -> None:
    """The extracted helper joins the command and appends ``; exec bash -l``.

    With no prelude to externalize it must not touch the executor and must
    produce exactly the string that tmux runs as its window command -- the
    contract the confined (sbatch) path also depends on.
    """
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    out = manager._build_remote_agent_cmd_str(
        ctx,
        ["claude", "--foo", "a b"],
        remote_prelude_text=None,
        prelude_sentinel="__UNUSED__",
    )

    assert out == "claude --foo 'a b'; exec bash -l"


def test_build_remote_agent_cmd_str_externalizes_prelude(tmp_path: Path) -> None:
    """When a prelude is supplied the helper routes it through
    ``_externalize_prelude`` (file write + ``$(cat ...)`` substitution)."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    recorded: dict = {}

    def spy(args, *, input=None, **kwargs):
        recorded["args"] = list(args)
        recorded["input"] = input

    manager.executor.run_agent = spy  # type: ignore[assignment]

    sentinel = "__SUCODER_PRELUDE_FROM_FILE__"
    out = manager._build_remote_agent_cmd_str(
        ctx,
        ["claude", "--system-prompt", sentinel],
        remote_prelude_text="SYSTEM PROMPT\n" + ("x" * 4000),
        prelude_sentinel=sentinel,
    )

    # Prelude moved off the command line; cat-ref substituted; still wrapped.
    assert sentinel not in out
    assert "SYSTEM PROMPT" not in out
    assert "$(cat " in out
    assert out.rstrip().endswith("exec bash -l")
    # The prelude went over stdin, never into argv.
    assert recorded["input"].startswith("SYSTEM PROMPT")


def test_launch_agent_confined_delegates_to_launch_confined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``confined`` target routes through ``_launch_confined`` (sbatch).

    It must NOT fall through to the non-confined branch, which would wrap
    the agent in a ``tmux new-session`` on the *login* node (unconfined --
    the very thing confinement avoids).  We assert the delegation happens
    with the launch context threaded through, and that no bare ``tmux``
    launch is sent to the executor.
    """
    from sucoder.config import RemoteConfig, SlurmConfig

    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    ctx.settings.remote = RemoteConfig(
        gateway="brc.berkeley.edu",
        transfer_host="dtn.brc.berkeley.edu",
        slurm=SlurmConfig(
            partition="savio4_htc", account="co_carleton", confined=True,
        ),
    )
    assert ctx.confined is True

    # Record any executor command so we can assert the unconfined tmux
    # launch never fires; stub the agent system prompt away.
    calls: list = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        MirrorManager,
        "_default_system_prompt_path",
        staticmethod(lambda: Path("/nonexistent-system-prompt")),
    )
    manager.config.system_prompt = None

    captured: dict = {}

    def fake_launch_confined(self, ctx_arg, command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(MirrorManager, "_launch_confined", fake_launch_confined)

    rc = manager.launch_agent(ctx, sync=False, detached=True)

    assert rc == 0
    assert captured, "confined launch_agent did not delegate to _launch_confined"
    # The launch context is threaded through for the sbatch flow.
    assert captured["kwargs"]["detached"] is True
    assert "prelude_sentinel" in captured["kwargs"]
    assert "remote_prelude_text" in captured["kwargs"]
    assert "env" in captured["kwargs"]
    # The *agent* command (not a tmux wrapper) is what reaches the sbatch
    # builder -- a regression that pre-wraps it in tmux before delegating
    # would be caught here.
    assert captured["command"], "no command threaded to _launch_confined"
    assert captured["command"][0] != "tmux", captured["command"]
    # No bare login-node tmux launch was sent to the executor.
    assert not any(args and args[0] == "tmux" for args in calls), (
        f"Confined launch must not run tmux on the login node: {calls!r}"
    )


# ----------------------------------------------------------------------
# _launch_confined — the sbatch submit/poll/persist/attach orchestration
# ----------------------------------------------------------------------

def _confined_manager(tmp_path, monkeypatch, *, target_name=None):
    """A manager wired for confined-launch unit tests.

    Stubs the session directory and the remote-path/home resolvers so the
    tests exercise ``_launch_confined``'s orchestration (submit/poll/persist/
    attach), not path resolution or SSH.  Returns ``(manager, ctx)``.
    """
    import sucoder.session as session_mod
    from sucoder.config import RemoteConfig, SlurmConfig

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(session_mod, "_session_dir", lambda: sessions)
    monkeypatch.setattr(mirror.time, "sleep", lambda *_a, **_k: None)

    manager = build_manager(tmp_path)
    manager.target_name = target_name
    ctx = manager.context_for("sample")
    ctx.settings.remote = RemoteConfig(
        gateway="brc.berkeley.edu",
        transfer_host="dtn.brc.berkeley.edu",
        mirror_root=Path("~/mirrors"),
        slurm=SlurmConfig(
            partition="savio4_htc", account="co_carleton",
            qos="carleton_htc4_normal", cpus_per_task=4, mem="16G",
            confined=True,
        ),
    )
    monkeypatch.setattr(
        MirrorManager, "_resolve_remote_home",
        lambda self, ctx: "/global/home/users/coder",
    )
    monkeypatch.setattr(
        MirrorManager, "_resolve_remote_path",
        lambda self, ctx: "/global/home/users/coder/mirrors/sample",
    )
    return manager, ctx


def _confined_responder(calls, *, sbatch_out="12345", live_state="",
                        poll_states=None, ready_rc=0, attach_rc=0,
                        sacct_state="FAILED"):
    """run_agent stub dispatching on the confined command shapes."""
    poll_iter = iter(poll_states if poll_states is not None else ["RUNNING n0001.savio4"])

    def run_agent(args, *, check=True, capture_output=True, input=None, **kwargs):
        a = list(args)
        calls.append({"args": a, "input": input, "check": check,
                      "capture_output": capture_output})

        def res(stdout="", stderr="", rc=0):
            if check and rc != 0:
                raise CommandError(
                    f"stub command failed: {a}",
                    CommandResult(requested_args=a, executed_args=a,
                                  stdout=stdout, stderr=stderr, returncode=rc),
                )
            return CommandResult(requested_args=a, executed_args=a,
                                 stdout=stdout, stderr=stderr, returncode=rc)

        if a[:3] == ["bash", "-c", "echo $HOME"]:
            return res("/global/home/users/coder")
        if a[0] == "sh":
            return res("")
        if a[0] == "sbatch":
            return res(sbatch_out)
        if a[0] == "squeue":
            # Dispatch on the actual -o format value (not position), and
            # assert the argv shape, so a drift in the squeue command fails
            # LOUDLY here instead of silently mis-routing to a green pass.
            assert a[1] == "--job" and "--noheader" in a, f"squeue argv: {a}"
            fmt = a[a.index("-o") + 1] if "-o" in a else None
            if fmt == "%T %N":            # node-poll
                nxt = next(poll_iter)
                return res(nxt) if isinstance(nxt, str) else nxt(res)
            if fmt == "%T":               # live/state probe
                if isinstance(live_state, tuple):
                    stdout, stderr, rc = live_state
                    return res(stdout, stderr, rc)
                return res(live_state)
            raise AssertionError(f"unexpected squeue -o {fmt!r}: {a}")
        if a[0] == "srun":
            if "has-session" in a:
                return res("", rc=ready_rc)
            if "attach-session" in a:
                return res("", rc=attach_rc)
            if "capture-pane" in a:
                return res("agent pane output")
        if a[0] == "sacct":
            return res(sacct_state)
        return res("")

    return run_agent


def test_parse_sbatch_job_id_defensive():
    p = MirrorManager._parse_sbatch_job_id
    assert p("12345", ["sbatch"]) == 12345
    assert p("12345\n", ["sbatch"]) == 12345
    # Federated cluster: "<id>;<cluster>".
    assert p("12345;brc", ["sbatch"]) == 12345
    # MOTD / warning lines precede the id.
    assert p("Note: maintenance Sunday\n12345", ["sbatch"]) == 12345
    # Unparseable -> raise (the job may be queued; must not be swallowed).
    with pytest.raises(MirrorError, match="could not be parsed"):
        p("sbatch: error: nonsense", ["sbatch"])


def test_confined_job_state_distinguishes_failure_from_gone(tmp_path, monkeypatch):
    manager, _ = _confined_manager(tmp_path, monkeypatch)

    def stub(state):
        calls = []
        manager.executor.run_agent = _confined_responder(calls, live_state=state)
        return manager._confined_job_state(999)

    # squeue lists ONLY active jobs, so ANY non-empty state is live -- the
    # method returns the raw state word (not a hard-coded allow-list, which
    # would miss SUSPENDED/PREEMPTED and resubmit over a live job).
    assert stub("RUNNING") == "RUNNING"
    assert stub("PENDING") == "PENDING"
    assert stub("SUSPENDED") == "SUSPENDED"
    assert stub("") is None                        # ok + empty = gone
    assert stub(("", "Invalid job id specified", 1)) is None  # gone
    # A genuine probe failure must RAISE (never silently "gone" -> resubmit).
    with pytest.raises(MirrorError, match="Could not verify"):
        stub(("", "ssh: connect timeout", 255))


def test_poll_confined_node_three_way(tmp_path, monkeypatch):
    manager, _ = _confined_manager(tmp_path, monkeypatch)

    # PENDING (empty %N) then RUNNING with a node.
    calls = []
    manager.executor.run_agent = _confined_responder(
        calls, poll_states=["PENDING ", "RUNNING n0007.savio4"],
    )
    assert manager._poll_confined_node(42, attempts=5, delay=0) == "n0007.savio4"

    # Empty squeue output == job left the queue == terminal -> raise, with
    # the sacct state surfaced (not mislabeled PENDING).
    calls = []
    manager.executor.run_agent = _confined_responder(
        calls, poll_states=[""], sacct_state="FAILED",
    )
    with pytest.raises(MirrorError, match="ended before") as exc:
        manager._poll_confined_node(42, attempts=5, delay=0)
    assert "FAILED" in str(exc.value)

    # Never persists the literal state word as a node: PENDING forever -> raise
    # tagged PENDING, not a bogus node.
    calls = []
    manager.executor.run_agent = _confined_responder(
        calls, poll_states=["PENDING ", "PENDING ", "PENDING "],
    )
    with pytest.raises(MirrorError, match="still PENDING"):
        manager._poll_confined_node(42, attempts=3, delay=0)


def test_launch_confined_submits_persists_and_attaches(tmp_path, monkeypatch):
    from sucoder.session import RemoteSession

    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    calls = []
    manager.executor.run_agent = _confined_responder(
        calls, sbatch_out="55501", poll_states=["RUNNING n0009.savio4"],
    )

    rc = manager._launch_confined(
        ctx, ["claude", "--dangerously"], remote_prelude_text=None,
        prelude_sentinel="__X__", env=None, detached=False,
    )
    assert rc == 0

    kinds = [c["args"][0] for c in calls]
    assert "sbatch" in kinds
    assert "squeue" in kinds
    # Interactive launch attaches via srun --pty (no TTY capture).
    attach = [c for c in calls if c["args"][0] == "srun" and "attach-session" in c["args"]]
    assert attach, "interactive confined launch must attach"
    assert attach[0]["capture_output"] is False
    assert "--overlap" in attach[0]["args"] and "--pty" in attach[0]["args"]

    # Job id + node persisted to the session.
    sess = RemoteSession.load("sample", target_name=None)
    assert sess.slurm_job_id == 55501
    assert sess.compute_node == "n0009.savio4"


def test_launch_confined_detached_does_not_attach(tmp_path, monkeypatch):
    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    calls = []
    manager.executor.run_agent = _confined_responder(calls, sbatch_out="777")

    rc = manager._launch_confined(
        ctx, ["claude"], remote_prelude_text=None, prelude_sentinel="__X__",
        env=None, detached=True,
    )
    assert rc == 0
    assert not any(
        c["args"][0] == "srun" and "attach-session" in c["args"] for c in calls
    ), "a detached (renew) relaunch must NOT attach a terminal"


def test_launch_confined_persists_job_id_before_poll(tmp_path, monkeypatch):
    """Leak-safety: the id is persisted BEFORE the poll, so a poll failure
    leaves the job recorded (recoverable) rather than orphaned."""
    from sucoder.session import RemoteSession

    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    calls = []
    manager.executor.run_agent = _confined_responder(calls, sbatch_out="6789")
    # Poll raises (as if PENDING-timeout / probe failure).
    monkeypatch.setattr(
        MirrorManager, "_poll_confined_node",
        lambda self, job_id, **kw: (_ for _ in ()).throw(MirrorError("boom")),
    )

    with pytest.raises(MirrorError, match="boom"):
        manager._launch_confined(
            ctx, ["claude"], remote_prelude_text=None,
            prelude_sentinel="__X__", env=None, detached=False,
        )

    sess = RemoteSession.load("sample", target_name=None)
    assert sess.slurm_job_id == 6789, "job id must be persisted before the poll"
    # ...and the failure path leaves it RECORDED, never scancelled (recoverable).
    assert sess.compute_node is None
    assert not any(c["args"][0] == "scancel" for c in calls), (
        "the launch path must never scancel on a poll failure"
    )


def test_launch_confined_reuse_probe_skips_resubmit(tmp_path, monkeypatch):
    from sucoder.session import RemoteSession

    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    # Pre-seed a live job.
    sess = RemoteSession.load("sample", target_name=None)
    sess.slurm_job_id = 4242
    sess.save()

    calls = []
    manager.executor.run_agent = _confined_responder(calls, live_state="RUNNING")

    rc = manager._launch_confined(
        ctx, ["claude"], remote_prelude_text=None, prelude_sentinel="__X__",
        env=None, detached=False,
    )
    assert rc == 0
    assert not any(c["args"][0] == "sbatch" for c in calls), (
        "must not submit a second job while one is live"
    )
    # It attaches to the EXISTING job.
    attach = [c for c in calls if c["args"][0] == "srun" and "attach-session" in c["args"]]
    assert attach and "--jobid=4242" in attach[0]["args"]


def test_launch_confined_probe_failure_does_not_resubmit(tmp_path, monkeypatch):
    from sucoder.session import RemoteSession

    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    sess = RemoteSession.load("sample", target_name=None)
    sess.slurm_job_id = 4242
    sess.save()

    calls = []
    # squeue probe fails (SSH blip): _confined_job_live must raise, and
    # _launch_confined must NOT fall through to a resubmit.
    manager.executor.run_agent = _confined_responder(
        calls, live_state=("", "ssh: connect timeout", 255),
    )

    with pytest.raises(MirrorError, match="Could not verify"):
        manager._launch_confined(
            ctx, ["claude"], remote_prelude_text=None,
            prelude_sentinel="__X__", env=None, detached=False,
        )
    assert not any(c["args"][0] == "sbatch" for c in calls), (
        "a probe failure must not trigger a resubmit (could orphan a live job)"
    )


def test_launch_confined_reuse_detached_returns_without_attach(tmp_path, monkeypatch):
    """A detached (renew) relaunch that finds a live job returns 0 WITHOUT
    resubmitting or attaching."""
    from sucoder.session import RemoteSession

    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    sess = RemoteSession.load("sample", target_name=None)
    sess.slurm_job_id = 4242
    sess.save()

    calls = []
    manager.executor.run_agent = _confined_responder(calls, live_state="RUNNING")

    rc = manager._launch_confined(
        ctx, ["claude"], remote_prelude_text=None, prelude_sentinel="__X__",
        env=None, detached=True,
    )
    assert rc == 0
    assert not any(c["args"][0] == "sbatch" for c in calls)
    assert not any(
        c["args"][0] == "srun" and "attach-session" in c["args"] for c in calls
    )


def test_launch_confined_reuse_pending_does_not_attach(tmp_path, monkeypatch):
    """Reusing a still-PENDING job interactively must NOT srun-attach (which
    would block until it starts) -- surface and bail instead."""
    from sucoder.session import RemoteSession

    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    sess = RemoteSession.load("sample", target_name=None)
    sess.slurm_job_id = 4242
    sess.save()

    calls = []
    manager.executor.run_agent = _confined_responder(calls, live_state="PENDING")

    with pytest.raises(MirrorError, match="queued, not yet running"):
        manager._launch_confined(
            ctx, ["claude"], remote_prelude_text=None,
            prelude_sentinel="__X__", env=None, detached=False,
        )
    assert not any(c["args"][0] == "sbatch" for c in calls)
    assert not any(
        c["args"][0] == "srun" and "attach-session" in c["args"] for c in calls
    )


def test_launch_confined_reuse_session_not_ready_still_attaches(tmp_path, monkeypatch):
    """Reusing a RUNNING job whose tmux session isn't up warns but still
    attaches (unlike a fresh launch, which raises) -- the agent may simply
    have exited, leaving the keeper's login shell."""
    from sucoder.session import RemoteSession

    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    sess = RemoteSession.load("sample", target_name=None)
    sess.slurm_job_id = 4242
    sess.save()

    calls = []
    manager.executor.run_agent = _confined_responder(
        calls, live_state="RUNNING", ready_rc=1, attach_rc=0,
    )

    rc = manager._launch_confined(
        ctx, ["claude"], remote_prelude_text=None, prelude_sentinel="__X__",
        env=None, detached=False,
    )
    assert rc == 0
    attach = [c for c in calls if c["args"][0] == "srun" and "attach-session" in c["args"]]
    assert attach and "--jobid=4242" in attach[0]["args"]


def test_launch_confined_save_failure_surfaces_scancel_hint(tmp_path, monkeypatch, caplog):
    """If persisting the just-submitted job id fails, the error log must name
    the job and the `scancel` recovery command (a job submitted but never
    recorded is a leak), and the error must propagate."""
    import logging
    from sucoder.session import RemoteSession

    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    calls = []
    manager.executor.run_agent = _confined_responder(calls, sbatch_out="8899")

    def boom(self):
        raise OSError("disk full")
    monkeypatch.setattr(RemoteSession, "save", boom)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(OSError):
            manager._launch_confined(
                ctx, ["claude"], remote_prelude_text=None,
                prelude_sentinel="__X__", env=None, detached=False,
            )
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "8899" in blob and "scancel 8899" in blob


def test_resolve_remote_home_resolves_and_caches(tmp_path):
    """`_resolve_remote_home` resolves $HOME via the executor, caches it, and
    raises on an empty result (a bad absolute path would be worse silent)."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    calls = []

    def stub(args, **kw):
        calls.append(list(args))
        return CommandResult(
            requested_args=list(args), executed_args=list(args),
            stdout="/global/home/users/coder\n", stderr="", returncode=0,
        )
    manager.executor.run_agent = stub

    assert manager._resolve_remote_home(ctx) == "/global/home/users/coder"
    # Cached: a second call does not re-probe.
    assert manager._resolve_remote_home(ctx) == "/global/home/users/coder"
    assert len(calls) == 1

    # Empty $HOME must raise rather than yield a bogus `None/.cache/...` path.
    (tmp_path / "m2").mkdir()
    manager2 = build_manager(tmp_path / "m2")
    ctx2 = manager2.context_for("sample")
    manager2.executor.run_agent = lambda args, **kw: CommandResult(
        requested_args=list(args), executed_args=list(args),
        stdout="\n", stderr="", returncode=0,
    )
    with pytest.raises(MirrorError, match="resolve the remote"):
        manager2._resolve_remote_home(ctx2)


def test_launch_confined_wraps_agent_in_bash_lc(tmp_path, monkeypatch):
    """sbatch drops the login env, so the agent is wrapped in `bash -lc` to
    resolve PATH/nvm; the staged batch script must contain it, and the
    agent_launcher.env must be threaded through (sbatch drops env too)."""
    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    calls = []
    manager.executor.run_agent = _confined_responder(calls, sbatch_out="9")

    manager._launch_confined(
        ctx, ["claude", "--foo"], remote_prelude_text=None,
        prelude_sentinel="__X__", env={"SUCODER_TOK": "abc 123"}, detached=True,
    )
    writes = [c for c in calls if c["args"][0] == "sh" and c["input"]]
    assert writes, "batch script must be staged to NFS via the executor"
    script = writes[0]["input"]
    assert "bash -lc " in script
    assert "claude --foo" in script
    # env=env is actually threaded into _build_batch_script (a dropped
    # `env=env` would silently regress env-carry to nothing).  The export is
    # inside the shlex-quoted window command, so assert the threaded var name
    # + value text appear (execution is covered by test_batch_script.py).
    assert "export SUCODER_TOK=" in script and "abc 123" in script
    # Dedicated socket + sanitized session name in the staged script.
    assert "tmux -L sucoder-sample new-session -A -d -s sucoder-sample" in script


def test_launch_confined_session_not_ready_surfaces_log(tmp_path, monkeypatch):
    """If the job is RUNNING but the tmux session never came up, fail loudly
    (with the job-log pointer) instead of attaching into nothing."""
    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    calls = []
    manager.executor.run_agent = _confined_responder(
        calls, sbatch_out="13", poll_states=["RUNNING n0001.savio4"], ready_rc=1,
    )
    with pytest.raises(MirrorError, match="did not come up") as exc:
        manager._launch_confined(
            ctx, ["claude"], remote_prelude_text=None,
            prelude_sentinel="__X__", env=None, detached=False,
        )
    msg = str(exc.value)
    # The agent's pane (its stderr lives there, not on the batch stdout) and
    # the job-log pointer are surfaced for diagnosis.
    assert "agent pane output" in msg
    assert "job-sample-13.out" in msg
    # No attach attempted when the session is not ready.
    assert not any(
        c["args"][0] == "srun" and "attach-session" in c["args"] for c in calls
    )


def test_launch_agent_raises_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    def fail_run_agent(args, **kwargs):
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="boom",
            returncode=7,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fail_run_agent)
    monkeypatch.setattr(
        MirrorManager,
        "_default_system_prompt_path",
        staticmethod(lambda: Path("/nonexistent-system-prompt")),
    )
    manager.config.system_prompt = None

    with pytest.raises(MirrorError):
        manager.launch_agent(ctx, sync=False)


def test_start_task_raises_with_missing_base(tmp_path: Path) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    ctx.settings.default_base_branch = "missing"

    with pytest.raises(MirrorError):
        manager.start_task(ctx, task_name="demo")


def test_launch_agent_reads_skills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    skill_file = skills_dir / "SKILL.org"
    skill_file.write_text(
        "#+TITLE: Demo Skill\n#+DESCRIPTION: Helpful instructions\nBody\n",
        encoding="utf-8",
    )
    ctx.settings.skills = [skills_dir]

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append({"args": list(args), "kwargs": kwargs})
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        MirrorManager,
        "_default_system_prompt_path",
        staticmethod(lambda: Path("/nonexistent-system-prompt")),
    )
    manager.config.system_prompt = None

    manager.launch_agent(ctx, sync=False)

    assert calls
    args = calls[0]["args"]
    assert args[0] == "claude"
    # Claude profile uses --dangerously-skip-permissions
    assert "--dangerously-skip-permissions" in args
    prelude = _extract_prelude(args)
    assert "SKILL" in prelude
    assert "Demo Skill" in prelude
    assert "Helpful instructions" in prelude


def test_launch_agent_runs_poetry_install_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    pyproject = ctx.mirror_path / "pyproject.toml"
    pyproject.write_text("[tool.poetry]\nname = \"demo\"\n", encoding="utf-8")

    prefs = WorkspacePrefs.load(ctx.mirror_path)
    prefs.set_poetry_auto_install(True)
    prefs.save()

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    manager.launch_agent(ctx, sync=False)

    assert len(calls) >= 2
    assert calls[0][:3] == ["poetry", "lock", "--no-update"]
    assert calls[1][:2] == ["poetry", "install"]


def test_launch_agent_prompts_and_records_preference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    decisions = []

    def prompt_handler(message: str) -> bool:
        decisions.append(message)
        return True

    manager = build_manager(tmp_path, prompt_handler=prompt_handler)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    (ctx.mirror_path / "pyproject.toml").write_text(
        "[tool.poetry]\nname = \"demo\"\n",
        encoding="utf-8",
    )

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    manager.launch_agent(ctx, sync=False)

    assert decisions
    prefs = WorkspacePrefs.load(ctx.mirror_path)
    assert prefs.poetry_auto_install() is True
    assert calls[0][:3] == ["poetry", "lock", "--no-update"]
    assert calls[1][:2] == ["poetry", "install"]


def test_poetry_install_python_mismatch_disables_auto_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    (ctx.mirror_path / "pyproject.toml").write_text(
        "[tool.poetry]\nname = \"demo\"\n",
        encoding="utf-8",
    )

    prefs = WorkspacePrefs.load(ctx.mirror_path)
    prefs.set_poetry_auto_install(True)
    prefs.save()

    agent_calls = []

    def fake_run_agent(args, **kwargs):
        if list(args)[:2] == ["poetry", "install"]:
            result = CommandResult(
                requested_args=list(args),
                executed_args=list(args),
                stdout=(
                    "The currently activated Python version 3.11.2 is not supported by the project (>=3.13,<4.0).\n"
                    "Poetry was unable to find a compatible version.\n"
                ),
                stderr="",
                returncode=1,
            )
            raise CommandError("poetry failed", result)
        agent_calls.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    with caplog.at_level(logging.WARNING):
        manager.launch_agent(ctx, sync=False)

    prefs = WorkspacePrefs.load(ctx.mirror_path)
    assert prefs.poetry_auto_install() is False
    non_poetry = [c for c in agent_calls if c[0] != "poetry"]
    assert non_poetry, "Agent command should still execute after poetry failure."
    assert non_poetry[0][0] == "claude"
    assert any("Poetry auto-install disabled" in message for message in caplog.messages)


def test_launch_agent_skips_poetry_install_when_declined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def prompt_handler(message: str) -> bool:
        return False

    manager = build_manager(tmp_path, prompt_handler=prompt_handler)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    (ctx.mirror_path / "pyproject.toml").write_text(
        "[tool.poetry]\nname = \"demo\"\n",
        encoding="utf-8",
    )

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    manager.launch_agent(ctx, sync=False)

    prefs = WorkspacePrefs.load(ctx.mirror_path)
    assert prefs.poetry_auto_install() is False
    assert all(call[0] != "poetry" for call in calls)


def test_poetry_lock_no_update_failure_falls_back_to_plain_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``poetry lock --no-update`` fails (old Poetry), fall back to ``poetry lock``."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    (ctx.mirror_path / "pyproject.toml").write_text(
        "[tool.poetry]\nname = \"demo\"\n",
        encoding="utf-8",
    )

    prefs = WorkspacePrefs.load(ctx.mirror_path)
    prefs.set_poetry_auto_install(True)
    prefs.save()

    calls = []

    def fake_run_agent(args, **kwargs):
        args_list = list(args)
        calls.append(args_list)
        if args_list[:3] == ["poetry", "lock", "--no-update"]:
            return CommandResult(
                requested_args=args_list,
                executed_args=args_list,
                stdout="",
                stderr='The option "--no-update" does not exist',
                returncode=1,
            )
        return CommandResult(
            requested_args=args_list,
            executed_args=args_list,
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    manager.launch_agent(ctx, sync=False)

    assert calls[0][:3] == ["poetry", "lock", "--no-update"]
    assert calls[1] == ["poetry", "lock"]
    assert calls[2][:2] == ["poetry", "install"]


def test_poetry_lock_total_failure_still_attempts_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When both lock variants fail, we still attempt ``poetry install``."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    (ctx.mirror_path / "pyproject.toml").write_text(
        "[tool.poetry]\nname = \"demo\"\n",
        encoding="utf-8",
    )

    prefs = WorkspacePrefs.load(ctx.mirror_path)
    prefs.set_poetry_auto_install(True)
    prefs.save()

    calls = []

    def fake_run_agent(args, **kwargs):
        args_list = list(args)
        calls.append(args_list)
        if args_list[:3] == ["poetry", "lock", "--no-update"]:
            # --no-update uses check=False, so return a failed result
            return CommandResult(
                requested_args=args_list,
                executed_args=args_list,
                stdout="",
                stderr="lock failed",
                returncode=1,
            )
        if args_list == ["poetry", "lock"]:
            # plain lock uses check=True, so raise
            result = CommandResult(
                requested_args=args_list,
                executed_args=args_list,
                stdout="",
                stderr="lock failed",
                returncode=1,
            )
            raise CommandError("poetry lock failed", result)
        return CommandResult(
            requested_args=args_list,
            executed_args=args_list,
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    manager.launch_agent(ctx, sync=False)

    assert calls[0][:3] == ["poetry", "lock", "--no-update"]
    assert calls[1] == ["poetry", "lock"]
    assert calls[2][:2] == ["poetry", "install"]


def test_auto_commit_agent_skills_after_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After a subprocess-mode session, changes in agent skills dir are auto-committed."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    # Set up a fake skills dir with git tracking.
    skills_dir = tmp_path / "agent_skills"
    skills_dir.mkdir()
    run_git(["init", "-b", "main"], skills_dir)
    run_git(["config", "user.email", "test@example.com"], skills_dir)
    run_git(["config", "user.name", "Test"], skills_dir)
    (skills_dir / "initial.md").write_text("# Skill\n", encoding="utf-8")
    run_git(["add", "-A"], skills_dir)
    run_git(["commit", "-m", "initial"], skills_dir)

    # Point the manager at our test skills dir.
    monkeypatch.setattr(type(manager), "_agent_skills_dir", property(lambda self: skills_dir))

    # Simulate agent writing a new skill file.
    (skills_dir / "new-skill.md").write_text("# New\n", encoding="utf-8")

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="?? new-skill.md\n" if _git_is(args, "status") else "",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    manager.launch_agent(ctx, sync=False)

    # Verify git add + commit happened for skills.
    git_calls = [c for c in calls if c[0] == "git"]
    status_calls = [c for c in git_calls if c[1] == "status"]
    add_calls = [c for c in git_calls if c[1] == "add"]
    commit_calls = [c for c in git_calls if c[1] == "commit"]
    assert status_calls, "Should check skills dir for changes"
    assert add_calls, "Should stage skill changes"
    assert commit_calls, "Should commit skill changes"
    assert any("Auto-snapshot" in str(c) for c in commit_calls)


def test_launch_agent_fires_auto_audit_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``audit.auto_after_session`` is True, ``launch_agent`` calls
    ``_maybe_run_audit`` after the auto-commit step.

    This is the integration check that the hook is wired into the
    session-teardown path.  The unit tests for ``_maybe_run_audit``
    itself live elsewhere (see "Post-session auto-audit hook"
    section).
    """
    manager = build_manager(tmp_path)
    object.__setattr__(
        manager.config, "audit",
        AuditConfig(auto_after_session=True, scope="all"),
    )
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    # Stub run_agent so the test doesn't require a real agent CLI.
    def fake_run_agent(args, **kwargs):
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )
    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    # Capture the call to _maybe_run_audit instead of letting it run.
    invocations: list = []
    def fake_maybe(ctx_arg):
        invocations.append(ctx_arg.settings.name)
    monkeypatch.setattr(manager, "_maybe_run_audit", fake_maybe)

    manager.launch_agent(ctx, sync=False)

    assert invocations == ["sample"], (
        "_maybe_run_audit must be called exactly once after a session, "
        f"with the session's mirror context (got {invocations!r})"
    )


def test_launch_agent_no_audit_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default config (auto_after_session=False) → hook is still called,
    but it short-circuits to a no-op without touching the audit functions.

    We exercise the *real* ``_maybe_run_audit`` here (no monkeypatch)
    to confirm the opt-in default genuinely produces zero side effects.
    """
    manager = build_manager(tmp_path)
    # Default Config().audit has auto_after_session=False; no need to set.
    assert manager.config.audit.auto_after_session is False

    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    def fake_run_agent(args, **kwargs):
        return CommandResult(
            requested_args=list(args), executed_args=list(args),
            stdout="", stderr="", returncode=0,
        )
    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    audit_called: list = []
    monkeypatch.setattr(
        manager, "audit_agent_skills",
        lambda **kw: audit_called.append("skills") or "",
    )
    monkeypatch.setattr(
        manager, "audit_code_changes",
        lambda mn, **kw: audit_called.append("code") or "",
    )

    manager.launch_agent(ctx, sync=False)

    assert audit_called == [], (
        "Default opt-out must not invoke any audit function; "
        f"got {audit_called!r}"
    )


def test_auto_commit_skipped_when_no_skills_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When agent skills dir does not exist, auto-commit is silently skipped."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    # Point at a nonexistent dir.
    monkeypatch.setattr(type(manager), "_agent_skills_dir", property(lambda self, p=tmp_path / "nonexistent": p))

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    manager.launch_agent(ctx, sync=False)

    # No git status/add/commit calls for skills — only the agent launch.
    git_status_calls = [c for c in calls if c[:2] == ["git", "status"] and "--porcelain" in c]
    assert not git_status_calls


def test_auto_commit_skipped_when_no_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When skills dir has no changes, no commit is created."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    # Set up a tracked skills dir.
    skills_dir = tmp_path / "agent_skills"
    skills_dir.mkdir()
    run_git(["init", "-b", "main"], skills_dir)
    run_git(["config", "user.email", "test@example.com"], skills_dir)
    run_git(["config", "user.name", "Test"], skills_dir)
    (skills_dir / "existing.md").write_text("# Skill\n", encoding="utf-8")
    run_git(["add", "-A"], skills_dir)
    run_git(["commit", "-m", "initial"], skills_dir)

    monkeypatch.setattr(type(manager), "_agent_skills_dir", property(lambda self: skills_dir))

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="" if _git_is(args, "status") else "",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    manager.launch_agent(ctx, sync=False)

    # Should check status but not commit.
    commit_calls = [c for c in calls if c[0] == "git" and c[1] == "commit"]
    assert not commit_calls


def test_skills_repo_initialized_on_first_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If skills dir exists but is not a git repo, it gets initialized."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    # Untracked skills dir.
    skills_dir = tmp_path / "agent_skills"
    skills_dir.mkdir()
    (skills_dir / "my-skill.md").write_text("# Skill\n", encoding="utf-8")

    monkeypatch.setattr(type(manager), "_agent_skills_dir", property(lambda self: skills_dir))

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="" if _git_is(args, "status") else "",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    manager.launch_agent(ctx, sync=False)

    # Should have git init + initial add + initial commit.
    init_calls = [c for c in calls if c[:2] == ["git", "init"]]
    assert init_calls, "Should initialize git repo"
    initial_commits = [c for c in calls if c[0] == "git" and c[1] == "commit" and "Initial" in str(c)]
    assert initial_commits, "Should create initial snapshot commit"


def _make_skills_dir(tmp_path: Path) -> Path:
    """Create a git-tracked skills directory with world-readable perms."""
    skills_dir = tmp_path / "agent_skills"
    skills_dir.mkdir(mode=0o755)
    run_git(["init", "-b", "main"], skills_dir)
    run_git(["config", "user.email", "test@example.com"], skills_dir)
    run_git(["config", "user.name", "Test"], skills_dir)
    return skills_dir


def _write_skill(skills_dir: Path, name: str, content: str) -> Path:
    path = skills_dir / name
    path.write_text(content, encoding="utf-8")
    path.chmod(0o644)  # Ensure o+r for audit checks.
    return path


def test_audit_full_review_when_no_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full review triggers when no refs/audited baseline exists."""
    manager = build_manager(tmp_path)

    skills_dir = _make_skills_dir(tmp_path)
    _write_skill(skills_dir, "skill.md", "# A Skill\nDo stuff.\n")
    run_git(["add", "-A"], skills_dir)
    run_git(["commit", "-m", "initial"], skills_dir)

    monkeypatch.setattr(type(manager), "_agent_skills_dir", property(lambda self: skills_dir))

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        if _git_is(args, "rev-parse", "--verify"):
            # No baseline ref exists.
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="", stderr="", returncode=1,
            )
        if args[0] == "claude" and "-p" in args:
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="No concerns.", stderr="", returncode=0,
            )
        return CommandResult(
            requested_args=list(args), executed_args=list(args),
            stdout="", stderr="", returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    report = manager.audit_agent_skills()

    assert report is not None
    assert "No concerns" in report
    # Should have invoked claude -p.
    claude_calls = [c for c in calls if c[0] == "claude"]
    assert claude_calls
    assert "-p" in claude_calls[0]


def test_audit_diff_review_with_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Diff review triggers when refs/audited baseline exists."""
    manager = build_manager(tmp_path)

    skills_dir = _make_skills_dir(tmp_path)
    _write_skill(skills_dir, "skill.md", "# Skill\n")
    run_git(["add", "-A"], skills_dir)
    run_git(["commit", "-m", "initial"], skills_dir)

    monkeypatch.setattr(type(manager), "_agent_skills_dir", property(lambda self: skills_dir))

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        if _git_is(args, "rev-parse", "--verify"):
            # Baseline exists.
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="abc1234\n", stderr="", returncode=0,
            )
        if _git_is(args, "diff"):
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="+new line in skill\n", stderr="", returncode=0,
            )
        if args[0] == "claude" and "-p" in args:
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="No concerns.", stderr="", returncode=0,
            )
        return CommandResult(
            requested_args=list(args), executed_args=list(args),
            stdout="", stderr="", returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    report = manager.audit_agent_skills()

    assert report is not None
    assert "No concerns" in report


def test_audit_returns_none_when_no_skills_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit returns None when skills dir does not exist."""
    manager = build_manager(tmp_path)
    monkeypatch.setattr(type(manager), "_agent_skills_dir", property(lambda self, p=tmp_path / "nonexistent": p))

    report = manager.audit_agent_skills()
    assert report is None


def test_audit_flags_unreadable_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit flags tracked files the auditor cannot read."""
    manager = build_manager(tmp_path)

    skills_dir = _make_skills_dir(tmp_path)
    hidden = _write_skill(skills_dir, "hidden.md", "# Secret\n")
    run_git(["add", "-A"], skills_dir)
    run_git(["commit", "-m", "initial"], skills_dir)
    # Mode 000 means even the file owner is denied by `[ -r ]`, so the
    # readability test fails regardless of which user the test runs as.
    hidden.chmod(0o000)

    monkeypatch.setattr(type(manager), "_agent_skills_dir", property(lambda self: skills_dir))

    try:
        report = manager.audit_agent_skills()
    finally:
        # Restore perms so pytest's tmp_path cleanup can remove the file.
        hidden.chmod(0o644)

    assert report is not None
    assert "PERMISSIONS AUDIT FAILURE" in report
    assert "hidden.md" in report


def test_audit_skips_gitignored_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit ignores virtualenvs / caches / build output (not git-tracked)."""
    manager = build_manager(tmp_path)

    skills_dir = _make_skills_dir(tmp_path)
    # Gitignore .venv (matches reality — virtualenvs are never tracked).
    (skills_dir / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    # A perfectly normal tracked file.
    _write_skill(skills_dir, "skill.md", "# Skill\n")
    run_git(["add", "-A"], skills_dir)
    run_git(["commit", "-m", "initial"], skills_dir)
    # An untracked, gitignored virtualenv-shaped file with restrictive
    # perms — exactly the false-positive class that flooded the audit
    # before the fix.
    venv_dir = skills_dir / ".venv" / "lib"
    venv_dir.mkdir(parents=True)
    venv_file = venv_dir / "blob.so"
    venv_file.write_bytes(b"\x7fELF...")
    venv_file.chmod(0o000)

    monkeypatch.setattr(type(manager), "_agent_skills_dir", property(lambda self: skills_dir))

    # Stub out the LLM call so the audit doesn't try to invoke claude.
    def fake_run_agent(args, **kwargs):
        if args[:1] == ["claude"]:
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="No concerns.", stderr="", returncode=0,
            )
        # Fall through to the real executor for git/sh — needed for the
        # readability check.
        return original_run_agent(args, **kwargs)

    original_run_agent = manager.executor.run_agent
    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    try:
        report = manager.audit_agent_skills(full=True)
    finally:
        venv_file.chmod(0o644)

    # We should NOT have flagged the .venv blob.
    assert report is None or "PERMISSIONS AUDIT FAILURE" not in report
    if report is not None:
        assert "blob.so" not in report


def test_audit_skips_git_crypt_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit hard-skips .git-crypt/keys/ even though those files are tracked."""
    manager = build_manager(tmp_path)

    skills_dir = _make_skills_dir(tmp_path)
    _write_skill(skills_dir, "skill.md", "# Skill\n")
    keys_dir = skills_dir / ".git-crypt" / "keys" / "default" / "0"
    keys_dir.mkdir(parents=True)
    keyfile = keys_dir / "ABCDEF.gpg"
    keyfile.write_bytes(b"encrypted-key-bytes")
    run_git(["add", "-A"], skills_dir)
    run_git(["commit", "-m", "initial"], skills_dir)
    # Realistic perms for a git-crypt key after commit — secret, denies
    # even the owner so the readability test would flag it without the
    # _AUDIT_SKIP_PREFIXES exclusion.
    keyfile.chmod(0o000)

    monkeypatch.setattr(type(manager), "_agent_skills_dir", property(lambda self: skills_dir))

    def fake_run_agent(args, **kwargs):
        if args[:1] == ["claude"]:
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="No concerns.", stderr="", returncode=0,
            )
        return original_run_agent(args, **kwargs)

    original_run_agent = manager.executor.run_agent
    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    try:
        report = manager.audit_agent_skills(full=True)
    finally:
        keyfile.chmod(0o644)

    assert report is None or "PERMISSIONS AUDIT FAILURE" not in report
    if report is not None:
        assert "ABCDEF.gpg" not in report


def test_audit_returns_none_when_no_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit returns None when baseline exists and there are no changes."""
    manager = build_manager(tmp_path)

    skills_dir = _make_skills_dir(tmp_path)
    _write_skill(skills_dir, "skill.md", "# Skill\n")
    run_git(["add", "-A"], skills_dir)
    run_git(["commit", "-m", "initial"], skills_dir)

    monkeypatch.setattr(type(manager), "_agent_skills_dir", property(lambda self: skills_dir))

    def fake_run_agent(args, **kwargs):
        if _git_is(args, "rev-parse", "--verify"):
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="abc1234\n", stderr="", returncode=0,
            )
        if _git_is(args, "diff"):
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="", stderr="", returncode=0,
            )
        return CommandResult(
            requested_args=list(args), executed_args=list(args),
            stdout="", stderr="", returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    report = manager.audit_agent_skills()
    assert report is None


# -- Code audit tests ---------------------------------------------------


def _setup_code_audit_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MirrorManager:
    """Build a MirrorManager with a cloned 'sample' mirror ready for code audit."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)
    # Make mirror world-readable so the permissions check passes.
    mirror = ctx.mirror_path
    for p in mirror.rglob("*"):
        if p.is_file():
            p.chmod(p.stat().st_mode | 0o004)
        elif p.is_dir():
            p.chmod(p.stat().st_mode | 0o005)
    # Neutralize skills dir so auto-commit doesn't interfere.
    monkeypatch.setattr(type(manager), "_agent_skills_dir", property(lambda self, p=tmp_path / "no_skills": p))
    return manager


def test_code_audit_full_review_when_no_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full code review triggers when no refs/audited-code baseline exists."""
    manager = _setup_code_audit_manager(tmp_path, monkeypatch)

    calls: list = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        if _git_is(args, "rev-parse", "--verify"):
            # No baseline ref exists.
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="", stderr="", returncode=1,
            )
        if _git_is(args, "diff"):
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="+hello code\n", stderr="", returncode=0,
            )
        if args[0] == "claude" and "-p" in args:
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="No concerns.", stderr="", returncode=0,
            )
        return CommandResult(
            requested_args=list(args), executed_args=list(args),
            stdout="", stderr="", returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    report = manager.audit_code_changes("sample")

    assert report is not None
    assert "No concerns" in report
    # Should have invoked claude -p.
    claude_calls = [c for c in calls if c[0] == "claude"]
    assert claude_calls
    assert "-p" in claude_calls[0]


def test_code_audit_diff_review_with_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Diff code review triggers when refs/audited-code baseline exists."""
    manager = _setup_code_audit_manager(tmp_path, monkeypatch)

    calls: list = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        if _git_is(args, "rev-parse", "--verify"):
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="abc1234\n", stderr="", returncode=0,
            )
        if _git_is(args, "diff"):
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="+changed code\n", stderr="", returncode=0,
            )
        if args[0] == "claude" and "-p" in args:
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="No concerns.", stderr="", returncode=0,
            )
        return CommandResult(
            requested_args=list(args), executed_args=list(args),
            stdout="", stderr="", returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    report = manager.audit_code_changes("sample")

    assert report is not None
    assert "No concerns" in report
    # Diff calls should reference refs/audited-code.
    diff_calls = [c for c in calls if _git_is(c, "diff")]
    assert any("refs/audited-code" in c for c in diff_calls)


def test_code_audit_returns_none_when_no_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Code audit returns None when the mirror is not a git repo."""
    manager = build_manager(tmp_path)
    monkeypatch.setattr(type(manager), "_agent_skills_dir", property(lambda self, p=tmp_path / "no_skills": p))
    # Don't clone — the mirror path won't be a git repo.
    report = manager.audit_code_changes("sample")
    assert report is None


def test_code_audit_returns_none_when_no_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Code audit returns None when baseline exists and there are no changes."""
    manager = _setup_code_audit_manager(tmp_path, monkeypatch)

    def fake_run_agent(args, **kwargs):
        if _git_is(args, "rev-parse", "--verify"):
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="abc1234\n", stderr="", returncode=0,
            )
        if _git_is(args, "diff"):
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="", stderr="", returncode=0,
            )
        return CommandResult(
            requested_args=list(args), executed_args=list(args),
            stdout="", stderr="", returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    report = manager.audit_code_changes("sample")
    assert report is None


def test_code_audit_flags_unreadable_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Code audit flags tracked files the auditor cannot read."""
    manager = _setup_code_audit_manager(tmp_path, monkeypatch)

    ctx = manager.context_for("sample")
    secret = ctx.mirror_path / "secret.py"
    secret.write_text("API_KEY = 'hunter2'\n", encoding="utf-8")
    # Track the file — the audit only looks at git-tracked content now.
    run_git(["config", "user.email", "test@example.com"], ctx.mirror_path)
    run_git(["config", "user.name", "Test"], ctx.mirror_path)
    run_git(["add", "secret.py"], ctx.mirror_path)
    run_git(["commit", "-m", "add secret"], ctx.mirror_path)
    # Mode 000 — denies even the owner, so the readability test fails
    # regardless of which user runs the test.
    secret.chmod(0o000)

    try:
        report = manager.audit_code_changes("sample")
    finally:
        secret.chmod(0o644)

    assert report is not None
    assert "PERMISSIONS AUDIT FAILURE" in report
    assert "secret.py" in report


def test_advance_audited_code_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """advance_audited_code_ref calls git update-ref with the right ref."""
    manager = _setup_code_audit_manager(tmp_path, monkeypatch)

    calls: list = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        return CommandResult(
            requested_args=list(args), executed_args=list(args),
            stdout="", stderr="", returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    manager.advance_audited_code_ref("sample")

    update_ref_calls = [c for c in calls if _git_is(c, "update-ref")]
    assert update_ref_calls
    assert "refs/audited-code" in update_ref_calls[0]


def test_code_audit_full_uses_empty_tree_diff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full code audit diffs against the empty tree hash."""
    manager = _setup_code_audit_manager(tmp_path, monkeypatch)

    calls: list = []
    # The fake mktree returns this stand-in for "the empty tree".  The
    # real empty-tree SHA-1 is ``4b825dc642cb6eb9a060e54bf8d69288fbee4904``
    # but the test only checks that whatever mktree returned is what the
    # subsequent diff was passed.
    empty_tree = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        if _git_is(args, "rev-parse", "--verify"):
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="", stderr="", returncode=1,
            )
        if _git_is(args, "mktree"):
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout=empty_tree + "\n", stderr="", returncode=0,
            )
        if _git_is(args, "diff"):
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="+all the code\n", stderr="", returncode=0,
            )
        if args[0] == "claude" and "-p" in args:
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="No concerns.", stderr="", returncode=0,
            )
        return CommandResult(
            requested_args=list(args), executed_args=list(args),
            stdout="", stderr="", returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    report = manager.audit_code_changes("sample", full=True)

    assert report is not None
    diff_calls = [c for c in calls if _git_is(c, "diff")]
    assert any(empty_tree in c for c in diff_calls), f"Expected empty tree hash in diff calls: {diff_calls}"


def test_code_audit_prompt_contains_security_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Code audit prompt includes security-specific review criteria."""
    manager = _setup_code_audit_manager(tmp_path, monkeypatch)

    captured_prompts: list = []

    def fake_run_agent(args, **kwargs):
        if _git_is(args, "rev-parse", "--verify"):
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="", stderr="", returncode=1,
            )
        if _git_is(args, "diff"):
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="+code\n", stderr="", returncode=0,
            )
        if args[0] == "claude" and "-p" in args:
            idx = args.index("-p")
            captured_prompts.append(args[idx + 1])
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="No concerns.", stderr="", returncode=0,
            )
        return CommandResult(
            requested_args=list(args), executed_args=list(args),
            stdout="", stderr="", returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    manager.audit_code_changes("sample")

    assert captured_prompts
    prompt = captured_prompts[0]
    assert "Credential leakage" in prompt
    assert "Unsafe subprocess" in prompt
    assert "Supply-chain" in prompt
    assert "Permission escalation" in prompt


# -- Post-session auto-audit hook ---------------------------------------


def _make_audit_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    auto_after_session: bool,
    scope: str = "all",
) -> MirrorManager:
    """Build a manager with the auto-audit hook configured.

    Replaces the auditor-existence check (``pwd.getpwnam("auditor")``)
    with a stub that always succeeds, since CI/test environments rarely
    have a real ``auditor`` user.
    """
    manager = build_manager(tmp_path)
    # Replace the frozen audit config — Config itself is mutable, but
    # AuditConfig is frozen, so swap the whole instance.
    object.__setattr__(
        manager.config,
        "audit",
        AuditConfig(auto_after_session=auto_after_session, scope=scope),
    )
    monkeypatch.setattr(
        "sucoder.mirror.pwd.getpwnam",
        lambda name: types.SimpleNamespace(pw_name=name),
    )
    return manager


def test_maybe_run_audit_noop_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto-audit is opt-in; disabled config = no audit calls."""
    manager = _make_audit_manager(
        tmp_path, monkeypatch, auto_after_session=False,
    )
    ctx = manager.context_for("sample")

    skills_called: list = []
    code_called: list = []
    monkeypatch.setattr(
        manager, "audit_agent_skills",
        lambda **kw: skills_called.append(kw) or "",
    )
    monkeypatch.setattr(
        manager, "audit_code_changes",
        lambda mn, **kw: code_called.append((mn, kw)) or "",
    )

    manager._maybe_run_audit(ctx)

    assert skills_called == []
    assert code_called == []


def test_maybe_run_audit_silent_when_auditor_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the auditor user doesn't exist, log + skip; don't raise."""
    manager = build_manager(tmp_path)
    object.__setattr__(
        manager.config, "audit",
        AuditConfig(auto_after_session=True, scope="all"),
    )
    # Simulate missing auditor user.
    def _raise(name: str):
        raise KeyError(name)
    monkeypatch.setattr("sucoder.mirror.pwd.getpwnam", _raise)

    audit_called: list = []
    monkeypatch.setattr(
        manager, "audit_agent_skills",
        lambda **kw: audit_called.append("skills") or "",
    )
    monkeypatch.setattr(
        manager, "audit_code_changes",
        lambda mn, **kw: audit_called.append("code") or "",
    )

    ctx = manager.context_for("sample")
    with caplog.at_level("INFO", logger="sucoder.test"):
        manager._maybe_run_audit(ctx)

    assert audit_called == []
    assert any(
        "auditor" in r.message and "does not exist" in r.message
        for r in caplog.records
    )


def test_maybe_run_audit_runs_both_audits_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``scope: all`` triggers skills AND code audits and saves both reports."""
    manager = _make_audit_manager(
        tmp_path, monkeypatch, auto_after_session=True, scope="all",
    )
    object.__setattr__(
        manager.config, "log_dir", tmp_path / "logs",
    )
    ctx = manager.context_for("sample")

    monkeypatch.setattr(
        manager, "audit_agent_skills",
        lambda **kw: "No concerns.\n",
    )
    monkeypatch.setattr(
        manager, "audit_code_changes",
        lambda mn, **kw: "Found a token at line 42.\n",
    )

    manager._maybe_run_audit(ctx)

    audits_dir = tmp_path / "logs" / "audits"
    assert audits_dir.is_dir()
    written = sorted(p.name for p in audits_dir.iterdir())
    assert any(name.startswith("sample-skills-") for name in written), written
    assert any(name.startswith("sample-code-") for name in written), written


def test_maybe_run_audit_skills_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``scope: skills`` runs the skills audit, not the code audit."""
    manager = _make_audit_manager(
        tmp_path, monkeypatch, auto_after_session=True, scope="skills",
    )
    object.__setattr__(
        manager.config, "log_dir", tmp_path / "logs",
    )
    ctx = manager.context_for("sample")

    code_called: list = []
    monkeypatch.setattr(
        manager, "audit_agent_skills",
        lambda **kw: "No concerns.\n",
    )
    monkeypatch.setattr(
        manager, "audit_code_changes",
        lambda mn, **kw: code_called.append("code") or "",
    )

    manager._maybe_run_audit(ctx)

    assert code_called == []
    audits_dir = tmp_path / "logs" / "audits"
    written = sorted(p.name for p in audits_dir.iterdir())
    assert all(name.startswith("sample-skills-") for name in written)


def test_maybe_run_audit_swallows_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An audit raising must not propagate; session teardown continues."""
    manager = _make_audit_manager(
        tmp_path, monkeypatch, auto_after_session=True, scope="all",
    )
    ctx = manager.context_for("sample")

    def _boom(**kw):
        raise RuntimeError("auditor token expired")
    def _boom_code(mn, **kw):
        raise RuntimeError("disk full")
    monkeypatch.setattr(manager, "audit_agent_skills", _boom)
    monkeypatch.setattr(manager, "audit_code_changes", _boom_code)

    with caplog.at_level("WARNING", logger="sucoder.test"):
        manager._maybe_run_audit(ctx)  # must not raise

    messages = " ".join(r.message for r in caplog.records)
    assert "skills audit failed" in messages
    assert "code audit failed" in messages


def test_maybe_run_audit_skips_when_audit_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the audit returns None (nothing to audit), no file is written."""
    manager = _make_audit_manager(
        tmp_path, monkeypatch, auto_after_session=True, scope="all",
    )
    object.__setattr__(
        manager.config, "log_dir", tmp_path / "logs",
    )
    ctx = manager.context_for("sample")

    monkeypatch.setattr(manager, "audit_agent_skills", lambda **kw: None)
    monkeypatch.setattr(manager, "audit_code_changes", lambda mn, **kw: None)

    with caplog.at_level("INFO", logger="sucoder.test"):
        manager._maybe_run_audit(ctx)

    audits_dir = tmp_path / "logs" / "audits"
    assert not audits_dir.exists() or list(audits_dir.iterdir()) == []
    messages = [r.message for r in caplog.records]
    assert any("nothing to audit" in m for m in messages)


def test_maybe_run_audit_log_dir_falls_back_to_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When config.log_dir is None, reports land under ~/.sucoder/logs/audits/."""
    manager = _make_audit_manager(
        tmp_path, monkeypatch, auto_after_session=True, scope="skills",
    )
    # Force log_dir to None to exercise the fallback.
    object.__setattr__(manager.config, "log_dir", None)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    monkeypatch.setattr(manager, "audit_agent_skills", lambda **kw: "Findings\n")

    ctx = manager.context_for("sample")
    manager._maybe_run_audit(ctx)

    fallback = fake_home / ".sucoder" / "logs" / "audits"
    assert fallback.is_dir()
    assert list(fallback.iterdir())  # at least one report written


def test_save_audit_report_writes_to_disk(
    tmp_path: Path,
) -> None:
    """``_save_audit_report`` returns the path it wrote to."""
    manager = build_manager(tmp_path)
    object.__setattr__(manager.config, "log_dir", tmp_path / "logs")
    ctx = manager.context_for("sample")

    path = manager._save_audit_report(ctx, "code", "REPORT BODY")

    assert path.exists()
    assert path.read_text(encoding="utf-8") == "REPORT BODY"
    assert path.parent.name == "audits"
    assert path.name.startswith("sample-code-")


def test_launch_agent_wraps_command_with_nvm_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    nvm_dir = tmp_path / "nvm"
    nvm_dir.mkdir()
    (nvm_dir / "nvm.sh").write_text("return 0\n", encoding="utf-8")

    ctx.settings.agent_launcher = AgentLauncher(
        command=["codex"],
        env={},
        nvm=NvmConfig(version="22.11.0", dir=nvm_dir),
    )

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    manager.launch_agent(ctx, sync=False)

    assert calls
    command = calls[-1]
    assert command[0] == "bash"
    assert command[1] == "-lc"
    script = command[2]
    assert str(nvm_dir) in script
    assert "nvm use" in script and "22.11.0" in script
    assert "exec codex" in script
    # Codex profile uses --sandbox instead of --yolo
    assert "--sandbox" in script
    assert "danger-full-access" in script
    # Codex profile doesn't use --add-dir (sandbox handles permissions)


def test_launch_agent_defaults_nvm_dir_to_agent_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    ctx.settings.agent_launcher = AgentLauncher(
        command=["codex"],
        env={},
        nvm=NvmConfig(version="lts/hydrogen", dir=None),
    )

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    manager.launch_agent(ctx, sync=False)

    command = calls[-1]
    script = command[2]
    home = manager._agent_home_directory()
    if home:
        expected_dir = home / ".nvm"
        assert str(expected_dir) in script
    assert "nvm use" in script


def test_launch_agent_preserves_user_extra_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that user-provided extra_args are preserved in the command."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    recorded = []

    def fake_run_agent(args, **kwargs):
        recorded.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    manager.launch_agent(ctx, sync=False, extra_args=["--add-dir", str(Path.home()), "--foo"])

    agent_call = recorded[-1]
    # Claude profile uses --dangerously-skip-permissions
    assert "--dangerously-skip-permissions" in agent_call
    # User-provided args are preserved
    assert "--add-dir" in agent_call
    assert str(Path.home()) in agent_call
    assert "--foo" in agent_call


def test_launch_agent_respects_existing_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that sandbox flag is not duplicated if user already provides it."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    captured: list[list[str]] = []

    def fake_run_agent(args, **kwargs):
        captured.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    # User provides --sandbox flag
    manager.launch_agent(ctx, sync=False, extra_args=["--sandbox", "workspace-write", "--foo"])

    codex_call = captured[-1]
    # Should only have one --sandbox (user's, not duplicated)
    assert codex_call.count("--sandbox") == 1


def test_launch_agent_adds_skills_flag_with_default_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    default_skills = tmp_path / "default_skills"
    default_skills.mkdir()

    ctx.settings.skills = []
    ctx.settings.agent_launcher.flags.skills = "--skills {path}"

    monkeypatch.setattr(
        MirrorManager,
        "_default_skills_dir",
        staticmethod(lambda: default_skills),
    )

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        manager,
        "_default_system_prompt_path",
        lambda: Path("/nonexistent-system-prompt"),
    )
    manager.config.system_prompt = None

    manager.launch_agent(ctx, sync=False)

    assert calls
    args = calls[0]
    assert "--skills" in args
    skills_index = args.index("--skills")
    assert args[skills_index + 1] == str(default_skills)


def test_launch_agent_reads_system_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    prompt = tmp_path / "prompt.org"
    prompt.write_text("Prompt\n", encoding="utf-8")
    manager.config.system_prompt = prompt

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append({"args": list(args), "kwargs": kwargs})
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    manager.launch_agent(ctx, sync=False)

    assert calls
    args = calls[0]["args"]
    assert args[0] == "claude"
    # Claude uses --system-prompt flag; the prelude content is a separate arg
    assert "--system-prompt" in args
    sp_idx = args.index("--system-prompt")
    prelude = args[sp_idx + 1]
    assert "SYSTEM PROMPT" in prelude
    assert "Prompt" in prelude


def test_launch_agent_reads_default_system_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    default_prompt = tmp_path / "default.org"
    default_prompt.write_text("Default\n", encoding="utf-8")

    manager.config.system_prompt = None

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append({"args": list(args), "kwargs": kwargs})
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(
        MirrorManager,
        "_default_system_prompt_path",
        staticmethod(lambda: default_prompt),
    )
    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    manager.launch_agent(ctx, sync=False)

    assert calls
    args = calls[0]["args"]
    assert "--system-prompt" in args
    sp_idx = args.index("--system-prompt")
    prelude = args[sp_idx + 1]
    assert "SYSTEM PROMPT" in prelude
    assert "Default" in prelude


def test_skill_catalog_expands_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    catalog_dir = tmp_path / "skills_catalog"
    catalog_dir.mkdir()
    catalog_file = catalog_dir / "SKILLS.md"
    catalog_file.write_text(
        """#+TITLE: Catalog Skill
#+DESCRIPTION: A list of additional capabilities.
- file:detail/SKILL.org
""",
        encoding="utf-8",
    )
    detail_dir = catalog_dir / "detail"
    detail_dir.mkdir()
    detail_skill = detail_dir / "SKILL.org"
    detail_skill.write_text(
        """#+TITLE: Detail Skill
#+DESCRIPTION: Additional context.
Content body.
""",
        encoding="utf-8",
    )

    ctx.settings.skills = [catalog_dir]

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append({"args": list(args), "kwargs": kwargs})
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        manager,
        "_default_system_prompt_path",
        lambda: Path("/nonexistent-system-prompt"),
    )
    monkeypatch.setattr(manager, "_default_skills_catalog_path", lambda: None)

    manager.launch_agent(ctx, sync=False)

    args = calls[0]["args"]
    prelude = _extract_prelude(args)
    assert "SKILL CATALOG" in prelude
    assert "Catalog Skill" in prelude
    assert "Detail Skill" in prelude
    assert "load with `Read tool:" in prelude or "load with" in prelude


def test_markdown_skill_file_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    skill_dir = tmp_path / "markdown_skill"
    skill_dir.mkdir()
    skill_file = skill_dir / "Skill.md"
    skill_file.write_text(
        """---
name: Markdown Skill
description: Example markdown-based skill.
---
Body content here.
""",
        encoding="utf-8",
    )

    ctx.settings.skills = [skill_dir]

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append({"args": list(args), "kwargs": kwargs})
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        manager,
        "_default_system_prompt_path",
        lambda: Path("/nonexistent"),
    )
    monkeypatch.setattr(manager, "_default_skills_catalog_path", lambda: None)

    manager.launch_agent(ctx, sync=False)

    prelude = _extract_prelude(calls[0]["args"])
    assert "Markdown Skill" in prelude
    assert "Example markdown-based skill." in prelude
    assert "Body content here." in prelude


def test_skill_resources_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    skill_dir = tmp_path / "resource_skill"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.org"
    skill_file.write_text(
        """#+TITLE: Resource Skill
#+DESCRIPTION: Skill with bundled resources.
Instructions here.
""",
        encoding="utf-8",
    )

    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "guide.md").write_text("Reference content", encoding="utf-8")

    scripts = skill_dir / "scripts"
    scripts.mkdir()
    (scripts / "run.py").write_text("print('hi')", encoding="utf-8")

    assets = skill_dir / "assets"
    assets.mkdir()
    (assets / "logo.png").write_bytes(b"\x89PNG\r\n")

    ctx.settings.skills = [skill_dir]

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append({"args": list(args), "kwargs": kwargs})
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        manager,
        "_default_system_prompt_path",
        lambda: Path("/nonexistent"),
    )
    monkeypatch.setattr(manager, "_default_skills_catalog_path", lambda: None)

    manager.launch_agent(ctx, sync=False)

    prelude = _extract_prelude(calls[0]["args"])
    assert "Resource Skill" in prelude
    assert "RESOURCES" in prelude
    assert "references/guide.md" in prelude
    assert "scripts/run.py" in prelude
    assert "assets/logo.png" in prelude


def test_prepare_canonical_skip_agent_remote(tmp_path: Path) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    canonical = ctx.canonical_path
    manager.prepare_canonical(ctx, use_sudo=False, setup_agent_remote=False)

    remote_result = subprocess.run(
        ["git", "remote", "get-url", ctx.agent_prefix],
        cwd=canonical,
        capture_output=True,
        text=True,
    )
    assert remote_result.returncode != 0

    helper_script = canonical / "scripts" / "fetch-agent-branches.sh"
    assert not helper_script.exists()


def test_pull_from_local_fast_forwards_canonical(tmp_path: Path) -> None:
    """_pull_from_local brings agent commits in the local mirror back into canonical."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    manager.ensure_clone(ctx)
    mirror_path = ctx.mirror_path
    canonical = ctx.canonical_path

    # Make a commit on the mirror's main branch that does not exist in
    # canonical — the realistic case where the agent did some work
    # locally and we want to pull it back.
    run_git(["checkout", "main"], mirror_path)
    run_git(["config", "user.email", "agent@example.com"], mirror_path)
    run_git(["config", "user.name", "Agent"], mirror_path)
    (mirror_path / "AGENT_NOTE.md").write_text("from the agent\n", encoding="utf-8")
    run_git(["add", "AGENT_NOTE.md"], mirror_path)
    run_git(["commit", "-m", "agent note"], mirror_path)
    mirror_head = run_git(["rev-parse", "HEAD"], mirror_path).stdout.strip()

    # Canonical was made writable by ensure_clone via prepare_canonical;
    # make sure we can still update refs (git update-ref needs +w on
    # .git, which prepare_canonical preserves).
    canonical.chmod(canonical.stat().st_mode | 0o200)

    manager._pull_from_local(ctx)

    canonical_head = run_git(["rev-parse", "HEAD"], canonical).stdout.strip()
    assert canonical_head == mirror_head
    assert (canonical / "AGENT_NOTE.md").exists()


def test_pull_from_local_raises_when_mirror_missing(tmp_path: Path) -> None:
    """_pull_from_local raises a clear MirrorError when no mirror exists yet."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    # Deliberately skip ensure_clone — mirror_path does not exist.
    with pytest.raises(MirrorError, match="not a git repository"):
        manager._pull_from_local(ctx)


def test_bootstrap_invokes_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    order: list[str] = []

    def fake_prepare(c, use_sudo=False, setup_agent_remote=True):
        assert c is ctx
        order.append(f"prepare:{use_sudo}:{setup_agent_remote}")

    def fake_ensure(c, *, skip_lfs=True):
        assert c is ctx
        order.append("clone")

    def fake_launch(c, **kwargs):
        assert c is ctx
        order.append(f"launch:{kwargs}")
        return 0

    monkeypatch.setattr(manager, "prepare_canonical", fake_prepare)
    monkeypatch.setattr(manager, "ensure_clone", fake_ensure)
    monkeypatch.setattr(manager, "launch_agent", fake_launch)

    manager.bootstrap(
        ctx,
        use_sudo=True,
        setup_agent_remote=False,
        sync=False,
        task_name="demo",
        base_branch="dev",
        extra_args=["--flag"],
    )

    assert order == [
        "prepare:True:False",
        "clone",
        (
            "launch:{'sync': False, 'task_name': 'demo', 'base_branch': 'dev', "
            "'extra_args': ['--flag'], 'command_override': None, 'env_override': None, "
            "'supports_inline_prompt': None}"
        ),
    ]


# ============================================================================
# Agent Profile Tests
# ============================================================================


def test_detect_agent_type_claude() -> None:
    """Test that Claude CLI is correctly detected."""
    from sucoder.config import AgentType
    from sucoder.mirror import _detect_agent_type

    assert _detect_agent_type(["claude"]) == AgentType.CLAUDE
    assert _detect_agent_type(["claude", "--flag"]) == AgentType.CLAUDE
    assert _detect_agent_type(["/usr/bin/claude"]) == AgentType.CLAUDE


def test_detect_agent_type_codex() -> None:
    """Test that Codex CLI is correctly detected."""
    from sucoder.config import AgentType
    from sucoder.mirror import _detect_agent_type

    assert _detect_agent_type(["codex"]) == AgentType.CODEX
    assert _detect_agent_type(["codex", "--prompt", "hello"]) == AgentType.CODEX


def test_detect_agent_type_gemini() -> None:
    """Test that Gemini CLI is correctly detected."""
    from sucoder.config import AgentType
    from sucoder.mirror import _detect_agent_type

    assert _detect_agent_type(["gemini"]) == AgentType.GEMINI
    assert _detect_agent_type(["/opt/gemini"]) == AgentType.GEMINI


def test_detect_agent_type_unknown() -> None:
    """Test that unknown CLIs return UNKNOWN type."""
    from sucoder.config import AgentType
    from sucoder.mirror import _detect_agent_type

    assert _detect_agent_type([]) == AgentType.UNKNOWN
    assert _detect_agent_type(["other-cli"]) == AgentType.UNKNOWN
    assert _detect_agent_type(["my-custom-agent"]) == AgentType.UNKNOWN


def test_merge_flag_templates_precedence() -> None:
    """Test that flag template merging respects precedence order."""
    from sucoder.config import AgentFlagTemplates
    from sucoder.mirror import _merge_flag_templates

    per_mirror = AgentFlagTemplates(yolo="--per-mirror", writable_dir=None)
    global_config = AgentFlagTemplates(yolo="--global", writable_dir="--global-dir {path}")
    profile = AgentFlagTemplates(yolo="--profile", writable_dir="--profile-dir {path}", system_prompt="--sys {content}")

    merged = _merge_flag_templates(per_mirror, global_config, profile)

    # Per-mirror wins for yolo
    assert merged.yolo == "--per-mirror"
    # Global wins for writable_dir (per-mirror is None)
    assert merged.writable_dir == "--global-dir {path}"
    # Profile wins for system_prompt (per-mirror and global are None)
    assert merged.system_prompt == "--sys {content}"


def test_merge_flag_templates_without_global() -> None:
    """Test merging when global config is None."""
    from sucoder.config import AgentFlagTemplates
    from sucoder.mirror import _merge_flag_templates

    per_mirror = AgentFlagTemplates(yolo=None)
    profile = AgentFlagTemplates(yolo="--profile-yolo", system_prompt="--sys {content}")

    merged = _merge_flag_templates(per_mirror, None, profile)

    # Profile wins when per-mirror is None and no global
    assert merged.yolo == "--profile-yolo"
    assert merged.system_prompt == "--sys {content}"


def test_launch_agent_claude_uses_system_prompt_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Claude CLI uses --system-prompt flag instead of trailing text."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    # Use Claude as the command
    ctx.settings.agent_launcher = AgentLauncher(command=["claude"])

    # Set up a system prompt
    prompt_file = tmp_path / "prompt.org"
    prompt_file.write_text("Test system prompt content", encoding="utf-8")
    manager.config.system_prompt = prompt_file

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    manager.launch_agent(ctx, sync=False)

    assert calls
    cmd = calls[-1]
    assert cmd[0] == "claude"
    # Claude should use --system-prompt flag (Claude profile provides this template)
    assert "--system-prompt" in cmd
    # Find the --system-prompt value
    idx = cmd.index("--system-prompt")
    prompt_content = cmd[idx + 1]
    # The prompt content includes a header and the file content
    assert "SYSTEM PROMPT" in prompt_content
    assert "Test system prompt content" in prompt_content
    # Should NOT have prompt as trailing text (since --system-prompt is used)
    assert cmd[-1] != prompt_content  # Last arg should be something else or same as prompt via flag



def test_write_agent_fetch_helper_quotes_defaults(tmp_path: Path) -> None:
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    # Restore owner-write so the helper script directory can be created.
    ctx.canonical_path.chmod(ctx.canonical_path.stat().st_mode | 0o200)
    ctx.settings.branch_prefixes.agent = 'agent"; $(echo hacked); #'
    manager._write_agent_fetch_helper(ctx)

    script_path = ctx.canonical_path / "scripts" / "fetch-agent-branches.sh"
    contents = script_path.read_text(encoding="utf-8")

    assert "remote=${1:-'agent\"; $(echo hacked); #'}" in contents
    assert "prefix=${2:-'agent\"; $(echo hacked); #'}" in contents


def test_tokens_present_requires_all_tokens() -> None:
    command = ["codex", "--sandbox", "workspace-write"]
    tokens = ["--sandbox", "workspace-write", "--ask-for-approval", "never"]
    assert not MirrorManager._tokens_present(command, tokens)


class TrackingExecutor(CommandExecutor):
    def __init__(self, agent_user: str, log: list[str], *, human_user: str, agent_group: str) -> None:
        super().__init__(human_user=human_user, agent_user=agent_user, agent_group=agent_group, logger=logging.getLogger("test"), dry_run=False, use_sudo_for_agent=False)
        self.log = log

    def run_human(self, args, **kwargs):
        self.log.append("human:" + " ".join(args))
        return CommandResult(args, args, "", "", 0)

    def run_agent(self, args, **kwargs):
        self.log.append("agent:" + " ".join(args))
        # Simulate `test -w` failing (canonical should not be writable).
        rc = 1 if list(args)[:2] == ["test", "-w"] else 0
        return CommandResult(args, args, "", "", rc)


def test_ensure_clone_sets_parent_permissions(tmp_path: Path) -> None:
    log: list[str] = []
    user = pwd.getpwuid(os.getuid()).pw_name
    group = grp.getgrgid(os.getgid()).gr_name
    executor = TrackingExecutor(agent_user="coder", log=log, human_user=user, agent_group=group)
    manager = build_manager(tmp_path, executor=executor)
    ctx = manager.context_for("sample")

    manager.ensure_clone(ctx)

    chmod_calls = [entry for entry in log if entry.startswith("human:chmod") and "2770" in entry]
    assert chmod_calls


def test_ensure_clone_creates_claude_md_symlink(tmp_path: Path) -> None:
    """AGENT.md in the canonical repo produces a CLAUDE.md symlink in the mirror."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    # Add AGENT.md to canonical before cloning.
    canonical = tmp_path / "canonical"
    canonical.chmod(canonical.stat().st_mode | 0o200)
    (canonical / "AGENT.md").write_text("# Project instructions\n", encoding="utf-8")
    run_git(["add", "AGENT.md"], canonical)
    run_git(["commit", "-m", "add agent doc"], canonical)
    canonical.chmod(canonical.stat().st_mode & ~0o200)

    manager.ensure_clone(ctx)

    claude_md = ctx.mirror_path / "CLAUDE.md"
    assert claude_md.is_symlink()
    assert os.readlink(str(claude_md)) == "AGENT.md"
    assert claude_md.read_text(encoding="utf-8").startswith("# Project instructions")


def test_ensure_clone_creates_claude_md_symlink_for_org(tmp_path: Path) -> None:
    """AGENT.org also gets a CLAUDE.md symlink."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    canonical = tmp_path / "canonical"
    canonical.chmod(canonical.stat().st_mode | 0o200)
    (canonical / "AGENT.org").write_text("#+TITLE: Instructions\n", encoding="utf-8")
    run_git(["add", "AGENT.org"], canonical)
    run_git(["commit", "-m", "add agent doc"], canonical)
    canonical.chmod(canonical.stat().st_mode & ~0o200)

    manager.ensure_clone(ctx)

    claude_md = ctx.mirror_path / "CLAUDE.md"
    assert claude_md.is_symlink()
    assert os.readlink(str(claude_md)) == "AGENT.org"


def test_ensure_clone_skips_symlink_when_claude_md_exists(tmp_path: Path) -> None:
    """If CLAUDE.md already exists, do not create a symlink."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    canonical = tmp_path / "canonical"
    canonical.chmod(canonical.stat().st_mode | 0o200)
    (canonical / "AGENT.md").write_text("agent\n", encoding="utf-8")
    (canonical / "CLAUDE.md").write_text("claude\n", encoding="utf-8")
    run_git(["add", "AGENT.md", "CLAUDE.md"], canonical)
    run_git(["commit", "-m", "add both docs"], canonical)
    canonical.chmod(canonical.stat().st_mode & ~0o200)

    manager.ensure_clone(ctx)

    claude_md = ctx.mirror_path / "CLAUDE.md"
    assert not claude_md.is_symlink()
    assert claude_md.read_text(encoding="utf-8") == "claude\n"


def test_ensure_clone_creates_skills_symlink(tmp_path: Path) -> None:
    """.skills/ in the repo produces a .claude/skills symlink."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    canonical = tmp_path / "canonical"
    canonical.chmod(canonical.stat().st_mode | 0o200)
    skills_dir = canonical / ".skills"
    skills_dir.mkdir()
    (skills_dir / "my-skill.md").write_text("# Skill\n", encoding="utf-8")
    run_git(["add", ".skills"], canonical)
    run_git(["commit", "-m", "add skills"], canonical)
    canonical.chmod(canonical.stat().st_mode & ~0o200)

    manager.ensure_clone(ctx)

    claude_skills = ctx.mirror_path / ".claude" / "skills"
    assert claude_skills.is_symlink()
    assert os.readlink(str(claude_skills)) == "../.skills"
    assert (claude_skills / "my-skill.md").exists()


def test_ensure_clone_skips_skills_symlink_when_claude_skills_exists(tmp_path: Path) -> None:
    """If .claude/skills already exists, do not create a symlink."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    canonical = tmp_path / "canonical"
    canonical.chmod(canonical.stat().st_mode | 0o200)
    (canonical / ".skills").mkdir()
    (canonical / ".claude").mkdir()
    (canonical / ".claude" / "skills").mkdir()
    (canonical / ".claude" / "skills" / "native.md").write_text("# Native\n", encoding="utf-8")
    run_git(["add", ".skills", ".claude"], canonical)
    run_git(["commit", "-m", "add both skills dirs"], canonical)
    canonical.chmod(canonical.stat().st_mode & ~0o200)

    manager.ensure_clone(ctx)

    claude_skills = ctx.mirror_path / ".claude" / "skills"
    assert not claude_skills.is_symlink()
    assert (claude_skills / "native.md").exists()


def test_symlinks_created_on_existing_mirror(tmp_path: Path) -> None:
    """Symlinks are created on subsequent ensure_clone calls for existing mirrors."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    # Now add AGENT.md to the mirror (simulating a fetch that brought new files).
    (ctx.mirror_path / "AGENT.md").write_text("# Late addition\n", encoding="utf-8")

    # Call ensure_clone again — mirror already exists path.
    manager.ensure_clone(ctx)

    claude_md = ctx.mirror_path / "CLAUDE.md"
    assert claude_md.is_symlink()
    assert os.readlink(str(claude_md)) == "AGENT.md"


def test_agent_doc_injected_for_non_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AGENT.md content is injected into the system prompt for non-Claude agents."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    (ctx.mirror_path / "AGENT.md").write_text("Use pytest for all tests.\n", encoding="utf-8")

    # Use Codex (non-Claude) agent.
    ctx.settings.agent_launcher = AgentLauncher(command=["codex"], launch_mode="subprocess")

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    manager.launch_agent(ctx, sync=False)

    # The trailing arg is the system prompt (codex uses inline prompt).
    prelude = _extract_prelude(calls[-1])
    assert "PROJECT INSTRUCTIONS (AGENT.md)" in prelude
    assert "Use pytest for all tests." in prelude


def test_agent_doc_not_injected_for_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AGENT.md is NOT injected for Claude (it discovers via CLAUDE.md symlink)."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    (ctx.mirror_path / "AGENT.md").write_text("Use pytest for all tests.\n", encoding="utf-8")

    # Use Claude agent.
    ctx.settings.agent_launcher = AgentLauncher(command=["claude"], launch_mode="subprocess")

    prompt_file = tmp_path / "prompt.org"
    prompt_file.write_text("System prompt", encoding="utf-8")
    manager.config.system_prompt = prompt_file

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    manager.launch_agent(ctx, sync=False)

    prelude = _extract_prelude(calls[-1])
    assert "PROJECT INSTRUCTIONS" not in prelude


def test_launch_agent_gemini_uses_prompt_interactive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Gemini CLI uses --prompt-interactive for system prompt."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    # Use Gemini as the command (force subprocess mode to allow mocking)
    ctx.settings.agent_launcher = AgentLauncher(command=["gemini"], launch_mode="subprocess")

    # Set up a system prompt
    prompt_file = tmp_path / "prompt.org"
    prompt_file.write_text("Test system prompt", encoding="utf-8")
    manager.config.system_prompt = prompt_file

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    manager.launch_agent(ctx, sync=False)

    assert calls
    cmd = calls[-1]
    assert cmd[0] == "gemini"
    # Gemini uses --yolo for permissions (from profile)
    assert "--yolo" in cmd
    # Gemini uses --prompt-interactive for system prompt (stays interactive after prompt)
    assert "--prompt-interactive" in cmd
    # Find the prompt content (follows --prompt-interactive flag)
    pi_idx = cmd.index("--prompt-interactive")
    prompt_content = cmd[pi_idx + 1]
    assert "SYSTEM PROMPT" in prompt_content
    assert "Test system prompt" in prompt_content


def test_launch_mode_default_for_gemini(tmp_path: Path) -> None:
    """Test that Gemini defaults to exec launch mode."""
    from sucoder.config import DEFAULT_LAUNCH_MODES, AgentType

    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    # Gemini should default to exec mode
    ctx.settings.agent_launcher = AgentLauncher(command=["gemini"])
    effective_mode = manager._get_effective_launch_mode(["gemini"], ctx.settings.agent_launcher)
    assert effective_mode == "exec"
    assert DEFAULT_LAUNCH_MODES[AgentType.GEMINI] == "exec"


def test_launch_mode_default_for_claude(tmp_path: Path) -> None:
    """Test that Claude defaults to subprocess launch mode."""
    from sucoder.config import DEFAULT_LAUNCH_MODES, AgentType

    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    # Claude should default to subprocess mode
    ctx.settings.agent_launcher = AgentLauncher(command=["claude"])
    effective_mode = manager._get_effective_launch_mode(["claude"], ctx.settings.agent_launcher)
    assert effective_mode == "subprocess"
    assert DEFAULT_LAUNCH_MODES[AgentType.CLAUDE] == "subprocess"


def test_launch_mode_default_for_codex(tmp_path: Path) -> None:
    """Test that Codex defaults to subprocess launch mode."""
    from sucoder.config import DEFAULT_LAUNCH_MODES, AgentType

    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    # Codex should default to subprocess mode
    ctx.settings.agent_launcher = AgentLauncher(command=["codex"])
    effective_mode = manager._get_effective_launch_mode(["codex"], ctx.settings.agent_launcher)
    assert effective_mode == "subprocess"
    assert DEFAULT_LAUNCH_MODES[AgentType.CODEX] == "subprocess"


def test_launch_mode_explicit_override(tmp_path: Path) -> None:
    """Test that explicit launch_mode overrides agent default."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    # Gemini with explicit subprocess mode override
    ctx.settings.agent_launcher = AgentLauncher(command=["gemini"], launch_mode="subprocess")
    effective_mode = manager._get_effective_launch_mode(["gemini"], ctx.settings.agent_launcher)
    assert effective_mode == "subprocess"

    # Claude with explicit exec mode override
    ctx.settings.agent_launcher = AgentLauncher(command=["claude"], launch_mode="exec")
    effective_mode = manager._get_effective_launch_mode(["claude"], ctx.settings.agent_launcher)
    assert effective_mode == "exec"


def test_launch_agent_exec_mode_calls_execvp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that exec launch mode calls os.execvp."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    # Use a command that would default to exec mode
    ctx.settings.agent_launcher = AgentLauncher(command=["gemini"])

    exec_calls = []

    def fake_execvp(file, args):
        exec_calls.append((file, list(args)))
        # Raise an exception to prevent actually replacing the process
        raise SystemExit(0)

    monkeypatch.setattr("os.execvp", fake_execvp)

    # Also need to mock chdir since _exec_agent calls it
    chdir_calls = []
    original_chdir = os.chdir

    def fake_chdir(path):
        chdir_calls.append(path)

    monkeypatch.setattr("os.chdir", fake_chdir)

    with pytest.raises(SystemExit):
        manager.launch_agent(ctx, sync=False)

    # Verify execvp was called with the right command
    assert len(exec_calls) == 1
    assert exec_calls[0][0] == "bash"
    assert exec_calls[0][1][1] == "-lc"
    # Check if script contains the command and user check
    script = exec_calls[0][1][2]
    assert "gemini" in script
    assert "whoami" in script

    # Verify chdir was called to set working directory
    assert len(chdir_calls) == 1


def test_launch_agent_subprocess_mode_does_not_call_execvp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that subprocess launch mode does not call os.execvp."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    # Use subprocess mode explicitly
    ctx.settings.agent_launcher = AgentLauncher(command=["claude"], launch_mode="subprocess")

    exec_calls = []

    def fake_execvp(file, args):
        exec_calls.append((file, list(args)))
        raise SystemExit(0)

    monkeypatch.setattr("os.execvp", fake_execvp)

    # Mock run_agent to return success
    def fake_run_agent(args, **kwargs):
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    result = manager.launch_agent(ctx, sync=False)

    # execvp should NOT be called
    assert len(exec_calls) == 0
    assert result == 0
# ------------------------------------------------------------------
# Worktree inspection tests
# ------------------------------------------------------------------


PORCELAIN_SAMPLE = """\
worktree /tmp/mirrors/project
HEAD abc1234567890abcdef1234567890abcdef123456
branch refs/heads/main

worktree /tmp/mirrors/project/.claude/worktrees/fix-auth
HEAD def4567890abcdef1234567890abcdef12345678
branch refs/heads/worktree-fix-auth

worktree /tmp/mirrors/project/.claude/worktrees/detached
HEAD 9999999999abcdef1234567890abcdef12345678
detached

"""


def test_parse_worktree_porcelain_basic() -> None:
    entries = _parse_worktree_porcelain(PORCELAIN_SAMPLE)
    assert len(entries) == 3

    assert entries[0]["worktree"] == "/tmp/mirrors/project"
    assert entries[0]["branch"] == "refs/heads/main"
    assert entries[0]["HEAD"].startswith("abc1234")

    assert entries[1]["worktree"] == "/tmp/mirrors/project/.claude/worktrees/fix-auth"
    assert entries[1]["branch"] == "refs/heads/worktree-fix-auth"

    assert entries[2]["worktree"] == "/tmp/mirrors/project/.claude/worktrees/detached"
    assert "detached" in entries[2]
    assert "branch" not in entries[2]


def test_parse_worktree_porcelain_empty() -> None:
    assert _parse_worktree_porcelain("") == []
    assert _parse_worktree_porcelain("\n\n") == []


def test_list_worktrees_no_extra_worktrees(tmp_path: Path) -> None:
    """When no worktrees have been added, list_worktrees returns only the main one."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    infos = manager.list_worktrees(ctx)
    assert len(infos) == 1
    assert infos[0].is_main is True
    assert infos[0].branch == "main"


def test_list_worktrees_with_worktree(tmp_path: Path) -> None:
    """Adding a git worktree makes it visible in list_worktrees."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    mirror_path = ctx.mirror_path

    # Create a worktree manually (simulating what claude --worktree does).
    wt_dir = mirror_path / ".claude" / "worktrees" / "fix-auth"
    wt_dir.parent.mkdir(parents=True, exist_ok=True)
    run_git(
        ["worktree", "add", str(wt_dir), "-b", "worktree-fix-auth"],
        cwd=mirror_path,
    )

    # Make a commit in the worktree so it's ahead of main.
    run_git(["config", "user.email", "test@example.com"], cwd=wt_dir)
    run_git(["config", "user.name", "Test User"], cwd=wt_dir)
    (wt_dir / "new_file.txt").write_text("hello\n", encoding="utf-8")
    run_git(["add", "new_file.txt"], cwd=wt_dir)
    run_git(["commit", "-m", "worktree commit"], cwd=wt_dir)

    infos = manager.list_worktrees(ctx)
    assert len(infos) == 2

    main_info = next(i for i in infos if i.is_main)
    wt_info = next(i for i in infos if not i.is_main)

    assert main_info.branch == "main"
    assert wt_info.branch == "worktree-fix-auth"
    assert wt_info.commits_ahead >= 1
    assert wt_info.last_commit_summary == "worktree commit"
    assert wt_info.is_dirty is False


def test_list_worktrees_dirty_detection(tmp_path: Path) -> None:
    """Uncommitted changes are reported as dirty."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)
    mirror_path = ctx.mirror_path

    wt_dir = mirror_path / ".claude" / "worktrees" / "dirty-wt"
    wt_dir.parent.mkdir(parents=True, exist_ok=True)
    run_git(["worktree", "add", str(wt_dir), "-b", "dirty-branch"], cwd=mirror_path)

    # Create tracked modification and untracked file.
    (wt_dir / "README.md").write_text("modified\n", encoding="utf-8")
    (wt_dir / "untracked.txt").write_text("new\n", encoding="utf-8")

    infos = manager.list_worktrees(ctx)
    wt_info = next(i for i in infos if not i.is_main)

    assert wt_info.is_dirty is True
    assert wt_info.modified_count >= 1
    assert wt_info.untracked_count == 1


def test_worktrees_summary_no_worktrees(tmp_path: Path) -> None:
    """Summary includes 'No active worktrees' when only main exists."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    output = manager.worktrees_summary(ctx)
    assert "No active worktrees" in output
    assert "sample" in output


def test_worktrees_summary_with_worktree(tmp_path: Path) -> None:
    """Summary shows worktree details."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)
    mirror_path = ctx.mirror_path

    wt_dir = mirror_path / ".claude" / "worktrees" / "add-tests"
    wt_dir.parent.mkdir(parents=True, exist_ok=True)
    run_git(["worktree", "add", str(wt_dir), "-b", "worktree-add-tests"], cwd=mirror_path)

    output = manager.worktrees_summary(ctx)
    assert "worktree-add-tests" in output
    assert "Branch:" in output
    assert "Status:" in output


def test_worktrees_summary_with_diff(tmp_path: Path) -> None:
    """--diff populates the diff_stat field."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)
    mirror_path = ctx.mirror_path

    wt_dir = mirror_path / ".claude" / "worktrees" / "diff-wt"
    wt_dir.parent.mkdir(parents=True, exist_ok=True)
    run_git(["worktree", "add", str(wt_dir), "-b", "diff-branch"], cwd=mirror_path)

    run_git(["config", "user.email", "test@example.com"], cwd=wt_dir)
    run_git(["config", "user.name", "Test User"], cwd=wt_dir)
    (wt_dir / "new_file.txt").write_text("content\n", encoding="utf-8")
    run_git(["add", "new_file.txt"], cwd=wt_dir)
    run_git(["commit", "-m", "add file"], cwd=wt_dir)

    infos = manager.list_worktrees(ctx, include_diff=True)
    wt_info = next(i for i in infos if not i.is_main)
    assert wt_info.diff_stat is not None
    assert "new_file.txt" in wt_info.diff_stat


def test_list_worktrees_detached_head(tmp_path: Path) -> None:
    """A detached-HEAD worktree should have branch=None."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)
    mirror_path = ctx.mirror_path

    wt_dir = mirror_path / ".claude" / "worktrees" / "detached-wt"
    wt_dir.parent.mkdir(parents=True, exist_ok=True)
    run_git(["worktree", "add", "--detach", str(wt_dir)], cwd=mirror_path)

    infos = manager.list_worktrees(ctx)
    wt_info = next(i for i in infos if not i.is_main)
    assert wt_info.branch is None


def test_list_worktrees_nonexistent_base_branch(tmp_path: Path) -> None:
    """_wt_commits_ahead returns -1 when the base branch ref does not exist."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)
    mirror_path = ctx.mirror_path

    wt_dir = mirror_path / ".claude" / "worktrees" / "ahead-wt"
    wt_dir.parent.mkdir(parents=True, exist_ok=True)
    run_git(["worktree", "add", str(wt_dir), "-b", "wt-ahead"], cwd=mirror_path)

    infos = manager.list_worktrees(ctx, base_branch="nonexistent")
    wt_info = next(i for i in infos if not i.is_main)
    assert wt_info.commits_ahead == -1


def test_worktrees_summary_include_main(tmp_path: Path) -> None:
    """With include_main=True the main worktree appears in the summary."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    output = manager.worktrees_summary(ctx, include_main=True)
    # The main worktree block should contain the main branch name
    assert "main" in output
    # The "(main worktree)" label should appear
    assert "(main worktree)" in output


# ---- MCP config tests ----

import json


def test_resolve_mcp_config_returns_none_when_empty(tmp_path: Path) -> None:
    """No MCP servers configured means no config file generated."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    assert ctx.settings.mcp_servers == {}
    result = manager._resolve_mcp_config(ctx)
    assert result is None


def test_resolve_mcp_config_generates_json(tmp_path: Path) -> None:
    """MCP servers in config produce a valid .sucoder-mcp.json file."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    ctx.settings.mcp_servers = {
        "filesystem": McpServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/data"],
        ),
        "github": McpServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": "tok-123"},
        ),
    }

    result = manager._resolve_mcp_config(ctx)
    assert result is not None
    assert result.name == ".sucoder-mcp.json"
    assert result.exists()

    data = json.loads(result.read_text(encoding="utf-8"))
    assert "mcpServers" in data
    assert "filesystem" in data["mcpServers"]
    assert data["mcpServers"]["filesystem"]["command"] == "npx"
    assert data["mcpServers"]["filesystem"]["args"] == [
        "-y", "@modelcontextprotocol/server-filesystem", "/data",
    ]
    # env omitted when empty
    assert "env" not in data["mcpServers"]["filesystem"]

    assert data["mcpServers"]["github"]["env"] == {"GITHUB_TOKEN": "tok-123"}

    # Verify git exclude
    exclude_file = ctx.mirror_path / ".git" / "info" / "exclude"
    assert ".sucoder-mcp.json" in exclude_file.read_text(encoding="utf-8")


def test_resolve_mcp_config_idempotent_exclude(tmp_path: Path) -> None:
    """Calling _resolve_mcp_config twice doesn't duplicate the exclude entry."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    ctx.settings.mcp_servers = {
        "tool": McpServerConfig(command="my-tool"),
    }

    manager._resolve_mcp_config(ctx)
    manager._resolve_mcp_config(ctx)

    exclude_file = ctx.mirror_path / ".git" / "info" / "exclude"
    content = exclude_file.read_text(encoding="utf-8")
    assert content.count(".sucoder-mcp.json") == 1


def test_merge_flag_templates_includes_mcp_config() -> None:
    """mcp_config participates in the three-level merge."""
    from sucoder.config import AgentFlagTemplates
    from sucoder.mirror import _merge_flag_templates

    per_mirror = AgentFlagTemplates()
    global_config = AgentFlagTemplates()
    profile = AgentFlagTemplates(mcp_config="--mcp-config {path}")

    merged = _merge_flag_templates(per_mirror, global_config, profile)
    assert merged.mcp_config == "--mcp-config {path}"

    # Per-mirror overrides profile
    per_mirror2 = AgentFlagTemplates(mcp_config="--custom-mcp {path}")
    merged2 = _merge_flag_templates(per_mirror2, global_config, profile)
    assert merged2.mcp_config == "--custom-mcp {path}"


def test_launch_agent_includes_mcp_config_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When MCP servers are configured, --mcp-config appears in the Claude command."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    ctx.settings.agent_launcher = AgentLauncher(command=["claude"])
    ctx.settings.mcp_servers = {
        "test-tool": McpServerConfig(command="test-server", args=["--port", "8080"]),
    }

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    manager.launch_agent(ctx, sync=False)

    assert len(calls) == 1
    cmd = calls[0]
    assert "--mcp-config" in cmd
    mcp_idx = cmd.index("--mcp-config")
    mcp_path = cmd[mcp_idx + 1]
    assert mcp_path.endswith(".sucoder-mcp.json")

    # Verify the file contents
    data = json.loads(Path(mcp_path).read_text(encoding="utf-8"))
    assert data["mcpServers"]["test-tool"]["command"] == "test-server"


def test_launch_agent_no_mcp_flag_without_servers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without MCP servers, no --mcp-config flag is added."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    ctx.settings.agent_launcher = AgentLauncher(command=["claude"])

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    manager.launch_agent(ctx, sync=False)

    assert len(calls) == 1
    assert "--mcp-config" not in calls[0]


# ------------------------------------------------------------------
# Interactive helpers: _prompt_choice, _unique_branch_name
# ------------------------------------------------------------------


class TestPromptChoice:
    """Tests for MirrorManager._prompt_choice."""

    def test_returns_user_choice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _prompt: "d")
        result = MirrorManager._prompt_choice(
            "Pick one:",
            [("a", "Alpha"), ("d", "Delta")],
            default="a",
        )
        assert result == "d"

    def test_returns_default_on_empty_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _prompt: "")
        result = MirrorManager._prompt_choice(
            "Pick one:",
            [("m", "Merge"), ("n", "Abort")],
            default="m",
        )
        assert result == "m"

    def test_lowercases_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _prompt: " S ")
        result = MirrorManager._prompt_choice(
            "Pick:", [("s", "Stash")], default="n",
        )
        assert result == "s"


class TestUniqueBranchName:
    """Tests for MirrorManager._unique_branch_name."""

    def test_first_name_available(self) -> None:
        branch = MirrorManager._unique_branch_name(
            "rescue", exists_fn=lambda _name: False,
        )
        # Should be rescue/YYYY-MM-DD with no suffix.
        assert branch.startswith("rescue/")
        assert "-" in branch  # date contains dashes
        parts = branch.split("/", 1)[1].split("-")
        assert len(parts) == 3  # YYYY-MM-DD

    def test_appends_suffix_when_taken(self) -> None:
        taken: set[str] = set()

        def _exists(name: str) -> bool:
            # First two names are taken.
            if len(taken) < 2:
                taken.add(name)
                return True
            return False

        branch = MirrorManager._unique_branch_name(
            "mirror-stash/main", exists_fn=_exists,
        )
        assert branch.endswith("-2")
        assert branch.startswith("mirror-stash/main/")


# ------------------------------------------------------------------
# _ensure_remote_worktree_clean
# ------------------------------------------------------------------


def _make_fake_run(status_output: str = "", stash_fail: bool = False):
    """Return a fake ``run`` callable and a log of calls made to it.

    *status_output* is returned as stdout for ``git status --porcelain``.
    """
    calls: list[list[str]] = []

    def fake_run(args, *, check=False, cwd=None, **_kwargs):
        calls.append(list(args))
        cmd = " ".join(args)

        if "status --porcelain" in cmd:
            return CommandResult(
                requested_args=list(args),
                executed_args=list(args),
                stdout=status_output,
                stderr="",
                returncode=0,
            )

        if "rev-parse --verify" in cmd:
            # Branch never exists — first name is always available.
            return CommandResult(
                requested_args=list(args),
                executed_args=list(args),
                stdout="",
                stderr="",
                returncode=1,
            )

        if "symbolic-ref" in cmd:
            return CommandResult(
                requested_args=list(args),
                executed_args=list(args),
                stdout="main\n",
                stderr="",
                returncode=0,
            )

        if "stash" in cmd and stash_fail:
            raise CommandError("stash failed", CommandResult(
                requested_args=list(args),
                executed_args=list(args),
                stdout="",
                stderr="stash failed",
                returncode=1,
            ))

        # Default: succeed.
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="",
            stderr="",
            returncode=0,
        )

    return fake_run, calls


class TestEnsureRemoteWorktreeClean:
    """Tests for MirrorManager._ensure_remote_worktree_clean."""

    def _manager(self) -> MirrorManager:
        """Build a minimal MirrorManager (no real repos needed)."""
        logger = logging.getLogger("sucoder.test.worktree_clean")
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        # We only call static/instance methods that don't touch config,
        # so we can pass None for fields we don't need.
        return MirrorManager.__new__(MirrorManager)

    def _init_manager(self) -> MirrorManager:
        mgr = self._manager()
        mgr.logger = logging.getLogger("sucoder.test.worktree_clean")
        mgr.logger.setLevel(logging.DEBUG)
        if not mgr.logger.handlers:
            mgr.logger.addHandler(logging.NullHandler())
        return mgr

    def test_clean_worktree_is_noop(self) -> None:
        mgr = self._init_manager()
        run, calls = _make_fake_run(status_output="")
        # Should return without prompting, signalling "ok to push".
        assert mgr._ensure_remote_worktree_clean(run, "/fake/path") is True
        assert len(calls) == 1  # only the status call
        assert "status" in " ".join(calls[0])

    def test_rescue_commit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = self._init_manager()
        run, calls = _make_fake_run(status_output=" M file.txt\n?? new.py\n")
        monkeypatch.setattr("builtins.input", lambda _prompt: "c")

        result = mgr._ensure_remote_worktree_clean(run, "/fake/path")

        cmds = [" ".join(c) for c in calls]
        assert any("checkout -b" in c and "rescue/" in c for c in cmds)
        assert any("git add -A" in c for c in cmds)
        assert any("git commit" in c for c in cmds)
        # Must switch back to original branch.
        assert any(c == "git checkout main" for c in cmds)
        # Caller should proceed with the push.
        assert result is True

    def test_stash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = self._init_manager()
        run, calls = _make_fake_run(status_output=" M dirty.txt\n")
        monkeypatch.setattr("builtins.input", lambda _prompt: "s")

        result = mgr._ensure_remote_worktree_clean(run, "/fake/path")

        cmds = [" ".join(c) for c in calls]
        assert any("stash" in c for c in cmds)
        assert result is True

    def test_discard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = self._init_manager()
        run, calls = _make_fake_run(status_output=" M dirty.txt\n")
        monkeypatch.setattr("builtins.input", lambda _prompt: "d")

        result = mgr._ensure_remote_worktree_clean(run, "/fake/path")

        cmds = [" ".join(c) for c in calls]
        assert any("checkout -- ." in c for c in cmds)
        assert any("clean -fd" in c for c in cmds)
        assert result is True

    def test_skip_push(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """'k' leaves the remote untouched and returns False."""
        mgr = self._init_manager()
        run, calls = _make_fake_run(status_output=" M dirty.txt\n?? new.py\n")
        monkeypatch.setattr("builtins.input", lambda _prompt: "k")

        result = mgr._ensure_remote_worktree_clean(run, "/fake/path")

        cmds = [" ".join(c) for c in calls]
        # Only the initial status check should have run — no
        # checkout/stash/clean/commit on the remote.
        assert len(calls) == 1
        assert "status" in cmds[0]
        # And the caller is told to skip the push.
        assert result is False

    def test_abort_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = self._init_manager()
        run, _calls = _make_fake_run(status_output=" M dirty.txt\n")
        monkeypatch.setattr("builtins.input", lambda _prompt: "n")

        with pytest.raises(MirrorError, match="uncommitted changes"):
            mgr._ensure_remote_worktree_clean(run, "/fake/path")

    def test_default_is_rescue_commit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty input should default to 'c' (rescue commit)."""
        mgr = self._init_manager()
        run, calls = _make_fake_run(status_output=" M file.txt\n")
        monkeypatch.setattr("builtins.input", lambda _prompt: "")

        result = mgr._ensure_remote_worktree_clean(run, "/fake/path")

        cmds = [" ".join(c) for c in calls]
        assert any("checkout -b" in c and "rescue/" in c for c in cmds)
        # Default path still pushes.
        assert result is True

    def test_skip_push_offered_in_prompt(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The 'skip push' option should appear in the menu."""
        mgr = self._init_manager()
        run, _calls = _make_fake_run(status_output=" M dirty.txt\n")
        prompts: list[str] = []

        def _capture(prompt: str) -> str:
            prompts.append(prompt)
            return "k"

        monkeypatch.setattr("builtins.input", _capture)

        mgr._ensure_remote_worktree_clean(run, "/fake/path")

        assert prompts, "input() was never called"
        menu = prompts[-1]
        assert "[k]" in menu
        assert "Skip the push" in menu

    def test_truncates_long_file_list(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """More than 20 dirty files should show a '… and N more' line."""
        mgr = self._init_manager()
        lines = "\n".join(f" M file{i}.txt" for i in range(25))
        run, _calls = _make_fake_run(status_output=lines)
        monkeypatch.setattr("builtins.input", lambda _prompt: "d")

        mgr._ensure_remote_worktree_clean(run, "/fake/path")

        captured = capsys.readouterr().out
        assert "… and 5 more files" in captured


# -- _resolve_base_branch -------------------------------------------------


def test_resolve_base_branch_configured_wins(tmp_path: Path) -> None:
    """An explicit default_base_branch bypasses the probe entirely."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    ctx.settings.default_base_branch = "release"
    assert manager._resolve_base_branch(ctx) == "release"


def test_resolve_base_branch_ignores_feature_checkout(tmp_path: Path) -> None:
    """Auto-detect must not follow a transient feature-branch checkout."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    ctx.settings.default_base_branch = None
    run_git(["checkout", "-b", "feat/wip"], ctx.canonical_path)
    assert manager._resolve_base_branch(ctx) == "main"


def test_resolve_base_branch_master_repo(tmp_path: Path) -> None:
    """A master-based canonical (no main) auto-detects master.

    Regression: the old hardcoded "main" fallback made the pre-push
    safety fetch a silent no-op for master-based repos (defeating the
    agent-commit rescue) and broke the post-push `reset --hard main`
    on the remote mirror.
    """
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    ctx.settings.default_base_branch = None
    run_git(["branch", "-m", "main", "master"], ctx.canonical_path)
    assert manager._resolve_base_branch(ctx) == "master"


def test_resolve_base_branch_prefers_origin_head(tmp_path: Path) -> None:
    """origin/HEAD (the upstream's default) outranks a local main."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    ctx.settings.default_base_branch = None
    run_git(
        ["symbolic-ref", "refs/remotes/origin/HEAD",
         "refs/remotes/origin/trunk"],
        ctx.canonical_path,
    )
    assert manager._resolve_base_branch(ctx) == "trunk"


def test_resolve_base_branch_probe_is_cached(tmp_path: Path) -> None:
    """The probe runs once per mirror per manager instance."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    ctx.settings.default_base_branch = None
    assert manager._resolve_base_branch(ctx) == "main"

    def boom(*args, **kwargs):  # pragma: no cover - fails the test if hit
        raise AssertionError("second resolve should be served from cache")

    manager.executor.run_human = boom  # type: ignore[method-assign]
    assert manager._resolve_base_branch(ctx) == "main"


# -- _sync_remote error wrapping ------------------------------------------


def test_sync_remote_unpacker_error_mentions_disk(tmp_path: Path) -> None:
    """The BRC field failure: remote index-pack dies with EIO.

    The raised MirrorError must keep the git detail and hint at
    checking disk space/quota on the remote host, so the CLI prints
    an actionable message instead of a traceback.
    """
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")

    manager._git_transports = (  # type: ignore[method-assign]
        lambda _ctx: [("login node", "ln000:/global/home/u/mirrors/sample", None)]
    )

    stderr = (
        "remote: fatal: write error: Input/output error\n"
        "error: remote unpack failed: index-pack abnormal exit\n"
        " ! [remote rejected] master -> master (unpacker error)\n"
        "error: failed to push some refs to "
        "'ln000:/global/home/u/mirrors/sample'\n"
    )

    def fail_push(args, **kwargs):
        raise CommandError(
            "Command failed with exit code 1: git push …",
            CommandResult(list(args), list(args), "", stderr, 1),
        )

    manager.executor.run_human = fail_push  # type: ignore[method-assign]

    with pytest.raises(MirrorError) as excinfo:
        manager._sync_remote(ctx)

    msg = str(excinfo.value)
    assert "failed to push some refs" in msg
    assert "quota" in msg
    assert isinstance(excinfo.value.__cause__, CommandError)


# ---------------------------------------------------------------------------
# Agent binary resolution reporting
# ---------------------------------------------------------------------------


def _write_fake_binary(path: Path, version_output: str, *, returncode: int = 0) -> Path:
    """Create an executable that prints ``version_output`` for --version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!/bin/sh\nprintf '%s\\n' {version_output!r}\nexit {returncode}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


@pytest.mark.parametrize(
    "text,expected",
    [
        ("codex-cli 0.77.0", (0, 77, 0)),
        ("2.1.233 (Claude Code)", (2, 1, 233)),
        ("0.50.0", (0, 50, 0)),
        ("v1.2", (1, 2)),
        ("v22.22.3", (22, 22, 3)),
        ("codex-cli 1.2.3.4", (1, 2, 3, 4)),
        ("no digits here", None),
        ("build 20260815", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_version_handles_agent_version_shapes(text, expected) -> None:
    assert _parse_version(text) == expected


def test_parse_version_orders_numerically_not_lexically() -> None:
    # The bug this guard exists for: 0.77.0 shadowing 0.147.0, where a string
    # comparison would wrongly call the stale one newer.
    assert _parse_version("codex-cli 0.77.0") < _parse_version("codex-cli 0.147.0")


def test_probe_binary_version_reads_first_line(tmp_path: Path) -> None:
    binary = _write_fake_binary(tmp_path / "codex", "codex-cli 0.147.0")
    assert _probe_binary_version(str(binary)) == "codex-cli 0.147.0"


def test_probe_binary_version_returns_none_on_nonzero_exit(tmp_path: Path) -> None:
    binary = _write_fake_binary(tmp_path / "codex", "nope", returncode=1)
    assert _probe_binary_version(str(binary)) is None


def test_probe_binary_version_returns_none_for_missing_binary(tmp_path: Path) -> None:
    assert _probe_binary_version(str(tmp_path / "absent")) is None


def test_report_agent_binary_logs_resolved_path_and_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = build_manager(tmp_path, report_agent_binary=True)
    ctx = manager.context_for("sample")

    bin_dir = tmp_path / "path-bin"
    _write_fake_binary(bin_dir / "codex", "codex-cli 0.147.0")
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setattr(manager, "_agent_home_directory", lambda: None)

    with caplog.at_level(logging.INFO):
        manager._report_agent_binary(ctx, ["codex"], AgentLauncher(command=["codex"], env={}))

    assert f"{bin_dir / 'codex'}" in caplog.text
    assert "codex-cli 0.147.0" in caplog.text
    assert "shadowing" not in caplog.text


def test_report_agent_binary_warns_when_newer_install_is_shadowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The real failure: a stale system install wins the PATH lookup.

    Mirrors the observed host state -- /usr/bin/codex at 0.77.0 on PATH while
    the current 0.147.0 sits in the agent user's nvm tree, invisible because
    ~/.bashrc's nvm block short-circuits in non-interactive shells.
    """
    manager = build_manager(tmp_path, report_agent_binary=True)
    ctx = manager.context_for("sample")

    stale_dir = tmp_path / "usr-bin"
    _write_fake_binary(stale_dir / "codex", "codex-cli 0.77.0")
    monkeypatch.setenv("PATH", str(stale_dir))

    agent_home = tmp_path / "agent-home"
    current = _write_fake_binary(
        agent_home / ".nvm" / "versions" / "node" / "v22.22.3" / "bin" / "codex",
        "codex-cli 0.147.0",
    )
    monkeypatch.setattr(manager, "_agent_home_directory", lambda: agent_home)

    with caplog.at_level(logging.WARNING):
        manager._report_agent_binary(ctx, ["codex"], AgentLauncher(command=["codex"], env={}))

    assert "shadowing a newer install" in caplog.text
    assert str(current) in caplog.text
    assert "0.147.0" in caplog.text


def test_report_agent_binary_ignores_older_and_identical_installs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = build_manager(tmp_path, report_agent_binary=True)
    ctx = manager.context_for("sample")

    bin_dir = tmp_path / "path-bin"
    _write_fake_binary(bin_dir / "codex", "codex-cli 0.147.0")
    monkeypatch.setenv("PATH", str(bin_dir))

    agent_home = tmp_path / "agent-home"
    _write_fake_binary(
        agent_home / ".nvm" / "versions" / "node" / "v20.0.0" / "bin" / "codex",
        "codex-cli 0.77.0",
    )
    _write_fake_binary(agent_home / ".local" / "bin" / "codex", "codex-cli 0.147.0")
    monkeypatch.setattr(manager, "_agent_home_directory", lambda: agent_home)

    with caplog.at_level(logging.WARNING):
        manager._report_agent_binary(ctx, ["codex"], AgentLauncher(command=["codex"], env={}))

    assert "shadowing" not in caplog.text


def test_report_agent_binary_ignores_symlink_to_same_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A symlink chain to one install must not look like two competing ones."""
    manager = build_manager(tmp_path, report_agent_binary=True)
    ctx = manager.context_for("sample")

    real = _write_fake_binary(tmp_path / "real" / "codex", "codex-cli 0.147.0")

    bin_dir = tmp_path / "path-bin"
    bin_dir.mkdir()
    (bin_dir / "codex").symlink_to(real)
    monkeypatch.setenv("PATH", str(bin_dir))

    agent_home = tmp_path / "agent-home"
    link_dir = agent_home / ".local" / "bin"
    link_dir.mkdir(parents=True)
    (link_dir / "codex").symlink_to(real)
    monkeypatch.setattr(manager, "_agent_home_directory", lambda: agent_home)

    with caplog.at_level(logging.WARNING):
        manager._report_agent_binary(ctx, ["codex"], AgentLauncher(command=["codex"], env={}))

    assert "shadowing" not in caplog.text


def test_report_agent_binary_warns_when_command_not_on_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = build_manager(tmp_path, report_agent_binary=True)
    ctx = manager.context_for("sample")

    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    with caplog.at_level(logging.WARNING):
        manager._report_agent_binary(ctx, ["codex"], AgentLauncher(command=["codex"], env={}))

    assert "was not found on PATH" in caplog.text


def test_report_agent_binary_skips_when_nvm_is_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An explicit nvm pin resolves inside the nvm shell, so our lookup would lie."""
    manager = build_manager(tmp_path, report_agent_binary=True)
    ctx = manager.context_for("sample")

    stale_dir = tmp_path / "usr-bin"
    _write_fake_binary(stale_dir / "codex", "codex-cli 0.77.0")
    monkeypatch.setenv("PATH", str(stale_dir))

    launcher = AgentLauncher(
        command=["codex"],
        env={},
        nvm=NvmConfig(version="22.11.0", dir=tmp_path / "nvm"),
    )

    with caplog.at_level(logging.INFO):
        manager._report_agent_binary(ctx, ["codex"], launcher)

    assert "Agent binary" not in caplog.text


def test_report_agent_binary_reports_explicit_path_without_shadow_check(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = build_manager(tmp_path, report_agent_binary=True)
    ctx = manager.context_for("sample")

    explicit = _write_fake_binary(tmp_path / "opt" / "codex", "codex-cli 0.147.0")

    with caplog.at_level(logging.INFO):
        manager._report_agent_binary(ctx, [str(explicit)], AgentLauncher(command=[str(explicit)], env={}))

    assert str(explicit) in caplog.text
    assert "shadowing" not in caplog.text


def test_report_agent_binary_resolves_via_login_shell_not_our_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The launch resolves in a profile-sourcing shell, so the report must too.

    Stock Debian ``~/.profile`` prepends ``$HOME/.local/bin``, so the agent's
    login PATH can name a different binary than this process's PATH.  Resolving
    with ``shutil.which`` reports the wrong one -- and then calls the binary
    that actually runs a "shadowed" rival, which is backwards.
    """
    manager = build_manager(tmp_path, report_agent_binary=True)
    ctx = manager.context_for("sample")

    ours = tmp_path / "our-path" / "codex"
    _write_fake_binary(ours, "codex-cli 0.77.0")
    theirs = tmp_path / "login-path" / "codex"
    _write_fake_binary(theirs, "codex-cli 0.147.0")

    # This process sees only the stale one; the login shell prefers the new one.
    monkeypatch.setenv("PATH", str(ours.parent))
    monkeypatch.setattr(manager, "_agent_home_directory", lambda: None)

    def fake_run_agent(args, **kwargs):
        assert args[0] == "bash" and args[1] == "-lc"
        assert "command -v codex" in args[2]
        return CommandResult(
            requested_args=list(args), executed_args=list(args),
            stdout=f"{theirs}\n", stderr="", returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    with caplog.at_level(logging.INFO):
        manager._report_agent_binary(ctx, ["codex"], AgentLauncher(command=["codex"], env={}))

    assert str(theirs) in caplog.text
    assert "0.147.0" in caplog.text
    # The one that actually launches is the newest, so nothing is shadowed.
    assert "shadowing" not in caplog.text


def test_report_agent_binary_falls_back_to_which_when_shell_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A shell we cannot run degrades the diagnostic; it does not remove it."""
    manager = build_manager(tmp_path, report_agent_binary=True)
    ctx = manager.context_for("sample")

    bin_dir = tmp_path / "path-bin"
    _write_fake_binary(bin_dir / "codex", "codex-cli 0.147.0")
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setattr(manager, "_agent_home_directory", lambda: None)

    def boom(args, **kwargs):
        raise OSError("no shell here")

    monkeypatch.setattr(manager.executor, "run_agent", boom)

    with caplog.at_level(logging.INFO):
        manager._report_agent_binary(ctx, ["codex"], AgentLauncher(command=["codex"], env={}))

    assert str(bin_dir / "codex") in caplog.text


def test_report_agent_binary_ignores_shell_builtin_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`command -v` echoes a bare word for builtins; only a path is probeable."""
    manager = build_manager(tmp_path)

    bin_dir = tmp_path / "path-bin"
    _write_fake_binary(bin_dir / "codex", "codex-cli 0.147.0")
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setattr(
        manager.executor,
        "run_agent",
        lambda args, **kw: CommandResult(
            requested_args=list(args), executed_args=list(args),
            stdout="codex\n", stderr="", returncode=0,
        ),
    )

    assert manager._resolve_agent_binary("codex") == str(bin_dir / "codex")


def test_probe_binary_version_survives_non_utf8_output(tmp_path: Path) -> None:
    """Strict decoding raises UnicodeDecodeError -- neither OSError nor
    SubprocessError -- so it must not escape a best-effort diagnostic."""
    path = tmp_path / "codex"
    path.write_bytes(b'#!/bin/sh\nprintf "agent \\377\\376 1.2.3\\n"\n')
    path.chmod(0o755)

    assert "1.2.3" in (_probe_binary_version(str(path)) or "")


def test_launch_agent_survives_a_failing_binary_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken diagnostic must never be the reason a launch fails."""
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    def explode(*args, **kwargs):
        raise RuntimeError("diagnostic blew up")

    monkeypatch.setattr(manager, "_report_agent_binary", explode)

    recorded: Dict[str, object] = {}

    def fake_run_agent(args, **kwargs):
        recorded["args"] = list(args)
        return CommandResult(
            requested_args=list(args), executed_args=list(args),
            stdout="", stderr="", returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    assert manager.launch_agent(ctx, sync=False) == 0
    assert recorded.get("args"), "the agent should still have been launched"
