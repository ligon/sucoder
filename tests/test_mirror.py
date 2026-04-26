import grp
import logging
import os
import pwd
import subprocess
from pathlib import Path
from typing import Callable, Optional

import pytest

import sucoder.mirror as mirror
from sucoder.config import AgentLauncher, BranchPrefixes, Config, McpServerConfig, MirrorSettings, NvmConfig
from sucoder.executor import CommandError, CommandExecutor, CommandResult
from sucoder.mirror import (
    MirrorError,
    MirrorManager,
    WorktreeInfo,
    _detect_agent_type,
    _merge_flag_templates,
    _parse_worktree_porcelain,
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
            stdout="?? new-skill.md\n" if args[:2] == ["git", "status"] else "",
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
            stdout="" if args[:2] == ["git", "status"] else "",
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
            stdout="" if args[:2] == ["git", "status"] else "",
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
        if args[:3] == ["git", "rev-parse", "--verify"]:
            # No baseline ref exists.
            result = CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="", stderr="", returncode=1,
            )
            raise CommandError("ref not found", result)
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
        if args[:3] == ["git", "rev-parse", "--verify"]:
            # Baseline exists.
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="abc1234\n", stderr="", returncode=0,
            )
        if args[:2] == ["git", "diff"]:
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
        if args[:3] == ["git", "rev-parse", "--verify"]:
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="abc1234\n", stderr="", returncode=0,
            )
        if args[:2] == ["git", "diff"]:
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
        if args[:3] == ["git", "rev-parse", "--verify"]:
            # No baseline ref exists.
            result = CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="", stderr="", returncode=1,
            )
            raise CommandError("ref not found", result)
        if args[:2] == ["git", "diff"]:
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
        if args[:3] == ["git", "rev-parse", "--verify"]:
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="abc1234\n", stderr="", returncode=0,
            )
        if args[:2] == ["git", "diff"]:
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
    diff_calls = [c for c in calls if c[:2] == ["git", "diff"]]
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
        if args[:3] == ["git", "rev-parse", "--verify"]:
            return CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="abc1234\n", stderr="", returncode=0,
            )
        if args[:2] == ["git", "diff"]:
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

    update_ref_calls = [c for c in calls if c[:2] == ["git", "update-ref"]]
    assert update_ref_calls
    assert "refs/audited-code" in update_ref_calls[0]


def test_code_audit_full_uses_empty_tree_diff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full code audit diffs against the empty tree hash."""
    manager = _setup_code_audit_manager(tmp_path, monkeypatch)

    calls: list = []
    empty_tree = "4b825dc642cb6eb9a060e54bf899d15f3780fcaa"

    def fake_run_agent(args, **kwargs):
        calls.append(list(args))
        if args[:3] == ["git", "rev-parse", "--verify"]:
            result = CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="", stderr="", returncode=1,
            )
            raise CommandError("ref not found", result)
        if args[:2] == ["git", "diff"]:
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
    diff_calls = [c for c in calls if c[:2] == ["git", "diff"]]
    assert any(empty_tree in c for c in diff_calls), f"Expected empty tree hash in diff calls: {diff_calls}"


def test_code_audit_prompt_contains_security_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Code audit prompt includes security-specific review criteria."""
    manager = _setup_code_audit_manager(tmp_path, monkeypatch)

    captured_prompts: list = []

    def fake_run_agent(args, **kwargs):
        if args[:3] == ["git", "rev-parse", "--verify"]:
            result = CommandResult(
                requested_args=list(args), executed_args=list(args),
                stdout="", stderr="", returncode=1,
            )
            raise CommandError("ref not found", result)
        if args[:2] == ["git", "diff"]:
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
        # Should return without prompting.
        mgr._ensure_remote_worktree_clean(run, "/fake/path")
        assert len(calls) == 1  # only the status call
        assert "status" in " ".join(calls[0])

    def test_rescue_commit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = self._init_manager()
        run, calls = _make_fake_run(status_output=" M file.txt\n?? new.py\n")
        monkeypatch.setattr("builtins.input", lambda _prompt: "c")

        mgr._ensure_remote_worktree_clean(run, "/fake/path")

        cmds = [" ".join(c) for c in calls]
        assert any("checkout -b" in c and "rescue/" in c for c in cmds)
        assert any("git add -A" in c for c in cmds)
        assert any("git commit" in c for c in cmds)
        # Must switch back to original branch.
        assert any(c == "git checkout main" for c in cmds)

    def test_stash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = self._init_manager()
        run, calls = _make_fake_run(status_output=" M dirty.txt\n")
        monkeypatch.setattr("builtins.input", lambda _prompt: "s")

        mgr._ensure_remote_worktree_clean(run, "/fake/path")

        cmds = [" ".join(c) for c in calls]
        assert any("stash" in c for c in cmds)

    def test_discard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = self._init_manager()
        run, calls = _make_fake_run(status_output=" M dirty.txt\n")
        monkeypatch.setattr("builtins.input", lambda _prompt: "d")

        mgr._ensure_remote_worktree_clean(run, "/fake/path")

        cmds = [" ".join(c) for c in calls]
        assert any("checkout -- ." in c for c in cmds)
        assert any("clean -fd" in c for c in cmds)

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

        mgr._ensure_remote_worktree_clean(run, "/fake/path")

        cmds = [" ".join(c) for c in calls]
        assert any("checkout -b" in c and "rescue/" in c for c in cmds)

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
