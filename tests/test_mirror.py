import grp
import logging
import os
import pwd
import subprocess
from pathlib import Path
from typing import Callable, Optional

import pytest

import sucoder.mirror as mirror
from sucoder.config import AgentLauncher, BranchPrefixes, Config, MirrorSettings, NvmConfig
from sucoder.executor import CommandError, CommandExecutor, CommandResult
from sucoder.mirror import (
    MirrorError,
    MirrorManager,
    _detect_agent_type,
    _merge_flag_templates,
    _sanitize_task_name,
)
from sucoder.workspace_prefs import WorkspacePrefs


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

    return MirrorManager(config, executor, logger, prompt_handler=prompt_handler)


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
    assert args[0] == "codex"
    # Codex profile uses --sandbox and --ask-for-approval instead of --yolo
    assert "--sandbox" in args
    assert "danger-full-access" in args
    assert "--ask-for-approval" in args
    assert "never" in args
    # Codex profile doesn't use writable_dir template (sandbox handles it)
    assert "--add-dir" not in args
    prelude = args[-1]
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

    assert calls
    assert calls[0][:2] == ["poetry", "install"]


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
    assert calls[0][:2] == ["poetry", "install"]


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
    assert agent_calls, "Agent command should still execute after poetry failure."
    assert agent_calls[0][0] == "codex"
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
    assert all(call[:2] != ["poetry", "install"] for call in calls)


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

    codex_call = recorded[-1]
    # Codex profile uses --sandbox flags
    assert "--sandbox" in codex_call
    # User-provided args are preserved
    assert "--add-dir" in codex_call
    assert str(Path.home()) in codex_call
    assert "--foo" in codex_call


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
    assert args[0] == "codex"
    assert str(prompt) not in args  # prompt should be merged, not passed as file
    prelude = args[-1]
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

    assert any("SYSTEM PROMPT" in call["args"][-1] and "Default" in call["args"][-1] for call in calls)


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

    prelude = calls[0]["args"][-1]
    assert "SKILL CATALOG" in prelude
    assert "Catalog Skill" in prelude
    assert "Detail Skill" in prelude
    assert "load with `codex read" in prelude


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

    prelude = calls[0]["args"][-1]
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

    prelude = calls[0]["args"][-1]
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

    def fake_ensure(c):
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
