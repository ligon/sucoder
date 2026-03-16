"""Tests for remote execution: config parsing, session, tunnel, and RemoteExecutor."""

import os
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from sucoder.config import (
    ConfigError,
    MirrorSettings,
    RemoteConfig,
    _parse_remote_config,
    load_config,
)
from sucoder.executor import RemoteExecutor
from sucoder.session import RemoteSession


# ------------------------------------------------------------------
# RemoteConfig parsing
# ------------------------------------------------------------------


def test_parse_remote_config_valid() -> None:
    raw = {
        "gateway": "brc.berkeley.edu",
        "transfer_host": "dtn.brc.berkeley.edu",
        "mirror_root": "~/mirrors",
    }
    rc = _parse_remote_config(raw)
    assert rc is not None
    assert rc.gateway == "brc.berkeley.edu"
    assert rc.transfer_host == "dtn.brc.berkeley.edu"
    assert rc.mirror_root == Path("~/mirrors")
    assert rc.ssh_options == {}


def test_parse_remote_config_none() -> None:
    assert _parse_remote_config(None) is None


def test_parse_remote_config_missing_gateway() -> None:
    with pytest.raises(ConfigError, match="gateway"):
        _parse_remote_config({"transfer_host": "dtn"})


def test_parse_remote_config_missing_transfer_host() -> None:
    with pytest.raises(ConfigError, match="transfer_host"):
        _parse_remote_config({"gateway": "gw"})


def test_parse_remote_config_with_ssh_options() -> None:
    raw = {
        "gateway": "gw",
        "transfer_host": "dtn",
        "ssh_options": {"StrictHostKeyChecking": "no"},
    }
    rc = _parse_remote_config(raw)
    assert rc is not None
    assert rc.ssh_options == {"StrictHostKeyChecking": "no"}


def test_parse_remote_config_bad_type() -> None:
    with pytest.raises(ConfigError, match="mapping"):
        _parse_remote_config("not-a-dict")


def test_mirror_settings_is_remote() -> None:
    from sucoder.config import BranchPrefixes

    remote = RemoteConfig(gateway="gw", transfer_host="dtn")
    settings = MirrorSettings(
        name="test",
        canonical_repo=Path("/tmp/test"),
        mirror_name="test",
        branch_prefixes=BranchPrefixes(human="ligon", agent="coder"),
        remote=remote,
    )
    assert settings.is_remote is True


def test_mirror_settings_not_remote() -> None:
    from sucoder.config import BranchPrefixes

    settings = MirrorSettings(
        name="test",
        canonical_repo=Path("/tmp/test"),
        mirror_name="test",
        branch_prefixes=BranchPrefixes(human="ligon", agent="coder"),
    )
    assert settings.is_remote is False


def test_load_config_with_remote(tmp_path: Path) -> None:
    """A full config file with a remote block parses correctly."""
    config_data: Dict[str, Any] = {
        "human_user": "ligon",
        "mirror_root": str(tmp_path / "mirrors"),
        "mirrors": {
            "cluster_project": {
                "canonical_repo": str(tmp_path),
                "remote": {
                    "gateway": "brc.berkeley.edu",
                    "transfer_host": "dtn.brc.berkeley.edu",
                    "mirror_root": "~/mirrors",
                },
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    config = load_config(config_path)
    settings = config.mirrors["cluster_project"]
    assert settings.remote is not None
    assert settings.remote.gateway == "brc.berkeley.edu"
    assert settings.is_remote is True


# ------------------------------------------------------------------
# RemoteSession
# ------------------------------------------------------------------


def test_session_save_load_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path)

    session = RemoteSession(mirror_name="test", login_node="ln003", tunnel_port=2222)
    session.save()

    loaded = RemoteSession.load("test")
    assert loaded.login_node == "ln003"
    assert loaded.tunnel_port == 2222
    assert loaded.created is not None


def test_session_load_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path)
    loaded = RemoteSession.load("nonexistent")
    assert loaded.login_node is None
    assert loaded.tunnel_port is None


def test_session_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path)

    session = RemoteSession(mirror_name="test", login_node="ln003")
    session.save()
    assert (tmp_path / "test.yaml").exists()

    session.clear()
    assert not (tmp_path / "test.yaml").exists()


def test_session_clear_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Clearing a non-existent session is a no-op."""
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path)
    session = RemoteSession(mirror_name="test")
    session.clear()  # Should not raise


def test_session_tunnel_alive_no_pid() -> None:
    session = RemoteSession(mirror_name="test")
    assert session.tunnel_alive() is False


def test_session_tunnel_alive_dead_pid() -> None:
    session = RemoteSession(mirror_name="test", tunnel_pid=999999999)
    assert session.tunnel_alive() is False


# ------------------------------------------------------------------
# RemoteExecutor
# ------------------------------------------------------------------


def _make_remote_executor(**kwargs) -> RemoteExecutor:
    import logging

    logger = logging.getLogger("test.remote")
    defaults = dict(
        human_user="ligon",
        agent_user="ligon",
        agent_group="ligon",
        logger=logger,
        dry_run=False,
        use_sudo_for_agent=False,
        gateway="brc.berkeley.edu",
        login_node="ln003",
        remote_mirror_root="~/mirrors",
        local_mirror_root="/var/tmp/coder-mirrors",
    )
    defaults.update(kwargs)
    return RemoteExecutor(**defaults)


def test_build_ssh_command_basic() -> None:
    executor = _make_remote_executor()
    cmd = executor._build_ssh_command(["git", "status"])
    assert cmd[:2] == ["ssh", "-J"]
    assert "brc.berkeley.edu" in cmd
    assert "ln003" in cmd
    assert "git status" in cmd[-1]


def test_build_ssh_command_with_cwd() -> None:
    executor = _make_remote_executor()
    cmd = executor._build_ssh_command(["git", "log"], cwd="/home/ligon/project")
    remote_cmd = cmd[-1]
    assert "cd" in remote_cmd
    assert "/home/ligon/project" in remote_cmd
    assert "git log" in remote_cmd


def test_build_ssh_command_with_env() -> None:
    executor = _make_remote_executor()
    cmd = executor._build_ssh_command(["echo", "hi"], env={"FOO": "bar"})
    remote_cmd = cmd[-1]
    assert "FOO" in remote_cmd
    assert "bar" in remote_cmd


def test_build_ssh_command_with_tty() -> None:
    executor = _make_remote_executor()
    cmd = executor._build_ssh_command(["bash"], allocate_tty=True)
    assert "-t" in cmd


def test_build_ssh_command_with_options() -> None:
    executor = _make_remote_executor(ssh_options={"StrictHostKeyChecking": "no"})
    cmd = executor._build_ssh_command(["ls"])
    assert "-o" in cmd
    idx = cmd.index("-o")
    assert cmd[idx + 1] == "StrictHostKeyChecking=no"


def test_translate_path_rewrites_mirror_root() -> None:
    executor = _make_remote_executor()
    result = executor._translate_path("/var/tmp/coder-mirrors/MyProject")
    assert result == "~/mirrors/MyProject"


def test_translate_path_passthrough() -> None:
    executor = _make_remote_executor()
    result = executor._translate_path("/home/ligon/something")
    assert result == "/home/ligon/something"


def test_translate_path_preserves_subdirs() -> None:
    executor = _make_remote_executor()
    result = executor._translate_path("/var/tmp/coder-mirrors/MyProject/.claude/worktrees/fix")
    assert result == "~/mirrors/MyProject/.claude/worktrees/fix"


def test_build_ssh_command_with_control_socket() -> None:
    executor = _make_remote_executor(control_socket_path="/tmp/test.sock")
    cmd = executor._build_ssh_command(["git", "status"])
    assert "-o" in cmd
    assert "ControlMaster=auto" in cmd
    assert "ControlPath=/tmp/test.sock" in cmd


def test_build_ssh_command_without_control_socket() -> None:
    executor = _make_remote_executor(control_socket_path=None)
    cmd = executor._build_ssh_command(["git", "status"])
    joined = " ".join(cmd)
    assert "ControlMaster" not in joined


# ------------------------------------------------------------------
# SshControl
# ------------------------------------------------------------------


def test_ssh_control_socket_path() -> None:
    from sucoder.tunnel import SshControl

    control = SshControl(gateway="brc.berkeley.edu")
    path = control.socket_path
    assert "brc.berkeley.edu" in str(path)
    assert path.suffix == ".sock"


def test_ssh_control_options() -> None:
    from sucoder.tunnel import SshControl

    control = SshControl(gateway="gw")
    opts = control.ssh_options()
    assert "-o" in opts
    assert "ControlMaster=auto" in opts
    assert any("ControlPath=" in o for o in opts)


# ------------------------------------------------------------------
# Targets (top-level config)
# ------------------------------------------------------------------


def test_parse_targets_valid() -> None:
    from sucoder.config import _parse_targets

    raw = {
        "savio": {
            "gateway": "brc.berkeley.edu",
            "transfer_host": "dtn.brc.berkeley.edu",
            "mirror_root": "~/mirrors",
            "control_persist": "24h",
        },
        "lab": {
            "gateway": "lab.example.com",
            "transfer_host": "lab.example.com",
        },
    }
    targets = _parse_targets(raw)
    assert len(targets) == 2
    assert targets["savio"].gateway == "brc.berkeley.edu"
    assert targets["savio"].control_persist == "24h"
    assert targets["lab"].mirror_root == Path("~/mirrors")  # default


def test_parse_targets_none() -> None:
    from sucoder.config import _parse_targets

    assert _parse_targets(None) == {}


def test_parse_targets_bad_type() -> None:
    from sucoder.config import _parse_targets

    with pytest.raises(ConfigError, match="mapping"):
        _parse_targets("not-a-dict")


def test_config_resolve_target(tmp_path: Path) -> None:
    config_data: Dict[str, Any] = {
        "human_user": "ligon",
        "mirror_root": str(tmp_path),
        "targets": {
            "savio": {
                "gateway": "brc.berkeley.edu",
                "transfer_host": "dtn.brc.berkeley.edu",
            },
        },
        "mirrors": {
            "Foo": {"canonical_repo": str(tmp_path)},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    config = load_config(config_path)

    # Resolve known target
    target = config.resolve_target("savio")
    assert target is not None
    assert target.gateway == "brc.berkeley.edu"

    # Resolve None → local
    assert config.resolve_target(None) is None

    # Unknown target → error
    with pytest.raises(ConfigError, match="Unknown target"):
        config.resolve_target("nonexistent")


def test_mirror_settings_not_remote_until_target_applied() -> None:
    """Mirror stays local until a target overlays remote config."""
    from dataclasses import replace
    from sucoder.config import BranchPrefixes

    settings = MirrorSettings(
        name="Foo",
        canonical_repo=Path("/tmp/Foo"),
        mirror_name="Foo",
        branch_prefixes=BranchPrefixes(human="ligon", agent="coder"),
    )
    assert settings.is_remote is False

    target = RemoteConfig(gateway="gw", transfer_host="dtn")
    settings_with_target = replace(settings, remote=target)
    assert settings_with_target.is_remote is True
    assert settings.is_remote is False  # original unchanged


# ------------------------------------------------------------------
# Remote mirror operations
# ------------------------------------------------------------------


class FakeTunnel:
    local_port = 2222

    def is_alive(self):
        return True


def _build_remote_manager(tmp_path: Path, *, executor=None):
    """Build a MirrorManager whose mirror settings carry a RemoteConfig."""
    import grp
    import logging
    import os as _os
    import pwd

    from sucoder.config import BranchPrefixes, Config, MirrorSettings, RemoteConfig
    from sucoder.executor import CommandExecutor
    from sucoder.mirror import MirrorManager

    canonical = tmp_path / "canonical"
    canonical.mkdir(exist_ok=True)
    # Create a minimal git repo so canonical_path validation can pass.
    import subprocess

    subprocess.run(["git", "init", "-b", "main"], cwd=canonical, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=canonical, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=canonical, check=True, capture_output=True)
    (canonical / "README.md").write_text("hi\n")
    subprocess.run(["git", "add", "README.md"], cwd=canonical, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=canonical, check=True, capture_output=True)
    canonical.chmod(canonical.stat().st_mode & ~0o200)

    _os.environ["GIT_CONFIG_GLOBAL"] = str(tmp_path / "gitconfig")

    mirror_root = tmp_path / "mirrors"
    mirror_root.mkdir(exist_ok=True)

    user = pwd.getpwuid(_os.getuid()).pw_name
    group = grp.getgrgid(_os.getgid()).gr_name

    remote = RemoteConfig(gateway="gw.example.com", transfer_host="dtn.example.com")
    settings = MirrorSettings(
        name="rproj",
        canonical_repo=canonical,
        mirror_name="rproj",
        branch_prefixes=BranchPrefixes(human="ligon", agent="coder"),
        default_base_branch="main",
        remote=remote,
    )

    config = Config(
        human_user=user,
        agent_user=user,
        agent_group=group,
        mirror_root=mirror_root,
        log_dir=None,
        mirrors={"rproj": settings},
    )

    logger = logging.getLogger("sucoder.test.remote")
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

    return MirrorManager(config, executor, logger)


def test_sync_remote_calls_push_with_tunnel_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_sync_remote should push to ssh://localhost:<tunnel_port><remote_path>."""
    from sucoder.executor import CommandResult

    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")

    # Mock _ensure_tunnel to return our fake tunnel.
    monkeypatch.setattr(manager, "_ensure_tunnel", lambda remote: FakeTunnel())

    calls: list = []

    def fake_run_human(args, **kwargs):
        calls.append(list(args))
        return CommandResult(list(args), list(args), "", "", 0)

    monkeypatch.setattr(manager.executor, "run_human", fake_run_human)

    manager._sync_remote(ctx)

    assert len(calls) == 1
    push_cmd = calls[0]
    assert push_cmd[0] == "git"
    assert push_cmd[1] == "push"
    # SCP-style URL: localhost:~/mirrors/rproj (port via GIT_SSH_COMMAND)
    url = push_cmd[2]
    assert url.startswith("localhost:")
    assert "rproj" in url
    assert "--all" in push_cmd
    assert "--force" in push_cmd


def test_ensure_remote_clone_mirror_exists_skips_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the remote mirror already exists, ensure_remote_clone skips git init."""
    from sucoder.executor import CommandResult

    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")

    agent_calls: list = []

    def fake_run_agent(args, **kwargs):
        agent_calls.append(list(args))
        # test -d succeeds → mirror already exists
        return CommandResult(list(args), list(args), "", "", 0)

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    # Mock _sync_remote since we don't want actual sync
    sync_called = []
    monkeypatch.setattr(manager, "_sync_remote", lambda ctx: sync_called.append(True))

    manager.ensure_remote_clone(ctx)

    # Should have called test -d but NOT bash -c (git init)
    assert any("test" in c and "-d" in c for c in agent_calls)
    assert not any(c[0] == "bash" for c in agent_calls)
    # Sync should still be called
    assert sync_called


def test_ensure_remote_clone_mirror_not_exists_inits_and_syncs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the remote mirror does NOT exist, ensure_remote_clone inits then syncs."""
    from sucoder.executor import CommandResult

    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")

    agent_calls: list = []
    call_counter = [0]

    def fake_run_agent(args, **kwargs):
        agent_calls.append(list(args))
        call_counter[0] += 1
        # First call is test -d → fail (mirror does not exist)
        if call_counter[0] == 1:
            return CommandResult(list(args), list(args), "", "", 1)
        # Second call is bash -c init → succeed
        return CommandResult(list(args), list(args), "", "", 0)

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    sync_called = []
    monkeypatch.setattr(manager, "_sync_remote", lambda ctx: sync_called.append(True))

    manager.ensure_remote_clone(ctx)

    # Should have both test -d and bash -c calls
    assert any("test" in c and "-d" in c for c in agent_calls)
    assert any(c[0] == "bash" for c in agent_calls)
    assert sync_called


def test_ensure_remote_mirror_exists_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_ensure_mirror_exists returns ctx.mirror_path when remote check succeeds."""
    from sucoder.executor import CommandResult

    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")

    def fake_run_agent(args, **kwargs):
        return CommandResult(list(args), list(args), "", "", 0)

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    result = manager._ensure_mirror_exists(ctx)
    assert result == ctx.mirror_path


def test_ensure_remote_mirror_exists_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_ensure_mirror_exists raises MirrorError when remote check fails."""
    from sucoder.executor import CommandResult
    from sucoder.mirror import MirrorError

    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")

    def fake_run_agent(args, **kwargs):
        return CommandResult(list(args), list(args), "", "", 1)

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    with pytest.raises(MirrorError):
        manager._ensure_mirror_exists(ctx)


def test_run_query_dispatch_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_run_query calls run_human for non-remote contexts."""
    import grp
    import logging
    import os as _os
    import pwd

    from sucoder.config import BranchPrefixes, Config, MirrorSettings
    from sucoder.executor import CommandExecutor, CommandResult
    from sucoder.mirror import MirrorManager

    _os.environ["GIT_CONFIG_GLOBAL"] = str(tmp_path / "gitconfig")
    mirror_root = tmp_path / "mirrors"
    mirror_root.mkdir(exist_ok=True)

    user = pwd.getpwuid(_os.getuid()).pw_name
    group = grp.getgrgid(_os.getgid()).gr_name

    # Local settings (no remote)
    settings = MirrorSettings(
        name="local",
        canonical_repo=tmp_path,
        mirror_name="local",
        branch_prefixes=BranchPrefixes(human="ligon", agent="coder"),
    )
    config = Config(
        human_user=user,
        agent_user=user,
        agent_group=group,
        mirror_root=mirror_root,
        log_dir=None,
        mirrors={"local": settings},
    )
    logger = logging.getLogger("sucoder.test.dispatch")
    executor = CommandExecutor(
        human_user=user, agent_user=user, agent_group=group,
        logger=logger, dry_run=False, use_sudo_for_agent=False,
    )
    manager = MirrorManager(config, executor, logger)
    ctx = manager.context_for("local")

    assert not ctx.is_remote

    called = {"human": False, "agent": False}

    def track_human(args, **kwargs):
        called["human"] = True
        return CommandResult(list(args), list(args), "", "", 0)

    def track_agent(args, **kwargs):
        called["agent"] = True
        return CommandResult(list(args), list(args), "", "", 0)

    monkeypatch.setattr(executor, "run_human", track_human)
    monkeypatch.setattr(executor, "run_agent", track_agent)

    manager._run_query(ctx, ["echo", "hi"])
    assert called["human"] is True
    assert called["agent"] is False


def test_run_query_dispatch_remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_run_query calls run_agent for remote contexts."""
    from sucoder.executor import CommandResult

    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")

    assert ctx.is_remote

    called = {"human": False, "agent": False}

    def track_human(args, **kwargs):
        called["human"] = True
        return CommandResult(list(args), list(args), "", "", 0)

    def track_agent(args, **kwargs):
        called["agent"] = True
        return CommandResult(list(args), list(args), "", "", 0)

    monkeypatch.setattr(manager.executor, "run_human", track_human)
    monkeypatch.setattr(manager.executor, "run_agent", track_agent)

    manager._run_query(ctx, ["echo", "hi"])
    assert called["agent"] is True
    assert called["human"] is False
