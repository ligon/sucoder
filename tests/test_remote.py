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
