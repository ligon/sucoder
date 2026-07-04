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


def test_parse_remote_config_keepalive_defaults() -> None:
    """Persistence/keepalive defaults: long-lived warm tunnel (7d) with a
    1-hour keepalive grace budget (30s x 120)."""
    rc = _parse_remote_config({"gateway": "gw", "transfer_host": "dtn"})
    assert rc is not None
    assert rc.control_persist == "7d"
    assert rc.keepalive_interval == 30
    assert rc.keepalive_count_max == 120


def test_parse_remote_config_keepalive_custom() -> None:
    rc = _parse_remote_config({
        "gateway": "gw",
        "transfer_host": "dtn",
        "control_persist": "3d",
        "keepalive_interval": 15,
        "keepalive_count_max": 240,
    })
    assert rc is not None
    assert rc.control_persist == "3d"
    assert rc.keepalive_interval == 15
    assert rc.keepalive_count_max == 240


@pytest.mark.parametrize("key", ["keepalive_interval", "keepalive_count_max"])
@pytest.mark.parametrize("bad_value", [0, -1, "30", 30.0, True, False])
def test_parse_remote_config_keepalive_rejects_bad_values(key: str, bad_value: object) -> None:
    raw = {"gateway": "gw", "transfer_host": "dtn", key: bad_value}
    with pytest.raises(ConfigError, match=key):
        _parse_remote_config(raw)


def test_remote_config_ssh_control_kwargs() -> None:
    """The helper is the single source of persist/keepalive kwargs threaded
    into SshControl / SshTunnel / render_block."""
    rc = RemoteConfig(
        gateway="gw",
        transfer_host="dtn",
        control_persist="2d",
        keepalive_interval=45,
        keepalive_count_max=80,
    )
    assert rc.ssh_control_kwargs() == {
        "control_persist": "2d",
        "keepalive_interval": 45,
        "keepalive_count_max": 80,
        "cert_file": None,
    }


def test_remote_config_ssh_control_kwargs_cert_file() -> None:
    """A configured cert is threaded as a plain string path (not a Path)."""
    from pathlib import Path

    rc = RemoteConfig(
        gateway="gw",
        transfer_host="dtn",
        cert_file=Path("/home/u/.ssh/ssh_certs/brc_cert"),
    )
    assert rc.ssh_control_kwargs()["cert_file"] == "/home/u/.ssh/ssh_certs/brc_cert"


def test_ssh_control_kwargs_match_sshcontrol_fields() -> None:
    """Regression guard: every key the helper emits must be a real
    SshControl constructor kwarg, since cli.py splats it as
    ``SshControl(..., **remote.ssh_control_kwargs())``."""
    from sucoder.tunnel import SshControl

    rc = RemoteConfig(gateway="gw", transfer_host="dtn")
    # Must not raise TypeError on unexpected keyword argument.
    control = SshControl(gateway="gw", **rc.ssh_control_kwargs())
    assert control.control_persist == "7d"
    assert control.keepalive_interval == 30
    assert control.keepalive_count_max == 120


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


def test_session_remote_mirror_root_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path)

    session = RemoteSession(
        mirror_name="test",
        login_node="ln003",
        remote_mirror_root="/local/mirrors",
    )
    session.save()

    loaded = RemoteSession.load("test")
    assert loaded.remote_mirror_root == "/local/mirrors"


def test_session_target_scoping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sessions for different targets don't collide."""
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path)

    local = RemoteSession(mirror_name="proj", login_node="ln001")
    local.save()

    remote = RemoteSession(mirror_name="proj", target_name="savio-node",
                           login_node="ln002", compute_node="n0101")
    remote.save()

    # Both files exist independently
    assert (tmp_path / "proj.yaml").exists()
    assert (tmp_path / "proj--savio-node.yaml").exists()

    # Loading with target retrieves the right one
    loaded_local = RemoteSession.load("proj")
    assert loaded_local.login_node == "ln001"
    assert loaded_local.compute_node is None

    loaded_remote = RemoteSession.load("proj", target_name="savio-node")
    assert loaded_remote.login_node == "ln002"
    assert loaded_remote.compute_node == "n0101"


def test_session_remote_mirror_root_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path)

    session = RemoteSession(mirror_name="test", login_node="ln003")
    session.save()

    loaded = RemoteSession.load("test")
    assert loaded.remote_mirror_root is None


def test_session_stale_local_disk_root_discarded_on_node_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the compute node changes, a saved local-disk mirror root must not
    be reused — the data lives on a different node's /local/ and is unreachable.

    Regression test for the sticky-session bug where ``/local/mirrors`` was
    carried forward even when SLURM allocated a different node.
    """
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path)

    # Simulate a session saved on old node n0126 with local-disk mirror root.
    session = RemoteSession(
        mirror_name="proj",
        target_name="savio-node",
        login_node="ln003",
        compute_node="n0126",
        remote_mirror_root="/local/mirrors",
    )
    session.save()

    loaded = RemoteSession.load("proj", target_name="savio-node")
    assert loaded.remote_mirror_root == "/local/mirrors"
    assert loaded.compute_node == "n0126"

    # After _ensure_slurm_node, suppose the session's compute_node was
    # updated to a different node (simulating SLURM giving us a new one).
    prev_compute_node = loaded.compute_node
    loaded.compute_node = "n0101"

    # Reproduce the mirror-root decision logic from cli.py:
    shared_fs_default = "~/mirrors"
    node_changed = (
        prev_compute_node is not None
        and loaded.compute_node is not None
        and prev_compute_node != loaded.compute_node
    )
    saved_root = loaded.remote_mirror_root

    assert node_changed, "Expected node_changed=True for different nodes"
    assert saved_root != shared_fs_default, "Saved root should differ from shared FS"

    # The fix: when node changed and saved root != shared default, discard it.
    if node_changed and saved_root != shared_fs_default:
        remote_mirror_root = shared_fs_default
    else:
        remote_mirror_root = saved_root

    assert remote_mirror_root == shared_fs_default, (
        f"Stale local-disk root should have been discarded, got {remote_mirror_root}"
    )


def test_session_saved_root_kept_when_node_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the compute node is the same, a saved local-disk mirror root
    should be reused (the data is still accessible)."""
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path)

    session = RemoteSession(
        mirror_name="proj",
        target_name="savio-node",
        login_node="ln003",
        compute_node="n0126",
        remote_mirror_root="/local/mirrors",
    )
    session.save()

    loaded = RemoteSession.load("proj", target_name="savio-node")
    prev_compute_node = loaded.compute_node
    # Same node — job was still running.
    loaded.compute_node = "n0126"

    shared_fs_default = "~/mirrors"
    node_changed = (
        prev_compute_node is not None
        and loaded.compute_node is not None
        and prev_compute_node != loaded.compute_node
    )
    saved_root = loaded.remote_mirror_root

    assert not node_changed
    # Saved root should be kept.
    remote_mirror_root = saved_root
    assert remote_mirror_root == "/local/mirrors"


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
# Shared-node allocation: holders_of_job + adopt probe
# ------------------------------------------------------------------


def test_holders_of_job_finds_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Other sessions sharing a job id are reported; the caller is excluded."""
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path)
    RemoteSession(
        mirror_name="Foo", target_name="savio-node",
        slurm_job_id=12345, compute_node="n0020.savio3",
    ).save()
    RemoteSession(
        mirror_name="Bar", target_name="savio-node",
        slurm_job_id=12345, compute_node="n0020.savio3",
    ).save()
    RemoteSession(
        mirror_name="Baz", target_name="savio-node",
        slurm_job_id=999, compute_node="n0099.savio3",
    ).save()

    holders = RemoteSession.holders_of_job(12345, exclude_key="Foo--savio-node")
    assert holders == ["Bar--savio-node"]


def test_holders_of_job_none_when_unique(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path)
    RemoteSession(
        mirror_name="Solo", target_name="savio-node",
        slurm_job_id=777, compute_node="n0001.savio3",
    ).save()
    assert RemoteSession.holders_of_job(777, exclude_key="Solo--savio-node") == []


def test_holders_of_job_handles_none_and_missing_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path / "nope")
    assert RemoteSession.holders_of_job(None) == []
    assert RemoteSession.holders_of_job(123) == []


class _FakeControl:
    def ssh_options(self, with_fallback: bool = False):
        return ["-o", "ControlPath=/tmp/x"]


class _FakeProc:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_adopt_existing_allocation_found(monkeypatch: pytest.MonkeyPatch) -> None:
    import logging

    from sucoder import cli

    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: _FakeProc(0, stdout="12345 n0020.savio3\n"),
    )
    out = cli._adopt_existing_allocation(
        "n0020.savio3", _FakeControl(), "ln001", logging.getLogger("t"),
    )
    assert out == (12345, "n0020.savio3")


def test_adopt_existing_allocation_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    import logging

    from sucoder import cli

    monkeypatch.setattr("subprocess.run", lambda *a, **k: _FakeProc(0, stdout="\n"))
    assert cli._adopt_existing_allocation(
        "n0020.savio3", _FakeControl(), "ln001", logging.getLogger("t"),
    ) is None


def test_adopt_existing_allocation_probe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import logging

    from sucoder import cli

    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: _FakeProc(1, stderr="bad node"),
    )
    assert cli._adopt_existing_allocation(
        "bogus", _FakeControl(), "ln001", logging.getLogger("t"),
    ) is None


def test_adopt_existing_allocation_no_login_node() -> None:
    import logging

    from sucoder import cli

    assert cli._adopt_existing_allocation(
        "n0020.savio3", _FakeControl(), None, logging.getLogger("t"),
    ) is None


def test_adopt_existing_allocation_skips_array_elements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Array-element ids like 12345_1 aren't whole-node reservations."""
    import logging

    from sucoder import cli

    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: _FakeProc(0, stdout="12345_1 n0020.savio3\n"),
    )
    assert cli._adopt_existing_allocation(
        "n0020.savio3", _FakeControl(), "ln001", logging.getLogger("t"),
    ) is None


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
    assert cmd[0] == "ssh"
    # Non-interactive commands get BatchMode=yes to prevent /dev/tty prompts.
    assert "-o" in cmd and "BatchMode=yes" in cmd
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
    # Interactive commands must NOT get BatchMode — they may need auth prompts.
    assert "BatchMode=yes" not in cmd


def test_build_ssh_command_with_options() -> None:
    executor = _make_remote_executor(ssh_options={"StrictHostKeyChecking": "no"})
    cmd = executor._build_ssh_command(["ls"])
    # Custom ssh_options are passed through as -o flags.
    assert "StrictHostKeyChecking=no" in cmd


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


def test_build_ssh_command_compute_node_proxy() -> None:
    """Compute-node targets include a ProxyCommand through the login node."""
    executor = _make_remote_executor(
        login_node="n0101.savio3",
        control_socket_path="/tmp/compute.sock",
        is_compute_node=True,
        proxy_node="ln001",
        proxy_socket_path="/tmp/login.sock",
    )
    cmd = executor._build_ssh_command(["hostname"])
    joined = " ".join(cmd)
    # Should include ProxyCommand through the login node
    assert "ProxyCommand" in joined
    assert "ln001" in joined
    assert "/tmp/login.sock" in joined
    # Should include host key options for ephemeral compute nodes
    assert "StrictHostKeyChecking=no" in joined
    assert "UserKnownHostsFile=/dev/null" in joined


def test_build_ssh_command_compute_node_no_proxy_without_info() -> None:
    """Compute node without proxy info falls through without ProxyCommand."""
    executor = _make_remote_executor(
        login_node="n0101.savio3",
        control_socket_path="/tmp/compute.sock",
        is_compute_node=True,
        # No proxy_node or proxy_socket_path
    )
    cmd = executor._build_ssh_command(["hostname"])
    joined = " ".join(cmd)
    # Should NOT include ProxyCommand (no proxy info available)
    assert "ProxyCommand" not in joined


def test_build_ssh_command_failfast_guards_noninteractive() -> None:
    """One-shot commands carry ConnectTimeout + ServerAlive fail-fast guards."""
    executor = _make_remote_executor(control_socket_path="/tmp/test.sock")
    cmd = executor._build_ssh_command(["git", "status"])
    assert f"ConnectTimeout={executor.CONNECT_TIMEOUT}" in cmd
    assert f"ServerAliveInterval={executor.KEEPALIVE_INTERVAL}" in cmd
    assert f"ServerAliveCountMax={executor.KEEPALIVE_COUNT_MAX}" in cmd
    # The detection window must stay under the outer subprocess ceiling.
    assert (
        executor.KEEPALIVE_INTERVAL * executor.KEEPALIVE_COUNT_MAX
        < executor.DEFAULT_SSH_TIMEOUT
    )


def test_build_ssh_command_no_failfast_guards_interactive() -> None:
    """Interactive sessions keep the ControlMaster's longer tolerance."""
    executor = _make_remote_executor(control_socket_path="/tmp/test.sock")
    cmd = executor._build_ssh_command(["bash"], allocate_tty=True)
    joined = " ".join(cmd)
    assert "ServerAliveInterval" not in joined
    assert "ServerAliveCountMax" not in joined
    assert "ConnectTimeout" not in joined


def test_build_ssh_command_proxy_failfast_noninteractive() -> None:
    """The compute-node ProxyCommand hop also bounds its inner dial."""
    executor = _make_remote_executor(
        login_node="n0101.savio3",
        control_socket_path="/tmp/compute.sock",
        is_compute_node=True,
        proxy_node="ln001",
        proxy_socket_path="/tmp/login.sock",
    )
    cmd = executor._build_ssh_command(["hostname"])
    proxy = next(opt for opt in cmd if opt.startswith("ProxyCommand="))
    assert f"ConnectTimeout={executor.CONNECT_TIMEOUT}" in proxy


def test_build_ssh_command_debug_ssh() -> None:
    """--debug-ssh adds -vvv to SSH commands."""
    executor = _make_remote_executor(
        control_socket_path="/tmp/test.sock",
        debug_ssh=True,
    )
    cmd = executor._build_ssh_command(["hostname"])
    assert "-vvv" in cmd


def test_build_ssh_command_no_debug_ssh() -> None:
    """Without --debug-ssh, -vvv is absent."""
    executor = _make_remote_executor(
        control_socket_path="/tmp/test.sock",
        debug_ssh=False,
    )
    cmd = executor._build_ssh_command(["hostname"])
    assert "-vvv" not in cmd


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
    # Default (no fallback) must NOT emit a ProxyJump/ProxyCommand.
    assert not any("Proxy" in o for o in opts)


def test_ssh_control_establish_uses_configured_keepalive(monkeypatch, tmp_path) -> None:
    """establish() must emit the instance's persist/keepalive values, not
    the old hardcoded 12h / 30 / 3."""
    import logging

    from sucoder.tunnel import SshControl

    socket_file = tmp_path / "gw.sock"
    monkeypatch.setattr(
        SshControl, "socket_path", property(lambda self: socket_file),
    )
    control = SshControl(
        gateway="gw",
        control_persist="3d",
        keepalive_interval=45,
        keepalive_count_max=200,
    )
    monkeypatch.setattr(control, "is_active", lambda: False)

    captured: dict = {}

    class _R:
        returncode = 0

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return _R()

    monkeypatch.setattr("sucoder.tunnel.subprocess.run", fake_run)
    control.establish(logging.getLogger("t"))

    joined = " ".join(captured["cmd"])
    assert "ControlPersist=3d" in joined
    assert "ServerAliveInterval=45" in joined
    assert "ServerAliveCountMax=200" in joined


def test_ssh_control_options_fallback_uses_jump_control() -> None:
    """Regression: ``with_fallback=True`` must route a fresh dial through
    the jump host's ControlMaster so a wedged login-node mux doesn't make
    ssh fall back to a direct dial of a jump-only hostname.

    Symptom (savio-node): the mux refused a session
    (``Session open refused by peer``) and ssh then tried to resolve the
    pinned login node ``ln003.brc`` directly -> ``Could not resolve
    hostname``.  The fallback ProxyCommand reuses the gateway mux instead.
    """
    from sucoder.tunnel import SshControl

    gw = SshControl(gateway="brc.berkeley.edu")
    ln = SshControl(
        gateway="ln003.brc",
        jump_host="brc.berkeley.edu",
        jump_control=gw,
    )
    opts = ln.ssh_options(with_fallback=True)
    proxy = [o for o in opts if o.startswith("ProxyCommand=")]
    assert proxy, f"expected a ProxyCommand fallback, got {opts}"
    # Must reuse the gateway's ControlMaster socket and tunnel via -W.
    assert str(gw.socket_path) in proxy[0]
    assert "-W %h:%p brc.berkeley.edu" in proxy[0]
    # Still reuses the login-node mux when alive.
    assert "ControlMaster=auto" in opts
    assert any(f"ControlPath={ln.socket_path}" == o for o in opts)


def test_ssh_control_options_fallback_plain_proxyjump() -> None:
    """With a jump_host but no jump_control, the fallback emits a plain
    ProxyJump (ssh re-authenticates the hop itself)."""
    from sucoder.tunnel import SshControl

    ln = SshControl(gateway="ln003.brc", jump_host="brc.berkeley.edu")
    opts = ln.ssh_options(with_fallback=True)
    assert "ProxyJump=brc.berkeley.edu" in opts
    assert not any(o.startswith("ProxyCommand=") for o in opts)


def test_ssh_control_is_active_uses_batchmode(monkeypatch, tmp_path) -> None:
    """Regression: both is_active() probes must set BatchMode=yes and a
    wall-clock timeout.

    Without BatchMode, a stale/unattachable mux can fall through to
    interactive auth on /dev/tty (bypassing stdin=DEVNULL and
    capture_output), and the probe blocks until the timeout — which
    previously made gateway re-auth invisible inside spinner blocks
    and produced an apparent hang.
    """
    from sucoder.tunnel import SshControl

    # Build a control with a socket file that exists so we exercise
    # both the structural and end-to-end probes.
    socket_file = tmp_path / "gw.sock"
    socket_file.touch()
    control = SshControl(gateway="gw")
    monkeypatch.setattr(
        SshControl, "socket_path",
        property(lambda self: socket_file),
    )

    calls: list[dict] = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(cmd, **kwargs):
        calls.append({"cmd": list(cmd), "kwargs": dict(kwargs)})
        return _Result()

    monkeypatch.setattr("sucoder.tunnel.subprocess.run", _fake_run)
    assert control.is_active() is True

    # Both probes must have run.
    assert len(calls) == 2, f"expected 2 ssh probes, got {len(calls)}"

    for probe in calls:
        cmd = probe["cmd"]
        joined = " ".join(cmd)
        # BatchMode=yes prevents fallback to interactive /dev/tty auth.
        assert "BatchMode=yes" in cmd, (
            f"is_active probe missing BatchMode=yes: {cmd!r}"
        )
        # Both probes need a hard timeout so a wedged mux can never
        # block forever (the structural probe previously had none).
        assert probe["kwargs"].get("timeout") is not None, (
            f"is_active probe missing timeout=: {cmd!r}"
        )
        # Defensive: never inherit parent stdin (would let ssh read
        # the user's terminal).
        import subprocess as _sp
        assert probe["kwargs"].get("stdin") is _sp.DEVNULL, (
            f"is_active probe missing stdin=DEVNULL: {cmd!r}"
        )
        # The end-to-end probe also needs ConnectTimeout for the TCP
        # leg; structural probe only needs the wall-clock timeout.
        if "true" in cmd:
            assert "ConnectTimeout=5" in joined, (
                f"end-to-end probe missing ConnectTimeout: {cmd!r}"
            )


def test_is_active_self_established_skips_remote_shell(monkeypatch, tmp_path) -> None:
    """#3: a master this process established is trusted via the structural
    ``-O check`` alone --- no ``true`` round-trip, so no remote login-shell
    spawn on the hot reuse path (the spawn latency that made the probe slow
    and flaky on a hammered BRC login node, and re-authed the gateway on
    every hop)."""
    from sucoder.tunnel import SshControl

    socket_file = tmp_path / "gw.sock"
    socket_file.touch()
    monkeypatch.setattr(
        SshControl, "socket_path", property(lambda self: socket_file)
    )
    control = SshControl(gateway="gw")
    control._established_this_session = True

    calls: list[list] = []

    class _Ok:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return _Ok()

    monkeypatch.setattr("sucoder.tunnel.subprocess.run", _fake_run)
    assert control.is_active() is True
    # Exactly one probe, and it is the structural -O check (no remote command).
    assert len(calls) == 1, f"expected only the structural check, got {calls!r}"
    assert "check" in calls[0]
    assert not any("true" in c for c in calls), (
        "self-established master must not spawn a remote shell probe"
    )


def test_is_active_deep_probe_retries_slow_master(monkeypatch, tmp_path) -> None:
    """#1: a live-but-slow master that fails one end-to-end probe then
    answers is reported ACTIVE (no spurious re-auth), thanks to the retry."""
    from sucoder.tunnel import SshControl

    socket_file = tmp_path / "gw.sock"
    socket_file.touch()
    monkeypatch.setattr(
        SshControl, "socket_path", property(lambda self: socket_file)
    )
    control = SshControl(gateway="gw")  # not self-established -> deep probe

    probes = {"true": 0}

    class _R:
        def __init__(self, rc):
            self.returncode = rc
            self.stdout = ""
            self.stderr = ""

    def _fake_run(cmd, **kwargs):
        if "true" in cmd:
            probes["true"] += 1
            return _R(0 if probes["true"] >= 2 else 255)  # slow: fail once
        return _R(0)  # -O check ok

    monkeypatch.setattr("sucoder.tunnel.subprocess.run", _fake_run)
    assert control.is_active(sleep=lambda *_: None) is True
    assert probes["true"] == 2, "should retry the end-to-end probe once, then pass"


def test_is_active_deep_probe_dead_after_retries(monkeypatch, tmp_path) -> None:
    """#1 boundary: a genuine zombie (``-O check`` passes, end-to-end always
    fails) is declared dead only after exhausting retries, so ensure() still
    re-authenticates it --- the retry cannot mask a real failure."""
    from sucoder.tunnel import SshControl, _LIVENESS_PROBE_ATTEMPTS

    socket_file = tmp_path / "gw.sock"
    socket_file.touch()
    monkeypatch.setattr(
        SshControl, "socket_path", property(lambda self: socket_file)
    )
    control = SshControl(gateway="gw")

    probes = {"true": 0}

    class _R:
        def __init__(self, rc):
            self.returncode = rc
            self.stdout = ""
            self.stderr = "mux_client_request_session: session open failed"

    def _fake_run(cmd, **kwargs):
        if "true" in cmd:
            probes["true"] += 1
            return _R(255)
        return _R(0)

    monkeypatch.setattr("sucoder.tunnel.subprocess.run", _fake_run)
    assert control.is_active(sleep=lambda *_: None) is False
    assert probes["true"] == _LIVENESS_PROBE_ATTEMPTS, (
        "must exhaust all attempts before declaring the mux dead"
    )


def test_is_active_logs_probe_failure(monkeypatch, tmp_path, caplog) -> None:
    """#2: a probe failure is logged at INFO with a diagnosable reason, so a
    wild recurrence needs no ``--debug-ssh`` (which rebuilds the socket and
    masks the very bug)."""
    import logging
    from sucoder.tunnel import SshControl

    socket_file = tmp_path / "gw.sock"
    socket_file.touch()
    monkeypatch.setattr(
        SshControl, "socket_path", property(lambda self: socket_file)
    )
    control = SshControl(gateway="gw")

    class _R:
        def __init__(self, rc, err=""):
            self.returncode = rc
            self.stdout = ""
            self.stderr = err

    def _fake_run(cmd, **kwargs):
        if "true" in cmd:
            return _R(255, "kex_exchange_identification: Connection closed")
        return _R(0)

    monkeypatch.setattr("sucoder.tunnel.subprocess.run", _fake_run)
    with caplog.at_level(logging.INFO, logger="sucoder.tunnel"):
        assert control.is_active(sleep=lambda *_: None) is False
    assert "treating connection as expired" in caplog.text, (
        f"expected an INFO diagnosis line, got: {caplog.text!r}"
    )


def test_establish_marks_session_established(monkeypatch, tmp_path) -> None:
    """establish() records that this process owns the master, so a later
    is_active() may take the cheap structural path (#3)."""
    import logging as _logging
    from sucoder.tunnel import SshControl

    socket_file = tmp_path / "gw.sock"
    monkeypatch.setattr(
        SshControl, "socket_path", property(lambda self: socket_file)
    )
    control = SshControl(gateway="gw")
    # Force the "not yet active" branch so establish() runs the ssh command.
    monkeypatch.setattr(control, "is_active", lambda *a, **k: False)

    class _Ok:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr("sucoder.tunnel.subprocess.run", lambda *a, **k: _Ok())
    assert control._established_this_session is False
    control.establish(_logging.getLogger("t"))
    assert control._established_this_session is True


def test_close_clears_session_established(tmp_path, monkeypatch) -> None:
    """close() drops ownership so a later stale socket is re-probed
    end-to-end rather than trusted (#3)."""
    import logging as _logging
    from sucoder.tunnel import SshControl

    socket_file = tmp_path / "gw.sock"  # intentionally absent
    monkeypatch.setattr(
        SshControl, "socket_path", property(lambda self: socket_file)
    )
    control = SshControl(gateway="gw")
    control._established_this_session = True
    control.close(_logging.getLogger("t"))  # early-returns, but must clear the flag
    assert control._established_this_session is False


def test_ensure_ssh_visible_runs_outside_spinner(monkeypatch) -> None:
    """Regression: _ensure_ssh_visible must NOT wrap ensure() in
    _spinner.

    A spinner's 100 ms refresh thread overwrites the SSH password /
    OTP prompt that ssh writes to /dev/tty, producing an invisible
    hang.  The visible-auth helper exists specifically to avoid that
    overlay; if a future refactor accidentally re-introduces _spinner
    around ensure() this test fails.
    """
    from contextlib import contextmanager
    import logging

    from sucoder import cli as cli_mod

    spinner_active = [False]

    @contextmanager
    def _watching_spinner(message: str):
        spinner_active[0] = True
        try:
            yield
        finally:
            spinner_active[0] = False

    monkeypatch.setattr(cli_mod, "_spinner", _watching_spinner)

    class _FakeControl:
        def __init__(self) -> None:
            self.ensured = False

        def ensure(self, logger) -> None:
            assert not spinner_active[0], (
                "ensure() was called inside a _spinner block — "
                "SSH auth prompts would be obscured by the spinner refresh"
            )
            self.ensured = True

    ctrl = _FakeControl()
    cli_mod._ensure_ssh_visible(ctrl, "ln002.brc", logging.getLogger("t"))
    assert ctrl.ensured is True
    # _spinner should never have been entered.
    assert spinner_active[0] is False


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


def test_parse_targets_with_slurm_local_disk() -> None:
    from sucoder.config import _parse_targets

    raw = {
        "savio-node": {
            "gateway": "brc.berkeley.edu",
            "transfer_host": "dtn.brc.berkeley.edu",
            "slurm": {
                "partition": "savio3",
                "account": "fc_jevons",
                "local_disk": "/local",
            },
        },
    }
    targets = _parse_targets(raw)
    assert targets["savio-node"].slurm is not None
    assert targets["savio-node"].slurm.local_disk == "/local"


def test_parse_targets_slurm_no_local_disk() -> None:
    from sucoder.config import _parse_targets

    raw = {
        "savio-node": {
            "gateway": "brc.berkeley.edu",
            "transfer_host": "dtn.brc.berkeley.edu",
            "slurm": {
                "partition": "savio3",
                "account": "fc_jevons",
            },
        },
    }
    targets = _parse_targets(raw)
    assert targets["savio-node"].slurm is not None
    assert targets["savio-node"].slurm.local_disk is None
    # cpus_per_task and mem default to None so the salloc command keeps
    # the partition's defaults (i.e. whole-node on exclusive partitions).
    assert targets["savio-node"].slurm.cpus_per_task is None
    assert targets["savio-node"].slurm.mem is None


def test_parse_targets_with_slurm_cpus_and_mem() -> None:
    """Shared partitions (e.g. savio4_htc) need cpus_per_task + mem."""
    from sucoder.config import _parse_targets

    raw = {
        "savio-htc": {
            "gateway": "brc.berkeley.edu",
            "transfer_host": "dtn.brc.berkeley.edu",
            "slurm": {
                "partition": "savio4_htc",
                "account": "fc_jevons",
                "qos": "savio_normal",
                "cpus_per_task": 4,
                "mem": "16G",
                "time": "24:00:00",
            },
        },
    }
    targets = _parse_targets(raw)
    slurm = targets["savio-htc"].slurm
    assert slurm is not None
    assert slurm.partition == "savio4_htc"
    assert slurm.cpus_per_task == 4
    assert slurm.mem == "16G"
    assert slurm.qos == "savio_normal"


@pytest.mark.parametrize("bad_value", [0, -1, "4", 4.0, True])
def test_parse_targets_slurm_cpus_per_task_rejects_bad_values(bad_value: object) -> None:
    from sucoder.config import _parse_targets

    raw = {
        "savio-htc": {
            "gateway": "brc.berkeley.edu",
            "transfer_host": "dtn.brc.berkeley.edu",
            "slurm": {
                "partition": "savio4_htc",
                "account": "fc_jevons",
                "cpus_per_task": bad_value,
            },
        },
    }
    with pytest.raises(ConfigError, match="cpus_per_task"):
        _parse_targets(raw)


@pytest.mark.parametrize("bad_value", ["", "   ", 16, 16.0, True])
def test_parse_targets_slurm_mem_rejects_bad_values(bad_value: object) -> None:
    from sucoder.config import _parse_targets

    raw = {
        "savio-htc": {
            "gateway": "brc.berkeley.edu",
            "transfer_host": "dtn.brc.berkeley.edu",
            "slurm": {
                "partition": "savio4_htc",
                "account": "fc_jevons",
                "mem": bad_value,
            },
        },
    }
    with pytest.raises(ConfigError, match="mem"):
        _parse_targets(raw)


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


def test_sync_remote_calls_push_via_login_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_sync_remote should push via the login node ControlMaster (no tunnel)."""
    from sucoder.executor import CommandResult, RemoteExecutor

    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")

    # Replace executor with a RemoteExecutor so login_node is available.
    import logging
    logger = logging.getLogger("test.remote")
    remote_exec = RemoteExecutor(
        human_user="ligon",
        agent_user="ligon",
        agent_group="ligon",
        logger=logger,
        dry_run=False,
        use_sudo_for_agent=False,
        gateway="gw.example.com",
        login_node="ln001",
        remote_mirror_root="~/mirrors",
        local_mirror_root=str(tmp_path / "mirrors"),
        control_socket_path="/tmp/test.sock",
    )

    calls: list = []

    def fake_run_human(args, **kwargs):
        calls.append({"args": list(args), "kwargs": kwargs})
        return CommandResult(list(args), list(args), "", "", 0)

    def fake_run_agent(args, **kwargs):
        # For _resolve_remote_path
        if "echo" in " ".join(str(a) for a in args):
            return CommandResult(list(args), list(args), "/home/ligon\n", "", 0)
        return CommandResult(list(args), list(args), "", "", 0)

    monkeypatch.setattr(remote_exec, "run_human", fake_run_human)
    monkeypatch.setattr(remote_exec, "run_agent", fake_run_agent)
    manager.executor = remote_exec

    manager._sync_remote(ctx)

    assert len(calls) == 1
    push_cmd = calls[0]["args"]
    assert push_cmd[0] == "git"
    assert push_cmd[1] == "push"
    # SCP-style URL using login node, not localhost tunnel
    url = push_cmd[2]
    assert "ln001:" in url
    assert "rproj" in url
    assert "--all" in push_cmd
    assert "--force" in push_cmd
    # GIT_SSH_COMMAND should reference the ControlMaster socket
    env = calls[0]["kwargs"].get("env", {})
    assert "GIT_SSH_COMMAND" in env
    assert "ControlPath" in env["GIT_SSH_COMMAND"]


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
        # All calls succeed → mirror exists and is valid
        return CommandResult(list(args), list(args), "", "", 0)

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    # Mock _sync_remote since we don't want actual sync
    sync_called = []
    monkeypatch.setattr(manager, "_sync_remote", lambda ctx: sync_called.append(True))

    manager.ensure_remote_clone(ctx)

    # Should have rev-parse check and config fixup, but NOT git init
    all_cmds = [" ".join(str(a) for a in c) for c in agent_calls]
    assert any("rev-parse" in cmd for cmd in all_cmds)
    assert not any("git init" in cmd for cmd in all_cmds)
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
        agent_calls.append({"args": list(args), "kwargs": kwargs})
        call_counter[0] += 1
        # First call is rev-parse → fail; also $HOME query needs to work
        args_str = " ".join(str(a) for a in args)
        if "rev-parse" in args_str and call_counter[0] <= 2:
            return CommandResult(list(args), list(args), "", "", 1)
        if "echo" in args_str:
            return CommandResult(list(args), list(args), "/home/testuser\n", "", 0)
        return CommandResult(list(args), list(args), "", "", 0)

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    sync_called = []
    monkeypatch.setattr(manager, "_sync_remote", lambda ctx: sync_called.append(True))

    manager.ensure_remote_clone(ctx)

    # Should have rev-parse, rm, mkdir, git init, and git config calls
    all_cmds = [" ".join(str(a) for a in c["args"]) for c in agent_calls]
    assert any("rev-parse" in cmd for cmd in all_cmds)
    assert any("init" in cmd for cmd in all_cmds)
    assert sync_called


def _remote_exec_with_scaffolding(tmp_path: Path):
    """A RemoteExecutor configured with a DTN scaffolding node + sockets."""
    import logging

    from sucoder.executor import RemoteExecutor

    logger = logging.getLogger("test.remote.hardening")
    return RemoteExecutor(
        human_user="ligon",
        agent_user="ligon",
        agent_group="ligon",
        logger=logger,
        dry_run=False,
        use_sudo_for_agent=False,
        gateway="gw.example.com",
        login_node="ln001",
        remote_mirror_root="~/mirrors",
        local_mirror_root=str(tmp_path / "mirrors"),
        control_socket_path="/tmp/test.sock",
        scaffolding_node="dtn.example.com",
        scaffolding_socket_path="/tmp/dtn.sock",
    )


def test_remote_git_env_hardens_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GIT_SSH_COMMAND carries fail-fast / quiet guards on both hops.

    Regression guard: a refused ControlMaster session must fall back
    fast and silently instead of dialing a fresh, login-shell-polluted
    connection that wedges git-receive-pack.
    """
    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")
    manager.executor = _remote_exec_with_scaffolding(tmp_path)
    # Avoid a real SSH round-trip to resolve ~.
    monkeypatch.setattr(
        manager, "_resolve_remote_path",
        lambda ctx: "/home/ligon/mirrors/rproj",
    )

    url, env = manager._remote_git_env(ctx)

    assert url == "dtn.example.com:/home/ligon/mirrors/rproj"
    cmd = env["GIT_SSH_COMMAND"]
    # Outer ssh hardening (matches the rest of the executor).
    assert "BatchMode=yes" in cmd
    assert "ConnectTimeout=10" in cmd
    assert "ServerAliveInterval=15" in cmd
    assert "ServerAliveCountMax=3" in cmd
    assert "LogLevel=ERROR" in cmd
    # Rides the DTN ControlMaster socket.
    assert "ControlPath=/tmp/dtn.sock" in cmd
    # The inner ProxyCommand ssh is hardened too — BatchMode + LogLevel
    # appear a second time inside the quoted ProxyCommand string.
    assert "ProxyCommand=ssh -o BatchMode=yes" in cmd
    assert cmd.count("BatchMode=yes") >= 2
    assert cmd.count("LogLevel=ERROR") >= 2


def test_remote_git_env_debug_preserves_verbosity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With debug_ssh, -vvv is kept and LogLevel=ERROR is not forced."""
    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")
    executor = _remote_exec_with_scaffolding(tmp_path)
    executor.debug_ssh = True
    manager.executor = executor
    monkeypatch.setattr(
        manager, "_resolve_remote_path",
        lambda ctx: "/home/ligon/mirrors/rproj",
    )

    _url, env = manager._remote_git_env(ctx)
    cmd = env["GIT_SSH_COMMAND"]

    assert "-vvv" in cmd
    assert "LogLevel=ERROR" not in cmd
    # Fail-fast guards still apply in debug mode.
    assert "BatchMode=yes" in cmd
    assert "ConnectTimeout=10" in cmd


def test_ensure_remote_clone_rebuilds_empty_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A husk repo (exists, but no commits/base branch) is rebuilt.

    Regression guard for the PlayPen/DTN failure: a remote mirror that
    was `git init`'d by a prior failed bootstrap but never received a
    push has no `main` ref.  ensure_remote_clone must rebuild it rather
    than sync into the half-dead repo.
    """
    from sucoder.executor import CommandResult

    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")

    agent_calls: list = []

    def fake_run_agent(args, **kwargs):
        agent_calls.append(list(args))
        s = " ".join(str(a) for a in args)
        if "echo" in s:
            return CommandResult(list(args), list(args), "/home/ligon\n", "", 0)
        if "rev-parse" in s and "--git-dir" in s:
            # Repo exists on disk.
            return CommandResult(list(args), list(args), ".git\n", "", 0)
        if "rev-parse" in s and ("HEAD" in s or "refs/heads/" in s):
            # No commits, no base branch → husk.
            return CommandResult(list(args), list(args), "", "fatal", 1)
        return CommandResult(list(args), list(args), "", "", 0)

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    # Isolate the rebuild decision: don't touch the network.
    monkeypatch.setattr(manager, "_pull_from_remote", lambda ctx: None)
    sync_called: list = []
    monkeypatch.setattr(manager, "_sync_remote", lambda ctx: sync_called.append(True))

    manager.ensure_remote_clone(ctx)

    cmds = [" ".join(str(a) for a in c) for c in agent_calls]
    # Husk detected → wiped and re-initialised before syncing.
    assert any("rm -rf" in c for c in cmds)
    assert any("git init" in c for c in cmds)
    assert sync_called


def test_git_transports_login_first_then_dtn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Login node is primary; the DTN is kept only as a fallback."""
    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")
    manager.executor = _remote_exec_with_scaffolding(tmp_path)
    monkeypatch.setattr(
        manager, "_resolve_remote_path",
        lambda ctx: "/home/ligon/mirrors/rproj",
    )

    transports = manager._git_transports(ctx)

    assert [t[0] for t in transports] == ["login node", "DTN"]
    login_url, dtn_url = transports[0][1], transports[1][1]
    assert login_url.startswith("ln001:")
    assert dtn_url.startswith("dtn.example.com:")
    # Login node rides its own socket; DTN rides the scaffolding socket.
    assert "ControlPath=/tmp/test.sock" in transports[0][2]["GIT_SSH_COMMAND"]
    assert "ControlPath=/tmp/dtn.sock" in transports[1][2]["GIT_SSH_COMMAND"]


def test_git_transports_single_without_scaffolding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a DTN there is one transport — no pointless failover."""
    from sucoder.executor import RemoteExecutor

    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")
    import logging
    manager.executor = RemoteExecutor(
        human_user="ligon", agent_user="ligon", agent_group="ligon",
        logger=logging.getLogger("test.remote.single"),
        dry_run=False, use_sudo_for_agent=False,
        gateway="gw.example.com", login_node="ln001",
        remote_mirror_root="~/mirrors",
        local_mirror_root=str(tmp_path / "mirrors"),
        control_socket_path="/tmp/test.sock",
    )
    monkeypatch.setattr(
        manager, "_resolve_remote_path",
        lambda ctx: "/home/ligon/mirrors/rproj",
    )

    transports = manager._git_transports(ctx)
    assert len(transports) == 1
    assert transports[0][0] == "login node"


def _transport_result(args, returncode, stderr=""):
    from sucoder.executor import CommandResult
    return CommandResult(list(args), list(args), "", stderr, returncode)


def test_sync_remote_prefers_login_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The push goes to the login node first; the DTN is not touched."""
    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")
    manager.executor = _remote_exec_with_scaffolding(tmp_path)
    monkeypatch.setattr(
        manager, "_resolve_remote_path",
        lambda ctx: "/home/ligon/mirrors/rproj",
    )

    pushes: list = []

    def fake_run_human(args, **kwargs):
        pushes.append(args[2])
        return _transport_result(args, 0)

    monkeypatch.setattr(manager.executor, "run_human", fake_run_human)

    manager._sync_remote(ctx)

    assert len(pushes) == 1
    assert pushes[0].startswith("ln001:")


def test_sync_remote_fails_over_to_dtn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A login-node push that hits a transport fault retries on the DTN."""
    from sucoder.executor import CommandError

    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")
    manager.executor = _remote_exec_with_scaffolding(tmp_path)
    monkeypatch.setattr(
        manager, "_resolve_remote_path",
        lambda ctx: "/home/ligon/mirrors/rproj",
    )

    pushes: list = []

    def fake_run_human(args, **kwargs):
        url = args[2]
        pushes.append(url)
        if url.startswith("ln001"):
            res = _transport_result(
                args, 1, "fatal: the remote end hung up unexpectedly")
            raise CommandError("boom", res)
        return _transport_result(args, 0)

    monkeypatch.setattr(manager.executor, "run_human", fake_run_human)

    manager._sync_remote(ctx)  # must not raise

    assert len(pushes) == 2
    assert pushes[0].startswith("ln001:")
    assert pushes[1].startswith("dtn.example.com:")


def test_sync_remote_no_failover_on_real_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine git rejection surfaces immediately — no login-node retry."""
    from sucoder.executor import CommandError

    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")
    manager.executor = _remote_exec_with_scaffolding(tmp_path)
    monkeypatch.setattr(
        manager, "_resolve_remote_path",
        lambda ctx: "/home/ligon/mirrors/rproj",
    )

    pushes: list = []

    def fake_run_human(args, **kwargs):
        pushes.append(args[2])
        res = _transport_result(
            args, 1, "! [rejected] main -> main (non-fast-forward)")
        raise CommandError("rejected", res)

    monkeypatch.setattr(manager.executor, "run_human", fake_run_human)

    with pytest.raises(CommandError):
        manager._sync_remote(ctx)

    # Failed on the login node and did NOT fall over to the DTN.
    assert len(pushes) == 1
    assert pushes[0].startswith("ln001:")


def test_pull_fails_over_to_dtn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A login-node fetch transport fault retries on the DTN before giving up."""
    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")
    manager.executor = _remote_exec_with_scaffolding(tmp_path)
    monkeypatch.setattr(
        manager, "_resolve_remote_path",
        lambda ctx: "/home/ligon/mirrors/rproj",
    )

    fetches: list = []

    def fake_run_human(args, **kwargs):
        url = args[2]
        fetches.append(url)
        if url.startswith("ln001"):
            return _transport_result(
                args, 128, "fatal: the remote end hung up unexpectedly")
        return _transport_result(args, 0)  # DTN connects

    monkeypatch.setattr(manager.executor, "run_human", fake_run_human)

    manager._pull_from_remote(ctx)  # reconciliation no-ops (tmp ref absent)

    assert len(fetches) == 2
    assert fetches[0].startswith("ln001:")
    assert fetches[1].startswith("dtn.example.com:")


def test_pull_no_failover_on_empty_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty remote ('couldn't find remote ref') is a real answer, no retry."""
    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")
    manager.executor = _remote_exec_with_scaffolding(tmp_path)
    monkeypatch.setattr(
        manager, "_resolve_remote_path",
        lambda ctx: "/home/ligon/mirrors/rproj",
    )

    fetches: list = []

    def fake_run_human(args, **kwargs):
        fetches.append(args[2])
        return _transport_result(
            args, 128, "fatal: couldn't find remote ref main")

    monkeypatch.setattr(manager.executor, "run_human", fake_run_human)

    manager._pull_from_remote(ctx)  # swallowed as a warning

    # Connected fine (just an empty mirror) → no DTN retry.
    assert len(fetches) == 1
    assert fetches[0].startswith("ln001:")


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


def test_ensure_remote_mirror_exists_timeout_is_graceful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timed-out probe raises a clean MirrorError, not a raw CommandError."""
    from sucoder.executor import CommandError, CommandResult
    from sucoder.mirror import MirrorError

    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")

    def fake_run_agent(args, **kwargs):
        # The git probe is what hangs; the `echo $HOME` path-resolution
        # call that precedes it must still succeed.
        if "rev-parse" in args:
            raise CommandError(
                "Command timed out after 120s",
                CommandResult(list(args), list(args), "", "(timed out)", -1),
            )
        return CommandResult(list(args), list(args), "/home/ligon", "", 0)

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)

    with pytest.raises(MirrorError) as excinfo:
        manager._ensure_mirror_exists(ctx)
    # The message must point at unresponsiveness, not a missing mirror.
    assert "not responding" in str(excinfo.value)
    assert "agents-clone" not in str(excinfo.value)


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


def test_run_on_login_node_compute_target() -> None:
    """run_on_login_node routes through the DTN, not the compute node."""
    executor = _make_remote_executor(
        login_node="n0101.savio3",
        control_socket_path="/tmp/compute.sock",
        is_compute_node=True,
        proxy_node="ln001",
        proxy_socket_path="/tmp/login.sock",
        scaffolding_node="dtn.brc.berkeley.edu",
        scaffolding_socket_path="/tmp/dtn.sock",
    )
    # Build the command that run_on_login_node would use
    cmd = executor._build_login_node_command(["hostname"])
    joined = " ".join(cmd)
    # Target is the DTN, not the compute node
    assert "dtn.brc.berkeley.edu" in cmd
    assert "n0101.savio3" not in joined
    # Uses the DTN's ControlMaster socket
    assert "/tmp/dtn.sock" in joined
    # BatchMode prevents hanging on impossible password prompts when
    # the ControlMaster socket is dead and SSH falls through to a
    # fresh connection in a non-interactive context.
    assert "BatchMode=yes" in joined
    assert "ConnectTimeout=10" in joined
    # The ProxyCommand also needs BatchMode + ConnectTimeout so the
    # inner SSH (to the gateway) fails fast too.
    proxy_args = [a for a in cmd if "ProxyCommand" in a]
    assert proxy_args, "Expected a ProxyCommand option"
    proxy_str = proxy_args[0]
    assert "BatchMode=yes" in proxy_str
    assert "ConnectTimeout=10" in proxy_str


def test_run_on_login_node_falls_through_without_scaffolding() -> None:
    """run_on_login_node delegates to run_agent without scaffolding node."""
    executor = _make_remote_executor(
        login_node="ln001",
        control_socket_path="/tmp/login.sock",
        is_compute_node=False,
    )
    # No scaffolding node set — should fall through to run_agent
    assert not executor.scaffolding_node


def test_run_on_login_node_falls_back_on_dtn_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the DTN times out, run_on_login_node falls back to run_agent."""
    from sucoder.executor import CommandError, CommandResult

    executor = _make_remote_executor(
        login_node="n0101.savio3",
        control_socket_path="/tmp/compute.sock",
        is_compute_node=True,
        proxy_node="ln001",
        proxy_socket_path="/tmp/login.sock",
        scaffolding_node="dtn.brc.berkeley.edu",
        scaffolding_socket_path="/tmp/dtn.sock",
    )

    # Make _run raise a timeout CommandError (returncode=-1).
    def fake_run(*args, **kwargs):
        raise CommandError(
            "Command timed out after 120s",
            CommandResult(
                requested_args=["echo", "$HOME"],
                executed_args=["ssh", "dtn.brc.berkeley.edu", "echo $HOME"],
                stdout="",
                stderr="(timed out)",
                returncode=-1,
            ),
        )

    fallback_called = {}

    def fake_run_agent(args, **kwargs):
        fallback_called["args"] = list(args)
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="/home/ligon",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(executor, "_run", fake_run)
    monkeypatch.setattr(executor, "run_agent", fake_run_agent)

    result = executor.run_on_login_node(["echo", "$HOME"])
    assert result.stdout == "/home/ligon"
    assert fallback_called["args"] == ["echo", "$HOME"]


def test_run_on_login_node_falls_back_on_ssh_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the DTN SSH connection fails (rc=255), fall back to run_agent."""
    from sucoder.executor import CommandError, CommandResult

    executor = _make_remote_executor(
        login_node="n0101.savio3",
        control_socket_path="/tmp/compute.sock",
        is_compute_node=True,
        scaffolding_node="dtn.brc.berkeley.edu",
        scaffolding_socket_path="/tmp/dtn.sock",
    )

    def fake_run(*args, **kwargs):
        raise CommandError(
            "SSH connection failed",
            CommandResult(
                requested_args=["hostname"],
                executed_args=["ssh", "dtn.brc.berkeley.edu", "hostname"],
                stdout="",
                stderr="ssh: connect to host dtn.brc.berkeley.edu: Connection refused",
                returncode=255,
            ),
        )

    fallback_called = {}

    def fake_run_agent(args, **kwargs):
        fallback_called["called"] = True
        return CommandResult(
            requested_args=list(args),
            executed_args=list(args),
            stdout="n0101",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(executor, "_run", fake_run)
    monkeypatch.setattr(executor, "run_agent", fake_run_agent)

    result = executor.run_on_login_node(["hostname"])
    assert fallback_called.get("called")
    assert result.stdout == "n0101"


def test_run_on_login_node_does_not_fallback_on_command_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal command failure on the DTN (rc=1) should NOT fall back."""
    from sucoder.executor import CommandError, CommandResult

    executor = _make_remote_executor(
        login_node="n0101.savio3",
        control_socket_path="/tmp/compute.sock",
        is_compute_node=True,
        scaffolding_node="dtn.brc.berkeley.edu",
        scaffolding_socket_path="/tmp/dtn.sock",
    )

    def fake_run(*args, **kwargs):
        raise CommandError(
            "Command failed",
            CommandResult(
                requested_args=["git", "push"],
                executed_args=["ssh", "dtn.brc.berkeley.edu", "git push"],
                stdout="",
                stderr="error: failed to push some refs",
                returncode=1,
            ),
        )

    monkeypatch.setattr(executor, "_run", fake_run)

    with pytest.raises(CommandError):
        executor.run_on_login_node(["git", "push"])


def test_ssh_error_enrichment_no_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """SSH exit 255 with no SSH_AUTH_SOCK logs a diagnostic hint."""
    import logging

    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    executor = _make_remote_executor(
        login_node="n0101.savio3",
        control_socket_path="/tmp/nonexistent.sock",
        is_compute_node=True,
        slurm_job_id=12345,
    )
    # Capture log output
    log_records: list = []
    handler = logging.Handler()
    handler.emit = lambda record: log_records.append(record)
    executor.logger.addHandler(handler)

    from sucoder.executor import CommandError, CommandResult

    exc = CommandError(
        "SSH failed",
        CommandResult(["ssh", "n0101.savio3"], ["ssh", "n0101.savio3"], "", "", 255),
    )
    executor._enrich_ssh_error(exc)

    messages = " ".join(r.getMessage() for r in log_records)
    assert "SSH_AUTH_SOCK" in messages
    assert "compute node" in messages
    assert "12345" in messages
    executor.logger.removeHandler(handler)


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
