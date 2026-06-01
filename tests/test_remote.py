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
