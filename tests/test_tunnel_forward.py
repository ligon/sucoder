"""Tests for `sucoder tunnel forward` / `tunnel forwards`.

The commands ride mux control requests (`ssh -O forward|cancel`) against
the compute-node ControlMaster socket, so the tests mock two seams:

* ``cli._mux_forward``      — the mux request itself
* ``cli._connect_with_retry`` — ControlMaster establishment

Everything else (target resolution, node resolution from session files,
record persistence in ``tunnel-<target>.yaml``) runs for real against a
temp ``$HOME``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

pytest.importorskip("typer")

from typer.testing import CliRunner

from sucoder import cli
from sucoder.session import RemoteSession
from sucoder.tunnel import SshControl

TARGET = "carleton"


def _setup(tmp_path: Path, monkeypatch, *, sessions=None) -> Path:
    """Write a config with one mirror + one target; seed session files.

    *sessions* maps filename stems (e.g. ``sample--carleton`` or
    ``tunnel-carleton``) to dicts dumped as the session YAML.
    Returns the config path.  ``$HOME`` is redirected to a temp dir so
    ``~/.sucoder/{sessions,ssh}`` are isolated.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    human = os.environ.get("USER", "coder")
    mirror_root = tmp_path / "mirrors"
    mirror_root.mkdir(exist_ok=True)
    canonical = tmp_path / "canonical"
    canonical.mkdir(exist_ok=True)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
human_user: {human}
agent_user: {human}
agent_group: {human}
mirror_root: {mirror_root}
mirrors:
  sample:
    canonical_repo: {canonical}
    mirror_name: sample
    branch_prefixes:
      human: {human}
      agent: {human}
targets:
  {TARGET}:
    gateway: gw.example.org
    transfer_host: dtn.example.org
""",
        encoding="utf-8",
    )

    sess_dir = fake_home / ".sucoder" / "sessions"
    sess_dir.mkdir(parents=True, exist_ok=True)
    for stem, data in (sessions or {}).items():
        (sess_dir / f"{stem}.yaml").write_text(
            yaml.safe_dump(data), encoding="utf-8",
        )
    return config_path


def _invoke(config_path: Path, *args: str):
    runner = CliRunner()
    return runner.invoke(
        cli.app, ["--config", str(config_path), "-T", TARGET, *args],
    )


def _combined(result) -> str:
    return result.stdout + (result.output or "")


def _tunnel_session_yaml(tmp_path: Path) -> dict:
    path = tmp_path / "home" / ".sucoder" / "sessions" / f"tunnel-{TARGET}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


SESSIONS_ONE_NODE = {
    f"sample--{TARGET}": {"compute_node": "n0030.savio4", "slurm_job_id": 1},
    f"tunnel-{TARGET}": {"login_node": "ln001.brc"},
}


def test_forward_requires_target(tmp_path, monkeypatch):
    config_path = _setup(tmp_path, monkeypatch)
    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["--config", str(config_path), "tunnel", "forward", "8888"],
    )
    assert result.exit_code == 2
    assert "-T" in _combined(result)


def test_forward_resolves_node_and_records(tmp_path, monkeypatch):
    """Happy path: node from the session, mux forward, record persisted."""
    config_path = _setup(tmp_path, monkeypatch, sessions=SESSIONS_ONE_NODE)

    connects: list = []
    monkeypatch.setattr(
        cli, "_connect_with_retry",
        lambda control, label, logger, **kw: connects.append(label),
    )
    mux_calls: list = []

    def fake_mux(action, spec, socket_path, node):
        mux_calls.append((action, spec, socket_path, node))
        return 0, ""

    monkeypatch.setattr(cli, "_mux_forward", fake_mux)

    result = _invoke(config_path, "tunnel", "forward", "8888")

    assert result.exit_code == 0, _combined(result)
    # Master chain walked gw -> ln -> node (no live node master in tests).
    assert connects == ["gw.example.org", "ln001.brc", "n0030.savio4"]
    assert mux_calls == [(
        "forward",
        "8888:localhost:8888",
        str(tmp_path / "home" / ".sucoder" / "ssh" / "n0030.savio4.sock"),
        "n0030.savio4",
    )]
    data = _tunnel_session_yaml(tmp_path)
    assert data["forwards"] == [
        {"local_port": 8888, "node": "n0030.savio4", "remote_port": 8888},
    ]
    # The pinned login node must survive the record write.
    assert data["login_node"] == "ln001.brc"
    assert "http://localhost:8888/" in _combined(result)


def test_forward_reuses_live_node_master(tmp_path, monkeypatch):
    """A live node master (e.g. collaborate's) means zero new connects."""
    config_path = _setup(tmp_path, monkeypatch, sessions=SESSIONS_ONE_NODE)

    monkeypatch.setattr(SshControl, "is_active", lambda self, **kw: True)

    def no_connect(*a, **kw):  # pragma: no cover - fails the test if hit
        raise AssertionError("must not (re)connect when the master is live")

    monkeypatch.setattr(cli, "_connect_with_retry", no_connect)
    monkeypatch.setattr(cli, "_mux_forward", lambda *a: (0, ""))

    result = _invoke(config_path, "tunnel", "forward", "8888")
    assert result.exit_code == 0, _combined(result)


def test_forward_local_port_override(tmp_path, monkeypatch):
    config_path = _setup(tmp_path, monkeypatch, sessions=SESSIONS_ONE_NODE)
    monkeypatch.setattr(cli, "_connect_with_retry", lambda *a, **kw: None)
    mux_calls: list = []
    monkeypatch.setattr(
        cli, "_mux_forward",
        lambda *a: mux_calls.append(a) or (0, ""),
    )

    result = _invoke(
        config_path, "tunnel", "forward", "8888", "--local-port", "9999",
    )
    assert result.exit_code == 0, _combined(result)
    assert mux_calls[0][1] == "9999:localhost:8888"
    data = _tunnel_session_yaml(tmp_path)
    assert data["forwards"][0]["local_port"] == 9999
    assert data["forwards"][0]["remote_port"] == 8888


def test_forward_ambiguous_nodes_require_flag(tmp_path, monkeypatch):
    config_path = _setup(tmp_path, monkeypatch, sessions={
        f"a--{TARGET}": {"compute_node": "n0030.savio4"},
        f"b--{TARGET}": {"compute_node": "n0047.savio4"},
        f"tunnel-{TARGET}": {"login_node": "ln001.brc"},
    })
    result = _invoke(config_path, "tunnel", "forward", "8888")
    combined = _combined(result)
    assert result.exit_code == 2
    assert "--node" in combined
    assert "n0030.savio4" in combined and "n0047.savio4" in combined


def test_forward_no_session_requires_flag(tmp_path, monkeypatch):
    config_path = _setup(tmp_path, monkeypatch, sessions={
        f"tunnel-{TARGET}": {"login_node": "ln001.brc"},
    })
    result = _invoke(config_path, "tunnel", "forward", "8888")
    assert result.exit_code == 2
    assert "--node" in _combined(result)


def test_forward_explicit_node_wins(tmp_path, monkeypatch):
    """--node bypasses session resolution (works with zero session files)."""
    config_path = _setup(tmp_path, monkeypatch, sessions={
        f"tunnel-{TARGET}": {"login_node": "ln001.brc"},
    })
    monkeypatch.setattr(cli, "_connect_with_retry", lambda *a, **kw: None)
    mux_calls: list = []
    monkeypatch.setattr(
        cli, "_mux_forward", lambda *a: mux_calls.append(a) or (0, ""),
    )

    result = _invoke(
        config_path, "tunnel", "forward", "8888", "--node", "n0099.savio4",
    )
    assert result.exit_code == 0, _combined(result)
    assert mux_calls[0][3] == "n0099.savio4"


def test_forward_unpinned_login_node_errors(tmp_path, monkeypatch):
    """No pinned login node and no live master -> point at `tunnel up`."""
    config_path = _setup(tmp_path, monkeypatch, sessions={
        f"sample--{TARGET}": {"compute_node": "n0030.savio4"},
    })

    def no_connect(*a, **kw):  # pragma: no cover - fails the test if hit
        raise AssertionError("must not connect without a pinned login node")

    monkeypatch.setattr(cli, "_connect_with_retry", no_connect)

    result = _invoke(config_path, "tunnel", "forward", "8888")
    assert result.exit_code == 1
    assert "tunnel up" in _combined(result)


def test_forward_duplicate_local_port_rejected(tmp_path, monkeypatch):
    config_path = _setup(tmp_path, monkeypatch, sessions={
        **SESSIONS_ONE_NODE,
        f"tunnel-{TARGET}": {
            "login_node": "ln001.brc",
            "forwards": [
                {"local_port": 8888, "node": "n0030.savio4",
                 "remote_port": 8888},
            ],
        },
    })
    result = _invoke(config_path, "tunnel", "forward", "8888")
    assert result.exit_code == 1
    assert "--cancel" in _combined(result)


def test_forward_mux_failure_surfaces_stderr(tmp_path, monkeypatch):
    config_path = _setup(tmp_path, monkeypatch, sessions=SESSIONS_ONE_NODE)
    monkeypatch.setattr(cli, "_connect_with_retry", lambda *a, **kw: None)
    monkeypatch.setattr(
        cli, "_mux_forward",
        lambda *a: (1, "bind [127.0.0.1]:8888: Address already in use"),
    )

    result = _invoke(config_path, "tunnel", "forward", "8888")
    combined = _combined(result)
    assert result.exit_code == 1
    assert "Address already in use" in combined
    assert "--local-port" in combined
    # A failed forward must not be recorded.
    assert _tunnel_session_yaml(tmp_path).get("forwards") in ([], None)


def test_forward_cancel_removes_record(tmp_path, monkeypatch):
    config_path = _setup(tmp_path, monkeypatch, sessions={
        f"tunnel-{TARGET}": {
            "login_node": "ln001.brc",
            "forwards": [
                {"local_port": 9999, "node": "n0030.savio4",
                 "remote_port": 8888},
            ],
        },
    })
    mux_calls: list = []
    monkeypatch.setattr(
        cli, "_mux_forward", lambda *a: mux_calls.append(a) or (0, ""),
    )

    result = _invoke(config_path, "tunnel", "forward", "9999", "--cancel")

    assert result.exit_code == 0, _combined(result)
    assert mux_calls[0][0] == "cancel"
    assert mux_calls[0][1] == "9999:localhost:8888"
    assert mux_calls[0][3] == "n0030.savio4"
    assert _tunnel_session_yaml(tmp_path)["forwards"] == []


def test_forward_cancel_unknown_port_errors(tmp_path, monkeypatch):
    config_path = _setup(tmp_path, monkeypatch, sessions={
        f"tunnel-{TARGET}": {"login_node": "ln001.brc"},
    })
    result = _invoke(config_path, "tunnel", "forward", "8888", "--cancel")
    assert result.exit_code == 1
    assert "No recorded forward" in _combined(result)


def test_forward_cancel_drops_record_even_if_mux_dead(tmp_path, monkeypatch):
    """A dead master already dropped the listener; the record must go too."""
    config_path = _setup(tmp_path, monkeypatch, sessions={
        f"tunnel-{TARGET}": {
            "login_node": "ln001.brc",
            "forwards": [
                {"local_port": 8888, "node": "n0030.savio4",
                 "remote_port": 8888},
            ],
        },
    })
    monkeypatch.setattr(
        cli, "_mux_forward", lambda *a: (255, "No such file or directory"),
    )
    result = _invoke(config_path, "tunnel", "forward", "8888", "--cancel")
    assert result.exit_code == 0, _combined(result)
    assert _tunnel_session_yaml(tmp_path)["forwards"] == []


def test_forwards_lists_records_with_liveness(tmp_path, monkeypatch):
    config_path = _setup(tmp_path, monkeypatch, sessions={
        f"tunnel-{TARGET}": {
            "login_node": "ln001.brc",
            "forwards": [
                {"local_port": 8888, "node": "n0030.savio4",
                 "remote_port": 8888},
                {"local_port": 9999, "node": "n0047.savio4",
                 "remote_port": 8080},
            ],
        },
    })
    monkeypatch.setattr(
        SshControl, "is_active",
        lambda self, **kw: self.gateway == "n0030.savio4",
    )

    result = _invoke(config_path, "tunnel", "forwards")
    combined = _combined(result)
    assert result.exit_code == 0
    assert "✓ localhost:8888 → n0030.savio4:8888" in combined
    assert "✗ localhost:9999 → n0047.savio4:8080" in combined


def test_forwards_empty(tmp_path, monkeypatch):
    config_path = _setup(tmp_path, monkeypatch)
    result = _invoke(config_path, "tunnel", "forwards")
    assert result.exit_code == 0
    assert "No forwards recorded" in _combined(result)


def test_session_forwards_roundtrip(tmp_path, monkeypatch):
    """The forwards field persists through save/load (and defaults to [])."""
    monkeypatch.setenv("HOME", str(tmp_path))

    session = RemoteSession.load("tunnel-x")
    assert session.forwards == []
    session.login_node = "ln001.brc"
    session.forwards.append(
        {"local_port": 1, "node": "n1", "remote_port": 2},
    )
    session.save()

    reloaded = RemoteSession.load("tunnel-x")
    assert reloaded.forwards == [
        {"local_port": 1, "node": "n1", "remote_port": 2},
    ]
    assert reloaded.login_node == "ln001.brc"
