"""Tests for gateway SSH-certificate auth (per-target ``cert_file``).

Covers the four wiring points: config parsing/expansion, ``SshControl.
establish()`` presenting the cert on the gateway hop only, ``render_block``
emitting it on the ``-gw`` alias only, and the ``tunnel doctor`` cert-status
helpers.
"""

from __future__ import annotations

import datetime
import logging

import pytest

from sucoder import cli, sshconfig
from sucoder.config import ConfigError, ConfigWarning, _parse_remote_config, _parse_slurm_config
from sucoder.tunnel import SshControl

CERT = "/home/u/.ssh/ssh_certs/brc_cert"


# --------------------------------------------------------------------------
# Config parsing
# --------------------------------------------------------------------------

def test_parse_remote_config_expands_cert_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    rc = _parse_remote_config(
        {"gateway": "gw", "transfer_host": "dtn", "cert_file": "~/.ssh/ssh_certs/brc_cert"}
    )
    assert rc.cert_file is not None
    assert rc.cert_file.is_absolute()
    assert "~" not in str(rc.cert_file)
    assert str(rc.cert_file).endswith(".ssh/ssh_certs/brc_cert")


def test_parse_remote_config_cert_file_absent_is_none() -> None:
    rc = _parse_remote_config({"gateway": "gw", "transfer_host": "dtn"})
    assert rc.cert_file is None


def test_parse_remote_config_cert_file_wrong_type() -> None:
    with pytest.raises(ConfigError, match="cert_file"):
        _parse_remote_config({"gateway": "gw", "transfer_host": "dtn", "cert_file": 123})


def test_cert_file_misplaced_under_slurm_warns() -> None:
    """cert_file is a target-level key; nested under `slurm:` it warns + is ignored."""
    with pytest.warns(ConfigWarning, match="target-level"):
        cfg = _parse_slurm_config(
            {"partition": "savio4_htc", "account": "co_carleton", "cert_file": CERT}
        )
    assert not hasattr(cfg, "cert_file")


# --------------------------------------------------------------------------
# establish(): cert on the direct/gateway hop only
# --------------------------------------------------------------------------

def _establish_capture_cmd(monkeypatch, control) -> list:
    """Run establish() with subprocess mocked; return the ssh command list."""
    captured: dict = {}

    class _Ok:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return _Ok()

    monkeypatch.setattr("sucoder.tunnel.subprocess.run", _fake_run)
    control.establish(logging.getLogger("t"))
    return captured["cmd"]


def test_establish_presents_cert_on_gateway(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(SshControl, "socket_path", property(lambda self: tmp_path / "gw.sock"))
    control = SshControl(gateway="gw", cert_file=CERT)  # no jump_host -> gateway hop
    cmd = _establish_capture_cmd(monkeypatch, control)
    assert f"IdentityFile={CERT}" in cmd
    assert f"CertificateFile={CERT}-cert.pub" in cmd
    assert "IdentitiesOnly=yes" in cmd


def test_establish_no_cert_on_jumped_hop(monkeypatch, tmp_path) -> None:
    """Login/DTN hops (jump_host set) authenticate by publickey through the
    mux and must NOT be forced onto the gateway cert."""
    monkeypatch.setattr(SshControl, "socket_path", property(lambda self: tmp_path / "ln.sock"))
    control = SshControl(gateway="ln001", jump_host="gw", cert_file=CERT)
    cmd = _establish_capture_cmd(monkeypatch, control)
    assert not any(str(c).startswith("IdentityFile=") for c in cmd)
    assert "IdentitiesOnly=yes" not in cmd


def test_establish_no_cert_when_unset(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(SshControl, "socket_path", property(lambda self: tmp_path / "gw.sock"))
    control = SshControl(gateway="gw")  # cert_file defaults to None
    cmd = _establish_capture_cmd(monkeypatch, control)
    assert not any(str(c).startswith("IdentityFile=") for c in cmd)


# --------------------------------------------------------------------------
# render_block(): cert on the -gw alias only
# --------------------------------------------------------------------------

def test_render_block_emits_cert_on_gateway_only() -> None:
    block = sshconfig.render_block(
        "savio-node", "hpc.brc.berkeley.edu", "dtn.brc.berkeley.edu",
        login_node="ln003.brc", cert_file=CERT,
    )
    gw_stanza = block.split("Host savio-node-ln")[0]  # everything before the ln alias
    assert f"IdentityFile {CERT}" in gw_stanza
    assert f"CertificateFile {CERT}-cert.pub" in gw_stanza
    assert "IdentitiesOnly yes" in gw_stanza
    # Exactly once each => gateway only, never the proxied ln/dtn aliases.
    assert block.count("IdentityFile ") == 1
    assert block.count("IdentitiesOnly yes") == 1


def test_render_block_no_cert_by_default() -> None:
    block = sshconfig.render_block("savio-node", "gw", "dtn", login_node="ln003.brc")
    assert "IdentityFile" not in block
    assert "CertificateFile" not in block
    assert "IdentitiesOnly" not in block


# --------------------------------------------------------------------------
# tunnel doctor helpers
# --------------------------------------------------------------------------

def test_parse_cert_time_formats() -> None:
    assert cli._parse_cert_time("2026-06-30T12:00:00") == datetime.datetime(2026, 6, 30, 12, 0, 0)
    assert cli._parse_cert_time("2026/06/30 12:00:00") == datetime.datetime(2026, 6, 30, 12, 0, 0)
    assert cli._parse_cert_time("not-a-time") is None


def test_cert_status_not_minted(tmp_path) -> None:
    glyph, msg = cli._cert_status(str(tmp_path / "missing" / "brc_cert"))
    assert glyph == "•"
    assert "not minted" in msg


def test_cert_status_expired(monkeypatch, tmp_path) -> None:
    (tmp_path / "brc_cert-cert.pub").write_text("dummy")

    class _R:
        stdout = "        Valid: from 2020-01-01T00:00:00 to 2020-01-01T12:00:00\n"

    monkeypatch.setattr("subprocess.run", lambda *a, **k: _R())
    glyph, msg = cli._cert_status(str(tmp_path / "brc_cert"))
    assert glyph == "⚠"
    assert "EXPIRED" in msg


def test_cert_status_valid(monkeypatch, tmp_path) -> None:
    (tmp_path / "brc_cert-cert.pub").write_text("dummy")

    class _R:
        stdout = "        Valid: from 2099-01-01T00:00:00 to 2099-01-01T12:00:00\n"

    monkeypatch.setattr("subprocess.run", lambda *a, **k: _R())
    glyph, msg = cli._cert_status(str(tmp_path / "brc_cert"))
    assert glyph == "✓"
    assert "2099-01-01T12:00:00" in msg
