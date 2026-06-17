"""Tests for the managed ~/.ssh/config block generation (tunnel spike)."""

from __future__ import annotations

import os

from sucoder import sshconfig


def test_alias_names_derive_from_target() -> None:
    aliases = sshconfig.alias_names("savio-node")
    assert aliases == {
        "gw": "savio-node-gw",
        "ln": "savio-node-ln",
        "dtn": "savio-node-dtn",
    }


def test_render_block_full() -> None:
    block = sshconfig.render_block(
        "savio-node",
        "hpc.brc.berkeley.edu",
        "dtn.brc.berkeley.edu",
        login_node="ln003.brc",
        user="eligon",
        control_persist="168h",
    )
    # All three aliases present.
    assert "Host savio-node-gw" in block
    assert "Host savio-node-ln" in block
    assert "Host savio-node-dtn" in block
    # Login + DTN ride the gateway alias via ProxyJump; gateway does not.
    assert "ProxyJump savio-node-gw" in block
    gw_stanza = block.split("Host savio-node-ln")[0]
    assert "ProxyJump" not in gw_stanza
    # Pinned login node lands as a HostName.
    assert "HostName ln003.brc" in block
    assert "User eligon" in block
    assert "ControlPersist 168h" in block
    assert "ControlMaster auto" in block
    # Keepalive defaults to the long (1h) grace budget: 30s x 120.
    assert "ServerAliveInterval 30" in block
    assert "ServerAliveCountMax 120" in block
    # ControlPath must match the SshControl socket for the same host so
    # SuCoder, ssh, and TRAMP share one mux.
    from sucoder.tunnel import _control_socket_path
    assert f"ControlPath {_control_socket_path('ln003.brc')}" in block
    # Fenced by per-target sentinels.
    assert ">>> sucoder managed block: savio-node >>>" in block
    assert "<<< sucoder managed block: savio-node <<<" in block


def test_render_block_custom_keepalive() -> None:
    """Custom keepalive values flow into every alias stanza so plain ssh /
    TRAMP match SuCoder's own keepalive tolerance."""
    block = sshconfig.render_block(
        "savio-node",
        "hpc.brc.berkeley.edu",
        "dtn.brc.berkeley.edu",
        login_node="ln003.brc",
        control_persist="3d",
        keepalive_interval=60,
        keepalive_count_max=30,
    )
    # One stanza per hop (gw/ln/dtn) => three of each directive.
    assert block.count("ServerAliveInterval 60") == 3
    assert block.count("ServerAliveCountMax 30") == 3
    assert block.count("ControlPersist 3d") == 3
    assert "ServerAliveInterval 30" not in block


def test_render_block_unpinned_login_node() -> None:
    block = sshconfig.render_block(
        "savio-node",
        "hpc.brc.berkeley.edu",
        "dtn.brc.berkeley.edu",
        login_node=None,
    )
    assert "Host savio-node-ln" in block
    # No actual HostName directive for the login alias until it is pinned;
    # only the placeholder comment.
    ln_stanza = block.split("Host savio-node-ln")[1].split("Host savio-node-dtn")[0]
    assert "    HostName " not in ln_stanza
    assert "HostName pending" in ln_stanza


def test_write_block_preserves_user_content_and_other_targets(tmp_path) -> None:
    cfg = tmp_path / "config"
    cfg.write_text(
        "Host myserver\n    HostName example.com\n    User me\n",
        encoding="utf-8",
    )

    block_a = sshconfig.render_block(
        "savio-node", "gw.a", "dtn.a", login_node="ln1",
    )
    sshconfig.write_block(block_a, "savio-node", path=cfg)

    block_b = sshconfig.render_block(
        "savio-htc", "gw.b", "dtn.b", login_node="ln2",
    )
    sshconfig.write_block(block_b, "savio-htc", path=cfg)

    text = cfg.read_text(encoding="utf-8")
    # User's own host survived untouched.
    assert "Host myserver" in text
    assert "HostName example.com" in text
    # Both target blocks present.
    assert "Host savio-node-gw" in text
    assert "Host savio-htc-gw" in text
    # Managed blocks are PREPENDED, ahead of the user's own content, so a
    # later `Host *` cannot shadow our ControlPath.
    assert text.index("Host savio-node-gw") < text.index("Host myserver")
    assert text.index("Host savio-htc-gw") < text.index("Host myserver")
    # Mode tightened to 0600.
    assert (os.stat(cfg).st_mode & 0o777) == 0o600


def test_write_block_precedes_wildcard_controlpath(tmp_path) -> None:
    """Regression: a user's ``Host *`` ControlPath must not shadow ours.

    ssh uses the first value it obtains per keyword, so the managed block
    has to be written *before* a general ``Host *`` default.  The bug:
    appending the block left the wildcard's ControlPath winning, so ssh
    looked for a socket SuCoder never created and re-authenticated.
    """
    cfg = tmp_path / "config"
    cfg.write_text(
        "Host *\n"
        "    ControlMaster auto\n"
        "    ControlPath ~/.ssh/sockets/%r@%h-%p\n"
        "    ControlPersist 10m\n",
        encoding="utf-8",
    )

    block = sshconfig.render_block(
        "savio-node", "hpc.brc.berkeley.edu", "dtn.brc.berkeley.edu",
        login_node="ln003.brc",
    )
    sshconfig.write_block(block, "savio-node", path=cfg)

    text = cfg.read_text(encoding="utf-8")
    # Our specific aliases must come before the wildcard so their
    # ControlPath is the first value ssh obtains.
    assert text.index("Host savio-node-ln") < text.index("Host *")
    # Wildcard block preserved (not clobbered).
    assert "ControlPath ~/.ssh/sockets/%r@%h-%p" in text


def test_write_block_replaces_in_place_no_duplication(tmp_path) -> None:
    cfg = tmp_path / "config"

    sshconfig.write_block(
        sshconfig.render_block("savio-node", "gw.a", "dtn.a", login_node=None),
        "savio-node",
        path=cfg,
    )
    # Re-run after the login node is pinned.
    sshconfig.write_block(
        sshconfig.render_block("savio-node", "gw.a", "dtn.a", login_node="ln003.brc"),
        "savio-node",
        path=cfg,
    )

    text = cfg.read_text(encoding="utf-8")
    # Exactly one managed block for the target (no accumulation).
    assert text.count(">>> sucoder managed block: savio-node >>>") == 1
    assert text.count("<<< sucoder managed block: savio-node <<<") == 1
    # Reflects the latest (pinned) state.
    assert "HostName ln003.brc" in text
    assert "HostName pending" not in text


def test_remove_block_and_block_present(tmp_path) -> None:
    cfg = tmp_path / "config"
    assert sshconfig.block_present("savio-node", path=cfg) is False

    sshconfig.write_block(
        sshconfig.render_block("savio-node", "gw.a", "dtn.a", login_node="ln1"),
        "savio-node",
        path=cfg,
    )
    assert sshconfig.block_present("savio-node", path=cfg) is True

    assert sshconfig.remove_block("savio-node", path=cfg) is True
    assert sshconfig.block_present("savio-node", path=cfg) is False
    # Removing again is a no-op.
    assert sshconfig.remove_block("savio-node", path=cfg) is False


# ------------------------------------------------------------------
# CLI: `sucoder -T <target> tunnel ...`
# ------------------------------------------------------------------


def _write_tunnel_config(tmp_path, human):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
human_user: {human}
agent_user: {human}
agent_group: {human}
mirror_root: {tmp_path / 'mirrors'}
targets:
  savio-node:
    gateway: hpc.brc.berkeley.edu
    transfer_host: dtn.brc.berkeley.edu
    control_persist: 168h
""",
        encoding="utf-8",
    )
    return config_path


def test_tunnel_requires_target(tmp_path, monkeypatch) -> None:
    from typer.testing import CliRunner
    from sucoder import cli

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)
    config_path = _write_tunnel_config(tmp_path, os.environ.get("USER", "coder"))

    runner = CliRunner()
    # No -T => exit code 2 with a helpful message.
    result = runner.invoke(cli.app, ["--config", str(config_path), "tunnel", "status"])
    assert result.exit_code == 2, result.output
    assert "requires a target" in (result.output + (result.stdout or ""))


def test_tunnel_status_reports_dead_when_no_sockets(tmp_path, monkeypatch) -> None:
    """`tunnel status` is read-only: with no warm sockets every hop is
    DEAD and the ssh_config block is absent — no network, no auth."""
    from typer.testing import CliRunner
    from sucoder import cli

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)
    config_path = _write_tunnel_config(tmp_path, os.environ.get("USER", "coder"))

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["--config", str(config_path), "-T", "savio-node", "tunnel", "status", "--json"],
    )
    assert result.exit_code == 0, result.output
    import json
    payload = json.loads(result.stdout)
    assert payload["target"] == "savio-node"
    assert payload["ssh_config"] is False
    # Gateway + DTN hops present; login not pinned yet so it's omitted.
    hop_names = {h["hop"] for h in payload["hops"]}
    assert {"gateway", "dtn"} <= hop_names
    assert all(h["active"] is False for h in payload["hops"])


# ------------------------------------------------------------------
# doctor: shadowing / pin-drift detection
# ------------------------------------------------------------------


def test_find_shadowing_hosts_detects_preceding_wildcard(tmp_path) -> None:
    """A `Host *` ControlPath BEFORE the managed block must be flagged."""
    cfg = tmp_path / "config"
    cfg.write_text(
        "Host *\n    ControlPath ~/.ssh/sockets/%r@%h-%p\n\n",
        encoding="utf-8",
    )
    sshconfig.write_block(
        sshconfig.render_block("savio-node", "gw", "dtn", login_node="ln003.brc"),
        "savio-node", path=cfg,
    )
    # write_block prepends, so re-add the wildcard ABOVE to simulate the
    # bad ordering a hand-edited config could have.
    text = cfg.read_text(encoding="utf-8")
    cfg.write_text("Host *\n    ControlPath ~/.ssh/sockets/%r@%h-%p\n\n" + text,
                   encoding="utf-8")

    shadow = sshconfig.find_shadowing_hosts("savio-node", path=cfg)
    assert shadow, "expected the preceding Host * ControlPath to be flagged"
    assert any(key == "controlpath" for _, key in shadow)
    assert any("Host *" in label for label, _ in shadow)


def test_find_shadowing_hosts_ignores_following_and_nonmatching(tmp_path) -> None:
    cfg = tmp_path / "config"
    # Managed block first (the correct, post-fix layout), then a Host *.
    sshconfig.write_block(
        sshconfig.render_block("savio-node", "gw", "dtn", login_node="ln003.brc"),
        "savio-node", path=cfg,
    )
    with cfg.open("a", encoding="utf-8") as fh:
        fh.write("\nHost *\n    ControlPath ~/.ssh/sockets/%r@%h-%p\n")
    # A Host * AFTER the block can't win the first-value race → not flagged.
    assert sshconfig.find_shadowing_hosts("savio-node", path=cfg) == []

    # A preceding but non-matching Host pattern is also fine.
    cfg2 = tmp_path / "config2"
    sshconfig.write_block(
        sshconfig.render_block("savio-node", "gw", "dtn", login_node="ln1"),
        "savio-node", path=cfg2,
    )
    cfg2.write_text(
        "Host buildbox\n    ControlPath ~/.ssh/sockets/bb\n\n" + cfg2.read_text(),
        encoding="utf-8",
    )
    # `buildbox` doesn't glob-match any savio-node-* alias → no shadow.
    assert sshconfig.find_shadowing_hosts("savio-node", path=cfg2) == []


def test_find_shadowing_hosts_matches_glob_alias(tmp_path) -> None:
    """A wildcard that globs the alias (e.g. `Host savio-*`) is flagged."""
    cfg = tmp_path / "config"
    sshconfig.write_block(
        sshconfig.render_block("savio-node", "gw", "dtn", login_node="ln1"),
        "savio-node", path=cfg,
    )
    cfg.write_text(
        "Host savio-*\n    ControlMaster no\n\n" + cfg.read_text(),
        encoding="utf-8",
    )
    shadow = sshconfig.find_shadowing_hosts("savio-node", path=cfg)
    assert any(key == "controlmaster" for _, key in shadow)


def test_managed_hostnames_parses_block(tmp_path) -> None:
    cfg = tmp_path / "config"
    sshconfig.write_block(
        sshconfig.render_block("savio-node", "gw.h", "dtn.h", login_node="ln003.brc"),
        "savio-node", path=cfg,
    )
    names = sshconfig.managed_hostnames("savio-node", path=cfg)
    assert names["savio-node-gw"] == "gw.h"
    assert names["savio-node-ln"] == "ln003.brc"
    assert names["savio-node-dtn"] == "dtn.h"


def test_tunnel_doctor_flags_shadowing(tmp_path, monkeypatch) -> None:
    """`tunnel doctor` exits non-zero and names the shadowing stanza."""
    from typer.testing import CliRunner
    from sucoder import cli

    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)
    config_path = _write_tunnel_config(tmp_path, os.environ.get("USER", "coder"))

    # Managed block present but a Host * ControlPath sits ABOVE it.
    from sucoder import sshconfig
    ssh_cfg = home / ".ssh" / "config"
    sshconfig.write_block(
        sshconfig.render_block(
            "savio-node", "hpc.brc.berkeley.edu", "dtn.brc.berkeley.edu",
            login_node="ln003.brc",
        ),
        "savio-node", path=ssh_cfg,
    )
    ssh_cfg.write_text(
        "Host *\n    ControlPath ~/.ssh/sockets/%r@%h-%p\n\n" + ssh_cfg.read_text(),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["--config", str(config_path), "-T", "savio-node", "tunnel", "doctor"],
    )
    assert result.exit_code == 1, result.output
    out = result.output + (result.stdout or "")
    assert "shadows" in out.lower()
    assert "controlpath" in out.lower()
