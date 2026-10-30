"""Tunnel diagnostics must remain useful when remote sessions are wedged."""

import json
import subprocess

import pytest
from typer.testing import CliRunner

from sucoder import cli, tunnel
from sucoder.session import RemoteSession


HOSTS = ("gw.example.org", "ln001.brc", "dtn.example.org")


@pytest.fixture
def status_target(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)
    config = tmp_path / "config.yaml"
    config.write_text(f"""human_user: tester
mirror_root: {tmp_path}/mirrors
targets:
  cluster:
    gateway: {HOSTS[0]}
    transfer_host: {HOSTS[2]}
""")
    RemoteSession("tunnel-cluster", login_node=HOSTS[1]).save()
    for host in HOSTS:
        tunnel.SshControl(gateway=host).socket_path.touch()
    return ["--config", str(config), "-T", "cluster", "tunnel", "status"]


@pytest.mark.parametrize("json_output", [False, True])
def test_status_uses_only_local_checks(status_target, monkeypatch, json_output):
    checked = []

    def local_check(cmd, **kwargs):
        assert cmd[-3:-1] == ["-O", "check"]
        checked.append(cmd[-1])
        return subprocess.CompletedProcess(cmd, 255 if cmd[-1] == HOSTS[2] else 0)

    monkeypatch.setattr(tunnel.subprocess, "run", local_check)
    monkeypatch.setattr(
        tunnel.SshControl, "_probe_end_to_end",
        lambda *a, **kw: pytest.fail("default status opened a remote session"),
    )
    result = CliRunner().invoke(cli.app, status_target + (["--json"] if json_output else []))
    assert result.exit_code == 0, (result.output, result.exception)
    assert checked == list(HOSTS)
    if json_output:
        payload = json.loads(result.stdout)
        assert payload["target"] == "cluster"
        assert payload["hops"] == [
            {"hop": name, "host": host, "active": host != HOSTS[2]}
            for name, host in zip(("gateway", "login", "dtn"), HOSTS)
        ]
    else:
        assert "Local master checks only" in result.stdout
        assert "--deep" in result.stdout
        assert "ACTIVE" in result.stdout and "DEAD" in result.stdout


@pytest.mark.parametrize("json_output", [False, True])
def test_deep_status_announces_each_probe_and_bypasses_cache(
    status_target, monkeypatch, json_output,
):
    for host in HOSTS:
        tunnel.SshControl(gateway=host)._note_verified()
    monkeypatch.setattr(
        tunnel.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0),
    )
    announcements = []
    original_echo = cli.typer.echo

    def echo(message=None, **kwargs):
        if str(message).startswith("Checking "):
            assert kwargs.get("err") is True
            announcements.append(str(message))
        return original_echo(message, **kwargs)

    probed = []

    def probe(control, **kwargs):
        # This assertion runs inside the probe: progress must precede the wait.
        assert control.gateway in announcements[-1]
        probed.append(control.gateway)
        return control.gateway != HOSTS[1]

    monkeypatch.setattr(cli.typer, "echo", echo)
    monkeypatch.setattr(tunnel.SshControl, "_probe_end_to_end", probe)
    args = status_target + ["--deep"] + (["--json"] if json_output else [])
    # Match test_cli's stream-isolation pattern across Click versions.
    import inspect
    from click.testing import CliRunner as ClickRunner
    from typer.main import get_command

    if "mix_stderr" in inspect.signature(ClickRunner.__init__).parameters:
        runner = ClickRunner(mix_stderr=False)
    else:
        runner = ClickRunner()
    result = runner.invoke(get_command(cli.app), args)
    assert result.exit_code == 0, (result.output, result.exception)
    assert probed == list(HOSTS)
    assert len(announcements) == len(HOSTS)
    assert "Checking " in result.stderr
    assert "Checking " not in result.stdout
    if json_output:
        assert [hop["active"] for hop in json.loads(result.stdout)["hops"]] == [True, False, True]
    else:
        assert "Local master checks only" not in result.stdout


def test_status_bounds_unresponsive_local_master(status_target, monkeypatch):
    def wedged_mux(cmd, **kwargs):
        assert cmd[-3:-1] == ["-O", "check"]
        assert kwargs["timeout"] == 5
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(tunnel.subprocess, "run", wedged_mux)
    result = CliRunner().invoke(cli.app, status_target + ["--json"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert all(not hop["active"] for hop in json.loads(result.stdout)["hops"])
