"""Issue 52: session discovery and messaging from the agent account.

An agent account whose own SSH route fails must be told that discovery or
delivery was *unavailable* (exit 3), never "no live session found" (exit 1),
and may reach live sessions through ``session_relay``: ``sudo -n -u HUMAN
sucoder relay ...``, which runs with the human's config and warm SSH.
"""

import os
import pwd
import subprocess
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from sucoder import cli
from sucoder.config import ConfigError, SessionRelayConfig, _parse_session_relay
from sucoder.messaging import Recipient
from sucoder.sessions_report import Report

GATEWAY = "gw.example.org"


def _config(tmp_path, relay=None):
    human = os.environ.get("USER", "coder")
    (tmp_path / "mirrors").mkdir(exist_ok=True)
    (tmp_path / "canonical").mkdir(exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(f"""
human_user: {human}
agent_user: {human}
agent_group: {human}
mirror_root: {tmp_path / 'mirrors'}
mirrors:
  sample:
    canonical_repo: {tmp_path / 'canonical'}
    mirror_name: sample
targets:
  fake-slurm:
    gateway: {GATEWAY}
    transfer_host: dtn.example.org
    slurm:
      partition: test
      account: test_acct
      confined: true
{relay or ''}""", encoding="utf-8")
    return path


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **k: None)
    monkeypatch.setattr(cli, "set_progress", lambda *a, **k: None)

    def invoke(*argv, relay=None):
        path = _config(tmp_path, relay)
        return CliRunner().invoke(cli.app, ["--config", str(path), *argv])
    return invoke


def _collected(monkeypatch, report):
    monkeypatch.setattr(cli, "_collect_sessions", lambda config, **kw: (report, {}))


def _unreached():
    return Report(errors=[f"{GATEWAY}: could not connect (Too many authentication failures)"],
                  uninspected=[GATEWAY])


# ---------------------------------------------------------- unavailable vs absent

def test_message_says_discovery_was_unavailable_not_that_no_session_exists(run, monkeypatch):
    _collected(monkeypatch, _unreached())
    result = run("-T", "fake-slurm", "message", "sample", "hello")
    assert result.exit_code == cli.EXIT_UNAVAILABLE, result.output
    assert "discovery unavailable" in result.output and "no live session found" not in result.output
    assert "sucoder -T <target> cert" in result.output and "sudo -iu" in result.output     # the agent can ask for a cert


def test_message_still_says_no_live_session_after_a_successful_inspection(run, monkeypatch):
    _collected(monkeypatch, Report())
    result = run("-T", "fake-slurm", "message", "sample", "hello")
    assert result.exit_code == 1
    assert "no live session found for mirror sample on target fake-slurm" in result.output


def test_peek_says_discovery_was_unavailable(run, monkeypatch):
    _collected(monkeypatch, _unreached())
    result = run("peek", "sample")
    assert result.exit_code == cli.EXIT_UNAVAILABLE and "Nothing was read" in result.output


def test_sessions_exits_unavailable_when_a_cluster_was_not_inspected(run, monkeypatch):
    _collected(monkeypatch, _unreached())
    monkeypatch.setattr("sucoder.sessions_report.render_report", lambda report, probed: "listing")
    result = run("sessions")
    assert result.exit_code == cli.EXIT_UNAVAILABLE
    assert f"not inspected: {GATEWAY}" in result.output


def test_a_send_that_cannot_connect_is_unavailable_not_undelivered(run, monkeypatch):
    target = Recipient(label="sample (job 7)", host="ln001.brc", session="sucoder-sample",
                       job_id=7, target="fake-slurm", pane="codex")
    monkeypatch.setattr(cli, "_plan_session_recipients", lambda config, **kw: ([target], [], Report(), {}))
    monkeypatch.setattr(cli, "_control_for_recipient", lambda *a: object())

    def refuse(*a, **k):
        raise RuntimeError("Failed to establish SSH connection")
    monkeypatch.setattr(cli, "_capture_over_tunnel", refuse)
    result = run("message", "sample", "hello")
    assert result.exit_code == cli.EXIT_UNAVAILABLE and "delivery unavailable" in result.output


def test_a_delivered_message_exits_zero(run, monkeypatch):
    target = Recipient(label="sample (job 7)", host="ln001.brc", session="sucoder-sample",
                       job_id=7, target="fake-slurm", pane="codex")
    monkeypatch.setattr(cli, "_plan_session_recipients", lambda config, **kw: ([target], [], Report(), {}))
    monkeypatch.setattr(cli, "_control_for_recipient", lambda *a: object())
    monkeypatch.setattr("sucoder.messaging.parse_send_output", lambda out: ({"sucoder-sample"}, set()))
    monkeypatch.setattr(cli, "_capture_over_tunnel",
                        lambda *a, **k: SimpleNamespace(stdout="", stderr="", returncode=0))
    result = run("message", "sample", "hello")
    assert result.exit_code == 0 and "sent to sample (job 7)" in result.output


# ---------------------------------------------------------- session_relay config

def test_session_relay_parses_a_path_or_a_list():
    assert _parse_session_relay(None) is None
    assert _parse_session_relay({"user": "ligon", "command": "/usr/local/bin/sucoder"}) == \
        SessionRelayConfig(user="ligon", command=["/usr/local/bin/sucoder"])
    assert _parse_session_relay({"user": "ligon", "command": ["/opt/uv/sucoder", "--no-x11"]}).command == \
        ["/opt/uv/sucoder", "--no-x11"]


@pytest.mark.parametrize("raw, message", [
    ({"user": "ligon", "command": "sucoder"}, "absolute path"),
    ({"user": "ligon"}, "path or a non-empty list"),
    ({"user": "Bad User", "command": "/x"}, "user name"),
    ({"user": "ligon", "command": "/x", "host": "y"}, "Unknown key"),
    ("ligon", "must be a mapping"),
])
def test_session_relay_rejections(raw, message):
    with pytest.raises(ConfigError, match=message):
        _parse_session_relay(raw)


# ---------------------------------------------------------- delegation (agent side)

_RELAY = "session_relay:\n  user: somebody-else\n  command: /opt/sucoder/bin/sucoder\n"


def _sudo(monkeypatch, *, allowed=0, ran=0):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        code = allowed if "-l" in argv else ran
        return subprocess.CompletedProcess(argv, code, stdout="", stderr="" if code == 0 else "a password is required")
    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_message_goes_through_the_relay_with_its_arguments(run, monkeypatch):
    calls = _sudo(monkeypatch, ran=0)
    result = run("-T", "fake-slurm", "message", "--from", "Sue", "--wait", "45", "--", "sample", "-x looks like a flag",
                 relay=_RELAY)
    assert result.exit_code == 0, result.output
    sudo = ["sudo", "-n", "-u", "somebody-else"]
    relayed = ["/opt/sucoder/bin/sucoder", "relay", "message", "-T", "fake-slurm", "--from", "Sue",
               "--wait", "45.0", "--", "sample", "-x looks like a flag"]
    assert calls == [[*sudo, "-l", *relayed], [*sudo, *relayed]]


def test_the_relays_exit_status_is_the_callers(run, monkeypatch):
    _sudo(monkeypatch, ran=cli.EXIT_UNAVAILABLE)
    assert run("sessions", relay=_RELAY).exit_code == cli.EXIT_UNAVAILABLE
    _sudo(monkeypatch, ran=1)
    assert run("peek", "sample", relay=_RELAY).exit_code == 1


def test_a_missing_sudo_rule_is_unavailable_not_no_session(run, monkeypatch):
    calls = _sudo(monkeypatch, allowed=1)
    result = run("message", "sample", "hello", relay=_RELAY)
    assert result.exit_code == cli.EXIT_UNAVAILABLE and "session_relay unavailable" in result.output
    assert len(calls) == 1          # never ran the command itself


def test_force_is_refused_through_the_relay(run, monkeypatch):
    calls = _sudo(monkeypatch)
    result = run("message", "sample", "hello", "--force", relay=_RELAY)
    assert result.exit_code == 2 and calls == []


def test_no_relay_when_already_the_relay_user(run, monkeypatch):
    me = pwd.getpwuid(os.getuid()).pw_name
    calls = _sudo(monkeypatch)
    _collected(monkeypatch, Report())
    result = run("message", "sample", "hello", relay=f"session_relay:\n  user: {me}\n  command: /x/sucoder\n")
    assert result.exit_code == 1 and calls == []


# ---------------------------------------------------------- the relay (human side)

def _as_default_config(monkeypatch, tmp_path, relay=None):
    path = _config(tmp_path, relay)
    monkeypatch.setattr(cli, "_default_config_path", lambda: path)
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **k: None)
    monkeypatch.setattr(cli, "set_progress", lambda *a, **k: None)
    return path


def test_the_relay_tags_the_sender_with_the_caller_and_never_forces(tmp_path, monkeypatch):
    _as_default_config(monkeypatch, tmp_path)
    monkeypatch.setenv("SUDO_USER", "coder")
    seen = {}
    monkeypatch.setattr(cli, "message", lambda ctx, **kw: seen.update(kw, target=ctx.obj["target_name"]))
    result = CliRunner().invoke(cli.app, ["relay", "message", "-T", "fake-slurm", "--from", "Sue/Admin@lyn",
                                          "--", "sample", "hi"])
    assert result.exit_code == 0, result.output
    assert seen["sender"] == "Sue/Admin@lyn (via coder)" and seen["force"] is False
    assert (seen["recipient"], seen["text"], seen["target"]) == ("sample", "hi", "fake-slurm")


def test_the_relay_reports_unavailable_discovery_like_the_command(tmp_path, monkeypatch):
    _as_default_config(monkeypatch, tmp_path)
    _collected(monkeypatch, _unreached())
    result = CliRunner().invoke(cli.app, ["relay", "peek", "sample"])
    assert result.exit_code == cli.EXIT_UNAVAILABLE


def test_the_relay_refuses_another_config(tmp_path, monkeypatch):
    _as_default_config(monkeypatch, tmp_path)
    (tmp_path / "other").mkdir()
    other = _config(tmp_path / "other")
    result = CliRunner().invoke(cli.app, ["--config", str(other), "relay", "sessions"])
    assert result.exit_code == 2 and "reads only" in result.output


def test_the_relay_refuses_a_home_that_is_not_its_own(tmp_path, monkeypatch):
    _as_default_config(monkeypatch, tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    result = CliRunner().invoke(cli.app, ["relay", "sessions"])
    assert result.exit_code == 2 and "HOME is not" in result.output
