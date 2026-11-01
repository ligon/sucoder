"""Status records for ``sucoder attention`` (docs/attention.org)."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sucoder import attention as att
from sucoder import cli
from sucoder.config import AgentLauncher, AttentionConfig, ConfigError, _parse_attention

from tests.test_reply_launch import _claude_launch


def _hook(key, payload, directory, *, event=None):
    """Run the hook exactly as Claude would: ``sh -c`` with the payload on stdin."""
    return subprocess.run(
        ["sh", "-c", att.hook_shell_command(key, event=event)],
        input=payload, text=True, capture_output=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(directory), att.DIR_ENV: str(directory)},
        timeout=10,
    )


def _payload(event, **extra):
    return json.dumps({"hook_event_name": event, "session_id": "s-1", "cwd": "/w",
                       "transcript_path": "/home/coder/.claude/projects/-w/s-1.jsonl", **extra})


# -- the hook ---------------------------------------------------------------

@pytest.mark.parametrize("event, state", [
    ("SessionStart", "idle"), ("UserPromptSubmit", "working"), ("Stop", "idle"),
    ("Notification", "waiting"), ("SessionEnd", "ended"),
])
def test_the_hook_writes_a_record_that_parses_to_the_event_state(tmp_path, event, state):
    result = _hook("Proj--savio", _payload(event, message="Claude needs your permission to use Bash"), tmp_path)
    assert result.returncode == 0
    record = tmp_path / "Proj--savio.rec"
    status = att.parse_record("Proj--savio", record.read_text())
    assert (status.state, status.event, status.mirror, status.target) == (state, event, "Proj", "savio")
    assert status.session_id == "s-1" and status.cwd == "/w" and status.host
    assert status.transcript_path == "/home/coder/.claude/projects/-w/s-1.jsonl"
    assert abs(status.at - time.time()) < 30
    assert [p.name for p in tmp_path.iterdir()] == ["Proj--savio.rec"]  # no temp file left


def test_each_event_replaces_the_record(tmp_path):
    _hook("P--local", _payload("Notification", message="waiting"), tmp_path)
    _hook("P--local", _payload("UserPromptSubmit"), tmp_path)
    assert att.parse_record("P--local", (tmp_path / "P--local.rec").read_text()).state == "working"


def test_post_tool_use_records_the_event_name_and_not_the_tool_output(tmp_path):
    secret = "tool output that must not be copied"
    result = _hook("P--local", _payload("PostToolUse", tool_response=secret), tmp_path, event="PostToolUse")
    text = (tmp_path / "P--local.rec").read_text()
    assert result.returncode == 0 and secret not in text
    assert att.parse_record("P--local", text).state == "working"


def test_the_hook_never_fails_the_agent(tmp_path):
    blocked = tmp_path / "file-not-dir"
    blocked.write_text("")
    result = subprocess.run(
        ["sh", "-c", att.hook_shell_command("P--local")], input=_payload("Stop"), text=True,
        capture_output=True, env={"PATH": "/usr/bin:/bin", att.DIR_ENV: str(blocked / "sub")},
    )
    assert result.returncode == 0 and result.stdout == ""


@pytest.mark.parametrize("key", ["a b", "x;rm", "../up", "$(id)", ""])
def test_unsafe_keys_are_refused(key):
    with pytest.raises(ValueError):
        att.hook_shell_command(key)


def test_keys_are_sanitized_and_pair_mirror_with_target():
    assert att.attention_key("SuCoder", "savio") == "SuCoder--savio"
    assert att.attention_key("SuCoder", None) == "SuCoder--local"
    assert att.attention_key("my repo;x", "t/1") == "my_repo_x--t_1"
    assert att.attention_key(".hidden", None) == "_.hidden--local"
    assert att.split_key("K-Aggregators--carleton-htc") == ("K-Aggregators", "carleton-htc")


def test_settings_install_one_hook_per_event():
    settings = json.loads(att.claude_settings_json("P--t"))
    assert set(settings["hooks"]) == set(att.EVENT_STATES)
    for event, groups in settings["hooks"].items():
        (hook,) = groups[0]["hooks"]
        assert hook["type"] == "command" and "P--t.rec" in hook["command"]


# -- parsing and rendering -------------------------------------------------

@pytest.mark.parametrize("text", [
    "", "garbage", "sucoder-attention-v1\t1\n", "sucoder-attention-v1\tx\th\n{not json",
    "sucoder-attention-v1\t1\th\n[1, 2]", "other-v9\t1\th\n" + _payload("Stop"),
])
def test_a_malformed_record_reads_as_unknown(text):
    assert att.parse_record("M--t", text).state == "unknown"


def test_the_remote_listing_script_round_trips_through_the_parser(tmp_path):
    _hook("A--savio", _payload("Notification", message="perm"), tmp_path)
    _hook("B--savio", _payload("Stop"), tmp_path)
    (tmp_path / ".A--savio.123").write_text("temp file, not a record")
    out = subprocess.run(["sh", "-c", att.LIST_RECORDS_SH], capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", att.DIR_ENV: str(tmp_path)}).stdout
    statuses = {s.key: s for s in att.parse_listing(out, source="savio")}
    assert set(statuses) == {"A--savio", "B--savio"}
    assert statuses["A--savio"].state == "waiting" and statuses["A--savio"].message == "perm"
    assert statuses["B--savio"].source == "savio"


def test_an_empty_directory_lists_nothing(tmp_path):
    out = subprocess.run(["sh", "-c", att.LIST_RECORDS_SH], capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", att.DIR_ENV: str(tmp_path / "absent")})
    assert out.returncode == 0 and att.parse_listing(out.stdout, source="x") == []


def _status(key, state, at, message=None):
    mirror, target = att.split_key(key)
    return att.Status(key=key, mirror=mirror, target=target, state=state, at=at, host="n1.savio", message=message)


def test_waiting_sorts_first_and_longest_waiting_leads():
    now = 1000.0
    rows = [_status("W1--t", "waiting", 990), _status("R--t", "working", 100),
            _status("W2--t", "waiting", 500), _status("I--t", "idle", 900), _status("E--t", "ended", 1)]
    assert [s.key for s in att.sort_statuses(rows, now)] == ["W2--t", "W1--t", "I--t", "R--t", "E--t"]
    line = att.render_tmux(rows, now)
    assert line == "WAIT W2(8m) W1(10s) | run 1 | idle 1"
    table = att.render_table(rows, now)
    assert table.splitlines()[1].startswith("waiting  8m")
    assert "[permission]" not in table  # no kind recorded, none shown


def test_the_newest_record_wins_for_a_key_seen_twice():
    merged = att.merge_statuses([_status("A--t", "working", 10), _status("A--t", "waiting", 20)])
    assert [(s.key, s.state) for s in merged] == [("A--t", "waiting")]


def _waiting(key, at, kind="permission"):
    status = _status(key, "waiting", at)
    status.kind = kind
    return status


def test_a_request_notifies_once_and_a_new_event_notifies_again():
    notified = att.waiting_requests([_waiting("B--t", 5), _status("D--t", "idle", 4)])
    assert notified == {("B--t", 5)}
    now = [_waiting("A--t", 9), _waiting("B--t", 5), _waiting("C--t", 3), _status("D--t", "idle", 4)]
    assert [s.key for s in att.due_notifications(now, notified, 10)] == ["A--t", "C--t"]
    assert att.due_notifications(now, notified, 10) == []
    assert [s.key for s in att.due_notifications([_waiting("B--t", 6)], notified, 10)] == ["B--t"]


def test_notifications_filter_by_kind_and_wait_until_due():
    notified = set()
    rows = [_waiting("P--t", 100, "permission"), _waiting("I--t", 100, "idle"), _waiting("Q--t", 100, None)]
    assert [s.key for s in att.due_notifications(rows, notified, 200, kinds=["permission", "other"],
                                                  after_seconds=300)] == []
    assert notified == set()  # not yet due is not notified
    due = att.due_notifications(rows, notified, 400, kinds=["permission", "other"], after_seconds=300)
    assert [s.key for s in due] == ["P--t", "Q--t"]


@pytest.mark.parametrize("ntype, message, state, kind", [
    ("permission_prompt", "Claude needs your permission to use Bash", "waiting", "permission"),
    ("idle_prompt", "Claude is waiting for your input", "waiting", "idle"),
    ("elicitation_dialog", "Pick one", "waiting", "question"),
    ("auth_success", "Logged in", "unknown", None),
    (None, "Claude needs your permission to use Edit", "waiting", "permission"),
    (None, "Claude is waiting for your input", "waiting", "idle"),
    (None, "Something new", "waiting", "other"),
])
def test_notification_kinds(tmp_path, ntype, message, state, kind):
    extra = {"message": message, **({"notification_type": ntype} if ntype else {})}
    _hook("K--t", _payload("Notification", **extra), tmp_path)
    status = att.parse_record("K--t", (tmp_path / "K--t.rec").read_text())
    assert (status.state, status.kind) == (state, kind)


def test_only_blocking_notification_types_fire_the_hook():
    settings = json.loads(att.claude_settings_json("P--t"))
    matcher = settings["hooks"]["Notification"][0]["matcher"]
    assert set(matcher.split("|")) == set(att.WAITING_TYPES)
    assert "auth_success" not in matcher
    assert all("matcher" not in groups[0] for event, groups in settings["hooks"].items() if event != "Notification")


def test_the_cache_round_trips():
    rows = [_waiting("A--t", 9)]
    payload = json.loads(json.dumps(att.cache_payload(rows, errors=["x"], now=10)))
    assert att.statuses_from_cache(payload) == rows


# -- configuration ---------------------------------------------------------

def test_attention_config_defaults_off_and_notifies_every_kind_at_once():
    parsed = _parse_attention(None)
    assert parsed == AttentionConfig(hooks=False, notify_command=[])
    assert parsed.notify_kinds == list(att.NOTIFY_KINDS) and parsed.notify_after_minutes == 0


def test_attention_config_parses_the_push_filter():
    parsed = _parse_attention({"notify_kinds": ["permission", "question"], "notify_after_minutes": 2.5})
    assert parsed.notify_kinds == ["permission", "question"] and parsed.notify_after_minutes == 2.5
    assert _parse_attention({"notify_kinds": "permission"}).notify_kinds == ["permission"]


def test_attention_config_parses_and_splits_a_command_string():
    parsed = _parse_attention({"hooks": True, "notify_command": "curl -s -d hi https://ntfy.sh/x"})
    assert parsed == AttentionConfig(hooks=True, notify_command=["curl", "-s", "-d", "hi", "https://ntfy.sh/x"])


@pytest.mark.parametrize("raw", [
    [], {"hooks": "yes"}, {"hook": True}, {"notify_command": [1]}, {"notify_command": ["", "x"]},
    {"notify_kinds": []}, {"notify_kinds": ["urgent"]}, {"notify_after_minutes": -1},
    {"notify_after_minutes": "5"}, {"notify_after_minutes": True},
])
def test_attention_config_rejects_malformed_input(raw):
    with pytest.raises(ConfigError):
        _parse_attention(raw)


# -- launch ------------------------------------------------------------------

def _hooked_launch(tmp_path, monkeypatch, command, *, hooks=True, target="savio"):
    import sucoder.mirror as mirror_module

    original = mirror_module.MirrorManager.launch_agent

    def launch(self, *args, **kwargs):
        self.config.attention = AttentionConfig(hooks=hooks)
        self.target_name = target
        return original(self, *args, **kwargs)

    monkeypatch.setattr(mirror_module.MirrorManager, "launch_agent", launch)
    return _claude_launch(tmp_path, monkeypatch, command)


def test_a_claude_launch_gets_hook_settings_keyed_by_mirror_and_target(tmp_path, monkeypatch):
    _, argv = _hooked_launch(tmp_path, monkeypatch, ["claude"])
    assert argv.count("--settings") == 1
    settings = json.loads(argv[argv.index("--settings") + 1])
    assert set(settings["hooks"]) == set(att.EVENT_STATES)
    assert "sample--savio.rec" in settings["hooks"]["Stop"][0]["hooks"][0]["command"]


def test_a_remote_launch_without_a_target_name_is_not_labelled_local(tmp_path, monkeypatch):
    _, argv = _hooked_launch(tmp_path, monkeypatch, ["claude"], target=None)
    assert "sample--remote.rec" in argv[argv.index("--settings") + 1]


def test_hooks_are_off_unless_configured(tmp_path, monkeypatch):
    _, argv = _hooked_launch(tmp_path, monkeypatch, ["claude"], hooks=False)
    assert "--settings" not in argv


@pytest.mark.parametrize("command", [["gemini"], ["claude", "--settings", "mine.json"], ["claude", "--settings=x"]])
def test_other_harnesses_and_operator_settings_are_left_alone(tmp_path, monkeypatch, command):
    _, argv = _hooked_launch(tmp_path, monkeypatch, command)
    assert sum(1 for a in argv if a.startswith("--settings")) == (0 if command == ["gemini"] else 1)
    if command[0] == "claude":
        assert command[-1] in argv


# -- the command -------------------------------------------------------------

def _cli_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        f"human_user: coder\nagent_user: coder\nmirror_root: {tmp_path}\n"
        "attention:\n  hooks: true\n"
    )
    return path


def test_the_command_lists_local_records_and_writes_the_cache(tmp_path, monkeypatch):
    records = tmp_path / "records"
    records.mkdir()
    _hook("Proj--local", _payload("Notification", message="Claude is waiting for your input"), records)
    monkeypatch.setenv(att.DIR_ENV, str(records))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **k: None)
    cache = tmp_path / "cache.json"
    result = CliRunner().invoke(cli.app, ["--config", str(_cli_config(tmp_path)), "attention",
                                          "--local-only", "--cache", str(cache)])
    assert result.exit_code == 0, result.output
    assert "waiting" in result.output and "Proj" in result.output and "waiting for your input" in result.output
    assert json.loads(cache.read_text())["sessions"][0]["key"] == "Proj--local"

    line = CliRunner().invoke(cli.app, ["attention", "--tmux", "--cache", str(cache)])
    assert line.exit_code == 0 and line.output.startswith("WAIT Proj(")


def test_the_tmux_line_marks_a_stale_or_missing_cache(tmp_path):
    cache = tmp_path / "cache.json"
    assert CliRunner().invoke(cli.app, ["attention", "--tmux", "--cache", str(cache)]).output.strip() == "attn: no data"
    cache.write_text(json.dumps(att.cache_payload([], errors=["savio: not inspected"], now=time.time() - 3600)))
    out = CliRunner().invoke(cli.app, ["attention", "--tmux", "--cache", str(cache)]).output.strip()
    assert out.startswith("attn STALE") and out.endswith("!1")


def test_notify_describes_the_session_in_the_environment(tmp_path):
    import logging

    out = tmp_path / "env.txt"
    command = ["sh", "-c", f'printf "%s|%s|%s|%s" "$SUCODER_ATTENTION_MIRROR" '
                           f'"$SUCODER_ATTENTION_TARGET" "$SUCODER_ATTENTION_KIND" "$SUCODER_ATTENTION_MESSAGE" > {out}']
    status = _waiting("Proj--savio", 100)
    status.message = "perm"
    cli._attention_notify(command, status, 160, logging.getLogger("t"))
    assert out.read_text() == "Proj|savio|permission|perm"


def test_cached_lists_without_collecting(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_collect_attention", lambda *a, **k: pytest.fail("must not collect"))
    cache = tmp_path / "cache.json"
    args = ["--config", str(_cli_config(tmp_path)), "attention", "--cached", "--cache", str(cache)]
    missing = CliRunner().invoke(cli.app, args)
    assert missing.exit_code == cli.EXIT_UNAVAILABLE and "--watch" in missing.output
    cache.write_text(json.dumps(att.cache_payload([_waiting("Proj--savio", time.time() - 90)], errors=[])))
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0 and "(cached" in result.output and "[permission]" in result.output


def test_choosing_two_modes_is_refused(tmp_path):
    result = CliRunner().invoke(cli.app, ["attention", "--tmux", "--cached"])
    assert result.exit_code == 2


# -- through the session relay ---------------------------------------------

_RELAY = "session_relay:\n  user: somebody-else\n  command: /opt/sucoder/bin/sucoder\n"


def _relay_config(tmp_path):
    path = tmp_path / "relay.yaml"
    path.write_text(f"human_user: coder\nagent_user: coder\nmirror_root: {tmp_path}\n" + _RELAY)
    return path


def _sudo(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


@pytest.mark.parametrize("flags, relayed", [
    ([], []), (["--json", "--cached"], ["--json", "--cached"]), (["-v"], ["--verbose"]),
])
def test_the_agent_account_reads_through_the_relay(tmp_path, monkeypatch, flags, relayed):
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **k: None)
    calls = _sudo(monkeypatch)
    result = CliRunner().invoke(cli.app, ["--config", str(_relay_config(tmp_path)), "attention", *flags])
    assert result.exit_code == 0, result.output
    argv = ["/opt/sucoder/bin/sucoder", "relay", "attention", *relayed]
    assert calls == [["sudo", "-n", "-u", "somebody-else", "-l", *argv], ["sudo", "-n", "-u", "somebody-else", *argv]]


@pytest.mark.parametrize("flags", [["--watch", "30"], ["--cache", "/tmp/x.json"]])
def test_watch_and_a_chosen_cache_are_not_relayed(tmp_path, monkeypatch, flags):
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **k: None)
    calls = _sudo(monkeypatch)
    result = CliRunner().invoke(cli.app, ["--config", str(_relay_config(tmp_path)), "attention", *flags])
    assert result.exit_code == 2 and calls == []


def test_local_only_reads_this_accounts_records_without_the_relay(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **k: None)
    monkeypatch.setenv(att.DIR_ENV, str(tmp_path / "none"))
    calls = _sudo(monkeypatch)
    result = CliRunner().invoke(cli.app, ["--config", str(_relay_config(tmp_path)), "attention",
                                          "--local-only", "--cache", str(tmp_path / "c.json")])
    assert result.exit_code == 0 and calls == [] and "No attention records" in result.output


def test_the_relay_side_runs_one_collection_and_never_watches(tmp_path, monkeypatch):
    from sucoder import tunnel
    monkeypatch.setattr(tunnel, "_NONINTERACTIVE", False)  # the relay sets it; restored after
    path = tmp_path / "config.yaml"
    path.write_text(f"human_user: coder\nagent_user: coder\nmirror_root: {tmp_path}\n")
    monkeypatch.setattr(cli, "_default_config_path", lambda: path)
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **k: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    seen = {}
    monkeypatch.setattr(cli, "_collect_attention",
                        lambda config, **kw: (seen.update(kw) or ([_waiting("A--t", time.time())], [])))
    result = CliRunner().invoke(cli.app, ["relay", "attention", "--json"])
    assert result.exit_code == 0, result.output
    assert seen["warm_only"] is False and seen["local_only"] is False
    assert json.loads(result.output)["sessions"][0]["key"] == "A--t"
    assert (tmp_path / ".sucoder" / "attention.json").exists()


# -- examples/coordinator/attention-notify -----------------------------------

NOTIFIER = Path(__file__).resolve().parent.parent / "examples" / "coordinator" / "attention-notify"


def _run_notifier(tmp_path, **env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "tmux.log"
    fake = bin_dir / "tmux"
    fake.write_text(f'#!/bin/sh\nfor a in "$@"; do printf "%s\\037" "$a"; done >> {log}\nprintf "\\n" >> {log}\n')
    fake.chmod(0o755)
    base = {"PATH": f"{bin_dir}:/usr/bin:/bin", "SUCODER_ATTENTION_KEY": "LSMS--savio",
            "SUCODER_ATTENTION_MIRROR": "LSMS", "SUCODER_ATTENTION_KIND": "permission",
            "SUCODER_ATTENTION_AGE": "3m"}
    result = subprocess.run(["sh", str(NOTIFIER)], env={**base, **env}, capture_output=True, text=True)
    calls = [line.split("\x1f")[:-1] for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls


def test_the_notifier_types_one_clean_line_into_the_coordinator(tmp_path):
    result, calls = _run_notifier(tmp_path, COORDINATOR_PANE="coordinator",
                                  SUCODER_ATTENTION_MESSAGE="needs\nyour\x1b[31m permission")
    assert result.returncode == 0
    assert calls == [
        ["send-keys", "-t", "coordinator", "-l",
         "[attention] LSMS--savio is waiting (permission, 3m): needs your [31m permission"],
        ["send-keys", "-t", "coordinator", "Enter"],
    ]


def test_the_notifier_never_reports_the_coordinator_to_itself(tmp_path):
    result, calls = _run_notifier(tmp_path, COORDINATOR_PANE="coordinator",
                                  SUCODER_ATTENTION_MIRROR="Coordinator", SUCODER_ATTENTION_MESSAGE="m")
    assert result.returncode == 0 and calls == []


def test_the_notifier_does_nothing_unconfigured_and_survives_a_failed_post(tmp_path):
    result, calls = _run_notifier(tmp_path, SUCODER_ATTENTION_MESSAGE="m")
    assert result.returncode == 0 and calls == []
    result, _ = _run_notifier(tmp_path, NTFY_URL="http://127.0.0.1:9/x", SUCODER_ATTENTION_MESSAGE="m")
    assert result.returncode == 0
