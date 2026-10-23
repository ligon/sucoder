"""``message --wait`` and ``peek --reply`` over the transcript probe, with the
transport mocked (docs/reply-from-transcript.org)."""

import getpass
import json
import logging
import os
import re
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from sucoder import cli, messaging, replies
from sucoder.messaging import sanitize_reply
from sucoder.session import RemoteSession

from tests.test_cli import _SnapshotsHarness, _message_report
from tests.test_session_relay import _as_default_config

ID = "8e25ab12"
SID = "83cd9e50-d565-4635-95a2-da82d4b334cd"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A private HOME: session records and the default sent-log path live there."""
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    monkeypatch.setattr(messaging, "new_message_id", lambda: ID)
    return path


def _record(mirror="M", target="t0", **fields):
    session = RemoteSession(mirror_name=mirror, target_name=target, remote_mirror_root="~/mirrors", **fields)
    session.save()
    return session


def _answer(status, **fields):
    out = {"schema": 1, "status": status, "harness": "claude", "replies": []}
    out.update(fields)
    return json.dumps(out) + "\n"


def _run(monkeypatch, *, answers=(), panes=(), wait=30, report=None, recipient="M", everyone=False,
         text="hello", times=None, raise_probe=False, ctx_obj=None, probe_rc=0, reconnects=None):
    harness = _SnapshotsHarness(times=("12-00:00:00",))
    harness.config.mirrors = {"M": SimpleNamespace(mirror_dirname="M"), "L": SimpleNamespace(mirror_dirname="L")}
    monkeypatch.setattr(cli, "setup_logger", lambda *a, **k: logging.getLogger("t"))
    monkeypatch.setattr(cli, "_connect_with_retry",
                        lambda *a, **kw: None if reconnects is None else reconnects.append(a[1]))
    monkeypatch.setattr(cli, "_message_sleep", lambda s: None)
    if times is not None:
        clock = iter(times)
        monkeypatch.setattr(cli.time, "monotonic", lambda: next(clock))
    control = object()
    monkeypatch.setattr(cli, "_collect_sessions",
                        lambda config, **kw: (report or _message_report(), {"hpc.brc": ("ln001.brc", control)}))
    trips = []
    probe_out = iter(answers)
    pane_out = iter(panes)

    def fake(ctl, host, command, **kw):
        trips.append((host, command))
        if "python3 -I -c" in command:
            if raise_probe:
                raise RuntimeError("ssh: connection refused")
            if probe_rc:
                return SimpleNamespace(returncode=probe_rc, stdout="", stderr="")
            return SimpleNamespace(returncode=0, stdout=next(probe_out, _answer("seen")), stderr="")
        if "capture-pane" in command:
            return SimpleNamespace(returncode=0, stdout=next(pane_out, ""), stderr="")
        return SimpleNamespace(returncode=0, stdout="SENT\tsucoder-M\nSENT\tsucoder-L\n", stderr="")
    monkeypatch.setattr(cli, "_capture_over_tunnel", fake)
    monkeypatch.setattr(cli, "_run_remote_capture", fake)   # the probe's own transport
    monkeypatch.setattr(cli, "_control_for_host", lambda remote, host, debug, **kw: control)
    obj = {"config": harness.config, "target_name": None}
    obj.update(ctx_obj or {})
    ctx = SimpleNamespace(obj=obj, params={})
    code = 0
    try:
        cli.message(ctx, recipient=recipient, text=None if everyone else text, everyone=everyone,
                    sender="me@lyn", wait=wait, force=False, dry_run=False, verbose=False)
    except typer.Exit as exc:
        code = exc.exit_code
    return code, trips


def _probes(trips):
    return [c for _, c in trips if "python3 -I -c" in c]


# ----------------------------------------------------------------- the sent log

def test_each_delivered_recipient_appends_one_line_to_the_sent_log(home, monkeypatch):
    monkeypatch.setattr(replies, "sent_log_path", replies.default_sent_log_path)
    code, _ = _run(monkeypatch, everyone=True, recipient="hello all", wait=0)
    assert code == 0
    log = home / ".sucoder" / "sent.jsonl"
    assert oct(log.stat().st_mode & 0o777) == "0o600"
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(entries) == 2
    for entry in entries:
        assert re.fullmatch(r"[0-9a-f]{8}", entry["id"]) and entry["id"] == ID
        assert entry["frame"].startswith(f"[message {ID} from me@lyn via sucoder, ") and entry["frame"].endswith("] hello all")
        assert entry["frame"].startswith(entry["header"])
        assert entry["sent_at"].endswith("Z") and isinstance(entry["sent_epoch"], float)
        assert entry["account"] == getpass.getuser() and entry["relay_account"] is None
    labels = {e["recipient"]["label"] for e in entries}
    assert labels == {"M (job 41 on n0035, t0)", "L (login session on ln003.brc, t0)"}
    job = next(e for e in entries if e["recipient"]["job_id"] == 41)
    assert job["recipient"]["host"] == "ln001.brc" and job["recipient"]["session"] == "sucoder-M"
    assert job["recipient"]["mirror"] == "M" and job["recipient"]["session_keys"] == ["M--t0"]
    assert replies.lookup_sent(ID) == entries


def test_message_ids_are_eight_hex_characters():
    assert re.fullmatch(r"[0-9a-f]{8}", messaging.new_message_id())


# ----------------------------------------------------------------- exit codes

@pytest.mark.parametrize("status, extra, code", [
    ("replied", {"reply": "ok.", "source": "transcript", "replies": [{"source": "transcript", "text": "ok."}]}, 0),
    ("seen", {}, 4),
    ("queued", {}, 4),
    ("absent", {}, 4),
    ("ambiguous", {"candidates": ["/a.jsonl", "/b.jsonl"]}, 1),
    ("unreadable", {"why": "denied"}, 3),
    ("unrecognized", {"why": "frame in an entry no adapter rule names"}, 3),
])
def test_wait_maps_each_status_to_its_exit_code(home, monkeypatch, capsys, status, extra, code):
    _record(slurm_job_id=41, harness="claude", claude_session_id=SID)
    got, trips = _run(monkeypatch, answers=[_answer(status, **extra)] * 5, times=[0, 0, 100])
    out = capsys.readouterr()
    assert got == code, out
    assert _probes(trips)
    screen = any("capture-pane" in c for _, c in trips)
    assert not screen                    # only no-adapter reads the pane
    if status == "replied":
        assert "M (job 41 on n0035, t0) replied (transcript): ok." in out.out
    else:
        assert f"{status}:" in out.out + out.err


def test_every_poll_failing_in_transport_is_exit_3(home, monkeypatch, capsys):
    _record(slurm_job_id=41, harness="claude", claude_session_id=SID)
    code, _ = _run(monkeypatch, raise_probe=True, times=[0, 0, 100])
    assert code == 3 and "every reply probe failed in transport" in capsys.readouterr().err


def test_several_recipients_exit_with_the_worst_status(home, monkeypatch, capsys):
    _record(slurm_job_id=41, harness="claude", claude_session_id=SID)
    _record(mirror="L", harness="claude", claude_session_id="0" * 8 + SID[8:])
    code, trips = _run(monkeypatch, everyone=True, recipient="hi", times=[0, 0, 100],
                       answers=[_answer("seen"), _answer("ambiguous", candidates=["/x"])] * 3)
    assert code == 1                     # seen (4) and ambiguous (1): 1
    assert len(_probes(trips)) == 4      # both probed, on each of two polls
    assert replies.worst_exit([0, 4]) == 4 and replies.worst_exit([4, 1]) == 1 and replies.worst_exit([1, 3, 4]) == 3


def test_a_recipient_with_no_record_keeps_the_screen_path(home, monkeypatch, capsys):
    _record(slurm_job_id=41, harness="claude", claude_session_id=SID)
    clock = [0, 0, 100]
    code, trips = _run(monkeypatch, everyone=True, recipient="hi", times=clock, answers=[_answer("seen")] * 3)
    out = capsys.readouterr().out
    assert code == 4
    assert "no reply from L (login session on ln003.brc, t0) within 30s" in out
    assert any("capture-pane" in c and "sucoder-L" in c for _, c in trips)
    assert all("sucoder-L" not in c for c in _probes(trips))


def test_a_record_for_another_job_is_not_this_sessions(home, monkeypatch):
    _record(slurm_job_id=7, harness="claude", claude_session_id=SID)
    code, trips = _run(monkeypatch, times=[0, 0, 100])
    assert code == 0 and not _probes(trips)


def test_no_adapter_falls_back_to_the_screen(home, monkeypatch, capsys):
    _record(slurm_job_id=41, harness="claude", claude_session_id=SID)
    reply_pane = " ✨ [message 8e25ab12 from me@lyn via sucoder, t] hello\n ● REPLY 8e25ab12: from the screen\n"
    code, trips = _run(monkeypatch, answers=[_answer("no-adapter")] * 8, panes=["nothing\n", reply_pane])
    out = capsys.readouterr().out
    assert code == 0 and "M (job 41 on n0035, t0) replied: from the screen" in out
    # Login host first, then inside the job; the screen read only after.
    probes = _probes(trips)
    assert "srun --jobid=41" not in probes[0] and "srun --jobid=41" in probes[1]


@pytest.mark.parametrize("status", ["unrecognized", "unreadable"])
def test_a_transcript_that_could_not_be_read_is_exit_3_whatever_the_screen_shows(
        home, monkeypatch, capsys, status):
    # A pane row naming the id may be a tool call that quotes the protocol;
    # it does not overrule the reader's "could not look".
    _record(slurm_job_id=41, harness="claude", claude_session_id=SID)
    pane = ' ● Bash(echo "REPLY 8e25ab12: not really")\n'
    code, trips = _run(monkeypatch, answers=[_answer(status, why="x")] * 9, panes=[pane] * 9,
                       times=[0, 0, 0, 100])
    out = capsys.readouterr()
    assert code == 3 and f"{status}:" in out.err and "replied" not in out.out
    assert not any("capture-pane" in c for _, c in trips)


def test_a_probe_timeout_is_not_rerun_blind(home, monkeypatch, capsys):
    _record(slurm_job_id=41, harness="claude", claude_session_id=SID)
    code, trips = _run(monkeypatch, answers=[_answer("unrecognized", why="ProbeTimeout")] * 4)
    assert code == 3 and len(_probes(trips)) == 1
    assert "timed out" in capsys.readouterr().err


def test_a_transport_timeout_is_not_rerun_on_the_next_poll_or_the_job_route(home, monkeypatch, capsys):
    _record(slurm_job_id=41, harness="claude", claude_session_id=SID)
    reconnects = []
    code, trips = _run(monkeypatch, times=[0, 0, 0, 0, 100], probe_rc=124, reconnects=reconnects)
    err = capsys.readouterr().err
    assert code == 3 and "timed out" in err
    probes = _probes(trips)
    assert len(probes) == 1 and "srun" not in probes[0]   # one run, never the job route after it
    assert reconnects == []                               # and no reconnect-and-rerun


def test_an_ssh_failure_reconnects_and_tries_the_probe_once_more(home, monkeypatch, capsys):
    _record(slurm_job_id=41, harness="claude", claude_session_id=SID)
    reconnects = []
    code, trips = _run(monkeypatch, times=[0, 0, 100], probe_rc=255, reconnects=reconnects)
    assert code == 3 and "every reply probe failed in transport" in capsys.readouterr().err
    assert reconnects and len(_probes(trips)) == 2 * len(reconnects)


def test_the_shipped_command_answers_a_timeout_itself_with_exit_0():
    command = replies.probe_command({"id": ID})
    assert command.startswith("if command -v timeout >/dev/null 2>&1; then timeout -s TERM 75 python3 -I -c ")
    late = command.rsplit("echo ", 1)[1].strip("'")
    assert replies.parse_probe_output(late)["why"] == "ProbeTimeout"


def test_no_adapter_at_the_deadline_keeps_the_screen_exit(home, monkeypatch, capsys):
    _record(slurm_job_id=41, harness="claude", claude_session_id=SID)
    code, _ = _run(monkeypatch, answers=[_answer("no-adapter")] * 8, times=[0, 0, 100])
    assert code == 0 and "no reply from M (job 41 on n0035, t0) within 30s" in capsys.readouterr().out


def test_polling_stops_early_at_replied_and_at_a_turn_that_ended(home, monkeypatch, capsys):
    _record(slurm_job_id=41, harness="claude", claude_session_id=SID)
    code, trips = _run(monkeypatch, answers=[_answer("seen"), _answer("replied", reply="ok", source="transcript")]
                       + [_answer("seen")] * 5)
    assert code == 0 and len(_probes(trips)) == 2
    code, trips = _run(monkeypatch, answers=[_answer("seen", turn_ended=True)] + [_answer("seen")] * 5)
    out = capsys.readouterr().out
    assert code == 4 and len(_probes(trips)) == 1 and "its turn has ended" in out
    # Without the turn ending it polls until the deadline.
    code, trips = _run(monkeypatch, answers=[_answer("seen")] * 9, times=[0, 0, 0, 0, 100])
    assert code == 4 and len(_probes(trips)) == 4


def test_the_probe_carries_the_recorded_identity(home, monkeypatch):
    import base64
    _record(slurm_job_id=41, harness="claude", claude_session_id=SID)
    _, trips = _run(monkeypatch, answers=[_answer("replied", reply="ok", source="transcript")])
    command = _probes(trips)[0]
    word = command.split('"')[2].split()[0]
    spec = json.loads(base64.b64decode(word))
    assert spec["id"] == ID and spec["harness"] == "claude" and spec["claude_session_id"] == SID
    assert spec["header"].startswith(f"[message {ID} from me@lyn via sucoder, ")
    assert spec["cwds"] == ["~/mirrors/M", "/local/job41/mirrors/M"]
    assert "~/.sucoder/mail" in spec["mailboxes"]


def test_managed_codex_identity_comes_from_the_owner_record(home, monkeypatch):
    owner = {"command": ["codex"], "scope": "abc", "durable_root": "/g/mirrors/.sucoder-state"}
    session = _record(slurm_job_id=41, managed_owner=owner)
    from sucoder.messaging import Recipient
    r = Recipient("M", "ln001.brc", "sucoder-M", job_id=41, target="t0")
    identity = replies.transcript_identity(session, r, mirror_path="~/mirrors/M", local_root="/local/job41", token="M")
    assert identity["harness"] == "codex"
    assert identity["codex"] == {"durable_root": "/g/mirrors/.sucoder-state", "scope": "abc", "job": "41",
                                 "local_root": "/local/job41"}


def _identity(**fields):
    from sucoder.messaging import Recipient
    session = RemoteSession(mirror_name="M", target_name="t0", slurm_job_id=41, **fields)
    r = Recipient("M", "ln001.brc", "sucoder-M", job_id=41, target="t0")
    return replies.transcript_identity(session, r, mirror_path="~/mirrors/M", local_root="/local/job41", token="M")


def test_the_harness_the_last_launch_recorded_beats_a_leftover_identity():
    owner = {"command": ["codex"], "scope": "abc", "durable_root": "/g/state"}
    # A Codex launch after a Claude one: the Claude id is a leftover.
    got = _identity(harness="codex", claude_session_id=SID, managed_owner=owner)
    assert got["harness"] == "codex" and got["codex"]["scope"] == "abc" and "claude_session_id" not in got
    # A record older than the harness field: the managed owner first.
    assert _identity(claude_session_id=SID, managed_owner=owner)["harness"] == "codex"
    assert _identity(claude_session_id=SID)["harness"] == "claude"
    # A Claude launch after a managed Codex one keeps no Codex identity.
    got = _identity(harness="claude", claude_session_id=SID, managed_owner=owner)
    assert got["harness"] == "claude" and "codex" not in got
    # Unmanaged Codex, and a harness with no adapter (the screen path).
    unmanaged = _identity(harness="codex")
    assert unmanaged["harness"] == "codex" and "codex" not in unmanaged
    assert _identity(harness="gemini", managed_owner=owner) is None


def test_an_ambiguous_answer_never_prints_a_file_name_raw(home, monkeypatch, capsys):
    _record(slurm_job_id=41, harness="claude", claude_session_id=SID)
    osc = "\x1b]52;c;SGVsbG8=\x07"
    code, _ = _run(monkeypatch, answers=[_answer("ambiguous", candidates=[f"/p/{osc}.jsonl", "/p/b\u202e.jsonl"],
                                                 why=f"x{osc}")] * 3, times=[0, 0, 100])
    out = capsys.readouterr()
    text = out.out + out.err
    assert code == 1 and "\x1b" not in text and "\x07" not in text and "\u202e" not in text
    assert "/p/]52;c;SGVsbG8=.jsonl, /p/b.jsonl" in out.err
    for status in ("unreadable", "unrecognized"):
        cleaned = replies.parse_probe_output(_answer(status, why=f"denied: /p/{osc}\nnext"))
        assert cleaned["why"] == "denied: /p/]52;c;SGVsbG8= next"
    got = replies.parse_probe_output(_answer("replied", reply="ok", replies=[{"source": f"mail{osc}", "text": "ok"}]))
    assert got["replies"] == [{"source": "mail]52;c;SGVsbG8=", "text": "ok"}]


# ----------------------------------------------------------------- sanitising

def test_sanitize_strips_controls_and_bidi_and_caps():
    osc52 = "\x1b]52;c;ZWNobyBwd25lZA==\x07"
    text = f"ok {osc52}done‮⁦evil⁩\x9b31m\tcol\r\nnext​"
    clean = sanitize_reply(text)
    assert clean == "ok ]52;c;ZWNobyBwd25lZA==doneevil31m\tcol\nnext"
    assert not any(ord(c) < 0x20 and c not in "\n\t" for c in clean)
    long = sanitize_reply("\n".join(f"line {i}" for i in range(50)))
    assert len(long.split("\n")) == 20 and long.endswith("[truncated]")
    wide = sanitize_reply("é" * 3000)
    assert len(wide.encode("utf-8")) <= 2048 and wide.endswith("[truncated]")


def test_wait_prints_the_reply_sanitised(home, monkeypatch, capsys):
    _record(slurm_job_id=41, harness="claude", claude_session_id=SID)
    evil = "ok\x1b]52;c;cGF5bG9hZA==\x07 ‮sdrawkcab\n" + "\n".join(f"l{i}" for i in range(40))
    code, _ = _run(monkeypatch, answers=[_answer("replied", reply=evil, source="transcript",
                                                 replies=[{"source": "transcript", "text": evil}])])
    out = capsys.readouterr().out
    assert code == 0
    assert "\x1b" not in out and "\x07" not in out and "‮" not in out
    assert "replied (transcript): ok]52;c;cGF5bG9hZA== sdrawkcab" in out
    assert "l18" in out and "l19" not in out and "[truncated]" in out


# ----------------------------------------------------------------- peek --reply

def _peek(monkeypatch, reply, *, answers=(), mirror=None, relay_caller=None):
    harness = _SnapshotsHarness(times=("12-00:00:00",))
    monkeypatch.setattr(cli, "setup_logger", lambda *a, **k: logging.getLogger("t"))
    control = object()
    trips = []
    out = iter(answers)

    def fake(ctl, host, command, **kw):
        trips.append((host, command))
        if "python3 -I -c" in command:
            return SimpleNamespace(returncode=0, stdout=next(out, _answer("seen")), stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(cli, "_capture_over_tunnel", fake)
    monkeypatch.setattr(cli, "_run_remote_capture", fake)
    monkeypatch.setattr(cli, "_control_for_host", lambda remote, host, debug, **kw: control)
    obj = {"config": harness.config, "target_name": None}
    if relay_caller:
        obj["relay_caller"] = relay_caller
    ctx = SimpleNamespace(obj=obj, params={})
    code = 0
    try:
        cli.peek(ctx, mirror=mirror, lines=40, reply=reply, verbose=False)
    except typer.Exit as exc:
        code = exc.exit_code
    return code, trips


def _logged(msg_id=ID, account="ligon", identity=True, label="M (job 41 on n0035, t0)"):
    replies.append_sent({
        "id": msg_id, "frame": f"[message {msg_id} from x via sucoder, t] hi",
        "header": f"[message {msg_id} from x via sucoder, t]", "sent_at": "2026-10-23T13:59:58Z",
        "sent_epoch": 1792763998.0, "account": account,
        "recipient": {"label": label, "host": "ln001.brc", "session": "sucoder-M", "socket": "sucoder-M",
                      "job_id": 41, "node": "n0035", "target": "t0", "mirror": "M", "session_keys": ["M--t0"]},
        "identity": {"harness": "claude", "claude_session_id": SID, "cwds": ["~/mirrors/M"],
                     "mailboxes": ["~/.sucoder/mail"]} if identity else None,
    })


def test_peek_reply_reads_the_sent_log_and_asks_once(monkeypatch, capsys):
    _logged()
    code, trips = _peek(monkeypatch, ID, answers=[_answer("replied", reply="all quiet", source="transcript")])
    assert code == 0 and len(_probes(trips)) == 1
    assert "M (job 41 on n0035, t0) replied (transcript): all quiet" in capsys.readouterr().out
    code, _ = _peek(monkeypatch, ID, answers=[_answer("seen")])
    assert code == 4


def test_peek_reply_with_an_unknown_id_is_a_usage_error(monkeypatch, capsys):
    _logged()
    assert _peek(monkeypatch, "deadbeef")[0] == 2
    assert "unknown message id deadbeef" in capsys.readouterr().err
    assert _peek(monkeypatch, "../etc")[0] == 2


def test_relay_peek_reply_answers_only_the_callers_own_ids(monkeypatch, capsys):
    _logged(account="coder")
    _logged(msg_id="0badf00d", account="ligon")
    assert _peek(monkeypatch, ID, relay_caller="coder", answers=[_answer("seen")])[0] == 4
    code, trips = _peek(monkeypatch, "0badf00d", relay_caller="coder")
    assert code == 2 and trips == []
    assert "unknown message id 0badf00d" in capsys.readouterr().err


def test_the_relay_process_sanitises_what_reaches_the_caller(tmp_path, monkeypatch):
    _as_default_config(monkeypatch, tmp_path)
    monkeypatch.setenv("SUDO_USER", "coder")
    _logged(account="coder")
    control = object()
    evil = "ok\x1b]52;c;cGF5bG9hZA==\x07‮!"
    monkeypatch.setattr(cli, "_control_for_recipient", lambda *a: control)
    answer = lambda *a, **k: SimpleNamespace(   # noqa: E731
        returncode=0, stderr="", stdout=_answer("replied", reply=evil, source="transcript",
                                                replies=[{"source": "transcript", "text": evil}]))
    monkeypatch.setattr(cli, "_capture_over_tunnel", answer)
    monkeypatch.setattr(cli, "_run_remote_capture", answer)
    result = CliRunner().invoke(cli.app, ["relay", "peek", "--reply", ID])
    assert result.exit_code == 0, result.output
    assert "replied (transcript): ok]52;c;cGF5bG9hZA==!" in result.output
    assert "\x1b" not in result.output and "\x07" not in result.output and "‮" not in result.output
    # Another account's id is unknown to this caller.
    _logged(msg_id="0badf00d", account="ligon")
    assert CliRunner().invoke(cli.app, ["relay", "peek", "--reply", "0badf00d"]).exit_code == 2


def test_relayed_sends_log_the_caller_as_the_sending_account(home, monkeypatch):
    code, _ = _run(monkeypatch, wait=0, ctx_obj={"relay_caller": "coder"})
    assert code == 0
    (entry,) = replies.lookup_sent(ID)
    assert entry["account"] == "coder" and entry["relay_account"]


def test_peek_delegates_reply_to_the_relay(tmp_path, monkeypatch):
    import subprocess
    from tests.test_session_relay import _RELAY, _config
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: calls.append(argv) or subprocess.CompletedProcess(argv, 0, "", ""))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **k: None)
    path = _config(tmp_path, _RELAY)
    result = CliRunner().invoke(cli.app, ["--config", str(path), "peek", "--reply", ID])
    assert result.exit_code == 0, result.output
    assert calls[-1][-6:] == ["peek", "--lines", "40", "--reply", ID, "--"]
