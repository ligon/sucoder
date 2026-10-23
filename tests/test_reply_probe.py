"""The transcript reader and its remote probe (docs/reply-from-transcript.org).

Fixture lines follow the shapes recorded in docs/reply-transcripts-evidence.org
(Claude Code 2.1.309, Codex 0.163.0) and in two lyn transcripts, trimmed to the
keys the reader looks at plus enough of the rest to stay faithful; every text
is invented.
"""

import ast
import base64
import calendar
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sucoder import reply_probe as rp
from sucoder.replies import probe_command, probe_spec
from sucoder.sessions_report import job_tmux_command

PROBE = Path(rp.__file__)
ID = "8e25ab12"
SENDER = "coder@lyn (relay)"
HEADER = f"[message {ID} from {SENDER} via sucoder, 2026-10-23 06:59]"
FRAME = f"{HEADER} Relay test from the SuCoder agent. No action needed; please just reply ok."
SENT = calendar.timegm((2026, 10, 23, 13, 59, 58, 0, 0, 0))
SID = "83cd9e50-d565-4635-95a2-da82d4b334cd"


def ts(seconds_after):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(SENT + seconds_after)) + ".057Z"


# ----------------------------------------------------------------- Claude shapes

def c_user(content, at=2, **extra):
    entry = {
        "parentUuid": None, "isSidechain": False, "promptId": "926b7031-0000", "type": "user",
        "message": {"role": "user", "content": content},
        "uuid": "40183fc8-0000", "timestamp": ts(at), "permissionMode": "bypassPermissions",
        "origin": {"kind": "human"}, "promptSource": "typed", "userType": "external",
        "entrypoint": "cli", "cwd": "/home/u/mirrors/M", "sessionId": SID, "version": "2.1.309",
    }
    entry.update(extra)
    return entry


def c_paste(text):
    return f'\n\n<pasted_content id="935c">\n{text}\n</pasted_content id="935c">\n'


def c_queue(operation, at, content=None, **extra):
    entry = {"type": "queue-operation", "operation": operation, "timestamp": ts(at), "sessionId": SID}
    if content is not None:
        entry["content"] = content
    entry.update(extra)
    return entry


def c_queued_command(prompt, at=40):
    return {
        "parentUuid": "9ff816c9-0000", "isSidechain": False, "type": "attachment",
        "attachment": {"type": "queued_command", "prompt": prompt, "source_uuid": "4b15136f-0000",
                       "delivery_id": "b74cf907-0000", "commandMode": "prompt", "origin": {"kind": "human"},
                       "timestamp": ts(1), "humanTurn": True},
        "rendered": [{"content": "<system-reminder>\nThe user sent a new message while you were working:\n"
                                 + prompt}],
        "renderedRole": "system", "timestamp": ts(1), "sessionId": SID,
    }


def c_tool_result(text, at=30):
    return c_user([{"tool_use_id": "toolu_01", "type": "tool_result", "content": text}], at=at)


def c_text(text, at=5, stop="tool_use", **extra):
    entry = {
        "parentUuid": "c58a866f-0000", "isSidechain": False,
        "message": {"model": "claude-opus-5-5", "id": "msg_011", "type": "message", "role": "assistant",
                    "content": [{"type": "text", "text": text}], "stop_reason": stop},
        "requestId": "req_011", "type": "assistant", "uuid": "ba473610-0000", "timestamp": ts(at),
        "sessionId": SID,
    }
    entry.update(extra)
    return entry


def c_block(block, at=5, stop="tool_use"):
    entry = c_text("", at=at, stop=stop)
    entry["message"]["content"] = [block]
    return entry


def c_tool_use(command, at=6):
    return c_block({"type": "tool_use", "id": "toolu_01", "name": "Bash", "input": {"command": command}}, at=at)


def c_last_prompt(text):
    return {"type": "last-prompt", "lastPrompt": text[:150] + "…", "leafUuid": "x", "sessionId": SID}


def c_blob(text, role="user"):
    return {"type": "api-request-blob", "sessionId": SID, "hash": "0" * 64,
            "message": {"role": role, "content": [{"type": "text", "text": text}]}}


def c_compact_summary(text, at=900):
    return c_user(text, at=at, isCompactSummary=True)


def lines(*entries, tail=True):
    data = "".join(json.dumps(e) + "\n" for e in entries)
    return data.encode("utf-8")


def claude(*entries, data=None):
    return rp.read_transcript(data if data is not None else lines(*entries), "claude", ID, HEADER, SENT)


# ----------------------------------------------------------------- Codex shapes

def x_meta(source="user", cwd="/home/u/mirrors/M", at=-3600, thread="01a168ea-5cfb-76b1-98d0-ab68f8fd2af6"):
    payload = {"session_id": thread, "id": thread, "timestamp": ts(at), "cwd": cwd, "originator": "sucoder",
               "cli_version": "0.163.0", "source": "vscode", "thread_source": source, "model_provider": "openai"}
    if source == "subagent":
        payload["forked_from_id"] = "01a168ea-5cfb-76b1-98d0-ab68f8fd2af6"
    return {"timestamp": ts(at), "ordinal": 0, "type": "session_meta", "payload": payload}


def x_user(text, at=1, ordinal=756):
    return {"timestamp": ts(at), "ordinal": ordinal, "type": "response_item",
            "payload": {"type": "message", "id": "msg_01a1690f", "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                        "internal_chat_message_metadata_passthrough": {"turn_id": "01a16900", "create_time": 1.0,
                                                                       "content_item_kinds": ["user.text"]}}}


def x_twin(text, at=1):
    return {"timestamp": ts(at), "ordinal": 757, "type": "event_msg",
            "payload": {"type": "item_completed", "thread_id": "t", "turn_id": "01a16900",
                        "item": {"type": "UserMessage", "id": "i", "client_id": "c",
                                 "content": [{"type": "text", "text": text, "text_elements": []}]}}}


def x_assistant(text, phase="commentary", at=3):
    return {"timestamp": ts(at), "ordinal": 761, "type": "response_item",
            "payload": {"type": "message", "status": "completed", "id": "msg_096a", "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}],
                        "phase": phase,
                        "internal_chat_message_metadata_passthrough": {"turn_id": "01a16900"}}}


def x_agent_message(text, at=4):
    return {"timestamp": ts(at), "ordinal": 762, "type": "response_item",
            "payload": {"type": "agent_message", "author": "/root/euler", "recipient": "/root",
                        "content": [{"type": "output_text", "text": text}]}}


def x_compacted(text, at=900):
    return {"timestamp": ts(at), "ordinal": 2371, "type": "compacted",
            "payload": {"message": "", "replacement_history": [
                {"type": "message", "id": "msg_01a1690f", "role": "user",
                 "content": [{"type": "input_text", "text": text}]}], "window_number": 2}}


def x_task_complete(at=60):
    return {"timestamp": ts(at), "ordinal": 800, "type": "event_msg",
            "payload": {"type": "task_complete", "turn_id": "01a16900", "last_agent_message": "done"}}


def x_reasoning(at=2):
    return {"timestamp": ts(at), "ordinal": 758, "type": "response_item",
            "payload": {"type": "reasoning", "summary": [], "encrypted_content": "xx"}}


def codex(*entries, data=None):
    return rp.read_transcript(data if data is not None else lines(*entries), "codex", ID, HEADER, SENT)


# ================================================================= Claude reader

def test_claude_idle_delivery_is_a_user_string_and_its_reply_is_found():
    got = claude(c_user(FRAME), c_last_prompt(FRAME), c_blob(FRAME),
                 c_block({"type": "thinking", "thinking": "..."}), c_text("REPLY 8e25ab12: ok.", stop="end_turn"))
    assert got["status"] == "replied" and got["reply"] == "ok."


def test_claude_pasted_delivery_is_unwrapped_once():
    got = claude(c_user(c_paste(FRAME)), c_text("REPLY 8e25ab12: ok, pasted."))
    assert got == {"status": "replied", "reply": "ok, pasted.", "source": "transcript"}


def test_claude_mid_turn_delivery_is_a_queued_command_attachment():
    got = claude(
        c_queue("enqueue", 1, FRAME), c_tool_result("tests passed", at=36),
        c_queue("remove", 36, FRAME, reason="absorbed_mid_turn", commandUuid="4b", deliveryId="b7"),
        c_queued_command(FRAME), c_tool_use("pytest -q", at=40), c_text("REPLY 8e25ab12: ok, mid-turn.", at=41),
    )
    assert got["status"] == "replied" and got["reply"] == "ok, mid-turn."


def test_claude_queue_only_is_queued():
    got = claude(c_text("working", at=-60), c_queue("enqueue", 1, FRAME), c_tool_use("make", at=3))
    assert got == {"status": "queued"}


def test_claude_turn_end_dequeue_delivers_as_a_user_entry():
    # Observed on lyn: enqueue -> dequeue (no content) -> a user entry byte-equal to the frame.
    got = claude(c_queue("enqueue", 1, FRAME), c_text("done", at=3, stop="end_turn"),
                 c_queue("dequeue", 4), c_user(FRAME, at=4), c_text("REPLY 8e25ab12: ok after the turn", at=6))
    assert got["status"] == "replied" and got["reply"] == "ok after the turn"


def test_a_compaction_summary_claiming_a_reply_is_not_one():
    # LSMS_Library, message 1733: delivered, never answered, and the summary
    # (with its api-request-blob copies) says it was.
    claim = f"Summary: ... the relay message {HEADER} (answered \"REPLY {ID}: ok\")"
    got = claude(c_queue("enqueue", 1, FRAME), c_queued_command(FRAME), c_tool_use("bash work.sh", at=41),
                 c_blob(claim), c_compact_summary("\nREPLY 8e25ab12: ok\n" + claim), c_blob(claim, role="assistant"))
    assert got["status"] == "seen"


def test_request_blob_and_last_prompt_copies_are_ignored():
    # The only copies of the frame and of a reply are a last-prompt and blobs: no delivery.
    got = claude(c_text("unrelated", at=10), c_last_prompt(FRAME), c_blob(FRAME),
                 c_blob(f"REPLY {ID}: ok", role="assistant"))
    assert got == {"status": "absent"}
    # Beside a real delivery they change nothing: still no reply.
    got = claude(c_user(FRAME), c_last_prompt(FRAME), c_blob(FRAME), c_blob(f"REPLY {ID}: ok", role="assistant"))
    assert got["status"] == "seen"


def test_a_tool_result_holding_the_frame_is_not_a_delivery():
    got = claude(c_tool_use("grep message ~/.sucoder/sent.jsonl", at=5), c_tool_result(FRAME, at=6))
    assert got == {"status": "absent"}


def test_a_tool_use_input_holding_the_reply_is_not_a_reply():
    got = claude(c_user(FRAME), c_tool_use(f"echo 'REPLY {ID}: ok' >> notes.org"))
    assert got == {"status": "seen", "turn_ended": False}


def test_a_reply_line_inside_a_longer_text_block_counts():
    got = claude(c_user(FRAME), c_text(f"I'm Sue, on Claude.\nFirst some context.\n● REPLY {ID}: ok -- the build\nis green."))
    assert got["status"] == "replied" and got["reply"] == "ok -- the build\nis green."


def test_a_bare_reply_counts_only_with_no_other_input_between():
    assert claude(c_user(FRAME), c_text("REPLY: ok"))["reply"] == "ok"
    got = claude(c_user(FRAME), c_user("what is the status?", at=10), c_text("REPLY: busy", at=12))
    assert got["status"] == "seen"


def test_reading_stops_at_the_next_delivered_input():
    got = claude(c_user(FRAME), c_queued_command("please also check the logs", at=20),
                 c_text(f"REPLY {ID}: ok", at=30))
    assert got["status"] == "seen"


def test_sidechain_entries_are_ignored():
    got = claude(c_text("main thread", at=10),
                 c_user(FRAME, isSidechain=True), c_text(f"REPLY {ID}: ok from a sub-agent", isSidechain=True))
    assert got == {"status": "absent"}


def test_claude_turn_end_after_the_delivery_is_seen_with_turn_ended():
    got = claude(c_user(FRAME), c_text("Done with the refactor.", stop="end_turn"))
    assert got == {"status": "seen", "turn_ended": True}


# ================================================================= Codex reader

def test_codex_main_thread_delivery_and_final_reply():
    got = codex(x_meta(), x_user(FRAME), x_twin(FRAME), x_reasoning(), x_assistant(f"REPLY {ID}: ok.", "final_answer"))
    assert got["status"] == "replied" and got["reply"] == "ok."


def test_codex_commentary_reply_mid_turn_counts():
    # CerealDemand, message 8e25: commentary, 2s after delivery, an hour before task_complete.
    got = codex(x_user(FRAME), x_twin(FRAME), x_reasoning(), x_assistant(f"REPLY {ID}: ok.", "commentary"),
                x_assistant("Now running the estimation.", "commentary", at=5), x_task_complete(at=3600))
    assert got == {"status": "replied", "reply": "ok.", "source": "transcript"}


def test_a_compacted_replay_is_not_a_delivery():
    got = codex(x_assistant("unrelated", at=800), x_compacted(FRAME))
    assert got == {"status": "absent"}


def test_agent_message_items_are_ignored():
    got = codex(x_user(FRAME), x_agent_message(f"REPLY {ID}: ok from a sub-agent"))
    assert got == {"status": "seen", "turn_ended": False}


def test_codex_task_complete_after_the_delivery_ends_the_turn():
    got = codex(x_user(FRAME), x_assistant("Working on it", "commentary"), x_task_complete())
    assert got == {"status": "seen", "turn_ended": True}


# ================================================================= shared rules

def test_an_embedded_frame_after_other_text_does_not_match():
    forwarded = f"Forwarding this for you: {FRAME}"
    assert not rp.anchored(forwarded, HEADER)
    assert rp.anchored("  \n" + FRAME, HEADER) and rp.anchored(c_paste(FRAME), HEADER)
    got = claude(c_user(forwarded), c_text(f"REPLY {ID}: ok", at=6))
    assert got == {"status": "absent"}
    # The same id under another sender is not our frame either.
    other = FRAME.replace(SENDER, "mallory@elsewhere")
    assert codex(x_user(other), x_assistant(f"REPLY {ID}: ok"))["status"] == "absent"


def test_absent_needs_an_entry_after_the_send():
    assert claude(c_text("after the send", at=5))["status"] == "absent"
    stale = claude(c_text("long before", at=-3600))
    assert stale["status"] == "unrecognized" and "after the send" in stale["why"]
    # Within the skew allowance still counts as after.
    assert claude(c_text("just before", at=-10))["status"] == "absent"


def test_the_frame_in_an_unknown_entry_type_is_unrecognized_not_absent():
    got = claude({"type": "brand-new-entry", "timestamp": ts(5), "copy": FRAME})
    assert got["status"] == "unrecognized" and got["why"] == rp.WHY_SHAPE


def test_a_partial_last_line_is_skipped_not_a_parse_error():
    data = lines(c_user(FRAME)) + json.dumps(c_text(f"REPLY {ID}: ok"))[:60].encode()
    assert claude(data=data) == {"status": "seen", "turn_ended": False}
    whole = lines(c_user(FRAME)) + json.dumps(c_text(f"REPLY {ID}: ok")).encode()   # complete but unterminated
    assert claude(data=whole)["status"] == "seen"
    assert claude(data=whole + b"\n")["status"] == "replied"


def test_timestamps_parse_as_utc():
    assert rp.entry_time({"timestamp": "2026-10-23T13:59:58Z"}) == SENT
    assert rp.entry_time({"timestamp": "2026-10-23T06:59:58-07:00"}) == SENT
    assert rp.entry_time({"timestamp": "2026-10-23T13:59:58.5Z"}) == SENT + 0.5
    assert rp.entry_time({"timestamp": 5}) is None


# ================================================================= the probe file

def test_the_probe_is_a_standalone_python36_module():
    source = PROBE.read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 6))
    tree = ast.parse(source)
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported <= {"base64", "calendar", "errno", "glob", "json", "os", "re", "signal", "stat", "sys", "time"}
    old = shutil.which("python3.6")
    if old is None:
        pytest.skip("no python3.6 to compile against")
    subprocess.run([old, "-m", "py_compile", str(PROBE)], check=True)


def _claude_home(tmp_path, *entries, session_id=SID, cwd="/home/u/mirrors/M"):
    home = tmp_path / "home"
    project = home / ".claude" / "projects" / rp.encode_project(cwd)
    project.mkdir(parents=True)
    path = project / f"{session_id}.jsonl"
    path.write_bytes(lines(*entries))
    return home, path


def _spec(**identity):
    base = {"harness": "claude", "claude_session_id": SID, "cwds": ["/home/u/mirrors/M"],
            "mailboxes": ["~/.sucoder/mail"], "start_date": "2026-10-23"}
    base.update(identity)
    return probe_spec(base, msg_id=ID, header=HEADER, sent_at=SENT)


def test_the_shipped_command_runs_through_the_job_route_quoting(tmp_path):
    home, _ = _claude_home(tmp_path, c_user(c_paste(FRAME)), c_text(f"REPLY {ID}: it's \"quoted\" & $fine"))
    # job_tmux_command's srun branch, with a stand-in srun that drops its
    # options and runs the rest: the quoting is exactly what reaches a job.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "srun").write_text('#!/bin/sh\nwhile [ $# -gt 0 ]; do case "$1" in --*) shift;; *) break;; esac; done\n'
                                  'exec "$@"\n')
    (bin_dir / "srun").chmod(0o755)
    command = job_tmux_command(probe_command(_spec()), job_id=41, session="sucoder-M", socket="sucoder-M")
    assert command.startswith("TMPDIR=/tmp srun --jobid=41 --overlap")
    env = {"HOME": str(home), "PATH": f"{bin_dir}:{Path(sys.executable).parent}:/usr/bin:/bin"}
    done = subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True, timeout=60)
    out = done.stdout.splitlines()
    assert len(out) == 1, done.stdout + done.stderr
    answer = json.loads(out[0])
    assert answer["schema"] == rp.SCHEMA and answer["status"] == "replied"
    assert answer["reply"] == 'it\'s "quoted" & $fine' and answer["harness"] == "claude"
    # The same command through a plain shell (the login-host route).
    plain = subprocess.run(["bash", "-c", probe_command(_spec())], env=env, capture_output=True, text=True, timeout=60)
    assert json.loads(plain.stdout)["status"] == "replied"


def test_any_exception_in_the_probe_is_one_unrecognized_line(tmp_path, capsys, monkeypatch):
    rp.main(["not base64 json"])
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and json.loads(out[0])["status"] == "unrecognized"

    def boom(spec, proc_root="/proc"):
        raise RuntimeError("secret detail")
    monkeypatch.setattr(rp, "probe", boom)
    rp.main([base64.b64encode(json.dumps(_spec()).encode()).decode()])
    line = capsys.readouterr().out
    assert json.loads(line) == {"schema": rp.SCHEMA, "status": "unrecognized", "why": "RuntimeError"}
    assert "Traceback" not in line and "secret detail" not in line

    # And as shipped: a spec the probe rejects never prints a traceback.
    bad = _spec()
    bad["header"] = "[message deadbeef from x via sucoder, t]"
    done = subprocess.run(["bash", "-c", probe_command(bad)], capture_output=True, text=True, timeout=60,
                          env={"HOME": str(tmp_path), "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin"})
    assert json.loads(done.stdout) == {"schema": rp.SCHEMA, "status": "unrecognized", "why": "ValueError"}


def test_the_probe_keeps_only_two_variables_from_proc_environ(tmp_path):
    proc = tmp_path / "proc"
    (proc / "4242").mkdir(parents=True)
    (proc / "4242" / "environ").write_bytes(
        b"HOME=/home/u\0ANTHROPIC_API_KEY=sk-secret\0CLAUDE_CONFIG_DIR=/cfg/claude\0"
        b"OPENAI_API_KEY=sk-other\0CODEX_HOME=/cfg/codex\0GITHUB_TOKEN=ghp_x\0")
    (proc / "self").mkdir()
    (proc / "self" / "environ").write_bytes(b"CLAUDE_CONFIG_DIR=/not-a-pid\0")
    found = rp.environ_dirs(str(proc))
    assert found == {"CLAUDE_CONFIG_DIR": ["/cfg/claude"], "CODEX_HOME": ["/cfg/codex"]}
    assert "secret" not in repr(found) and "ghp_" not in repr(found)
    assert rp.ENV_KEYS == ("CLAUDE_CONFIG_DIR", "CODEX_HOME")
    # Code review, mechanised: environ_dirs is the only reader of an environ file.
    source = PROBE.read_text(encoding="utf-8")
    assert source.count('"environ"') == 1
    assert source.index('"environ"') > source.index("def environ_dirs")
    assert source.index('"environ"') < source.index("def encode_project")


def test_claude_falls_back_to_the_project_directory_and_reports_ambiguity(tmp_path, monkeypatch):
    home, named = _claude_home(tmp_path, c_text("before the send", at=-3600))
    os.utime(named, (SENT - 3600, SENT - 3600))
    monkeypatch.setenv("HOME", str(home))
    proc = tmp_path / "noproc"
    # The recorded id names a stale file (a --continue started a new one).
    other = named.parent / "5d1c0de0-0000-4000-8000-000000000000.jsonl"
    other.write_bytes(lines(c_user(FRAME), c_text(f"REPLY {ID}: ok from the new file")))
    got = rp.probe(_spec(), str(proc))
    assert got["status"] == "replied" and got["transcript"] == str(other)
    third = named.parent / "6d1c0de0-0000-4000-8000-000000000000.jsonl"
    third.write_bytes(lines(c_user(FRAME)))
    got = rp.probe(_spec(), str(proc))
    assert got["status"] == "ambiguous" and sorted(got["candidates"]) == sorted([str(other), str(third)])


def test_nothing_identified_is_no_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert rp.probe(_spec(), str(tmp_path / "noproc"))["status"] == "no-adapter"
    assert rp.probe(_spec(harness="gemini"), str(tmp_path / "noproc"))["status"] == "no-adapter"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_a_transcript_that_cannot_be_read_is_unreadable_not_absent(tmp_path, monkeypatch):
    home, named = _claude_home(tmp_path, c_user(FRAME))
    monkeypatch.setenv("HOME", str(home))
    named.chmod(0)
    try:
        got = rp.probe(_spec(), str(tmp_path / "noproc"))
    finally:
        named.chmod(0o600)
    assert got["status"] == "unreadable"


# ----------------------------------------------------------------- managed and unmanaged Codex

SCOPE = "0f536c4f"
USER_THREAD = "01a168ea-5cfb-76b1-98d0-ab68f8fd2af6"
SUB_THREAD = "01a16936-0000-7000-8000-000000000000"


def _managed_run(tmp_path, *, job="41", run="c48a7b09", user_entries=(), lag=0):
    durable = tmp_path / "state"
    run_dir = durable / "codex" / SCOPE / "runs" / run
    (run_dir / "rollouts").mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "format": 1, "scope": SCOPE, "run": run, "allocation": {"job": job, "node": "n0198.savio4"},
        "codex_home": "/x", "created": SENT - 3600}))
    user = run_dir / "rollouts" / f"{USER_THREAD}.jsonl"
    user.write_bytes(lines(x_meta("user", thread=USER_THREAD), *user_entries))
    # A forked sub-agent carries a copy of the delivery with the parent's item id.
    sub = run_dir / "rollouts" / f"{SUB_THREAD}.jsonl"
    sub.write_bytes(lines(x_meta("subagent", thread=SUB_THREAD), x_user(FRAME, ordinal=5),
                          x_assistant(f"REPLY {ID}: ok from the fork")))
    local = tmp_path / "local" / f"job{job}"
    local_run = local / "state" / "codex" / SCOPE / run
    (local_run / "home" / "sessions" / "2026" / "10" / "23").mkdir(parents=True)
    rel = f"2026/10/23/rollout-2026-10-23T06-19-12-{USER_THREAD}.jsonl"
    (local_run / "status.json").write_text(json.dumps({"replica": {"lag_bytes": lag, "threads": {
        USER_THREAD: {"local_bytes": 100 + lag, "replicated_bytes": 100, "relative_path": rel},
        SUB_THREAD: {"local_bytes": 50, "replicated_bytes": 50,
                     "relative_path": f"2026/10/23/rollout-x-{SUB_THREAD}.jsonl"}}}}))
    return durable, local, user, local_run / "home" / "sessions" / rel


def _codex_spec(durable, local, job="41"):
    return _spec(harness="codex", claude_session_id=None,
                 codex={"durable_root": str(durable), "scope": SCOPE, "job": job, "local_root": str(local)})


def test_managed_codex_picks_the_user_thread_of_this_job(tmp_path):
    durable, local, user, _ = _managed_run(tmp_path, user_entries=[x_user(FRAME), x_assistant(f"REPLY {ID}: ok.")])
    # Another allocation's run of the same scope is not this session.
    _managed_run(tmp_path / "older", job="40", run="aaaa0000")
    paths = rp.managed_codex_threads({"durable_root": str(durable), "scope": SCOPE, "job": "41",
                                      "local_root": str(local)}, SENT - 30)
    assert paths == [str(user)]
    got = rp.probe(_codex_spec(durable, local), str(tmp_path / "noproc"))
    assert got["status"] == "replied" and got["reply"] == "ok." and got["transcript"] == str(user)


def test_managed_codex_reads_the_node_local_rollout_when_the_replica_lags(tmp_path):
    durable, local, user, local_path = _managed_run(tmp_path, user_entries=[x_user(FRAME)], lag=10)
    local_path.write_bytes(lines(x_meta("user", thread=USER_THREAD), x_user(FRAME), x_assistant(f"REPLY {ID}: fresh")))
    got = rp.probe(_codex_spec(durable, local), str(tmp_path / "noproc"))
    assert got["status"] == "replied" and got["transcript"] == str(local_path)


def test_unmanaged_codex_is_found_by_cwd_and_thread_source(tmp_path, monkeypatch):
    home = tmp_path / "home"
    day = home / ".codex" / "sessions" / "2026" / "10" / "23"
    day.mkdir(parents=True)
    mine = day / f"rollout-2026-10-23T06-19-12-{USER_THREAD}.jsonl"
    mine.write_bytes(lines(x_meta("user"), x_user(FRAME), x_assistant(f"REPLY {ID}: ok")))
    (day / f"rollout-2026-10-23T06-20-00-{SUB_THREAD}.jsonl").write_bytes(
        lines(x_meta("subagent"), x_user(FRAME), x_assistant(f"REPLY {ID}: fork")))
    (day / "rollout-2026-10-23T06-21-00-elsewhere.jsonl").write_bytes(
        lines(x_meta("user", cwd="/home/u/mirrors/Other"), x_user(FRAME)))
    monkeypatch.setenv("HOME", str(home))
    spec = _spec(harness="codex", claude_session_id=None)
    paths = rp.unmanaged_codex_threads(spec, str(tmp_path / "noproc"), now=SENT + 60)
    assert paths == [str(mine)]
    got = rp.find_codex(spec, str(tmp_path / "noproc"))
    assert got["status"] == "replied" and got["transcript"] == str(mine)


# ----------------------------------------------------------------- mailbox

def _mail(tmp_path, body=b"ok from the mailbox\n", name=f"{ID}.reply"):
    box = tmp_path / "mail"
    box.mkdir(exist_ok=True)
    path = box / name
    path.write_bytes(body)
    os.utime(path, (SENT + 5, SENT + 5))
    return box, path


def test_a_mailbox_file_is_a_reply(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    box, _ = _mail(tmp_path)
    got = rp.probe(_spec(mailboxes=[str(box)]), str(tmp_path / "noproc"))
    assert got["status"] == "replied" and got["source"] == "mailbox" and got["reply"] == "ok from the mailbox"


def test_a_mailbox_reply_beside_a_transcript_reply_lists_the_transcript_first(tmp_path, monkeypatch):
    home, _ = _claude_home(tmp_path, c_user(FRAME), c_text(f"REPLY {ID}: ok"))
    monkeypatch.setenv("HOME", str(home))
    box, _ = _mail(tmp_path)
    got = rp.probe(_spec(mailboxes=[str(box)]), str(tmp_path / "noproc"))
    assert got["source"] == "transcript"
    assert [r["source"] for r in got["replies"]] == ["transcript", "mailbox"]


def test_a_symlinked_stale_or_oversize_mailbox_file_is_ignored(tmp_path):
    box, path = _mail(tmp_path)
    assert rp.read_mailbox([str(box)], ID, SENT) == "ok from the mailbox"
    os.utime(path, (SENT - 3600, SENT - 3600))
    assert rp.read_mailbox([str(box)], ID, SENT) is None                    # stale
    path.write_bytes(b"x" * (rp.MAILBOX_MAX + 1))
    os.utime(path, (SENT + 5, SENT + 5))
    assert rp.read_mailbox([str(box)], ID, SENT) is None                    # oversize
    path.unlink()
    real = tmp_path / "elsewhere.txt"
    real.write_bytes(b"planted")
    os.utime(real, (SENT + 5, SENT + 5))
    path.symlink_to(real)
    assert rp.read_mailbox([str(box)], ID, SENT) is None                    # symlink
    assert rp.read_mailbox([str(box)], "../../etc", SENT) is None           # not an id
