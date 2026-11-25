"""Unattended dreaming records human questions without waiting for an answer."""
import json
import os
import sys
import time

import click
import pytest
import typer
from typer.testing import CliRunner

from sucoder import cli, dream_native, dream_reader
from tests.test_dream import git, workspace  # noqa: F401
from tests.test_dream_wiki import EditorTransport, POLICY, TOPIC, planned


QUESTION = (
    "\n*** WAITING Choose transcript-identity launch policy\n"
    ":PROPERTIES:\n:CUSTOM_ID: open-launch-question\n:OWNER: User\n"
    ":REQUIRES_HUMAN: yes\n"
    ":BLOCKER: The admitted evidence contains no maintainer decision for this policy.\n"
    ":RESUME_WHEN: A maintainer records the launch-policy decision.\n:END:\n"
    "Should a launch require an explicitly recorded transcript session identity?\n"
    "The documented launch capability does not establish that policy; this question remains open.\n"
)


def test_native_capture_has_no_human_terminal_and_only_finite_assignment_stdin(tmp_path):
    script = """
import json,os,sys
payload=sys.stdin.buffer.read()
try:
    fd=os.open('/dev/tty', os.O_RDWR)
except OSError:
    human_terminal=False
else:
    os.close(fd)
    human_terminal=True
print(json.dumps(dict(payload=payload.decode(), at_eof=sys.stdin.buffer.read()==b'',
                     tty=[os.isatty(fd) for fd in (0,1,2)], human_terminal=human_terminal)))
"""
    payload = b'Frozen role assignment.\n'
    code, stdout, stderr = dream_native.capture([sys.executable, "-S", "-c", script], payload,
        tmp_path, dict(PATH=os.defpath), time.monotonic() + 5)
    assert code == 0 and stderr == b""
    assert json.loads(stdout) == dict(payload=payload.decode(), at_eof=True,
        tty=[False, False, False], human_terminal=False)


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_headless_cli_records_human_question_and_finishes_independent_work(workspace, monkeypatch, harness):
    plan = planned(workspace)

    def record_question(task, result):
        if task["role"] == "reviewer":
            return
        content = result["pages"][0]["content"]
        result["pages"][0]["content"] = content.replace(POLICY, "** Talk :editorial:\n" + QUESTION + "\n" + POLICY, 1)
        source = next(s for s in task["input"]["sources"] if s["unit_id"] == "evidence-b")
        result["references"].append(dict(path=TOPIC, target="open-launch-question", evidence_status="inference",
            citations=[dict(unit_id=source["unit_id"], quote=source["text"])]))

    transport = EditorTransport(mutate=record_question)
    monkeypatch.setattr(dream_native, "Transport", lambda *args, **kwargs: transport)

    def no_human_prompt(*args, **kwargs):
        raise AssertionError("Unattended dream attempted to question the human")

    monkeypatch.setattr("builtins.input", no_human_prompt)
    for module in (click, typer):
        monkeypatch.setattr(module, "prompt", no_human_prompt)
        monkeypatch.setattr(module, "confirm", no_human_prompt)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["--config", str(workspace[1]), "dream", "run", "sample",
        "--plan", plan["plan_path"], "--harness", harness, "--model", "explicit-fixture-model"], input="")
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["state"] == "candidate_ready"
    assert [task["role"] for task in transport.calls] == ["author", "reviewer"]
    content = git(workspace[0], "show", value["candidate_commit"] + ":" + TOPIC)
    assert "A compaction summary's assertion is not proof" in content
    assert POLICY.strip() in content and QUESTION.strip() in content
    assert git(workspace[0], "status", "--porcelain") == ""

    active = dream_reader.read(workspace[2], "sample", "reply-evidence", target="open-launch-question",
        base=value["candidate_commit"])
    assert "Should a launch require" in active["text"]
    assert "WAITING" in active["text"] and "this question remains open" in active["text"]

    status = runner.invoke(cli.app, ["--config", str(workspace[1]), "dream", "status", "sample",
        "--plan", plan["plan_path"], "--run", value["run_id"]], input="")
    assert status.exit_code == 0, status.output
    assert json.loads(status.output)["state"] == "candidate_ready"
    assert len(transport.calls) == 2


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_native_command_disables_interactive_approval(harness):
    command = dream_native.command(harness, "native-fixture", "explicit-model", dict(type="object"))
    if harness == "codex":
        assert "exec" in command and 'approval_policy="never"' in command and command[-1] == "-"
    else:
        assert "--print" in command
        assert command[command.index("--permission-mode") + 1] == "dontAsk"
        assert command[command.index("--permission-prompts") + 1] == "none"


def test_missing_model_returns_error_without_question_or_provider_start(workspace, monkeypatch):
    plan = planned(workspace)

    def forbidden(*args, **kwargs):
        raise AssertionError("Missing setup attempted human interaction or provider startup")

    monkeypatch.setattr("builtins.input", forbidden)
    for module in (click, typer):
        monkeypatch.setattr(module, "prompt", forbidden)
        monkeypatch.setattr(module, "confirm", forbidden)
    monkeypatch.setattr(dream_native, "Transport", forbidden)
    result = CliRunner().invoke(cli.app, ["--config", str(workspace[1]), "dream", "run", "sample",
        "--plan", plan["plan_path"], "--harness", "codex"], input="")
    assert result.exit_code == 2, result.output
    value = json.loads(result.output)
    assert value["state"] == "rejected" and "model" in value["error"].lower()
