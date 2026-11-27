"""Native transport controls, kernel enforcement and observed trace shapes."""
import json
import os
from pathlib import Path
import shutil
import sys
import time

import pytest

from sucoder import dream, dream_native as native


def boundary_or_skip(path):
    try:
        return native.FilesystemBoundary(path)
    except dream.DreamError as exc:
        pytest.skip(str(exc))


def test_kernel_boundary_denies_outside_reads_and_writes(tmp_path):
    scratch = tmp_path / "role"
    scratch.mkdir()
    secret = tmp_path / "unrelated-secret"
    secret.write_text("canary outside the role")
    boundary = boundary_or_skip(scratch)
    try:
        # Production admits the harness executable it starts (invoke).
        # An interpreter also reads its standard library, which is outside
        # /usr when setup-python installs it under /opt/hostedtoolcache.
        python = Path(sys.executable).resolve()
        for path in (python, Path(sys.base_prefix).resolve()):
            assert path not in secret.resolve().parents
            boundary.allow(path, native.READ)
        script = """
import os,sys
open('allowed','w').write('role output')
for operation in ('read','write'):
    try:
        open(sys.argv[1], 'r' if operation == 'read' else 'w')
    except PermissionError:
        print(operation + ':denied')
    else:
        raise SystemExit('boundary failed')
try:
    open('/proc/1/environ','rb')
except PermissionError:
    print('process-environment:denied')
else:
    raise SystemExit('process read escaped')
"""
        code, stdout, _ = native.capture([str(python), "-S", "-c", script, str(secret)], b"", scratch,
            dict(PATH=os.defpath), time.monotonic() + 5, boundary)
        assert code == 0, stdout
        assert stdout == b"read:denied\nwrite:denied\nprocess-environment:denied\n"
        assert secret.read_text() == "canary outside the role"
        assert (scratch / "allowed").read_text() == "role output"
    finally:
        boundary.close()


@pytest.mark.parametrize("mode", ["deadline", "output"])
def test_owned_process_is_bounded_and_stdin_is_exact(tmp_path, mode):
    script = ("import time; time.sleep(30)" if mode == "deadline" else
              "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read() * 100)")
    with pytest.raises(dream.DreamError, match="deadline|byte allowance"):
        native.capture([sys.executable, "-S", "-c", script], b"input", tmp_path, dict(PATH=os.defpath),
                       time.monotonic() + (0.1 if mode == "deadline" else 3), limit=100)


def test_environment_copies_only_auth_and_clears_parent_session(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "auth.json").write_text('{"token":"sanitized-fixture"}')
    (source / "config.toml").write_text('dangerous = "settings"')
    monkeypatch.setenv("CODEX_HOME", str(source))
    monkeypatch.setenv("SUCODER_CODEX_REMOTE_ID", "parent-socket")
    monkeypatch.setenv("CODEX_THREAD_ID", "parent-thread")
    scratch = tmp_path / "role"
    scratch.mkdir()
    env = native.environment(scratch, "codex")
    assert "CODEX_THREAD_ID" not in env
    assert "SUCODER_CODEX_REMOTE_ID" not in env
    assert (scratch / ".codex/auth.json").read_bytes() == (source / "auth.json").read_bytes()
    assert not (scratch / ".codex/config.toml").exists()
    assert (scratch / ".codex/auth.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_native_commands_never_use_dispatch_bypass_modes(harness):
    command = native.command(harness, "native", "explicit-model", dict(type="object"))
    assert not any("bypass" in arg or "skip-permissions" in arg for arg in command)
    assert command[command.index("--model") + 1] == "explicit-model"
    if harness == "codex":
        assert "read-only" in command
        for feature in ("shell_tool", "view_image", "apps", "plugins", "hooks", "multi_agent"):
            assert command[command.index(feature) - 1] == "--disable"
        assert "--ignore-user-config" in command
    else:
        assert command[command.index("--tools") + 1] == ""
        assert "--restricted" in command
        assert "--safe-mode" in command


def jsonl(*events):
    return b"".join(json.dumps(event).encode() + b"\n" for event in events)


@pytest.mark.parametrize("effort", [None, "low", "medium", "high", "xhigh", "max", "ultra"])
def test_codex_command_changes_only_the_requested_effort(effort):
    # Literal argv from c0fa15c. In particular, the TOML quotes are part of one argument.
    before = ["codex", "exec", "--ignore-user-config", "--ignore-rules", "--strict-config", "--ephemeral",
              "--skip-git-repo-check", "--sandbox", "read-only", "--json", "--model", "fixture"]
    after = ["--output-schema", "result.schema.json", "-c", 'approval_policy="never"', "-c", 'web_search="disabled"',
             "-c", "project_doc_max_bytes=0", "-c", 'developer_instructions="Use only the inline assignment. Do not call tools."',
             "-c", "agents.enabled=false", "-c", "mcp_servers={}"]
    for feature in ("shell_tool", "view_image", "apps", "plugins", "hooks", "multi_agent", "browser_use",
                    "computer_use", "image_generation", "code_mode", "shell_snapshot"):
        after.extend(["--disable", feature])
    after.append("-")
    assert native.command("codex", "codex", "fixture", {}) == before + after
    extra = [] if effort is None else ["-c", 'model_reasoning_effort="' + effort + '"']
    assert native.command("codex", "codex", "fixture", {}, effort) == before + extra + after


@pytest.mark.parametrize("harness,effort", [("claude", "high"), ("codex", "default"), ("codex", "")])
def test_invalid_effort_is_refused(harness, effort):
    with pytest.raises(dream.DreamError, match="effort"):
        native.command(harness, "native", "fixture", {}, effort)


def test_native_invocation_binds_effort_to_command_assignment_and_trace(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from sucoder import dream_submission

    task = dict(role="reviewer", effort="max")
    monkeypatch.setattr(native.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(native.shutil, "which", lambda name: "/fixture/codex")
    monkeypatch.setattr(native, "environment", lambda *args: {})
    monkeypatch.setattr(native, "FilesystemBoundary", Mock())

    def capture(args, stdin, *rest):
        if "--help" in args:
            return 0, b"--ignore-user-config --output-schema --ephemeral --ignore-rules", b""
        if "--version" in args:
            return 0, b"fixture", b""
        assert args[args.index('model_reasoning_effort="max"') - 1] == "-c"
        assert stdin == dream.canonical(task)
        return 0, jsonl(dict(type="item.completed", item=dict(type="agent_message", text='{"checked":true}')),
                        dict(type="turn.completed", usage={})), b""

    monkeypatch.setattr(native, "capture", capture)
    result, usage, trace = native.invoke("codex", "fixture", task, {}, time.monotonic() + 5)
    assert result == {"checked": True} and trace["effort"] == "max"
    assert dream_submission.parse(task, trace) == (result, usage)
    with pytest.raises(dream.DreamError, match="reasoning effort"):
        dream_submission.parse(task, dict(trace, effort="low"))


def test_codex_trace_uses_completed_final_result_and_native_usage():
    raw = jsonl(dict(type="thread.started", thread_id="native"),
                dict(type="item.completed", item=dict(type="agent_message", text="Commentary is not the final result.")),
                dict(type="item.completed", item=dict(type="agent_message", text='{"answer":"checked"}')),
                dict(type="turn.completed", usage=dict(input_tokens=11, output_tokens=7)))
    result, usage = native.parse_trace(raw, "codex")
    assert result == dict(answer="checked")
    assert usage["native"]["input_tokens"] == 11
    assert usage["currency_usd"] is None


@pytest.mark.parametrize("item", ["command_execution", "mcp_tool_call", "web_search", "file_change", "unknown"])
def test_codex_external_tool_or_unknown_items_reject(item):
    with pytest.raises(dream.DreamError, match="external tools|unknown item"):
        native.parse_trace(jsonl(dict(type="item.completed", item=dict(type=item)), dict(type="turn.completed")), "codex")


def test_claude_structured_output_is_internal_but_external_tools_are_rejected():
    raw = jsonl(dict(type="system", subtype="init", tools=["StructuredOutput"], mcp_servers=[]),
                dict(type="assistant", message=dict(content=[dict(type="tool_use", name="StructuredOutput", input=dict(answer="checked"))])),
                dict(type="result", subtype="success", is_error=False, structured_output=dict(answer="checked"),
                     usage=dict(input_tokens=9), total_cost_usd=.01))
    assert native.parse_trace(raw, "claude")[0] == dict(answer="checked")
    with pytest.raises(dream.DreamError, match="unexpected external tools"):
        native.parse_trace(jsonl(dict(type="system", subtype="init", tools=["Read"], mcp_servers=[])), "claude")
    with pytest.raises(dream.DreamError, match="attempted a tool"):
        native.parse_trace(jsonl(dict(type="assistant", message=dict(content=[dict(type="tool_use", name="Bash")]))), "claude")


def test_claude_thinking_progress_preserves_completed_result_and_usage():
    final = dict(type="result", subtype="success", is_error=False, structured_output=dict(answer="checked"),
        usage=dict(input_tokens=3, cache_creation_input_tokens=6130, cache_read_input_tokens=0, output_tokens=516),
        total_cost_usd=.044529)
    progress = dict(type="system", subtype="thinking_tokens", estimated_tokens=50, estimated_tokens_delta=50,
        session_id="fixture-session", uuid="fixture-event")
    assert native.parse_trace(jsonl(progress, final), "claude") == native.parse_trace(jsonl(final), "claude")
    with pytest.raises(dream.DreamError, match="no completed"):
        native.parse_trace(jsonl(progress), "claude")
    with pytest.raises(dream.DreamError, match="attempted a tool"):
        native.parse_trace(jsonl(progress, dict(type="assistant", message=dict(content=[dict(type="tool_use", name="Bash")]))), "claude")


@pytest.mark.parametrize("mutation", [dict(estimated_tokens=-1), dict(estimated_tokens=True),
    dict(estimated_tokens_delta=None), dict(tool_name="Bash")])
def test_claude_thinking_progress_rejects_malformed_or_action_fields(mutation):
    event = dict(type="system", subtype="thinking_tokens", estimated_tokens=50, estimated_tokens_delta=50)
    with pytest.raises(dream.DreamError, match="thinking-progress"):
        native.parse_trace(jsonl(dict(event, **mutation)), "claude")


@pytest.mark.parametrize("subtype", ["compact_boundary", "hook_started", "permission_request", "unknown"])
def test_claude_other_system_events_still_reject(subtype):
    with pytest.raises(dream.DreamError, match="Unexpected Claude system"):
        native.parse_trace(jsonl(dict(type="system", subtype=subtype)), "claude")


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_exit_without_completed_structured_result_is_not_success(harness):
    with pytest.raises(dream.DreamError):
        native.parse_trace(jsonl(dict(type="system", subtype="init", tools=[])), harness)


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_installed_native_version_starts_inside_boundary(tmp_path, harness):
    executable = shutil.which(harness)
    if executable is None:
        pytest.skip("Native CLI not installed")
    boundary = boundary_or_skip(tmp_path)
    try:
        boundary.allow(Path(executable).resolve(), native.READ)
        code, stdout, _ = native.capture([executable, "--version"], b"", tmp_path,
            native.environment(tmp_path, harness), time.monotonic() + 10, boundary)
        assert code == 0
        assert stdout.strip()
    finally:
        boundary.close()
