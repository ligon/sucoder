"""Native instructions and the actual Codex client-override regression (ledger 4).

The protocol test uses only a loopback fake model endpoint and an isolated home.
It skips when Codex is absent; no remote-control pairing or paid request occurs.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from sucoder.codex_context import CONTEXT_BEGIN, CONTEXT_END, install_context
from tests.test_remote_control import PRELUDE, _manager


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "project"
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


def test_native_context_preserves_project_and_refreshes_only_generated_regions(workspace):
    project = workspace / "AGENTS.md"
    project.write_text("PROJECT_FIRST\n", encoding="utf-8")
    path = install_context(workspace, PRELUDE)
    assert PRELUDE in path.read_text(encoding="utf-8")
    assert "PROJECT_FIRST" in path.read_text(encoding="utf-8")
    assert project.read_text() == "PROJECT_FIRST\n"
    with path.open("a") as stream:
        stream.write("USER_ADDITION\n")
    project.write_text("PROJECT_UPDATED\n")
    install_context(workspace, "NEW_PRELUDE")
    content = path.read_text()
    assert "NEW_PRELUDE" in content and PRELUDE not in content
    assert "PROJECT_UPDATED" in content and "PROJECT_FIRST" not in content
    assert content.endswith("USER_ADDITION\n")
    assert content.count(CONTEXT_BEGIN) == 1
    install_context(workspace, "NEW_PRELUDE")
    assert path.read_text() == content
    assert (workspace / ".git/info/exclude").read_text().count("/AGENTS.override.md") == 1
    ignored = subprocess.run(["git", "check-ignore", "AGENTS.override.md"],
                             cwd=workspace, capture_output=True)
    assert ignored.returncode == 0


def test_existing_local_override_keeps_its_precedence_and_exact_text(workspace):
    (workspace / "AGENTS.md").write_text("SHADOWED_PROJECT\n")
    override = workspace / "AGENTS.override.md"
    original = b"CUSTOM_OVERRIDE\r\nkeep spacing  \r\n"
    override.write_bytes(original)
    install_context(workspace, PRELUDE)
    assert override.read_bytes().endswith(original)
    assert "SHADOWED_PROJECT" not in override.read_text()
    install_context(workspace, "UPDATED")
    assert override.read_bytes().endswith(original)


@pytest.mark.parametrize("text", [CONTEXT_BEGIN + "unfinished", CONTEXT_END + CONTEXT_BEGIN,
                                 CONTEXT_BEGIN + CONTEXT_END + CONTEXT_END])
def test_malformed_override_is_preserved_and_rejected(workspace, text):
    path = workspace / "AGENTS.override.md"
    path.write_text(text)
    with pytest.raises(ValueError):
        install_context(workspace, PRELUDE)
    assert path.read_text() == text


def test_symlinked_override_is_not_replaced(workspace, tmp_path):
    source = tmp_path / "shared-instructions"
    source.write_text("SHARED\n")
    (workspace / "AGENTS.override.md").symlink_to(source)
    with pytest.raises(ValueError, match="symlinked"):
        install_context(workspace, PRELUDE)
    assert source.read_text() == "SHARED\n"
    assert (workspace / "AGENTS.override.md").is_symlink()


@pytest.mark.parametrize("edited", ["SUCODER_ORIGINAL", "PROJECT_ORIGINAL"])
def test_edits_within_generated_regions_are_preserved(workspace, edited):
    (workspace / "AGENTS.md").write_text("PROJECT_ORIGINAL\n")
    path = install_context(workspace, "SUCODER_ORIGINAL")
    changed = path.read_text().replace(edited, "USER_EDIT")
    path.write_text(changed)
    with pytest.raises(ValueError, match="was edited"):
        install_context(workspace, "NEW_PRELUDE")
    assert path.read_text() == changed


def test_concurrent_edit_is_preserved(workspace, monkeypatch):
    path = install_context(workspace, "ORIGINAL")
    real_fsync = os.fsync

    def edit_during_write(fd):
        real_fsync(fd)
        path.write_text("CONCURRENT_EDIT\n")

    monkeypatch.setattr(os, "fsync", edit_during_write)
    with pytest.raises(ValueError, match="changed during installation"):
        install_context(workspace, "UPDATED")
    assert path.read_text() == "CONCURRENT_EDIT\n"


def test_tracked_override_is_not_modified(workspace):
    path = workspace / "AGENTS.override.md"
    path.write_text("TRACKED_INSTRUCTIONS\n")
    subprocess.run(["git", "add", "AGENTS.override.md"], cwd=workspace, check=True)
    with pytest.raises(ValueError, match="tracked by Git"):
        install_context(workspace, PRELUDE)
    assert path.read_text() == "TRACKED_INSTRUCTIONS\n"


def test_linked_worktree_gets_its_own_context(workspace, tmp_path):
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "-q", "--allow-empty", "-m", "fixture"],
                   cwd=workspace, check=True)
    linked = tmp_path / "linked"
    subprocess.run(["git", "worktree", "add", "-q", "--detach", str(linked)],
                   cwd=workspace, check=True)
    path = install_context(linked, PRELUDE)
    assert path.parent == linked
    assert not (workspace / "AGENTS.override.md").exists()
    ignored = subprocess.run(["git", "check-ignore", "AGENTS.override.md"],
                             cwd=linked, capture_output=True)
    assert ignored.returncode == 0


def test_staged_installer_runs_in_final_clone_and_keeps_launches_separate(
    tmp_path, monkeypatch, workspace,
):
    manager, ctx = _manager(tmp_path, monkeypatch)
    manager._service_launch = {"command": ["codex", "remote-control"], "model": None}

    def execute(args, **kwargs):
        return subprocess.run(args, input=kwargs.get("input"), check=True,
                              capture_output=True, text=True)

    manager.executor.run_agent = execute
    command = manager._wrap_codex_service_context(ctx, ["true"], PRELUDE)
    other = manager._wrap_codex_service_context(ctx, ["true"], "OTHER_LAUNCH")
    assert command[1] != other[1]
    subprocess.run(command, cwd=workspace, check=True, capture_output=True)
    assert PRELUDE in (workspace / "AGENTS.override.md").read_text(encoding="utf-8")
    assert not (tmp_path / "AGENTS.override.md").exists()
    assert not (workspace / "INJECTED").exists()


@pytest.mark.skipif(not shutil.which("codex"), reason="Codex CLI not installed")
def test_codex_model_receives_prelude_when_client_replaces_developer_instructions(
    workspace, tmp_path,
):
    """Exercise thread/start through the installed app-server, not mocked argv."""
    requests = queue.Queue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.put(json.loads(body))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            event = {"type": "response.completed", "response": {
                "id": "resp_test", "object": "response", "status": "completed", "output": [],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }}
            self.wfile.write(("event: response.completed\ndata: "
                              + json.dumps(event) + "\n\n").encode())

    # Exceed Codex's usual 32 KiB document budget and check the last bytes too.
    prelude = PRELUDE + "workspace rule\n" * 3000 + "END_OF_SUCODER_PRELUDE"
    (workspace / "AGENTS.md").write_text("PROJECT_INSTRUCTIONS\n")
    install_context(workspace, prelude)
    home = tmp_path / "codex-home"
    home.mkdir()
    (home / "AGENTS.md").write_text("GLOBAL_INSTRUCTIONS\n")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    options = [
        "-c", 'developer_instructions="SERVER_VALUE_REPLACED_BY_CLIENT"',
        "-c", "project_doc_max_bytes=1048576",
        "-c", 'model_provider="offline"', "-c", 'model="fake"',
        "-c", "features.enable_request_compression=false",
        "-c", "features.apps=false",
        "-c", ('model_providers.offline={name="offline",'
               f'base_url="http://127.0.0.1:{server.server_port}",wire_api="responses"}}'),
    ]
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "CODEX_HOME": str(home),
           "NO_PROXY": "127.0.0.1", "RUST_LOG": "error"}
    process = None
    with (tmp_path / "codex-stderr.log").open("w") as log:
        try:
            process = subprocess.Popen(
                [shutil.which("codex"), *options, "app-server"], cwd=workspace, env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
            )
            messages = queue.Queue()

            def read_messages():
                for line in process.stdout:
                    messages.put(json.loads(line))

            reader = threading.Thread(target=read_messages, daemon=True)
            reader.start()

            def rpc(method, params, request_id):
                process.stdin.write(json.dumps({"id": request_id, "method": method,
                                                "params": params}) + "\n")
                process.stdin.flush()
                while True:
                    message = messages.get(timeout=15)
                    if message.get("id") == request_id:
                        assert "error" not in message, message
                        return message["result"]

            rpc("initialize", {"clientInfo": {"name": "sucoder_test", "version": "1"},
                               "capabilities": {"experimentalApi": True}}, 1)
            process.stdin.write('{"method":"initialized","params":{}}\n')
            process.stdin.flush()
            thread = rpc("thread/start", {"cwd": str(workspace), "ephemeral": True,
                                          "developerInstructions": "CLIENT_INSTRUCTIONS"}, 2)
            rpc("turn/start", {"threadId": thread["thread"]["id"],
                               "input": [{"type": "text", "text": "probe"}]}, 3)
            request = requests.get(timeout=15)
            texts = [part.get("text", "") for item in request["input"]
                     for part in item.get("content", [])]
            model_input = "\n".join(texts)
            assert prelude in model_input
            assert "CLIENT_INSTRUCTIONS" in model_input
            assert "PROJECT_INSTRUCTIONS" in model_input
            assert "GLOBAL_INSTRUCTIONS" in model_input
            assert "SERVER_VALUE_REPLACED_BY_CLIENT" not in model_input
            assert request["instructions"]  # Preserve Codex's native base instructions.
        finally:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                process.stdin.close()
                process.stdout.close()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)
