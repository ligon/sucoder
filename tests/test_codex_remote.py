import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from sucoder import cli
from sucoder.codex_remote import build_remote_command
from sucoder.config import AgentLauncher, ConfigError, _parse_agent_launcher
from sucoder.executor import CommandResult
from sucoder.mirror import MirrorError
from tests.test_cli import _write_config
from tests.test_mirror import build_manager
from tests.test_mirror import _confined_manager, _confined_responder


@pytest.mark.parametrize("mode", [None, "off", "terminal", "server"])
def test_remote_config_modes(mode):
    assert _parse_agent_launcher({"codex_remote_control": mode}).codex_remote_control == mode


@pytest.mark.parametrize("mode", [True, False, 1, [], "yes", "daemon"])
def test_remote_config_invalid(mode):
    with pytest.raises(ConfigError, match="codex_remote_control"):
        _parse_agent_launcher({"codex_remote_control": mode})


@pytest.mark.parametrize("subcommand", ["collaborate", "agents-run"])
@pytest.mark.parametrize("flags, mode", [([], None), (["--remote-control"], "terminal"),
                                        (["--remote-control-only"], "server"),
                                        (["--no-remote-control"], "off")])
def test_cli_forwards_remote_mode(tmp_path, monkeypatch, subcommand, flags, mode):
    path = _write_config(tmp_path, skills_entry=tmp_path / "skills")
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **k: None)
    calls = []
    manager = SimpleNamespace(
        context_for=lambda name: name,
        bootstrap=lambda *a, **kw: calls.append(kw),
        launch_agent=lambda *a, **kw: calls.append(kw),
    )
    monkeypatch.setattr(cli, "_build_manager_for_mirror", lambda *a, **kw: manager)
    result = CliRunner().invoke(cli.app, ["--config", str(path), subcommand,
                                         "sample", "--agent", "codex", *flags])
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls[0]["codex_remote_control"] == mode
    assert calls[0]["command_override"] == ["codex"]


def test_cli_rejects_conflicting_remote_flags():
    with pytest.raises(Exception, match="conflicts"):
        cli._codex_remote_mode(False, True)


@pytest.mark.parametrize("configured, override, expected", [
    ("terminal", None, "terminal"), ("server", "off", None),
    ("off", "server", "server"),
])
def test_launch_config_precedence_and_context(tmp_path, monkeypatch, configured, override, expected):
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)
    ctx.settings.agent_launcher = AgentLauncher(command=["codex"])
    manager.config.agent_launcher = AgentLauncher(codex_remote_control=configured)
    prelude = 'SuCoder instructions\nwith "quotes", $HOME and `literal backticks`.'
    monkeypatch.setattr(manager, "_compose_context_prelude", lambda ctx: prelude)
    calls = []

    def capture(args, **kwargs):
        calls.append(list(args))
        return CommandResult(list(args), list(args), "", "", 0)

    monkeypatch.setattr(manager.executor, "run_agent", capture)
    manager.launch_agent(ctx, sync=False, codex_remote_control=override, model_override="a-model")
    command = calls[-1]
    if expected:
        assert "--temporary-local-root" in command and "--remote-helper" in command
        at = command.index("sucoder-codex-remote")
        assert command[at + 1] == expected
        assert 'model="a-model"' in command
        assert "developer_instructions=" + json.dumps(prelude) in command
        assert prelude not in command  # instructions belong to every server-created thread
    else:
        assert command[0] == "codex"
        assert prelude in command


def test_mirror_off_overrides_global_and_other_harnesses_are_unchanged(tmp_path, monkeypatch):
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)
    manager.config.agent_launcher = AgentLauncher(codex_remote_control="terminal")
    calls = []
    monkeypatch.setattr(manager.executor, "run_agent", lambda args, **kw:
                        calls.append(list(args)) or CommandResult(list(args), list(args), "", "", 0))
    for harness in ("codex", "claude"):
        ctx.settings.agent_launcher = AgentLauncher(command=[harness], codex_remote_control="off")
        manager.launch_agent(ctx, sync=False)
        assert calls[-1][0] == harness
    with pytest.raises(MirrorError, match="requires"):
        manager.launch_agent(ctx, sync=False, codex_remote_control="terminal")


def test_unknown_remote_args_fail_instead_of_dropping_settings():
    for args in (["--remote", "unix://other"], ["resume"], ["--model"], ["--profile", "custom"]):
        with pytest.raises(ValueError):
            build_remote_command(["codex", *args], "terminal")


@pytest.mark.parametrize("mode", ["terminal", "server"])
def test_confined_remote_launch_stages_toml_context_inside_batch(tmp_path, monkeypatch, mode):
    manager, ctx = _confined_manager(tmp_path, monkeypatch)
    ctx.settings.agent_launcher = AgentLauncher(command=["codex"], codex_remote_control=mode)
    manager.executor.local_disk_root = "/local"
    manager.executor.slurm_job_id = None
    prelude = 'Instructions with "quotes" and\na newline and $(not-a-command).'
    monkeypatch.setattr(manager, "_compose_context_prelude", lambda ctx: prelude)
    calls = []
    manager.executor.run_agent = _confined_responder(calls, sbatch_out="9")
    manager.launch_agent(ctx, sync=False, detached=True)
    staged = next(c["input"] for c in calls if c["args"][0] == "sh" and "prelude-" in c["args"][2])
    assert staged == "developer_instructions=" + json.dumps(prelude)
    batch = next(c["input"] for c in calls if c["args"][0] == "sh" and "job-sample.sh" in c["args"][2])
    assert "sucoder-codex-remote" in batch
    assert "--remote-control" in batch
    assert "__SUCODER_PRELUDE_FROM_FILE__" not in batch
    assert "$(not-a-command)" not in batch
    assert "prelude-sample.txt" in batch
    assert "tmux -L" in batch


@pytest.fixture
def fake_codex(tmp_path):
    binary = tmp_path / "fake codex"
    binary.write_text("#!" + sys.executable + "\n" + r'''
import json, os, signal, socket, sys, time
from pathlib import Path
args = sys.argv[1:]
root = Path(os.environ['FAKE_RECORD'])
if '--help' in args:
    sys.exit(int(os.environ.get('FAKE_PROBE_EXIT', '0')))
role = 'server' if args[0] == 'app-server' else 'client'
url = args[args.index('--listen' if role == 'server' else '--remote') + 1]
path = Path(url.removeprefix('unix://'))
(root / (role + '.json')).write_text(json.dumps({
    'pid': os.getpid(), 'url': url, 'args': args,
    'mode': path.parent.stat().st_mode & 0o777,
    'cwd': os.getcwd(), 'env': os.environ.get('FAKE_SHARED_ENV'),
}))
def stop(*_):
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
try:
    if role == 'server':
        if os.environ.get('FAKE_SERVER_FAIL'):
            print('server startup failed', file=sys.stderr, flush=True)
            sys.exit(17)
        with socket.socket(socket.AF_UNIX) as sock:
            sock.bind(str(path))
            sock.listen()
            if os.environ.get('FAKE_SERVER_IGNORE_TERM'):
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
            if os.environ.get('FAKE_SERVER_CRASH'):
                time.sleep(0.2)
                sys.exit(17)
            while True:
                time.sleep(0.1)
    else:
        assert path.is_socket()
        if os.environ.get('FAKE_CLIENT_WAIT'):
            while True:
                time.sleep(0.1)
        time.sleep(float(os.environ.get('FAKE_CLIENT_DELAY', '0')))
        sys.exit(int(os.environ.get('FAKE_CLIENT_EXIT', '0')))
finally:
    (root / (role + '.stopped')).touch()
''')
    binary.chmod(0o700)
    return binary


def test_terminal_uses_same_server_and_cleans_up(fake_codex, tmp_path):
    command = build_remote_command([str(fake_codex), "--model", "test-model"], "terminal",
                                   instructions='developer_instructions="context"')
    result = subprocess.run(command, env=dict(os.environ, FAKE_RECORD=str(tmp_path),
                            FAKE_SHARED_ENV="inherited", FAKE_CLIENT_EXIT="7"),
                            cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert result.returncode == 7, result.stderr
    server = json.loads((tmp_path / "server.json").read_text())
    client = json.loads((tmp_path / "client.json").read_text())
    assert server["url"] == client["url"]
    assert server["mode"] == client["mode"] == 0o700
    assert server["cwd"] == client["cwd"] == str(tmp_path)
    assert server["env"] == client["env"] == "inherited"
    assert 'model="test-model"' in server["args"]
    assert 'developer_instructions="context"' in server["args"]
    assert "--model" in client["args"]
    assert (tmp_path / "server.stopped").exists()
    assert not Path(server["url"][7:]).parent.exists()


@pytest.mark.parametrize("mode", ["terminal", "server"])
def test_signal_stops_only_owned_processes(fake_codex, tmp_path, mode):
    command = build_remote_command([str(fake_codex)], mode)
    process = subprocess.Popen(command, env=dict(os.environ, FAKE_RECORD=str(tmp_path),
                               FAKE_CLIENT_WAIT="1"), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    try:
        ready = tmp_path / ("client.json" if mode == "terminal" else "server.json")
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists()
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
        assert process.returncode == (0 if mode == "server" else 143)
        assert (tmp_path / "server.stopped").exists()
        assert (tmp_path / "client.json").exists() == (mode == "terminal")
        if mode == "terminal":
            assert (tmp_path / "client.stopped").exists()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()


def test_startup_failure_never_launches_client(fake_codex, tmp_path):
    result = subprocess.run(build_remote_command([str(fake_codex)], "terminal"),
                            env=dict(os.environ, FAKE_RECORD=str(tmp_path), FAKE_SERVER_FAIL="1"),
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 17
    assert "server startup failed" in result.stderr
    assert not (tmp_path / "client.json").exists()


def test_unsupported_codex_fails_before_starting_any_server(fake_codex, tmp_path):
    result = subprocess.run(build_remote_command([str(fake_codex)], "terminal"),
                            env=dict(os.environ, FAKE_RECORD=str(tmp_path), FAKE_PROBE_EXIT="2"),
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert "does not support" in result.stderr
    assert not (tmp_path / "server.json").exists()


@pytest.mark.parametrize("settings, expected", [
    ({"FAKE_SERVER_CRASH": "1", "FAKE_CLIENT_DELAY": "0.5"}, 17),
    ({"FAKE_SERVER_IGNORE_TERM": "1", "FAKE_CLIENT_DELAY": "0.2"}, 137),
])
def test_clean_client_cannot_hide_failed_server(fake_codex, tmp_path, settings, expected):
    result = subprocess.run(build_remote_command([str(fake_codex)], "terminal"),
                            env=dict(os.environ, FAKE_RECORD=str(tmp_path), **settings),
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == expected, result.stderr
