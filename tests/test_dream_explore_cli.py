"""Exploratory CLI routing and detached launch without provider processes."""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import click
import pytest
import typer
from typer.testing import CliRunner
import yaml

import sucoder
from sucoder import cli, dream, dream_cli, dream_run, dream_wiki
from tests.test_dream import workspace  # noqa: F401


@pytest.fixture
def exploratory_runtime(monkeypatch):
    calls = []

    def operation(name):
        def execute(config, *args, **kwargs):
            calls.append((name, config, args, kwargs))
            return dict(schema_version=1, operation=name, state="fixture")
        return execute

    runtime = SimpleNamespace(**{name: operation(name)
        for name in ("start", "prepare", "status", "budget_status")})
    monkeypatch.setattr(sucoder, "dream_explore", runtime, raising=False)
    monkeypatch.setitem(sys.modules, "sucoder.dream_explore", runtime)
    return runtime, calls


def invoke(workspace, *arguments):
    return CliRunner().invoke(cli.app, ["--config", str(workspace[1]), "dream", *arguments], input="")


def test_bare_dream_starts_once_without_interactive_questions(workspace, exploratory_runtime, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Exploratory launch attempted a human prompt")
    monkeypatch.setattr("builtins.input", forbidden)
    for module in (click, typer):
        monkeypatch.setattr(module, "prompt", forbidden)
        monkeypatch.setattr(module, "confirm", forbidden)
    result = invoke(workspace)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["operation"] == "start"
    calls = exploratory_runtime[1]
    assert len(calls) == 1 and calls[0][0] == "start"
    assert calls[0][2] == (None,)
    assert calls[0][3] == dict(overrides={}, budget_path=None, attempt=1)


def test_bare_dream_forwards_explicit_limits_and_frozen_job_selection(workspace, exploratory_runtime):
    result = invoke(workspace, "--harness", "codex", "--model", "fixture-model", "--purpose", "Follow useful connections",
        "--job", "/private/job", "--budget", "/private/wallet", "--attempt", "2",
        "--max-total-seconds", "120", "--max-total-calls", "8", "--reserve-seconds", "10", "--reserve-calls", "2",
        "--discovery-calls", "3", "--discovery-bytes", "8192", "--max-total-tokens", "1000", "--max-cost-usd", "1.00")
    assert result.exit_code == 0, result.output
    _, _, args, kwargs = exploratory_runtime[1][0]
    assert args == (Path("/private/job"),)
    assert kwargs == dict(budget_path=Path("/private/wallet"), attempt=2, overrides=dict(
        harness="codex", model="fixture-model", purpose="Follow useful connections", max_total_seconds=120,
        max_total_calls=8, reserve_seconds=10, reserve_calls=2, max_discovery_calls=3, max_discovery_bytes=8192,
        max_total_tokens=1000, max_cost_usd="1.00"))


@pytest.mark.parametrize("arguments", [[], ["status"], ["budget"]])
def test_help_requires_no_configuration_or_exploration(monkeypatch, arguments):
    def forbidden(*args, **kwargs):
        raise AssertionError("Help attempted configuration or execution")
    monkeypatch.setattr(dream_cli, "load_config", forbidden)
    result = CliRunner().invoke(cli.app, ["--config", "/missing/config.yaml", "dream", *arguments, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


@pytest.mark.parametrize("command,operation", [("status", "status"), ("budget", "budget_status")])
@pytest.mark.parametrize("job", [None, "/private/frozen-job"])
def test_job_inspection_does_not_enter_exploration(workspace, exploratory_runtime, command, operation, job):
    arguments = [command] + ([] if job is None else ["--job", job])
    result = invoke(workspace, *arguments)
    assert result.exit_code == 0, result.output
    calls = exploratory_runtime[1]
    assert len(calls) == 1 and calls[0][0] == operation
    assert calls[0][2] == (None if job is None else Path(job),)
    assert calls[0][3] == {}


def test_explicit_native_run_preserves_old_dispatch_and_never_starts_exploration(workspace, exploratory_runtime, monkeypatch):
    calls = []
    def explicit(config, *args, **kwargs):
        calls.append((args, kwargs))
        return dict(state="candidate_ready")
    monkeypatch.setattr(dream_wiki, "run", explicit)
    result = invoke(workspace, "run", "sample", "--plan", "/private/plan.json", "--harness", "codex", "--model", "model")
    assert result.exit_code == 0, result.output
    assert exploratory_runtime[1] == []
    assert calls[0][0][:5] == ("sample", Path("/private/plan.json"), "codex", "model", 1)
    assert calls[0][1]["budget_path"] is None


@pytest.mark.parametrize("command", ["status", "budget"])
def test_explicit_inspection_preserves_mirror_plan_dispatch(workspace, exploratory_runtime, monkeypatch, command):
    calls = []
    def inspect(config, *args):
        calls.append(args)
        return dict(state="candidate_ready")
    monkeypatch.setattr(dream_run if command == "status" else dream_cli.dream_budget,
        "status" if command == "status" else "inspect", inspect)
    result = invoke(workspace, command, "sample", "--plan", "/private/plan.json")
    assert result.exit_code == 0, result.output
    assert calls == [("sample", Path("/private/plan.json"), None)]
    assert exploratory_runtime[1] == []


@pytest.mark.parametrize("arguments", [
    ["--model", "ignored", "status"], ["--attempt", "1", "budget"],
    ["status", "sample"], ["status", "--plan", "/private/plan.json"],
    ["status", "--run", "unknown"], ["status", "sample", "--plan", "/private/plan.json", "--job", "/private/job"],
    ["budget", "sample"], ["budget", "--budget", "/private/wallet"],
    ["budget", "sample", "--plan", "/private/plan.json", "--job", "/private/job"],
    ["--max-total-calls", "0"], ["--discovery-bytes", "0"],
])
def test_incomplete_or_ignored_option_combinations_are_rejected(workspace, exploratory_runtime, arguments):
    result = invoke(workspace, *arguments)
    assert result.exit_code == 2, result.output
    assert exploratory_runtime[1] == []


def test_launch_errors_keep_structured_failure_output(workspace, exploratory_runtime):
    def failed(*args, **kwargs):
        raise dream.DreamError("Missing configured model; no provider work", 3)
    exploratory_runtime[0].start = failed
    result = invoke(workspace)
    assert result.exit_code == 3
    assert json.loads(result.output) == dict(schema_version=1, operation="dream_job", state="rejected",
        error="Missing configured model; no provider work")


def test_detach_prepares_then_spawns_same_job_with_private_logs(workspace, exploratory_runtime, monkeypatch):
    job = workspace[1].parent / "private-job"
    job.mkdir(mode=0o700)
    prepare_calls, process_calls = [], []
    def prepare(config, *args, **kwargs):
        prepare_calls.append((args, kwargs))
        return dict(job_path=str(job), job_id="fixture-job")
    def popen(argv, **kwargs):
        process_calls.append((argv, dict(kwargs)))
        assert kwargs["stdin"] == subprocess.DEVNULL and kwargs["start_new_session"] is True
        assert kwargs["close_fds"] is True and not kwargs.get("shell", False)
        for name in ("stdout", "stderr"):
            info = os.fstat(kwargs[name].fileno())
            assert stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600
            kwargs[name].write(b"detached fixture\n")
        return SimpleNamespace(pid=4321)
    exploratory_runtime[0].prepare = prepare
    monkeypatch.setattr(dream_cli.subprocess, "Popen", popen)
    result = invoke(workspace, "--detach", "--model", "fixture-model", "--attempt", "2")
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["state"] == "detached" and value["pid"] == 4321 and value["job_path"] == str(job)
    assert prepare_calls == [((None,), dict(overrides=dict(model="fixture-model"), budget_path=None, attempt=2))]
    assert exploratory_runtime[1] == []
    assert process_calls[0][0] == [sys.executable, "-c", "from sucoder.cli import app; app()",
        "--config", str(workspace[1].absolute()), "dream", "--job", str(job), "--attempt", "2"]
    assert (job / "worker.stdout.log").read_bytes() == (job / "worker.stderr.log").read_bytes() == b"detached fixture\n"


@pytest.mark.parametrize("defect", ["public-directory", "symlink-log", "public-log"])
def test_detach_rejects_unsafe_log_destinations_before_starting_worker(workspace, exploratory_runtime, monkeypatch, defect):
    job = workspace[1].parent / "job"
    job.mkdir(mode=0o755 if defect == "public-directory" else 0o700)
    if defect == "symlink-log":
        (job / "worker.stdout.log").symlink_to(workspace[1])
    elif defect == "public-log":
        (job / "worker.stdout.log").write_text("preserve")
    exploratory_runtime[0].prepare = lambda *args, **kwargs: dict(job_path=str(job))
    def forbidden(*args, **kwargs):
        raise AssertionError("Unsafe log destination started a worker")
    monkeypatch.setattr(dream_cli.subprocess, "Popen", forbidden)
    result = invoke(workspace, "--detach")
    assert result.exit_code == 3, result.output
    assert json.loads(result.output)["state"] == "rejected"


def test_bare_cli_executes_exploration_and_inspects_current_project_without_more_calls(workspace, monkeypatch):
    from sucoder import dream_native
    from tests.test_dream_explore import ExploringTransport
    subdirectory = workspace[0] / "nested"
    subdirectory.mkdir()
    monkeypatch.chdir(subdirectory)
    native = ExploringTransport(no_selection=True)
    monkeypatch.setattr(dream_native, "Transport", lambda *args, **kwargs: native)
    result = invoke(workspace, "--harness", "codex", "--model", "fixture-model")
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["state"] == "no_selection" and value["project"]["path"] == str(workspace[0])
    assert len(native.calls) == 1 and native.calls[0]["role"] == "explorer"
    inspected = invoke(workspace, "status")
    assert inspected.exit_code == 0, inspected.output
    assert json.loads(inspected.output)["state"] == "no_selection"
    balance = invoke(workspace, "budget")
    assert balance.exit_code == 0, balance.output
    assert json.loads(balance.output)["reserved_calls"] == 1 and len(native.calls) == 1


def test_detached_child_from_subdirectory_uses_frozen_job_and_finite_stdin(workspace, monkeypatch):
    # This fixture executable exercises the real detached Python child, native
    # launcher, Landlock confinement, staged dummy auth and persistent wallet.
    scratch = workspace[1].parent
    executable = scratch / "codex"
    executable.write_text('''#!/usr/bin/python3 -S
import json,os,sys
from pathlib import Path
if '--version' in sys.argv:
    print('detached-native-fixture 1.0'); raise SystemExit
if '--help' in sys.argv:
    print('--ignore-user-config --output-schema --ephemeral --ignore-rules')
    raise SystemExit
assert sys.argv[sys.argv.index('--model')+1]=='fixture-model'
assert json.loads((Path(os.environ['CODEX_HOME'])/'auth.json').read_text())['token']=='dummy-detached-fixture'
payload=sys.stdin.buffer.read()
assert payload and sys.stdin.buffer.read()==b''
assert not any(os.isatty(fd) for fd in (0,1,2))
task=json.loads(payload)
assert task['role']=='explorer'
result={key:task[key] for key in ('assignment_id','input_sha256','role')}
result.update(failures=[],action='finish',path='',query='',offset=0,start=0,end=0,
    selections=[],topic='',rationale='Fixture consumed finite non-TTY stdin and selected no evidence.',
    questions=['Which connection deserves attention in a later session?'])
print(json.dumps(dict(type='item.completed',item=dict(type='agent_message',text=json.dumps(result)))))
print(json.dumps(dict(type='turn.completed',usage={})))
''')
    executable.chmod(0o700)
    auth = scratch / "dummy-native-home"
    auth.mkdir(mode=0o700)
    (auth / "auth.json").write_text('{"token":"dummy-detached-fixture"}')
    monkeypatch.setenv("CODEX_HOME", str(auth))
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-detached-fixture")
    monkeypatch.setenv("PATH", str(scratch) + os.pathsep + os.environ["PATH"])
    subdirectory = workspace[0] / "nested"
    subdirectory.mkdir()
    monkeypatch.chdir(subdirectory)

    real_popen = subprocess.Popen
    workers = []
    def start_worker(argv, *args, **kwargs):
        is_worker = argv[:3] == [sys.executable, "-c", "from sucoder.cli import app; app()"]
        if is_worker:
            # A worker must use the model frozen by prepare, even when the
            # caller's defaults change before the child loads configuration.
            config = yaml.safe_load(workspace[1].read_text())
            config["dream"] = dict(harness="codex", model="changed-default-model")
            workspace[1].write_text(yaml.safe_dump(config))
        process = real_popen(argv, *args, **kwargs)
        if is_worker:
            workers.append(process)
        return process
    monkeypatch.setattr(dream_cli.subprocess, "Popen", start_worker)
    launched_at = time.monotonic()
    result = invoke(workspace, "--detach", "--harness", "codex", "--model", "fixture-model",
        "--max-total-seconds", "20", "--max-total-calls", "6", "--reserve-seconds", "2", "--discovery-calls", "2")
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["state"] == "detached" and len(workers) == 1
    job = Path(value["job_path"])
    try:
        exit_code = workers[0].wait(timeout=25)
    except subprocess.TimeoutExpired:
        workers[0].terminate()
        workers[0].wait(timeout=3)
        raise
    assert time.monotonic() - launched_at < 30
    stderr = (job / "worker.stderr.log").read_text()
    assert exit_code == 0, stderr + (job / "worker.stdout.log").read_text()
    completed = json.loads((job / "result-000001.json").read_text())
    assert completed["state"] == "no_selection"
    assert "finite non-TTY stdin" in completed["rationale"]
    assert completed["questions"] == ["Which connection deserves attention in a later session?"]
    header = json.loads((job / "job.json").read_text())
    assert header["project"]["path"] == str(workspace[0])
    assert header["settings"]["model"] == "fixture-model" and header["settings"]["max_total_seconds"] == 20
    assert len(list((job / "budget").glob("reservation-*.json"))) == 1
    returned = json.loads((job / "budget" / "returned-000000.json").read_text())
    assert returned["state"] == "completed"
    for name in ("worker.stdout.log", "worker.stderr.log"):
        assert stat.S_IMODE((job / name).stat().st_mode) == 0o600
    assert json.loads((job / "worker.stdout.log").read_text())["state"] == "no_selection"
