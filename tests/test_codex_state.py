"""Managed Codex state, Slice 1: placement and process ownership.

Launcher side: the scope digest, the ``sqlite_home`` placeholder, the
staged runner outermost, and the typed ``slurm.codex_state`` configuration.
Node side: the runner against a fake ``codex`` (placement verified through
``/proc``, refusals, signal forwarding, exit codes) and, when the real
``codex`` and ``bwrap`` are installed, an actual app-server whose databases
must land under the managed ``db`` directory.  No real Codex home is used.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from sucoder import cli, codex_state
from sucoder.codex_state import (
    PLACEHOLDER, check_durable_root, durable_root_for, runner_command, runner_source, scope_id,
    validate_restore_selector, with_sqlite_home_placeholder,
)
from sucoder.config import AgentLauncher, CodexStateConfig, ConfigError, _parse_slurm_config
from sucoder.mirror import MirrorError
from tests.test_cli import _write_config
from tests.test_remote_control import _manager

SELECTOR = "0123456789abcdef0123456789abcdef/000412-0a1b2c3d"

RUNNER = Path(codex_state.__file__).with_name("codex_state_runner.py")
FAKE_CODEX = r'''#!/usr/bin/env python3
import os, re, signal, sqlite3, sys, time
home = os.environ["CODEX_HOME"]
sqlite_home = None
for arg in sys.argv[1:]:
    m = re.match(r'sqlite_home="([^"]+)"', arg)
    if m:
        sqlite_home = m.group(1)
if os.environ.get("FAKE_EXIT_NOW"):
    sys.exit(5)
target = home if os.environ.get("FAKE_IGNORE_SQLITE_HOME") else sqlite_home
conn = sqlite3.connect(os.path.join(target, "state_5.sqlite"))
conn.execute("create table if not exists t(x)")
conn.commit()
os.makedirs(os.path.join(home, "sessions", "2026"), exist_ok=True)
rollout = open(os.path.join(home, "sessions", "2026", "rollout-2026-fake.jsonl"), "a")
rollout.write('{"fake": 1}\n')
rollout.flush()
code = {"value": None}
signal.signal(signal.SIGTERM, lambda s, f: code.update(value=37))
sys.stdout.write("FAKE_READY\n")
sys.stdout.flush()
while code["value"] is None:
    time.sleep(0.05)
sys.exit(code["value"])
'''
needs_bwrap = pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap not installed")


# ----------------------------------------------------------------- helpers

def test_scope_id_is_a_stable_digest_of_the_unsanitized_tuple():
    a = scope_id("hpc.example", "ligon", "/global/scratch/u/mirrors/SuCoder")
    assert re.fullmatch(r"[0-9a-f]{64}", a)
    assert a == scope_id("hpc.example", "ligon", "/global/scratch/u/mirrors/SuCoder")
    assert a != scope_id("hpc.example", "other", "/global/scratch/u/mirrors/SuCoder")
    assert a != scope_id("hpc.example", "ligon", "/global/scratch/u/mirrors/SuCoder", slot="two")
    # A display token is not an identity: paths that sanitize alike stay distinct.
    assert scope_id("c", "u", "/m/a b") != scope_id("c", "u", "/m/a_b")


def test_placeholder_goes_right_after_the_executable_and_explicit_sqlite_home_is_refused():
    assert with_sqlite_home_placeholder(["codex", "remote-control"]) == [
        "codex", "-c", f'sqlite_home="{PLACEHOLDER}"', "remote-control"]
    assert with_sqlite_home_placeholder(["/opt/codex", "-c", 'model="x"', "prompt"])[:3] == [
        "/opt/codex", "-c", f'sqlite_home="{PLACEHOLDER}"']
    with pytest.raises(ValueError, match="already sets sqlite_home"):
        with_sqlite_home_placeholder(["codex", "-c", 'sqlite_home="/elsewhere"'])
    with pytest.raises(ValueError):
        with_sqlite_home_placeholder([])


def test_placeholder_survives_shell_quoting_and_substitution_keeps_the_argv():
    command = with_sqlite_home_placeholder(["codex", "-c", 'developer_instructions="a b"', "prompt with spaces"])
    wrapped = ["bash", "-lc", f"export NVM_DIR=/x; exec {shlex.join(command)}"]
    assert PLACEHOLDER in wrapped[2]
    substituted = [arg.replace(PLACEHOLDER, "/local/job7/state/codex/ab/cd/db") for arg in wrapped]
    inner = shlex.split(substituted[2].split("exec ", 1)[1])
    assert inner[2] == 'sqlite_home="/local/job7/state/codex/ab/cd/db"'
    assert inner[4] == 'developer_instructions="a b"' and inner[5] == "prompt with spaces"


def test_runner_command_is_outermost_and_passes_the_command_after_a_separator():
    kwargs = dict(durable_root="/g/mirrors/.sucoder-state", checkpoint_minutes=10, checkpoint_timeout_seconds=30)
    prefix = ["python3", "/h/.cache/sucoder/r.py", "--scope", "ab" * 32, "--placeholder", PLACEHOLDER,
              "--durable-root", "/g/mirrors/.sucoder-state", "--checkpoint-minutes", "10", "--checkpoint-timeout", "30"]
    assert runner_command("/h/.cache/sucoder/r.py", "ab" * 32, ["python3", "/i.py", "codex"], **kwargs) == [
        *prefix, "--", "python3", "/i.py", "codex"]
    assert runner_command("/h/.cache/sucoder/r.py", "ab" * 32, ["codex"], restore=SELECTOR, **kwargs) == [
        *prefix, "--restore", SELECTOR, "--", "codex"]
    assert runner_command("/h/.cache/sucoder/r.py", "ab" * 32, ["codex"], fresh=True, **kwargs) == [
        *prefix, "--fresh", "--", "codex"]
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner_command("/h/.cache/sucoder/r.py", "ab" * 32, ["codex"], restore=SELECTOR, fresh=True, **kwargs)
    with pytest.raises(ValueError, match="expected RUN/GENERATION"):
        runner_command("/h/.cache/sucoder/r.py", "ab" * 32, ["codex"], restore="latest", **kwargs)


def test_durable_root_is_configured_or_the_sibling_of_the_mirrors():
    assert durable_root_for("/global/home/users/ligon/mirrors/SuCoder", None) == "/global/home/users/ligon/mirrors/.sucoder-state"
    assert durable_root_for("/global/home/users/ligon/mirrors/SuCoder/", None) == "/global/home/users/ligon/mirrors/.sucoder-state"
    assert durable_root_for("/global/home/users/ligon/mirrors/SuCoder", "/global/scratch/ligon/state/") == "/global/scratch/ligon/state"


@pytest.mark.parametrize("root, message", [
    ("/g/mirrors/sample", "is the shared mirror repository itself"),
    ("/g/mirrors/sample/", "is the shared mirror repository itself"),
    ("/g/mirrors/sample/.sucoder-state", "is inside the shared mirror repository"),
    ("/g/mirrors", "contains the shared mirror repository"),
    ("/g", "contains the shared mirror repository"),
    ("/local/state", "allocation-local storage is not durable"),
    ("/local", "allocation-local storage is not durable"),
])
def test_durable_root_rejections(root, message):
    with pytest.raises(ValueError, match=message):
        check_durable_root(root, "/g/mirrors/sample", "/local")


def test_durable_root_accepts_siblings_and_prefix_lookalikes():
    assert check_durable_root("/g/mirrors/.sucoder-state", "/g/mirrors/sample", "/local") == "/g/mirrors/.sucoder-state"
    assert check_durable_root("/g/mirrors/sample2", "/g/mirrors/sample", "/local") == "/g/mirrors/sample2"
    assert check_durable_root("/g/mirrors/sample-state", "/g/mirrors/sample", "/local") == "/g/mirrors/sample-state"
    assert check_durable_root("/localdisk/state", "/g/mirrors/sample", "/local") == "/localdisk/state"
    assert check_durable_root("/local/state", "/g/mirrors/sample", None) == "/local/state"


@pytest.mark.parametrize("value", [SELECTOR, "a" * 32 + "/000001-deadbeef"])
def test_restore_selector_accepts_run_slash_generation(value):
    assert validate_restore_selector(value) == value


@pytest.mark.parametrize("value", [
    "", "latest", "a" * 32, "000412-0a1b2c3d", "a" * 31 + "/000001-deadbeef", SELECTOR.upper(),
    "a" * 32 + "/1-deadbeef", "a" * 32 + "/000001-deadbee", SELECTOR + "\n", " " + SELECTOR,
    "../" + SELECTOR, "/global/x/" + SELECTOR, SELECTOR + "/db", SELECTOR + "; rm -rf /", "$(id)",
])
def test_restore_selector_rejects_everything_else(value):
    with pytest.raises(ValueError, match="expected RUN/GENERATION"):
        validate_restore_selector(value)


def test_runner_source_is_the_staged_file_and_stays_valid_for_python_3_6():
    assert runner_source() == RUNNER.read_text(encoding="utf-8")
    old = shutil.which("python3.6")
    if old is None:
        pytest.skip("no python3.6 to compile against")
    subprocess.run([old, "-m", "py_compile", str(RUNNER)], check=True)


# ----------------------------------------------------------------- config

def _slurm(**codex_state):
    return _parse_slurm_config({"partition": "p", "account": "a", "local_disk": True, "codex_state": codex_state})


def test_codex_state_config_parses_defaults_and_values():
    assert _parse_slurm_config({"partition": "p", "account": "a"}).codex_state is None
    parsed = _slurm(enabled=True).codex_state
    assert parsed == CodexStateConfig(enabled=True, checkpoint_minutes=10, checkpoint_timeout_seconds=30, durable_root=None)
    parsed = _slurm(enabled=False, checkpoint_minutes=5, checkpoint_timeout_seconds=60, durable_root="/global/scratch/x").codex_state
    assert (parsed.enabled, parsed.checkpoint_minutes, parsed.checkpoint_timeout_seconds, parsed.durable_root) == (False, 5, 60, "/global/scratch/x")


@pytest.mark.parametrize("raw, message", [
    ({"partition": "p", "account": "a", "codex_state": {"enabled": True}}, "requires `slurm.local_disk`"),
    ({"partition": "p", "account": "a", "local_disk": True, "codex_state": {"enabled": "yes"}}, "must be a boolean"),
    ({"partition": "p", "account": "a", "local_disk": True, "codex_state": {"checkpoint_minutes": True}}, "positive integer"),
    ({"partition": "p", "account": "a", "local_disk": True, "codex_state": {"checkpoint_minutes": 0}}, "positive integer"),
    ({"partition": "p", "account": "a", "local_disk": True, "codex_state": {"checkpoint_timeout_seconds": 4000}}, "no greater than 3600"),
    ({"partition": "p", "account": "a", "local_disk": True, "codex_state": {"durable_root": "relative"}}, "absolute path"),
    ({"partition": "p", "account": "a", "local_disk": True, "codex_state": {"durable_root": ""}}, "absolute path"),
    ({"partition": "p", "account": "a", "local_disk": True, "codex_state": {"bogus": 1}}, "Unknown key"),
    ({"partition": "p", "account": "a", "local_disk": True, "codex_state": "on"}, "must be a mapping"),
])
def test_codex_state_config_rejections(raw, message):
    with pytest.raises(ConfigError, match=message):
        _parse_slurm_config(raw)


# ----------------------------------------------------------------- launch wiring

def _state_manager(tmp_path, monkeypatch, *, enabled=True, local_disk_root="/local"):
    manager, ctx = _manager(tmp_path, monkeypatch, remote=True)
    ctx.settings.remote.slurm.local_disk = "/local"
    ctx.settings.remote.slurm.codex_state = CodexStateConfig(enabled=enabled)
    manager.executor.local_disk_root = local_disk_root
    manager.executor.agent_user = "coder"
    monkeypatch.setattr(manager, "_resolve_remote_path", lambda ctx: "/global/scratch/u/mirrors/sample")
    confined = Mock(return_value=0)
    monkeypatch.setattr(manager, "_launch_confined", confined)
    return manager, ctx, confined


@pytest.mark.parametrize("command, remote_control", [
    (["codex", "remote-control"], None), (["codex"], None), (["codex"], "terminal"), (["codex"], "server"),
])
def test_launch_wraps_codex_in_the_staged_runner_with_the_placeholder_after_codex(tmp_path, monkeypatch, command, remote_control):
    manager, ctx, confined = _state_manager(tmp_path, monkeypatch)
    ctx.settings.agent_launcher = AgentLauncher(command=["claude"])
    manager.launch_agent(ctx, sync=False, command_override=command, supports_inline_prompt=True, detached=True,
                         codex_remote_control=remote_control)
    launched = confined.call_args.args[1]
    assert launched[0] == "python3" and re.search(r"codex-state-runner-sample-[0-9a-f]{16}\.py$", launched[1])
    assert launched[2:12] == [
        "--scope", scope_id("cluster", "coder", "/global/scratch/u/mirrors/sample"), "--placeholder", PLACEHOLDER,
        "--durable-root", "/global/scratch/u/mirrors/.sucoder-state", "--checkpoint-minutes", "10", "--checkpoint-timeout", "30"]
    assert launched[12] == "--"
    inner = launched[13:]
    marker = f'sqlite_home="{PLACEHOLDER}"'
    if remote_control is None:
        assert inner.count(marker) == 1
        codex_at = inner.index("codex")
        assert inner[codex_at + 1:codex_at + 3] == ["-c", marker]
    else:
        # The remote-control supervisor carries two Codex invocations, the
        # app-server and the terminal client; both run inside the runner's
        # namespace and both must carry the placeholder.
        at = inner.index("sucoder-codex-remote")
        count = int(inner[at + 2])
        server, client = inner[at + 3:at + 3 + count], inner[at + 3 + count:]
        assert server[:3] == ["codex", "app-server", "--remote-control"] and server[3:5] == ["-c", marker]
        assert client[:3] == ["codex", "-c", marker]
    if command == ["codex", "remote-control"]:
        assert inner[0] == "python3" and inner[1].endswith(".txt")   # the context installer stays inside
    staged = [c for c in manager.executor.run_agent.call_args_list if c.kwargs.get("input") == runner_source()]
    assert len(staged) == 1 and staged[0].args[0][-1] == launched[1]


def test_launch_without_local_disk_refuses_managed_state(tmp_path, monkeypatch):
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch, local_disk_root=None)
    with pytest.raises(MirrorError, match="local-disk tiering is off"):
        manager.launch_agent(ctx, sync=False, command_override=["codex"], supports_inline_prompt=True, detached=True)


def test_launch_with_explicit_sqlite_home_refuses(tmp_path, monkeypatch):
    manager, ctx, _ = _state_manager(tmp_path, monkeypatch)
    with pytest.raises(MirrorError, match="already sets sqlite_home"):
        manager.launch_agent(ctx, sync=False, command_override=["codex", "-c", 'sqlite_home="/x"'],
                             supports_inline_prompt=True, detached=True)


@pytest.mark.parametrize("enabled, command", [(False, ["codex"]), (True, ["claude"])])
def test_launch_leaves_other_cases_alone(tmp_path, monkeypatch, enabled, command):
    manager, ctx, confined = _state_manager(tmp_path, monkeypatch, enabled=enabled)
    ctx.settings.agent_launcher = AgentLauncher(command=["claude"])
    manager.launch_agent(ctx, sync=False, command_override=command, supports_inline_prompt=True, detached=True)
    launched = confined.call_args.args[1]
    assert "codex-state-runner" not in " ".join(launched) and PLACEHOLDER not in " ".join(launched)


def _runner_argv(confined):
    """The staged runner's own options: after its path, before the separator."""
    launched = confined.call_args.args[1]
    return launched[2:launched.index("--")]


def _launch(manager, ctx, command=("codex",), **kwargs):
    return manager.launch_agent(ctx, sync=False, command_override=list(command), supports_inline_prompt=True,
                                detached=True, **kwargs)


def test_launch_honours_a_configured_durable_root_and_cadence(tmp_path, monkeypatch):
    manager, ctx, confined = _state_manager(tmp_path, monkeypatch)
    ctx.settings.remote.slurm.codex_state = CodexStateConfig(
        enabled=True, checkpoint_minutes=3, checkpoint_timeout_seconds=45, durable_root="/global/scratch/u/state/")
    _launch(manager, ctx)
    assert _runner_argv(confined)[2:] == [
        "--placeholder", PLACEHOLDER, "--durable-root", "/global/scratch/u/state",
        "--checkpoint-minutes", "3", "--checkpoint-timeout", "45"]


@pytest.mark.parametrize("root, message", [
    ("/global/scratch/u/mirrors/sample", "is the shared mirror repository itself"),
    ("/global/scratch/u/mirrors/sample/.sucoder-state", "inside the shared mirror repository"),
    ("/global/scratch/u/mirrors", "contains the shared mirror repository"),
    ("/local/codex-state", "allocation-local storage is not durable"),
])
def test_launch_refuses_a_durable_root_nested_with_the_mirror_or_on_local_disk(tmp_path, monkeypatch, root, message):
    manager, ctx, confined = _state_manager(tmp_path, monkeypatch)
    ctx.settings.remote.slurm.codex_state = CodexStateConfig(enabled=True, durable_root=root)
    with pytest.raises(MirrorError, match=message):
        _launch(manager, ctx)
    assert not confined.called


def test_launch_passes_a_valid_selector_as_restore(tmp_path, monkeypatch):
    manager, ctx, confined = _state_manager(tmp_path, monkeypatch)
    _launch(manager, ctx, codex_state_from=SELECTOR)
    argv = _runner_argv(confined)
    assert argv[-2:] == ["--restore", SELECTOR] and "--fresh" not in argv


@pytest.mark.parametrize("value", ["latest", "../x/000001-deadbeef", "/global/x/" + SELECTOR, SELECTOR + "; rm -rf /",
                                   SELECTOR.upper(), SELECTOR + "/db"])
def test_launch_rejects_a_bad_selector_before_staging_anything(tmp_path, monkeypatch, value):
    manager, ctx, confined = _state_manager(tmp_path, monkeypatch)
    with pytest.raises(MirrorError, match="expected RUN/GENERATION"):
        _launch(manager, ctx, codex_state_from=value)
    assert not confined.called
    assert not any(c.kwargs.get("input") == runner_source() for c in manager.executor.run_agent.call_args_list)


def test_launch_passes_fresh(tmp_path, monkeypatch):
    manager, ctx, confined = _state_manager(tmp_path, monkeypatch)
    _launch(manager, ctx, codex_state_fresh=True)
    argv = _runner_argv(confined)
    assert argv[-1] == "--fresh" and "--restore" not in argv


def test_launch_rejects_restore_and_fresh_together(tmp_path, monkeypatch):
    manager, ctx, confined = _state_manager(tmp_path, monkeypatch)
    with pytest.raises(MirrorError, match="mutually exclusive"):
        _launch(manager, ctx, codex_state_from=SELECTOR, codex_state_fresh=True)
    assert not confined.called


@pytest.mark.parametrize("selection", [{"codex_state_from": SELECTOR}, {"codex_state_fresh": True}])
@pytest.mark.parametrize("enabled, command, message", [
    (False, ["codex"], "not enabled for this target"),
    (True, ["claude"], "apply to a Codex launch only"),
])
def test_launch_rejects_a_selection_without_managed_state(tmp_path, monkeypatch, selection, enabled, command, message):
    manager, ctx, confined = _state_manager(tmp_path, monkeypatch, enabled=enabled)
    ctx.settings.agent_launcher = AgentLauncher(command=["claude"])
    with pytest.raises(MirrorError, match=message):
        _launch(manager, ctx, command, **selection)
    assert not confined.called


def _cli_launch(tmp_path, monkeypatch, subcommand, flags):
    path = _write_config(tmp_path, skills_entry=tmp_path / "skills")
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **k: None)
    calls = []
    manager = SimpleNamespace(
        context_for=lambda name: name,
        bootstrap=lambda *a, **kw: calls.append(kw),
        launch_agent=lambda *a, **kw: calls.append(kw),
    )
    monkeypatch.setattr(cli, "_build_manager_for_mirror", lambda *a, **kw: manager)
    result = CliRunner().invoke(cli.app, ["--config", str(path), subcommand, "sample", "--agent", "codex", *flags])
    return result, calls


@pytest.mark.parametrize("subcommand", ["collaborate", "agents-run"])
@pytest.mark.parametrize("flags, expected", [
    ([], {"codex_state_from": None, "codex_state_fresh": False}),
    (["--codex-state-from", SELECTOR], {"codex_state_from": SELECTOR, "codex_state_fresh": False}),
    (["--codex-state-fresh"], {"codex_state_from": None, "codex_state_fresh": True}),
])
def test_cli_forwards_codex_state_selection(tmp_path, monkeypatch, subcommand, flags, expected):
    result, calls = _cli_launch(tmp_path, monkeypatch, subcommand, flags)
    assert result.exit_code == 0, (result.output, result.exception)
    assert {key: calls[0][key] for key in expected} == expected
    assert calls[0]["command_override"] == ["codex"]


@pytest.mark.parametrize("subcommand", ["collaborate", "agents-run"])
def test_cli_rejects_both_codex_state_selections_before_touching_anything(tmp_path, monkeypatch, subcommand):
    result, calls = _cli_launch(tmp_path, monkeypatch, subcommand, ["--codex-state-from", SELECTOR, "--codex-state-fresh"])
    assert result.exit_code == 2 and calls == []
    with pytest.raises(Exception, match="conflicts"):
        cli._codex_state_selection(SELECTOR, True)


# ----------------------------------------------------------------- the runner, on this node

@pytest.fixture
def home(tmp_path):
    codex_home = tmp_path / "home"
    for d in ("sessions", "thread-writer-locks", "skills"):
        (codex_home / d).mkdir(parents=True)
    (codex_home / "skills" / "s.md").write_text("skill\n")
    (codex_home / "AGENTS.md").write_text("GLOBAL\n")
    (codex_home / "config.toml").write_text('model = "x"\n')
    local = tmp_path / "local"
    local.mkdir()
    durable = tmp_path / "durable"      # the runner's --durable-root; never under the local root
    durable.mkdir()
    fake = tmp_path / "fake_codex.py"
    fake.write_text(FAKE_CODEX)
    fake.chmod(0o700)
    return {"codex_home": codex_home, "local": local, "durable": durable, "fake": fake, "scope": "ab" * 32}


def _env(home, **extra):
    env = {k: v for k, v in os.environ.items() if k not in ("CODEX_SQLITE_HOME",)}
    env.update({"SUCODER_LOCAL_ROOT": str(home["local"]), "CODEX_HOME": str(home["codex_home"])})
    env.update(extra)
    return env


def _runner(home, env, placeholder=True):
    arg = f'sqlite_home="{PLACEHOLDER}"' if placeholder else 'sqlite_home="/elsewhere"'
    return subprocess.Popen(
        [sys.executable, str(RUNNER), "--scope", home["scope"], "--durable-root", str(home["durable"]), "--",
         sys.executable, str(home["fake"]), "-c", arg, "app-server"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _status(home):
    scope_dir = home["local"] / "state" / "codex" / home["scope"]
    run = (scope_dir / "latest-run").read_text().strip()
    return json.loads((scope_dir / run / "status.json").read_text()), scope_dir / run


def _wait_state(home, wanted, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status, run_dir = _status(home)
            if status["state"] in wanted:
                return status, run_dir
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(0.1)
    raise AssertionError(f"runner never reached {wanted}")


@needs_bwrap
def test_runner_places_databases_and_rollout_locally_and_forwards_sigterm(home):
    proc = _runner(home, _env(home))
    try:
        status, run_dir = _wait_state(home, {"placed"})
        placed = status["placed"]
        assert str(run_dir / "db" / "state_5.sqlite") in placed
        assert (run_dir / "db" / "state_5.sqlite").exists()
        assert list((run_dir / "home" / "sessions").rglob("rollout-*.jsonl")), "rollout must land in the bound local directory"
        assert not list((home["codex_home"] / "sessions").rglob("*")), "the shared sessions directory stays untouched"
        assert sorted(status["skipped"]) == sorted([".tmp", "tmp", "shell_snapshots", "app-server-control", "AGENTS.override.md", "history.jsonl"])
        owner = json.loads((run_dir / "owner.json").read_text())
        assert {b["target"] for b in owner["bound"]} == {str(home["codex_home"] / d) for d in ("sessions", "thread-writer-locks", "skills", "AGENTS.md")}
        assert (run_dir / "runner.log").read_text().count("SUCODER codex-state") >= 2
        assert oct((run_dir / "db").stat().st_mode & 0o777) == "0o700"
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=15) == 37, proc.stderr.read()
        status, _ = _status(home)
        assert status["state"] == "exited" and status["exit"] == 37
    finally:
        if proc.poll() is None:
            proc.kill()


@needs_bwrap
def test_runner_stops_codex_that_opens_a_database_elsewhere(home):
    proc = _runner(home, _env(home, FAKE_IGNORE_SQLITE_HOME="1"))
    assert proc.wait(timeout=30) == 71
    assert "outside the managed placement" in proc.stderr.read()
    status, _ = _status(home)
    assert status["state"] == "stopped_misplaced" and status["misplaced"] == [str(home["codex_home"] / "state_5.sqlite")]


@needs_bwrap
def test_runner_reports_a_child_that_exits_before_opening_a_store(home):
    proc = _runner(home, _env(home, FAKE_EXIT_NOW="1"))
    assert proc.wait(timeout=30) == 5
    status, _ = _status(home)
    assert status["state"] == "exited" and status["outcome"] == "no_stores_opened"


@needs_bwrap
@pytest.mark.parametrize("case", ["env", "config", "profile", "argv", "no_sessions", "no_placeholder"])
def test_runner_refuses_conflicts_and_missing_prerequisites(home, case):
    env = _env(home)
    placeholder = True
    if case == "env":
        env["CODEX_SQLITE_HOME"] = "/x"
    elif case == "config":
        (home["codex_home"] / "config.toml").write_text('sqlite_home = "/x"\n')
    elif case == "profile":
        (home["codex_home"] / "fast.config.toml").write_text('sqlite_home = "/x"\n')
        proc = subprocess.Popen([sys.executable, str(RUNNER), "--scope", home["scope"], "--durable-root", str(home["durable"]), "--",
                                 sys.executable, str(home["fake"]), "--profile", "fast", "-c", f'sqlite_home="{PLACEHOLDER}"'],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert proc.wait(timeout=30) == 70 and "fast.config.toml" in proc.stderr.read()
        return
    elif case == "argv":
        placeholder = False
    elif case == "no_sessions":
        shutil.rmtree(home["codex_home"] / "sessions")
    if case == "no_placeholder":
        proc = subprocess.Popen([sys.executable, str(RUNNER), "--scope", home["scope"], "--durable-root", str(home["durable"]), "--",
                                 sys.executable, str(home["fake"]), "app-server"],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    else:
        proc = _runner(home, env, placeholder=placeholder)
    assert proc.wait(timeout=30) == 70
    assert "refusing to launch" in proc.stderr.read()
    assert not list(home["local"].rglob("state_5.sqlite"))
    assert not (home["codex_home"] / "sessions" / "2026").exists()


def test_runner_refuses_without_local_root(home):
    env = _env(home)
    env.pop("SUCODER_LOCAL_ROOT")
    proc = _runner(home, env)
    assert proc.wait(timeout=30) == 70
    assert "SUCODER_LOCAL_ROOT" in proc.stderr.read()


@needs_bwrap
@pytest.mark.skipif(not shutil.which("codex"), reason="Codex CLI not installed")
def test_runner_places_a_real_codex_app_server(home, tmp_path):
    """The actual database paths of an actual Codex, as Slice 1 requires."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            event = {"type": "response.completed", "response": {
                "id": "resp_test", "object": "response", "status": "completed", "output": [],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}}
            self.wfile.write(("event: response.completed\ndata: " + json.dumps(event) + "\n\n").encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    options = ["-c", 'model_provider="offline"', "-c", 'model="fake"',
               "-c", "features.plugins=false", "-c", "features.remote_plugin=false", "-c", "features.apps=false",
               "-c", ('model_providers.offline={name="offline",'
                      f'base_url="http://127.0.0.1:{server.server_port}",wire_api="responses"}}')]
    command = with_sqlite_home_placeholder([shutil.which("codex"), *options, "app-server"])
    (tmp_path / "user-home").mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "user-home"), "NO_PROXY": "127.0.0.1", "RUST_LOG": "error",
           "SUCODER_LOCAL_ROOT": str(home["local"]), "CODEX_HOME": str(home["codex_home"]), "TMPDIR": str(tmp_path)}
    proc = subprocess.Popen([sys.executable, str(RUNNER), "--scope", home["scope"], "--durable-root", str(home["durable"]), "--", *command],
                            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        proc.stdin.write(json.dumps({"id": 1, "method": "initialize", "params": {
            "clientInfo": {"name": "sucoder_test", "version": "1"}, "capabilities": {"experimentalApi": True}}}) + "\n")
        proc.stdin.flush()
        proc.stdout.readline()
        proc.stdin.write('{"method":"initialized","params":{}}\n')
        proc.stdin.write(json.dumps({"id": 2, "method": "thread/start", "params": {"cwd": str(tmp_path)}}) + "\n")
        proc.stdin.flush()
        status, run_dir = _wait_state(home, {"placed", "misplaced"}, timeout=60)
        assert status["state"] == "placed", status
        names = {Path(p).name for p in status["placed"]}
        assert any(n.endswith(".sqlite") for n in names)
        assert all(p.startswith(str(run_dir / "db")) or Path(p).name.startswith("rollout-") for p in status["placed"])
        assert not list(home["codex_home"].glob("*.sqlite")), "no database in the shared home"
        proc.stdin.close()
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        server.shutdown()
        server.server_close()
