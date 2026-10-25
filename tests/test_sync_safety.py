"""Git-state regressions for independently managed, filtered followers."""
import os
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sucoder import cli
from sucoder.config import RemoteConfig
from sucoder.executor import CommandResult
from sucoder.mirror import MirrorError
from sucoder.remote_bootstrap import VERIFY_LAUNCH_MIRROR_SH
from tests.test_mirror import build_manager


def git(path, *args, check=True):
    return subprocess.run(["git", *args], cwd=path, text=True,
                          capture_output=True, check=check, timeout=30)


def snapshot(path):
    # Includes the index, refs, config, object store, and untracked files.
    return {str(p.relative_to(path)): p.read_bytes()
            for p in path.rglob("*") if p.is_file() and not p.is_symlink()}


def topology(tmp_path, monkeypatch):
    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    gate = ctx.canonical_path
    gate.chmod(gate.stat().st_mode | 0o200)
    (gate / "formal/model/lib").mkdir(parents=True)
    (gate / "formal/model/lib/old.txt").write_text("excluded baseline\n")
    git(gate, "add", ".")
    git(gate, "commit", "-qm", "accepted baseline")
    git(gate, "config", "uploadpack.allowFilter", "true")
    satellite = tmp_path / "satellite"
    git(tmp_path, "clone", "-q", "--no-hardlinks", str(gate), str(satellite))
    laptop = tmp_path / "laptop"
    git(tmp_path, "clone", "-q", "--filter=blob:none", "--no-checkout",
        "--origin=gate", gate.as_uri(), str(laptop))
    patterns = "/*\n!/formal/model/lib/\n!/formal/migration/baseline-sources.jsonl\n!/formal/evidence/\n!/formal/parameter_cell_*/\n!/results/*/data/\n!/results/*/cells/\n"
    subprocess.run(["git", "sparse-checkout", "set", "--no-cone", "--stdin"],
                   cwd=laptop, input=patterns, text=True, check=True, capture_output=True)
    git(laptop, "checkout", "-q", "main")
    git(laptop, "config", "remote.gate.fetch", "+refs/heads/main:refs/remotes/gate/main")
    git(laptop, "config", "remote.gate.tagOpt", "--no-tags")
    git(laptop, "config", "user.name", "Test")
    git(laptop, "config", "user.email", "test@example.invalid")
    ctx.settings.canonical_repo = laptop
    ctx.settings.remote = RemoteConfig(gateway="unused", transfer_host="unused",
                                       mirror_root=tmp_path)
    monkeypatch.setattr(manager, "_resolve_remote_path", lambda c: str(satellite))
    monkeypatch.setattr(manager, "_git_transports", lambda c: [("test", satellite.as_uri(), None)])
    return manager, ctx, laptop, satellite


def advance(path, name="new.txt"):
    git(path, "config", "user.name", "Test")
    git(path, "config", "user.email", "test@example.invalid")
    (path / name).write_text("new work\n")
    git(path, "add", name)
    git(path, "commit", "-qm", "independent work")
    return git(path, "rev-parse", "HEAD").stdout.strip()


@pytest.mark.parametrize("diverged", [False, True])
@pytest.mark.parametrize("dirty", ["clean", "tracked", "staged", "topic"])
def test_no_sync_bootstrap_preserves_both_repositories(tmp_path, monkeypatch, diverged, dirty):
    manager, ctx, laptop, satellite = topology(tmp_path, monkeypatch)
    tip = advance(satellite, "formal/model/lib/new.txt")
    if diverged:
        advance(laptop, "local.txt")
    if dirty == "topic":
        git(laptop, "checkout", "-qb", "work/topic")
    elif dirty != "clean":
        (laptop / "README.md").write_text("unsaved edit\n")
        if dirty == "staged":
            git(laptop, "add", "README.md")
    (laptop / "untracked.txt").write_text("preserve me\n")
    git(laptop, "update-ref", "refs/sucoder/mirror-head", "HEAD")
    before = snapshot(laptop), snapshot(satellite)
    calls = []
    def launch(c, **kwargs):
        calls.append(kwargs)
        manager._existing_launch_mirror(c)
        return 0
    monkeypatch.setattr(manager, "launch_agent", launch)
    for method in ("ensure_remote_clone", "_configure_target_remote", "_resolve_base_branch",
                   "_pull_from_remote", "_sync_remote"):
        monkeypatch.setattr(manager, method, lambda *a, **k: pytest.fail("synchronization ran"))
    assert manager.bootstrap(ctx, sync=False) == 0
    assert calls[0]["sync"] is False
    assert (snapshot(laptop), snapshot(satellite)) == before
    assert git(satellite, "rev-parse", "HEAD").stdout.strip() == tip


@pytest.mark.parametrize("state", ["missing", "empty", "unborn", "bare", "subdirectory"])
def test_no_sync_refuses_uninitialized_mirror(tmp_path, monkeypatch, state):
    manager, ctx, laptop, satellite = topology(tmp_path, monkeypatch)
    path = tmp_path / state
    if state == "subdirectory":
        path = satellite / state
    if state != "missing":
        path.mkdir()
    if state == "unborn":
        git(path, "init", "-q")
    elif state == "bare":
        git(path, "init", "--bare", "-q")
    monkeypatch.setattr(manager, "_resolve_remote_path", lambda c: str(path))
    before = snapshot(laptop), snapshot(path) if path.exists() else None
    with pytest.raises(MirrorError, match="absent|unverified"):
        manager.launch_agent(ctx, sync=False)
    assert (snapshot(laptop), snapshot(path) if path.exists() else None) == before


@pytest.mark.parametrize("entry", ["bootstrap", "launch_agent"])
def test_no_sync_dry_run_does_not_probe_or_launch(tmp_path, monkeypatch, entry):
    manager, ctx, laptop, satellite = topology(tmp_path, monkeypatch)
    manager.executor.dry_run = True
    before = snapshot(laptop), snapshot(satellite)
    monkeypatch.setattr(manager, "_resolve_remote_path", lambda *a: pytest.fail("probed remote"))
    assert getattr(manager, entry)(ctx, sync=False) == 0
    assert (snapshot(laptop), snapshot(satellite)) == before


@pytest.mark.parametrize("entry", ["bootstrap", "launch_agent"])
def test_no_sync_task_refuses_before_setup(tmp_path, monkeypatch, entry):
    manager, ctx, laptop, satellite = topology(tmp_path, monkeypatch)
    before = snapshot(laptop), snapshot(satellite)
    with pytest.raises(MirrorError, match="--task cannot"):
        getattr(manager, entry)(ctx, sync=False, task_name="demo")
    assert (snapshot(laptop), snapshot(satellite)) == before


@pytest.mark.parametrize("command", ["collaborate", "agents-run"])
def test_no_sync_task_cli_refuses_before_setup(tmp_path, monkeypatch, command):
    from tests.test_cli import _plain_output, _write_config
    config = _write_config(tmp_path, skills_entry=tmp_path / "skills")
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_tmux_launch_preamble", lambda *a, **k: pytest.fail("created session"))
    result = CliRunner().invoke(cli.app, ["--config", str(config), command, "sample", "--no-sync", "--task", "demo"])
    assert result.exit_code != 0
    assert "--task cannot" in _plain_output(result)


def test_unavailable_is_not_reported_as_missing(tmp_path, monkeypatch):
    manager, ctx, laptop, satellite = topology(tmp_path, monkeypatch)
    monkeypatch.setattr(manager.executor, "run_agent",
                        lambda args, **kw: CommandResult(args, args, "", "SSH unavailable", 255))
    with pytest.raises(MirrorError, match="unavailable or unverified"):
        manager._existing_launch_mirror(ctx)
