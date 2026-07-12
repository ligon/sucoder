from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("typer")

from typer.testing import CliRunner

from sucoder import cli
from sucoder.config import BranchPrefixes, Config, MirrorSettings

try:
    from click.shell_completion import CompletionItem as ClickCompletionItem
except (ImportError, AttributeError):  # pragma: no cover - defensive
    ClickCompletionItem = None  # type: ignore[assignment]


def _write_config(tmp_path: Path, *, skills_entry: Path) -> Path:
    human = os.environ.get("USER", "coder")
    agent = os.environ.get("USER", "coder")
    mirror_root = tmp_path / "mirrors"
    mirror_root.mkdir(exist_ok=True)
    canonical_repo = tmp_path / "canonical"
    canonical_repo.mkdir(exist_ok=True)

    config_content = f"""
human_user: {human}
agent_user: {agent}
agent_group: {agent}
mirror_root: {mirror_root}
mirrors:
  sample:
    canonical_repo: {canonical_repo}
    mirror_name: sample
    branch_prefixes:
      human: {human}
      agent: {agent}
    skills:
      - {skills_entry}
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content, encoding="utf-8")
    return config_path


def test_mirrors_list_outputs_configured_entries(tmp_path, monkeypatch):
    runner = CliRunner()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    monkeypatch.setattr(cli, "run_startup_checks", lambda *args, **kwargs: None)

    config_path = _write_config(tmp_path, skills_entry=skills_dir)

    result = runner.invoke(cli.app, ["--config", str(config_path), "mirrors-list"])

    assert result.exit_code == 0
    stdout = result.stdout
    assert "Mirror" in stdout
    assert "sample" in stdout
    assert str(tmp_path / "canonical") in stdout
    assert str(tmp_path / "mirrors" / "sample") in stdout


def test_skills_list_reports_accessible_paths(tmp_path, monkeypatch):
    runner = CliRunner()

    home_dir = tmp_path / "home"
    skills_dir = home_dir / ".sucoder" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "orgmode").mkdir()
    (skills_dir / "SKILL.md").write_text("name: sample\n", encoding="utf-8")
    catalog = home_dir / ".sucoder" / "SKILLS.md"
    catalog.write_text("# Catalog\n", encoding="utf-8")

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *args, **kwargs: None)

    config_path = _write_config(tmp_path, skills_entry=skills_dir)

    result = runner.invoke(cli.app, ["--config", str(config_path), "skills-list"])

    assert result.exit_code == 0
    stdout = result.stdout
    assert str(skills_dir) in stdout
    assert "[OK]" in stdout
    assert "sample" in stdout or "SKILL.md" in stdout


def test_skills_list_reports_missing_path(tmp_path, monkeypatch):
    runner = CliRunner()

    home_dir = tmp_path / "home"
    skills_dir = home_dir / ".sucoder" / "skills"
    skills_dir.mkdir(parents=True)
    catalog = home_dir / ".sucoder" / "SKILLS.md"
    catalog.write_text("# Catalog\n", encoding="utf-8")

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *args, **kwargs: None)

    missing_path = tmp_path / "missing-skills"
    config_path = _write_config(tmp_path, skills_entry=missing_path)

    result = runner.invoke(cli.app, ["--config", str(config_path), "skills-list"])

    assert result.exit_code == 1
    assert "[MISSING]" in result.stdout
    assert str(missing_path) in result.stdout


def test_mirror_completion_uses_click_completion_items(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    config_path = _write_config(tmp_path, skills_entry=skills_dir)
    ctx = SimpleNamespace(obj={}, params={"config": config_path})

    completions = cli._mirror_completion(ctx, None, "sam")

    assert completions, "Expected at least one completion candidate."
    first = completions[0]
    if ClickCompletionItem is not None:
        assert isinstance(first, ClickCompletionItem)
        assert first.value == "sample"
    else:
        assert first == "sample"


# ---------------------------------------------------------------------------
# Zero-config callback flow
# ---------------------------------------------------------------------------


def _fake_default_config(tmp_path: Path) -> Config:
    """Build a minimal Config like build_default_config would produce."""
    user = os.environ.get("USER", "testuser")
    mirror = MirrorSettings(
        name="myrepo",
        canonical_repo=tmp_path,
        mirror_name="myrepo",
        branch_prefixes=BranchPrefixes(human=user, agent="coder"),
    )
    return Config(
        human_user=user,
        agent_user="coder",
        agent_group="coder",
        mirror_root=Path("/var/tmp/coder-mirrors"),
        mirrors={"myrepo": mirror},
    )


def test_zero_config_mirrors_list(tmp_path, monkeypatch):
    """mirrors-list works without a config file when build_default_config succeeds."""
    runner = CliRunner()
    # Ensure default config path does not exist.
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    cfg = _fake_default_config(tmp_path)
    monkeypatch.setattr(cli, "build_default_config", lambda: cfg)
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    result = runner.invoke(cli.app, ["mirrors-list"])
    assert result.exit_code == 0
    assert "myrepo" in result.stdout


def test_zero_config_startup_warning(tmp_path, monkeypatch):
    """In zero-config mode, startup check failures become warnings instead of errors."""
    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    cfg = _fake_default_config(tmp_path)
    monkeypatch.setattr(cli, "build_default_config", lambda: cfg)

    from sucoder.startup_checks import StartupError
    monkeypatch.setattr(
        cli, "run_startup_checks",
        lambda *a, **kw: (_ for _ in ()).throw(StartupError("agent user not found")),
    )

    result = runner.invoke(cli.app, ["mirrors-list"])
    # Should NOT exit with code 2 — warning only.
    assert result.exit_code == 0
    assert "Warning" in result.output or "agent user not found" in result.output


# ---------------------------------------------------------------------------
# _resolve_mirror_name
# ---------------------------------------------------------------------------


def test_resolve_mirror_name_single(tmp_path, monkeypatch):
    """When config has exactly one mirror, omitting the name succeeds."""
    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    cfg = _fake_default_config(tmp_path)
    monkeypatch.setattr(cli, "build_default_config", lambda: cfg)
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    # mirrors-list doesn't take a mirror arg, so test via status which
    # does require a mirror.  It will fail at the MirrorManager level, but
    # the important thing is it gets past _resolve_mirror_name.
    result = runner.invoke(cli.app, ["status"])
    # Should not fail due to "specify one of" (mirror resolution worked).
    assert "specify one of" not in (result.stdout + (result.output or ""))


def test_resolve_mirror_name_explicit(tmp_path, monkeypatch):
    """Explicit mirror name is passed through."""
    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    cfg = _fake_default_config(tmp_path)
    monkeypatch.setattr(cli, "build_default_config", lambda: cfg)
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    result = runner.invoke(cli.app, ["status", "myrepo"])
    assert "specify one of" not in (result.stdout + (result.output or ""))


# ---------------------------------------------------------------------------
# _resolve_mirror_name – git-based auto-detection (multi-mirror configs)
# ---------------------------------------------------------------------------


def _multi_mirror_config(tmp_path: Path) -> Config:
    """Build a Config with two mirrors so the single-mirror shortcut is skipped."""
    user = os.environ.get("USER", "testuser")
    repo_a = tmp_path / "RepoA"
    repo_a.mkdir(exist_ok=True)
    repo_b = tmp_path / "RepoB"
    repo_b.mkdir(exist_ok=True)

    def _mirror(name: str, repo: Path) -> MirrorSettings:
        return MirrorSettings(
            name=name,
            canonical_repo=repo,
            mirror_name=name,
            branch_prefixes=BranchPrefixes(human=user, agent="coder"),
        )

    return Config(
        human_user=user,
        agent_user="coder",
        agent_group="coder",
        mirror_root=Path("/var/tmp/coder-mirrors"),
        mirrors={
            "repo-a": _mirror("repo-a", repo_a),
            "repo-b": _mirror("repo-b", repo_b),
        },
    )


def test_resolve_mirror_name_matches_configured_canonical(tmp_path, monkeypatch):
    """When cwd's git root matches a configured mirror's canonical_repo, use it."""
    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    cfg = _multi_mirror_config(tmp_path)
    monkeypatch.setattr(cli, "build_default_config", lambda: cfg)
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    # Simulate git returning the canonical_repo path of repo-b.
    target_repo = cfg.mirrors["repo-b"].canonical_repo
    monkeypatch.setattr(
        cli, "_detect_git_toplevel", lambda: target_repo,
    )

    # Use status command; it will fail at MirrorManager level but should
    # get past _resolve_mirror_name without "specify one of".
    result = runner.invoke(cli.app, ["status"])
    assert "specify one of" not in (result.stdout + (result.output or ""))


def test_resolve_mirror_name_creates_ephemeral_for_unconfigured_repo(tmp_path, monkeypatch):
    """When cwd is a git repo not in config, an ephemeral mirror is created."""
    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    cfg = _multi_mirror_config(tmp_path)
    monkeypatch.setattr(cli, "build_default_config", lambda: cfg)
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    # Simulate git returning a repo that is NOT in the config.
    unconfigured_repo = tmp_path / "VESDemand"
    unconfigured_repo.mkdir()
    monkeypatch.setattr(
        cli, "_detect_git_toplevel", lambda: unconfigured_repo,
    )

    result = runner.invoke(cli.app, ["status"])
    # Should not hit the "Multiple mirrors" error.
    assert "specify one of" not in (result.stdout + (result.output or ""))
    # The ephemeral mirror should have been injected.
    assert "VESDemand" in cfg.mirrors


def test_resolve_mirror_name_not_in_git_repo(tmp_path, monkeypatch):
    """When not in a git repo and multiple mirrors exist, show the error."""
    from sucoder.config import ConfigError as CfgError

    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    cfg = _multi_mirror_config(tmp_path)
    monkeypatch.setattr(cli, "build_default_config", lambda: cfg)
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    def _raise_not_git():
        raise CfgError("Not inside a git repository.")

    monkeypatch.setattr(cli, "_detect_git_toplevel", _raise_not_git)

    result = runner.invoke(cli.app, ["status"])
    assert "specify one of" in (result.stdout + (result.output or "")).lower()


def test_resolve_mirror_name_explicit_unconfigured_matches_cwd(tmp_path, monkeypatch):
    """Regression: an explicit mirror name that isn't configured but names
    the git repo we're standing in must get an ephemeral entry — the same
    one a no-arg `collaborate` auto-creates.

    Field symptom: a no-arg `collaborate` from ~/Projects/Emu-GMM created
    the mirror and ran, but `attach Emu-GMM` (explicit) then reported
    'Mirror is not configured for remote execution' because the explicit
    path returned the name blindly and skipped ephemeral creation.
    """
    from types import SimpleNamespace

    cfg = _multi_mirror_config(tmp_path)
    repo = tmp_path / "Emu-GMM"
    repo.mkdir()
    monkeypatch.setattr(cli, "_detect_git_toplevel", lambda: repo)

    ctx = SimpleNamespace(obj={"config": cfg})
    resolved = cli._resolve_mirror_name(ctx, "Emu-GMM")

    assert resolved == "Emu-GMM"
    # The ephemeral mirror must now exist so attach/release can use it.
    assert "Emu-GMM" in cfg.mirrors
    assert cfg.mirrors["Emu-GMM"].canonical_repo == repo


def test_resolve_mirror_name_explicit_unconfigured_name_mismatch(tmp_path, monkeypatch):
    """An explicit name that does NOT match the cwd repo is returned as-is
    (no fabricated ephemeral) so downstream reports 'not configured'."""
    from types import SimpleNamespace

    cfg = _multi_mirror_config(tmp_path)
    repo = tmp_path / "SomethingElse"
    repo.mkdir()
    monkeypatch.setattr(cli, "_detect_git_toplevel", lambda: repo)

    ctx = SimpleNamespace(obj={"config": cfg})
    resolved = cli._resolve_mirror_name(ctx, "Emu-GMM")

    assert resolved == "Emu-GMM"
    # No ephemeral fabricated for a name that doesn't match the cwd repo.
    assert "Emu-GMM" not in cfg.mirrors
    assert "SomethingElse" not in cfg.mirrors


def test_attach_refuses_login_node_when_compute_unknown(tmp_path, monkeypatch):
    """Regression: `attach` on a SLURM target with a recorded job but an
    UNKNOWN compute node (and no --via-srun) must refuse, not silently
    drop the user onto the login node.

    This is the gap the earlier `slurm_job_id: null` test didn't cover:
    a job IS recorded, but `compute_node` is null and the caller didn't
    ask to join via srun.  Pre-fix this fell through the `else` branch to
    a bare login-node tmux.
    """
    from sucoder import session as session_mod

    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    config_path = _slurm_config(tmp_path)

    sessions_dir = fake_home / ".sucoder" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    # Job recorded, but compute node unknown.
    (sessions_dir / "sample--fake-slurm.yaml").write_text(
        "login_node: ln001\nslurm_job_id: 7654321\ncompute_node: null\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(session_mod, "_session_dir", lambda: sessions_dir)

    def _no_real_ssh(*a, **kw):
        raise AssertionError("attach must bail out before exec/SSH")
    monkeypatch.setattr(os, "execvp", _no_real_ssh)

    result = runner.invoke(
        cli.app,
        ["--config", str(config_path), "-T", "fake-slurm", "attach", "sample"],
    )
    assert result.exit_code != 0
    combined = (result.stdout + (result.output or "")).lower()
    assert "compute node is unknown" in combined, combined
    assert "via-srun" in combined, combined


# ------------------------------------------------------------------
# Detach / scancel-lifecycle regressions
# ------------------------------------------------------------------


def test_slurm_timer_script_omits_scancel():
    """Regression: the backstop timer must NOT auto-scancel.

    Previously the on-compute-node monitor script ran
    ``scancel $JOB`` both on tmux-startup-timeout and on tmux-session-
    gone.  Either auto-cancel turned a transient agent failure into a
    catastrophic teardown (allocation released, no reattach possible).
    User now owns the SLURM lifecycle via ``sucoder release``.

    This is a source-inspection guard: it scans the body of
    ``_start_slurm_timer`` for any bare ``scancel <something>`` line
    that would execute as a shell command at runtime.  ``scancel``
    *can* appear in user-facing warning strings (e.g. "Run
    `scancel {q_job}` to free the allocation") — those are fine
    because they're inside an ``echo``/string, not a shell statement.
    """
    import inspect
    src = inspect.getsource(cli._start_slurm_timer)
    for raw in src.splitlines():
        stripped = raw.strip()
        # Skip strings that mention scancel for documentation/warnings.
        if not stripped.startswith("scancel"):
            continue
        # If we got here, a bare `scancel ...` shell command remains.
        pytest.fail(
            "_start_slurm_timer still emits a `scancel` shell "
            f"command line: {stripped!r}.  User owns SLURM lifecycle "
            "now; use `sucoder release` for explicit cancel."
        )


def _slurm_config(tmp_path: Path, *, with_session_jobid: bool = False) -> Path:
    """Write a config with a SLURM-backed target and (optionally) a
    saved RemoteSession for the sample mirror."""
    human = os.environ.get("USER", "coder")
    mirror_root = tmp_path / "mirrors"
    mirror_root.mkdir(exist_ok=True)
    canonical_repo = tmp_path / "canonical"
    canonical_repo.mkdir(exist_ok=True)

    config_content = f"""
human_user: {human}
agent_user: {human}
agent_group: {human}
mirror_root: {mirror_root}
mirrors:
  sample:
    canonical_repo: {canonical_repo}
    mirror_name: sample
    branch_prefixes:
      human: {human}
      agent: {human}
targets:
  fake-slurm:
    gateway: gw.example.org
    transfer_host: dtn.example.org
    slurm:
      partition: test
      account: test_acct
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content, encoding="utf-8")
    return config_path


def test_release_command_rejects_non_slurm_target(tmp_path, monkeypatch):
    """`sucoder release` should fail clearly when the target has no
    SLURM config (nothing to release)."""
    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    human = os.environ.get("USER", "coder")
    mirror_root = tmp_path / "mirrors"
    mirror_root.mkdir(exist_ok=True)
    canonical_repo = tmp_path / "canonical"
    canonical_repo.mkdir(exist_ok=True)

    # No remote / no slurm — pure local mirror.
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
human_user: {human}
agent_user: {human}
agent_group: {human}
mirror_root: {mirror_root}
mirrors:
  sample:
    canonical_repo: {canonical_repo}
    mirror_name: sample
    branch_prefixes:
      human: {human}
      agent: {human}
""",
        encoding="utf-8",
    )

    result = runner.invoke(cli.app, ["--config", str(config_path), "release", "sample"])
    assert result.exit_code != 0
    combined = (result.stdout + (result.output or "") + (str(result.stderr_bytes or b""))).lower()
    assert "not configured for remote" in combined or "no slurm" in combined


def test_release_command_no_recorded_job(tmp_path, monkeypatch):
    """`sucoder release` should exit cleanly (code 0) saying nothing
    to release when no SLURM job is recorded in the session."""
    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    config_path = _slurm_config(tmp_path)

    # No session file written, so session.slurm_job_id is None.
    result = runner.invoke(
        cli.app,
        ["--config", str(config_path), "-T", "fake-slurm", "release", "sample"],
    )
    assert result.exit_code == 0
    combined = result.stdout + (result.output or "")
    assert "nothing to release" in combined.lower()


def test_release_scancels_via_gateway(tmp_path, monkeypatch):
    """`release` cancels the job over the GATEWAY control (round-robin ->
    a healthy login node), never dialing the mirror's pinned login node --
    so a single wedged login node can't block a release."""
    from sucoder import session as session_mod

    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    config_path = _slurm_config(tmp_path)

    sessions_dir = fake_home / ".sucoder" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    # login_node pinned to a node we must NOT dial for the scancel.
    (sessions_dir / "sample--fake-slurm.yaml").write_text(
        "login_node: ln002\nslurm_job_id: 7654321\ncompute_node: n0032\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(session_mod, "_session_dir", lambda: sessions_dir)

    # The gateway ControlMaster "connects" without real ssh.
    monkeypatch.setattr(cli, "_connect_with_retry", lambda *a, **kw: None)

    # Capture where scancel is routed.
    captured: dict = {}
    def _fake_capture(control, host, command, **kw):
        captured["host"] = host
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(cli, "_run_remote_capture", _fake_capture)

    result = runner.invoke(
        cli.app,
        ["--config", str(config_path), "-T", "fake-slurm", "release", "sample", "-f"],
    )
    assert result.exit_code == 0, (result.stdout, result.exception)
    # Routed to the gateway, NOT the pinned login node.
    assert captured["host"] == "gw.example.org", captured
    assert "ln002" not in captured["host"]
    assert "scancel 7654321" in captured["command"], captured
    assert "Released SLURM job 7654321" in result.stdout

    # SLURM fields cleared; login_node retained for future attaches.
    reloaded = session_mod.RemoteSession.load("sample", target_name="fake-slurm")
    assert reloaded.slurm_job_id is None
    assert reloaded.login_node == "ln002"


def test_release_reports_gateway_scancel_failure(tmp_path, monkeypatch):
    """A non-zero scancel (SSH/auth/timeout, not 'no such job') is surfaced,
    exits non-zero, and does NOT clear the session (job may still be alive)."""
    from sucoder import session as session_mod

    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    config_path = _slurm_config(tmp_path)

    sessions_dir = fake_home / ".sucoder" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "sample--fake-slurm.yaml").write_text(
        "login_node: ln002\nslurm_job_id: 7654321\ncompute_node: n0032\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(session_mod, "_session_dir", lambda: sessions_dir)
    monkeypatch.setattr(cli, "_connect_with_retry", lambda *a, **kw: None)
    monkeypatch.setattr(
        cli, "_run_remote_capture",
        lambda *a, **kw: SimpleNamespace(
            returncode=255, stdout="",
            stderr="kex_exchange_identification: Connection closed",
        ),
    )

    result = runner.invoke(
        cli.app,
        ["--config", str(config_path), "-T", "fake-slurm", "release", "sample", "-f"],
    )
    assert result.exit_code != 0
    combined = (
        result.stdout + (result.output or "") + str(result.stderr_bytes or b"")
    )
    assert "scancel returned 255" in combined
    # Session NOT cleared on failure.
    reloaded = session_mod.RemoteSession.load("sample", target_name="fake-slurm")
    assert reloaded.slurm_job_id == 7654321


def test_reconcile_login_node_adopts_warm_tunnel(tmp_path, monkeypatch):
    """A SLURM mirror session stuck on a stale login node adopts the warm
    tunnel session's node -- and persists it for the next command."""
    from sucoder import session as session_mod

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr(session_mod, "_session_dir", lambda: sessions_dir)

    # Warm tunnel session pinned to ln003; mirror session stuck on ln002.
    (sessions_dir / "tunnel-fake-slurm.yaml").write_text(
        "login_node: ln003\n", encoding="utf-8",
    )
    session = session_mod.RemoteSession(
        mirror_name="sample", target_name="fake-slurm", login_node="ln002",
    )
    remote = SimpleNamespace(slurm=SimpleNamespace())  # SLURM-backed
    logger = SimpleNamespace(info=lambda *a, **k: None)

    changed = cli._reconcile_login_node(remote, session, "fake-slurm", logger)
    assert changed is True
    assert session.login_node == "ln003"
    reloaded = session_mod.RemoteSession.load("sample", target_name="fake-slurm")
    assert reloaded.login_node == "ln003"


def test_reconcile_login_node_noop_for_non_slurm(tmp_path, monkeypatch):
    """Non-SLURM sessions keep their pin -- the agent tmux lives ON the
    login node, so it is not a swappable routing hop."""
    from sucoder import session as session_mod

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr(session_mod, "_session_dir", lambda: sessions_dir)
    (sessions_dir / "tunnel-fake-slurm.yaml").write_text(
        "login_node: ln003\n", encoding="utf-8",
    )
    session = session_mod.RemoteSession(
        mirror_name="sample", target_name="fake-slurm", login_node="ln002",
    )
    remote = SimpleNamespace(slurm=None)
    logger = SimpleNamespace(info=lambda *a, **k: None)

    changed = cli._reconcile_login_node(remote, session, "fake-slurm", logger)
    assert changed is False
    assert session.login_node == "ln002"


def test_reconcile_login_node_noop_without_tunnel_pin(tmp_path, monkeypatch):
    """No warm tunnel node recorded -> nothing to adopt, pin unchanged."""
    from sucoder import session as session_mod

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr(session_mod, "_session_dir", lambda: sessions_dir)
    session = session_mod.RemoteSession(
        mirror_name="sample", target_name="fake-slurm", login_node="ln002",
    )
    remote = SimpleNamespace(slurm=SimpleNamespace())
    logger = SimpleNamespace(info=lambda *a, **k: None)

    changed = cli._reconcile_login_node(remote, session, "fake-slurm", logger)
    assert changed is False
    assert session.login_node == "ln002"


def test_login_node_via_gateway(monkeypatch):
    """Probe returns the gateway mux's backend node, or '' on failure."""
    gw = SimpleNamespace(ssh_options=lambda **kw: [])
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="ln005.brc\n", stderr=""),
    )
    assert cli._login_node_via_gateway(gw, "gw.example.org") == "ln005.brc"
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=255, stdout="", stderr="boom"),
    )
    assert cli._login_node_via_gateway(gw, "gw.example.org") == ""


def _cert_config(tmp_path: Path, cert_path: Path) -> Path:
    """A SLURM config whose target carries a ``cert_file``."""
    human = os.environ.get("USER", "coder")
    mirror_root = tmp_path / "mirrors"
    mirror_root.mkdir(exist_ok=True)
    canonical = tmp_path / "canonical"
    canonical.mkdir(exist_ok=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
human_user: {human}
agent_user: {human}
agent_group: {human}
mirror_root: {mirror_root}
mirrors:
  sample:
    canonical_repo: {canonical}
    mirror_name: sample
    branch_prefixes:
      human: {human}
      agent: {human}
targets:
  fake-slurm:
    gateway: gw.example.org
    transfer_host: dtn.example.org
    cert_file: {cert_path}
    slurm:
      partition: test
      account: test_acct
""",
        encoding="utf-8",
    )
    return config_path


def test_cert_command_mints(tmp_path, monkeypatch):
    """`sucoder -T <t> cert` POSTs to the CA (mocked) and writes the cert."""
    from sucoder import cert as cert_mod

    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("BRC_USER", "ligon")
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    cert_path = fake_home / ".ssh" / "ssh_certs" / "brc_cert"
    config_path = _cert_config(tmp_path, cert_path)

    # Fake the CA call (write_cert still runs for real); avoid ssh-keygen and
    # the interactive prompts.
    monkeypatch.setattr(
        cert_mod, "request_cert",
        lambda *a, **k: {
            "key_id": "kid123", "private_key": "PRIV", "public_key": "PUB",
            "signed_public_key": "SIGNED", "expires_at": "2026",
        },
    )
    monkeypatch.setattr(cli, "_cert_status", lambda cf: ("✓", "cert valid to 2026"))
    answers = iter(["1234", "567890"])
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: next(answers))

    result = runner.invoke(
        cli.app, ["--config", str(config_path), "-T", "fake-slurm", "cert"],
    )
    assert result.exit_code == 0, (result.stdout, result.exception)
    assert "Minted for ligon" in result.stdout
    assert "kid123" in result.stdout
    assert list(tmp_path.rglob("brc_cert-cert.pub")), "signed cert not written"


def test_cert_command_requires_cert_file(tmp_path, monkeypatch):
    """No `cert_file` on the target -> clear error, exit 2, no prompt."""
    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    config_path = _slurm_config(tmp_path)  # fake-slurm, no cert_file
    result = runner.invoke(
        cli.app, ["--config", str(config_path), "-T", "fake-slurm", "cert"],
    )
    assert result.exit_code == 2
    combined = (
        result.stdout + (result.output or "") + str(result.stderr_bytes or b"")
    )
    assert "cert_file" in combined


def _fake_control(**kw):
    kw.setdefault("jump_host", None)
    kw.setdefault("cert_file", "/x/brc_cert")
    kw.setdefault("is_active", lambda: False)
    return SimpleNamespace(**kw)


def _tty(is_tty):
    return SimpleNamespace(
        stdin=SimpleNamespace(isatty=lambda: is_tty),
        stdout=SimpleNamespace(isatty=lambda: is_tty),
    )


def test_maybe_offer_cert_mint_mints_when_stale(monkeypatch):
    """Gateway hop + cold mux + TTY + stale cert + confirm -> mint fires."""
    from sucoder import cert as cert_mod

    monkeypatch.setenv("BRC_USER", "ligon")
    monkeypatch.setattr(cli, "sys", _tty(True))
    monkeypatch.setattr(cli, "_cert_status", lambda cf: ("⚠", "cert EXPIRED (x)"))
    monkeypatch.setattr(cli.typer, "confirm", lambda *a, **k: True)
    answers = iter(["1234", "567890"])
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: next(answers))
    monkeypatch.setattr(cli.typer, "echo", lambda *a, **k: None)

    minted = {}
    def fake_mint(cert_file, ca_url, username, pin, otp, lifetime):
        minted.update(
            cert_file=cert_file, username=username, pin=pin, otp=otp, lifetime=lifetime,
        )
        return {"key_id": "k"}
    monkeypatch.setattr(cert_mod, "mint", fake_mint)

    cli._maybe_offer_cert_mint(_fake_control(), logger=SimpleNamespace(info=lambda *a, **k: None))
    assert minted == {
        "cert_file": "/x/brc_cert", "username": "ligon",
        "pin": "1234", "otp": "567890", "lifetime": cert_mod.DEFAULT_LIFETIME,
    }


@pytest.mark.parametrize("control_kw, tty, status, confirm", [
    ({"jump_host": "gw.example.org"}, True, ("⚠", "x"), True),   # not the gateway hop
    ({"cert_file": None}, True, ("⚠", "x"), True),               # no cert configured
    ({}, False, ("⚠", "x"), True),                               # not a TTY
    ({}, True, ("✓", "valid"), True),                            # cert still valid
    ({}, True, ("⚠", "x"), False),                               # user declines
    ({"is_active": lambda: True}, True, ("⚠", "x"), True),       # warm mux
])
def test_maybe_offer_cert_mint_skips(monkeypatch, control_kw, tty, status, confirm):
    from sucoder import cert as cert_mod

    monkeypatch.setattr(cli, "sys", _tty(tty))
    monkeypatch.setattr(cli, "_cert_status", lambda cf: status)
    monkeypatch.setattr(cli.typer, "confirm", lambda *a, **k: confirm)
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: "x")
    monkeypatch.setattr(cli.typer, "echo", lambda *a, **k: None)

    called = {"mint": False}
    monkeypatch.setattr(
        cert_mod, "mint",
        lambda *a, **k: called.__setitem__("mint", True) or {"key_id": "k"},
    )
    cli._maybe_offer_cert_mint(_fake_control(**control_kw), logger=SimpleNamespace(info=lambda *a, **k: None))
    assert called["mint"] is False


def test_attach_refuses_silent_login_node_fallback(tmp_path, monkeypatch):
    """Regression: `sucoder attach` on a SLURM target without a
    recorded SLURM job must NOT silently drop the user onto the login
    node — that masks the underlying problem (allocation died, or the
    session was never set up properly).  It must exit with a clear
    'run sucoder collaborate' message.
    """
    from sucoder import session as session_mod

    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    config_path = _slurm_config(tmp_path)

    # Write a session that records a login_node but NO slurm_job_id —
    # the regression target.  Pre-fix, attach would fall through to
    # `ssh -t -J gw ln_node 'tmux attach || tmux new-session'`,
    # leaving the user in a fresh shell on the login node.
    sessions_dir = fake_home / ".sucoder" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "sample--fake-slurm.yaml").write_text(
        "login_node: ln001\nslurm_job_id: null\ncompute_node: null\n",
        encoding="utf-8",
    )

    # Belt-and-suspenders: also patch _session_dir in case HOME isn't
    # honored by some path normalization.
    monkeypatch.setattr(session_mod, "_session_dir", lambda: sessions_dir)

    # Make sure ensure_ssh_visible / SshControl don't try to actually
    # SSH anywhere.  We expect attach to bail out BEFORE the SSH
    # exec, so this is just a safety net.
    def _no_real_ssh(*a, **kw):
        raise AssertionError("attach should not reach exec/SSH path")
    monkeypatch.setattr(os, "execvp", _no_real_ssh)

    result = runner.invoke(
        cli.app,
        ["--config", str(config_path), "-T", "fake-slurm", "attach", "sample"],
    )
    assert result.exit_code != 0
    combined = result.stdout + (result.output or "")
    # Should reference SLURM and suggest collaborate.
    assert "slurm" in combined.lower(), combined
    assert "collaborate" in combined.lower(), combined


def test_attach_via_srun_uses_overlap_step(tmp_path, monkeypatch):
    """`sucoder attach --via-srun` should stop at the login node and
    join the allocation with `srun --jobid=<JOB> --overlap --pty`
    rather than SSHing directly to the compute node.  This is the
    recovery path for orphaned sessions and for clusters that block
    direct login -> compute SSH.
    """
    from sucoder import session as session_mod

    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    config_path = _slurm_config(tmp_path)

    # Healthy session: login node, jobid, compute node all recorded.
    sessions_dir = fake_home / ".sucoder" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "sample--fake-slurm.yaml").write_text(
        "login_node: ln001\nslurm_job_id: 1234567\ncompute_node: n0148\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(session_mod, "_session_dir", lambda: sessions_dir)

    # Pretend squeue says the job is still RUNNING.
    def _fake_squeue(cmd, **kw):
        return SimpleNamespace(stdout="RUNNING\n", stderr="", returncode=0)
    monkeypatch.setattr(subprocess, "run", _fake_squeue)

    # Capture the execvp args instead of actually exec'ing ssh.
    captured: dict = {}
    def _fake_execvp(prog, argv):
        captured["prog"] = prog
        captured["argv"] = list(argv)
        raise SystemExit(0)  # halt the command cleanly
    monkeypatch.setattr(os, "execvp", _fake_execvp)

    result = runner.invoke(
        cli.app,
        [
            "--config", str(config_path),
            "-T", "fake-slurm",
            "attach", "sample", "--via-srun",
        ],
    )
    # SystemExit(0) from our fake_execvp bubbles up as exit_code 0.
    assert result.exit_code == 0, (result.stdout, result.exception)

    argv = captured["argv"]
    # Single hop via gateway to the login node — NOT a two-hop jump to
    # the compute node.
    assert "-J" in argv
    jump = argv[argv.index("-J") + 1]
    assert jump == "gw.example.org", argv
    # Target host is the login node, not the compute node.
    assert "ln001" in argv, argv
    assert not any("n0148" in part for part in argv), argv
    # The remote command must include `srun --jobid=1234567 --overlap --pty`
    # in front of tmux attach.
    remote_cmd = argv[-1]
    assert "srun --jobid=1234567 --overlap --pty" in remote_cmd, remote_cmd
    assert "tmux attach-session -t sucoder-sample" in remote_cmd, remote_cmd


def _confined_config(tmp_path: Path) -> Path:
    """Write a config with a ``confined`` SLURM target named fake-confined."""
    human = os.environ.get("USER", "coder")
    mirror_root = tmp_path / "mirrors"
    mirror_root.mkdir(exist_ok=True)
    canonical_repo = tmp_path / "canonical"
    canonical_repo.mkdir(exist_ok=True)
    config_content = f"""
human_user: {human}
agent_user: {human}
agent_group: {human}
mirror_root: {mirror_root}
mirrors:
  sample:
    canonical_repo: {canonical_repo}
    mirror_name: sample
    branch_prefixes:
      human: {human}
      agent: {human}
targets:
  fake-confined:
    gateway: gw.example.org
    transfer_host: dtn.example.org
    mirror_root: ~/mirrors
    slurm:
      partition: savio4_htc
      account: co_carleton
      qos: carleton_htc4_normal
      cpus_per_task: 4
      mem: 16G
      confined: true
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content, encoding="utf-8")
    return config_path


def test_attach_confined_uses_srun_overlap_dedicated_socket(tmp_path, monkeypatch):
    """`attach` on a confined target must join via `srun --overlap` on the
    dedicated `-L` socket (so it lands INSIDE the job cgroup) and must NOT
    carry the `|| tmux new-session` fallback (which would spawn an
    unconfined orphan).  via-srun is auto-selected -- the user need not pass
    it."""
    from sucoder import session as session_mod

    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    config_path = _confined_config(tmp_path)

    sessions_dir = fake_home / ".sucoder" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "sample--fake-confined.yaml").write_text(
        "login_node: ln001\nslurm_job_id: 1234567\ncompute_node: n0148\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(session_mod, "_session_dir", lambda: sessions_dir)

    def _fake_squeue(cmd, **kw):
        return SimpleNamespace(stdout="RUNNING\n", stderr="", returncode=0)
    monkeypatch.setattr(subprocess, "run", _fake_squeue)

    captured: dict = {}
    def _fake_execvp(prog, argv):
        captured["argv"] = list(argv)
        raise SystemExit(0)
    monkeypatch.setattr(os, "execvp", _fake_execvp)

    # NOTE: no --via-srun; confined must force it.
    result = runner.invoke(
        cli.app,
        ["--config", str(config_path), "-T", "fake-confined", "attach", "sample"],
    )
    assert result.exit_code == 0, (result.stdout, result.exception)

    argv = captured["argv"]
    # Single hop via gateway to the login node (srun routes by jobid).
    assert argv[argv.index("-J") + 1] == "gw.example.org", argv
    assert "ln001" in argv and not any("n0148" in p for p in argv), argv
    remote_cmd = argv[-1]
    assert "srun --jobid=1234567 --overlap --pty" in remote_cmd, remote_cmd
    # Dedicated socket + sanitized session name; attach-session only.
    assert "tmux -L sucoder-sample attach-session -t sucoder-sample" in remote_cmd, remote_cmd
    # NO new-session fallback for confined.
    assert "new-session" not in remote_cmd, remote_cmd


def test_attach_unconfined_keeps_new_session_fallback(tmp_path, monkeypatch):
    """Regression: the confined-only attach branch must NOT perturb the
    unconfined attach command -- it still carries the `|| tmux new-session`
    fallback and uses the default socket (no `-L`)."""
    from sucoder import session as session_mod

    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    config_path = _slurm_config(tmp_path)
    sessions_dir = fake_home / ".sucoder" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "sample--fake-slurm.yaml").write_text(
        "login_node: ln001\nslurm_job_id: 1234567\ncompute_node: n0148\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(session_mod, "_session_dir", lambda: sessions_dir)

    def _fake_squeue(cmd, **kw):
        return SimpleNamespace(stdout="RUNNING\n", stderr="", returncode=0)
    monkeypatch.setattr(subprocess, "run", _fake_squeue)

    captured: dict = {}
    def _fake_execvp(prog, argv):
        captured["argv"] = list(argv)
        raise SystemExit(0)
    monkeypatch.setattr(os, "execvp", _fake_execvp)

    result = runner.invoke(
        cli.app,
        ["--config", str(config_path), "-T", "fake-slurm", "attach", "sample", "--via-srun"],
    )
    assert result.exit_code == 0, (result.stdout, result.exception)
    remote_cmd = captured["argv"][-1]
    # Pin the EXACT unconfined command (byte-identity): the srun --overlap
    # prefix must be applied to BOTH the attach AND the new-session fallback
    # (so the fallback runs inside the allocation, not as a login-node
    # orphan -- the field bug this form fixes).  A substring check would miss
    # a dropped prefix on the fallback.
    expected = (
        "srun --jobid=1234567 --overlap --pty tmux attach-session -t sucoder-sample "
        "|| srun --jobid=1234567 --overlap --pty tmux new-session -s sucoder-sample"
    )
    assert remote_cmd == expected, remote_cmd
    assert "-L sucoder-sample" not in remote_cmd, remote_cmd


def test_attach_via_srun_rejects_non_slurm_target(tmp_path, monkeypatch):
    """`--via-srun` only makes sense for SLURM targets — refuse it on
    a plain remote target so the user doesn't think it did something
    silent."""
    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    human = os.environ.get("USER", "coder")
    mirror_root = tmp_path / "mirrors"
    mirror_root.mkdir(exist_ok=True)
    canonical_repo = tmp_path / "canonical"
    canonical_repo.mkdir(exist_ok=True)

    # Remote target WITHOUT a slurm: stanza.
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
human_user: {human}
agent_user: {human}
agent_group: {human}
mirror_root: {mirror_root}
mirrors:
  sample:
    canonical_repo: {canonical_repo}
    mirror_name: sample
    branch_prefixes:
      human: {human}
      agent: {human}
targets:
  plain-remote:
    gateway: gw.example.org
    transfer_host: dtn.example.org
""",
        encoding="utf-8",
    )

    # Need a session file so we get past the "no session" check and
    # reach the --via-srun validation.
    sessions_dir = fake_home / ".sucoder" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "sample--plain-remote.yaml").write_text(
        "login_node: ln001\nslurm_job_id: null\ncompute_node: null\n",
        encoding="utf-8",
    )
    from sucoder import session as session_mod
    monkeypatch.setattr(session_mod, "_session_dir", lambda: sessions_dir)

    def _no_real_ssh(*a, **kw):
        raise AssertionError("attach should not reach exec/SSH path")
    monkeypatch.setattr(os, "execvp", _no_real_ssh)

    result = runner.invoke(
        cli.app,
        [
            "--config", str(config_path),
            "-T", "plain-remote",
            "attach", "sample", "--via-srun",
        ],
    )
    assert result.exit_code != 0
    combined = result.stdout + (result.output or "")
    assert "slurm" in combined.lower(), combined


def test_collaborate_applies_target_overlay(tmp_path, monkeypatch):
    """``sucoder -T <target> collaborate <mirror>`` must overlay the
    target's RemoteConfig onto the mirror settings so the bootstrap
    flow takes the remote branch.

    Regression test: typer >=0.21 stopped pushing its Context onto
    Click's global stack, so ``click.get_current_context()`` raises
    ``RuntimeError`` inside subcommand bodies.  The previous CLI code
    relied on that call to fish ``-T`` out of ``ctx.obj`` -- which
    silently dropped the overlay and routed every ``-T <target>
    collaborate`` invocation through the local executor.  The user-
    visible symptom was ``sucoder -T savio-node collaborate``
    reporting ``Mirror already exists at /home/coder/mirrors/<name>``
    (the LOCAL mirror) instead of clone/sync against the remote.

    The fix threads the typer ``ctx`` through the helper chain as
    ``cli_ctx=``.  This test pins the contract: bootstrap must
    receive ``ctx.is_remote=True`` and ``ctx.settings.remote`` set
    to the resolved target.
    """
    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    human = os.environ.get("USER", "coder")
    mirror_root = tmp_path / "mirrors"
    mirror_root.mkdir(exist_ok=True)
    canonical_repo = tmp_path / "canonical"
    canonical_repo.mkdir(exist_ok=True)
    # Init the canonical so prepare_canonical doesn't choke -- though
    # we short-circuit before it actually runs.
    subprocess.run(
        ["git", "init", "-b", "main", str(canonical_repo)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(canonical_repo), "config", "user.email", "t@t"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(canonical_repo), "config", "user.name", "t"],
        check=True,
    )
    (canonical_repo / "README.md").write_text("hi\n")
    subprocess.run(
        ["git", "-C", str(canonical_repo), "add", "README.md"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(canonical_repo), "commit", "-m", "init"],
        check=True, capture_output=True,
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
human_user: {human}
agent_user: {human}
agent_group: {human}
mirror_root: {mirror_root}
mirrors:
  sample:
    canonical_repo: {canonical_repo}
    mirror_name: sample
    branch_prefixes:
      human: {human}
      agent: {human}
targets:
  plain-remote:
    gateway: gw.example.org
    transfer_host: dtn.example.org
""",
        encoding="utf-8",
    )

    # Intercept _build_manager_for_mirror after the target overlay has
    # been applied but BEFORE _build_executor establishes an SSH
    # ControlMaster (which would try to reach the fake gateway).  The
    # overlay writes the resolved RemoteConfig back to
    # ``config.mirrors[mirror_name].remote``; that's what we inspect.
    captured: dict = {}
    original_bmfm = cli._build_manager_for_mirror

    def spy_bmfm(config, logger, dry_run, mirror_name, *, cli_ctx=None):
        # Re-run the overlay logic just like the helper would do, then
        # raise before constructing a RemoteExecutor (which would dial
        # the fake gateway).
        settings = config.mirrors.get(mirror_name)
        target = cli._get_active_target(cli_ctx)
        if target is not None and settings is not None:
            from dataclasses import replace
            settings = replace(settings, remote=target)
            config.mirrors[mirror_name] = settings  # type: ignore[index]
        captured["is_remote"] = bool(settings and settings.remote)
        captured["remote_gateway"] = (
            settings.remote.gateway if settings and settings.remote else None
        )
        captured["cli_ctx_obj_target"] = (
            (cli_ctx.obj or {}).get("target") if cli_ctx else None
        )
        raise SystemExit(99)

    monkeypatch.setattr(cli, "_build_manager_for_mirror", spy_bmfm)

    result = runner.invoke(
        cli.app,
        [
            "--config", str(config_path),
            "-T", "plain-remote",
            "collaborate", "sample",
        ],
    )

    # SystemExit(99) bubbles up through the typer command wrapper.
    assert result.exit_code == 99, (result.stdout, result.exception)
    assert captured.get("cli_ctx_obj_target") is not None, (
        "Subcommand failed to forward its typer.Context to "
        f"_build_manager_for_mirror as cli_ctx=; got: {captured}"
    )
    assert captured.get("is_remote") is True, (
        "Expected `-T plain-remote collaborate` to overlay the target's "
        f"RemoteConfig onto mirror settings; got: {captured}"
    )
    assert captured.get("remote_gateway") == "gw.example.org", (
        f"Expected target overlay to apply correctly; got: {captured}"
    )


def test_ensure_slurm_node_persists_job_id_before_node_query(tmp_path, monkeypatch):
    """Regression: a granted SLURM allocation must be recorded BEFORE the
    squeue node-query.

    salloc bills from the moment the job is granted.  The historical code
    only persisted ``slurm_job_id`` *after* resolving the node via squeue,
    so a node-query failure (the original mux-refusal bug) left a
    granted-but-unrecorded 24h allocation that ``release``/``scancel``
    could not find -- a silent compute-budget leak.  This pins that the
    job id is on disk even when the node-query then fails.
    """
    import logging

    import typer

    from sucoder import session as session_mod
    from sucoder.config import RemoteConfig, SlurmConfig

    sessions = tmp_path / "sessions"
    monkeypatch.setattr(session_mod, "_session_dir", lambda: sessions)

    remote = RemoteConfig(
        gateway="gw",
        transfer_host="dtn",
        slurm=SlurmConfig(partition="savio3", account="acct", time="24:00:00"),
    )
    sess = session_mod.RemoteSession(
        mirror_name="Emu-GMM", target_name="savio-node", login_node="ln003.brc",
    )

    class _FakeControl:
        def ssh_options(self, **kw):
            return []

    granted = "salloc: Granted job allocation 34688352\n"

    def fake_run(cmd, *a, **kw):
        joined = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
        if "salloc" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=granted)
        if "squeue --job" in joined:
            # Simulate the node-query failing (e.g. wedged mux).
            raise subprocess.CalledProcessError(
                1, cmd, stderr="Session open refused by peer",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(typer.Exit):
        cli._ensure_slurm_node(
            remote, sess, _FakeControl(), _FakeControl(), logging.getLogger("t"),
        )

    # Despite the node-query failure, the job id must be recoverable.
    reloaded = session_mod.RemoteSession.load("Emu-GMM", target_name="savio-node")
    assert reloaded.slurm_job_id == 34688352
    assert reloaded.compute_node is None


def _slurm_recovery_fixture(tmp_path, monkeypatch, *, squeue_lines):
    """Build a session stuck in the persist-before-query state.

    ``slurm_job_id`` set, ``compute_node`` None -- exactly what
    ``test_ensure_slurm_node_persists_job_id_before_node_query`` pins as
    the on-disk outcome of an interrupted allocation.  *squeue_lines* is a
    list of ``CompletedProcess``-shaped responses for successive ``squeue
    --job`` calls.  Returns ``(remote, session, run_calls)``.
    """
    from sucoder import session as session_mod
    from sucoder.config import RemoteConfig, SlurmConfig

    sessions = tmp_path / "sessions"
    monkeypatch.setattr(session_mod, "_session_dir", lambda: sessions)

    remote = RemoteConfig(
        gateway="gw",
        transfer_host="dtn",
        slurm=SlurmConfig(partition="savio3", account="acct", time="24:00:00"),
    )
    sess = session_mod.RemoteSession(
        mirror_name="LSMS_Library", target_name="savio-node",
        login_node="ln003.brc", slurm_job_id=35141648,
    )
    sess.save()
    assert sess.compute_node is None

    run_calls: list = []
    pending = list(squeue_lines)

    def fake_run(cmd, *a, **kw):
        joined = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
        run_calls.append(joined)
        if "squeue --job" in joined:
            return pending.pop(0)
        if "salloc" in joined:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="",
                stderr="salloc: Granted job allocation 99999999\n",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Neither the SSH connect nor the on-node deadline timer is under test.
    monkeypatch.setattr(cli, "_connect_with_retry", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_start_slurm_timer", lambda *a, **kw: None)
    return remote, sess, run_calls


def _ok(stdout):
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


class _StubControl:
    def ssh_options(self, **kw):
        return []


def test_ensure_slurm_node_recovers_node_for_recorded_job(tmp_path, monkeypatch):
    """Regression: a job recorded WITHOUT a node must resolve, not crash.

    ``salloc`` bills on grant, so the job id is persisted before the node
    is queried; an interrupt in that window leaves ``slurm_job_id`` set and
    ``compute_node`` None.  That state used to satisfy none of the three
    gates in ``_ensure_slurm_node`` -- reuse needs a node, adopt and salloc
    need no job id -- so it fell through to ``SshControl(gateway=None)`` and
    died with ``TypeError: expected str, bytes or os.PathLike object, not
    NoneType`` from inside Popen.  Sticky: nothing cleared it, so every
    later run of that mirror crashed identically.
    """
    import logging

    remote, sess, run_calls = _slurm_recovery_fixture(
        tmp_path, monkeypatch,
        squeue_lines=[_ok("RUNNING n0123.savio3\n"), _ok("RUNNING\n")],
    )

    node, _control = cli._ensure_slurm_node(
        remote, sess, _StubControl(), _StubControl(), logging.getLogger("t"),
    )

    assert node == "n0123.savio3"
    assert not any("salloc" in c for c in run_calls), (
        "The recorded job is alive -- reallocating would LEAK it (a 24h job "
        f"`release` can no longer find).  Calls: {run_calls}"
    )

    from sucoder import session as session_mod
    reloaded = session_mod.RemoteSession.load(
        "LSMS_Library", target_name="savio-node",
    )
    assert reloaded.slurm_job_id == 35141648
    assert reloaded.compute_node == "n0123.savio3"


def test_ensure_slurm_node_reallocates_when_recorded_job_is_gone(
    tmp_path, monkeypatch,
):
    """A recorded job that has LEFT the queue is cleared, then reallocated.

    Empty output from a successful ``squeue --job`` is the one signal we
    accept as "gone" -- so the stale id is dropped and a fresh node
    allocated, rather than wedging the mirror forever.
    """
    import logging

    remote, sess, run_calls = _slurm_recovery_fixture(
        tmp_path, monkeypatch,
        squeue_lines=[
            _ok(""),                      # recovery probe: job is gone
            _ok("n0456.savio3\n"),        # post-salloc node query
        ],
    )

    node, _control = cli._ensure_slurm_node(
        remote, sess, _StubControl(), _StubControl(), logging.getLogger("t"),
    )

    assert node == "n0456.savio3"
    assert any("salloc" in c for c in run_calls)

    from sucoder import session as session_mod
    reloaded = session_mod.RemoteSession.load(
        "LSMS_Library", target_name="savio-node",
    )
    assert reloaded.slurm_job_id == 99999999
    assert reloaded.compute_node == "n0456.savio3"


def test_ensure_slurm_node_refuses_to_reallocate_over_a_pending_job(
    tmp_path, monkeypatch,
):
    """PENDING (state word, empty %N) must never be read as "gone".

    A pending job is LIVE.  Clearing its id to allocate a replacement would
    leak it.  Bail instead, leaving the id on disk for retry/`release`.
    """
    import logging

    import typer

    monkeypatch.setattr(cli.time, "sleep", lambda *_a: None)
    remote, sess, run_calls = _slurm_recovery_fixture(
        tmp_path, monkeypatch,
        squeue_lines=[_ok("PENDING\n")] * 5,
    )

    with pytest.raises(typer.Exit):
        cli._ensure_slurm_node(
            remote, sess, _StubControl(), _StubControl(), logging.getLogger("t"),
        )

    assert not any("salloc" in c for c in run_calls), (
        f"Reallocated over a live PENDING job -- that leaks it.  {run_calls}"
    )
    from sucoder import session as session_mod
    reloaded = session_mod.RemoteSession.load(
        "LSMS_Library", target_name="savio-node",
    )
    assert reloaded.slurm_job_id == 35141648, "The live job id must stay recoverable."


def test_ensure_slurm_node_keeps_job_when_node_query_errors(tmp_path, monkeypatch):
    """An ssh/squeue *failure* is not evidence the job is dead."""
    import logging

    import typer

    monkeypatch.setattr(cli.time, "sleep", lambda *_a: None)
    remote, sess, run_calls = _slurm_recovery_fixture(
        tmp_path, monkeypatch,
        squeue_lines=[
            subprocess.CompletedProcess(
                [], 255, stdout="", stderr="Session open refused by peer",
            ),
        ],
    )

    with pytest.raises(typer.Exit):
        cli._ensure_slurm_node(
            remote, sess, _StubControl(), _StubControl(), logging.getLogger("t"),
        )

    assert not any("salloc" in c for c in run_calls)
    from sucoder import session as session_mod
    reloaded = session_mod.RemoteSession.load(
        "LSMS_Library", target_name="savio-node",
    )
    assert reloaded.slurm_job_id == 35141648


class _FakeSshControl:
    """Stand-in for ``SshControl`` in ``_build_executor`` tests.

    Accepts the full constructor kwarg surface (gateway, persistence
    knobs, jump host/control, extra_options, debug) and provides the few
    attributes/methods ``_build_executor`` reaches for: ``ensure``,
    ``ssh_options``, ``gateway``, and ``socket_path``.
    """

    def __init__(self, *, gateway=None, **kwargs):
        self.gateway = gateway
        self._kwargs = kwargs

    def ensure(self, logger):
        return None

    def ssh_options(self, **kwargs):
        return []

    @property
    def socket_path(self):
        return f"/tmp/sucoder-sock-{self.gateway}"


def _install_build_executor_fakes(monkeypatch, tmp_path, *, login_node="ln001.brc"):
    """Stub the SSH/session/executor layer for a ``_build_executor`` test.

    Pre-seeds a session with ``login_node`` set (so the login-node pin
    subprocess is skipped), fakes ``SshControl`` and ``_ensure_ssh_visible``,
    captures the ``RemoteExecutor`` kwargs, and spies on
    ``_ensure_slurm_node``.  Returns ``(captured, slurm_calls)`` where
    ``captured["kwargs"]`` is the RemoteExecutor kwarg dict.
    """
    from sucoder import session as session_mod
    import sucoder.tunnel as tunnel_mod
    import sucoder.executor as executor_mod

    sessions = tmp_path / "sessions"
    monkeypatch.setattr(session_mod, "_session_dir", lambda: sessions)

    monkeypatch.setattr(tunnel_mod, "SshControl", _FakeSshControl)
    monkeypatch.setattr(cli, "_ensure_ssh_visible", lambda *a, **k: None)

    slurm_calls: list = []

    def spy_ensure_slurm_node(remote, session, ln_control, gw_control, logger, **kw):
        slurm_calls.append(True)
        # A non-confined allocation resolves a compute node and its control.
        session.slurm_job_id = 999
        session.compute_node = "n0001.savio4"
        session.save()
        return "n0001.savio4", _FakeSshControl(gateway="n0001.savio4")

    monkeypatch.setattr(cli, "_ensure_slurm_node", spy_ensure_slurm_node)

    captured: dict = {}

    class _FakeRemoteExecutor:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(executor_mod, "RemoteExecutor", _FakeRemoteExecutor)

    # Seed the session so the login node is already pinned.
    sess = session_mod.RemoteSession(
        mirror_name="sample", target_name=None, login_node=login_node,
    )
    sess.save()

    return captured, slurm_calls


def _confined_mirror_settings(tmp_path, *, confined: bool):
    from sucoder.config import RemoteConfig, SlurmConfig

    return MirrorSettings(
        name="sample",
        canonical_repo=tmp_path / "canonical",
        mirror_name="sample",
        branch_prefixes=BranchPrefixes(human="ligon", agent="coder"),
        remote=RemoteConfig(
            gateway="brc.berkeley.edu",
            transfer_host="dtn.brc.berkeley.edu",
            slurm=SlurmConfig(
                partition="savio4_htc", account="co_carleton", confined=confined,
            ),
        ),
    )


def test_build_executor_confined_skips_salloc(tmp_path, monkeypatch):
    """A ``confined`` target fuses allocate+launch into a later ``sbatch``.

    ``_build_executor`` must therefore NOT call ``_ensure_slurm_node``
    (salloc) and must return a *login-node* executor: ``is_compute_node``
    False, ``login_node`` pointing at the login node (not a compute node).
    """
    import logging

    captured, slurm_calls = _install_build_executor_fakes(monkeypatch, tmp_path)
    config = Config(human_user="coder", mirror_root=tmp_path / "mirrors")
    settings = _confined_mirror_settings(tmp_path, confined=True)

    cli._build_executor(
        config, logging.getLogger("t"), dry_run=False, mirror_settings=settings,
    )

    assert slurm_calls == [], "confined launch must not salloc a compute node"
    kwargs = captured["kwargs"]
    assert kwargs["is_compute_node"] is False
    assert kwargs["login_node"] == "ln001.brc"
    # No compute-node ProxyCommand fallback for a login-node executor.
    assert "proxy_node" not in kwargs
    # Confined runs on NFS, never a compute-node-local mirror root.
    assert kwargs["remote_mirror_root"] == str(settings.remote.mirror_root)


def test_build_executor_unconfined_slurm_allocates(tmp_path, monkeypatch):
    """Control: an unconfined SLURM target still allocates a compute node
    and returns a compute-node executor (the salloc path is unchanged)."""
    import logging

    captured, slurm_calls = _install_build_executor_fakes(monkeypatch, tmp_path)
    config = Config(human_user="coder", mirror_root=tmp_path / "mirrors")
    settings = _confined_mirror_settings(tmp_path, confined=False)

    cli._build_executor(
        config, logging.getLogger("t"), dry_run=False, mirror_settings=settings,
    )

    assert slurm_calls == [True], "unconfined SLURM target must salloc"
    kwargs = captured["kwargs"]
    assert kwargs["is_compute_node"] is True
    assert kwargs["login_node"] == "n0001.savio4"
    assert kwargs["proxy_node"] == "ln001.brc"


# ----------------------------------------------------------------------
# `sucoder nodes` — read-only SLURM node-availability query
# ----------------------------------------------------------------------


def _write_nodes_config(tmp_path: Path) -> Path:
    """Config with a SLURM target (`savio-node`) and a plain target."""
    human = os.environ.get("USER", "coder")
    agent = os.environ.get("USER", "coder")
    mirror_root = tmp_path / "mirrors"
    mirror_root.mkdir(exist_ok=True)
    canonical_repo = tmp_path / "canonical"
    canonical_repo.mkdir(exist_ok=True)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(exist_ok=True)

    config_content = f"""
human_user: {human}
agent_user: {agent}
agent_group: {agent}
mirror_root: {mirror_root}
targets:
  savio-node:
    gateway: hpc.example.edu
    transfer_host: dtn.example.edu
    slurm:
      partition: savio3
      account: fc_test
  plain:
    gateway: gw.example.edu
    transfer_host: dtn.example.edu
mirrors:
  sample:
    canonical_repo: {canonical_repo}
    mirror_name: sample
    branch_prefixes:
      human: {human}
      agent: {agent}
    skills:
      - {skills_dir}
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content, encoding="utf-8")
    return config_path


_SINFO_AVAIL = (
    "NODELIST       STATE      CPUS(A/I/O/T)  CPU_LOAD\n"
    "n0000.savio3   idle             0/32/0/32      0.01\n"
    "n0001.savio3   mix             16/16/0/32      8.20\n"
)
_SINFO_DRAIN = (
    "REASON               USER      TIMESTAMP           NODELIST\n"
    "Lustre client hung   root      2026-06-15T09:12:00 n0123.savio3\n"
)


_SINFO_DRAIN_NONE = "REASON               USER      TIMESTAMP           NODELIST\n"


def _install_nodes_fakes(
    monkeypatch,
    *,
    avail_rc: int = 0,
    avail_stderr: str = "",
    drain_rc: int = 0,
    drain_stdout: str = _SINFO_DRAIN,
    drain_stderr: str = "",
):
    """Stub startup checks, SSH setup, and the remote sinfo runner.

    Returns a list that records each ``(host, command)`` actually sent
    to :func:`cli._run_remote_capture`.
    """
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_ensure_ssh_visible", lambda *a, **k: None)

    calls: list = []

    def fake_run(control, host, command, *, debug=False, timeout=30):
        calls.append((host, command))
        if " -R" in command:  # drain query
            return SimpleNamespace(
                returncode=drain_rc, stdout=drain_stdout, stderr=drain_stderr
            )
        return SimpleNamespace(
            returncode=avail_rc,
            stdout="" if avail_rc else _SINFO_AVAIL,
            stderr=avail_stderr,
        )

    monkeypatch.setattr(cli, "_run_remote_capture", fake_run)
    return calls


def test_nodes_defaults_partition_from_target(tmp_path, monkeypatch):
    runner = CliRunner()
    calls = _install_nodes_fakes(monkeypatch)
    cfg = _write_nodes_config(tmp_path)

    result = runner.invoke(
        cli.app, ["--config", str(cfg), "-T", "savio-node", "nodes"]
    )

    assert result.exit_code == 0, result.output
    # Partition defaulted from the target's slurm.partition, and the
    # exact columnar format is what produces the documented output.
    avail_cmd = calls[0][1]
    assert "-p savio3 " in avail_cmd and "-N" in avail_cmd
    assert '-o "%N %6t %.15C %.6O"' in avail_cmd
    assert any(" -R" in cmd for _, cmd in calls)
    assert "n0000.savio3" in result.output
    assert "n0123.savio3" in result.output  # drain section


def test_nodes_positional_overrides_partition(tmp_path, monkeypatch):
    runner = CliRunner()
    calls = _install_nodes_fakes(monkeypatch)
    cfg = _write_nodes_config(tmp_path)

    result = runner.invoke(
        cli.app,
        ["--config", str(cfg), "-T", "savio-node", "nodes", "savio3_gpu"],
    )

    assert result.exit_code == 0, result.output
    # Positional overrides the default; the target's `savio3` is unused.
    assert all("-p savio3_gpu" in cmd for _, cmd in calls)
    assert all("-p savio3 " not in cmd for _, cmd in calls)


def test_nodes_requires_target(tmp_path, monkeypatch):
    runner = CliRunner()
    _install_nodes_fakes(monkeypatch)
    cfg = _write_nodes_config(tmp_path)

    result = runner.invoke(cli.app, ["--config", str(cfg), "nodes"])

    assert result.exit_code == 2  # usage error, not a runtime failure
    assert "remote target" in result.output.lower()


def test_nodes_requires_partition_without_slurm(tmp_path, monkeypatch):
    runner = CliRunner()
    _install_nodes_fakes(monkeypatch)
    cfg = _write_nodes_config(tmp_path)

    result = runner.invoke(
        cli.app, ["--config", str(cfg), "-T", "plain", "nodes"]
    )

    assert result.exit_code == 2  # usage error, not a runtime failure
    assert "partition" in result.output.lower()


def test_nodes_surfaces_sinfo_failure(tmp_path, monkeypatch):
    runner = CliRunner()
    _install_nodes_fakes(
        monkeypatch, avail_rc=1, avail_stderr="Invalid partition name specified"
    )
    cfg = _write_nodes_config(tmp_path)

    result = runner.invoke(
        cli.app, ["--config", str(cfg), "-T", "savio-node", "nodes", "bogus"]
    )

    assert result.exit_code == 1
    assert "Invalid partition name specified" in result.output


def test_partition_re_accepts_and_rejects():
    accept = ["savio3", "savio4_htc", "savio3_gpu", "savio2_bigmem",
              "savio3,savio4_htc", "a", "P1.2"]
    reject = ["", "-N", "--help", "a b", "a;b", "a|b", "a&b", "$(x)",
              "`x`", "a'b", 'a"b', "savio3\n", "\nsavio3", ",savio3", ".savio3"]
    for p in accept:
        assert cli._PARTITION_RE.match(p), f"should accept {p!r}"
    for p in reject:
        assert not cli._PARTITION_RE.match(p), f"should reject {p!r}"


def test_nodes_reports_none_when_no_drained_nodes(tmp_path, monkeypatch):
    """`sinfo -R` header-only output must read as 'none', not a bare header."""
    runner = CliRunner()
    _install_nodes_fakes(monkeypatch, drain_stdout=_SINFO_DRAIN_NONE)
    cfg = _write_nodes_config(tmp_path)

    result = runner.invoke(
        cli.app, ["--config", str(cfg), "-T", "savio-node", "nodes"]
    )

    assert result.exit_code == 0, result.output
    assert "(none reported)" in result.output
    assert "n0123.savio3" not in result.output


def test_nodes_handles_drain_query_failure(tmp_path, monkeypatch):
    """A failed drain query must not masquerade as a healthy partition."""
    runner = CliRunner()
    _install_nodes_fakes(
        monkeypatch, drain_rc=1, drain_stdout="", drain_stderr="sinfo: error"
    )
    cfg = _write_nodes_config(tmp_path)

    result = runner.invoke(
        cli.app, ["--config", str(cfg), "-T", "savio-node", "nodes"]
    )

    # Availability succeeded, so the command still exits 0...
    assert result.exit_code == 0, result.output
    # ...but the drain failure is surfaced, not swallowed as "(none reported)".
    assert "could not query drain reasons" in result.output
    assert "(none reported)" not in result.output


def test_nodes_rejects_partition_with_metacharacters(tmp_path, monkeypatch):
    """A partition carrying shell metacharacters is rejected before any ssh."""
    runner = CliRunner()
    calls = _install_nodes_fakes(monkeypatch)
    cfg = _write_nodes_config(tmp_path)

    result = runner.invoke(
        cli.app, ["--config", str(cfg), "-T", "savio-node", "nodes", "a;rm -rf ~"]
    )

    assert result.exit_code == 2
    assert "invalid partition" in result.output.lower()
    assert calls == []  # never reached the remote


def test_nodes_rejects_option_like_partition(tmp_path, monkeypatch):
    """A `-`-prefixed value can't slip through as an sinfo flag."""
    runner = CliRunner()
    calls = _install_nodes_fakes(monkeypatch)
    cfg = _write_nodes_config(tmp_path)

    # `--` stops Typer option parsing so the value reaches the command.
    result = runner.invoke(
        cli.app, ["--config", str(cfg), "-T", "savio-node", "nodes", "--", "-N"]
    )

    assert result.exit_code == 2
    assert "invalid partition" in result.output.lower()
    assert calls == []


def test_nodes_stdout_carries_data_stderr_carries_caveat(tmp_path, monkeypatch):
    """Piping hygiene: sinfo rows go to stdout, the Lustre caveat to stderr."""
    import inspect
    from click.testing import CliRunner as ClickRunner
    from typer.main import get_command

    _install_nodes_fakes(monkeypatch)
    cfg = _write_nodes_config(tmp_path)

    # click <8.2 needs mix_stderr=False to split streams; >=8.2 splits
    # unconditionally and dropped the kwarg.
    if "mix_stderr" in inspect.signature(ClickRunner.__init__).parameters:
        runner = ClickRunner(mix_stderr=False)
    else:  # pragma: no cover - depends on installed click
        runner = ClickRunner()
    result = runner.invoke(
        get_command(cli.app),
        ["--config", str(cfg), "-T", "savio-node", "nodes"],
    )

    assert result.exit_code == 0, result.stderr
    assert "n0000.savio3" in result.stdout          # data on stdout
    assert "Lustre health" in result.stderr         # caveat on stderr
    assert "Lustre health" not in result.stdout     # ...and not polluting stdout


def test_run_remote_capture_builds_batchmode_command(monkeypatch):
    """The query reuses the mux (ControlMaster=auto) under BatchMode=yes."""

    class _Ctl:
        def ssh_options(self, **kwargs):
            return ["-o", "ControlMaster=auto", "-o", "ControlPath=/tmp/x.sock"]

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    out = cli._run_remote_capture(_Ctl(), "host.example", "sinfo -p p -R")

    cmd = captured["cmd"]
    assert cmd[0] == "ssh"
    assert "BatchMode=yes" in cmd
    assert "ControlMaster=auto" in cmd
    assert cmd[-2:] == ["host.example", "sinfo -p p -R"]
    assert captured["kwargs"].get("capture_output") is True
    assert captured["kwargs"].get("check") is False
    assert captured["kwargs"].get("timeout")  # bounded, never unbounded
    assert out.stdout == "ok"


def test_run_remote_capture_timeout_returns_124(monkeypatch):
    """A wedged tunnel is bounded and surfaced as a synthetic failure."""

    class _Ctl:
        def ssh_options(self, **kwargs):
            return ["-o", "ControlMaster=auto"]

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1))

    monkeypatch.setattr(subprocess, "run", fake_run)

    out = cli._run_remote_capture(_Ctl(), "host", "sinfo -p p -R", timeout=2)

    assert out.returncode == 124
    assert "timed out" in out.stderr


def test_collaborate_command_error_prints_clean_message(tmp_path, monkeypatch):
    """A CommandError escaping bootstrap exits 1 with a clean message.

    Field failure: a remote `git push` died with `remote unpack failed`
    and the raw CommandError surfaced as a full Python traceback.  The
    collaborate command must render it as an error message (including
    the tail of the failing command's stderr) instead.
    """
    from sucoder.executor import CommandError, CommandResult

    runner = CliRunner()
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)

    human = os.environ.get("USER", "coder")
    mirror_root = tmp_path / "mirrors"
    mirror_root.mkdir(exist_ok=True)
    canonical_repo = tmp_path / "canonical"
    canonical_repo.mkdir(exist_ok=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
human_user: {human}
agent_user: {human}
agent_group: {human}
mirror_root: {mirror_root}
mirrors:
  sample:
    canonical_repo: {canonical_repo}
    mirror_name: sample
    branch_prefixes:
      human: {human}
      agent: {human}
""",
        encoding="utf-8",
    )

    class StubManager:
        def context_for(self, name):
            return SimpleNamespace(name=name)

        def bootstrap(self, *args, **kwargs):
            raise CommandError(
                "Command failed with exit code 1: git push ln000:… --all --force",
                CommandResult(
                    ["git", "push"], ["git", "push"], "",
                    "remote: fatal: write error: Input/output error\n"
                    "error: remote unpack failed: index-pack abnormal exit\n",
                    1,
                ),
            )

    monkeypatch.setattr(
        cli, "_build_manager_for_mirror", lambda *a, **kw: StubManager(),
    )

    result = runner.invoke(
        cli.app,
        ["--config", str(config_path), "collaborate", "sample"],
    )

    assert result.exit_code == 1, (result.output, result.exception)
    # The handler converted the CommandError into a clean exit — the
    # exception reaching the runner is SystemExit, not CommandError.
    assert not isinstance(result.exception, CommandError), result.exception
    combined = result.stdout + (result.output or "")
    assert "Command failed with exit code 1" in combined
    assert "Input/output error" in combined
