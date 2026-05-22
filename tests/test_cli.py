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
