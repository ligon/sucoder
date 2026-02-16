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
