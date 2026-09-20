from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("typer")

import typer
from typer.testing import CliRunner

from sucoder import cli
from sucoder.config import BranchPrefixes, Config, MirrorSettings


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_BOX_DRAWING = re.compile(r"[\u2500-\u257f]")


def _plain_output(result) -> str:
    """CLI output with styling, panel borders, and line wraps removed.

    typer renders ``BadParameter`` through a rich panel and forces terminal
    mode when ``GITHUB_ACTIONS`` (or ``FORCE_COLOR``) is set, so under CI the
    message is coloured and wrapped at 80 columns even though no terminal is
    attached.  A substring assertion on the raw output therefore depends on
    the console width; assert against this normalised text instead.
    """
    text = _ANSI_ESCAPE.sub("", result.output)
    text = _BOX_DRAWING.sub(" ", text)
    return " ".join(text.split())

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


def test_nested_list_mirrors_preserves_flat_command(tmp_path, monkeypatch):
    runner = CliRunner()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    config_path = _write_config(tmp_path, skills_entry=skills_dir)
    monkeypatch.setattr(cli, "run_startup_checks", lambda *args, **kwargs: None)

    flat = runner.invoke(cli.app, ["--config", str(config_path), "mirrors-list"])
    nested = runner.invoke(cli.app, ["--config", str(config_path), "list", "mirrors"])

    assert flat.exit_code == nested.exit_code == 0
    assert nested.stdout == flat.stdout


def test_list_harnesses_shows_builtins_and_configured_default(tmp_path, monkeypatch):
    runner = CliRunner()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    config_path = _write_config(tmp_path, skills_entry=skills_dir)
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "    agent_launcher:\n"
            "      command: [aider]\n"
            "      model: openrouter/deepseek/deepseek-chat\n"
        )
    monkeypatch.setattr(cli, "run_startup_checks", lambda *args, **kwargs: None)

    result = runner.invoke(
        cli.app, ["--config", str(config_path), "list", "harnesses"],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    for harness in ("aider", "claude", "codex", "gemini", "goose", "kimi", "opencode"):
        assert harness in result.stdout
    for capability in (
        "Shell", "Files", "Skills", "MCP", "Subagents", "Providers", "Approval",
    ):
        assert capability in result.stdout
    aider_row = next(line for line in result.stdout.splitlines() if line.startswith("aider"))
    assert "suggest" in aider_row
    assert "explicit" in aider_row
    opencode_row = next(
        line for line in result.stdout.splitlines() if line.startswith("opencode")
    )
    assert "multi" in opencode_row
    assert "auto" in opencode_row
    kimi_row = next(line for line in result.stdout.splitlines() if line.startswith("kimi"))
    assert "yes" in kimi_row
    assert "auto" in kimi_row
    assert "sample" in result.stdout
    assert "openrouter/deepseek/deepseek-chat" in result.stdout


def test_list_models_shows_configured_defaults(tmp_path, monkeypatch):
    runner = CliRunner()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    config_path = _write_config(tmp_path, skills_entry=skills_dir)
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "    agent_launcher:\n"
            "      command: [aider]\n"
            "      model: anthropic/claude-sonnet-4\n"
        )
    monkeypatch.setattr(cli, "run_startup_checks", lambda *args, **kwargs: None)

    result = runner.invoke(
        cli.app, ["--config", str(config_path), "list", "models"],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    assert "sample" in result.stdout
    assert "aider" in result.stdout
    assert "anthropic/claude-sonnet-4" in result.stdout


def test_list_models_queries_sole_configured_provider(tmp_path, monkeypatch):
    from sucoder.executor import CommandResult

    runner = CliRunner()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    config_path = _write_config(tmp_path, skills_entry=skills_dir)
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "credentials:\n"
            "  openrouter:\n"
            "    pass: openrouter.ai/apikey\n"
        )
    monkeypatch.setattr(cli, "run_startup_checks", lambda *args, **kwargs: None)
    human_calls = []

    class StubExecutor:
        def run_human(self, args, **kwargs):
            human_calls.append((list(args), kwargs))
            return CommandResult(list(args), list(args), "test-secret\nmetadata\n", "", 0)

    class StubResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({
                "data": [
                    {"id": "moonshotai/kimi-k3", "name": "Kimi K3"},
                    {"id": "openai/gpt-5", "name": "GPT-5"},
                ]
            }).encode()

    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return StubResponse()

    monkeypatch.setattr(cli, "_build_executor", lambda *args, **kwargs: StubExecutor())
    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)

    result = runner.invoke(
        cli.app,
        ["--config", str(config_path), "list", "models", "kimi"],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    assert human_calls == [
        (
            ["pass", "show", "openrouter.ai/apikey"],
            {"check": False, "capture_output": True},
        )
    ]
    assert requests[0][0].full_url.endswith("/models?q=kimi")
    assert requests[0][0].get_header("Authorization") == "Bearer test-secret"
    assert requests[0][1] == 30
    assert result.stdout.strip() == "openrouter/moonshotai/kimi-k3"
    assert "test-secret" not in result.stdout


def test_list_models_rejects_harness_and_provider_together(tmp_path, monkeypatch):
    runner = CliRunner()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    config_path = _write_config(tmp_path, skills_entry=skills_dir)
    monkeypatch.setattr(cli, "run_startup_checks", lambda *args, **kwargs: None)

    result = runner.invoke(
        cli.app,
        [
            "--config", str(config_path), "list", "models",
            "--harness", "aider", "--provider", "openrouter",
        ],
    )

    assert result.exit_code == 2
    assert "either --harness or --provider" in result.output


def test_list_models_delegates_available_catalog_to_aider(tmp_path, monkeypatch):
    from sucoder.executor import CommandResult

    runner = CliRunner()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    config_path = _write_config(tmp_path, skills_entry=skills_dir)
    monkeypatch.setattr(cli, "run_startup_checks", lambda *args, **kwargs: None)
    calls = []

    class StubExecutor:
        def run_agent(self, args, **kwargs):
            calls.append((list(args), kwargs))
            return CommandResult(list(args), list(args), "gpt-5\ngpt-5-mini\n", "", 0)

    monkeypatch.setattr(cli, "_build_executor", lambda *args, **kwargs: StubExecutor())

    result = runner.invoke(
        cli.app,
        [
            "--config", str(config_path), "list", "models",
            "--harness", "aider", "gpt",
        ],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == [(["aider", "--list-models", "gpt"], {"check": False, "capture_output": True})]
    assert "gpt-5-mini" in result.stdout


def test_list_models_filters_opencode_catalog(tmp_path, monkeypatch):
    from sucoder.executor import CommandResult

    runner = CliRunner()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    config_path = _write_config(tmp_path, skills_entry=skills_dir)
    monkeypatch.setattr(cli, "run_startup_checks", lambda *args, **kwargs: None)
    calls = []

    class StubExecutor:
        def run_agent(self, args, **kwargs):
            calls.append((list(args), kwargs))
            return CommandResult(
                list(args), list(args),
                "openrouter/moonshotai/kimi-k3\nopenai/gpt-5\n", "", 0,
            )

    monkeypatch.setattr(cli, "_build_executor", lambda *args, **kwargs: StubExecutor())

    result = runner.invoke(
        cli.app,
        [
            "--config", str(config_path), "list", "models",
            "--harness", "opencode", "kimi",
        ],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == [(["opencode", "models"], {"check": False, "capture_output": True})]
    assert "openrouter/moonshotai/kimi-k3" in result.stdout
    assert "openai/gpt-5" not in result.stdout


def test_list_models_filters_kimi_configured_aliases(tmp_path, monkeypatch):
    from sucoder.executor import CommandResult

    runner = CliRunner()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    config_path = _write_config(tmp_path, skills_entry=skills_dir)
    monkeypatch.setattr(cli, "run_startup_checks", lambda *args, **kwargs: None)
    calls = []

    class StubExecutor:
        def run_agent(self, args, **kwargs):
            calls.append((list(args), kwargs))
            return CommandResult(
                list(args), list(args),
                '{"providers": {}, "models": {'
                '"kimi-code/k3": {"provider": "kimi-code"},'
                '"anthropic/claude-opus": {"provider": "anthropic"}'
                '}}\n',
                "", 0,
            )

    monkeypatch.setattr(cli, "_build_executor", lambda *args, **kwargs: StubExecutor())

    result = runner.invoke(
        cli.app,
        [
            "--config", str(config_path), "list", "models",
            "--harness", "kimi", "kimi",
        ],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == [
        (["kimi", "provider", "list", "--json"], {"check": False, "capture_output": True})
    ]
    assert "kimi-code/k3" in result.stdout
    assert "anthropic/claude-opus" not in result.stdout


def test_list_providers_shows_pass_reference_without_resolving_it(tmp_path, monkeypatch):
    runner = CliRunner()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    config_path = _write_config(tmp_path, skills_entry=skills_dir)
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "credentials:\n"
            "  openrouter:\n"
            "    pass: openrouter.ai/apikey\n"
        )
    monkeypatch.setattr(cli, "run_startup_checks", lambda *args, **kwargs: None)

    result = runner.invoke(
        cli.app, ["--config", str(config_path), "list", "providers"],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    assert "openrouter" in result.stdout
    assert "https://openrouter.ai/api/v1" in result.stdout
    assert "pass:openrouter.ai/apikey" in result.stdout


def test_goose_harness_shorthand_starts_interactive_run() -> None:
    assert cli._resolve_harness_command(
        harness="goose", agent=None, agent_command=None,
    ) == ["goose", "run", "--interactive"]


@pytest.mark.parametrize("subcommand, method", [("agents-run", "launch_agent"), ("collaborate", "bootstrap")])
def test_launch_commands_forward_harness_and_model(
    tmp_path, monkeypatch, subcommand, method,
):
    runner = CliRunner()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    config_path = _write_config(tmp_path, skills_entry=skills_dir)
    monkeypatch.setattr(cli, "run_startup_checks", lambda *args, **kwargs: None)

    calls = []

    class StubManager:
        def context_for(self, name):
            return SimpleNamespace(name=name)

        def launch_agent(self, ctx, **kwargs):
            calls.append(("launch_agent", ctx.name, kwargs))

        def bootstrap(self, ctx, **kwargs):
            calls.append(("bootstrap", ctx.name, kwargs))

    monkeypatch.setattr(cli, "_build_manager_for_mirror", lambda *args, **kwargs: StubManager())

    result = runner.invoke(
        cli.app,
        [
            "--config", str(config_path), subcommand, "sample", "--no-sync",
            "--harness", "aider", "--model", "openrouter/deepseek/deepseek-chat",
        ],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    assert calls[0][0] == method
    assert calls[0][2]["command_override"] == ["aider"]
    assert calls[0][2]["model_override"] == "openrouter/deepseek/deepseek-chat"


def test_harness_and_legacy_agent_are_mutually_exclusive(tmp_path, monkeypatch):
    runner = CliRunner()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    config_path = _write_config(tmp_path, skills_entry=skills_dir)
    monkeypatch.setattr(cli, "run_startup_checks", lambda *args, **kwargs: None)

    result = runner.invoke(
        cli.app,
        [
            "--config", str(config_path), "agents-run", "sample",
            "--harness", "aider", "--agent", "codex",
        ],
    )

    assert result.exit_code != 0
    assert "either --harness or the legacy --agent" in _plain_output(result)


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


def test_slurm_timer_script_omits_scancel(monkeypatch):
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
    # The script body now comes from ``slurm_timer.build_timer_script``;
    # inspecting ``_start_slurm_timer``'s own source would check nothing.
    # Capture what it actually ships to the node instead.
    from sucoder import slurm_timer

    rendered = []
    real_build = slurm_timer.build_timer_script

    def capture(**kw):
        script = real_build(**kw)
        rendered.append(script)
        return script

    monkeypatch.setattr(cli, "build_timer_script", capture)
    # ``_start_slurm_timer`` imports subprocess locally as ``_sp``; patch
    # the module so its ssh write/start calls are swallowed.
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
    session = SimpleNamespace(slurm_job_id=7, mirror_name="sample", compute_node="n0")
    control = SimpleNamespace(ssh_options=lambda **kw: [])
    cli._start_slurm_timer(session, control, control, mock.Mock())

    assert rendered, "the timer script was not rendered"
    for raw in rendered[0].splitlines():
        stripped = raw.strip()
        # Strings may *mention* scancel ("Run `scancel N` to free ..."),
        # but no line may execute it.  Stripping quoted strings and
        # comments leaves only shell code; ``startswith`` missed every
        # realistic reintroduction (``then scancel``, ``$(scancel ...)``).
        code = re.sub(r"'[^']*'|\"[^\"]*\"", "", stripped).split("#", 1)[0]
        assert "scancel" not in code, (
            "the deadline timer still emits a `scancel` shell command: "
            f"{stripped!r}.  The user owns the SLURM lifecycle; use "
            "`sucoder release` for explicit cancel."
        )
    assert "JOB=7\n" in rendered[0]


def test_salloc_timer_is_per_mirror_and_staged_atomically(monkeypatch):
    """Preserve main's atomic staging and readiness handshake (ledger 4)."""
    calls = []

    def record(argv, *a, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", record)
    session = SimpleNamespace(slurm_job_id=7, mirror_name="K Agg", compute_node="n0")
    control = SimpleNamespace(ssh_options=lambda **kw: [])
    cli._start_slurm_timer(session, control, control, mock.Mock())

    write, start = calls[0][-1], calls[1][-1]
    # The script name hashes its contents, including mirror/allocation identity.
    script = re.search(r"slurm-timer-[0-9a-f]{24}\.sh", write).group()
    assert script in start
    for cmd in (write, start):
        assert '"$HOME/.cache/sucoder/slurm-timer.sh"' not in cmd
    # Staged to a temp file, chmod'd, then renamed over the destination.
    assert 'mktemp' in write
    assert 'cat > "$tmp"' in write
    assert 'chmod 700 "$tmp"' in write
    assert 'mv "$tmp"' in write
    assert write.index('cat > "$tmp"') < write.index('mv "$tmp"')
    assert '--ensure' in start
    assert 'pkill' not in start and 'nohup' not in start


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


def test_resolve_cert_username_precedence(monkeypatch):
    """username -> $BRC_USER -> config.human_user -> getpass.getuser()."""
    cfg = SimpleNamespace(human_user="hu")

    # An explicit control user wins over everything else.
    monkeypatch.setenv("BRC_USER", "envu")
    assert cli._resolve_cert_username("ctlu", cfg) == "ctlu"

    # $BRC_USER wins over the configured human_user.
    assert cli._resolve_cert_username(None, cfg) == "envu"

    # With no control user and no env, fall back to the configured human_user.
    monkeypatch.delenv("BRC_USER", raising=False)
    assert cli._resolve_cert_username(None, cfg) == "hu"

    # No config threaded at all -> local OS user, never a crash.
    monkeypatch.setattr(cli.getpass, "getuser", lambda: "localu", raising=False)
    assert cli._resolve_cert_username(None, None) == "localu"


def test_maybe_offer_cert_mint_defaults_to_human_user(monkeypatch):
    """No control user and no $BRC_USER -> mint with config.human_user.

    Regression guard: the fallback used to call ``typer.get_current_context``
    (which does not exist -- it is a ``click`` function) and crashed with
    ``AttributeError`` before the cert could be minted.
    """
    from sucoder import cert as cert_mod

    monkeypatch.delenv("BRC_USER", raising=False)
    monkeypatch.setattr(cli, "sys", _tty(True))
    monkeypatch.setattr(cli, "_cert_status", lambda cf: ("⚠", "cert EXPIRED (x)"))
    monkeypatch.setattr(cli.typer, "confirm", lambda *a, **k: True)
    answers = iter(["1234", "567890"])
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: next(answers))
    monkeypatch.setattr(cli.typer, "echo", lambda *a, **k: None)

    minted = {}
    def fake_mint(cert_file, ca_url, username, pin, otp, lifetime):
        minted.update(username=username)
        return {"key_id": "k"}
    monkeypatch.setattr(cert_mod, "mint", fake_mint)

    cli._maybe_offer_cert_mint(
        _fake_control(),
        logger=SimpleNamespace(info=lambda *a, **k: None),
        config=SimpleNamespace(human_user="ligon"),
    )
    assert minted == {"username": "ligon"}


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
        captured["slurm_kwargs"] = dict(kw)
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


def test_build_executor_confined_local_disk_is_tiering(tmp_path, monkeypatch):
    """``slurm.local_disk`` on a confined target keeps the *mirror root* on
    the shared FS (it is the durable repo and the staging area) and hands
    the local-disk root to the executor for ``_launch_confined``."""
    import logging

    captured, slurm_calls = _install_build_executor_fakes(monkeypatch, tmp_path)
    config = Config(human_user="coder", mirror_root=tmp_path / "mirrors")
    settings = _confined_mirror_settings(tmp_path, confined=True)
    settings.remote.slurm.local_disk = "/local"

    cli._build_executor(
        config, logging.getLogger("t"), dry_run=False, mirror_settings=settings,
    )

    assert slurm_calls == []
    kwargs = captured["kwargs"]
    assert kwargs["remote_mirror_root"] == str(settings.remote.mirror_root)
    assert kwargs["local_disk_root"] == "/local"
    # Scaffolding still goes through the DTN: the shared mirror is reachable.
    assert kwargs.get("scaffolding_node")


def test_build_executor_confined_local_disk_flag_without_config(tmp_path, monkeypatch):
    """``sucoder --local-disk -T <confined>`` turns tiering on with the
    default root even when the target config says nothing about it."""
    import logging

    captured, _ = _install_build_executor_fakes(monkeypatch, tmp_path)
    config = Config(human_user="coder", mirror_root=tmp_path / "mirrors")
    settings = _confined_mirror_settings(tmp_path, confined=True)
    assert settings.remote.slurm.local_disk is None

    cli._build_executor(
        config, logging.getLogger("t"), dry_run=False, mirror_settings=settings,
        local_disk_override=True,
    )
    kwargs = captured["kwargs"]
    assert kwargs["local_disk_root"] == "/local"
    assert kwargs["remote_mirror_root"] == str(settings.remote.mirror_root)


@pytest.mark.parametrize("cfg_root,override_root,expected", [
    (None, "/scratch/x", "/scratch/x"),       # root alone implies --local-disk
    ("/local", "/scratch/x", "/scratch/x"),   # CLI root beats config root
])
def test_build_executor_confined_local_disk_root_override(tmp_path, monkeypatch, cfg_root, override_root, expected):
    import logging

    captured, _ = _install_build_executor_fakes(monkeypatch, tmp_path)
    config = Config(human_user="coder", mirror_root=tmp_path / "mirrors")
    settings = _confined_mirror_settings(tmp_path, confined=True)
    settings.remote.slurm.local_disk = cfg_root

    cli._build_executor(
        config, logging.getLogger("t"), dry_run=False, mirror_settings=settings,
        local_disk_root_override=override_root,
    )
    kwargs = captured["kwargs"]
    assert kwargs["local_disk_root"] == expected
    assert kwargs["remote_mirror_root"] == str(settings.remote.mirror_root)


def test_local_disk_root_flag_rejects_no_local_disk(tmp_path, monkeypatch):
    runner = CliRunner()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    config_path = _write_config(tmp_path, skills_entry=skills_dir)
    monkeypatch.setattr(cli, "run_startup_checks", lambda *args, **kwargs: None)

    result = runner.invoke(
        cli.app,
        ["--config", str(config_path), "--no-local-disk", "--local-disk-root", "/x",
         "list", "models"],
    )
    assert result.exit_code != 0
    assert "implies --local-disk" in _plain_output(result)


def test_local_disk_root_flag_reaches_build_executor(tmp_path, monkeypatch):
    """The value is normalised (trailing slash stripped) and handed to
    ``_build_executor`` as ``local_disk_root_override``; the bool flag
    stays None so config keeps its say on everything but the root."""
    runner = CliRunner()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    config_path = _write_config(tmp_path, skills_entry=skills_dir)
    monkeypatch.setattr(cli, "run_startup_checks", lambda *args, **kwargs: None)
    seen = {}

    def fake_build_executor(*args, **kwargs):
        seen.update(kwargs)
        raise typer.Exit(code=0)      # captured what we need; skip the command body

    monkeypatch.setattr(cli, "_build_executor", fake_build_executor)
    result = runner.invoke(
        cli.app,
        ["--config", str(config_path), "--local-disk-root", "/scratch/l/", "list", "models", "--harness", "aider"],
    )
    assert seen, f"_build_executor was never reached: {_plain_output(result)}"
    obj = seen["cli_ctx"].obj
    assert obj["local_disk_root"] == "/scratch/l"
    assert obj["local_disk"] is None
    assert cli._get_local_disk_root_override(seen["cli_ctx"]) == "/scratch/l"


def test_build_executor_unconfined_local_disk_is_tiering(tmp_path, monkeypatch):
    """An unconfined (salloc) target with slurm.local_disk keeps the mirror
    root shared, hands the root to the executor, tells the allocation step
    (so the timer snapshots the clone), and still routes scaffolding via
    the DTN."""
    import logging

    captured, slurm_calls = _install_build_executor_fakes(monkeypatch, tmp_path)
    settings = _confined_mirror_settings(tmp_path, confined=False)
    settings.remote.slurm.local_disk = "/local"
    config = Config(human_user="coder", mirror_root=tmp_path / "mirrors")

    cli._build_executor(
        config, logging.getLogger("t"), dry_run=False, mirror_settings=settings,
    )
    assert slurm_calls and slurm_calls[-1] is True
    kwargs = captured["kwargs"]
    assert kwargs["remote_mirror_root"] == str(settings.remote.mirror_root)
    assert kwargs["local_disk_root"] == "/local"
    assert kwargs.get("scaffolding_node"), "shared mirror: scaffolding stays on the DTN"
    assert captured.get("slurm_kwargs", {}).get("local_disk_root") == "/local"


def test_build_executor_warns_once_about_retired_local_mirror_root(tmp_path, monkeypatch, caplog):
    """A session saved by the retired all-on-/local layout names its node
    in a warning: commits there were never published to the shared mirror."""
    import logging
    from sucoder.session import RemoteSession

    captured, _ = _install_build_executor_fakes(monkeypatch, tmp_path)
    settings = _confined_mirror_settings(tmp_path, confined=False)
    config = Config(human_user="coder", mirror_root=tmp_path / "mirrors")
    sess = RemoteSession.load(settings.name, target_name=None)
    sess.remote_mirror_root = "/local/mirrors"
    sess.compute_node = "n0099.savio3"
    sess.save()

    with caplog.at_level(logging.WARNING):
        cli._build_executor(
            config, logging.getLogger("t"), dry_run=False, mirror_settings=settings,
        )
    assert captured["kwargs"]["remote_mirror_root"] == str(settings.remote.mirror_root)
    msgs = [r.getMessage() for r in caplog.records if "retired" in r.getMessage()]
    assert len(msgs) == 1 and "/local/mirrors" in msgs[0] and "n0099.savio3" in msgs[0]
    assert RemoteSession.load(settings.name, target_name=None).remote_mirror_root == str(settings.remote.mirror_root)


def test_start_slurm_timer_snapshots_the_local_clone_and_retires_the_old_timer(monkeypatch):
    from sucoder import slurm_timer
    rendered, ssh_cmds = [], []
    real_build = slurm_timer.build_timer_script

    def capture(**kw):
        script = real_build(**kw)
        rendered.append(script)
        return script

    monkeypatch.setattr(cli, "build_timer_script", capture)

    def fake_run(argv, *a, **k):
        ssh_cmds.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    session = SimpleNamespace(slurm_job_id=7, mirror_name="sample", compute_node="n0")
    control = SimpleNamespace(ssh_options=lambda **kw: [])
    cli._start_slurm_timer(session, control, control, mock.Mock(), local_disk_root="/local")

    assert "SNAPSHOT_DIR=/local/job7/mirrors/sample\n" in rendered[0]
    write_cmd, start_cmd = ssh_cmds[-2][-1], ssh_cmds[-1][-1]
    assert "mktemp" in write_cmd and "mv" in write_cmd
    assert "slurm-timer-" in start_cmd and "--ensure" in start_cmd
    assert "pkill" not in start_cmd


def test_build_executor_confined_no_local_disk_override_wins(tmp_path, monkeypatch):
    import logging

    captured, _ = _install_build_executor_fakes(monkeypatch, tmp_path)
    config = Config(human_user="coder", mirror_root=tmp_path / "mirrors")
    settings = _confined_mirror_settings(tmp_path, confined=True)
    settings.remote.slurm.local_disk = "/local"

    cli._build_executor(
        config, logging.getLogger("t"), dry_run=False, mirror_settings=settings,
        local_disk_override=False,
    )
    assert "local_disk_root" not in captured["kwargs"]


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


def test_push_and_sync_are_the_same_operation(tmp_path, monkeypatch):
    """`push` is the documented name; `sync` stays as a compatible alias.

    Guards against the two commands drifting apart: they must accept the
    same options and reach MirrorManager.sync with identical arguments.
    """
    import typer.main

    runner = CliRunner()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)
    config_path = _write_config(tmp_path, skills_entry=skills_dir)

    command = typer.main.get_command(cli.app)
    push_opts = {o for p in command.commands["push"].params for o in p.opts}
    sync_opts = {o for p in command.commands["sync"].params for o in p.opts}
    assert push_opts == sync_opts, (push_opts ^ sync_opts)

    calls: list = []

    class StubManager:
        def context_for(self, name):
            return SimpleNamespace(name=name)

        def sync(self, ctx, *, allow_unverified_mirror=False):
            calls.append((ctx.name, allow_unverified_mirror))

    monkeypatch.setattr(
        cli, "_build_manager_for_mirror", lambda *a, **kw: StubManager(),
    )

    for name in ("push", "sync"):
        result = runner.invoke(
            cli.app,
            ["--config", str(config_path), name, "sample",
             "--allow-unverified-mirror"],
        )
        assert result.exit_code == 0, (name, result.output, result.exception)

    assert calls == [("sample", True), ("sample", True)]


def test_push_reports_local_mirror_branches_were_not_moved(tmp_path, caplog):
    """The local path must not imply the mirror now matches canonical."""
    import logging

    from tests.test_mirror import build_manager

    manager = build_manager(tmp_path)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    with caplog.at_level(logging.INFO):
        manager.sync(ctx)

    assert any(
        "own branches were not moved" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_salloc_job_carries_the_same_name_a_confined_launch_uses(tmp_path, monkeypatch):
    """An unnamed allocation is invisible to everything that looks jobs up
    by mirror.  `sucoder sessions` filters on the `sucoder-` prefix, so an
    salloc job did not appear in it at all, and the name-keyed reuse probe
    that closed issue 19 for confined launches has nothing to match on.
    The name is sanitized identically to `confined_tmux_target`'s, or the
    two paths disagree about what a mirror is called.
    """
    import logging

    import typer

    from sucoder import session as session_mod
    from sucoder.config import RemoteConfig, SlurmConfig

    sessions = tmp_path / "sessions"
    monkeypatch.setattr(session_mod, "_session_dir", lambda: sessions)

    remote = RemoteConfig(
        gateway="gw", transfer_host="dtn",
        slurm=SlurmConfig(partition="savio3", account="acct", time="24:00:00"),
    )
    sess = session_mod.RemoteSession(
        mirror_name="K Aggregators", target_name="savio-node", login_node="ln003.brc",
    )

    class _FakeControl:
        def ssh_options(self, **kw):
            return []

    seen = []

    def fake_run(cmd, *a, **kw):
        joined = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
        if "salloc" in joined:
            seen.append(joined)
            return subprocess.CompletedProcess(
                cmd, 0, stdout="",
                stderr="salloc: Granted job allocation 34688352\n",
            )
        if "squeue --job" in joined:
            raise subprocess.CalledProcessError(1, cmd, stderr="stop here")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(typer.Exit):
        cli._ensure_slurm_node(
            remote, sess, _FakeControl(), _FakeControl(), logging.getLogger("t"),
        )

    assert seen, "salloc was never invoked"
    # The space collapses, exactly as confined_tmux_target would render it.
    assert "--job-name=sucoder-K_Aggregators" in seen[0], seen[0]


# -- the destructive prompt's default ------------------------------------------
#
# Every other `release` test passes `-f`, so the confirmation path had no
# coverage at all: the prompt read `[y/N]` while `typer.confirm` was called
# with `default=True`, and bare Enter cancelled the allocation.  A regression
# flipping that back would otherwise pass the whole suite in silence, because
# the tests that do stub `typer.confirm` return a fixed bool and never see the
# default.

def test_prompt_yes_no_defaults_to_yes_for_the_opt_in_callers(monkeypatch):
    """MirrorManager's prompt_handler (poetry install, MCP discovery) reaches
    this with no `default`, and those are opt-in conveniences."""
    seen: dict = {}
    monkeypatch.setattr(
        cli.typer, "confirm",
        lambda msg, **kw: seen.update(kw) or True,
    )
    cli._prompt_yes_no("enable the thing?")
    assert seen["default"] is True


def test_prompt_yes_no_can_default_to_no(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(
        cli.typer, "confirm",
        lambda msg, **kw: seen.update(kw) or False,
    )
    cli._prompt_yes_no("cancel the job?", default=False)
    assert seen["default"] is False


def test_release_declines_on_a_bare_enter(tmp_path, monkeypatch):
    """Enter must NOT cancel the allocation, and nothing may be dialed."""
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

    def _must_not_run(*a, **kw):
        raise AssertionError("release dialed the cluster after declining")
    monkeypatch.setattr(cli, "_run_remote_capture", _must_not_run)

    # Bare Enter on the confirmation.
    result = runner.invoke(
        cli.app,
        ["--config", str(config_path), "-T", "fake-slurm", "release", "sample"],
        input="\n",
    )
    assert result.exit_code == 0, (result.stdout, result.exception)
    assert "Aborted." in result.stdout
    # The prompt must not advertise a default it does not have.
    assert "[y/N]  [Y/n]" not in result.stdout
    # And the allocation is still recorded.
    reloaded = session_mod.RemoteSession.load("sample", target_name="fake-slurm")
    assert reloaded.slurm_job_id == 7654321


# -- the pane probe's per-group mirror_root ------------------------------------
#
# `_probe_session_panes` had no tests.  It read `mirror_root` off the
# cluster's FIRST target and applied it to every group in that cluster, so
# two targets sharing a gateway but not a mirror root sent one target's WIP
# lookup at the other's path.

def _probe_fixture(monkeypatch, roots):
    """Drive _probe_session_panes over two targets on one gateway.

    *roots* maps target name -> mirror_root.  Returns the remote script the
    probe would have run.
    """
    import logging as _logging

    from sucoder import tunnel as tunnel_mod
    from sucoder.sessions_report import (
        JobRow, Report, SessionEntry, TargetGroup,
    )

    def _job(job_id):
        return JobRow(job_id=job_id, name="sucoder-M", partition="p", account="a",
                      qos="q", state="RUNNING", time_left="1:00", node="n1")

    names = sorted(roots)
    report = Report(groups=[
        TargetGroup(name=n, signature="sig", entries=[
            SessionEntry(job=_job(1000 + i), mirror=f"mir{i}", target=n),
        ])
        for i, n in enumerate(names)
    ])

    class _Slurm:
        confined = True

    class _Remote:
        def __init__(self, root):
            self.slurm = _Slurm()
            self.mirror_root = root
            self.gateway = "gw.example.org"
        def ssh_control_kwargs(self):
            return {}

    class _Config:
        targets = {n: _Remote(roots[n]) for n in names}

    class _Control:
        def __init__(self, **kw):
            pass
        def is_active(self):
            return True

    monkeypatch.setattr(tunnel_mod, "SshControl", _Control)
    sent: dict = {}

    def _fake_capture(control, host, command, **kw):
        sent["command"] = command
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cli, "_run_remote_capture", _fake_capture)
    cli._probe_session_panes(
        report, _Config(), {"gw.example.org": names}, _logging.getLogger("t"), False,
    )
    return sent.get("command", "")


def test_probe_uses_each_targets_own_mirror_root(monkeypatch):
    """Two targets, one gateway, different roots: each WIP lookup must use
    its own, not whichever target sorted first."""
    command = _probe_fixture(
        monkeypatch, {"alpha": "~/mirrors", "beta": "/scratch/mirrors"},
    )
    assert '"$HOME"/mirrors/mir0' in command, command
    assert "/scratch/mirrors/mir1" in command, command
    # The bug: beta's entry rendered under alpha's root.
    assert "/scratch/mirrors/mir0" not in command
    assert '"$HOME"/mirrors/mir1' not in command


# -- ephemeral mirrors must not collide on the sanitized token (issue #16) -----
#
# `_reject_token_collisions` runs at config load, but `_create_ephemeral_mirror`
# injects into config.mirrors afterwards, from a *directory* name -- which is
# far less constrained than a configured one. The collision is silent: the tmux
# session and socket, the staged scripts and the warn file are all keyed on the
# sanitized token.

def _ephemeral_config(mirror_names):
    from sucoder.config import BranchPrefixes, Config, MirrorSettings

    mirrors = {
        n: MirrorSettings(
            name=n,
            canonical_repo=pathlib.Path(f"/nowhere/{n}"),
            mirror_name=n,
            branch_prefixes=BranchPrefixes(human="h", agent="a"),
        )
        for n in mirror_names
    }
    cfg = Config.__new__(Config)
    cfg.mirrors = mirrors
    cfg.human_user = "h"
    cfg.agent_user = "a"
    cfg.agent_launcher = None
    cfg.skills = []
    return cfg


def test_ephemeral_mirror_refuses_a_token_collision(tmp_path):
    """`~/work/K Agg` sanitizes to `K_Agg` and would silently share a tmux
    session, socket, staged scripts and warn file with a configured `K_Agg`."""
    config = _ephemeral_config(["K_Agg"])
    toplevel = tmp_path / "K Agg"
    toplevel.mkdir()
    from sucoder.config import ConfigError

    with pytest.raises(ConfigError) as exc:
        cli._create_ephemeral_mirror(config, toplevel)
    assert "K_Agg" in str(exc.value)
    # And nothing was injected on the way out.
    assert "K Agg" not in config.mirrors


def test_ephemeral_mirror_with_no_collision_is_injected(tmp_path):
    config = _ephemeral_config(["Something-Else"])
    toplevel = tmp_path / "K Agg"
    toplevel.mkdir()
    name = cli._create_ephemeral_mirror(config, toplevel)
    assert name == "K Agg"
    assert config.mirrors["K Agg"].canonical_repo == toplevel


def test_ephemeral_mirror_does_not_collide_with_itself(tmp_path):
    """Both callers guarantee the name is not already configured; if that
    ever changed, a self-collision would block every command in the repo."""
    config = _ephemeral_config([])
    toplevel = tmp_path / "PlainName"
    toplevel.mkdir()
    assert cli._create_ephemeral_mirror(config, toplevel) == "PlainName"


# -- an unusable SSH config must not sink the whole listing --------------------
#
# `ssh_control_kwargs()` resolves a direct target through `ssh -G` and raises
# ConfigError when that config cannot be evaluated (a ControlPath ending in a
# bare `%`, say).  It was called while *building* each SshControl, outside the
# try that guards the connect, so one such target aborted `sessions` with a
# traceback -- discarding the clusters that had already answered.

def _raising_remote(gateway="gw.example.org"):
    from sucoder.config import ConfigError

    class _Remote:
        slurm = None
        mirror_root = "~/mirrors"

        def __init__(self):
            self.gateway = gateway
            self.host = gateway

        def ssh_control_kwargs(self):
            raise ConfigError("Cannot resolve direct SSH configuration for x.")

    return _Remote()


def test_pane_probe_survives_unresolvable_ssh_config(monkeypatch):
    import logging as _logging

    from sucoder.sessions_report import JobRow, Report, SessionEntry, TargetGroup

    job = JobRow(job_id=7, name="sucoder-M", partition="p", account="a", qos="q",
                 state="RUNNING", time_left="1:00", node="n1")
    report = Report(groups=[
        TargetGroup(name="alpha", signature="sig",
                    entries=[SessionEntry(job=job, mirror="mir", target="alpha")]),
    ])

    class _Config:
        targets = {"alpha": _raising_remote()}

    def _no_capture(*args, **kwargs):  # pragma: no cover -- must not be reached
        raise AssertionError("probed a target whose SSH config cannot be read")

    monkeypatch.setattr(cli, "_run_remote_capture", _no_capture)
    cli._probe_session_panes(
        report, _Config(), {"gw.example.org": ["alpha"]},
        _logging.getLogger("t"), False,
    )
    assert any("pane probe skipped" in e and "Cannot resolve" in e
               for e in report.errors), report.errors
    # An unanswered probe is not evidence: no entry is marked agent-exited.
    assert report.groups[0].entries[0].pane is None


def test_login_probe_survives_unresolvable_ssh_config(monkeypatch):
    import logging as _logging

    from sucoder.sessions_report import Report

    report = Report(groups=[])

    class _Config:
        targets = {"droplet": _raising_remote(gateway="direct.example.org")}

    def _no_capture(*args, **kwargs):  # pragma: no cover -- must not be reached
        raise AssertionError("probed a host whose SSH config cannot be read")

    monkeypatch.setattr(cli, "_run_remote_capture", _no_capture)
    cli._probe_login_sessions(
        report, _Config(), ["droplet"], _logging.getLogger("t"), False,
    )
    assert any("Cannot resolve" in e for e in report.errors), report.errors
    assert report.login_probed


# -- the warm path must not pay for a liveness probe (perf regression guard) ---
#
# is_active()'s end-to-end probe opens a real ssh session, and on a BRC login
# node a session *open* is ~10s before the command is even exec'd.  Probing
# first and then running the command paid that toll twice per host.  The work
# is BatchMode, so it is its own probe: connect only when the transport fails.

def _capture_harness(monkeypatch, results):
    """Drive _capture_over_tunnel over a scripted sequence of ssh results.

    Returns (captured_commands, connect_calls).
    """
    calls, connects = [], []
    pending = list(results)

    def _fake_capture(control, host, command, **kw):
        calls.append(command)
        return pending.pop(0)

    def _fake_connect(control, label, logger, **kw):
        connects.append(label)

    monkeypatch.setattr(cli, "_run_remote_capture", _fake_capture)
    monkeypatch.setattr(cli, "_connect_with_retry", _fake_connect)
    return calls, connects


def _done(rc, stderr=""):
    return SimpleNamespace(returncode=rc, stdout="out", stderr=stderr)


def test_warm_tunnel_runs_command_without_connecting(monkeypatch):
    import logging as _logging

    calls, connects = _capture_harness(monkeypatch, [_done(0)])
    result = cli._capture_over_tunnel(
        object(), "ln001.brc", "tmux ls", logger=_logging.getLogger("t"),
    )
    assert result.returncode == 0
    assert calls == ["tmux ls"]      # ran once
    assert connects == []            # and never opened a probe session


def test_remote_command_failure_does_not_trigger_reconnect(monkeypatch):
    """A non-zero exit from the *command* proves the tunnel works; re-authing
    on it would turn every `tmux ls` on a host with no server into a re-auth."""
    import logging as _logging

    calls, connects = _capture_harness(monkeypatch, [_done(1, "no server running")])
    result = cli._capture_over_tunnel(
        object(), "ln001.brc", "tmux ls", logger=_logging.getLogger("t"),
    )
    assert result.returncode == 1
    assert connects == []
    assert len(calls) == 1


@pytest.mark.parametrize("rc,stderr", [
    (255, "ssh: connect to host ln001.brc port 22: Connection refused"),
    (124, "timed out after 60s (wedged tunnel?)"),
    (1, "mux_client_request_session: session open refused by peer"),
])
def test_transport_failure_connects_and_retries_once(monkeypatch, rc, stderr):
    import logging as _logging

    calls, connects = _capture_harness(
        monkeypatch, [_done(rc, stderr), _done(0)],
    )
    result = cli._capture_over_tunnel(
        object(), "ln001.brc", "tmux ls", logger=_logging.getLogger("t"),
    )
    assert result.returncode == 0
    assert connects == ["ln001.brc"]   # authenticated once
    assert len(calls) == 2             # and retried exactly once


# -- login-node probes run concurrently ---------------------------------------
#
# Hosts are independent and the cost is almost all remote session setup, so
# probing them one after another made the wall time the sum rather than the
# max.  Phase 1 fans out on the warm path only; anything needing credentials
# is done serially in phase 2, so two OTP prompts can never interleave.

def _login_probe_fixture(monkeypatch, hosts, capture):
    """Drive _probe_login_sessions over *hosts*, one schedulerless target each."""
    import logging as _logging

    from sucoder import tunnel as tunnel_mod
    from sucoder.session import RemoteSession
    from sucoder.sessions_report import Report

    class _Remote:
        slurm = None
        mirror_root = "~/mirrors"

        def __init__(self, host):
            self.gateway = host
            self.host = host

        def ssh_control_kwargs(self):
            return {}

    class _Config:
        targets = {h: _Remote(h) for h in hosts}

    class _Control:
        def __init__(self, **kw):
            self.gateway = kw.get("gateway")

    monkeypatch.setattr(tunnel_mod, "SshControl", _Control)
    monkeypatch.setattr(
        RemoteSession, "login_nodes_for_target", staticmethod(lambda name: {}),
    )
    monkeypatch.setattr(cli, "_run_remote_capture", capture)

    report = Report(groups=[])
    cli._probe_login_sessions(
        report, _Config(), list(hosts), _logging.getLogger("t"), False,
    )
    return report


def test_login_probes_run_in_parallel(monkeypatch):
    """Deterministic concurrency check: every host must reach the barrier
    before any is released.  A serial sweep deadlocks it and raises."""
    import threading

    hosts = ["ln001.brc", "ln002.brc", "ln003.brc"]
    barrier = threading.Barrier(len(hosts), timeout=10)

    def _capture(control, host, command, **kw):
        barrier.wait()          # BrokenBarrierError if the sweep is serial
        return SimpleNamespace(
            returncode=0, stdout=f"sucoder-{host}\tclaude\n", stderr="",
        )

    report = _login_probe_fixture(monkeypatch, hosts, _capture)
    assert sorted(s.host for s in report.logins) == hosts
    assert [s.pane for s in report.logins] == ["claude"] * 3
    assert report.errors == []


def test_warm_login_probes_never_authenticate(monkeypatch):
    """The whole point: a reachable host costs one round trip, not a probe
    session plus a command session plus a pane session."""
    calls = []

    def _capture(control, host, command, **kw):
        calls.append(host)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def _no_connect(*args, **kwargs):  # pragma: no cover -- must not be reached
        raise AssertionError("warm host was made to authenticate")

    monkeypatch.setattr(cli, "_connect_with_retry", _no_connect)
    report = _login_probe_fixture(monkeypatch, ["ln001.brc", "ln002.brc"], _capture)
    assert sorted(calls) == ["ln001.brc", "ln002.brc"]   # one each, no more
    assert report.errors == []


def test_cold_login_probe_authenticates_serially_then_retries(monkeypatch):
    """A transport failure in the parallel sweep must be retried *after* the
    pool closes, so the credential prompt is not competing with other threads."""
    import logging as _logging
    import threading

    attempts = []
    connects = []

    def _capture(control, host, command, **kw):
        attempts.append(host)
        if attempts.count(host) == 1:
            return SimpleNamespace(
                returncode=255, stdout="", stderr="ssh: connect: Connection refused",
            )
        return SimpleNamespace(returncode=0, stdout="sucoder-x\tclaude\n", stderr="")

    def _fake_connect(control, label, logger, **kw):
        connects.append((label, threading.current_thread().name))

    monkeypatch.setattr(cli, "_connect_with_retry", _fake_connect)
    report = _login_probe_fixture(monkeypatch, ["ln001.brc"], _capture)

    assert connects and connects[0][0] == "ln001.brc"
    # The retry happens on the main thread, i.e. after the pool has closed.
    assert connects[0][1] == threading.main_thread().name
    assert attempts == ["ln001.brc", "ln001.brc"]        # warm, then retry
    assert [s.name for s in report.logins] == ["sucoder-x"]


def test_remote_capture_routes_its_fallback_through_the_jump_host(monkeypatch):
    """A jump-only login node does not resolve locally, so the fresh
    connection ssh opens when the mux refuses a session must go via the
    gateway -- otherwise a wedged mux becomes "Could not resolve hostname"
    and costs a full re-auth."""
    from sucoder.tunnel import SshControl

    gw = SshControl(gateway="hpc.brc.berkeley.edu")
    ln = SshControl(gateway="ln003.brc", jump_host="hpc.brc.berkeley.edu",
                    jump_control=gw)

    seen = {}

    def _fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    cli._run_remote_capture(ln, "ln003.brc", "tmux ls")

    joined = " ".join(seen["cmd"])
    assert "ProxyCommand=" in joined
    assert "hpc.brc.berkeley.edu" in joined
    assert str(gw.socket_path) in joined


# -- the login sweep overlaps the scheduler queries ---------------------------
#
# A login-node session has no allocation, so nothing in that sweep reads
# squeue; it was sequenced after the cluster queries only by accident of where
# the call sat.  Started first and collected last, its remote session setup
# (the dominant cost) runs *during* the scheduler queries instead of after.

def test_start_login_probes_does_not_wait_for_the_answers(monkeypatch):
    """The enabling property: starting the sweep must return immediately,
    leaving the queries in flight for the caller to get on with its work."""
    import threading

    release = threading.Event()
    started = threading.Event()

    def _capture(control, host, command, **kw):
        started.set()
        assert release.wait(timeout=10), "probe was never released"
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cli, "_run_remote_capture", _capture)

    class _Control:
        pass

    plan = cli._LoginProbes(
        probes=[("t", "ln001.brc", _Control())], errors=[], pool=None, futures=[],
    )
    live = cli._start_login_probes(plan, False)
    # Returned while the query is still running -- that is the whole point.
    assert started.wait(timeout=10)
    assert not live.futures[0].done()
    release.set()
    assert live.futures[0].result().returncode == 0
    live.pool.shutdown(wait=True)


def test_sessions_queries_login_nodes_while_squeue_is_still_running(monkeypatch):
    """Deterministic overlap check on the command itself.

    The fake squeue refuses to return until a login probe has started.  If
    `sessions` went back to sweeping the login nodes *after* the cluster
    queries, nothing would ever set that event and this deadlocks out.
    """
    import logging as _logging
    import threading

    from sucoder.session import RemoteSession

    login_started = threading.Event()

    class _Slurm:
        confined = True

    class _Remote:
        mirror_root = "~/mirrors"
        partition = account = qos = None

        def __init__(self, host, slurm):
            self.gateway = host
            self.host = host
            self.slurm = slurm

        def ssh_control_kwargs(self):
            return {}

    # One scheduler-backed target (queried with squeue) and one schedulerless
    # (swept for login-node tmux sessions).  Both are needed, or there is no
    # squeue to overlap with and the test proves nothing.
    config = SimpleNamespace(
        targets={
            "savio": _Remote("hpc.brc", _Slurm()),
            "droplet": _Remote("direct.example", None),
        },
        mirrors={},
        log_dir=None,
    )

    # Recorded and asserted on AFTER the command returns: the cluster loop
    # catches every exception into its error list, so a bare assert in here
    # would be swallowed and the test would pass regardless.
    events = []

    def _fake_squeue(control, host, command, **kw):
        events.append("squeue:start")
        login_started.wait(timeout=5)     # give the sweep room to reach the wire
        events.append("squeue:end")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def _fake_login_capture(control, host, command, **kw):
        events.append("login")
        login_started.set()
        return SimpleNamespace(returncode=0, stdout="sucoder-x\tclaude\n", stderr="")

    monkeypatch.setattr(cli, "_capture_over_tunnel", _fake_squeue)
    monkeypatch.setattr(cli, "_run_remote_capture", _fake_login_capture)
    monkeypatch.setattr(cli, "setup_logger", lambda *a, **k: _logging.getLogger("t"))
    monkeypatch.setattr(
        RemoteSession, "login_nodes_for_target", staticmethod(lambda name: {}),
    )
    monkeypatch.setattr(RemoteSession, "recorded_jobs", staticmethod(lambda: {}))
    monkeypatch.setattr(
        RemoteSession, "holders_of_job", staticmethod(lambda job_id: []),
    )

    ctx = SimpleNamespace(obj={"config": config}, params={})
    cli.sessions(ctx, fast=False, login_nodes=True, verbose=False)

    assert "login" in events and "squeue:end" in events
    # The login node was asked while squeue was still outstanding.  Sweeping
    # afterwards puts "login" last and fails here.
    assert events.index("login") < events.index("squeue:end"), events


# -- a cold start must not crawl through the login nodes ----------------------
#
# The warm sweep only helps when masters already exist.  A freshly minted
# certificate leaves none, so every host fails phase 1 fast and lands in the
# reconnect path -- which connected them one at a time.  The gateway is the
# only hop that can prompt (and the only one that can earn "Too many
# authentication failures" if raced), so it goes first, alone; the nodes
# behind it are publickey hops through its live mux and go together.

def test_cold_login_connects_gateway_once_then_nodes_in_parallel(monkeypatch):
    import threading

    order = []
    order_lock = threading.Lock()
    hosts = ["ln001.brc", "ln002.brc", "ln003.brc"]
    barrier = threading.Barrier(len(hosts), timeout=10)

    class _Gateway:
        gateway = "hpc.brc"
        jump_control = None

    gw = _Gateway()

    class _Node:
        def __init__(self, host):
            self.gateway = host
            self.jump_control = gw

    pending = [(i, "savio", h, _Node(h)) for i, h in enumerate(hosts)]

    def _fake_connect(control, label, logger, **kw):
        with order_lock:
            order.append(label)
        if control is not gw:
            # Every node must be connecting at once; serial trips the barrier.
            barrier.wait()

    monkeypatch.setattr(cli, "_connect_with_retry", _fake_connect)
    monkeypatch.setattr(
        cli, "_run_remote_capture",
        lambda control, host, command, **kw: SimpleNamespace(
            returncode=0, stdout="", stderr="",
        ),
    )

    cli._open_login_gateways(pending, None, None)
    out = cli._reconnect_login_probes(pending, None, None, False)

    assert order[0] == "hpc.brc"            # gateway before anything else
    assert order.count("hpc.brc") == 1      # and authenticated once, not per node
    assert sorted(order[1:]) == hosts
    assert not any(isinstance(v, Exception) for v in out.values()), out


def test_unreachable_login_host_does_not_sink_the_others(monkeypatch):
    """One host's raise is carried, not thrown: the rest still answer."""
    class _Node:
        jump_control = None

        def __init__(self, host):
            self.gateway = host

    pending = [(0, "t", "good.example", _Node("good.example")),
               (1, "t", "bad.example", _Node("bad.example"))]

    def _fake_connect(control, label, logger, **kw):
        if label == "bad.example":
            raise RuntimeError("host is down")

    monkeypatch.setattr(cli, "_connect_with_retry", _fake_connect)
    monkeypatch.setattr(
        cli, "_run_remote_capture",
        lambda control, host, command, **kw: SimpleNamespace(
            returncode=0, stdout="", stderr="",
        ),
    )
    out = cli._reconnect_login_probes(pending, None, None, False)
    assert out[0].returncode == 0
    assert isinstance(out[1], Exception)


def test_no_login_nodes_flag_skips_the_sweep_without_claiming_absence(monkeypatch):
    """Skipping must leave `login_probed` False, so the report says "not
    inspected" rather than "no sessions" -- a claim the sweep never made."""
    import logging as _logging

    from sucoder.session import RemoteSession

    touched = []

    class _Remote:
        mirror_root = "~/mirrors"
        partition = account = qos = None
        slurm = None

        def __init__(self, host):
            self.gateway = host
            self.host = host

        def ssh_control_kwargs(self):
            return {}

    config = SimpleNamespace(
        targets={"droplet": _Remote("direct.example")}, mirrors={}, log_dir=None,
    )
    monkeypatch.setattr(
        cli, "_run_remote_capture",
        lambda *a, **kw: touched.append(a) or SimpleNamespace(
            returncode=0, stdout="", stderr="",
        ),
    )
    monkeypatch.setattr(cli, "setup_logger", lambda *a, **k: _logging.getLogger("t"))
    monkeypatch.setattr(
        RemoteSession, "login_nodes_for_target", staticmethod(lambda name: {}),
    )
    monkeypatch.setattr(RemoteSession, "recorded_jobs", staticmethod(lambda: {}))
    monkeypatch.setattr(
        RemoteSession, "holders_of_job", staticmethod(lambda job_id: []),
    )

    ctx = SimpleNamespace(obj={"config": config}, params={})
    cli.sessions(ctx, fast=False, login_nodes=False, verbose=False)
    assert touched == []          # nothing was asked of any login node
