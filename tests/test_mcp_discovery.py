import json
import os
from pathlib import Path
from typing import Dict

import pytest

from sucoder.config import McpServerConfig
from sucoder.mcp_discovery import (
    MCP_SUGGESTIONS,
    McpSuggestion,
    detect_suggestions,
)
from sucoder.workspace_prefs import WorkspacePrefs


# -- detect_suggestions tests -------------------------------------------------


def test_detect_suggestions_empty_dir(tmp_path: Path) -> None:
    """A bare directory still gets always-on suggestions (memory)."""
    results = detect_suggestions(tmp_path, {})
    names = [s.name for s in results]
    assert "memory" in names
    # No github suggestion without .github/
    assert "github" not in names


def test_detect_suggestions_github_dir(tmp_path: Path) -> None:
    """A .github/ directory triggers the github suggestion."""
    (tmp_path / ".github").mkdir()
    results = detect_suggestions(tmp_path, {})
    names = [s.name for s in results]
    assert "github" in names
    assert "memory" in names


def test_detect_suggestions_package_json(tmp_path: Path) -> None:
    """A package.json triggers the fetch suggestion."""
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    results = detect_suggestions(tmp_path, {})
    names = [s.name for s in results]
    assert "fetch" in names


def test_detect_suggestions_docker_compose(tmp_path: Path) -> None:
    """docker-compose.yml triggers the postgres suggestion."""
    (tmp_path / "docker-compose.yml").write_text("version: '3'", encoding="utf-8")
    results = detect_suggestions(tmp_path, {})
    names = [s.name for s in results]
    assert "postgres" in names


def test_detect_suggestions_dedup_existing_config(tmp_path: Path) -> None:
    """Servers already in config are excluded from suggestions."""
    (tmp_path / ".github").mkdir()
    existing = {"github": McpServerConfig(command="npx")}
    results = detect_suggestions(tmp_path, existing)
    names = [s.name for s in results]
    assert "github" not in names
    # Memory should still be suggested
    assert "memory" in names


def test_detect_suggestions_dedup_repo_mcp_json(tmp_path: Path) -> None:
    """Servers defined in repo's .mcp.json are excluded from suggestions."""
    (tmp_path / ".github").mkdir()
    mcp_data = {"mcpServers": {"github": {"command": "npx", "args": []}}}
    (tmp_path / ".mcp.json").write_text(json.dumps(mcp_data), encoding="utf-8")

    results = detect_suggestions(tmp_path, {})
    names = [s.name for s in results]
    assert "github" not in names


def test_detect_suggestions_invalid_mcp_json(tmp_path: Path) -> None:
    """Invalid .mcp.json is handled gracefully."""
    (tmp_path / ".github").mkdir()
    (tmp_path / ".mcp.json").write_text("not json", encoding="utf-8")
    results = detect_suggestions(tmp_path, {})
    names = [s.name for s in results]
    # github should still appear since .mcp.json couldn't be parsed
    assert "github" in names


def test_detect_suggestions_openapi(tmp_path: Path) -> None:
    """openapi.yaml triggers the fetch suggestion."""
    (tmp_path / "openapi.yaml").write_text("openapi: 3.0", encoding="utf-8")
    results = detect_suggestions(tmp_path, {})
    names = [s.name for s in results]
    assert "fetch" in names


# -- WorkspacePrefs MCP discovery tests ---------------------------------------


def test_workspace_prefs_mcp_discovery_default(tmp_path: Path) -> None:
    """Default prefs have no MCP discovery decisions."""
    prefs = WorkspacePrefs.load(tmp_path)
    assert prefs.mcp_discovery() is None


def test_workspace_prefs_mcp_discovery_roundtrip(tmp_path: Path) -> None:
    """set_mcp_discovery / mcp_discovery round-trips correctly."""
    prefs = WorkspacePrefs.load(tmp_path)
    prefs.set_mcp_discovery({"github": True, "docker": False})
    prefs.save()

    prefs2 = WorkspacePrefs.load(tmp_path)
    decisions = prefs2.mcp_discovery()
    assert decisions == {"github": True, "docker": False}


def test_workspace_prefs_mcp_discovery_merge(tmp_path: Path) -> None:
    """Subsequent set_mcp_discovery calls merge into existing decisions."""
    prefs = WorkspacePrefs.load(tmp_path)
    prefs.set_mcp_discovery({"github": True})
    prefs.set_mcp_discovery({"memory": True})
    prefs.save()

    prefs2 = WorkspacePrefs.load(tmp_path)
    decisions = prefs2.mcp_discovery()
    assert decisions == {"github": True, "memory": True}


# -- _maybe_suggest_mcp_servers integration tests ----------------------------


def test_maybe_suggest_accepts_servers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Accepted suggestions are injected into ctx.settings.mcp_servers."""
    from tests.test_mirror import build_manager, CommandResult

    manager = build_manager(tmp_path, prompt_handler=lambda _msg: True)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    mirror_path = ctx.mirror_path
    (mirror_path / ".github").mkdir(exist_ok=True)

    # Set GITHUB_TOKEN so the server is not skipped
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")

    manager._maybe_suggest_mcp_servers(ctx, mirror_path)

    assert "github" in ctx.settings.mcp_servers
    assert ctx.settings.mcp_servers["github"].env["GITHUB_TOKEN"] == "test-token"


def test_maybe_suggest_declines_servers(tmp_path: Path) -> None:
    """Declined suggestions are not injected."""
    from tests.test_mirror import build_manager

    manager = build_manager(tmp_path, prompt_handler=lambda _msg: False)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    mirror_path = ctx.mirror_path
    (mirror_path / ".github").mkdir(exist_ok=True)

    manager._maybe_suggest_mcp_servers(ctx, mirror_path)

    assert "github" not in ctx.settings.mcp_servers


def test_maybe_suggest_skips_missing_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Servers with missing required env vars are skipped even when accepted."""
    from tests.test_mirror import build_manager

    manager = build_manager(tmp_path, prompt_handler=lambda _msg: True)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    mirror_path = ctx.mirror_path
    (mirror_path / ".github").mkdir(exist_ok=True)

    # Ensure GITHUB_TOKEN is NOT set
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    manager._maybe_suggest_mcp_servers(ctx, mirror_path)

    # github should be skipped (missing token), but memory should be added
    assert "github" not in ctx.settings.mcp_servers
    assert "memory" in ctx.settings.mcp_servers


def test_maybe_suggest_remembers_decisions(tmp_path: Path) -> None:
    """Decisions are persisted and not re-prompted on second call."""
    from tests.test_mirror import build_manager

    call_count = 0

    def counting_handler(msg: str) -> bool:
        nonlocal call_count
        call_count += 1
        return False

    manager = build_manager(tmp_path, prompt_handler=counting_handler)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    mirror_path = ctx.mirror_path
    (mirror_path / ".github").mkdir(exist_ok=True)

    manager._maybe_suggest_mcp_servers(ctx, mirror_path)
    assert call_count == 1

    # Second call should not prompt again
    manager._maybe_suggest_mcp_servers(ctx, mirror_path)
    assert call_count == 1


def test_maybe_suggest_no_prompt_handler(tmp_path: Path) -> None:
    """Without a prompt handler, suggestions are silently skipped."""
    from tests.test_mirror import build_manager

    manager = build_manager(tmp_path, prompt_handler=None)
    # Explicitly clear the prompt handler since build_manager may set a default
    manager._prompt_handler = None
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)

    mirror_path = ctx.mirror_path
    (mirror_path / ".github").mkdir(exist_ok=True)

    manager._maybe_suggest_mcp_servers(ctx, mirror_path)
    assert "github" not in ctx.settings.mcp_servers
