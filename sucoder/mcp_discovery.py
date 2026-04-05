"""Automatic MCP server discovery based on repository tech stack."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .config import McpServerConfig


@dataclass
class McpSuggestion:
    """A single discoverable MCP server tied to file-based indicators."""

    name: str
    indicator_patterns: List[str]
    server: McpServerConfig
    required_env: List[str] = field(default_factory=list)
    description: str = ""


# Static mapping of file indicators to MCP server suggestions.
# Order: specific indicators first, always-on suggestions last.
MCP_SUGGESTIONS: List[McpSuggestion] = [
    McpSuggestion(
        name="github",
        indicator_patterns=[".github/"],
        server=McpServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": ""},
        ),
        required_env=["GITHUB_TOKEN"],
        description="GitHub API (issues, PRs, actions)",
    ),
    McpSuggestion(
        name="postgres",
        indicator_patterns=["docker-compose.yml", "docker-compose.yaml"],
        server=McpServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-postgres"],
            env={"DATABASE_URL": ""},
        ),
        required_env=["DATABASE_URL"],
        description="PostgreSQL database access",
    ),
    McpSuggestion(
        name="fetch",
        indicator_patterns=["package.json", "openapi.yaml", "openapi.json", "swagger.json"],
        server=McpServerConfig(
            command="npx",
            args=["-y", "@anthropic-ai/mcp-server-fetch"],
        ),
        description="Fetch web pages and URLs",
    ),
    McpSuggestion(
        name="memory",
        indicator_patterns=[],  # Always suggested
        server=McpServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-memory"],
        ),
        description="Persistent memory across sessions",
    ),
]


def _pattern_matches(mirror_path: Path, pattern: str) -> bool:
    """Check whether an indicator pattern matches something in mirror_path."""
    if pattern.endswith("/"):
        return (mirror_path / pattern.rstrip("/")).is_dir()
    return (mirror_path / pattern).exists()


def _read_repo_mcp_server_names(mirror_path: Path) -> set[str]:
    """Return server names already defined in the repo's ``.mcp.json``."""
    mcp_file = mirror_path / ".mcp.json"
    if not mcp_file.exists():
        return set()
    try:
        data = json.loads(mcp_file.read_text(encoding="utf-8"))
        return set(data.get("mcpServers", {}).keys())
    except (json.JSONDecodeError, OSError):
        return set()


def detect_suggestions(
    mirror_path: Path,
    existing_servers: Dict[str, McpServerConfig],
) -> List[McpSuggestion]:
    """Return MCP suggestions for *mirror_path*, excluding already-configured servers.

    *existing_servers* should include servers from both the sucoder config
    (``mcp_servers``) and any other source the caller wants to deduplicate against.
    The repo's ``.mcp.json`` is also read for deduplication.
    """
    skip_names = set(existing_servers.keys()) | _read_repo_mcp_server_names(mirror_path)

    results: List[McpSuggestion] = []
    for suggestion in MCP_SUGGESTIONS:
        if suggestion.name in skip_names:
            continue
        # Empty indicator_patterns means "always suggest"
        if not suggestion.indicator_patterns:
            results.append(suggestion)
            continue
        if any(_pattern_matches(mirror_path, p) for p in suggestion.indicator_patterns):
            results.append(suggestion)

    return results
