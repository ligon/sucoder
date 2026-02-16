"""Build a unified skills catalog from local and curated sources.

This script is intentionally tool-agnostic so any LLM agent can read the
rendered catalog without Codex-specific glue.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CURATED_API_URL = "https://api.github.com/repos/openai/skills/contents/skills/.curated"
CURATED_SKILL_URL = (
    "https://raw.githubusercontent.com/openai/skills/main/skills/.curated/{name}/SKILL.md"
)
EM_DASH = "\u2014"


@dataclass
class SkillEntry:
    name: str
    description: Optional[str]
    source: str  # human-readable source name
    source_kind: str
    location: Optional[str] = None
    installed: bool = False
    url: Optional[str] = None


@dataclass
class Source:
    name: str
    kind: str  # local_catalog, curated_repo
    path: Optional[Path] = None  # local SKILLS.md or repo subpath
    repo: Optional[str] = None
    ref: str = "main"
    extra: Dict[str, str] = field(default_factory=dict)


def _parse_local_line(line: str) -> Optional[dict]:
    if not line.startswith("-"):
        return None
    text = line.lstrip("- ").strip()
    if not text:
        return None

    description: Optional[str] = None
    separator = f" {EM_DASH} "
    alt_separator = " -- "
    if separator in text:
        head, description = text.split(separator, 1)
    elif alt_separator in text:
        head, description = text.split(alt_separator, 1)
    else:
        head = text

    head = head.strip()
    description = description.strip() if description else None

    location: Optional[str] = None
    name = head
    if head.startswith("file:"):
        location = head.split()[0][5:]
        path = Path(location)
        if path.name.lower() == "skill.md":
            name = path.parent.name
        else:
            name = path.stem or path.name

    return {
        "name": name,
        "description": description,
        "location": location,
    }


def load_local_skills(catalog_path: Path, source: Source) -> List[SkillEntry]:
    if not catalog_path.exists():
        return []
    entries: List[SkillEntry] = []
    for line in catalog_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_local_line(line.strip())
        if parsed:
            entries.append(
                SkillEntry(
                    source=source.name,
                    source_kind=source.kind,
                    installed=True,
                    url=None,
                    **parsed,
                )
            )
    return entries


def _fetch_json(url: str) -> List[dict]:
    request = Request(url, headers={"Accept": "application/vnd.github.v3+json"})
    with urlopen(request) as response:
        return json.load(response)


def _fetch_skill_description_from_raw(url: str) -> Optional[str]:
    try:
        with urlopen(url) as response:
            text = response.read().decode("utf-8")
    except (HTTPError, URLError):
        return None

    if not text.startswith("---"):
        return None

    for line in text.splitlines()[1:]:
        if line.strip() == "---":
            break
        if line.startswith("description:"):
            return line.split(":", 1)[1].strip()
    return None


def _discover_local_catalog(explicit: Optional[Path]) -> Path:
    if explicit:
        return explicit.expanduser()

    candidates: List[Path] = [
        Path("~/.sucoder/skills/SKILLS.md").expanduser(),
    ]

    config_path = Path("~/.sucoder/config.yaml").expanduser()
    if config_path.exists():
        try:
            import yaml

            config_data = yaml.safe_load(config_path.read_text())
            mirrors = config_data.get("mirrors", {}) if config_data else {}
            for settings in mirrors.values():
                for skill_path in settings.get("skills", []) or []:
                    candidates.append(Path(skill_path).expanduser() / "SKILLS.md")
        except Exception:
            # Fall back to default candidate list when config parsing fails.
            pass

    # Local development mirror path fallback (e.g., ../Skills/)
    candidates.append(Path("../Skills/SKILLS.md").resolve())

    for path in candidates:
        if path.exists():
            return path

    # Return the first candidate even if it does not exist so the caller
    # can surface the missing path in the generated catalog.
    return candidates[0]


def _default_sources_config() -> Path:
    return Path("../Skills/trusted_skill_sources.yaml").resolve()


def load_sources(config_path: Optional[Path], installed_dirs: List[Path]) -> List[Source]:
    # If config path is provided or default exists, load it; otherwise build defaults.
    resolved_path = config_path.expanduser() if config_path else _default_sources_config()
    if resolved_path.exists():
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit("pyyaml is required to parse the sources config") from exc

        data = yaml.safe_load(resolved_path.read_text())
        sources_cfg = data.get("sources", []) if data else []
        sources: List[Source] = []
        for entry in sources_cfg:
            name = entry.get("name")
            kind = entry.get("kind")
            if not name or not kind:
                continue
            path = entry.get("path")
            repo = entry.get("repo")
            ref = entry.get("ref", "main")
            extra = {k: v for k, v in entry.items() if k not in {"name", "kind", "path", "repo"}}
            sources.append(
                Source(
                    name=name,
                    kind=kind,
                    path=Path(path).expanduser() if path else None,
                    repo=repo,
                    ref=ref,
                    extra=extra,
                )
            )
        if sources:
            return sources

    # Default sources when config is missing or empty.
    local_catalog_path = _discover_local_catalog(None)
    default_sources = [
        Source(
            name="local-catalog",
            kind="local_catalog",
            path=local_catalog_path,
        ),
        Source(
            name="curated-openai",
            kind="curated_repo",
            repo="openai/skills",
            path=Path("skills/.curated"),
        ),
    ]
    return default_sources


def load_curated_skills(source: Source, installed_dirs: Iterable[Path]) -> List[SkillEntry]:
    if not source.repo:
        return []

    repo_path_obj = source.path or Path("skills/.curated")
    repo_path = repo_path_obj.as_posix()
    ref = source.ref or "main"
    api_url = f"https://api.github.com/repos/{source.repo}/contents/{repo_path}"
    raw_template = f"https://raw.githubusercontent.com/{source.repo}/{ref}/{repo_path}/{{name}}/SKILL.md"

    try:
        data = _fetch_json(api_url)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to fetch curated skills listing from {api_url}: {exc}") from exc

    installed_names = set()
    for installed_dir in installed_dirs:
        if not installed_dir.exists():
            continue
        for path in installed_dir.iterdir():
            if path.name.startswith("."):
                continue
            if path.is_dir():
                installed_names.add(path.name)

    entries: List[SkillEntry] = []
    for item in data:
        if item.get("type") != "dir":
            continue
        name = item["name"]
        raw_url = raw_template.format(name=name)
        description = _fetch_skill_description_from_raw(raw_url)
        entries.append(
            SkillEntry(
                name=name,
                description=description,
                source=source.name,
                source_kind=source.kind,
                installed=name in installed_names,
                url=f"https://github.com/{source.repo}/tree/{ref}/{repo_path}/{name}",
            )
        )
    return entries


def render_catalog(
    source_entries: Dict[str, List[SkillEntry]],
    conflicts: List[str],
    sources: List[Source],
    output_path: Path,
    installed_dirs: Iterable[Path],
) -> None:
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    lines: List[str] = []
    lines.append("# Unified Skills Catalog")
    lines.append("")
    lines.append(f"Generated: {timestamp}")
    lines.append(
        "Sources: "
        + ", ".join(f"{src.name} ({src.kind})" for src in sources)
    )
    install_dirs_text = ", ".join(str(path) for path in installed_dirs) or "none"
    lines.append(f"Installed directories inspected: {install_dirs_text}")
    lines.append("")
    for src in sources:
        entries = source_entries.get(src.name, [])
        lines.append(f"## Source: {src.name} ({src.kind})")
        if src.path:
            lines.append(f"- path: `{src.path}`")
        if src.repo:
            lines.append(f"- repo: {src.repo}")
        if not entries:
            lines.append("- (no entries found)")
        else:
            for entry in sorted(entries, key=lambda e: e.name.lower()):
                status = "installed" if entry.installed else "available"
                desc = entry.description or "(no description found)"
                lines.append(f"- {entry.name} ({status}) {EM_DASH} {desc}")
                if entry.location:
                    lines.append(f"  - source: `{entry.location}`")
                if entry.url:
                    lines.append(f"  - source: {entry.url}")
        lines.append("")

    lines.append("## Conflicts (same skill name in multiple sources)")
    if conflicts:
        for name in conflicts:
            lines.append(f"- {name}")
    else:
        lines.append("- None detected")
    lines.append("")
    lines.append("## Usage")
    lines.append(
        "Any LLM can read this catalog, open a `SKILL.md`, and follow its instructions."
    )
    lines.append(
        "When sources overlap, prefer the version you trust and consider removing or renaming "
        "duplicates to avoid conflicting instructions."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a unified skills catalog (local + curated)."
    )
    parser.add_argument(
        "--installed-dir",
        action="append",
        dest="installed_dirs",
        type=Path,
        default=[Path("~/.sucoder/skills").expanduser()],
        help="Directory to check for installed curated skills (can be repeated).",
    )
    parser.add_argument(
        "--sources-config",
        type=Path,
        default=None,
        help="YAML file listing trusted skill sources (defaults to ../Skills/trusted_skill_sources.yaml when present).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the generated catalog (defaults to alongside the local catalog).",
    )
    args = parser.parse_args()

    installed_dirs = [path.expanduser() for path in args.installed_dirs]
    sources = load_sources(args.sources_config, installed_dirs)

    all_entries: Dict[str, List[SkillEntry]] = {}
    conflicts: List[str] = []

    for src in sources:
        entries: List[SkillEntry] = []
        if src.kind == "local_catalog":
            catalog_path = src.path or _discover_local_catalog(None)
            entries = load_local_skills(catalog_path, src)
        elif src.kind == "curated_repo":
            entries = load_curated_skills(src, installed_dirs)
        else:
            # Unknown source kinds are ignored to keep behavior predictable.
            entries = []

        all_entries[src.name] = entries

    # Detect conflicts by name across sources.
    name_to_sources: Dict[str, List[str]] = {}
    for src_name, entries in all_entries.items():
        for entry in entries:
            name_to_sources.setdefault(entry.name, []).append(src_name)
    conflicts = sorted(name for name, srcs in name_to_sources.items() if len(srcs) > 1)

    # Choose output path: alongside the first local catalog when possible.
    output_path = args.output
    if output_path is None:
        # Find first local_catalog source with a path.
        target_path = None
        for src in sources:
            if src.kind == "local_catalog" and src.path:
                target_path = src.path.with_name("UNIFIED_SKILLS_CATALOG.md")
                break
        output_path = target_path or Path("UNIFIED_SKILLS_CATALOG.md")

    try:
        render_catalog(
            source_entries=all_entries,
            conflicts=conflicts,
            sources=sources,
            output_path=output_path,
            installed_dirs=installed_dirs,
        )
    except PermissionError as exc:
        raise SystemExit(
            f"Permission denied writing catalog to {output_path}. "
            "Use --output to specify a writable location."
        ) from exc


if __name__ == "__main__":
    main()
