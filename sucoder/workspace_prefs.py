"""Persisted workspace preferences stored alongside the mirror."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class WorkspacePrefs:
    """Lightweight helper for storing mirror-scoped preferences."""

    root: Path
    data: Dict[str, Any] = field(default_factory=dict)
    filename: str = "agent_prefs.json"

    @property
    def path(self) -> Path:
        return self.root / self.filename

    @property
    def directory(self) -> Path:
        return self.root

    @classmethod
    def load(cls, mirror_path: Path) -> "WorkspacePrefs":
        prefs_dir = mirror_path / ".coder-session"
        prefs_path = prefs_dir / "agent_prefs.json"
        try:
            with prefs_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            data = {}
        except json.JSONDecodeError:
            data = {}
        return cls(root=prefs_dir, data=data)

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def poetry_auto_install(self) -> Optional[bool]:
        raw = self.data.get("poetry_auto_install")
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, dict):
            enabled = raw.get("enabled")
            if isinstance(enabled, bool):
                return enabled
        return None

    def set_poetry_auto_install(self, enabled: bool) -> None:
        self.data["poetry_auto_install"] = {
            "enabled": enabled,
            "decided_at": _utc_now_iso(),
        }
