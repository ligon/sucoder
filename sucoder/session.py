"""Persistent SSH session state for remote mirrors."""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


def _session_dir() -> Path:
    return Path("~/.sucoder/sessions").expanduser()


@dataclass
class RemoteSession:
    """Tracks the pinned login node and SSH tunnel for a remote mirror."""

    mirror_name: str
    target_name: Optional[str] = None  # e.g. "savio-node"; None = local
    login_node: Optional[str] = None
    tunnel_port: Optional[int] = None
    tunnel_pid: Optional[int] = None
    created: Optional[str] = None
    slurm_job_id: Optional[int] = None
    compute_node: Optional[str] = None
    remote_mirror_root: Optional[str] = None  # e.g. "/local/mirrors" or "~/mirrors"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def _session_key(self) -> str:
        """Filename stem: ``mirror`` or ``mirror--target``."""
        if self.target_name:
            return f"{self.mirror_name}--{self.target_name}"
        return self.mirror_name

    @classmethod
    def load(cls, mirror_name: str, target_name: Optional[str] = None) -> "RemoteSession":
        """Load an existing session or return a blank one.

        *target_name* scopes the session to a named target (e.g.
        ``savio-node``) so that local and remote sessions for the
        same mirror don't collide.
        """
        key = f"{mirror_name}--{target_name}" if target_name else mirror_name
        path = _session_dir() / f"{key}.yaml"
        if not path.is_file():
            return cls(mirror_name=mirror_name, target_name=target_name)
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except (yaml.YAMLError, OSError):
            return cls(mirror_name=mirror_name, target_name=target_name)
        if not isinstance(data, dict):
            return cls(mirror_name=mirror_name, target_name=target_name)
        return cls(
            mirror_name=mirror_name,
            target_name=target_name,
            login_node=data.get("login_node"),
            tunnel_port=data.get("tunnel_port"),
            tunnel_pid=data.get("tunnel_pid"),
            created=data.get("created"),
            slurm_job_id=data.get("slurm_job_id"),
            compute_node=data.get("compute_node"),
            remote_mirror_root=data.get("remote_mirror_root"),
        )

    def save(self) -> None:
        """Write session state to disk."""
        directory = _session_dir()
        directory.mkdir(parents=True, exist_ok=True)
        if not self.created:
            self.created = _dt.datetime.now(_dt.timezone.utc).isoformat()
        path = directory / f"{self._session_key}.yaml"
        data = {
            "login_node": self.login_node,
            "tunnel_port": self.tunnel_port,
            "tunnel_pid": self.tunnel_pid,
            "created": self.created,
            "slurm_job_id": self.slurm_job_id,
            "compute_node": self.compute_node,
            "remote_mirror_root": self.remote_mirror_root,
        }
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, default_flow_style=False)

    def clear(self) -> None:
        """Remove the session file."""
        path = _session_dir() / f"{self._session_key}.yaml"
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    # Tunnel liveness
    # ------------------------------------------------------------------

    def tunnel_alive(self) -> bool:
        """Return True if the recorded tunnel PID is still running."""
        if self.tunnel_pid is None:
            return False
        try:
            os.kill(self.tunnel_pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False
