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
    # Port forwards created by `tunnel forward` (kept on the target's
    # ``tunnel-<target>`` session).  Each entry is a dict:
    # ``{"local_port": int, "node": str, "remote_port": int}``.
    forwards: list = field(default_factory=list)

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
            forwards=list(data.get("forwards") or []),
        )

    def save(self) -> None:
        """Write session state to disk *atomically*.

        Writes to a temp file in the same directory and ``os.replace``s it
        into place, so a mid-write failure (disk full, crash) can never leave
        a truncated session file.  This matters because a half-written file
        loads as a blank session (``load`` swallows parse errors), which for
        a SLURM target would drop the recorded ``slurm_job_id`` and make the
        next ``collaborate`` resubmit -- leaking the running job.
        """
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
            "forwards": self.forwards,
        }
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, default_flow_style=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    @classmethod
    def holders_of_job(cls, job_id, exclude_key: Optional[str] = None) -> list:
        """Return session keys that reference *job_id*.

        A key is ``mirror`` or ``mirror--target`` (the filename stem).
        Used by ``sucoder release`` to detect when a SLURM job is shared
        by several sessions (e.g. two mirrors co-resident on one node) so
        it can detach this mirror instead of cancelling a job the others
        still need.  *exclude_key* omits the caller's own session.
        """
        holders: list = []
        if job_id is None:
            return holders
        directory = _session_dir()
        if not directory.is_dir():
            return holders
        for path in sorted(directory.glob("*.yaml")):
            stem = path.stem
            if exclude_key is not None and stem == exclude_key:
                continue
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
            except (yaml.YAMLError, OSError):
                continue
            if isinstance(data, dict) and data.get("slurm_job_id") == job_id:
                holders.append(stem)
        return holders

    @classmethod
    def compute_nodes_for_target(cls, target_name: str) -> dict:
        """Map session key -> compute node for sessions on *target_name*.

        Scans ``~/.sucoder/sessions/<mirror>--<target>.yaml`` (collaborate
        sessions; the target's own ``tunnel-<target>.yaml`` has no ``--``
        and is naturally excluded) and returns the entries that record a
        compute node.  Used by ``tunnel forward`` to default the target
        node to "the node my session is on" without a ``--node`` flag.
        Entries may be stale if a job ended without ``release``; callers
        surface connection errors rather than pre-validating here.
        """
        nodes: dict = {}
        directory = _session_dir()
        if not directory.is_dir():
            return nodes
        for path in sorted(directory.glob(f"*--{target_name}.yaml")):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
            except (yaml.YAMLError, OSError):
                continue
            if isinstance(data, dict) and data.get("compute_node"):
                nodes[path.stem] = data["compute_node"]
        return nodes

    @classmethod
    def login_nodes_for_target(cls, target_name: str) -> dict:
        """Map session key -> login node for mirror sessions on *target_name*.

        Scans ``<mirror>--<target>.yaml`` (the ``tunnel-<target>.yaml`` has no
        ``--`` and is excluded) and returns entries that record a login node.
        Used by ``tunnel doctor`` to flag a mirror pin that has drifted from
        the warm tunnel node.
        """
        nodes: dict = {}
        directory = _session_dir()
        if not directory.is_dir():
            return nodes
        for path in sorted(directory.glob(f"*--{target_name}.yaml")):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
            except (yaml.YAMLError, OSError):
                continue
            if isinstance(data, dict) and data.get("login_node"):
                nodes[path.stem] = data["login_node"]
        return nodes

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
