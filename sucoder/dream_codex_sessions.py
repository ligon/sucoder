"""Bounded, mirror-scoped inventory for Codex session reader jobs."""
from __future__ import annotations

import os
from pathlib import Path
import re
import stat
import time

from . import codex_state, dream_codex_view, dream_experience, dream_sessions, dream_transcript, reply_probe
from .dream import DreamError, _json

MAX_ENTRIES = 16384
MAX_IDENTITY_BYTES = dream_sessions.MAX_IDENTITY_BYTES
MAX_SCAN = 32 * MAX_IDENTITY_BYTES
LOCAL_NAME = re.compile(r"rollout-[^/]+-(" + dream_experience.SESSION.pattern + r")\.jsonl")


def live_threads(proc_root=Path("/proc")):
    """Codex can hold an idle rollout open without naming its UUID on the launch command line."""
    found, inspected = set(), 0
    if not proc_root.is_dir():
        return found
    with os.scandir(proc_root) as processes:
        for proc in processes:
            inspected += 1
            if inspected > MAX_ENTRIES:
                raise DreamError("Codex live thread scan exceeds its entry allowance", 3)
            if not proc.name.isdigit():
                continue
            try:
                if proc.stat().st_uid != os.getuid():
                    continue
                executable = Path(os.readlink(Path(proc.path) / "exe")).name
                if executable != "codex" and not executable.startswith("codex-"):
                    continue
                with os.scandir(Path(proc.path) / "fd") as descriptors:
                    for fd in descriptors:
                        inspected += 1
                        if inspected > MAX_ENTRIES:
                            raise DreamError("Codex live thread scan exceeds its entry allowance", 3)
                        try:
                            match = LOCAL_NAME.fullmatch(Path(os.readlink(fd.path)).name)
                        except OSError:
                            continue
                        if match:
                            found.add(match[1])
            except OSError:
                continue  # The process exited or the kernel hides its descriptors.
    return found


class _Scan:
    def __init__(self):
        self.entries = 0
        self.remaining = MAX_SCAN

    def children(self, directory):
        if not directory.is_dir() or directory.resolve() != directory:
            return
        with os.scandir(directory) as entries:
            for entry in entries:
                self.entries += 1
                if self.entries > MAX_ENTRIES:
                    raise DreamError("Codex session inventory exceeds its entry allowance", 3)
                if not entry.is_symlink():
                    yield Path(entry.path)

    def read(self, stream, first_line=False):
        limit = min(MAX_IDENTITY_BYTES, self.remaining)
        if not limit:
            raise DreamError("Codex session identity scan allowance exhausted", 3)
        data = stream.readline(limit) if first_line else stream.read(limit)
        self.remaining -= len(data)
        if len(data) == limit and not (first_line and data.endswith(b"\n")):
            raise DreamError("Codex session metadata exceeds its scan allowance", 3)
        return data

    def metadata(self, path):
        try:
            with os.fdopen(dream_transcript._open_source(path), "rb") as stream:
                if stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raw = self.read(stream)
                    try:
                        value = _json(raw)
                    except DreamError:
                        return {}
                    return value if isinstance(value, dict) else {}
        except OSError:
            pass
        return {}


def _roots(source, config):
    """Configured Slurm roots and locally visible allocation disks, plus the default replica root."""
    roots = {}
    if config is not None:
        mirror = config.mirrors.get(source.name)
        for target in [*config.targets.values(), getattr(mirror, "remote", None)]:
            if target is None or target.slurm is None:
                continue
            settings = target.slurm
            configured = settings.codex_state.durable_root if settings.codex_state else None
            root = Path(codex_state.durable_root_for(str(source.path), configured))
            disks = roots.setdefault(root, set())
            if settings.local_disk:
                disks.add(Path(settings.local_disk).expanduser())
    return roots or {Path(codex_state.durable_root_for(str(source.path), None)): set()}


def _node_copies(scan, run, disks):
    scope = run.parent.parent.name
    allocation = scan.metadata(run / "run.json").get("allocation")
    job = allocation.get("job") if isinstance(allocation, dict) else None
    if not re.fullmatch(r"[0-9]{1,20}", str(job)):
        return
    for disk in sorted(disks):
        local = Path(codex_state.local_scope_dir(str(disk), int(job), scope)) / run.name
        replica = scan.metadata(local / "status.json").get("replica")
        threads = replica.get("threads") if isinstance(replica, dict) else None
        if not isinstance(threads, dict):
            continue
        for tid, info in threads.items():
            if not dream_experience.SESSION.fullmatch(tid) or not isinstance(info, dict):
                continue
            if any(type(info.get(key, 0)) is not int for key in ("local_bytes", "replicated_bytes")):
                continue
            rel = info.get("relative_path")
            if not isinstance(rel, str):
                continue
            match = re.fullmatch(r"[0-9]{4}/[0-9]{2}/[0-9]{2}/" + LOCAL_NAME.pattern, rel)
            if not match or match[1] != tid:
                continue
            candidate = local / "home/sessions" / rel
            durable = run / "rollouts" / (tid + ".jsonl")
            if (candidate.resolve() == candidate and candidate.is_file() and os.access(candidate, os.R_OK) and
                    reply_probe.replica_needs_local(info, durable)):
                yield tid, candidate


def _managed(scan, roots):
    for root, disks in roots.items():
        for scope in scan.children(root / "codex"):
            for run in scan.children(scope / "runs"):
                marker = run / "run.json"
                if marker.is_symlink() or not marker.is_file():
                    continue
                paths = {p.stem: p for p in scan.children(run / "rollouts")
                         if p.suffix == ".jsonl" and dream_experience.SESSION.fullmatch(p.stem)}
                if disks:
                    paths.update(_node_copies(scan, run, disks))
                yield from paths.items()


def _unmanaged(scan, home):
    dirs = [home / ".codex/sessions"]
    for pattern in (r"[0-9]{4}", r"[0-9]{2}", r"[0-9]{2}"):
        dirs = [path for parent in dirs for path in scan.children(parent) if re.fullmatch(pattern, path.name)]
    for day in dirs:
        for path in scan.children(day):
            match = LOCAL_NAME.fullmatch(path.name)
            if match:
                yield match[1], path


def _complete(stream, size, scan):
    """Charge any partial-line search to the shared scan budget."""
    end, step = size, 1
    while end:
        length = min(end, step, scan.remaining)
        if not length:
            raise DreamError("Codex complete-line scan allowance exhausted", 3)
        end -= length
        stream.seek(end)
        data = stream.read(length)
        if len(data) != length:
            raise DreamError("Codex rollout shrank during discovery", 3)
        scan.remaining -= length
        cut = data.rfind(b"\n")
        if cut >= 0:
            return end + cut + 1
        step = 65536
    return 0


def _inspect(scan, filename, tid, workspace, live, excluded, quiet_seconds, now):
    with os.fdopen(dream_transcript._open_source(Path(filename)), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            return None
        line = scan.read(stream, first_line=True)
        if not line.endswith(b"\n"):
            return None
        try:
            payload = dream_codex_view.meta(line)
        except DreamError:
            return None
        if payload.get("id") != tid or payload.get("cwd") != workspace:
            return None
        reason = dream_codex_view.exclusion(payload)
        if any(tid.startswith(prefix) for prefix in excluded):
            reason = "dream.reader_exclude_sessions"
        row = dict(harness="codex", thread_id=tid, path=filename, workspace=workspace,
                   modified=info.st_mtime, consumed=[], device=info.st_dev, inode=info.st_ino)
        if reason:
            return dict(row, state="excluded", reason=reason, bytes=info.st_size)
        state = "live" if tid in live else "recent" if now - info.st_mtime < quiet_seconds else "finished"
        return dict(row, state=state, bytes=_complete(stream, info.st_size, scan),
                    binding=dream_codex_view.identity(line, tid))


def sessions(source, *, config=None, home=None, quiet_minutes=30, live=(), excluded=(), now=None):
    if type(quiet_minutes) is not int or quiet_minutes < 1:
        raise DreamError("session_quiet_minutes must be a positive integer")
    now = time.time() if now is None else now
    scan, rows = _Scan(), {}
    for inventory in (_managed(scan, _roots(source, config)), _unmanaged(scan, Path(home or Path.home()))):
        candidates = []
        for tid, path in inventory:
            if tid in rows or path.resolve() != path:
                continue
            try:
                info = path.stat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode):
                candidates.append((info.st_mtime_ns, str(path), tid))
        for _, filename, tid in sorted(candidates, reverse=True):
            if tid in rows:
                continue
            try:
                row = _inspect(scan, filename, tid, str(source.path), live, excluded, quiet_minutes * 60, now)
                if row is not None:
                    rows[tid] = row
            except FileNotFoundError:
                continue
    return sorted(rows.values(), key=lambda row: (-row["modified"], row["thread_id"]))
