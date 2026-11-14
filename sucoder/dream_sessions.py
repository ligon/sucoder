"""Recognize historical job-clone session directories without crawling homes."""
from __future__ import annotations

import os
from pathlib import Path
import re
import stat
import time

from . import dream, dream_transcript
from .config import sanitize_session_token
from .dream import DreamError
from .local_tier import work_path


MAX_PROJECT_DIRS = 4096
MAX_IDENTITY_BYTES = 1024 * 1024
MAX_IDENTITY_SCAN = 16 * MAX_IDENTITY_BYTES


def workspaces(source, config, home):
    """Only configured local-tier roots/tokens for this project can match."""
    if config is None:
        return []
    from .dream_experience import project_dir
    layouts, selected, mirrors = {}, set(), {}
    for name, mirror in config.mirrors.items():
        for target in config.targets.values():
            if not target.slurm or not target.slurm.local_disk:
                continue
            disk = str(Path(target.slurm.local_disk).expanduser().resolve())
            token = sanitize_session_token(name)
            durable = (target.mirror_root.expanduser() / mirror.mirror_dirname).resolve()
            key = disk, token
            layouts.setdefault(key, set()).add(str(durable))
            roots = {durable, mirror.canonical_repo.resolve(),
                (config.mirror_root.expanduser() / mirror.mirror_dirname).resolve()}
            local_job = source.path.parent.parent.name
            local_match = (source.path.parent.name == "mirrors" and re.fullmatch(r"job[0-9]+", local_job) and
                Path(work_path(disk, token, int(local_job[3:]))).resolve() == source.path)
            if source.path in roots or (local_match and name == source.name):
                if local_match:
                    origin = source.git("config", "--get", "remote.origin.url", missing=True)
                    if not origin or not Path(origin.decode().strip()).is_absolute() or Path(origin.decode().strip()).resolve() != durable:
                        continue
                selected.add(key)
                mirrors.setdefault(key, set()).add(str(durable))
    # A token collision between configured repositories cannot establish identity.
    selected = {key for key in selected if len(layouts[key]) == 1}
    if not selected:
        return []
    directory = Path(home) / ".claude/projects"
    if not directory.is_dir() or directory.is_symlink():
        return []
    slug = lambda value: re.sub(r"[^A-Za-z0-9]", "-", value)
    patterns = [(disk, token, re.compile(re.escape(slug(disk.rstrip("/") + "/job")) +
        r"([0-9]+)" + re.escape(slug("/mirrors/" + token)))) for disk, token in sorted(selected)]
    found = {path for key in selected for path in mirrors[key]}
    with os.scandir(directory) as entries:
        for count, entry in enumerate(entries, 1):
            if count > MAX_PROJECT_DIRS:
                raise DreamError("Claude project directory inventory exceeds its bounded allowance", 3)
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                continue
            for disk, token, pattern in patterns:
                match = pattern.fullmatch(entry.name)
                if match:
                    workspace = work_path(disk, token, int(match[1]))
                    expected = project_dir(home, workspace)
                    if expected is not None and expected.name == entry.name:
                        found.add(workspace)
    return sorted(found)


def belongs(path, workspace, thread_id, deadline, allowance):
    """Require native cwd, branch and session identity; the slug alone is lossy."""
    used = 0
    path = Path(path).absolute()
    if path != path.resolve():
        return False, used
    limit = min(MAX_IDENTITY_BYTES, allowance)
    if limit <= 0:
        raise DreamError("Historical session identity scan allowance exhausted", 3)
    try:
        with os.fdopen(dream_transcript._open_source(path), "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                return False, used
            while used < limit:
                if time.monotonic() >= deadline:
                    raise DreamError("Historical session identity inspection deadline exhausted", 3)
                line = stream.readline(limit - used + 1)
                used += len(line)
                if not line or used > limit or not line.endswith(b"\n"):
                    break
                try:
                    record = dream._json(line)
                except DreamError:
                    return False, used
                if not isinstance(record, dict) or record.get("sessionId") != thread_id:
                    continue
                cwd, branch = record.get("cwd"), record.get("gitBranch")
                if (cwd == workspace and isinstance(branch, str) and branch.strip() and
                        len(branch) <= 4096 and not any(ord(char) < 32 for char in branch)):
                    return True, used
    except OSError:
        return False, used
    return False, used
