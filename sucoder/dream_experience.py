"""This mirror's finished agent sessions, for experience dreams (essay-alignment S3).

Discovery is scoped, not a home-directory crawl: Claude keeps a session's
transcript in a project directory derived from its working directory
(attention.claude_transcript_path), so only the directories derived from the
mirror and its registered git worktrees are listed. A session counts as
finished when no live local launch uses it and its file has been quiet for
``dream.session_quiet_minutes``. Consumption comes from the receipts already
on the memory branch, which record each admitted window's raw range.
Codex discovery is deferred (S3b): managed rollouts live in node-local state.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import time

from .dream import DreamError, _json
from . import dream_view

SESSION = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
MAX_RECEIPTS = 256


def project_dir(home, cwd):
    """Claude's project directory for ``cwd`` (attention.claude_transcript_path's rule)."""
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(cwd))
    return None if len(slug) > 200 else Path(home) / ".claude" / "projects" / slug


def workspaces(source):
    """The mirror and its registered worktrees, as git reports them."""
    listed = source.git("worktree", "list", "--porcelain").decode("utf-8", errors="replace")
    paths = [line[len("worktree "):] for line in listed.splitlines() if line.startswith("worktree ")]
    return [str(source.path)] + [p for p in paths if p != str(source.path)]


def consumed(source, memory_commit):
    """Raw ranges of session windows already admitted by published generations."""
    ranges = {}
    if memory_commit is None:
        return ranges
    names = source.git("ls-tree", "-z", "--name-only", memory_commit, "--", ".sucoder/wiki/receipts/")
    for name in sorted(n for n in names.decode().split("\0") if n.endswith(".json"))[-MAX_RECEIPTS:]:
        oid = source.entry(memory_commit, name)
        receipt = _json(source.blob(oid))
        for unit in receipt.get("units", []) if isinstance(receipt, dict) else []:
            if isinstance(unit, dict) and unit.get("kind") == dream_view.KIND and isinstance(unit.get("raw"), dict):
                ranges.setdefault(unit.get("thread_id"), []).append(
                    [unit["raw"].get("byte_start"), unit["raw"].get("byte_end")])
    return {key: sorted(value) for key, value in ranges.items()}


def sessions(source, *, home=None, quiet_minutes=30, live=(), memory_commit=None, now=None):
    """Finished Claude sessions of this mirror, newest first, with consumed ranges.

    ``live`` holds session IDs of running launches (attention.scan_local_launches);
    a live or recently written session is listed as excluded, never admitted.
    """
    if type(quiet_minutes) is not int or quiet_minutes < 1:
        raise DreamError("session_quiet_minutes must be a positive integer")
    home = Path(home or os.path.expanduser("~"))
    now = time.time() if now is None else now
    done = consumed(source, memory_commit)
    found, seen = [], set()
    for workspace in workspaces(source):
        directory = project_dir(home, workspace)
        if directory is None or not directory.is_dir() or directory.is_symlink():
            continue
        for entry in sorted(directory.iterdir()):
            if entry.suffix != ".jsonl" or not SESSION.fullmatch(entry.stem) or entry.is_symlink() or entry.stem in seen:
                continue
            info = entry.stat()
            if not entry.is_file() or not info.st_size:
                continue
            seen.add(entry.stem)
            state = ("live" if entry.stem in live else
                     "recent" if now - info.st_mtime < quiet_minutes * 60 else "finished")
            found.append(dict(path=str(entry), thread_id=entry.stem, workspace=workspace, bytes=info.st_size,
                              modified=info.st_mtime, state=state, consumed=done.get(entry.stem, [])))
    return sorted(found, key=lambda item: (-item["modified"], item["thread_id"]))
