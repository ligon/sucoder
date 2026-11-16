"""Bounded preparation of the existing scoped session inventory (ledger 2, 5)."""
from __future__ import annotations

import os
from pathlib import Path
import stat
import time

from . import dream_experience as experience, dream_isolate, dream_sessions, dream_transcript
from .dream import DreamError


def _tail(path, info, deadline, limits, coverage):
    end, used = info.st_size, 0
    with os.fdopen(dream_transcript._open_source(path), "rb", buffering=0) as stream:
        current = os.fstat(stream.fileno())
        if (not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino) or
                current.st_size < end):
            raise DreamError("source_identity_changed")
        while end and used < limits["tail_file_bytes"]:
            if time.monotonic() >= deadline - 0.025:
                raise DreamError("inventory_deadline")
            count = min(limits["chunk_bytes"], end, limits["tail_file_bytes"] - used,
                        limits["tail_total_bytes"] - coverage["tail_bytes"])
            if count <= 0:
                raise DreamError("inventory_probe_limit")
            stream.seek(end - count)
            data = stream.read(count)
            coverage["tail_bytes"] += len(data)
            used += len(data)
            if len(data) != count:
                raise DreamError("source_shrank")
            cut = data.rfind(b"\n")
            if cut >= 0:
                return end - count + cut + 1
            end -= count
    if end == 0:
        return 0
    raise DreamError("complete_prefix_probe_limit")


def collect(source, *, home, quiet_minutes, live, memory_commit, config, limits, deadline):
    coverage = dict(state="complete", entries=0, metadata_operations=0, candidates=0, tail_bytes=0,
                    unprobed_candidates_lower_bound=0, undiscovered_sources=None, reasons=[])
    candidates, rows, seen = [], [], set()

    def take(kind, ceiling):
        if time.monotonic() >= deadline - 0.025:
            reason = "inventory_deadline"
        elif coverage[kind] >= limits[ceiling]:
            reason = ceiling + "_limit"
        else:
            coverage[kind] += 1
            return True
        coverage["state"] = "incomplete"
        if reason not in coverage["reasons"]:
            coverage["reasons"].append(reason)
        return False

    if time.monotonic() >= deadline - 0.025:
        coverage.update(state="incomplete", reasons=["inventory_deadline"])
        return dict(sessions=[], coverage=coverage)
    done = experience.consumed(source, memory_commit)
    known = experience.workspaces(source)
    historical = dream_sessions.workspaces(source, config, home)
    stop = False
    for workspace in known + [p for p in historical if p not in known]:
        if stop:
            break
        directory = experience.project_dir(home, workspace)
        if directory is None:
            continue
        if not take("metadata_operations", "inventory_metadata"):
            break
        try:
            info = os.stat(directory, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode):
                continue
            if directory.resolve() != directory:
                coverage["state"] = "incomplete"
                coverage["reasons"] = list(dict.fromkeys(coverage["reasons"] + ["directory_noncanonical"]))
                continue
            with os.scandir(directory) as entries:
                while True:
                    # Check before next(): even invalid names spend the entry allowance.
                    if not take("entries", "inventory_entries"):
                        stop = True
                        break
                    try:
                        entry = next(entries)
                    except StopIteration:
                        coverage["entries"] -= 1
                        break
                    if not entry.name.endswith(".jsonl") or not experience.SESSION.fullmatch(entry.name[:-6]):
                        continue
                    if coverage["candidates"] >= limits["inventory_candidates"]:
                        coverage.update(state="incomplete")
                        coverage["reasons"].append("inventory_candidates_limit")
                        stop = True
                        break
                    if not take("metadata_operations", "inventory_metadata"):
                        stop = True
                        break
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except OSError:
                        coverage["state"] = "incomplete"
                        coverage["reasons"] = list(dict.fromkeys(coverage["reasons"] + ["metadata_unavailable"]))
                        continue
                    if stat.S_ISREG(info.st_mode) and info.st_size and entry.name[:-6] not in seen:
                        seen.add(entry.name[:-6])
                        candidates.append((entry.name[:-6], Path(entry.path), workspace, info))
                        coverage["candidates"] += 1
        except FileNotFoundError:
            continue
        except OSError:
            coverage["state"] = "incomplete"
            coverage["reasons"] = list(dict.fromkeys(coverage["reasons"] + ["directory_unavailable"]))
    identity_bytes = 0
    now = time.time()
    for number, (sid, path, workspace, info) in enumerate(sorted(candidates)):
        if time.monotonic() >= deadline - 0.025 or coverage["tail_bytes"] >= limits["tail_total_bytes"]:
            coverage["state"] = "incomplete"
            coverage["unprobed_candidates_lower_bound"] = len(candidates) - number
            coverage["reasons"] = list(dict.fromkeys(coverage["reasons"] + ["inventory_preparation_limit"]))
            break
        if workspace not in known:
            try:
                confirmed, used = dream_sessions.belongs(path, workspace, sid, deadline,
                    dream_sessions.MAX_IDENTITY_SCAN - identity_bytes)
                identity_bytes += used
            except DreamError:
                coverage["state"] = "incomplete"
                coverage["reasons"] = list(dict.fromkeys(coverage["reasons"] + ["historical_identity_limit"]))
                continue
            if not confirmed:
                continue
        row = dict(path=str(path), thread_id=sid, workspace=workspace, bytes=info.st_size,
            modified=info.st_mtime, consumed=done.get(sid, []), source_identity=[info.st_dev, info.st_ino])
        try:
            row["bytes"] = _tail(path, info, deadline, limits, coverage)
            if not row["bytes"]:
                raise DreamError("no_complete_prefix")
            row["state"] = "live" if sid in live else "recent" if now - info.st_mtime < quiet_minutes * 60 else "finished"
        except (OSError, DreamError) as exc:
            row.update(state="unavailable", reason=str(exc) if isinstance(exc, DreamError) else "source_unavailable")
            coverage["state"] = "incomplete"
        rows.append(row)
    if coverage["state"] == "complete":
        coverage["undiscovered_sources"] = 0
    return dict(sessions=sorted(rows, key=lambda r: (-r["modified"], r["thread_id"])), coverage=coverage)


def prepare(source, *, config, header, live, home=None):
    limits = header["navigation_limits"]
    policy = header["processing"]
    deadline = min(source.deadline, time.monotonic() + policy["max_seconds"])
    try:
        return dream_isolate.run(lambda: collect(source, home=Path(home or os.path.expanduser("~")),
            quiet_minutes=header["settings"].get("session_quiet_minutes", 30), live=live() if callable(live) else live,
            memory_commit=header.get("inventory_memory_commit"), config=config, limits=limits, deadline=deadline),
            deadline, policy)
    except dream_isolate.ProcessingError as exc:
        return dict(sessions=[], coverage=dict(state="unavailable", reasons=[exc.reason], entries=None,
            metadata_operations=None, candidates=None, tail_bytes=None, undiscovered_sources=None,
            unprobed_candidates_lower_bound=0, actual_reads="unknown", limits=limits))
