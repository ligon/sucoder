"""Selected private proof outputs, independent of Git and conversation state."""
import hashlib
import json
import os
import re
import stat
import time

try:
    from .dispatch_protocol import SCHEMA_VERSION, atomic_json
except ImportError:  # staged standard-library helper
    from dispatch_protocol import SCHEMA_VERSION, atomic_json


def publish_evidence(directory, required=None, max_file_bytes=64 * 1024 * 1024,
                     max_total_bytes=256 * 1024 * 1024):
    """Seal metadata for explicitly selected outputs already in evidence/.

    The private directory is the selection boundary: never scan the working
    tree, home, data cache or ignored files. A missing required file or a file
    changing during hashing is an incomplete publication, not lost evidence.
    """
    root = os.path.join(directory, "evidence")
    os.makedirs(root, mode=0o700, exist_ok=True)
    files, errors, total = [], [], 0
    if os.path.islink(root):
        errors.append("evidence root must not be a symlink")
    else:
        for parent, dirs, names in os.walk(root, followlinks=False):
            for name in list(dirs):
                if os.path.islink(os.path.join(parent, name)):
                    errors.append("symlink directory: " + os.path.relpath(os.path.join(parent, name), root))
                    dirs.remove(name)
            for name in sorted(names):
                path = os.path.join(parent, name)
                relative = os.path.relpath(path, root).replace(os.sep, "/")
                try:
                    before = os.lstat(path)
                    if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                            or before.st_mode & 0o077):
                        raise ValueError("not a private owned regular file")
                    if re.search(r"\.(?:sqlite[0-9]*|db)(?:-(?:wal|shm|journal))?$", name, re.I):
                        raise ValueError("database state is not a verification artifact")
                    if before.st_size > max_file_bytes or total + before.st_size > max_total_bytes:
                        raise ValueError("evidence size limit exceeded")
                    digest = hashlib.sha256()
                    read_bytes = 0
                    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                    with os.fdopen(fd, "rb") as stream:
                        opened = os.fstat(stream.fileno())
                        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                            raise ValueError("evidence changed before hashing")
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            read_bytes += len(block)
                            if read_bytes > max_file_bytes or total + read_bytes > max_total_bytes:
                                raise ValueError("evidence size limit exceeded during hashing")
                            digest.update(block)
                        after = os.fstat(stream.fileno())
                    named = os.lstat(path)
                    if (named.st_dev, named.st_ino) != (after.st_dev, after.st_ino):
                        raise ValueError("evidence replaced during hashing")
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError("evidence changed during hashing")
                    files.append({"name": relative, "size": after.st_size, "sha256": digest.hexdigest()})
                    total += after.st_size
                except (OSError, ValueError) as exc:
                    errors.append(relative + ": " + str(exc))
    present = {entry["name"] for entry in files}
    missing = sorted(set(required or []) - present)
    path = os.path.join(directory, "evidence-manifest.json")
    result = {"schema_version": SCHEMA_VERSION, "state": "incomplete" if errors or missing else "complete",
              "manifest_path": path, "published_at": time.time(), "files": sorted(files, key=lambda value: value["name"]),
              "count": len(files), "bytes": total, "missing": missing, "errors": errors,
              "selection": "private evidence directory only"}
    atomic_json(path, result)
    return result


def read_task_outcome(directory):
    """A worker's report is a claim; coordinator scientific acceptance is separate."""
    path = os.path.join(directory, "task-result.json")
    try:
        info = os.lstat(path)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or info.st_size > 65536):
            raise ValueError("task report is not a private bounded regular file")
        with open(path) as stream:
            report = json.load(stream)
        if (not isinstance(report, dict) or report.get("status") not in
                ("completed", "blocked", "unfinished", "scientific_disagreement")):
            raise ValueError("invalid task outcome")
        if not isinstance(report.get("summary", ""), str):
            raise ValueError("task summary must be text")
        unexecuted = report.get("unexecuted", [])
        if not isinstance(unexecuted, list) or any(not isinstance(value, str) for value in unexecuted):
            raise ValueError("unexecuted checks must be a list of text")
        return {"status": report["status"], "summary": report.get("summary", ""),
                "unexecuted": unexecuted, "source": "worker_report", "coordinator_acceptance": "pending"}
    except FileNotFoundError:
        return {"status": "not_reported", "coordinator_acceptance": "pending"}
    except (OSError, ValueError) as exc:
        return {"status": "unreadable", "reason": str(exc), "coordinator_acceptance": "pending"}
