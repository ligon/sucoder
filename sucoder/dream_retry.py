"""Explicit elapsed windows; original budget identity and call cap stay frozen."""
import math
import os
import re

from . import dream, dream_run
from .dream import DreamError, canonical, digest


def validate_request(job_path, attempt, seconds):
    if seconds is not None and (job_path is None or type(attempt) is not int or attempt < 2 or
            type(seconds) is not int or not 1 <= seconds <= 86400):
        raise DreamError("--retry-seconds requires --job, --attempt 2 or later and 1-86400 seconds")


def windows(fd, header, policy_sha256):
    result = []
    previous = policy_sha256
    for name in sorted(n for n in os.listdir(fd) if re.fullmatch(r"retry-[0-9]{7}\.json", n)):
        saved = dream._json(dream_run._read(fd, name, 4096))
        dream._object(saved, ("window", "sha256"))
        row = saved["window"]
        dream._object(row, ("job_id", "policy_sha256", "previous_sha256", "attempt", "seconds",
            "started", "deadline", "monotonic_deadline"))
        validate_request(True, row["attempt"], row["seconds"])
        started = row["started"]
        dream._object(started, ("wall", "monotonic", "boot_id"))
        if (not isinstance(started["boot_id"], str) or not started["boot_id"] or
                any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                    for v in (started["wall"], started["monotonic"], row["deadline"], row["monotonic_deadline"])) or
                row["deadline"] != started["wall"] + row["seconds"] or
                row["monotonic_deadline"] != started["monotonic"] + row["seconds"] or
                row["seconds"] <= header["limits"]["reserve_seconds"] or
                row["job_id"] != header["job_id"] or row["policy_sha256"] != policy_sha256 or
                row["previous_sha256"] != previous or saved["sha256"] != digest(canonical(row)) or
                name != f"retry-{row['attempt']:07d}.json"):
            raise DreamError("Invalid recorded dream retry window", 3)
        result.append(row)
        previous = saved["sha256"]
    return result


def renew(job, attempt, seconds, now):
    validate_request(True, attempt, seconds)
    if job.readonly or seconds <= job.header["limits"]["reserve_seconds"]:
        raise DreamError("Retry allowance must exceed closeout reserves and requires a writable budget", 3)
    if job.retry_windows and attempt <= job.retry_windows[-1]["attempt"]:
        row = next((r for r in job.retry_windows if r["attempt"] == attempt), None)
        if row is not None and row["seconds"] == seconds:
            return  # Explicit repeats retain the recorded deadline; never top it up.
        raise DreamError("Retry window is immutable; choose a new --attempt", 3)
    if now["wall"] < max([job.header["started"]["wall"]] +
            [r["reservation"]["reserved_at"] for r in job.calls] +
            [r["started"]["wall"] for r in job.retry_windows]):
        raise DreamError("Cannot authorize a retry after the wall clock moved backwards", 3)
    previous = digest(canonical(job.retry_windows[-1])) if job.retry_windows else job.policy_sha256
    row = dict(job_id=job.header["job_id"], policy_sha256=job.policy_sha256,
        previous_sha256=previous, attempt=attempt, seconds=seconds, started=now,
        deadline=now["wall"] + seconds, monotonic_deadline=now["monotonic"] + seconds)
    dream_run._store(job.fd, f"retry-{attempt:07d}.json", canonical(dict(window=row, sha256=digest(canonical(row)))) + b"\n")
    job.retry_windows.append(row)
