"""Interruptible local evidence processing (navigation ledger, sections 4-5).

The renderer's bytes are an existing contract. Isolate its execution instead
of changing redaction semantics or relying on checks between JSONL records.
Only trusted local callables run here; no provider or shell is invoked.
"""
from __future__ import annotations

import math
import os
import resource
import selectors
import signal
import time

from .dream import DreamError, _json, _object, canonical


POLICY = dict(contract="bounded-dream-processing-v1", max_seconds=5,
    memory_headroom=256 * 1024 * 1024, max_output_bytes=2 * 1024 * 1024)
CODES = frozenset(("processing_deadline", "processing_memory_limit", "processing_failed", "processing_output_limit",
    "processing_isolation_unavailable", "source_processing_error", "continued_record_changed", "source_identity_changed",
    "source_shrank", "selection_processing_limit", "selection_record_limit"))


class ProcessingError(DreamError):
    def __init__(self, reason, detail=""):
        super().__init__(detail or reason, 3)
        self.reason, self.detail = reason, detail


def validate(policy):
    _object(policy, tuple(POLICY))
    if policy["contract"] != POLICY["contract"]:
        raise DreamError("Unknown frozen evidence processing policy", 3)
    for key in ("max_seconds", "memory_headroom", "max_output_bytes"):
        value = policy[key]
        if (type(value) not in (int, float) or not math.isfinite(value) or
                not 0 < value <= POLICY[key] or (key != "max_seconds" and type(value) is not int)):
            raise DreamError("Invalid frozen evidence processing allowance", 3)
    return policy


def _child(function, fd, seconds, policy):
    os.setsid()
    # Group cleanup also covers Git subprocesses if the coordinator disappears.
    def expired(signum, frame):
        try:
            os.set_blocking(fd, False)
            os.write(fd, b'{"state":"error","reason":"processing_deadline"}')
        except OSError:
            pass
        os.killpg(os.getpgrp(), signal.SIGKILL)
    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGXCPU, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        with open("/proc/self/statm", "rb") as stream:
            virtual = int(stream.read(128).split()[0]) * os.sysconf("SC_PAGE_SIZE")
        ceiling = virtual + policy["memory_headroom"]
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            ceiling = min(ceiling, hard)
        resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))
        cpu = max(1, math.ceil(seconds))
        _, hard = resource.getrlimit(resource.RLIMIT_CPU)
        if hard != resource.RLIM_INFINITY:
            cpu = min(cpu, hard)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, min(cpu + 1, hard) if hard != resource.RLIM_INFINITY else cpu + 1))
        try:
            payload = canonical(dict(state="complete", value=function()))
        except DreamError as exc:
            detail = str(exc)[:4096]
            code = detail if detail in CODES else "source_processing_error"
            while len(canonical(detail)) > 256:
                detail = detail[:len(detail) // 2]
            payload = canonical(dict(state="error", reason=code, detail=detail))
        except (MemoryError, RecursionError):
            payload = canonical(dict(state="error", reason="processing_memory_limit"))
        except Exception:
            payload = canonical(dict(state="error", reason="processing_failed"))
        if len(payload) > policy["max_output_bytes"]:
            payload = canonical(dict(state="error", reason="processing_output_limit"))
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
    finally:
        os.close(fd)


def _kill(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        # The parent may time out before the child has reached setsid().
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run(function, deadline, policy=None):
    """Return bounded JSON data, or terminate and reap the local worker.

    Pipes carry JSON, never pickle. The parent enforces time and output even
    when the child cannot allocate, blocks in I/O, or hangs inside a regex.
    """
    policy = validate(POLICY if policy is None else policy)
    started = time.monotonic()
    stop = min(deadline, started + policy["max_seconds"])
    seconds = stop - started
    if seconds <= 0:
        raise ProcessingError("processing_deadline")
    if not hasattr(os, "fork"):
        raise ProcessingError("processing_isolation_unavailable")
    read_fd, write_fd = os.pipe()
    try:
        pid = os.fork()
    except BaseException:
        os.close(read_fd)
        os.close(write_fd)
        raise
    if pid == 0:
        os.close(read_fd)
        try:
            _child(function, write_fd, max(0.001, stop - time.monotonic()), policy)
        finally:
            os._exit(0)
    os.close(write_fd)
    data = bytearray()
    status = None
    try:
        os.set_blocking(read_fd, False)
        with selectors.DefaultSelector() as selector:
            selector.register(read_fd, selectors.EVENT_READ)
            while True:
                left = stop - time.monotonic()
                if left <= 0 or not selector.select(left):
                    raise ProcessingError("processing_deadline")
                block = os.read(read_fd, min(65536, policy["max_output_bytes"] + 1 - len(data)))
                if not block:
                    break
                data.extend(block)
                if len(data) > policy["max_output_bytes"]:
                    raise ProcessingError("processing_output_limit")
        # Kill descendants before reaping the leader, whose PID still owns the
        # process-group identity. A callable cannot leave background work behind.
        _kill(pid)
        _, status = os.waitpid(pid, 0)
        if not data:
            timed = os.WIFSIGNALED(status) and os.WTERMSIG(status) in (signal.SIGALRM, signal.SIGXCPU)
            raise ProcessingError("processing_deadline" if timed else "processing_failed")
        try:
            value = _json(bytes(data))
        except DreamError:
            raise ProcessingError("processing_deadline" if time.monotonic() >= stop - 0.05 else "processing_failed") from None
        if value.get("state") != "complete":
            code = value.get("reason", "processing_failed")
            raise ProcessingError(code if code in CODES else "processing_failed", value.get("detail", ""))
        return value["value"]
    finally:
        os.close(read_fd)
        if status is None:
            _kill(pid)
            os.waitpid(pid, 0)
