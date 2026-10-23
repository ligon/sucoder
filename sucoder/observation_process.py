"""Bounded, noninteractive subprocesses for observation only (Python 3.9)."""
from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass


ISOLATED_WORKER = False


class ObservationFailure(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class Budget:
    def __init__(self, seconds):
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.cutoff = self.deadline - min(2, seconds * .2)

    def remaining(self):
        return max(0, self.cutoff - time.monotonic())

    def check(self):
        if not self.remaining():
            raise ObservationFailure("TIMEOUT")


@dataclass
class Capture:
    returncode: int
    stdout: str
    stderr: str = ""


def capture(argv, budget, *, cwd=None, env=None, limit=1024 * 1024):
    """Drain both pipes without communicate's unbounded buffers/waits.

    Each child gets its own process group; existing SSH masters are never
    signalled. SIGKILL is used only for owned children at cleanup. Kernel
    uninterruptible I/O remains an OS limit; no wait without a deadline.
    """
    budget.check()
    try:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=not ISOLATED_WORKER)
    except OSError:
        raise ObservationFailure("ROUTE_UNAVAILABLE") from None
    streams = [bytearray(), bytearray()]
    selector = selectors.DefaultSelector()
    try:
        for i, pipe in enumerate((proc.stdout, proc.stderr)):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, i)
        while selector.get_map():
            budget.check()
            for key, _ in selector.select(min(.05, budget.remaining())):
                data = os.read(key.fileobj.fileno(), 65536)
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                streams[key.data].extend(data)
                if len(streams[key.data]) > limit:
                    raise ObservationFailure("OUTPUT_LIMIT")
        while proc.poll() is None:
            budget.check()
            time.sleep(min(.01, budget.remaining()))
        return Capture(proc.returncode, *(b.decode("utf-8", "replace") for b in streams))
    finally:
        selector.close()
        # Also remove descendants that inherited no pipe and outlived the leader.
        try:
            (os.kill if ISOLATED_WORKER else os.killpg)(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # Privileged relay children own a separate bounded supervisor.
            pass
        proc.stdout.close()
        proc.stderr.close()
        try:
            proc.wait(timeout=max(0, budget.deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
