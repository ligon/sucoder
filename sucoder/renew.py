"""Auto-renew / respawn controller for persistent SLURM sessions.

A ``sucoder collaborate`` session is pinned to a single SLURM allocation
and dies when that allocation ends -- at the courtesy ``--time`` or at a
maintenance reboot.  On a no-wall-clock condo QOS (e.g. ``co_carleton``
on ``savio4_htc``) the natural way to stay present is to *re-allocate*
when the slice turns over and relaunch the agent on the fresh node.

This module is the operator-side controller for that loop.  It is
deliberately free of any ``sucoder`` imports: all I/O (probing SLURM,
re-allocating, cancelling, signalling the agent, sleeping) is injected
by the caller, so the decision logic here is pure and unit-testable.
The CLI (`sucoder renew`) wires the real implementations.

Design notes live in ``docs/persistent-presence.org`` (Deliverable B).

Key robustness rule: a *failed probe* (e.g. a transient SSH blip to the
login node) is NOT treated as "job lost".  Only a probe that succeeds
and reports the job gone/terminal triggers a re-allocation -- otherwise
a network hiccup would needlessly tear down and replace a healthy
slice.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from typing import Callable, Optional

__all__ = [
    "parse_time_left",
    "JobStatus",
    "RenewSettings",
    "Action",
    "decide_action",
    "Backoff",
    "run_renew_loop",
]


# SLURM job states that mean "this allocation is finished"; a job in any
# of these (or absent from squeue) should trigger a fresh allocation.
# ``CANCELLED`` may appear as "CANCELLED by 12345", so match by prefix.
_TERMINAL_PREFIXES = (
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "SPECIAL_EXIT",
    "REVOKED",
)

# States that mean "still alive, keep holding" (running or on its way).
_ALIVE_PREFIXES = (
    "RUNNING",
    "PENDING",
    "CONFIGURING",
    "COMPLETING",
    "RESV_DEL_HOLD",
    "REQUEUED",
    "RESIZING",
    "SUSPENDED",
)


def parse_time_left(value: Optional[str]) -> Optional[int]:
    """Convert SLURM ``squeue -o %L`` (TIME_LEFT) to whole minutes.

    Mirrors the on-node ``left_to_mins`` bash helper.  ``%L`` renders as
    ``D-HH:MM:SS`` once a day or more remains, ``HH:MM:SS`` under a day,
    and ``MM:SS`` under an hour.  Returns ``None`` for unbounded/unknown
    values (``UNLIMITED``, ``INVALID``, ``N/A``, empty) so the caller
    treats them as "no finite deadline" rather than "expires now".
    """
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None

    days = 0
    if "-" in s:
        head, _, s = s.partition("-")
        if not head.isdigit():
            return None
        days = int(head)

    parts = s.split(":")
    if not parts or not all(p.isdigit() for p in parts):
        return None
    if len(parts) == 3:
        hours, minutes = int(parts[0]), int(parts[1])
    elif len(parts) == 2:
        # MM:SS -- seconds are dropped (we round down to whole minutes).
        hours, minutes = 0, int(parts[0])
    else:
        return None
    return days * 1440 + hours * 60 + minutes


@dataclass(frozen=True)
class JobStatus:
    """Outcome of probing a SLURM job.

    ``ok`` is False when the probe itself failed (SSH error, timeout) --
    distinct from a successful probe that found the job gone
    (``ok=True, state=None``).  The distinction is load-bearing: only a
    *successful* probe reporting absence/termination triggers relaunch.
    """

    ok: bool
    state: Optional[str] = None        # SLURM state token, or None if absent
    mins_left: Optional[int] = None    # minutes remaining, or None if unbounded

    @property
    def alive(self) -> bool:
        if not self.state:
            return False
        st = self.state.upper()
        return any(st.startswith(p) for p in _ALIVE_PREFIXES)

    @property
    def terminal(self) -> bool:
        if not self.state:
            return False
        st = self.state.upper()
        return any(st.startswith(p) for p in _TERMINAL_PREFIXES)


@dataclass(frozen=True)
class RenewSettings:
    poll_interval: int = 60        # seconds between probes while holding
    drain_minutes: int = 20        # start a proactive relaunch this far from --time
    checkpoint_grace: int = 60     # seconds to let the agent commit/push after the nudge
    backoff_initial: int = 30      # seconds, first retry after a failed (re)allocation
    backoff_max: int = 1800        # seconds, cap on the exponential backoff
    overlap: bool = True           # allocate the new slice before cancelling the old


class Action(enum.Enum):
    HOLD = "hold"                      # job healthy (or probe failed) -> keep polling
    RELAUNCH_DRAINING = "drain"        # approaching --time -> checkpoint, then re-allocate
    RELAUNCH_LOST = "lost"             # job gone/terminal -> re-allocate (no checkpoint window)


def decide_action(status: JobStatus, settings: RenewSettings) -> Action:
    """Pure decision: given a probe outcome, what should the loop do?

    - Probe failed (``not ok``)            -> HOLD (never relaunch on a blip).
    - Alive with a finite deadline at/under the drain threshold
                                            -> RELAUNCH_DRAINING.
    - Alive otherwise (incl. unbounded)    -> HOLD.
    - Successful probe, job absent/terminal -> RELAUNCH_LOST.
    """
    if not status.ok:
        return Action.HOLD
    if status.alive:
        if status.mins_left is not None and status.mins_left <= settings.drain_minutes:
            return Action.RELAUNCH_DRAINING
        return Action.HOLD
    # Successful probe, not alive: gone (state None) or terminal.
    return Action.RELAUNCH_LOST


class Backoff:
    """Exponential backoff with reset, for failed (re)allocations."""

    def __init__(self, initial: int, maximum: int) -> None:
        self._initial = max(1, initial)
        self._maximum = max(self._initial, maximum)
        self._current = self._initial

    def reset(self) -> None:
        self._current = self._initial

    def next(self) -> int:
        delay = self._current
        self._current = min(self._current * 2, self._maximum)
        return delay


def run_renew_loop(
    job_id: int,
    settings: RenewSettings,
    *,
    probe: Callable[[int], JobStatus],
    relaunch: Callable[[], Optional[int]],
    scancel: Callable[[int], None],
    request_checkpoint: Optional[Callable[[], None]] = None,
    stop_requested: Callable[[], bool] = lambda: False,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda _msg: None,
    max_iterations: Optional[int] = None,
) -> int:
    """Drive the monitor/relaunch loop until stopped.

    Injected I/O:
      - ``probe(job_id) -> JobStatus``     : query SLURM for the job.
      - ``relaunch() -> Optional[int]``    : re-allocate + relaunch the
        agent (detached); returns the new job id, or ``None`` on failure.
      - ``scancel(job_id)``                : cancel a (now-stale) job.
      - ``request_checkpoint()``           : nudge the agent to commit/push
        (write the sentinel) ahead of a proactive drain.
      - ``stop_requested() -> bool``       : True to exit (release/Ctrl-C).
      - ``sleep(seconds)`` / ``log(msg)``  : injectable for tests.

    ``max_iterations`` bounds the loop for tests; ``None`` runs until
    ``stop_requested`` returns True.  Returns the last job id held.
    """
    backoff = Backoff(settings.backoff_initial, settings.backoff_max)
    iterations = 0

    while not stop_requested():
        if max_iterations is not None and iterations >= max_iterations:
            break
        iterations += 1

        status = probe(job_id)
        action = decide_action(status, settings)

        if action is Action.HOLD:
            if status.ok:
                backoff.reset()
            sleep(settings.poll_interval)
            continue

        if action is Action.RELAUNCH_DRAINING:
            mins = status.mins_left
            log(f"job {job_id}: ~{mins} min to --time; checkpointing then re-allocating")
            if request_checkpoint is not None:
                request_checkpoint()
                sleep(settings.checkpoint_grace)
        else:  # RELAUNCH_LOST
            log(f"job {job_id}: gone/terminal (state={status.state}); re-allocating")

        if stop_requested():
            break

        new_job = relaunch()
        if new_job is None:
            delay = backoff.next()
            log(f"re-allocation failed; retrying in {delay}s")
            sleep(delay)
            continue

        # New slice is up.  Release the old one (a no-op for LOST, where
        # it is already gone) when overlapping.
        if settings.overlap and action is Action.RELAUNCH_DRAINING:
            try:
                scancel(job_id)
            except Exception as exc:  # pragma: no cover - best effort
                log(f"scancel of old job {job_id} failed: {exc}")

        log(f"relaunched: job {job_id} -> {new_job}")
        job_id = new_job
        backoff.reset()
        sleep(settings.poll_interval)

    return job_id
