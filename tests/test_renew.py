"""Tests for the auto-renew controller (`sucoder.renew`).

All I/O is injected, so the decision logic and loop are exercised here
without touching SLURM, SSH, or the filesystem.
"""
import pytest

from sucoder.renew import (
    Action,
    Backoff,
    JobStatus,
    RenewSettings,
    decide_action,
    parse_time_left,
    run_renew_loop,
)


# --------------------------------------------------------------------------- #
# parse_time_left
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value, expected",
    [
        ("3-00:00:00", 4320),
        ("1-00:00:00", 1440),
        ("2-12:30:00", 3630),
        ("23:59:00", 1439),
        ("01:30:00", 90),
        ("5:00:00", 300),
        ("45:00", 45),
        ("5:00", 5),
        ("0:30", 0),
    ],
)
def test_parse_time_left_durations(value, expected):
    assert parse_time_left(value) == expected


@pytest.mark.parametrize("value", ["UNLIMITED", "INVALID", "N/A", "", "  ", None])
def test_parse_time_left_unbounded_or_invalid(value):
    assert parse_time_left(value) is None


# --------------------------------------------------------------------------- #
# JobStatus
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "state, alive, terminal",
    [
        ("RUNNING", True, False),
        ("PENDING", True, False),
        ("CONFIGURING", True, False),
        ("COMPLETED", False, True),
        ("TIMEOUT", False, True),
        ("CANCELLED by 12345", False, True),
        ("PREEMPTED", False, True),
        (None, False, False),
    ],
)
def test_jobstatus_classification(state, alive, terminal):
    s = JobStatus(ok=True, state=state)
    assert s.alive is alive
    assert s.terminal is terminal


# --------------------------------------------------------------------------- #
# decide_action
# --------------------------------------------------------------------------- #
def test_decide_probe_failure_holds():
    s = JobStatus(ok=False)
    assert decide_action(s, RenewSettings()) is Action.HOLD


def test_decide_running_with_headroom_holds():
    s = JobStatus(ok=True, state="RUNNING", mins_left=600)
    assert decide_action(s, RenewSettings(drain_minutes=20)) is Action.HOLD


def test_decide_running_near_deadline_drains():
    s = JobStatus(ok=True, state="RUNNING", mins_left=20)
    assert decide_action(s, RenewSettings(drain_minutes=20)) is Action.RELAUNCH_DRAINING


def test_decide_running_unbounded_holds():
    # No --time (UNLIMITED QOS): mins_left is None -> never drains.
    s = JobStatus(ok=True, state="RUNNING", mins_left=None)
    assert decide_action(s, RenewSettings()) is Action.HOLD


def test_decide_absent_is_lost():
    s = JobStatus(ok=True, state=None)
    assert decide_action(s, RenewSettings()) is Action.RELAUNCH_LOST


def test_decide_terminal_is_lost():
    s = JobStatus(ok=True, state="COMPLETED")
    assert decide_action(s, RenewSettings()) is Action.RELAUNCH_LOST


# --------------------------------------------------------------------------- #
# Backoff
# --------------------------------------------------------------------------- #
def test_backoff_exponential_and_reset():
    b = Backoff(initial=30, maximum=200)
    assert [b.next() for _ in range(5)] == [30, 60, 120, 200, 200]
    b.reset()
    assert b.next() == 30


# --------------------------------------------------------------------------- #
# run_renew_loop
# --------------------------------------------------------------------------- #
class _Harness:
    """Records loop interactions; feeds scripted probe results."""

    def __init__(self, statuses, relaunch_ids):
        self._statuses = list(statuses)
        self._relaunch_ids = list(relaunch_ids)
        self.probed = []
        self.relaunch_calls = 0
        self.scancelled = []
        self.checkpoints = 0
        self.slept = []

    def probe(self, job_id):
        self.probed.append(job_id)
        return self._statuses.pop(0) if self._statuses else JobStatus(ok=False)

    def relaunch(self):
        self.relaunch_calls += 1
        return self._relaunch_ids.pop(0) if self._relaunch_ids else None

    def scancel(self, job_id):
        self.scancelled.append(job_id)

    def checkpoint(self):
        self.checkpoints += 1

    def sleep(self, seconds):
        self.slept.append(seconds)


def _run(h, job_id=111, settings=None, max_iterations=1, stop=None):
    return run_renew_loop(
        job_id,
        settings or RenewSettings(),
        probe=h.probe,
        relaunch=h.relaunch,
        scancel=h.scancel,
        request_checkpoint=h.checkpoint,
        stop_requested=stop or (lambda: False),
        sleep=h.sleep,
        max_iterations=max_iterations,
    )


def test_loop_holds_on_healthy_job():
    h = _Harness([JobStatus(ok=True, state="RUNNING", mins_left=600)], [])
    final = _run(h, max_iterations=1)
    assert final == 111
    assert h.relaunch_calls == 0
    assert h.scancelled == []


def test_loop_draining_relaunches_checkpoints_and_cancels_old():
    h = _Harness(
        [
            JobStatus(ok=True, state="RUNNING", mins_left=10),   # drain
            JobStatus(ok=True, state="RUNNING", mins_left=600),  # new job healthy
        ],
        [222],
    )
    final = _run(h, settings=RenewSettings(drain_minutes=20), max_iterations=2)
    assert final == 222
    assert h.relaunch_calls == 1
    assert h.checkpoints == 1
    assert h.scancelled == [111]          # old job released after overlap
    assert h.probed == [111, 222]         # second poll watches the new job


def test_loop_lost_relaunches_without_checkpoint_or_cancel():
    h = _Harness([JobStatus(ok=True, state=None)], [333])
    final = _run(h, max_iterations=1)
    assert final == 333
    assert h.relaunch_calls == 1
    assert h.checkpoints == 0              # no checkpoint window when already gone
    assert h.scancelled == []             # nothing to cancel


def test_loop_probe_failure_does_not_relaunch():
    h = _Harness([JobStatus(ok=False)], [999])
    final = _run(h, max_iterations=1)
    assert final == 111                   # unchanged
    assert h.relaunch_calls == 0          # a transient blip must not tear down


def test_loop_relaunch_failure_backs_off():
    h = _Harness([JobStatus(ok=True, state=None)], [None])  # relaunch fails
    settings = RenewSettings(backoff_initial=30, backoff_max=200)
    final = _run(h, settings=settings, max_iterations=1)
    assert final == 111                   # still holding the (dead) id to retry
    assert h.relaunch_calls == 1
    assert 30 in h.slept                  # backoff delay applied


def test_loop_stop_requested_exits_immediately():
    h = _Harness([JobStatus(ok=True, state="RUNNING", mins_left=5)], [222])
    final = _run(h, stop=lambda: True, max_iterations=5)
    assert final == 111
    assert h.probed == []                 # never even polled
    assert h.relaunch_calls == 0


def test_loop_no_overlap_skips_cancel():
    h = _Harness(
        [JobStatus(ok=True, state="RUNNING", mins_left=5)],
        [222],
    )
    final = _run(h, settings=RenewSettings(overlap=False), max_iterations=1)
    assert final == 222
    assert h.scancelled == []             # overlap off -> caller cancels elsewhere
