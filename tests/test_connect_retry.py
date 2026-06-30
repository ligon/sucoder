"""Tests for `_connect_with_retry` (cli.py).

A freshly-allocated SLURM node can refuse SSH for a few seconds while
`sshd`/`pam_slurm_adopt` register the job; the connect helper rides that
out with capped exponential backoff instead of failing the launch.
"""
import logging

import pytest

from sucoder.cli import _connect_with_retry
from sucoder.tunnel import TunnelError, is_transient_ssh_error

_LOG = logging.getLogger("test.connect_retry")


class _FakeControl:
    """A control whose `ensure` fails `fail_times` times, then succeeds."""

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def ensure(self, logger):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TunnelError("kex_exchange_identification: Connection closed")


def test_connects_first_try():
    ctl = _FakeControl(fail_times=0)
    slept = []
    _connect_with_retry(ctl, "compute node nX", _LOG, sleep=slept.append)
    assert ctl.calls == 1
    assert slept == []


def test_rides_out_transient_failures():
    ctl = _FakeControl(fail_times=2)
    slept = []
    _connect_with_retry(ctl, "compute node nX", _LOG, sleep=slept.append)
    assert ctl.calls == 3          # 2 failures + 1 success
    assert slept == [3, 6]         # capped exponential backoff between tries


def test_gives_up_after_max_wait():
    ctl = _FakeControl(fail_times=999)  # never recovers
    slept = []
    with pytest.raises(TunnelError):
        _connect_with_retry(
            ctl, "compute node nX", _LOG,
            max_wait=20, initial_delay=3, sleep=slept.append,
        )
    # delays 3, 6, 12 -> waited 3, 9, 21 (>= 20) -> re-raise on the 4th try
    assert ctl.calls == 4
    assert slept == [3, 6, 12]


def test_backoff_is_capped_at_15s():
    ctl = _FakeControl(fail_times=999)
    slept = []
    with pytest.raises(TunnelError):
        _connect_with_retry(
            ctl, "n", _LOG, max_wait=120, initial_delay=3, sleep=slept.append,
        )
    assert max(slept) == 15        # 3, 6, 12, 15, 15, ... never exceeds 15


class _FailingControl:
    """A control that always fails with a fixed message + stderr."""

    def __init__(self, message="", stderr=""):
        self.message = message
        self.stderr = stderr
        self.calls = 0

    def ensure(self, logger):
        self.calls += 1
        raise TunnelError(self.message, stderr=self.stderr)


def test_fails_fast_on_nontransient_error():
    # A genuine auth failure is not transient: surface immediately with
    # no backoff, even though max_wait would otherwise allow retries.
    ctl = _FailingControl(
        message="Failed to establish SSH connection to gw",
        stderr="Permission denied (publickey,keyboard-interactive).",
    )
    slept = []
    with pytest.raises(TunnelError):
        _connect_with_retry(ctl, "gw", _LOG, sleep=slept.append)
    assert ctl.calls == 1          # no retry
    assert slept == []


def test_retries_transient_marker_carried_on_stderr():
    # Real establish() wraps a generic message and puts the transient
    # reason on ``stderr``; the classifier must look there too.
    class _Ctl:
        def __init__(self):
            self.calls = 0

        def ensure(self, logger):
            self.calls += 1
            if self.calls <= 2:
                raise TunnelError(
                    "Failed to establish SSH connection to ln001.brc",
                    stderr="kex_exchange_identification: "
                           "Connection closed by remote host",
                )

    ctl = _Ctl()
    slept = []
    _connect_with_retry(ctl, "ln001.brc", _LOG, sleep=slept.append)
    assert ctl.calls == 3          # 2 transient failures + 1 success
    assert slept == [3, 6]


def test_is_transient_ssh_error_classification():
    assert is_transient_ssh_error(
        "kex_exchange_identification: Connection closed by remote host"
    )
    assert is_transient_ssh_error(
        "ssh: connect to host x port 22: Connection refused"
    )
    # Not transient: a real auth failure, an unresolvable host, and empty.
    assert not is_transient_ssh_error("Permission denied (publickey).")
    assert not is_transient_ssh_error("ssh: Could not resolve hostname ln001.brc")
    assert not is_transient_ssh_error("")
