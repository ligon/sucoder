"""Tests for the on-node SLURM deadline-timer helpers in ``sucoder.cli``.

The deadline watchdog embeds a small bash function, ``left_to_mins``,
that converts ``squeue -o %L`` (TIME_LEFT) into whole minutes.  The
previous inline parser split on ``:`` only and mis-handled the
``D-HH:MM:SS`` day format -- a multi-day job collapsed to a tiny value,
firing every warning at once and then going silent.  These tests
exercise the extracted helper directly under bash.
"""
import shutil
import subprocess

import pytest

from sucoder.cli import _SLURM_TIME_LEFT_TO_MINS_SH

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)


def _mins(time_left: str) -> str:
    """Run the embedded ``left_to_mins`` helper on *time_left*."""
    script = _SLURM_TIME_LEFT_TO_MINS_SH + '\nleft_to_mins "$1"\n'
    result = subprocess.run(
        ["bash", "-c", script, "_", time_left],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.mark.parametrize(
    "time_left, expected",
    [
        ("3-00:00:00", "4320"),  # 3 days -- the original bug case
        ("1-00:00:00", "1440"),  # exactly 1 day (day format)
        ("2-12:30:00", "3630"),  # 2d 12h 30m
        ("23:59:00", "1439"),    # under a day, HH:MM:SS
        ("01:30:00", "90"),      # leading zeros (octal trap)
        ("5:00:00", "300"),      # 5h
        ("45:00", "45"),         # under an hour, MM:SS
        ("5:00", "5"),
        ("0:30", "0"),           # 30s rounds down to 0 whole minutes
    ],
)
def test_left_to_mins_durations(time_left, expected):
    assert _mins(time_left) == expected


@pytest.mark.parametrize("value", ["UNLIMITED", "INVALID", "", "N/A"])
def test_left_to_mins_non_numeric_sentinel(value):
    # Non-numeric / unlimited time-left -> large sentinel so the timer
    # never fires a spurious deadline warning.
    assert _mins(value) == "999999"


def test_left_to_mins_day_format_regression():
    # The core regression: a multi-day job must NOT parse to a tiny
    # number that would trip the 5-minute "commit NOW" warning at the
    # very start of the allocation.
    assert int(_mins("2-00:00:00")) > 30
    assert int(_mins("7-00:00:00")) == 10080
