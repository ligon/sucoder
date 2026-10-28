"""Tests for the tmux launch-command builder used by the renew loop.

``MirrorManager._build_tmux_launch_command`` is the one bit of launch
logic the auto-renew feature added: a detached relaunch must use
``tmux new-session -A -d`` (create-or-attach, but do not attach a
terminal) instead of the interactive ``-A``.
"""
import shutil
import subprocess

import pytest

from sucoder.mirror import MirrorManager

_AGENT = "claude --foo; exec bash -l"
_NAME = "sucoder-proj"


def test_attached_launch_uses_plain_A():
    cmd = MirrorManager._build_tmux_launch_command(_NAME, _AGENT, detached=False)
    assert cmd == ["tmux", "new-session", "-A", "-s", _NAME, _AGENT]
    assert "-d" not in cmd


def test_detached_launch_adds_d_flag():
    cmd = MirrorManager._build_tmux_launch_command(_NAME, _AGENT, detached=True)
    assert cmd == ["tmux", "new-session", "-A", "-d", "-s", _NAME, _AGENT]
    # -A keeps it idempotent if the session somehow already exists.
    assert cmd.index("-A") < cmd.index("-d")


def test_window_name_labels_the_window_after_the_session():
    cmd = MirrorManager._build_tmux_launch_command(
        _NAME, _AGENT, detached=True, window_name="proj",
    )
    assert cmd == ["tmux", "new-session", "-A", "-d", "-s", _NAME, "-n", "proj", _AGENT]


def test_no_window_name_leaves_the_command_unchanged():
    for name in (None, ""):
        cmd = MirrorManager._build_tmux_launch_command(
            _NAME, _AGENT, detached=False, window_name=name,
        )
        assert "-n" not in cmd


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux not available")
def test_window_name_survives_the_running_command(tmp_path):
    # A real server: the label must stick while the command runs, which
    # needs tmux to have turned automatic-rename off for the window.
    sock = str(tmp_path / "sock")
    cmd = MirrorManager._build_tmux_launch_command(
        "sucoder-t", "sleep 30", detached=True, window_name="My.Repo",
    )
    subprocess.run(["tmux", "-S", sock, *cmd[1:]], check=True)
    try:
        out = subprocess.run(
            ["tmux", "-S", sock, "display", "-p", "-t", "=sucoder-t:",
             "#{window_name} #{automatic-rename}"],
            check=True, capture_output=True, text=True,
        ).stdout.split()
        assert out == ["My.Repo", "0"]
    finally:
        subprocess.run(["tmux", "-S", sock, "kill-server"], check=False)
