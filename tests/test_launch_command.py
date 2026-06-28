"""Tests for the tmux launch-command builder used by the renew loop.

``MirrorManager._build_tmux_launch_command`` is the one bit of launch
logic the auto-renew feature added: a detached relaunch must use
``tmux new-session -A -d`` (create-or-attach, but do not attach a
terminal) instead of the interactive ``-A``.
"""
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
