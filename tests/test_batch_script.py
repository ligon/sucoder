"""Batch-script builder for `confined` (sbatch) launches.

Spike-validated on savio4_htc 2026-06-29: the batch task runs in the job
cgroup and, with a DEDICATED tmux socket (`-L`), starts its own server
confined to the reserved cores (nproc=4 in the pane).  These tests lock
the script's shape + quoting; runtime confinement is cluster-validated.
"""
import os
import shutil
import subprocess
import tempfile

import pytest

from sucoder.mirror import MirrorManager

_BASE = dict(
    tmux_session="sucoder-K-Aggregators",
    socket="sucoder-K-Aggregators",
    mirror_path="/global/home/users/ligon/mirrors/K-Aggregators",
    agent_cmd_str='claude --system-prompt "$(cat "$HOME/.cache/sucoder/prelude-K.txt")"; exec bash -l',
)

_bash_only = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _bash_n(script):
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        return subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    finally:
        os.unlink(path)


def test_structure():
    s = MirrorManager._build_batch_script(**_BASE)
    assert s.startswith("#!/bin/bash\n")
    assert "cd /global/home/users/ligon/mirrors/K-Aggregators || " in s
    # Dedicated -L socket on BOTH new-session and the keeper -- the
    # spike-required detail (else tmux reuses a shared server, wrong cgroup).
    assert "tmux -L sucoder-K-Aggregators new-session -A -d -s sucoder-K-Aggregators " in s
    assert "tmux -L sucoder-K-Aggregators has-session -t sucoder-K-Aggregators" in s
    assert "sleep 15" in s
    assert "claude --system-prompt" in s          # agent command embedded
    # new-session failure must be caught + marked, not silently exit-0.
    assert "SUCODER: tmux new-session failed" in s
    assert "exit 1" in s


@_bash_only
def test_new_session_failure_marks_and_exits_nonzero(tmp_path):
    # If new-session fails, the script must emit a SUCODER marker to stderr
    # and exit non-zero -- NOT fall through to the keeper loop and COMPLETE
    # with exit 0 (which would mask an agent-never-started failure).
    s = MirrorManager._build_batch_script(
        tmux_session="t", socket="t", mirror_path=str(tmp_path),
        agent_cmd_str="true",
    )
    # Stub tmux so new-session fails (rc=3); has-session would say "yes" if
    # reached, which would prove the guard did NOT stop the fall-through.
    stub = (
        "tmux() {\n"
        '  for w in "$@"; do [ "$w" = new-session ] && return 3; done\n'
        "  return 0\n"
        "}\n"
    )
    r = subprocess.run(
        ["bash", "-c", stub + s], capture_output=True, text=True,
    )
    assert r.returncode == 1
    assert "SUCODER: tmux new-session failed (rc=3)" in r.stderr


@_bash_only
def test_env_reaches_window_command(tmp_path):
    # sbatch does not carry agent_launcher.env, so the builder prepends
    # `export K=V;` to the window command.  Verify it actually reaches the
    # executed command (with a spaced value, to exercise the quoting).
    out = tmp_path / "tok.txt"
    s = MirrorManager._build_batch_script(
        tmux_session="t", socket="t", mirror_path=str(tmp_path),
        agent_cmd_str=f'printf "TOK=%s" "$SUCODER_TOK" > {out}',
        env={"SUCODER_TOK": "abc 123"},
    )
    # Stub tmux: has-session -> false (keeper exits); new-session -> run the
    # window command (last arg) so we can observe the exported env.
    stub = (
        "tmux() {\n"
        '  for w in "$@"; do [ "$w" = has-session ] && return 1; done\n'
        '  bash -c "${@: -1}"\n'
        "}\n"
    )
    subprocess.run(["bash", "-c", stub + s], check=True, capture_output=True, text=True)
    assert out.read_text() == "TOK=abc 123"


@_bash_only
def test_valid_bash():
    assert _bash_n(MirrorManager._build_batch_script(**_BASE)).returncode == 0
    assert _bash_n(MirrorManager._build_batch_script(**_BASE, env={"A": "b c"})).returncode == 0


@_bash_only
def test_quoting_survives_metacharacters():
    nasty = dict(
        tmux_session="s'q",
        socket="so'ck",
        mirror_path="/p a/t`h",
        agent_cmd_str='X "$(cat f)" `b` $V & ; | exec bash -l',
    )
    r = _bash_n(MirrorManager._build_batch_script(**nasty, env={"K": "v'1 `x`"}))
    assert r.returncode == 0, r.stderr
