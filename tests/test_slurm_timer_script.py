"""The shared deadline timer + WIP snapshotter (``sucoder.slurm_timer``).

The script is rendered for both launch modes and exercised under bash:
syntax, socket threading, the no-``scancel`` invariant, and the snapshot
function against a real temporary repository (mirroring the manual spike
on n0036.savio4, 2026-09-09).  Nothing here execs ``tmux`` or ``squeue``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from sucoder.slurm_timer import WIP_SNAPSHOT_SH, build_timer_script

_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def _render(**kw) -> str:
    base = dict(mirror_token="K-Aggregators", tmux_session="sucoder-K-Aggregators")
    base.update(kw)
    return build_timer_script(**base)


def _bash_n(script: str, tmp_path: Path) -> subprocess.CompletedProcess:
    path = tmp_path / "timer.sh"
    path.write_text(script)
    return subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)


# -- rendering ----------------------------------------------------------------

def test_no_unresolved_tokens_in_either_mode():
    import re
    for script in (_render(job_id=123), _render(tmux_socket="s", snapshot_dir="/d")):
        assert not re.search(r"@[A-Z_]+@", script), script


def test_unconfined_mode_literal_job_id_and_plain_tmux():
    s = _render(job_id=38661192)
    assert "JOB=38661192\n" in s
    assert "TMUX_BIN=(tmux)\n" in s
    assert "SNAPSHOT_DIR=''\n" in s


def test_confined_mode_reads_job_id_at_runtime_and_threads_socket():
    s = _render(tmux_socket="sucoder-K-Aggregators", snapshot_dir="/global/home/u/mirrors/K")
    assert 'JOB="${SLURM_JOB_ID:-}"\n' in s
    assert "TMUX_BIN=(tmux -L sucoder-K-Aggregators)\n" in s
    assert "SNAPSHOT_DIR=/global/home/u/mirrors/K\n" in s
    # Every tmux invocation goes through the array, so the socket cannot be
    # dropped from any one call (has-session, display-message, set-option).
    import re
    bare_tmux = re.compile(r"(?:^|[;&|(]\s*|\$\(\s*)tmux\s")   # tmux in command position
    for line in s.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("TMUX_BIN=("):
            continue
        assert not bare_tmux.search(stripped), f"tmux called without the socket array: {line}"
    assert s.count('"${TMUX_BIN[@]}"') >= 4


def test_state_files_are_per_mirror_and_legacy_warn_kept():
    s = _render(mirror_token="alpha")
    assert 'WARN_FILE="$STATE_DIR/slurm-deadline-$MIRROR_TOKEN.warn"' in s
    assert 'LEGACY_WARN_FILE="$CACHE_DIR/slurm-deadline.warn"' in s
    assert "MIRROR_TOKEN=alpha\n" in s
    for n in (5, 15, 30):
        assert f'WARN{n}="$STATE_DIR/.slurm-warn-{n}-$MIRROR_TOKEN"' in s


def test_user_values_are_shell_quoted():
    s = _render(mirror_token="x y", tmux_session="s;rm -rf /", tmux_socket="a b",
                snapshot_dir="/p q")
    assert "MIRROR_TOKEN='x y'\n" in s
    assert "TMUX_SESSION='s;rm -rf /'\n" in s
    assert "TMUX_BIN=(tmux -L 'a b')\n" in s
    assert "SNAPSHOT_DIR='/p q'\n" in s


def test_snapshot_dir_shell_is_inserted_unquoted():
    s = _render(tmux_socket="s", snapshot_dir_shell='/local/job"${SLURM_JOB_ID}"/mirrors/K')
    assert 'SNAPSHOT_DIR=/local/job"${SLURM_JOB_ID}"/mirrors/K\n' in s
    with pytest.raises(ValueError):
        _render(snapshot_dir="/a", snapshot_dir_shell="/b")


def test_negative_snapshot_minutes_rejected():
    with pytest.raises(ValueError):
        _render(snapshot_minutes=-1)


def test_never_emits_a_bare_scancel():
    """The user owns the SLURM lifecycle (``sucoder release``); the timer
    may *mention* scancel in a warning string but never run it."""
    for script in (_render(job_id=1), _render(tmux_socket="s")):
        for raw in script.splitlines():
            assert not raw.strip().startswith("scancel"), raw


@_bash
def test_bash_syntax_both_modes(tmp_path):
    assert _bash_n(_render(job_id=1), tmp_path).returncode == 0
    r = _bash_n(_render(tmux_socket="s", snapshot_dir="/d", snapshot_minutes=0), tmp_path)
    assert r.returncode == 0, r.stderr


# -- snapshot_wip under bash --------------------------------------------------

def _git_run(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@x", *args],
        cwd=cwd, capture_output=True, text=True, check=True,
    ).stdout.strip()


def _snapshot(work: Path, token: str = "mirror") -> subprocess.CompletedProcess:
    script = (
        "set -u\n"
        f"SNAPSHOT_DIR={work}\nMIRROR_TOKEN={token}\nJOB=42\n"
        + WIP_SNAPSHOT_SH + "\nsnapshot_wip\n"
    )
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


@pytest.fixture
def repo_pair(tmp_path):
    """A non-bare ``origin`` with ``updateInstead`` and a working clone of it."""
    origin = tmp_path / "origin"
    _git_run(tmp_path, "init", "-q", "-b", "main", str(origin))
    (origin / "README").write_text("hello\n")
    (origin / ".gitignore").write_text(".venv/\n")
    _git_run(origin, "add", "-A")
    _git_run(origin, "commit", "-q", "-m", "init")
    _git_run(origin, "config", "receive.denyCurrentBranch", "updateInstead")
    work = tmp_path / "work"
    _git_run(tmp_path, "clone", "-q", str(origin), str(work))
    return origin, work


@_bash
@_git
def test_snapshot_clean_tree_is_a_noop(repo_pair):
    origin, work = repo_pair
    assert _snapshot(work).returncode == 0
    assert subprocess.run(["git", "show-ref", "refs/sucoder/wip/mirror"], cwd=origin).returncode != 0


@_bash
@_git
def test_snapshot_dirty_tree_lands_on_origin_and_skips_ignored(repo_pair):
    origin, work = repo_pair
    (work / "README").write_text("hello\nedit\n")          # tracked edit
    (work / "notes.org").write_text("new\n")               # untracked
    (work / ".venv").mkdir(); (work / ".venv" / "x").write_text("ignored\n")
    r = _snapshot(work)
    assert r.returncode == 0, r.stderr
    wip = _git_run(origin, "rev-parse", "refs/sucoder/wip/mirror")
    assert _git_run(origin, "rev-parse", "refs/sucoder/wip/mirror^") == _git_run(work, "rev-parse", "HEAD")
    files = _git_run(origin, "ls-tree", "-r", "--name-only", wip).splitlines()
    assert "notes.org" in files and "README" in files
    assert not any(f.startswith(".venv") for f in files)
    # The agent's own index and HEAD are untouched.
    status = subprocess.run(["git", "status", "--porcelain"], cwd=work,
                            capture_output=True, text=True, check=True).stdout
    assert status.splitlines() == [" M README", "?? notes.org"]
    # origin's checked-out branch did not move (the WIP ref is not a branch).
    assert _git_run(origin, "rev-parse", "main") == _git_run(work, "rev-parse", "HEAD")


@_bash
@_git
def test_snapshot_unchanged_tree_skips_via_marker(repo_pair):
    origin, work = repo_pair
    (work / "notes.org").write_text("new\n")
    assert _snapshot(work).returncode == 0
    first = _git_run(origin, "rev-parse", "refs/sucoder/wip/mirror")
    marker = work / ".git" / "sucoder-last-wip-tree"
    assert marker.exists() and marker.read_text().strip() == _git_run(origin, "rev-parse", f"{first}^{{tree}}")
    assert _snapshot(work).returncode == 0
    assert _git_run(origin, "rev-parse", "refs/sucoder/wip/mirror") == first
    # The marker lives under .git/, so it never appears in a snapshot.
    assert "sucoder-last-wip-tree" not in _git_run(origin, "ls-tree", "-r", "--name-only", first)


@_bash
@_git
def test_snapshot_without_origin_is_a_noop(tmp_path):
    solo = tmp_path / "solo"
    _git_run(tmp_path, "init", "-q", "-b", "main", str(solo))
    (solo / "a").write_text("x\n")
    _git_run(solo, "add", "-A"); _git_run(solo, "commit", "-q", "-m", "init")
    (solo / "b").write_text("dirty\n")
    assert _snapshot(solo).returncode == 0
    assert subprocess.run(["git", "show-ref", "refs/sucoder/wip/mirror"], cwd=solo).returncode != 0
    assert not (solo / ".git" / "sucoder-last-wip-tree").exists()


@_bash
def test_snapshot_missing_or_non_git_dir_is_a_noop(tmp_path):
    assert _snapshot(tmp_path / "nope").returncode == 0
    assert _snapshot(tmp_path).returncode == 0
