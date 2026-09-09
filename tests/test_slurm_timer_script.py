"""The shared deadline timer + WIP snapshotter (``sucoder.slurm_timer``).

The script is rendered for both launch modes and exercised under bash:
syntax, socket threading, the no-``scancel`` invariant, and the snapshot
function against a real temporary repository (mirroring the manual spike
on n0036.savio4, 2026-09-09).  Nothing here execs ``tmux`` or ``squeue``.
"""
from __future__ import annotations

import os
import re
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


def test_both_bash_helpers_reach_the_rendered_script():
    """The one thing ``build_timer_script`` uniquely does is assemble the
    two helpers into the script.  Tested elsewhere only as standalone
    constants, so rendering them as empty strings left the suite green
    while shipping a script that warns never and snapshots never --
    ``bash -n`` does not flag a call to an undefined function."""
    for script in (_render(job_id=1), _render(tmux_socket="s", snapshot_dir="/d")):
        assert "left_to_mins() {" in script
        assert "snapshot_wip() {" in script
        # ...and they are defined before the loop that calls them.
        assert script.index("left_to_mins() {") < script.index("mins=$(left_to_mins")
        assert script.index("snapshot_wip() {") < script.index("        snapshot_wip\n")


def test_state_files_are_per_mirror_and_legacy_warn_kept():
    s = _render(mirror_token="alpha")
    assert 'WARN_FILE="$STATE_DIR/slurm-deadline-$MIRROR_TOKEN.warn"' in s
    assert 'LEGACY_WARN_FILE="$STATE_DIR/slurm-deadline.warn"' in s
    assert "MIRROR_TOKEN=alpha\n" in s
    for n in (5, 15, 30):
        assert f'WARN{n}="$STATE_DIR/.slurm-warn-{n}-$MIRROR_TOKEN"' in s


def test_startup_clears_the_legacy_warn_file_too():
    """Startup clears every warn file it may later write, the legacy one
    included.  Clearing only the per-mirror file let a previous job's
    \"allocation may have ended\" survive on the legacy path into a
    healthy new session -- read by exactly the older prompts that path
    is kept for."""
    s = _render(mirror_token="alpha")
    rm = next(ln for ln in s.splitlines() if ln.startswith("rm -f "))
    for var in ("$WARN5", "$WARN15", "$WARN30", "$WARN_FILE", "$LEGACY_WARN_FILE"):
        assert f'"{var}"' in rm, f"{var} not cleared at startup: {rm}"
    # Cleared before any warning could be written.
    assert s.index(rm) < s.index("warn() {")


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


def _runs_scancel(line: str) -> bool:
    """True if ``line`` would *execute* scancel.

    The timer legitimately names scancel inside warning messages ("Run
    'scancel 7' to free the allocation"), so a guard cannot just look for
    the word.  Stripping quoted strings and comments leaves only shell
    code, and any scancel surviving that is a real command -- which
    ``startswith("scancel")`` was not: it missed ``then scancel ...``,
    ``&& /usr/bin/scancel ...``, ``$(scancel ...)`` and ``timeout 5
    scancel ...``, i.e. every plausible way it would come back.
    """
    code = re.sub(r"'[^']*'|\"[^\"]*\"", "", line).split("#", 1)[0]
    return "scancel" in code


def test_never_emits_a_bare_scancel():
    """The user owns the SLURM lifecycle (``sucoder release``); the timer
    may *mention* scancel in a warning string but never run it."""
    for script in (_render(job_id=1), _render(tmux_socket="s")):
        for raw in script.splitlines():
            assert not _runs_scancel(raw), raw


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


# -- the warning threshold chain, driven under bash ---------------------------

def _drive(tmp_path: Path, time_left: list, **render) -> list:
    """Run the rendered timer against stubbed squeue/tmux/sleep.

    ``time_left`` is fed to successive ``squeue -o %L`` polls; once it is
    exhausted the job reads as gone and the loop ends.  Returns the
    messages the human would have seen, in order.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    counter, log = tmp_path / "n", tmp_path / "msgs"
    (bin_dir / "squeue").write_text(
        '#!/bin/bash\n'
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        'vals=(' + " ".join(f'"{v}"' for v in time_left) + ')\n'
        'if [ "$n" -le "${#vals[@]}" ]; then echo "${vals[$((n-1))]}"; fi\n'
    )
    # has-session always succeeds; only display-message is recorded.
    (bin_dir / "tmux").write_text(
        '#!/bin/bash\n'
        'case "$1" in\n'
        f'  display-message) echo "${{@: -1}}" >> {log} ;;\n'
        '  has-session) exit 0 ;;\n'
        'esac\n'
        'exit 0\n'
    )
    (bin_dir / "sleep").write_text("#!/bin/bash\nexit 0\n")
    for f in bin_dir.iterdir():
        f.chmod(0o755)

    script = tmp_path / "timer.sh"
    script.write_text(_render(job_id=42, **render))
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
               HOME=str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    subprocess.run(["bash", str(script)], env=env, timeout=60,
                   capture_output=True, text=True)
    if not log.exists():
        return []
    return [ln for ln in log.read_text().splitlines() if ln.strip()]


@_bash
def test_warnings_escalate_and_never_repeat(tmp_path):
    """Each threshold fires once, in increasing urgency, and a job that
    starts inside a threshold does not walk back down the ladder.

    The chain fires the highest *unfired* threshold, so marking only the
    one that fired let later polls fall through to the *less* urgent
    branches: a job with 3 minutes left warned "Commit and save NOW",
    then "Start wrapping up" at 2 minutes, then the bare 30-minute
    notice at 1 minute -- urgency running backwards as the deadline
    approached, with a ``git add -A`` sweep behind each spurious
    warning.  A threshold firing must therefore also mark every coarser
    one as spent.
    """
    # A long job passes each threshold in turn: one warning each, escalating.
    msgs = _drive(tmp_path, ["2:00:00", "40:00", "25:00", "12:00", "4:00"])
    deadline = [m for m in msgs if "min left" in m]
    assert len(deadline) == 3, deadline
    assert "Start wrapping up" not in deadline[0]
    assert "Commit and save NOW" not in deadline[0]
    assert "Start wrapping up" in deadline[1]
    assert "Commit and save NOW" in deadline[2]


@_bash
def test_short_job_warns_once_at_its_true_urgency(tmp_path):
    """A job that starts with 3 minutes left gets the 5-minute warning and
    nothing else -- not a de-escalating sequence down to the 30-minute
    notice."""
    msgs = _drive(tmp_path, ["3:00", "2:00", "1:00"])
    deadline = [m for m in msgs if "min left" in m]
    assert len(deadline) == 1, f"expected one warning, got {deadline}"
    assert "Commit and save NOW" in deadline[0]


@_bash
def test_skipped_poll_does_not_walk_back_down(tmp_path):
    """A poll gap that jumps 31 -> 14 minutes fires the 15-minute warning,
    then escalates to the 5-minute one; it must not emit the 30-minute
    notice afterwards."""
    msgs = _drive(tmp_path, ["31:00", "14:00", "13:00", "4:00"])
    deadline = [m for m in msgs if "min left" in m]
    assert len(deadline) == 2, deadline
    assert "Start wrapping up" in deadline[0]
    assert "Commit and save NOW" in deadline[1]


@_bash
def test_transient_squeue_failure_does_not_retire_the_watchdog(tmp_path):
    """One empty squeue read must not end the loop.

    squeue prints nothing both when the job is gone and when the
    controller RPC times out, so breaking on the first empty read let a
    single transient failure remove every remaining deadline warning
    from a job with hours left -- the exact failure this script exists
    to prevent.
    """
    # 10h left, one transient blank, then the countdown resumes and
    # every threshold still fires.
    msgs = _drive(tmp_path, ["10:00:00", "", "25:00", "12:00", "4:00"])
    deadline = [m for m in msgs if "min left" in m]
    assert len(deadline) == 3, deadline
    assert "Commit and save NOW" in deadline[-1]


@_bash
def test_sustained_squeue_silence_still_reports_the_job_gone(tmp_path):
    """Tolerating blips must not mean never noticing a finished job."""
    msgs = _drive(tmp_path, ["10:00:00"])
    assert any("no longer queued" in m for m in msgs), msgs


def test_lifecycle_hint_matches_the_launch_mode():
    """Under sbatch the batch body's keeper loop exits with the tmux
    session, so the job ends with it; telling that user to `scancel` a
    job that already completed sends them after a ghost.  Under salloc
    the allocation really does survive."""
    def hints(script):
        # Only the two warning messages, not the surrounding comments.
        return [l for l in script.splitlines()
                if l.strip().startswith("echo ") and "$WARN_FILE" in l
                and ("kept alive" in l or "ends with it" in l)]

    confined = hints(_render(tmux_socket="s"))   # job id read at run time
    unconfined = hints(_render(job_id=42))
    # Both places say it: the startup timeout and the session-gone exit.
    assert len(confined) == 2 and len(unconfined) == 2
    for line in confined:
        assert "ends with it" in line
        assert "kept alive" not in line and "sucoder release" not in line
    for line in unconfined:
        assert "kept alive" in line and "sucoder release" in line
        assert "ends with it" not in line
