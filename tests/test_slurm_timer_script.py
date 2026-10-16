"""The shared deadline timer + WIP snapshotter (``sucoder.slurm_timer``).

The script is rendered for both launch modes and exercised under bash:
syntax, socket threading, the no-``scancel`` invariant, and the snapshot
function against a real temporary repository (mirroring the manual spike
on n0036.savio4, 2026-09-09).  Nothing here execs ``tmux`` or ``squeue``.
"""
from __future__ import annotations

import difflib
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from sucoder.slurm_timer import CODEX_CHECKPOINT_SH, WIP_SNAPSHOT_SH, build_timer_script

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
    assert 'LEGACY_WARN_FILE="$CACHE_DIR/slurm-deadline.warn"' in s
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
    for var in ("$WARN5", "$WARN15", "$WARN30", "$WARN_FILE", "$LEGACY_WARN_FILE", "$MIRROR_WARN_FILE"):
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


def _wip_ref(token: str = "mirror", job: int = 42) -> str:
    """Where a snapshot lands: one ref per (mirror, job).

    Keyed by job because two jobs on one mirror have two clones on two
    nodes and one slot per mirror let the later force-push over the
    earlier one's only copy (issue 19).  A SIBLING of the legacy
    ``refs/sucoder/wip/<token>``, never a child: a ref at the parent path
    blocks creating one below it.
    """
    return f"refs/sucoder/wip-job/{token}/{job}"


def _snapshot(work: Path, token: str = "mirror", job: int = 42) -> subprocess.CompletedProcess:
    script = (
        "set -u\n"
        f"SNAPSHOT_DIR={work}\nMIRROR_TOKEN={token}\nJOB={job}\n"
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
    assert subprocess.run(["git", "show-ref", _wip_ref()], cwd=origin).returncode != 0


@_bash
@_git
def test_snapshot_dirty_tree_lands_on_origin_and_skips_ignored(repo_pair):
    origin, work = repo_pair
    (work / "README").write_text("hello\nedit\n")          # tracked edit
    (work / "notes.org").write_text("new\n")               # untracked
    (work / ".venv").mkdir(); (work / ".venv" / "x").write_text("ignored\n")
    r = _snapshot(work)
    assert r.returncode == 0, r.stderr
    wip = _git_run(origin, "rev-parse", _wip_ref())
    assert _git_run(origin, "rev-parse", _wip_ref() + "^") == _git_run(work, "rev-parse", "HEAD")
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
    first = _git_run(origin, "rev-parse", _wip_ref())
    marker = work / ".git" / "sucoder-last-wip-tree"
    assert marker.exists() and marker.read_text().strip() == _git_run(origin, "rev-parse", f"{first}^{{tree}}")
    assert _snapshot(work).returncode == 0
    assert _git_run(origin, "rev-parse", _wip_ref()) == first
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
    assert subprocess.run(["git", "show-ref", _wip_ref()], cwd=solo).returncode != 0
    assert not (solo / ".git" / "sucoder-last-wip-tree").exists()


@_bash
def test_snapshot_missing_or_non_git_dir_is_a_noop(tmp_path):
    assert _snapshot(tmp_path / "nope").returncode == 0
    assert _snapshot(tmp_path).returncode == 0


# -- the warning threshold chain, driven under bash ---------------------------

def _drive(tmp_path: Path, time_left: list, **render) -> list:
    """Run the rendered timer against stubbed squeue/tmux/sleep.

    ``time_left`` is fed to successive ``squeue -o %L`` polls; once it is
    exhausted the job reads as gone and the loop ends. None represents
    a failed query; (rc, output) also permits partial output on failure.
    Returns the messages the human would have seen, in order.
    """
    return _drive_run(tmp_path, time_left, **render)[0]


def _drive_run(tmp_path: Path, time_left: list, **render) -> tuple:
    """:func:`_drive`, also returning the finished process: its stdout is
    what ``--ensure`` would have appended to ``timer.log``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    counter, log = tmp_path / "n", tmp_path / "msgs"
    polls = [(1, "") if v is None else v if isinstance(v, tuple) else (0, v)
             for v in time_left]
    (bin_dir / "squeue").write_text(
        '#!/bin/bash\n'
        f'n=$(cat {shlex.quote(str(counter))} 2>/dev/null || echo 0); '
        f'n=$((n+1)); echo $n > {shlex.quote(str(counter))}\n'
        'vals=(' + " ".join(shlex.quote(v) for _, v in polls) + ')\n'
        'codes=(' + " ".join(str(rc) for rc, _ in polls) + ')\n'
        'if [ "$n" -le "${#vals[@]}" ]; then\n'
        '  echo "${vals[$((n-1))]}"; exit "${codes[$((n-1))]}"\n'
        'fi\n'
    )
    # has-session always succeeds; only display-message is recorded.
    (bin_dir / "tmux").write_text(
        '#!/bin/bash\n'
        'case "$1" in\n'
        f'  display-message) echo "${{@: -1}}" >> {shlex.quote(str(log))} ;;\n'
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
    result = subprocess.run(["bash", str(script)], env=env, timeout=10,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    if not log.exists():
        return [], result
    return [ln for ln in log.read_text().splitlines() if ln.strip()], result


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


@_bash
@pytest.mark.parametrize("failures", [[None] * 4, [(1, "1:00")] * 4])
def test_scheduler_errors_keep_monitoring_until_recovery(tmp_path, failures):
    # Errors are unknown state, even when squeue emits partial output (ledger 3).
    msgs = _drive(tmp_path, ["2:00:00", *failures, "25:00", "12:00", "4:00"])
    deadline = [m for m in msgs if "min left" in m]
    assert len(deadline) == 3, msgs
    assert "Commit and save NOW" in deadline[-1]
    assert int((tmp_path / "n").read_text()) == 11


@_bash
def test_failed_query_interrupts_consecutive_empty_observations(tmp_path):
    msgs = _drive(tmp_path, ["", "", None, "", "", "4:00"])
    assert any("Commit and save NOW" in m for m in msgs), msgs
    assert int((tmp_path / "n").read_text()) == 9


@_bash
@_git
def test_scheduler_outage_does_not_suspend_periodic_snapshots(repo_pair, tmp_path):
    origin, work = repo_pair
    (work / "a.txt").write_text("recover this during the outage\n")
    # All cadence ticks occur during failed queries; recovery is outside a
    # threshold and outside the cadence, so only the outage can save this tree.
    _drive(tmp_path, [None, None, None, "2:00:00"],
           snapshot_dir=str(work), snapshot_minutes=2)
    saved = subprocess.run(
        ["git", "show", _wip_ref("K-Aggregators") + ":a.txt"], cwd=origin,
        capture_output=True, text=True, check=True,
    )
    assert saved.stdout == "recover this during the outage\n"


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


@_bash
@_git
def test_snapshot_force_updates_a_diverged_wip_ref(repo_pair):
    """The second snapshot of a tree whose HEAD has diverged must land.

    Every other push here creates the ref for the first time, so
    ``--force`` never does any work and dropping it left the suite
    green.  It is load-bearing in reality: the WIP ref is built with
    ``commit-tree -p HEAD``, so once the agent rebases, resets, or
    switches branch, the new snapshot is not a descendant of the old one
    and an unforced push is rejected -- silently, every step being
    best-effort -- in exactly the scenario the snapshotter exists for.
    """
    origin, work = repo_pair
    ref = _wip_ref()
    show = lambda: subprocess.run(
        ["git", "show-ref", "-s", ref], cwd=origin,
        capture_output=True, text=True).stdout.strip()

    (work / "a.txt").write_text("first\n")
    assert _snapshot(work).returncode == 0
    first = show()
    assert first, "first snapshot did not create the ref"

    # Diverge: commit onto HEAD, so the next WIP commit's parent is no
    # longer an ancestor of the ref just pushed.
    _git_run(work, "add", "-A")
    _git_run(work, "-c", "user.email=a@b", "-c", "user.name=a",
             "commit", "-q", "-m", "work")
    (work / "a.txt").write_text("second\n")

    assert _snapshot(work).returncode == 0
    second = show()
    assert second and second != first, (
        f"diverged snapshot did not update the ref ({first} -> {second}); "
        "an unforced push would be rejected here"
    )
    # The new snapshot really is not a descendant of the old one.
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", first, second],
        cwd=origin).returncode != 0


# -- a missing git must not be a silent no-op (issue #15) ----------------------
#
# Every git step in snapshot_wip is `|| exit 0` so a snapshot failure can
# never kill the watchdog.  The cost was that if git is absent the whole
# WIP-snapshot feature is a permanent no-op with zero diagnostics --
# indistinguishable from "the tree was clean".

def _snapshot_without_git(tmp_path, *, token="mirror", job=42, runs=1):
    """Drive snapshot_wip with a PATH that has no git. Returns the warn text."""
    work = tmp_path / "work"
    (work / ".git").mkdir(parents=True)
    state = tmp_path / "state"
    state.mkdir()
    warn = state / "warn"
    # A PATH containing only a coreutils shim dir: no git anywhere on it.
    stub = tmp_path / "bin"
    stub.mkdir()
    for name in ("printf", "mktemp", "date", "cat", "rm"):
        found = shutil.which(name)
        if found:
            (stub / name).symlink_to(found)
    script = (
        "set -u\n"
        f"PATH={shlex.quote(str(stub))}\n"
        f"SNAPSHOT_DIR={shlex.quote(str(work))}\n"
        f"MIRROR_TOKEN={token}\nJOB={job}\n"
        f"STATE_DIR={shlex.quote(str(state))}\n"
        f"WARN_FILE={shlex.quote(str(warn))}\n"
        + WIP_SNAPSHOT_SH
        + "\n" + ("snapshot_wip\n" * runs)
    )
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return warn.read_text() if warn.exists() else ""


@_bash
def test_missing_git_is_reported_to_the_warn_file(tmp_path):
    text = _snapshot_without_git(tmp_path)
    assert "git is not on PATH" in text, text
    assert "job 42" in text, text


@_bash
def test_missing_git_is_reported_only_once_per_job(tmp_path):
    """At the default 10-minute interval a multi-day allocation would
    otherwise append hundreds of identical lines."""
    text = _snapshot_without_git(tmp_path, runs=5)
    assert text.count("git is not on PATH") == 1, text


@_bash
def test_a_later_job_reports_again_rather_than_inheriting_silence(tmp_path):
    """The marker carries $JOB, so it is self-clearing across jobs sharing
    one $HOME -- otherwise the second job would be silently unprotected."""
    first = _snapshot_without_git(tmp_path, job=111)
    assert first.count("git is not on PATH") == 1
    # Same state dir and warn file, new job id.
    work = tmp_path / "work"
    state = tmp_path / "state"
    warn = state / "warn"
    stub = tmp_path / "bin"
    script = (
        "set -u\n"
        f"PATH={shlex.quote(str(stub))}\n"
        f"SNAPSHOT_DIR={shlex.quote(str(work))}\n"
        f"MIRROR_TOKEN=mirror\nJOB=222\n"
        f"STATE_DIR={shlex.quote(str(state))}\n"
        f"WARN_FILE={shlex.quote(str(warn))}\n"
        + WIP_SNAPSHOT_SH + "\nsnapshot_wip\n"
    )
    subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    text = warn.read_text()
    assert "job 111" in text and "job 222" in text, text


@_bash
@_git
def test_a_present_git_writes_no_such_warning(tmp_path):
    """The diagnostic must not fire on the healthy path."""
    work = tmp_path / "work"
    (work / ".git").mkdir(parents=True)
    state = tmp_path / "state"
    state.mkdir()
    warn = state / "warn"
    script = (
        "set -u\n"
        f"SNAPSHOT_DIR={shlex.quote(str(work))}\n"
        "MIRROR_TOKEN=mirror\nJOB=42\n"
        f"STATE_DIR={shlex.quote(str(state))}\n"
        f"WARN_FILE={shlex.quote(str(warn))}\n"
        + WIP_SNAPSHOT_SH + "\nsnapshot_wip\n"
    )
    subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    assert not warn.exists() or "git is not on PATH" not in warn.read_text()


# -- managed Codex state: each deadline warning requests a capture -------------
#
# A managed Codex launch (docs/codex-local-state.org) stages a runner that
# owns the Codex process and captures its databases on its own cadence.  The
# timer's part is one more capture at each deadline warning: it touches
# <local root>/state/codex/<scope>/<run>/checkpoint.request, which the runner
# polls, consumes and services single-flight.  The timer never captures,
# hashes or writes anything durable, and without a scope it is unchanged.

SCOPE = "0123456789abcdef" * 4            # a 64-hex scope digest
RUN = "fedcba9876543210" * 2              # a 32-hex run id
_RUNTIME_ROOT = '/local/job"${SLURM_JOB_ID}"'
_CLONE = _RUNTIME_ROOT + "/mirrors/K-Aggregators"


def _managed(**kw) -> str:
    base = dict(tmux_socket="s", snapshot_dir_shell=_CLONE,
                codex_state_scope=SCOPE, local_disk_root="/local")
    base.update(kw)
    return _render(**base)


def test_without_a_scope_the_script_is_byte_identical():
    """The feature is additive: a Claude launch, or a target with
    codex_state off, ships exactly the script it shipped before."""
    plain = _render(tmux_socket="s", snapshot_dir_shell=_CLONE)
    assert plain == _render(tmux_socket="s", snapshot_dir_shell=_CLONE, codex_state_scope=None)
    assert "codex" not in plain.lower() and "checkpoint" not in plain.lower()
    diff = list(difflib.ndiff(plain.splitlines(), _managed().splitlines()))
    assert not [ln for ln in diff if ln.startswith("- ")], "a scope may only add lines"
    added = [ln[2:] for ln in diff if ln.startswith("+ ")]
    assert added[0] == f"CODEX_STATE_DIR={_RUNTIME_ROOT}/state/codex/{SCOPE}"
    assert added.count("        request_codex_checkpoint") == 3
    assert set(added[1:]) - {"        request_codex_checkpoint", ""} == set(CODEX_CHECKPOINT_SH.splitlines())


def test_scope_renders_the_request_function_and_calls_it_at_each_warning():
    s = _managed()
    assert not re.search(r"@[A-Z_]+@", s), s
    assert f"\nCODEX_STATE_DIR={_RUNTIME_ROOT}/state/codex/{SCOPE}\n" in s
    assert "request_codex_checkpoint() {" in s
    assert s.index("request_codex_checkpoint() {") < s.index("        request_codex_checkpoint\n")
    # Right after snapshot_wip in each of the three warning branches...
    assert s.count("        snapshot_wip\n        request_codex_checkpoint\n") == 3
    # ...and nowhere else: one definition, three calls -- not on the
    # periodic tick (the runner has its own cadence) and not once the job
    # has left the queue (nothing left to request).
    assert s.count("request_codex_checkpoint") == 4
    assert "request_codex_checkpoint" not in s[s.index('if [ "$SNAPSHOT_MINUTES" -gt 0 ]'):]
    # A literal job id spells the root as local_root_shell does for it.
    literal = _render(job_id=42, codex_state_scope=SCOPE, local_disk_root="/local/")
    assert f"\nCODEX_STATE_DIR=/local/job42/state/codex/{SCOPE}\n" in literal


def test_request_function_touches_local_control_files_only():
    """No durable path, no backup, no hashing, nothing created but the
    request: the runner owns all of that.  Everything the function reads
    or writes hangs off $CODEX_STATE_DIR, and the only commands it runs are
    cat, the ``:`` redirection and echo."""
    for token in (".sucoder-state", "generations", "rollouts", "durable", "sha256",
                  "sqlite", "manifest", "mkdir", "git ", "flock", "sleep"):
        assert token not in CODEX_CHECKPOINT_SH, token
    # "could not touch" is a message; no line runs touch, git, or the like.
    assert not re.search(r"^\s*(touch|git|mkdir|cp|mv|rm|find|tar)\b", CODEX_CHECKPOINT_SH, re.M)
    assert set(re.findall(r"\$\w+", CODEX_CHECKPOINT_SH)) == {
        "$CODEX_STATE_DIR", "$run", "$why", "$request",
        "$_codex_request_said_ok", "$_codex_request_said_fail",
    }
    assert "return 0" in CODEX_CHECKPOINT_SH and "exit" not in CODEX_CHECKPOINT_SH


def test_scope_is_validated_and_needs_the_local_root():
    """``shlex.quote`` is the identity on a hex digest, so "the scope is
    quoted" is observable only as the refusal of anything that is not one
    -- a path component is built from it."""
    assert shlex.quote(SCOPE) == SCOPE
    for bad in ("a b", "../" + SCOPE[3:], SCOPE[:-1], SCOPE + "0", SCOPE.upper(), SCOPE + "'", ""):
        with pytest.raises(ValueError, match="64-hex"):
            _managed(codex_state_scope=bad)
    for root in (None, ""):
        with pytest.raises(ValueError, match="local_disk_root"):
            _render(tmux_socket="s", codex_state_scope=SCOPE, local_disk_root=root)


@_bash
def test_bash_syntax_with_a_scope(tmp_path):
    r = _bash_n(_managed(), tmp_path)
    assert r.returncode == 0, r.stderr
    r = _bash_n(_render(job_id=42, codex_state_scope=SCOPE, local_disk_root="/local"), tmp_path)
    assert r.returncode == 0, r.stderr


def _request(state_dir: Path, *, runs: int = 1, between: str = "") -> subprocess.CompletedProcess:
    """Run ``request_codex_checkpoint`` ``runs`` times against ``state_dir``
    (the rendered ``$CODEX_STATE_DIR``), with ``between`` executed after
    the first call.  stdout is what ``timer.log`` would receive."""
    script = (
        "set -u\n"
        f"CODEX_STATE_DIR={shlex.quote(str(state_dir))}\n"
        + CODEX_CHECKPOINT_SH + "\n"
        + "request_codex_checkpoint\n" + between
        + "request_codex_checkpoint\n" * (runs - 1)
    )
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def _tree(root: Path) -> set:
    return {str(p.relative_to(root)) for p in root.rglob("*")} if root.exists() else set()


@pytest.fixture
def managed_root(tmp_path):
    """``<local root>/state/codex/<scope>`` as the runner leaves it under
    ``<tmp>/local/job42``: a run directory and ``latest-run`` naming it."""
    state_dir = tmp_path / "local" / "job42" / "state" / "codex" / SCOPE
    (state_dir / RUN).mkdir(parents=True)
    (state_dir / "latest-run").write_text(RUN + "\n")
    return state_dir


@_bash
def test_request_touches_the_latest_runs_checkpoint_request(managed_root):
    before = _tree(managed_root)
    r = _request(managed_root)
    assert r.returncode == 0 and r.stderr == ""
    request = managed_root / RUN / "checkpoint.request"
    assert request.is_file() and request.stat().st_size == 0
    assert _tree(managed_root) == before | {f"{RUN}/checkpoint.request"}
    assert r.stdout.count("requested a Codex state checkpoint") == 1
    assert str(request) in r.stdout


@_bash
def test_request_logs_once_per_outcome_not_per_warning(managed_root):
    r = _request(managed_root, runs=3)
    assert r.returncode == 0
    assert r.stdout.count("sucoder timer:") == 1
    assert (managed_root / RUN / "checkpoint.request").is_file()


@_bash
def test_request_without_a_run_leaves_nothing_behind_and_does_not_fail(tmp_path):
    state_dir = tmp_path / "local" / "job42" / "state" / "codex" / SCOPE
    r = _request(state_dir, runs=3)                 # no scope directory at all
    assert r.returncode == 0 and r.stderr == ""
    assert not (tmp_path / "local").exists()
    assert r.stdout.count("could not request a Codex state checkpoint") == 1
    assert "no managed run is recorded" in r.stdout
    state_dir.mkdir(parents=True)                   # the runner never recorded a run
    r = _request(state_dir)
    assert r.returncode == 0 and _tree(state_dir) == set()


@_bash
@pytest.mark.parametrize("latest, why", [
    ("not-a-run\n", "does not name a run"),
    ("../../escape\n", "does not name a run"),
    (RUN[::-1] + "\n", "is missing"),       # well formed, but no such run directory
])
def test_request_refuses_a_bad_or_absent_run(tmp_path, latest, why):
    state_dir = tmp_path / "state" / "codex" / SCOPE
    state_dir.mkdir(parents=True)
    (state_dir / "latest-run").write_text(latest)
    before = _tree(tmp_path)
    r = _request(state_dir, runs=2)
    assert r.returncode == 0 and r.stderr == ""
    assert _tree(tmp_path) == before                # nothing created, nowhere
    assert r.stdout.count("could not request") == 1 and why in r.stdout


@_bash
def test_request_recovers_once_the_runner_records_a_run(tmp_path):
    """Each outcome logs on its first occurrence: a warning before the
    runner placed its run logs the failure, a later one logs the success."""
    state_dir = tmp_path / "state" / "codex" / SCOPE
    state_dir.mkdir(parents=True)
    between = (
        f"mkdir -p {shlex.quote(str(state_dir / RUN))}\n"
        f"echo {RUN} > {shlex.quote(str(state_dir / 'latest-run'))}\n"
    )
    r = _request(state_dir, runs=3, between=between)
    assert r.returncode == 0
    assert (state_dir / RUN / "checkpoint.request").is_file()
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("sucoder timer:")]
    assert len(lines) == 2 and "could not request" in lines[0] and "requested" in lines[1]


@_bash
def test_deadline_warnings_request_a_checkpoint_from_the_runner(tmp_path, managed_root):
    """End to end: the rendered script under real bash with squeue and tmux
    stubbed.  Three warnings fire, the request lands on local disk, and
    timer.log carries one line about it."""
    msgs, result = _drive_run(tmp_path, ["2:00:00", "25:00", "12:00", "4:00"],
                              codex_state_scope=SCOPE, local_disk_root=str(tmp_path / "local"))
    assert len([m for m in msgs if "min left" in m]) == 3
    assert (managed_root / RUN / "checkpoint.request").is_file()
    assert result.stdout.count("requested a Codex state checkpoint") == 1


@_bash
def test_periodic_tick_and_job_gone_do_not_request_a_checkpoint(tmp_path, managed_root):
    """The runner has its own cadence; the timer asks at the warnings only,
    and a job that has left the queue has nothing left to request."""
    msgs, result = _drive_run(tmp_path, ["2:00:00", "1:59:00", "1:58:00"], snapshot_minutes=1,
                              codex_state_scope=SCOPE, local_disk_root=str(tmp_path / "local"))
    assert not [m for m in msgs if "min left" in m]
    assert any("no longer queued" in m for m in msgs), msgs
    assert not (managed_root / RUN / "checkpoint.request").exists()
    assert "Codex state checkpoint" not in result.stdout


@_bash
def test_warnings_without_a_managed_run_leave_the_local_root_untouched(tmp_path):
    local = tmp_path / "local" / "job42"
    local.mkdir(parents=True)
    msgs, result = _drive_run(tmp_path, ["25:00", "12:00", "4:00"],
                              codex_state_scope=SCOPE, local_disk_root=str(tmp_path / "local"))
    assert len([m for m in msgs if "min left" in m]) == 3
    assert _tree(local) == set()
    assert result.stdout.count("could not request a Codex state checkpoint") == 1


def test_all_scopes_needs_the_local_root_and_excludes_a_specific_scope():
    for root in (None, ""):
        with pytest.raises(ValueError, match="local_disk_root"):
            _render(codex_state_all_scopes=True, local_disk_root=root)
    with pytest.raises(ValueError, match="not both"):
        _managed(codex_state_all_scopes=True)
    assert _render(local_disk_root="/local") == _render()


@_bash
@pytest.mark.parametrize("job_id", [None, 42])
def test_all_scopes_shell_syntax_and_root_quoting(tmp_path, job_id):
    root = tmp_path / "local ' $HOME"
    script = _render(job_id=job_id, codex_state_all_scopes=True, local_disk_root=str(root) + "/")
    result = _bash_n(script, tmp_path)
    assert result.returncode == 0, result.stderr
    assignment = next(ln for ln in script.splitlines() if ln.startswith("CODEX_STATE_ROOT="))
    result = subprocess.run(
        ["sh", "-c", 'SLURM_JOB_ID=42\n' + assignment + '\nprintf "%s" "$CODEX_STATE_ROOT"'],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout == str(root / "job42" / "state" / "codex")
    assert script.count("        snapshot_wip\n        request_codex_checkpoint\n") == 3
    assert script.count("request_codex_checkpoint") == 4
    assert not re.search(r"@[A-Z_]+@", script)


@_bash
@pytest.mark.parametrize("time_left", ["25:00", "12:00", "4:00", "2:00:00"])
def test_all_scopes_warnings_request_only_each_scopes_latest_run(tmp_path, time_left):
    root = tmp_path / "local ' $HOME"
    expected = set()
    for job in (42, 43):
        for scope in (SCOPE, SCOPE[::-1]):
            state = root / f"job{job}" / "state" / "codex" / scope
            for run in (RUN, RUN[::-1]):
                (state / run).mkdir(parents=True)
            (state / "latest-run").write_text(RUN + "\n")
            if job == 42 and time_left != "2:00:00":
                expected.add(str((state / RUN / "checkpoint.request").relative_to(root)))
    before = _tree(root)
    msgs, result = _drive_run(tmp_path, [time_left], snapshot_minutes=1,
                              codex_state_all_scopes=True, local_disk_root=str(root))
    assert len([m for m in msgs if "min left" in m]) == bool(expected)
    assert _tree(root) == before | expected
    assert result.stdout.count("requested a Codex state checkpoint") == bool(expected)


@_bash
@pytest.mark.parametrize("existing", [None, "job42", "job42/state/codex", f"job42/state/codex/{SCOPE}"])
def test_all_scopes_without_managed_runs_creates_nothing(tmp_path, existing):
    root = tmp_path / "local"
    if existing is not None:
        (root / existing).mkdir(parents=True)
    before = _tree(root)
    msgs, result = _drive_run(tmp_path, ["25:00", "12:00", "4:00"],
                              codex_state_all_scopes=True, local_disk_root=str(root))
    assert len([m for m in msgs if "min left" in m]) == 3
    assert _tree(root) == before
    assert root.exists() == (existing is not None)
    assert "Codex state checkpoint" not in result.stdout


@pytest.mark.parametrize("shell", ["sh", "bash"])
def test_all_scopes_skips_bad_runs_and_failed_requests_and_repeats(tmp_path, shell):
    root = tmp_path / "local"
    states = root / "job42" / "state" / "codex"
    bad = ("", "../escape", RUN[:-1], RUN + "0", RUN.upper(), RUN + "\nextra", RUN[::-1])
    for i, latest in enumerate(bad):
        state = states / f"{i:064x}"
        (state / RUN).mkdir(parents=True)
        (state / "latest-run").write_text(latest + "\n")
    # A directory at the request path forces a write failure even as root.
    failed = states / ("0" * 63 + "f")
    (failed / RUN / "checkpoint.request").mkdir(parents=True)
    (failed / "latest-run").write_text(RUN + "\n")
    good = states / SCOPE
    (good / RUN).mkdir(parents=True)
    (good / "latest-run").write_text(RUN + "\n")
    request = good / RUN / "checkpoint.request"
    before = _tree(root)
    rendered = _render(job_id=42, codex_state_all_scopes=True, local_disk_root=str(root))
    assignment = next(ln for ln in rendered.splitlines() if ln.startswith("CODEX_STATE_ROOT="))
    helper = rendered[rendered.index("_codex_request_said_ok=0"):rendered.index("\n\nwarn()")]
    # Consume the first request as the runner does; the next warning must
    # request again even though its success message has already been logged.
    result = subprocess.run(
        [shell, "-c", "set -eu\n" + assignment + "\n" + helper
         + "\nrequest_codex_checkpoint\n"
         + f"rm {shlex.quote(str(request))}\n"
         + "request_codex_checkpoint\nrequest_codex_checkpoint\n"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0 and result.stderr == ""
    assert _tree(root) == before | {str(request.relative_to(root))}
    assert request.is_file() and request.stat().st_size == 0
    assert result.stdout.count("requested a Codex state checkpoint") == 1
    assert result.stdout.count("could not request a Codex state checkpoint") == 1


# -- the launcher hands the scope to the timer for a managed Codex launch ----

def _confined_launch(tmp_path, monkeypatch, command):
    """A confined launch end to end through ``_launch_confined`` with the
    scheduler stubbed (``tests.test_mirror._confined_responder``), managed
    Codex state enabled on the target the way ``tests.test_codex_state``
    wires it, and ``mirror.build_timer_script`` wrapped to capture its
    arguments.  Returns (captured kwargs, staged batch script, staged timer).
    """
    from sucoder import mirror as mirror_mod
    from sucoder.config import AgentLauncher, CodexStateConfig
    from tests.test_mirror import _confined_responder
    from tests.test_remote_control import _manager

    manager, ctx = _manager(tmp_path, monkeypatch, remote=True)
    ctx.settings.remote.slurm.local_disk = "/local"
    ctx.settings.remote.slurm.codex_state = CodexStateConfig(enabled=True)
    ctx.settings.agent_launcher = AgentLauncher(command=["claude"])
    manager.executor.local_disk_root = "/local"
    manager.executor.agent_user = "coder"
    monkeypatch.setattr("sucoder.session._session_dir", lambda: tmp_path / "sessions")
    monkeypatch.setattr(mirror_mod.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(manager, "_resolve_remote_path", lambda ctx: "/global/scratch/u/mirrors/sample")
    calls = []
    manager.executor.run_agent = _confined_responder(calls, sbatch_out="9")
    captured = []
    real = mirror_mod.build_timer_script

    def capture(**kw):
        captured.append(kw)
        return real(**kw)

    monkeypatch.setattr(mirror_mod, "build_timer_script", capture)
    manager.launch_agent(ctx, sync=False, command_override=list(command),
                         supports_inline_prompt=True, detached=True)
    batch = next(c["input"] for c in calls
                 if c["args"][0] == "sh" and c["input"] and "/job-sample.sh" in c["args"][2])
    timer = next(c["input"] for c in calls if "sucoder-slurm-timer" in c["args"] and c["input"])
    assert len(captured) == 1
    return captured[0], batch, timer


def test_managed_codex_launch_hands_the_scope_to_the_timer(tmp_path, monkeypatch):
    from sucoder.codex_state import scope_id

    kw, batch, timer = _confined_launch(tmp_path, monkeypatch, ["codex"])
    scope = scope_id("cluster", "coder", "/global/scratch/u/mirrors/sample")
    assert kw["codex_state_scope"] == scope and kw["local_disk_root"] == "/local"
    assert f'\nCODEX_STATE_DIR=/local/job"${{SLURM_JOB_ID}}"/state/codex/{scope}\n' in timer
    assert timer.count("        request_codex_checkpoint\n") == 3
    # The same digest the staged runner is told, so the two meet on disk.
    assert "codex-state-runner" in batch and f"--scope {scope}" in batch


def test_claude_launch_hands_no_scope_to_the_timer(tmp_path, monkeypatch):
    kw, batch, timer = _confined_launch(tmp_path, monkeypatch, ["claude"])
    assert kw["codex_state_scope"] is None
    assert "codex" not in timer.lower() and "checkpoint" not in timer.lower()
    assert "codex-state-runner" not in batch
