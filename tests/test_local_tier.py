"""Local-disk tiering (``sucoder.local_tier``): path helpers, the prepare
script rendered for a confined launch, and the script driven under bash
against a real temporary shared mirror.  Mirrors the manual spike in
docs/local-disk-tiering.org.  Needs bash and git; never execs tmux/squeue.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from sucoder.local_tier import (
    POST_COMMIT_HOOK,
    build_prepare_script,
    cache_exports_sh,
    local_root_shell,
    work_path_shell,
)

_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
_needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


# -- path helpers --------------------------------------------------------------

def test_paths_runtime_job_id_expands_and_literal_is_quoted():
    assert local_root_shell("/local") == '/local/job"${SLURM_JOB_ID}"'
    assert local_root_shell("/local/", 42) == "/local/job42"
    assert work_path_shell("/local", "SuCoder") == '/local/job"${SLURM_JOB_ID}"/mirrors/SuCoder'
    assert work_path_shell("/scratch x", "a b", 7) == "'/scratch x/job7'/mirrors/'a b'"


def test_cache_exports_cover_tools_and_tmpdir_but_not_installs():
    line = cache_exports_sh("/local")
    for var in ("SUCODER_LOCAL_ROOT", "UV_CACHE_DIR", "PIP_CACHE_DIR", "npm_config_cache", "TMPDIR"):
        assert f" {var}=" in f" {line}"
    assert "UV_TOOL_DIR" not in line and "npm_config_prefix" not in line


# -- rendering -----------------------------------------------------------------

def _render(**kw) -> str:
    base = dict(mirror_path="/global/home/u/mirrors/K", mirror_token="K")
    base.update(kw)
    return build_prepare_script(**base)


def test_prepare_script_resolves_all_tokens_and_embeds_hook():
    s = _render()
    assert not re.search(r"@[A-Z_]+@", s)
    assert "MIRROR=/global/home/u/mirrors/K\n" in s
    assert 'LOCAL_ROOT=/local/job"${SLURM_JOB_ID}"\n' in s
    assert POST_COMMIT_HOOK.strip() in s
    assert "git push --quiet --force" not in POST_COMMIT_HOOK   # never force-publish


def test_prepare_script_quotes_user_values():
    s = _render(mirror_path="/p q/m", mirror_token="a b", local_disk_root="/l r", job_id=3)
    assert "MIRROR='/p q/m'\n" in s and "TOKEN='a b'\n" in s and "LOCAL_ROOT='/l r/job3'\n" in s


@_bash
def test_prepare_script_bash_syntax(tmp_path):
    path = tmp_path / "p.sh"
    path.write_text(_render())
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True).returncode == 0


# -- driven under bash ----------------------------------------------------------

def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@x", *args],
        cwd=cwd, capture_output=True, text=True, check=True,
    ).stdout.rstrip("\n")


@pytest.fixture
def shared(tmp_path):
    """A shared mirror (non-bare, main checked out, updateInstead) plus a
    local-disk stand-in root."""
    mirror = tmp_path / "mirrors" / "K"
    _git(tmp_path, "init", "-q", "-b", "main", str(mirror))
    (mirror / "README").write_text("hello\n")
    (mirror / ".gitignore").write_text(".venv/\n")
    _git(mirror, "add", "-A"); _git(mirror, "commit", "-q", "-m", "init")
    _git(mirror, "config", "receive.denyCurrentBranch", "updateInstead")
    local = tmp_path / "local"
    local.mkdir()
    return mirror, local


# The restore gate asks the scheduler whether a snapshot's job is still
# running, so these tests control whether ``squeue`` exists and what it
# says.  PATH is rebuilt from scratch rather than prepended to: the host
# running the suite may be a real Slurm cluster, and a test for "squeue is
# unavailable" that silently finds /usr/bin/squeue tests nothing.
_PATH_TOOLS = (
    "bash", "sh", "git", "grep", "sed", "cat", "ls", "mkdir", "rm", "chmod",
    "cut", "tr", "head", "tail", "sort", "uniq", "wc", "env", "dirname",
    "basename", "uname", "date", "mktemp", "expr", "printf", "true", "false",
)


class _SqueueInherit:
    """Sentinel: leave PATH alone (tests that do not exercise the gate)."""


_SQUEUE_INHERIT = _SqueueInherit()


def _stub_path(tmp: Path, squeue: str | None) -> tuple[Path, dict]:
    """A bin dir holding only the tools the prepare script may use.

    ``squeue`` None leaves it off PATH entirely; otherwise it is a stub
    whose body is ``squeue`` (it receives the job id as ``$2``).

    One directory per distinct stub body, never one mutated in place: two
    prepares in a single test may need different answers, and a rewritten
    shared stub would silently give the second one the first one's.
    """
    key = "nosq" if squeue is None else hashlib.sha256(squeue.encode()).hexdigest()[:12]
    bin_dir = tmp / f"bin-{key}"
    if bin_dir.exists():
        return bin_dir, dict(os.environ, PATH=str(bin_dir))
    bin_dir.mkdir(parents=True)
    for name in _PATH_TOOLS:
        found = shutil.which(name)
        if found:
            (bin_dir / name).symlink_to(found)
    if squeue is not None:
        stub = bin_dir / "squeue"
        stub.write_text("#!/bin/sh\n" + squeue + "\n")
        stub.chmod(0o755)
    return bin_dir, dict(os.environ, PATH=str(bin_dir))


def _run_prepare(
    mirror: Path, local: Path, job_id: int = 5, *, squeue=_SQUEUE_INHERIT,
) -> subprocess.CompletedProcess:
    script = build_prepare_script(
        mirror_path=str(mirror), mirror_token="K", local_disk_root=str(local), job_id=job_id,
    )
    env = os.environ
    if squeue is not _SQUEUE_INHERIT:
        _, env = _stub_path(local.parent, squeue)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)


def _push_snapshot(
    work: Path, mirror: Path, ref: str, subject: str, when: str | None = None,
) -> str:
    """Take a snapshot the way the timer does and push it to the mirror.

    ``when`` pins the committer date.  Retention picks by
    ``--sort=-committerdate``, and snapshots taken in the same second sort
    ambiguously, so any test about *which* ref survives must set it.
    """
    env = dict(os.environ, GIT_INDEX_FILE=str(work / ".git" / f"tmpidx-{ref.replace('/', '_')}"))
    subprocess.run(["git", "read-tree", "HEAD"], cwd=work, env=env, check=True)
    subprocess.run(["git", "add", "-A"], cwd=work, env=env, check=True)
    tree = subprocess.run(["git", "write-tree"], cwd=work, env=env,
                          capture_output=True, text=True, check=True).stdout.strip()
    if when is None:
        wip = _git(work, "commit-tree", tree, "-p", "HEAD", "-m", subject)
    else:
        wip = subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@x",
             "commit-tree", tree, "-p", "HEAD", "-m", subject],
            cwd=work, capture_output=True, text=True, check=True,
            env=dict(os.environ, GIT_COMMITTER_DATE=when, GIT_AUTHOR_DATE=when),
        ).stdout.strip()
    _git(work, "push", "-q", "--force", "origin", f"{wip}:{ref}")
    return wip


@_bash
@_needs_git
def test_fresh_clone_with_hook_publishes_commits(shared):
    mirror, local = shared
    r = _run_prepare(mirror, local)
    assert r.returncode == 0, r.stderr
    work = local / "job5" / "mirrors" / "K"
    assert (work / ".git").is_dir()
    for sub in ("cache/uv", "cache/pip", "cache/npm", "tmp"):
        assert (local / "job5" / sub).is_dir()
    assert _git(work, "remote", "get-url", "origin") == str(mirror)
    hook = work / ".git" / "hooks" / "post-commit"
    assert hook.exists() and os.access(hook, os.X_OK)
    assert "SUCODER: local-tier working clone ready" in r.stdout

    (work / "new.txt").write_text("x\n")
    _git(work, "add", "new.txt")
    out = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-q", "-m", "agent"],
                         cwd=work, capture_output=True, text=True, check=True)
    assert "SUCODER: published" in out.stdout + out.stderr
    assert _git(mirror, "rev-parse", "main") == _git(work, "rev-parse", "HEAD")
    assert (mirror / "new.txt").exists()          # updateInstead moved the shared checkout


@_bash
@_needs_git
def test_rerun_is_idempotent_and_fast_forwards(shared):
    mirror, local = shared
    assert _run_prepare(mirror, local).returncode == 0
    work = local / "job5" / "mirrors" / "K"
    (work / "keep.txt").write_text("dirty\n")     # uncommitted work must survive a re-run
    # laptop pushes a new commit into the shared mirror
    (mirror / "README").write_text("hello\nlaptop\n")
    _git(mirror, "commit", "-q", "-am", "laptop")
    r = _run_prepare(mirror, local)
    assert r.returncode == 0, r.stderr
    assert (work / "keep.txt").exists()
    assert _git(work, "rev-parse", "HEAD") == _git(mirror, "rev-parse", "main")


# -- restore gate (issue 19) ----------------------------------------------------
#
# The old gate was "parent == HEAD and the tree is clean".  Two jobs on one
# mirror are normally on the same branch at the same commit, so that gate
# passed for the OTHER job's snapshot and imported its dirty tree, announcing
# it as restored work with nothing saying whose it was.  What follows pins
# both halves: pick by job, and name the job in the message.

_RUNNING = "echo RUNNING"
_GONE = 'echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1'
_GONE_EMPTY = "exit 0"
_UNREACHABLE = 'echo "slurm_load_jobs error: Unable to contact slurm controller" >&2; exit 1'


def _ref(job: int, token: str = "K") -> str:
    return f"refs/sucoder/wip-job/{token}/{job}"


def _subject(job: int) -> str:
    return f"WIP snapshot 2026-09-18T00:00:00-07:00 job {job}"


@_bash
@_needs_git
def test_wip_snapshot_restored_only_onto_its_parent(shared):
    mirror, local = shared
    assert _run_prepare(mirror, local, job_id=1, squeue=_GONE).returncode == 0
    work1 = local / "job1" / "mirrors" / "K"
    (work1 / "wip.txt").write_text("in progress\n")
    wip = _push_snapshot(work1, mirror, _ref(1), _subject(1))

    # New job: job 1 has ended, so its snapshot is this job's to resume.
    r = _run_prepare(mirror, local, job_id=2, squeue=_GONE)
    assert r.returncode == 0, r.stderr
    work2 = local / "job2" / "mirrors" / "K"
    assert (work2 / "wip.txt").read_text() == "in progress\n"
    assert "restored uncommitted work" in r.stdout
    assert _git(work2, "status", "--porcelain") == "?? wip.txt"
    assert _git(mirror, "rev-parse", _ref(1)) == wip        # ref kept

    # A commit lands after the snapshot: the next job must NOT restore it.
    (mirror / "README").write_text("hello\nlater\n")
    _git(mirror, "commit", "-q", "-am", "later")
    r = _run_prepare(mirror, local, job_id=3, squeue=_GONE)
    assert r.returncode == 0, r.stderr
    assert not (local / "job3" / "mirrors" / "K" / "wip.txt").exists()
    assert "not restored" in r.stderr


@_bash
@_needs_git
def test_snapshot_of_a_still_running_job_is_not_restored(shared):
    """Issue 19 mode 2.  Job 1 is alive on another node; job 2 must not
    adopt its uncommitted tree just because they sit on the same commit."""
    mirror, local = shared
    assert _run_prepare(mirror, local, job_id=1, squeue=_GONE).returncode == 0
    work1 = local / "job1" / "mirrors" / "K"
    (work1 / "theirs.txt").write_text("job 1 is still working on this\n")
    _push_snapshot(work1, mirror, _ref(1), _subject(1))

    r = _run_prepare(mirror, local, job_id=2, squeue=_RUNNING)
    assert r.returncode == 0, r.stderr
    assert not (local / "job2" / "mirrors" / "K" / "theirs.txt").exists()
    assert "restored uncommitted work" not in r.stdout
    assert "belongs to job 1" in r.stderr and "still running" in r.stderr


@_bash
@_needs_git
def test_restore_names_the_job_it_came_from(shared):
    """The silent half of mode 2: the message never said whose tree it was."""
    mirror, local = shared
    assert _run_prepare(mirror, local, job_id=1, squeue=_GONE).returncode == 0
    work1 = local / "job1" / "mirrors" / "K"
    (work1 / "wip.txt").write_text("x\n")
    _push_snapshot(work1, mirror, _ref(1), _subject(1))

    r = _run_prepare(mirror, local, job_id=2, squeue=_GONE)
    assert "restored uncommitted work from job 1" in r.stdout, r.stdout
    assert _ref(1) in r.stdout


@_bash
@_needs_git
def test_scheduler_error_is_not_read_as_the_job_having_ended(shared):
    """A failed query is unknown state, not evidence the job is gone
    (ledger 3).  Unknown must decline, or the gate is decorative during
    exactly the controller outage that makes two live jobs likely."""
    mirror, local = shared
    assert _run_prepare(mirror, local, job_id=1, squeue=_GONE).returncode == 0
    work1 = local / "job1" / "mirrors" / "K"
    (work1 / "theirs.txt").write_text("unknown owner\n")
    _push_snapshot(work1, mirror, _ref(1), _subject(1))

    r = _run_prepare(mirror, local, job_id=2, squeue=_UNREACHABLE)
    assert r.returncode == 0, r.stderr
    assert not (local / "job2" / "mirrors" / "K" / "theirs.txt").exists()
    assert "could not be checked" in r.stderr


@_bash
@_needs_git
def test_successful_empty_query_counts_as_gone(shared):
    """The other accepted 'gone' answer: squeue succeeds and says nothing."""
    mirror, local = shared
    assert _run_prepare(mirror, local, job_id=1, squeue=_GONE).returncode == 0
    work1 = local / "job1" / "mirrors" / "K"
    (work1 / "wip.txt").write_text("y\n")
    _push_snapshot(work1, mirror, _ref(1), _subject(1))

    r = _run_prepare(mirror, local, job_id=2, squeue=_GONE_EMPTY)
    assert (local / "job2" / "mirrors" / "K" / "wip.txt").exists(), r.stderr


@_bash
@_needs_git
def test_this_jobs_own_snapshot_wins_over_a_newer_foreign_one(shared):
    """A re-run of prepare inside one job resumes that job's own work even
    when another job snapshotted more recently."""
    mirror, local = shared
    assert _run_prepare(mirror, local, job_id=7, squeue=_GONE).returncode == 0
    work = local / "job7" / "mirrors" / "K"
    (work / "mine.txt").write_text("mine\n")
    _push_snapshot(work, mirror, _ref(7), _subject(7))
    (work / "mine.txt").unlink()
    (work / "theirs.txt").write_text("theirs\n")
    _push_snapshot(work, mirror, _ref(8), _subject(8))       # newer
    (work / "theirs.txt").unlink()

    r = _run_prepare(mirror, local, job_id=7, squeue=_GONE)
    assert r.returncode == 0, r.stderr
    assert "restored uncommitted work from job 7" in r.stdout, r.stdout
    assert (work / "mine.txt").exists() and not (work / "theirs.txt").exists()


@_bash
@_needs_git
def test_legacy_shared_ref_is_still_read_and_gated_by_its_subject(shared):
    """A pre-upgrade timer on a live job keeps writing the shared ref, and
    its job id exists only in the snapshot's subject.  Both still apply."""
    mirror, local = shared
    assert _run_prepare(mirror, local, job_id=1, squeue=_GONE).returncode == 0
    work1 = local / "job1" / "mirrors" / "K"
    (work1 / "legacy.txt").write_text("from an old timer\n")
    _push_snapshot(work1, mirror, "refs/sucoder/wip/K", _subject(77))

    r = _run_prepare(mirror, local, job_id=2, squeue=_RUNNING)
    assert not (local / "job2" / "mirrors" / "K" / "legacy.txt").exists()
    assert "belongs to job 77" in r.stderr

    r = _run_prepare(mirror, local, job_id=3, squeue=_GONE)
    assert (local / "job3" / "mirrors" / "K" / "legacy.txt").exists(), r.stderr
    assert "restored uncommitted work from job 77" in r.stdout


@_bash
@_needs_git
def test_snapshot_without_a_job_id_is_not_restored(shared):
    """Every snapshot this snapshotter writes carries `job <ID>`.  One that
    does not was written by something else; do not import it blind."""
    mirror, local = shared
    assert _run_prepare(mirror, local, job_id=1, squeue=_GONE).returncode == 0
    work1 = local / "job1" / "mirrors" / "K"
    (work1 / "mystery.txt").write_text("?\n")
    _push_snapshot(work1, mirror, "refs/sucoder/wip/K", "WIP")

    r = _run_prepare(mirror, local, job_id=2, squeue=_GONE)
    assert not (local / "job2" / "mirrors" / "K" / "mystery.txt").exists()
    assert "records no job id" in r.stderr


@_bash
@_needs_git
def test_without_squeue_the_restore_says_liveness_was_unchecked(shared):
    """Declining outright would break every relaunch wherever squeue is not
    on PATH -- which is issue 15's environment.  Restore, but say so: the
    complaint in issue 19 is that the wrong restore was SILENT."""
    mirror, local = shared
    assert _run_prepare(mirror, local, job_id=1, squeue=None).returncode == 0
    work1 = local / "job1" / "mirrors" / "K"
    (work1 / "wip.txt").write_text("z\n")
    _push_snapshot(work1, mirror, _ref(1), _subject(1))

    r = _run_prepare(mirror, local, job_id=2, squeue=None)
    assert (local / "job2" / "mirrors" / "K" / "wip.txt").exists(), r.stderr
    assert "squeue unavailable" in r.stdout
    assert "could not check whether job 1 is still running" in r.stdout


@_bash
@_needs_git
def test_two_jobs_snapshots_do_not_overwrite_each_other(shared):
    """Issue 19 mode 1.  One slot per mirror meant the later snapshot was
    the only one that survived; per-job refs keep both."""
    mirror, local = shared
    assert _run_prepare(mirror, local, job_id=1, squeue=_GONE).returncode == 0
    work1 = local / "job1" / "mirrors" / "K"
    (work1 / "a.txt").write_text("job 1\n")
    first = _push_snapshot(work1, mirror, _ref(1), _subject(1))
    (work1 / "a.txt").unlink()
    (work1 / "b.txt").write_text("job 2\n")
    second = _push_snapshot(work1, mirror, _ref(2), _subject(2))

    assert first != second
    assert _git(mirror, "rev-parse", _ref(1)) == first
    assert _git(mirror, "rev-parse", _ref(2)) == second
    assert _git(mirror, "show", f"{_ref(1)}:a.txt") == "job 1"
    assert _git(mirror, "show", f"{_ref(2)}:b.txt") == "job 2"


@_bash
@_needs_git
def test_foreign_post_commit_hook_left_alone_with_warning(shared):
    mirror, local = shared
    work = local / "job5" / "mirrors" / "K"
    # Pre-create the clone with somebody else's hook.
    work.parent.mkdir(parents=True)
    _git(work.parent, "clone", "-q", str(mirror), str(work))
    hook = work / ".git" / "hooks" / "post-commit"
    hook.write_text("#!/bin/sh\necho theirs\n"); hook.chmod(0o755)
    r = _run_prepare(mirror, local)
    assert r.returncode == 0, r.stderr
    assert hook.read_text() == "#!/bin/sh\necho theirs\n"
    assert "left alone" in r.stderr and "NOT auto-publish" in r.stderr


@_bash
@_needs_git
def test_dirty_shared_mirror_warns(shared):
    mirror, local = shared
    (mirror / "README").write_text("hello\nhand edit\n")   # tracked edit on the mailbox
    r = _run_prepare(mirror, local)
    assert r.returncode == 0, r.stderr
    assert "uncommitted changes to tracked files" in r.stderr and "REJECTED" in r.stderr


@_bash
@_needs_git
def test_missing_or_detached_mirror_fails_loudly(shared, tmp_path):
    mirror, local = shared
    r = _run_prepare(tmp_path / "nowhere", local)
    assert r.returncode == 1 and "not a git repository" in r.stderr
    _git(mirror, "checkout", "-q", "--detach")
    r = _run_prepare(mirror, local)
    assert r.returncode == 1 and "detached HEAD" in r.stderr


# -- retention (issue 14) -------------------------------------------------------
#
# Per-job refs would otherwise accumulate one per allocation forever.
# Deleting them frees nothing by itself -- each snapshot is commit-tree -p
# HEAD and already orphans its predecessor -- but it is what lets the
# gc --auto receive-pack runs on every hook push reap them at all.

# Job 1 is still on a node; everything else has ended.
_ONE_LIVE = 'case "$2" in 1) echo RUNNING;; *) echo "Invalid job id specified" >&2; exit 1;; esac'


def _remote_wip_refs(mirror: Path) -> set[str]:
    out = _git(mirror, "for-each-ref", "--format=%(refname)", "refs/sucoder")
    return {ln for ln in out.splitlines() if ln}


def _seed(shared, refs, subjects=None, squeue=_GONE):
    """Prepare job 1, then push one snapshot per (job, timestamp) in *refs*."""
    mirror, local = shared
    assert _run_prepare(mirror, local, job_id=1, squeue=squeue).returncode == 0
    work = local / "job1" / "mirrors" / "K"
    for job, when in refs:
        (work / f"f{job}.txt").write_text(f"job {job}\n")
        subject = (subjects or {}).get(job, _subject(job))
        target = (subjects or {}).get(f"ref{job}", _ref(job))
        _push_snapshot(work, mirror, target, subject, when)
        (work / f"f{job}.txt").unlink()
    return mirror, local


@_bash
@_needs_git
def test_ended_jobs_snapshots_are_retired_keeping_the_newest(shared):
    mirror, local = _seed(shared, [(1, "2026-09-10T00:00:00+00:00"),
                                   (2, "2026-09-11T00:00:00+00:00"),
                                   (3, "2026-09-12T00:00:00+00:00")])
    assert _remote_wip_refs(mirror) == {_ref(1), _ref(2), _ref(3)}

    r = _run_prepare(mirror, local, job_id=9, squeue=_GONE)
    assert r.returncode == 0, r.stderr
    # The newest ended job's snapshot survives: job 9 has not taken its own
    # yet and will not for up to wip_snapshot_minutes.
    assert _remote_wip_refs(mirror) == {_ref(3)}
    assert "retired WIP snapshot" in r.stdout
    assert _ref(1) in r.stdout and _ref(2) in r.stdout


@_bash
@_needs_git
def test_a_live_jobs_snapshot_is_never_retired(shared):
    mirror, local = _seed(shared, [(1, "2026-09-10T00:00:00+00:00"),
                                   (2, "2026-09-11T00:00:00+00:00"),
                                   (3, "2026-09-12T00:00:00+00:00")])
    r = _run_prepare(mirror, local, job_id=9, squeue=_ONE_LIVE)
    assert r.returncode == 0, r.stderr
    # 3 is newest-ended and kept; 2 is retired; 1 is still running.
    assert _remote_wip_refs(mirror) == {_ref(1), _ref(3)}
    assert _ref(1) not in r.stdout


@_bash
@_needs_git
def test_unknown_scheduler_answer_retires_nothing(shared):
    """Deleting on an unknown answer is the same mistake as restoring on
    one, and it deletes the only durable copy of someone's work."""
    mirror, local = _seed(shared, [(1, "2026-09-10T00:00:00+00:00"),
                                   (2, "2026-09-11T00:00:00+00:00")])
    r = _run_prepare(mirror, local, job_id=9, squeue=_UNREACHABLE)
    assert r.returncode == 0, r.stderr
    assert _remote_wip_refs(mirror) == {_ref(1), _ref(2)}
    assert "retired" not in r.stdout


@_bash
@_needs_git
def test_without_squeue_nothing_is_retired(shared):
    mirror, local = _seed(shared, [(1, "2026-09-10T00:00:00+00:00"),
                                   (2, "2026-09-11T00:00:00+00:00")],
                          squeue=None)
    r = _run_prepare(mirror, local, job_id=9, squeue=None)
    assert r.returncode == 0, r.stderr
    assert _remote_wip_refs(mirror) == {_ref(1), _ref(2)}

    # ...and it says so.  Retention is the only thing keeping the ref set
    # finite; silence here is indistinguishable from "nothing to retire",
    # which is how unbounded accumulation would come back unnoticed.
    assert "squeue is not on PATH" in r.stdout
    assert "2 left" in r.stdout
    assert "issue 14" in r.stdout


@_bash
@_needs_git
def test_an_unreachable_controller_is_reported_not_swallowed(shared):
    """A scheduler that answers with an error is the dangerous case: squeue
    exists, so nothing looks wrong, and every ref is conservatively kept."""
    mirror, local = _seed(shared, [(1, "2026-09-10T00:00:00+00:00"),
                                   (2, "2026-09-11T00:00:00+00:00")])
    r = _run_prepare(mirror, local, job_id=9, squeue=_UNREACHABLE)
    assert r.returncode == 0, r.stderr
    assert _remote_wip_refs(mirror) == {_ref(1), _ref(2)}
    assert "no usable answer for 2 WIP" in r.stdout
    assert "retired" not in r.stdout


@_bash
@_needs_git
def test_a_running_job_is_not_reported_as_unchecked(shared):
    """The distinction the report rests on: a ref left alone because its job
    is running is retention working, and must not read as a failure to
    check.  Job 1 runs; 2 and 3 have ended."""
    mirror, local = _seed(shared, [(1, "2026-09-10T00:00:00+00:00"),
                                   (2, "2026-09-11T00:00:00+00:00"),
                                   (3, "2026-09-12T00:00:00+00:00")])
    r = _run_prepare(mirror, local, job_id=9, squeue=_ONE_LIVE)
    assert r.returncode == 0, r.stderr
    assert _ref(1) in _remote_wip_refs(mirror)
    assert "no usable answer" not in r.stdout
    assert "squeue is not on PATH" not in r.stdout


@_bash
@_needs_git
def test_this_jobs_own_snapshot_is_never_retired(shared):
    """A re-run of prepare inside one job must not delete that job's own
    snapshot: it is the live record of the tree being worked on."""
    mirror, local = _seed(shared, [(1, "2026-09-10T00:00:00+00:00"),
                                   (7, "2026-09-11T00:00:00+00:00"),
                                   (8, "2026-09-12T00:00:00+00:00")])
    r = _run_prepare(mirror, local, job_id=7, squeue=_GONE)
    assert r.returncode == 0, r.stderr
    assert _ref(7) in _remote_wip_refs(mirror)


@_bash
@_needs_git
def test_legacy_shared_ref_is_retired_once_a_newer_job_ref_exists(shared):
    """The legacy ref goes through the same rule as any other candidate:
    it survives while it is the newest ended snapshot and is retired once
    a per-job one supersedes it."""
    mirror, local = shared
    assert _run_prepare(mirror, local, job_id=1, squeue=_GONE).returncode == 0
    work = local / "job1" / "mirrors" / "K"
    (work / "old.txt").write_text("from a pre-upgrade timer\n")
    _push_snapshot(work, mirror, "refs/sucoder/wip/K", _subject(77),
                   "2026-09-10T00:00:00+00:00")
    (work / "old.txt").unlink()

    # Alone, it is the newest ended snapshot and is kept.
    r = _run_prepare(mirror, local, job_id=9, squeue=_GONE)
    assert "refs/sucoder/wip/K" in _remote_wip_refs(mirror), r.stdout

    (work / "new.txt").write_text("from a current timer\n")
    _push_snapshot(work, mirror, _ref(5), _subject(5), "2026-09-12T00:00:00+00:00")
    (work / "new.txt").unlink()

    r = _run_prepare(mirror, local, job_id=10, squeue=_GONE)
    assert r.returncode == 0, r.stderr
    assert _remote_wip_refs(mirror) == {_ref(5)}
    assert "refs/sucoder/wip/K" in r.stdout          # named as retired
