"""Local-disk tiering (``sucoder.local_tier``): path helpers, the prepare
script rendered for a confined launch, and the script driven under bash
against a real temporary shared mirror.  Mirrors the manual spike in
docs/local-disk-tiering.org.  Needs bash and git; never execs tmux/squeue.
"""
from __future__ import annotations

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


def _run_prepare(mirror: Path, local: Path, job_id: int = 5) -> subprocess.CompletedProcess:
    script = build_prepare_script(
        mirror_path=str(mirror), mirror_token="K", local_disk_root=str(local), job_id=job_id,
    )
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


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


@_bash
@_needs_git
def test_wip_snapshot_restored_only_onto_its_parent(shared):
    mirror, local = shared
    assert _run_prepare(mirror, local, job_id=1).returncode == 0
    work1 = local / "job1" / "mirrors" / "K"
    # Take a snapshot the way the timer does: tree with an untracked file,
    # parented on HEAD, pushed to refs/sucoder/wip/K on the mirror.
    (work1 / "wip.txt").write_text("in progress\n")
    env = dict(os.environ, GIT_INDEX_FILE=str(work1 / ".git" / "tmpidx"))
    subprocess.run(["git", "read-tree", "HEAD"], cwd=work1, env=env, check=True)
    subprocess.run(["git", "add", "-A"], cwd=work1, env=env, check=True)
    tree = subprocess.run(["git", "write-tree"], cwd=work1, env=env, capture_output=True, text=True, check=True).stdout.strip()
    wip = _git(work1, "commit-tree", tree, "-p", "HEAD", "-m", "WIP")
    _git(work1, "push", "-q", "--force", "origin", f"{wip}:refs/sucoder/wip/K")

    # New job: fresh clone restores it.
    r = _run_prepare(mirror, local, job_id=2)
    assert r.returncode == 0, r.stderr
    work2 = local / "job2" / "mirrors" / "K"
    assert (work2 / "wip.txt").read_text() == "in progress\n"
    assert "restored uncommitted work" in r.stdout
    assert _git(work2, "status", "--porcelain") == "?? wip.txt"
    assert _git(mirror, "rev-parse", "refs/sucoder/wip/K") == wip   # ref kept

    # A commit lands after the snapshot: the next job must NOT restore it.
    (mirror / "README").write_text("hello\nlater\n")
    _git(mirror, "commit", "-q", "-am", "later")
    r = _run_prepare(mirror, local, job_id=3)
    assert r.returncode == 0, r.stderr
    assert not (local / "job3" / "mirrors" / "K" / "wip.txt").exists()
    assert "not restored" in r.stderr


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
