"""Local-disk tiering: a disposable working clone on node-local disk.

Design and spike results: ``docs/local-disk-tiering.org``.  The shared
mirror (``~/mirrors/<name>``, the laptop's push/pull target) stays the
durable repository; the agent works in a full clone under
``<local_disk>/job<ID>/mirrors/<token>`` whose ``origin`` points back at
the shared mirror.  A ``post-commit`` hook publishes every commit to the
mirror the moment it exists, and the deadline timer
(``slurm_timer``) snapshots the dirty tree to ``refs/sucoder/wip/<token>``
there.  Slurm's epilog wipes ``<local_disk>/job<ID>`` when the job ends,
so there is nothing to clean up and nothing to orphan.

Layout is a pure function of three inputs (local disk root, job id,
mirror token) so the batch body, the timer, and the launcher agree on
paths without passing them around:

- ``<root>/job<ID>/mirrors/<token>``  the working clone (agent cwd)
- ``<root>/job<ID>/cache/{uv,pip,npm}``  tool caches
- ``<root>/job<ID>/tmp``  ``$TMPDIR``

The job id is a runtime ``${SLURM_JOB_ID}`` for confined (sbatch)
launches, where it does not exist when the scripts are rendered, so the
path helpers come in a literal and a shell-expression flavour.
"""

from __future__ import annotations

import shlex
from typing import Optional

DEFAULT_LOCAL_DISK = "/local"


def local_root_shell(local_disk_root: str, job_id: Optional[int] = None) -> str:
    """``<root>/job<ID>`` as a shell word: quoted literal when ``job_id`` is
    known, otherwise ``<root>/job"${SLURM_JOB_ID}"`` for the batch body."""
    root = local_disk_root.rstrip("/") or "/"
    if job_id is not None:
        return shlex.quote(f"{root}/job{job_id}")
    return f'{shlex.quote(root)}/job"${{SLURM_JOB_ID}}"'


def work_path(local_disk_root: str, mirror_token: str, job_id: int) -> str:
    """The working clone's path as a plain string (job id known)."""
    root = local_disk_root.rstrip("/") or "/"
    return f"{root}/job{job_id}/mirrors/{mirror_token}"


def work_path_shell(local_disk_root: str, mirror_token: str, job_id: Optional[int] = None) -> str:
    """The working clone's path as a shell word (see :func:`local_root_shell`)."""
    return f"{local_root_shell(local_disk_root, job_id)}/mirrors/{shlex.quote(mirror_token)}"


def cache_exports_sh(local_disk_root: str, job_id: Optional[int] = None) -> str:
    """One ``export`` line pointing tool caches and ``$TMPDIR`` at local disk.

    Emitted by the batch body *before* ``tmux new-session`` so the new tmux
    server inherits it; it cannot go through the launcher's ``env`` dict,
    which is shell-quoted into the window command and would make
    ``${SLURM_JOB_ID}`` literal.  Tool *installs* (``UV_TOOL_DIR``, the npm
    prefix) are deliberately not redirected: they must outlive the job.
    """
    root = local_root_shell(local_disk_root, job_id)
    return (
        f'export SUCODER_LOCAL_ROOT={root} '
        f'UV_CACHE_DIR={root}/cache/uv PIP_CACHE_DIR={root}/cache/pip '
        f'npm_config_cache={root}/cache/npm TMPDIR={root}/tmp'
    )


# Installed into the working clone.  Never forces (a force would clobber
# the laptop's push into the shared mirror); never fails the commit.
# The ``sucoder local-tier`` marker is how the prepare script recognises
# its own hook on a re-run and leaves any other hook alone.
POST_COMMIT_HOOK = r'''#!/bin/bash
# sucoder local-tier: publish every commit to the shared mirror (origin).
branch=$(git symbolic-ref --short -q HEAD) || exit 0   # detached: nothing to publish
if ! out=$(git push --quiet origin HEAD:refs/heads/"$branch" 2>&1); then
    echo "SUCODER: post-commit push to origin REJECTED; run 'git pull --ff-only' and commit again" >&2
    echo "$out" >&2
    exit 0
fi
echo "SUCODER: published $(git rev-parse --short HEAD) -> origin/$branch"
'''

_PREPARE_TEMPLATE = r'''#!/bin/bash
# sucoder local-tier prepare (generated; do not edit).  Idempotent: a
# re-run in the same job refreshes the clone instead of recreating it.
set -u
MIRROR=@MIRROR@
TOKEN=@TOKEN@
LOCAL_ROOT=@LOCAL_ROOT@
WORK="$LOCAL_ROOT/mirrors/$TOKEN"
umask 077
mkdir -p "$LOCAL_ROOT/mirrors" "$LOCAL_ROOT/cache/uv" "$LOCAL_ROOT/cache/pip" \
         "$LOCAL_ROOT/cache/npm" "$LOCAL_ROOT/tmp" || exit 1

if [ ! -e "$MIRROR/.git" ]; then
    echo "SUCODER: shared mirror $MIRROR is not a git repository" >&2
    exit 1
fi
branch=$(git -C "$MIRROR" symbolic-ref --short -q HEAD)
if [ -z "$branch" ]; then
    echo "SUCODER: shared mirror $MIRROR has a detached HEAD; cannot pick a branch" >&2
    exit 1
fi
# The shared mirror is a mailbox, not a desk: a tracked edit there makes
# updateInstead refuse every publish from the working clone.
if ! git -C "$MIRROR" diff --quiet || ! git -C "$MIRROR" diff --cached --quiet; then
    echo "SUCODER: shared mirror $MIRROR has uncommitted changes to tracked files; commits from the local clone will be REJECTED until it is clean" >&2
fi

if [ ! -e "$WORK/.git" ]; then
    # Invariant before the only destructive step: WORK is always
    # <root>/job<ID>/mirrors/<token>.  set -u already aborts on an unset
    # ${SLURM_JOB_ID}; this documents the shape and refuses anything else.
    case "$WORK" in
        */job*/mirrors/?*) ;;
        *) echo "SUCODER: refusing to clear unexpected work path $WORK" >&2; exit 1 ;;
    esac
    rm -rf "$WORK"
    if ! git clone --quiet --no-hardlinks "$MIRROR" "$WORK"; then
        echo "SUCODER: clone of $MIRROR to $WORK failed" >&2
        exit 1
    fi
fi
cd "$WORK" || exit 1
git remote set-url origin "$MIRROR"
git fetch --quiet origin || echo "SUCODER: fetch from $MIRROR failed; continuing with the clone as is" >&2
if ! git checkout --quiet "$branch" 2>/dev/null; then
    git checkout --quiet -b "$branch" "origin/$branch" || exit 1
fi
if ! git merge --quiet --ff-only "origin/$branch" 2>/dev/null; then
    echo "SUCODER: $WORK and $MIRROR ($branch) have diverged; reconcile with 'git pull --ff-only' before committing" >&2
fi

hook="$(git rev-parse --git-dir)/hooks/post-commit"
if [ ! -e "$hook" ] || grep -q 'sucoder local-tier' "$hook" 2>/dev/null; then
    cat > "$hook" <<'SUCODER_HOOK'
@HOOK@
SUCODER_HOOK
    chmod 700 "$hook"
else
    echo "SUCODER: existing post-commit hook at $hook left alone; commits will NOT auto-publish to $MIRROR" >&2
fi

# Restore uncommitted work from the last WIP snapshot, but only onto the
# exact commit it was taken from and only into a clean tree.  The ref is
# left in place: the next snapshot overwrites it.
wip="refs/sucoder/wip/$TOKEN"
if git fetch --quiet origin "+$wip:$wip" 2>/dev/null; then
    parent=$(git rev-parse -q --verify "$wip^" 2>/dev/null || true)
    if [ -n "$parent" ] && [ "$parent" = "$(git rev-parse HEAD)" ] && [ -z "$(git status --porcelain)" ]; then
        if git read-tree -m -u "$wip" && git reset --quiet; then
            echo "SUCODER: restored uncommitted work from $wip ($(git log -1 --format=%s "$wip"))"
        fi
    elif [ -n "$parent" ]; then
        echo "SUCODER: WIP snapshot $wip is from commit ${parent:0:7}, not the current $(git rev-parse --short HEAD); not restored" >&2
    fi
fi
echo "SUCODER: local-tier working clone ready at $WORK ($branch)"
'''


def build_prepare_script(
    *,
    mirror_path: str,
    mirror_token: str,
    local_disk_root: str = DEFAULT_LOCAL_DISK,
    job_id: Optional[int] = None,
) -> str:
    """Render the prepare script for one mirror.

    ``mirror_path`` is the shared mirror's absolute path; ``mirror_token``
    the sanitized mirror name (``mirror._sanitize_session_token``).  The
    script is *executed*, not sourced: every early ``exit`` is its own,
    and the batch body recomputes the same paths from the same inputs
    (:func:`work_path_shell`, :func:`cache_exports_sh`).
    """
    hook = POST_COMMIT_HOOK.rstrip("\n")
    return (
        _PREPARE_TEMPLATE
        .replace("@MIRROR@", shlex.quote(mirror_path))
        .replace("@TOKEN@", shlex.quote(mirror_token))
        .replace("@LOCAL_ROOT@", local_root_shell(local_disk_root, job_id))
        .replace("@HOOK@", hook)
    )
