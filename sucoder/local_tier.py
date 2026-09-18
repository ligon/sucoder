"""Local-disk tiering: a disposable working clone on node-local disk.

Design and spike results: ``docs/local-disk-tiering.org``.  The shared
mirror (``~/mirrors/<name>``, the laptop's push/pull target) stays the
durable repository; the agent works in a full clone under
``<local_disk>/job<ID>/mirrors/<token>`` whose ``origin`` points back at
the shared mirror.  A ``post-commit`` hook publishes every commit to the
mirror the moment it exists, and the deadline timer
(``slurm_timer``) snapshots the dirty tree to
``refs/sucoder/wip-job/<token>/<job>`` there.  Slurm's epilog wipes
``<local_disk>/job<ID>`` when the job ends, so there is nothing to clean
up and nothing to orphan.

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
JOB=@JOB_REF@
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

# Restore uncommitted work from a WIP snapshot -- but only onto the exact
# commit it was taken from, only into a clean tree, and only from a job
# that is not still running somewhere else.
#
# That last gate is issue 19.  Two jobs on one mirror have two separate
# clones on two nodes, and are normally on the same branch at the same
# commit, so parent==HEAD alone happily imported the OTHER job's dirty
# tree and announced it as restored work.  Nothing in the message said
# whose it was.  Both halves are fixed here: pick by job, and say which.
#
# Two ref shapes are candidates.  Current snapshots are per job,
# refs/sucoder/wip-job/<token>/<job>.  The legacy shared
# refs/sucoder/wip/<token> is still read because a pre-upgrade timer on
# another live job keeps writing it; its job id exists only in the
# snapshot's subject.
WIP_NS="refs/sucoder/wip-job/$TOKEN"
WIP_LEGACY="refs/sucoder/wip/$TOKEN"
# One wildcard refspec rather than two explicit ones: naming a ref that
# origin does not have fails the whole fetch, and a wildcard that matches
# nothing does not.  Origin is this mirror's own repository, so everything
# under refs/sucoder/ there is this mirror's.
git fetch --quiet --prune origin "+refs/sucoder/*:refs/sucoder/*" 2>/dev/null || true

wip_job_of() {
    # A per-job ref carries the id in its name; a legacy one only in the
    # subject the snapshotter writes, "WIP snapshot <date> job <ID>".
    case "$1" in
        "$WIP_NS"/*) printf '%s\n' "${1##*/}" ;;
        *) git log -1 --format=%s "$1" 2>/dev/null |
               sed -n 's/^.*[[:space:]]job[[:space:]]\{1,\}\([0-9][0-9]*\)[[:space:]]*$/\1/p' ;;
    esac
}

wip_job_gone() {
    # Mirrors the launcher's scheduler convention (cli.py, _slurm_job_node):
    # a successful EMPTY query and an "invalid job id" error are the only
    # answers accepted as "the job is gone".  Any other failure is unknown
    # state and must not be read as gone.
    local out rc
    out=$(squeue --job "$1" --noheader -o '%T' 2>&1); rc=$?
    if [ "$rc" -eq 0 ]; then
        [ -z "$out" ]
        return
    fi
    case "$out" in
        *[Ii]nvalid\ job\ id*) return 0 ;;
        *) return 1 ;;
    esac
}

wip_head=$(git rev-parse HEAD)
wip_have_squeue=0
command -v squeue >/dev/null 2>&1 && wip_have_squeue=1
wip_own="" wip_own_job=""
wip_best="" wip_best_job="" wip_best_unchecked=0
wip_stale=""
while read -r _when ref; do
    [ -n "$ref" ] || continue
    parent=$(git rev-parse -q --verify "$ref^" 2>/dev/null) || continue
    if [ "$parent" != "$wip_head" ]; then
        wip_stale="${wip_stale:+$wip_stale }$ref"
        continue
    fi
    job=$(wip_job_of "$ref")
    if [ -n "$JOB" ] && [ "$job" = "$JOB" ]; then
        # This job's own snapshot: a re-run of prepare inside one job.
        wip_own="$ref" wip_own_job="$job"
        continue
    fi
    [ -n "$wip_best" ] && continue      # a newer eligible one already won
    if [ -z "$job" ]; then
        echo "SUCODER: WIP snapshot $ref records no job id; not restored" >&2
        continue
    fi
    if [ "$wip_have_squeue" -eq 0 ]; then
        wip_best="$ref" wip_best_job="$job" wip_best_unchecked=1
    elif wip_job_gone "$job"; then
        wip_best="$ref" wip_best_job="$job"
    else
        echo "SUCODER: WIP snapshot $ref belongs to job $job, which is still running or could not be checked; not restored" >&2
    fi
done <<WIP_CANDIDATES
$(git for-each-ref --sort=-committerdate --format='%(committerdate:unix) %(refname)' "$WIP_NS" "$WIP_LEGACY" 2>/dev/null)
WIP_CANDIDATES

wip_pick="$wip_own" wip_pick_job="$wip_own_job" wip_unchecked=0
if [ -z "$wip_pick" ]; then
    wip_pick="$wip_best" wip_pick_job="$wip_best_job" wip_unchecked="$wip_best_unchecked"
fi
if [ -n "$wip_pick" ] && [ -z "$(git status --porcelain)" ]; then
    if git read-tree -m -u "$wip_pick" && git reset --quiet; then
        note=""
        if [ "$wip_unchecked" -eq 1 ]; then
            note=" [squeue unavailable: could not check whether job $wip_pick_job is still running]"
        fi
        echo "SUCODER: restored uncommitted work from job $wip_pick_job, $wip_pick ($(git log -1 --format=%s "$wip_pick"))$note"
    fi
elif [ -n "$wip_pick" ]; then
    echo "SUCODER: WIP snapshot $wip_pick not restored: the working tree is not clean" >&2
else
    for ref in $wip_stale; do
        parent=$(git rev-parse -q --verify "$ref^" 2>/dev/null || true)
        echo "SUCODER: WIP snapshot $ref is from commit ${parent:0:7}, not the current $(git rev-parse --short HEAD); not restored" >&2
    done
fi

# Retire snapshots belonging to jobs that have ended (issue 14).
#
# Keyed by job, the refs are per job rather than one per mirror, so without
# this they accumulate one per allocation forever.  Deleting them does not
# itself free anything -- every snapshot is commit-tree -p HEAD, so each one
# already orphans its predecessor regardless of ref shape -- but it is what
# makes the orphans collectable at all: the gc --auto that receive-pack
# already runs on every hook push then reaps them on the mirror's own
# gc.pruneExpire.  That turns unbounded growth into one expiry window of
# churn.  No gc is run from here: it would take a lock on the human's
# repository at every launch, and on the measured workload it would find
# nothing to do.
#
# The newest ended job's snapshot is kept as a fallback: this job has not
# taken its own yet, and will not for up to wip_snapshot_minutes.  A live
# job's snapshot is never touched, and neither is one whose job could not be
# checked -- deleting on an unknown answer is the same mistake as restoring
# on one.
wip_kept=""
for ref in $(git for-each-ref --sort=-committerdate --format='%(refname)' \
             "$WIP_NS" "$WIP_LEGACY" 2>/dev/null); do
    job=$(wip_job_of "$ref")
    [ -n "$job" ] || continue
    [ -n "$JOB" ] && [ "$job" = "$JOB" ] && continue     # our own, still in use
    [ "$wip_have_squeue" -eq 1 ] || continue             # cannot tell: leave it
    wip_job_gone "$job" || continue                      # live or unknown: leave it
    if [ -z "$wip_kept" ]; then
        wip_kept="$ref"                                  # newest ended job: fallback
        continue
    fi
    if git push --quiet origin ":$ref" 2>/dev/null; then
        git update-ref -d "$ref" 2>/dev/null || true
        echo "SUCODER: retired WIP snapshot $ref (job $job has ended)"
    fi
done
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
    the sanitized mirror name (``config.sanitize_session_token``).  The
    script is *executed*, not sourced: every early ``exit`` is its own,
    and the batch body recomputes the same paths from the same inputs
    (:func:`work_path_shell`, :func:`cache_exports_sh`).

    ``job_id`` ``None`` means "read ``$SLURM_JOB_ID`` at run time", as in
    :func:`slurm_timer.build_timer_script`.  The restore gate needs it to
    recognise this job's own snapshot; with no id it still runs, and simply
    has no own-snapshot case to prefer.
    """
    hook = POST_COMMIT_HOOK.rstrip("\n")
    return (
        _PREPARE_TEMPLATE
        .replace("@MIRROR@", shlex.quote(mirror_path))
        .replace("@TOKEN@", shlex.quote(mirror_token))
        .replace("@LOCAL_ROOT@", local_root_shell(local_disk_root, job_id))
        .replace("@JOB_REF@", '"${SLURM_JOB_ID:-}"' if job_id is None else shlex.quote(str(job_id)))
        .replace("@HOOK@", hook)
    )
