"""Deadline watchdog + WIP snapshotter for SLURM-backed agent sessions.

One bash script serves both launch modes:

- *unconfined* (``salloc`` + compute-node SSH): ``cli._start_slurm_timer``
  renders it with a literal job id and starts it over SSH;
- *confined* (``sbatch``): ``MirrorManager._launch_confined`` stages it to
  NFS next to the batch script, and the batch body starts it *inside the
  job cgroup* once the tmux session is up.  The job id is read from
  ``$SLURM_JOB_ID`` because it does not exist when the script is rendered.

The script warns at 30/15/5 minutes before the allocation's ``--time``
(``tmux display-message`` for the human, a sentinel file for the agent)
and, when ``snapshot_dir`` is set, snapshots that working tree's dirty
state to ``refs/sucoder/wip-job/<mirror>/<job>`` on its ``origin`` every
``snapshot_minutes`` and at each warning.  The job id is in the ref, not
just in the snapshot's subject: two jobs on one mirror have two separate
working clones on two nodes, so one slot per mirror meant the later one
force-pushed over the earlier one's only copy (issue 19).  A tree with no ``origin`` is
never snapshotted: there is nowhere durable for the snapshot to go, and
``git add -A`` on a shared filesystem is not free (see
``docs/local-disk-tiering.org``).

With ``codex_state_scope`` set (a managed Codex launch,
``docs/codex-local-state.org``) each deadline warning also asks the
staged state runner for a capture by touching the current run's
``checkpoint.request`` on local disk.  The timer never does the capture
itself: the runner polls that file, consumes it and runs one
single-flight backup to the durable root ("Scheduling and outages": the
timer enqueues through a nonblocking local control path and does no
backup, checksumming or upload inline).

It lives in its own module because ``cli`` imports ``mirror``, not the
reverse, so a builder both can share cannot live in either.  The script
is a plain template with ``@TOKEN@`` placeholders rather than an
f-string: it is dense with ``$`` and ``{}``.
"""

from __future__ import annotations

import shlex
import hashlib
import re
from typing import Optional

from .local_tier import local_root_shell
from .timer_lifecycle import TIMER_LIFECYCLE_SH

# The managed-state scope digest (``codex_state.scope_id``): the runner
# fails closed on every identifier it is handed, and this one becomes a
# path component here, so the builder does too.
_SCOPE_DIGEST = re.compile(r"[0-9a-f]{64}")


def timer_identity(mirror_name: str, target_name: Optional[str]) -> str:
    """Avoid collisions between sanitized names and targets sharing HOME."""
    return hashlib.sha256(repr((mirror_name, target_name)).encode()).hexdigest()[:24]

# Converts SLURM ``squeue -o %L`` time-left into whole minutes.  ``%L``
# renders as ``D-HH:MM:SS`` once a day or more remains, ``HH:MM:SS`` under
# a day, ``MM:SS`` under an hour; a job with no limit prints ``UNLIMITED``.
# The day component is split off on ``-`` first (bash reads ``D-HH`` as
# arithmetic), leading zeros are forced to base 10 (``08``/``09`` are not
# octal), and non-numeric values return a large sentinel so no warning
# ever fires.  Unit-tested under bash in tests/test_slurm_timer.py.
TIME_LEFT_TO_MINS_SH = r'''
left_to_mins() {
    local s="$1" days=0 rest a b c
    if [ -z "$s" ]; then echo 999999; return; fi
    case "$s" in
        *-*) days="${s%%-*}"; rest="${s#*-}" ;;
        *)   rest="$s" ;;
    esac
    IFS=: read -r a b c <<< "$rest"
    if [ -z "$c" ]; then b="$a"; a=0; fi
    case "${days}${a}${b}" in
        *[!0-9]*) echo 999999; return ;;
    esac
    echo $(( 10#${days:-0}*1440 + 10#${a:-0}*60 + 10#${b:-0} ))
}
'''.strip("\n")

# Snapshot the dirty tree of $SNAPSHOT_DIR to
# refs/sucoder/wip-job/$MIRROR_TOKEN/$JOB on origin.  Keyed by job as well
# as mirror: the ref is the only durable record of an uncommitted tree, and
# one slot per mirror let a second job on the same mirror overwrite it
# (issue 19).  It is a SIBLING of the legacy refs/sucoder/wip/$MIRROR_TOKEN,
# not a child, because an existing ref at the parent path blocks creating
# one below it -- and --atomic does not rescue a delete-plus-create, so
# nesting would need a window with no snapshot at all.  The prepare script
# reads both shapes.
# Every step is best-effort: the timer must never die because a snapshot
# could not be taken.  Runs in a subshell so cd/trap/export do
# not leak.  A temporary index leaves the agent's real index untouched;
# the "last tree" marker lives under .git/ because ``add -A`` would sweep
# up a marker in the working tree and defeat the unchanged check.
# Behaviour verified by hand on n0036.savio4, 2026-09-09.
WIP_SNAPSHOT_SH = r'''
snapshot_wip() {
    [ -n "$SNAPSHOT_DIR" ] || return 0
    [ -e "$SNAPSHOT_DIR/.git" ] || return 0
    if ! command -v git >/dev/null 2>&1; then
        # Every git step below is `|| exit 0`, deliberately, so a snapshot
        # failure can never kill the watchdog.  The cost is that a missing
        # git makes the whole feature a permanent no-op with no
        # diagnostics -- indistinguishable from "the tree was clean".
        # Say so once per job, to the warn file the prompts already poll.
        # Once, not per cycle: at the default 10-minute interval a
        # multi-day allocation would otherwise write hundreds of copies.
        # The marker carries $JOB, so it is self-clearing -- a later job
        # in the same $HOME reports again rather than inheriting silence.
        _nogit_marker="${STATE_DIR:-${TMPDIR:-/tmp}}/.sucoder-nogit-$MIRROR_TOKEN-$JOB"
        [ -e "$_nogit_marker" ] && return 0
        : > "$_nogit_marker" 2>/dev/null || true
        if [ -n "${WARN_FILE:-}" ]; then
            printf '%s\n' "sucoder: git is not on PATH in this job's environment; WIP snapshots are disabled for job $JOB." \
                >> "$WARN_FILE" 2>/dev/null || true
        fi
        return 0
    fi
    (
        cd "$SNAPSHOT_DIR" || exit 0
        git remote get-url origin >/dev/null 2>&1 || exit 0
        marker="$(git rev-parse --git-dir)/sucoder-last-wip-tree"
        GIT_INDEX_FILE=$(mktemp) || exit 0
        export GIT_INDEX_FILE
        trap 'rm -f "$GIT_INDEX_FILE"' EXIT
        git read-tree HEAD 2>/dev/null || exit 0
        git add -A 2>/dev/null || exit 0
        tree=$(git write-tree 2>/dev/null) || exit 0
        [ "$tree" = "$(cat "$marker" 2>/dev/null)" ] && exit 0
        if [ "$tree" = "$(git rev-parse 'HEAD^{tree}')" ]; then
            echo "$tree" > "$marker"
            exit 0
        fi
        wip=$(git -c user.name=sucoder-wip -c user.email=sucoder-wip@localhost \
                  commit-tree "$tree" -p HEAD \
                  -m "WIP snapshot $(date -Is) job $JOB") || exit 0
        ref="refs/sucoder/wip-job/$MIRROR_TOKEN/$JOB"
        git update-ref "$ref" "$wip" || exit 0
        if git push --quiet --force origin "$ref" >/dev/null 2>&1; then
            echo "$tree" > "$marker"
        fi
    )
}
'''.strip("\n")

# Ask the managed Codex state runner (codex_state_runner.py) for a
# capture.  The runner records this allocation's run id in
# $CODEX_STATE_DIR/latest-run and polls <that run>/checkpoint.request once
# a second, consuming it and running one single-flight capture; a request
# that lands during a capture is dropped and counted, never queued.  The
# timer only touches that file: no backup, hashing or durable I/O happens
# here, everything read or written is on local disk, and nothing is ever
# created but the request itself (a missing run or run directory means
# there is nothing to ask).  Every step is best-effort so the watchdog
# can never die because of it, and it runs in the main shell, not a
# subshell, because the two flags make each outcome log once -- first
# success, first failure -- rather than at every warning.  The log is
# timer.log (``--ensure`` redirects ``--run`` there), the only trace a
# request leaves.  Called at the deadline warnings only: the runner has
# its own capture cadence, and a job that has left the queue has nothing
# left to request.
CODEX_CHECKPOINT_SH = r'''
_codex_request_said_ok=0
_codex_request_said_fail=0
request_codex_checkpoint() {
    local run="" why="" request=""
    run=$(cat "$CODEX_STATE_DIR/latest-run" 2>/dev/null) || run=""
    if [ -z "$run" ]; then
        why="no managed run is recorded in $CODEX_STATE_DIR/latest-run"
    elif ! [[ "$run" =~ ^[0-9a-f]{32}$ ]]; then
        why="$CODEX_STATE_DIR/latest-run does not name a run"
    elif [ ! -d "$CODEX_STATE_DIR/$run" ]; then
        why="run directory $CODEX_STATE_DIR/$run is missing"
    else
        request="$CODEX_STATE_DIR/$run/checkpoint.request"
        { : > "$request"; } 2>/dev/null || why="could not touch $request"
    fi
    if [ -z "$why" ]; then
        if [ "$_codex_request_said_ok" -eq 0 ]; then
            _codex_request_said_ok=1
            echo "sucoder timer: requested a Codex state checkpoint ($request); later requests are not logged."
        fi
    elif [ "$_codex_request_said_fail" -eq 0 ]; then
        _codex_request_said_fail=1
        echo "sucoder timer: could not request a Codex state checkpoint: $why"
    fi
    return 0
}
'''.strip("\n")

# State files are per mirror: several confined mirrors share one $HOME,
# and a second timer's startup ``rm -f`` must not clear the first's
# markers.  The un-suffixed ``slurm-deadline.warn`` is still written for
# prompts that poll the legacy path; it is cleared at startup like the
# rest, or a warning from a previous job ("allocation may have ended")
# survives into a healthy new session.  It is deliberately NOT per
# mirror -- that is what the legacy path means -- so with several
# confined mirrors it is last-writer-wins; the suffixed file is the
# one to poll.
#
# The three ``@CODEX_*@`` tokens render as nothing unless a managed Codex
# scope is given.  Each sits at the end of an existing line and its
# replacement starts with the newline(s) it needs, so a script rendered
# without a scope is byte-identical to one rendered before the feature.
_TEMPLATE = r'''#!/bin/bash
# sucoder SLURM deadline timer + WIP snapshotter (generated; do not edit).
set -u
STATE_DIR="${HOME}/.cache/sucoder"
mkdir -p "$STATE_DIR"
chmod 700 "$STATE_DIR" 2>/dev/null || true
MIRROR_TOKEN=@MIRROR_TOKEN@
TMUX_SESSION=@TMUX_SESSION@
TMUX_BIN=(@TMUX_CMD@)
SNAPSHOT_DIR=@SNAPSHOT_DIR@
SNAPSHOT_MINUTES=@SNAPSHOT_MINUTES@
JOB=@JOB_REF@
TIMER_SCOPE=@TIMER_SCOPE@@CODEX_STATE_VARS@
@TIMER_LIFECYCLE@
WARN_FILE="$STATE_DIR/slurm-deadline-$MIRROR_TOKEN.warn"
LEGACY_WARN_FILE="$CACHE_DIR/slurm-deadline.warn"
MIRROR_WARN_FILE="$CACHE_DIR/slurm-deadline-$MIRROR_TOKEN.warn"
WARN5="$STATE_DIR/.slurm-warn-5-$MIRROR_TOKEN"
WARN15="$STATE_DIR/.slurm-warn-15-$MIRROR_TOKEN"
WARN30="$STATE_DIR/.slurm-warn-30-$MIRROR_TOKEN"
rm -f "$WARN5" "$WARN15" "$WARN30" "$WARN_FILE" "$LEGACY_WARN_FILE" "$MIRROR_WARN_FILE"

if [ -z "$JOB" ]; then
    echo "sucoder timer: no SLURM job id (not inside a job?); exiting." > "$WARN_FILE"
    exit 1
fi

@LEFT_TO_MINS@

@SNAPSHOT_WIP@@CODEX_CHECKPOINT@

warn() {
    echo "$1" > "$WARN_FILE"
    echo "$1" > "$LEGACY_WARN_FILE"
    echo "$1" > "$MIRROR_WARN_FILE"
    "${TMUX_BIN[@]}" display-message -t "$TMUX_SESSION" "$1" 2>/dev/null
}

# Wait for the agent tmux session to appear before monitoring.  The
# unconfined timer starts before the session is created, so its absence
# must not be read as "agent exited".
TMUX_READY=0
for i in $(seq 1 120); do
    if "${TMUX_BIN[@]}" has-session -t "$TMUX_SESSION" 2>/dev/null; then
        TMUX_READY=1
        break
    fi
    sleep 5
done
if [ "$TMUX_READY" -eq 0 ]; then
    # The user owns the SLURM lifecycle (see `sucoder release`): leave the
    # allocation alone even though the agent never appeared.
    echo "Timed out waiting for tmux session $TMUX_SESSION; @LIFECYCLE@" > "$WARN_FILE"
    exit 1
fi

# Make each warning linger on the status line so a full-screen agent TUI
# does not redraw over it before the human notices.
"${TMUX_BIN[@]}" set-option -t "$TMUX_SESSION" display-time 15000 2>/dev/null || true
echo monitoring > "$STATE_DIR/status"

elapsed=0
missed=0
while true; do
    if ! left=$(squeue --job "$JOB" --noheader -o "%L" 2>/dev/null); then
        # Failed queries are unknown state, not evidence the job ended
        # (ledger 3). Discard partial output and interrupt the empty-query
        # streak. Still check tmux and take periodic snapshots below.
        left=""
        missed=0
    elif [ -z "$left" ]; then
        # Only consecutive successful empty queries establish disappearance.
        missed=$((missed + 1))
        if [ "$missed" -ge 3 ]; then
            warn "SLURM job $JOB is no longer queued -- allocation may have ended."
            break
        fi
        sleep 60
        elapsed=$((elapsed + 1))
        continue
    else
        missed=0
    fi

    # Agent gone: record it but do NOT scancel (see above).
    if ! "${TMUX_BIN[@]}" has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "Agent tmux session is gone; @LIFECYCLE@" > "$WARN_FILE"
        break
    fi

    # A threshold that fires also marks every COARSER one spent.  Marking
    # only the one that fired let the chain fall through to a less urgent
    # branch on the next poll, so a job that started with 3 minutes left
    # warned "Commit and save NOW", then "Start wrapping up", then the
    # bare 30-minute notice -- urgency running backwards, one `git add -A`
    # sweep per spurious warning.  Same for any skipped poll (31 -> 14).
    mins=$(left_to_mins "$left")
    if [ "$mins" -le 5 ] && [ ! -f "$WARN5" ]; then
        warn "SLURM: ~${mins} min left (job $JOB). Commit and save NOW."
        touch "$WARN5" "$WARN15" "$WARN30"
        snapshot_wip@CODEX_REQUEST@
    elif [ "$mins" -le 15 ] && [ ! -f "$WARN15" ]; then
        warn "SLURM: ~${mins} min left (job $JOB). Start wrapping up."
        touch "$WARN15" "$WARN30"
        snapshot_wip@CODEX_REQUEST@
    elif [ "$mins" -le 30 ] && [ ! -f "$WARN30" ]; then
        warn "SLURM: ~${mins} min left (job $JOB)."
        touch "$WARN30"
        snapshot_wip@CODEX_REQUEST@
    fi

    if [ "$SNAPSHOT_MINUTES" -gt 0 ] && [ $((elapsed % SNAPSHOT_MINUTES)) -eq 0 ]; then
        snapshot_wip
    fi
    sleep 60
    elapsed=$((elapsed + 1))
done
'''


def build_timer_script(
    *,
    mirror_token: str,
    tmux_session: str,
    job_id: Optional[int] = None,
    tmux_socket: Optional[str] = None,
    snapshot_dir: Optional[str] = None,
    snapshot_dir_shell: Optional[str] = None,
    snapshot_minutes: int = 10,
    timer_scope: Optional[str] = None,
    codex_state_scope: Optional[str] = None,
    local_disk_root: Optional[str] = None,
) -> str:
    """Render the timer script.

    ``mirror_token`` names the per-mirror state files and the WIP ref's
    mirror component; the caller sanitizes it
    (``config.sanitize_session_token``).  ``job_id``
    ``None`` means "read ``$SLURM_JOB_ID`` at run time" (confined launches,
    where the id is unknown until sbatch assigns it).  ``tmux_socket``
    adds ``-L <socket>`` to *every* tmux call, which a confined launch
    requires: without it the timer waits on a session it cannot see and
    exits as "timed out".  ``snapshot_dir`` ``None`` or ``snapshot_minutes``
    ``0`` disables the periodic snapshot (the deadline snapshots still run
    when a directory is given).

    ``snapshot_dir_shell`` is the *unquoted* alternative to
    ``snapshot_dir``: a shell word the caller has already quoted piecewise
    (``local_tier.work_path_shell``), so a runtime ``${SLURM_JOB_ID}`` in
    it still expands.  Give at most one of the two.

    ``codex_state_scope`` (the 64-hex managed-state scope digest,
    ``codex_state.scope_id``) makes every deadline warning also request a
    capture from the staged Codex state runner (``request_codex_checkpoint``
    above).  It needs ``local_disk_root``, the tiering root the launch
    uses: the runner's control files live under
    ``<root>/job<ID>/state/codex/<scope>/`` (the same ``SUCODER_LOCAL_ROOT``
    it is given, rendered by :func:`local_tier.local_root_shell` with the
    literal or runtime job id), and the periodic cadence stays the
    runner's own.  Without a scope the rendered script is byte-identical
    to one without the feature.

    Every user-controlled value is shell-quoted.  The only unquoted
    substitutions are the ``$SLURM_JOB_ID`` reference itself and a
    caller-quoted ``snapshot_dir_shell``.
    """
    if snapshot_minutes < 0:
        raise ValueError("snapshot_minutes must be >= 0")
    if snapshot_dir is not None and snapshot_dir_shell is not None:
        raise ValueError("give snapshot_dir or snapshot_dir_shell, not both")
    codex_vars = codex_fn = codex_call = ""
    if codex_state_scope is not None:
        if not _SCOPE_DIGEST.fullmatch(codex_state_scope):
            raise ValueError("codex_state_scope must be the 64-hex scope digest")
        if not local_disk_root:
            raise ValueError("codex_state_scope needs local_disk_root (managed state is local-disk only)")
        # Layout owned by codex_state_runner.py (``scope_dir``): the local
        # root the runner resolves from SUCODER_LOCAL_ROOT, then
        # state/codex/<scope>.  Only ${SLURM_JOB_ID} is unquoted.
        state_dir = (
            f"{local_root_shell(local_disk_root, job_id)}/state/codex/"
            f"{shlex.quote(codex_state_scope)}"
        )
        codex_vars = f"\nCODEX_STATE_DIR={state_dir}"
        codex_fn = "\n\n" + CODEX_CHECKPOINT_SH
        codex_call = "\n        request_codex_checkpoint"
    snapshot_word = (
        snapshot_dir_shell if snapshot_dir_shell is not None
        else shlex.quote(snapshot_dir or "")
    )
    tmux_cmd = "tmux" if tmux_socket is None else f"tmux -L {shlex.quote(tmux_socket)}"
    job_ref = '"${SLURM_JOB_ID:-}"' if job_id is None else shlex.quote(str(job_id))
    # What happens to the allocation when the agent's session goes away is
    # the opposite in the two modes, and telling a confined user to run
    # `scancel` on a job that already completed is worse than saying
    # nothing.  Under sbatch the batch body's keeper loop polls the same
    # session, so the job ends with it; under salloc the user owns the
    # allocation and it survives.
    if job_id is None:
        lifecycle = (
            "SLURM job $JOB ends with it (the batch body exits when the "
            "session does)."
        )
    else:
        lifecycle = (
            "SLURM job $JOB kept alive. Run 'sucoder release' or "
            "'scancel $JOB' to free the allocation."
        )
    return (
        _TEMPLATE
        # First, so a later user value can never be re-expanded by them.
        .replace("@CODEX_STATE_VARS@", codex_vars)
        .replace("@CODEX_CHECKPOINT@", codex_fn)
        .replace("@CODEX_REQUEST@", codex_call)
        .replace("@TIMER_SCOPE@", shlex.quote(timer_scope or timer_identity(mirror_token, None)))
        .replace("@TIMER_LIFECYCLE@", TIMER_LIFECYCLE_SH)
        .replace("@MIRROR_TOKEN@", shlex.quote(mirror_token))
        .replace("@TMUX_SESSION@", shlex.quote(tmux_session))
        .replace("@TMUX_CMD@", tmux_cmd)
        .replace("@SNAPSHOT_DIR@", snapshot_word)
        .replace("@SNAPSHOT_MINUTES@", str(int(snapshot_minutes)))
        .replace("@JOB_REF@", job_ref)
        .replace("@LIFECYCLE@", lifecycle)
        .replace("@LEFT_TO_MINS@", TIME_LEFT_TO_MINS_SH)
        .replace("@SNAPSHOT_WIP@", WIP_SNAPSHOT_SH)
    )
