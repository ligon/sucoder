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
state to ``refs/sucoder/wip/<mirror>`` on its ``origin`` every
``snapshot_minutes`` and at each warning.  A tree with no ``origin`` is
never snapshotted: there is nowhere durable for the snapshot to go, and
``git add -A`` on a shared filesystem is not free (see
``docs/local-disk-tiering.org``).

It lives in its own module because ``cli`` imports ``mirror``, not the
reverse, so a builder both can share cannot live in either.  The script
is a plain template with ``@TOKEN@`` placeholders rather than an
f-string: it is dense with ``$`` and ``{}``.
"""

from __future__ import annotations

import shlex
import hashlib
from typing import Optional

from .timer_lifecycle import TIMER_LIFECYCLE_SH


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

# Snapshot the dirty tree of $SNAPSHOT_DIR to refs/sucoder/wip/$MIRROR_TOKEN
# on origin.  Every step is best-effort: the timer must never die because
# a snapshot could not be taken.  Runs in a subshell so cd/trap/export do
# not leak.  A temporary index leaves the agent's real index untouched;
# the "last tree" marker lives under .git/ because ``add -A`` would sweep
# up a marker in the working tree and defeat the unchanged check.
# Behaviour verified by hand on n0036.savio4, 2026-09-09.
WIP_SNAPSHOT_SH = r'''
snapshot_wip() {
    [ -n "$SNAPSHOT_DIR" ] || return 0
    [ -e "$SNAPSHOT_DIR/.git" ] || return 0
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
        git update-ref "refs/sucoder/wip/$MIRROR_TOKEN" "$wip" || exit 0
        if git push --quiet --force origin "refs/sucoder/wip/$MIRROR_TOKEN" >/dev/null 2>&1; then
            echo "$tree" > "$marker"
        fi
    )
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
TIMER_SCOPE=@TIMER_SCOPE@
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

@SNAPSHOT_WIP@

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
        snapshot_wip
    elif [ "$mins" -le 15 ] && [ ! -f "$WARN15" ]; then
        warn "SLURM: ~${mins} min left (job $JOB). Start wrapping up."
        touch "$WARN15" "$WARN30"
        snapshot_wip
    elif [ "$mins" -le 30 ] && [ ! -f "$WARN30" ]; then
        warn "SLURM: ~${mins} min left (job $JOB)."
        touch "$WARN30"
        snapshot_wip
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
) -> str:
    """Render the timer script.

    ``mirror_token`` names the per-mirror state files and the WIP ref; the
    caller sanitizes it (``mirror._sanitize_session_token``).  ``job_id``
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

    Every user-controlled value is shell-quoted.  The only unquoted
    substitutions are the ``$SLURM_JOB_ID`` reference itself and a
    caller-quoted ``snapshot_dir_shell``.
    """
    if snapshot_minutes < 0:
        raise ValueError("snapshot_minutes must be >= 0")
    if snapshot_dir is not None and snapshot_dir_shell is not None:
        raise ValueError("give snapshot_dir or snapshot_dir_shell, not both")
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
