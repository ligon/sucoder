"""Bash timer supervision without process-name matching or PID-based killing."""

TIMER_LIFECYCLE_SH = r'''
CACHE_DIR="$STATE_DIR"
node=$(hostname) || exit 1
[[ "$node" =~ ^[a-zA-Z0-9._-]+$ && "$JOB" =~ ^[0-9]+$ ]] || {
    echo "sucoder timer: invalid node or allocation identity" >&2; exit 1;
}
# Locks and liveness records are node-local, just like /proc. Shared HOME
# may be NFS without flock support; TMPDIR may also point at shared storage.
# Never follow or chmod a pre-created path in world-writable /tmp.
RUNTIME_DIR="/tmp/sucoder-$UID"
umask 077
mkdir -m 700 "$RUNTIME_DIR" 2>/dev/null || true
if [ -L "$RUNTIME_DIR" ] || [ ! -d "$RUNTIME_DIR" ] ||
   [ ! -O "$RUNTIME_DIR" ] || [ "$(stat -c %a "$RUNTIME_DIR")" != 700 ]; then
    echo "sucoder timer: unsafe runtime directory: $RUNTIME_DIR" >&2
    exit 1
fi
STATE_DIR="$RUNTIME_DIR/timers/$TIMER_SCOPE/$node-$JOB"
mkdir -p "$STATE_DIR" || exit 1
chmod 700 "$STATE_DIR" || exit 1
command -v flock >/dev/null || { echo "sucoder timer: flock unavailable" >&2; exit 1; }

process_start() {
    local stat
    stat=$(cat "/proc/$1/stat" 2>/dev/null) || return 1
    # Strip comm (which can contain spaces/parentheses); starttime is the
    # twentieth field after it. No PID is ever used to signal a process.
    stat=${stat##*) }
    local fields=($stat)
    printf '%s\n' "${fields[19]}"
}
live_timer() {
    local pid stamp
    read -r pid stamp < "$STATE_DIR/owner" || return 1
    [[ "$pid" =~ ^[0-9]+$ && -n "$stamp" ]] || return 1
    [ "$(process_start "$pid")" = "$stamp" ]
}

if [ "${1:-}" = --ensure ]; then
    # Serialize starters. The watchdog closes this fd so it cannot retain
    # the startup lock for its lifetime. Locks are scoped by node and job.
    exec 8>"$STATE_DIR/start.lock"
    flock -w 10 8 || { echo "sucoder timer: startup lock busy" >&2; exit 1; }
    exec 9>"$STATE_DIR/run.lock"
    if ! flock -n 9; then
        if live_timer && [ -s "$STATE_DIR/status" ]; then
            echo "SUCODER_TIMER_REUSED $(cat "$STATE_DIR/status")"
            exit 0
        fi
        echo "sucoder timer: lock held but owner is not healthy; retry" >&2
        exit 1
    fi
    flock -u 9
    rm -f "$STATE_DIR/owner" "$STATE_DIR/status"
    nohup bash "$0" --run </dev/null >>"$STATE_DIR/timer.log" 2>&1 8>&- 9>&- &
    child=$!
    for attempt in {1..50}; do
        if live_timer 2>/dev/null && [ -s "$STATE_DIR/status" ]; then
            echo "SUCODER_TIMER_STARTED $(cat "$STATE_DIR/status")"
            exit 0
        fi
        kill -0 "$child" 2>/dev/null || break
        sleep 0.1
    done
    echo "sucoder timer: startup failed; see $STATE_DIR/timer.log" >&2
    tail -n 5 "$STATE_DIR/timer.log" >&2
    exit 1
fi

exec 9>"$STATE_DIR/run.lock"
flock -n 9 || exit 0
for executable in tmux squeue; do
    command -v "$executable" >/dev/null || {
        echo "sucoder timer: $executable unavailable" >&2; exit 1;
    }
done
printf '%s %s\n' "$$" "$(process_start "$$")" > "$STATE_DIR/owner"
trap 'rm -f "$STATE_DIR/owner" "$STATE_DIR/status"' EXIT
echo waiting-for-tmux > "$STATE_DIR/status"
'''
