#!/usr/bin/env bash
#
# claude-science.sh -- serve claude-science on a compute node and open it
# in the local browser, over SuCoder's warm tunnels.
#
# claude-science runs on its OWN dedicated thin SLURM slice (job csci-<target>)
# so it never shares — and fights — a compute node with heavy jobs.  The slice
# is allocated on demand and held open in a detached login-node tmux.
#
# Usage:
#   scripts/claude-science.sh [up]   warm the tunnels, reuse-or-allocate the
#                                    dedicated slice, start `claude-science
#                                    serve` on it, forward its port, and open
#                                    the URL locally.
#   scripts/claude-science.sh url    fetch a (possibly new) URL from the
#                                    RUNNING claude-science, reuse/repair the
#                                    port forward, and open it.  Fails if the
#                                    service is down (run `up` instead).
#   scripts/claude-science.sh down   release the slice: cancel the SLURM job,
#                                    kill its tmux, and remove the forward.
#
# up/url are idempotent: re-running reuses the warm sockets, the slice
# (running, or still queued — re-running attaches rather than resubmitting,
# so the job keeps its place in line), the server (one that exists but has
# not answered yet — e.g. a slow cold start — is waited on, NEVER
# double-started: two servers would mean two writers on one SQLite DB), and
# the existing forward.
#
# The URL is opened with garcon-url-handler (ChromeOS / Crostini); set
# OPENER to something else (xdg-open, firefox, ...) or OPENER=- to only
# print the localhost URL on stdout.
#
# Env overrides:
#   TARGET      sucoder target            (default: carleton-htc)
#   NODE        pin a node, skip the slice (default: reuse/allocate the slice)
#   APP_BIN     app path on the node      (default: $HOME/bin/claude-science)
#   SERVE_ARGS  args for `serve`          (default: --dangerously-no-sandbox)
#   OPENER      local URL opener, or "-"  (default: garcon-url-handler)
#   WAIT_SECS   serve readiness timeout   (default: 120s, wall-clock)
#   CS_SCHED_WAIT  slice scheduling ceiling  (default: 300s, wall-clock).  On
#               timeout a QUEUED slice stays queued (priority keeps accruing);
#               re-run `up` later to attach to it.
#   CS_PARTITION/CS_ACCOUNT/CS_QOS/CS_CPUS/CS_MEM/CS_TIME
#               override the slice's SLURM params (default: from sucoder config)
#
set -euo pipefail

MODE=${1:-up}
case "$MODE" in
  up|url|down) : ;;
  *) echo "usage: $0 [up|url|down]" >&2; exit 2 ;;
esac

TARGET=${TARGET:-carleton-htc}
APP_BIN=${APP_BIN:-'$HOME/bin/claude-science'}  # literal $HOME; expands REMOTE-side
SERVE_ARGS=${SERVE_ARGS:---dangerously-no-sandbox}
OPENER=${OPENER:-garcon-url-handler}
WAIT_SECS=${WAIT_SECS:-120}   # wall-clock; cold NFS DB-open can take >60s
LN_ALIAS="${TARGET}-ln"
REMOTE_LOG=".claude-science.serve.log"        # in the remote $HOME

say() { printf '%s\n' "$*" >&2; }
die() { say "ERROR: $*"; exit 1; }

# ---------------------------------------------------------------- tunnels
# Idempotent: reuses warm sockets; prompts (OTP) only when everything is
# cold.  Also (re)writes the ${TARGET}-ln ssh alias we ride below.
sucoder -T "$TARGET" tunnel up

# ----------------------------------------------------------- dedicated slice
# claude-science runs on its OWN thin SLURM slice (a job named csci-<target>)
# so it never has to share — and fight — a compute node with heavy jobs (that
# is exactly how it got starved before).  The slice's SLURM params come from
# the target's `slurm:` block in ~/.sucoder/config.yaml; CS_* env vars override.
SLICE_JOB="csci-${TARGET}"
SLICE_TMUX="csci-${TARGET}"        # login-node tmux that holds the srun client
SLICE_ERR=".csci-${TARGET}.srun.err"  # login-node file: the srun client's stderr
SCHED_WAIT=${CS_SCHED_WAIT:-300}   # wall-clock ceiling for the slice to schedule

slice_node() {  # node of my RUNNING dedicated slice, or empty
  ssh "$LN_ALIAS" "squeue --me -n '$SLICE_JOB' -h -t RUNNING -o %N" 2>/dev/null \
    | grep -vE '^[[:space:]]*$' | sort -u | head -1
}

slice_status() {  # "STATE (Reason)" of my slice in ANY queue state, or empty
  ssh "$LN_ALIAS" "squeue --me -n '$SLICE_JOB' -h -o '%T (%r)'" 2>/dev/null \
    | grep -vE '^[[:space:]]*$' | head -1
}

client_alive() {  # is the login-node tmux holding the srun client still up?
  ssh "$LN_ALIAS" "tmux has-session -t '$SLICE_TMUX'" >/dev/null 2>&1
}

launch_slice() {  # submit the slice srun inside a detached login-node tmux
  # SLURM params from the sucoder config (env overrides win); PyYAML ships with
  # the sucoder venv, so this parse mirrors whatever `collaborate` would use.
  local cfg P A Q C M T
  cfg=$(python3 - "$TARGET" <<'PY' 2>/dev/null || true
import yaml, sys, os
t = sys.argv[1]
p = os.path.expanduser("~/.sucoder/config.yaml")
s = ((yaml.safe_load(open(p)) or {}).get("targets", {}).get(t, {}) or {}).get("slurm", {}) or {}
def g(k, d=""):
    v = s.get(k, d)
    return "" if v is None else str(v)
print("P=" + g("partition"))
print("A=" + g("account"))
print("Q=" + g("qos"))
print("C=" + g("cpus_per_task", "4"))
print("M=" + g("mem", "24G"))
print("T=" + g("time", "7-00:00:00"))
PY
)
  while IFS='=' read -r k v; do case "$k" in
    P) P=$v ;; A) A=$v ;; Q) Q=$v ;; C) C=$v ;; M) M=$v ;; T) T=$v ;;
  esac; done <<<"$cfg"
  P=${CS_PARTITION:-${P:-}}; A=${CS_ACCOUNT:-${A:-}}; Q=${CS_QOS:-${Q:-}}
  C=${CS_CPUS:-${C:-4}};     M=${CS_MEM:-${M:-24G}};  T=${CS_TIME:-${T:-7-00:00:00}}
  [ -n "$P" ] || die "no SLURM partition for target '$TARGET' — set CS_PARTITION= or fix ~/.sucoder/config.yaml"
  # Keep the slice off any node already running my jobs — never co-locate with
  # heavy compute again.
  local excl; excl=$(ssh "$LN_ALIAS" 'squeue --me -h -o %N' 2>/dev/null \
                     | grep -vE '^[[:space:]]*$' | sort -u | paste -sd, -)
  local srun="srun --job-name=$SLICE_JOB --partition=$P${A:+ --account=$A}${Q:+ --qos=$Q} --cpus-per-task=$C --mem=$M --time=$T${excl:+ --exclude=$excl} --pty bash"
  say "Allocating dedicated slice '$SLICE_JOB' on $P (${C} cpu / ${M})${excl:+, off $excl} ..."
  # Held open in a detached login-node tmux so the allocation survives our
  # disconnects.  The srun client's stderr is captured in ~/$SLICE_ERR so a
  # REJECTED submission (bad partition/mem/account/...) leaves a message we
  # can show, instead of dying unseen with the tmux (blind spot found
  # 2026-07-06: a savio2,savio3 submission was rejected and the error just
  # vanished).  Clear any stale same-named session first.
  ssh "$LN_ALIAS" "tmux kill-session -t '$SLICE_TMUX' 2>/dev/null; tmux new-session -d -s '$SLICE_TMUX' '$srun 2>\$HOME/$SLICE_ERR'" \
    || die "could not launch the allocation tmux on $LN_ALIAS"
}

wait_for_slice() {  # echo the slice node once its job is RUNNING
  local node st last=""
  local deadline=$((SECONDS + SCHED_WAIT))
  while [ "$SECONDS" -lt "$deadline" ]; do
    node=$(slice_node || true)
    [ -n "$node" ] && { say "Slice scheduled on $node."; printf '%s\n' "$node"; return 0; }
    st=$(slice_status || true)
    if [ -n "$st" ]; then
      # Queued but not running yet: narrate state changes, keep waiting.
      if [ "$st" != "$last" ]; then
        say "Slice '$SLICE_JOB' is $st — waiting up to ${SCHED_WAIT}s (CS_SCHED_WAIT= raises) ..."
        last=$st
      fi
    elif ! client_alive; then
      # No job in the queue AND the srun client is gone: the submission was
      # rejected outright.  Surface the stderr we captured for it.
      local err
      err=$(ssh "$LN_ALIAS" "cat \$HOME/$SLICE_ERR 2>/dev/null" | sed 's/^/  /' || true)
      die "slice submission was rejected — srun said:
${err:-  (nothing captured in ~/$SLICE_ERR)}"
    fi  # else: client alive, job not visible yet — srun is still submitting
    sleep 5
  done
  st=$(slice_status || true)
  if [ -n "$st" ]; then
    die "slice '$SLICE_JOB' is still $st after ${SCHED_WAIT}s — it STAYS QUEUED (priority keeps accruing); re-run \`$0 up\` later to attach, or \`$0 down\` to give up the spot."
  fi
  die "slice '$SLICE_JOB' neither scheduled nor errored within ${SCHED_WAIT}s — check \`squeue --me\` and \`tmux ls\` on $LN_ALIAS"
}

ensure_slice() {  # echo the slice node; reuse running/queued, else allocate
  local node st
  node=$(slice_node || true)
  if [ -n "$node" ]; then
    say "Reusing dedicated slice '$SLICE_JOB' on $node."
    printf '%s\n' "$node"; return 0
  fi
  st=$(slice_status || true)
  if [ -n "$st" ]; then
    # A previous run left the slice queued.  Waiting keeps its place in line;
    # killing and resubmitting would reset its queue position every retry.
    say "Slice '$SLICE_JOB' is already queued ($st) — attaching to it."
  else
    launch_slice
  fi
  wait_for_slice
}

# ------------------------------------------------------------------- down
# Tear down what `up` created: the port forward, the SLURM slice, its tmux.
if [ "$MODE" = down ]; then
  node=$(slice_node || true)
  if [ -n "$node" ]; then
    sucoder -T "$TARGET" tunnel forwards 2>/dev/null \
      | sed -nE "s#.*localhost:([0-9]+) .*${node}:.*#\1#p" \
      | while read -r p; do
          [ -n "$p" ] && sucoder -T "$TARGET" tunnel forward "$p" --cancel >/dev/null 2>&1
        done
  fi
  ssh "$LN_ALIAS" "scancel --me -n '$SLICE_JOB' 2>/dev/null; tmux kill-session -t '$SLICE_TMUX' 2>/dev/null" || true
  say "Released dedicated slice '$SLICE_JOB'${node:+ (was on $node)} — job cancelled, tmux killed, forward removed."
  exit 0
fi

# ------------------------------------------------------------------ node
# $NODE overrides entirely (use that node, skip the slice); otherwise reuse or
# allocate the dedicated slice.
if [ -z "${NODE:-}" ]; then
  NODE=$(ensure_slice)
fi
say "Compute node: $NODE"

rnode() {  # run a command on the compute node via the warm login-node hop
  ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -J "$LN_ALIAS" "$NODE" "$1"
}

# ------------------------------------------------------------------- app
LAST_ERR=$(mktemp 2>/dev/null || printf '/tmp/claude-science.%s.err' "$$")
trap 'rm -f "$LAST_ERR"' EXIT

app_url() {  # echo the service URL if any; keep this try's stderr in $LAST_ERR
  rnode "timeout 10 $APP_BIN url" 2>"$LAST_ERR" \
    | grep -Eo 'https?://[^[:space:]]+' | head -n 1
}

APP_NAME=${APP_BIN##*/}   # basename, for matching the serve process

serve_alive() {  # is a `... serve` process alive on the node?
  # [s]erve keeps pgrep -f from matching the ssh-spawned remote shell, whose
  # own cmdline contains this very pattern.
  rnode "pgrep -u \$USER -f '$APP_NAME [s]erve' >/dev/null 2>&1"
}

# The remote log is append-only, so a naive tail shows a *previous* run's
# errors too.  Print only the lines from this launch's marker onward.
log_since_marker() {
  rnode "awk '/^===== launch /{b=\"\"} {b=b \$0 ORS} END{printf \"%s\", b}' \$HOME/$REMOTE_LOG 2>/dev/null"
}

URL=$(app_url || true)
if [ -z "$URL" ]; then
  if [ "$MODE" = url ]; then
    [ -s "$LAST_ERR" ] && { say "last url/ssh error:"; sed 's/^/  /' "$LAST_ERR" >&2; }
    if serve_alive; then
      die "serve on $NODE is ALIVE but not answering \`url\` — still starting (cold NFS DB-open) or overloaded; retry in a minute"
    fi
    die "no serve process on $NODE — run \`$0 up\` to start it"
  fi
  if serve_alive; then
    # A serve process exists but has not answered `url` yet — almost always a
    # slow cold start (NFS DB-open, WAL recovery).  Starting a second one
    # would put two writers on the same SQLite DB, so wait on the one we have.
    say "A $APP_NAME serve process already exists on $NODE — waiting for it (NOT starting a second one) ..."
  else
    # Stamp the log so its tail is never confused with an earlier run.
    rnode "printf '\n===== launch %s =====\n' \"\$(date '+%F %T')\" >>\$HOME/$REMOTE_LOG" || true
    say "Starting claude-science serve on $NODE (log: ~/$REMOTE_LOG) ..."
    rnode "nohup $APP_BIN serve $SERVE_ARGS >>\$HOME/$REMOTE_LOG 2>&1 </dev/null &"
  fi
  say "Waiting up to ${WAIT_SECS}s for the URL (cold DB-open on NFS is slow; raise WAIT_SECS= if needed):"
  deadline=$((SECONDS + WAIT_SECS))
  while [ "$SECONDS" -lt "$deadline" ]; do
    printf '.' >&2                       # heartbeat: a slow node is not a freeze
    URL=$(app_url || true)
    [ -n "$URL" ] && break
    serve_alive || break                 # process died — stop waiting on a corpse
    sleep 3
  done
  printf '\n' >&2
fi
if [ -z "$URL" ]; then
  say "---- ~/$REMOTE_LOG on $NODE (this launch) ----"
  launchlog=$(log_since_marker 2>/dev/null || true)
  printf '%s\n' "$launchlog" >&2
  load=$(rnode "timeout 8 uptime" 2>/dev/null \
         | sed -En 's/.*load average: *([0-9.]+).*/\1/p' || true)
  if serve_alive; then
    # The process is alive; it just couldn't answer `url` in time — a slow
    # cold start (NFS DB-open, WAL recovery) or node load, not a failure.
    die "serve on $NODE is ALIVE but answered no \`url\` within ${WAIT_SECS}s${load:+ (node load ${load})} — re-run \`$0 up\` with a larger WAIT_SECS=; it will wait on this process, not double-start it."
  else
    [ -s "$LAST_ERR" ] && { say "last url/ssh error:"; sed 's/^/  /' "$LAST_ERR" >&2; }
    die "the serve process on $NODE is GONE — it died without answering \`url\`${load:+ (node load ${load})}.  A clean log usually means an OOM kill or a stale lock in the app's data dir; watch it fail in the foreground:
  ssh -J $LN_ALIAS $NODE '$APP_BIN serve $SERVE_ARGS'"
  fi
fi
say "Service URL on node: $URL"

# --------------------------------------------------------------- forward
PORT=$(printf '%s' "$URL" | sed -En 's#^https?://[^/:]+:([0-9]+).*#\1#p')
[ -n "$PORT" ] || die "could not parse a port from: $URL"

# The localhost URL we ultimately hand to the browser — computed early so we
# can probe an existing forward before deciding whether to add our own.
LOCAL_URL=$(printf '%s' "$URL" \
  | sed -E "s#^(https?://)[^/:]+:[0-9]+#\1localhost:$PORT#")

port_bound() {  # is anything already listening on localhost:$PORT here?
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | grep -qE ":$PORT([[:space:]]|$)"
  else
    (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null \
      && { exec 3>&- 3<&-; return 0; } || return 1
  fi
}

serves_app() {  # does localhost:$PORT reach the very server we fetched $URL from?
  command -v curl >/dev/null 2>&1 || return 0   # no curl: trust the listener
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "$LOCAL_URL" \
         2>/dev/null || echo 000)
  # 2xx means our nonce was accepted — proof the port reaches this same server.
  # (401/403 would be *a* claude-science but possibly a different node, so it
  # is not a safe reuse target; a refused connection gives 000.)
  case "$code" in 2??) return 0 ;; *) return 1 ;; esac
}

FORWARDS=$(sucoder -T "$TARGET" tunnel forwards)
if printf '%s' "$FORWARDS" | grep -q "localhost:$PORT → $NODE:$PORT"; then
  say "Forward localhost:$PORT → $NODE already in place (sucoder-managed)."
elif printf '%s' "$FORWARDS" | grep -q "localhost:$PORT →"; then
  # sucoder tracks a forward on this local port to a *different* destination
  # (old node or old port) — replace it.
  sucoder -T "$TARGET" tunnel forward "$PORT" --cancel
  sucoder -T "$TARGET" tunnel forward "$PORT" --node "$NODE"
elif port_bound; then
  # localhost:$PORT is occupied by something sucoder does NOT track — usually a
  # hand-rolled `ssh -L`.  Adding another forward would just fail with an opaque
  # mux error, so reuse the existing one if it reaches our server, else stop
  # with a clear message instead of clobbering the user's process.
  if serves_app; then
    say "localhost:$PORT already forwards to this claude-science (untracked) — reusing it."
  else
    die "localhost:$PORT is in use but does not reach $NODE's claude-science — free it (or set NODE/port) and retry."
  fi
else
  sucoder -T "$TARGET" tunnel forward "$PORT" --node "$NODE"
fi

# ------------------------------------------------------------------ open
printf '%s\n' "$LOCAL_URL"
if [ "$OPENER" = "-" ]; then
  exit 0
fi
if command -v "$OPENER" >/dev/null 2>&1; then
  "$OPENER" "$LOCAL_URL"
elif command -v xdg-open >/dev/null 2>&1; then
  say "$OPENER not found; falling back to xdg-open"
  xdg-open "$LOCAL_URL"
else
  say "no URL opener found — open the printed URL by hand"
fi
