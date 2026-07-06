#!/usr/bin/env bash
#
# claude-science.sh -- serve claude-science on a compute node and open it
# in the local browser, over SuCoder's warm tunnels.
#
# Usage:
#   scripts/claude-science.sh [up]   warm the tunnels, start `claude-science
#                                    serve` on the compute node if it is not
#                                    already running, forward its port, and
#                                    open the URL locally.
#   scripts/claude-science.sh url    fetch a (possibly new) URL from the
#                                    RUNNING claude-science, reuse/repair the
#                                    port forward, and open it.  Fails if the
#                                    service is down (run `up` instead).
#
# Both modes are idempotent: re-running them reuses the warm sockets, the
# running server, and the existing forward.
#
# The URL is opened with garcon-url-handler (ChromeOS / Crostini); set
# OPENER to something else (xdg-open, firefox, ...) or OPENER=- to only
# print the localhost URL on stdout.
#
# Env overrides:
#   TARGET      sucoder target            (default: carleton-htc)
#   NODE        compute node              (default: session file, then squeue)
#   APP_BIN     app path on the node      (default: ~/bin/claude-science)
#   SERVE_ARGS  args for `serve`          (default: --dangerously-no-sandbox)
#   OPENER      local URL opener, or "-"  (default: garcon-url-handler)
#   WAIT_SECS   serve readiness timeout   (default: 30)
#
set -euo pipefail

MODE=${1:-up}
case "$MODE" in
  up|url) : ;;
  *) echo "usage: $0 [up|url]" >&2; exit 2 ;;
esac

TARGET=${TARGET:-carleton-htc}
APP_BIN=${APP_BIN:-'$HOME/bin/claude-science'}  # literal $HOME; expands REMOTE-side
SERVE_ARGS=${SERVE_ARGS:---dangerously-no-sandbox}
OPENER=${OPENER:-garcon-url-handler}
WAIT_SECS=${WAIT_SECS:-30}
LN_ALIAS="${TARGET}-ln"
REMOTE_LOG=".claude-science.serve.log"        # in the remote $HOME

say() { printf '%s\n' "$*" >&2; }
die() { say "ERROR: $*"; exit 1; }

# ---------------------------------------------------------------- tunnels
# Idempotent: reuses warm sockets; prompts (OTP) only when everything is
# cold.  Also (re)writes the ${TARGET}-ln ssh alias we ride below.
sucoder -T "$TARGET" tunnel up

# ------------------------------------------------------------------ node
# Resolution order: $NODE override > live `squeue` > recorded sessions.
# squeue is authoritative for what is *actually* allocated right now;
# collaborate session files can still name nodes from jobs that have since
# ended, and a stale record used to trip the ambiguity guard below — so we
# consult the session files only when squeue can't answer.
if [ -z "${NODE:-}" ]; then
  NODE=$(ssh "$LN_ALIAS" 'squeue --me --noheader -o %N' 2>/dev/null \
         | grep -vE '^[[:space:]]*$' | sort -u || true)
fi
if [ -z "${NODE:-}" ]; then
  say "squeue named no nodes; falling back to recorded sessions ..."
  NODE=$(grep -hs '^compute_node:' ~/.sucoder/sessions/*--"$TARGET".yaml \
         | awk '{print $2}' | grep -v '^null$' | sort -u || true)
fi
case $(printf '%s' "$NODE" | grep -c .) in
  0) die "no compute node found — allocate one first, or set NODE=..." ;;
  1) ;;
  *) die "multiple compute nodes allocated ($(printf '%s' "$NODE" | tr '\n' ' ')) — pick one with NODE=..." ;;
esac
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

# The remote log is append-only, so a naive tail shows a *previous* run's
# errors too.  Print only the lines from this launch's marker onward.
log_since_marker() {
  rnode "awk '/^===== launch /{b=\"\"} {b=b \$0 ORS} END{printf \"%s\", b}' \$HOME/$REMOTE_LOG 2>/dev/null"
}

URL=$(app_url || true)
if [ -z "$URL" ]; then
  if [ "$MODE" = url ]; then
    [ -s "$LAST_ERR" ] && { say "last url/ssh error:"; sed 's/^/  /' "$LAST_ERR" >&2; }
    die "claude-science on $NODE is not answering \`url\` — run \`$0 up\` to start it (or the node may be overloaded)"
  fi
  # Stamp the log so its tail is never confused with an earlier run.
  rnode "printf '\n===== launch %s =====\n' \"\$(date '+%F %T')\" >>\$HOME/$REMOTE_LOG" || true
  say "Starting claude-science serve on $NODE (log: ~/$REMOTE_LOG) ..."
  rnode "nohup $APP_BIN serve $SERVE_ARGS >>\$HOME/$REMOTE_LOG 2>&1 </dev/null &"
  say "Waiting for the URL (up to ~${WAIT_SECS} tries; raise WAIT_SECS= if the node is busy):"
  for _ in $(seq "$WAIT_SECS"); do
    sleep 1
    printf '.' >&2                       # heartbeat: a slow node is not a freeze
    URL=$(app_url || true)
    [ -n "$URL" ] && break
  done
  printf '\n' >&2
fi
if [ -z "$URL" ]; then
  say "---- ~/$REMOTE_LOG on $NODE (this launch) ----"
  launchlog=$(log_since_marker 2>/dev/null || true)
  printf '%s\n' "$launchlog" >&2
  load=$(rnode "timeout 8 uptime" 2>/dev/null \
         | sed -En 's/.*load average: *([0-9.]+).*/\1/p' || true)
  if printf '%s' "$launchlog" | grep -qiE 'daemon already running|listening on|https?://'; then
    # The daemon exists; it just couldn't answer `url` in time — almost always
    # node load, not a startup failure.
    die "daemon on $NODE is up but returned no URL within ~${WAIT_SECS} tries${load:+ (node load ${load})} — the node is likely overloaded; let load drop and retry, or raise WAIT_SECS=."
  else
    [ -s "$LAST_ERR" ] && { say "last url/ssh error:"; sed 's/^/  /' "$LAST_ERR" >&2; }
    die "claude-science did not start on $NODE within ~${WAIT_SECS} tries${load:+ (node load ${load})} (log above)."
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
