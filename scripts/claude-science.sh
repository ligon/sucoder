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
APP_BIN=${APP_BIN:-~/bin/claude-science}     # expanded on the REMOTE side
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
# Resolution order: $NODE override > collaborate session files > squeue.
if [ -z "${NODE:-}" ]; then
  NODE=$(grep -hs '^compute_node:' ~/.sucoder/sessions/*--"$TARGET".yaml \
         | awk '{print $2}' | grep -v '^null$' | sort -u || true)
fi
if [ -z "${NODE:-}" ]; then
  say "No session records a compute node; asking squeue ..."
  NODE=$(ssh "$LN_ALIAS" 'squeue --me --noheader -o %N' | sort -u || true)
fi
case $(printf '%s' "$NODE" | grep -c .) in
  0) die "no compute node found — allocate one first, or set NODE=..." ;;
  1) ;;
  *) die "multiple compute nodes in play ($(printf '%s' "$NODE" | tr '\n' ' ')) — pick one with NODE=..." ;;
esac
say "Compute node: $NODE"

rnode() {  # run a command on the compute node via the warm login-node hop
  ssh -o StrictHostKeyChecking=accept-new -J "$LN_ALIAS" "$NODE" "$1"
}

# ------------------------------------------------------------------- app
app_url() {
  rnode "timeout 10 $APP_BIN url" 2>/dev/null \
    | grep -Eo 'https?://[^[:space:]]+' | head -n 1
}

URL=$(app_url || true)
if [ -z "$URL" ]; then
  if [ "$MODE" = url ]; then
    die "claude-science on $NODE is not answering \`url\` — run \`$0 up\` to start it"
  fi
  say "Starting claude-science serve on $NODE (log: ~/$REMOTE_LOG) ..."
  rnode "nohup $APP_BIN serve $SERVE_ARGS >>\$HOME/$REMOTE_LOG 2>&1 </dev/null &"
  for _ in $(seq "$WAIT_SECS"); do
    sleep 1
    URL=$(app_url || true)
    [ -n "$URL" ] && break
  done
fi
if [ -z "$URL" ]; then
  say "---- tail of ~/$REMOTE_LOG on $NODE ----"
  rnode "tail -n 20 \$HOME/$REMOTE_LOG" >&2 || true
  die "claude-science reported no URL within ${WAIT_SECS}s (log tail above)"
fi
say "Service URL on node: $URL"

# --------------------------------------------------------------- forward
PORT=$(printf '%s' "$URL" | sed -En 's#^https?://[^/:]+:([0-9]+).*#\1#p')
[ -n "$PORT" ] || die "could not parse a port from: $URL"

FORWARDS=$(sucoder -T "$TARGET" tunnel forwards)
if printf '%s' "$FORWARDS" | grep -q "localhost:$PORT → $NODE:$PORT"; then
  say "Forward localhost:$PORT already in place."
else
  if printf '%s' "$FORWARDS" | grep -q "localhost:$PORT →"; then
    # Same local port, stale destination (old node or old port) — replace.
    sucoder -T "$TARGET" tunnel forward "$PORT" --cancel
  fi
  sucoder -T "$TARGET" tunnel forward "$PORT" --node "$NODE"
fi

# ------------------------------------------------------------------ open
LOCAL_URL=$(printf '%s' "$URL" \
  | sed -E "s#^(https?://)[^/:]+:[0-9]+#\1localhost:$PORT#")
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
