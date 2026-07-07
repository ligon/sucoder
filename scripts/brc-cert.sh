#!/usr/bin/env bash
#
# brc-cert.sh -- acquire a short-lived BRC/Savio SSH certificate non-interactively
# and verify it, in ONE fast command, so a fresh OTP is consumed before it expires
# (no interactive round-trip / agent latency).
#
# The MSM CA hard-caps the lifetime at 12h -- confirmed 2026-06-29, the server
# returns: {"error":"Requested certificate lifetime larger than maximum lifetime (12h)."}
# So 12h is the ceiling: one OTP buys at most a 12h passwordless window.
#
# Usage:
#   scripts/brc-cert.sh <PIN> <OTP>
#   BRC_PIN=1234 scripts/brc-cert.sh <OTP>
#
# Env overrides: BRC_USER (default ligon), BRC_LIFETIME (default 12h),
#                LRC_SCRIPTS (default ~/lrc-scripts),
#                VERIFY_TIMEOUT (login-test hard ceiling, default 40s).
#
# The verify step tests the freshly-written cert DIRECTLY (ssh -i "$CERT"),
# so it needs no ~/.ssh/config stanza.  sucoder presents the same cert to
# `tunnel up` via a target's `cert_file:` option, so a green test here means
# `sucoder -T <target> tunnel up` (with cert_file set) is OTP-free too.
#
# Exit codes: 0 cert issued AND verified; 1 cert issued but the gateway
# GENUINELY rejected it (publickey denied -- re-mint); 2 usage error; 3 cert
# issued but verification was INCONCLUSIVE (login-test timeout / host-key
# mismatch / pre-auth connection failure -- NOT a bad cert: it is written and
# usable).  A non-zero ssh exit on its own is never read as a cert rejection --
# that false verdict (seen 2026-07-07: a flaky login node made a perfectly good
# cert look rejected) is exactly what the classification below removes.
#
set -euo pipefail

BRC_USER="${BRC_USER:-ligon}"
BRC_LIFETIME="${BRC_LIFETIME:-12h}"
LRC_SCRIPTS="${LRC_SCRIPTS:-$HOME/lrc-scripts}"
GATEWAY="hpc.brc.berkeley.edu"
CERT="$HOME/.ssh/ssh_certs/brc_cert"

if   [ "$#" -eq 2 ]; then PIN="$1"; OTP="$2"
elif [ "$#" -eq 1 ]; then OTP="$1"; PIN="${BRC_PIN:?provide PIN as first arg or set BRC_PIN}"
else echo "usage: $0 <PIN> <OTP>   (or: BRC_PIN=.. $0 <OTP>)" >&2; exit 2
fi

if [ ! -x "$LRC_SCRIPTS/request_cert.sh" ]; then
  echo ">> cloning lrc-scripts into $LRC_SCRIPTS"
  git clone --depth 1 https://github.com/lbnl-science-it/lrc-scripts.git "$LRC_SCRIPTS"
fi

echo ">> requesting ${BRC_LIFETIME} cert for ${BRC_USER} @ ${GATEWAY} ..."
out="$(printf '%s\n%s\n%s\n' "$BRC_USER" "$PIN" "$OTP" \
        | "$LRC_SCRIPTS/request_cert.sh" -p brc -l "$BRC_LIFETIME" 2>&1)" || true
echo "$out"

if ! printf '%s' "$out" | grep -q "wrote key"; then
  echo "!! cert NOT issued -- OTP likely stale/wrong (regenerate and re-run fast)." >&2
  exit 1
fi

echo
echo ">> granted validity (CA caps at 12h):"
# Principals live on indented continuation lines *under* the "Principals:"
# header; the old `grep Principals` matched the header and silently dropped the
# names beneath it -- which hid the one field a real rejection turns on (seen
# 2026-07-07).  Collect Key ID + Valid, then the principals onto one line.
cert_info=$(ssh-keygen -L -f "$CERT-cert.pub")
printf '%s\n' "$cert_info" | grep -iE 'Key ID|Valid:' | sed 's/^ */   /'
princ=$(printf '%s\n' "$cert_info" \
  | awk '/Principals:/{f=1;next} /Critical Options:|Extensions:|Signing CA:/{f=0} f{gsub(/^[[:space:]]+/,"");print}' \
  | paste -sd, -)
printf '   Principals: %s\n' "${princ:-<none -- cert is valid for ANY principal>}"

echo ">> passwordless login test (cert presented directly -- no ssh_config needed):"
# Present the cert exactly the way sucoder's establish() does (IdentityFile +
# CertificateFile + IdentitiesOnly), but strip everything a *verification* has
# no business doing: no agent/X11 forwarding, no mux socket -- and wrap it all
# in a hard `timeout` so a slow/flaky login node can neither hang the probe nor
# masquerade as a cert problem.  A unique marker echoed back over the session
# is our proof the cert authenticated AND a shell ran end to end.  `-v` keeps
# the auth trace so we can classify a *failure* by its actual cause.
MARKER="__brc_cert_ok_$$__"
rc=0
probe=$(timeout "${VERIFY_TIMEOUT:-40}" \
  ssh -v -i "$CERT" -o "CertificateFile=$CERT-cert.pub" -o IdentitiesOnly=yes \
      -o BatchMode=yes -o ConnectTimeout=25 \
      -o ControlPath=none -o ControlMaster=no -o ClearAllForwardings=yes \
      -o ForwardAgent=no -o ForwardX11=no \
      -l "$BRC_USER" "$GATEWAY" "echo $MARKER host=\$(hostname)" 2>&1) || rc=$?

if printf '%s' "$probe" | grep -q "$MARKER" \
   || printf '%s' "$probe" | grep -qE 'Authenticated to .* using "publickey"'; then
  host=$(printf '%s' "$probe" | sed -nE "s/.*$MARKER host=([^[:space:]]+).*/\1/p" | head -1)
  echo "   OK -- cert authenticated${host:+ (login node ${host})}."
  echo ">> SUCCESS: ${GATEWAY} accepted the cert.  Point sucoder at it with"
  echo "   cert_file: ${CERT}   (under the target) so \`tunnel up\` is OTP-free until 'Valid to'."
  exit 0
fi

# Not authenticated.  A non-zero ssh exit is NOT proof the cert is bad -- so
# classify WHY, and only call it a rejection when the server actually denied
# the publickey.  Everything else leaves the (already written) cert usable.
if printf '%s' "$probe" | grep -qiE 'Permission denied \(publickey'; then
  echo "!! cert issued but the gateway GENUINELY REJECTED it (publickey denied)." >&2
  echo "   Check the principal(s) above map to your login on ${GATEWAY}; re-mint if stale." >&2
  exit 1
elif [ "$rc" -eq 124 ]; then
  echo "!! could NOT verify: the login test timed out after ${VERIFY_TIMEOUT:-40}s." >&2
  echo "   That is a SLOW/FLAKY gateway, not a bad cert -- the cert is written and" >&2
  echo "   usable.  Retry, or just run \`sucoder -T <target> tunnel up\`." >&2
  exit 3
elif printf '%s' "$probe" | grep -qiE 'REMOTE HOST IDENTIFICATION HAS CHANGED|Host key verification failed|No .* host key is known'; then
  echo "!! could NOT verify: host-key check failed for ${GATEWAY} (NOT a cert problem)." >&2
  echo "   ${GATEWAY} is round-robin across login nodes with distinct host keys; this is" >&2
  echo "   a known_hosts issue.  The cert is written and usable.  Fix known_hosts (or set" >&2
  echo "   a HostKeyAlias for the pool) and retry." >&2
  exit 3
else
  echo "!! could NOT verify: the connection to ${GATEWAY} failed before auth completed." >&2
  echo "   No evidence the cert is bad -- it is written and usable.  Trace tail:" >&2
  printf '%s\n' "$probe" | tail -4 | sed 's/^/     /' >&2
  exit 3
fi
