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
#                LRC_SCRIPTS (default ~/lrc-scripts).
#
# Requires the ~/.ssh/config "brc-login" stanza (cert IdentityFile + ControlMaster)
# for the passwordless test to use the cert.
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
ssh-keygen -L -f "$CERT-cert.pub" | grep -iE "Valid|Principals|Key ID" | sed 's/^/   /'
echo ">> passwordless login test:"
if ssh -o BatchMode=yes -o ConnectTimeout=25 "$GATEWAY" 'echo "   OK -- host=$(hostname) user=$(whoami)"'; then
  echo ">> SUCCESS: cert in place + ControlMaster warm; sucoder reaches Savio with no OTP until 'Valid to'."
else
  echo "!! cert issued but passwordless login failed -- check the ~/.ssh/config brc-login stanza." >&2
  exit 1
fi
