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
# The verify step tests the freshly-written cert DIRECTLY (ssh -i "$CERT"),
# so it needs no ~/.ssh/config stanza.  sucoder presents the same cert to
# `tunnel up` via a target's `cert_file:` option, so a green test here means
# `sucoder -T <target> tunnel up` (with cert_file set) is OTP-free too.
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
echo ">> passwordless login test (cert presented directly -- no ssh_config needed):"
# Present the cert we just wrote exactly the way sucoder's establish() does:
# IdentityFile + CertificateFile + IdentitiesOnly, so this verifies the cert
# itself rather than whatever the default ssh_config happens to offer.
# BatchMode=yes: a rejected cert must fail loudly, never fall back to a prompt.
if ssh -i "$CERT" -o "CertificateFile=$CERT-cert.pub" -o IdentitiesOnly=yes \
       -o BatchMode=yes -o ConnectTimeout=25 -l "$BRC_USER" "$GATEWAY" \
       'echo "   OK -- host=$(hostname) user=$(whoami)"'; then
  echo ">> SUCCESS: ${GATEWAY} accepted the cert.  Point sucoder at it with"
  echo "   cert_file: ${CERT}   (under the target) so \`tunnel up\` is OTP-free until 'Valid to'."
else
  echo "!! cert issued but the gateway REJECTED it (tested: ssh -i \"$CERT\")." >&2
  echo "   Check the cert's principals / that your login maps to it; re-mint if stale." >&2
  exit 1
fi
