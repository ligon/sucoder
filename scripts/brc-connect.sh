#!/usr/bin/env bash
#
# brc-connect.sh -- interactive front-end to brc-cert.sh.
#
# Prompts for your BRC PIN (hidden) and a FRESH OTP, then immediately
# consumes them in one shot to acquire a 12h passwordless SSH certificate
# (the MSM CA hard-caps lifetime at 12h).  Run this in YOUR terminal: ssh
# reads the OTP locally, so there is no chat round-trip to blow the TOTP
# window, and no secret ever lands in the agent transcript.
#
# Once it prints SUCCESS, the cert + warm ControlMaster let any later ssh
# (including the agent's non-interactive probes) reach Savio with no OTP
# until the cert's "Valid to" time.
#
# Env overrides forwarded to brc-cert.sh: BRC_USER (default ligon),
# BRC_LIFETIME (default 12h), LRC_SCRIPTS (default ~/lrc-scripts).
#
# Usage:
#   scripts/brc-connect.sh
#
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cert_script="$here/brc-cert.sh"

if [ ! -x "$cert_script" ]; then
  echo "!! $cert_script not found or not executable." >&2
  exit 1
fi

echo "Acquiring a BRC/Savio SSH cert (user: ${BRC_USER:-ligon}, lifetime: ${BRC_LIFETIME:-12h})."
echo "Generate a FRESH OTP right before you type it -- the token is consumed immediately."
echo

# Hidden PIN entry (static secret; never echoed).
read -rsp "BRC PIN: " BRC_PIN_INPUT; echo
# OTP shown as you type so a typo is obvious; it is single-use and expires fast.
read -rp  "BRC OTP (fresh): " BRC_OTP_INPUT

if [ -z "${BRC_PIN_INPUT}" ] || [ -z "${BRC_OTP_INPUT}" ]; then
  echo "!! PIN and OTP are both required." >&2
  exit 2
fi

echo
exec "$cert_script" "$BRC_PIN_INPUT" "$BRC_OTP_INPUT"
