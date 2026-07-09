"""Mint short-lived BRC/Savio SSH certificates via the MSM CA.

sucoder only ever *presented* a gateway SSH cert (see ``cert_file`` in
:mod:`sucoder.config` and ``SshControl.establish``); *acquiring* one meant
running ``scripts/brc-cert.sh`` by hand, which shells out to a cloned
``lrc-scripts/request_cert.sh``.  That script is, underneath, a single HTTPS
POST to the LBNL MSM CA.  This module reimplements just that POST in pure
stdlib (no external clone, no third-party deps) so ``sucoder cert`` and the
connect-time auto-mint hook can acquire a cert directly.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict

# The ``brc`` preset from lrc-scripts/request_cert.sh: HOST=https://msm.brc.lbl.gov,
# endpoint /v1/cert.  The MSM CA hard-caps the lifetime at 12h (it returns
# ``Requested certificate lifetime larger than maximum lifetime (12h)`` above
# that -- see scripts/brc-cert.sh:7-9).
DEFAULT_CA_URL = "https://msm.brc.lbl.gov/v1/cert"
DEFAULT_LIFETIME = "12h"

# Fields the CA returns in its success JSON body (see request_cert.sh's q()).
_REQUIRED_FIELDS = ("private_key", "public_key", "signed_public_key")


class CertError(Exception):
    """A cert could not be minted (CA rejected the request or was unreachable)."""


def request_cert(
    ca_url: str,
    username: str,
    pin: str,
    otp: str,
    lifetime: str,
    *,
    timeout: int = 30,
) -> Dict:
    """POST credentials to the MSM CA and return the parsed JSON cert payload.

    Mirrors ``request_cert.sh``: the JSON body is
    ``{username, password, mfa, lifetime}`` where ``password`` is the BRC PIN
    and ``mfa`` is the one-time code.  Raises :class:`CertError` on any
    non-success, carrying the CA's own message so a stale OTP / wrong PIN is
    legible rather than a bare status code.
    """
    body = json.dumps(
        {"username": username, "password": pin, "mfa": otp, "lifetime": lifetime}
    ).encode("utf-8")
    req = urllib.request.Request(
        ca_url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            payload = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:  # non-2xx: the CA rejected us
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace").strip()
        except Exception:  # noqa: BLE001 -- best-effort error body
            pass
        raise CertError(
            f"CA rejected the request (HTTP {exc.code})"
            f"{': ' + detail if detail else ''} "
            "-- check the PIN and that the OTP is fresh."
        ) from exc
    except urllib.error.URLError as exc:  # network / TLS / DNS
        raise CertError(f"could not reach CA {ca_url}: {exc.reason}") from exc

    if status not in (200, 201):
        raise CertError(f"CA returned HTTP {status}: {payload[:200]}")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CertError(f"CA response was not JSON: {payload[:200]}") from exc
    if not isinstance(data, dict):
        raise CertError(f"CA response was not a JSON object: {payload[:200]}")
    missing = [f for f in _REQUIRED_FIELDS if not data.get(f)]
    if missing:
        raise CertError(f"CA response missing field(s): {', '.join(missing)}")
    return data


def write_cert(cert_file, data: Dict) -> None:
    """Write the private key, public key, and signed cert around *cert_file*.

    *cert_file* is the private-key path (the target's ``cert_file`` config
    value).  Its ``.pub`` and ``-cert.pub`` siblings are what
    ``SshControl.establish`` presents on the gateway hop.  The private key is
    created ``0o600`` from the start so it is never briefly world-readable.
    """
    cert_file = Path(cert_file)
    cert_file.parent.mkdir(parents=True, exist_ok=True)
    _write(cert_file, data["private_key"], 0o600)
    _write(cert_file.with_name(cert_file.name + ".pub"), data["public_key"], 0o644)
    _write(
        cert_file.with_name(cert_file.name + "-cert.pub"),
        data["signed_public_key"],
        0o644,
    )


def _write(path: Path, content: str, mode: int) -> None:
    if not content.endswith("\n"):
        content += "\n"
    # O_CREAT with the target mode so a secret key is never momentarily 0644;
    # chmod afterwards pins the mode even when the file already existed (the
    # open-mode is ignored for a pre-existing file, and umask can clear bits).
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.write(fd, content.encode("utf-8"))
    finally:
        os.close(fd)
    os.chmod(path, mode)


def mint(
    cert_file,
    ca_url: str,
    username: str,
    pin: str,
    otp: str,
    lifetime: str,
) -> Dict:
    """Request a cert from the CA and write it to disk; return the CA payload."""
    data = request_cert(ca_url, username, pin, otp, lifetime)
    write_cert(cert_file, data)
    return data
