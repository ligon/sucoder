"""Unit tests for the pure-stdlib MSM CA client in :mod:`sucoder.cert`."""
import io
import json
import stat
import urllib.error

import pytest

from sucoder import cert


class _FakeResp:
    """Minimal stand-in for the object urlopen returns as a context manager."""

    def __init__(self, code, body):
        self._code = code
        self._body = body.encode() if isinstance(body, str) else body

    def getcode(self):
        return self._code

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_request_cert_success(monkeypatch):
    payload = {
        "key_id": "abc123",
        "private_key": "PRIV",
        "public_key": "PUB",
        "signed_public_key": "SIGNED",
        "expires_at": "2026-07-10T06:39:37",
    }
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = req.data
        captured["ctype"] = req.headers.get("Content-type")
        return _FakeResp(201, json.dumps(payload))

    monkeypatch.setattr(cert.urllib.request, "urlopen", fake_urlopen)
    out = cert.request_cert("https://ca.test/v1/cert", "ligon", "1234", "567890", "12h")

    assert out == payload
    assert captured["url"] == "https://ca.test/v1/cert"
    assert captured["method"] == "POST"
    assert captured["ctype"] == "application/json"
    assert json.loads(captured["body"]) == {
        "username": "ligon", "password": "1234", "mfa": "567890", "lifetime": "12h",
    }


def test_request_cert_http_error_surfaces_ca_message(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 401, "Unauthorized", {},
            io.BytesIO(b'{"error":"invalid otp"}'),
        )

    monkeypatch.setattr(cert.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(cert.CertError) as ei:
        cert.request_cert("https://ca.test/v1/cert", "ligon", "1234", "000000", "12h")
    assert "401" in str(ei.value)
    assert "invalid otp" in str(ei.value)


def test_request_cert_url_error(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("name resolution failed")

    monkeypatch.setattr(cert.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(cert.CertError) as ei:
        cert.request_cert("https://ca.test/v1/cert", "u", "p", "o", "12h")
    assert "could not reach CA" in str(ei.value)


def test_request_cert_missing_fields(monkeypatch):
    monkeypatch.setattr(
        cert.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeResp(201, json.dumps({"key_id": "x"})),
    )
    with pytest.raises(cert.CertError) as ei:
        cert.request_cert("https://ca.test/v1/cert", "u", "p", "o", "12h")
    assert "missing field" in str(ei.value)


def test_write_cert_files_and_modes(tmp_path):
    cert_file = tmp_path / "certs" / "brc_cert"
    cert.write_cert(
        cert_file,
        {"private_key": "PRIV", "public_key": "PUB", "signed_public_key": "SIGNED"},
    )
    assert cert_file.read_text() == "PRIV\n"
    assert (cert_file.parent / "brc_cert.pub").read_text() == "PUB\n"
    assert (cert_file.parent / "brc_cert-cert.pub").read_text() == "SIGNED\n"
    # Private key must not be group/other readable.
    assert stat.S_IMODE(cert_file.stat().st_mode) == 0o600
    assert stat.S_IMODE((cert_file.parent / "brc_cert.pub").stat().st_mode) == 0o644


def test_write_cert_tightens_mode_on_preexisting_file(tmp_path):
    cert_file = tmp_path / "brc_cert"
    cert_file.write_text("stale")
    cert_file.chmod(0o644)  # was world-readable
    cert.write_cert(
        cert_file,
        {"private_key": "PRIV", "public_key": "PUB", "signed_public_key": "SIGNED"},
    )
    assert stat.S_IMODE(cert_file.stat().st_mode) == 0o600


def test_mint_writes_and_returns(monkeypatch, tmp_path):
    payload = {
        "key_id": "k", "private_key": "PRIV", "public_key": "PUB",
        "signed_public_key": "SIGNED", "expires_at": "2026",
    }
    monkeypatch.setattr(
        cert.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeResp(201, json.dumps(payload)),
    )
    cert_file = tmp_path / "brc_cert"
    out = cert.mint(cert_file, "https://ca.test/v1/cert", "u", "p", "o", "12h")
    assert out == payload
    assert (tmp_path / "brc_cert-cert.pub").read_text() == "SIGNED\n"
