"""Shared fixtures.

Keeps process-global caches from leaking between tests: they are keyed by
socket path, and a test that reuses a path would otherwise inherit another
test's verdict.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_tunnel_caches():
    from sucoder import tunnel

    tunnel._VERIFIED.clear()
    yield
    tunnel._VERIFIED.clear()


@pytest.fixture(autouse=True)
def _private_sent_log(tmp_path, monkeypatch):
    """``message`` appends every delivery to ``~/.sucoder/sent.jsonl``; no
    test may write the real one.  A test of the default path restores
    ``replies.default_sent_log_path`` under its own temporary HOME."""
    from sucoder import replies

    monkeypatch.setattr(replies, "sent_log_path", lambda: tmp_path / "sent-log" / "sent.jsonl")
