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
