"""Shared fixtures.

Keeps process-global caches from leaking between tests: they are keyed by
socket path, and a test that reuses a path would otherwise inherit another
test's verdict.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config):
    """Run the suite under umask 022, as CI does.

    Fixtures create directories and files with default modes (``Path.mkdir()``,
    ``write_text``), and the code under test rightly refuses group-writable
    state: the runner refuses a durable root that is ``0770``, ``ssh -G``
    refuses a group-writable config, and the timer tests assert the ``0755``
    they asked for.  An account whose umask is 0007 (the shared-mirror
    convention, ``CommandExecutor.default_umask``) failed 48 such tests here
    (2026-11-12) without any code defect.  Tests about umask set it themselves.
    """
    os.umask(0o022)


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


@pytest.fixture
def foreign_sucoder(tmp_path, monkeypatch):
    """Put another, unimportable ``sucoder`` first on child Pythons' path.

    It stands in for the copy a child finds on its own -- an older installed
    one, or a checkout in its working directory -- even where CI installs the
    package under test.  A child that imports it fails, so a test passing
    with this fixture shows the child imported this process's package
    (sucoder/child_python.py).
    """
    directory = tmp_path / "foreign-sucoder"
    (directory / "sucoder").mkdir(parents=True)
    (directory / "sucoder" / "__init__.py").write_text('raise ImportError("foreign sucoder imported")\n')
    inherited = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", str(directory) + (os.pathsep + inherited if inherited else ""))
    return directory
