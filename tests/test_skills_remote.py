"""Remote-mode guard for post-session agent-skills auto-commit.

`_agent_skills_dir` resolves the *agent user's local* home (`~coder`),
so running the skills git commands through the remote executor (on the
compute node, as a different user) fails with a path that doesn't
exist.  In remote mode the auto-commit must be skipped, not attempted.
"""
from tests.test_remote import _build_remote_manager


def test_auto_commit_skills_skipped_on_remote(tmp_path):
    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")
    assert ctx.is_remote

    calls = []

    def _spy(*args, **kwargs):  # would run git over SSH if reached
        calls.append((args, kwargs))
        raise AssertionError("executor.run_agent must not be called on remote")

    manager.executor.run_agent = _spy
    manager._auto_commit_agent_skills(ctx)  # must early-return, touch nothing

    assert calls == []
