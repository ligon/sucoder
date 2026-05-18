"""Tests for portable path rendering in the injected prelude.

Covers the fix that stops the prelude from baking in a specific
``$HOME``/username (e.g. ``/home/ligon/Projects/sucoder-skills/...``)
and instead emits paths valid on the host that consumes the prelude:

* ``_portable_skill_path`` -- skill-tree paths expressed *through* the
  stable ``<home>/.sucoder/skills`` symlink (target varies per machine),
  re-rooted on the remote ``$HOME`` for remote sessions; absolute so
  Claude's Read tool keeps working.
* ``_collapse_home`` -- informational headers collapse to ``$HOME/...``.
* ``_file_read_hint`` / ``_render_scripts_section`` route through the
  portable helper so the catalog *and* the auto-generated Scripts
  section stay consistent.
"""

import logging
import types
from pathlib import Path

import pytest

import sucoder.mirror as mirror
from sucoder.config import AgentType
from sucoder.executor import CommandResult
from sucoder.mirror import MirrorManager

from .test_mirror import build_manager


def _result(stdout: str, returncode: int = 0) -> CommandResult:
    return CommandResult(
        requested_args=["bash", "-lc", "probe"],
        executed_args=["bash", "-lc", "probe"],
        stdout=stdout,
        stderr="",
        returncode=returncode,
    )


def _make_skills_tree(home: Path) -> Path:
    """Create ``home/.sucoder/skills`` as a symlink to a varying target."""
    target = home / "Projects" / "sucoder-skills"
    (target / "orgmode" / "exporting" / "scripts").mkdir(parents=True)
    (target / "orgmode" / "exporting" / "SKILL.md").write_text("x\n")
    (target / "orgmode" / "exporting" / "scripts" / "export-pdf.sh").write_text("#!\n")
    link = home / ".sucoder" / "skills"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    return link


@pytest.fixture()
def manager_with_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    _make_skills_tree(home)
    mgr = build_manager(tmp_path)
    return mgr, home


def test_local_skill_path_uses_symlink_not_resolved_target(manager_with_home):
    mgr, home = manager_with_home
    mgr._render_is_remote = False
    skill = home / ".sucoder" / "skills" / "orgmode" / "exporting" / "SKILL.md"

    out = mgr._portable_skill_path(skill)

    # Expressed THROUGH the stable .sucoder/skills symlink, never the
    # per-machine-varying resolved Projects/ target.
    assert out == f"{home}/.sucoder/skills/orgmode/exporting/SKILL.md"
    assert "Projects/sucoder-skills" not in out
    assert Path(out).is_absolute()


def test_remote_skill_path_rerooted_on_remote_home(manager_with_home):
    mgr, _home = manager_with_home
    mgr._render_is_remote = True
    mgr._resolved_remote_home = "/remote/home/bob"
    skill = (
        Path.home() / ".sucoder" / "skills"
        / "orgmode" / "exporting" / "scripts" / "export-pdf.sh"
    )

    out = mgr._portable_skill_path(skill)

    assert out == "/remote/home/bob/.sucoder/skills/orgmode/exporting/scripts/export-pdf.sh"
    assert Path(out).is_absolute()  # Claude Read tool needs absolute


def test_remote_home_unknown_falls_back_to_original(manager_with_home):
    mgr, home = manager_with_home
    mgr._render_is_remote = True
    mgr._resolved_remote_home = None
    skill = home / ".sucoder" / "skills" / "orgmode" / "exporting" / "SKILL.md"

    # No regression: when remote home can't be resolved we keep today's
    # absolute path rather than emitting a wrong one.
    assert mgr._portable_skill_path(skill) == str(skill)


def test_path_outside_home_unchanged(manager_with_home):
    mgr, _home = manager_with_home
    mgr._render_is_remote = False
    outside = Path("/opt/elsewhere/thing.md")
    assert mgr._portable_skill_path(outside) == str(outside)


def test_collapse_home_for_headers(manager_with_home):
    mgr, home = manager_with_home
    inside = home / ".sucoder" / "system_prompt.org"
    assert mgr._collapse_home(inside) == "$HOME/.sucoder/system_prompt.org"
    assert mgr._collapse_home(Path("/etc/hosts")) == "/etc/hosts"


def test_file_read_hint_claude_absolute_and_portable(manager_with_home):
    mgr, home = manager_with_home
    mgr._render_is_remote = True
    mgr._resolved_remote_home = "/remote/home/bob"
    mgr._detected_agent_type = AgentType.CLAUDE
    skill = Path.home() / ".sucoder" / "skills" / "orgmode" / "exporting" / "SKILL.md"

    hint = mgr._file_read_hint(skill)

    assert hint == "Read tool: /remote/home/bob/.sucoder/skills/orgmode/exporting/SKILL.md"
    assert "~" not in hint  # Claude does not expand ~
    assert "Projects/sucoder-skills" not in hint


def test_classify_skills_base_pure():
    assert MirrorManager._classify_skills_base(False, False) == "MISSING"
    assert MirrorManager._classify_skills_base(True, False) == "PERMS"
    assert MirrorManager._classify_skills_base(True, True) == "OK"


def test_local_skills_base_missing_warns(manager_with_home, caplog, monkeypatch):
    mgr, _home = manager_with_home
    ctx = mgr.context_for("sample")  # local mirror
    missing = Path("/no/such/.sucoder/skills")
    monkeypatch.setattr(type(mgr), "_default_skills_dir", staticmethod(lambda: missing))
    with caplog.at_level(logging.WARNING, logger="sucoder.test"):
        mgr._warn_if_skills_base_unusable(ctx)
    assert any(
        "missing or a broken symlink" in r.message and str(missing) in r.message
        for r in caplog.records
    )


def test_local_skills_base_bad_perms_warns(manager_with_home, caplog, monkeypatch):
    mgr, home = manager_with_home
    ctx = mgr.context_for("sample")
    # Exists but not readable/traversable.
    monkeypatch.setattr(mirror.os, "access", lambda p, m: False)
    with caplog.at_level(logging.WARNING, logger="sucoder.test"):
        mgr._warn_if_skills_base_unusable(ctx)
    assert any("not readable" in r.message for r in caplog.records)


def test_local_skills_base_ok_is_silent(manager_with_home, caplog):
    mgr, _home = manager_with_home  # fixture created a valid skills symlink
    ctx = mgr.context_for("sample")
    with caplog.at_level(logging.WARNING, logger="sucoder.test"):
        mgr._warn_if_skills_base_unusable(ctx)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_local_check_makes_no_subprocess(manager_with_home):
    mgr, _home = manager_with_home
    ctx = mgr.context_for("sample")

    def boom(*a, **k):  # would fire if the local path shelled out
        raise AssertionError("local skills check must not call run_agent")

    mgr.executor.run_agent = boom
    mgr._warn_if_skills_base_unusable(ctx)  # must not raise


def test_remote_skills_base_probe_warns(manager_with_home, caplog):
    mgr, _home = manager_with_home
    mgr._resolved_remote_home = "/remote/home/bob"
    fake_ctx = types.SimpleNamespace(is_remote=True)
    mgr.executor.run_agent = lambda *a, **k: _result("MISSING\n")
    with caplog.at_level(logging.WARNING, logger="sucoder.test"):
        mgr._warn_if_skills_base_unusable(fake_ctx)
    assert any(
        "missing or a broken symlink" in r.message
        and "/remote/home/bob/.sucoder/skills" in r.message
        for r in caplog.records
    )


def test_remote_probe_failure_is_nonfatal(manager_with_home):
    mgr, _home = manager_with_home
    mgr._resolved_remote_home = "/remote/home/bob"
    fake_ctx = types.SimpleNamespace(is_remote=True)

    def boom(*a, **k):
        raise RuntimeError("ssh down")

    mgr.executor.run_agent = boom
    mgr._warn_if_skills_base_unusable(fake_ctx)  # must not raise


def test_scripts_section_renders_portable_path(manager_with_home):
    mgr, home = manager_with_home
    mgr._render_is_remote = True
    mgr._resolved_remote_home = "/remote/home/bob"
    skill_dir = Path.home() / ".sucoder" / "skills" / "orgmode" / "exporting"

    section = mgr._render_scripts_section(skill_dir)

    assert "bash /remote/home/bob/.sucoder/skills/orgmode/exporting/scripts/export-pdf.sh" in section
    assert "Projects/sucoder-skills" not in section
