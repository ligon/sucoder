"""Tool-version preflight (``sucoder.tool_preflight``, GH #20): the version
parser against the spellings real tools actually use, the report evaluator,
the probe script driven under a real bash with stub binaries on ``PATH``,
and the two guarantees that matter --- it warns, and it never gates.

Needs bash for the script tests; the parser/evaluator tests are pure Python.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from sucoder.config import ConfigError, ToolPreflightConfig, _parse_tool_preflight
from sucoder.tool_preflight import (
    BELOW,
    DEFAULT_TOOL_FLOORS,
    HOST_TAG,
    LINE_TAG,
    MISSING,
    OK,
    UNKNOWN,
    VERSION_FLAGS,
    build_probe_script,
    evaluate,
    format_report,
    parse_version,
    version_at_least,
)

_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


# -- version parsing -----------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        # The five real spellings named in the issue.  None is clean semver.
        ("gh version 2.67.0 (2025-02-11)", (2, 67, 0)),
        ("git version 2.39.3", (2, 39, 3)),
        ("ripgrep 14.1.0", (14, 1, 0)),
        ("tmux 3.3a", (3, 3)),
        ("jq-1.6", (1, 6)),
        # Observed in the wild while building this.
        ("gh version 2.101.0 (2026-09-15)", (2, 101, 0)),
        ("tmux 3.7c", (3, 7)),
        ("git version 2.43.7", (2, 43, 7)),
        ("jq-1.7.1", (1, 7, 1)),
        ("git version 2.39.3 (Apple Git-145)", (2, 39, 3)),
    ],
)
def test_parse_version_handles_the_real_spellings(raw, expected):
    assert parse_version(raw) == expected


@pytest.mark.parametrize("raw", ["", None, "not found", "jq-1", "unknown", "vNext"])
def test_parse_version_returns_none_rather_than_guessing(raw):
    """Unparseable is ``None`` -- recorded, never a pass and never a raise."""
    assert parse_version(raw) is None


def test_the_date_in_gh_version_cannot_win_the_match():
    # Belt and braces for the one real ambiguity: the date uses hyphens,
    # so the dotted-numeric regex cannot see it at all.
    assert parse_version("gh version 2.67.0 (2025-02-11)") == (2, 67, 0)
    assert parse_version("something (2025-02-11)") is None


def test_version_comparison_pads_the_shorter_side():
    assert version_at_least((3, 3), (3, 0))
    assert version_at_least((3, 0), (3, 0, 0))       # 3.0 == 3.0.0
    assert version_at_least((2, 101, 0), (2, 83, 0)) # not string order
    assert not version_at_least((2, 67, 0), (2, 83, 0))
    assert not version_at_least((2, 9), (2, 9, 1))


# -- evaluation ----------------------------------------------------------------

def _line(name, path, raw):
    return "\t".join([LINE_TAG, name, path, raw])


def test_evaluate_grades_below_at_and_above_a_floor():
    out = "\n".join([
        f"{HOST_TAG}\tn0043.savio4",
        _line("gh", "/home/u/bin/gh", "gh version 2.67.0 (2025-02-11)"),   # below
        _line("git", "/usr/bin/git", "git version 2.34.0"),                # exactly at
        _line("tmux", "/usr/bin/tmux", "tmux 3.3a"),                       # above
    ])
    host, reports = evaluate(out, {"gh": "2.83.0", "git": "2.34.0", "tmux": "3.0"})
    assert host == "n0043.savio4"
    assert [(r.name, r.status) for r in reports] == [
        ("gh", BELOW), ("git", OK), ("tmux", OK),
    ]
    assert reports[0].is_warning and not reports[1].is_warning


def test_a_missing_binary_is_recorded_not_raised():
    host, reports = evaluate(_line("jq", "", "not found"), {"jq": "1.6"})
    (report,) = reports
    assert report.status == MISSING
    assert report.is_warning
    assert "jq: not found" in report.describe()


def test_an_unparseable_version_is_neither_a_pass_nor_a_warning():
    """``unknown`` is its own state: recorded, not graded, not shouted about."""
    host, reports = evaluate(_line("rg", "/usr/bin/rg", "ripgrep"), {"rg": "13.0.0"})
    (report,) = reports
    assert report.status == UNKNOWN
    assert report.version is None
    assert not report.is_warning
    assert "unreadable" in report.describe()


def test_a_null_floor_records_the_version_without_judging_it():
    host, reports = evaluate(
        _line("tmux", "/usr/bin/tmux", "tmux 1.8"), {"tmux": None},
    )
    (report,) = reports
    assert report.status == OK and report.floor is None


def test_login_shell_banners_are_ignored_not_parsed():
    noisy = "\n".join([
        "Welcome to the cluster!  Scheduled maintenance 2026-10-01.",
        "  * jq is deprecated, use jaq",
        f"{HOST_TAG}\tlogin1",
        _line("jq", "/usr/bin/jq", "jq-1.6"),
        "Last login: Thu Sep 17",
    ])
    host, reports = evaluate(noisy, {"jq": "1.6"})
    assert host == "login1"
    assert [r.name for r in reports] == ["jq"]


def test_no_output_yields_no_reports():
    """Under --dry-run the executor hands back empty stdout.  Say nothing."""
    assert evaluate("", DEFAULT_TOOL_FLOORS) == (None, [])


def test_format_report_names_the_host_that_was_probed():
    host, reports = evaluate(
        f"{HOST_TAG}\tn0043\n" + _line("gh", "/u/bin/gh", "gh version 2.67.0"),
        {"gh": "2.83.0"},
    )
    text = format_report(host, reports)
    assert "n0043" in text and "BELOW FLOOR" in text and "/u/bin/gh" in text


# -- the probe script ----------------------------------------------------------

def test_probe_script_resolves_all_tokens_and_uses_the_right_version_flag():
    script = build_probe_script(["gh", "tmux", "shellcheck"])
    assert not re.search(r"@[A-Z_]+@", script)
    assert "probe gh gh --version" in script
    assert "probe tmux tmux -V" in script          # NOT --version; tmux rejects it
    assert "probe shellcheck shellcheck --version" in script   # unknown tool default


def test_every_default_tool_has_a_version_flag_entry():
    assert set(DEFAULT_TOOL_FLOORS) <= set(VERSION_FLAGS)


def _stub(bin_dir: Path, name: str, output: str, *, exit_code: int = 0) -> None:
    path = bin_dir / name
    path.write_text(f'#!/bin/bash\necho "{output}"\nexit {exit_code}\n', encoding="utf-8")
    path.chmod(0o755)


@_bash
def test_probe_under_bash_reports_versions_and_a_missing_tool(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir, "gh", "gh version 2.67.0 (2025-02-11)")
    _stub(bin_dir, "tmux", "tmux 3.3a")
    # A tool that cannot be on any PATH: a missing tool must produce
    # "not found", not a traceback and not a non-zero exit.  (A real name
    # like `jq` would be found in /usr/bin on most developer machines,
    # which is the opposite of what this asserts.)
    absent = "sucoder-absent-tool"
    script = build_probe_script(["gh", "tmux", absent])
    proc = subprocess.run(
        ["bash", "-s"], input=script, capture_output=True, text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert proc.returncode == 0, proc.stderr

    host, reports = evaluate(
        proc.stdout, {"gh": "2.83.0", "tmux": "3.0", absent: "1.0"},
    )
    by_name = {r.name: r for r in reports}
    assert by_name["gh"].status == BELOW
    assert by_name["gh"].path == str(bin_dir / "gh")
    assert by_name["tmux"].status == OK
    assert by_name[absent].status == MISSING
    assert "not found" in by_name[absent].describe()


@_bash
def test_probe_survives_a_tool_that_fails_or_prints_nothing(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir, "gh", "boom", exit_code=1)
    (bin_dir / "rg").write_text("#!/bin/bash\nexit 3\n", encoding="utf-8")
    (bin_dir / "rg").chmod(0o755)
    proc = subprocess.run(
        ["bash", "-s"], input=build_probe_script(["gh", "rg"]),
        capture_output=True, text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert proc.returncode == 0, proc.stderr
    host, reports = evaluate(proc.stdout, {"gh": "2.83.0", "rg": "13.0.0"})
    # Both resolved on PATH, so neither is "missing"; both are unreadable.
    assert {r.status for r in reports} == {UNKNOWN}


# -- config --------------------------------------------------------------------

def test_floors_default_to_the_shipped_table():
    cfg = _parse_tool_preflight(None, path=Path("cfg.yaml"))
    assert cfg == ToolPreflightConfig()
    assert cfg.enabled is True
    assert cfg.floors == DEFAULT_TOOL_FLOORS


def test_configured_floors_merge_over_the_defaults_rather_than_replacing_them():
    cfg = _parse_tool_preflight(
        {"floors": {"gh": "2.90.0", "tmux": None, "shellcheck": "0.9.0"}},
        path=Path("cfg.yaml"),
    )
    assert cfg.floors["gh"] == "2.90.0"     # overridden
    assert cfg.floors["tmux"] is None       # floor silenced, still probed
    assert cfg.floors["shellcheck"] == "0.9.0"   # added
    assert cfg.floors["git"] == DEFAULT_TOOL_FLOORS["git"]   # untouched


def test_preflight_can_be_switched_off():
    assert _parse_tool_preflight({"enabled": False}, path=Path("cfg.yaml")).enabled is False


@pytest.mark.parametrize(
    "raw",
    [
        "not-a-mapping",
        {"enabled": "yes"},
        {"floors": ["gh"]},
        {"floors": {"gh": 2}},
        {"floors": {"gh": "latest"}},      # unreadable floor: caught at load time
    ],
)
def test_malformed_tool_preflight_config_is_a_config_error(raw):
    with pytest.raises(ConfigError):
        _parse_tool_preflight(raw, path=Path("cfg.yaml"))


# -- the launch-path guarantee -------------------------------------------------
#
# "Report, never gate" is a promise about the LAUNCH, so it is tested at
# the launch, not asserted in a docstring.

def _launch_with_preflight(tmp_path, monkeypatch, probe_result, **cfg):
    """Drive ``launch_agent`` with a fake executor and return its calls.

    *probe_result* is either a ``CommandResult`` to hand back for the probe
    or an exception instance to raise from it.
    """
    from tests.test_mirror import build_manager
    from sucoder.executor import CommandResult
    from sucoder.mirror import MirrorManager

    manager = build_manager(tmp_path, tool_preflight=True)
    ctx = manager.context_for("sample")
    manager.ensure_clone(ctx)
    manager.config.tool_preflight = ToolPreflightConfig(**cfg) if cfg else ToolPreflightConfig()
    manager.config.mirrors["sample"].agent_launcher.command = ["echo", "hello"]
    manager.config.system_prompt = None
    monkeypatch.setattr(
        MirrorManager, "_default_system_prompt_path",
        staticmethod(lambda: Path("/nonexistent-system-prompt")),
    )

    calls = []

    def fake_run_agent(args, **kwargs):
        calls.append((list(args), kwargs))
        if list(args)[:3] == ["bash", "-l", "-s"]:
            if isinstance(probe_result, BaseException):
                raise probe_result
            return probe_result
        return CommandResult(
            requested_args=list(args), executed_args=list(args),
            stdout="", stderr="", returncode=0,
        )

    monkeypatch.setattr(manager.executor, "run_agent", fake_run_agent)
    manager.launch_agent(ctx, sync=False)
    return manager, calls


def _probe_stdout():
    return "\n".join([
        f"{HOST_TAG}\tn0043.savio4",
        _line("gh", "/home/u/bin/gh", "gh version 2.67.0 (2025-02-11)"),
    ])


def test_launch_logs_the_versions_and_warns_below_floor(tmp_path, monkeypatch, caplog):
    from sucoder.executor import CommandResult

    result = CommandResult(
        requested_args=[], executed_args=[], stdout=_probe_stdout(),
        stderr="", returncode=0,
    )
    with caplog.at_level(logging.INFO, logger="sucoder.test"):
        manager, calls = _launch_with_preflight(tmp_path, monkeypatch, result)

    probe = [a for a, _ in calls if a[:3] == ["bash", "-l", "-s"]]
    assert probe, "the preflight probe did not run at launch"
    text = caplog.text
    assert "gh version 2.67.0" in text
    assert "BELOW FLOOR" in text
    assert "n0043.savio4" in text
    # The warning says WHOSE problem it is -- the misattribution is the bug.
    assert "not a property of the repository" in text
    assert "did not block the launch" in text
    # and the agent was still launched, last, after the probe
    assert calls[-1][0] == ["echo", "hello"]


def test_a_failing_probe_does_not_fail_the_launch(tmp_path, monkeypatch):
    """The whole point: a broken preflight must not stop a session."""
    manager, calls = _launch_with_preflight(
        tmp_path, monkeypatch, RuntimeError("ssh exploded"),
    )
    assert calls[-1][0] == ["echo", "hello"]   # agent still launched


def test_garbage_from_the_probe_does_not_fail_the_launch(tmp_path, monkeypatch):
    from sucoder.executor import CommandResult

    result = CommandResult(
        requested_args=[], executed_args=[],
        stdout="\x00 not tab separated at all\n", stderr="boom", returncode=127,
    )
    manager, calls = _launch_with_preflight(tmp_path, monkeypatch, result)
    assert calls[-1][0] == ["echo", "hello"]


def test_disabled_preflight_does_not_probe(tmp_path, monkeypatch):
    from sucoder.executor import CommandResult

    result = CommandResult(
        requested_args=[], executed_args=[], stdout=_probe_stdout(),
        stderr="", returncode=0,
    )
    manager, calls = _launch_with_preflight(
        tmp_path, monkeypatch, result, enabled=False,
    )
    assert not [a for a, _ in calls if a[:3] == ["bash", "-l", "-s"]]
    assert calls[-1][0] == ["echo", "hello"]


# -- `sucoder doctor` ----------------------------------------------------------

def _doctor_config(tmp_path: Path, *, slurm: str = "") -> Path:
    import os

    user = os.environ.get("USER", "coder")
    (tmp_path / "mirrors").mkdir(exist_ok=True)
    (tmp_path / "canonical").mkdir(exist_ok=True)
    body = f"""
human_user: {user}
agent_user: {user}
agent_group: {user}
mirror_root: {tmp_path / "mirrors"}
{slurm}
mirrors:
  sample:
    canonical_repo: {tmp_path / "canonical"}
    mirror_name: sample
"""
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_doctor_refuses_to_allocate_a_compute_node(tmp_path, monkeypatch):
    """A diagnostic must not bill an salloc just to read `--version` output."""
    from typer.testing import CliRunner

    from sucoder import cli

    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)
    config = _doctor_config(tmp_path, slurm="""
targets:
  savio:
    gateway: brc.example.edu
    transfer_host: dtn.example.edu
    mirror_root: ~/mirrors
    slurm:
      partition: savio3
      account: acct
""")
    result = CliRunner().invoke(
        cli.app, ["--config", str(config), "-T", "savio", "doctor", "sample"],
    )
    assert result.exit_code == 2
    assert "will not allocate a compute node" in result.output


def test_doctor_reports_and_exits_non_zero_below_floor(tmp_path, monkeypatch):
    """Unlike the launch hook, the manual entry point *is* allowed to fail."""
    from typer.testing import CliRunner

    from sucoder import cli
    from sucoder.mirror import MirrorManager

    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)
    monkeypatch.setattr(
        MirrorManager, "tool_preflight",
        lambda self, ctx: evaluate(_probe_stdout(), DEFAULT_TOOL_FLOORS),
    )
    config = _doctor_config(tmp_path)
    result = CliRunner().invoke(cli.app, ["--config", str(config), "doctor", "sample"])
    assert result.exit_code == 1
    assert "gh version 2.67.0" in result.output
    assert "BELOW FLOOR" in result.output
    assert "`mv`" in result.output or "mv" in result.output


def test_doctor_exits_zero_when_everything_is_current(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from sucoder import cli
    from sucoder.mirror import MirrorManager

    current = "\n".join([
        f"{HOST_TAG}\tlogin1",
        _line("gh", "/u/bin/gh", "gh version 2.101.0 (2026-09-15)"),
    ])
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)
    monkeypatch.setattr(
        MirrorManager, "tool_preflight",
        lambda self, ctx: evaluate(current, DEFAULT_TOOL_FLOORS),
    )
    config = _doctor_config(tmp_path)
    result = CliRunner().invoke(cli.app, ["--config", str(config), "doctor", "sample"])
    assert result.exit_code == 0, result.output
    assert "gh version 2.101.0" in result.output


def test_doctor_refuses_a_mirrors_own_slurm_block_too(tmp_path, monkeypatch):
    """The -T target is not the only way a mirror acquires a `slurm` block.

    ``_build_manager_for_mirror`` overlays an explicit target ONTO the
    mirror's own ``remote:``; consult only the target and a mirror that
    carries its own non-confined SLURM config walks past the guard into
    ``salloc``.
    """
    from typer.testing import CliRunner

    from sucoder import cli

    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **kw: None)
    import os

    user = os.environ.get("USER", "coder")
    (tmp_path / "mirrors").mkdir(exist_ok=True)
    (tmp_path / "canonical").mkdir(exist_ok=True)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
human_user: {user}
agent_user: {user}
agent_group: {user}
mirror_root: {tmp_path / "mirrors"}
mirrors:
  sample:
    canonical_repo: {tmp_path / "canonical"}
    mirror_name: sample
    remote:
      gateway: brc.example.edu
      transfer_host: dtn.example.edu
      mirror_root: ~/mirrors
      slurm:
        partition: savio3
        account: acct
""",
        encoding="utf-8",
    )
    result = CliRunner().invoke(cli.app, ["--config", str(config), "doctor", "sample"])
    assert result.exit_code == 2
    assert "will not allocate a compute node" in result.output
