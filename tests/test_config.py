import os
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from sucoder.config import BranchPrefixes, ConfigError, build_default_config, load_config


def write_config(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_load_config_success(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        """
human_user: ligon
agent_user: coder
agent_group: coder
mirror_root: ./mirrors
mirrors:
  sample:
    canonical_repo: ./canonical
    branch_prefixes:
      human: ligon
      agent: coder
""",
    )
    cfg = load_config(config_path)

    assert cfg.human_user == "ligon"
    assert cfg.agent_user == "coder"
    sample = cfg.mirrors["sample"]
    assert sample.branch_prefixes == BranchPrefixes(human="ligon", agent="coder")
    assert sample.agent_launcher.command == ["codex"]
    assert sample.agent_launcher.env == {}
    assert sample.agent_launcher.nvm is None
    assert sample.agent_launcher.accepts_inline_prompt is None
    assert sample.agent_launcher.needs_yolo is None
    assert sample.agent_launcher.writable_dirs == []
    assert sample.agent_launcher.workdir is None
    assert sample.agent_launcher.default_flags == []
    # Flag templates default to None - actual values come from AGENT_PROFILES at runtime
    assert sample.agent_launcher.flags.yolo is None
    assert sample.agent_launcher.flags.writable_dir is None
    assert sample.agent_launcher.flags.workdir is None
    assert sample.agent_launcher.flags.default_flag == "{flag}"
    assert sample.agent_launcher.flags.skills is None
    assert sample.agent_launcher.flags.system_prompt is None
    assert sample.skills == []
    assert cfg.system_prompt is None


@pytest.mark.parametrize(
    "yaml_content, message",
    [
        ("{}", "`human_user` must be set"),
        (
            """
human_user: ligon
mirror_root: ./mirrors
""",
            "`mirrors` must be defined",
        ),
        (
            """
human_user: ligon
mirror_root: ./mirrors
mirrors: {}
""",
            "At least one mirror must be configured.",
        ),
        (
            """
human_user: ligon
mirror_root: ./mirrors
mirrors:
  sample: {}
""",
            "Mirror `sample` requires `canonical_repo`.",
        ),
        (
            """
human_user: ligon
mirror_root: ./mirrors
mirrors:
  sample:
    canonical_repo: ./canonical
    skills: ./skill
""",
            "`skills` must be a list of paths when provided.",
        ),
        (
            """
human_user: ligon
mirror_root: ./mirrors
mirrors:
  sample:
    canonical_repo: ./canonical
    agent_launcher: 123
""",
            "`agent_launcher` must be a mapping when provided.",
        ),
        (
            """
human_user: ligon
mirror_root: ./mirrors
mirrors:
  sample:
    canonical_repo: ./canonical
    agent_launcher:
      env: [1, 2, 3]
""",
            "`agent_launcher.env` must be a mapping of string keys to string values.",
        ),
        (
            """
human_user: ligon
mirror_root: ./mirrors
mirrors:
  sample:
    canonical_repo: ./canonical
    agent_launcher:
      nvm: 123
""",
            "`agent_launcher.nvm` must be a mapping when provided.",
        ),
        (
            """
human_user: ligon
mirror_root: ./mirrors
mirrors:
  sample:
    canonical_repo: ./canonical
    agent_launcher:
      nvm:
        version: ""
""",
            "`agent_launcher.nvm.version` must be a non-empty string.",
        ),
        (
            """
human_user: ligon
mirror_root: ./mirrors
mirrors:
  sample:
    canonical_repo: ./canonical
    agent_launcher:
      nvm:
        version: "22"
        dir: [1,2]
""",
            "`agent_launcher.nvm.dir` must be a path string when provided.",
        ),
    ],
)
def test_load_config_failures(tmp_path: Path, yaml_content: str, message: str) -> None:
    config_path = write_config(tmp_path, yaml_content)

    with pytest.raises(ConfigError) as excinfo:
        load_config(config_path)

    assert message in str(excinfo.value)


def test_load_config_custom_agent_launcher(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        """
human_user: ligon
mirror_root: ./mirrors
mirrors:
  sample:
    canonical_repo: ./canonical
    agent_launcher:
      command:
        - codex
        - "--headless"
      env:
        CODEX_DEBUG: "1"
""",
    )

    cfg = load_config(config_path)
    launcher = cfg.mirrors["sample"].agent_launcher
    assert launcher.command == ["codex", "--headless"]
    assert launcher.env == {"CODEX_DEBUG": "1"}
    assert launcher.nvm is None


def test_load_config_agent_launcher_with_nvm(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        """
human_user: ligon
mirror_root: ./mirrors
mirrors:
  sample:
    canonical_repo: ./canonical
    agent_launcher:
      nvm:
        version: "22.11.0"
        dir: ~/.nvm
""",
    )

    cfg = load_config(config_path)
    launcher = cfg.mirrors["sample"].agent_launcher
    assert launcher.nvm is not None
    assert launcher.nvm.version == "22.11.0"
    assert str(launcher.nvm.dir).endswith(".nvm")


def test_load_config_agent_launcher_with_flags(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    workdir_str = str(workdir)
    writable_one = tmp_path / "data"
    config_path = write_config(
        tmp_path,
        f"""
human_user: ligon
mirror_root: ./mirrors
mirrors:
  sample:
    canonical_repo: ./canonical
    agent_launcher:
      accepts_inline_prompt: false
      needs_yolo: false
      writable_dirs:
        - {writable_one}
      workdir: {workdir_str}
      default_flags:
        - "--quiet"
      flags:
        yolo: "--permit"
        writable_dir: "--mount {{path}}:{{path}}:rw"
        workdir: "--here {{path}}"
        default_flag: "--add {{flag}}"
        skills: "--skills {{path}}"
""",
    )

    cfg = load_config(config_path)
    launcher = cfg.mirrors["sample"].agent_launcher
    assert launcher.accepts_inline_prompt is False
    assert launcher.needs_yolo is False
    assert launcher.writable_dirs == [writable_one.resolve()]
    assert launcher.workdir == workdir.resolve()
    assert launcher.default_flags == ["--quiet"]
    flags = launcher.flags
    assert flags.yolo == "--permit"
    assert flags.writable_dir == "--mount {path}:{path}:rw"
    assert flags.workdir == "--here {path}"
    assert flags.default_flag == "--add {flag}"
    assert flags.skills == "--skills {path}"


def test_load_config_with_system_prompt(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.org"
    prompt_file.write_text("Sample\n", encoding="utf-8")
    config_path = write_config(
        tmp_path,
        f"""
human_user: ligon
mirror_root: ./mirrors
system_prompt: {prompt_file}
mirrors:
  sample:
    canonical_repo: ./canonical
""",
    )

    cfg = load_config(config_path)
    assert cfg.system_prompt == prompt_file.resolve()


def test_load_config_missing_system_prompt(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        """
human_user: ligon
mirror_root: ./mirrors
system_prompt: ./missing-file.org
mirrors:
  sample:
    canonical_repo: ./canonical
""",
    )

    with pytest.raises(ConfigError) as excinfo:
        load_config(config_path)

    assert "system_prompt file not found" in str(excinfo.value)


def test_load_config_skills(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    config_path = write_config(
        tmp_path,
        f"""
human_user: ligon
mirror_root: ./mirrors
mirrors:
  sample:
    canonical_repo: ./canonical
    skills:
      - {skill_dir}
""",
    )

    cfg = load_config(config_path)
    skills = cfg.mirrors["sample"].skills
    assert len(skills) == 1
    assert skills[0] == skill_dir.resolve()


def test_load_config_launch_mode_subprocess(tmp_path: Path) -> None:
    """Test parsing launch_mode: subprocess."""
    config_path = write_config(
        tmp_path,
        """
human_user: ligon
mirror_root: ./mirrors
mirrors:
  sample:
    canonical_repo: ./canonical
    agent_launcher:
      command: [gemini]
      launch_mode: subprocess
""",
    )

    cfg = load_config(config_path)
    launcher = cfg.mirrors["sample"].agent_launcher
    assert launcher.launch_mode == "subprocess"


def test_load_config_launch_mode_exec(tmp_path: Path) -> None:
    """Test parsing launch_mode: exec."""
    config_path = write_config(
        tmp_path,
        """
human_user: ligon
mirror_root: ./mirrors
mirrors:
  sample:
    canonical_repo: ./canonical
    agent_launcher:
      command: [claude]
      launch_mode: exec
""",
    )

    cfg = load_config(config_path)
    launcher = cfg.mirrors["sample"].agent_launcher
    assert launcher.launch_mode == "exec"


def test_load_config_launch_mode_default(tmp_path: Path) -> None:
    """Test that launch_mode defaults to None when not specified."""
    config_path = write_config(
        tmp_path,
        """
human_user: ligon
mirror_root: ./mirrors
mirrors:
  sample:
    canonical_repo: ./canonical
    agent_launcher:
      command: [codex]
""",
    )

    cfg = load_config(config_path)
    launcher = cfg.mirrors["sample"].agent_launcher
    assert launcher.launch_mode is None


def test_load_config_launch_mode_invalid(tmp_path: Path) -> None:
    """Test that invalid launch_mode raises ConfigError."""
    config_path = write_config(
        tmp_path,
        """
human_user: ligon
mirror_root: ./mirrors
mirrors:
  sample:
    canonical_repo: ./canonical
    agent_launcher:
      launch_mode: invalid
""",
    )

    with pytest.raises(ConfigError) as excinfo:
        load_config(config_path)

    assert "launch_mode" in str(excinfo.value)
    assert "'subprocess' or 'exec'" in str(excinfo.value)


# ---------------------------------------------------------------------------
# BranchPrefixes default derives from $USER
# ---------------------------------------------------------------------------


def test_branch_prefixes_default_uses_user_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER", "alice")
    bp = BranchPrefixes()
    assert bp.human == "alice"
    assert bp.agent == "coder"


def test_branch_prefixes_explicit_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER", "alice")
    bp = BranchPrefixes(human="bob")
    assert bp.human == "bob"


def test_branch_prefixes_empty_user_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("USER", raising=False)
    bp = BranchPrefixes()
    assert bp.human == ""


# ---------------------------------------------------------------------------
# build_default_config
# ---------------------------------------------------------------------------


def test_build_default_config_in_git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER", "testuser")
    # Simulate git returning tmp_path as the repo root.
    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=str(tmp_path) + "\n", stderr=""
    )
    with mock.patch("sucoder.config.subprocess.run", return_value=fake_result):
        cfg = build_default_config()

    assert cfg.human_user == "testuser"
    assert cfg.agent_user == "coder"
    assert cfg.agent_group == "coder"
    assert cfg.mirror_root == Path("/var/tmp/coder-mirrors")

    mirror_name = tmp_path.name
    assert mirror_name in cfg.mirrors
    mirror = cfg.mirrors[mirror_name]
    assert mirror.canonical_repo == tmp_path
    assert mirror.branch_prefixes.human == "testuser"
    assert mirror.branch_prefixes.agent == "coder"


def test_build_default_config_not_in_git_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER", "testuser")
    with mock.patch(
        "sucoder.config.subprocess.run",
        side_effect=subprocess.CalledProcessError(128, "git"),
    ):
        with pytest.raises(ConfigError, match="Not inside a git repository"):
            build_default_config()


def test_build_default_config_user_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("USER", raising=False)
    with pytest.raises(ConfigError, match="\\$USER is not set"):
        build_default_config()
