import grp
import os
import pwd
from pathlib import Path

import pytest

from sucoder.config import BranchPrefixes, Config, MirrorSettings
from sucoder.startup_checks import StartupError, run_startup_checks


def _build_config(tmp_path: Path, *, agent_user: str, agent_group: str) -> Config:
    canonical = tmp_path / "canonical.git"
    canonical.mkdir()
    mirrors = {
        "sample": MirrorSettings(
            name="sample",
            canonical_repo=canonical,
            mirror_name="sample",
            branch_prefixes=BranchPrefixes(),
        )
    }
    return Config(
        human_user=agent_user,
        agent_user=agent_user,
        agent_group=agent_group,
        mirror_root=tmp_path,
        log_dir=None,
        mirrors=mirrors,
    )


def test_startup_checks_pass_when_agent_can_read_but_not_write(tmp_path: Path) -> None:
    user = pwd.getpwuid(os.getuid()).pw_name
    group = grp.getgrgid(os.getgid()).gr_name
    config = _build_config(tmp_path, agent_user=user, agent_group=group)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("sample: value\n", encoding="utf-8")
    config_path.chmod(0o440)

    run_startup_checks(config, config_path, use_sudo=False)


def test_startup_checks_fail_when_agent_can_write(tmp_path: Path) -> None:
    user = pwd.getpwuid(os.getuid()).pw_name
    group = grp.getgrgid(os.getgid()).gr_name
    config = _build_config(tmp_path, agent_user=user, agent_group=group)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("sample: value\n", encoding="utf-8")
    config_path.chmod(0o640)

    with pytest.raises(StartupError):
        run_startup_checks(config, config_path, use_sudo=False)


def test_startup_checks_fail_for_missing_agent(tmp_path: Path) -> None:
    user = pwd.getpwuid(os.getuid()).pw_name
    group = grp.getgrgid(os.getgid()).gr_name
    config = _build_config(tmp_path, agent_user=user, agent_group=group)
    config.agent_user = "nonexistent-user"

    config_path = tmp_path / "config.yaml"
    config_path.write_text("sample: value\n", encoding="utf-8")
    config_path.chmod(0o440)

    with pytest.raises(StartupError):
        run_startup_checks(config, config_path, use_sudo=False)


def test_startup_checks_fail_when_config_not_readable(tmp_path: Path) -> None:
    user = pwd.getpwuid(os.getuid()).pw_name
    group = grp.getgrgid(os.getgid()).gr_name
    config = _build_config(tmp_path, agent_user=user, agent_group=group)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("sample: value\n", encoding="utf-8")
    config_path.chmod(0o000)

    with pytest.raises(StartupError):
        run_startup_checks(config, config_path, use_sudo=False)
