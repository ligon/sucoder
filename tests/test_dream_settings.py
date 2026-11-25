"""Exploratory dream defaults and inherited project policy validation."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import re

import pytest
import yaml

from sucoder.config import BranchPrefixes, Config, ConfigError, DreamSettings, MirrorSettings, load_config


def configured(tmp_path, global_dream=None, mirror_dream=None):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(dict(
        human_user="fixture", mirror_root=str(tmp_path / "mirrors"), dream=global_dream,
        mirrors=dict(sample=dict(canonical_repo=str(tmp_path / "project"), dream=mirror_dream),
            other=dict(canonical_repo=str(tmp_path / "other")))), sort_keys=False))
    return load_config(path)


def test_existing_constructors_and_configurations_keep_usable_defaults(tmp_path):
    cfg = configured(tmp_path)
    expected = DreamSettings()
    assert cfg.dream == cfg.mirrors["sample"].dream == cfg.mirrors["other"].dream == expected
    assert (expected.max_total_seconds, expected.max_total_calls) == (900, 40)
    assert (expected.reserve_seconds, expected.reserve_calls) == (30, 3)
    assert (expected.max_discovery_calls, expected.max_discovery_bytes) == (8, 131072)
    assert expected.harness is expected.model is None
    assert Config("fixture").dream == expected
    direct = MirrorSettings("sample", Path("/project"), "sample", BranchPrefixes())
    assert direct.dream is None  # Runtime-created mirrors inherit config.dream.
    assert cfg.agent_launcher is None
    assert cfg.mirrors["sample"].agent_launcher.command == ["claude"]


def test_partial_project_overrides_inherit_global_defaults_without_mutation(tmp_path):
    cfg = configured(tmp_path, dict(harness="codex", model="  configured-model  ",
        max_total_seconds=1200, max_total_calls=24, reserve_calls=4),
        dict(max_discovery_calls=5, reserve_seconds=90))
    project = cfg.mirrors["sample"].dream
    assert project == DreamSettings(harness="codex", model="configured-model",
        max_total_seconds=1200, max_total_calls=24, reserve_calls=4,
        max_discovery_calls=5, reserve_seconds=90)
    assert cfg.mirrors["other"].dream == cfg.dream
    assert cfg.dream.reserve_seconds == 30 and cfg.dream.max_discovery_calls == 8
    with pytest.raises(FrozenInstanceError):
        cfg.dream.max_total_calls = 1000


def test_project_can_clear_configured_backend_inheritance_and_disable_reserves(tmp_path):
    cfg = configured(tmp_path, dict(harness="codex", model="global-model"),
        dict(harness=None, model=None, max_total_seconds=1, max_total_calls=1,
            reserve_seconds=0, reserve_calls=0))
    project = cfg.mirrors["sample"].dream
    assert project.harness is project.model is None
    assert (project.reserve_seconds, project.reserve_calls) == (0, 0)
    assert cfg.dream.model == "global-model"


@pytest.mark.parametrize("scope", ["global", "project"])
@pytest.mark.parametrize("raw", [[], False, "codex"])
def test_dream_block_requires_a_mapping(tmp_path, scope, raw):
    label = "dream" if scope == "global" else "mirrors.sample.dream"
    with pytest.raises(ConfigError, match=re.escape("`" + label + "` must be a mapping")):
        configured(tmp_path, raw if scope == "global" else None, raw if scope == "project" else None)


@pytest.mark.parametrize("scope", ["global", "project"])
@pytest.mark.parametrize("raw,field", [
    ({"harness": "gemini"}, "harness"),
    ({"harness": ""}, "harness"),
    ({"harness": True}, "harness"),
    ({"harness": []}, "harness"),
    ({"model": ""}, "model"),
    ({"model": "  "}, "model"),
    ({"model": False}, "model"),
    ({"model": []}, "model"),
    ({"max_total_seconds": 0}, "max_total_seconds"),
    ({"max_total_seconds": -1}, "max_total_seconds"),
    ({"max_total_seconds": True}, "max_total_seconds"),
    ({"max_total_seconds": 900.0}, "max_total_seconds"),
    ({"max_total_seconds": "900"}, "max_total_seconds"),
    ({"max_total_seconds": None}, "max_total_seconds"),
    ({"max_total_calls": 0}, "max_total_calls"),
    ({"max_total_calls": False}, "max_total_calls"),
    ({"reserve_seconds": -1}, "reserve_seconds"),
    ({"reserve_seconds": False}, "reserve_seconds"),
    ({"reserve_calls": -1}, "reserve_calls"),
    ({"reserve_calls": True}, "reserve_calls"),
    ({"max_discovery_calls": 0}, "max_discovery_calls"),
    ({"max_discovery_calls": True}, "max_discovery_calls"),
    ({"max_discovery_bytes": 0}, "max_discovery_bytes"),
    ({"max_discovery_bytes": False}, "max_discovery_bytes"),
])
def test_dream_values_reject_invalid_types_and_unbounded_allowances(tmp_path, scope, raw, field):
    label = "dream" if scope == "global" else "mirrors.sample.dream"
    with pytest.raises(ConfigError, match=re.escape(label + "." + field)):
        configured(tmp_path, raw if scope == "global" else None, raw if scope == "project" else None)


@pytest.mark.parametrize("scope", ["global", "project"])
def test_unknown_settings_fail_with_their_configuration_path(tmp_path, scope):
    label = "dream" if scope == "global" else "mirrors.sample.dream"
    with pytest.raises(ConfigError, match=re.escape("Unknown key(s) under `" + label + "`: max_tokenz")):
        configured(tmp_path, dict(max_tokenz=10) if scope == "global" else None,
            dict(max_tokenz=10) if scope == "project" else None)


@pytest.mark.parametrize("override,field", [
    (dict(max_total_seconds=30), "reserve_seconds"),
    (dict(max_total_calls=2), "reserve_calls"),
    (dict(reserve_seconds=900), "reserve_seconds"),
    (dict(reserve_calls=41), "reserve_calls"),
])
@pytest.mark.parametrize("scope", ["global", "project"])
def test_closeout_reserves_fit_the_resolved_inherited_limits(tmp_path, scope, override, field):
    label = "dream" if scope == "global" else "mirrors.sample.dream"
    with pytest.raises(ConfigError, match=re.escape(label + "." + field)):
        configured(tmp_path, override if scope == "global" else None,
            override if scope == "project" else None)


def test_claude_backend_and_exact_call_reserve_are_supported(tmp_path):
    cfg = configured(tmp_path, dict(harness="claude", model="claude-model",
        max_total_calls=3, reserve_calls=3))
    assert cfg.dream.harness == "claude"
    assert cfg.dream.max_total_calls == cfg.dream.reserve_calls == 3
