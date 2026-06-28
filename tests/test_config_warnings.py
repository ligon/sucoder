"""Tests for non-fatal config warnings (`ConfigWarning`).

A target-level option (notably ``system_prompt_extra``) accidentally
nested under ``slurm:`` used to be dropped silently.  The parser now
warns rather than erroring, so existing configs keep loading.
"""
import warnings

import pytest

from sucoder.config import ConfigWarning, _parse_slurm_config

BASE = {"partition": "savio4_htc", "account": "co_carleton"}


def test_valid_slurm_block_emits_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigWarning)  # any ConfigWarning -> failure
        cfg = _parse_slurm_config(
            {
                **BASE,
                "qos": "carleton_htc4_normal",
                "cpus_per_task": 4,
                "mem": "16G",
                "time": "3-00:00:00",
                "local_disk": "/local",
            }
        )
    assert cfg.partition == "savio4_htc"
    assert cfg.account == "co_carleton"


def test_misplaced_target_key_warns_and_is_ignored():
    with pytest.warns(ConfigWarning, match="target-level"):
        cfg = _parse_slurm_config({**BASE, "system_prompt_extra": "~/p.org"})
    # Parsing still succeeds and the misplaced key changed nothing.
    assert cfg.partition == "savio4_htc"
    assert not hasattr(cfg, "system_prompt_extra")


def test_unknown_slurm_key_warns():
    with pytest.warns(ConfigWarning, match="Unknown key"):
        cfg = _parse_slurm_config({**BASE, "partiton": "typo"})  # misspelled
    assert cfg.partition == "savio4_htc"
