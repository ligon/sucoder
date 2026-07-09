"""Tests for the ``scripts/savio-run`` helper.

``savio-run`` is a bash script that dispatches a command to a Savio compute
node over SuCoder's warm tunnels.  The cluster-facing half (``sucoder tunnel
up``, ``ssh``, ``srun``/``sbatch``) can't run in CI, so the script exposes a
``--dry-run`` seam that resolves everything locally — arg parsing, mode
selection, SLURM-resource resolution from the config, and faithful command
reconstruction — then prints a ``key=value`` plan and exits before any ssh.
These tests drive that seam as a subprocess, pointing ``SUCODER_CONFIG`` at a
fixture so nothing depends on the developer's real ``~/.sucoder/config.yaml``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "savio-run"

CONFIG = """\
targets:
  testcluster:
    slurm:
      partition: testpart
      account: testacct
      qos: testqos
      cpus_per_task: 4
      mem: 16G
      time: "1-00:00:00"
  noslurm:
    gateway: gw.example.org
"""


@pytest.fixture()
def cfg(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(CONFIG, encoding="utf-8")
    return p


def run(cfg: Path, *args: str, expect_rc: int = 0, extra_env=None) -> subprocess.CompletedProcess:
    env = {**os.environ, "SUCODER_CONFIG": str(cfg)}
    env.pop("TARGET", None)  # keep tests hermetic regardless of the caller's env
    # The script's config parse needs PyYAML; the ambient `python3` may lack it
    # (this repo's system python does), so point it at the interpreter running
    # the tests, which has the project's deps.
    env.setdefault("SUCODER_PYTHON", sys.executable)
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == expect_rc, (
        f"rc={proc.returncode}\n--stdout--\n{proc.stdout}\n--stderr--\n{proc.stderr}"
    )
    return proc


def plan(proc: subprocess.CompletedProcess) -> dict:
    """Parse the dry-run ``key=value`` lines into a dict."""
    out = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k] = v
    return out


# ---------------------------------------------------------------- run mode

def test_run_defaults_from_config(cfg):
    p = plan(run(cfg, "-T", "testcluster", "-n", "--", "python3", "-c", "print(1)"))
    assert p["mode"] == "run"
    assert p["target"] == "testcluster"
    assert p["ln"] == "testcluster-ln"
    s = p["slurm"]
    for flag in (
        "--partition=testpart",
        "--account=testacct",
        "--qos=testqos",
        "--cpus-per-task=4",
        "--mem=16G",
        "--time=1-00:00:00",
        "--job-name=savio-run",
    ):
        assert flag in s, f"missing {flag} in {s!r}"


def test_resource_overrides(cfg):
    p = plan(run(
        cfg, "-T", "testcluster", "-n",
        "-c", "8", "--mem", "32G", "-p", "other",
        "-t", "2:00:00", "-J", "job", "-d", "/w",
        "--", "echo", "hi",
    ))
    s = p["slurm"]
    for flag in (
        "--cpus-per-task=8", "--mem=32G", "--partition=other",
        "--time=2:00:00", "--job-name=job", "--chdir=/w",
    ):
        assert flag in s, f"missing {flag} in {s!r}"
    # account/qos untouched by the overrides
    assert "--account=testacct" in s and "--qos=testqos" in s


# -------------------------------------------------- command reconstruction

def test_argv_is_requoted(cfg):
    # multiple args -> argv; re-quoted so the parens survive `bash -lc`
    p = plan(run(cfg, "-T", "testcluster", "-n", "--", "python3", "-c", "print(1)"))
    assert p["payload"] == r"python3 -c print\(1\)"


def test_single_arg_is_shell_string(cfg):
    # a lone arg is a shell string: pipes/operators preserved verbatim
    p = plan(run(cfg, "-T", "testcluster", "-n", "--", "a | b | c"))
    assert p["payload"] == "a | b | c"


def test_bare_command_without_dashdash(cfg):
    p = plan(run(cfg, "-T", "testcluster", "-n", "echo", "hi"))
    assert p["payload"] == "echo hi"


# -------------------------------------------------------------- other modes

def test_batch_mode(cfg):
    p = plan(run(cfg, "-T", "testcluster", "-n", "--batch", "--", "echo", "hi"))
    assert p["mode"] == "batch"
    assert p["payload"] == "echo hi"
    assert "--partition=testpart" in p["slurm"]


def test_fetch_mode(cfg):
    p = plan(run(cfg, "-T", "testcluster", "-n", "--fetch", "12345"))
    assert p["mode"] == "fetch"
    assert p["jobid"] == "12345"
    assert p["ln"] == "testcluster-ln"


def test_status_without_jobid(cfg):
    p = plan(run(cfg, "-T", "testcluster", "-n", "--status"))
    assert p["mode"] == "status"
    assert p["jobid"] == ""


def test_status_with_jobid(cfg):
    p = plan(run(cfg, "-T", "testcluster", "-n", "--status", "999"))
    assert p["mode"] == "status"
    assert p["jobid"] == "999"


# ------------------------------------------------------ target resolution

def test_target_from_env(cfg):
    proc = run(cfg, "-n", "--", "echo", "hi", extra_env={"TARGET": "testcluster"})
    assert plan(proc)["target"] == "testcluster"


# ---------------------------------------------------------------- errors

def test_no_command_errors(cfg):
    proc = run(cfg, "-T", "testcluster", "-n", expect_rc=1)
    assert "no command given" in proc.stderr


def test_unknown_option_errors(cfg):
    proc = run(cfg, "--bogus", expect_rc=1)
    assert "unknown option" in proc.stderr


def test_missing_slurm_partition_errors(cfg):
    proc = run(cfg, "-T", "noslurm", "-n", "--", "echo", "hi", expect_rc=1)
    assert "no SLURM partition" in proc.stderr


def test_help_lists_modes(cfg):
    proc = run(cfg, "--help")
    assert "savio-run" in proc.stdout
    assert "--batch" in proc.stdout
    assert "--fetch" in proc.stdout
