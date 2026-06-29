"""`_build_sbatch_command` -- the sbatch argv for a `confined` launch."""
from sucoder.cli import _build_sbatch_command
from sucoder.config import SlurmConfig


def _slurm(**kw):
    base = dict(partition="savio4_htc", account="co_carleton", time="3-00:00:00")
    base.update(kw)
    return SlurmConfig(**base)


def test_minimal_required_flags():
    cmd = _build_sbatch_command(
        _slurm(), job_name="sucoder-K", log_path="/h/log.out", script_path="/h/run.sh"
    )
    assert cmd[0] == "sbatch"
    assert "--parsable" in cmd and "--no-requeue" in cmd
    assert "--job-name=sucoder-K" in cmd
    assert "--output=/h/log.out" in cmd
    assert "--partition=savio4_htc" in cmd
    assert "--account=co_carleton" in cmd
    assert "--time=3-00:00:00" in cmd
    assert cmd[-1] == "/h/run.sh"                 # the script is the final arg
    # optional flags absent when unset
    assert not any(p.startswith("--qos") for p in cmd)
    assert not any(p.startswith("--cpus-per-task") for p in cmd)
    assert not any(p.startswith("--mem") for p in cmd)
    assert not any(p.startswith("--nodelist") for p in cmd)


def test_optional_flags_and_nodelist():
    cmd = _build_sbatch_command(
        _slurm(qos="carleton_htc4_normal", cpus_per_task=4, mem="16G"),
        job_name="j", log_path="l", script_path="s", nodelist="n0055.savio4",
    )
    assert "--qos=carleton_htc4_normal" in cmd
    assert "--cpus-per-task=4" in cmd
    assert "--mem=16G" in cmd
    assert "--nodelist=n0055.savio4" in cmd
    assert cmd[-1] == "s"
