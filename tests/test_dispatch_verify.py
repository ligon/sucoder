"""An independent deterministic job tests private commits, not a moving mirror."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sucoder import dispatch, dispatch_verify as verify
from tests.test_dispatch_project import profile_file
from tests.test_dispatch_worker import FakeJobOwnership, git


@pytest.fixture
def verification(tmp_path):
    source = tmp_path / "human"
    source.mkdir()
    git(source, "init", "-q")
    git(source, "config", "user.name", "Test")
    git(source, "config", "user.email", "test@example.invalid")
    (source / "answer.py").write_text("OK = False\n")
    git(source, "add", ".")
    git(source, "commit", "-qm", "control")
    control = git(source, "rev-parse", "HEAD")
    results = tmp_path / "results.git"
    subprocess.check_call(["git", "clone", "--bare", "-q", str(source), str(results)])
    author = tmp_path / "implementer"
    subprocess.check_call(["git", "clone", "-q", str(results), str(author)])
    git(author, "config", "user.name", "Test")
    git(author, "config", "user.email", "test@example.invalid")
    (author / "answer.py").write_text("OK = True\n")
    git(author, "commit", "-qam", "candidate")
    candidate = git(author, "rev-parse", "HEAD")
    git(author, "push", "-q", "origin", "HEAD:refs/heads/private-candidate")
    script = '''set -eu
python3 - <<'PY'
import json, os, sys
with open(os.environ['SUCODER_PROJECT_RESULT'], 'w') as f:
    json.dump({'schema_version': 1, 'state': 'ready',
               'environment': {'PROJECT_PYTHON': sys.executable},
               'runtime': {'cwd': os.getcwd()}}, f)
PY
'''
    profile = profile_file(tmp_path, script=script, output_environment=["PROJECT_PYTHON"])
    command = tmp_path / "command.json"
    command.write_text(json.dumps(["{python}", "-c", "import answer; assert answer.OK"]))
    mounts = tmp_path / "mountinfo"
    mounts.write_text("1 0 8:1 / / rw - ext4 /dev/test rw\n")
    disk = tmp_path / "disk"
    disk.mkdir()
    return dict(source=source, results=results, control=control, candidate=candidate,
                profile=profile, command=command, mounts=mounts, disk=disk)


def freeze(case, tmp_path, commit, name):
    return verify.freeze(case["results"], commit, case["profile"], case["command"], tmp_path / name,
                         lambda p, label: p.resolve(), dispatch._git,
                         local_disk_root=str(case["disk"]))


def run(case, report, job):
    return verify.run(report["directory"], {"SLURM_JOB_ID": str(job), "SLURM_CPUS_PER_TASK": "1"},
                      str(case["mounts"]), ownership_factory=FakeJobOwnership)


def test_control_fails_private_candidate_passes_and_source_is_unchanged(verification, tmp_path):
    case = verification
    before = (git(case["source"], "show-ref"), (case["source"] / ".git/index").read_bytes(),
              (case["source"] / ".git/config").read_bytes(), git(case["source"], "status", "--porcelain"))
    control = freeze(case, tmp_path, case["control"], "check-control")
    candidate = freeze(case, tmp_path, case["candidate"], "check-candidate")
    assert run(case, control, 234) == 1
    assert run(case, candidate, 235) == 0
    results = [json.loads((Path(r["directory"]) / "verification-result.json").read_text()) for r in (control, candidate)]
    assert [r["command_exit_code"] for r in results] == [1, 0]
    assert [r["commit"] for r in results] == [case["control"], case["candidate"]]
    assert results[0]["project"]["project_root"] != results[1]["project"]["project_root"]
    assert results[1]["coordinator_acceptance"] == "pending"
    assert before == (git(case["source"], "show-ref"), (case["source"] / ".git/index").read_bytes(),
                      (case["source"] / ".git/config").read_bytes(), git(case["source"], "status", "--porcelain"))
    assert git(case["source"], "rev-parse", "HEAD") == case["control"]
    assert case["candidate"] not in git(case["source"], "rev-list", "--all")


def test_queued_profile_script_and_command_edits_do_not_change_request(verification, tmp_path):
    case = verification
    report = freeze(case, tmp_path, case["candidate"], "frozen")
    case["profile"].write_text("{}")
    case["command"].write_text(json.dumps(["/bin/false"]))
    (tmp_path / "bootstrap.sh").write_text("exit 91")
    assert run(case, report, 234) == 0


def test_uncommitted_changes_are_not_used_by_verifier(verification, tmp_path):
    case = verification
    (case["source"] / "answer.py").write_text("OK = True\n")
    report = verify.freeze(case["source"], case["control"], case["profile"], case["command"], tmp_path / "check",
                           lambda p, label: p.resolve(), dispatch._git, local_disk_root=str(case["disk"]))
    assert run(case, report, 234) == 1
    assert (case["source"] / "answer.py").read_text() == "OK = True\n"


def test_tampered_frozen_helper_refuses_before_checkout(verification, tmp_path):
    report = freeze(verification, tmp_path, verification["candidate"], "tampered")
    (Path(report["directory"]) / "project-bootstrap.sh").write_text("exit 0")
    with pytest.raises(ValueError, match="frozen assignment"):
        run(verification, report, 234)
    assert not (verification["disk"] / "job234").exists()


def test_prepared_directory_cannot_be_reused(verification, tmp_path):
    case = verification
    freeze(case, tmp_path, case["candidate"], "one-job")
    with pytest.raises(ValueError, match="already exists"):
        freeze(case, tmp_path, case["candidate"], "one-job")


def test_legacy_run_script_is_frozen_before_job(verification, tmp_path):
    case = verification
    script = tmp_path / "run.sh"
    script.write_text('test -n "$PY"\ntest "$CLONE" = "$PWD"\n"$PY" -c "import answer; assert answer.OK"\n')
    report = verify.freeze(case["results"], case["candidate"], case["profile"], None, tmp_path / "legacy",
                           lambda p, label: p.resolve(), dispatch._git, script_file=script,
                           local_disk_root=str(case["disk"]))
    script.write_text("exit 88")
    assert run(case, report, 234) == 0


def test_exit_zero_without_required_proof_is_failed_verification(verification, tmp_path):
    case = verification
    report = verify.freeze(case["results"], case["candidate"], case["profile"], case["command"], tmp_path / "missing-proof",
                           lambda p, label: p.resolve(), dispatch._git, local_disk_root=str(case["disk"]),
                           required_evidence=["junit.xml"])
    assert run(case, report, 234) == 70
    result = json.loads((Path(report["directory"]) / "verification-result.json").read_text())
    assert result["command_exit_code"] == 0
    assert result["state"] == "failed"
    assert result["evidence"]["state"] == "incomplete"
