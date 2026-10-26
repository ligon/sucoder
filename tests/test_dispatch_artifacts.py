import hashlib
import json
import os

from sucoder.dispatch_artifacts import publish_evidence, read_task_outcome


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o600)


def test_evidence_hashes_private_selected_proofs_but_not_git_or_data(tmp_path):
    _write(tmp_path / "evidence" / "junit.xml", "<testsuite/>")
    _write(tmp_path / "ignored-data.parquet", "not selected")
    result = publish_evidence(str(tmp_path), required=["junit.xml"])
    assert result["state"] == "complete"
    assert result["files"] == [{"name": "junit.xml", "size": 12,
        "sha256": hashlib.sha256(b"<testsuite/>").hexdigest()}]
    assert json.loads((tmp_path / "evidence-manifest.json").read_text()) == result
    assert (tmp_path / "evidence-manifest.json").stat().st_mode & 0o077 == 0


def test_required_missing_evidence_is_incomplete(tmp_path):
    result = publish_evidence(str(tmp_path), ["control.log", "candidate.xml"])
    assert result["state"] == "incomplete"
    assert result["missing"] == ["candidate.xml", "control.log"]


def test_symlinks_database_files_and_public_files_are_not_evidence(tmp_path):
    _write(tmp_path / "secret", "do not follow")
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "link").symlink_to(tmp_path / "secret")
    (root / "dir-link").symlink_to(tmp_path, target_is_directory=True)
    _write(root / "state.sqlite-wal", "not copied")
    _write(root / "public.txt", "wrong permissions")
    (root / "public.txt").chmod(0o644)
    result = publish_evidence(str(tmp_path))
    assert result["state"] == "incomplete"
    assert result["count"] == 0
    assert len(result["errors"]) == 4


def test_evidence_size_limit_is_explicit(tmp_path):
    _write(tmp_path / "evidence" / "big.txt", "12345")
    result = publish_evidence(str(tmp_path), max_file_bytes=4)
    assert result["state"] == "incomplete"
    assert "size limit" in result["errors"][0]


def test_absent_task_report_is_not_inferred_from_agent_exit(tmp_path):
    assert read_task_outcome(str(tmp_path))["status"] == "not_reported"


def test_task_claim_never_sets_coordinator_acceptance(tmp_path):
    _write(tmp_path / "task-result.json", json.dumps({"status": "completed", "summary": "worker claim",
        "coordinator_acceptance": "accepted", "unexecuted": ["S3 tests"]}))
    result = read_task_outcome(str(tmp_path))
    assert result["status"] == "completed"
    assert result["unexecuted"] == ["S3 tests"]
    assert result["coordinator_acceptance"] == "pending"


def test_malformed_or_unsafe_task_report_is_explicit(tmp_path):
    _write(tmp_path / "task-result.json", "not json")
    assert read_task_outcome(str(tmp_path))["status"] == "unreadable"
    (tmp_path / "task-result.json").unlink()
    (tmp_path / "task-result.json").symlink_to(tmp_path / "other")
    assert read_task_outcome(str(tmp_path))["status"] == "unreadable"
