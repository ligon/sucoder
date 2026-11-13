"""Dream planning contracts exercised against disposable git repositories."""
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import uuid

import pytest
from typer.testing import CliRunner
import yaml

from sucoder import cli, dream
from sucoder.config import load_config


runner = CliRunner()


def git(repo, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.check_output(["git", "-C", str(repo), *args], env=env).decode().strip()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    repo = tmp_path / "mirrors" / "sample"
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.name", "Fixture")
    git(repo, "config", "user.email", "fixture@example.test")
    (repo / "note.org").write_bytes("* Context\r\nFirst qualification.\r\nSecond: caf\u00e9.\nFinal".encode())
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "fixture")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(dict(human_user="fixture", mirror_root=str(repo.parent),
        log_dir=str(tmp_path / "logs"), mirrors=dict(sample=dict(canonical_repo=str(tmp_path / "canonical"))))))
    config = load_config(config_path)
    request = dict(schema_version=1, repository="sample", integration_ref="refs/heads/main", units=[
        dict(unit_id="note", kind="git", shareable=True,
             locator=dict(repository="sample", commit="HEAD", path="note.org", lines=[2, 3]))])

    def no_startup(*args, **kwargs):
        raise AssertionError("Dream planning entered interactive startup")
    monkeypatch.setattr(cli, "run_startup_checks", no_startup)
    monkeypatch.setattr(cli, "build_default_config", no_startup)
    return repo, config_path, config, request


def invoke(workspace, request=None, *extra):
    repo, config_path, config, default = workspace
    catalog = config_path.parent / "sources.json"
    catalog.write_text(json.dumps(default if request is None else request))
    return runner.invoke(cli.app, ["--config", str(config_path), "dream", "plan", "sample",
        "--sources", str(catalog), "--base", "refs/heads/main", *extra])


def test_plan_freezes_committed_passage_and_preserves_all_git_state(workspace):
    repo, _, config, request = workspace
    original = (repo / "note.org").read_bytes()
    (repo / "note.org").write_text("Human edits that must survive\n")
    git(repo, "add", "note.org")
    (repo / "untracked").write_text("do not touch")
    before = (git(repo, "status", "--porcelain"), git(repo, "show-ref"), (repo / ".git/index").read_bytes())
    result = invoke(workspace)
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    unit = value["units"][0]
    selected = b"First qualification.\r\nSecond: caf\xc3\xa9.\n"
    assert original[unit["byte_start"]:unit["byte_end"]] == selected
    assert unit["sha256"] == hashlib.sha256(selected).hexdigest()
    assert unit["commit"] == git(repo, "rev-parse", "HEAD")
    assert unit["blob"] == git(repo, "rev-parse", "HEAD:note.org")
    assert value["work"]["required_role_invocations"] == 3
    assert value["work"]["repair_reserve"] == 1
    assert value["usage"] == dict(role_invocations=0, tokens=None, currency=None)
    assert value["coverage"][0]["examined_bytes"] == 0
    assert value["state"] == "planned" and value["execution"]["available"] is False
    assert value["candidate_commit"] is None and value["candidate_ref"] is None
    assert value["integration"]["state"] == "not_integrated"
    assert before == (git(repo, "status", "--porcelain"), git(repo, "show-ref"), (repo / ".git/index").read_bytes())
    assert (repo / "note.org").read_text() == "Human edits that must survive\n"
    plan_path = Path(value.pop("plan_path"))
    assert json.loads(plan_path.read_bytes()) == value
    identity = value.pop("plan_id")
    assert dream.digest(dream.canonical(value)) == identity
    assert stat.S_IMODE(plan_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(plan_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(plan_path.parent.parent.stat().st_mode) == 0o700


@pytest.mark.parametrize("kind", ["generic", "unknown"])
def test_unimplemented_evidence_adapters_reject_without_a_backend(workspace, kind):
    request = copy.deepcopy(workspace[3])
    request["units"][0]["kind"] = kind
    result = invoke(workspace, request)
    assert result.exit_code == 2
    assert "Unsupported evidence kind" in result.output
    assert not (workspace[2].log_dir / "dream").exists()


@pytest.mark.parametrize("change", [
    lambda r: r.update(schema_version=2),
    lambda r: r.update(schema_version=True),
    lambda r: r.update(unknown=True),
    lambda r: r.update(repository="another"),
    lambda r: r.update(units=[]),
    lambda r: r["units"].append(copy.deepcopy(r["units"][0])),
    lambda r: r["units"][0].update(shareable="yes"),
    lambda r: r["units"][0].update(extra="ignored?"),
    lambda r: r["units"][0]["locator"].update(extra="ignored?"),
    lambda r: r["units"][0]["locator"].update(lines=[0, 2]),
    lambda r: r["units"][0]["locator"].update(lines=[1, 500]),
    lambda r: r["units"][0]["locator"].update(lines=[True, 2]),
    lambda r: r["units"][0]["locator"].update(lines=[3, 2]),
    lambda r: r.update(limits=dict(max_repairs=2)),
    lambda r: r.update(limits=dict(max_units=True)),
    lambda r: r.update(limits=dict(chunk_bytes=3)),
    lambda r: r.update(limits=dict(unlimited=True)),
    lambda r: r.update(limits=dict(max_source_bytes=4, chunk_bytes=4)),
    lambda r: r.update(limits=dict(max_role_invocations=3)),
    lambda r: r.update(limits=dict(max_consolidation_bytes=128 * 1024)),
])
def test_invalid_requests_fail_before_private_plan_creation(workspace, change):
    request = copy.deepcopy(workspace[3])
    change(request)
    result = invoke(workspace, request)
    assert result.exit_code == 2, result.output
    assert json.loads(result.output)["state"] == "rejected"
    assert not (workspace[2].log_dir / "dream").exists()


@pytest.mark.parametrize("path", ["../note.org", "/note.org", "a/../note.org", "./note.org", "a//b", "a\\b", "a\nb"])
def test_traversal_and_noncanonical_paths_reject(workspace, path):
    request = copy.deepcopy(workspace[3])
    request["units"][0]["locator"]["path"] = path
    result = invoke(workspace, request)
    assert result.exit_code == 2


def test_git_pathspecs_are_literal(workspace):
    repo, _, _, request = workspace
    (repo / ":(glob)*").write_text("Literal source\n")
    git(repo, "add", "--", ":(literal):(glob)*")
    git(repo, "commit", "-qm", "literal path")
    request["units"][0]["locator"].update(path=":(glob)*", lines=[1, 1])
    result = invoke(workspace)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["units"][0]["bytes"] == len(b"Literal source\n")


@pytest.mark.parametrize("path", ["link", "directory"])
def test_symlinks_and_directories_are_not_admitted(workspace, path):
    repo, _, _, request = workspace
    (repo / "link").symlink_to("note.org")
    (repo / "directory").mkdir()
    (repo / "directory/child").write_text("text\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "non regular sources")
    request["units"][0]["locator"].update(path=path)
    result = invoke(workspace)
    assert result.exit_code == 2, result.output


@pytest.mark.parametrize("content", [b"\xff\n", b"\x00\n", b""])
def test_invalid_or_empty_text_is_explicit(workspace, content):
    repo, _, _, request = workspace
    (repo / "note.org").write_bytes(content)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "invalid source")
    request["units"][0]["locator"].pop("lines")
    assert invoke(workspace).exit_code == 2


def test_chunk_coverage_preserves_utf8_bytes_and_final_line(workspace):
    repo, _, _, request = workspace
    content = "\u00e9" * 5 + "\nA\n" + "\U0001f600" * 30 + "last"
    (repo / "note.org").write_bytes(content.encode())
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "chunked source")
    request["units"][0]["locator"].pop("lines")
    request["limits"] = dict(chunk_bytes=9)
    result = invoke(workspace)
    assert result.exit_code == 0, result.output
    unit = json.loads(result.output)["units"][0]
    position = 0
    for chunk in unit["chunks"]:
        assert chunk["byte_start"] == position
        raw = content.encode()[position:chunk["byte_end"]]
        raw.decode("utf-8")
        assert 0 < len(raw) <= 9
        assert dream.digest(raw) == chunk["sha256"]
        position = chunk["byte_end"]
    assert position == len(content.encode())


def add_registry(repo, identity):
    path = repo / ".sucoder/wiki/sources.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(schema_version=1, repository_uuid=identity)))
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "accepted registry")


def test_accepted_uuid_survives_revisions_and_existing_plans_are_idempotent(workspace):
    repo, _, _, request = workspace
    identity = str(uuid.uuid4())
    add_registry(repo, identity)
    first = json.loads(invoke(workspace).output)
    second = json.loads(invoke(workspace).output)
    assert first == second
    assert first["repositories"]["sample"]["provisional"] is False
    (repo / "note.org").write_text("different\nsource\nrevision\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "new revision")
    third = json.loads(invoke(workspace).output)
    assert first["units"][0]["source_id"] == third["units"][0]["source_id"]
    assert first["plan_id"] != third["plan_id"]
    assert first["units"][0]["sha256"] != third["units"][0]["sha256"]
    request["limits"] = dict(max_elapsed_seconds=901)
    assert json.loads(invoke(workspace).output)["plan_id"] != third["plan_id"]


def test_prior_wiki_freezes_active_pages_and_reports_omitted_archives(workspace):
    repo, _, _, request = workspace
    wiki = repo / ".sucoder/wiki"
    (wiki / "archive").mkdir(parents=True)
    (wiki / "index.org").write_text("* Active\n** Talk :editorial:\n*** WAITING Human ruling\n")
    (wiki / "archive/old.org").write_text("* DONE Old task\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "prior wiki")
    result = invoke(workspace)
    assert result.exit_code == 0, result.output
    prior = json.loads(result.output)["prior_wiki"]
    assert prior["progress"] == "not_evaluated"
    assert [f["path"] for f in prior["selected"]] == [".sucoder/wiki/index.org"]
    assert prior["omitted"] == [dict(path=".sucoder/wiki/archive/old.org", reason="archive_or_metadata")]
    request["limits"] = dict(max_prior_bytes=8)
    result = invoke(workspace)
    assert result.exit_code != 0
    assert "byte allowance" in result.output or "max_prior_bytes" in result.output


def test_maintenance_requires_reason_topic_and_sources(workspace):
    assert invoke(workspace, None, "--maintenance", "repair").exit_code == 2
    assert invoke(workspace, None, "--topic", "reply").exit_code == 2
    result = invoke(workspace, None, "--maintenance", "reader missed a qualification", "--topic", "reply")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["topic"] == "reply"


def test_missing_sources_are_unavailable_not_empty(workspace):
    request = copy.deepcopy(workspace[3])
    request["units"][0]["locator"]["path"] = "absent"
    result = invoke(workspace, request)
    assert result.exit_code == 3
    assert json.loads(result.output)["state"] == "rejected"


def test_duplicate_json_fields_reject(workspace):
    repo, config_path, config, request = workspace
    path = config_path.parent / "sources.json"
    path.write_text(json.dumps(request).replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'))
    with pytest.raises(dream.DreamError, match="Duplicate JSON field"):
        dream.create_plan(config, "sample", path, "refs/heads/main")


def test_logs_inside_any_source_repository_are_rejected(workspace):
    repo, config_path, config, request = workspace
    config.log_dir = repo / "logs"
    path = config_path.parent / "sources.json"
    path.write_text(json.dumps(request))
    with pytest.raises(dream.DreamError, match="outside every inspected repository"):
        dream.create_plan(config, "sample", path, "refs/heads/main")
    assert not (repo / "logs").exists()


@pytest.mark.parametrize("mode", ["shared", "symlink"])
def test_existing_unsafe_private_root_is_not_repaired(workspace, mode):
    repo, _, config, _ = workspace
    config.log_dir.mkdir()
    root = config.log_dir / "dream"
    if mode == "shared":
        root.mkdir()
        root.chmod(0o770)
    else:
        target = config.log_dir / "target"
        target.mkdir()
        root.symlink_to(target, target_is_directory=True)
    result = invoke(workspace)
    assert result.exit_code != 0
    if mode == "shared":
        assert stat.S_IMODE(root.stat().st_mode) == 0o770
    else:
        assert root.is_symlink() and not list(target.iterdir())


def test_ref_change_during_planning_requires_replan(workspace, monkeypatch):
    repo = workspace[0]
    first = git(repo, "rev-parse", "HEAD")
    (repo / "extra").write_text("new commit")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "extra")
    second = git(repo, "rev-parse", "HEAD")
    git(repo, "update-ref", "refs/heads/main", first)
    original = dream.GitSource.check_binding
    calls = []

    def change_ref(source):
        original(source)
        calls.append(source)
        if len(calls) == 2:
            git(repo, "update-ref", "refs/heads/main", second)
    monkeypatch.setattr(dream.GitSource, "check_binding", change_ref)
    result = invoke(workspace)
    assert result.exit_code == 3
    assert "Integration ref changed" in result.output
    assert not (workspace[2].log_dir / "dream").exists()


def test_dream_candidate_and_symbolic_integration_refs_reject(workspace):
    repo, config_path, config, request = workspace
    for base in ("HEAD", "main", "refs/heads/dream/candidate", "refs/remotes/upstream/HEAD"):
        request["integration_ref"] = base
        path = config_path.parent / "sources.json"
        path.write_text(json.dumps(request))
        if base == "refs/remotes/upstream/HEAD":
            git(repo, "symbolic-ref", base, "refs/heads/main")
        with pytest.raises(dream.DreamError):
            dream.create_plan(config, "sample", path, base)


def test_help_needs_no_configuration_and_rejects_remote_startup(workspace, monkeypatch):
    def no_config(*args, **kwargs):
        raise AssertionError("help read configuration")
    monkeypatch.setattr("sucoder.dream_cli.load_config", no_config)
    assert runner.invoke(cli.app, ["dream", "--help"]).exit_code == 0
    assert runner.invoke(cli.app, ["dream", "plan", "--help"]).exit_code == 0
    result = runner.invoke(cli.app, ["--target", "cluster", "dream", "--help"])
    assert result.exit_code == 2


def test_small_chunks_reject_role_overflow_before_allocating_a_manifest(workspace):
    repo, _, _, request = workspace
    (repo / "note.org").write_bytes(b"x" * 100000)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "long source")
    request["units"][0]["locator"].pop("lines")
    request["limits"] = dict(chunk_bytes=4, max_role_invocations=4)
    result = invoke(workspace)
    assert result.exit_code == 2
    assert "max_role_invocations" in result.output
    assert not (workspace[2].log_dir / "dream").exists()


def test_interrupted_plan_publication_can_be_retried(workspace, monkeypatch):
    repo, _, config, _ = workspace
    add_registry(repo, str(uuid.uuid4()))
    original = dream.os.link

    def interrupt(*args, **kwargs):
        raise OSError("simulated interruption before atomic publication")
    monkeypatch.setattr(dream.os, "link", interrupt)
    first = invoke(workspace)
    assert first.exit_code == 3
    assert not list((config.log_dir / "dream").rglob("plan.json"))
    monkeypatch.setattr(dream.os, "link", original)
    second = invoke(workspace)
    assert second.exit_code == 0, second.output
    assert len(list((config.log_dir / "dream").rglob("plan.json"))) == 1


def test_tampered_existing_plan_is_not_overwritten(workspace):
    repo, _, _, _ = workspace
    add_registry(repo, str(uuid.uuid4()))
    first = json.loads(invoke(workspace).output)
    path = Path(first["plan_path"])
    path.write_text("human inspection note; preserve this")
    second = invoke(workspace)
    assert second.exit_code == 2
    assert path.read_text() == "human inspection note; preserve this"


def test_catalog_fifo_does_not_block(workspace):
    repo, config_path, config, _ = workspace
    fifo = config_path.parent / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(dream.DreamError, match="regular JSON file"):
        dream.create_plan(config, "sample", fifo, "refs/heads/main")


def test_planning_does_not_leak_umask_or_change_parent_permissions(workspace):
    config = workspace[2]
    config.log_dir.mkdir(mode=0o770)
    config.log_dir.chmod(0o770)
    initial = os.umask(0o027)
    try:
        result = invoke(workspace)
        assert result.exit_code == 0, result.output
        current = os.umask(0o027)
        assert current == 0o027
        assert stat.S_IMODE(config.log_dir.stat().st_mode) == 0o770
    finally:
        os.umask(initial)


def test_inherited_git_redirection_does_not_change_evidence(workspace, monkeypatch, tmp_path):
    repo = workspace[0]
    other = tmp_path / "other"
    other.mkdir()
    git(other, "init", "-q")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    result = invoke(workspace)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["base_commit"] == git(repo, "rev-parse", "HEAD")


def test_configured_local_source_alias_has_separate_provenance(workspace):
    repo, config_path, _, request = workspace
    other = repo.parent / "other"
    other.mkdir()
    git(other, "init", "-q", "-b", "main")
    git(other, "config", "user.name", "Fixture")
    git(other, "config", "user.email", "fixture@example.test")
    (other / "note.org").write_text("Other source\n")
    git(other, "add", ".")
    git(other, "commit", "-qm", "other source")
    raw = yaml.safe_load(config_path.read_text())
    raw["mirrors"]["other"] = dict(canonical_repo=str(config_path.parent / "other-canonical"))
    config_path.write_text(yaml.safe_dump(raw))
    request["units"].append(dict(unit_id="other", kind="git", shareable=False,
        locator=dict(repository="other", commit="HEAD", path="note.org")))
    result = invoke(workspace)
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert set(value["repositories"]) == {"sample", "other"}
    assert value["units"][1]["shareable"] is False
    assert value["units"][0]["source_id"] != value["units"][1]["source_id"]


def test_unreadable_configuration_is_a_json_failure(workspace, monkeypatch):
    def refuse(*args, **kwargs):
        raise PermissionError("test configuration is unreadable")
    monkeypatch.setattr("sucoder.dream_cli.load_config", refuse)
    result = invoke(workspace)
    assert result.exit_code == 3
    assert json.loads(result.output)["state"] == "rejected"
