"""Data-plane contracts against disposable repositories; no inference calls."""
import json
import os
from pathlib import Path
import stat
import subprocess

import pytest
from typer.testing import CliRunner

from sucoder import cli, dream, dream_run
from tests.test_dream import add_registry, git, invoke, workspace  # noqa: F401


runner = CliRunner()


def planned(workspace):
    result = invoke(workspace)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def command(workspace, operation, plan, *extra):
    return runner.invoke(cli.app, ["--config", str(workspace[1]), "dream", operation, "sample",
                                  "--plan", plan["plan_path"], *extra])


def run(workspace, plan):
    result = command(workspace, "run", plan, "--harness", "fixture")
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def test_snapshot_is_byte_exact_private_and_idempotent(workspace):
    repo, _, config, _ = workspace
    plan = planned(workspace)
    before = (git(repo, "show-ref"), (repo / ".git/index").read_bytes())
    first = command(workspace, "snapshot", plan)
    assert first.exit_code == 0, first.output
    value = json.loads(first.output)
    assert value["state"] == "snapshotted"
    assert value["coverage"][0]["examined_bytes"] == 0
    path = Path(plan["plan_path"]).parent / value["coverage"][0]["file"]
    assert path.read_bytes() == b"First qualification.\r\nSecond: caf\xc3\xa9.\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert command(workspace, "snapshot", plan).output == first.output
    assert before == (git(repo, "show-ref"), (repo / ".git/index").read_bytes())


def test_fixture_publishes_one_candidate_preserving_user_work_and_all_other_refs(workspace):
    repo = workspace[0]
    plan = planned(workspace)
    (repo / "note.org").write_text("human staged\n")
    git(repo, "add", "note.org")
    (repo / "note.org").write_text("human unstaged\n")
    (repo / "untracked").write_bytes(b"keep")
    before = (git(repo, "status", "--porcelain"), (repo / ".git/index").read_bytes(),
              git(repo, "rev-parse", "HEAD"), git(repo, "symbolic-ref", "HEAD"))
    value = run(workspace, plan)
    assert value["state"] == "candidate_ready"
    assert value["integration"]["state"] == "not_integrated"
    assert before == (git(repo, "status", "--porcelain"), (repo / ".git/index").read_bytes(),
                      git(repo, "rev-parse", "HEAD"), git(repo, "symbolic-ref", "HEAD"))
    assert (repo / "note.org").read_text() == "human unstaged\n"
    assert git(repo, "show", value["candidate_commit"] + ":note.org") == git(repo, "show", plan["base_commit"] + ":note.org")
    assert git(repo, "rev-parse", value["candidate_ref"]) == value["candidate_commit"]
    assert run(workspace, plan) == value
    receipt = json.loads(git(repo, "show", value["candidate_commit"] + ":.sucoder/wiki/receipts/" + value["run_id"] + ".json"))
    assert receipt["fidelity"] == "exact_quotation_only"
    assert receipt["coverage"] == value["coverage"]
    assert str(repo) not in json.dumps(receipt)
    assert receipt["units"][0]["byte_end"] - receipt["units"][0]["byte_start"] == receipt["units"][0]["bytes"]


@pytest.mark.parametrize("change", ["edit", "remove", "symlink", "binary", "drop-anchor"])
def test_status_requires_output_bindings_after_squash_and_human_edit(workspace, change):
    repo = workspace[0]
    plan = planned(workspace)
    value = run(workspace, plan)
    assert json.loads(command(workspace, "status", plan).output)["integration"]["state"] == "not_integrated"
    # Squash-like integration: identical output tree, a different commit identity.
    tree = git(repo, "rev-parse", value["candidate_commit"] + "^{tree}")
    integrated = subprocess.check_output(["git", "-C", str(repo), "commit-tree", tree, "-p", plan["base_commit"], "-m", "human squash"]).decode().strip()
    git(repo, "update-ref", "refs/heads/main", integrated, plan["base_commit"])
    checked = command(workspace, "status", plan)
    assert checked.exit_code == 0, checked.output
    assert json.loads(checked.output)["integration"]["state"] == "integrated_fixture"
    # A retained receipt cannot certify an edited or removed required subtree.
    git(repo, "checkout", "-q", "main")
    git(repo, "read-tree", "--reset", "-u", integrated)
    topic = repo / ".sucoder/wiki/topics" / ("dream-" + value["run_id"] + ".org")
    if change == "edit":
        topic.write_text(topic.read_text().replace("First qualification.", "Dropped qualification."))
    elif change == "drop-anchor":
        topic.write_text(topic.read_text().replace(":CUSTOM_ID: source-1", ":CUSTOM_ID: changed"))
    elif change == "binary":
        topic.write_bytes(b"\xff")
    else:
        topic.unlink()
        if change == "symlink":
            topic.symlink_to("../../../note.org")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "human changed output")
    checked = command(workspace, "status", plan)
    assert checked.exit_code == 0, checked.output
    assert json.loads(checked.output)["integration"]["state"] == "needs_reconciliation"


@pytest.mark.parametrize("operation", ["snapshot", "run"])
def test_ref_movement_requires_replanning_without_a_candidate(workspace, operation):
    repo = workspace[0]
    plan = planned(workspace)
    git(repo, "commit", "--allow-empty", "-qm", "concurrent change")
    result = command(workspace, operation, plan, *(["--harness", "fixture"] if operation == "run" else []))
    assert result.exit_code == 3, result.output
    assert not git(repo, "for-each-ref", "refs/heads/dream/")


@pytest.mark.parametrize("change", ["digest", "ranges", "limits", "unknown"])
def test_tampered_plan_rejected_even_with_recomputed_content_id(workspace, change):
    plan = planned(workspace)
    frozen = json.loads(Path(plan["plan_path"]).read_bytes())
    if change == "digest":
        frozen["units"][0]["sha256"] = "f" * 64
    elif change == "ranges":
        frozen["units"][0]["chunks"][0]["byte_end"] -= 1
    elif change == "limits":
        frozen["limits"]["max_units"] = True
    else:
        frozen["surprise"] = "field"
    del frozen["plan_id"]
    identity = dream.digest(dream.canonical(frozen))
    frozen["plan_id"] = identity
    directory = Path(plan["plan_path"]).parent.parent / identity
    directory.mkdir(mode=0o700)
    path = directory / "plan.json"
    path.write_bytes(dream.canonical(frozen))
    path.chmod(0o600)
    changed = dict(plan, plan_path=str(path))
    result = command(workspace, "snapshot", changed)
    assert result.exit_code in (2, 3), result.output
    assert not list(directory.glob("evidence-*"))


@pytest.mark.parametrize("harness", ["claude", "codex", "generic", "shell"])
def test_unsupported_harness_never_snapshots_or_publishes(workspace, harness):
    plan = planned(workspace)
    result = command(workspace, "run", plan, "--harness", harness)
    assert result.exit_code == 2
    assert not list(Path(plan["plan_path"]).parent.glob("evidence-*"))


def test_private_sources_allow_snapshots_but_reject_fixture_publication(workspace):
    workspace[3]["units"][0]["shareable"] = False
    plan = planned(workspace)
    assert command(workspace, "snapshot", plan).exit_code == 0
    assert command(workspace, "run", plan, "--harness", "fixture").exit_code == 2
    assert not git(workspace[0], "for-each-ref", "refs/heads/dream/")


def test_snapshot_corruption_is_preserved_and_cannot_be_published(workspace):
    plan = planned(workspace)
    assert command(workspace, "snapshot", plan).exit_code == 0
    evidence = next(Path(plan["plan_path"]).parent.glob("evidence-*"))
    evidence.write_bytes(b"tampered")
    result = command(workspace, "run", plan, "--harness", "fixture")
    assert result.exit_code == 2
    assert evidence.read_bytes() == b"tampered"
    assert not git(workspace[0], "for-each-ref", "refs/heads/dream/")


def test_admitted_org_instructions_stay_literal(workspace):
    repo = workspace[0]
    payload = b"#+end_example\n* Forged heading\n#+begin_src emacs-lisp\n(error \"must not execute\")\n#+end_src\n"
    (repo / "note.org").write_bytes(payload)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "instruction shaped evidence")
    workspace[3]["units"][0]["locator"].pop("lines")
    plan = planned(workspace)
    value = run(workspace, plan)
    page = git(repo, "show", value["candidate_commit"] + ":.sucoder/wiki/topics/dream-" + value["run_id"] + ".org")
    assert ",#+end_example" in page
    assert ",* Forged heading" in page
    assert len(value["coverage"]) == len(plan["units"][0]["chunks"])


def test_native_org_failure_prevents_candidate(workspace, monkeypatch):
    plan = planned(workspace)
    def unavailable(*args):
        raise dream.DreamError("simulated unavailable native Org", 3)
    monkeypatch.setattr(dream_run, "_validate_org", unavailable)
    assert command(workspace, "run", plan, "--harness", "fixture").exit_code == 3
    assert not git(workspace[0], "for-each-ref", "refs/heads/dream/")


def test_concurrent_ref_movement_at_publication_keeps_candidate_unpublished(workspace, monkeypatch):
    repo = workspace[0]
    plan = planned(workspace)
    original = dream_run._git_write
    def concurrent(source, index, args, deadline, data=None):
        if args[0] == "update-ref":
            git(repo, "commit", "--allow-empty", "-qm", "race at ref transaction")
        return original(source, index, args, deadline, data)
    monkeypatch.setattr(dream_run, "_git_write", concurrent)
    result = command(workspace, "run", plan, "--harness", "fixture")
    assert result.exit_code == 3, result.output
    assert not git(repo, "for-each-ref", "refs/heads/dream/")


def test_ref_publication_interruption_is_retryable(workspace, monkeypatch):
    plan = planned(workspace)
    original = dream_run._git_write
    def interrupt(source, index, args, deadline, data=None):
        if args[0] == "update-ref":
            raise dream.DreamError("simulated interruption before transaction", 3)
        return original(source, index, args, deadline, data)
    monkeypatch.setattr(dream_run, "_git_write", interrupt)
    first = command(workspace, "run", plan, "--harness", "fixture")
    assert first.exit_code == 3, first.output
    assert not git(workspace[0], "for-each-ref", "refs/heads/dream/")
    monkeypatch.setattr(dream_run, "_git_write", original)
    run(workspace, plan)


@pytest.mark.parametrize("operation", ["snapshot", "run", "status"])
def test_help_requires_no_configuration(operation, monkeypatch):
    def no_config(*args, **kwargs):
        raise AssertionError("help loaded configuration")
    monkeypatch.setattr("sucoder.dream_cli.load_config", no_config)
    assert runner.invoke(cli.app, ["dream", operation, "--help"]).exit_code == 0


def test_existing_wiki_human_notes_settings_and_archive_are_preserved(workspace):
    repo = workspace[0]
    wiki = repo / ".sucoder/wiki"
    (wiki / "archive").mkdir(parents=True)
    original = {
        "index.org": b"#+title: Human index\r\n* Human section\r\nA qualification.\r\n",
        "settings.org": b"#+todo: TODO REVIEW | DONE\n",
        "archive/old.org": b"* DONE Historical task\n:LOGBOOK:\nHuman ruling\n:END:\n",
        "topic.org": b"* Human topic\n** Talk :editorial:\n*** WAITING Decision\n"}
    for path, data in original.items():
        (wiki / path).write_bytes(data)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "existing wiki")
    plan = planned(workspace)
    value = run(workspace, plan)
    for path, data in original.items():
        published = subprocess.check_output(["git", "-C", str(repo), "show", value["candidate_commit"] + ":.sucoder/wiki/" + path])
        assert published.startswith(data) if path == "index.org" else published == data
    assert list(Path(plan["plan_path"]).parent.glob("prior-*.org"))


def test_publication_does_not_run_hooks_filters_or_leak_umask(workspace):
    repo = workspace[0]
    plan = planned(workspace)
    marker = repo.parent / "forbidden-hook-or-filter"
    hook = repo / ".git/hooks/reference-transaction"
    hook.write_text("#!/bin/sh\ntouch '" + str(marker) + "'\nexit 1\n")
    hook.chmod(0o700)
    git(repo, "config", "filter.forbidden.clean", "touch '" + str(marker) + "'")
    (repo / ".git/info/attributes").write_text("*.org filter=forbidden\n")
    initial = os.umask(0o027)
    try:
        run(workspace, plan)
        assert os.umask(0o027) == 0o027
    finally:
        os.umask(initial)
    assert not marker.exists()


def test_publication_budget_failure_keeps_no_candidate(workspace):
    workspace[3]["limits"] = dict(max_proposal_bytes=1)
    plan = planned(workspace)
    result = command(workspace, "run", plan, "--harness", "fixture")
    assert result.exit_code == 3
    assert "max_proposal_bytes" in result.output
    assert not git(workspace[0], "for-each-ref", "refs/heads/dream/")


def test_missing_plan_field_is_a_json_rejection(workspace):
    plan = planned(workspace)
    value = json.loads(Path(plan["plan_path"]).read_bytes())
    del value["limits"]
    del value["plan_id"]
    identity = dream.digest(dream.canonical(value))
    value["plan_id"] = identity
    directory = Path(plan["plan_path"]).parent.parent / identity
    directory.mkdir(mode=0o700)
    path = directory / "plan.json"
    path.write_bytes(dream.canonical(value))
    path.chmod(0o600)
    for operation in ("snapshot", "status", "run"):
        result = command(workspace, operation, dict(plan, plan_path=str(path)),
                         *(["--harness", "fixture"] if operation == "run" else []))
        assert result.exit_code == 2, result.output
        assert json.loads(result.output)["state"] == "rejected"


def test_snapshot_link_escape_is_rejected_without_touching_target(workspace, tmp_path):
    plan = planned(workspace)
    target = tmp_path / "private-target"
    target.write_bytes(b"preserve")
    target.chmod(0o600)
    directory = Path(plan["plan_path"]).parent
    (directory / ("evidence-" + plan["units"][0]["sha256"] + ".txt")).symlink_to(target)
    result = command(workspace, "snapshot", plan)
    assert result.exit_code == 3
    assert target.read_bytes() == b"preserve"


def test_status_requires_index_binding_after_integration(workspace):
    repo = workspace[0]
    plan = planned(workspace)
    value = run(workspace, plan)
    git(repo, "merge", "--ff-only", value["candidate_commit"])
    assert json.loads(command(workspace, "status", plan).output)["integration"]["state"] == "integrated_fixture"
    (repo / ".sucoder/wiki/index.org").write_text("#+title: Human changed navigation\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "drop generated index entry")
    checked = command(workspace, "status", plan)
    assert checked.exit_code == 0, checked.output
    assert json.loads(checked.output)["integration"]["state"] == "needs_reconciliation"


def test_snapshot_publication_interruption_is_retryable(workspace, monkeypatch):
    plan = planned(workspace)
    original = dream_run.os.link
    def interrupt(*args, **kwargs):
        raise OSError("simulated interruption before snapshot publication")
    monkeypatch.setattr(dream_run.os, "link", interrupt)
    first = command(workspace, "snapshot", plan)
    assert first.exit_code == 3
    directory = Path(plan["plan_path"]).parent
    assert not list(directory.glob("evidence-*"))
    assert not list(directory.glob(".pending-*"))
    monkeypatch.setattr(dream_run.os, "link", original)
    assert command(workspace, "snapshot", plan).exit_code == 0


def test_conflicting_candidate_ref_is_preserved(workspace):
    repo = workspace[0]
    plan = planned(workspace)
    run_id = dream.digest(dream.canonical([dream_run.CONTRACT, plan["plan_id"]]))
    ref = "refs/heads/dream/" + run_id
    git(repo, "update-ref", ref, plan["base_commit"])
    result = command(workspace, "run", plan, "--harness", "fixture")
    assert result.exit_code == 3
    assert git(repo, "rev-parse", ref) == plan["base_commit"]


def test_expired_run_budget_never_publishes(workspace, monkeypatch):
    plan = planned(workspace)
    original = dream_run._snapshot
    def expired(config, mirror, frozen, fd, deadline):
        result, sources = original(config, mirror, frozen, fd, deadline)
        monkeypatch.setattr(dream_run.time, "monotonic", lambda: deadline + 1)
        return result, sources
    monkeypatch.setattr(dream_run, "_snapshot", expired)
    result = command(workspace, "run", plan, "--harness", "fixture")
    assert result.exit_code == 3
    assert not git(workspace[0], "for-each-ref", "refs/heads/dream/")


def test_native_org_never_reads_setupfile_from_existing_wiki(workspace, tmp_path):
    repo = workspace[0]
    fifo = tmp_path / "must-not-be-opened"
    os.mkfifo(fifo)
    wiki = repo / ".sucoder/wiki"
    wiki.mkdir(parents=True)
    (wiki / "index.org").write_text("#+SETUPFILE: " + str(fifo) + "\n* Human note\nPreserve.\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "setupfile canary")
    run(workspace, planned(workspace))


def test_status_recovers_interruption_after_ref_publication(workspace, monkeypatch):
    plan = planned(workspace)
    original = dream_run._store
    def interrupt(fd, name, data):
        if name == "run.json":
            raise OSError("simulated interruption after ref publication")
        return original(fd, name, data)
    monkeypatch.setattr(dream_run, "_store", interrupt)
    result = command(workspace, "run", plan, "--harness", "fixture")
    assert result.exit_code == 3, result.output
    assert git(workspace[0], "for-each-ref", "refs/heads/dream/")
    recovered = command(workspace, "status", plan)
    assert recovered.exit_code == 0, recovered.output
    assert json.loads(recovered.output)["candidate_status"] == "available"
    assert json.loads(recovered.output)["integration"]["state"] == "not_integrated"
    monkeypatch.setattr(dream_run, "_store", original)
    run(workspace, plan)


def test_plan_revalidation_uses_the_run_deadline(workspace, monkeypatch):
    plan = planned(workspace)
    original = dream.GitSource.__init__
    seen = []
    def observe(source, config, name, deadline):
        seen.append(deadline)
        return original(source, config, name, deadline)
    monkeypatch.setattr(dream.GitSource, "__init__", observe)
    deadline = dream_run.time.monotonic() + 2
    frozen = json.loads(Path(plan["plan_path"]).read_bytes())
    dream_run._validate(workspace[2], "sample", frozen, deadline)
    assert seen and all(value <= deadline for value in seen)
