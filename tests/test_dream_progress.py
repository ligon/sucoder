"""Only accepted, compatible and output-bound shared receipts save inference."""
import json
from pathlib import Path
import time

import pytest

from sucoder import dream, dream_progress as progress, dream_run, dream_wiki as wiki
from tests.test_dream import git, invoke, workspace  # noqa: F401
from tests.test_dream_wiki import EditorTransport, TOPIC, planned


class NoCalls:
    version = EditorTransport.version

    def __call__(self, *args):
        raise AssertionError("Compatible accepted work must not invoke inference")


def replan(workspace):
    request = dict(workspace[3], units=[dict(unit_id=name, kind="git", shareable=True,
        locator=dict(repository="sample", commit="HEAD", path=name + ".org")) for name in ("evidence-a", "evidence-b")])
    result = invoke(workspace, request)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def accepted(workspace, harness="codex"):
    first = planned(workspace)
    run = wiki.run(workspace[2], "sample", Path(first["plan_path"]), harness, "explicit-model", transport=EditorTransport())
    git(workspace[0], "merge", "--ff-only", run["candidate_commit"])
    return replan(workspace), run


def discovery(workspace, envelope, *, model="explicit-model", version=NoCalls.version):
    deadline = time.monotonic() + 30
    with dream_run._directory(workspace[2], Path(envelope["plan_path"])) as (fd, directory):
        plan = dream_run._load(fd, directory.name, "sample")
        _, sources = dream_run._snapshot(workspace[2], "sample", plan, fd, deadline)
        return progress.discover(plan, fd, sources["sample"], "codex", model, version, wiki.PURPOSE, deadline)


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_accepted_work_is_reused_without_model_calls_or_duplicate_topic(workspace, harness):
    plan, original = accepted(workspace, harness)
    repo = workspace[0]
    before = (git(repo, "show-ref"), git(repo, "status", "--porcelain"), (repo / ".git/index").read_bytes())
    result = wiki.run(workspace[2], "sample", Path(plan["plan_path"]), harness, "explicit-model", transport=NoCalls())
    assert result["state"] == "already_integrated"
    assert result["usage"]["role_invocations"] == 0
    assert result["candidate_commit"] is result["candidate_ref"] is None
    assert result["receipts"][0]["run_id"] == original["run_id"]
    assert all(c["extraction"] == "accepted_receipt_reused" for c in result["coverage"])
    assert not list(Path(plan["plan_path"]).parent.glob("*-assignment.json"))
    assert before == (git(repo, "show-ref"), git(repo, "status", "--porcelain"), (repo / ".git/index").read_bytes())
    status = dream_run.status(workspace[2], "sample", Path(plan["plan_path"]))
    assert status["state"] == "already_integrated"
    assert status["candidate_status"] == "not_created"
    assert wiki.run(workspace[2], "sample", Path(plan["plan_path"]), harness, "explicit-model", transport=NoCalls()) == result


def test_reused_progress_reports_reconciliation_after_required_output_edit(workspace):
    plan, _ = accepted(workspace)
    wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "explicit-model", transport=NoCalls())
    repo = workspace[0]
    page = repo / TOPIC
    page.write_text(page.read_text().replace("#+title: Reply evidence", "#+title: Human changed orientation"))
    git(repo, "add", TOPIC)
    git(repo, "commit", "-qm", "Human edits outside bound subtrees")
    assert dream_run.status(workspace[2], "sample", Path(plan["plan_path"]))["state"] == "needs_reconciliation"
    current = replan(workspace)
    with pytest.raises(dream.DreamError, match="need reconciliation"):
        wiki.run(workspace[2], "sample", Path(current["plan_path"]), "codex", "explicit-model", transport=NoCalls())


@pytest.mark.parametrize("defect", ["coverage", "bindings", "review"])
def test_receipt_presence_without_complete_valid_proof_is_not_coverage(workspace, defect):
    _, run = accepted(workspace)
    repo = workspace[0]
    path = ".sucoder/wiki/receipts/" + run["run_id"] + ".json"
    receipt = json.loads((repo / path).read_text())
    if defect == "coverage": receipt["coverage"] = receipt["coverage"][:-1]
    if defect == "bindings": receipt["outputs"] = []
    if defect == "review": receipt["review"] = None
    (repo / path).write_bytes(dream.canonical(receipt) + b"\n")
    git(repo, "add", path)
    git(repo, "commit", "-qm", "Replace shared receipt metadata")
    assert discovery(workspace, replan(workspace)) is None


def test_model_or_native_build_change_is_not_compatible_progress(workspace):
    plan, _ = accepted(workspace)
    assert discovery(workspace, plan, model="another-model") is None
    assert discovery(workspace, plan, version="another-native-build") is None


def test_changed_source_dependency_prevents_false_partial_reuse(workspace):
    _, _ = accepted(workspace)
    repo = workspace[0]
    page = repo / "evidence-b.org"
    page.write_text(page.read_text() + "A changed condition needs another review.\n")
    git(repo, "add", "evidence-b.org")
    git(repo, "commit", "-qm", "Change one explanation dependency")
    assert discovery(workspace, replan(workspace)) is None


def test_reused_status_rejects_replaced_repository_identity(workspace):
    import uuid
    plan, _ = accepted(workspace)
    wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "explicit-model", transport=NoCalls())
    repo = workspace[0]
    path = ".sucoder/wiki/sources.json"
    registry = json.loads((repo / path).read_text())
    registry["repository_uuid"] = str(uuid.uuid4())
    (repo / path).write_bytes(dream.canonical(registry) + b"\n")
    git(repo, "add", path)
    git(repo, "commit", "-qm", "Human replaces repository source identity")
    assert dream_run.status(workspace[2], "sample", Path(plan["plan_path"]))["state"] == "needs_reconciliation"


def test_reuse_spec_cannot_be_substituted_under_another_run_filename(workspace):
    plan, _ = accepted(workspace)
    value = wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "explicit-model", transport=NoCalls())
    wrong_id = "f" * 64
    with dream_run._directory(workspace[2], Path(plan["plan_path"])) as (fd, _):
        data = dream_run._read(fd, "native-" + value["run_id"] + "-spec.json", 16384)
        dream_run._store(fd, "native-" + wrong_id + "-spec.json", data)
    with pytest.raises(dream.DreamError, match="storage identity"):
        dream_run.status(workspace[2], "sample", Path(plan["plan_path"]), run_id=wrong_id)
