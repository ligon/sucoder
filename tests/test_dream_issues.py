"""Correction drafts, disclosure gating and marker reconciliation without GitHub writes."""
from dataclasses import replace
import json
from pathlib import Path
import uuid

import pytest
from typer.testing import CliRunner

from sucoder import cli, dream, dream_issues, dream_memory, dream_wiki
from tests.test_dream import git, workspace  # noqa: F401
from tests.test_dream_memory import MEMORY, memory_plan, source
from tests.test_dream_wiki import EditorTransport, TOPIC
from tests.dream_submission_fixture import source_text


CUSTOM = "correct-summary"


def reviewed(workspace):
    plan = memory_plan(workspace)
    def correction(task, result):
        if task["role"] == "reviewer":
            return
        assert "Corrections are tasks" in task["brief"]
        page = next(p for p in result["pages"] if p["path"] == TOPIC)
        page["content"] += ("* TODO Correct the summary documentation\n:PROPERTIES:\n:CUSTOM_ID: correct-summary\n"
            ":TARGET: evidence-a.org:1-1\n:ACCEPTANCE: Say that summaries do not prove replies.\n:END:\n"
            "Correct the documented claim using the native delivery evidence.\n")
        result["references"].append(dict(path=TOPIC, target=CUSTOM, evidence_status="inference",
            citations=[dict(unit_id=u["unit_id"], quote=source_text(u)) for u in task["input"]["sources"]]))
    result = dream_wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "fixture",
        transport=EditorTransport(mutate=correction))
    assert result["state"] == "published"
    return result


def test_reviewed_correction_produces_a_draft_and_marker_prevents_refiling(workspace):
    generation = reviewed(workspace)
    first = dream_issues.drafts(source(workspace), generation["candidate_commit"], "example/project")
    assert len(first["drafts"]) == 1
    draft = first["drafts"][0]
    assert draft["labels"] == ["dream-memory"] and draft["target_url"].endswith("/evidence-a.org#L1-L1")
    assert "topics/reply-evidence.org#correct-summary" in draft["marker"]
    assert draft["marker"] in draft["body"] and "Acceptance:" in draft["body"]
    existing = [dict(number=7, state="open", html_url="https://github.com/example/project/issues/7", body=draft["body"])]
    second = dream_issues.drafts(source(workspace), generation["candidate_commit"], "example/project", existing)
    assert second["drafts"] == [] and second["tasks"][0]["state"] == "open_task_with_open_issue"


def seeded(workspace, *, private=False, missing=False):
    """A pinned reviewed-memory fixture with open and terminal correction states."""
    repo = workspace[0]
    memory = repo / ".sucoder/wiki"
    (memory / "topics").mkdir(parents=True)
    (memory / "receipts").mkdir()
    identity = str(uuid.uuid4())
    (memory / "sources.json").write_text(json.dumps(dict(schema_version=1, repository_uuid=identity)))
    page = "#+TODO: TODO NEXT ACTIVE WAITING REVIEW | DONE MOOT\n" + "".join(
        f"* {state} Correction {target}\n:PROPERTIES:\n:CUSTOM_ID: {target}\n:TARGET: note.org:2-3\n"
        ":ACCEPTANCE: Preserve the source qualification.\n:END:\nCorrect the missing qualification.\n"
        for state, target in (("TODO", "unfiled"), ("TODO", "open"), ("TODO", "closed"), ("DONE", "done")))
    path = ".sucoder/wiki/topics/corrections.org"
    (repo / path).write_text(page)
    project_commit = git(repo, "rev-parse", "HEAD")
    unit = dict(unit_id="evidence", kind="claude-view" if private else "git", repository_uuid=identity,
        commit=project_commit, path="note.org", sha256="a" * 64)
    receipt = dict(repository_uuid=identity, project_commit=project_commit,
        review=dict(verdict="accept", failures=[], findings=[]),
        outputs=[dict(path=path, sha256=dream.digest(page.encode()), rule="file-utf8-v1")],
        units=[] if missing else [unit], references=[dict(path=path, target=target,
            citations=[dict(unit_id="evidence", byte_start=0, byte_end=1)]) for target in ("unfiled", "open", "closed", "done")])
    (memory / "receipts" / ("b" * 64 + ".json")).write_bytes(dream.canonical(receipt) + b"\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "seed reviewed correction records")
    commit = dream_memory.ensure(source(workspace), MEMORY, git(repo, "rev-parse", "HEAD"), repo.parent, source(workspace).deadline)
    markers = {target: f"<!-- sucoder-dream: {identity}/topics/corrections.org#{target} -->" for target in ("open", "closed", "done")}
    existing = [dict(number=i + 1, state="closed" if key == "closed" else "open", body=marker,
        html_url=f"https://github.com/example/project/issues/{i + 1}") for i, (key, marker) in enumerate(markers.items())]
    return commit, existing


@pytest.mark.parametrize("missing", [False, True])
def test_session_or_missing_provenance_is_withheld_without_disclosure_flag(workspace, missing):
    commit, _ = seeded(workspace, private=True, missing=missing)
    result = dream_issues.drafts(source(workspace), commit, "example/project")
    assert result["drafts"] == []
    assert all("withheld" in task for task in result["tasks"] if task["task_state"] == "TODO")
    overridden = dream_issues.drafts(source(workspace), commit, "example/project", allow_private=True)
    assert len(overridden["drafts"]) == 3


def test_list_reports_all_four_required_task_issue_cases(workspace):
    commit, existing = seeded(workspace)
    result = dream_issues.drafts(source(workspace), commit, "example/project", existing)
    assert {task["state"] for task in result["tasks"]} == {
        "open_task_without_issue", "open_task_with_open_issue", "closed_issue_with_open_task", "terminal_task_with_open_issue"}


def test_cli_defaults_to_drafts_and_files_only_on_explicit_request(workspace, monkeypatch):
    commit, existing = seeded(workspace)
    monkeypatch.chdir(workspace[0])
    calls, filed = [], []
    def gh(*args, data=None):
        calls.append(args)
        if args[0] == "api" and "issues?" in args[-1]:
            return json.dumps([existing + filed])
        if args[0] == "api" and "labels?" in args[-1]:
            return json.dumps([[dict(name="dream-memory")]])
        assert args[:2] == ("issue", "create")
        body = Path(args[args.index("--body-file") + 1]).read_text()
        filed.append(dict(number=8, state="open", body=body, html_url="https://github.com/example/project/issues/8"))
        return filed[-1]["html_url"] + "\n"
    monkeypatch.setattr(dream_issues, "_gh", gh)
    base = ["--config", str(workspace[1]), "dream", "issues", "--repository", "example/project"]
    runner = CliRunner()
    before = git(workspace[0], "rev-parse", MEMORY)
    result = runner.invoke(cli.app, base)
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.output)["drafts"]) == 1 and not filed
    result = runner.invoke(cli.app, base + ["--file"])
    assert result.exit_code == 0, result.output
    assert len(filed) == 1 and json.loads(result.output)["state"] == "filed"
    result = runner.invoke(cli.app, base + ["--file"])
    assert result.exit_code == 0, result.output
    assert len(filed) == 1 and json.loads(result.output)["drafts"] == []
    result = runner.invoke(cli.app, base + ["--list"])
    assert result.exit_code == 0, result.output
    assert "drafts" not in json.loads(result.output)
    assert git(workspace[0], "rev-parse", MEMORY) == before
