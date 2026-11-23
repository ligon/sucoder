"""V6 diagnostics repair independent defects together without expanding authority."""
from pathlib import Path
import json
import os
import time

import pytest

from sucoder import dream, dream_diagnostics as diag, dream_editor, dream_run, dream_wiki as wiki
from tests.test_dream import workspace  # noqa: F401
from tests.test_dream_wiki import EditorTransport, POLICY, TOPIC, planned
from tests.test_dream_prompt import profile


def case():
    text = "caf\u00e9 twice then twice."
    unit = dict(unit_id="source", text=text, byte_start=17, byte_end=17 + len(text.encode()))
    pages = {TOPIC: "fixture"}
    parsed = {TOPIC: dict(targets=dict(explanation="new", talk="new"))}
    proposal = dict(references=[dict(path=TOPIC, target="explanation", evidence_status="inference",
        citations=[dict(unit_id="source", quote="twice")])], omissions=[])
    return proposal, pages, parsed, {}, [unit], set(), {"source": [(17, unit["byte_end"])]}


def test_reports_ambiguous_quote_and_uncited_talk_together():
    with pytest.raises(diag.InvalidProposal) as caught:
        diag.references(*case(), time.monotonic() + 5)
    findings = caught.value.findings
    assert [(f["code"], f["path"], f["target"]) for f in findings] == [
        ("citation_quote", TOPIC, "explanation"), ("reference_missing", TOPIC, "talk")]
    assert findings[0]["unit_id"] == "source" and findings[0]["match_count"] == 2


def test_overlapping_windows_deduplicate_absolute_utf8_position():
    args = list(case())
    args[0]["references"][0]["citations"][0]["quote"] = "caf\u00e9"
    args[2][TOPIC]["targets"].pop("talk")
    args[6]["source"] = [(17, 25), (17, args[4][0]["byte_end"])]
    result = diag.references(*args, time.monotonic() + 5)
    assert result[0]["citations"][0] == dict(unit_id="source", quote="caf\u00e9", byte_start=17, byte_end=22)


def test_collects_unknown_unit_empty_reference_wrong_target_and_missing_anchor():
    args = list(case())
    args[0]["references"][0]["citations"][0]["unit_id"] = "not-admitted"
    args[0]["references"].append(dict(path=TOPIC, target="absent", evidence_status="inference", citations=[]))
    with pytest.raises(diag.InvalidProposal) as caught:
        diag.references(*args, time.monotonic() + 5)
    assert {f["code"] for f in caught.value.findings} == {"citation_unit", "reference_empty", "reference_target", "reference_missing"}
    # Invalid citation dependency is already reported; don't invent an extra missing reference for explanation.
    assert [f["target"] for f in caught.value.findings if f["code"] == "reference_missing"] == ["talk"]


def test_diagnostic_overflow_is_explicit_and_never_accepts():
    args = list(case())
    args[2][TOPIC]["targets"] = {str(i): "new" for i in range(diag.MAX_FINDINGS + 10)}
    with pytest.raises(diag.InvalidProposal) as caught:
        diag.references(*args, time.monotonic() + 5)
    assert len(caught.value.findings) == diag.MAX_FINDINGS
    assert caught.value.findings[-1]["code"] == "diagnostics_incomplete"


def test_deadline_is_operational_not_a_repair_finding():
    with pytest.raises(dream.DreamError) as caught:
        diag.references(*case(), time.monotonic() - 1)
    assert caught.value.exit_code == 3 and not isinstance(caught.value, diag.InvalidProposal)


def test_clipped_location_is_explicitly_incomplete():
    report = diag.Report()
    report.add("reference_missing", "Supply a reference.", path="p" * 300, target="talk")
    with pytest.raises(diag.InvalidProposal) as caught:
        report.check()
    assert caught.value.findings[0]["location_truncated"]
    assert caught.value.findings[-1]["code"] == "diagnostics_incomplete"


@pytest.mark.parametrize("extra", ["", "link", "duplicate"])
def test_one_full_repair_fixes_both_defects_then_reaches_review_and_replays(workspace, extra):
    plan = planned(workspace)
    def defects(task, result):
        if task["role"] == "reviewer":
            return
        page = next(p for p in result["pages"] if p["path"] == TOPIC)
        talk = "* Talk\n:PROPERTIES:\n:CUSTOM_ID: talk\n:END:\nWhat establishes native reply evidence?\n\n"
        page["content"] = page["content"].replace(POLICY, talk + POLICY)
        if task["role"] == "author":
            result["references"][0]["citations"][0]["quote"] = " "
            if extra == "link":
                page["content"] = page["content"].replace(POLICY, "[[file:../../../missing.txt][Missing source]]\n\n" + POLICY)
            if extra == "duplicate":
                result["references"].append(dict(result["references"][0]))
        else:
            findings = task["input"]["findings"]
            assert {f["code"] for f in findings} == {"citation_quote", "reference_missing"} | ({"link_invalid"} if extra == "link" else {"reference_duplicate"} if extra == "duplicate" else set())
            assert all(f["path"] == TOPIC for f in findings)
            result["references"].append(dict(result["references"][0], target="talk"))
    transport = EditorTransport(mutate=defects)
    result = wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "claude", "fixture", transport=transport)
    assert result["fidelity"] == diag.CONTRACT
    assert [t["role"] for t in transport.calls] == ["author", "repair", "reviewer"]
    assert dream_run.status(workspace[2], "sample", Path(plan["plan_path"]))["state"] == "candidate_ready"
    assert len(transport.calls) == 3


def test_v5_retains_first_error_and_saved_status(workspace, monkeypatch):
    planned(workspace)
    plan = dream.create_plan(workspace[2], "sample", workspace[1].parent / "sources.json", "refs/heads/main", authoring_version=5)
    assert "authoring_version" not in plan
    def old_error(task, result):
        if task["role"] == "author":
            result["references"][0]["citations"][0]["quote"] = " "
            page = next(p for p in result["pages"] if p["path"] == TOPIC)
            page["content"] = page["content"].replace(POLICY, "* Talk\n:PROPERTIES:\n:CUSTOM_ID: talk\n:END:\nA question.\n\n" + POLICY)
        if task["role"] == "repair":
            findings = task["input"]["findings"]
            assert len(findings) == 1 and findings[0]["path"] == "" and "code" not in findings[0]
            assert "unique among delivered" in findings[0]["message"]
    transport = EditorTransport(mutate=old_error)
    result = wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "fixture", transport=transport)
    assert result["fidelity"] == dream_editor.CONTRACT
    assert dream_run.status(workspace[2], "sample", Path(plan["plan_path"]))["state"] == "candidate_ready"
    assert len(transport.calls) == 3


def test_v5_interrupted_attempt_cannot_silently_start_a_v6_author(workspace):
    planned(workspace)
    plan = dream.create_plan(workspace[2], "sample", workspace[1].parent / "sources.json", "refs/heads/main", authoring_version=5)
    class Interrupted(EditorTransport):
        def __call__(self, *args, **kwargs):
            self.calls.append(args[2])
            raise RuntimeError("simulated interrupted transport")
    native = Interrupted()
    with pytest.raises(RuntimeError, match="interrupted"):
        wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "fixture", transport=native)
    assert len(native.calls) == 1 and native.calls[0]["contract"] == dream_editor.CONTRACT
    with pytest.raises(dream.DreamError, match="earlier role attempt"):
        wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "fixture", transport=native)
    assert len(native.calls) == 1


def test_malformed_inventory_blocks_dependent_coverage(workspace):
    plan = planned(workspace)
    def malformed(task, result):
        if task["role"] != "reviewer":
            page = next(p for p in result["pages"] if p["path"] == TOPIC)
            page["content"] += "\n* Duplicate\n:PROPERTIES:\n:CUSTOM_ID: reply-proof\n:END:\n"
        if task["role"] == "repair":
            assert [f["code"] for f in task["input"]["findings"]] == ["validation_blocked"]
    native = EditorTransport(mutate=malformed)
    with pytest.raises(dream.DreamError, match="did not pass"):
        wiki.run(workspace[2], "sample", Path(plan["plan_path"]), "codex", "fixture", transport=native)
    assert [t["role"] for t in native.calls] == ["author", "repair"]


@pytest.mark.parametrize("layout,explicit,attempt", [("stable", False, 1), ("stable", False, 2), ("legacy", False, 1), ("stable", True, 1)])
def test_stable_resume_uses_original_profile_and_rejects_layout_or_profile_change(tmp_path, monkeypatch, layout, explicit, attempt):
    frozen = profile("claude")
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        spec = dict(harness="claude", model="fixture", attempt=1, purpose="purpose", prompt_profile=frozen)
        dream_run._store(fd, "native-original-spec.json", dream.canonical(spec) + b"\n")
        def no_catalog(*args):
            raise AssertionError("Resume loaded today's profile")
        from sucoder import dream_prompt_native
        monkeypatch.setattr(dream_prompt_native, "load_profile", no_catalog)
        replacement = dict(frozen, sha256="0" * 64) if explicit else None
        if layout == "stable" and not explicit:
            assert wiki.frozen_profile(None, {}, fd, "claude", "fixture", attempt, "purpose", layout, replacement) == frozen
        else:
            with pytest.raises(dream.DreamError, match="different frozen prompt"):
                wiki.frozen_profile(None, {}, fd, "claude", "fixture", 1, "purpose", layout, replacement)
    finally:
        os.close(fd)


def test_historical_job_without_editorial_plan_keeps_v5(workspace):
    from sucoder import dream_explore
    from tests.test_dream_explore import ExploringTransport, options
    planned(workspace)
    prepared = dream_explore.prepare(workspace[2], cwd=workspace[0], overrides=options())
    directory = Path(prepared["job_path"])
    header = json.loads((directory / "job.json").read_bytes())
    assert header.pop("authoring_version") == 6
    # Construct the pre-v6 frozen header fixture before any provider work.
    (directory / "job.json").write_bytes(dream.canonical(header) + b"\n")
    (directory / "identity.json").write_bytes(dream.canonical(dict(sha256=dream.digest(dream.canonical(header)))) + b"\n")
    native = ExploringTransport()
    result = dream_explore.run(workspace[2], directory, transport=native)
    assert result["state"] == "published"
    assert all(t["contract"] == dream_editor.CONTRACT for t in native.calls if t["role"] != "explorer")
    assert "authoring_version" not in json.loads(Path(result["plan_path"]).read_bytes())
