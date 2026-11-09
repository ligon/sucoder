"""Opt-in native editorial/archive and fresh-reader check over sanitized data.

poetry run python scripts/dream_editorial_acceptance.py --harness codex --model MODEL

All source history is an explicitly synthetic fixture in a private repository.
This never discovers real transcripts or records a real maintainer decision.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pwd
import subprocess
import tempfile

from sucoder import dream, dream_reader, dream_reader_agent, dream_run, dream_wiki
from sucoder.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, choices=("codex", "claude"))
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="sucoder-dream-editorial-acceptance-"))
    print(json.dumps(dict(state="started", private_artifacts=str(root))), flush=True)
    repo = root / "mirrors/canary"
    repo.mkdir(parents=True)
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(repo), *arguments]).decode().strip()
    git("init", "-q", "-b", "main")
    git("config", "user.name", "Sanitized editorial acceptance fixture")
    git("config", "user.email", "fixture@example.test")
    topic = ".sucoder/wiki/topics/replies.org"
    archive = ".sucoder/wiki/archive/replies.org"
    (repo / topic).parent.mkdir(parents=True)
    (repo / "evidence.org").write_text("Sanitized evidence only. A compaction summary does not establish a native reply. "
        "Native delivery plus subsequent native assistant text is required. An absorbed queued input may establish delivery through the queued_command attachment. "
        "This records documented/static behavior, not an executed launch.\n")
    (repo / topic).write_text("#+title: Sample reply evidence\n#+TODO: TODO NEXT ACTIVE WAITING REVIEW | DONE MOOT\n"
        "Sanitized acceptance fixture: the task history below is illustrative, not actual project or human review.\n"
        "* Reply evidence\n:PROPERTIES:\n:CUSTOM_ID: reply-proof\n:END:\n"
        "A summary match is not reply proof. Require native delivery and subsequent native assistant text. "
        "The queued_command attachment can establish absorbed-input delivery; this is documented/static scope, not a newly executed launch.\n"
        "[[file:../../../evidence.org][Maintained fixture evidence]]\n** Talk :editorial:\n"
        "*** DONE Correct the reply rule\nCLOSED: [2026-11-08 Sun 08:10]\n:PROPERTIES:\n:CUSTOM_ID: corrected-reply-rule\n:OWNER: Example editor\n:REVIEWER: Example verifier\n:END:\n"
        ":LOGBOOK:\n- State \"DONE\" from \"REVIEW\" [2026-11-08 Sun 08:10] \\\n  Example verifier checked the restored queued-input qualification.\n"
        "- State \"ACTIVE\" from \"REVIEW\" [2026-11-08 Sun 08:08] \\\n  Example verifier returned the first proposal because it omitted queued-command attachment delivery.\n:END:\n"
        "Preserve the native-event rule and its queued-input qualification in the active article.\n"
        "*** MOOT Infer a reply from a summary\nCLOSED: [2026-11-08 Sun 08:10]\n:PROPERTIES:\n:CUSTOM_ID: infer-from-summary\n:OWNER: Example editor\n:END:\n"
        ":LOGBOOK:\n- State \"MOOT\" from \"TODO\" [2026-11-08 Sun 08:10] \\\n  Superseded by [[#reply-proof][the native-event rule]]; summary claims do not establish assistant delivery.\n:END:\n"
        "*** WAITING Maintainer timeout decision\n:PROPERTIES:\n:CUSTOM_ID: maintainer-timeout\n:OWNER: User\n:REVIEWER: User\n:REQUIRES_HUMAN: yes\n"
        ":BLOCKER: No maintainer decision supplied\n:RESUME_WHEN: A human decision is recorded\n:END:\nKeep the ten-second timeout until a recorded human decision.\n")
    incoming = ".sucoder/wiki/topics/navigation.org"
    (repo / incoming).write_text("* Navigation\n:PROPERTIES:\n:CUSTOM_ID: navigation\n:END:\n[[file:replies.org::#corrected-reply-rule][Why the correction returned for review]].\n")
    (repo / dream_wiki.INDEX).write_text("#+title: Sample companion\n[[file:topics/replies.org::#reply-proof][Reply evidence]]\n[[file:topics/navigation.org::#navigation][History navigation]]\n")
    (repo / dream_wiki.SETTINGS_PATH).write_text(dream_wiki.SETTINGS)
    git("add", ".")
    git("commit", "-qm", "Seed explicit synthetic editorial history and evidence")
    configuration = root / "config.json"
    configuration.write_bytes(dream.canonical(dict(human_user=pwd.getpwuid(os.geteuid()).pw_name,
        mirror_root=str(repo.parent), log_dir=str(root / "logs"), mirrors={"canary": {"canonical_repo": str(root / "canonical")}})))
    config = load_config(configuration)
    catalog = root / "sources.json"
    catalog.write_bytes(dream.canonical(dict(schema_version=1, repository="canary", integration_ref="refs/heads/main",
        units=[dict(unit_id="evidence", kind="git", shareable=True, locator=dict(repository="canary", commit="HEAD", path="evidence.org"))])))
    plan = dream.create_plan(config, "canary", catalog, "refs/heads/main", maintenance=
        "Archive the eligible terminal correction and superseded heuristic, preserving active qualifications, history, incoming links and the unresolved human decision.", topic="replies")
    before = (git("status", "--porcelain"), (repo / ".git/index").read_bytes(), git("symbolic-ref", "HEAD"))
    result = dream_wiki.run(config, "canary", Path(plan["plan_path"]), args.harness, args.model)
    if before != (git("status", "--porcelain"), (repo / ".git/index").read_bytes(), git("symbolic-ref", "HEAD")):
        raise RuntimeError("Editorial candidate changed the invoking worktree/index/branch")
    if result["state"] != "candidate_ready":
        raise RuntimeError("Expected an archival candidate for this fixture; inspect retained artifacts")
    candidate = result["candidate_commit"]
    active = git("show", candidate + ":" + topic)
    historical = git("show", candidate + ":" + archive)
    navigation = git("show", candidate + ":" + incoming)
    if ("queued_command" not in active or "WAITING Maintainer timeout decision" not in active or
            "* DONE Correct the reply rule" not in historical or "* MOOT Infer a reply from a summary" not in historical or
            ":ARCHIVE_ITAGS: editorial" not in historical or ":ARCHIVE_FILE: " + topic not in historical or
            "file:../archive/replies.org::#corrected-reply-rule" not in navigation):
        raise RuntimeError("Archival acceptance lost a task, qualification, human constraint or incoming link")
    if 'State "ACTIVE" from "REVIEW"' in active or 'State "ACTIVE" from "REVIEW"' not in historical:
        raise RuntimeError("Returned-review history was not moved completely")
    if dream_run.status(config, "canary", Path(plan["plan_path"]))["integration"]["state"] != "not_integrated":
        raise RuntimeError("Unintegrated editorial output was treated as accepted")
    git("merge", "--ff-only", candidate)
    ordinary = dream_reader.read(config, "canary", "replies")
    if 'State "ACTIVE" from "REVIEW"' in ordinary["text"]:
        raise RuntimeError("Ordinary reading included archived history")
    question = "Why did the correction return for review, what qualification was restored, and where is that decision recorded?"
    answer = dream_reader_agent.ask(config, "canary", Path(plan["plan_path"]), question, args.harness, args.model)
    if not answer["topic_delivered"] or not any(entry["request"]["kind"] == "archive" and entry["request"]["target"] == "corrected-reply-rule"
            and entry["response"]["delivered_bytes"] for entry in answer["retrievals"]):
        raise RuntimeError("Fresh reader did not follow the active page to the specific archived correction")
    engine = Path(__file__).resolve().parent.parent
    record = dict(engine_commit=subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"]).decode().strip(),
        plan=plan, result=result, answer=answer, active=active, archive=historical, incoming=navigation, ordinary=ordinary)
    (root / "acceptance.json").write_bytes(dream.canonical(record) + b"\n")
    print(json.dumps(dict(state="passed", private_artifacts=str(root), run_id=result["run_id"],
        builder_calls=result["usage"]["role_invocations"], reader_calls=answer["usage"]["role_invocations"])), flush=True)


if __name__ == "__main__":
    main()
