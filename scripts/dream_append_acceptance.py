"""Opt-in live append acceptance over sanitized transcripts in a private repo.

poetry run python scripts/dream_append_acceptance.py --harness codex --model MODEL

Uses the existing native transport and planner; never discovers real sessions.
Live Claude remains separately opt-in. All evidence and integration are fixtures.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pwd
import subprocess
import tempfile
from unittest.mock import patch

from sucoder import dream, dream_native, dream_run, dream_wiki
from sucoder.config import load_config


def evidence(kind):
    first = ("Sanitized acceptance fixture: the sample publishing procedure requires --dry-run. "
             "This records documentation, not observed execution. The pending tool call below is synthetic.")
    revised = ("The same sample publishing procedure now additionally requires --scope repo. "
               "Keep --dry-run. These are documented requirements; no successful publication has been observed.")
    if kind == "codex-jsonl":
        prefix = [dict(type="session_meta", payload=dict(id="sanitized-append-thread")),
            dict(type="response_item", payload=dict(type="message", role="user", content=[dict(type="input_text", text=first)])),
            dict(type="response_item", payload=dict(type="function_call", call_id="pending", name="exec", arguments='{"command":"sample-publish --dry-run"}'))]
        tail = [dict(type="response_item", payload=dict(type="function_call_output", call_id="pending", output="Synthetic fixture result: exit 1; missing scope.")),
            dict(type="response_item", payload=dict(type="message", role="user", content=[dict(type="input_text", text=revised)]))]
    else:
        prefix = [dict(type="user", sessionId="sanitized-append-thread", uuid="u1", message=dict(content=first)),
            dict(type="assistant", sessionId="sanitized-append-thread", uuid="a1", message=dict(content=[dict(type="tool_use", id="pending", name="Bash", input=dict(command="sample-publish --dry-run"))]))]
        tail = [dict(type="user", sessionId="sanitized-append-thread", uuid="r1", parentUuid="a1", message=dict(content=[dict(type="tool_result", tool_use_id="pending", is_error=True, content="Synthetic fixture result: exit 1; missing scope.")])),
            dict(type="user", sessionId="sanitized-append-thread", uuid="u2", message=dict(content=revised))]
    return tuple(b"".join(dream.canonical(row) + b"\n" for row in rows) for rows in (prefix, tail))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, choices=("codex", "claude"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", choices=("codex-jsonl", "claude-jsonl"), default="codex-jsonl")
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="sucoder-dream-append-acceptance-"))
    print(json.dumps(dict(state="started", private_artifacts=str(root))), flush=True)
    repository = root / "mirrors" / "canary"
    repository.mkdir(parents=True)
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(repository), *arguments]).decode().strip()
    git("init", "-q", "-b", "main")
    git("config", "user.name", "Dream append acceptance fixture")
    git("config", "user.email", "fixture@example.test")
    (repository / "README.txt").write_text("Sanitized append acceptance only; no real source corpus or procedure execution.\n")
    git("add", ".")
    git("commit", "-qm", "Initialize disposable sanitized append fixture")
    configuration = root / "config.json"
    configuration.write_bytes(dream.canonical(dict(human_user=pwd.getpwuid(os.geteuid()).pw_name,
        mirror_root=str(repository.parent), log_dir=str(root / "logs"),
        mirrors={"canary": {"canonical_repo": str(root / "canonical")}})))
    config = load_config(configuration)
    transcript = root / "sanitized.jsonl"
    prefix, tail = evidence(args.kind)
    transcript.write_bytes(prefix)
    catalog = root / "sources.json"
    catalog.write_bytes(dream.canonical(dict(schema_version=1, repository="canary", integration_ref="refs/heads/main",
        units=[dict(unit_id="procedure", kind=args.kind, shareable=True,
            locator=dict(path=str(transcript), thread_id="sanitized-append-thread"))])))
    purpose = "Maintain one concise explanation of the sample publishing procedure, its required flags, and what has actually been observed."
    def run():
        plan = dream.create_plan(config, "canary", catalog, "refs/heads/main")
        before = (git("status", "--porcelain"), (repository / ".git/index").read_bytes(), git("symbolic-ref", "HEAD"))
        result = dream_wiki.run(config, "canary", Path(plan["plan_path"]), args.harness, args.model, purpose=purpose)
        if before != (git("status", "--porcelain"), (repository / ".git/index").read_bytes(), git("symbolic-ref", "HEAD")):
            raise RuntimeError("Candidate publication changed user worktree/index/branch")
        if result["state"] != "candidate_ready":
            raise RuntimeError("Expected a substantive candidate for this fixture; inspect retained role traces")
        status = dream_run.status(config, "canary", Path(plan["plan_path"]))
        if status["integration"]["state"] != "not_integrated":
            raise RuntimeError("An unintegrated candidate was treated as accepted")
        receipt = dream._json(git("show", result["candidate_commit"] + ":.sucoder/wiki/receipts/" + result["run_id"] + ".json").encode())
        return plan, result, receipt
    initial_plan, initial, _ = run()
    original_paths = git("ls-tree", "-r", "--name-only", initial["candidate_commit"], "--", ".sucoder/wiki/topics/").splitlines()
    if not any("--dry-run" in git("show", initial["candidate_commit"] + ":" + path) for path in original_paths):
        raise RuntimeError("Initial live companion lost the required flag")
    git("merge", "--ff-only", initial["candidate_commit"])
    transcript.write_bytes(prefix + tail)
    update_plan, update, receipt = run()
    paths = git("ls-tree", "-r", "--name-only", update["candidate_commit"], "--", ".sucoder/wiki/topics/").splitlines()
    if paths != original_paths or not any("--dry-run" in git("show", update["candidate_commit"] + ":" + path) and
            "--scope repo" in git("show", update["candidate_commit"] + ":" + path) for path in original_paths):
        raise RuntimeError("Append did not update the established procedure topic with both required qualifications")
    if receipt.get("source_delivery", {}).get("initial_bytes") != len(tail) or receipt["source_delivery"]["accepted_prefix_bytes"] != len(prefix):
        raise RuntimeError("Append did not account for accepted prefix and suffix delivery")
    directory = Path(update_plan["plan_path"]).parent
    for trace in receipt["assignments"]:
        task = dream._json((directory / (update["run_id"] + "-" + trace["assignment_id"] + "-assignment.json")).read_bytes())
        if task["input"]["sources"][0]["text"].encode() != tail:
            raise RuntimeError("A native role was resupplied the original raw prefix")
    git("merge", "--ff-only", update["candidate_commit"])
    fresh = dream.create_plan(config, "canary", catalog, "refs/heads/main")
    with patch.object(dream_native.Transport, "__call__", side_effect=AssertionError("Unchanged accepted append invoked a provider")):
        reused = dream_wiki.run(config, "canary", Path(fresh["plan_path"]), args.harness, args.model, purpose=purpose)
        status = dream_run.status(config, "canary", Path(fresh["plan_path"]))
    if reused["state"] != "already_integrated" or reused["usage"]["role_invocations"] != 0 or status["state"] != "already_integrated":
        raise RuntimeError("Accepted appended revision did not establish zero-call shared progress")
    engine = Path(__file__).resolve().parent.parent
    record = dict(engine_commit=subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"]).decode().strip(),
        harness=args.harness, model=args.model, kind=args.kind, initial_plan=initial_plan, initial=initial,
        update_plan=update_plan, update=update, receipt=receipt, fresh_plan=fresh, reused=reused, status=status,
        final_topics={path: git("show", "HEAD:" + path) for path in paths})
    (root / "acceptance.json").write_bytes(dream.canonical(record) + b"\n")
    print(json.dumps(dict(state="passed", private_artifacts=str(root), initial_run=initial["run_id"],
        update_run=update["run_id"], source_delivery=receipt["source_delivery"], unchanged_provider_calls=0)), flush=True)


if __name__ == "__main__":
    main()
