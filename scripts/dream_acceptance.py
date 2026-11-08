"""Opt-in live dream acceptance using synthetic evidence in a disposable repo.

Run with `poetry run python scripts/dream_acceptance.py --harness codex
--model MODEL`. Artifacts are retained privately for inspection. This invokes
the explicitly selected provider; it never discovers or reads real transcripts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pwd
import subprocess
import tempfile

from sucoder import dream, dream_roles, dream_run
from sucoder.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, choices=("codex", "claude"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", choices=("git", "codex-jsonl", "claude-jsonl"), default="git")
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="sucoder-dream-acceptance-"))
    print(json.dumps(dict(state="started", private_artifacts=str(root))), flush=True)
    repository = root / "mirrors" / "canary"
    repository.mkdir(parents=True)

    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(repository), *arguments]).decode().strip()

    git("init", "-q", "-b", "main")
    git("config", "user.name", "Dream acceptance fixture")
    git("config", "user.email", "fixture@example.test")
    text = ("Synthetic acceptance fixture only.\n"
            "The sample procedure requires flag --dry-run before publication.\n"
            "This records a requirement, not a successful experiment.\n")
    (repository / "procedure.txt").write_text(text)
    git("add", ".")
    git("commit", "-qm", "Synthetic dream acceptance evidence")
    if args.kind == "git":
        locator = dict(repository="canary", commit="HEAD", path="procedure.txt")
    else:
        if args.kind == "codex-jsonl":
            entries = [dict(type="session_meta", payload=dict(id="synthetic-thread")),
                       dict(type="response_item", payload=dict(type="message", role="user",
                           content=[dict(type="input_text", text=text)]))]
        else:
            entries = [dict(type="user", sessionId="synthetic-thread", uuid="synthetic-message",
                            message=dict(content=text))]
        transcript = root / "synthetic.jsonl"
        transcript.write_bytes(b"".join(dream.canonical(entry) + b"\n" for entry in entries))
        locator = dict(path=str(transcript), thread_id="synthetic-thread")
    catalog = root / "sources.json"
    catalog.write_bytes(dream.canonical(dict(schema_version=1, repository="canary", integration_ref="refs/heads/main",
        units=[dict(unit_id="procedure", kind=args.kind, shareable=True, locator=locator)])))
    configuration = root / "config.json"
    settings = dict(human_user=pwd.getpwuid(os.geteuid()).pw_name,
        mirror_root=str(repository.parent), log_dir=str(root / "logs"),
        mirrors={"canary": {"canonical_repo": str(root / "canonical")}})
    configuration.write_bytes(dream.canonical(settings))
    config = load_config(configuration)
    plan = dream.create_plan(config, "canary", catalog, "refs/heads/main")
    before = (git("status", "--porcelain"), (repository / ".git/index").read_bytes(), git("symbolic-ref", "HEAD"))
    result = dream_roles.run(config, "canary", Path(plan["plan_path"]), args.harness, args.model)
    if before != (git("status", "--porcelain"), (repository / ".git/index").read_bytes(), git("symbolic-ref", "HEAD")):
        raise RuntimeError("Candidate publication changed user worktree/index/branch")
    page = git("show", result["candidate_commit"] + ":.sucoder/wiki/topics/dream-" + result["run_id"] + ".org")
    if "--dry-run" not in page:
        raise RuntimeError("Live semantic acceptance lost the required command flag")
    status = dream_run.status(config, "canary", Path(plan["plan_path"]))
    if status["integration"]["state"] != "not_integrated":
        raise RuntimeError("Unintegrated candidate was reported as accepted")
    tree = git("rev-parse", result["candidate_commit"] + "^{tree}")
    squash = git("commit-tree", tree, "-p", plan["base_commit"], "-m", "Synthetic squash integration")
    git("update-ref", "refs/heads/main", squash, plan["base_commit"])
    status = dream_run.status(config, "canary", Path(plan["plan_path"]))
    if status["integration"]["state"] != "integrated_candidate":
        raise RuntimeError("Squash-integrated output bindings did not validate")
    (root / "acceptance.json").write_bytes(dream.canonical(dict(result=result, status=status)) + b"\n")
    print(json.dumps(dict(state="passed", harness=args.harness, model=args.model, kind=args.kind,
                         run_id=result["run_id"], private_artifacts=str(root))), flush=True)


if __name__ == "__main__":
    main()
