"""Prepare and verify the scoped dream pilot without provider requests.

Supply exact Git blobs, obtained from the pinned local repositories or the
connected GitHub file API, under BLOBS/GIT_BLOB_ID. This never discovers
transcripts, runs survey code, or starts the separately authorized experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time

from sucoder import dream, dream_native


CONDITIONS = ("source-only", "wiki-search", "wiki-only")


def verify(manifest, blobs):
    verified = []
    for corpus in manifest["corpora"]:
        units = []
        for source in corpus["sources"]:
            raw = (blobs / source["git_blob"]).read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            selected = raw[source["byte_start"]:source["byte_end"]]
            lines = raw.splitlines(keepends=True)
            lo, hi = source["lines"]
            if (blob != source["git_blob"] or len(selected) != source["bytes"] or
                    selected != b"".join(lines[lo - 1:hi]) or dream.digest(selected) != source["sha256"]):
                raise dream.DreamError("Frozen pilot source differs: " + source["unit_id"])
            units.append(dict(source, original_commit=corpus["commit"]))
        if len(units) != 6 or sum(unit["bytes"] for unit in units) != corpus["selected_bytes"]:
            raise dream.DreamError("Pilot corpus must reproduce all six declared passages")
        verified.append(dict(repository=corpus["repository"], commit=corpus["commit"], units=units))
    return verified


def isolation_canary(directory):
    evaluator = directory / "evaluator-only"
    evaluator.mkdir(mode=0o700)
    secret = evaluator / "boundary-canary"
    secret.write_text("EVALUATOR_ONLY_" + os.urandom(16).hex())
    with tempfile.TemporaryDirectory(prefix="dream-pilot-role-") as scratch:
        boundary = dream_native.FilesystemBoundary(scratch)
        try:
            boundary.allow(Path(sys.executable).resolve(), dream_native.READ)
            code, stdout, stderr = dream_native.capture([sys.executable, "-S", "-c",
                "import sys\nfor mode in ('r','w'):\n try: open(sys.argv[1],mode)\n except PermissionError: print('denied')\n else: raise SystemExit(1)", str(secret)],
                b"", scratch, dict(PATH=os.defpath), time.monotonic() + 10, boundary)
            if code or stdout != b"denied\ndenied\n":
                raise dream.DreamError("Pilot evaluator isolation canary failed", 3)
        finally:
            boundary.close()
    return dict(state="passed", boundary="linux-landlock-inline-v1", read="denied", write="denied",
        note="Same kernel boundary class as native construction/readers; their external tools are disabled.")


def prepare(blobs, output, *, model, seed=20261109):
    manifest_path = Path(__file__).resolve().parent.parent / "docs/dream/corpus.json"
    manifest = json.loads(manifest_path.read_bytes())
    verified = verify(manifest, blobs)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    boundary = isolation_canary(output)
    rng = random.Random(seed)
    schedule: list[dict] = []
    for corpus in verified:
        for question in range(1, 7):
            order = list(CONDITIONS)
            rng.shuffle(order)
            schedule.extend(dict(repository=corpus["repository"], question_id="q" + str(question), condition=condition) for condition in order)
    proposal = dict(schema_version=1, state="prepared_not_executed", provider_calls=0, harness="codex", model=model,
        corpus_manifest_sha256=dream.digest(manifest_path.read_bytes()), sources=verified,
        selected_bytes=sum(unit["bytes"] for corpus in verified for unit in corpus["units"]),
        cases=["initial first three passages", "incremental all six passages", "labeled seeded-error repair"],
        evaluator="Independent six held-out questions/rubrics per repository before construction; evaluator-only scoring after fresh answers.",
        isolation=boundary, seed=seed, reader_schedule=schedule,
        limits=dict(aggregate_elapsed_seconds=7200, aggregate_native_calls=180, reader_elapsed_seconds=120,
            reader_role_calls=8, construction_role_calls=12, construction_elapsed_seconds=900,
            currency_ceiling=None, currency_accounting="unknown; reject requests for an unenforceable dollar ceiling"),
        publication="Disposable candidates only; no changes to source repositories, private transcript admission or scientific decisions.",
        prerequisite="Separate scoped experiment authorization; this preparation performs no inference.")
    (output / "pilot-proposal.json").write_bytes(dream.canonical(proposal) + b"\n")
    return proposal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blobs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New private directory; existing paths are rejected.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    result = prepare(args.blobs, args.output, model=args.model, seed=args.seed)
    print(json.dumps(dict(state=result["state"], selected_bytes=result["selected_bytes"],
        answer_attempts=len(result["reader_schedule"]), provider_calls=0, proposal=str(args.output / "pilot-proposal.json"))))


if __name__ == "__main__":
    main()
