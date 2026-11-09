"""Accepted shared wiki progress, grounded in implementation ledger 3-6.

Conservatively reuse complete unchanged git selections. Receipt presence,
candidate refs, partial dependencies and private model summaries do not count.
"""
from __future__ import annotations

import re

from . import dream, dream_roles, dream_run, dream_wiki
from .dream import DreamError, canonical, digest


CONTRACT = "accepted-wiki-progress-v1"


def identity(unit):
    return tuple(unit[key] for key in ("source_id", "repository_uuid", "kind", "byte_start", "byte_end", "sha256"))


def compatible(plan, receipt, fd, harness, model, version, purpose):
    from . import dream_curate, dream_author
    contract = receipt.get("contract")
    briefs = {dream_wiki.CONTRACT: dream_wiki.BRIEFS, dream_curate.CONTRACT: dream_curate.BRIEFS,
              dream_author.CONTRACT: dream_author.BRIEFS}.get(contract)
    if (type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1 or briefs is None or
            receipt.get("repository_uuid") != plan["repository_uuid"] or receipt.get("harness") != harness or
            receipt.get("model") != model or receipt.get("native_version") != version or receipt.get("purpose") != purpose or
            receipt.get("brief_sha256") != digest(canonical(briefs))):
        return set()
    if contract in (dream_curate.CONTRACT, dream_author.CONTRACT) and receipt.get("decision") != "change":
        return set()
    dream_roles.validate_schema(receipt["review"], dream_author.schema("reviewer") if contract == dream_author.CONTRACT else dream_wiki.schema("reviewer"))
    if contract == dream_author.CONTRACT and receipt["review"]["requests"]:
        return set()
    if (receipt["review"]["verdict"] != "accept" or receipt["review"]["failures"] or
            any(f["severity"] == "blocking" for f in receipt["review"]["findings"])):
        return set()
    if not receipt.get("units") or not receipt.get("outputs") or not receipt.get("assignments"):
        return set()
    if receipt["assignments"][-1]["role"] != "reviewer" or not any(t["role"] == "author" for t in receipt["assignments"]):
        return set()
    for output in receipt["outputs"]:
        dream._object(output, ("path", "target", "sha256", "rule"))
        dream._path(output["path"])
        if not output["path"].startswith(".sucoder/wiki/") or not output["path"].endswith(".org"):
            return set()
        if output["rule"] != "file-utf8-v1" or not re.fullmatch(r"[0-9a-f]{64}", output["sha256"]):
            return set()
    admitted = {identity(unit): unit for unit in plan["units"]}
    covered, old_ids = set(), set()
    for old in receipt["units"]:
        unit = admitted.get(identity(old))
        if unit is None or old["kind"] != "git" or old["unit_id"] in old_ids:
            return set()  # Every dependency must still be admitted unchanged.
        old_ids.add(old["unit_id"])
        raw = dream_run._read(fd, "evidence-" + unit["sha256"] + ".txt", unit["bytes"])
        if len(raw) != unit["bytes"] or digest(raw) != unit["sha256"]:
            raise DreamError("Reuse source snapshot differs from its frozen digest", 3)
        cursor = unit["byte_start"]
        chunks = sorted((c for c in receipt["coverage"] if c["unit_id"] == old["unit_id"]), key=lambda c: c["byte_start"])
        for chunk in chunks:
            lo, hi = chunk["byte_start"], chunk["byte_end"]
            if (lo != cursor or not lo < hi <= unit["byte_end"] or
                    chunk["extraction"] != "author_inline_delivered" or chunk["verification"] != "whole_candidate_review" or
                    digest(raw[lo - unit["byte_start"]:hi - unit["byte_start"]]) != chunk["sha256"]):
                return set()
            cursor = hi
        if cursor != unit["byte_end"]:
            return set()
        covered.add(unit["unit_id"])
    return covered


def discover(plan, fd, source, harness, model, version, purpose, deadline):
    if any(unit["kind"] != "git" for unit in plan["units"]):
        return None  # Transcript deltas need their own normalization contract.
    valid, coverage, broken = [], set(), set()
    required = {unit["unit_id"] for unit in plan["units"]}
    inspected = 0
    for item in sorted(plan["prior_wiki"]["files"], key=lambda item: item["path"]):
        path = item["path"]
        match = re.fullmatch(r"\.sucoder/wiki/receipts/([0-9a-f]{64})\.json", path)
        if not match:
            continue
        try:
            remaining = dream_run.MAX_PLAN_BYTES - inspected
            if remaining <= 0:
                raise DreamError("Shared receipt inspection allowance exhausted; use explicit targeted maintenance", 3)
            raw = source.blob(item["blob"], remaining)
            inspected += len(raw)
            receipt = dream._json(raw)
            if (not isinstance(receipt, dict) or receipt.get("run_id") != match[1] or
                    raw != canonical(receipt) + b"\n"):
                continue
            matches = compatible(plan, receipt, fd, harness, model, version, purpose)
            if not matches:
                continue
            if not dream_run.bound_outputs(source, plan["base_commit"], receipt["outputs"], deadline):
                broken.update(matches)
                continue
        except (KeyError, TypeError, UnicodeError):
            continue  # Malformed shared metadata never establishes coverage.
        except DreamError as exc:
            if exc.exit_code not in (1, 2):
                raise
            continue
        if matches - coverage:
            coverage.update(matches)
            valid.append(dict(path=path, sha256=digest(raw), run_id=receipt["run_id"]))
        if coverage == required:
            break  # At most one proof per newly covered unit; bound metadata.
    if coverage == required:
        return valid
    if (required - coverage) & broken:
        raise DreamError("Accepted shared outputs need reconciliation; use explicit --maintenance and --topic for a checked repair", 3)
    return None


def result(plan, proof, harness, model, version, purpose):
    run_id = digest(canonical([CONTRACT, plan["plan_id"], proof, harness, model, version, purpose]))
    return dict(schema_version=1, operation="dream_run", state="already_integrated", run_id=run_id,
        plan_id=plan["plan_id"], base_commit=plan["base_commit"], harness=harness, model=model,
        native_version=version, fidelity=CONTRACT, candidate_commit=None, candidate_ref=None,
        receipts=proof, coverage=[dict(unit_id=unit["unit_id"], **chunk,
            extraction="accepted_receipt_reused", verification="validated_output_bindings")
            for unit in plan["units"] for chunk in unit["chunks"]],
        usage=dict(role_invocations=0, native=[], tokens=0, currency=None),
        integration=dict(state="integrated_candidate", evidence=dict(integration_commit=plan["base_commit"],
            receipt_run_ids=[p["run_id"] for p in proof])))


def try_reuse(plan, fd, source, harness, model, version, purpose, deadline):
    if plan["maintenance"] is not None:
        return None
    proof = discover(plan, fd, source, harness, model, version, purpose, deadline)
    if proof is None:
        return None
    value = result(plan, proof, harness, model, version, purpose)
    source.check_binding()
    if source.commit(plan["integration_ref"]) != plan["base_commit"]:
        raise DreamError("Integration ref changed during shared progress validation; re-plan", 3)
    prefix = "native-" + value["run_id"]
    spec = dict(contract=CONTRACT, run_id=value["run_id"], harness=harness, model=model,
                native_version=version, purpose=purpose, proof=proof)
    with dream_roles.run_lock(fd, prefix):
        dream_run._store(fd, prefix + "-spec.json", canonical(spec) + b"\n")
        dream_run._store(fd, prefix + "-run.json", canonical(value) + b"\n")
    return value


def restore(config, mirror, plan, fd, spec, deadline):
    dream._object(spec, ("contract", "run_id", "harness", "model", "native_version", "purpose", "proof"))
    source = dream_run._sources(config, plan, deadline)[mirror]
    proof = discover(plan, fd, source, spec["harness"], spec["model"], spec["native_version"], spec["purpose"], deadline)
    if proof is None:
        raise DreamError("No compatible accepted progress remains in the frozen tree")
    expected = result(plan, proof, spec["harness"], spec["model"], spec["native_version"], spec["purpose"])
    if proof != spec["proof"] or expected["run_id"] != spec["run_id"]:
        raise DreamError("Shared-progress proof differs from the frozen accepted tree")
    recorded = dream._json(dream_run._read(fd, "native-" + spec["run_id"] + "-run.json", dream_run.MAX_PLAN_BYTES))
    if canonical(recorded) != canonical(expected):
        raise DreamError("Stored shared-progress record differs from its checked proof")
    base = source.commit(plan["integration_ref"])
    current_identity = dream._repository_uuid(source, base)
    state = ("needs_reconciliation" if current_identity["provisional"] or
             current_identity["repository_uuid"] != plan["repository_uuid"] else "integrated_candidate")
    for item in proof:
        blob = source.entry(base, item["path"])
        if blob is None:
            state = "needs_reconciliation"
            break
        raw = source.blob(blob, dream_run.MAX_PLAN_BYTES)
        if digest(raw) != item["sha256"]:
            state = "needs_reconciliation"
            break
        receipt = dream._json(raw)
        if not dream_run.bound_outputs(source, base, receipt["outputs"], deadline):
            state = "needs_reconciliation"
            break
    source.check_binding()
    if source.commit(plan["integration_ref"]) != base:
        raise DreamError("Integration ref changed during shared progress status; retry", 3)
    return dict(expected, operation="dream_status", state="already_integrated" if state == "integrated_candidate" else state,
        candidate_status="not_created", integration=dict(state=state, evidence=dict(integration_commit=base,
            receipt_run_ids=[p["run_id"] for p in proof])))
