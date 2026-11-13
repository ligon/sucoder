"""Accepted transcript prefixes (implementation ledger, sections 2-6).

Shared acceptance, exact normalization and required outputs justify reuse.
A private no-op, copied path or incomplete tool call never establishes it.
"""
from __future__ import annotations

import re

from . import dream, dream_context, dream_progress, dream_run, dream_transcript
from .dream import DreamError, canonical, digest


CONTRACT = "accepted-transcript-prefix-v1"


def normalization(raw, unit):
    value = dream_transcript.normalize(raw, unit["kind"], unit["thread_id"])
    return dict(contract=dream_transcript.CONTRACT, sha256=digest(canonical(value)), records=len(value["records"]),
        unknown_events=sum(event["kind"] == "unknown" for record in value["records"] for event in record["events"]))


def snapshot(fd, unit):
    raw = dream_run._read(fd, "evidence-" + unit["sha256"] + ".txt", unit["bytes"])
    if len(raw) != unit["bytes"] or digest(raw) != unit["sha256"]:
        raise DreamError("Reuse source snapshot differs from its frozen digest", 3)
    return raw


def coverage_matches(receipt, old, raw):
    cursor, origin = old["byte_start"], old["byte_start"]
    chunks = sorted((chunk for chunk in receipt["coverage"] if chunk["unit_id"] == old["unit_id"]),
                    key=lambda chunk: chunk["byte_start"])
    for chunk in chunks:
        lo, hi = chunk["byte_start"], chunk["byte_end"]
        if (type(lo) is not int or type(hi) is not int or lo != cursor or not lo < hi <= old["byte_end"] or
                digest(raw[lo - origin:hi - origin]) != chunk["sha256"]):
            return False
        normal = chunk["extraction"] == "author_inline_delivered" and chunk["verification"] == "whole_candidate_review"
        reused = (receipt["contract"] in ("source-grounded-wiki-v4", "source-grounded-wiki-v5") and old["kind"] in dream_transcript.KINDS and
            chunk["extraction"] == "accepted_receipt_reused" and chunk["verification"] == "validated_output_bindings" and
            any(proof["contract"] == CONTRACT and proof["unit_id"] == old["unit_id"] and
                proof["byte_start"] == lo == 0 and proof["byte_end"] == hi and proof["sha256"] == chunk["sha256"] and
                re.fullmatch(r"[0-9a-f]{64}", proof["receipt"]["sha256"]) and
                re.fullmatch(r"\.sucoder/wiki/receipts/[0-9a-f]{64}\.json", proof["receipt"]["path"])
                for proof in receipt.get("reused_prefixes", [])))
        if not (normal or reused):
            return False
        cursor = hi
    return cursor == old["byte_end"]


def matches(plan, receipt, fd, *, allow_prefix, incompatible=None):
    """All parent dependencies must match; local unit aliases may change."""
    admitted = {}
    for unit in plan["units"]:
        key = tuple(unit[k] for k in ("source_id", "repository_uuid", "kind", "byte_start"))
        if key in admitted:
            return None  # Ambiguous repeated selections cannot justify reuse.
        admitted[key] = unit
    result, old_ids = {}, set()
    for old in receipt["units"]:
        key = tuple(old[k] for k in ("source_id", "repository_uuid", "kind", "byte_start"))
        unit = admitted.get(key)
        if unit is None or old["unit_id"] in old_ids:
            return None
        old_ids.add(old["unit_id"])
        raw = snapshot(fd, unit)
        origin, cut = old["byte_start"], old["byte_end"]
        if type(cut) is not int or not origin < cut <= unit["byte_end"]:
            if incompatible is not None and old["kind"] in dream_transcript.KINDS:
                incompatible.add(unit["unit_id"])
            return None
        if not allow_prefix and cut != unit["byte_end"]:
            return None
        prefix = raw[:cut - origin]
        valid = digest(prefix) == old["sha256"]
        if old["kind"] in dream_transcript.KINDS:
            valid = (valid and origin == 0 and prefix.endswith(b"\n") and
                receipt["contract"] in ("source-grounded-wiki-v3", "source-grounded-wiki-v4", "source-grounded-wiki-v5") and
                old["thread_id"] == unit["thread_id"] and
                unit.get("normalization") == normalization(raw, unit) and
                old.get("normalization") == normalization(prefix, unit))
        elif old["kind"] == "git":
            valid = valid and cut == unit["byte_end"]
        else:
            valid = False
        if not valid:
            if incompatible is not None and old["kind"] in dream_transcript.KINDS:
                incompatible.add(unit["unit_id"])
            return None
        if not coverage_matches(receipt, old, prefix):
            return None
        result[unit["unit_id"]] = dict(old=old, byte_end=cut)
    if any(chunk["unit_id"] not in old_ids for chunk in receipt["coverage"]):
        return None
    return result


def entry_shape(entry):
    dream._object(entry, ("contract", "unit_id", "parent_unit_id", "byte_start", "byte_end", "sha256", "receipt", "outputs"))
    if entry["contract"] != CONTRACT or entry["byte_start"] != 0 or type(entry["byte_end"]) is not int or entry["byte_end"] <= 0:
        raise DreamError("Invalid accepted prefix range")
    for key in ("unit_id", "parent_unit_id"):
        dream._text(entry[key], key)
    if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
        raise DreamError("Invalid accepted prefix digest")
    parent = entry["receipt"]
    dream._object(parent, ("path", "blob", "sha256", "run_id"))
    if (not re.fullmatch(r"[0-9a-f]{64}", parent["run_id"]) or
            parent["path"] != ".sucoder/wiki/receipts/" + parent["run_id"] + ".json" or
            not re.fullmatch(r"[0-9a-f]{64}", parent["sha256"])):
        raise DreamError("Invalid accepted parent receipt")
    if not isinstance(entry["outputs"], list) or not entry["outputs"]:
        raise DreamError("Accepted prefix needs required output objects")
    for output in entry["outputs"]:
        dream._object(output, ("path", "blob"))
        dream._path(output["path"])
    for obj in [parent, *entry["outputs"]]:
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", obj["blob"]):
            raise DreamError("Invalid accepted proof Git object")


def validate(plan, fd, entries, harness, model, version, purpose, deadline):
    """Replay acceptance at the frozen planning tree using cached objects only."""
    files = {item["path"]: item["blob"] for item in plan["prior_wiki"]["files"]}
    reader = dream_context.Session(plan, fd, "prefix-proof", [], {}, deadline)
    checked, units = {}, set()
    if not isinstance(entries, list) or len(entries) > len(plan["units"]):
        raise DreamError("Invalid accepted prefix manifest")
    for entry in entries:
        entry_shape(entry)
        parent = entry["receipt"]
        if entry["unit_id"] in units or files.get(parent["path"]) != parent["blob"]:
            raise DreamError("Accepted prefix differs from its frozen tree")
        units.add(entry["unit_id"])
        try:
            if parent["sha256"] not in checked:
                raw = dream_run._read(fd, "prefix-parent-" + parent["sha256"] + ".json", dream_run.MAX_PLAN_BYTES)
                dream_context.verify_blob(raw, parent["blob"])
                receipt = dream._json(raw)
                if (digest(raw) != parent["sha256"] or raw != canonical(receipt) + b"\n" or
                        receipt["run_id"] != parent["run_id"] or
                        not dream_progress.envelope(plan, receipt, harness, model, version, purpose)):
                    raise DreamError("Cached parent receipt differs from accepted prefix proof")
                matched = matches(plan, receipt, fd, allow_prefix=True)
                if matched is None:
                    raise DreamError("Accepted prefix dependencies no longer match")
                expected = [dict(path=path, blob=files.get(path)) for path in sorted({output["path"] for output in receipt["outputs"]})]
                if entry["outputs"] != expected:
                    raise DreamError("Accepted prefix required-output inventory differs")
                contents = {output["path"]: reader.blob(output["path"]).decode() for output in expected}
                if not dream_run.output_bindings_match(contents, dream_run._validate_org(contents, deadline), receipt["outputs"]):
                    raise DreamError("Accepted prefix required outputs differ from frozen bindings")
                checked[parent["sha256"]] = matched, expected
            matched, expected = checked[parent["sha256"]]
            old = matched[entry["unit_id"]]["old"]
            if (old["kind"] not in dream_transcript.KINDS or entry["parent_unit_id"] != old["unit_id"] or
                    entry["byte_end"] != old["byte_end"] or entry["sha256"] != old["sha256"] or entry["outputs"] != expected):
                raise DreamError("Accepted prefix manifest differs from its matched source")
        except FileNotFoundError:
            raise DreamError("Accepted prefix proof is incomplete; status cannot fetch missing objects", 3) from None


def discover(plan, fd, source, harness, model, version, purpose, deadline):
    if plan["maintenance"] is not None or not any(unit["kind"] in dream_transcript.KINDS for unit in plan["units"]):
        return []
    files = {item["path"]: item["blob"] for item in plan["prior_wiki"]["files"]}
    chosen, broken, inspected = {}, set(), 0
    for path, blob in sorted(files.items()):
        match = re.fullmatch(r"\.sucoder/wiki/receipts/([0-9a-f]{64})\.json", path)
        if not match:
            continue
        try:
            remaining = dream_run.MAX_PLAN_BYTES - inspected
            if remaining <= 0:
                raise DreamError("Accepted prefix receipt inspection allowance exhausted; use targeted maintenance", 3)
            raw = source.blob(blob, remaining)
            inspected += len(raw)
            receipt = dream._json(raw)
            if (not isinstance(receipt, dict) or receipt.get("run_id") != match[1] or raw != canonical(receipt) + b"\n" or
                    not dream_progress.envelope(plan, receipt, harness, model, version, purpose)):
                continue
            matched = matches(plan, receipt, fd, allow_prefix=True, incompatible=broken)
            if not matched:
                continue
            candidates = {key: value for key, value in matched.items() if value["old"]["kind"] in dream_transcript.KINDS}
            if not dream_run.bound_outputs(source, plan["base_commit"], receipt["outputs"], deadline):
                broken.update(candidates)
                continue
            outputs = [dict(path=name, blob=files[name]) for name in sorted({output["path"] for output in receipt["outputs"]})]
            for key, value in candidates.items():
                entry = dict(contract=CONTRACT, unit_id=key, parent_unit_id=value["old"]["unit_id"], byte_start=0,
                    byte_end=value["byte_end"], sha256=value["old"]["sha256"],
                    receipt=dict(path=path, blob=blob, sha256=digest(raw), run_id=receipt["run_id"]), outputs=outputs)
                if key not in chosen or entry["byte_end"] > chosen[key][0]["byte_end"]:
                    chosen[key] = entry, raw
        except (KeyError, TypeError, UnicodeError):
            continue
        except DreamError as exc:
            if exc.exit_code not in (1, 2):
                raise
    if broken - set(chosen):
        raise DreamError("Accepted transcript prefix or required outputs need reconciliation; use explicit --maintenance and --topic", 3)
    entries = []
    reader = dream_context.Session(plan, fd, "prefix-proof", [], {}, deadline, source)
    for key, (entry, raw) in sorted(chosen.items()):
        dream_run._store(fd, "prefix-parent-" + entry["receipt"]["sha256"] + ".json", raw)
        for output in entry["outputs"]:
            reader.blob(output["path"])
        entries.append(entry)
    validate(plan, fd, entries, harness, model, version, purpose, deadline)
    return entries


def views(sources, entries, context):
    by_id, result = {entry["unit_id"]: entry for entry in entries}, []
    for unit in sources:
        entry = by_id.get(unit["unit_id"])
        if entry is None:
            result.append(unit)
            continue
        cut, end = entry["byte_end"], unit["byte_end"]
        raw, normalized = unit["text"].encode(), unit["provenance"]
        prefix = dream_transcript.normalize(raw[:cut], unit["kind"], unit["thread_id"])
        records = [dict(byte_start=record["byte_start"], byte_end=record["byte_end"], line=record["line"],
            sha256=record["sha256"], timestamp=record["timestamp"], provenance=record["provenance"],
            event_kinds=[event["kind"] for event in record["events"]])
            for record in normalized["records"] if record["byte_start"] >= cut]
        boundary = [dict(call, byte_start=prefix["records"][call["line"] - 1]["byte_start"],
            byte_end=prefix["records"][call["line"] - 1]["byte_end"]) for call in prefix["pending_calls"]]
        view = {key: value for key, value in unit.items() if key not in ("text", "provenance", "chunks", "sha256", "byte_start")}
        view.update(byte_start=cut, text=raw[cut:].decode(), sha256=digest(raw[cut:]), unit_sha256=unit["sha256"],
            accepted_prefix=entry, native_records=records, boundary_pending_calls=boundary,
            prefix_ambiguous_call_ids=prefix["ambiguous_call_ids"], current_pending_calls=normalized["pending_calls"],
            current_orphan_results=normalized["orphan_results"], current_ambiguous_call_ids=normalized["ambiguous_call_ids"],
            session_completion="not_established", independence="not_established")
        view["window_id"] = digest(canonical(["source-window-v1", unit["unit_id"], cut, end, view["sha256"]]))
        context.windows[unit["unit_id"]] = [(cut, end)] if cut < end else []
        result.append(view)
    return result


def coverage(plan, fd, entries):
    cuts = {entry["unit_id"]: entry for entry in entries}
    result = []
    for unit in plan["units"]:
        raw, cursor = snapshot(fd, unit), unit["byte_start"]
        entry = cuts.get(unit["unit_id"])
        if entry:
            cursor = entry["byte_end"]
            result.append(dict(unit_id=unit["unit_id"], byte_start=0, byte_end=cursor, sha256=entry["sha256"],
                extraction="accepted_receipt_reused", verification="validated_output_bindings"))
        for chunk in unit["chunks"]:
            lo, hi = max(cursor, chunk["byte_start"]), chunk["byte_end"]
            if lo < hi:
                result.append(dict(unit_id=unit["unit_id"], byte_start=lo, byte_end=hi,
                    sha256=digest(raw[lo - unit["byte_start"]:hi - unit["byte_start"]]),
                    extraction="author_inline_delivered", verification="whole_candidate_review"))
    return result
