"""V7 exact evidence handles; grounded in dream-authoring-ledger sections 2-6.

The registry indexes delivered bytes, never grants reads or editorial authority.
Historical quote contracts and authenticated Session accounting remain intact.
"""
from __future__ import annotations

import copy
import json
import re
import time

from . import dream_context, dream_diagnostics as diag, dream_roles, dream_support, dream_view, dream_wiki as wiki
from .dream import DreamError, canonical, digest


CONTRACT = "source-grounded-wiki-v7"
POLICY = "delivered-evidence-handles-v1"
MAX_ENTRIES = 2048
SPAN_BYTES = 2048
REGISTRY_BYTES = 512 * 1024
EXPANSION_BYTES = 256 * 1024
BRIEFS = {role: brief.replace(
    "Quotes must occur at exactly one absolute position in the delivered windows of their named source. ", "").replace(
    "Cite each new/revised explanatory section with exact source ranges/quotes in references",
    "Cite each new/revised explanatory section with supplied evidence handles in references") + (
    " Select citations and task-action evidence using only {handle: supplied_handle}. "
    "Do not copy quotes or compute offsets. Initial sources contain ordered passages instead of text; "
    "concatenating their texts recovers the exact delivered source. Context lookups retain exact text "
    "and context.evidence supplies handles with absolute source spans. "
    "A handle proves delivered provenance, not semantic support, human authority, current status or a tool/result association. "
    "Preserve source dates and distinguish native user decisions, agent reports, proposals and observed outcomes. "
    "Missing tool-call context or timing remains unknown. Never infer a full-suite pass or publication from an isolated result."
    ) for role, brief in diag.BRIEFS.items()}


def schema(role):
    value = copy.deepcopy(diag.schema(role))
    if role != "reviewer":
        citation = dream_roles.object_schema(dict(handle=dream_roles.TEXT))
        value["properties"]["references"]["items"]["properties"]["citations"] = dream_roles.array_schema(citation, 16)
        value["properties"]["task_actions"]["items"]["properties"]["evidence"] = dream_roles.array_schema(citation, 16)
    return value


def action_receipt_schema():
    value = copy.deepcopy(diag.schema("author")["properties"]["task_actions"])
    value["items"]["properties"]["evidence"]["items"] = dream_roles.object_schema(dict(
        handle=dream_roles.TEXT, unit_id=dream_roles.TEXT, quote=dream_roles.TEXT,
        byte_start=dict(type="integer", minimum=0), byte_end=dict(type="integer", minimum=0)))
    return value


def check_receipt(receipt):
    """Check canonical handle proofs before reusing an accepted v7 receipt."""
    try:
        snapshot = receipt["context"]["evidence"]
        payload = {key: value for key, value in snapshot.items() if key != "sha256"}
        raw = canonical(payload)
        entries = snapshot["entries"]
        registry_id = snapshot["registry_id"]
        if (set(snapshot) != {"contract", "registry_id", "entries", "sha256"} or snapshot["contract"] != POLICY or
                not re.fullmatch(r"[0-9a-f]{64}", registry_id) or not isinstance(entries, list) or
                len(entries) > MAX_ENTRIES or len(raw) > REGISTRY_BYTES or snapshot["sha256"] != digest(raw) or
                receipt["evidence_registry"] != dict(contract=POLICY, registry_id=registry_id,
                    sha256=snapshot["sha256"], entries=len(entries))):
            raise DreamError("Accepted evidence registry proof differs")
        units = {unit["unit_id"]: unit for unit in receipt["units"]}
        by_handle = {}
        for i, entry in enumerate(entries, 1):
            unit = units.get(entry["unit_id"])
            lo, hi = entry["byte_start"], entry["byte_end"]
            if (set(entry) != {"handle", "unit_id", "byte_start", "byte_end", "sha256", "provenance"} or
                    entry["handle"] != "ev-" + registry_id[:16] + "-" + str(i) or unit is None or
                    type(lo) is not int or type(hi) is not int or
                    not unit["byte_start"] <= lo < hi <= unit["byte_end"] or hi - lo > SPAN_BYTES or
                    not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
                raise DreamError("Accepted evidence registry entry differs")
            by_handle[entry["handle"]] = entry
        expanded = 0
        for owner in [*receipt["references"], *receipt["editorial"]["task_actions"]]:
            for citation in owner["citations" if "citations" in owner else "evidence"]:
                entry = by_handle.get(citation["handle"])
                if (entry is None or any(citation[field] != entry[field] for field in ("unit_id", "byte_start", "byte_end")) or
                        len(citation["quote"].encode()) != entry["byte_end"] - entry["byte_start"] or
                        digest(citation["quote"].encode()) != entry["sha256"]):
                    raise DreamError("Accepted citation differs from its evidence handle")
                expanded += len(canonical(citation))
        if expanded > EXPANSION_BYTES:
            raise DreamError("Accepted evidence citation expansion exceeds its allowance")
    except (KeyError, TypeError, AttributeError, ValueError):
        raise DreamError("Malformed accepted evidence registry proof") from None


class Registry:
    def __init__(self, run_id, sources, deadline):
        self.sources = {unit["unit_id"]: copy.deepcopy(unit) for unit in sources}
        self.deadline = deadline
        identities = [{k: v for k, v in unit.items() if k != "text"} for unit in sources]
        self.registry_id = digest(canonical([POLICY, run_id, identities]))
        self.entries: dict[str, dict] = {}
        self.spans: dict[tuple, str] = {}

    def check_time(self):
        if time.monotonic() >= self.deadline:
            raise DreamError("Evidence registry deadline expired", 3)

    def provenance(self, unit, lo, hi):
        if unit["kind"] == "git":
            return dict(kind="maintained_source")
        if unit["kind"] == dream_view.KIND:
            for block in unit["provenance"]["blocks"]:
                if block["start"] <= lo < hi <= block["end"]:
                    return dict(kind="rendered_transcript", role=block["role"],
                        summary=block["summary"], sidechain=block["sidechain"], raw_offset=block["offset"],
                        tool_association="not_established")
            return dict(kind="render_framing", tool_association="not_established")
        for record in unit["provenance"]["records"]:
            if record["byte_start"] <= lo < hi <= record["byte_end"]:
                return dict(kind="native_transcript_record", timestamp=record["timestamp"],
                    record_sha256=record["sha256"], provenance=record["provenance"],
                    event_kinds=[event["kind"] for event in record["events"]],
                    recorded_roles=[event["payload"].get("role") for event in record["events"] if event["kind"] == "message"],
                    tool_association="not_established")
        return dict(kind="partial_native_record", tool_association="not_established")

    def deliver(self, unit_id, lo, hi):
        """Label one actual delivery; no fetching and no widening to other bytes."""
        self.check_time()
        unit = self.sources[unit_id]
        origin, raw = unit["byte_start"], unit["text"].encode()
        if not origin <= lo <= hi <= unit["byte_end"]:
            raise DreamError("Handle delivery is outside its admitted source", 3)
        try:
            raw[lo - origin:hi - origin].decode()
        except UnicodeError:
            raise DreamError("Handle delivery must use UTF-8 boundaries", 3) from None
        boundaries = {lo, hi}
        if unit["kind"] == dream_view.KIND:
            for block in unit["provenance"]["blocks"]:
                boundaries.update(p for p in (block["start"], block["end"]) if lo < p < hi)
        elif unit["kind"] != "git":
            for record in unit["provenance"]["records"]:
                if record["byte_end"] <= lo or record["byte_start"] >= hi:
                    continue
                boundaries.update(p for p in (record["byte_start"], record["byte_end"]) if lo < p < hi)
                # Reuse the human-source check's literal/JSON text forms. Body
                # boundaries keep long user messages citable without JSON framing.
                record_lo, record_hi = record["byte_start"], record["byte_end"]
                for event in record["events"]:
                    if event["kind"] != "message":
                        continue
                    payload = event["payload"]
                    texts = [payload.get("text"), payload.get("content")]
                    if isinstance(payload.get("content"), list):
                        texts.extend(block.get("text") for block in payload["content"] if isinstance(block, dict))
                    for text in texts:
                        if not isinstance(text, str) or not text:
                            continue
                        for ascii_only in (False, True):
                            literal = json.dumps(text, ensure_ascii=ascii_only)[1:-1].encode()
                            cursor = max(lo, record_lo) - origin
                            while True:
                                self.check_time()
                                found = raw.find(literal, cursor, min(hi, record_hi) - origin)
                                if found < 0:
                                    break
                                boundaries.update((found + origin, found + origin + len(literal)))
                                cursor = found + max(1, len(literal))
        result = []
        cuts = sorted(boundaries)
        for start, end in zip(cuts, cuts[1:]):
            # Paragraphs retain blank lines; long paragraphs split at UTF-8 boundaries.
            position, paragraph = start, []
            parts = []
            for line in raw[start - origin:end - origin].splitlines(keepends=True):
                self.check_time()
                paragraph.append(line)
                if not line.strip():
                    parts.append(b"".join(paragraph))
                    paragraph = []
            if paragraph:
                parts.append(b"".join(paragraph))
            for part in parts:
                while part:
                    self.check_time()
                    text = part[:SPAN_BYTES].decode("utf-8", errors="ignore")
                    size = len(text.encode())
                    stop = position + size
                    key = unit_id, position, stop
                    if key not in self.spans:
                        if len(self.entries) >= MAX_ENTRIES:
                            raise DreamError("Evidence handle entry allowance exhausted", 3)
                        handle = "ev-" + self.registry_id[:16] + "-" + str(len(self.entries) + 1)
                        self.spans[key] = handle
                        self.entries[handle] = dict(handle=handle, unit_id=unit_id, byte_start=position, byte_end=stop,
                            sha256=digest(text.encode()), provenance=self.provenance(unit, position, stop))
                    handle = self.spans[key]
                    result.append(dict(self.entries[handle], text=text))
                    position, part = stop, part[size:]
        self.snapshot()
        return result

    def snapshot(self):
        value = dict(contract=POLICY, registry_id=self.registry_id, entries=list(self.entries.values()))
        raw = canonical(value)
        if len(raw) > REGISTRY_BYTES:
            raise DreamError("Evidence handle registry byte allowance exhausted", 3)
        return copy.deepcopy(dict(value, sha256=digest(raw)))

    def resolve(self, citation):
        self.check_time()
        entry = self.entries.get(citation.get("handle"))
        if entry is None:
            raise DreamError("Unknown, stale or undelivered evidence handle")
        unit = self.sources[entry["unit_id"]]
        lo, hi = entry["byte_start"], entry["byte_end"]
        text = unit["text"].encode()[lo - unit["byte_start"]:hi - unit["byte_start"]].decode()
        if digest(text.encode()) != entry["sha256"]:
            raise DreamError("Evidence handle source bytes changed", 3)
        item = dict(handle=entry["handle"], unit_id=entry["unit_id"], quote=text, byte_start=lo, byte_end=hi)
        dream_roles.citations([item], unit, unit, unit["text"].encode())
        return item


class Session(dream_context.Session):
    evidence: Registry

    def check_cached(self, response):
        if response.get("state") == "unavailable" and (
                set(response) != {"state", "error", "delivered_bytes", "searched_bytes", "read_id"} or
                not isinstance(response.get("error"), str) or
                any(type(response.get(field)) is not int or response[field] != 0
                    for field in ("delivered_bytes", "searched_bytes"))):
            raise DreamError("Malformed unavailable context response", 3)

    def lookup(self, request):
        response = super().lookup(request)
        if request["kind"] in ("source", "search") and response["state"] in ("complete", "incomplete"):
            for excerpt in response.get("excerpts", []):
                unit_id, lo, hi = excerpt["unit_id"], excerpt["byte_start"], excerpt["byte_end"]
                unit = self.evidence.sources[unit_id]
                raw = unit["text"].encode()[lo - unit["byte_start"]:hi - unit["byte_start"]]
                if (not any(start <= lo < hi <= end for start, end in self.windows[unit_id]) or
                        raw != excerpt["text"].encode() or digest(raw) != excerpt["sha256"]):
                    raise DreamError("Handle lookup differs from authenticated delivered windows", 3)
                self.evidence.deliver(unit_id, lo, hi)
        return response

    def payload(self):
        return dict(super().payload(), evidence=self.evidence.snapshot())


class Editor(diag.Editor):
    context_type = Session
    context: Session

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract = CONTRACT
        self.editing = dict(self.editing, citation_selection="Select supplied evidence handles; Python supplies exact quotes and UTF-8 ranges.")
        self.run_id = digest(canonical([CONTRACT, self.run_id, BRIEFS, self.editing,
            {role: schema(role) for role in BRIEFS}, POLICY, MAX_ENTRIES, SPAN_BYTES, REGISTRY_BYTES, EXPANSION_BYTES]))

    def briefs(self):
        return dream_support.briefs(BRIEFS) if self.plan.get("editorial_version") == 2 else BRIEFS

    def result_schema(self, role):
        return schema(role)

    def input_sources(self, sources):
        delivered = super().input_sources(sources)
        self.context.evidence = Registry(self.run_id, sources, self.deadline)
        result = []
        for unit in delivered:
            passages = self.context.evidence.deliver(unit["unit_id"], unit["byte_start"], unit["byte_end"])
            value = dict({k: v for k, v in unit.items() if k not in ("text", "provenance")},
                passages=[dict(handle=p["handle"], text=p["text"]) for p in passages])
            if unit["kind"] not in ("git", dream_view.KIND) and "provenance" in unit:
                # Keep pending/orphan/ambiguous-call metadata without duplicating
                # raw records. Prefix views already supply their scoped metadata.
                value["native_context"] = {k: v for k, v in unit["provenance"].items() if k != "records"}
            result.append(value)
        return result

    def proposal(self, proposal, prior, sources):
        # Keep saved model replies immutable; normalize only the private materialization.
        value = copy.deepcopy(proposal)
        report, expanded = diag.Report(), 0
        for owner in [*value["references"], *value["task_actions"]]:
            field = "citations" if "citations" in owner else "evidence"
            for i, citation in enumerate(owner[field]):
                try:
                    resolved = self.context.evidence.resolve(citation)
                except DreamError as exc:
                    if exc.exit_code != 2:
                        raise
                    # Reference defects join the structural/coverage pass below.
                    # Actions need resolved evidence before editorial materialization.
                    if field == "evidence":
                        report.add("citation_handle", str(exc), path=owner["path"], target=owner["target"])
                    continue
                expanded += len(canonical(resolved))
                if expanded > EXPANSION_BYTES:
                    raise DreamError("Expanded evidence citation byte allowance exhausted", 3)
                owner[field][i] = resolved
        for action in value["task_actions"]:
            if action["authority"] == "recorded_human" and not any(
                    "unit_id" in citation and self._human_source(citation, sources) for citation in action["evidence"]):
                report.add("action_authority", "Recorded human authority needs the action's own native user evidence or maintained decision record.",
                    path=action["path"], target=action["target"])
        report.check()
        return super().proposal(value, prior, sources)

    def validate(self, proposal, prior, sources):
        report = diag.Report()
        expanded = sum(len(canonical(citation)) for action in proposal["task_actions"] for citation in action["evidence"])
        def resolve(citation):
            nonlocal expanded
            item = self.context.evidence.resolve(citation)
            expanded += len(canonical(item))
            if expanded > EXPANSION_BYTES:
                raise DreamError("Expanded evidence citation byte allowance exhausted", 3)
            return item
        def check(proposal, pages, parsed, old, sources, derived):
            return diag.references(proposal, pages, parsed, old, sources, derived,
                self.context.windows, self.deadline, self.duplicates, report,
                citation_resolver=resolve)
        return wiki.validate_proposal(self.plan, proposal, self.validation_prior, sources, self.deadline,
            wiki.LEGACY_CONTRACT, reference_validator=check, report_issue=report.add, **self.permissions)

    def prepare(self):
        pages, receipt = super().prepare()
        snapshot = self.context.evidence.snapshot()
        receipt["evidence_registry"] = dict(contract=POLICY, registry_id=snapshot["registry_id"],
            sha256=snapshot["sha256"], entries=len(snapshot["entries"]))
        check_receipt(receipt)
        return pages, receipt
