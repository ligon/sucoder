"""Explicit semantic review, grounded in dream-authoring-ledger sections 2-6.

The matrix records judgments, not proofs of truth. Exact evidence selection,
editorial authority, bounded repair and publication remain the v7 machinery.
"""
from __future__ import annotations

import copy
import difflib
import re

from . import dream_handles as handles, dream_roles, dream_support, dream_wiki as wiki
from .dream import DreamError, canonical, digest


CONTRACT = "source-grounded-wiki-v8"
POLICY = "page-semantic-review-v1"
MAX_UNITS = 128
DIFF_BYTES = 128 * 1024
GATE_BYTES = 256 * 1024
CATEGORIES = ("support", "scope", "time", "causality", "outcomes", "attribution", "preservation", "summary")
BRIEFS = dict(handles.BRIEFS)
BRIEFS["reviewer"] += (
    " The coordinator supplies candidate.review_gate: a complete list of changed output pages, including the index, "
    "with before/after hashes and exact unified differences. It is navigation data, not evidence or a verdict. "
    "Inspect EVERY unit against the supplied sources, prior pages and full candidate. Return exactly one page_checks row "
    "per unit_id, with a concrete reason for EACH category; choose fail or uncertain whenever support is insufficient. "
    "support: identify the evidence for new factual claims and distinguish a report from a demonstrated fact. "
    "scope: preserve project, host, sample and experiment boundaries; one observation cannot establish fleet-wide behavior. "
    "time: preserve evidence dates and proposed/then/current distinctions; an old report cannot establish present state. "
    "causality: check each asserted cause independently; several failed attempts may have different causes. "
    "outcomes: absence of a recorded result is unknown, not failure, nonoccurrence or proof of completion/publication; "
    "aggregate outcomes do not identify the outcome of a particular process without linkage. "
    "attribution: preserve who observed or claimed what, project identities and table-row provenance; do not assume "
    "missing table columns are recoverable from prose without finding that exact identity in the prose. "
    "preservation: inspect removed caveats, counterexamples, uncertainty and source detail, not just added statements. "
    "summary: independently check index blurbs, titles, tables and summaries against their sources and detailed pages; "
    "an accurate topic page cannot rescue an overclaim in the index. Evidence handles may point to supplied passages; "
    "their presence alone proves no semantic judgment. Explain categories without applicable claims explicitly. "
    "An unchanged unit still requires reviewing the no-change disposition against useful omissions and prior errors. "
    "Any failed or uncertain check blocks acceptance regardless of verdict or finding severity. "
    "For lookup return page_checks=[] and defer all final checks until the requested context arrives."
)


def schema(role):
    value = copy.deepcopy(handles.schema(role))
    if role == "reviewer":
        check = dream_roles.object_schema(dict(
            status=dict(type="string", enum=["pass", "fail", "uncertain"]),
            reason=dict(type="string", maxLength=512),
            evidence=dream_roles.array_schema(dream_roles.object_schema(dict(handle=dream_roles.TEXT)), 4)))
        row = dream_roles.object_schema(dict(unit_id=dream_roles.TEXT,
            **{name: copy.deepcopy(check) for name in CATEGORIES}))
        value["properties"]["page_checks"] = dream_roles.array_schema(row, MAX_UNITS)
        value["required"].append("page_checks")
    return value


def manifest(run_id, proposal, pages, prior):
    """Coordinator-owned, bounded diff coverage (ledger sections 4-5)."""
    before = {path: digest(text.encode()) for path, text in sorted(prior.items())}
    candidate = digest(canonical([POLICY, run_id, proposal, before, pages]))
    units, used = [], 0
    if len(pages) > MAX_UNITS:
        raise DreamError("Semantic review page allowance exhausted", 3)
    for i, (path, text) in enumerate(sorted(pages.items()), 1):
        diff = "".join(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
            for line in difflib.unified_diff(prior.get(path, "").splitlines(keepends=True),
                text.splitlines(keepends=True), fromfile="before/" + path, tofile="after/" + path))
        used += len(diff.encode())
        if used > DIFF_BYTES:
            raise DreamError("Semantic review diff allowance exhausted", 3)
        units.append(dict(unit_id="change-" + candidate[:16] + "-" + str(i), path=path,
            kind="index" if path == wiki.INDEX else "page", before_sha256=before.get(path, ""),
            after_sha256=digest(text.encode()), diff=diff))
    if not units:
        sha = digest(canonical(before))
        units.append(dict(unit_id="change-" + candidate[:16] + "-1", path="", kind="unchanged",
            before_sha256=sha, after_sha256=sha, diff=""))
    value = dict(contract=POLICY, candidate_sha256=candidate, prior=before, units=units)
    if len(canonical(value)) > GATE_BYTES:
        raise DreamError("Semantic review manifest allowance exhausted", 3)
    return dict(value, sha256=digest(canonical(value)))


def findings(review, gate, available):
    """An accepting verdict cannot override incomplete/failed semantic checks."""
    result = []
    def block(message, path=""):
        result.append(dict(severity="blocking", path=path, message=message))
    rows = review["page_checks"]
    expected = {unit["unit_id"]: unit for unit in gate["units"]}
    if len(rows) != len(expected) or {row["unit_id"] for row in rows} != set(expected):
        block("Semantic review must check each current page unit exactly once")
        return result
    for row in rows:
        path = expected[row["unit_id"]]["path"]
        for name in CATEGORIES:
            check = row[name]
            if not check["reason"].strip():
                block("Semantic " + name + " check needs a concrete reason", path)
            if any(citation["handle"] not in available for citation in check["evidence"]):
                block("Semantic " + name + " check names an undelivered evidence handle", path)
            if check["status"] != "pass":
                block("Semantic " + name + " " + check["status"] + ": " + check["reason"], path)
    return result


def check_receipt(receipt, *, reviewer_schema=None):
    """Bounded shared proof consistency; replay still reconstructs assignments."""
    handles.check_receipt(receipt)
    try:
        gate = receipt["semantic_review"]
        payload = {key: value for key, value in gate.items() if key != "sha256"}
        units = gate["units"]
        if (set(gate) != {"contract", "candidate_sha256", "prior", "units", "sha256"} or gate["contract"] != POLICY or
                not re.fullmatch(r"[0-9a-f]{64}", gate["candidate_sha256"]) or
                gate["sha256"] != digest(canonical(payload)) or len(canonical(payload)) > GATE_BYTES or
                not isinstance(units, list) or not 1 <= len(units) <= MAX_UNITS):
            raise DreamError("Accepted semantic review manifest differs")
        outputs = {row["path"]: row["sha256"] for row in receipt["outputs"]}
        changes = {row["path"]: row["before_sha256"] for row in receipt["changes"]}
        for path, sha in gate["prior"].items():
            if (not path.startswith(".sucoder/wiki/") or not path.endswith(".org") or
                    not re.fullmatch(r"[0-9a-f]{64}", sha) or
                    sha != (changes[path] if path in changes else outputs.get(path))):
                raise DreamError("Accepted semantic review prior binding differs")
        paths, diff_bytes = set(), 0
        for i, unit in enumerate(units, 1):
            if (set(unit) != {"unit_id", "path", "kind", "before_sha256", "after_sha256", "diff"} or
                    unit["unit_id"] != "change-" + gate["candidate_sha256"][:16] + "-" + str(i) or
                    unit["path"] in paths or not isinstance(unit["diff"], str) or
                    not re.fullmatch(r"[0-9a-f]{64}", unit["after_sha256"]) or
                    (unit["before_sha256"] and not re.fullmatch(r"[0-9a-f]{64}", unit["before_sha256"]))):
                raise DreamError("Accepted semantic review unit differs")
            paths.add(unit["path"])
            diff_bytes += len(unit["diff"].encode())
            if receipt["decision"] == "unchanged":
                # Reviewer lookups can add full frozen pages to the final
                # dependency footprint after this disposition unit was made.
                sha = digest(canonical(gate["prior"]))
                if unit != dict(unit_id=unit["unit_id"], path="", kind="unchanged",
                                before_sha256=sha, after_sha256=sha, diff="") or len(units) != 1:
                    raise DreamError("Accepted unchanged semantic review differs")
            elif (unit["kind"] != ("index" if unit["path"] == wiki.INDEX else "page") or
                    outputs.get(unit["path"]) != unit["after_sha256"] or
                    unit["before_sha256"] != changes.get(unit["path"], "") or
                    (unit["path"] not in changes and (unit["path"] != wiki.SETTINGS_PATH or
                        unit["after_sha256"] != digest(wiki.SETTINGS.encode())))):
                raise DreamError("Accepted semantic review output binding differs")
        expected = set(changes)
        if receipt["decision"] != "unchanged" and paths != expected and paths != expected | {wiki.SETTINGS_PATH}:
            raise DreamError("Accepted semantic review omits changed output pages")
        if diff_bytes > DIFF_BYTES:
            raise DreamError("Accepted semantic review diff allowance differs")
        dream_roles.validate_schema(receipt["review"], schema("reviewer") if reviewer_schema is None else reviewer_schema)
        if receipt["review"]["requests"] or findings(receipt["review"], gate,
                {entry["handle"] for entry in receipt["context"]["evidence"]["entries"]}):
            raise DreamError("Accepted semantic review contains unresolved checks")
    except DreamError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError):
        raise DreamError("Malformed accepted semantic review proof") from None


class Editor(handles.Editor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract = CONTRACT
        self.run_id = digest(canonical([CONTRACT, self.run_id, BRIEFS,
            {role: schema(role) for role in BRIEFS}, POLICY, MAX_UNITS, DIFF_BYTES, GATE_BYTES]))
        self.review_gate: dict = {}

    def briefs(self):
        return dream_support.briefs(BRIEFS) if self.plan.get("editorial_version") == 2 else BRIEFS

    def result_schema(self, role):
        return schema(role)

    def check_result(self, task, result):
        super().check_result(task, result)
        if task["role"] == "reviewer" and result["verdict"] == "lookup" and result["page_checks"]:
            raise DreamError("A reviewer lookup cannot contain premature semantic checks")

    def review_candidate(self, proposal, pages):
        candidate = super().review_candidate(proposal, pages)
        self.review_gate = manifest(self.run_id, candidate, pages, dict(self.context.prior, **self.extra_prior))
        return dict(candidate, review_gate=copy.deepcopy(self.review_gate))

    def review_findings(self, review, proposal):
        return super().review_findings(review, proposal) + findings(review, self.review_gate,
            set(self.context.evidence.entries))

    def prepare(self):
        pages, receipt = super().prepare()
        receipt["semantic_review"] = copy.deepcopy(self.review_gate)
        self.receipt_check(receipt)
        return pages, receipt

    def receipt_check(self, receipt):
        check_receipt(receipt)
