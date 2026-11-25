"""V6 complete reference diagnostics (dream-authoring-ledger, sections 2-7).

Structural validation and editorial authority remain prerequisites. Historical
editors retain their original policies, identities and first-error behavior.
"""
from __future__ import annotations

import time

from . import dream_editor, dream_roles, dream_support, dream_wiki as wiki
from .dream import DreamError, canonical, digest


CONTRACT = "source-grounded-wiki-v6"
MAX_FINDINGS = 32
POLICY = "located-reference-diagnostics-v1"
BRIEFS = {role: brief + (
    " Every new or revised CUSTOM_ID section outside the index needs a source reference, including Talk and questions. "
    "Only unchanged sections and coordinator-verified derived editorial changes are exempt. "
    "Quotes must occur at exactly one absolute position in the delivered windows of their named source. "
    "Mechanical findings have codes and path/target/unit locations; fix every finding together. "
    "A diagnostics_incomplete finding means further defects may remain. "
    "Repair returns a complete proposal, including all still-needed references and dispositions."
    ) for role, brief in dream_editor.BRIEFS.items()}
schema = dream_editor.schema


class InvalidProposal(DreamError):
    def __init__(self, findings):
        self.findings = findings
        super().__init__("; ".join(f["message"] for f in findings))


class Report:
    def __init__(self):
        self.findings: list[dict] = []
        self.incomplete = False

    def add(self, code, message, *, path="", target="", unit_id="", **details):
        if len(self.findings) >= MAX_FINDINGS:
            self.incomplete = True
            return
        clipped = any(len(value) > 256 for value in (path, target, unit_id))
        self.incomplete = self.incomplete or clipped
        location = (path + ("#" + target if target else ""))[:512]
        self.findings.append(dict(severity="blocking", code=code, path=path[:256], target=target[:256],
            unit_id=unit_id[:256], message=(code + (" " + location if location else "") +
                (" source " + unit_id[:128] if unit_id else "") + ": " + message)[:1024], location_truncated=clipped, **details))

    def check(self):
        if self.incomplete:
            marker = dict(severity="blocking", code="diagnostics_incomplete", path="", target="", unit_id="",
                message="diagnostics_incomplete: finding/field limit reached; additional diagnostic details remain unreported.")
            if len(self.findings) < MAX_FINDINGS:
                self.findings.append(marker)
            else:
                self.findings[-1] = marker
        if self.findings:
            raise InvalidProposal(self.findings)


def references(proposal, pages, parsed, old, sources, derived_targets, windows, deadline, duplicates=(), report=None, *, citation_resolver=None):
    """Collect only checks whose structural prerequisites have succeeded."""
    report = report if report is not None else Report()
    for path, target in duplicates:
        report.add("reference_duplicate", "Consolidate this page/target into one reference retaining necessary citations.", path=path, target=target)
    units = {s["unit_id"]: s for s in sources}
    declared, resolved = set(), []
    for reference in proposal["references"]:
        if time.monotonic() >= deadline:
            raise DreamError("Reference validation deadline expired", 3)
        path, target = reference["path"], reference["target"]
        key = path, target
        if key in declared:
            report.add("reference_duplicate", "Use one reference per page/target.", path=path, target=target)
        declared.add(key)
        if path not in pages or target not in parsed.get(path, {}).get("targets", {}):
            report.add("reference_target", "Reference must name an anchor in a changed page.", path=path, target=target)
        if not reference["citations"]:
            report.add("reference_empty", "Supply at least one exact source citation.", path=path, target=target)
        citations = []
        for citation in reference["citations"]:
            if citation_resolver is not None:
                try:
                    citations.append(citation_resolver(citation))
                except DreamError as exc:
                    if exc.exit_code != 2:
                        raise
                    report.add("citation_handle", str(exc), path=path, target=target)
                continue
            unit_id = citation["unit_id"]
            unit = units.get(unit_id)
            if unit is None:
                report.add("citation_unit", "Name an admitted source unit.", path=path, target=target, unit_id=unit_id)
                continue
            raw, quote = unit["text"].encode(), citation["quote"].encode()
            positions: set[int] = set()
            if quote:
                for lo, hi in windows[unit_id]:
                    if time.monotonic() >= deadline:
                        raise DreamError("Reference validation deadline expired", 3)
                    cursor = lo - unit["byte_start"]
                    while len(positions) < 2:
                        found = raw.find(quote, cursor, hi - unit["byte_start"])
                        if found < 0:
                            break
                        positions.add(found + unit["byte_start"])
                        cursor = found + 1
                    if len(positions) == 2:
                        break
            if len(positions) != 1:
                report.add("citation_quote", "Choose a nonempty exact quote occurring once in delivered windows; " +
                    ("at least two positions match." if positions else "no position matches."),
                    path=path, target=target, unit_id=unit_id, match_count=len(positions),
                    match_count_is_lower_bound=len(positions) == 2)
                continue
            start = positions.pop()
            item = dict(citation, byte_start=start, byte_end=start + len(quote))
            dream_roles.citations([item], unit, unit, raw)
            citations.append(item)
        resolved.append(dict(reference, citations=citations))
    for path in pages:
        for target, sha in parsed[path]["targets"].items():
            unchanged = old.get(path, {}).get("targets", {}).get(target) == sha
            if path != wiki.INDEX and not unchanged and (path, target) not in declared and (path, target) not in derived_targets:
                report.add("reference_missing", "Add a source reference for this new or revised anchored section, including Talk.", path=path, target=target)
        if path not in (wiki.INDEX, wiki.SETTINGS_PATH) and not parsed[path]["targets"]:
            report.add("anchor_missing", "Topic pages need stable CUSTOM_ID anchors.", path=path)
    for omission in proposal["omissions"]:
        if omission["unit_id"] not in units or not omission["reason"].strip():
            report.add("omission_invalid", "An omission needs an admitted source unit and a reason.", unit_id=omission["unit_id"])
    report.check()
    return resolved


class Editor(dream_editor.Editor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract = CONTRACT
        editing: dict = dict(self.editing)
        editing.update(reference_coverage="Every changed non-index CUSTOM_ID, including Talk, needs a reference unless unchanged or coordinator-derived.",
            diagnostics=dict(contract=POLICY, max_findings=MAX_FINDINGS))
        self.editing = editing
        self.run_id = digest(canonical([CONTRACT, self.run_id, BRIEFS, self.editing, POLICY, MAX_FINDINGS]))
        self.duplicates: list[tuple[str, str]] = []

    def briefs(self):
        return dream_support.briefs(BRIEFS) if self.plan.get("editorial_version") == 2 else BRIEFS

    def reference_map(self, values):
        refs: dict = {}
        self.duplicates = []
        for reference in values:
            key = reference["path"], reference["target"]
            if key in refs:
                self.duplicates.append(key)
                refs[key]["citations"].extend(reference["citations"])
            else:
                refs[key] = dict(reference, citations=list(reference["citations"]))
        return refs

    def resolve_references(self, values, sources):
        # Resolve only after structural validation, together with coverage.
        return values

    def validate(self, proposal, prior, sources):
        report = Report()
        def check(proposal, pages, parsed, old, sources, derived):
            return references(proposal, pages, parsed, old, sources, derived,
                self.context.windows, self.deadline, self.duplicates, report)
        return wiki.validate_proposal(self.plan, proposal, self.validation_prior, sources, self.deadline,
            wiki.LEGACY_CONTRACT, reference_validator=check, report_issue=report.add, **self.permissions)

    def proposal(self, proposal, prior, sources):
        try:
            return super().proposal(proposal, prior, sources)
        except DreamError as exc:
            if exc.exit_code != 2 or isinstance(exc, InvalidProposal):
                raise
            report = Report()
            report.add("validation_blocked", str(exc) + "; dependent reference/coverage validation was blocked.")
            report.check()
