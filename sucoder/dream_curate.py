"""Agent curation and checked no-op convergence (implementation ledger 3-6).

Wiki-v1/v2 remain the original writer contracts. V3 adds a disposition and
reason, using the same bounded author/reviewer/repair and publication machinery.
"""
from __future__ import annotations

from . import dream, dream_roles, dream_run, dream_wiki as wiki
from .dream import DreamError, canonical, digest


CONTRACT = "source-grounded-wiki-v3"
BRIEFS = dict(wiki.BRIEFS)
BRIEFS["author"] += (" You may return decision unchanged when the useful knowledge is already represented or no worthwhile edit is supported. "
    "Explain the reason and material omissions; unchanged requires empty pages, changes and references. Otherwise return decision change and the full proposed edits. "
    "Do not create a duplicate topic just to demonstrate activity. Explicit maintenance may conclude unchanged.")
BRIEFS["reviewer"] += (" Review an unchanged decision against the complete evidence and prior wiki too: reject it if it leaves a material wrong note, omission or unresolved necessary repair. "
    "Accept a reasoned no-op when no useful supported change is needed; no output does not mean the selected evidence was unread.")
BRIEFS["repair"] += (" Return decision change or unchanged with a reason. An unchanged result must have no proposed pages, changes or references; "
    "address a blocking finding about a mistaken no-op by making the supported correction.")


def schema(role):
    original = wiki.schema(role)
    if role == "reviewer":
        return original
    fields = dict(original["properties"], decision=dict(type="string", enum=["change", "unchanged"]), rationale=dream_roles.TEXT)
    return dream_roles.object_schema(fields)


class Curator(wiki.WikiCoordinator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, contract=CONTRACT, **kwargs)
        self.run_id = digest(canonical([CONTRACT, self.run_id, BRIEFS, {role: schema(role) for role in BRIEFS}]))
        self.accepted_proposal: dict | None = None

    def result_schema(self, role):
        return schema(role)

    def briefs(self):
        return BRIEFS

    def proposal(self, proposal, prior, sources):
        if not proposal["rationale"].strip():
            raise DreamError("A curation decision needs a reason")
        if proposal["decision"] == "change":
            value = super().proposal(proposal, prior, sources)
        else:
            if proposal["pages"] or proposal["changes"] or proposal["references"]:
                raise DreamError("An unchanged decision cannot propose page edits or references")
            ids = {unit["unit_id"] for unit in sources}
            for omission in proposal["omissions"]:
                if omission["unit_id"] not in ids or not omission["reason"].strip():
                    raise DreamError("An omission needs an admitted source and a reason")
            value = {}, wiki.inventory(prior, self.deadline) if prior else {}, []
        self.accepted_proposal = proposal
        return value

    def bindings(self, pages, prior):
        return super().bindings(pages or prior, prior)

    def prepare(self):
        pages, receipt = super().prepare()
        assert self.accepted_proposal is not None
        receipt.update(decision=self.accepted_proposal["decision"], rationale=self.accepted_proposal["rationale"])
        units = {unit["unit_id"]: unit for unit in self.plan["units"]}
        for unit in receipt["units"]:
            if "normalization" in units[unit["unit_id"]]:
                unit["normalization"] = dict(units[unit["unit_id"]]["normalization"])
        return pages, receipt


def unchanged_value(plan, receipt):
    return dict(schema_version=1, operation="dream_run", state="unchanged", plan_id=plan["plan_id"],
        run_id=receipt["run_id"], harness=receipt["harness"], model=receipt["model"], native_version=receipt["native_version"],
        fidelity=receipt["contract"], base_commit=plan["base_commit"], candidate_commit=None, candidate_ref=None,
        rationale=receipt["rationale"], coverage=receipt["coverage"], usage=receipt["usage"],
        integration=dict(state="no_changes_proposed", evidence=dict(inspected_commit=plan["base_commit"])))


def record_unchanged(plan, fd, sources, receipt, deadline):
    from . import dream_author
    if receipt["contract"] not in (CONTRACT, dream_author.CONTRACT) or receipt["decision"] != "unchanged":
        raise DreamError("Expected a checked unchanged curation result")
    source = sources[plan["repository"]]
    if receipt["outputs"] and not dream_run.bound_outputs(source, plan["base_commit"], receipt["outputs"], deadline):
        raise DreamError("Unchanged wiki footprint differs from the inspected base")
    for item in sources.values():
        item.check_binding()
    if source.commit(plan["integration_ref"]) != plan["base_commit"]:
        raise DreamError("Integration ref changed during no-op validation; re-plan", 3)
    prefix = "native-" + receipt["run_id"]
    dream_run._store(fd, prefix + "-receipt.json", canonical(receipt) + b"\n")
    dream_run._store(fd, prefix + "-unchanged.json", canonical(dict(run_id=receipt["run_id"], receipt_sha256=digest(canonical(receipt)))) + b"\n")
    result = unchanged_value(plan, receipt)
    dream_run._store(fd, prefix + "-run.json", canonical(result) + b"\n")
    return result


def unchanged_status(config, mirror, plan, fd, run_id, marker, deadline):
    receipt = wiki.restore(plan, fd, run_id, deadline)
    expected_marker = dict(run_id=run_id, receipt_sha256=digest(canonical(receipt)))
    if receipt.get("decision") != "unchanged" or canonical(marker) != canonical(expected_marker):
        raise DreamError("Private no-op marker differs from its checked role results")
    prefix = "native-" + run_id
    if dream_run._read(fd, prefix + "-receipt.json", dream_run.MAX_PLAN_BYTES) != canonical(receipt) + b"\n":
        raise DreamError("Private no-op receipt differs from its checked role results")
    expected = unchanged_value(plan, receipt)
    try:
        recorded = dream._json(dream_run._read(fd, prefix + "-run.json", dream_run.MAX_PLAN_BYTES))
    except FileNotFoundError:
        pass  # Interruption after atomic no-op marker publication.
    else:
        if canonical(recorded) != canonical(expected):
            raise DreamError("Stored no-op result differs from its frozen role plan")
    source = dream_run._sources(config, plan, deadline)[mirror]
    base = source.commit(plan["integration_ref"])
    tree = lambda commit: source.git("rev-parse", "--verify", "--end-of-options", commit + ":.sucoder/wiki", missing=True)
    valid = tree(base) == tree(plan["base_commit"])
    if receipt["outputs"]:
        valid = valid and dream_run.bound_outputs(source, base, receipt["outputs"], deadline)
    # Without an accepted registry, keep the frozen proposed ID instead of
    # generating a fresh UUID as evidence that the wiki context was altered.
    if not plan["repositories"][mirror]["provisional"]:
        current_identity = dream._repository_uuid(source, base)
        valid = valid and not current_identity["provisional"] and current_identity["repository_uuid"] == plan["repository_uuid"]
    source.check_binding()
    if source.commit(plan["integration_ref"]) != base:
        raise DreamError("Integration ref changed during no-op status; retry", 3)
    return dict(expected, operation="dream_status", state="unchanged" if valid else "needs_reconciliation",
        candidate_status="not_created", integration=dict(state="no_changes_proposed" if valid else "needs_reconciliation",
            evidence=dict(inspected_commit=plan["base_commit"], integration_commit=base)))
