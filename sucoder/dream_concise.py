"""V9 concise review; reuses campaign ledger sections 2-5 and byte accounting."""
from __future__ import annotations

import copy

from . import dream_review, dream_roles, dream_support
from .dream import canonical, digest


CONTRACT = "source-grounded-wiki-v9"
POLICY = "concise-semantic-review-v1"
BRIEFS = dict(dream_review.BRIEFS)
BRIEFS["reviewer"] += (
    " Keep each check reason to one short sentence, normally at most 40 words. "
    "Point to the specific assertion or missing qualification and use supplied handles instead of copying paragraphs. "
    "Do not repeat a narrative for every category. State concrete failures precisely; concise wording must not hide "
    "uncertainty or omit a check. The existing cumulative max_proposal_bytes allowance still bounds all role replies."
)


def schema(role):
    value = copy.deepcopy(dream_review.schema(role))
    if role == "reviewer":
        row = value["properties"]["page_checks"]["items"]["properties"]
        for name in dream_review.CATEGORIES:
            row[name]["properties"]["reason"] = copy.deepcopy(dream_roles.TEXT)
    return value


def check_receipt(receipt):
    dream_review.check_receipt(receipt, reviewer_schema=schema("reviewer"))


class Editor(dream_review.Editor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract = CONTRACT
        self.run_id = digest(canonical([CONTRACT, self.run_id, BRIEFS, POLICY,
            {role: schema(role) for role in BRIEFS}]))

    def briefs(self):
        return dream_support.briefs(BRIEFS) if self.plan.get("editorial_version") == 2 else BRIEFS

    def result_schema(self, role):
        return schema(role)

    def receipt_check(self, receipt):
        check_receipt(receipt)
