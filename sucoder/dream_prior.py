"""Deterministic bounded prior context, grounded in ledger sections 2-6.

Associations select context, not accepted progress. Omitted pages remain in
the frozen inventory and cannot be overwritten as if they were new pages.
"""
from __future__ import annotations

import re

from . import dream
from .dream import DreamError, canonical


CONTRACT = "bounded-prior-v1"
LEGACY = "all-active-v1"
INDEX = ".sucoder/wiki/index.org"
SETTINGS = ".sucoder/wiki/settings.org"
TOPICS = ".sucoder/wiki/topics/"
RECEIPT_BYTES = 2 * 1024 * 1024
CATALOG_BYTES = 8192


def active(path):
    return path.endswith(".org") and not path.startswith(".sucoder/wiki/archive/")


def associations(source, files, units):
    source_ids = {unit["source_id"] for unit in units}
    paths = {item["path"] for item in files if item["path"].startswith(TOPICS) and active(item["path"])}
    selected, used = set(), 0
    for item in files:
        if not re.fullmatch(r"\.sucoder/wiki/receipts/[0-9a-f]{64}\.json", item["path"]):
            continue
        remaining = RECEIPT_BYTES - used
        if remaining <= 0:
            break  # A hint scan is incomplete, never falsely accepted coverage.
        if int(source.git("cat-file", "-s", item["blob"]).strip()) > remaining:
            continue  # Oversized metadata cannot make a relevance hint mandatory.
        try:
            raw = source.blob(item["blob"], remaining)
            used += len(raw)
            receipt = dream._json(raw)
            if not isinstance(receipt, dict) or not any(u["source_id"] in source_ids for u in receipt.get("units", [])):
                continue
            # A human may have edited the page since this association was
            # recorded. Select its current frozen bytes; do not infer reuse.
            selected.update(o["path"] for o in receipt.get("outputs", []) if o["path"] in paths)
        except (KeyError, TypeError):
            continue
        except DreamError as exc:
            if exc.exit_code not in (1, 2):
                raise
    return selected


def choose(source, files, limits, topic, units, policy):
    paths = {item["path"]: item for item in files if active(item["path"])}
    if policy == LEGACY:
        return set(paths), None
    if policy != CONTRACT:
        raise DreamError("Unsupported prior-context selection policy")
    bootstrap = set(paths) & {INDEX, SETTINGS}
    named = TOPICS + topic + ".org" if topic else None
    if named in paths:
        seeds, strategy = {named}, "topic"
    else:
        seeds = associations(source, files, units)
        strategy = "source_associations"
    if seeds:
        from . import dream_wiki
        contents, used = {}, 0
        for path in sorted(seeds):
            raw = source.blob(paths[path]["blob"], limits.max_prior_bytes - used)
            contents[path] = raw.decode()
            used += len(raw)
        inventory = dream_wiki.inventory(contents, source.deadline)
        neighbors = {link["path"] for page in inventory.values() for link in page["links"]
                     if link["type"] == "file" and link["path"] in paths and link["path"].startswith(TOPICS)}
        selected = bootstrap | seeds | neighbors
    else:
        sizes = [int(source.git("cat-file", "-s", item["blob"]).strip()) for item in paths.values()]
        if sum(sizes) <= limits.max_prior_bytes:
            selected, strategy = set(paths), "all"
        else:
            selected, strategy = bootstrap, "orientation"
    return selected, dict(contract=CONTRACT, strategy=strategy, seed_paths=sorted(seeds))


def context(plan):
    """Bounded model catalog; full omission records remain in the plan."""
    prior = plan["prior_wiki"]
    omitted = [item for item in prior["omitted"] if item["reason"] == "unselected_active_topic"]
    catalog = []
    for item in omitted:
        candidate = catalog + [item]
        if len(canonical(candidate)) > CATALOG_BYTES:
            break
        catalog = candidate
    return dict(selection=prior["selection"], omitted_active=catalog,
        catalog_entries_omitted=len(omitted) - len(catalog), selected_bytes=prior["selected_bytes"],
        allowance_bytes=plan["limits"]["max_prior_bytes"],
        rules="Only fully delivered prior pages may be rewritten. Unselected pages may contain relevant decisions. If necessary context is unavailable or cannot fit, report incomplete work rather than assuming no other knowledge exists.")
