"""Versioned native usage observations (prompt-caching ledger, sections 3-5)."""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
import math


CONTRACT = "dream-usage-v2"
MAPPINGS = ("codex-exec-inclusive-v1", "claude-print-exclusive-v1", "unverified")
FIELDS = ("input_total", "input_uncached", "input_cache_read", "input_cache_write", "output")


def normalize(usage, mapping):
    """Normalize one final aggregate; missing counts are never presumed zero."""
    value = dict(contract=CONTRACT, mapping=mapping, state="partial", errors=[],
                 cost=dict(provider_reported_usd=None, native_estimate_usd=None, basis="unknown"),
                 **{key: None for key in FIELDS})
    amount = usage.get("currency_usd") if isinstance(usage, dict) else None
    if (mapping == "claude-print-exclusive-v1" and
            (type(amount) is int and amount >= 0 or type(amount) is float and math.isfinite(amount) and amount >= 0)):
        # Claude Code computes this at list prices; it is not a billed amount.
        value["cost"].update(native_estimate_usd=amount, basis="native-list-estimate")
    if mapping not in MAPPINGS:
        value.update(state="invalid", errors=["unknown_mapping"])
        return value
    native = usage.get("native") if isinstance(usage, dict) else None
    if not isinstance(native, dict):
        return value

    def count(name):
        item = native.get(name)
        if item is None:
            return None
        if type(item) is not int or item < 0:
            value["errors"].append("invalid_" + name)
            return None
        return item

    value["output"] = count("output_tokens")
    if mapping == "unverified":
        if value["errors"]:
            value["state"] = "invalid"
        return value
    base = count("input_tokens")
    if mapping == "codex-exec-inclusive-v1":
        read, write = count("cached_input_tokens"), count("cache_write_input_tokens")
        value.update(input_total=base, input_cache_read=read, input_cache_write=write)
        if base is not None:
            if ((read is not None and read > base) or (write is not None and write > base) or
                    (read is not None and write is not None and read + write > base)):
                value["errors"].append("cache_exceeds_total")
            elif read is not None and write is not None:
                value["input_uncached"] = base - read - write
    else:
        read, write = count("cache_read_input_tokens"), count("cache_creation_input_tokens")
        value.update(input_uncached=base, input_cache_read=read, input_cache_write=write)
        if base is not None and read is not None and write is not None:
            value["input_total"] = base + read + write
        breakdown = native.get("cache_creation")
        if isinstance(breakdown, dict):
            parts = [breakdown.get("ephemeral_5m_input_tokens"), breakdown.get("ephemeral_1h_input_tokens")]
            if any(item is not None and (type(item) is not int or item < 0) for item in parts):
                value["errors"].append("invalid_write_breakdown")
            elif all(item is not None for item in parts) and write is not None and sum(parts) != write:
                value["errors"].append("write_breakdown_differs")
    if value["errors"]:
        value.update(state="invalid", **{key: None for key in FIELDS})
    elif all(value[key] is not None for key in FIELDS):
        value["state"] = "complete"
    return value


def aggregate(calls, *, by_role=True):
    """Known subtotals plus coverage; do not relabel partial sums as totals."""
    result = dict(contract=CONTRACT, calls=len(calls), invalid_calls=0,
                  known={key: 0 for key in FIELDS}, calls_missing={key: 0 for key in FIELDS},
                  cache_ratio_input=0, cache_ratio_read=0, cache_ratio_calls=0)
    money = {key: dict(known=Decimal(0), calls_missing=0)
             for key in ("provider_reported_usd", "native_estimate_usd")}
    roles = defaultdict(list)
    for row in calls:
        roles[row["reservation"]["role"]].append(row)
        value = row.get("usage_v2") or {}
        if value.get("state") == "invalid":
            result["invalid_calls"] += 1
        for key, subtotal in money.items():
            amount = value.get("cost", {}).get(key)
            if type(amount) in (int, float) and amount >= 0 and (type(amount) is int or math.isfinite(amount)):
                subtotal["known"] += Decimal(str(amount))
            else:
                subtotal["calls_missing"] += 1
        for key in FIELDS:
            item = value.get(key)
            if type(item) is int and item >= 0:
                result["known"][key] += item
            else:
                result["calls_missing"][key] += 1
        total, read = value.get("input_total"), value.get("input_cache_read")
        if type(total) is int and type(read) is int and 0 <= read <= total:
            result["cache_ratio_input"] += total
            result["cache_ratio_read"] += read
            result["cache_ratio_calls"] += 1
    result["input_cache_read_fraction"] = (result["cache_ratio_read"] / result["cache_ratio_input"]
        if result["cache_ratio_input"] else None)
    result["cost"] = {key: dict(known=str(value["known"]), calls_missing=value["calls_missing"])
                      for key, value in money.items()}
    if by_role:
        result["by_role"] = {role: aggregate(rows, by_role=False) for role, rows in sorted(roles.items())}
    return result
