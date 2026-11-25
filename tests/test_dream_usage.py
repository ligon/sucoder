import pytest

from sucoder import dream_usage as usage


def test_provider_totals_and_nonzero_writes_have_different_definitions():
    codex = usage.normalize(dict(native=dict(input_tokens=100, cached_input_tokens=60,
        cache_write_input_tokens=10, output_tokens=5)), "codex-exec-inclusive-v1")
    claude = usage.normalize(dict(native=dict(input_tokens=30, cache_read_input_tokens=60,
        cache_creation_input_tokens=10, output_tokens=5,
        cache_creation=dict(ephemeral_5m_input_tokens=7, ephemeral_1h_input_tokens=3))), "claude-print-exclusive-v1")
    assert codex["state"] == claude["state"] == "complete"
    assert {key: codex[key] for key in usage.FIELDS} == {key: claude[key] for key in usage.FIELDS}
    assert claude["input_total"] == 100 and claude["input_uncached"] == 30


@pytest.mark.parametrize("bad", [True, False, -1, 1.5, "4", float("inf")])
def test_invalid_counts_never_become_savings(bad):
    result = usage.normalize(dict(native=dict(input_tokens=100, cached_input_tokens=bad,
        cache_write_input_tokens=0, output_tokens=1)), "codex-exec-inclusive-v1")
    assert result["state"] == "invalid"
    assert all(result[key] is None for key in usage.FIELDS)


def test_missing_write_is_unknown_while_explicit_zero_is_observed():
    raw = dict(native=dict(input_tokens=100, cached_input_tokens=20, output_tokens=1))
    result = usage.normalize(raw, "codex-exec-inclusive-v1")
    assert result["state"] == "partial" and result["input_cache_write"] is None
    assert result["input_total"] == 100 and result["input_uncached"] is None
    raw["native"]["cache_write_input_tokens"] = 0
    assert usage.normalize(raw, "codex-exec-inclusive-v1")["input_uncached"] == 80


@pytest.mark.parametrize("native,mapping", [
    (dict(input_tokens=10, cached_input_tokens=9, cache_write_input_tokens=2), "codex-exec-inclusive-v1"),
    (dict(input_tokens=10, cache_read_input_tokens=20, cache_creation_input_tokens=3,
        cache_creation=dict(ephemeral_5m_input_tokens=2, ephemeral_1h_input_tokens=2)), "claude-print-exclusive-v1"),
])
def test_contradictory_counts_retain_no_invented_total(native, mapping):
    assert usage.normalize(dict(native=native), mapping)["state"] == "invalid"


def test_exclusive_missing_reads_and_unverified_mapping_leave_total_unknown():
    raw = dict(native=dict(input_tokens=10, cache_creation_input_tokens=2, output_tokens=3))
    assert usage.normalize(raw, "claude-print-exclusive-v1")["input_total"] is None
    result = usage.normalize(raw, "unverified")
    assert result["input_total"] is None and result["output"] == 3


def test_aggregate_keeps_partial_coverage_by_role_and_does_not_count_breakdowns_twice():
    full = usage.normalize(dict(native=dict(input_tokens=100, cached_input_tokens=60,
        cache_write_input_tokens=10, output_tokens=5)), "codex-exec-inclusive-v1")
    partial = usage.normalize(dict(native=dict(input_tokens=2, output_tokens=1)), "claude-print-exclusive-v1")
    rows = [dict(reservation=dict(role="explorer"), usage_v2=full),
            dict(reservation=dict(role="reviewer"), usage_v2=partial),
            dict(reservation=dict(role="explorer"), usage=None)]
    result = usage.aggregate(rows)
    assert result["known"]["input_total"] == 100
    assert result["known"]["input_uncached"] == 32
    assert result["calls_missing"]["input_total"] == 2
    assert result["cache_ratio_calls"] == 1
    assert result["input_cache_read_fraction"] == .6
    assert result["by_role"]["explorer"]["calls"] == 2


def test_raw_usage_is_never_modified_or_summed_with_iteration_copies():
    raw = dict(native=dict(input_tokens=10, cache_read_input_tokens=20,
        cache_creation_input_tokens=3, output_tokens=4, iterations=[dict(input_tokens=10)]))
    assert usage.normalize(raw, "claude-print-exclusive-v1")["input_total"] == 33
    assert raw["native"]["input_tokens"] == 10


def test_native_list_price_estimate_is_never_reported_as_billed_cost():
    result = usage.normalize(dict(native={}, currency_usd=.125), "claude-print-exclusive-v1")
    assert result["cost"] == dict(provider_reported_usd=None, native_estimate_usd=.125, basis="native-list-estimate")
    summary = usage.aggregate([dict(reservation=dict(role="author"), usage_v2=result)])
    assert summary["cost"]["native_estimate_usd"] == dict(known="0.125", calls_missing=0)
    assert summary["cost"]["provider_reported_usd"] == dict(known="0", calls_missing=1)


def test_invalid_unverified_output_counter_is_invalid():
    assert usage.normalize(dict(native=dict(output_tokens=-1)), "unverified")["state"] == "invalid"
