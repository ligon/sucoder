#!/usr/bin/env python3
"""Benchmark the network/server cost of a Claude API call from this host.

Measures time-to-first-token (TTFB) and total wall time for N identical
streaming requests, then prints a one-screen summary plus optional
JSONL of the raw per-trial data.

Designed for paired comparison: run it from two hosts (e.g. a laptop
and a Savio compute node) and diff the medians.  The script is
self-labelling --- hostname, ISO timestamp, model, and trial count are
all in the header --- so you can pipe output to a file per host and
compare later.

Usage::

    # Default: 10 trials of Haiku, ~100 output tokens each.
    ./scripts/bench_api_wire.py

    # Heavier workload, custom prompt, JSONL trace to a file.
    ./scripts/bench_api_wire.py --trials 20 --max-tokens 300 \\
        --prompt 'Write a haiku about Berkeley.' \\
        --jsonl /tmp/wire-savio.jsonl

Requirements: ``anthropic`` Python package and ``ANTHROPIC_API_KEY``
in the environment.  No other dependencies.

Cost: ~$0.005 per default run with Haiku.  Negligible.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Optional, TextIO

try:
    import anthropic
except ImportError:
    sys.exit(
        "Missing dependency: install with `pip install anthropic`.\n"
        "(Or run inside a venv where the SDK is already available.)"
    )


# A short fixed prompt with a small bounded output keeps per-trial cost
# trivial and per-trial variance dominated by network/server latency
# rather than by the model's content choices.  At ~50 tokens of input
# the request is well below Anthropic's 1024-token caching threshold,
# so successive trials genuinely re-traverse the inference path.
DEFAULT_PROMPT = (
    "Reply with exactly three short sentences about why benchmarks matter."
)
DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_TRIALS = 10
DEFAULT_MAX_TOKENS = 120
# A small gap between trials keeps us comfortably under the per-minute
# rate limit on shared API keys without materially extending wall time.
DEFAULT_PAUSE_S = 0.5


@dataclass
class TrialResult:
    """Per-trial timing + token-count record."""

    trial: int
    ttfb_ms: float            # request send -> first content delta
    total_ms: float           # request send -> stream done
    input_tokens: int
    output_tokens: int
    streaming_tps: float      # output_tokens / (total - TTFB), 0 if <=0 denom

    @property
    def server_ms(self) -> float:
        """Time after first byte spent streaming the body."""
        return self.total_ms - self.ttfb_ms


def run_one_trial(
    client: anthropic.Anthropic,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    trial: int,
) -> TrialResult:
    """Send one streaming request, return timing + usage."""
    t_start = time.perf_counter()
    t_first: Optional[float] = None
    input_tokens = 0
    output_tokens = 0

    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for event in stream:
            # message_start arrives almost immediately with the message
            # header; we care about the first actual text token.
            if t_first is None and event.type == "content_block_delta":
                t_first = time.perf_counter()
            # The final usage counts arrive on message_delta /
            # message_stop; the SDK exposes them on get_final_message().
        final = stream.get_final_message()
        input_tokens = final.usage.input_tokens
        output_tokens = final.usage.output_tokens

    t_end = time.perf_counter()
    if t_first is None:
        # No content deltas — model returned an empty message or
        # only refused.  Treat ttfb as total so the row is still
        # comparable, but flag with zero output tokens.
        t_first = t_end

    ttfb_ms = (t_first - t_start) * 1000.0
    total_ms = (t_end - t_start) * 1000.0
    server_s = (t_end - t_first)
    tps = (output_tokens / server_s) if server_s > 0 else 0.0

    return TrialResult(
        trial=trial,
        ttfb_ms=ttfb_ms,
        total_ms=total_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        streaming_tps=tps,
    )


def percentile(values: List[float], q: float) -> float:
    """Linear-interp percentile.  q in [0, 100]."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def summarise(results: List[TrialResult]) -> str:
    """Render a human-readable summary block."""
    if not results:
        return "(no successful trials)"

    ttfbs = [r.ttfb_ms for r in results]
    totals = [r.total_ms for r in results]
    tps = [r.streaming_tps for r in results if r.streaming_tps > 0]
    in_tok = [r.input_tokens for r in results]
    out_tok = [r.output_tokens for r in results]

    def stats(label: str, xs: List[float], unit: str) -> str:
        if not xs:
            return f"  {label:<22} (no data)"
        return (
            f"  {label:<22} "
            f"min={min(xs):.1f} med={statistics.median(xs):.1f} "
            f"p95={percentile(xs, 95):.1f} max={max(xs):.1f} {unit}"
        )

    lines = [
        stats("TTFB",              ttfbs,  "ms"),
        stats("total wall time",   totals, "ms"),
        stats("streaming rate",    tps,    "tok/s"),
        stats("input tokens",      [float(x) for x in in_tok], "tok"),
        stats("output tokens",     [float(x) for x in out_tok], "tok"),
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark Claude API wire time from this host.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Compare hosts by piping output to per-host files:\n"
            "  ./scripts/bench_api_wire.py > /tmp/wire-laptop.txt\n"
            "  # then on the savio node:\n"
            "  ./scripts/bench_api_wire.py > /tmp/wire-savio.txt\n"
            "  diff -u /tmp/wire-laptop.txt /tmp/wire-savio.txt"
        ),
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Claude model (default: {DEFAULT_MODEL}). "
                   "Use Haiku for cheap latency tests; Sonnet/Opus to "
                   "measure their characteristic streaming rates.")
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                   help=f"Number of requests to send (default: {DEFAULT_TRIALS}).")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help=f"max_tokens per request (default: {DEFAULT_MAX_TOKENS}).")
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="Override the fixed user prompt. Keep it short to "
                   "minimise input-token cost and stay below the cache threshold.")
    p.add_argument("--pause", type=float, default=DEFAULT_PAUSE_S,
                   help=f"Seconds to sleep between trials "
                   f"(default: {DEFAULT_PAUSE_S}; raise if you hit 429s).")
    p.add_argument("--jsonl", type=argparse.FileType("w"), default=None,
                   help="Also write one JSON record per trial to this file "
                   "for later analysis.")
    p.add_argument("--label", default=None,
                   help="Extra free-form label for the header "
                   "(e.g. 'wifi', 'ethernet', 'savio4_htc-n0042').")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set in the environment.")

    client = anthropic.Anthropic()

    header_bits = [
        f"host={socket.gethostname()}",
        f"platform={platform.system()}-{platform.machine()}",
        f"model={args.model}",
        f"trials={args.trials}",
        f"max_tokens={args.max_tokens}",
        f"timestamp={datetime.now(timezone.utc).isoformat(timespec='seconds')}",
    ]
    if args.label:
        header_bits.append(f"label={args.label}")
    print("# " + " ".join(header_bits))
    print(f"# prompt={args.prompt!r}")
    print()

    results: List[TrialResult] = []
    failures: List[tuple[int, str]] = []

    for i in range(1, args.trials + 1):
        try:
            r = run_one_trial(
                client,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                trial=i,
            )
        except anthropic.APIError as exc:
            failures.append((i, f"{type(exc).__name__}: {exc}"))
            print(f"  trial {i:>3}: FAIL ({type(exc).__name__})", file=sys.stderr)
            continue

        results.append(r)
        print(
            f"  trial {i:>3}: TTFB={r.ttfb_ms:7.1f}ms "
            f"total={r.total_ms:7.1f}ms "
            f"in={r.input_tokens:>4} out={r.output_tokens:>4} "
            f"@{r.streaming_tps:5.1f} tok/s",
            file=sys.stderr,
        )

        if args.jsonl is not None:
            args.jsonl.write(json.dumps(asdict(r)) + "\n")
            args.jsonl.flush()

        if i < args.trials:
            time.sleep(args.pause)

    print()
    print("# Summary")
    print(summarise(results))
    if failures:
        print()
        print(f"# {len(failures)} failure(s):")
        for i, msg in failures:
            print(f"  trial {i}: {msg}")

    if args.jsonl is not None:
        args.jsonl.close()

    # Exit nonzero only if every trial failed --- partial failures are
    # informative noise, not script bugs.
    return 0 if results else 2


if __name__ == "__main__":
    sys.exit(main())
