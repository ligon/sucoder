"""Bounded wiki/source retrieval, reusing ledger sections 2-6 machinery.

Reading never admits new sources, checks out files, or follows source links.
Native Org supplies projections; exact source snapshots supply evidence.
"""
from __future__ import annotations

import subprocess
import time

from . import dream, dream_run
from .dream import DreamError, canonical, digest


ORIENTATION_BYTES = 2048
TOPIC_BYTES = 12288
SOURCE_BYTES = 32768
INSTRUCTIONS = ("Project memory is advisory; project instructions and maintained sources govern. "
    "Start with this orientation, choose a topic, and retrieve original evidence when correctness, date or scope matters. "
    "Use sucoder dream read MIRROR --topic NAME [--target CUSTOM_ID], or --archives with a specific target. "
    "Use sucoder dream source MIRROR --plan PLAN --unit UNIT, or --search TEXT, for admitted frozen evidence. "
    "Missing sources remain unavailable. Ordinary reads omit archives and LOGBOOK history.\n\n")


def allowance(value, default):
    if value is None:
        return default
    if type(value) is not int or not 1 <= value <= dream.MAX_TREE_BYTES:
        raise DreamError("Reader allowance must be between one and 262144 UTF-8 bytes")
    return value


def bounded(text, limit):
    raw = text.encode()
    delivered = raw[:limit].decode("utf-8", errors="ignore")
    return delivered, len(raw) - len(delivered.encode())


def projection(path, text, deadline, *, history=False):
    command = ["emacs", "--batch", "-Q", "--script", str(dream_run.ORG_VALIDATOR), "--reader"]
    if history:
        command.append("--history")
    try:
        result = subprocess.run(command, input=canonical({path: text}), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=dream_run._remaining(deadline), check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise DreamError("Native reader projection unavailable or timed out", 3) from None
    if result.returncode or len(result.stdout) > dream_run.MAX_PLAN_BYTES:
        raise DreamError("Native reader rejected Org or exceeded its allowance", 3)
    return dream._json(result.stdout)[path]


class View:
    def __init__(self, config, mirror, base="HEAD", deadline=None):
        self.deadline = deadline or time.monotonic() + 30
        self.source = dream.GitSource(config, mirror, self.deadline)
        self.commit = self.source.commit(base)
        self.mirror = mirror

    def read(self, topic=None, archives=False, target=None, max_bytes=None):
        if archives and (not topic or not target):
            raise DreamError("Archive reads require a topic and specific custom target")
        if not topic:
            if target:
                raise DreamError("A target requires a topic")
            path = ".sucoder/wiki/index.org"
            default = ORIENTATION_BYTES
        else:
            dream._path(topic)
            if not topic.endswith(".org"):
                topic += ".org"
            path = ".sucoder/wiki/" + ("archive/" if archives else "topics/") + topic
            default = TOPIC_BYTES
        limit = allowance(max_bytes, default)
        blob = self.source.entry(self.commit, path)
        if blob is None:
            raise DreamError("Requested wiki page is unavailable at this revision", 3)
        raw = self.source.blob(blob, dream_run.MAX_PLAN_BYTES)
        try:
            parsed = projection(path, raw.decode("utf-8"), self.deadline, history=archives)
        except UnicodeError:
            raise DreamError("Wiki page is not UTF-8", 3) from None
        text = parsed["text"]
        start, end = 0, len(text.encode())
        if target:
            candidates = [s for s in parsed["sections"] if s["target"] == target]
            if len(candidates) != 1:
                raise DreamError("Requested wiki target is unavailable or ambiguous", 3)
            start, end = candidates[0]["begin"], candidates[0]["end"]
            text = text.encode()[start:end].decode()
        if not topic:
            text = INSTRUCTIONS + text
        text, omitted = bounded(text, limit)
        result = dict(schema_version=1, operation="dream_read", state="incomplete" if omitted else "complete",
            repository=self.mirror, commit=self.commit, path=path, target=target, blob=blob,
            page_sha256=digest(raw), projection="org-with-history" if archives else "org-without-logbook",
            projected_range=dict(byte_start=start, byte_end=end), text=text,
            delivered_bytes=len(text.encode()), allowance_bytes=limit, budget_bytes_omitted=omitted,
            logbook_bytes_omitted=parsed["logbook_bytes_omitted"],
            archives="specific_entry" if archives else "excluded")
        result["read_id"] = digest(canonical(result))
        return result


def read(config, mirror, topic=None, archives=False, target=None, max_bytes=None, base="HEAD"):
    return View(config, mirror, base).read(topic, archives, target, max_bytes)


def frozen_sources(config, mirror, plan_path):
    """Read explicit private snapshots only; a shared receipt grants no access."""
    with dream_run._directory(config, plan_path) as (fd, directory):
        plan = dream_run._load(fd, directory.name, mirror)
        sources = []
        for unit in plan["units"]:
            try:
                raw = dream_run._read(fd, "evidence-" + unit["sha256"] + ".txt", unit["bytes"])
            except FileNotFoundError:
                raise DreamError("Frozen source unavailable locally; explicitly snapshot the admitted plan first", 3) from None
            if len(raw) != unit["bytes"] or digest(raw) != unit["sha256"]:
                raise DreamError("Frozen source differs from its admitted digest", 3)
            sources.append((unit, raw))
    return plan, sources


def search_ranges(raw, query):
    """Literal byte search with one neighboring line, merging overlaps."""
    lines = raw.splitlines(keepends=True)
    positions = [0]
    for line in lines:
        positions.append(positions[-1] + len(line))
    ranges = []
    for number, line in enumerate(lines):
        if query.lower() not in line.lower():
            continue
        start, end = positions[max(0, number - 1)], positions[min(len(lines), number + 2)]
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], end)
        else:
            ranges.append((start, end))
    return ranges


def source(config, mirror, plan_path, unit_id=None, query=None, start=None, end=None, max_bytes=None):
    limit = allowance(max_bytes, SOURCE_BYTES)
    if query is not None:
        dream._text(query, "source search")
        if start is not None or end is not None:
            raise DreamError("Choose a search or an explicit source range")
    if (start is not None or end is not None) and unit_id is None:
        raise DreamError("A byte range requires a source unit")
    plan, sources = frozen_sources(config, mirror, plan_path)
    selected = [(u, raw) for u, raw in sources if unit_id is None or u["unit_id"] == unit_id]
    if not selected:
        raise DreamError("Requested source unit is not admitted")
    excerpts, omitted, used, scanned = [], [], 0, 0
    for unit, raw in selected:
        origin = unit["byte_start"]
        if query is not None:
            scanned += len(raw)
            ranges = search_ranges(raw, query.encode())
        else:
            lo = origin if start is None else start
            hi = unit["byte_end"] if end is None else end
            if type(lo) is not int or type(hi) is not int or not origin <= lo < hi <= unit["byte_end"]:
                raise DreamError("Requested range is outside the admitted source unit")
            ranges = [(lo - origin, hi - origin)]
        for lo, hi in ranges:
            try:
                text = raw[lo:hi].decode("utf-8")
            except UnicodeError:
                raise DreamError("Source ranges must end at UTF-8 boundaries") from None
            text, missing = bounded(text, limit - used)
            if text:
                excerpt = {key: unit[key] for key in ("unit_id", "source_id", "kind", "repository_uuid")}
                if unit["kind"] == "git":
                    excerpt.update({key: unit[key] for key in ("repository", "commit", "blob", "path")})
                excerpt.update(byte_start=origin + lo, byte_end=origin + lo + len(text.encode()),
                               sha256=digest(text.encode()), text=text)
                excerpts.append(excerpt)
                used += len(text.encode())
            if missing:
                omitted.append(dict(unit_id=unit["unit_id"], byte_start=origin + lo + len(text.encode()),
                                    byte_end=origin + hi, reason="source_allowance"))
    result = dict(schema_version=1, operation="dream_source", state="incomplete" if omitted else "complete",
        plan_id=plan["plan_id"], query=query, excerpts=excerpts, omitted=omitted,
        delivered_bytes=used, allowance_bytes=limit, searched_bytes=scanned,
        snapshot_bytes_checked=sum(len(raw) for _, raw in sources))
    result["read_id"] = digest(canonical(result))
    return result
