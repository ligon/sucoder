"""Frozen author lookups, grounded in implementation ledger sections 2-6.

The agent requests context; this service alone reads admitted snapshots and
frozen wiki blobs. A partial read never authorizes a whole-page replacement.
"""
from __future__ import annotations

import hashlib

from . import dream, dream_reader, dream_roles, dream_run, dream_wiki
from .dream import DreamError, canonical, digest


CONTRACT = "frozen-author-context-v1"
MAX_REQUESTS = 12
SOURCE_BYTES = 32768
SEARCH_BYTES = 2 * 1024 * 1024


def request_schema():
    return dream_roles.object_schema(dict(kind=dict(type="string", enum=["topic", "archive", "topic-search", "source", "search"]),
        key=dream_roles.TEXT, target=dream_roles.TEXT, query=dream_roles.TEXT,
        start=dict(type="integer", minimum=0), end=dict(type="integer", minimum=0)))


def verify_blob(raw, oid):
    algorithm = "sha1" if len(oid) == 40 else "sha256"
    if hashlib.new(algorithm, b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != oid:
        raise DreamError("Cached wiki context differs from its frozen Git object", 3)


class Session:
    def __init__(self, plan, fd, run_id, sources, prior, deadline, source=None):
        self.plan, self.fd, self.run_id = plan, fd, run_id
        self.sources, self.prior, self.deadline, self.source = sources, prior, deadline, source
        self.files = {item["path"]: item["blob"] for item in plan["prior_wiki"]["files"]}
        self.trace, self.archives = [], {}
        self.used = dict(prior=sum(len(text.encode()) for text in prior.values()), source=0, searched=0)
        self.windows = {unit["unit_id"]: [(unit["byte_start"], unit["byte_end"])] for unit in sources}

    def blob(self, path):
        oid = self.files.get(path)
        if oid is None or not path.endswith(".org"):
            raise DreamError("Requested wiki page is absent from the frozen inventory")
        name = "context-blob-" + oid + ".org"
        try:
            raw = dream_run._read(self.fd, name, dream_run.MAX_PLAN_BYTES)
        except FileNotFoundError:
            if self.source is None:
                raise DreamError("Status cannot fetch missing frozen context", 3) from None
            raw = self.source.blob(oid, dream_run.MAX_PLAN_BYTES)
            dream_run._store(self.fd, name, raw)
        verify_blob(raw, oid)
        try:
            raw.decode()
        except UnicodeError:
            raise DreamError("Frozen wiki context is not UTF-8") from None
        return raw

    def wiki_lookup(self, request):
        remaining = self.plan["limits"]["max_prior_bytes"] - self.used["prior"]
        if remaining <= 0:
            raise DreamError("Cumulative prior-context allowance exhausted", 3)
        if request["start"] or request["end"]:
            raise DreamError("Wiki context uses topic/target, not source byte offsets")
        if request["kind"] == "topic-search":
            if request["key"] or request["target"] or not request["query"]:
                raise DreamError("Topic search requires only a literal query")
            excerpts, used, scan, omitted = [], 0, 0, []
            for path in sorted(self.files):
                if not path.startswith(".sucoder/wiki/topics/") or not path.endswith(".org"):
                    continue
                if self.used["searched"] + scan >= SEARCH_BYTES:
                    omitted.append(dict(path=path, reason="search_scan_allowance"))
                    continue
                raw = self.blob(path)
                if self.used["searched"] + scan + len(raw) > SEARCH_BYTES:
                    omitted.append(dict(path=path, reason="search_scan_allowance"))
                    continue
                scan += len(raw)
                for lo, hi in dream_reader.search_ranges(raw, request["query"].encode()):
                    text, missing = dream_reader.bounded(raw[lo:hi].decode(), remaining - used)
                    if text:
                        excerpts.append(dict(path=path, blob=self.files[path], byte_start=lo,
                            byte_end=lo + len(text.encode()), text=text, editable=False))
                        used += len(text.encode())
                    if missing:
                        omitted.append(dict(path=path, byte_start=lo + len(text.encode()), byte_end=hi, reason="prior_allowance"))
            return dict(state="incomplete" if omitted else "complete", excerpts=excerpts,
                omitted=omitted, delivered_bytes=used, searched_bytes=scan)
        if request["query"] or not request["key"]:
            raise DreamError("Topic/archive lookup requires a topic name and no query")
        key = dream._path(request["key"])
        if not key.endswith(".org"):
            key += ".org"
        archived = request["kind"] == "archive"
        if archived and not request["target"]:
            raise DreamError("Archive context requires a specific CUSTOM_ID")
        path = ".sucoder/wiki/" + ("archive/" if archived else "topics/") + key
        raw = self.blob(path)
        text = raw.decode()
        full = not request["target"] and not archived
        if not full:
            parsed = dream_reader.projection(path, text, self.deadline, history=True)
            matches = [section for section in parsed["sections"] if section["target"] == request["target"]]
            if len(matches) != 1:
                raise DreamError("Requested frozen wiki target is absent or ambiguous")
            section = matches[0]
            text = parsed["text"].encode()[section["begin"]:section["end"]].decode()
        if full and len(text.encode()) > remaining:
            raise DreamError("Full topic exceeds remaining prior allowance; request a target or narrow the plan", 3)
        text, omitted = dream_reader.bounded(text, remaining)
        return dict(state="incomplete" if omitted else "complete", path=path, blob=self.files[path],
            page_sha256=digest(raw), text=text, target=request["target"], editable=full,
            delivered_bytes=len(text.encode()), budget_bytes_omitted=omitted, searched_bytes=0)

    def source_lookup(self, request):
        if request["target"]:
            raise DreamError("Source lookup has no wiki target")
        search = request["kind"] == "search"
        if search and (not request["query"] or request["start"] or request["end"]):
            raise DreamError("Source search requires a literal query without offsets")
        if not search and (request["query"] or not request["key"]):
            raise DreamError("Source range lookup requires a unit ID without a query")
        units = [unit for unit in self.sources if not request["key"] or unit["unit_id"] == request["key"]]
        if not units:
            raise DreamError("Source unit was not admitted")
        remaining = SOURCE_BYTES - self.used["source"]
        if remaining <= 0:
            raise DreamError("Cumulative extra source allowance exhausted", 3)
        excerpts, omitted, used, scanned = [], [], 0, 0
        for unit in units:
            raw, origin = unit["text"].encode(), unit["byte_start"]
            if search:
                ranges = dream_reader.search_ranges(raw, request["query"].encode())
                scanned += len(raw)
            else:
                lo, hi = (request["start"], request["end"]) if request["start"] or request["end"] else (origin, unit["byte_end"])
                if not origin <= lo < hi <= unit["byte_end"]:
                    raise DreamError("Requested range is outside the frozen admitted unit")
                ranges = [(lo - origin, hi - origin)]
            for lo, hi in ranges:
                try:
                    text, missing = dream_reader.bounded(raw[lo:hi].decode(), remaining - used)
                except UnicodeError:
                    raise DreamError("Source offsets must be UTF-8 boundaries") from None
                if text:
                    excerpts.append(dict(unit_id=unit["unit_id"], byte_start=origin + lo,
                        byte_end=origin + lo + len(text.encode()), sha256=digest(text.encode()), text=text))
                    used += len(text.encode())
                if missing:
                    omitted.append(dict(unit_id=unit["unit_id"], byte_start=origin + lo + len(text.encode()),
                        byte_end=origin + hi, reason="source_allowance"))
        return dict(state="incomplete" if omitted else "complete", excerpts=excerpts, omitted=omitted,
            delivered_bytes=used, searched_bytes=scanned)

    def lookup(self, request):
        if len(self.trace) >= MAX_REQUESTS:
            raise DreamError("Author context request-count allowance exhausted", 3)
        dream_roles.validate_schema(request, request_schema())
        key = "context-" + digest(canonical([self.run_id, request, self.used])) + ".json"
        try:
            cached = dream._json(dream_run._read(self.fd, key, dream_run.MAX_PLAN_BYTES))
        except FileNotFoundError:
            cached = None
            if self.source is None:
                raise DreamError("Status cannot serve an unrecorded context request", 3) from None
        if cached is not None and not isinstance(cached, dict):
            raise DreamError("Invalid cached context response")
        if cached is not None and cached.get("state") == "unavailable":
            response = cached
        else:
            try:
                response = (self.wiki_lookup(request) if request["kind"] in ("topic", "archive", "topic-search")
                            else self.source_lookup(request))
            except DreamError as exc:
                response = dict(state="unavailable", error=str(exc), delivered_bytes=0, searched_bytes=0)
            response["read_id"] = digest(canonical([request, response]))
            if cached is not None and canonical(cached) != canonical(response):
                raise DreamError("Cached context differs from its frozen data", 3)
        if response["read_id"] != digest(canonical([request, {k: v for k, v in response.items() if k != "read_id"}])):
            raise DreamError("Invalid cached context response")
        if cached is None:
            dream_run._store(self.fd, key, canonical(response) + b"\n")
        category = "prior" if request["kind"] in ("topic", "archive", "topic-search") else "source"
        self.used[category] += response["delivered_bytes"]
        if request["kind"] == "topic-search":
            self.used["searched"] += response["searched_bytes"]
        if response["state"] != "unavailable":
            if response.get("editable"):
                self.prior[response["path"]] = response["text"]
            elif request["kind"] == "archive" and response["state"] == "complete":
                path, text = response["path"], response["text"]
                old = self.archives.get(path, "")
                if text not in old:
                    self.archives[path] = text if old in text else old.rstrip() + "\n" + text
            for excerpt in response.get("excerpts", []):
                if "unit_id" in excerpt:
                    self.windows[excerpt["unit_id"]].append((excerpt["byte_start"], excerpt["byte_end"]))
        self.trace.append(dict(request=request, response=response))
        return response

    def payload(self):
        return dict(contract=CONTRACT, requests=self.trace, used=dict(self.used),
            allowances=dict(prior=self.plan["limits"]["max_prior_bytes"], extra_source=SOURCE_BYTES,
                            topic_search_scan=SEARCH_BYTES, requests=MAX_REQUESTS))
