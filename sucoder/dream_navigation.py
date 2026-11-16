"""Version 3 navigation, durable observations and advice (navigation ledger 5).

Old jobs retain dream_discover's original schemas, briefs and recomputation.
New cursors and observations share a single publish-once commit point.
"""
from __future__ import annotations

import copy
import os
import time

from . import dream, dream_isolate, dream_run, dream_session_scan as scan
from .dream import DreamError, canonical, digest


BRIEF = (
    " Navigation version 3: always supply cursor, an empty string for a new operation. "
    "Session read/search require offset=0; initial search also requires start=end=0 and a literal query. "
    "Continue a session operation with next_cursor, the same path/query/action, and zero numeric fields. "
    "A page without hits can still be incomplete. complete_unfrozen means traversal ended over mutable files; "
    "it never proves query absence in a coherent source revision. Gaps, exclusions and worker failures leave "
    "unsearched material. A processing timeout terminates that lookup; query another session ID if useful. "
    "Project list/read/search retain their file offsets and line semantics with cursor empty. "
    "Select only exact nonzero raw session ranges at finish. Offered actions and remaining allowances are binding. "
)


def checksum(value):
    return digest(canonical({k: v for k, v in value.items() if k != "sha256"}))


def checked(value):
    if not isinstance(value, dict) or value.get("sha256") != checksum(value):
        raise DreamError("Frozen navigation artifact checksum differs", 3)
    return value


class Navigation:
    def __init__(self, inventory, limits=None, processing=None):
        self.inventory = inventory
        self.limits = scan.validate_limits(dict(scan.LIMITS if limits is None else limits))
        self.processing = dream_isolate.validate(dict(dream_isolate.POLICY if processing is None else processing))
        self.binding = dict(inventory_sha256=digest(canonical(inventory.manifest)), navigation_version=3)
        self.cursors, self.last, self.selection_receipts = {}, None, []

    def bind(self, header):
        self.binding["header_sha256"] = digest(canonical(header))

    def invalid(self, request, reason):
        example = dict(action="search", path="@sessions", query="literal phrase", start=0, end=0, offset=0, cursor="")
        return dict(state="invalid_request", reason=reason, example=example)

    def validate(self, request):
        action = request["action"]
        session = request["path"] == "@sessions" or request["path"].startswith("@sessions/")
        if request.get("cursor") and (not session or action not in ("read", "search")):
            return self.invalid(request, "Only session read/search accept a continuation cursor")
        if session and action in ("read", "search"):
            if request["offset"] or (action == "search" and (request["start"] or request["end"] or not request["query"])):
                return self.invalid(request, "Session search uses query and cursor, with zero numeric fields")
            if action == "read" and request["query"]:
                return self.invalid(request, "Session read requires an empty query")
            if request.get("cursor") and (request["start"] or request["end"]):
                return self.invalid(request, "Continuation numeric fields must be zero")
        return None

    def register(self, envelope):
        body = envelope.get("continuation")
        observation = envelope["observation"]
        if body is None:
            if observation.get("next_cursor"):
                raise DreamError("Saved navigation observation has an orphan cursor", 3)
            return
        token = digest(canonical(body))
        base = {k: v for k, v in observation.items() if k != "next_cursor"}
        if (body.get("binding") != self.binding or observation.get("next_cursor") != token or
                body.get("receipts_sha256") != digest(canonical(envelope["receipts"])) or
                body.get("observation_sha256") != digest(canonical(base)) or
                body.get("producer") != envelope["binding"]["assignment_id"]):
            raise DreamError("Frozen navigation cursor differs from its committed observation", 3)
        self.cursors[token] = body

    def window(self, row, start, end, cap):
        deadline = min(self.inventory.source.deadline, time.monotonic() + self.processing["max_seconds"])
        value = dream_isolate.run(lambda: scan.normalize(row, start, end, cap, deadline, self.limits),
                                  deadline, self.processing)
        self.last_window = value
        self.selection_receipts.extend(value["receipts"])
        return value["start"], value["end"], value["text"]

    def lookup(self, request, remaining, producer="local"):
        self.last = None
        private_error = ""
        cap = min(16384, remaining)
        if cap < 1024:
            return dict(state="discovery_delivery_exhausted", delivered_bytes=0)
        invalid = self.validate(request)
        if invalid:
            return invalid
        path, action = request["path"], request["action"]
        rows = [row for row in self.inventory.sessions.values()
                if path == "@sessions" or path == "@sessions/" + row["thread_id"]]
        if action == "list":
            if path != "@sessions" or request.get("cursor") or request["query"] or request["start"] or request["end"]:
                return self.invalid(request, "List @sessions with an offset and no other bounds")
            shown, offset = [], request["offset"]
            for row in rows[offset:offset + 32]:
                item = {k: row[k] for k in ("bytes", "state", "modified", "consumed")}
                item.update(path="@sessions/" + row["thread_id"], kind="session")
                if "reason" in row:
                    item["reason"] = row["reason"]
                if len(canonical(shown + [item])) > cap - 768:
                    break
                shown.append(item)
            result = dict(state="complete" if offset + len(shown) >= len(rows) else "incomplete", entries=shown,
                total_entries=len(rows), next_offset=offset + len(shown), inventory_coverage=self.inventory.session_coverage)
            if len(canonical(result)) > cap:
                result = dict(state="response_metadata_limit", unsearched_scope=path)
            return result
        if action not in ("read", "search") or (action == "read" and path == "@sessions"):
            return self.invalid(request, "Read requires @sessions/ID; search accepts @sessions or @sessions/ID")
        if path != "@sessions" and not rows:
            return dict(state="unavailable", reason="session_absent", unsearched_scope=path)
        operation = dict(action=action, path=path, query=request["query"])
        token = request.get("cursor", "")
        if token:
            prior = self.cursors.get(token)
            if prior is None or prior["binding"] != self.binding or prior["operation"] != operation:
                return self.invalid(request, "Cursor does not bind this job, operation, scope and query")
            state = prior["state"]
        else:
            if action == "read" and not (0 <= request["start"] < rows[0]["bytes"] and
                    (not request["end"] or request["start"] < request["end"] <= rows[0]["bytes"])):
                return self.invalid(request, "Read bounds must lie in the pinned complete prefix")
            state = dict(session=0, position=request["start"] if action == "read" else 0,
                range_end=(request["end"] or rows[0]["bytes"]) if action == "read" else 0,
                block=0, record_sha256="", align=action == "read", skip_start=-1,
                had_gaps=self.inventory.session_coverage.get("state", "complete") != "complete")
        deadline = min(self.inventory.source.deadline, time.monotonic() + self.processing["max_seconds"])
        try:
            value = dream_isolate.run(lambda: scan.page(rows, action, request["query"], state, cap,
                deadline, self.limits), deadline, self.processing)
        except dream_isolate.ProcessingError as exc:
            private_error = exc.detail
            value = dict(observation=dict(state=exc.reason, unsearched_scope=path, actual_reads="unknown",
                raw_scanned_bytes=None, raw_scan_upper_bound=self.limits["scan_bytes"] + self.limits["alignment_bytes"],
                source_consistency=scan.QUALIFICATION), continuation=None, receipts=None)
        observation = value["observation"]
        observation["inventory_complete"] = self.inventory.session_coverage.get("state", "complete") == "complete"
        observation["next_cursor"] = "0" * 64 if value["continuation"] else ""
        while len(canonical(observation)) > cap and observation.get("gaps"):
            observation["gaps"].pop()
            observation["omitted_gap_details"] += 1
        body = None
        if value["continuation"] is not None:
            base = {k: v for k, v in observation.items() if k != "next_cursor"}
            body = dict(binding=self.binding, producer=producer, operation=operation, state=value["continuation"],
                receipts_sha256=digest(canonical(value["receipts"])), observation_sha256=digest(canonical(base)))
        observation["next_cursor"] = digest(canonical(body)) if body else ""
        if len(canonical(observation)) > cap:
            observation = dict(state="response_metadata_limit", unsearched_scope=path, next_cursor="",
                actual_reads="unknown" if value["receipts"] is None else "recorded_privately",
                raw_scanned_bytes=value["observation"].get("raw_scanned_bytes"),
                alignment_bytes=value["observation"].get("alignment_bytes"),
                raw_scan_upper_bound=self.limits["scan_bytes"] + self.limits["alignment_bytes"])
            body = None
        self.last = dict(observation=observation, continuation=body, receipts=value["receipts"], processing_error=private_error)
        if body:
            # Direct local navigation also resolves tokens; the coordinator commits before exposing one to a successor.
            self.cursors[observation["next_cursor"]] = body
        return observation


def _read(fd, name):
    return dream._json(dream_run._read(fd, name, dream_run.MAX_PLAN_BYTES))


def _store(fd, name, value):
    dream_run._store(fd, name, canonical(value) + b"\n")


def turn(explorer, step, history, used):
    key = "navigation-" + str(explorer.attempt).zfill(6) + "-" + str(step).zfill(6)
    binding = dict(header_sha256=digest(canonical(explorer.header)),
        inventory_sha256=digest(canonical(explorer.inventory.manifest)), run_id=explorer.run_id,
        attempt=explorer.attempt, step=step, history_sha256=digest(canonical(history)))
    try:
        saved = checked(_read(explorer.fd, key + ".json"))
    except FileNotFoundError:
        try:
            _read(explorer.fd, key + "-marker.json")
        except FileNotFoundError:
            pass
        else:
            raise DreamError("Frozen navigation advice is missing after its turn marker", 3)
        if explorer.budget.readonly:
            raise DreamError("Read-only inspection cannot recreate navigation advice", 3)
        reserved = sum(row["reservation"]["run_id"] == explorer.run_id for row in explorer.budget.calls)
        if reserved > step:
            raise DreamError("Frozen navigation advice is missing after a native reservation", 3)
        # The marker and advice may both be lost. An already published assignment
        # must still prevent a different assignment from being minted for this step.
        with os.scandir(explorer.fd) as entries:
            for count, entry in enumerate(entries, 1):
                if count > 16384:
                    raise DreamError("Cannot establish frozen advice within the private journal allowance", 3)
                if entry.name.startswith(explorer.run_id + "-") and entry.name.endswith("-assignment.json"):
                    task = _read(explorer.fd, entry.name)
                    if (task.get("role") == "explorer" and task.get("run_id") == explorer.run_id and
                            len(task.get("input", {}).get("history", [])) == step):
                        raise DreamError("Frozen navigation advice is missing after assignment publication", 3)
        budget = explorer.budget
        summary = budget.summary()
        available = max(0, summary["remaining_calls"] - budget.header["limits"]["reserve_calls"] - 1)
        left = min(explorer.cap - step, available)
        if left <= 0 or budget.remaining() <= budget.header["limits"]["reserve_seconds"]:
            budget.fail("no unreserved discovery capacity")
        remaining = max(0, explorer.header["settings"]["max_discovery_bytes"] - used)
        navigation = dict(actions=["finish"] if left == 1 or remaining < 1024 else ["list", "read", "search", "finish"],
            root="", files=len(explorer.inventory.files), selected_bytes=65536,
            remaining_delivery_bytes=remaining, remaining_discovery_calls=left,
            wallet_calls_left=summary["remaining_calls"], wallet_seconds_left=summary["remaining_seconds"],
            reserve_calls=budget.header["limits"]["reserve_calls"], author_calls_reserved=1,
            reserve_seconds=budget.header["limits"]["reserve_seconds"], page_limits=explorer.inventory.navigator.limits,
            processing=explorer.inventory.navigator.processing, source_consistency=scan.QUALIFICATION)
        if "leads" in explorer.header:
            navigation["memory"] = explorer.header["leads"]
        if explorer.experience:
            navigation.update(sessions_root="@sessions", sessions=len(explorer.inventory.sessions),
                notes=explorer.inventory.notes(), inventory_coverage=explorer.inventory.session_coverage)
        saved = dict(binding=binding, navigation=navigation)
        saved["sha256"] = checksum(saved)
        _store(explorer.fd, key + "-marker.json", binding)
        _store(explorer.fd, key + ".json", saved)
    try:
        marker = _read(explorer.fd, key + "-marker.json")
    except FileNotFoundError:
        raise DreamError("Frozen navigation turn marker is missing", 3) from None
    if saved.get("binding") != binding or marker != binding:
        raise DreamError("Frozen navigation advice binding differs", 3)
    return saved["navigation"]


def observe(explorer, step, history, request, remaining):
    assignment = explorer.traces[-1]["assignment_id"]
    name = "lookup-" + assignment + ".json"
    navigator = explorer.inventory.navigator
    binding = dict(assignment_id=assignment, header_sha256=digest(canonical(explorer.header)),
        inventory_sha256=digest(canonical(explorer.inventory.manifest)), attempt=explorer.attempt, step=step,
        history_sha256=digest(canonical(history)), request_sha256=digest(canonical(request)))
    try:
        saved = checked(_read(explorer.fd, name))
    except FileNotFoundError:
        if explorer.budget.readonly:
            raise DreamError("Committed navigation observation is missing", 3)
        successor = "navigation-" + str(explorer.attempt).zfill(6) + "-" + str(step + 1).zfill(6) + "-marker.json"
        try:
            _read(explorer.fd, successor)
        except FileNotFoundError:
            pass
        else:
            raise DreamError("Missing committed observation already has a successor turn", 3)
        navigator.last, navigator.selection_receipts = None, []
        selection = None
        try:
            if request["action"] == "finish":
                from .dream_discover import topic_name
                topic = topic_name(request["topic"])
                if not request["rationale"].strip():
                    raise DreamError("Explain the exploratory selection or no-selection decision")
                units = explorer.inventory.selected(request["selections"])
                selection = dict(units=units, topic=topic or "exploration", rationale=request["rationale"],
                                 questions=request["questions"])
                observation = dict(state="selected")
            else:
                navigator.producer = assignment
                observation = explorer.inventory.lookup(request, remaining)
        except (DreamError, UnicodeError) as exc:
            observation = dict(state="invalid_selection" if request["action"] == "finish" else "unavailable",
                               error=exc.reason if isinstance(exc, dream_isolate.ProcessingError) else str(exc)[:512])
            if isinstance(exc, dream_isolate.ProcessingError):
                observation["actual_reads"] = "unknown"
        saved = dict(binding=binding, observation=observation, continuation=None,
            receipts=navigator.selection_receipts, selection=selection)
        if navigator.last is not None:
            saved.update(navigator.last)
        saved["sha256"] = checksum(saved)
        _store(explorer.fd, name, saved)
    if saved.get("binding") != binding:
        raise DreamError("Frozen navigation observation binding differs", 3)
    if saved["selection"] is None and len(canonical(saved["observation"])) > min(16384, remaining):
        raise DreamError("Frozen navigation observation exceeds its delivery allowance", 3)
    navigator.register(saved)
    return saved


def prepare(explorer):
    history = []
    used = len(canonical(explorer.header["leads"])) if "leads" in explorer.header else 0
    raw_scanned, unknown_reads = 0, False
    for step in range(explorer.cap):
        navigation = turn(explorer, step, history, used)
        explorer.turn_schema = copy.deepcopy(explorer.schema)
        explorer.turn_schema["properties"]["action"]["enum"] = navigation["actions"]
        result = explorer.call("explorer", dict(project=explorer.header["project"], purpose=explorer.header["purpose"],
            variation=explorer.header["job_id"], navigation=navigation, history=history))
        saved = observe(explorer, step, history, result, explorer.header["settings"]["max_discovery_bytes"] - used)
        observation = saved["observation"]
        if saved["selection"] is not None:
            return dict(saved["selection"], history=history, assignments=explorer.traces,
                selection_sha256=saved["sha256"], navigation_coverage=dict(raw_scanned_bytes=None if unknown_reads else raw_scanned,
                    known_raw_scanned_bytes=raw_scanned, unknown_actual_reads=unknown_reads, delivered_bytes=used,
                    inventory_coverage=explorer.inventory.session_coverage, source_consistency=scan.QUALIFICATION))
        raw_scanned += observation.get("raw_scanned_bytes") or 0
        unknown_reads |= observation.get("actual_reads") == "unknown"
        used += len(canonical(observation))
        history.append(dict(action=result, observation=observation, lookup_sha256=saved["sha256"]))
    explorer.budget.fail("exploration allowance exhausted; author/reviewer capacity remains reserved")
