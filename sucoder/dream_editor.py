"""V5 editorial actions and materialized review (implementation ledger P5)."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import posixpath

from . import dream, dream_author, dream_context, dream_editorial as org, dream_roles, dream_run, dream_view, dream_wiki as wiki
from .dream import DreamError, canonical, digest


CONTRACT = "source-grounded-wiki-v5"
POLICY = "recorded-editorial-actions-v1"
BRIEFS = {role: brief + (
    " Editorial work uses task_actions; leave it empty for prose-only changes. "
    "Keep existing task states, properties, completion stamps and LOGBOOK text unchanged in authored pages. "
    "Request transition/archive actions using the supplied frozen subtree hash; the coordinator writes their history and move metadata. "
    "Every transition needs an owner and a reason. WAITING needs blocker and resume_when; REVIEW needs reviewer and acceptance; DONE needs acceptance. "
    "Use authority agent, independent_review or recorded_human, with exact source evidence where required. "
    "Never invent a human approval or drop a task's human requirement. Existing prose may impose authority conditions even without a property. "
    "A task requiring a human decision stays WAITING/REVIEW without recorded human authority. "
    "Archive only terminal trees without open descendants; preserve necessary qualifications on the active page. "
    "The coordinator retains forwarding anchors, archive context and tracked incoming links. "
    "Review the materialized candidate, including generated pages and history, and return one task_check for every action. "
    "Check authority_verified and acceptance_satisfied separately from the verdict; requires_human must reflect the prior task too. "
    "Lookup replies have no task actions/checks. Unchanged has no task actions. "
    "omissions entries must name an admitted source unit_id, never a wiki path. Explain unchanged wiki material in rationale instead. "
    "Private transcript evidence has no public source-file link: cite it in references and plain prose, never invent a file path. "
    "CUSTOM_ID links use [[#target]] or [[file:relative.org::#target][label]]; id: links are unsupported."
    ) for role, brief in dream_author.BRIEFS.items()}


def schema(role):
    original = dream_author.schema(role)
    fields = dict(original["properties"])
    if role == "reviewer":
        fields["task_checks"] = dream_roles.array_schema(dream_roles.object_schema(dict(
            kind=dict(type="string", enum=["transition", "archive"]), path=dream_roles.TEXT, target=dream_roles.TEXT,
            requires_human=dict(type="string", enum=["yes", "no"]), authority_verified=dict(type="string", enum=["yes", "no"]),
            acceptance_satisfied=dict(type="string", enum=["yes", "no"]), reason=dream_roles.TEXT)), org.MAX_ACTIONS)
    else:
        fields["task_actions"] = dream_roles.array_schema(dream_roles.object_schema(dict(
            kind=dict(type="string", enum=["transition", "archive"]), path=dream_roles.TEXT, target=dream_roles.TEXT,
            before_sha256=dream_roles.TEXT, state=dream_roles.TEXT, note=dream_roles.TEXT,
            owner=dream_roles.TEXT, reviewer=dream_roles.TEXT, acceptance=dream_roles.TEXT,
            blocker=dream_roles.TEXT, resume_when=dream_roles.TEXT, destination=dream_roles.TEXT,
            authority=dict(type="string", enum=["agent", "independent_review", "recorded_human"]),
            evidence=dream_roles.array_schema(dream_roles.object_schema(dict(unit_id=dream_roles.TEXT, quote=dream_roles.TEXT)), 16))), org.MAX_ACTIONS)
    return dream_roles.object_schema(fields)


def heading_map(inventory):
    return {(path, heading["target"]): heading for path, page in inventory.items()
            for heading in page["headings"] if heading["target"]}


def declared_human(heading):
    props = heading["properties"]
    required = props.get("REQUIRES_HUMAN", "").strip().lower()
    return (bool(required and required not in ("no", "false", "0")) or
            props.get("DECISION_AUTHORITY", "").strip().lower() in ("human", "user", "maintainer") or
            props.get("REVIEWER", "").strip().lower() in ("human", "user", "maintainer"))


def check_actions(actions, checks, required_human=()):
    """Shared mechanical check of the independent action assessment."""
    keys = lambda items: Counter((item["kind"], item["path"], item["target"]) for item in items)
    if keys(actions) != keys(checks) or any(count != 1 for count in keys(actions).values()):
        raise DreamError("Independent review needs exactly one check for every editorial action")
    if set(required_human) - set(keys(actions)):
        raise DreamError("Unknown declared human action in editorial review")
    by_key = {(item["kind"], item["path"], item["target"]): item for item in checks}
    for action in actions:
        key = action["kind"], action["path"], action["target"]
        check = by_key[key]
        if not check["reason"].strip() or check["acceptance_satisfied"] != "yes" or check["authority_verified"] != "yes":
            raise DreamError("Editorial action lacks verified authority or satisfied acceptance: " + check["reason"])
        human = key in required_human or check["requires_human"] == "yes"
        if key in required_human and check["requires_human"] != "yes":
            raise DreamError("Independent review weakened a declared human decision requirement")
        if human and (action["authority"] != "recorded_human" or not action["evidence"]):
            raise DreamError("A required human decision needs recorded source evidence")


class Editor(dream_author.Author):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.contract = CONTRACT
        self.editing = dict(dream_author.EDITING, task_workflow=dict(states=list(org.STATES),
            task_actions="Copy before_sha256 from the prior inventory target. The coordinator materializes states/history/archives before review.",
            metadata="Declare REQUIRES_HUMAN: yes or DECISION_AUTHORITY: human for tasks requiring a human. Preserve existing requirements.",
            archive_destination="Use .sucoder/wiki/archive/TOPIC.org; keep explanatory qualifications active."))
        self.run_id = digest(canonical([CONTRACT, self.run_id, BRIEFS, self.editing,
            {role: schema(role) for role in BRIEFS}, POLICY, org.CONTRACT]))
        self.extra_prior: dict[str, str] = {}
        self.required_human: set[tuple[str, str, str]] = set()
        self.moves: list[dict] = []
        self.materialized: dict | None = None
        self.validation_prior: dict[str, str] = {}
        self.permissions: dict = {}
        self.action_evidence: dict[tuple[str, str, str], tuple[str, str]] = {}

    def briefs(self):
        return BRIEFS

    def result_schema(self, role):
        return schema(role)

    def bundles(self):
        sources, prior = super().bundles()
        name = "editorial-time-" + self.run_id + ".json"
        try:
            value = dream._json(dream_run._read(self.fd, name, 1024))
        except FileNotFoundError:
            if self.source is None:
                raise DreamError("Editorial replay lacks its frozen execution timestamp", 3) from None
            value = dict(timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
            dream_run._store(self.fd, name, canonical(value) + b"\n")
        dream._object(value, ("timestamp",))
        self.timestamp = org.timestamp(value["timestamp"])
        return sources, prior

    def call(self, role, inputs):
        return super().call(role, dict(inputs, editorial=dict(contract=POLICY, proposed_at=self.timestamp)))

    def check_result(self, task, result):
        super().check_result(task, result)
        if task["role"] == "reviewer":
            if result["verdict"] == "lookup" and result["task_checks"]:
                raise DreamError("A reviewer lookup cannot contain premature task checks")
        elif result["decision"] in ("lookup", "unchanged") and result["task_actions"]:
            raise DreamError("Lookup/unchanged results cannot contain editorial actions")

    def _archive_context(self, path):
        if path in self.extra_prior or path not in self.context.files:
            return
        raw = self.context.blob(path)
        if len(self.context.trace) >= dream_context.MAX_REQUESTS or self.context.used["prior"] + len(raw) > self.plan["limits"]["max_prior_bytes"]:
            raise DreamError("Necessary archive context cannot fit the prior/request allowance", 3)
        response = dict(path=path, blob=self.context.files[path], text=raw.decode(), sha256=digest(raw),
            editable=False, state="complete", delivered_bytes=len(raw), searched_bytes=0)
        request = dict(kind="archive-destination", key=path)
        response["read_id"] = digest(canonical([request, response]))
        self.context.trace.append(dict(request=request, response=response))
        self.context.used["prior"] += len(raw)
        self.extra_prior[path] = raw.decode()

    def _incoming(self, moved, prior):
        used = 0
        for path in sorted(self.context.files):
            if path != wiki.INDEX and not (path.startswith(".sucoder/wiki/topics/") and path.endswith(".org")):
                continue
            raw = prior[path].encode() if path in prior else self.context.blob(path)
            used += len(raw)
            if used > dream_run.MAX_PLAN_BYTES:
                raise DreamError("Incoming-link inspection allowance exhausted; narrow archive work", 3)
            details = org.inspect({path: raw.decode()}, self.deadline)[path]
            needed = any(((path, link["path"]) if link["type"] == "custom-id" else
                (link["path"], link["search"][1:]) if link["type"] == "file" and link["search"] and link["search"].startswith("#") else None)
                in moved for link in details["links"])
            if needed and path not in prior:
                response = self.context.lookup(dict(kind="topic", key=path[len(".sucoder/wiki/topics/"):], target="", query="", start=0, end=0))
                if response["state"] != "complete" or not response.get("editable"):
                    raise DreamError("Incoming-link page could not be fully delivered: " + path, 3)

    def _authored_tasks(self, old, authored):
        before, after = heading_map(old), heading_map(authored)
        for path, details in authored.items():
            counts = lambda page: Counter(heading["sha256"] for heading in page["headings"] if heading["state"] and not heading["target"])
            if counts(details) - counts(old.get(path, dict(headings=[]))):
                raise DreamError("New editorial tasks require stable CUSTOM_ID targets")
        for path, details in old.items():
            for task in details["headings"]:
                if not task["state"]:
                    continue
                key = path, task["target"]
                current = after.get(key)
                if not task["target"]:
                    if task["sha256"] not in {heading["sha256"] for heading in authored[path]["headings"]}:
                        raise DreamError("Unanchored existing tasks must be preserved")
                elif current is None or any(current[field] != task[field] for field in ("state", "properties", "logbook", "closed")):
                    raise DreamError("Author must preserve existing task state, properties and history; use task_actions")
        for key, task in after.items():
            if key in before or not task["state"]:
                continue
            props = task["properties"]
            if task["state"] in org.TERMINAL or task["logbook"] or task["closed"]:
                raise DreamError("New tasks must start open without invented completion/history")
            if task["state"] not in org.STATES:
                raise DreamError("New editorial tasks must use the approved seven-state workflow")
            if task["state"] != "TODO" and not props.get("OWNER"):
                raise DreamError("Selected editorial tasks need an owner")
            if task["state"] == "REVIEW" and not (props.get("REVIEWER") and props.get("ACCEPTANCE")):
                raise DreamError("New REVIEW work needs a named reviewer and acceptance criteria")
            if task["state"] == "WAITING" and not (props.get("BLOCKER") and props.get("RESUME_WHEN")):
                raise DreamError("New WAITING work needs its blocker and resumption condition")

    def _human_source(self, citation, sources):
        unit = next(source for source in sources if source["unit_id"] == citation["unit_id"])
        if unit["kind"] == "git":
            return True  # Independent review must verify the recorded human attribution in maintained prose.
        if unit["kind"] == dream_view.KIND:
            return dream_view.user_quote(unit["provenance"], citation)
        for record in unit["provenance"]["records"]:
            if not record["byte_start"] <= citation["byte_start"] < citation["byte_end"] <= record["byte_end"]:
                continue
            if record["provenance"].get("summary") or record["provenance"].get("sidechain"):
                continue
            if not record["events"] or not all(event["kind"] == "message" and event["payload"].get("role") == "user" for event in record["events"]):
                continue
            for event in record["events"]:
                payload = event["payload"]
                texts = [payload.get("text"), payload.get("content")]
                if isinstance(payload.get("content"), list):
                    texts.extend(block.get("text") for block in payload["content"] if isinstance(block, dict))
                for text in texts:
                    if isinstance(text, str) and text and any(citation["quote"] in value or value in citation["quote"]
                            for value in (text, json.dumps(text)[1:-1], json.dumps(text, ensure_ascii=False)[1:-1])):
                        return True
        return False

    def proposal(self, proposal, prior, sources):
        self.required_human, self.moves, self.action_evidence = set(), [], {}
        inspected = {}
        def inspect(files):
            key = digest(canonical(files))
            if key not in inspected:
                inspected[key] = org.inspect(files, self.deadline)
            return inspected[key]
        if proposal["decision"] == "unchanged":
            self.materialized = proposal
            return super().proposal(proposal, prior, sources)
        if not proposal["rationale"].strip():
            raise DreamError("An editorial disposition needs a reason")
        authored = {}
        for page in proposal["pages"]:
            path = wiki.editable_path(page["path"])
            if path in authored:
                raise DreamError("Duplicate authored page")
            if path in self.context.files and path not in prior:
                raise DreamError("Existing page is outside the delivered prior context; include it before editing")
            authored[path] = page["content"]
        original = inspect(prior)
        initial = inspect(dict(prior, **authored))
        self._authored_tasks(original, initial)
        old_tasks, initial_tasks = heading_map(original), heading_map(initial)
        moved = set()
        for action in proposal["task_actions"]:
            key = action["path"], action["target"]
            old = old_tasks.get(key)
            if old is None or not old["state"] or action["before_sha256"] != old["sha256"]:
                raise DreamError("Editorial action must bind an existing frozen task")
            dream._text(action["note"], "editorial action reason")
            if action["kind"] == "archive":
                if initial_tasks[key]["sha256"] != old["sha256"]:
                    raise DreamError("Archive must preserve the frozen task body; accept any prose revision before moving it")
                org.action_shape({field: action[field] for field in ("kind", "path", "target", "before_sha256", "destination")})
                self._archive_context(action["destination"])
                for child in original[action["path"]]["headings"]:
                    if child["target"] and old["begin"] <= child["begin"] < old["end"]:
                        moved.add((action["path"], child["target"]))
            else:
                if action["state"] not in org.STATES or action["destination"]:
                    raise DreamError("Invalid transition state/destination")
                if not action["owner"].strip():
                    raise DreamError("An editorial transition needs a named owner")
                if action["state"] == "WAITING" and not (action["blocker"].strip() and action["resume_when"].strip()):
                    raise DreamError("WAITING needs a blocker and condition for resumption")
                if action["state"] != "WAITING" and (action["blocker"] or action["resume_when"]):
                    raise DreamError("Resolved work cannot introduce a current waiting blocker")
                if action["state"] == "REVIEW" and not (action["reviewer"].strip() and action["acceptance"].strip()):
                    raise DreamError("REVIEW needs a named reviewer and acceptance criteria")
                if action["state"] == "DONE" and not action["acceptance"].strip():
                    raise DreamError("DONE needs a checked acceptance criterion")
                if action["state"] in org.TERMINAL and declared_human(old):
                    self.required_human.add((action["kind"], *key))
                    if action["authority"] != "recorded_human" or not action["evidence"]:
                        raise DreamError("A declared human decision cannot be completed without recorded authority")
                if declared_human(old) and action["reviewer"] and old["properties"].get("REVIEWER") not in (None, action["reviewer"]):
                    raise DreamError("A declared human reviewer cannot be replaced by the agent")
                reviewer = old["properties"].get("REVIEWER")
                if action["state"] == "DONE" and reviewer:
                    if action["reviewer"] and action["reviewer"] != reviewer:
                        raise DreamError("Completion cannot replace its required named reviewer")
                    if action["authority"] == "agent":
                        raise DreamError("DONE requires its named review, not the author's assertion")
                    if reviewer.lower() not in ("independent native reviewer", "independent reviewer", "sue") and not action["evidence"]:
                        raise DreamError("Completion needs evidence of the required named review")
        if moved:
            self._incoming(moved, prior)
        base = dict(prior, **self.extra_prior)
        working = dict(base, **authored)
        current = heading_map(inspect(working))
        compiled = []
        for action in proposal["task_actions"]:
            native = {field: action[field] for field in ("kind", "path", "target")}
            native["before_sha256"] = current[action["path"], action["target"]]["sha256"]
            if action["kind"] == "archive":
                native["destination"] = action["destination"]
                working.setdefault(action["destination"], "#+title: Editorial archive\n#+SETUPFILE: " +
                    posixpath.relpath(wiki.SETTINGS_PATH, posixpath.dirname(action["destination"])) + "\n")
            else:
                fields = dict(owner="OWNER", reviewer="REVIEWER", acceptance="ACCEPTANCE", blocker="BLOCKER", resume_when="RESUME_WHEN")
                native.update(state=action["state"], note=action["note"],
                    properties={prop: action[field] for field, prop in fields.items() if action[field]})
                if action["state"] != "WAITING":
                    native["clear_properties"] = ["BLOCKER", "RESUME_WHEN"]
            compiled.append(native)
        before_native = inspect(working)
        transformed = (org.native(working, compiled, at=self.timestamp, deadline=self.deadline) if compiled else
            dict(inventory=before_native, files={}, moves=[]))
        self.moves = transformed["moves"]
        final = dict(working, **transformed["files"])
        pages = {path: text for path, text in final.items() if path not in base or text != base[path]}
        for path in list(pages):
            if any(heading["state"] for heading in transformed["inventory"][path]["headings"]):
                keywords = wiki.inventory({path: pages[path]}, self.deadline)[path]["keywords"]
                if not any(keyword["key"] in ("TODO", "SETUPFILE") for keyword in keywords):
                    pages[path] = "#+SETUPFILE: " + posixpath.relpath(wiki.SETTINGS_PATH, posixpath.dirname(path)) + "\n" + pages[path]
        old_changes = {change["path"]: change for change in proposal["changes"]}
        if set(old_changes) != set(authored) or len(old_changes) != len(proposal["changes"]):
            raise DreamError("Each authored page needs exactly one change disposition")
        for path, change in old_changes.items():
            if change["before_sha256"] != (digest(prior[path].encode()) if path in prior else "") or not change["reason"].strip():
                raise DreamError("Authored change differs from its frozen page")
        changes = [dict(path=path, before_sha256=digest(base[path].encode()) if path in base else "",
            reason=old_changes[path]["reason"] if path in old_changes else "Coordinator materialized validated editorial history/archive/link changes.") for path in sorted(pages)]
        refs = {(reference["path"], reference["target"]): dict(reference, citations=list(reference["citations"])) for reference in proposal["references"]}
        if len(refs) != len(proposal["references"]):
            raise DreamError("Duplicate authored explanatory-section reference")
        destinations = {(move["source"], target): move["destination"] for move in self.moves for target in move["targets"]}
        for action in proposal["task_actions"]:
            if not action["evidence"]:
                continue
            path = destinations.get((action["path"], action["target"]), action["path"])
            key = path, action["target"]
            reference = refs.setdefault(key, dict(path=path, target=action["target"], evidence_status="inference", citations=[]))
            if action["authority"] == "recorded_human":
                reference["evidence_status"] = "human_decision"
            reference["citations"].extend(citation for citation in action["evidence"] if citation not in reference["citations"])
            self.action_evidence[action["kind"], action["path"], action["target"]] = key
        self.materialized = dict(proposal, pages=[dict(path=path, content=text) for path, text in sorted(pages.items())],
            changes=changes, references=list(refs.values()))
        prior_inventory = inspect(base)
        derived = set()
        for path, details in transformed["inventory"].items():
            for target, sha in details["targets"].items():
                previous = before_native[path]["targets"].get(target)
                if sha != previous and (previous == prior_inventory.get(path, {}).get("targets", {}).get(target) or path.startswith(".sucoder/wiki/archive/")):
                    derived.add((path, target))
        self.validation_prior = dict(prior, **self.context.archives, **self.extra_prior)
        self.permissions = dict(approved_tasks={path: [heading["sha256"] for heading in details["headings"] if heading["state"]]
            for path, details in prior_inventory.items()}, archive_paths={path for path in pages if path.startswith(".sucoder/wiki/archive/")}, derived_targets=derived)
        value = super().proposal(self.materialized, prior, sources)
        resolved = {(reference["path"], reference["target"]): reference for reference in value[2]}
        for action in proposal["task_actions"]:
            key = action["kind"], action["path"], action["target"]
            if action["authority"] == "recorded_human":
                reference = resolved.get(self.action_evidence.get(key), {})
                own = {(citation["unit_id"], citation["quote"]) for citation in action["evidence"]}
                if not any((citation["unit_id"], citation["quote"]) in own and self._human_source(citation, sources)
                           for citation in reference.get("citations", [])):
                    raise DreamError("Recorded human authority needs a maintained decision record or native non-summary user evidence")
        return value

    def validate(self, proposal, prior, sources):
        return wiki.validate_proposal(self.plan, proposal, self.validation_prior, sources, self.deadline,
            wiki.LEGACY_CONTRACT, **self.permissions)

    def review_candidate(self, proposal, pages):
        return dict(self.materialized, editorial_moves=self.moves)

    def receipt_proposal(self, proposal):
        return self.materialized

    def review_findings(self, review, proposal):
        findings = list(review["findings"])
        if review["verdict"] == "accept":
            try:
                check_actions(proposal["task_actions"], review["task_checks"], self.required_human)
            except DreamError as exc:
                findings.append(dict(severity="blocking", path="", message=str(exc)))
        return findings

    def bindings(self, pages, prior):
        return super().bindings(dict(prior, **self.extra_prior, **pages) if pages else dict(prior, **self.extra_prior), prior)

    def prepare(self):
        pages, receipt = super().prepare()
        receipt["editorial"] = dict(contract=POLICY, proposed_at=self.timestamp,
            task_actions=self.materialized["task_actions"], required_human=[list(key) for key in sorted(self.required_human)], moves=self.moves)
        return pages, receipt
