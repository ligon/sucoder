"""Durable project dreaming, grounded in the exploratory launch ledger.

Bind the actual checkout, reuse private artifacts/locks and native role traces,
and hand selected evidence to the existing independently reviewed wiki writer.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
import errno
import os
from pathlib import Path
import re
import time
import uuid

from . import (attention, dream, dream_budget, dream_discover, dream_experience, dream_memory, dream_native,
               dream_roles, dream_run, dream_view, dream_wiki, dream_retry, dream_leads, dream_local,
               dream_inventory, dream_isolate, dream_session_scan)
from .config import AgentLauncher, BranchPrefixes, DreamSettings, MirrorSettings, _parse_dream
from .dream import DreamError, canonical, digest
from .observation_process import Budget
from .observations import WorkspaceBinding


CONTRACT = "exploratory-dream-job-v1"
# The charter's experience purpose (docs/dream/essay-alignment.org, S4); no fixed salience rule.
EXPERIENCE_PURPOSE = (
    "Keep this project's agent memory correct and useful. Verify existing memory pages and agent notes against "
    "what sessions actually did. Record what sessions learned that the repository does not already say: commands "
    "and flags that worked, failed approaches and traps, decisions with their dates and who made them, and open "
    "questions. Prefer correcting and pruning to adding.")
MODES = ("experience", "project")
# inventory.json holds the canonical manifest and a newline.
INVENTORY_FILE_BYTES = dream_discover.MAX_INVENTORY_BYTES + 1


def _read(fd, name):
    return dream._json(dream_run._read(fd, name, dream_run.MAX_PLAN_BYTES))


def _store(fd, name, value):
    dream_run._store(fd, name, canonical(value) + b"\n")


def _probe_locks(fd, directory):
    # Reuse private/no-follow lock validation before creating a job (ledger, 2).
    try:
        with dream_roles.run_lock(fd, "storage-probe"):
            pass
    except OSError as exc:
        if exc.errno != errno.ENOLCK:
            raise
        raise DreamError(f"Dream storage {directory} has no advisory lock support; "
            "set log_dir to storage with lock support", 3) from exc


def project(config, cwd=None):
    value = WorkspaceBinding(cwd or Path.cwd()).inspect(config, Budget(5))
    if value["status"] != "complete" or value["data"]["binding"] != "valid":
        raise DreamError("Run dream inside an available committed Git project", 3)
    root = Path(value["data"]["git_root"]["value"])
    matches = [(name, item) for name, item in config.mirrors.items()
        if root in (item.canonical_repo.resolve(), (config.mirror_root / item.mirror_dirname).resolve())]
    observed = value["data"]["mirror"]["value"]
    if not matches and observed in config.mirrors:
        matches = [(observed, config.mirrors[observed])]
    if len(matches) > 1:
        raise DreamError("Project matches multiple configured mirrors; resolve the ambiguous configuration")
    if matches:
        name, item = matches[0]
    else:
        name = root.name
        item = MirrorSettings(name=name, canonical_repo=root, mirror_name=root.name,
            branch_prefixes=BranchPrefixes(human=config.human_user, agent=config.agent_user))
    return root, name, item, value["data"]["branch"]["value"], value["data"]["head"]["value"]


def _binding(config, header):
    root = Path(header["project"]["path"])
    name = header["project"]["name"]
    item = MirrorSettings(name=name, canonical_repo=root, mirror_name=root.name,
        branch_prefixes=BranchPrefixes(human=config.human_user, agent=config.agent_user))
    return replace(config, mirror_root=root.parent, mirrors={name: item}, targets={}, log_dir=Path(header["directory"]))


def _limits(settings):
    return dict(max_elapsed_seconds=settings["max_total_seconds"], max_native_calls=settings["max_total_calls"],
        reserve_seconds=settings["reserve_seconds"], reserve_calls=settings["reserve_calls"])


def _budget_plan(header):
    return dict(limits=dict(max_elapsed_seconds=header["settings"]["max_total_seconds"],
        max_role_invocations=header["settings"]["max_total_calls"]),
        repositories={header["project"]["name"]: dict(path=header["project"]["path"])})


@contextmanager
def _job(path):
    path = Path(path).expanduser().absolute()
    if not re.fullmatch(r"[0-9a-f]{32}", path.name):
        raise DreamError("Use the private job directory returned by dream")
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise DreamError("Dream job directory must be private and account-owned", 3)
        header = _read(fd, "job.json")
        dream._object(header, ("contract", "job_id", "directory", "created_at", "project", "settings", "purpose", "budget_path"),
                      ("mode", "navigation_version", "leads", "memory_origin", "navigation_limits", "processing",
                       "inventory_sha256", "discovery_allocation"))
        if "leads" in header:
            dream_leads.validate(header["leads"])
        if header.get("mode", "project") not in MODES:
            raise DreamError("Invalid frozen exploratory job mode", 3)
        navigation = header.get("navigation_version", 1)
        if type(navigation) is not int or not 1 <= navigation <= dream_discover.NAVIGATION:
            raise DreamError("Invalid frozen exploratory navigation version", 3)
        if navigation >= 3:
            dream_session_scan.validate_limits(header.get("navigation_limits"))
            dream_isolate.validate(header.get("processing"))
            allocation = header.get("discovery_allocation", {})
            if (allocation.get("contract") != "automatic-discovery-v1" or
                    allocation.get("resolved") != header["settings"]["max_discovery_calls"] or
                    allocation.get("mode") not in ("auto", "fixed") or
                    not isinstance(header.get("inventory_sha256"), str)):
                raise DreamError("Invalid frozen navigation allocation or inventory binding", 3)
        if header["contract"] != CONTRACT or header["job_id"] != path.name or header["directory"] != str(path):
            raise DreamError("Exploratory job identity changed", 3)
        if _read(fd, "identity.json") != dict(sha256=digest(canonical(header))):
            raise DreamError("Frozen exploratory job settings changed", 3)
        settings = _parse_dream(header["settings"])
        # Jobs frozen before later settings existed record fewer keys; compare what they recorded.
        if ({k: v for k, v in asdict(settings).items() if k in header["settings"]} != header["settings"] or
                settings.harness not in ("codex", "claude") or not settings.model):
            raise DreamError("Invalid frozen exploratory job settings", 3)
        prepared = _read(fd, "prepared.json")
        if prepared.get("job_id") != header["job_id"] or prepared.get("budget") != _budget(header):
            raise DreamError("Exploratory job preparation or original budget identity changed", 3)
        yield fd, path, header
    finally:
        os.close(fd)


def prepare(config, job_path=None, *, cwd=None, overrides=None, budget_path=None, attempt=1, retry_seconds=None):
    dream_retry.validate_request(job_path, attempt, retry_seconds)
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    if type(attempt) is not int or not 1 <= attempt <= 1000000:
        raise DreamError("--attempt must be between 1 and 1000000")
    if job_path is not None:
        with _job(_resolve_job(config, job_path)) as (_, path, header):
            if "memory_origin" in header:
                expected_origin = {key: value for key, value in header["memory_origin"].items() if key != "expected_tip"}
                if dream_local.recognize(config, Path(header["project"]["path"])) != expected_origin:
                    raise DreamError("Configured local-tier memory binding changed; preserve this job", 3)
            expected = dict(header["settings"], purpose=header["purpose"], mode=header.get("mode", "project"))
            if any(key not in expected or value != expected[key] for key, value in overrides.items()):
                raise DreamError("Job settings are immutable; resume without changing its options", 3)
            if budget_path is not None and str(Path(budget_path).expanduser().absolute()) != header["budget_path"]:
                raise DreamError("A resumed job cannot select a different budget", 3)
            return dict(job_path=str(path), job_id=header["job_id"])
    root, name, item, branch, commit = project(config, cwd)
    if not branch or not commit or branch.startswith("dream/"):
        raise DreamError("Exploratory launch requires a committed non-dream branch; select one before launch", 3)
    mode = overrides.pop("mode", "experience")
    if mode not in MODES:
        raise DreamError("Dream mode must be experience or project")
    purpose = overrides.pop("purpose", EXPERIENCE_PURPOSE if mode == "experience" else dream_wiki.PURPOSE)
    dream._text(purpose, "dream purpose")
    hard = {key: overrides.pop(key) for key in ("max_total_tokens", "max_cost_usd") if key in overrides}
    settings = _parse_dream(overrides, item.dream or config.dream)
    launcher = item.agent_launcher if item.agent_launcher != AgentLauncher() else (config.agent_launcher or item.agent_launcher)
    harness = settings.harness or Path(launcher.command[0]).name
    model = settings.model or launcher.model
    if harness not in ("codex", "claude"):
        raise DreamError("Configure dream.harness as codex or claude, or supply --harness")
    if not model:
        raise DreamError("Configure dream.model (or agent_launcher.model), or supply --model; dreaming never prompts")
    dream._text(model, "configured native model")
    settings = replace(settings, harness=harness, model=model)
    memory_ref = dream_memory.branch_ref(settings.memory_branch) if settings.memory_branch else None
    if memory_ref == "refs/heads/" + branch:
        raise DreamError("dream.memory_branch must differ from the launch branch", 3)
    frozen = asdict(settings)
    # Capability rejection precedes job creation, native probes and all calls.
    dream_budget.limits(dict(limits=dict(max_elapsed_seconds=settings.max_total_seconds,
        max_role_invocations=settings.max_total_calls)), dict(_limits(frozen), **hard))
    logs = (config.log_dir or Path("~/.sucoder/logs")).expanduser().resolve() / "dream-jobs"
    if logs == root or logs.is_relative_to(root):
        raise DreamError("Keep dream job storage outside the project")
    logs.parent.mkdir(parents=True, exist_ok=True)
    parent_fd = dream._open_private_directory(logs)
    try:
        _probe_locks(parent_fd, logs)
    finally:
        os.close(parent_fd)
    job_id = uuid.uuid4().hex
    directory = logs / job_id
    fd = dream._open_private_directory(directory)
    try:
        target = Path(budget_path).expanduser().absolute() if budget_path is not None else directory / "budget"
        policy_keys = dict(max_total_seconds="max_elapsed_seconds", max_total_calls="max_native_calls",
            reserve_seconds="reserve_seconds", reserve_calls="reserve_calls")
        supplied = ({policy_keys[key]: value for key, value in overrides.items() if key in policy_keys}
            if (target / "policy.json").exists() else _limits(frozen))
        plan = dict(limits=dict(max_elapsed_seconds=settings.max_total_seconds, max_role_invocations=settings.max_total_calls),
            repositories={name: dict(path=str(root))})
        with dream_budget.open_job(plan, directory, target, supplied) as budget:
            frozen.update({key: budget.header["limits"][value] for key, value in policy_keys.items()})
            allocation = None
            if dream_discover.NAVIGATION >= 3:
                available = budget.summary()["remaining_calls"]
                reserve = budget.header["limits"]["reserve_calls"]
                requested = frozen["max_discovery_calls"]
                capacity = max(0, available - reserve - 1)
                resolved = min(available // 2 if requested is None else requested, capacity)
                if not resolved or budget.remaining() <= budget.header["limits"]["reserve_seconds"]:
                    budget.fail("no unreserved discovery and author capacity at preparation")
                frozen["max_discovery_calls"] = resolved
                allocation = dict(contract="automatic-discovery-v1", mode="auto" if requested is None else "fixed",
                    requested=requested, resolved=resolved, basis_remaining_calls=available,
                    reserve_calls=reserve, author_calls=1, wallet=budget.descriptor())
            elif frozen["max_discovery_calls"] is None:
                frozen["max_discovery_calls"] = 8
            project_binding = dict(name=name, path=str(root), base_ref="refs/heads/" + branch, base_commit=commit)
            if memory_ref:
                project_binding["memory_ref"] = memory_ref
            header = dict(contract=CONTRACT, job_id=job_id, directory=str(directory), created_at=time.time(),
                project=project_binding, settings=frozen, purpose=purpose, budget_path=str(target), mode=mode)
            if dream_discover.NAVIGATION > 1:
                # Version 1 stays implicit, so earlier headers (and their digests) are unchanged.
                header["navigation_version"] = dream_discover.NAVIGATION
            if allocation is not None:
                header.update(discovery_allocation=allocation, navigation_limits=dict(dream_session_scan.LIMITS),
                              processing=dict(dream_isolate.POLICY))
            recognized = dream_local.recognize(config, root) if memory_ref else None
            if recognized is not None:
                header["memory_origin"] = dream_local.synchronize(_source(config, header, time.monotonic() + 30),
                    recognized, memory_ref, directory)
            if dream_leads.VERSION or allocation is not None:
                source = _source(config, header, min(time.monotonic() + 30,
                    budget.deadline(dict(role="explorer")) if allocation is not None else budget.deadline()))
                sessions = _sessions(config, source, header) if mode == "experience" else None
                inventory = _inventory(source, header, sessions)
                if allocation is not None:
                    header["inventory_sha256"] = digest(canonical(inventory.manifest))
                    dream_run._store(fd, "inventory.json", canonical(inventory.manifest) + b"\n")
                if dream_leads.VERSION:
                    header["leads"] = dream_leads.snapshot(source, header, inventory)
            _store(fd, "job.json", header)
            _store(fd, "identity.json", dict(sha256=digest(canonical(header))))
            _store(fd, "prepared.json", dict(job_id=job_id, budget=budget.descriptor()))
    finally:
        os.close(fd)
    return dict(job_path=str(directory), job_id=job_id)


def _source(config, header, deadline):
    source = dream.GitSource(_binding(config, header), header["project"]["name"], deadline)
    if str(source.path) != header["project"]["path"]:
        raise DreamError("The frozen project path now resolves to a different checkout", 3)
    return source


def _base(source, header):
    project = header["project"]
    if project.get("memory_ref"):
        # The memory branch, not the launch branch, is the publication base;
        # evidence stays pinned to base_commit and publish checks the memory tip.
        return
    if (source.git("symbolic-ref", "--quiet", project["base_ref"], missing=True) is not None or
            source.commit(project["base_ref"]) != project["base_commit"]):
        raise DreamError("Project publication base changed during exploration; preserve this job and start a new plan", 3)


def _discovery(config, fd, header, budget, *, attempt=1, transport=None, readonly=False):
    deadline = time.monotonic() + 30 if readonly else budget.deadline(
        dict(role="explorer") if header.get("navigation_version", 1) >= 3 else None)
    source = _source(config, header, deadline)
    sessions = None
    frozen = None
    if header.get("navigation_version", 1) >= 3:
        frozen = dream._json(dream_run._read(fd, "inventory.json", INVENTORY_FILE_BYTES))
        if digest(canonical(frozen)) != header["inventory_sha256"]:
            raise DreamError("Frozen navigation inventory checksum differs", 3)
        if header.get("mode") == "experience":
            sessions = dict(sessions=frozen["sessions"], coverage=frozen["session_coverage"])
    elif header.get("mode") == "experience":
        if readonly:
            # Replay uses the sessions pinned at discovery; a resumed session may have grown since.
            sessions = dream._json(dream_run._read(fd, "inventory.json", INVENTORY_FILE_BYTES)).get("sessions", [])
        else:
            sessions = _sessions(config, source, header)
    inventory = _inventory(source, header, sessions)
    raw = canonical(inventory.manifest) + b"\n"
    if readonly:
        if dream_run._read(fd, "inventory.json", INVENTORY_FILE_BYTES) != raw:
            raise DreamError("Pinned project inventory changed", 3)
    else:
        dream_run._store(fd, "inventory.json", raw)
        _base(source, header)
    if readonly:
        version = _read(fd, "native.json")["version"]
        def no_inference(*args):
            raise DreamError("Incomplete exploratory job; inspection never invokes inference", 3)
        native = no_inference
    else:
        native = transport or dream_native.Transport(header["settings"]["harness"], deadline)
        version = native.version
        _store(fd, "native.json", dict(version=version))
    explorer = dream_discover.Explorer(header, fd, source, inventory, deadline, native, attempt=attempt, version=version)
    budget.attach(explorer)
    return explorer.prepare(), native


def _inventory(source, header, sessions):
    if header.get("navigation_version", 1) >= 3:
        return dream_discover.Inventory(source, header["project"]["base_commit"],
            sessions=sessions["sessions"] if sessions is not None else None,
            navigation=header["navigation_version"], limits=header["navigation_limits"], processing=header["processing"],
            session_coverage=sessions["coverage"] if sessions is not None else None)
    return dream_discover.Inventory(source, header["project"]["base_commit"], sessions=sessions,
                                   navigation=header.get("navigation_version", 1))


def _plan(config, fd, header, chosen, deadline):
    effective = _binding(config, header)
    source = _source(config, header, deadline)
    memory_ref = header["project"].get("memory_ref")
    request = dict(schema_version=1, repository=source.name, integration_ref=memory_ref or header["project"]["base_ref"],
        units=chosen["outcome"]["units"], limits=dict(max_elapsed_seconds=header["settings"]["max_total_seconds"]))
    if header.get("navigation_version", 1) >= 3:
        request["limits"]["max_source_bytes"] = dream_discover.MAX_SELECTED_BYTES
    topic = chosen["outcome"]["topic"]
    maintenance = ("Exploratory job " + header["job_id"] + ": " + _line(chosen["outcome"]["rationale"]))[:4096]
    _store(fd, "plan-intent.json", dict(request=request, maintenance=maintenance, topic=topic))
    try:
        pointer = _read(fd, "plan-pointer.json")
    except FileNotFoundError:
        _base(source, header)
        # Recovery of the exact saved plan avoids a new provisional repo UUID.
        directory = Path(header["directory"]) / "dream"
        existing = list(directory.glob("*/plan.json")) if directory.exists() else []
        if len(existing) > 1:
            raise DreamError("Exploratory plan handoff is ambiguous; preserved without another provider call", 3)
        if existing:
            path = existing[0]
        else:
            if memory_ref:
                # First use seeds the orphan memory branch from the launch commit's wiki.
                dream_memory.ensure(source, memory_ref, header["project"]["base_commit"],
                                    Path(header["directory"]), deadline)
            _store(fd, "sources.json", request)
            # The project tree the explorer inventoried is what pages may link (S5).
            value = dream.create_plan(effective, source.name, Path(header["directory"]) / "sources.json",
                request["integration_ref"], maintenance=maintenance, topic=topic,
                publication="in_place" if memory_ref else None,
                project_commit=header["project"]["base_commit"], memory_origin=header.get("memory_origin"),
                **(dict(inspection_deadline=deadline) if header.get("navigation_version", 1) >= 3 else {}))
            path = Path(value["plan_path"])
        pointer = dict(plan_path=str(path))
        _store(fd, "plan-pointer.json", pointer)
    path = Path(pointer["plan_path"])
    with dream_run._directory(effective, path) as (plan_fd, plan_directory):
        plan = dream_run._load(plan_fd, plan_directory.name, source.name)
        if memory_ref:
            bound = plan["integration_ref"] == memory_ref and dream_run.in_place_plan(plan)
        else:
            bound = plan["base_commit"] == header["project"]["base_commit"]
        if (not bound or plan.get("memory_origin") != header.get("memory_origin") or plan["maintenance"] != maintenance or
                plan["topic"] != topic or len(plan["units"]) != len(request["units"])):
            raise DreamError("Saved exploratory plan differs from the handoff intent", 3)
        for unit, proposed in zip(plan["units"], request["units"]):
            locator = proposed["locator"]
            if proposed["kind"] == dream_view.KIND:
                if (unit["kind"] != dream_view.KIND or unit["path"] != locator["path"] or
                        unit["thread_id"] != locator["thread_id"] or
                        [unit["raw"]["byte_start"], unit["raw"]["byte_end"]] != [locator["byte_start"], locator["byte_end"]]):
                    raise DreamError("Saved plan evidence differs from the exploratory choice", 3)
                continue
            raw = source.blob(source.entry(header["project"]["base_commit"], locator["path"]))
            lines = locator.get("lines", [1, raw.count(b"\n") + int(not raw.endswith(b"\n"))])
            if unit["path"] != locator["path"] or unit["commit"] != locator["commit"] or unit["lines"] != lines:
                raise DreamError("Saved plan evidence differs from the exploratory choice", 3)
    return path


def _line(text):
    return " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", text).split())


def _purpose(header, outcome):
    purpose = header["purpose"]
    if "navigation_coverage" in outcome:
        qualification = (" Navigation was selective over mutable sources; traversal does not establish coherent query absence. "
            "Scanned and delivered material is not accepted coverage; verify claims against the selected evidence.")
        purpose = purpose[:4096 - len(qualification)] + qualification
    value = " Exploration: " + _line(outcome["rationale"])
    if outcome["questions"]:
        value += " Important open questions (investigate and record, never assume answers): " + "; ".join(_line(q) for q in outcome["questions"])
    remaining = 4096 - len(purpose)
    if len(value) > remaining:
        note = " [Further discovery reasoning/questions are retained in the private job report.]"
        value = value[:max(0, remaining - len(note))] + note if remaining >= len(note) else ""
    return purpose + value


def _budget(header):
    path = Path(header["budget_path"])
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        policy = dream_budget._read(fd, "policy.json")
    finally:
        os.close(fd)
    return dict(contract=dream_budget.CONTRACT, directory=str(path), job_id=policy["job_id"],
        policy_sha256=digest(canonical(policy)))


def _result(header, outcome, attempt, written=None, plan_path=None):
    common = dict(operation="dream_job", job_id=header["job_id"], job_path=header["directory"],
        project=header["project"], attempt=attempt, questions=outcome["questions"], discovery=outcome)
    if written is not None:
        return dict(written, **common, plan_path=str(plan_path))
    return dict(schema_version=1, state="no_selection", rationale=outcome["rationale"], **common,
        note="Discovery selected no evidence; no independently reviewed wiki candidate was proposed.")


def run(config, job_path, attempt=1, *, transport=None, retry_seconds=None):
    job_path = _resolve_job(config, job_path)
    prepare(config, job_path, attempt=attempt, retry_seconds=retry_seconds)
    with _job(job_path) as (fd, path, header), dream_roles.run_lock(fd, "orchestrator"):
        name = "result-" + str(attempt).zfill(6) + ".json"
        try:
            saved = _read(fd, name)
        except FileNotFoundError:
            pass
        else:
            checked = _inspect(config, fd, header, attempt)
            if "memory_origin" in header and checked["state"] == "conflict":
                return dict(saved, state="conflict", validation=checked)
            return saved
        _store(fd, "attempt-" + str(attempt).zfill(6) + ".json", dict(attempt=attempt))
        try:
            with dream_budget.open_job(_budget_plan(header), path, Path(header["budget_path"])) as budget:
                if retry_seconds is not None:
                    budget.renew(attempt, retry_seconds)
                try:
                    chosen = _read(fd, "chosen.json")
                except FileNotFoundError:
                    if not budget.remaining():
                        budget.fail("exploratory job elapsed allowance exhausted")
                    outcome, transport = _discovery(config, fd, header, budget, attempt=attempt, transport=transport)
                    chosen = dict(attempt=attempt, outcome=outcome)
                    _store(fd, "chosen.json", chosen)
                else:
                    with dream_budget.restore(budget.descriptor()) as replay:
                        outcome, _ = _discovery(config, fd, header, replay, attempt=chosen["attempt"], readonly=True)
                    if outcome != chosen["outcome"]:
                        raise DreamError("Saved exploratory choice differs from its native evidence", 3)
                outcome = chosen["outcome"]
                if outcome["units"]:
                    try:
                        _read(fd, "plan-pointer.json")
                    except FileNotFoundError:
                        if not budget.remaining():
                            budget.fail("elapsed allowance exhausted before editorial handoff")
                    plan_path = _plan(config, fd, header, chosen,
                        budget.deadline() if budget.remaining() else time.monotonic() + 30)
                else:
                    result = _result(header, outcome, attempt)
                    _store(fd, name, result)
                    return result
            # Keep orchestration ownership while the writer reacquires the wallet.
            written = dream_wiki.run(_binding(config, header), header["project"]["name"], plan_path,
                header["settings"]["harness"], header["settings"]["model"], attempt, purpose=_purpose(header, outcome),
                transport=transport, budget_path=Path(header["budget_path"]))
            result = _result(header, outcome, attempt, written, plan_path)
            _store(fd, name, result)
            return result
        except (DreamError, OSError, UnicodeError) as exc:
            _store(fd, "failure-" + uuid.uuid4().hex + ".json", dict(error=str(exc), attempt=attempt,
                state="budget_exhausted" if isinstance(exc, dream_budget.Exhausted) else "incomplete"))
            raise


def start(config, job_path=None, *, cwd=None, overrides=None, budget_path=None, attempt=1, transport=None, retry_seconds=None):
    prepared = prepare(config, job_path, cwd=cwd, overrides=overrides, budget_path=budget_path,
        attempt=attempt, retry_seconds=retry_seconds)
    return run(config, Path(prepared["job_path"]), attempt, transport=transport, retry_seconds=retry_seconds)


def _resolve_job(config, path):
    path = Path(path).expanduser()
    if re.fullmatch(r"[0-9a-f]{32}", str(path)):
        path = (config.log_dir or Path("~/.sucoder/logs")).expanduser() / "dream-jobs" / path
    return path.absolute()


def _latest(config, cwd):
    root, _, _, _, _ = project(config, cwd)
    directory = (config.log_dir or Path("~/.sucoder/logs")).expanduser().resolve() / "dream-jobs"
    found = []
    if directory.exists():
        for path in directory.iterdir():
            if re.fullmatch(r"[0-9a-f]{32}", path.name) and not path.is_symlink():
                # Only published preparations participate in automatic lookup.
                # Explicit --job inspection still diagnoses incomplete jobs.
                handle = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    try:
                        dream_run._read(handle, "prepared.json", dream_run.MAX_PLAN_BYTES)
                    except FileNotFoundError:
                        continue
                finally:
                    os.close(handle)
                with _job(path) as (_, _, header):
                    if header["project"]["path"] == str(root):
                        found.append((header["created_at"], header["job_id"], path))
    if not found:
        raise DreamError("No exploratory dream job exists for this project", 3)
    return max(found)[2]


def _live_sessions(config):
    """Session IDs of running local agent launches (attention.scan_local_launches)."""
    return {sid for _, _, sid in attention.scan_local_launches(agent_user=config.agent_user, agent_uid=os.getuid()) if sid}


def _sessions(config, source, header):
    memory_ref = header["project"].get("memory_ref")
    memory_commit = dream_memory.tip(source, memory_ref) if memory_ref else None
    if header.get("navigation_version", 1) >= 3:
        return dream_inventory.prepare(source, config=config, header=dict(header, inventory_memory_commit=memory_commit),
                                       live=lambda: _live_sessions(config))
    return dream_experience.sessions(source, quiet_minutes=header["settings"].get("session_quiet_minutes", 30),
        live=_live_sessions(config), memory_commit=memory_commit, config=config)


def _inspect(config, fd, header, attempt):
    chosen = _read(fd, "chosen.json")
    with dream_budget.restore(_budget(header)) as budget:
        outcome, _ = _discovery(config, fd, header, budget, attempt=chosen["attempt"], readonly=True)
    if chosen["outcome"] != outcome:
        raise DreamError("Saved exploratory choice differs from its frozen native evidence", 3)
    saved = _read(fd, "result-" + str(attempt).zfill(6) + ".json")
    if outcome["units"]:
        pointer = _read(fd, "plan-pointer.json")
        path = Path(pointer["plan_path"])
        effective = _binding(config, header)
        checked = dream_run.status(effective, header["project"]["name"], path, saved["run_id"])
        with dream_run._directory(effective, path) as (plan_fd, _):
            written = _read(plan_fd, "native-" + saved["run_id"] + "-run.json")
        expected = _result(header, outcome, attempt, written, path)
    else:
        checked = dict(state="no_selection", discovery=outcome)
        expected = _result(header, outcome, attempt)
    if canonical(saved) != canonical(expected):
        raise DreamError("Exploratory result differs from its checked native evidence", 3)
    return checked


def _pending_publication(config, fd, header, attempt):
    """Inspect a retained reviewed candidate even when durability prevented a result."""
    try:
        pointer = _read(fd, "plan-pointer.json")
    except FileNotFoundError:
        return None
    effective = _binding(config, header)
    path = Path(pointer["plan_path"])
    with dream_run._directory(effective, path) as (plan_fd, _):
        for name in sorted(os.listdir(plan_fd)):
            if not re.fullmatch(r"native-[0-9a-f]{64}-spec\.json", name):
                continue
            spec = dream._json(dream_run._read(plan_fd, name, 16384))
            if spec.get("attempt") != attempt:
                continue
            try:
                dream_run._read(plan_fd, "candidate-" + spec["run_id"] + ".txt", 128)
            except FileNotFoundError:
                continue
            return dream_run.status(effective, header["project"]["name"], path, spec["run_id"])
    return None


def status(config, job_path=None, *, cwd=None):
    path = _resolve_job(config, job_path) if job_path is not None else _latest(config, cwd)
    with _job(path) as (fd, _, header):
        attempts = [int(name[8:-5]) for name in os.listdir(fd) if re.fullmatch(r"attempt-[0-9]{6,7}\.json", name)]
        attempt = max(attempts, default=1)
        result = dict(schema_version=1, operation="dream_job_status", state="prepared", job_id=header["job_id"],
            job_path=str(path), project=header["project"], attempt=attempt)
        try:
            saved = _read(fd, "result-" + str(attempt).zfill(6) + ".json")
        except FileNotFoundError:
            result["state"] = "incomplete" if attempts else "prepared"
            if "memory_origin" in header:
                checked = _pending_publication(config, fd, header, attempt)
                if checked is not None:
                    result.update(state=checked["state"], validation=checked)
        else:
            checked = _inspect(config, fd, header, attempt)
            result.update(saved)
            result.update(operation="dream_job_status", state=checked["state"], validation=checked)
        with dream_budget.restore(_budget(header)) as budget:
            result["budget"] = budget.summary()
        return result


def budget_status(config, job_path=None, *, cwd=None):
    path = _resolve_job(config, job_path) if job_path is not None else _latest(config, cwd)
    with _job(path) as (_, _, header), dream_budget.restore(_budget(header)) as budget:
        return dict(budget.summary(), job_path=str(path), exploratory_job_id=header["job_id"])
