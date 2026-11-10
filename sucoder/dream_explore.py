"""Durable project dreaming, grounded in the exploratory launch ledger.

Bind the actual checkout, reuse private artifacts/locks and native role traces,
and hand selected evidence to the existing independently reviewed wiki writer.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
import os
from pathlib import Path
import re
import time
import uuid

from . import dream, dream_budget, dream_discover, dream_native, dream_roles, dream_run, dream_wiki
from .config import AgentLauncher, BranchPrefixes, DreamSettings, MirrorSettings, _parse_dream
from .dream import DreamError, canonical, digest
from .observation_process import Budget
from .observations import WorkspaceBinding


CONTRACT = "exploratory-dream-job-v1"


def _read(fd, name):
    return dream._json(dream_run._read(fd, name, dream_run.MAX_PLAN_BYTES))


def _store(fd, name, value):
    dream_run._store(fd, name, canonical(value) + b"\n")


def project(config, cwd=None):
    value = WorkspaceBinding(cwd or Path.cwd()).inspect(config, Budget(5))
    if value["status"] != "complete" or value["data"]["binding"] != "valid":
        raise DreamError("Run dream inside an available committed Git project", 3)
    root = Path(value["data"]["git_root"]["value"])
    matches = [(name, item) for name, item in config.mirrors.items()
        if root in (item.canonical_repo.resolve(), (config.mirror_root / item.mirror_dirname).resolve())]
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
    return replace(config, mirror_root=root.parent, mirrors={name: item}, log_dir=Path(header["directory"]))


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
        dream._object(header, ("contract", "job_id", "directory", "created_at", "project", "settings", "purpose", "budget_path"))
        if header["contract"] != CONTRACT or header["job_id"] != path.name or header["directory"] != str(path):
            raise DreamError("Exploratory job identity changed", 3)
        if _read(fd, "identity.json") != dict(sha256=digest(canonical(header))):
            raise DreamError("Frozen exploratory job settings changed", 3)
        settings = _parse_dream(header["settings"])
        if asdict(settings) != header["settings"] or settings.harness not in ("codex", "claude") or not settings.model:
            raise DreamError("Invalid frozen exploratory job settings", 3)
        prepared = _read(fd, "prepared.json")
        if prepared.get("job_id") != header["job_id"] or prepared.get("budget") != _budget(header):
            raise DreamError("Exploratory job preparation or original budget identity changed", 3)
        yield fd, path, header
    finally:
        os.close(fd)


def prepare(config, job_path=None, *, cwd=None, overrides=None, budget_path=None, attempt=1):
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    if type(attempt) is not int or not 1 <= attempt <= 1000000:
        raise DreamError("--attempt must be between 1 and 1000000")
    if job_path is not None:
        with _job(job_path) as (_, path, header):
            expected = dict(header["settings"], purpose=header["purpose"])
            if any(key not in expected or value != expected[key] for key, value in overrides.items()):
                raise DreamError("Job settings are immutable; resume without changing its options", 3)
            if budget_path is not None and str(Path(budget_path).expanduser().absolute()) != header["budget_path"]:
                raise DreamError("A resumed job cannot select a different budget", 3)
            return dict(job_path=str(path), job_id=header["job_id"])
    root, name, item, branch, commit = project(config, cwd)
    if not branch or not commit or branch.startswith("dream/"):
        raise DreamError("Exploratory launch requires a committed non-dream branch; select one before launch", 3)
    purpose = overrides.pop("purpose", dream_wiki.PURPOSE)
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
    frozen = asdict(settings)
    # Capability rejection precedes job creation, native probes and all calls.
    dream_budget.limits(dict(limits=dict(max_elapsed_seconds=settings.max_total_seconds,
        max_role_invocations=settings.max_total_calls)), dict(_limits(frozen), **hard))
    logs = (config.log_dir or Path("~/.sucoder/logs")).expanduser().resolve() / "dream-jobs"
    if logs == root or logs.is_relative_to(root):
        raise DreamError("Keep dream job storage outside the project")
    logs.parent.mkdir(parents=True, exist_ok=True)
    parent_fd = dream._open_private_directory(logs)
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
            header = dict(contract=CONTRACT, job_id=job_id, directory=str(directory), created_at=time.time(),
                project=dict(name=name, path=str(root), base_ref="refs/heads/" + branch, base_commit=commit),
                settings=frozen, purpose=purpose, budget_path=str(target))
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
    if (source.git("symbolic-ref", "--quiet", project["base_ref"], missing=True) is not None or
            source.commit(project["base_ref"]) != project["base_commit"]):
        raise DreamError("Project publication base changed during exploration; preserve this job and start a new plan", 3)


def _discovery(config, fd, header, budget, *, attempt=1, transport=None, readonly=False):
    deadline = time.monotonic() + 30 if readonly else budget.deadline()
    source = _source(config, header, deadline)
    inventory = dream_discover.Inventory(source, header["project"]["base_commit"])
    raw = canonical(inventory.manifest) + b"\n"
    if readonly:
        if dream_run._read(fd, "inventory.json", dream_run.MAX_PLAN_BYTES) != raw:
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


def _plan(config, fd, header, chosen, deadline):
    effective = _binding(config, header)
    source = _source(config, header, deadline)
    request = dict(schema_version=1, repository=source.name, integration_ref=header["project"]["base_ref"],
        units=chosen["outcome"]["units"], limits=dict(max_elapsed_seconds=header["settings"]["max_total_seconds"]))
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
            _store(fd, "sources.json", request)
            value = dream.create_plan(effective, source.name, Path(header["directory"]) / "sources.json",
                request["integration_ref"], maintenance=maintenance, topic=topic)
            path = Path(value["plan_path"])
        pointer = dict(plan_path=str(path))
        _store(fd, "plan-pointer.json", pointer)
    path = Path(pointer["plan_path"])
    with dream_run._directory(effective, path) as (plan_fd, plan_directory):
        plan = dream_run._load(plan_fd, plan_directory.name, source.name)
        if (plan["base_commit"] != header["project"]["base_commit"] or plan["maintenance"] != maintenance or
                plan["topic"] != topic or len(plan["units"]) != len(request["units"])):
            raise DreamError("Saved exploratory plan differs from the handoff intent", 3)
        for unit, proposed in zip(plan["units"], request["units"]):
            locator = proposed["locator"]
            raw = source.blob(source.entry(header["project"]["base_commit"], locator["path"]))
            lines = locator.get("lines", [1, raw.count(b"\n") + int(not raw.endswith(b"\n"))])
            if unit["path"] != locator["path"] or unit["commit"] != locator["commit"] or unit["lines"] != lines:
                raise DreamError("Saved plan evidence differs from the exploratory choice", 3)
    return path


def _line(text):
    return " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", text).split())


def _purpose(header, outcome):
    value = " Exploration: " + _line(outcome["rationale"])
    if outcome["questions"]:
        value += " Important open questions (investigate and record, never assume answers): " + "; ".join(_line(q) for q in outcome["questions"])
    remaining = 4096 - len(header["purpose"])
    if len(value) > remaining:
        note = " [Further discovery reasoning/questions are retained in the private job report.]"
        value = value[:max(0, remaining - len(note))] + note if remaining >= len(note) else ""
    return header["purpose"] + value


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


def run(config, job_path, attempt=1, *, transport=None):
    prepare(config, job_path, attempt=attempt)
    with _job(job_path) as (fd, path, header), dream_roles.run_lock(fd, "orchestrator"):
        name = "result-" + str(attempt).zfill(6) + ".json"
        try:
            saved = _read(fd, name)
        except FileNotFoundError:
            pass
        else:
            _inspect(config, fd, header, attempt)
            return saved
        _store(fd, "attempt-" + str(attempt).zfill(6) + ".json", dict(attempt=attempt))
        try:
            with dream_budget.open_job(_budget_plan(header), path, Path(header["budget_path"])) as budget:
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


def start(config, job_path=None, *, cwd=None, overrides=None, budget_path=None, attempt=1, transport=None):
    prepared = prepare(config, job_path, cwd=cwd, overrides=overrides, budget_path=budget_path, attempt=attempt)
    return run(config, Path(prepared["job_path"]), attempt, transport=transport)


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


def status(config, job_path=None, *, cwd=None):
    path = Path(job_path) if job_path is not None else _latest(config, cwd)
    with _job(path) as (fd, _, header):
        attempts = [int(name[8:-5]) for name in os.listdir(fd) if re.fullmatch(r"attempt-[0-9]{6,7}\.json", name)]
        attempt = max(attempts, default=1)
        result = dict(schema_version=1, operation="dream_job_status", state="prepared", job_id=header["job_id"],
            job_path=str(path), project=header["project"], attempt=attempt)
        try:
            saved = _read(fd, "result-" + str(attempt).zfill(6) + ".json")
        except FileNotFoundError:
            result["state"] = "incomplete" if attempts else "prepared"
        else:
            checked = _inspect(config, fd, header, attempt)
            result.update(saved)
            result.update(operation="dream_job_status", state=checked["state"], validation=checked)
        with dream_budget.restore(_budget(header)) as budget:
            result["budget"] = budget.summary()
        return result


def budget_status(config, job_path=None, *, cwd=None):
    path = Path(job_path) if job_path is not None else _latest(config, cwd)
    with _job(path) as (_, _, header), dream_budget.restore(_budget(header)) as budget:
        return dict(budget.summary(), job_path=str(path), exploratory_job_id=header["job_id"])
