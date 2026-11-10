"""Persistent execution budgets, grounded in the dream implementation ledger.

Reuse private atomic artifacts and the pilot's reserve-before-invoke pattern.
This ledger counts native invocations, not hidden HTTP requests. Current native
adapters cannot guarantee token/currency ceilings; those settings fail closed.
"""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid

from . import dream, dream_run
from .dream import DreamError, canonical, digest


CONTRACT = "persistent-dream-budget-v1"
CLOSEOUT = frozenset(("reviewer", "repair", "verifier", "consolidator"))
INSTRUCTIONS = ("This is a frozen allocation snapshot; the coordinator enforces the current shared balance. "
    "All roles, lookups, repairs and explicit attempts share this job. Finish useful work within the work allowance, "
    "leaving closeout capacity for independent review and finalization. Record important unresolved questions; "
    "do not ask the human, invent approval or publish unchecked work. Tokens and currency are observations, not enforced ceilings here.")


class Exhausted(DreamError):
    def __init__(self, reason, budget):
        super().__init__("Dream job budget exhausted: " + reason, 3)
        self.reason, self.budget = reason, budget


def _clock():
    return dict(wall=time.time(), monotonic=time.monotonic(),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def _read(fd, name):
    return dream._json(dream_run._read(fd, name, dream_run.MAX_PLAN_BYTES))


def _store(fd, name, value):
    dream_run._store(fd, name, canonical(value) + b"\n")


def limits(plan, supplied=None):
    supplied = {key: value for key, value in (supplied or {}).items() if value is not None}
    allowed = ("max_elapsed_seconds", "max_native_calls", "reserve_seconds", "reserve_calls", "max_total_tokens", "max_cost_usd")
    dream._object(supplied, (), allowed)
    for key in allowed[:-1]:
        if key in supplied and (type(supplied[key]) is not int or supplied[key] < (0 if key.startswith("reserve_") else 1)):
            raise DreamError(key + " must be a positive integer (reserves may be zero)")
    if "max_cost_usd" in supplied:
        try:
            value = Decimal(str(supplied["max_cost_usd"]))
        except InvalidOperation:
            raise DreamError("max_cost_usd must be a finite positive amount") from None
        if not value.is_finite() or value <= 0:
            raise DreamError("max_cost_usd must be a finite positive amount")
    if "max_total_tokens" in supplied or "max_cost_usd" in supplied:
        raise DreamError("Current native dream adapters cannot enforce a hard token or currency ceiling; no provider request was made", 3)
    seconds = supplied.get("max_elapsed_seconds", plan["limits"]["max_elapsed_seconds"])
    calls = supplied.get("max_native_calls", plan["limits"]["max_role_invocations"])
    result = dict(max_elapsed_seconds=seconds, max_native_calls=calls,
        reserve_seconds=supplied.get("reserve_seconds", min(30, seconds // 10)),
        reserve_calls=supplied.get("reserve_calls", 1),
        max_total_tokens=None, max_cost_usd=None)
    if result["reserve_seconds"] >= seconds or result["reserve_calls"] > calls:
        raise DreamError("Closeout reserves exceed the job allowance")
    return result


class Job:
    def __init__(self, fd, path, header, *, readonly=False):
        self.fd, self.path, self.header, self.readonly = fd, path, header, readonly
        dream._object(header, ("contract", "job_id", "directory", "limits", "started", "deadline", "monotonic_deadline"))
        if (header["contract"] != CONTRACT or not re.fullmatch(r"[0-9a-f]{32}", header["job_id"]) or
                header["directory"] != str(path)):
            raise DreamError("Dream budget identity or storage directory changed", 3)
        policy = header["limits"]
        checked = limits(dict(limits=dict(max_elapsed_seconds=policy["max_elapsed_seconds"],
            max_role_invocations=policy["max_native_calls"])), policy)
        if checked != policy:
            raise DreamError("Invalid frozen budget limits", 3)
        started = header["started"]
        dream._object(started, ("wall", "monotonic", "boot_id"))
        if (not isinstance(started["boot_id"], str) or not started["boot_id"] or
                any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    for value in (started["wall"], started["monotonic"], header["deadline"], header["monotonic_deadline"])) or
                header["deadline"] != started["wall"] + policy["max_elapsed_seconds"] or
                header["monotonic_deadline"] != started["monotonic"] + policy["max_elapsed_seconds"]):
            raise DreamError("Invalid frozen budget clock", 3)
        self.policy_sha256 = digest(canonical(header))
        self.calls: list[dict] = []
        self.tasks: dict[str, str] = {}
        self.invoked: set[str] = set()
        self.invoke_lock = threading.Lock()
        names = os.listdir(fd)
        reservations = sorted(name for name in names if name.startswith("reservation-"))
        if (len(reservations) > policy["max_native_calls"] or reservations !=
                ["reservation-" + str(i).zfill(6) + ".json" for i in range(len(reservations))]):
            raise DreamError("Dream budget reservation journal is incomplete or exceeds its cap", 3)
        if any(name.startswith("returned-") and name.replace("returned-", "reservation-", 1) not in reservations for name in names):
            raise DreamError("Dream budget has an unreserved return record", 3)
        for name in reservations:
            reservation = _read(fd, name)
            dream._object(reservation, ("job_id", "policy_sha256", "assignment_id", "base_assignment_id",
                "run_id", "role", "reserved_at", "assignment_bytes", "submitted_sha256"))
            if (any(not isinstance(reservation[k], str) or not re.fullmatch(r"[0-9a-f]{64}", reservation[k])
                    for k in ("assignment_id", "base_assignment_id", "submitted_sha256")) or
                    type(reservation["assignment_bytes"]) is not int or reservation["assignment_bytes"] <= 0 or
                    type(reservation["reserved_at"]) not in (int, float) or not math.isfinite(reservation["reserved_at"]) or
                    reservation["reserved_at"] < started["wall"] or not isinstance(reservation["role"], str)):
                raise DreamError("Invalid dream budget reservation", 3)
            if (reservation.get("job_id") != header["job_id"] or reservation.get("policy_sha256") != self.policy_sha256 or
                    any(row["reservation"]["assignment_id"] == reservation.get("assignment_id") or
                        row["reservation"]["base_assignment_id"] == reservation.get("base_assignment_id") for row in self.calls)):
                raise DreamError("Dream budget reservation binding changed", 3)
            returned = name.replace("reservation-", "returned-", 1)
            if returned in names:
                row = _read(fd, returned)
                dream._object(row, ("reservation", "state", "usage"), ("elapsed_seconds", "error", "trace_sha256", "reason"))
                if row.get("reservation") != reservation or row.get("state") not in ("completed", "failed", "uncertain", "denied_before_provider"):
                    raise DreamError("Dream budget return differs from its reservation", 3)
                if row["usage"] is not None and not isinstance(row["usage"], dict):
                    raise DreamError("Invalid native usage in budget journal", 3)
                if row["state"] in ("completed", "failed"):
                    from . import dream_native
                    trace = dream._json(dream_run._read(fd, name.replace("reservation-", "trace-", 1), dream_native.MAX_TRACE * 2))
                    if (row.get("trace_sha256") != digest(canonical(trace)) or
                            trace.get("input_sha256") != reservation["submitted_sha256"]):
                        raise DreamError("Budget native trace binding changed", 3)
                    if row["state"] == "completed":
                        _, traced_usage = dream_native.parse_trace(trace["stdout"].encode(), trace["harness"])
                        if traced_usage != row["usage"]:
                            raise DreamError("Budget usage differs from its native trace", 3)
            else:
                row = dict(reservation=reservation, state="uncertain", usage=None)
            self.calls.append(row)
            self.invoked.add(reservation["assignment_id"])

    def descriptor(self):
        return dict(contract=CONTRACT, directory=str(self.path), job_id=self.header["job_id"], policy_sha256=self.policy_sha256)

    def remaining(self):
        now = _clock()
        if now["boot_id"] != self.header["started"]["boot_id"]:
            return 0.0
        latest = max([self.header["started"]["wall"]] + [row["reservation"]["reserved_at"] for row in self.calls])
        if now["wall"] < latest:
            return 0.0
        return max(0.0, min(self.header["deadline"] - now["wall"], self.header["monotonic_deadline"] - now["monotonic"]))

    def summary(self):
        totals: dict = dict(reported_input_tokens=0, reported_output_tokens=0, calls_without_token_usage=0,
            reported_cost_usd=None, calls_without_currency=0)
        cost = Decimal(0)
        for row in self.calls:
            usage = row.get("usage") or {}
            native = usage.get("native") or {}
            if not isinstance(native, dict):
                native = {}
            if not all(type(native.get(k)) is int and native[k] >= 0 for k in ("input_tokens", "output_tokens")):
                totals["calls_without_token_usage"] += 1
            for key in ("input_tokens", "output_tokens"):
                if type(native.get(key)) is int and native[key] >= 0:
                    totals["reported_" + key] += native[key]
            try:
                amount = Decimal(str(usage.get("currency_usd")))
            except InvalidOperation:
                amount = Decimal("NaN")
            if amount.is_finite() and amount >= 0:
                cost += amount
            else:
                totals["calls_without_currency"] += 1
        if not totals["calls_without_currency"]:
            totals["reported_cost_usd"] = str(cost)
        left = self.remaining()
        cap = self.header["limits"]["max_native_calls"]
        return dict(schema_version=1, operation="dream_budget", contract=CONTRACT, job_id=self.header["job_id"],
            directory=str(self.path), state="budget_exhausted" if not left or len(self.calls) >= cap else "available",
            limits=self.header["limits"], started_at=self.header["started"]["wall"], deadline=self.header["deadline"],
            remaining_seconds=left, reserved_calls=len(self.calls), remaining_calls=max(0, cap - len(self.calls)),
            call_states={state: sum(row["state"] == state for row in self.calls)
                for state in ("completed", "failed", "uncertain", "denied_before_provider")},
            usage=totals, hard_token_ceiling_supported=False, hard_currency_ceiling_supported=False,
            note="Native invocation counts include reserved/unknown outcomes, not hidden HTTP retries. Native-reported tokens/currency may be incomplete; unknown usage is never zero.")

    def fail(self, reason, task=None):
        report = self.summary()
        if not self.readonly:
            _store(self.fd, "stop-" + uuid.uuid4().hex + ".json", dict(reason=reason, budget=report,
                run_id=task.get("run_id") if task else None, next_role=task.get("role") if task else None,
                note="Completed roles, source coverage and proposals remain in the private plan artifacts; no unchecked candidate is authorized."))
        raise Exhausted(reason, report)

    def check(self, task=None):
        if self.readonly:
            raise DreamError("Read-only budget replay cannot prepare another paid request", 3)
        if not self.remaining():
            self.fail("elapsed allowance exhausted or clock domain changed", task)
        cap = self.header["limits"]
        if len(self.calls) >= cap["max_native_calls"]:
            self.fail("native invocation allowance exhausted", task)
        closeout = task is not None and task["role"] in CLOSEOUT
        if task is not None and not closeout:
            if len(self.calls) >= cap["max_native_calls"] - cap["reserve_calls"]:
                self.fail("remaining invocations are reserved for review and closeout", task)
            if self.remaining() <= cap["reserve_seconds"]:
                self.fail("remaining time is reserved for review and closeout", task)

    def deadline(self, task=None):
        left = self.remaining()
        if task is not None and task["role"] not in CLOSEOUT:
            left -= self.header["limits"]["reserve_seconds"]
        return time.monotonic() + max(0.0, left)

    def attach(self, coordinator):
        if getattr(coordinator, "budget", None) is not None:
            raise DreamError("Coordinator already has a job budget")
        coordinator.budget = self
        coordinator.run_id = digest(canonical([CONTRACT, coordinator.run_id, self.descriptor()]))
        coordinator.transport = self.transport(coordinator.transport)

    def assignment(self, task):
        base = task["assignment_id"]
        name = "advice-" + base + ".json"
        try:
            saved = _read(self.fd, name)
        except FileNotFoundError:
            self.check(task)
            saved = dict(base_sha256=digest(canonical(task)), policy_sha256=self.policy_sha256,
                advice=dict(contract=CONTRACT, job_id=self.header["job_id"], instructions=INSTRUCTIONS,
                    as_of=_clock()["wall"], remaining_seconds=self.remaining(),
                    remaining_calls=self.header["limits"]["max_native_calls"] - len(self.calls),
                    reserve_calls=self.header["limits"]["reserve_calls"], reserve_seconds=self.header["limits"]["reserve_seconds"],
                    deadline=self.header["deadline"], usage=self.summary()["usage"]))
            _store(self.fd, name, saved)
        if saved.get("base_sha256") != digest(canonical(task)) or saved.get("policy_sha256") != self.policy_sha256:
            raise DreamError("Frozen budget advice differs from its assignment", 3)
        advice = saved["advice"]
        dream._object(advice, ("contract", "job_id", "instructions", "as_of", "remaining_seconds", "remaining_calls",
            "reserve_calls", "reserve_seconds", "deadline", "usage"))
        if (advice["contract"] != CONTRACT or advice["job_id"] != self.header["job_id"] or
                advice["instructions"] != INSTRUCTIONS or advice["deadline"] != self.header["deadline"] or
                any(advice[key] != self.header["limits"][key] for key in ("reserve_calls", "reserve_seconds"))):
            raise DreamError("Frozen budget advice policy changed", 3)
        value = dict(task, budget=saved["advice"])
        value.pop("assignment_id")
        value["assignment_id"] = digest(canonical(value))
        self.tasks[value["assignment_id"]] = base
        prior = next((row for row in self.calls if row["reservation"]["base_assignment_id"] == base), None)
        if prior is not None and prior["reservation"]["assignment_id"] != value["assignment_id"]:
            raise DreamError("Frozen budget advice changed after its native reservation", 3)
        return value

    def reserve(self, task):
        self.check(task)
        if task["assignment_id"] not in self.tasks:
            raise DreamError("Native request lacks frozen budget advice", 3)
        if any(row["reservation"]["assignment_id"] == task["assignment_id"] for row in self.calls):
            raise DreamError("Budget already reserved this request; inspect its outcome and choose an explicit new attempt", 3)
        record = dict(job_id=self.header["job_id"], policy_sha256=self.policy_sha256,
            assignment_id=task["assignment_id"], base_assignment_id=self.tasks[task["assignment_id"]],
            run_id=task["run_id"], role=task["role"], reserved_at=_clock()["wall"], assignment_bytes=len(canonical(task)),
            submitted_sha256=digest(canonical(task)))
        _store(self.fd, "reservation-" + str(len(self.calls)).zfill(6) + ".json", record)
        self.calls.append(dict(reservation=record, state="uncertain", usage=None))

    def denied(self, task):
        for row in self.calls:
            if row["reservation"]["assignment_id"] == task["assignment_id"] and row["state"] == "denied_before_provider":
                self.fail("request was denied before provider invocation", task)

    def transport(self, native):
        job = self
        class Scoped:
            version = getattr(native, "version", "budget-replay")

            def __call__(self, harness, model, task, schema, deadline):
                index = next((i for i, row in enumerate(job.calls) if row["reservation"]["assignment_id"] == task["assignment_id"]), None)
                if job.readonly or index is None:
                    raise DreamError("Native invocation has no writable job reservation", 3)
                row = job.calls[index]
                with job.invoke_lock:
                    if task["assignment_id"] in job.invoked:
                        raise DreamError("A reserved provider request cannot be silently repeated", 3)
                    job.invoked.add(task["assignment_id"])
                key = str(index).zfill(6)
                work_reserved = task["role"] not in CLOSEOUT and job.remaining() <= job.header["limits"]["reserve_seconds"]
                if not job.remaining() or work_reserved:
                    row.update(state="denied_before_provider", reason="job/work allowance exhausted before provider startup")
                    _store(job.fd, "returned-" + key + ".json", row)
                    job.fail(row["reason"], task)
                started = time.monotonic()
                try:
                    result, usage, trace = native(harness, model, task, schema, min(deadline, job.deadline(task)))
                except BaseException as exc:
                    row.update(state="uncertain", elapsed_seconds=time.monotonic() - started, error=type(exc).__name__)
                    _store(job.fd, "returned-" + key + ".json", row)
                    raise
                dream_run._store(job.fd, "trace-" + key + ".json", canonical(trace) + b"\n")
                row.update(state="completed" if result is not None else "failed", usage=usage,
                    elapsed_seconds=time.monotonic() - started, trace_sha256=digest(canonical(trace)))
                _store(job.fd, "returned-" + key + ".json", row)
                if not job.remaining():
                    job.fail("elapsed allowance exhausted during native work", task)
                if result is None and task["role"] not in CLOSEOUT and job.remaining() <= job.header["limits"]["reserve_seconds"]:
                    job.fail("ordinary work time exhausted; closeout time remains reserved", task)
                return result, usage, trace
        return Scoped()


def _path(plan, directory, path):
    path = Path(path).expanduser().absolute() if path is not None else directory / "job-budget"
    resolved = path.resolve()
    if any(resolved == Path(item["path"]).resolve() or resolved.is_relative_to(Path(item["path"]).resolve())
            for item in plan["repositories"].values()):
        raise DreamError("Keep job budgets outside inspected repositories")
    return path


@contextmanager
def open_job(plan, directory, path=None, supplied=None):
    from .dream_roles import run_lock
    requested = limits(plan, supplied)
    path = _path(plan, directory, path)
    fd = dream._open_private_directory(path)
    try:
        with run_lock(fd, "job"):
            try:
                header = _read(fd, "policy.json")
            except FileNotFoundError:
                if any(name.startswith(("reservation-", "advice-", "returned-")) for name in os.listdir(fd)):
                    raise DreamError("Existing job budget lost its policy", 3) from None
                now = _clock()
                header = dict(contract=CONTRACT, job_id=uuid.uuid4().hex, directory=str(path), limits=requested,
                    started=now, deadline=now["wall"] + requested["max_elapsed_seconds"],
                    monotonic_deadline=now["monotonic"] + requested["max_elapsed_seconds"])
                _store(fd, "policy.json", header)
            else:
                for key, value in (supplied or {}).items():
                    if value is not None and header["limits"].get(key) != value:
                        raise DreamError("Job budget limits are immutable; attempts/restarts cannot reset them", 3)
            expected = dict(policy_sha256=digest(canonical(header)))
            try:
                anchor = _read(fd, "identity.json")
            except FileNotFoundError:
                if any(name.startswith("reservation-") for name in os.listdir(fd)):
                    raise DreamError("Existing job budget lost its identity anchor", 3) from None
                _store(fd, "identity.json", expected)
            else:
                if anchor != expected:
                    raise DreamError("Frozen job budget policy changed", 3)
            job = Job(fd, path, header)
            try:
                yield job
            except (DreamError, OSError, UnicodeError) as exc:
                if not isinstance(exc, Exhausted) and not job.remaining():
                    job.fail("elapsed allowance exhausted during job: " + str(exc))
                raise
    finally:
        os.close(fd)


@contextmanager
def restore(descriptor):
    if descriptor is None:
        yield None
        return
    dream._object(descriptor, ("contract", "directory", "job_id", "policy_sha256"))
    path = Path(descriptor["directory"])
    if not path.is_absolute() or not path.is_dir():
        raise DreamError("Saved private job budget is unavailable", 3)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise DreamError("Saved job budget directory is not private and account-owned", 3)
        header = _read(fd, "policy.json")
        if _read(fd, "identity.json") != dict(policy_sha256=digest(canonical(header))):
            raise DreamError("Frozen job budget policy changed", 3)
        job = Job(fd, path, header, readonly=True)
        if descriptor != job.descriptor():
            raise DreamError("Saved job budget descriptor changed", 3)
        yield job
    finally:
        os.close(fd)


def inspect(config, mirror, plan_path, path=None):
    with dream_run._directory(config, plan_path) as (fd, directory):
        plan = dream_run._load(fd, directory.name, mirror)
        target = _path(plan, directory, path)
        if not target.is_dir():
            raise DreamError("No persisted job budget exists yet", 3)
        budget_fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            header = _read(budget_fd, "policy.json")
        finally:
            os.close(budget_fd)
        descriptor = dict(contract=CONTRACT, directory=str(target), job_id=header["job_id"], policy_sha256=digest(canonical(header)))
        with restore(descriptor) as job:
            return job.summary()


def completed(config, mirror, plan_path, fd, job, harness, model, attempt, purpose=None):
    """After expiry only replay a fully published result, with fresh validation."""
    from . import dream_progress
    for name in sorted(os.listdir(fd)):
        if not re.fullmatch(r"native-[0-9a-f]{64}-spec\.json", name):
            continue
        spec = _read(fd, name)
        # Accepted shared progress has no paid assignments and is independent of
        # attempt/budget identity. Its existing proof is still revalidated below.
        reused = spec.get("contract") == dream_progress.CONTRACT and purpose is not None
        if (spec.get("harness") != harness or spec.get("model") != model or spec.get("purpose") != purpose or
                (not reused and (spec.get("budget") != job.descriptor() or spec.get("attempt") != attempt))):
            continue
        try:
            result = _read(fd, "native-" + spec["run_id"] + "-run.json")
        except FileNotFoundError:
            continue
        if result.get("state") not in (("already_integrated",) if reused else ("candidate_ready", "unchanged")):
            continue
        checked = dream_run.status(config, mirror, plan_path, spec["run_id"])
        if checked["state"] not in ("candidate_ready", "already_integrated", "unchanged"):
            raise DreamError("Completed budgeted result needs reconciliation; inspect status", 3)
        return result
    return None
