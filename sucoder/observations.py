"""Version 1 observation service. No provisioning, lifecycle or record writes.

The human report and its interactive transports intentionally remain separate.
Sequential probes satisfy the concurrency bound and share one request budget.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import pwd
import shlex
import socket
import stat
import time
import uuid

import yaml

from .observation_process import Budget, Capture, ObservationFailure, capture
from .session import RemoteSession
from .sessions_report import (SQUEUE_FORMAT, SESSION_PANE_PROBE_SH,
                              parse_squeue, parse_pane_status, match_target)
from .mirror import _sanitize_session_token, confined_tmux_target

VERSION = "1.0"
MAX_BYTES = 256 * 1024
MESSAGES = {
    "TIMEOUT": "The observation budget expired; uninspected state remains unknown.",
    "OUTPUT_LIMIT": "Observation output was bounded; narrow the request and repeat.",
    "AUTH_UNAVAILABLE": "Existing noninteractive authority is unavailable.",
    "ROUTE_UNAVAILABLE": "The configured noninteractive route is unavailable.",
    "PROBE_FAILED": "A probe did not return a valid complete observation.",
    "RELAY_UNSUPPORTED": "The configured relay does not support structured observations.",
    "SCHEMA_UNSUPPORTED": "The response schema version is unsupported.",
    "INVALID_RESPONSE": "The observation response is malformed.",
    "INVALID_ARGUMENT": "The observation arguments are invalid.",
    "CONFIG_INVALID": "The observation configuration is invalid or unreadable.",
    "INTERNAL_ERROR": "The observation could not be completed due to an internal error.",
    "WORKSPACE_CHANGED": "The bound workspace changed; bind a new observer.",
    "WORKSPACE_UNAVAILABLE": "A readable git workspace could not be established.",
    "RECORD_CORRUPT": "A session record has an invalid structure.",
    "RECORD_UNREADABLE": "Session records could not be read completely.",
    "IDENTITY_AMBIGUOUS": "The observation cannot be uniquely associated with configured identity.",
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def identity():
    account = pwd.getpwuid(os.geteuid()).pw_name
    host = socket.gethostname()
    return dict(host=host, account=account, configuration_authority=account + "@" + host,
                discovery_account=None)


def fact(value=None, state="unknown", source="none"):
    return dict(value=value, state=state, source=source)


class Result:
    def __init__(self, operation, budget, observer=None):
        self.budget = budget
        self.value = dict(schema_version=VERSION, operation=operation,
                          observation_id=str(uuid.uuid4()), started_at=now(), finished_at=now(),
                          elapsed_ms=0, status="complete", observer=observer or identity(),
                          sources=[], coverage=[], errors=[], truncated=False, data=None)

    @property
    def authority(self):
        return self.value["observer"]["configuration_authority"]

    def source(self, sid, kind, targets=(), account=None):
        if not any(s["source_id"] == sid for s in self.value["sources"]):
            self.value["sources"].append(dict(source_id=sid, kind=kind, targets=list(targets),
                                             configuration_authority=self.authority,
                                             discovery_account=account))

    def error(self, code, sid=None):
        row = dict(code=code, source_id=sid, message=MESSAGES[code])
        if row not in self.value["errors"]:
            self.value["errors"].append(row)
        if code == "OUTPUT_LIMIT":
            self.value["truncated"] = True

    def coverage(self, sid, scope, state, count=None, reason=None):
        self.value["coverage"].append(dict(source_id=sid, scope=scope, state=state,
                                           count=count, reason=reason))
        if reason and state != "skipped":
            self.error(reason, sid)

    def finish(self):
        v = self.value
        if v["status"] != "error":
            failed = bool(v["errors"]) or any(c["state"] not in ("complete", "skipped")
                                                  for c in v["coverage"])
            useful = any(c["state"] == "complete" and c["scope"] != "records"
                         for c in v["coverage"])
            if v["operation"] == "list_sessions":
                useful = useful or bool(v["data"]["sessions"])
            v["status"] = ("partial" if useful else "unavailable") if failed else "complete"
        v["finished_at"] = now()
        v["elapsed_ms"] = max(0, int((time.monotonic() - self.budget.started) * 1000))
        return bounded(v)


def bounded(value):
    """On excess, retain a small valid envelope rather than broken references.

    Discarding the payload is deliberately conservative: omitted data cannot
    accidentally support an absence claim. The aggregate explicitly covers it.
    """
    def excessive(x):
        if isinstance(x, str):
            return len(x) > 4096
        if isinstance(x, list):
            return any(excessive(y) for y in x)
        if isinstance(x, dict):
            return any(excessive(y) for y in x.values())
        return False
    data = value.get("data") or {}
    overflow = (excessive(value) or len(value["sources"]) > 256 or
                len(value["coverage"]) > 256 or len(value["errors"]) > 100 or
                any(len(data.get(k, [])) > n for k, n in
                    (("sessions", 500), ("records", 500), ("selected_targets", 100))) or
                any(len(s["targets"]) > 100 for s in value["sources"]) or
                any(len(s[k]) > 100 for s in data.get("sessions", [])
                    for k in ("targets", "mirror_candidates", "record_keys")) or
                len(json.dumps(value, ensure_ascii=True).encode()) + 1 > MAX_BYTES)
    if overflow:
        value["truncated"] = True
        value["sources"] = [dict(source_id="aggregate:omitted", kind="aggregate", targets=[],
                                 configuration_authority=None, discovery_account=None)]
        value["coverage"] = [dict(source_id="aggregate:omitted", scope="omitted_sources",
                                  state="partial", count=None, reason="OUTPUT_LIMIT")]
        value["errors"] = [dict(code="OUTPUT_LIMIT", source_id="aggregate:omitted",
                                message=MESSAGES["OUTPUT_LIMIT"])]
        if value["operation"] == "list_sessions":
            value["data"] = dict(scope="configured_remote_targets", selected_targets=[],
                                  sessions=[], records=[])
        else:
            value["data"] = context_data("")
        for key in value["observer"]:
            if isinstance(value["observer"][key], str):
                value["observer"][key] = value["observer"][key][:4096]
    if value["truncated"]:
        value["status"] = "partial"
    return value


def context_data(requested):
    data = {key: fact() for key in ("git_root", "branch", "head", "mirror", "durable_mirror",
                                    "allocation_id", "local_root", "confinement")}
    data.update(binding="unavailable", requested_workspace=str(requested))
    return data


class WorkspaceBinding:
    """Reusable binding for a future adapter; no process-global cwd changes."""
    def __init__(self, workspace):
        self.requested = os.path.abspath(workspace)
        self.bound = None

    def inspect(self, config, budget, runner=capture):
        result = Result("session_context", budget)
        result.source("workspace", "workspace")
        data = context_data(self.requested)
        result.value["data"] = data
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")

        def git(*args):
            return runner(["git", "-c", "core.hooksPath=/dev/null", "-c", "credential.helper=",
                           *args], budget, cwd=self.requested, env=env)
        try:
            root = git("rev-parse", "--show-toplevel")
            directory = git("rev-parse", "--absolute-git-dir")
            if root.returncode or directory.returncode:
                raise ObservationFailure("WORKSPACE_UNAVAILABLE")
            paths = (Path(self.requested).resolve(strict=True),
                     Path(root.stdout.strip()).resolve(strict=True),
                     Path(directory.stdout.strip()).resolve(strict=True))
            current = tuple((str(p), p.stat().st_dev, p.stat().st_ino) for p in paths)
            if self.bound is not None and self.bound != current:
                raise ObservationFailure("WORKSPACE_CHANGED")
            self.bound = current
            branch = git("symbolic-ref", "--quiet", "--short", "HEAD")
            head = git("rev-parse", "--verify", "--quiet", "HEAD")
            # A null HEAD is valid only for a symbolic, unborn branch.
            if branch.returncode not in (0, 1) or head.returncode not in (0, 1):
                raise ObservationFailure("WORKSPACE_UNAVAILABLE")
            if head.returncode and branch.returncode:
                raise ObservationFailure("WORKSPACE_UNAVAILABLE")
            data.update(binding="valid", git_root=fact(str(paths[1]), "observed", "local"),
                        branch=fact(branch.stdout.strip() if branch.returncode == 0 else None,
                                    "observed", "local"),
                        head=fact(head.stdout.strip() if head.returncode == 0 else None,
                                  "observed", "local"))
            for key, variable in (("allocation_id", "SLURM_JOB_ID"),
                                  ("local_root", "SUCODER_LOCAL_ROOT")):
                if os.environ.get(variable):
                    data[key] = fact(os.environ[variable], "reported", "environment")
            matches = []
            for name, mirror in config.mirrors.items():
                durable = (config.mirror_root.expanduser() / mirror.mirror_dirname).resolve()
                if durable == paths[1]:
                    matches.append((name, str(durable), mirror.remote))
            # A local-tier origin alone is insufficient. Require the exact
            # configured work_path layout AND a local-path origin matching the
            # configured durable mirror. Environment claims are not used here.
            from .local_tier import work_path
            origin = None
            for name, mirror in config.mirrors.items():
                for remote in config.targets.values():
                    if not remote.slurm or not remote.slurm.local_disk:
                        continue
                    job_part = paths[1].parent.parent.name
                    if not job_part.startswith("job") or not job_part[3:].isdigit():
                        continue
                    expected = Path(work_path(remote.slurm.local_disk, _sanitize_session_token(name), int(job_part[3:])))
                    if expected.resolve() != paths[1]:
                        continue
                    durable = remote.mirror_root.expanduser() / mirror.mirror_dirname
                    if origin is None:
                        origin_reply = git("config", "--get", "remote.origin.url")
                        origin = origin_reply.stdout.strip() if origin_reply.returncode == 0 else ""
                    if origin and Path(origin).is_absolute() and Path(origin).resolve() == durable.resolve():
                        matches.append((name, str(durable.resolve()), remote))
            # Several targets giving the same path still leave confinement
            # ambiguous; never pick a target based on environment alone.
            if len(matches) == 1:
                name, durable, remote = matches[0]
                data["mirror"] = fact(name, "reported", "configuration")
                data["durable_mirror"] = fact(durable, "reported", "configuration")
                if remote and remote.slurm:
                    data["confinement"] = fact(str(remote.slurm.confined).lower(), "reported", "configuration")
            elif len(matches) > 1:
                data["mirror"] = fact(None, "ambiguous")
                data["durable_mirror"] = fact(None, "ambiguous")
                result.error("IDENTITY_AMBIGUOUS", "workspace")
            result.coverage("workspace", "git", "complete", 1)
        except (ObservationFailure, OSError) as exc:
            code = exc.code if isinstance(exc, ObservationFailure) else "WORKSPACE_UNAVAILABLE"
            if self.bound is not None and code == "WORKSPACE_UNAVAILABLE":
                code = "WORKSPACE_CHANGED"
            data["binding"] = "changed" if code == "WORKSPACE_CHANGED" else "unavailable"
            result.coverage("workspace", "git", "timeout" if code == "TIMEOUT" else "unavailable",
                            reason=code)
        return result.finish()


def source_id(authority, endpoint, account, target, kind):
    # Versioned, canonical tuple. Unknown accounts deliberately prevent dedup.
    value = [authority, endpoint, account, None if account else target, kind]
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def endpoint_for(remote, target):
    # Options can alter routing (HostName, Port, ProxyCommand). Rather than
    # fingerprint secrets or assume equivalent aliases, isolate such targets.
    distinct = any(k.lower() != "user" for k in remote.ssh_options)
    return [remote.gateway, target if distinct else None]


def account_for(remote):
    return remote.remote_user or next((v for k, v in remote.ssh_options.items()
                                       if k.lower() == "user"), None)


def pane_class(text):
    status = parse_pane_status(text)
    if status.command in {"claude", "codex", "gemini", "aider", "opencode", "goose", "kimi"}:
        return "recognized_agent_command"
    if status.agent_mode == "codex-remote-control":
        return "service_marker"
    if status.command in {"bash", "sh", "zsh", "dash", "ksh", "fish", "-bash", "login"}:
        return "shell_command"
    return "unknown"


class Transport:
    def __init__(self, budget, runner=capture):
        self.budget, self.runner = budget, runner
        self.direct = {}

    def options(self, remote, host):
        # First value wins in OpenSSH. Do not create masters or forwarding.
        opts = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ControlMaster=no",
                "-o", "ControlPersist=no", "-o", "ClearAllForwardings=yes", "-o", "RequestTTY=no",
                "-o", "ForwardAgent=no", "-o", "ForwardX11=no", "-o", "PermitLocalCommand=no"]
        if account_for(remote):
            opts += ["-l", account_for(remote)]
        socket_name = host
        if remote.host and host == remote.gateway:
            key = id(remote)
            if key not in self.direct:
                resolution = self.runner(["ssh", "-G", *remote.direct_ssh_options(), "--", remote.host],
                                         self.budget)
                if resolution.returncode:
                    raise ObservationFailure("ROUTE_UNAVAILABLE")
                value = repr((remote.host, remote.remote_user, sorted(remote.ssh_options.items()),
                              resolution.stdout))
                self.direct[key] = "direct-" + hashlib.sha256(value.encode()).hexdigest()[:24]
            socket_name = self.direct[key]
        path = Path("~/.sucoder/ssh").expanduser() / (socket_name + ".sock")
        # Legacy sockets do not encode accounts. Only reuse them when account is
        # unspecified; explicit accounts use independent nonpersistent SSH.
        lock_key = None
        direct_socket = remote.host and host == remote.gateway
        legacy_socket = (not remote.host and account_for(remote) is None and
                         not remote.ssh_options)
        if (direct_socket or legacy_socket) and path.is_socket():
            opts += ["-o", "ControlPath=" + str(path)]
            lock_key = str(path)
        else:
            opts += ["-o", "ControlPath=none"]
        if host == remote.gateway:
            opts += remote.direct_ssh_options()
        if host != remote.gateway:
            jump, _ = self.options(remote, remote.gateway)
            opts += ["-o", "ProxyCommand=" + shlex.join(["ssh", *jump, "-W", "%h:%p", "--", remote.gateway])]
        elif remote.cert_file:
            opts += ["-i", str(remote.cert_file)]
        return opts, lock_key

    def run(self, remote, host, script):
        from .tunnel import session_lock
        options, key = self.options(remote, host)
        lock = session_lock(key or (host, account_for(remote)))
        if not lock.acquire(timeout=self.budget.remaining()):
            raise ObservationFailure("TIMEOUT")
        try:
            self.budget.check()
            seconds = max(.01, self.budget.remaining())
            command = ("if command -v timeout >/dev/null 2>&1; then exec timeout -k 1 "
                       + str(round(seconds, 3)) + " sh -c " + shlex.quote(script)
                       + "; else exec sh -c " + shlex.quote(script) + "; fi")
            reply = self.runner(["ssh", *options, "--", host, command], self.budget)
            if reply.returncode:
                raise ObservationFailure("TIMEOUT" if reply.returncode == 124 else "PROBE_FAILED")
            return reply.stdout
        finally:
            lock.release()


# tmux's documented no-server diagnostics under C locale are a distinct empty
# observation. Missing binary, permission errors and other failures propagate.
LOGIN_PROBE = """out=$(LC_ALL=C tmux list-sessions -F '#{session_name}' 2>&1); rc=$?
if [ "$rc" -eq 0 ]; then printf '%s\n' "$out"; exit 0; fi
if [ "$rc" -eq 1 ]; then
  case "$out" in
    'no server running on '*|'error connecting to '*'(No such file or directory)') exit 0;;
  esac
fi
exit 1
"""


LOGIN_PANES_PROBE = "names=$(sh -c " + shlex.quote(LOGIN_PROBE) + ") || exit 1; " + (
    "printf '%s\\n' \"$names\" | while IFS= read -r name; do "
    "case \"$name\" in sucoder-*) "
    "printf '%s\\t' \"$name\"; bash -c " + shlex.quote(SESSION_PANE_PROBE_SH) +
    " _ \"$name\" ''; printf '\\n';; esac; done"
)


def read_records(config, selected, result):
    result.source("records", "records", selected)
    records = []
    directory = Path("~/.sucoder/sessions").expanduser()
    try:
        entries = os.scandir(directory)
    except FileNotFoundError:
        result.coverage("records", "records", "complete", 0)
        return records
    except OSError:
        result.coverage("records", "records", "unavailable", reason="RECORD_UNREADABLE")
        return records
    failed = False
    try:
        with entries:
            for entry in entries:
                result.budget.check()
                if not entry.name.endswith(".yaml"):
                    continue
                key = entry.name[:-5]
                candidates = [(name, target) for name in config.mirrors for target in config.targets
                              if key == name + "--" + target]
                if len(candidates) == 1 and candidates[0][1] not in selected:
                    continue
                if len(records) >= 500:
                    raise ObservationFailure("OUTPUT_LIMIT")
                item = dict(key=key, source_id=None, state="unknown", jobs=[])
                raw = {}
                try:
                    # A FIFO/device/symlink is not a session record. O_NONBLOCK
                    # closes the stat/open FIFO race without a blocking open.
                    fd = os.open(entry.path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
                    with os.fdopen(fd, "rb") as stream:
                        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                            raise OSError("not a regular record")
                        content = stream.read(65537)
                    if len(content) > 65536:
                        raise ObservationFailure("OUTPUT_LIMIT")
                    raw = yaml.safe_load(content)
                    if not isinstance(raw, dict):
                        raise ValueError()
                    main = raw.get("slurm_job_id")
                    if main is not None and (type(main) is not int or main <= 0):
                        raise ValueError()
                    replacement = raw.get("replacement")
                    if replacement is not None:
                        if not isinstance(replacement, dict) or replacement.get("state") not in ("drained", "submitted", "ready"):
                            raise ValueError()
                        required = {"submitted": "new_job", "ready": "old_job"}.get(replacement["state"])
                        for key in ("new_job", "old_job"):
                            value = replacement.get(key)
                            if (key == required or value is not None) and (type(value) is not int or value <= 0):
                                raise ValueError()
                    jobs = [main, RemoteSession.pending_replacement_job(raw)]
                    item["jobs"] = [dict(job_id=str(j), state="unknown")
                                    for j in dict.fromkeys(jobs) if j is not None]
                except (ValueError, yaml.YAMLError):
                    item["state"] = "corrupt"
                    result.error("RECORD_CORRUPT", "records")
                    failed = True
                    raw = {}
                except OSError:
                    item["state"] = "corrupt"
                    result.error("RECORD_UNREADABLE", "records")
                    failed = True
                    raw = {}
                if len(candidates) != 1 and item["state"] != "corrupt":
                    item["state"] = "ambiguous" if candidates else "unknown"
                    result.error("IDENTITY_AMBIGUOUS", "records")
                    failed = True
                records.append((item, candidates[0] if len(candidates) == 1 else None, raw))
    except ObservationFailure as exc:
        result.error(exc.code, "records")
        failed = True
    except OSError:
        result.error("RECORD_UNREADABLE", "records")
        failed = True
    result.coverage("records", "records", "partial" if failed else "complete", len(records))
    return records


def list_sessions(config, budget, *, target=None, fast=False, login_nodes=True, transport=None):
    result = Result("list_sessions", budget)
    selected = [target] if target else sorted(config.targets)
    data = dict(scope="configured_remote_targets", selected_targets=selected, sessions=[], records=[])
    result.value["data"] = data
    if target and target not in config.targets:
        result.value.update(status="error", data=None)
        result.error("INVALID_ARGUMENT")
        return result.finish()
    if not selected:
        return result.finish()
    transport = transport or Transport(budget)
    records = read_records(config, selected, result)
    groups = {}
    by_target = {}
    for name, remote in config.targets.items():
        if not remote.slurm:
            continue
        sid = source_id(result.authority, endpoint_for(remote, name), account_for(remote), name, "scheduler")
        groups.setdefault(sid, []).append(name)
        by_target[name] = sid
    complete = set()
    # Reconciliation uses ALL live rows of a source, before target narrowing.
    live = {}
    for sid, names in groups.items():
        if not set(names).intersection(selected):
            continue
        remote = config.targets[names[0]]
        result.source(sid, "scheduler", names, account_for(remote))
        try:
            output = transport.run(remote, remote.gateway,
                                   "squeue --me --noheader -o " + shlex.quote(SQUEUE_FORMAT))
            jobs, bad = parse_squeue(output)
            if bad:
                result.error("PROBE_FAILED", sid)
            else:
                complete.add(sid)
            live[sid] = {str(j.job_id) for j in jobs}
            result.coverage(sid, "scheduler", "partial" if bad else "complete", len(jobs))
            pane_scripts = []
            pane_rows = {}
            for job in jobs:
                budget.check()
                winner = match_target(job, config.targets, names)
                if winner and winner not in selected:
                    continue
                targets = [winner] if winner else list(names)
                mirrors = [m for m in config.mirrors if _sanitize_session_token(m) == job.token]
                if winner is None or len(mirrors) > 1:
                    result.error("IDENTITY_AMBIGUOUS", sid)
                row = dict(key=dict(source_id=sid, kind="slurm", job_id=str(job.job_id),
                                    socket=None, session_name=None), targets=targets,
                           mirror_candidates=mirrors, scheduler_state=job.state, node=job.node or None,
                           pane="not_probed" if fast else "unknown", record_keys=[])
                data["sessions"].append(row)
                if len(data["sessions"]) > 500:
                    raise ObservationFailure("OUTPUT_LIMIT")
                if not fast and winner and len(mirrors) == 1 and job.state == "RUNNING" and job.node:
                    name, dedicated_socket = confined_tmux_target(mirrors[0])
                    confined = config.targets[winner].slurm.confined
                    sock = dedicated_socket if confined else ""
                    row["key"].update(socket=sock, session_name=name)
                    script = "bash -c " + shlex.quote(SESSION_PANE_PROBE_SH) + " _ " + shlex.quote(name) + " " + shlex.quote(sock)
                    # Existing allocation route, also valid for unconfined
                    # Slurm jobs; never SSH to a scheduler nodelist.
                    script = "TMPDIR=/tmp srun --jobid=" + str(job.job_id) + " --overlap --quiet --chdir=/tmp " + script
                    pane_scripts.append("printf '%s\\t' " + shlex.quote(str(job.job_id)) + "; " + script + "; printf '\\n'")
                    pane_rows[str(job.job_id)] = row
            if pane_scripts:
                try:
                    panes = transport.run(remote, remote.gateway, "\n".join(pane_scripts))
                    seen = set()
                    for line in panes.splitlines():
                        number, sep, status = line.partition("\t")
                        if not line:
                            continue
                        if not sep or number not in pane_rows or number in seen:
                            result.error("PROBE_FAILED", sid)
                            continue
                        seen.add(number)
                        pane_rows[number]["pane"] = pane_class(status)
                except ObservationFailure as exc:
                    result.error(exc.code, sid)
            unknown = sum(r["pane"] == "unknown" for r in data["sessions"] if r["key"]["source_id"] == sid)
            result.coverage(sid, "panes", "skipped" if fast else ("partial" if unknown else "complete"),
                            None if fast else len(jobs) - unknown, "PROBE_FAILED" if unknown else None)
        except ObservationFailure as exc:
            result.coverage(sid, "scheduler", "timeout" if exc.code == "TIMEOUT" else "unavailable",
                            reason=exc.code)
            complete.discard(sid)
    # Login sources include configured gateway and recorded pins, separately
    # attributed. A failed tmux command is never rewritten into an empty list.
    logins = {}
    for name in config.targets:
        remote = config.targets[name]
        hosts = {remote.gateway}
        for _, candidate, raw in records:
            if candidate and candidate[1] == name and isinstance(raw.get("login_node"), str):
                hosts.add(raw["login_node"])
        for host in hosts:
            sid = source_id(result.authority, [endpoint_for(remote, name), host], account_for(remote), name, "login")
            if sid not in logins:
                logins[sid] = (remote, host, [])
            logins[sid][2].append(name)
    for sid, (remote, host, names) in logins.items():
        if not set(names).intersection(selected):
            continue
        result.source(sid, "login", names, account_for(remote))
        if fast or not login_nodes:
            result.coverage(sid, "login", "skipped")
            continue
        try:
            text = transport.run(remote, host, LOGIN_PANES_PROBE)
            count = 0
            for line in text.splitlines():
                name, sep, status = line.partition("\t")
                budget.check()
                if not name.startswith("sucoder-"):
                    continue
                if any(c.isspace() for c in name):
                    raise ObservationFailure("PROBE_FAILED")
                mirrors = [m for m in config.mirrors if "sucoder-" + _sanitize_session_token(m) == name]
                if len(names) > 1 or len(mirrors) > 1:
                    result.error("IDENTITY_AMBIGUOUS", sid)
                row = dict(key=dict(source_id=sid, kind="login", job_id=None, socket="", session_name=name),
                           targets=list(names), mirror_candidates=mirrors, scheduler_state=None,
                           node=host, pane="unknown", record_keys=[])
                data["sessions"].append(row)
                count += 1
                if len(data["sessions"]) > 500:
                    raise ObservationFailure("OUTPUT_LIMIT")
                row["pane"] = pane_class(status) if sep else "unknown"
                if row["pane"] == "unknown":
                    result.error("PROBE_FAILED", sid)
            result.coverage(sid, "login", "complete", count)
        except ObservationFailure as exc:
            result.coverage(sid, "login", "timeout" if exc.code == "TIMEOUT" else "unavailable", reason=exc.code)
    for record, candidate, _ in records:
        if candidate and record["state"] != "corrupt":
            mirror, target_name = candidate
            sid = by_target.get(target_name)
            record["source_id"] = sid
            for job in record["jobs"]:
                job["state"] = ("matched" if job["job_id"] in live.get(sid, set()) else
                                "absent" if sid in complete else "unknown")
                for row in data["sessions"]:
                    if row["key"]["source_id"] == sid and row["key"]["job_id"] == job["job_id"]:
                        row["record_keys"].append(record["key"])
            states = {j["state"] for j in record["jobs"]}
            record["state"] = next(iter(states)) if len(states) == 1 else "unknown"
            if record["state"] == "unknown":
                result.error("IDENTITY_AMBIGUOUS", "records")
        data["records"].append(record)
    return result.finish()
