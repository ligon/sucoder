"""Compute-side, headless Slurm dispatch with private durable artifacts.

Submission is a transaction with an explicit unknown outcome: a timed-out
sbatch can have accepted the job, so callers must reconcile its comment token
rather than retry. Source repositories are read offline and never written.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional

from . import __version__
from .codex_state import runner_source, scope_id
from .dispatch_protocol import ENVIRONMENT_ALLOWLIST, SCHEMA_VERSION, TERMINAL_STATES
from .local_tier import build_worker_prepare_script, build_worker_snapshot_script

DEFAULT_DIRECTORY_ROOT = Path.home() / ".sucoder" / "dispatch"
HEARTBEAT_STALE_SECONDS = 60


def _finite_seconds(value, label, *, allow_zero=False):
    """Reject invalid deadlines before an operation can contact Slurm."""
    try:
        numeric = float(value) if type(value) in (int, float) else float("nan")
    except OverflowError:
        numeric = float("inf")
    if not math.isfinite(numeric) or numeric < 0 or (numeric == 0 and not allow_zero):
        raise ValueError(label + " must be finite and " + ("nonnegative" if allow_zero else "positive"))
    return numeric


def _run(argv, *, timeout=30, env=None):
    return subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=timeout, env=env, check=False)


def _git(repo, *args, timeout=30, file_protocol=False):
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0",
               GIT_OPTIONAL_LOCKS="0", GIT_ALLOW_PROTOCOL="file" if file_protocol else "")
    # Git caller configuration must not redirect this read to the parent worktree.
    for name in tuple(env):
        if name.startswith("GIT_") and name not in (
                "GIT_NO_LAZY_FETCH", "GIT_TERMINAL_PROMPT", "GIT_OPTIONAL_LOCKS", "GIT_ALLOW_PROTOCOL"):
            env.pop(name)
    result = _run(["git", "-C", str(repo), *args], timeout=timeout, env=env)
    if result.returncode:
        raise ValueError("git source inspection failed: " + result.stderr.strip())
    return result.stdout.strip()


def _shared_path(path: Path, label: str) -> Path:
    """Reject known node-local paths, including symlinks into an old allocation.

    Lustre/NFS/GPFS paths pass. Mount information is best effort: an unknown
    filesystem is not proof of shared placement; absolute deployment paths
    still require the receiving node's availability check.
    """
    path = path.expanduser().resolve()
    if any(path == Path(root) or Path(root) in path.parents for root in ("/local", "/tmp", "/var/tmp", "/dev/shm")):
        raise ValueError(label + " must be on shared storage, not node-local: " + str(path))
    try:
        mounts = []
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            before, after = line.split(" - ", 1)
            mount = Path(before.split()[4].replace("\\040", " "))
            if mount == path or mount in path.parents:
                mounts.append((len(str(mount)), after.split()[0]))
        filesystem = max(mounts)[1] if mounts else None
        if filesystem in {"ext2", "ext3", "ext4", "xfs", "btrfs", "tmpfs", "ramfs", "overlay"}:
            raise ValueError(label + " is on a node-local filesystem: " + str(path))
    except OSError:
        pass
    return path


def _runtime_path(executable):
    """Supply only the selected shared toolchain and receiving system paths."""
    paths = [str(_shared_path(executable.parent, "agent runtime directory"))]
    with executable.open("rb") as stream:
        shebang = stream.readline(4096).decode("utf-8", errors="replace").strip()
    if re.match(r"^#!\s*/usr/bin/env\s+(?:-S\s+)?node(?:\s|$)", shebang):
        node = shutil.which("node")
        if not node:
            raise ValueError("agent uses env node but no shared Node executable was found")
        node_path = Path(node).resolve()
        if str(node_path.parent) not in ("/usr/local/bin", "/usr/bin", "/bin"):
            _shared_path(node_path, "Node executable")
        paths.append(str(node_path.parent))
    paths.extend(("/usr/local/bin", "/usr/bin", "/bin"))
    return list(dict.fromkeys(paths))


def _private_write(path: Path, content: str, executable=False):
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700 if executable else 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(content)


def _write_json(path: Path, data: Mapping[str, Any]):
    _private_write(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path):
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            return None, "unreadable"
        return value, "present"
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError):
        return None, "unreadable"


def _resources(values):
    required = {"account", "partition", "qos", "cpus", "mem", "time"}
    if not required.issubset(values) or set(values) - required - {"exclude"}:
        raise ValueError("resources require account, partition, qos, cpus, mem and time; optional exclude")
    values = dict(values)
    values.setdefault("exclude", None)
    if values["exclude"] is not None and (not isinstance(values["exclude"], str) or not re.fullmatch(r"[A-Za-z0-9_.,-]+", values["exclude"])):
        raise ValueError("invalid excluded node list")
    if isinstance(values["cpus"], bool) or not isinstance(values["cpus"], int) or values["cpus"] < 1:
        raise ValueError("cpus must be a positive integer")
    for name in ("account", "partition", "qos"):
        if name == "qos" and values[name] is None:
            continue
        if not isinstance(values[name], str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", values[name]):
            raise ValueError("invalid resource " + name)
    if not isinstance(values["mem"], str) or not re.fullmatch(r"[1-9][0-9]*[KMGT]?", values["mem"], re.I):
        raise ValueError("invalid memory request")
    if not isinstance(values["time"], str) or not re.fullmatch(r"(?:[0-9]+-)?[0-9]+(?::[0-9]{1,2}){0,2}", values["time"]):
        raise ValueError("invalid time request")
    days, clock = values["time"].split("-", 1) if "-" in values["time"] else (None, values["time"])
    fields = [int(part) for part in clock.split(":")]
    if days is not None:
        fields += [0] * (3 - len(fields))
        seconds = int(days) * 86400 + fields[0] * 3600 + fields[1] * 60 + fields[2]
    elif len(fields) == 1:
        seconds = fields[0] * 60
    elif len(fields) == 2:
        seconds = fields[0] * 60 + fields[1]
    else:
        seconds = fields[0] * 3600 + fields[1] * 60 + fields[2]
    if seconds < 120:
        raise ValueError("dispatch time must allow the 120-second shutdown signal window")
    return values


def submit(repo, ref, task_file, agent, executable, resources, *, model=None,
           directory_root=None, local_disk_root="/local", source_agent_home=None,
           source_claude_user_config=None, required_evidence=None, snapshot_minutes=10,
           dry_run=False, timeout=30):
    """Pin an offline commit, stage a private worker and submit exactly once.

    Dry runs validate and report the planned command without writing artifacts
    or contacting Slurm. They do not reserve their reported random dispatch ID.
    """
    timeout = _finite_seconds(timeout, "timeout")
    if type(snapshot_minutes) not in (int, float) or not math.isfinite(snapshot_minutes) or snapshot_minutes <= 0:
        raise ValueError("snapshot_minutes must be finite and positive")
    if agent not in ("codex", "claude"):
        raise ValueError("headless agent must be codex or claude")
    resources = _resources(resources)
    if required_evidence is not None and not isinstance(required_evidence, (list, tuple)):
        raise ValueError("required evidence must be a list of relative POSIX file paths")
    required_evidence = [] if required_evidence is None else list(required_evidence)
    if any(not isinstance(name, str) or not name or "\x00" in name or
           not PurePosixPath(name).parts or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or
           str(PurePosixPath(name)) != name or "\\" in name
           for name in required_evidence):
        raise ValueError("required evidence names must be relative POSIX file paths without traversal")
    if len(set(required_evidence)) != len(required_evidence):
        raise ValueError("required evidence names must be unique")
    if not Path(repo).is_absolute() or not Path(executable).is_absolute():
        raise ValueError("source repository and executable require absolute shared paths")
    repo = _shared_path(Path(repo), "source repository")
    root = _shared_path(Path(directory_root) if directory_root is not None else DEFAULT_DIRECTORY_ROOT, "result directory")
    executable = _shared_path(Path(executable), "agent executable")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("agent executable must be an executable file")
    local_disk_root = str(Path(local_disk_root))
    if not Path(local_disk_root).is_absolute() or local_disk_root in ("/", "/tmp", "/var/tmp"):
        raise ValueError("local disk root must be an explicit node-local base")
    task = Path(task_file).read_text()
    if not task.strip():
        raise ValueError("task file must contain a task")
    if not isinstance(ref, str) or not ref or ref.startswith("-"):
        raise ValueError("ref must name a committed Git object")
    config = _git(repo, "config", "--local", "--list").lower()
    gitdir = Path(_git(repo, "rev-parse", "--absolute-git-dir"))
    common = Path(_git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    if ("extensions.partialclone=" in config or
            re.search(r"^remote\..*\.promisor=(?:true|yes|on|1)$", config, re.M) or
            list((common / "objects" / "pack").glob("*.promisor"))):
        raise ValueError("partial/promisor source repositories are unsupported; no hydration attempted")
    # Git metadata can itself be redirected into node-local storage by a worktree.
    _shared_path(gitdir, "source Git metadata")
    _shared_path(common, "source common Git metadata")
    commit = _git(repo, "rev-parse", "--verify", ref + "^{commit}")
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ValueError("source did not resolve to a commit")
    dispatch_id = uuid.uuid4().hex
    directory = root / dispatch_id
    cluster = os.environ.get("SLURM_CLUSTER_NAME") or None
    manifest = dict(schema_version=SCHEMA_VERSION, dispatcher_version=__version__, adapter_version=1,
                    requested_model=model, required_evidence=required_evidence, created_at=time.time(), dispatch_id=dispatch_id, uid=os.getuid(),
                    directory=str(directory), cluster=cluster, repo=str(repo), commit=commit,
                    branch="dispatch/" + dispatch_id, agent=agent, executable=str(executable),
                    model=model, runtime_path=_runtime_path(executable), local_disk_root=local_disk_root,
                    task_file=str(directory / "task.txt"), prepare_script=str(directory / "prepare.sh"),
                    snapshot_script=str(directory / "snapshot.sh"), runner_path=str(directory / "runner.py"),
                    worker_path=str(directory / "worker.py"), environment_file=str(directory / "environment.json"),
                    source_agent_home=None, source_claude_user_config=None, resources=resources,
                    scope=scope_id(cluster or "unknown", str(os.getuid()), str(repo), dispatch_id),
                    checkpoint_minutes=10, checkpoint_timeout=30, snapshot_minutes=float(snapshot_minutes))
    source_home = source_agent_home if source_agent_home is not None else Path.home() / (".codex" if agent == "codex" else ".claude")
    source_home_path = _shared_path(Path(source_home), "source agent home")
    if source_agent_home is not None and not source_home_path.is_dir():
        raise ValueError("source agent home must be an existing directory")
    if source_home_path.exists():
        manifest["source_agent_home"] = str(source_home_path)
    claude_config = Path(source_claude_user_config) if source_claude_user_config is not None else Path.home() / ".claude.json"
    if source_claude_user_config is not None and not claude_config.is_file():
        raise ValueError("Claude user config must be an existing file")
    if agent == "claude" and claude_config.exists():
        manifest["source_claude_user_config"] = str(_shared_path(claude_config, "Claude user config"))
    argv = ["sbatch", "--parsable", "--no-requeue", "--signal=B:TERM@120", "--export=NONE", "--chdir=/tmp", "--nodes=1", "--ntasks=1",
            "--account=" + resources["account"], "--partition=" + resources["partition"],
            "--cpus-per-task=" + str(resources["cpus"]),
            "--mem=" + resources["mem"], "--time=" + resources["time"],
            "--comment=sucoder-dispatch:" + dispatch_id, "--job-name=sucoder-" + dispatch_id[:12],
            "--output=" + str(directory / "slurm.log"), "--error=" + str(directory / "slurm.stderr.log"),
            str(directory / "batch.sh")]
    if resources["qos"] is not None:
        argv.insert(3, "--qos=" + resources["qos"])
    if resources["exclude"] is not None:
        argv.insert(3, "--exclude=" + resources["exclude"])
    report = dict(schema_version=SCHEMA_VERSION, dispatch_id=dispatch_id, uid=os.getuid(), cluster=cluster,
                  directory=str(directory), commit=commit, command=argv, state="dry_run" if dry_run else "preparing")
    if dry_run:
        return report
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.stat().st_uid != os.getuid():
        raise ValueError("result root must be owned by the submitting uid")
    directory.mkdir(mode=0o700)
    (directory / "evidence").mkdir(mode=0o700)
    # Private self-contained source snapshot. No source ref/config/index changes.
    _git(repo, "clone", "--bare", "--no-local", "--no-hardlinks", str(repo), str(directory / "results.git"), timeout=timeout, file_protocol=True)
    _git(directory / "results.git", "cat-file", "-e", commit + "^{commit}")
    # Make the private initial reference explicit; this never touches source.
    _git(directory / "results.git", "update-ref", "refs/heads/" + manifest["branch"], commit)
    _write_json(directory / "manifest.json", manifest)
    _private_write(directory / "task.txt", task)
    # Mutable coordinator channels are deliberately separate from frozen task
    # and helper digests: workers on another node must see later steering.
    _private_write(directory / "MESSAGES_TO_AGENT.txt", "Coordinator steering messages appear here.\n")
    _private_write(directory / "QUESTIONS_FROM_AGENT.txt", "Record questions here; continue independent work.\n")
    _write_json(directory / "environment.json", {key: os.environ[key] for key in ENVIRONMENT_ALLOWLIST if key in os.environ})
    _private_write(directory / "prepare.sh", build_worker_prepare_script(results_path=str(directory / "results.git"), dispatch_id=dispatch_id, commit=commit, local_disk_root=local_disk_root), True)
    _private_write(directory / "snapshot.sh", build_worker_snapshot_script(results_path=str(directory / "results.git"), dispatch_id=dispatch_id, local_disk_root=local_disk_root), True)
    _private_write(directory / "runner.py", runner_source())
    _private_write(directory / "worker.py", Path(__file__).with_name("dispatch_worker.py").read_text())
    _private_write(directory / "dispatch_protocol.py", Path(__file__).with_name("dispatch_protocol.py").read_text())
    _private_write(directory / "dispatch_artifacts.py", Path(__file__).with_name("dispatch_artifacts.py").read_text())
    _private_write(directory / "batch.sh", "#!/bin/bash\nset -eu\numask 077\nexec python3 " + shlex.quote(str(directory / "worker.py")) + " " + shlex.quote(str(directory / "manifest.json")) + "\n", True)
    _write_json(directory / "digests.json", {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in ("task.txt", "prepare.sh", "snapshot.sh", "runner.py", "worker.py", "dispatch_protocol.py", "dispatch_artifacts.py", "environment.json", "manifest.json")})
    # Persist the token before contacting Slurm. A crash here is conservatively unknown.
    report["state"] = "submission_unknown"
    report["submission_started_at"] = time.time()
    _write_json(directory / "submission.json", report)
    try:
        response = _run(argv, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as exc:
        # OSError before execution is known rejection; timeout is ambiguous.
        report["state"] = "submission_unknown" if isinstance(exc, subprocess.TimeoutExpired) else "submission_failed"
        report["reason"] = "sbatch timed out; reconcile the dispatch comment before retrying" if isinstance(exc, subprocess.TimeoutExpired) else "sbatch unavailable"
    else:
        report["receipt_observed_at"] = time.time()
        answer = response.stdout.strip()
        match = re.fullmatch(r"([1-9][0-9]*)(?:;([A-Za-z0-9_.-]+))?", answer)
        if match:
            receipt_cluster = match.group(2) or cluster
            source = dict(schema_version=SCHEMA_VERSION, dispatch_id=dispatch_id, uid=os.getuid(),
                          job_id=int(match.group(1)), cluster=receipt_cluster)
            _write_json(directory / "scheduler-source.json", source)
            report.update(job_id=source["job_id"], cluster=receipt_cluster)
            if cluster is not None and receipt_cluster != cluster:
                report.update(state="submission_unknown", reason="sbatch receipt cluster differs from submitting cluster; reconcile before retrying")
            elif response.returncode:
                report.update(state="submission_unknown", reason="sbatch returned a receipt with nonzero exit; reconcile before retrying", returncode=response.returncode)
            else:
                report["state"] = "submitted"
        else:
            report.update(state="submission_unknown", reason="sbatch returned no trustworthy receipt; reconcile comment before retrying", returncode=response.returncode)
    # Atomic replacement preserves the original pre-submission receipt on failure.
    temp = directory / "submission.updated.json"
    _write_json(temp, report)
    temp.replace(directory / "submission.json")
    return report


def _directory(value):
    requested = Path(value).expanduser()
    if requested.is_symlink():
        raise ValueError("dispatch directory must not redirect to another record")
    path = requested.resolve()
    if not path.is_dir() or path.stat().st_uid != os.getuid():
        raise ValueError("dispatch directory is absent or belongs to another uid")
    manifest, state = _read_json(path / "manifest.json")
    if (state != "present" or manifest.get("schema_version") != SCHEMA_VERSION or
            manifest.get("uid") != os.getuid() or
            manifest.get("directory") != str(path) or
            manifest.get("dispatch_id") != path.name or
            not re.fullmatch(r"[0-9a-f]{32}", str(manifest.get("dispatch_id")))):
        raise ValueError("invalid or unreadable dispatch manifest")
    return path, manifest


def _receipt(path, manifest):
    receipt, state = _read_json(path / "submission.json")
    if receipt is None:
        return None, state
    if (receipt.get("schema_version") != SCHEMA_VERSION or
            receipt.get("dispatch_id") != manifest["dispatch_id"] or
            receipt.get("uid") != manifest["uid"] or
            receipt.get("directory") != manifest["directory"]):
        return None, "unreadable"
    cluster = receipt.get("cluster")
    if cluster is not None and (not isinstance(cluster, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", cluster)):
        return None, "unreadable"
    if manifest.get("cluster") is not None and cluster != manifest["cluster"]:
        return None, "unreadable"
    job = receipt.get("job_id")
    if job is not None:
        source, source_state = _read_json(path / "scheduler-source.json")
        if (type(job) is not int or job <= 0 or source_state != "present" or
                source.get("schema_version") != SCHEMA_VERSION or
                any(source.get(key) != receipt.get(key) for key in ("dispatch_id", "uid", "job_id", "cluster"))):
            return None, "unreadable"
    return receipt, state


def _matches_worker(value, manifest, receipt):
    if (value is None or receipt is None or not receipt.get("cluster") or
            type(receipt.get("job_id")) is not int or receipt["job_id"] <= 0):
        return False
    return (value.get("schema_version") == SCHEMA_VERSION and
            value.get("dispatch_id") == manifest["dispatch_id"] and
            value.get("uid") == manifest["uid"] and
            str(value.get("job_id")) == str(receipt.get("job_id")) and
            value.get("cluster") == receipt["cluster"] and
            all(value.get(key) == manifest[key] for key in ("commit", "branch", "agent")))


def read_result(directory):
    """Return an explicit present/missing/unreadable result, never guess success."""
    path, manifest = _directory(directory)
    receipt, receipt_state = _receipt(path, manifest)
    result, state = _read_json(path / "result.json")
    if result is not None and not _matches_worker(result, manifest, receipt):
        result, state = None, "unreadable"
    return {"schema_version": SCHEMA_VERSION, "dispatch_id": manifest["dispatch_id"],
            "result_state": state, "result": result}


def _scheduler(receipt, timeout):
    timeout = _finite_seconds(timeout, "scheduler timeout")
    if receipt is None or not isinstance(receipt.get("job_id"), int):
        return {"available": False, "state": "not_submitted"}
    job = str(receipt["job_id"])
    if not receipt.get("cluster"):
        return {"available": False, "state": "unknown_source", "reason": "receipt names no cluster; no current-cluster job lookup attempted"}
    cluster_args = ["--clusters=" + receipt["cluster"]]
    deadline = time.monotonic() + timeout
    try:
        query = _run(["squeue", *cluster_args, "--jobs=" + job, "--noheader", "--format=%i|%U|%T|%N"], timeout=max(0.001, deadline - time.monotonic()))
        if query.returncode:
            return {"available": False, "state": "unavailable"}
        for line in query.stdout.splitlines():
            fields = line.strip().split("|")
            if len(fields) == 4 and fields[0] == job and fields[1] == str(receipt["uid"]):
                return {"available": True, "state": fields[2], "node": fields[3], "source": "squeue"}
        if time.monotonic() >= deadline:
            return {"available": False, "state": "unavailable", "reason": "scheduler deadline exhausted"}
        query = _run(["sacct", *cluster_args, "--jobs=" + job, "--noheader", "--parsable2", "--format=JobIDRaw,UID,State,ExitCode"], timeout=max(0.001, deadline - time.monotonic()))
        if query.returncode:
            return {"available": False, "state": "unavailable"}
        for line in query.stdout.splitlines():
            fields = line.strip().split("|")
            if len(fields) >= 4 and fields[0] == job and fields[1] == str(receipt["uid"]):
                return {"available": True, "state": fields[2], "exit_code": fields[3], "source": "sacct"}
        return {"available": True, "state": "unknown", "reason": "job absent from queue and accounting; no result inferred"}
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False, "state": "unavailable"}


def status(directory, *, timeout=10):
    """Combine distinct scheduler, placement and process-result observations."""
    timeout = _finite_seconds(timeout, "timeout")
    path, manifest = _directory(directory)
    receipt, receipt_state = _receipt(path, manifest)
    worker, worker_state = _read_json(path / "status.json")
    result = read_result(path)
    scheduler = _scheduler(receipt, timeout)
    state = receipt.get("state", "unknown") if receipt else "unknown"
    if worker and not _matches_worker(worker, manifest, receipt):
        worker, worker_state = None, "unreadable"
    if worker:
        state = worker.get("state", "unknown")
    if result["result"]:
        state = result["result"].get("state", "unknown")
    observation_at = time.time()
    heartbeat_at = worker.get("heartbeat_at") if worker else None
    heartbeat_age = (max(0.0, observation_at - heartbeat_at)
                     if type(heartbeat_at) in (int, float) and math.isfinite(heartbeat_at) else None)
    return dict(directory=str(path), state=state, observation_at=observation_at,
                heartbeat_age_seconds=heartbeat_age,
                stale=heartbeat_age > HEARTBEAT_STALE_SECONDS if heartbeat_age is not None else None,
                heartbeat_stale_after_seconds=HEARTBEAT_STALE_SECONDS,
                submission=receipt, submission_state=receipt_state, worker=worker,
                worker_state=worker_state, scheduler=scheduler, **result)


def wait(directory, *, timeout=300, poll_seconds=2):
    """Bounded observation; timeout is not a worker exit or an implicit retry."""
    timeout = _finite_seconds(timeout, "wait timeout", allow_zero=True)
    poll_seconds = _finite_seconds(poll_seconds, "poll interval")
    deadline = time.monotonic() + timeout
    while True:
        remaining = max(0, deadline - time.monotonic())
        report = status(directory, timeout=max(0.001, min(10, remaining)))
        if report["result_state"] == "present" and report["state"] in TERMINAL_STATES:
            report["wait_timed_out"] = False
            return report
        if time.monotonic() >= deadline:
            report["wait_timed_out"] = True
            return report
        time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
