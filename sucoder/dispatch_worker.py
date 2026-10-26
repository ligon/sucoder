"""Receiving-node headless dispatch supervisor (Python 3.6, stdlib only).

Staged with the existing managed Codex runner. No package installation, TTY,
coordinator runtime, or shared working checkout is needed on the worker node.
"""
import argparse
import errno
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid

try:
    from .dispatch_protocol import ENVIRONMENT_ALLOWLIST, SCHEMA_VERSION, TERMINAL_STATES, atomic_json
except ImportError:  # standalone staged receiving-node script
    from dispatch_protocol import ENVIRONMENT_ALLOWLIST, SCHEMA_VERSION, TERMINAL_STATES, atomic_json


EXIT_REFUSED = 70
EXIT_PLACEMENT = 71
EXIT_PRESERVATION = 74
LOCAL_FILESYSTEMS = frozenset(("ext4", "ext3", "ext2", "xfs", "btrfs", "tmpfs"))
CREDENTIAL_KEYS = frozenset(ENVIRONMENT_ALLOWLIST)


class Refused(Exception):
    pass


class JobOwnership:
    """A receiving allocation is the ownership boundary, including new steps.

    A descendant can change session or be reparented; Slurm cgroup membership
    survives both. Only matching uid, job prefix and PID incarnation authorize
    signals. The supervisor and kernel-owned Slurm processes are excluded.
    """
    def __init__(self, job_id, proc_root="/proc", uid=None, supervisor_pid=None):
        self.proc_root = proc_root
        self.uid = os.getuid() if uid is None else uid
        self.supervisor_pid = os.getpid() if supervisor_pid is None else supervisor_pid
        self.job_id = str(job_id)
        self.prefixes = {}
        for hierarchy, controllers, path in self._cgroups("self"):
            parts = path.split("/")
            requested = "job_" + self.job_id
            jobs = [part for part in parts if re.fullmatch(r"job_[0-9]+", part)]
            if jobs and jobs != [requested]:
                raise Refused("receiving SLURM_JOB_ID differs from the supervisor's actual cgroup")
            if requested in parts:
                prefix = "/".join(parts[:parts.index(requested) + 1])
                self.prefixes[(hierarchy, controllers)] = prefix
        if not self.prefixes:
            raise Refused("unable to prove receiving allocation ownership from /proc/self/cgroup")

    def _cgroups(self, pid):
        with open(os.path.join(self.proc_root, str(pid), "cgroup")) as stream:
            rows = []
            for line in stream:
                fields = line.rstrip("\n").split(":", 2)
                if len(fields) != 3 or not fields[2].startswith("/"):
                    raise Refused("unreadable process cgroup membership")
                rows.append(tuple(fields))
            if not rows:
                raise Refused("process cgroup membership is empty")
            return rows

    def _record(self, pid):
        path = os.path.join(self.proc_root, str(pid))
        try:
            if pid == self.supervisor_pid:
                return None
            # proc directory ownership changes to root for nondumpable
            # processes, even when their real uid remains this worker's uid.
            # Read the kernel's actual credentials instead of skipping such
            # daemons as if they belonged to a different account.
            with open(os.path.join(path, "status")) as stream:
                status_text = stream.read()
            credentials = re.search(r"^Uid:\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s*$", status_text, re.M)
            if credentials is None:
                raise Refused("cannot inspect process uid credentials")
            if int(credentials.group(1)) != self.uid:
                return None
            with open(os.path.join(path, "stat")) as stream:
                fields = stream.read().rsplit(")", 1)[1].split()
            if fields[0] in ("Z", "X"):
                return None
            started = int(fields[19])
            groups = self._cgroups(pid)
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ESRCH) and not os.path.exists(path):
                return None
            raise Refused("cannot inspect same-uid process allocation membership")
        except (IndexError, ValueError):
            raise Refused("cannot inspect process incarnation")
        matching = [(hierarchy, controllers, path) for hierarchy, controllers, path in groups
                    if (hierarchy, controllers) in self.prefixes and
                    (path == self.prefixes[(hierarchy, controllers)] or
                     path.startswith(self.prefixes[(hierarchy, controllers)] + "/"))]
        if not matching:
            return None
        if len(matching) != len(self.prefixes):
            raise Refused("process has inconsistent receiving-job cgroup membership")
        return {"pid": pid, "starttime": started, "cgroups": matching}

    def members(self):
        return [record for name in os.listdir(self.proc_root) if name.isdigit()
                for record in (self._record(int(name)),) if record is not None]

    def still_owned(self, record):
        current = self._record(record["pid"])
        return current is not None and current["starttime"] == record["starttime"]

    def describe(self):
        return {"job_id": self.job_id, "prefixes": sorted(set(self.prefixes.values()))}


def stop_allocation(ownership, timeout):
    """Stop only proven members, including daemons outside the leader group."""
    deadline = time.monotonic() + timeout
    signalled = set()
    while True:
        members = ownership.members()
        if not members:
            return
        for record in members:
            identity = (record["pid"], record["starttime"])
            if identity not in signalled and ownership.still_owned(record):
                try:
                    os.kill(record["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
                signalled.add(identity)
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    for record in ownership.members():
        if ownership.still_owned(record):
            try:
                os.kill(record["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
    tail = time.monotonic() + 1
    while time.monotonic() < tail:
        if not ownership.members():
            return
        time.sleep(0.05)
    if ownership.members():
        raise Refused("owned allocation processes remain after bounded stop; state cannot be copied")


def contained(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root.rstrip("/") + "/")


def beneath_text(path, root):
    """Compare already absolute proc link text without touching its target."""
    path, root = os.path.normpath(path), os.path.normpath(root)
    return path == root or path.startswith(root.rstrip("/") + "/")


def private_file(path):
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise Refused("control file must be a private regular file owned by this uid: " + path)


def read_json(path):
    with open(path) as stream:
        return json.load(stream)


def load_manifest(path):
    private_file(path)
    manifest = read_json(path)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("uid") != os.getuid():
        raise Refused("manifest schema or owning uid mismatch")
    if not re.fullmatch(r"[0-9a-f]{32}", manifest.get("dispatch_id", "")):
        raise Refused("invalid dispatch id")
    if manifest.get("agent") not in ("codex", "claude"):
        raise Refused("unsupported headless agent")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", manifest.get("commit", "")):
        raise Refused("source commit must be an immutable full hash")
    if manifest.get("branch") != "dispatch/" + manifest["dispatch_id"]:
        raise Refused("manifest branch does not belong to this dispatch")
    directory = manifest.get("directory", "")
    if not os.path.isabs(directory) or os.path.realpath(directory) != os.path.dirname(os.path.realpath(path)):
        raise Refused("manifest directory mismatch")
    info = os.lstat(directory)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise Refused("dispatch directory is not private and owned by this uid")
    for key in ("task_file", "prepare_script", "snapshot_script", "runner_path", "worker_path", "environment_file"):
        value = manifest.get(key)
        if not isinstance(value, str) or not contained(value, directory):
            raise Refused("control path escapes dispatch directory: " + key)
        private_file(value)
    executable = manifest.get("executable", "")
    if not os.path.isabs(executable) or not os.access(executable, os.X_OK):
        raise Refused("agent executable must be an accessible absolute path")
    if not re.fullmatch(r"[0-9a-f]{8,64}", manifest.get("scope", "")):
        raise Refused("invalid managed state scope")
    for key in ("checkpoint_minutes", "checkpoint_timeout"):
        if type(manifest.get(key)) is not int or manifest[key] <= 0:
            raise Refused("invalid " + key)
    cadence = manifest.get("snapshot_minutes", manifest["checkpoint_minutes"])
    if type(cadence) not in (int, float) or not math.isfinite(cadence) or cadence <= 0:
        raise Refused("invalid workspace snapshot cadence")
    digest_path = os.path.join(directory, "digests.json")
    private_file(digest_path)
    digests = read_json(digest_path)
    required = {os.path.basename(manifest[key]) for key in
                ("task_file", "prepare_script", "snapshot_script", "runner_path", "worker_path", "environment_file")}
    required.update((os.path.basename(path), "dispatch_protocol.py", "dispatch_artifacts.py"))
    if not isinstance(digests, dict) or not required.issubset(digests):
        raise Refused("dispatch integrity record is incomplete")
    for name, expected in digests.items():
        if not isinstance(name, str) or os.path.basename(name) != name or name in (".", ".."):
            raise Refused("invalid dispatch integrity path")
        staged = os.path.join(directory, name)
        private_file(staged)
        with open(staged, "rb") as stream:
            actual = hashlib.sha256(stream.read()).hexdigest()
        if actual != expected:
            raise Refused("staged dispatch file differs from its integrity record: " + name)
    return manifest


def mount_for(path, mountinfo_path="/proc/self/mountinfo"):
    """Resolve the actual longest mount, including mountinfo octal escapes."""
    path = os.path.realpath(path)
    found = None
    with open(mountinfo_path) as stream:
        for line in stream:
            fields = line.split()
            if "-" not in fields or len(fields) < 7:
                continue
            point = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), fields[4])
            if path == point or path.startswith(point.rstrip("/") + "/"):
                if found is None or len(point) > len(found["mount"]):
                    found = {"mount": point, "filesystem": fields[fields.index("-") + 1]}
    if found is None:
        raise Refused("unable to identify local filesystem mount")
    return found


def receiving_root(manifest, environment, mountinfo_path="/proc/self/mountinfo"):
    job_id = environment.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[1-9][0-9]*", job_id):
        raise Refused("dispatch requires a numeric receiving SLURM_JOB_ID")
    base = manifest.get("local_disk_root", "")
    if not os.path.isabs(base) or os.path.normpath(base) != base or os.path.realpath(base) != base:
        raise Refused("local_disk_root must be an absolute, non-symlink disk path")
    root = os.path.join(base, "job" + job_id)
    if os.path.islink(root) or os.path.realpath(root) != root:
        raise Refused("receiving allocation root must not be a symlink")
    mount = mount_for(root, mountinfo_path)
    if mount["filesystem"] not in LOCAL_FILESYSTEMS:
        raise Refused("receiving allocation root is not on a supported local filesystem: " + mount["filesystem"])
    os.makedirs(root, mode=0o700, exist_ok=True)
    if os.stat(root).st_uid != os.getuid() or not os.path.isdir(root):
        raise Refused("receiving allocation root is not owned by this uid")
    return root, job_id, mount


def environment_for(manifest, inherited, root):
    """Keep scheduler allocation identity; discard coordinator control state."""
    system_keys = {"PATH", "LANG", "LANGUAGE", "TZ", "USER", "LOGNAME", "SHELL",
                   "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                   "SSL_CERT_FILE", "SSL_CERT_DIR"}
    environment = {key: value for key, value in inherited.items()
                   if key in system_keys or key.startswith(("SLURM_", "SPANK_", "LC_"))}
    # A preceding allocation's toolchain can be present in an inherited PATH.
    paths = manifest.get("runtime_path", [])
    if not isinstance(paths, list) or not paths:
        raise Refused("dispatch has no explicit receiving-node runtime PATH")
    if any(not isinstance(part, str) or not os.path.isabs(part)
           or re.search(r"(?:^|/)job[0-9]+(?:/|$)", part) for part in paths):
        raise Refused("runtime PATH must contain only absolute, non-allocation tool directories")
    environment["PATH"] = os.pathsep.join(paths)
    credentials = read_json(manifest["environment_file"])
    if not isinstance(credentials, dict) or set(credentials) - CREDENTIAL_KEYS:
        raise Refused("credential file contains an unsupported environment key")
    if any(not isinstance(value, str) or "\x00" in value for value in credentials.values()):
        raise Refused("credential file values must be strings without NUL")
    environment.update(credentials)
    # A dispatch is single-use. Once loaded, this submitted credential copy
    # need not remain on shared storage; source credentials are never edited.
    os.unlink(manifest["environment_file"])
    home = os.path.join(root, "state", "dispatch", manifest["dispatch_id"], "home")
    if os.path.realpath(home) != home or not contained(home, root):
        raise Refused("worker HOME must not traverse a symlink outside its private local path")
    environment.update({"HOME": home, "SUCODER_LOCAL_ROOT": root,
                        "SUCODER_DISPATCH_ID": manifest["dispatch_id"], "TERM": "dumb",
                        "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1",
                        "GIT_CONFIG_GLOBAL": os.devnull, "GIT_NO_LAZY_FETCH": "1"})
    cpus = inherited.get("SLURM_CPUS_PER_TASK")
    if not isinstance(cpus, str) or not re.fullmatch(r"[1-9][0-9]*", cpus):
        cpus = str(manifest.get("resources", {}).get("cpus", 1))
    if not re.fullmatch(r"[1-9][0-9]*", cpus):
        raise Refused("invalid receiving CPU budget")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[key] = cpus
    evidence_directory = os.path.join(manifest["directory"], "evidence")
    if os.path.realpath(evidence_directory) != evidence_directory:
        raise Refused("dispatch evidence directory must not be a symlink")
    os.makedirs(evidence_directory, mode=0o700, exist_ok=True)
    environment["SUCODER_DISPATCH_EVIDENCE_DIR"] = evidence_directory
    environment["SUCODER_DISPATCH_TASK_RESULT"] = os.path.join(manifest["directory"], "task-result.json")
    for key, suffix in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"),
                        ("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"),
                        ("XDG_STATE_HOME", "state"), ("UV_CACHE_DIR", "cache/uv"),
                        ("PIP_CACHE_DIR", "cache/pip"), ("npm_config_cache", "cache/npm")):
        path = os.path.join(home, suffix)
        os.makedirs(path, mode=0o700, exist_ok=True)
        environment[key] = path
    return environment


def copy_private(source, target):
    """Copy a seed without following a source symlink into unrelated state."""
    if source and os.path.exists(source):
        info = os.lstat(source)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise Refused("agent seed must be an owned regular file")
        with open(source, "rb") as incoming, open(target, "xb") as outgoing:
            os.chmod(target, 0o600)
            shutil.copyfileobj(incoming, outgoing)


def seed_home(manifest, environment):
    home = environment["HOME"]
    source = manifest.get("source_agent_home")
    if source is not None and (not os.path.isabs(source) or not os.path.isdir(source)):
        raise Refused("source agent home is not an absolute directory")
    if manifest["agent"] == "codex":
        # The managed runner records canonical CODEX_HOME in its lineage.
        # Keep that private identity stable across allocations; the runner
        # bind-mounts every hot directory/file onto its node-local copies.
        # Source credentials/configuration are cold-copied, never rebound.
        config = os.path.join(manifest["directory"], "codex-home")
        if os.path.lexists(config):
            raise Refused("private canonical Codex home already exists; dispatch is single-use")
        os.mkdir(config, mode=0o700)
        runner = load_runner(manifest["runner_path"])
        for name in runner.BIND_DIRS:
            os.mkdir(os.path.join(config, name), mode=0o700)
        with open(os.path.join(config, "history.jsonl"), "x"):
            pass
        # Codex 0.164 locks this file during every initialization. Keep a
        # stable private seed, but let prepare_binds place the live lockfile
        # on node-local disk. Never reuse the coordinator's native identity.
        fd = os.open(os.path.join(config, "installation_id"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(str(uuid.uuid4()))
        for name in ("auth.json", "config.toml", "AGENTS.md", "AGENTS.override.md"):
            copy_private(os.path.join(source, name) if source else None, os.path.join(config, name))
        environment["CODEX_HOME"] = config
    else:
        config = os.path.join(home, ".claude")
        os.makedirs(config, mode=0o700, exist_ok=True)
        for name in (".credentials.json", "settings.json", "CLAUDE.md"):
            copy_private(os.path.join(source, name) if source else None, os.path.join(config, name))
        # With CLAUDE_CONFIG_DIR set, native Claude also relocates its global
        # .claude.json into that directory (vendor installation docs).
        copy_private(manifest.get("source_claude_user_config"), os.path.join(config, ".claude.json"))
        environment["CLAUDE_CONFIG_DIR"] = config
    return config


def load_runner(path):
    spec = importlib.util.spec_from_file_location("sucoder_dispatch_managed_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_artifacts(directory):
    path = os.path.join(directory, "dispatch_artifacts.py")
    module_name = (__package__ + ".dispatch_artifacts") if __package__ else "sucoder_dispatch_artifacts"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_version(manifest, environment):
    """Claude's read-only version command is safe under its private HOME.

    Codex's version comes from managed generation metadata; never launch a
    raw Codex command against the shared canonical home to discover it.
    """
    if manifest["agent"] != "claude":
        return {"state": "pending", "source": "managed_generation"}
    path = os.path.join(manifest["directory"], "native-version.log")
    with open(path, "wb") as output:
        process = subprocess.Popen([manifest["executable"], "--version"], env=environment,
                                   stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            stop_owned(process, 1)
            return {"state": "unavailable", "reason": "native version command timed out", "log": path}
    with open(path, "r", errors="replace") as stream:
        text = stream.read(256).strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9A-Za-z_-]+){1,3})(?:\s+\(Claude Code\))?", text)
    if code != 0 or match is None:
        return {"state": "unavailable", "reason": "native version command did not return a recognizable version", "log": path}
    return {"state": "observed", "version": match.group(1), "source": "native_version_command"}


def command_for(manifest, environment):
    executable, directory = manifest["executable"], manifest["directory"]
    if manifest["agent"] == "codex":
        runner = load_runner(manifest["runner_path"])
        command = [executable, "-c", 'sqlite_home="%s"' % runner.PLACEHOLDER_DEFAULT,
                   "exec", "--dangerously-bypass-approvals-and-sandbox",
                   "--output-last-message", os.path.join(directory, "final.txt")]
        if manifest.get("model"):
            command += ["--model", manifest["model"]]
        return [sys.executable, manifest["runner_path"], "--scope", manifest["scope"],
                "--codex-home", environment["CODEX_HOME"], "--durable-root", os.path.join(directory, "state"),
                "--checkpoint-minutes", str(manifest["checkpoint_minutes"]),
                "--checkpoint-timeout", str(manifest["checkpoint_timeout"]),
                "--fresh", "--unattended", "--"] + command + ["-"]
    command = [executable, "--print", "--dangerously-skip-permissions", "--output-format", "text"]
    if manifest.get("model"):
        command += ["--model", manifest["model"]]
    return command


def run_script(path, environment, timeout, cancellation=None, heartbeat=None):
    process = subprocess.Popen(["/bin/bash", path], env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    deadline = time.monotonic() + timeout
    def diagnostics(stdout, stderr):
        logs = []
        for suffix, data in (("stdout", stdout), ("stderr", stderr)):
            fd, log_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix="." + suffix + ".log",
                                            dir=os.path.dirname(path))
            with os.fdopen(fd, "wb") as stream:
                stream.write(data or b"")
            logs.append(os.path.basename(log_path))
        return ", ".join(logs)
    def stop_and_drain():
        cleanup_error = None
        try:
            stop_owned(process, max(0, min(1, deadline - time.monotonic())))
        except Exception as exc:
            cleanup_error = type(exc).__name__
        try:
            return process.communicate(timeout=0.25), cleanup_error
        except subprocess.TimeoutExpired as exc:
            # An escaped helper descendant may still own a pipe after the
            # leader/group stops. Never wait indefinitely for that writer.
            stdout, stderr = exc.output or b"", exc.stderr or b""
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            return (stdout, stderr), cleanup_error or "escaped output writer"
    while True:
        if heartbeat is not None:
            heartbeat()
        if cancellation is not None and cancellation[0] is not None:
            (stdout, stderr), cleanup_error = stop_and_drain()
            raise Refused("dispatch workspace helper interrupted; cleanup=%s; private logs: %s" %
                          (cleanup_error or "complete", diagnostics(stdout, stderr)))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            (stdout, stderr), cleanup_error = stop_and_drain()
            raise Refused("dispatch workspace helper timed out; cleanup=%s; private logs: %s" %
                          (cleanup_error or "complete", diagnostics(stdout, stderr)))
        try:
            stdout, stderr = process.communicate(timeout=min(remaining, 0.2))
            break
        except subprocess.TimeoutExpired:
            pass
    if process.returncode:
        # Helper output can contain credentials from inherited tools; keep the
        # failure provenance without placing raw diagnostics in shared JSON.
        raise Refused("dispatch workspace helper failed (exit %s): %s; private logs: %s" %
                      (process.returncode, os.path.basename(path), diagnostics(stdout, stderr)))
    return stdout.decode("utf-8").strip()


def verify_checkout(manifest, work, environment):
    def git(*args):
        return subprocess.check_output(["git", "-C", work] + list(args), env=environment,
                                       stderr=subprocess.DEVNULL,
                                       timeout=min(30, manifest["checkpoint_timeout"])).decode("utf-8").strip()
    if git("rev-parse", "HEAD") != manifest["commit"]:
        raise Refused("prepared checkout does not match immutable source commit")
    if git("symbolic-ref", "--short", "HEAD") != manifest["branch"]:
        raise Refused("prepared checkout has the wrong dispatch branch")
    origin = git("remote", "get-url", "origin")
    if os.path.realpath(origin) != os.path.join(manifest["directory"], "results.git"):
        raise Refused("worker origin is not the private results repository")
    # Native HOME isolation intentionally excludes shared user Git config.
    git("config", "user.name", "Sue the Coder")
    git("config", "user.email", "coder@sucoder.dev")


def prompt_for(manifest, work):
    channels = {}
    for name, text in (("MESSAGES_TO_AGENT.txt", "Coordinator steering messages appear here.\n"),
                       ("QUESTIONS_FROM_AGENT.txt", "Record questions here; continue independent work.\n")):
        shared = os.path.join(manifest["directory"], name)
        if not os.path.lexists(shared):
            try:
                fd = os.open(shared, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                pass  # another legitimate coordinator initialized the channel
            else:
                with os.fdopen(fd, "w") as stream:
                    stream.write(text)
        private_file(shared)
        channels[name] = shared
        path = os.path.join(work, name)
        if not os.path.lexists(path):
            # These are conveniences; absolute shared channels remain the
            # contract even when the source already tracks these filenames.
            try:
                os.symlink(shared, path)
            except FileExistsError:
                pass
    with open(manifest["task_file"]) as stream:
        task = stream.read()
    prelude = ("You are a headless dispatched worker. Work in %s.\n"
               "Initial commit: %s\nBranch: %s\nPrivate results repository: %s/results.git\n"
               "Read AGENTS.md and any existing project instructions. Read the mutable shared steering channel %s "
               "at checkpoints; record questions in the mutable shared channel %s. "
               "Use these absolute channels even if the source contains files with the same names. "
               "No human is attached to approve prompts. "
               "Commits publish to your private result repository; do not push to the source repository.\n"
               "Save reviewable evidence under SUCODER_DISPATCH_EVIDENCE_DIR (%s/evidence). "
               "Before finishing, write optional JSON to SUCODER_DISPATCH_TASK_RESULT (%s/task-result.json): "
               "{\"status\":\"completed\",\"summary\":\"...\",\"unexecuted\":[]}. "
               "Choose status completed, blocked, unfinished, or scientific_disagreement. "
               "This reports the task outcome separately from infrastructure execution; coordinator acceptance remains pending.\n\n")
    return (prelude % (work, manifest["commit"], manifest["branch"], manifest["directory"],
                      channels["MESSAGES_TO_AGENT.txt"], channels["QUESTIONS_FROM_AGENT.txt"],
                      manifest["directory"], manifest["directory"]) + task).encode("utf-8")


def inspect_placement(child, runner, root, home, proc_root="/proc", expected_environment=None,
                      ownership=None, manifest=None):
    """Reuse the managed runner's process/FD machinery, checking readability."""
    members = ownership.members() if ownership is not None else []
    pids = [record["pid"] for record in members] if ownership is not None else [child.pid] + runner.descendants(child.pid)
    readable = []
    for pid in pids:
        path = os.path.join(proc_root, str(pid), "fd")
        try:
            entries = os.listdir(path)
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ESRCH):
                continue  # process teardown can remove fd/ before its proc dir
            raise Refused("cannot inspect owned child file descriptors (pid=%s errno=%s)" % (pid, exc.errno))
        # Descriptor links are proc metadata: never stat a magic FD link,
        # which would touch the target filesystem and could block on NFS.
        for entry in entries:
            try:
                descriptor = os.path.join(path, entry)
                os.readlink(descriptor)
            except OSError as exc:
                if exc.errno not in (errno.ENOENT, errno.ESRCH):
                    raise Refused("cannot inspect owned child descriptor links (pid=%s errno=%s)" % (pid, exc.errno))
        readable.append(pid)
    environment_verified = False
    if expected_environment is not None and child.pid in readable:
        try:
            with open(os.path.join(proc_root, str(child.pid), "environ"), "rb") as stream:
                effective = dict(entry.split(b"=", 1) for entry in stream.read().split(b"\x00") if b"=" in entry)
        except OSError as exc:
            if exc.errno not in (errno.ENOENT, errno.ESRCH):
                raise Refused("cannot inspect owned child effective environment (pid=%s errno=%s)" % (child.pid, exc.errno))
        else:
            for key in ("HOME", "CODEX_HOME", "CLAUDE_CONFIG_DIR", "SUCODER_LOCAL_ROOT"):
                if key in expected_environment and effective.get(key.encode()) != expected_environment[key].encode():
                    raise Refused("owned child did not receive isolated " + key)
            environment_verified = True
    observed, misplaced, databases, checkpoint_copies = [], [], [], []
    for descriptor in runner.inspect_file_mounts(readable, proc_root=proc_root):
        pid, path = descriptor["pid"], descriptor["path"]
        path = re.sub(r" \(deleted\)$", "", path)
        base = os.path.basename(path)
        database = bool(re.search(r"\.(?:sqlite(?:3)?|db)(?:-(?:wal|shm|journal))?$", base))
        # stdin lives below HOME/tmp but is supplied by this supervisor: its
        # descriptor is not evidence that the native agent adopted tiering.
        relevant = database or any(beneath_text(path, os.path.join(home, name)) for name in (".claude", ".codex"))
        if not relevant:
            continue
        # The audited runner publishes closed SQLite backups by byte-copying
        # into private .incoming directories, then hashes those copies. Only
        # the runner supervisor may own those exact durable copy paths.
        durable_copy = False
        if manifest is not None and manifest["agent"] == "codex" and pid == child.pid:
            base_path = os.path.join(manifest["directory"], "state", "codex", manifest["scope"], "runs")
            relative = os.path.relpath(path, base_path)
            stores = tuple(runner.REQUIRED_STORES) + tuple(runner.OPTIONAL_STORES)
            durable_copy = bool(re.fullmatch(r"[0-9a-f]{32}/(?:\.incoming-[0-9a-f]{32}|generations/[0-9]{6}-[0-9a-f]{8})/db/[A-Za-z0-9_]+\.sqlite", relative)
                                and base in stores)
        if durable_copy:
            checkpoint_copies.append(path)
        elif descriptor["filesystem"] in LOCAL_FILESYSTEMS and beneath_text(path, root):
            observed.append(path)
            if database:
                databases.append(path)
        else:
            misplaced.append(path)
    return {"observed": sorted(set(observed)), "databases": sorted(set(databases)),
            "misplaced": sorted(set(misplaced)), "inspected_pids": readable,
            "environment_verified": environment_verified, "members": members,
            "durable_checkpoint_copies": sorted(set(checkpoint_copies))}


def stop_owned(child, timeout):
    # Popen(start_new_session=True) makes the child the owner of this group.
    # Signal the group even if its leader has exited but left descendants.
    if child.pid == os.getpgrp():
        raise Refused("refusing to signal the coordinator process group")
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        child.poll()  # reap the leader, so a zombie alone is not a live group
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=5)


def codex_preservation(manifest, root):
    scope = os.path.join(root, "state", "codex", manifest["scope"])
    statuses = []
    if os.path.isdir(scope):
        for name in os.listdir(scope):
            path = os.path.join(scope, name, "status.json")
            if os.path.isfile(path):
                statuses.append(read_json(path))
    if len(statuses) != 1:
        return {"state": "unverified", "reason": "managed runner status absent or ambiguous"}
    status = statuses[0]
    generation = status.get("last_generation")
    if not generation or not isinstance(generation, dict):
        return {"state": "failed", "reason": "no acknowledged managed state generation", "runner_state": status.get("state")}
    path = generation.get("path", "")
    if not path or not contained(path, os.path.join(manifest["directory"], "state")):
        return {"state": "failed", "reason": "managed generation escaped private durable state"}
    stored_path = os.path.join(path, "manifest.json")
    with open(stored_path, "rb") as stream:
        encoded = stream.read()
    if hashlib.sha256(encoded).hexdigest() != generation.get("manifest_sha256"):
        return {"state": "failed", "reason": "managed generation manifest hash mismatch"}
    stored = json.loads(encoded.decode("utf-8"))
    classification = stored.get("class")
    if stored.get("scope") != manifest["scope"] or classification != generation.get("class"):
        return {"state": "failed", "reason": "managed generation identity mismatch"}
    return {"state": "preserved" if classification == "sealed" else classification or "unverified",
            "class": classification, "generation": generation,
            "native_version": stored.get("codex_version"),
            "effective_model": stored.get("model"),
            "runner_state": status.get("state"), "runner_outcome": status.get("outcome")}


def preserve_claude(manifest, home, heartbeat=None):
    """Only stopped JSON/text state is copied; SQLite is never file-copied."""
    target = os.path.join(manifest["directory"], "state", "claude")
    os.makedirs(target, mode=0o700, exist_ok=True)
    copied, excluded, unsupported_stores, excluded_subtrees = [], [], [], []
    private_seeds = {".claude/.credentials.json", ".claude/settings.json", ".claude/.claude.json", ".claude.json",
                     ".codex/auth.json", ".codex/config.toml"}
    for directory, dirs, files in os.walk(home, followlinks=False):
        kept = []
        for name in dirs:
            path = os.path.join(directory, name)
            relative = os.path.relpath(path, home)
            if (os.path.islink(path) or name in ("cache", "tmp", "node_modules", "backups", ".git")
                    or relative in (os.path.join(".claude", "plugins"), os.path.join(".claude", "skills"))):
                excluded_subtrees.append(relative)
            else:
                kept.append(name)
        dirs[:] = kept
        for name in files:
            if heartbeat is not None:
                heartbeat()
            source = os.path.join(directory, name)
            relative = os.path.relpath(source, home)
            if relative in private_seeds:
                excluded.append(relative)
                continue
            if os.path.islink(source):
                excluded.append(relative)
                continue
            # Include known transcript/config formats only, never a live
            # database, sidecar, opaque binary, socket or named pipe.
            if re.search(r"\.(?:sqlite(?:3)?|db)(?:-(?:wal|shm|journal))?$", name):
                excluded.append(relative)
                unsupported_stores.append(relative)
                continue
            if not stat.S_ISREG(os.stat(source).st_mode) or not name.endswith((".json", ".jsonl", ".txt", ".md", ".yaml", ".yml", ".toml")):
                continue
            destination = os.path.join(target, relative)
            os.makedirs(os.path.dirname(destination), mode=0o700, exist_ok=True)
            copy_private(source, destination)
            copied.append(relative)
    return {"state": "unsupported" if unsupported_stores else "text_preserved", "path": target,
            "files": copied, "excluded": excluded, "excluded_subtrees": sorted(excluded_subtrees),
            "unsupported_stores": unsupported_stores,
            "sqlite": "unsupported; database files and sidecars are excluded"}


def native_state_files(home):
    """Metadata for local regular state files; seeds are compared before/after."""
    files = {}
    for name in (".claude", ".codex"):
        base = os.path.join(home, name)
        if not os.path.isdir(base) or os.path.islink(base):
            continue
        for directory, dirs, names in os.walk(base, followlinks=False):
            dirs[:] = [item for item in dirs if not os.path.islink(os.path.join(directory, item))]
            for item in names:
                path = os.path.join(directory, item)
                info = os.lstat(path)
                if stat.S_ISREG(info.st_mode):
                    files[path] = (info.st_ino, info.st_size, info.st_mtime_ns)
    return files


def run(manifest_path, inherited=None, mountinfo_path="/proc/self/mountinfo", ownership_factory=None):
    os.umask(0o077)
    manifest = load_manifest(manifest_path)
    directory = manifest["directory"]
    record = {"schema_version": SCHEMA_VERSION, "dispatch_id": manifest["dispatch_id"], "uid": os.getuid(),
              "agent": manifest["agent"], "commit": manifest["commit"], "branch": manifest["branch"],
              "node": socket.gethostname(), "state": "starting", "exit_code": None,
              "placement": {"state": "unverified", "sqlite": "not_observed"},
              "preservation": {"state": "unverified"}, "started_at": time.time(),
              "phase": "cloning", "coordinator_acceptance": "pending",
              "task": {"status": "not_reported"},
              "versions": {"dispatcher": manifest.get("dispatcher_version"),
                           "adapter": manifest.get("adapter_version"),
                           "requested_model": manifest.get("requested_model", manifest.get("model")),
                           "effective_model": None},
              "publication": {"state": "not_started", "incomplete": True,
                              "repository": os.path.join(directory, "results.git"),
                              "ref": "refs/heads/" + manifest["branch"], "snapshot_ref": None,
                              "published_at": None, "evidence": {"state": "not_published"}},
              "logs": {"stdout": os.path.join(directory, "stdout.log"), "stderr": os.path.join(directory, "stderr.log")},
              "result_path": os.path.join(directory, "result.json"), "final_path": os.path.join(directory, "final.txt")}
    child, work, environment, ownership, artifacts, stop_signal = None, None, None, None, None, [None]
    previous = {}
    last_published = [None, 0]
    def signalled(signum, frame):
        stop_signal[0] = signum
    def publish(phase=None, force=False):
        if phase is not None:
            record["phase"] = phase
        signature = (record["phase"], record["state"], record["placement"]["state"],
                     record["preservation"]["state"], record.get("reason"), record["publication"]["state"])
        now = time.monotonic()
        if not force and signature == last_published[0] and now - last_published[1] < 2:
            return
        record["observed_at"] = record["heartbeat_at"] = time.time()
        atomic_json(os.path.join(directory, "status.json"), record)
        last_published[:] = [signature, now]
    def final_publication():
        publish("final_publication", force=True)
        try:
            snapshot = json.loads(run_script(manifest["snapshot_script"], environment,
                                             manifest["checkpoint_timeout"], heartbeat=publish))
            record["workspace_snapshot"] = snapshot
            head = subprocess.check_output(["git", "-C", work, "rev-parse", "HEAD"], env=environment,
                                           stderr=subprocess.DEVNULL,
                                           timeout=min(10, manifest["checkpoint_timeout"])).decode().strip()
            record["final_head"] = head
            record["publication"].update(state="complete", incomplete=False, head=head,
                                         snapshot_ref=snapshot.get("ref"), published_at=time.time())
            return True
        except Exception as exc:
            record["publication"].update(state="incomplete", incomplete=True,
                                         reason="%s: %s" % (type(exc).__name__, exc))
            record["workspace_snapshot"] = {"state": "failed", "reason": str(exc)}
            return False
    try:
        artifacts = load_artifacts(directory)
        root, job_id, mount = receiving_root(manifest, os.environ if inherited is None else inherited, mountinfo_path)
        receiving_environment = os.environ if inherited is None else inherited
        cluster = receiving_environment.get("SLURM_CLUSTER_NAME") or None
        if manifest.get("cluster") is not None and cluster != manifest["cluster"]:
            raise Refused("receiving Slurm cluster differs from immutable submission source")
        record.update(local_root=root, job_id=job_id, cluster=cluster, mount=mount)
        ownership = (ownership_factory or JobOwnership)(job_id)
        record["allocation_ownership"] = ownership.describe()
        environment = environment_for(manifest, receiving_environment, root)
        home = seed_home(manifest, environment)
        if manifest["agent"] == "codex":
            record["canonical_home"] = environment["CODEX_HOME"]
            record["managed_state"] = {"canonical_home": environment["CODEX_HOME"], "scope": manifest["scope"],
                                       "durable_root": os.path.join(directory, "state"),
                                       "recovery": "explicit existing-runner recovery; no automatic resume",
                                       "seed_retention": "private cold auth/config seed retained for explicit recovery; excluded from generations"}
        work = os.path.join(root, "mirrors", "dispatch_" + manifest["dispatch_id"])
        record["work_path"] = work
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous[signum] = signal.signal(signum, signalled)
        publish("cloning", force=True)
        run_script(manifest["prepare_script"], environment, max(120, manifest["checkpoint_timeout"]), stop_signal,
                   heartbeat=publish)
        publish("agent_setup", force=True)
        verify_checkout(manifest, work, environment)
        prompt = prompt_for(manifest, work)
        runner = load_runner(manifest["runner_path"])
        command = command_for(manifest, environment)
        record["versions"]["native"] = native_version(manifest, environment)
        state_before = native_state_files(environment["HOME"])
        # File-backed input avoids a full pipe blocking the supervisor before
        # it can inspect placement or forward a scheduler termination.
        input_path = os.path.join(environment["TMPDIR"], "dispatch-input.txt")
        with open(input_path, "wb") as stream:
            stream.write(prompt)
        with open(input_path, "rb") as stdin, open(record["logs"]["stdout"], "wb") as stdout, open(record["logs"]["stderr"], "wb") as stderr:
            child = subprocess.Popen(command, cwd=work, env=environment, stdin=stdin,
                                     stdout=stdout, stderr=stderr, start_new_session=True)
            record.update(state="running", pid=child.pid)
            publish("agent_execution", force=True)
            observed, databases = set(), set()
            deadline = time.monotonic() + manifest.get("snapshot_minutes", manifest["checkpoint_minutes"]) * 60
            while child.poll() is None:
                evidence = inspect_placement(child, runner, root, environment["HOME"], expected_environment=environment,
                                             ownership=ownership, manifest=manifest)
                observed.update(evidence["observed"])
                databases.update(evidence["databases"])
                record["placement"].update(observed=sorted(observed), sqlite="observed_local" if databases else "not_observed",
                                           databases=sorted(databases), inspected_pids=evidence["inspected_pids"])
                record["placement"]["durable_checkpoint_copies"] = evidence["durable_checkpoint_copies"]
                record["placement"]["environment_verified"] = (record["placement"].get("environment_verified", False)
                                                                  or evidence["environment_verified"])
                if evidence["misplaced"]:
                    record["placement"].update(state="misplaced", misplaced=evidence["misplaced"])
                    stop_allocation(ownership, manifest["checkpoint_timeout"])
                    stop_owned(child, manifest["checkpoint_timeout"])
                    break
                if observed and record["placement"]["environment_verified"]:
                    record["placement"]["state"] = "verified_observations"
                if stop_signal[0] is not None:
                    stop_allocation(ownership, manifest["checkpoint_timeout"])
                    stop_owned(child, manifest["checkpoint_timeout"])
                    break
                if time.monotonic() >= deadline:
                    try:
                        record["workspace_snapshot"] = json.loads(run_script(manifest["snapshot_script"], environment,
                                                                             manifest["checkpoint_timeout"], stop_signal,
                                                                             heartbeat=publish))
                    except Exception as exc:
                        record["workspace_snapshot"] = {"state": "failed", "reason": str(exc)}
                    record["workspace_snapshot"]["observed_at"] = time.time()
                    deadline = time.monotonic() + manifest.get("snapshot_minutes", manifest["checkpoint_minutes"]) * 60
                publish()
                time.sleep(0.1)
            code = child.wait()
            # A native CLI must not leave an independent writer alive while
            # its supposedly stopped state is copied to durable storage.
            stop_allocation(ownership, manifest["checkpoint_timeout"])
            stop_owned(child, manifest["checkpoint_timeout"])
        record["exit_code"] = code if code >= 0 else 128 - code
        # Protect code and the dirty patch before conversation state copying,
        # which can fail or stall independently on durable storage.
        final_publication()
        changed = sorted(path for path, fingerprint in native_state_files(environment["HOME"]).items()
                         if state_before.get(path) != fingerprint)
        record["placement"]["post_exit_state_files"] = changed
        if (record["placement"]["state"] == "unverified" and changed
                and record["placement"].get("environment_verified")):
            record["placement"]["state"] = "verified_state_files"
        record["task"] = artifacts.read_task_outcome(directory)
        publish("preservation", force=True)
        if manifest["agent"] == "claude":
            shutil.copyfile(record["logs"]["stdout"], record["final_path"])
            record["preservation"] = preserve_claude(manifest, environment["HOME"], heartbeat=publish)
        else:
            record["preservation"] = codex_preservation(manifest, root)
            version = record["preservation"].get("native_version")
            record["versions"]["native"] = ({"state": "observed", "version": version, "source": "managed_generation"}
                                             if version else {"state": "unavailable", "source": "managed_generation"})
            record["versions"]["effective_model"] = record["preservation"].get("effective_model")
        if stop_signal[0] is not None:
            record["state"] = "interrupted"
        elif (record["exit_code"] == 0 and record["placement"]["state"] in ("verified_observations", "verified_state_files")
              and record["preservation"]["state"] in ("preserved", "text_preserved")):
            record["state"] = "succeeded"
        else:
            record["state"] = "failed"
    except Exception as exc:
        if child is not None:
            try:
                if ownership is not None:
                    stop_allocation(ownership, manifest["checkpoint_timeout"])
                stop_owned(child, manifest["checkpoint_timeout"])
            except Exception as cleanup_error:
                record["cleanup_error"] = "%s: %s" % (type(cleanup_error).__name__, cleanup_error)
            code = child.poll()
            record["exit_code"] = code if code is None or code >= 0 else 128 - code
        record.update(state="interrupted" if stop_signal[0] is not None else "failed" if child is not None else "refused", reason=str(exc),
                      failure_type=type(exc).__name__)
        # Attempt a final workspace publication even on agent-state failure.
        if work and os.path.isdir(work) and environment and record["publication"]["state"] != "complete":
            final_publication()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if artifacts is not None:
        try:
            record["task"] = artifacts.read_task_outcome(directory)
            record["publication"]["evidence"] = artifacts.publish_evidence(directory, required=manifest.get("required_evidence", []))
        except Exception as exc:
            record["publication"]["evidence"] = {"state": "incomplete", "reason": "%s: %s" % (type(exc).__name__, exc)}
    else:
        record["publication"]["evidence"] = {"state": "incomplete", "reason": "artifact helper unavailable"}
    # Workspace and proof publication are independent facts. Keep the
    # successful branch/snapshot receipt even when required proof is missing,
    # while making the aggregate marker describe the complete publication.
    publication = record["publication"]
    publication["workspace"] = {key: publication[key] for key in
                                ("state", "incomplete", "repository", "ref", "head", "snapshot_ref", "published_at", "reason")
                                if key in publication}
    incomplete = []
    if publication["workspace"]["state"] != "complete":
        incomplete.append("workspace")
    if publication["evidence"].get("state") != "complete":
        incomplete.append("evidence")
    publication.update(state="incomplete" if incomplete else "complete",
                       incomplete=bool(incomplete), incomplete_components=incomplete)
    if incomplete and record["state"] == "succeeded":
        record["state"] = "failed"
    if stop_signal[0] is not None:
        supervisor_exit = 128 + stop_signal[0]
        record["signal"] = stop_signal[0]
    elif record["placement"]["state"] == "misplaced":
        supervisor_exit = EXIT_PLACEMENT
    elif record["exit_code"]:
        supervisor_exit = record["exit_code"]
    elif record["state"] == "succeeded":
        supervisor_exit = 0
    else:
        supervisor_exit = EXIT_REFUSED if record["state"] == "refused" else EXIT_PRESERVATION
    record["supervisor_exit_code"] = supervisor_exit
    record["finished_at"] = time.time()
    publish("finished", force=True)
    atomic_json(record["result_path"], record)
    return supervisor_exit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="private staged dispatch manifest")
    args = parser.parse_args(argv)
    try:
        return run(os.path.abspath(args.manifest))
    except Exception as exc:
        # No untrusted manifest path is a safe location for a failure record.
        sys.stderr.write("dispatch worker refused (%s)\n" % type(exc).__name__)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
