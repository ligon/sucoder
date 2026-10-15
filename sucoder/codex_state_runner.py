#!/usr/bin/env python3
"""SuCoder managed Codex state runner: one owned Codex process on local disk.

Staged to the compute node and run as ``python3 <this file> --scope ...
--durable-root ... -- <command>``, like the native context installer, so it
has no dependency beyond the standard library and must stay valid for the
oldest ``python3`` a node image offers (3.6 has been seen).

What it does, per ``docs/codex-local-state.org`` ("Revision after Slice 0"):

- resolves ``SUCODER_LOCAL_ROOT`` and creates
  ``<root>/state/codex/<scope>/<run>/{db,home,snapshots}`` with mode 0700;
- takes a per-scope startup lock so one allocation runs one managed server
  per scope;
- substitutes the ``sqlite_home`` placeholder the launcher inserted into the
  command with the absolute ``db`` path, and refuses when the caller's argv,
  ``CODEX_SQLITE_HOME`` or ``$CODEX_HOME/config.toml`` already set it;
- binds local directories over the write-heavy subdirectories of the shared
  ``CODEX_HOME`` in a per-process user+mount namespace (``bwrap``), so the
  shared home is untouched for every other process; entries that do not
  exist in the home are skipped and recorded, never created;
- runs Codex as a child in the same process group, ignores SIGINT (the TTY
  delivers it to Codex), forwards SIGTERM and SIGHUP, and exits with the
  child's code;
- verifies placement through ``/proc``: every ``.sqlite`` descriptor must
  be under ``db``, and every rollout descriptor must be on local disk;
  anything else stops the child and fails with a specific error.  Nothing
  ever falls back to shared placement;
- replicates every rollout under ``home/sessions`` by file growth to
  ``DURABLE/runs/<run>/rollouts/<thread-id>.jsonl`` with fsync, from a
  thread of its own, so a slow durable filesystem delays the replica and
  never the conversation (``replica.json``, ``status.json["replica"]``);
- captures database generations with the SQLite backup API: an initial one
  once placement is verified, one per ``--checkpoint-minutes``, one per
  ``<run>/checkpoint.request``, and a final sealed or crash one after Codex
  exits.  A generation is published only after every thread's replica has
  reached the rollout offset the copied history projection records, as an
  atomically renamed ``DURABLE/runs/<run>/generations/<generation>/`` whose
  ``manifest.json`` is written last.

Restore (``--restore``) and unattended continuation are the second half of
Slice 2 and are not here yet.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid

PLACEHOLDER_DEFAULT = "__SUCODER_CODEX_STATE_DB__"
BIND_DIRS = ["sessions", ".tmp", "tmp", "thread-writer-locks", "shell_snapshots", "app-server-control"]
COPY_DIRS = ["skills"]
BIND_FILES = ["AGENTS.md", "AGENTS.override.md", "history.jsonl"]
REQUIRED_DIRS = ["sessions"]
SAFE_PATH = re.compile(r"^[A-Za-z0-9/_.-]+$")
EXIT_REFUSED = 70
EXIT_MISPLACED = 71
# The classified store set.  Required stores are required once seen; the
# optional ones are recorded as absent until Codex creates them.
REQUIRED_STORES = ("state_5.sqlite", "thread_history_1.sqlite")
OPTIONAL_STORES = ("goals_1.sqlite", "queue_1.sqlite", "memories_1.sqlite", "logs_2.sqlite")
ROLLOUT_NAME = re.compile(r"^rollout-.*-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.jsonl$")
REPLICATION_INTERVAL = 1.0
CHUNK = 1 << 20
LOCAL_BUNDLES_KEPT = 3


class Refused(Exception):
    """The launch cannot proceed under managed placement; nothing was started."""


class CaptureFailed(Exception):
    """The candidate generation is invalid; nothing was published."""


class CaptureSkipped(Exception):
    """There was nothing to capture."""


class NotDurable(Exception):
    """A replica did not reach its projection offset in budget; the candidate stays local."""


LOG_FILE = {"path": None, "quiet": False}


def log(message):
    """To stderr until Codex owns the terminal, to the run's log file always."""
    line = "SUCODER codex-state: " + message + "\n"
    if LOG_FILE["path"]:
        try:
            with open(LOG_FILE["path"], "a") as handle:
                handle.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + line)
        except OSError:
            pass
    if not LOG_FILE["quiet"]:
        sys.stderr.write(line)
        sys.stderr.flush()


def utc_iso(when):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when))


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        handle.write(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def fsync_path(path):
    """fsync a file or a directory by path (a directory for its entries)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sha256_file(path, limit=None):
    """(hex digest, bytes hashed) of a file, or of its first ``limit`` bytes."""
    digest = hashlib.sha256()
    hashed = 0
    with open(path, "rb") as handle:
        while limit is None or hashed < limit:
            size = CHUNK if limit is None else min(CHUNK, limit - hashed)
            chunk = handle.read(size)
            if not chunk:
                break
            digest.update(chunk)
            hashed += len(chunk)
    return digest.hexdigest(), hashed


def copy_file(source, target):
    """Exclusive creation, 0600, every byte written and fsynced before return."""
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with open(source, "rb") as handle:
            while True:
                chunk = handle.read(CHUNK)
                if not chunk:
                    break
                view = memoryview(chunk)
                while len(view):
                    view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)


class Status(object):
    """``status.json``: the placement state with its transitions, plus the
    replica, checkpoint and last-generation views the other threads merge in.
    Three threads write it, so every write happens under one lock."""

    def __init__(self, path):
        self.path = path
        self.data = {"transitions": []}
        self.lock = threading.Lock()
        self.durable_at = None     # monotonic time of the last durable generation

    def update(self, state, **fields):
        with self.lock:
            self.data.update(fields)
            self.data["state"] = state
            self.data["transitions"].append({"state": state, "at": time.time()})
            self._write()

    def note(self, **fields):
        """Merge fields without a state transition."""
        with self.lock:
            self.data.update(fields)
            self._write()

    def _write(self):
        age = None if self.durable_at is None else round(time.monotonic() - self.durable_at, 1)
        self.data["durable_age_seconds"] = age
        write_json(self.path, self.data)


def mkdir_private(path):
    """The allocation-local tier: ours entirely, so an existing directory is re-tightened."""
    if not os.path.isdir(path):
        os.makedirs(path, 0o700)
    os.chmod(path, 0o700)


def ensure_private_dir(path):
    """The durable tier: create with 0700 when missing, never chmod a directory we did not create."""
    if os.path.isdir(path):
        return False
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        return False
    os.chmod(path, 0o700)
    return True


def ensure_private_tree(path):
    missing = []
    current = path
    while not os.path.isdir(current):
        missing.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    for directory in reversed(missing):
        ensure_private_dir(directory)


def resolve_local_root():
    root = os.environ.get("SUCODER_LOCAL_ROOT")
    if not root:
        raise Refused("SUCODER_LOCAL_ROOT is not set; managed placement needs local-disk tiering")
    if not os.path.isdir(root):
        raise Refused("SUCODER_LOCAL_ROOT=%s is not a directory" % root)
    if not SAFE_PATH.match(root):
        raise Refused("SUCODER_LOCAL_ROOT=%s contains characters the placeholder substitution cannot carry" % root)
    return root


def resolve_durable_base(base, local_root):
    """The configured durable root: absolute, plain characters, not allocation-local."""
    if not os.path.isabs(base):
        raise Refused("--durable-root %s is not an absolute path" % base)
    if not SAFE_PATH.match(base):
        raise Refused("--durable-root %s contains characters outside [A-Za-z0-9/_.-]" % base)
    base = os.path.normpath(base)
    real, local_real = os.path.realpath(base), os.path.realpath(local_root)
    if real == local_real or real.startswith(local_real + "/"):
        raise Refused("--durable-root %s is under the allocation-local root %s; allocation-local storage is not durable" % (base, local_root))
    return base


def resolve_codex_home(explicit):
    home = explicit or os.environ.get("CODEX_HOME") or os.path.join(os.path.expanduser("~"), ".codex")
    real = os.path.realpath(home)
    if not os.path.isdir(real):
        raise Refused("CODEX_HOME %s (%s) is not a directory" % (home, real))
    return home, real


def profile_names(command):
    """``-p NAME`` / ``--profile NAME`` / ``--profile=NAME`` in the Codex argv."""
    names = []
    for i, arg in enumerate(command):
        if arg in ("-p", "--profile") and i + 1 < len(command):
            names.append(command[i + 1])
        elif arg.startswith("--profile="):
            names.append(arg.split("=", 1)[1])
    return names


def check_conflicts(command, placeholder, codex_home_real):
    if os.environ.get("CODEX_SQLITE_HOME"):
        raise Refused("CODEX_SQLITE_HOME is set in the launch environment; managed placement would silently win over it. Unset it or disable codex_state.")
    seen = 0
    for arg in command:
        if "sqlite_home=" in arg:
            if placeholder in arg:
                seen += 1
            else:
                raise Refused("the command already sets sqlite_home (%s); managed placement refuses to override an explicit choice" % arg[:80])
    if seen == 0:
        raise Refused("the command carries no sqlite_home placeholder; the launcher did not prepare it for managed placement")
    configs = [os.path.join(codex_home_real, "config.toml")]
    configs += [os.path.join(codex_home_real, name + ".config.toml") for name in profile_names(command)]
    for config in configs:
        if os.path.isfile(config):
            with open(config, "r") as handle:
                for line in handle:
                    if re.match(r"^\s*sqlite_home\s*=", line):
                        raise Refused("%s sets sqlite_home; managed placement would win over it. Remove that line or disable codex_state." % config)


def substitute(command, placeholder, db_dir):
    if not SAFE_PATH.match(db_dir):
        raise Refused("database path %s contains characters the substitution cannot carry" % db_dir)
    return [arg.replace(placeholder, db_dir) for arg in command]


def prepare_binds(codex_home_real, home_local):
    """Local directories and files bound over entries of the shared home."""
    binds = []
    skipped = []
    for name in BIND_DIRS:
        target = os.path.join(codex_home_real, name)
        if os.path.isdir(target) and not os.path.islink(target):
            local = os.path.join(home_local, name)
            mkdir_private(local)
            binds.append((local, target))
        else:
            skipped.append(name)
            if name in REQUIRED_DIRS:
                raise Refused("%s does not exist in CODEX_HOME; managed placement needs it and will not create entries in the shared home" % target)
    for name in COPY_DIRS:
        target = os.path.join(codex_home_real, name)
        if os.path.isdir(target) and not os.path.islink(target):
            local = os.path.join(home_local, name)
            if os.path.exists(local):
                shutil.rmtree(local)
            shutil.copytree(target, local, symlinks=True)
            os.chmod(local, 0o700)
            binds.append((local, target))
        else:
            skipped.append(name)
    for name in BIND_FILES:
        target = os.path.join(codex_home_real, name)
        if os.path.isfile(target) and not os.path.islink(target):
            local = os.path.join(home_local, name)
            shutil.copyfile(target, local)
            os.chmod(local, 0o600)
            binds.append((local, target))
        else:
            skipped.append(name)
    return binds, skipped


def bwrap_argv(binds, command):
    argv = ["bwrap", "--dev-bind", "/", "/"]
    for local, target in binds:
        argv += ["--bind", local, target]
    argv += ["--unshare-user", "--uid", str(os.getuid()), "--gid", str(os.getgid()), "--die-with-parent", "--"]
    return argv + list(command)


def descendants(root_pid):
    children = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % name) as handle:
                stat = handle.read()
        except OSError:
            continue
        try:
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(name))
    found = []
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        for child in children.get(pid, []):
            found.append(child)
            stack.append(child)
    return found


def parent_of(pid):
    try:
        with open("/proc/%d/stat" % pid) as handle:
            return int(handle.read().rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return None


def open_files(pids):
    """(pid, link path, st_dev) for every open file of the given pids."""
    out = []
    for pid in pids:
        fd_dir = "/proc/%d/fd" % pid
        try:
            names = os.listdir(fd_dir)
        except OSError:
            continue
        for name in names:
            entry = os.path.join(fd_dir, name)
            try:
                link = os.readlink(entry)
                if not link.startswith("/"):
                    continue
                st = os.stat(entry)          # follows the magic link into the child's namespace
            except OSError:
                continue
            out.append((pid, link, st.st_dev))
    return out


def holders_of(directory):
    """Pids (readable to us) with a descriptor open under the directory."""
    pids = [int(name) for name in os.listdir("/proc") if name.isdigit()]
    return sorted(set(pid for pid, link, _ in open_files(pids) if link.startswith(directory + "/")))


def classify(files, db_dir, db_dev, local_dev):
    """Return (placed, misplaced): descriptors on stores and rollouts."""
    placed = []
    misplaced = []
    for pid, link, dev in files:
        base = os.path.basename(link)
        if ".sqlite" in base:
            if dev == db_dev and (link == db_dir or link.startswith(db_dir + "/")):
                placed.append(link)
            else:
                misplaced.append(link)
        elif base.startswith("rollout-") and base.endswith(".jsonl"):
            if dev == local_dev:
                placed.append(link)
            else:
                misplaced.append(link)
    return placed, misplaced


def elide(command):
    """The argv for owner.json without transcripts: long values become a length marker."""
    out = []
    for arg in command:
        if len(arg) > 200:
            out.append("<%d bytes elided>" % len(arg))
        else:
            out.append(arg)
    return out


def monitor(child, db_dir, local_root, status, timeout, result, placed_event=None):
    db_dev = os.stat(db_dir).st_dev
    local_dev = os.stat(local_root).st_dev
    deadline = time.time() + timeout
    reported = False
    while child.poll() is None:
        pids = [child.pid] + descendants(child.pid)
        placed, misplaced = classify(open_files(pids), db_dir, db_dev, local_dev)
        if misplaced:
            result["outcome"] = "misplaced"
            result["misplaced"] = sorted(set(misplaced))
            status.update("misplaced", misplaced=sorted(set(misplaced)), placed=sorted(set(placed)))
            log("Codex opened state outside the managed placement: %s; stopping it" % ", ".join(sorted(set(misplaced))[:4]))
            try:
                child.terminate()
                for _ in range(50):
                    if child.poll() is not None:
                        break
                    time.sleep(0.1)
                if child.poll() is None:
                    child.kill()
            except OSError:
                pass
            return
        if placed and not reported:
            reported = True
            result["outcome"] = "placed"
            status.update("placed", placed=sorted(set(placed)))
            log("state placed under %s (%d descriptors)" % (db_dir, len(set(placed))))
            if placed_event is not None:
                placed_event.set()
        if not reported and time.time() > deadline:
            result["outcome"] = "unverified"
            status.update("unverified", note="no store descriptor appeared within %ds" % timeout)
            log("no state descriptor appeared within %ds; placement unverified" % timeout)
            return
        time.sleep(5.0 if reported else 0.5)   # keep watching for a later misplacement, cheaply
    if not reported and "outcome" not in result:
        result["outcome"] = "no_stores_opened"
        status.update("no_stores_opened")


# ----------------------------------------------------------------- replication

class Replicator(object):
    """Follows each rollout under ``home/sessions`` by growth and appends the
    new bytes to its durable replica, from a thread of its own.

    The replica file's own size is the truth for the replicated offset (a
    partial write or a failed fsync must never double-append).  A local file
    shorter than its replica is a "shrunk" anomaly that stops replication of
    that thread: the rollout is append-only on this build and a shrink is
    never replicated.  Errors on the durable side (a slow or unwritable
    filesystem) are retried on the next pass and shown as replica age and
    lag; they block this thread only.
    """

    def __init__(self, sessions, rollouts, state_path, status):
        self.sessions = sessions
        self.rollouts = rollouts
        self.state_path = state_path
        self.status = status
        self.threads = {}
        self.unrecognized = []
        self.state_lock = threading.Lock()     # the dicts, for snapshots
        self.pass_lock = threading.Lock()      # one pass at a time: the final pass waits for a stuck one
        self.caught_up_at = None
        self.dirty = True
        self.failures = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.loop)
        self.thread.daemon = True

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def loop(self):
        while not self.stop_event.is_set():
            try:
                self.pass_once()
            except Exception as exc:          # the watcher must not die silently
                self.failures += 1
                if self.failures <= 3:
                    log("replication pass failed: %s: %s" % (type(exc).__name__, exc))
            self.stop_event.wait(REPLICATION_INTERVAL)

    def discover(self):
        found = {}
        for dirpath, _, filenames in os.walk(self.sessions):
            for name in filenames:
                if not (name.startswith("rollout-") and name.endswith(".jsonl")):
                    continue
                relative = os.path.relpath(os.path.join(dirpath, name), self.sessions)
                match = ROLLOUT_NAME.match(name)
                if match is None:
                    if relative not in self.unrecognized:
                        self.unrecognized.append(relative)
                        self.dirty = True
                    continue
                found[match.group(1)] = relative
        return found

    def pass_once(self):
        with self.pass_lock:
            for tid, relative in sorted(self.discover().items()):
                self.follow(tid, relative)
            if self.lag_bytes(self.snapshot()) == 0:
                self.caught_up_at = time.monotonic()
            if self.dirty:
                self.dirty = False
                self.write_state()
            self.status.note(replica=self.view())

    def follow(self, tid, relative):
        with self.state_lock:
            entry = self.threads.get(tid)
            if entry is None:
                entry = {"relative_path": relative, "local_bytes": 0, "replicated_bytes": 0,
                         "replica_path": os.path.join(self.rollouts, tid + ".jsonl")}
                try:
                    entry["replicated_bytes"] = os.path.getsize(entry["replica_path"])   # a replica placed before this run
                except OSError:
                    pass
                self.threads[tid] = entry
                self.dirty = True
        if entry.get("anomaly"):
            return
        local_path = os.path.join(self.sessions, relative)
        try:
            local_bytes = os.path.getsize(local_path)
        except OSError:
            return
        if local_bytes != entry["local_bytes"]:
            entry["local_bytes"] = local_bytes
            self.dirty = True
        if local_bytes < entry["replicated_bytes"]:
            self.anomaly(tid, entry, local_bytes)
        elif local_bytes > entry["replicated_bytes"] or entry.get("error"):
            try:
                self.append(tid, entry, local_path, local_bytes)
                if entry.pop("error", None) is not None:
                    log("replication of %s resumed at %d bytes" % (tid, entry["replicated_bytes"]))
            except OSError as exc:
                text = "%s: %s" % (type(exc).__name__, exc)
                if entry.get("error") != text:
                    entry["error"] = text
                    log("replication of %s stalled: %s" % (tid, text))
            self.dirty = True

    def anomaly(self, tid, entry, local_bytes):
        entry["anomaly"] = "shrunk"
        entry["anomaly_at"] = time.time()
        self.dirty = True
        log("rollout of %s shrank to %d bytes below its %d-byte replica; replication of this thread stops" % (tid, local_bytes, entry["replicated_bytes"]))

    def append(self, tid, entry, local_path, local_bytes):
        fd = os.open(entry["replica_path"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            size = os.fstat(fd).st_size
            if size > local_bytes:
                entry["replicated_bytes"] = size
                self.anomaly(tid, entry, local_bytes)
                return
            with open(local_path, "rb") as local:
                local.seek(size)
                while size < local_bytes:
                    chunk = local.read(min(CHUNK, local_bytes - size))
                    if not chunk:
                        break
                    view = memoryview(chunk)
                    while len(view):
                        view = view[os.write(fd, view):]
                    size += len(chunk)
            os.fsync(fd)
            entry["replicated_bytes"] = os.fstat(fd).st_size
        finally:
            os.close(fd)

    def snapshot(self):
        # dict() is one C-level copy per entry: follow() adds keys to entries
        # outside the lock, which a Python-level deep copy could trip over.
        with self.state_lock:
            return dict((tid, dict(entry)) for tid, entry in self.threads.items())

    @staticmethod
    def lag_bytes(threads):
        return sum(e["local_bytes"] - e["replicated_bytes"] for e in threads.values() if not e.get("anomaly"))

    def write_state(self):
        write_json(self.state_path, {"format": 1, "sessions": self.sessions, "rollouts": self.rollouts,
                                     "threads": self.snapshot(), "unrecognized": list(self.unrecognized), "updated": time.time()})

    def view(self):
        threads = self.snapshot()
        age = None if self.caught_up_at is None else round(time.monotonic() - self.caught_up_at, 1)
        return {
            "age_seconds": age,
            "lag_bytes": self.lag_bytes(threads),
            "threads": dict((tid, {"local_bytes": e["local_bytes"], "replicated_bytes": e["replicated_bytes"],
                                   "relative_path": e["relative_path"]}) for tid, e in threads.items()),
            "anomalies": dict((tid, e["anomaly"]) for tid, e in threads.items() if e.get("anomaly")),
            "errors": dict((tid, e["error"]) for tid, e in threads.items() if e.get("error")),
            "unrecognized": list(self.unrecognized),
        }


# ----------------------------------------------------------------- capture

def backup_method():
    if hasattr(sqlite3.Connection, "backup"):
        return "connection"
    if shutil.which("sqlite3"):
        return "cli"
    raise CaptureFailed("no backup method: this python has no sqlite3 backup API and no sqlite3 CLI is installed")


def overdue(deadline):
    return time.monotonic() > deadline


def backup_with_connection(source, target, deadline):
    """The online backup API in-process: the source read-only first, then
    read-write when a WAL source refuses a read-only open; bounded by the
    deadline through the progress callback."""
    src = None
    try:
        try:
            src = sqlite3.connect("file:%s?mode=ro" % source, uri=True, timeout=5)
            src.execute("select count(*) from sqlite_master").fetchone()
        except sqlite3.Error:
            if src is not None:
                src.close()
            src = sqlite3.connect(source, timeout=5)
        dst = sqlite3.connect(target)
        try:
            os.chmod(target, 0o600)

            def progress(status, remaining, total):
                if overdue(deadline):
                    raise CaptureFailed("backup of %s overdue" % os.path.basename(source))
            src.backup(dst, pages=64, progress=progress, sleep=0.005)
        finally:
            dst.close()
    finally:
        if src is not None:
            src.close()


def backup_with_cli(source, target, deadline):
    """``sqlite3 <src> ".backup <dst>"`` for interpreters without the API (3.6)."""
    cli = shutil.which("sqlite3")
    attempt = 0
    while True:
        attempt += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CaptureFailed("backup of %s overdue" % os.path.basename(source))
        try:
            proc = subprocess.run([cli, "-bail", "-cmd", ".timeout 5000", source, ".backup " + target],
                                  stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=remaining)
        except subprocess.TimeoutExpired:
            raise CaptureFailed("sqlite3 .backup of %s timed out" % os.path.basename(source))
        error = proc.stderr.decode("utf-8", "replace").strip()
        if proc.returncode == 0 and not error:
            os.chmod(target, 0o600)
            return
        if attempt >= 3:
            raise CaptureFailed("sqlite3 .backup of %s failed (%d): %s" % (os.path.basename(source), proc.returncode, error[:200]))
        try:
            os.unlink(target)
        except OSError:
            pass
        time.sleep(0.2)


def connect_copy(path, deadline, readonly=True):
    """A connection to a copied store whose statements abort once the capture is overdue."""
    if readonly:
        conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True, timeout=5)
    else:
        conn = sqlite3.connect(path, timeout=5)
    conn.set_progress_handler(lambda: 1 if overdue(deadline) else 0, 10000)
    return conn


def backup_store(name, source, target, method, deadline):
    """One store into the local staging directory: backup, journal_mode=DELETE,
    quick_check, no sidecar left, then size and hash of the closed file."""
    tmp = target + ".incomplete"
    for stale in (tmp, tmp + "-wal", tmp + "-shm", tmp + "-journal"):
        if os.path.exists(stale):
            os.unlink(stale)
    if overdue(deadline):
        raise CaptureFailed("capture overdue before %s" % name)
    try:
        if method == "connection":
            backup_with_connection(source, tmp, deadline)
        else:
            backup_with_cli(source, tmp, deadline)
        conn = connect_copy(tmp, deadline, readonly=False)
        try:
            mode = conn.execute("pragma journal_mode=delete").fetchone()[0]
            rows = conn.execute("pragma quick_check").fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise CaptureFailed("%s: %s: %s" % (name, type(exc).__name__, exc))
    check = rows[0][0] if len(rows) == 1 else "; ".join(str(row[0]) for row in rows)[:300]
    if mode != "delete":
        raise CaptureFailed("%s: journal_mode is %s after the switch to delete" % (name, mode))
    if check != "ok":
        raise CaptureFailed("%s: quick_check: %s" % (name, check))
    for sidecar in ("-wal", "-shm", "-journal"):
        if os.path.exists(tmp + sidecar):
            raise CaptureFailed("%s: a %s sidecar remains after close" % (name, sidecar))
    os.chmod(tmp, 0o600)
    os.rename(tmp, target)
    digest, size = sha256_file(target)
    return {"sha256": digest, "bytes": size, "quick_check": check, "captured_at": time.time()}


def version_key(text):
    return tuple(int(part) for part in re.findall(r"\d+", text or ""))


def relative_rollout(path, homes):
    """A stored rollout path relative to ``<codex_home>/sessions``; refused elsewhere."""
    for home in homes:
        prefix = os.path.join(os.path.normpath(home), "sessions") + "/"
        if path.startswith(prefix):
            return path[len(prefix):]
    raise CaptureFailed("rollout path %s is not under %s" % (path, " or ".join(os.path.join(os.path.normpath(h), "sessions") for h in homes)))


class Checkpointer(object):
    """Single-flight database generations.

    A scheduler thread fires the triggers (initial after placement, timer,
    ``checkpoint.request``) and marks an overdue capture stuck from outside;
    each capture runs in a thread of its own under a nonblocking flock on
    ``checkpoint.lock``, so a capture blocked on the durable filesystem never
    holds up Codex, the placement monitor or the replicator.  Triggers that
    arrive while one is in flight are dropped and counted, never queued.
    """

    def __init__(self, run_dir, db_dir, sessions, durable_run, replicator, status, meta, minutes, timeout, placed):
        self.db_dir = db_dir
        self.sessions = sessions
        self.durable_run = durable_run
        self.snapshots = os.path.join(run_dir, "snapshots")
        self.generations = os.path.join(durable_run, "generations")
        self.request_path = os.path.join(run_dir, "checkpoint.request")
        self.lock_path = os.path.join(run_dir, "checkpoint.lock")
        self.replicator = replicator
        self.status = status
        self.meta = meta
        self.interval = minutes * 60.0
        self.timeout = timeout
        self.grace = max(2.0, 0.1 * timeout)    # a capture honouring its own deadline is not stuck
        self.placed = placed
        self.seq = self.highest_seq()
        self.seen = set()
        self.count = 0
        self.dropped = 0
        self.last_outcome = None
        self.reason = None
        self.last_generation = None
        self.inflight = None
        self.inflight_lock = threading.Lock()
        self.last_started = time.monotonic()
        self.errors = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.loop)
        self.thread.daemon = True

    def highest_seq(self):
        highest = 0
        for directory in (self.snapshots, self.generations):
            try:
                names = os.listdir(directory)
            except OSError:
                continue
            for name in names:
                match = re.match(r"^(\d{6})-[0-9a-f]{8}$", name)
                if match:
                    highest = max(highest, int(match.group(1)))
        return highest

    def start(self):
        self.thread.start()

    # -- scheduling

    def loop(self):
        initial_done = False
        while not self.stop_event.is_set():
            try:
                trigger = None
                if os.path.exists(self.request_path):
                    try:
                        os.unlink(self.request_path)
                    except OSError:
                        pass
                    trigger = "request"
                elif not initial_done and self.placed.is_set():
                    initial_done = True
                    trigger = "initial"
                elif time.monotonic() - self.last_started >= self.interval:
                    trigger = "interval"
                if trigger is not None:
                    self.begin(trigger, "warm")
                else:
                    self.watch_overdue()
            except Exception as exc:
                self.errors += 1
                if self.errors <= 3:
                    log("checkpoint scheduler error: %s: %s" % (type(exc).__name__, exc))
            self.stop_event.wait(1.0)

    def watch_overdue(self):
        with self.inflight_lock:
            flight = self.inflight
        if flight is not None and not flight["stuck"] and overdue(flight["deadline"] + self.grace):
            flight["stuck"] = True
            log("capture %s (%s) still running %ds past its %ds budget; marked stuck, no capture starts until it ends" % (
                flight["generation"], flight["trigger"], self.grace, self.timeout))
            self.status.note(checkpoint=self.view("stuck"))

    def begin(self, trigger, cls, final_pass=False):
        with self.inflight_lock:
            self.last_started = time.monotonic()
            if self.inflight is not None:
                self.dropped += 1
                blocker = self.inflight["generation"]
                flight = None
            else:
                self.count += 1
                self.seq += 1
                generation = "%06d-%s" % (self.seq, uuid.uuid4().hex[:8])
                deadline = time.monotonic() + self.timeout
                worker = threading.Thread(target=self.capture, args=(generation, trigger, cls, deadline, final_pass))
                worker.daemon = True
                flight = {"thread": worker, "deadline": deadline, "generation": generation, "trigger": trigger,
                          "class": cls, "stuck": False, "started": time.time()}
                self.inflight = flight
        if flight is None:
            log("checkpoint trigger %s dropped: capture %s is in flight" % (trigger, blocker))
            self.status.note(checkpoint=self.view(None))
            return None
        self.status.note(checkpoint=self.view("capturing"))
        flight["thread"].start()
        return flight["thread"]

    def view(self, state):
        with self.inflight_lock:
            flight = self.inflight
            if state is None:
                if flight is not None:
                    state = "stuck" if flight["stuck"] else "capturing"
                elif self.last_outcome in ("failed", "not_durable"):
                    state = "failed"
                else:
                    state = "idle"
            out = {"state": state, "last_outcome": self.last_outcome, "count": self.count, "dropped": self.dropped,
                   "interval_seconds": self.interval, "timeout_seconds": self.timeout}
            if self.reason:
                out["reason"] = self.reason
            if flight is not None:
                out["in_flight"] = {"generation": flight["generation"], "trigger": flight["trigger"],
                                    "class": flight["class"], "started": flight["started"]}
        return out

    def describe_last(self):
        return self.last_generation["name"] if self.last_generation else "none"

    # -- one capture

    def capture(self, generation, trigger, cls, deadline, final_pass):
        outcome, reason, published = "failed", None, None
        started = time.monotonic()
        lock = None
        try:
            lock = open(self.lock_path, "w")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self.dropped += 1
                outcome, reason = "dropped", "checkpoint.lock is held"
                return
            if final_pass:
                self.replicator.pass_once()
            staging = os.path.join(self.snapshots, generation)
            manifest = self.bundle(generation, trigger, cls, deadline, staging)
            published = self.publish(generation, manifest, staging, deadline)
            outcome = "published"
            self.prune_local()
        except CaptureSkipped as exc:
            outcome, reason = "skipped", str(exc)
        except NotDurable as exc:
            outcome, reason = "not_durable", str(exc)
        except CaptureFailed as exc:
            outcome, reason = "failed", str(exc)
        except Exception as exc:
            outcome, reason = "failed", "%s: %s" % (type(exc).__name__, exc)
        finally:
            if lock is not None:
                lock.close()
            self.last_outcome, self.reason = outcome, reason
            with self.inflight_lock:
                flight, self.inflight = self.inflight, None
            fields = {}
            if published is not None:
                self.last_generation = published
                self.status.durable_at = time.monotonic()
                fields["last_generation"] = published
            fields["checkpoint"] = self.view(None)
            self.status.note(**fields)
            self.report(generation, trigger, cls, outcome, reason, time.monotonic() - started, flight)

    def report(self, generation, trigger, cls, outcome, reason, elapsed, flight):
        prefix = "stuck " if flight is not None and flight["stuck"] else ""
        if outcome == "published":
            log("%sgeneration %s published (%s, %s, %.1fs)" % (prefix, generation, cls, trigger, elapsed))
        elif outcome == "not_durable":
            log("%sgeneration %s is not durable (%s, %s, %.1fs): %s; last durable generation: %s" % (prefix, generation, cls, trigger, elapsed, reason, self.describe_last()))
        elif outcome == "skipped":
            log("%scapture %s skipped (%s, %s): %s" % (prefix, generation, cls, trigger, reason))
        else:
            log("%scapture %s %s (%s, %s, %.1fs): %s; last durable generation: %s" % (prefix, generation, outcome, cls, trigger, elapsed, reason, self.describe_last()))

    def bundle(self, generation, trigger, cls, deadline, staging):
        names = sorted(n for n in os.listdir(self.db_dir) if n.endswith(".sqlite"))
        if not names:
            raise CaptureSkipped("nothing to capture: no store under %s" % self.db_dir)
        unclassified = [n for n in names if n not in REQUIRED_STORES and n not in OPTIONAL_STORES]
        if unclassified:
            raise CaptureFailed("unclassified store under db: %s" % ", ".join(unclassified))
        self.seen.update(n for n in REQUIRED_STORES if n in names)
        gone = sorted(n for n in self.seen if n not in names)
        if gone:
            raise CaptureFailed("required store disappeared: %s" % ", ".join(gone))
        method = backup_method()
        absent = sorted(n for n in REQUIRED_STORES + OPTIONAL_STORES if n not in names)
        if os.path.isdir(staging):
            shutil.rmtree(staging)
        mkdir_private(staging)
        mkdir_private(os.path.join(staging, "db"))
        try:
            databases = {}
            for name in names:
                databases[name] = backup_store(name, os.path.join(self.db_dir, name), os.path.join(staging, "db", name), method, deadline)
            threads, version, notes = self.inspect(os.path.join(staging, "db"), names, deadline)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        captured = time.time()
        manifest = {
            "format": 1, "scope": self.meta["scope"], "run": self.meta["run"], "generation": generation,
            "parent": self.meta["parent"], "class": cls, "allocation": self.meta["allocation"],
            "codex_home": self.meta["codex_home"], "codex_home_real": self.meta["codex_home_real"],
            "codex_version": version, "captured_at": captured, "captured_at_iso": utc_iso(captured),
            "backup_method": method, "databases": databases, "absent": absent, "threads": threads, "trigger": trigger,
        }
        if notes:
            manifest["notes"] = notes
        lagging = self.wait_for_replicas(threads, deadline)
        if lagging:
            manifest["not_durable"] = lagging
            write_json(os.path.join(staging, "manifest.json"), manifest)
            first = sorted(lagging.items())[0]
            raise NotDurable("%d thread(s) not replicated to their projection offset in %ds: %s has %d replicated bytes, the projection needs %d" % (
                len(lagging), self.timeout, first[0], first[1]["replica_bytes"], first[1]["next_rollout_byte_offset"]))
        write_json(os.path.join(staging, "manifest.json"), manifest)
        return manifest

    def inspect(self, db_copy, names, deadline):
        """Threads from the copies: projection offsets, rollout paths, the Codex version."""
        threads, paths, notes, version = {}, {}, [], None
        if "state_5.sqlite" in names:
            conn = connect_copy(os.path.join(db_copy, "state_5.sqlite"), deadline)
            try:
                try:
                    rows = conn.execute("select id, rollout_path, cli_version from threads").fetchall()
                except sqlite3.OperationalError as exc:
                    notes.append("state_5.sqlite: %s" % exc)
                    rows = []
            finally:
                conn.close()
            for tid, path, cli in rows:
                paths[tid] = path
                if cli and (version is None or version_key(cli) > version_key(version)):
                    version = cli
        if "thread_history_1.sqlite" in names:
            conn = connect_copy(os.path.join(db_copy, "thread_history_1.sqlite"), deadline)
            try:
                try:
                    rows = conn.execute("select thread_id, next_rollout_byte_offset, next_rollout_ordinal from thread_history_projection_state").fetchall()
                except sqlite3.OperationalError as exc:
                    notes.append("thread_history_1.sqlite: %s" % exc)
                    rows = []
            finally:
                conn.close()
            for tid, offset, ordinal in rows:
                threads[tid] = {"next_rollout_byte_offset": int(offset), "next_rollout_ordinal": int(ordinal)}
        for tid in paths:
            threads.setdefault(tid, {"next_rollout_byte_offset": 0, "next_rollout_ordinal": 0})
        replicas = self.replicator.snapshot()
        homes = (self.meta["codex_home"], self.meta["codex_home_real"])
        for tid, entry in sorted(threads.items()):
            replica = replicas.get(tid)
            if tid in paths:
                relative = relative_rollout(paths[tid], homes)
            elif replica is not None:
                relative = replica["relative_path"]
            else:
                raise CaptureFailed("thread %s has a history projection row but no rollout path" % tid)
            if replica is not None and replica["relative_path"] != relative:
                raise CaptureFailed("thread %s: rollout_path %s is not the replicated file %s" % (tid, relative, replica["relative_path"]))
            if replica is not None and replica.get("anomaly"):
                raise CaptureFailed("thread %s: its rollout %s, so its replica is not faithful" % (tid, replica["anomaly"]))
            entry["relative_path"] = relative
        return threads, version, notes

    def wait_for_replicas(self, threads, deadline):
        """The ordering invariant: every thread's replica at or past the offset
        its copied projection records, within the budget.  Records the replica
        length and the local rollout's hash up to it; returns the lagging ones."""
        while True:
            replicas = self.replicator.snapshot()
            lagging = {}
            for tid, entry in threads.items():
                have = replicas[tid]["replicated_bytes"] if tid in replicas else 0
                if have < entry["next_rollout_byte_offset"]:
                    lagging[tid] = {"replica_bytes": have, "next_rollout_byte_offset": entry["next_rollout_byte_offset"]}
            if not lagging or overdue(deadline):
                break
            time.sleep(0.2)
        for tid, entry in threads.items():
            have = replicas[tid]["replicated_bytes"] if tid in replicas else 0
            entry["replica_bytes"] = have
            if have == 0:
                entry["rollout_sha256_prefix"] = hashlib.sha256(b"").hexdigest()
                continue
            local = os.path.join(self.sessions, entry["relative_path"])
            try:
                digest, hashed = sha256_file(local, have)
            except OSError as exc:
                raise CaptureFailed("thread %s: cannot hash its local rollout: %s" % (tid, exc))
            if hashed != have:
                raise CaptureFailed("thread %s: local rollout has %d bytes, shorter than its %d-byte replica" % (tid, hashed, have))
            entry["rollout_sha256_prefix"] = digest
        return lagging

    def publish(self, generation, manifest, staging, deadline):
        """``.incoming-<uuid>`` with the copies verified by hash, the manifest
        last, everything synced, then one rename into ``generations/``."""
        incoming = os.path.join(self.durable_run, ".incoming-" + uuid.uuid4().hex)
        os.mkdir(incoming, 0o700)
        os.chmod(incoming, 0o700)
        db_copy = os.path.join(incoming, "db")
        os.mkdir(db_copy, 0o700)
        os.chmod(db_copy, 0o700)
        for name, record in sorted(manifest["databases"].items()):
            if overdue(deadline):
                raise CaptureFailed("publication overdue before %s" % name)
            target = os.path.join(db_copy, name)
            copy_file(os.path.join(staging, "db", name), target)
            if sha256_file(target)[0] != record["sha256"]:
                raise CaptureFailed("%s: the published copy's hash differs from the local copy" % name)
        fsync_path(db_copy)
        manifest_path = os.path.join(incoming, "manifest.json")
        write_json(manifest_path, manifest)
        fsync_path(manifest_path)
        fsync_path(incoming)
        manifest_sha = sha256_file(manifest_path)[0]
        if overdue(deadline):
            raise CaptureFailed("publication overdue before the rename")
        final = os.path.join(self.generations, generation)
        os.rename(incoming, final)
        fsync_path(self.generations)
        published = time.time()
        return {"name": generation, "class": manifest["class"], "published_at": published, "published_at_iso": utc_iso(published),
                "manifest_sha256": manifest_sha, "path": final, "trigger": manifest["trigger"]}

    def prune_local(self):
        try:
            names = sorted(n for n in os.listdir(self.snapshots) if re.match(r"^\d{6}-[0-9a-f]{8}$", n))
        except OSError:
            return
        for name in names[:-LOCAL_BUNDLES_KEPT]:
            shutil.rmtree(os.path.join(self.snapshots, name), ignore_errors=True)

    # -- after Codex exits

    def final(self, cls):
        """One last replication pass and the sealed or crash capture, bounded:
        a capture still in flight or one that overruns never delays the exit."""
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(2.0)
        self.replicator.stop()
        deadline = time.monotonic() + self.timeout
        while True:
            with self.inflight_lock:
                flight = self.inflight
            if flight is None or overdue(deadline):
                break
            time.sleep(0.1)
        if flight is not None:
            log("final %s capture skipped: capture %s is still in flight; last durable generation: %s" % (cls, flight["generation"], self.describe_last()))
            return
        worker = self.begin("final", cls, final_pass=True)
        if worker is None:
            return
        worker.join(self.timeout + 2.0)
        if worker.is_alive():
            log("final %s capture still running after %ds; exiting without it; last durable generation: %s" % (cls, self.timeout, self.describe_last()))
            self.status.note(checkpoint=self.view("stuck"))


# ----------------------------------------------------------------- the run

def run(args):
    status = None
    try:
        if args.restore:
            raise Refused("restore is not implemented yet (--restore %s is the second half of Slice 2)" % args.restore)
        if shutil.which("bwrap") is None:
            raise Refused("bwrap is not installed on this node; managed placement is unavailable and SuCoder will not fall back to shared SQLite")
        local_root = resolve_local_root()
        codex_home, codex_home_real = resolve_codex_home(args.codex_home)
        check_conflicts(args.command, args.placeholder, codex_home_real)
        durable_base = resolve_durable_base(args.durable_root, local_root)
        scope_dir = os.path.join(local_root, "state", "codex", args.scope)
        mkdir_private(os.path.join(local_root, "state"))
        mkdir_private(os.path.join(local_root, "state", "codex"))
        mkdir_private(scope_dir)
        lock_path = os.path.join(scope_dir, "startup.lock")
        lock = open(lock_path, "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Refused("another managed Codex server for this scope is running in this allocation (%s is held)" % lock_path)
        run_id = uuid.uuid4().hex
        local = os.path.join(scope_dir, run_id)
        db_dir = os.path.join(local, "db")
        home_local = os.path.join(local, "home")
        for path in (local, db_dir, home_local, os.path.join(local, "snapshots")):
            mkdir_private(path)
        status = Status(os.path.join(local, "status.json"))
        LOG_FILE["path"] = os.path.join(local, "runner.log")
        binds, skipped = prepare_binds(codex_home_real, home_local)
        command = substitute(args.command, args.placeholder, db_dir)
        argv = bwrap_argv(binds, command)
        env = dict(os.environ)
        env["CODEX_HOME"] = codex_home
        env["SUCODER_CODEX_STATE_RUN"] = local
        durable = os.path.join(durable_base, "codex", args.scope)
        durable_run = os.path.join(durable, "runs", run_id)
        try:
            ensure_private_tree(durable_run)
            ensure_private_dir(os.path.join(durable_run, "rollouts"))
            ensure_private_dir(os.path.join(durable_run, "generations"))
        except OSError as exc:
            raise Refused("cannot create the durable run directory %s: %s" % (durable_run, exc))
        created = time.time()
        meta = {
            "format": 1, "scope": args.scope, "run": run_id, "parent": None,
            "allocation": {"job": os.environ.get("SLURM_JOB_ID"), "node": os.uname()[1]},
            "codex_home": codex_home, "codex_home_real": codex_home_real,
            "created": created, "created_iso": utc_iso(created),
        }
        try:
            write_json(os.path.join(durable_run, "run.json"), meta)
            fsync_path(os.path.join(durable_run, "run.json"))
        except OSError as exc:
            raise Refused("cannot write %s: %s" % (os.path.join(durable_run, "run.json"), exc))
        owner = {
            "scope": args.scope, "run": run_id, "local": local, "db": db_dir, "codex_home": codex_home,
            "codex_home_real": codex_home_real, "job": meta["allocation"]["job"], "node": meta["allocation"]["node"],
            "durable_root": durable_base, "durable": durable_run,
            "bound": [{"local": l, "target": t} for l, t in binds], "skipped": skipped,
            "runner_pid": os.getpid(), "started": time.time(), "command": elide(command),
        }
        write_json(os.path.join(local, "owner.json"), owner)
        with open(os.path.join(scope_dir, "latest-run"), "w") as handle:
            handle.write(run_id + "\n")
        status.update("starting", scope=args.scope, run=run_id, db=db_dir, durable=durable_run,
                      bound=[t for _, t in binds], skipped=skipped)
        log("run %s: databases at %s; bound %s; skipped %s" % (run_id[:8], db_dir, ", ".join(os.path.basename(t) for _, t in binds) or "nothing", ", ".join(skipped) or "nothing"))
        log("the managed Codex history for this mirror starts empty and is separate from the shared history in %s; rollout replicas and database generations go to %s" % (codex_home, durable))
    except Refused as exc:
        log("refusing to launch: %s" % exc)
        if status is not None:
            status.update("refused", reason=str(exc))
        return EXIT_REFUSED

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # From here Codex may own the terminal (the TUI draws on it); keep the
    # runner's own lines in runner.log and speak on stderr again only after
    # the child has exited.
    LOG_FILE["quiet"] = sys.stderr.isatty()
    child = subprocess.Popen(argv, env=env)
    owner["child_pid"] = child.pid
    write_json(os.path.join(local, "owner.json"), owner)
    status.update("launched", child_pid=child.pid)

    def forward(signum, frame):
        # bwrap stays alive as the parent of the sandboxed command and does
        # not relay signals; delivered to bwrap itself, SIGTERM would end it
        # and --die-with-parent would then SIGKILL Codex before it could
        # shut down.  Signal the command instead, and bwrap propagates its
        # exit status.  Fall back to bwrap only when no child is visible.
        targets = [pid for pid in descendants(child.pid) if parent_of(pid) == child.pid] or [child.pid]
        for pid in targets:
            try:
                os.kill(pid, signum)
            except OSError:
                pass
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, forward)

    result = {}
    placed = threading.Event()
    watcher = threading.Thread(target=monitor, args=(child, db_dir, local_root, status, args.placement_timeout, result),
                               kwargs={"placed_event": placed})
    watcher.daemon = True
    watcher.start()
    replicator = Replicator(os.path.join(home_local, "sessions"), os.path.join(durable_run, "rollouts"),
                            os.path.join(local, "replica.json"), status)
    replicator.start()
    checkpointer = Checkpointer(local, db_dir, os.path.join(home_local, "sessions"), durable_run, replicator, status, meta,
                                args.checkpoint_minutes, args.checkpoint_timeout, placed)
    checkpointer.start()
    while True:
        try:
            code = child.wait()
            break
        except InterruptedError:
            continue
    watcher.join(timeout=5)
    LOG_FILE["quiet"] = False
    if code < 0:
        code = 128 - code
    misplaced = result.get("outcome") == "misplaced"
    if misplaced:
        status.update("stopped_misplaced", exit=code)
    else:
        status.update("exited", exit=code, outcome=result.get("outcome"))
        log("Codex exited with %d (%s)" % (code, result.get("outcome", "unknown")))
    cls = "sealed" if code == 0 and not misplaced else "crash"
    if cls == "sealed":
        holders = holders_of(db_dir)
        if holders:
            cls = "crash"
            log("pids %s still hold descriptors under %s after Codex exited; the final generation is a crash candidate, not sealed" % (
                ", ".join(str(pid) for pid in holders[:4]), db_dir))
    checkpointer.final(cls)
    if misplaced:
        return EXIT_MISPLACED
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description="SuCoder managed Codex state runner")
    parser.add_argument("--scope", required=True, help="stable scope digest from the launcher")
    parser.add_argument("--placeholder", default=PLACEHOLDER_DEFAULT)
    parser.add_argument("--codex-home", default=None)
    parser.add_argument("--placement-timeout", type=float, default=90.0)
    parser.add_argument("--durable-root", required=True, metavar="ABS_BASE",
                        help="absolute durable state root; this run's rollout replicas and database generations go under "
                             "ABS_BASE/codex/<scope>/runs/<run>; refused when relative, with unusual characters, or under SUCODER_LOCAL_ROOT")
    parser.add_argument("--checkpoint-minutes", type=float, default=10.0, metavar="N",
                        help="minutes between timer-driven database generations (default 10)")
    parser.add_argument("--checkpoint-timeout", type=float, default=30.0, metavar="S",
                        help="seconds one capture (backup, replica wait, publication) may take before it is marked stuck (default 30)")
    parser.add_argument("--restore", default=None, metavar="RUN/GENERATION",
                        help="continue from a published generation; accepted but not implemented yet (the second half of Slice 2): refuses with exit 70")
    parser.add_argument("--fresh", action="store_true",
                        help="start an empty managed history; today the only behaviour, so this is a no-op")
    parser.add_argument("--unattended", action="store_true",
                        help="no interactive client is attached; accepted and ignored until restore exists")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="the Codex command, after --")
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("no command given after --")
    if not re.match(r"^[0-9a-f]{8,64}$", args.scope):
        parser.error("--scope must be a hex digest")
    if args.checkpoint_minutes <= 0 or args.checkpoint_timeout <= 0:
        parser.error("--checkpoint-minutes and --checkpoint-timeout must be positive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
