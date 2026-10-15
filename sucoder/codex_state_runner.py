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
  reached the greater of the rollout offset the copied history projection
  records and the local rollout's length right after the copies (a thread
  the replicator has not discovered yet waits for its discovery), as an
  atomically renamed ``DURABLE/runs/<run>/generations/<generation>/`` whose
  ``manifest.json`` is written last.  ``status.json`` is best effort and
  never stops replication, capture or reporting.

- restores a published generation (``--restore RUN/GENERATION``): the
  manifest is checked against this launch (format, scope, the same
  ``CODEX_HOME``, class), every replica against the recorded projection
  offset and hash, the stores are copied into ``restore-staging/db/`` and
  verified there (size, sha256, ``quick_check``), the pending-work gate runs
  on the copies, each replica is staged whole at its recorded relative path
  under ``restore-staging/sessions`` and copied byte-identically to the new
  run's ``rollouts/.incoming-<thread-id>.jsonl``, ``run.json`` is written,
  and only then is everything activated by renames (the replicas to their
  final names, the sessions tree under ``home/``, the stores into ``db/``).
  A refusal at any point up to the ownership record (the bind preparation
  included) removes what the attempt created and nothing else; a durable
  side that does not answer within the restore deadline leaves the launch
  ``blocked`` (the abandoned restore thread is fenced: it writes nothing
  more and rolls back if it wakes up; a deadline that passes once the
  activation's local renames have landed does not fence, the launch takes
  the result).  Nothing verifies, nothing is placed and Codex does not
  start;
- reuses this allocation's latest run in place when it still has its
  databases and nobody holds them (recorded pids dead, unknown ownership
  refuses), under the same canonical ``CODEX_HOME`` and the same durable run
  it was created for: same run, same local directory, same durable run,
  history and generation numbering continuing;
  local state wins over ``--restore``, ``--fresh`` starts a new run;
- detects pending work (active goals in ``goals_1``, queued items in
  ``queue_1``) in the stores it is about to hand to Codex and records it; it
  never rewrites a row to hold it.  With ``--unattended`` it refuses instead
  (and accepts sealed generations only), because an active goal drives
  unprompted model requests and a queued item executes once on resume.
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
import stat
import subprocess
import sys
import threading
import time
import traceback
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
CHUNK = 1 << 20


def replication_interval():
    """Seconds between replication passes: 1.0, or the environment's override
    (tests pin the pass timing with it) when it is a finite number in
    (0, 3600]; anything else is the default (``Event.wait`` overflows on an
    infinite or huge wait, and the loop would die outside its handler)."""
    try:
        value = float(os.environ.get("SUCODER_CODEX_STATE_REPLICATION_INTERVAL") or 1.0)
    except ValueError:
        return 1.0
    if value != value or value <= 0 or value > 3600:     # NaN, non-positive, infinite or huge
        return 1.0
    return value


def restore_timeout(checkpoint_timeout):
    """Seconds a restore may spend before the launch is reported blocked:
    max(120, 4 x --checkpoint-timeout), or SUCODER_CODEX_STATE_RESTORE_TIMEOUT
    when it is a positive finite number no greater than a day; either is
    clamped to a day (``Thread.join`` overflows on a huge wait).  Read at
    call time."""
    try:
        value = float(os.environ.get("SUCODER_CODEX_STATE_RESTORE_TIMEOUT") or 0)
    except ValueError:
        value = 0.0
    if not (value > 0 and value <= RESTORE_TIMEOUT_MAX):     # NaN, non-positive, infinite or huge
        value = max(120.0, 4.0 * checkpoint_timeout)
    return min(value, RESTORE_TIMEOUT_MAX)


RESTORE_TIMEOUT_MAX = 86400.0
REPLICATION_INTERVAL = replication_interval()
LOCAL_BUNDLES_KEPT = 3
# ``--restore RUN/GENERATION`` names history under DURABLE/runs, never a path.
RESTORE_SELECTOR = re.compile(r"^([0-9a-f]{32})/([0-9]{6}-[0-9a-f]{8})$")
RUN_ID = re.compile(r"^[0-9a-f]{32}$")
GENERATION_NAME = re.compile(r"^[0-9]{6}-[0-9a-f]{8}$")
# The keys of the record ``Checkpointer.publish`` returns (status.json's last_generation and unsynced_generation).
PUBLISHED_RECORD_KEYS = ("name", "class", "published_at", "manifest_sha256", "path", "trigger")
GENERATION_CLASSES = ("sealed", "warm", "crash")


class Refused(Exception):
    """The launch cannot proceed under managed placement; nothing was started."""


class Blocked(Refused):
    """The durable side did not answer within the restore deadline; nothing was started."""


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


def copy_whole(source, targets, prefix=0):
    """One read of ``source`` written to every target (exclusive creation,
    0600, fsynced).  Returns (sha256 of every byte, sha256 of the first
    ``prefix`` bytes, size), so a replica can be placed locally and copied to
    the new run's durable replica from the same bytes and checked against
    the manifest's prefix hash without a second read of the durable side."""
    fds = []
    digest, head = hashlib.sha256(), hashlib.sha256()
    size = 0
    try:
        for target in targets:
            fds.append(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        with open(source, "rb") as handle:
            while True:
                chunk = handle.read(CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
                if size < prefix:
                    head.update(chunk[:prefix - size])
                size += len(chunk)
                for fd in fds:
                    view = memoryview(chunk)
                    while len(view):
                        view = view[os.write(fd, view):]
        for fd in fds:
            os.fsync(fd)
    finally:
        for fd in fds:
            os.close(fd)
    return digest.hexdigest(), head.hexdigest(), size


def pid_alive(pid):
    """A recorded pid still exists.  A zombie counts, and a recycled pid is
    taken as alive too: refusing is the safe side for a second writer."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if not os.path.isdir("/proc/%d" % pid):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def connect_readonly(path):
    """A SELECT-only connection: read-only when the sidecars allow it (a
    staged copy always does), a plain connection otherwise, as the backup
    takes its source.  No statement here ever writes a row."""
    conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True, timeout=5)
    try:
        conn.execute("select count(*) from sqlite_master").fetchone()
        return conn
    except sqlite3.Error:
        conn.close()
    return sqlite3.connect(path, timeout=5)


def quick_check(path):
    """``pragma quick_check`` of a closed store opened read-only; "ok" or the verdict."""
    try:
        conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True, timeout=5)
        try:
            rows = conn.execute("pragma quick_check").fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return "%s: %s" % (type(exc).__name__, exc)
    if len(rows) == 1:
        return str(rows[0][0])
    return "; ".join(str(row[0]) for row in rows)[:300]


def pending_work(db_dir):
    """Active goals and queued items in the stores under ``db_dir``: detect
    and report only.  This runner never pauses a goal or drops a queued item
    by rewriting a row; that hold is the operator's, in Codex."""
    found = {"active_goals": [], "queued": {}}
    unreadable = []
    goals = os.path.join(db_dir, "goals_1.sqlite")
    if os.path.isfile(goals):
        try:
            rows = select_rows(goals, "select thread_id from thread_goals where status = 'active' order by thread_id")
            found["active_goals"] = [str(row[0]) for row in rows]
        except sqlite3.Error as exc:
            unreadable.append("goals_1.sqlite: %s" % exc)
    queue = os.path.join(db_dir, "queue_1.sqlite")
    if os.path.isfile(queue):
        try:
            rows = select_rows(queue, "select thread_id, count(*) from queued_items group by thread_id order by thread_id")
            found["queued"] = dict((str(tid), int(count)) for tid, count in rows)
        except sqlite3.Error as exc:
            unreadable.append("queue_1.sqlite: %s" % exc)
    if unreadable:
        found["unreadable"] = unreadable
    return found


def select_rows(path, sql):
    conn = connect_readonly(path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def describe_pending(pending):
    parts = []
    if pending["active_goals"]:
        parts.append("an active goal on thread(s) %s" % ", ".join(pending["active_goals"]))
    for tid, count in sorted(pending["queued"].items()):
        parts.append("%d queued item(s) on thread %s" % (count, tid))
    for note in pending.get("unreadable", []):
        parts.append("a store this runner cannot read (%s)" % note)
    return "; ".join(parts) if parts else "none"


def gate_unattended(pending, where):
    """With no client attached, pending work is refused, never held by SQL."""
    if pending["active_goals"] or pending["queued"] or pending.get("unreadable"):
        raise Refused("unattended continuation refused: %s holds %s. An active goal drives unprompted model requests "
                      "from the moment the thread resumes: pause it in Codex (thread/goal/set with status paused) before "
                      "the next attempt. A queued item executes once on resume and has no hold." % (where, describe_pending(pending)))


# ----------------------------------------------------------------- restore

def parse_selector(value):
    match = RESTORE_SELECTOR.fullmatch(value or "")
    if match is None:
        if ".incoming" in (value or ""):
            raise Refused("--restore %s names an incomplete publication (.incoming-*), which is never a candidate; "
                          "a generation is RUN/GENERATION" % value)
        raise Refused("--restore %s is not RUN/GENERATION (32 hex characters, a slash, six digits, a dash, eight hex "
                      "characters); the selector names history under the durable root, never a path" % value)
    return match.group(1), match.group(2)


def selector_paths(durable, selector):
    """(name, source run directory, generation directory) of a parsed selector."""
    run, generation = selector
    source_run = os.path.join(durable, "runs", run)
    return run + "/" + generation, source_run, os.path.join(source_run, "generations", generation)


def load_manifest(gen_dir, name):
    """The manifest parsed from one read of its bytes, with the sha256 of
    those same bytes: a second read could describe a file that changed in
    between."""
    manifest_path = os.path.join(gen_dir, "manifest.json")
    if not os.path.isdir(gen_dir):
        raise Refused("generation %s does not exist under %s" % (name, os.path.dirname(os.path.dirname(gen_dir))))
    if not os.path.isfile(manifest_path):
        raise Refused("generation %s has no manifest.json; it was never completely published" % name)
    try:
        with open(manifest_path, "rb") as handle:
            raw = handle.read()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise Refused("generation %s: manifest.json is unreadable: %s" % (name, exc))
    if not isinstance(manifest, dict):
        raise Refused("generation %s: manifest.json is not an object" % name)
    return manifest, hashlib.sha256(raw).hexdigest()


def require_int(value, what):
    """``value`` when it is a non-negative int; a bool, a float, a numeric
    string or anything else is a TypeError (``int()`` would coerce 0.5 to 0,
    "7" to 7 and True to 1), which check_manifest reports as malformed."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError("%s is %r, not a non-negative integer" % (what, value))
    return value


def check_manifest(manifest, name, run, generation, scope, codex_home, codex_home_real, unattended):
    """The manifest against this launch: reads only, every refusal specific.
    ``threads`` is normalized in place (absent or null becomes {}), so every
    later step reads the same dict.  Every count is required to be an int."""
    try:
        if manifest.get("format") != 1:
            raise Refused("generation %s: manifest format %r is not 1; this runner does not claim compatibility with it" % (name, manifest.get("format")))
        if manifest.get("scope") != scope:
            raise Refused("generation %s belongs to scope %s, not %s; a generation never crosses scopes" % (name, manifest.get("scope"), scope))
        if manifest.get("run") != run or manifest.get("generation") != generation:
            raise Refused("generation %s: its manifest names %s/%s" % (name, manifest.get("run"), manifest.get("generation")))
        recorded_real = manifest.get("codex_home_real") or os.path.realpath(str(manifest.get("codex_home")))
        if recorded_real != codex_home_real:
            raise Refused("generation %s was captured under CODEX_HOME %s (%s), not %s (%s); the stored rollout paths are "
                          "absolute and valid only under the same home" % (name, manifest.get("codex_home"), recorded_real, codex_home, codex_home_real))
        cls = manifest.get("class")
        if cls not in GENERATION_CLASSES:
            raise Refused("generation %s: unknown class %r" % (name, cls))
        if unattended and cls != "sealed":
            raise Refused("unattended continuation accepts sealed generations only; %s is a %s generation, whose goals, "
                          "queue, memories and state rows are consistent per database only" % (name, cls))
        if manifest.get("not_durable"):
            raise Refused("generation %s records threads that were not durable at capture (%s); it is not a candidate" % (
                name, ", ".join(sorted(manifest["not_durable"]))))
        databases = manifest["databases"]
        if not isinstance(databases, dict) or not databases:
            raise Refused("generation %s lists no database" % name)
        for store, record in databases.items():
            if store not in REQUIRED_STORES and store not in OPTIONAL_STORES:
                raise Refused("generation %s lists an unclassified store %s" % (name, store))
            require_int(record["bytes"], "databases[%s].bytes" % store)
            str(record["sha256"])
        threads = manifest.get("threads")
        if threads is None:
            threads = {}
        if not isinstance(threads, dict):
            raise Refused("generation %s: threads is not an object" % name)
        manifest["threads"] = threads
        for tid, entry in threads.items():
            for key in ("next_rollout_byte_offset", "next_rollout_ordinal", "replica_bytes"):
                require_int(entry[key], "threads[%s].%s" % (tid, key))
            str(entry["rollout_sha256_prefix"])
            relative = entry["relative_path"]
            if not isinstance(relative, str):
                raise Refused("generation %s: thread %s records an unusable rollout path %r" % (name, tid, relative))
            match = ROLLOUT_NAME.fullmatch(os.path.basename(relative))
            if os.path.isabs(relative) or ".." in relative.split("/") or not SAFE_PATH.fullmatch(relative) or match is None:
                raise Refused("generation %s: thread %s records an unusable rollout path %r" % (name, tid, relative))
            if match.group(1) != tid:
                raise Refused("generation %s: thread %s: its rollout name %s does not carry the thread id; replicas are keyed by it" % (
                    name, tid, os.path.basename(relative)))
    except (KeyError, TypeError, ValueError) as exc:
        raise Refused("generation %s: manifest.json is malformed (%s: %s)" % (name, type(exc).__name__, exc))


def check_replicas(manifest, name, source_run):
    """Every thread's replica at least as long as the projection offset the
    generation records (equality passes; longer is the rollout-ahead case
    Codex rebuilds) and hashing to the recorded prefix, the empty prefix
    included.  Reads only.  Returns {tid: (replica path, size, exists)}."""
    sources = {}
    for tid, entry in sorted(manifest["threads"].items()):
        replica = os.path.join(source_run, "rollouts", tid + ".jsonl")
        offset = int(entry["next_rollout_byte_offset"])
        hashed_bytes = int(entry["replica_bytes"])
        exists = os.path.isfile(replica)
        size = os.path.getsize(replica) if exists else 0
        if size < offset:
            raise Refused("generation %s: thread %s: its replica %s holds %d bytes but the generation's history projection "
                          "records next_rollout_byte_offset %d; restoring it would skip real turns below that offset and show "
                          "phantom ones (the database-ahead hazard)" % (name, tid, replica, size, offset))
        if hashed_bytes > size:
            raise Refused("generation %s: thread %s: the manifest records %d replicated bytes but its replica %s holds %d" % (
                name, tid, hashed_bytes, replica, size))
        if hashed_bytes:
            digest, hashed = sha256_file(replica, hashed_bytes)
        else:
            digest, hashed = hashlib.sha256(b"").hexdigest(), 0
        if hashed != hashed_bytes or digest != entry["rollout_sha256_prefix"]:
            raise Refused("generation %s: thread %s: the first %d bytes of its replica %s do not hash to the manifest's "
                          "rollout_sha256_prefix; the generation was captured against different rollout bytes" % (name, tid, hashed_bytes, replica))
        sources[tid] = (replica, size, exists)
    return sources


def stage_databases(manifest, name, gen_dir, staging_db):
    """The generation's stores copied into the local staging directory and
    verified there: the inventory reconciled with the manifest (an unlisted
    store refuses), size and sha256 against the manifest, quick_check
    read-only, no sidecar.  Returns the bytes copied."""
    db_source = os.path.join(gen_dir, "db")
    try:
        present = sorted(n for n in os.listdir(db_source) if n.endswith(".sqlite"))
    except OSError as exc:
        raise Refused("generation %s: cannot list %s: %s" % (name, db_source, exc))
    unlisted = [n for n in present if n not in manifest["databases"]]
    if unlisted:
        raise Refused("generation %s: %s holds unlisted store(s) %s that its manifest does not describe; the generation was "
                      "altered after publication" % (name, db_source, ", ".join(unlisted)))
    total = 0
    for store, record in sorted(manifest["databases"].items()):
        source = os.path.join(db_source, store)
        if not os.path.isfile(source):
            raise Refused("generation %s: %s is listed in the manifest but missing from %s" % (name, store, db_source))
        for sidecar in ("-wal", "-shm", "-journal"):
            if os.path.exists(source + sidecar):
                raise Refused("generation %s: %s has a %s sidecar; a published generation never does" % (name, store, sidecar))
        size = os.path.getsize(source)
        if size != int(record["bytes"]):
            raise Refused("generation %s: %s is %d bytes, the manifest records %d; the generation is corrupt or was altered after publication" % (
                name, store, size, int(record["bytes"])))
        target = os.path.join(staging_db, store)
        try:
            copy_file(source, target)
        except OSError as exc:
            raise Refused("generation %s: cannot copy %s: %s" % (name, store, exc))
        digest, copied = sha256_file(target)
        if copied != size or digest != record["sha256"]:
            raise Refused("generation %s: %s does not hash to the manifest's sha256 %s; the generation is corrupt or was altered after publication" % (
                name, store, record["sha256"]))
        verdict = quick_check(target)
        if verdict != "ok":
            raise Refused("generation %s: %s fails quick_check: %s" % (name, store, verdict))
        total += size
    return total


def stage_replicas(manifest, name, sources, staging_sessions, rollouts, attempt):
    """Each replica, whole, from one read, to its recorded relative path under
    the staging sessions tree (0700 parents, 0600 file) and to the new run's
    rollouts directory as ``.incoming-<thread id>.jsonl``; both copies hashed
    back.  Nothing is at its final name yet.  Returns the bytes staged."""
    total = 0
    for tid, entry in sorted(manifest["threads"].items()):
        source, size, exists = sources[tid]
        if not exists:
            log("thread %s has no replica to place (the generation recorded none); Codex will find no rollout for it" % tid)
            continue
        local = os.path.join(staging_sessions, entry["relative_path"])
        incoming = os.path.join(rollouts, ".incoming-" + tid + ".jsonl")
        ensure_private_tree(os.path.dirname(local), staging_sessions)
        attempt.check_fence()           # before each durable write
        attempt.incoming.append(incoming)
        try:
            digest, head, copied = copy_whole(source, [local, incoming], int(entry["replica_bytes"]))
        except OSError as exc:
            raise Refused("generation %s: cannot stage the rollout of thread %s: %s" % (name, tid, exc))
        if copied < int(entry["next_rollout_byte_offset"]) or head != entry["rollout_sha256_prefix"]:
            raise Refused("generation %s: the replica of thread %s changed while it was being copied" % (name, tid))
        for path in (local, incoming):
            check, length = sha256_file(path)
            if length != copied or check != digest:
                raise Refused("generation %s: the staged copy %s does not hash to its source" % (name, path))
        total += copied
    return total


def activate_restore(name, attempt, staging, db_dir, rollouts):
    """The last step, after run.json: each staged replica renamed to its final
    name and ``rollouts/`` synced, then, under the attempt's lock, the staged
    sessions tree renamed under ``home/`` and the staged stores renamed into
    ``db/``.  Every rename stays within one directory or one local
    filesystem.  ``restore_bounded`` fences under the same lock, so the fence
    and the local renames exclude each other: a fence that lands first
    refuses here and nothing is installed; a deadline that passes once
    ``db_installed`` is set does not fence, the launch takes the result.
    Only local renames run under the lock; the durable ones above it are
    fenced one by one, so a stalled durable side never holds the lock."""
    attempt.check_fence()
    while attempt.incoming:
        incoming = attempt.incoming[0]
        final = os.path.join(rollouts, os.path.basename(incoming)[len(".incoming-"):])
        if os.path.lexists(final):
            raise Refused("generation %s: %s already exists; a restore never overwrites a replica of the new run" % (name, final))
        attempt.check_fence()
        os.rename(incoming, final)
        attempt.placed.append(final)
        attempt.incoming.pop(0)
    if attempt.placed:
        fsync_path(rollouts)        # the final names, not only the bytes
    staged_sessions = os.path.join(staging, "sessions")
    with attempt.lock:
        attempt.check_fence()
        if os.path.isdir(staged_sessions):
            os.rename(staged_sessions, attempt.sessions)
            attempt.sessions_installed = True
        os.rename(os.path.join(staging, "db"), db_dir)
        attempt.db_installed = True
    try:
        os.rmdir(staging)
    except OSError:
        pass


class RestoreAttempt(object):
    """What one launch attempt has created, so a refusal after its first
    durable write removes exactly that and nothing else: the staging tree,
    the ``.incoming-*`` replicas (or the replicas renamed from them), the
    durable run directory when this attempt created it, the two run.json
    copies, the sessions tree it renamed under ``home/`` and the ``db/`` it
    installed (from its staging, or created empty by a fresh launch).  The
    transaction lasts until owner.json and latest-run are written: the
    fresh path records its creations here too.  ``fenced`` is set by the
    launch, under ``lock``, when it reports the restore blocked: from then on
    every durable write of the abandoned restore thread refuses and rolls
    back.  The activation's local renames hold the same lock, so the fence
    either precedes them (nothing is installed) or finds ``db_installed``
    set and is not applied: the launch takes the result instead."""

    def __init__(self, local, home_local, durable_run, label="restore"):
        self.label = label
        self.lock = threading.Lock()
        self.step = "validating the selector"
        self.staging = os.path.join(local, "restore-staging")
        self.local_meta = os.path.join(local, "run.json")
        self.durable_meta = os.path.join(durable_run, "run.json")
        self.sessions = os.path.join(home_local, "sessions")
        self.db_dir = os.path.join(local, "db")
        self.durable_run = durable_run
        self.durable_created = False
        self.meta_written = False
        self.sessions_installed = False
        self.db_installed = False
        self.fenced = False
        self.incoming = []
        self.placed = []

    def check_fence(self):
        if self.fenced:
            raise Refused("fenced: the launch already reported this restore blocked, so its late thread writes nothing more")

    def rollback(self):
        removed, kept = [], []

        def tree(path):
            if os.path.lexists(path):
                shutil.rmtree(path, ignore_errors=True)
                (kept if os.path.lexists(path) else removed).append(path)

        def unlink(path):
            try:
                os.unlink(path)
                removed.append(path)
            except FileNotFoundError:
                pass
            except OSError:
                kept.append(path)

        tree(self.staging)
        if self.sessions_installed:
            tree(self.sessions)
        if self.db_installed:
            tree(self.db_dir)
        if self.meta_written:
            unlink(self.local_meta)
            unlink(self.local_meta + ".tmp")
        if self.durable_created:
            tree(self.durable_run)
        else:
            for path in self.incoming + self.placed:
                unlink(path)
            if self.meta_written:
                unlink(self.durable_meta)
                unlink(self.durable_meta + ".tmp")
        if removed or kept:
            log("%s rolled back: removed %s%s" % (self.label, ", ".join(removed) or "nothing",
                                                  "; could not remove " + ", ".join(kept) if kept else ""))


def restore_generation(selector, durable, durable_base, scope, meta, unattended, local, home_local, durable_run, status, attempt):
    """``--restore RUN/GENERATION``: validate on the durable side, stage the
    stores and the replicas (locally, and the durable replicas under their
    ``.incoming-`` names), gate, write run.json, activate last.  A refusal at
    any step leaves ``db/`` and ``home/sessions`` absent and removes what this
    attempt created (``attempt.rollback``), never the source generation; an
    OSError or sqlite error is a refusal naming the step.  Fills
    ``meta["parent"]``.  Returns (restored_from, pending_work)."""
    name, source_run, gen_dir = selector_paths(durable, selector)
    run, generation = selector
    db_dir = os.path.join(local, "db")
    rollouts = os.path.join(durable_run, "rollouts")
    staging = attempt.staging
    staging_db = os.path.join(staging, "db")
    try:
        attempt.step = "reading the manifest"
        manifest, manifest_sha = load_manifest(gen_dir, name)
        check_manifest(manifest, name, run, generation, scope, meta["codex_home"], meta["codex_home_real"], unattended)
        attempt.step = "checking the replicas"
        sources = check_replicas(manifest, name, source_run)
        if os.path.exists(db_dir):
            raise Refused("run directory %s already has db/; a restore never overwrites local databases" % local)
        if os.path.exists(attempt.sessions):
            raise Refused("run directory %s already has home/sessions; a restore places the rollouts into an empty run directory" % local)
        attempt.step = "staging the databases"
        if os.path.isdir(staging):
            shutil.rmtree(staging)
        mkdir_private(staging)
        mkdir_private(staging_db)
        expected_db = sum(int(record["bytes"]) for record in manifest["databases"].values())
        expected_rollouts = sum(size for _, size, _ in sources.values())
        started = time.monotonic()
        log("restoring generation %s (%s, captured %s): reading %d database(s), %d bytes, and %d replica(s), %d bytes, from %s" % (
            name, manifest["class"], manifest.get("captured_at_iso"), len(manifest["databases"]), expected_db, len(sources), expected_rollouts, source_run))
        db_bytes = stage_databases(manifest, name, gen_dir, staging_db)
        attempt.step = "reading the pending work"
        pending = pending_work(staging_db)
        status.note(pending_work=pending)      # recorded before the gate, so a refusal keeps it
        if unattended:
            gate_unattended(pending, "generation " + name)
        attempt.step = "creating the durable run directory"
        attempt.check_fence()           # the first durable write
        attempt.durable_created = not os.path.isdir(durable_run)
        ensure_durable_run(durable_run, durable_base)
        attempt.step = "staging the replicas"
        rollout_bytes = stage_replicas(manifest, name, sources, os.path.join(staging, "sessions"), rollouts, attempt)
        attempt.step = "writing run.json"
        meta["parent"] = {"run": run, "generation": generation, "class": manifest["class"], "manifest_sha256": manifest_sha}
        attempt.check_fence()
        attempt.meta_written = True
        write_run_meta(meta, local, durable_run)
        attempt.step = "activating"
        activate_restore(name, attempt, staging, db_dir, rollouts)
    except Refused:
        attempt.rollback()
        raise
    except (OSError, sqlite3.Error) as exc:
        attempt.rollback()
        raise Refused("restore of %s failed while %s: %s: %s" % (name, attempt.step, type(exc).__name__, exc))
    except Exception as exc:
        log("restore of %s failed while %s:\n%s" % (name, attempt.step, traceback.format_exc()))
        attempt.rollback()
        raise Refused("restore of %s failed while %s: %s: %s" % (name, attempt.step, type(exc).__name__, exc))
    log("placed %d database(s) (%d bytes) under %s and %d rollout(s) (%d bytes) under %s, replicas copied to %s, in %.1fs" % (
        len(manifest["databases"]), db_bytes, db_dir, len(sources), rollout_bytes, attempt.sessions, rollouts, time.monotonic() - started))
    restored = {"run": run, "generation": generation, "class": manifest["class"], "manifest_sha256": manifest_sha,
                "threads": len(manifest["threads"]), "bytes": db_bytes + rollout_bytes,
                "database_bytes": db_bytes, "rollout_bytes": rollout_bytes,
                "captured_at_iso": manifest.get("captured_at_iso"), "codex_version": manifest.get("codex_version")}
    return restored, pending


def restore_bounded(timeout, attempt, *args):
    """``restore_generation`` on a daemon thread, joined with the deadline: a
    durable side that does not answer (an episode on the shared filesystem)
    blocks that thread, which is abandoned, and the launch reports blocked
    instead of hanging; nothing was started.  The abandoned thread is fenced
    first, under the attempt's lock: should it wake up later, its next
    durable write (the activation included) refuses and it rolls back what it
    had created instead of installing a restore the launch has already
    reported blocked.  The activation's local renames hold the same lock, so
    a thread that has installed ``db/`` by the time the deadline is enforced
    is past every durable write and is not fenced: the launch takes its
    result (joined: only local work remains) rather than reporting as
    blocked a restore that is installed.  The normal path is the same call
    on another thread."""
    outcome = {}

    def work():
        try:
            outcome["result"] = restore_generation(*args)
        except BaseException as exc:
            outcome["error"] = exc
            if attempt.fenced:          # nobody reads the outcome of an abandoned thread
                log("the abandoned restore thread ended: %s" % exc)

    worker = threading.Thread(target=work, name="codex-state-restore")
    worker.daemon = True
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        with attempt.lock:
            installed = attempt.db_installed
            if not installed:
                attempt.fenced = True
        if not installed:
            raise Blocked("the durable side has not answered in %ds while %s; the restore is blocked, nothing is in use and Codex did not "
                          "start. Retry when the durable filesystem answers, or pass --fresh for a new, empty history" % (timeout, attempt.step))
        log("the restore deadline of %ds passed after the activation had landed; the launch takes the result" % timeout)
        worker.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


# ----------------------------------------------------------------- reuse in place

def reusable_run(scope_dir):
    """This allocation's latest run when it can be reused in place: a run
    directory with ``owner.json`` and ``db/`` whose recorded runner and child
    are both dead.  A run without ``db/`` (refused or never placed) is not a
    candidate; unreadable ownership or a live pid is a refusal, not a fresh start."""
    try:
        with open(os.path.join(scope_dir, "latest-run"), "r") as handle:
            run_id = handle.read().strip()
    except OSError:
        return None, None
    if not RUN_ID.fullmatch(run_id):
        return None, None
    local = os.path.join(scope_dir, run_id)
    owner_path = os.path.join(local, "owner.json")
    if not (os.path.isdir(os.path.join(local, "db")) and os.path.isfile(owner_path)):
        return None, None
    try:
        with open(owner_path, "r") as handle:
            owner = json.load(handle)
        if not isinstance(owner, dict):
            raise ValueError("not an object")
    except (OSError, ValueError) as exc:
        raise Refused("this allocation's latest run %s has local databases but an unreadable owner.json (%s); its ownership "
                      "is unknown, so nothing is reused, deleted or started" % (run_id, exc))
    for key in ("runner_pid", "child_pid"):      # both known before either is judged dead
        pid = owner.get(key)
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise Refused("this allocation's latest run %s has local databases but its owner.json records no usable %s (%r); its "
                          "ownership is unknown, so nothing is reused, deleted or started" % (run_id, key, pid))
    for key in ("runner_pid", "child_pid"):
        if pid_alive(owner[key]):
            raise Refused("this allocation's latest run %s records a live %s %d; a second writer on the same stores is "
                          "forbidden, so nothing starts until it has exited" % (run_id, key, owner[key]))
    return run_id, owner


def load_run_meta(local, durable_run, run_id, scope):
    """The reused run's immutable run.json: the local copy first, the durable one otherwise."""
    for path in (os.path.join(local, "run.json"), os.path.join(durable_run, "run.json")):
        try:
            with open(path, "r") as handle:
                meta = json.load(handle)
        except (OSError, ValueError):
            continue
        if isinstance(meta, dict) and meta.get("run") == run_id and meta.get("scope") == scope:
            return meta
    raise Refused("run %s cannot be reused: neither %s nor %s holds its run.json, so its provenance is unknown" % (
        run_id, os.path.join(local, "run.json"), os.path.join(durable_run, "run.json")))


def check_reusable(meta, owner, run_id, codex_home, codex_home_real, durable_base, durable_run):
    """A run is reused only under the canonical home it was created under
    (its stored rollout paths are absolute under that home) and the durable
    run its replicas and generations live in."""
    recorded_home = meta.get("codex_home_real")
    if recorded_home != codex_home_real:
        raise Refused("run %s cannot be reused under CODEX_HOME %s (%s): it was created under %s (%s), and its stored rollout "
                      "paths are absolute under that home; relaunch with that CODEX_HOME, or pass --fresh for a new history" % (
                          run_id, codex_home, codex_home_real, meta.get("codex_home"), recorded_home))
    recorded_durable = (owner or {}).get("durable")
    if recorded_durable != durable_run:
        raise Refused("run %s cannot be reused under --durable-root %s: its replicas and generations are recorded under %s, "
                      "not %s; relaunch with the durable root it was created under, or pass --fresh for a new history" % (
                          run_id, durable_base, recorded_durable, durable_run))


def ensure_durable_run(durable_run, durable_base):
    """``<BASE>/codex/<scope>/runs/<run>/{rollouts,generations}``: every component
    that exists from the base down is checked (a real directory, ours, not
    group- or world-writable), the rest are created 0700."""
    try:
        ensure_private_tree(durable_run, durable_base)
        ensure_private_dir(os.path.join(durable_run, "rollouts"))
        ensure_private_dir(os.path.join(durable_run, "generations"))
    except OSError as exc:
        raise Refused("cannot create the durable run directory %s: %s" % (durable_run, exc))


def write_run_meta(meta, local, durable_run):
    """run.json: the durable copy the generations hang off, and a local copy for reuse without the durable side."""
    for path in (os.path.join(durable_run, "run.json"), os.path.join(local, "run.json")):
        try:
            write_json(path, meta)
            fsync_path(path)
        except OSError as exc:
            raise Refused("cannot write %s: %s" % (path, exc))


class Status(object):
    """``status.json``: the placement state with its transitions, plus the
    replica, checkpoint and last-generation views the other threads merge in.
    Three threads write it, so every write happens under one lock."""

    def __init__(self, path):
        self.path = path
        self.data = {"transitions": []}
        self.lock = threading.Lock()
        self.durable_at = None     # monotonic time of the last durable generation
        self.write_errors = set()  # error texts already logged: status.json is best effort

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

    def note_latest(self, compute):
        """Merge the fields ``compute()`` returns, evaluated under the status
        lock right before the write: a writer delayed past a later completion
        then records the current acknowledgement, never its own stale one.
        ``compute`` may take the checkpointer's lock; nothing takes the two
        locks in the other order."""
        with self.lock:
            self.data.update(compute())
            self._write()

    def load(self):
        """A reused run's status.json, so its transitions accumulate across launches."""
        try:
            with open(self.path, "r") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return False
        if not isinstance(data, dict):
            return False
        if not isinstance(data.get("transitions"), list):
            data["transitions"] = []
        for key in ("exit", "outcome", "child_pid", "placed", "misplaced", "note"):   # the previous launch's, not this one's
            data.pop(key, None)
        with self.lock:
            self.data = data
        return True

    def _write(self):
        """Best effort: a status.json that cannot be written is logged once per
        distinct error and never stops replication, capture or reporting."""
        age = None if self.durable_at is None else round(time.monotonic() - self.durable_at, 1)
        self.data["durable_age_seconds"] = age
        try:
            write_json(self.path, self.data)
        except OSError as exc:
            text = "%s: %s" % (type(exc).__name__, exc)
            if text not in self.write_errors:
                self.write_errors.add(text)
                log("status.json not written (%s); the run continues without it until the write succeeds again" % text)


def mkdir_private(path):
    """The allocation-local tier: ours entirely, so an existing directory is re-tightened."""
    if not os.path.isdir(path):
        os.makedirs(path, 0o700)
    os.chmod(path, 0o700)


def check_private_dir(path):
    """An existing component of a tree this runner writes into must be a real
    directory (no symlink: it would redirect durable writes, possibly onto local
    disk), ours, and writable by nobody else.  Refuses otherwise; 0755 passes."""
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        raise Refused("%s is a symbolic link; the durable and session trees accept real directories only, because a link "
                      "would redirect this run's writes elsewhere. Replace it with a directory" % path)
    if not stat.S_ISDIR(st.st_mode):
        raise Refused("%s exists and is not a directory" % path)
    if st.st_uid != os.getuid():
        raise Refused("%s is owned by uid %d, not by this user (uid %d); this runner writes only into directories it owns" % (
            path, st.st_uid, os.getuid()))
    if st.st_mode & 0o022:
        raise Refused("%s is writable by its group or by everyone (mode %s); another writer could replace this run's state. "
                      "Fix it with: chmod g-w,o-w %s" % (path, oct(st.st_mode & 0o777), path))


def ensure_private_dir(path, parent_is_ours=True):
    """The durable tier: an existing directory is checked (never chmod'ed); a
    missing one is created with 0700 and its parent's entry synced, so a crash
    cannot lose the name of a directory a generation depends on.  The parent
    of the base is the operator's filesystem (it may be traversable but not
    readable, so its sync may be refused): that sync is best effort."""
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        check_private_dir(path)
        return False
    os.chmod(path, 0o700)
    try:
        fsync_path(os.path.dirname(path))
    except OSError as exc:
        if parent_is_ours:
            raise
        log("created %s but could not sync its parent directory (%s: %s); the parent is the operator's and the launch goes on" % (
            path, type(exc).__name__, exc))
    return True


def ensure_private_tree(path, base):
    """Every component of ``path`` from ``base`` downward: checked when it
    exists, created when it does not.  Components above ``base`` are the
    operator's filesystem and are not judged; the missing ones are created
    one by one from the first missing component down, each one's parent
    synced, so a generation under a brand-new durable root never depends on
    a name that was not synced.  Only the sync of a parent that existed
    before this call (the operator's directory: traversable, perhaps not
    readable) is best effort and logged when it fails; a parent this call
    created must sync, or the launch is refused."""
    base = os.path.normpath(base)
    path = os.path.normpath(path)
    if path != base and not path.startswith(base + "/"):
        raise Refused("%s is not under %s" % (path, base))
    missing = []
    above = os.path.dirname(base)
    while above and above != "/" and not os.path.isdir(above):
        missing.append(above)
        above = os.path.dirname(above)
    created = set()
    for directory in reversed(missing):
        try:
            os.mkdir(directory, 0o700)
        except FileExistsError:
            continue
        created.add(directory)
        parent = os.path.dirname(directory)
        try:
            fsync_path(parent)
        except OSError as exc:
            if parent in created:
                raise Refused("created %s but could not sync its parent directory %s (%s: %s), which this launch created as well; "
                              "a durable tree below it would depend on an unsynced name" % (directory, parent, type(exc).__name__, exc))
            log("created %s but could not sync its parent directory (%s: %s); the parent is the operator's and the launch goes on" % (
                directory, type(exc).__name__, exc))
    current = base
    ensure_private_dir(current, parent_is_ours=os.path.dirname(base) in created)
    if path != base:
        for component in path[len(base) + 1:].split("/"):
            current = os.path.join(current, component)
            ensure_private_dir(current)


def resolve_local_root():
    root = os.environ.get("SUCODER_LOCAL_ROOT")
    if not root:
        raise Refused("SUCODER_LOCAL_ROOT is not set; managed placement needs local-disk tiering")
    if not os.path.isdir(root):
        raise Refused("SUCODER_LOCAL_ROOT=%s is not a directory" % root)
    if not SAFE_PATH.fullmatch(root):
        raise Refused("SUCODER_LOCAL_ROOT=%s contains characters the placeholder substitution cannot carry" % root)
    return root


def resolve_durable_base(base, local_root):
    """The configured durable root: absolute, plain characters, not allocation-local."""
    if not os.path.isabs(base):
        raise Refused("--durable-root %s is not an absolute path" % base)
    if not SAFE_PATH.fullmatch(base):
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
    if not SAFE_PATH.fullmatch(db_dir):
        raise Refused("database path %s contains characters the substitution cannot carry" % db_dir)
    return [arg.replace(placeholder, db_dir) for arg in command]


def check_bind_targets(codex_home_real):
    """The entries managed placement needs in the shared home, checked before
    anything is created: ``prepare_binds`` runs after a restore has activated,
    where this refusal would come too late."""
    for name in REQUIRED_DIRS:
        target = os.path.join(codex_home_real, name)
        if not (os.path.isdir(target) and not os.path.islink(target)):
            raise Refused("%s does not exist in CODEX_HOME; managed placement needs it and will not create entries in the shared home" % target)


def prepare_binds(codex_home_real, home_local, keep_existing=False):
    """Local directories and files bound over entries of the shared home.

    ``keep_existing`` is the reuse-in-place case: a bound file that already
    exists locally (``history.jsonl`` holds the managed history, the AGENTS
    copies may have been edited) is kept, not overwritten from the shared
    home; the ``skills`` copy is refreshed either way."""
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
            if not (keep_existing and os.path.isfile(local)):
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
                text = handle.read()
        except OSError:
            continue
        try:
            ppid = int(text.rsplit(")", 1)[1].split()[1])
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
    """Pids (readable to us) with a descriptor open under the directory, by
    the link text alone: no stat of any descriptor (a stat follows the link
    onto whatever filesystem the file is on, and a Lustre file in an episode
    would block the exit).  A deleted store still held counts."""
    prefix = directory + "/"
    holders = set()
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        fd_dir = "/proc/%s/fd" % name
        try:
            entries = os.listdir(fd_dir)
        except OSError:
            continue
        for entry in entries:
            try:
                link = os.readlink(os.path.join(fd_dir, entry))
            except OSError:
                continue
            if link.startswith(prefix):
                holders.add(int(name))
                break
    return sorted(holders)


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


def monitor(child, db_dir, local_root, status, timeout, result, placed_event=None, exit_event=None):
    db_dev = os.stat(db_dir).st_dev
    local_dev = os.stat(local_root).st_dev
    deadline = time.time() + timeout
    reported = False
    # Between polls, sleep on the exit event when run() provides one: the
    # 5-second cadence of the placed state must not delay the exit by 5 s.
    pause = exit_event.wait if exit_event is not None else time.sleep
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
        pause(5.0 if reported else 0.5)   # keep watching for a later misplacement, cheaply
    if not reported and "outcome" not in result:
        result["outcome"] = "no_stores_opened"
        status.update("no_stores_opened")


# ----------------------------------------------------------------- replication

class Replicator(object):
    """Follows each rollout under ``home/sessions`` by growth and appends the
    new bytes to its durable replica, from a thread of its own.

    The replica file's own size is the truth for the replicated offset (a
    partial write or a failed fsync must never double-append).  A local file
    shorter than it was at the previous pass, or shorter than its replica, is
    a "shrunk" anomaly that stops replication of that thread: the rollout is
    append-only on this build, so after a shrink the unreplicated bytes are no
    longer the bytes that followed the replicated prefix.  Errors on the
    durable side (a slow or unwritable filesystem) are retried on the next
    pass and shown as replica age and lag; they block this thread only.
    """

    def __init__(self, sessions, rollouts, state_path, status):
        self.sessions = sessions
        self.rollouts = rollouts
        self.state_path = state_path
        self.status = status
        self.threads = {}
        self.dir_synced = set()     # threads whose replica's directory entry has been fsynced since this start
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
                match = ROLLOUT_NAME.fullmatch(name)
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
            replica_path = os.path.join(self.rollouts, tid + ".jsonl")
            try:
                replicated = os.path.getsize(replica_path)   # a replica placed before this run; durable, so not under the lock
            except OSError:
                replicated = 0
            entry = {"relative_path": relative, "local_bytes": 0, "replicated_bytes": replicated, "replica_path": replica_path}
            with self.state_lock:
                entry = self.threads.setdefault(tid, entry)
                self.dirty = True
        if entry.get("anomaly"):
            return
        local_path = os.path.join(self.sessions, relative)
        try:
            st = os.stat(local_path)
        except OSError:
            return
        local_bytes = st.st_size
        identity = [st.st_dev, st.st_ino]
        observed = entry["local_bytes"]
        if local_bytes != observed:
            entry["local_bytes"] = local_bytes
            self.dirty = True
        if entry.get("identity") is None:
            entry["identity"] = identity
            self.dirty = True
        if entry["identity"] != identity:
            # Another file at the rollout's path: whatever its length, its bytes
            # beyond the replica are not the ones that followed the replica.
            self.anomaly(tid, entry, "replaced", "was replaced by another file (device:inode %d:%d, was %d:%d; %d bytes, %d replicated)" % (
                identity[0], identity[1], entry["identity"][0], entry["identity"][1], local_bytes, entry["replicated_bytes"]))
        elif local_bytes < observed:
            # Shorter than at the previous pass: the bytes beyond the replica are
            # not the ones that followed it, whatever the replica's length.
            self.shrunk(tid, entry, local_bytes, observed, "than it was at the previous pass")
        elif local_bytes < entry["replicated_bytes"]:
            self.shrunk(tid, entry, local_bytes, entry["replicated_bytes"], "than its replica")
        elif local_bytes > entry["replicated_bytes"] or entry.get("error") or tid not in self.dir_synced:
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

    def anomaly(self, tid, entry, kind, detail):
        entry["anomaly"] = kind
        entry["anomaly_at"] = time.time()
        self.dirty = True
        log("rollout of %s %s; replication of this thread stops" % (tid, detail))

    def shrunk(self, tid, entry, local_bytes, previous, than):
        self.anomaly(tid, entry, "shrunk", "shrank to %d bytes, shorter %s (%d bytes; %d replicated)" % (
            local_bytes, than, previous, entry["replicated_bytes"]))

    def append(self, tid, entry, local_path, local_bytes):
        fd = os.open(entry["replica_path"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            size = os.fstat(fd).st_size
            if size > local_bytes:
                entry["replicated_bytes"] = size
                self.shrunk(tid, entry, local_bytes, size, "than its replica")
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
            size = os.fstat(fd).st_size
        finally:
            os.close(fd)
        if tid not in self.dir_synced:
            # The file's bytes are durable; its name is not until the directory
            # is, and the publication gate reads replicated_bytes, so the name
            # is synced before the count advances.  A failed sync leaves the
            # count as it was (the replica's own size is the truth): the next
            # pass re-enters here, writes nothing new, and retries the sync.
            fsync_path(self.rollouts)
            self.dir_synced.add(tid)
        entry["replicated_bytes"] = size

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
        self.seq = None                 # the highest published sequence: listed lazily, off run()'s path to child.wait()
        self.seen = set()
        self.count = 0
        self.dropped = 0
        self.last_outcome = None
        self.reason = None
        self.last_generation = None
        self.pending_sync = None        # published, but its directory entry's sync failed: not acknowledged yet
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
                match = re.fullmatch(r"^(\d{6})-[0-9a-f]{8}$", name)
                if match:
                    highest = max(highest, int(match.group(1)))
        return highest

    def ensure_seq(self):
        """The durable ``generations/`` listing, once, from the scheduler thread
        (or from the first ``begin`` when that comes first)."""
        if self.seq is None:
            self.seq = self.highest_seq()

    def start(self):
        self.thread.start()

    # -- scheduling

    def loop(self):
        initial_done = False
        try:
            self.ensure_seq()
        except Exception as exc:
            self.errors += 1
            log("checkpoint scheduler error: %s: %s" % (type(exc).__name__, exc))
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

    def begin(self, trigger, cls, final_pass=False, budget=None):
        """Start one capture unless one is in flight.  ``budget`` is the final
        capture's remaining share of the shutdown deadline; otherwise a capture
        gets the full timeout."""
        with self.inflight_lock:
            self.last_started = time.monotonic()
            if self.inflight is not None:
                self.dropped += 1
                blocker = self.inflight["generation"]
                flight = None
            else:
                self.ensure_seq()
                self.count += 1
                self.seq += 1
                generation = "%06d-%s" % (self.seq, uuid.uuid4().hex[:8])
                deadline = time.monotonic() + (self.timeout if budget is None else budget)
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
            self.retry_sync()
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
            # The acknowledgement is complete before the next capture may
            # begin: outcome, reason and the generation under the same lock
            # that clears in-flight.  The status write happens outside it and
            # records whatever is current at that moment (note_latest), so a
            # slow writer never overwrites a later acknowledgement.
            with self.inflight_lock:
                self.last_outcome, self.reason = outcome, reason
                if published is not None:
                    if published.get("sync_warning"):
                        # The rename succeeded but generations/ is not known to be
                        # durable: no acknowledgement until a later sync succeeds.
                        self.pending_sync = published
                    else:
                        self.last_generation = published
                        self.pending_sync = None    # the same directory sync covered an earlier unsynced one
                        self.status.durable_at = time.monotonic()
                flight, self.inflight = self.inflight, None
            self.status.note_latest(self.acknowledgement)
            self.report(generation, trigger, cls, outcome, reason, time.monotonic() - started, flight)

    def acknowledgement(self):
        """The status fields of the current checkpoint state: evaluated by
        ``Status.note_latest`` under its lock, right before the write."""
        fields = {"checkpoint": self.view(None)}
        with self.inflight_lock:
            last = self.last_generation
            fields["unsynced_generation"] = self.pending_sync
        if last is not None:
            fields["last_generation"] = last
        return fields

    def adopt_unsynced(self, record):
        """A reused run's outstanding unsynced generation, from the status.json
        the earlier launch left: taken over when it is shaped like a record
        this runner publishes (the published record's keys, a generation
        name, and exactly this run's ``generations/<name>`` as its path), so
        the next capture's retry acknowledges it; anything else is ignored
        with a log line.  Judged on the record alone, without touching the
        filesystem: this runs in run() before child.wait(), where a stalled
        durable stat would hold the runner.  Whether the generation is still
        there is the retry's business, on the capture's own worker.  Before
        the scheduler starts, so no lock."""
        usable = (isinstance(record, dict) and all(key in record for key in PUBLISHED_RECORD_KEYS)
                  and isinstance(record["name"], str) and GENERATION_NAME.fullmatch(record["name"]) is not None
                  and isinstance(record["path"], str) and record["path"] == os.path.join(self.generations, record["name"]))
        if not usable:
            log("status.json records an unusable unsynced_generation (%r); it is not taken over" % (record,))
            return False
        self.pending_sync = record
        log("generation %s of the reused run is not acknowledged yet (the sync of %s failed in the earlier launch); "
            "retried before the next capture" % (record["name"], self.generations))
        return True

    def retry_sync(self):
        """A generation published but whose directory entry's sync failed is
        acknowledged only once ``generations/`` syncs: retried at the start
        of every later capture (the final one included, under its deadline on
        its own worker; never on the exit path itself, where a blocked sync
        would hang the exit) and promoted to the last generation.  The
        generation directory and its manifest.json must still be there (a
        record taken over from an earlier launch may name one that is gone):
        a missing one is dropped, not acknowledged; a stat that fails
        otherwise (the durable side not answering) keeps the record for the
        next retry."""
        with self.inflight_lock:
            pending = self.pending_sync
        if pending is None:
            return
        try:
            present = (stat.S_ISDIR(os.stat(pending["path"]).st_mode)
                       and stat.S_ISREG(os.stat(os.path.join(pending["path"], "manifest.json")).st_mode))
        except (FileNotFoundError, NotADirectoryError):
            present = False
        except OSError as exc:
            log("generation %s is still not acknowledged: %s cannot be checked (%s: %s); retried before the next capture" % (
                pending["name"], pending["path"], type(exc).__name__, exc))
            return
        if not present:
            with self.inflight_lock:
                if self.pending_sync is pending:
                    self.pending_sync = None
            log("generation %s is dropped, not acknowledged: %s or its manifest.json is gone" % (pending["name"], pending["path"]))
            self.status.note_latest(self.acknowledgement)
            return
        try:
            fsync_path(self.generations)
        except OSError as exc:
            log("generation %s is still not acknowledged: the sync of %s failed again (%s: %s)" % (
                pending["name"], self.generations, type(exc).__name__, exc))
            return
        record = dict(pending)
        record.pop("sync_warning", None)
        record["synced_at"] = time.time()
        with self.inflight_lock:
            if self.pending_sync is pending:
                self.pending_sync = None
                self.last_generation = record
                self.status.durable_at = time.monotonic()
        log("generation %s acknowledged: %s synced on retry" % (record["name"], self.generations))
        self.status.note_latest(self.acknowledgement)

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
        unclassified = [n for n in names if n not in REQUIRED_STORES and n not in OPTIONAL_STORES]
        if unclassified:
            raise CaptureFailed("unclassified store under db: %s" % ", ".join(unclassified))
        self.seen.update(n for n in REQUIRED_STORES if n in names)
        gone = sorted(n for n in self.seen if n not in names)
        if gone:                      # before the empty-directory shortcut: every store vanishing is the worst case of this
            raise CaptureFailed("required store disappeared: %s" % ", ".join(gone))
        if not names:
            raise CaptureSkipped("nothing to capture: no store under %s" % self.db_dir)
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
            threads, version, required = self.inspect(os.path.join(staging, "db"), names, deadline)
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
        lagging = self.wait_for_replicas(threads, required, deadline)
        if lagging:
            manifest["not_durable"] = lagging
            write_json(os.path.join(staging, "manifest.json"), manifest)
            first = sorted(lagging.items())[0]
            raise NotDurable("%d thread(s) not replicated to their rollout's length at capture in %ds: %s has %d replicated bytes, "
                             "the projection needs %d and the rollout held %d" % (
                                 len(lagging), self.timeout, first[0], first[1]["replica_bytes"],
                                 first[1]["next_rollout_byte_offset"], first[1]["local_bytes"]))
        write_json(os.path.join(staging, "manifest.json"), manifest)
        return manifest

    def inspect(self, db_copy, names, deadline):
        """Threads from the copies: projection offsets, rollout paths, the Codex
        version, and per thread the bytes its replica must reach (the greater of
        the copy's watermark and the local rollout's length right now, just
        after the copies: the watermark never exceeds the rollout's length at a
        completed turn, and a thread the replicator has not discovered yet is
        covered by its length too).  A copy whose table cannot be read is a
        storage layout this runner does not know: the capture fails rather
        than publishing a generation without its watermarks."""
        threads, paths, required, version = {}, {}, {}, None
        if "state_5.sqlite" in names:
            rows = self.read_copy(db_copy, "state_5.sqlite", "select id, rollout_path, cli_version from threads", deadline)
            for tid, path, cli in rows:
                paths[tid] = path
                if cli and (version is None or version_key(cli) > version_key(version)):
                    version = cli
        if "thread_history_1.sqlite" in names:
            rows = self.read_copy(db_copy, "thread_history_1.sqlite",
                                  "select thread_id, next_rollout_byte_offset, next_rollout_ordinal from thread_history_projection_state", deadline)
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
            try:
                local_bytes = os.path.getsize(os.path.join(self.sessions, relative))
            except OSError:
                local_bytes = 0
            required[tid] = {"bytes": max(entry["next_rollout_byte_offset"], local_bytes), "local_bytes": local_bytes}
        return threads, version, required

    @staticmethod
    def read_copy(db_copy, name, sql, deadline):
        conn = connect_copy(os.path.join(db_copy, name), deadline)
        try:
            return conn.execute(sql).fetchall()
        except sqlite3.Error as exc:
            raise CaptureFailed("%s: cannot read %s (%s: %s); this copy passed quick_check, so its storage layout is not the "
                                "one this runner knows and no generation is published from it" % (
                                    name, sql.split(" from ", 1)[-1], type(exc).__name__, exc))
        finally:
            conn.close()

    def wait_for_replicas(self, threads, required, deadline):
        """The ordering invariant: every thread's replica at or past the bytes
        ``inspect`` required of it (the copy's watermark or the rollout's length
        at capture, whichever is greater), within the budget.  Records the
        replica length and the local rollout's hash up to it; returns the
        lagging ones."""
        while True:
            replicas = self.replicator.snapshot()
            lagging = {}
            for tid, entry in threads.items():
                have = replicas[tid]["replicated_bytes"] if tid in replicas else 0
                if have < required[tid]["bytes"]:
                    lagging[tid] = {"replica_bytes": have, "next_rollout_byte_offset": entry["next_rollout_byte_offset"],
                                    "local_bytes": required[tid]["local_bytes"]}
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
        published = time.time()
        record = {"name": generation, "class": manifest["class"], "published_at": published, "published_at_iso": utc_iso(published),
                  "manifest_sha256": manifest_sha, "path": final, "trigger": manifest["trigger"]}
        try:
            fsync_path(self.generations)
        except OSError as exc:
            # The rename succeeded and the manifest is complete (a restore
            # verifies by hash), so the generation is published; only its
            # directory entry is not yet known to be durable.
            record["sync_warning"] = "%s: %s" % (type(exc).__name__, exc)
            log("generation %s is published but the sync of %s failed (%s); its directory entry may not survive a crash of the "
                "filesystem until the next successful sync" % (generation, self.generations, record["sync_warning"]))
        return record

    def prune_local(self):
        try:
            names = sorted(n for n in os.listdir(self.snapshots) if re.fullmatch(r"^\d{6}-[0-9a-f]{8}$", n))
        except OSError:
            return
        for name in names[:-LOCAL_BUNDLES_KEPT]:
            shutil.rmtree(os.path.join(self.snapshots, name), ignore_errors=True)

    # -- after Codex exits

    def describe_unsynced(self):
        """The exit path's note on a generation still unacknowledged: its sync
        is retried only inside a capture, so when no final capture runs it
        stays as it is."""
        with self.inflight_lock:
            pending = self.pending_sync
        if pending is None:
            return ""
        return "; generation %s stays unacknowledged (its directory sync is retried only inside a capture)" % pending["name"]

    def final(self, cls):
        """One last replication pass and the sealed or crash capture under one
        shutdown deadline of the checkpoint timeout: waiting for a capture in
        flight spends from it, the final capture gets what remains, and when
        nothing remains it is skipped.  Release is never blocked: nothing here
        touches the durable side itself (the final capture's worker retries an
        unsynced generation's directory sync, under the deadline)."""
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
            log("final %s capture skipped: capture %s is still in flight; last durable generation: %s%s" % (
                cls, flight["generation"], self.describe_last(), self.describe_unsynced()))
            return
        with self.inflight_lock:
            listed = self.seq is not None
        if not listed:
            # The scheduler never got its listing of generations/ back: the
            # durable side is not answering, and the exit path does not ask it.
            log("final %s capture skipped: the durable generations listing never completed, so nothing could be published; "
                "last durable generation: %s%s" % (cls, self.describe_last(), self.describe_unsynced()))
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            log("final %s capture skipped: the %ds shutdown budget was spent waiting for the capture in flight; last durable "
                "generation: %s%s" % (cls, self.timeout, self.describe_last(), self.describe_unsynced()))
            return
        worker = self.begin("final", cls, final_pass=True, budget=remaining)
        if worker is None:
            return
        worker.join(max(0.0, deadline + 2.0 - time.monotonic()))
        if worker.is_alive():
            log("final %s capture still running after its %.1fs share of the %ds shutdown budget; exiting without it; last durable "
                "generation: %s" % (cls, remaining, self.timeout, self.describe_last()))
            self.status.note(checkpoint=self.view("stuck"))


# ----------------------------------------------------------------- the run

def run(args):
    status = None
    staging = None
    try:
        if shutil.which("bwrap") is None:
            raise Refused("bwrap is not installed on this node; managed placement is unavailable and SuCoder will not fall back to shared SQLite")
        local_root = resolve_local_root()
        codex_home, codex_home_real = resolve_codex_home(args.codex_home)
        check_conflicts(args.command, args.placeholder, codex_home_real)
        check_bind_targets(codex_home_real)
        durable_base = resolve_durable_base(args.durable_root, local_root)
        selector = parse_selector(args.restore) if args.restore else None    # a bad selector touches nothing
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
        durable = os.path.join(durable_base, "codex", args.scope)
        # Within one allocation local state takes precedence over every
        # archived generation: an intact run is reused in place unless --fresh.
        run_id, owner_before = (None, None) if args.fresh else reusable_run(scope_dir)
        if run_id is not None:
            mode = "reuse"
        else:
            run_id = uuid.uuid4().hex
            mode = "restore" if selector else "fresh"
        local = os.path.join(scope_dir, run_id)
        db_dir = os.path.join(local, "db")
        home_local = os.path.join(local, "home")
        for path in (local, home_local, os.path.join(local, "snapshots")):
            mkdir_private(path)
        status = Status(os.path.join(local, "status.json"))
        if mode == "reuse":
            status.load()
        LOG_FILE["path"] = os.path.join(local, "runner.log")
        command = substitute(args.command, args.placeholder, db_dir)
        env = dict(os.environ)
        env["CODEX_HOME"] = codex_home
        env["SUCODER_CODEX_STATE_RUN"] = local
        durable_run = os.path.join(durable, "runs", run_id)
        restored = None
        restore_ignored = None
        attempt = None
        unsynced_before = None
        pending = {"active_goals": [], "queued": {}}
        if mode == "reuse":
            unsynced_before = status.data.get("unsynced_generation")    # the earlier launch's, taken over below
            if args.restore:
                restore_ignored = args.restore
                log("IGNORING --restore %s: this allocation already holds run %s with its local databases, and within one "
                    "allocation local state takes precedence over every archived generation" % (args.restore, run_id))
            step = "reading run.json"
            try:
                meta = load_run_meta(local, durable_run, run_id, args.scope)
                check_reusable(meta, owner_before, run_id, codex_home, codex_home_real, durable_base, durable_run)
                step = "creating the durable run directory"
                ensure_durable_run(durable_run, durable_base)
                step = "reading the pending work"
                pending = pending_work(db_dir)
                status.note(pending_work=pending)      # recorded before the gate, so a refusal keeps it
                if args.unattended:
                    gate_unattended(pending, "this allocation's run %s" % run_id)
            except (OSError, sqlite3.Error) as exc:
                raise Refused("run %s cannot be reused: %s failed: %s: %s" % (run_id, step, type(exc).__name__, exc))
            restored = (status.data.get("restored_from") if isinstance(status.data.get("restored_from"), dict)
                        else (owner_before or {}).get("restored_from"))
        else:
            if args.fresh and os.path.isfile(os.path.join(scope_dir, "latest-run")):
                log("--fresh: starting a new run although this allocation holds an earlier run for the scope; that run stays on disk and is not reused")
            created = time.time()
            meta = {
                "format": 1, "scope": args.scope, "run": run_id, "parent": None,
                "allocation": {"job": os.environ.get("SLURM_JOB_ID"), "node": os.uname()[1]},
                "codex_home": codex_home, "codex_home_real": codex_home_real,
                "created": created, "created_iso": utc_iso(created),
            }
            attempt = RestoreAttempt(local, home_local, durable_run, label=mode if mode == "restore" else "fresh launch")
            staging = attempt.staging
            if mode == "restore":
                name, _, gen_dir = selector_paths(durable, selector)
                status.update("restoring", restore=name, source=gen_dir)    # before the first durable read
                restored, pending = restore_bounded(restore_timeout(args.checkpoint_timeout), attempt, selector, durable, durable_base,
                                                    args.scope, meta, args.unattended, local, home_local, durable_run, status, attempt)
            else:
                attempt.durable_created = not os.path.isdir(durable_run)
                ensure_durable_run(durable_run, durable_base)    # refused before anything local is created
                mkdir_private(db_dir)
                attempt.db_installed = True
                attempt.meta_written = True
                write_run_meta(meta, local, durable_run)
        # The binds last: a restore has renamed its sessions tree under home/
        # by now, so nothing created the directory before it.  Until owner.json
        # and latest-run are written the launch is still the attempt's
        # transaction: a refusal or an OSError here takes back what a restore
        # or a fresh launch created (a reused run has nothing to take back),
        # so the next launch starts clean.
        launch = {"restore": "restore of %s" % args.restore, "fresh": "the new run %s" % run_id[:8],
                  "reuse": "reuse of run %s" % run_id[:8]}[mode]
        step = "preparing the binds"
        try:
            binds, skipped = prepare_binds(codex_home_real, home_local, keep_existing=(mode == "reuse"))
            argv = bwrap_argv(binds, command)
            owner = {
                "scope": args.scope, "run": run_id, "local": local, "db": db_dir, "codex_home": codex_home,
                "codex_home_real": codex_home_real, "job": meta["allocation"]["job"], "node": meta["allocation"]["node"],
                "durable_root": durable_base, "durable": durable_run,
                "bound": [{"local": l, "target": t} for l, t in binds], "skipped": skipped,
                "runner_pid": os.getpid(), "started": time.time(), "command": elide(command),
                "reused_run": mode == "reuse", "restored_from": restored, "pending_work": pending,
            }
            if owner_before is not None:
                previous = list(owner_before.get("previous") or [])
                previous.append({"runner_pid": owner_before.get("runner_pid"), "child_pid": owner_before.get("child_pid"),
                                 "started": owner_before.get("started")})
                owner["previous"] = previous
                owner["first_started"] = owner_before.get("first_started", owner_before.get("started"))
            if restore_ignored:
                owner["restore_ignored"] = restore_ignored
            step = "writing owner.json"
            write_json(os.path.join(local, "owner.json"), owner)
            step = "recording the latest run"
            with open(os.path.join(scope_dir, "latest-run"), "w") as handle:
                handle.write(run_id + "\n")
        except Refused:
            if attempt is not None:
                attempt.rollback()
            raise
        except OSError as exc:
            if attempt is not None:
                attempt.rollback()
            raise Refused("%s failed while %s: %s: %s" % (launch, step, type(exc).__name__, exc))
        fields = {"scope": args.scope, "run": run_id, "db": db_dir, "durable": durable_run, "bound": [t for _, t in binds],
                  "skipped": skipped, "mode": mode, "reused_run": mode == "reuse", "restored_from": restored, "pending_work": pending}
        if restore_ignored:
            fields["restore_ignored"] = restore_ignored
        status.update("starting", **fields)
        log("run %s: databases at %s; bound %s; skipped %s" % (run_id[:8], db_dir, ", ".join(os.path.basename(t) for _, t in binds) or "nothing", ", ".join(skipped) or "nothing"))
        if mode == "restore":
            line = ("run %s restores generation %s/%s (%s, captured %s): %d thread(s), %d rollout bytes and %d database bytes placed "
                    "from %s; pending work: %s" % (run_id[:8], restored["run"], restored["generation"], restored["class"],
                                                    restored.get("captured_at_iso"), restored["threads"], restored["rollout_bytes"],
                                                    restored["database_bytes"], durable, describe_pending(pending)))
            if restored["class"] != "sealed":
                line += ("; a %s generation was captured while Codex ran, so its goals, queue, memories and state rows are "
                         "consistent per database only, not across databases" % restored["class"])
            log(line)
        elif mode == "reuse":
            rollouts = 0
            for _, _, filenames in os.walk(os.path.join(home_local, "sessions")):
                rollouts += len([n for n in filenames if n.startswith("rollout-") and n.endswith(".jsonl")])
            log("run %s: reusing this allocation's run with its local databases and %d rollout(s) at %s; its managed history, "
                "replicas and generations continue under %s; pending work: %s" % (run_id[:8], rollouts, local, durable_run, describe_pending(pending)))
        else:
            log("the managed Codex history for this mirror starts empty and is separate from the shared history in %s; rollout replicas and database generations go to %s" % (codex_home, durable))
    except Blocked as exc:
        # The restore thread may still be mid-write: nothing is cleaned from here.
        log("restore blocked: %s" % exc)
        if status is not None:
            status.update("blocked", reason=str(exc))
        return EXIT_REFUSED
    except Refused as exc:
        log("refusing to launch: %s" % exc)
        if staging is not None and os.path.isdir(staging):
            shutil.rmtree(staging, ignore_errors=True)
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
    exited = threading.Event()
    watcher = threading.Thread(target=monitor, args=(child, db_dir, local_root, status, args.placement_timeout, result),
                               kwargs={"placed_event": placed, "exit_event": exited})
    watcher.daemon = True
    watcher.start()
    replicator = Replicator(os.path.join(home_local, "sessions"), os.path.join(durable_run, "rollouts"),
                            os.path.join(local, "replica.json"), status)
    replicator.start()
    checkpointer = Checkpointer(local, db_dir, os.path.join(home_local, "sessions"), durable_run, replicator, status, meta,
                                args.checkpoint_minutes, args.checkpoint_timeout, placed)
    if unsynced_before is not None:
        checkpointer.adopt_unsynced(unsynced_before)
    checkpointer.start()
    while True:
        try:
            code = child.wait()
            break
        except InterruptedError:
            continue
    exited.set()
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
                        help="continue from a published generation of this scope (RUN: 32 hex characters, GENERATION: NNNNNN-hhhhhhhh, "
                             "under ABS_BASE/codex/<scope>/runs): its stores are staged and verified locally, its rollout replicas placed "
                             "at their recorded paths, and Codex starts on them; refused with exit 70 when anything does not verify, "
                             "with nothing of the attempt left behind. A durable side that does not answer within max(120 s, 4 x "
                             "--checkpoint-timeout) (SUCODER_CODEX_STATE_RESTORE_TIMEOUT overrides, in seconds, at most a day) leaves the launch "
                             "blocked, exit 70: retry, or pass --fresh. Ignored when this allocation already holds an intact run for "
                             "the scope, which is reused in place instead")
    parser.add_argument("--fresh", action="store_true",
                        help="start a new, empty managed history even when this allocation holds a run for the scope "
                             "(which is otherwise reused in place); exclusive with --restore")
    parser.add_argument("--unattended", action="store_true",
                        help="no interactive client is attached: a restore accepts sealed generations only, and both restore and reuse "
                             "refuse (exit 70) while the stores hold an active goal or queued items, because an active goal drives "
                             "unprompted model requests on resume and a queued item executes once; pause the goal in Codex first")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="the Codex command, after --")
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("no command given after --")
    if args.restore and args.fresh:
        parser.error("--restore and --fresh are mutually exclusive")
    if not re.fullmatch(r"^[0-9a-f]{8,64}$", args.scope):
        parser.error("--scope must be a hex digest")
    if args.checkpoint_minutes <= 0 or args.checkpoint_timeout <= 0:
        parser.error("--checkpoint-minutes and --checkpoint-timeout must be positive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
