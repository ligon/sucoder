#!/usr/bin/env python3
"""SuCoder managed Codex state runner: one owned Codex process on local disk.

Staged to the compute node and run as ``python3 <this file> --scope ... --
<command>``, like the native context installer, so it has no dependency
beyond the standard library and must stay valid for the oldest ``python3``
a node image offers (3.6 has been seen).

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
  ever falls back to shared placement.

Capture, replication and restore are Slice 2 and are not here.
"""
import argparse
import fcntl
import json
import os
import re
import shutil
import signal
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


class Refused(Exception):
    """The launch cannot proceed under managed placement; nothing was started."""


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


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        handle.write(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


class Status(object):
    def __init__(self, path):
        self.path = path
        self.data = {"transitions": []}

    def update(self, state, **fields):
        self.data.update(fields)
        self.data["state"] = state
        self.data["transitions"].append({"state": state, "at": time.time()})
        write_json(self.path, self.data)


def mkdir_private(path):
    if not os.path.isdir(path):
        os.makedirs(path, 0o700)
    os.chmod(path, 0o700)


def resolve_local_root():
    root = os.environ.get("SUCODER_LOCAL_ROOT")
    if not root:
        raise Refused("SUCODER_LOCAL_ROOT is not set; managed placement needs local-disk tiering")
    if not os.path.isdir(root):
        raise Refused("SUCODER_LOCAL_ROOT=%s is not a directory" % root)
    if not SAFE_PATH.match(root):
        raise Refused("SUCODER_LOCAL_ROOT=%s contains characters the placeholder substitution cannot carry" % root)
    return root


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


def monitor(child, db_dir, local_root, status, timeout, result):
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
        if not reported and time.time() > deadline:
            result["outcome"] = "unverified"
            status.update("unverified", note="no store descriptor appeared within %ds" % timeout)
            log("no state descriptor appeared within %ds; placement unverified" % timeout)
            return
        time.sleep(5.0 if reported else 0.5)   # keep watching for a later misplacement, cheaply
    if not reported and "outcome" not in result:
        result["outcome"] = "no_stores_opened"
        status.update("no_stores_opened")


def run(args):
    status = None
    try:
        if shutil.which("bwrap") is None:
            raise Refused("bwrap is not installed on this node; managed placement is unavailable and SuCoder will not fall back to shared SQLite")
        local_root = resolve_local_root()
        codex_home, codex_home_real = resolve_codex_home(args.codex_home)
        check_conflicts(args.command, args.placeholder, codex_home_real)
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
        owner = {
            "scope": args.scope, "run": run_id, "local": local, "db": db_dir, "codex_home": codex_home,
            "codex_home_real": codex_home_real, "job": os.environ.get("SLURM_JOB_ID"), "node": os.uname()[1],
            "bound": [{"local": l, "target": t} for l, t in binds], "skipped": skipped,
            "runner_pid": os.getpid(), "started": time.time(), "command": elide(command),
        }
        write_json(os.path.join(local, "owner.json"), owner)
        with open(os.path.join(scope_dir, "latest-run"), "w") as handle:
            handle.write(run_id + "\n")
        status.update("starting", scope=args.scope, run=run_id, db=db_dir, bound=[t for _, t in binds], skipped=skipped)
        log("run %s: databases at %s; bound %s; skipped %s" % (run_id[:8], db_dir, ", ".join(os.path.basename(t) for _, t in binds) or "nothing", ", ".join(skipped) or "nothing"))
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
    watcher = threading.Thread(target=monitor, args=(child, db_dir, local_root, status, args.placement_timeout, result))
    watcher.daemon = True
    watcher.start()
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
    if result.get("outcome") == "misplaced":
        status.update("stopped_misplaced", exit=code)
        return EXIT_MISPLACED
    status.update("exited", exit=code, outcome=result.get("outcome"))
    log("Codex exited with %d (%s)" % (code, result.get("outcome", "unknown")))
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description="SuCoder managed Codex state runner")
    parser.add_argument("--scope", required=True, help="stable scope digest from the launcher")
    parser.add_argument("--placeholder", default=PLACEHOLDER_DEFAULT)
    parser.add_argument("--codex-home", default=None)
    parser.add_argument("--placement-timeout", type=float, default=90.0)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="the Codex command, after --")
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("no command given after --")
    if not re.match(r"^[0-9a-f]{8,64}$", args.scope):
        parser.error("--scope must be a hex digest")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
