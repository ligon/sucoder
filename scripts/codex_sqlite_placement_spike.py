#!/usr/bin/env python3
"""Inventory where one Codex launch puts its SQLite state (design Slice 0).

See docs/codex-local-state-spike.org for the cases, results and limits.

Read-only with respect to the real account: every case runs with an isolated
HOME and CODEX_HOME under a private spike root, a loopback fake model (the
pattern of tests/test_codex_context.py:158), and strace recording every file,
connect, bind, flock and fcntl call.  The strace classification is the
evidence of what Codex opened, including anything under the real Codex home.
"""
import argparse
import json
import os
import pathlib
import queue
import re
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = pathlib.Path(os.environ.get("SUCODER_REPO") or pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, str(REPO))
from sucoder.agent_mode import config_override  # noqa: E402

SQLITE_NAME = re.compile(r"\.sqlite(-wal|-shm|-journal)?$|\.lock$|-locks(/|$)")
QUOTED_PATH = re.compile(r'"((?:/|\./|\.\./)[^"]*)"')
FD_PATH = re.compile(r"<(/[^>]*?)(?: \(deleted\))?>")
MUTATORS = {"unlink", "unlinkat", "rename", "renameat", "renameat2", "mkdir", "mkdirat",
            "symlink", "symlinkat", "link", "linkat", "truncate", "ftruncate", "rmdir"}
STRACE_FILTER = "trace=%file,connect,bind,flock,fcntl"


class FakeModel(ThreadingHTTPServer):
    """Responses-wire endpoint that completes every request with no output."""

    def __init__(self):
        self.requests = queue.Queue()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                try:
                    server.requests.put(json.loads(body))
                except ValueError:
                    server.requests.put({"raw": body[:200].decode("utf-8", "replace")})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                event = {"type": "response.completed", "response": {
                    "id": "resp_spike", "object": "response", "status": "completed", "output": [],
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}}
                self.wfile.write(("event: response.completed\ndata: "
                                  + json.dumps(event) + "\n\n").encode())

        super().__init__(("127.0.0.1", 0), Handler)


def descendants(root_pid):
    children = {}
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        ppid = int(stat.rsplit(")", 1)[1].split()[1])
        children.setdefault(ppid, []).append(int(entry.name))
    found, stack = [], [root_pid]
    while stack:
        pid = stack.pop()
        for child in children.get(pid, []):
            found.append(child)
            stack.append(child)
    return found


def fd_snapshot(pids):
    snapshot = []
    for pid in pids:
        proc = pathlib.Path(f"/proc/{pid}")
        try:
            comm = (proc / "comm").read_text().strip()
            argv = (proc / "cmdline").read_bytes().split(b"\0")
            fds = []
            for fd in (proc / "fd").iterdir():
                try:
                    fds.append(os.readlink(fd))
                except OSError:
                    pass
        except OSError:
            continue
        snapshot.append({"pid": pid, "comm": comm,
                         "argv0": argv[0].decode("utf-8", "replace") if argv else "",
                         "argv_len": len([a for a in argv if a]),
                         "paths": sorted({f for f in fds if f.startswith("/")})})
    return snapshot


def parse_strace(path):
    """Map every path strace saw to the set of syscall names and tags."""
    seen = {}
    with open(path, errors="replace") as handle:
        for line in handle:
            body = re.sub(r"^\d+\s+", "", line.rstrip("\n"))
            if body.startswith(("+++", "---", "<... ")) or "(" not in body:
                continue
            call = body.split("(", 1)[0].strip()
            if " <unfinished ...>" in body:
                body = body.replace(" <unfinished ...>", "")
            result = body.rsplit(" = ", 1)[1] if " = " in body else ""
            failed = result.startswith("-1")
            errno = result.split()[1] if failed and len(result.split()) > 1 else ""
            for found in set(QUOTED_PATH.findall(body)) | set(FD_PATH.findall(body)):
                tags = seen.setdefault(found, set())
                tags.add(call)
                if failed:
                    tags.add(errno or "failed")
                elif call in ("openat", "open", "creat") and re.search(r"O_WRONLY|O_RDWR|O_CREAT", body):
                    tags.add("write-open")
                elif call in MUTATORS:
                    tags.add("mutate")
                elif call == "fcntl" and "F_SETLK" in body:
                    tags.add("fcntl-lock")
                elif call == "flock":
                    tags.add("flock")
    return seen


def classify(path, roots):
    for name, root in roots:
        if path == root or path.startswith(root.rstrip("/") + "/"):
            return name
    if "node_modules/@openai" in path or "/.nvm/" in path:
        return "install"
    if path.startswith(("/usr/", "/lib", "/etc/", "/proc/", "/sys/", "/dev/", "/run/", "/bin/", "/sbin/")):
        return "system"
    return "other"


def tree(root):
    root = pathlib.Path(root)
    if not root.exists():
        return []
    out = []
    for item in sorted(root.rglob("*")):
        try:
            st = item.lstat()
        except OSError:
            continue
        kind = "dir" if item.is_dir() and not item.is_symlink() else ("link" if item.is_symlink() else "file")
        out.append({"path": str(item.relative_to(root)), "kind": kind, "bytes": st.st_size, "mode": oct(st.st_mode & 0o777)})
    return out


def schema_inventory(roots):
    found = {}
    for _, root in roots:
        root = pathlib.Path(root)
        if not root.exists():
            continue
        for db in sorted(root.rglob("*.sqlite")):
            info = {"bytes": db.stat().st_size}
            try:
                conn = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True, timeout=2)
                try:
                    info["tables"] = [r[0] for r in conn.execute("select name from sqlite_master where type='table' order by name")]
                    info["user_version"] = conn.execute("pragma user_version").fetchone()[0]
                    info["journal_mode"] = conn.execute("pragma journal_mode").fetchone()[0]
                finally:
                    conn.close()
            except sqlite3.Error as error:
                info["error"] = str(error)
            found[str(db)] = info
    return found


def marker_hits(marker, roots):
    hits = []
    needle = marker.encode()
    for _, root in roots:
        root = pathlib.Path(root)
        if not root.exists():
            continue
        for item in sorted(root.rglob("*")):
            if item.is_file() and not item.is_symlink():
                try:
                    if needle in item.read_bytes():
                        hits.append(str(item))
                except OSError:
                    pass
    return hits


def stop_group(proc, log, first=None):
    """Graceful first (optional callable), then SIGINT, SIGTERM, SIGKILL to the group."""
    steps = []
    if first is not None:
        first()
        steps.append("graceful")
        try:
            proc.wait(timeout=12)
            return steps, proc.returncode
        except subprocess.TimeoutExpired:
            pass
    for name, sig, wait in (("SIGINT", signal.SIGINT, 8), ("SIGTERM", signal.SIGTERM, 6), ("SIGKILL", signal.SIGKILL, 6)):
        if proc.poll() is not None:
            break
        steps.append(name)
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
    return steps, proc.poll()


def base_options(port, sqlite_config):
    options = ["-c", 'model_provider="offline"', "-c", 'model="fake"',
               "-c", "features.enable_request_compression=false", "-c", "features.apps=false",
               "-c", ('model_providers.offline={name="offline",'
                      f'base_url="http://127.0.0.1:{port}",wire_api="responses"}}')]
    if sqlite_config is not None:
        options += ["-c", config_override("sqlite_home", sqlite_config)]
    return options


def run_app_server(cmd, cwd, env, case_dir, marker, model, result):
    stderr = open(case_dir / "stderr.log", "w")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=stderr, text=True, start_new_session=True)
    messages, notifications = queue.Queue(), []

    def read():
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                notifications.append({"raw": line[:200]})
                continue
            if "id" in msg and ("result" in msg or "error" in msg):
                messages.put(msg)
            else:
                notifications.append(msg)
    threading.Thread(target=read, daemon=True).start()

    def rpc(method, params, request_id, timeout=30):
        proc.stdin.write(json.dumps({"id": request_id, "method": method, "params": params}) + "\n")
        proc.stdin.flush()
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{method} timed out")
            msg = messages.get(timeout=remaining)
            if msg.get("id") == request_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg["result"]

    try:
        result["initialize"] = rpc("initialize", {"clientInfo": {"name": "sucoder_spike", "version": "1"},
                                                  "capabilities": {"experimentalApi": True}}, 1)
        proc.stdin.write('{"method":"initialized","params":{}}\n')
        proc.stdin.flush()
        thread = rpc("thread/start", {"cwd": str(cwd)}, 2)
        thread_id = thread.get("thread", {}).get("id")
        result["thread_id"] = thread_id
        rpc("turn/start", {"threadId": thread_id, "input": [{"type": "text", "text": marker}]}, 3)
        try:
            request = model.requests.get(timeout=30)
            texts = [part.get("text", "") for item in request.get("input", []) for part in item.get("content", []) if isinstance(part, dict)]
            result["model_saw_marker"] = any(marker in t for t in texts)
        except queue.Empty:
            result["model_saw_marker"] = None
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if any("turn" in (n.get("method") or "") and "complet" in (n.get("method") or "") for n in notifications):
                break
            time.sleep(0.25)
        time.sleep(2)
        result["fd_snapshot"] = fd_snapshot(descendants(proc.pid))
        try:
            result["thread_list"] = rpc("thread/list", {}, 4, timeout=15)
        except Exception as error:  # noqa: BLE001
            result["thread_list_error"] = str(error)
    except Exception as error:  # noqa: BLE001
        result["driver_error"] = f"{type(error).__name__}: {error}"
        if not result.get("fd_snapshot"):
            result["fd_snapshot"] = fd_snapshot(descendants(proc.pid))
    result["notification_methods"] = sorted({n.get("method") for n in notifications if n.get("method")})
    steps, code = stop_group(proc, case_dir, first=lambda: proc.stdin.close())
    result["stop_steps"], result["exit_code"] = steps, code
    stderr.close()


def run_foreground(cmd, cwd, env, case_dir, result, settle=20):
    """remote-control --no-daemon: no stdio protocol; observe, then stop."""
    out = open(case_dir / "stdout.log", "w")
    err = open(case_dir / "stderr.log", "w")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                            start_new_session=True)
    deadline = time.monotonic() + settle
    snapshots = []
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(4)
        snapshots.append(fd_snapshot(descendants(proc.pid)))
    result["exited_early"] = proc.poll() is not None
    result["fd_snapshot"] = max(snapshots, key=lambda s: sum(len(p["paths"]) for p in s)) if snapshots else []
    steps, code = stop_group(proc, case_dir)
    result["stop_steps"], result["exit_code"] = steps, code
    out.close()
    err.close()


def run_tui(cmd, cwd, env, case_dir, case, result, settle=25):
    sock = f"spike-{case}"
    tmux = ["tmux", "-L", sock]
    shell = "exec env " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " " + shlex.join(cmd)
    subprocess.run([*tmux, "new-session", "-d", "-s", "c", "-x", "180", "-y", "45", "-c", str(cwd), shell], check=True)
    time.sleep(settle)
    pane_pid = int(subprocess.run([*tmux, "display", "-p", "-t", "c", "#{pane_pid}"], capture_output=True, text=True).stdout.strip() or 0)
    result["fd_snapshot"] = fd_snapshot(descendants(pane_pid) + [pane_pid])
    result["pane_before_quit"] = subprocess.run([*tmux, "capture-pane", "-p", "-t", "c"], capture_output=True, text=True).stdout
    steps = []
    for keys, wait in ((["/quit", "Enter"], 6), (["C-c"], 3), (["C-c"], 3), (["C-d"], 3)):
        alive = subprocess.run([*tmux, "has-session", "-t", "c"], capture_output=True).returncode == 0
        if not alive:
            break
        subprocess.run([*tmux, "send-keys", "-t", "c", *keys], capture_output=True)
        steps.append(" ".join(keys))
        time.sleep(wait)
    alive = subprocess.run([*tmux, "has-session", "-t", "c"], capture_output=True).returncode == 0
    if alive:
        steps.append("kill-server")
        subprocess.run([*tmux, "kill-server"], capture_output=True)
    result["stop_steps"] = steps
    result["exit_code"] = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode", choices=["app-server", "remote-control", "tui"], default="app-server")
    parser.add_argument("--root", default=os.environ.get("SPIKE_ROOT") or os.path.join(
        os.environ.get("SUCODER_LOCAL_ROOT", "/tmp"), "spike-codex-state"))
    parser.add_argument("--sqlite-config", help="absolute or relative value passed as -c sqlite_home")
    parser.add_argument("--sqlite-env", help="value exported as CODEX_SQLITE_HOME")
    parser.add_argument("--sqlite-toml", help="sqlite_home written into the isolated config.toml")
    parser.add_argument("--no-create", action="store_true", help="do not pre-create the sqlite_home target")
    parser.add_argument("--unwritable", action="store_true", help="pre-create the sqlite_home target with mode 0500")
    parser.add_argument("--real-codex-home", default=str(pathlib.Path("~/.codex").expanduser().resolve()))
    parser.add_argument("--codex", default=shutil.which("codex"))
    args = parser.parse_args()

    os.umask(0o077)
    case_dir = pathlib.Path(args.root) / "cases" / args.case
    if case_dir.exists():
        raise SystemExit(f"case directory exists, refusing to reuse: {case_dir}")
    codex_home = case_dir / "codex-home"
    user_home = case_dir / "user-home"
    workspace = case_dir / "workspace"
    case_tmp = case_dir / "tmp"
    for d in (case_dir, codex_home, user_home, workspace, case_tmp):
        d.mkdir(parents=True, mode=0o700)
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    (workspace / "README").write_text("spike workspace\n")
    marker = f"SUCODER_SPIKE_MARKER_{args.case}_{int(time.time())}"

    toml = [f'[projects."{workspace}"]', 'trust_level = "trusted"']
    if args.sqlite_toml:
        toml.insert(0, f"sqlite_home = {json.dumps(args.sqlite_toml)}")
    (codex_home / "config.toml").write_text("\n".join(toml) + "\n")

    targets = [t for t in (args.sqlite_config, args.sqlite_env, args.sqlite_toml) if t]
    for t in targets:
        p = pathlib.Path(t)
        if not p.is_absolute() or args.no_create:
            continue
        if args.unwritable:
            p.mkdir(parents=True, mode=0o500, exist_ok=True)
            os.chmod(p, 0o500)
        else:
            p.mkdir(parents=True, mode=0o700, exist_ok=True)

    model = FakeModel()
    threading.Thread(target=model.serve_forever, daemon=True).start()
    env = {"PATH": os.environ["PATH"], "HOME": str(user_home), "CODEX_HOME": str(codex_home),
           "NO_PROXY": "127.0.0.1", "RUST_LOG": "error", "TERM": "xterm-256color",
           "LANG": "C.UTF-8", "TMPDIR": str(case_tmp)}
    if args.sqlite_env:
        env["CODEX_SQLITE_HOME"] = args.sqlite_env
    options = base_options(model.server_port, args.sqlite_config)
    strace_log = case_dir / "strace.log"
    strace = ["strace", "-f", "-y", "-s", "256", "-o", str(strace_log), "-e", STRACE_FILTER, "--"]
    if args.mode == "app-server":
        cmd = [*strace, args.codex, *options, "app-server"]
    elif args.mode == "remote-control":
        cmd = [*strace, args.codex, *options, "remote-control", "--no-daemon"]
    else:
        cmd = [*strace, args.codex, *options, "--cd", str(workspace), marker]

    result = {"case": args.case, "mode": args.mode, "host": os.uname().nodename,
              "job": os.environ.get("SLURM_JOB_ID"), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "codex": args.codex, "codex_version": subprocess.run([args.codex, "--version"], capture_output=True, text=True).stdout.strip(),
              "command": cmd, "env": {k: v for k, v in env.items() if k != "PATH"}, "marker": marker,
              "sqlite_config": args.sqlite_config, "sqlite_env": args.sqlite_env, "sqlite_toml": args.sqlite_toml,
              "config_toml": (codex_home / "config.toml").read_text()}
    if args.mode == "app-server":
        run_app_server(cmd, workspace, env, case_dir, marker, model, result)
    elif args.mode == "remote-control":
        run_foreground(cmd, workspace, env, case_dir, result)
    else:
        run_tui(cmd, workspace, env, case_dir, args.case, result)
    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    model.shutdown()
    model.server_close()

    roots = []
    for name, value in (("sqlite_config", args.sqlite_config), ("sqlite_env", args.sqlite_env), ("sqlite_toml", args.sqlite_toml)):
        if value and pathlib.Path(value).is_absolute():
            roots.append((name, value))
    roots += [("codex_home", str(codex_home)), ("user_home", str(user_home)), ("workspace", str(workspace)),
              ("case_tmp", str(case_tmp)), ("case_dir", str(case_dir)),
              ("REAL_CODEX_HOME", args.real_codex_home), ("REAL_HOME", os.environ["HOME"]),
              ("daemon_tmp", f"/tmp/codex-daemon-{os.getuid()}"), ("tmp", "/tmp"),
              ("local_root", os.environ.get("SUCODER_LOCAL_ROOT", "/local")), ("scratch", "/global/scratch"), ("global_home", "/global/home")]
    seen = parse_strace(strace_log)
    classified = {}
    for path, tags in seen.items():
        bucket = classified.setdefault(classify(path, roots), {})
        bucket[path] = sorted(tags)
    result["strace_classes"] = {k: {"paths": len(v), "sqlite_like": sorted(p for p in v if SQLITE_NAME.search(p))} for k, v in classified.items()}
    result["strace_paths"] = {k: v for k, v in classified.items() if k not in ("system", "install")}
    result["connects"] = sorted({p for p, t in seen.items() if "connect" in t or "bind" in t})
    result["files_after"] = {name: tree(root) for name, root in roots[:5]}
    result["schemas"] = schema_inventory(roots[:5])
    result["marker_hits"] = marker_hits(marker, roots[:5])
    (case_dir / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")

    print(f"== {args.case} ({args.mode}) exit={result.get('exit_code')} stop={result.get('stop_steps')} driver_error={result.get('driver_error')}")
    for name, info in result["strace_classes"].items():
        if name in ("system", "install"):
            continue
        print(f"  {name}: {info['paths']} paths; sqlite-like: {len(info['sqlite_like'])}")
        for p in info["sqlite_like"]:
            print(f"      {p}  {classified[name][p]}")
    print("  connects:", result["connects"])
    print("  live fds (sqlite-like):")
    for proc in result.get("fd_snapshot", []):
        for p in proc["paths"]:
            if SQLITE_NAME.search(p):
                print(f"      pid {proc['pid']} {proc['comm']}: {p}")
    print("  schemas:", {pathlib.Path(k).name: v.get("tables", v.get("error")) for k, v in result["schemas"].items()})
    print("  marker hits:", result["marker_hits"])
    print(f"  result: {case_dir / 'result.json'}")


if __name__ == "__main__":
    main()
