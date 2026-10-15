#!/usr/bin/env python3
"""Slice 0 outcomes 3, 4, 7 and the Codex-facing half of 5 (docs/codex-local-state.org).

Everything here measures Codex's own behavior with relocated SQLite state:
restore and the rollout dependency (3), abrupt-stop/capture interleavings (3),
queue and goal replay after restore (4), restart-in-place and two writers (5,
Codex side only), and local-versus-Lustre placement latency (7).  The runner's
transfer, claim and publication protocol does not exist yet, so outcome 6 and
the protocol half of 5 are out of scope.

Reuses the fake model, process helpers and option builder of
codex_sqlite_placement_spike.py.  The capture follows the SQLite online-backup
pattern of the 2026-10-01 recovery-tools/snapshot.py under the real Codex home
(per-database backup, quick_check, sha256, manifest); it never copies live
database files.  Every launch uses an isolated HOME and CODEX_HOME; nothing
touches the real account.  See docs/codex-local-state-spike.org.
"""
import argparse
import hashlib
import json
import os
import pathlib
import queue
import random
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve()
REPO = pathlib.Path(os.environ.get("SUCODER_REPO") or HERE.parents[1])
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent))
from codex_sqlite_placement_spike import FakeModel, descendants, fd_snapshot, base_options  # noqa: E402

MARKER = re.compile(r"SPKM_[A-Za-z0-9]+_[0-9]+")
QUIET = ["-c", "features.plugins=false", "-c", "features.remote_plugin=false"]
DATABASES = ["state_5.sqlite", "thread_history_1.sqlite", "goals_1.sqlite",
             "queue_1.sqlite", "memories_1.sqlite", "logs_2.sqlite"]
CODEX = shutil.which("codex")


def now():
    return time.monotonic()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, round((len(s) - 1) * p)))
    return s[k]


def stats(values):
    return {"n": len(values), "p50": percentile(values, 0.5), "p95": percentile(values, 0.95),
            "min": min(values) if values else None, "max": max(values) if values else None}


class Home:
    """An isolated CODEX_HOME plus HOME and TMPDIR under one private root."""

    def __init__(self, root, trusted=(), tmp=None):
        self.root = pathlib.Path(root)
        self.codex_home = self.root / "codex-home"
        self.user_home = self.root / "user-home"
        self.tmp = pathlib.Path(tmp) if tmp else self.root / "tmp"
        for d in (self.root, self.codex_home, self.user_home, self.tmp):
            d.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.trusted = list(trusted)
        self.write_config()

    def write_config(self, extra=()):
        lines = list(extra)
        for w in self.trusted:
            lines += [f'[projects."{w}"]', 'trust_level = "trusted"']
        (self.codex_home / "config.toml").write_text("\n".join(lines) + "\n")

    def env(self, **extra):
        env = {"PATH": os.environ["PATH"], "HOME": str(self.user_home), "CODEX_HOME": str(self.codex_home),
               "TMPDIR": str(self.tmp), "NO_PROXY": "127.0.0.1", "RUST_LOG": "error",
               "TERM": "xterm-256color", "LANG": "C.UTF-8"}
        env.update(extra)
        return env

    def rollouts(self, thread_id=None):
        pattern = f"rollout-*-{thread_id}.jsonl" if thread_id else "rollout-*.jsonl"
        return sorted((self.codex_home / "sessions").rglob(pattern)) if (self.codex_home / "sessions").exists() else []

    def copy_rollout_from(self, other, thread_id):
        copied = []
        for src in other.rollouts(thread_id):
            rel = src.relative_to(other.codex_home)
            dst = self.codex_home / rel
            dst.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            shutil.copyfile(src, dst)
            copied.append(str(rel))
        return copied


def make_workspace(path):
    path = pathlib.Path(path)
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    if not (path / ".git").exists():
        subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "README").write_text("spike workspace\n")
    return path


class Server:
    """One isolated foreground ``codex app-server`` driven over stdio."""

    def __init__(self, label, home, sqlite_home, cwd, quiet=True, extra_options=(), prefix=(), strace_log=None):
        self.label = label
        self.home = home
        self.sqlite_home = pathlib.Path(sqlite_home)
        self.sqlite_home.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.cwd = pathlib.Path(cwd)
        self.model = FakeModel()
        threading.Thread(target=self.model.serve_forever, daemon=True).start()
        self.model_seen = []
        threading.Thread(target=self._drain_model, daemon=True).start()
        options = base_options(self.model.server_port, str(self.sqlite_home)) + (QUIET if quiet else []) + list(extra_options)
        cmd = [CODEX, *options, "app-server"]
        if strace_log:
            cmd = ["strace", "-f", "-tt", "-y", "-s", "256", "-o", str(strace_log), "-e", "trace=%file,connect,bind,flock,fcntl", "--", *cmd]
        self.cmd = [*prefix, *cmd]
        self.notifications = []
        self.cond = threading.Condition()
        self.replies = queue.Queue()
        self.next_id = 100
        self.stderr = open(self.home.root / f"{label}.stderr.log", "a")
        self.started_at = now()
        self.proc = subprocess.Popen(self.cmd, cwd=str(self.cwd), env=home.env(), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.stderr, text=True, start_new_session=True)
        threading.Thread(target=self._read, daemon=True).start()
        self.rpc("initialize", {"clientInfo": {"name": "sucoder_spike", "version": "1"},
                                "capabilities": {"experimentalApi": True}})
        self._send({"method": "initialized", "params": {}})
        self.ready_at = now()

    def _drain_model(self):
        while True:
            try:
                request = self.model.requests.get(timeout=1)
            except queue.Empty:
                if getattr(self, "proc", None) is not None and self.proc.poll() is not None:
                    return
                continue
            self.model_seen.append({"t": now(), "markers": sorted(set(MARKER.findall(json.dumps(request))))})

    def _read(self):
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "id" in msg and ("result" in msg or "error" in msg):
                self.replies.put(msg)
            else:
                msg["_t"] = now()
                with self.cond:
                    self.notifications.append(msg)
                    self.cond.notify_all()

    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def rpc(self, method, params, timeout=60):
        self.next_id += 1
        rid = self.next_id
        self._send({"id": rid, "method": method, "params": params})
        deadline = now() + timeout
        while True:
            remaining = deadline - now()
            if remaining <= 0:
                raise TimeoutError(f"{method} timed out after {timeout}s")
            if self.proc.poll() is not None:
                raise RuntimeError(f"{method}: server exited with {self.proc.returncode}")
            try:
                msg = self.replies.get(timeout=min(remaining, 0.2))
            except queue.Empty:
                continue
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {json.dumps(msg['error'])[:300]}")
                return msg["result"]

    def wait_for(self, predicate, timeout, start_index=0):
        deadline = now() + timeout
        with self.cond:
            while True:
                for n in self.notifications[start_index:]:
                    if predicate(n):
                        return n
                remaining = deadline - now()
                if remaining <= 0 or self.proc.poll() is not None:
                    return None
                self.cond.wait(min(remaining, 0.2))

    def start_thread(self, cwd=None):
        t0 = now()
        result = self.rpc("thread/start", {"cwd": str(cwd or self.cwd)})
        return result["thread"]["id"], now() - t0

    def turn(self, thread_id, text, timeout=60):
        index = len(self.notifications)
        t0 = now()
        self.rpc("turn/start", {"threadId": thread_id, "input": [{"type": "text", "text": text}]})
        done = self.wait_for(lambda n: n.get("method") == "turn/completed"
                             and (n.get("params") or {}).get("threadId", thread_id) == thread_id, timeout, index)
        t1 = now()
        return {"text": text, "started": t0, "completed": t1 if done else None, "ok": done is not None,
                "duration": (t1 - t0) if done else None}

    def inspect(self, thread_id, cwd=None):
        """Every read path the protocol offers, with the markers each exposes."""
        out = {}

        def call(name, method, params):
            t0 = now()
            try:
                r = self.rpc(method, params, timeout=60)
                dumped = json.dumps(r)
                entry = {"ok": True, "seconds": now() - t0, "markers": sorted(set(MARKER.findall(dumped)))}
                if method == "thread/list":
                    ids = [t.get("id") for t in r.get("data", [])]
                    entry["found"] = thread_id in ids
                    for t in r.get("data", []):
                        if t.get("id") == thread_id:
                            entry["path"] = t.get("path")
                            entry["historyMode"] = t.get("historyMode")
                if method in ("thread/resume", "thread/read"):
                    entry["path"] = (r.get("thread") or {}).get("path")
                    entry["turns"] = len((r.get("thread") or {}).get("turns") or [])
                if method in ("thread/items/list", "thread/turns/list"):
                    entry["count"] = len(r.get("data", []))
            except Exception as error:  # noqa: BLE001
                entry = {"ok": False, "seconds": now() - t0, "error": f"{type(error).__name__}: {error}"[:300], "markers": []}
            out[name] = entry
            return entry

        call("list", "thread/list", {"limit": 100})
        call("list_state_db_only", "thread/list", {"limit": 100, "useStateDbOnly": True})
        call("read", "thread/read", {"threadId": thread_id, "includeTurns": True})
        resume = {"threadId": thread_id}
        if cwd:
            resume["cwd"] = str(cwd)
        call("resume", "thread/resume", resume)
        call("items", "thread/items/list", {"threadId": thread_id, "limit": 500})
        call("turns", "thread/turns/list", {"threadId": thread_id, "limit": 500})
        return out

    def model_markers(self):
        return sorted({m for r in self.model_seen for m in r["markers"]})

    def pids(self):
        return [p for p in fd_snapshot(descendants(self.proc.pid)) if p["comm"] == "codex"]

    def stop(self, timeout=15):
        steps = ["stdin-close"]
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            for name, sig, wait in (("SIGTERM", signal.SIGTERM, 10), ("SIGKILL", signal.SIGKILL, 10)):
                if self.proc.poll() is not None:
                    break
                steps.append(name)
                try:
                    os.killpg(self.proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    self.proc.wait(timeout=wait)
                except subprocess.TimeoutExpired:
                    continue
        self._close()
        return {"steps": steps, "exit": self.proc.returncode, "at": now()}

    def kill(self):
        t = now()
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        self._close()
        return {"steps": ["SIGKILL"], "exit": self.proc.returncode, "at": t}

    def _close(self):
        try:
            self.model.shutdown()
            self.model.server_close()
        except OSError:
            pass
        self.stderr.close()


def backup(source, destination, pages=64, sleep=0.005):
    """Per-database SQLite online backup with quick_check and sha256 (snapshot.py pattern)."""
    source, destination = pathlib.Path(source), pathlib.Path(destination)
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    manifest = {"source": str(source), "destination": str(destination), "started": now(), "wall": utc(), "databases": {}}
    for path in sorted(source.glob("*.sqlite")):
        rec = {"started": now()}
        tmp = destination / (path.name + ".incomplete")
        src = dst = None
        try:
            try:
                src = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=5)
                src.execute("select count(*) from sqlite_master").fetchone()
                rec["opened_as"] = "ro"
            except sqlite3.Error as error:
                rec["ro_error"] = str(error)
                if src is not None:
                    src.close()
                src = sqlite3.connect(str(path), timeout=5)
                rec["opened_as"] = "rw"
            dst = sqlite3.connect(str(tmp), timeout=5)
            os.chmod(tmp, 0o600)
            src.backup(dst, pages=pages, sleep=sleep)
            rec["quick_check"] = dst.execute("pragma quick_check").fetchall()[0][0]
            rec["journal_mode"] = dst.execute("pragma journal_mode").fetchone()[0]
        except Exception as error:  # noqa: BLE001
            rec["error"] = f"{type(error).__name__}: {error}"[:300]
        finally:
            for c in (dst, src):
                if c is not None:
                    c.close()
        if "error" not in rec:
            tmp.rename(destination / path.name)
            rec["bytes"] = (destination / path.name).stat().st_size
            rec["sha256"] = sha256(destination / path.name)
        rec["ok"] = rec.get("quick_check") == "ok" and "error" not in rec
        rec["finished"] = now()
        manifest["databases"][path.name] = rec
    manifest["finished"] = now()
    manifest["seconds"] = manifest["finished"] - manifest["started"]
    manifest["bytes"] = sum(r.get("bytes", 0) for r in manifest["databases"].values())
    manifest["complete"] = set(manifest["databases"]) >= set(DATABASES)
    manifest["ok"] = manifest["complete"] and all(r["ok"] for r in manifest["databases"].values())
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def restore(backup_dir, target):
    backup_dir, target = pathlib.Path(backup_dir), pathlib.Path(target)
    target.mkdir(parents=True, mode=0o700, exist_ok=False)
    manifest = json.loads((backup_dir / "manifest.json").read_text())
    t0 = now()
    copied, verified = [], True
    for name, rec in manifest["databases"].items():
        if not rec.get("ok"):
            continue
        shutil.copyfile(backup_dir / name, target / name)
        os.chmod(target / name, 0o600)
        copied.append(name)
        verified = verified and sha256(target / name) == rec["sha256"]
    return {"from": str(backup_dir), "to": str(target), "copied": copied, "verified": verified, "seconds": now() - t0}


def quick_check_all(directory):
    out = {}
    for path in sorted(pathlib.Path(directory).glob("*.sqlite")):
        try:
            c = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=5)
            out[path.name] = c.execute("pragma quick_check").fetchall()[0][0]
            c.close()
        except sqlite3.Error as error:
            out[path.name] = f"error: {error}"
    return out


def write_result(evidence, name, result):
    d = pathlib.Path(evidence) / name
    d.mkdir(parents=True, mode=0o700, exist_ok=True)
    result.setdefault("name", name)
    result.setdefault("host", os.uname().nodename)
    result.setdefault("job", os.environ.get("SLURM_JOB_ID"))
    result.setdefault("finished_utc", utc())
    (d / "result.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    return d / "result.json"


# ---------------------------------------------------------------- outcome 3a

def restore_matrix(args, root, evidence):
    """Restore the databases with the rollout present, copied, or absent."""
    base = pathlib.Path(root) / "restore"
    w1 = make_workspace(base / "w1")
    w2 = make_workspace(base / "w2")
    h1 = Home(base / "h1", trusted=[w1, w2])
    result = {"started_utc": utc(), "turns": args.turns, "cells": {}}
    s1 = Server("source", h1, base / "sqlite-A", w1)
    tid, _ = s1.start_thread()
    sent = []
    for i in range(args.turns):
        t = s1.turn(tid, f"SPKM_src_{i}")
        sent.append(t["text"])
    result["thread"] = tid
    result["sent"] = sent
    result["source_inspect_live"] = s1.inspect(tid)
    warm = backup(base / "sqlite-A", base / "backup-warm")
    result["stop"] = s1.stop()
    sealed = backup(base / "sqlite-A", base / "backup-sealed")
    result["backup_warm"] = {k: warm[k] for k in ("ok", "complete", "seconds", "bytes")}
    result["backup_sealed"] = {k: sealed[k] for k in ("ok", "complete", "seconds", "bytes")}
    result["rollout_files"] = [str(p.relative_to(h1.codex_home)) for p in h1.rollouts(tid)]
    result["source_wal_after_stop"] = sorted(p.name for p in (base / "sqlite-A").glob("*-wal"))
    result["rollout_path_in_db"] = result["source_inspect_live"]["resume"].get("path")
    rollouts = h1.rollouts(tid)
    rollout = rollouts[0] if rollouts else None
    original = base / "rollout-original.jsonl"
    if rollout:
        shutil.copyfile(rollout, original)
    hidden = rollout.with_name(rollout.name + ".hidden") if rollout else None

    # (name, home, backup, resume cwd, rollout at the absolute path stored in the database)
    cells = [
        ("original_home_sealed", "h1", "backup-sealed", None, "present"),
        ("original_home_warm", "h1", "backup-warm", None, "present"),
        ("original_home_new_cwd", "h1", "backup-sealed", w2, "present"),
        ("fresh_home_rollout_at_db_path", "fresh", "backup-sealed", None, "present"),
        ("fresh_home_rollout_copied_relative", "copy", "backup-sealed", None, "hidden"),
        ("fresh_home_rollout_absent", "fresh", "backup-sealed", None, "hidden"),
        ("original_home_rollout_absent", "h1", "backup-sealed", None, "hidden"),
    ]
    for name, home_spec, bdir, resume_cwd, rollout_state in cells:
        cell = {"backup": bdir, "rollout_at_db_path": rollout_state}
        cwd = w1
        if rollout:
            shutil.copyfile(original, rollout)  # reset: earlier cells' follow-up turns appended to it
            if rollout_state == "hidden":
                rollout.rename(hidden)
        if home_spec == "h1":
            home = h1
        else:
            home = Home(base / f"home-{name}", trusted=[w1, w2])
            if home_spec == "copy" and rollout:
                rel = rollout.relative_to(h1.codex_home)
                (home.codex_home / rel).parent.mkdir(parents=True, mode=0o700, exist_ok=True)
                shutil.copyfile(original, home.codex_home / rel)
                cell["rollout_copied"] = [str(rel)]
        cell["restore"] = restore(base / bdir, base / f"sqlite-{name}")
        try:
            s = Server(name, home, base / f"sqlite-{name}", cwd)
            cell["inspect"] = s.inspect(tid, cwd=resume_cwd)
            follow = s.turn(tid, f"SPKM_follow_{name.replace('_', '')}_0", timeout=60)
            cell["follow_up_turn"] = {"ok": follow["ok"], "duration": follow["duration"]}
            time.sleep(1)
            cell["model_saw"] = s.model_markers()
            cell["history_to_model"] = sorted(m for m in cell["model_saw"] if m in sent)
            cell["stop"] = s.stop()
            cell["ok"] = True
        except Exception as error:  # noqa: BLE001
            cell["ok"] = False
            cell["error"] = f"{type(error).__name__}: {error}"[:400]
        finally:
            if hidden and hidden.exists():
                hidden.rename(rollout)
        result["cells"][name] = cell
        ins = cell.get("inspect", {})
        print(f"  restore cell {name}: list={ins.get('list', {}).get('found')} "
              f"list_db={ins.get('list_state_db_only', {}).get('found')} "
              f"read={ins.get('read', {}).get('ok')}/{ins.get('read', {}).get('turns')} "
              f"resume={ins.get('resume', {}).get('ok')}/{ins.get('resume', {}).get('turns')} "
              f"items={ins.get('items', {}).get('count')} turns={ins.get('turns', {}).get('count')} "
              f"history_to_model={len(cell.get('history_to_model', []))}/{len(sent)} "
              f"err={str(ins.get('resume', {}).get('error', ''))[:60]!r}", flush=True)
    return write_result(evidence, "restore_matrix", result)


# ---------------------------------------------------------------- outcome 3b

def interleavings(args, root, evidence):
    """Seeded abrupt-stop/capture interleavings on local disk."""
    base = pathlib.Path(root) / "interleave"
    summary = {"started_utc": utc(), "seed": args.seed, "runs": [], "turns": args.turns}
    for i in range(args.runs):
        seed = args.seed * 1000 + i
        rng = random.Random(seed)
        run_dir = base / f"run{i:02d}"
        w = make_workspace(run_dir / "w")
        h = Home(run_dir / "h", trusted=[w])
        rec = {"run": i, "seed": seed}
        s = Server(f"run{i}", h, run_dir / "sqlite-A", w)
        tid, _ = s.start_thread()
        rec["thread"] = tid
        capture_turn, capture_frac = rng.randint(0, args.turns - 1), rng.random()
        kill_turn, kill_frac = rng.randint(capture_turn, args.turns - 1), rng.random()
        rec["plan"] = {"capture_turn": capture_turn, "capture_frac": capture_frac, "kill_turn": kill_turn, "kill_frac": kill_frac}
        capture = {}
        killed = {}
        median = 0.3

        def do_capture():
            try:
                capture.update(backup(run_dir / "sqlite-A", run_dir / "backup-warm", pages=8, sleep=0.002))
            except Exception as error:  # noqa: BLE001
                capture["error"] = f"{type(error).__name__}: {error}"[:300]
                capture["ok"] = False

        def do_kill():
            killed.update(s.kill())

        timers, turns, durations = [], [], []
        for k in range(args.turns):
            if k == capture_turn:
                t = threading.Timer(capture_frac * median, do_capture)
                t.start()
                timers.append(t)
            if k == kill_turn:
                t = threading.Timer(kill_frac * median, do_kill)
                t.start()
                timers.append(t)
            try:
                turn = s.turn(tid, f"SPKM_r{i}_{k}", timeout=20)
            except Exception as error:  # noqa: BLE001
                turn = {"text": f"SPKM_r{i}_{k}", "started": now(), "completed": None, "ok": False, "error": str(error)[:120]}
            turns.append(turn)
            if turn["ok"]:
                durations.append(turn["duration"])
                median = percentile(durations, 0.5) or median
            if killed:
                break
        for t in timers:
            t.join(timeout=30)
        if not killed:
            killed.update(s.kill())
        rec["turns"] = turns
        rec["killed_at"] = killed.get("at")
        rec["capture"] = {k: capture.get(k) for k in ("ok", "complete", "seconds", "bytes", "error")}
        rec["capture_db"] = {n: {k: r.get(k) for k in ("started", "finished", "ok", "quick_check", "opened_as", "error")}
                             for n, r in capture.get("databases", {}).items()}
        th_start = capture.get("databases", {}).get("thread_history_1.sqlite", {}).get("started")
        completed_before = [t["text"] for t in turns if t["completed"] and th_start and t["completed"] < th_start]
        completed_after = [t["text"] for t in turns if t["completed"] and th_start and t["completed"] >= th_start]
        never_completed = [t["text"] for t in turns if not t["completed"]]
        rec["sets"] = {"completed_before_capture": completed_before, "completed_after_capture": completed_after,
                       "never_completed": never_completed}
        try:
            crash = backup(run_dir / "sqlite-A", run_dir / "backup-crash")
            rec["crash_capture"] = {k: crash[k] for k in ("ok", "complete", "seconds", "bytes")}
            rec["crash_capture_db"] = {n: {k: r.get(k) for k in ("ok", "quick_check", "opened_as", "error")} for n, r in crash["databases"].items()}
        except Exception as error:  # noqa: BLE001
            rec["crash_capture"] = {"ok": False, "error": str(error)[:300]}
        rec["source_quick_check"] = quick_check_all(run_dir / "sqlite-A")
        rollouts = h.rollouts(tid)
        postkill = run_dir / "rollout-postkill.jsonl"
        if rollouts:
            shutil.copyfile(rollouts[0], postkill)
            rec["rollout_markers_after_kill"] = sorted(set(MARKER.findall(postkill.read_text(errors="replace"))))
        rec["restores"] = {}
        passed = True
        for label in ("warm", "crash"):
            bdir = run_dir / f"backup-{label}"
            r = {}
            if rollouts:
                shutil.copyfile(postkill, rollouts[0])  # undo the previous restore's follow-up turn
            for db in ("thread_history_1.sqlite", "state_5.sqlite"):
                if (bdir / db).exists():
                    r[f"markers_in_backup_{db.split('_')[0]}"] = sorted(set(m.decode() for m in re.findall(rb"SPKM_[A-Za-z0-9]+_[0-9]+", (bdir / db).read_bytes())))
            if not (bdir / "manifest.json").exists() or not json.loads((bdir / "manifest.json").read_text()).get("ok"):
                r["skipped"] = "capture not ok"
                rec["restores"][label] = r
                if label == "warm" and capture.get("ok") is False:
                    r["capture_error"] = capture.get("error")
                continue
            r["restore"] = restore(bdir, run_dir / f"sqlite-{label}")
            try:
                s2 = Server(f"run{i}-{label}", h, run_dir / f"sqlite-{label}", w)
                ins = s2.inspect(tid)
                r["inspect"] = {k: {kk: v.get(kk) for kk in ("ok", "markers", "error", "count", "turns", "found")} for k, v in ins.items()}
                visible = set(ins["resume"]["markers"]) | set(ins["items"]["markers"])
                r["visible"] = sorted(visible)
                r["missing_completed_before_capture"] = sorted(set(completed_before) - visible) if label == "warm" else sorted(set(completed_before + completed_after) - visible)
                r["phantom"] = sorted(visible - {t["text"] for t in turns})
                r["turns_items_agree"] = set(ins["turns"]["markers"]) == set(ins["items"]["markers"])
                r["turns_only"] = sorted(set(ins["turns"]["markers"]) - set(ins["items"]["markers"]))
                r["items_only"] = sorted(set(ins["items"]["markers"]) - set(ins["turns"]["markers"]))
                follow = s2.turn(tid, f"SPKM_f{i}{label}_0", timeout=30)
                r["follow_up_ok"] = follow["ok"]
                time.sleep(0.5)
                r["history_to_model"] = sorted(m for m in s2.model_markers() if m != f"SPKM_f{i}{label}_0")
                r["stop"] = s2.stop()
                r["pass"] = (not r["phantom"]) and (label != "warm" or not r["missing_completed_before_capture"]) and follow["ok"]
            except Exception as error:  # noqa: BLE001
                r["error"] = f"{type(error).__name__}: {error}"[:300]
                r["pass"] = False
            passed = passed and r.get("pass", False)
            rec["restores"][label] = r
        rec["pass"] = passed
        summary["runs"].append(rec)
        print(f"  interleave run {i:02d} seed={seed} kill@turn{kill_turn} capture@turn{capture_turn} "
              f"capture_ok={rec['capture']['ok']} crash_ok={rec['crash_capture'].get('ok')} "
              f"before={len(completed_before)} after={len(completed_after)} "
              f"warm_pass={rec['restores'].get('warm', {}).get('pass')} crash_pass={rec['restores'].get('crash', {}).get('pass')} "
              f"agree={rec['restores'].get('warm', {}).get('turns_items_agree')}/{rec['restores'].get('crash', {}).get('turns_items_agree')}", flush=True)
        write_result(evidence, "interleavings", summary)
        if passed and not args.keep:
            shutil.rmtree(run_dir, ignore_errors=True)
    summary["passed"] = sum(1 for r in summary["runs"] if r["pass"])
    return write_result(evidence, "interleavings", summary)


# ---------------------------------------------------------------- outcome 4

def replay(args, root, evidence):
    """Persisted queue entry and/or active goal across stop, capture, restore and resume."""
    result = {"started_utc": utc(), "watch_seconds": args.watch, "variants": {}}

    def queued(directory):
        try:
            c = sqlite3.connect(f"{(pathlib.Path(directory) / 'queue_1.sqlite').as_uri()}?mode=ro", uri=True, timeout=5)
            rows = c.execute("select id, queue_order from queued_items").fetchall()
            c.close()
            return rows
        except sqlite3.Error as error:
            return f"error: {error}"

    for variant in ("queue_only", "goal_only", "queue_and_goal", "goal_paused"):
        base = pathlib.Path(root) / "replay" / variant
        w = make_workspace(base / "w")
        h = Home(base / "h", trusted=[w])
        rec = {"variant": variant}
        s1 = Server(f"{variant}-src", h, base / "sqlite-A", w)
        tid, _ = s1.start_thread()
        s1.turn(tid, "SPKM_rp_0")
        if "goal" in variant:
            try:
                rec["goal_set"] = s1.rpc("thread/goal/set", {"threadId": tid, "objective": "SPKM_goal_1", "status": "active", "origin": "user"})
                if variant == "goal_paused":
                    rec["goal_paused"] = s1.rpc("thread/goal/set", {"threadId": tid, "status": "paused", "origin": "user"})
            except Exception as error:  # noqa: BLE001
                rec["goal_error"] = str(error)[:300]
        rec["stop_source"] = s1.stop()
        if "queue" in variant:
            q = subprocess.run([CODEX, "-c", f'sqlite_home="{base / "sqlite-A"}"', *QUIET, "queue", "--thread", tid, "--message", "SPKM_queued_1"],
                               env=h.env(), capture_output=True, text=True, timeout=60)
            rec["queue_cli"] = {"exit": q.returncode, "stdout": q.stdout.strip()[:200], "stderr": q.stderr.strip()[:200]}
        rec["queued_rows_source"] = queued(base / "sqlite-A")
        b = backup(base / "sqlite-A", base / "backup")
        rec["backup"] = {k: b[k] for k in ("ok", "complete", "seconds", "bytes")}
        rec["restore"] = restore(base / "backup", base / "sqlite-B")
        s2 = Server(f"{variant}-restored", h, base / "sqlite-B", w)
        index = len(s2.notifications)
        t0 = now()
        try:
            resumed = s2.rpc("thread/resume", {"threadId": tid})
            rec["resume"] = {"ok": True, "turns": len(resumed.get("thread", {}).get("turns") or [])}
        except Exception as error:  # noqa: BLE001
            rec["resume"] = {"ok": False, "error": str(error)[:300]}
        time.sleep(args.watch)
        watched = s2.notifications[index:]
        reqs = [m for m in s2.model_seen if m["t"] >= t0]
        buckets = {}
        for m in reqs:
            buckets[int(m["t"] - t0)] = buckets.get(int(m["t"] - t0), 0) + 1
        rec["after_resume"] = {
            "notifications": sorted({n.get("method") for n in watched}),
            "turn_started": sum(1 for n in watched if n.get("method") == "turn/started"),
            "turn_completed": sum(1 for n in watched if n.get("method") == "turn/completed"),
            "model_requests": len(reqs),
            "model_requests_per_second": [buckets.get(i, 0) for i in range(args.watch)],
            "model_markers": sorted({x for m in reqs for x in m["markers"]}),
            "queued_rows": queued(base / "sqlite-B"),
        }
        if "goal" in variant:
            try:
                rec["after_resume"]["goal"] = s2.rpc("thread/goal/get", {"threadId": tid}).get("goal")
            except Exception as error:  # noqa: BLE001
                rec["after_resume"]["goal_error"] = str(error)[:300]
        index2 = len(s2.notifications)
        n_before = len(s2.model_seen)
        client = s2.turn(tid, "SPKM_rp_1", timeout=60)
        time.sleep(2)
        after = s2.model_seen[n_before:]
        rec["after_client_turn"] = {
            "turn_ok": client["ok"],
            "model_requests": len(after),
            "model_markers": sorted({x for m in after for x in m["markers"]}),
            "queued_rows": queued(base / "sqlite-B"),
            "notifications": sorted({n.get("method") for n in s2.notifications[index2:]}),
        }
        rec["stop_restored"] = s2.stop()
        result["variants"][variant] = rec
        a = rec["after_resume"]
        print(f"  replay {variant}: resume={rec['resume'].get('ok')} unprompted turns={a['turn_started']} "
              f"model_requests={a['model_requests']} per_second={a['model_requests_per_second'][:10]} "
              f"queued {rec['queued_rows_source']}->{a['queued_rows']} markers={a['model_markers']} "
              f"goal_status={(a.get('goal') or {}).get('status')} client_turn_requests={rec['after_client_turn']['model_requests']}", flush=True)
        write_result(evidence, "replay", result)
    return write_result(evidence, "replay", result)


# ---------------------------------------------------------------- outcome 5, Codex side

def writers(args, root, evidence):
    base = pathlib.Path(root) / "writers"
    w = make_workspace(base / "w")
    h1 = Home(base / "h1", trusted=[w])
    result = {"started_utc": utc(), "cases": {}}
    s = Server("w-src", h1, base / "sqlite-A", w)
    tid, _ = s.start_thread()
    s.turn(tid, "SPKM_wr_0")
    result["thread"] = tid
    result["stop_src"] = s.stop()

    case = {}
    s2 = Server("w-restart", h1, base / "sqlite-A", w)
    case["inspect"] = {k: {kk: v.get(kk) for kk in ("ok", "markers", "error")} for k, v in s2.inspect(tid).items()}
    t = s2.turn(tid, "SPKM_wr_1")
    time.sleep(0.5)
    case["turn_ok"], case["history_to_model"] = t["ok"], [m for m in s2.model_markers() if m != "SPKM_wr_1"]
    case["stop"] = s2.stop()
    result["cases"]["restart_in_place"] = case

    for label, second_home in (("same_home_same_sqlite", h1), ("different_home_same_sqlite", None)):
        case = {}
        if second_home is None:
            second_home = Home(base / f"h-{label}", trusted=[w])
            case["rollout_copied"] = second_home.copy_rollout_from(h1, tid)
        a = Server(f"{label}-a", h1, base / "sqlite-A", w)
        b = Server(f"{label}-b", second_home, base / "sqlite-A", w)
        for name, srv in (("a", a), ("b", b)):
            try:
                srv.rpc("thread/resume", {"threadId": tid})
                case[f"resume_{name}"] = {"ok": True}
            except Exception as error:  # noqa: BLE001
                case[f"resume_{name}"] = {"ok": False, "error": str(error)[:300]}
        for name, srv in (("a", a), ("b", b)):
            try:
                t = srv.turn(tid, f"SPKM_{'sh' if 'same_home' in label else 'dh'}_{0 if name == 'a' else 1}", timeout=30)
                case[f"turn_{name}"] = {"ok": t["ok"], "duration": t["duration"]}
            except Exception as error:  # noqa: BLE001
                case[f"turn_{name}"] = {"ok": False, "error": str(error)[:300]}
        case["lock_files"] = sorted(str(p.relative_to(h1.codex_home)) for p in (h1.codex_home / "thread-writer-locks").glob("*.lock"))
        case["stop_a"], case["stop_b"] = a.stop(), b.stop()
        case["quick_check"] = quick_check_all(base / "sqlite-A")
        c = Server(f"{label}-check", h1, base / "sqlite-A", w)
        ins = c.inspect(tid)
        case["after_items"] = ins["items"].get("markers")
        case["after_turns"] = ins["turns"].get("markers")
        case["stop_check"] = c.stop()
        result["cases"][label] = case
        print(f"  writers {label}: resume_a={case['resume_a']['ok']} resume_b={case['resume_b']['ok']} "
              f"turn_a={case['turn_a'].get('ok')} turn_b={case['turn_b'].get('ok')} "
              f"b_error={case['resume_b'].get('error', case['turn_b'].get('error', ''))[:80]!r}", flush=True)
    return write_result(evidence, "writers", result)


# ---------------------------------------------------------------- sessions symlink

def sessions_symlink(args, root, evidence):
    """Is the rollout path stored as written (through a CODEX_HOME/sessions symlink)?"""
    base = pathlib.Path(root) / "symlink"
    w = make_workspace(base / "w")
    h = Home(base / "h", trusted=[w])
    target = base / "sessions-local-1"
    target.mkdir(mode=0o700)
    (h.codex_home / "sessions").symlink_to(target)
    result = {"started_utc": utc(), "symlink": str(h.codex_home / "sessions"), "target_1": str(target)}
    s = Server("symlink", h, base / "sqlite", w)
    tid, _ = s.start_thread()
    s.turn(tid, "SPKM_sym_0")
    listed = s.rpc("thread/list", {"limit": 10, "useStateDbOnly": True})
    path = [t["path"] for t in listed["data"] if t["id"] == tid][0]
    result["stored_path"] = path
    result["stored_through_symlink"] = str(h.codex_home / "sessions") in path
    result["stored_canonical"] = str(target) in path
    result["stop_1"] = s.stop()
    copy = base / "sessions-local-2"
    shutil.copytree(target, copy)
    (h.codex_home / "sessions").unlink()
    (h.codex_home / "sessions").symlink_to(copy)
    shutil.rmtree(target)
    result["target_2"] = str(copy)
    s2 = Server("symlink-retargeted", h, base / "sqlite", w)
    try:
        r = s2.rpc("thread/resume", {"threadId": tid})
        result["resume_after_retarget"] = {"ok": True, "turns": len(r["thread"]["turns"])}
        t = s2.turn(tid, "SPKM_sym_1")
        time.sleep(0.5)
        result["follow_up_ok"] = t["ok"]
        result["history_to_model"] = [m for m in s2.model_markers() if m != "SPKM_sym_1"]
    except Exception as error:  # noqa: BLE001
        result["resume_after_retarget"] = {"ok": False, "error": str(error)[:300]}
    result["stop_2"] = s2.stop()
    print(f"  symlink: stored through symlink={result['stored_through_symlink']} canonical={result['stored_canonical']} "
          f"resume after retarget={result['resume_after_retarget']} history_to_model={result.get('history_to_model')}", flush=True)
    return write_result(evidence, "sessions_symlink", result)


# ---------------------------------------------------------------- bind mount

def bwrap_prefix(binds):
    """Per-process binds of local directories over CODEX_HOME subdirectories; the shared home is untouched."""
    cmd = ["bwrap", "--dev-bind", "/", "/"]
    for local, target in binds.items():
        cmd += ["--bind", str(local), str(target)]
    return cmd + ["--unshare-user", "--uid", str(os.getuid()), "--gid", str(os.getgid()), "--die-with-parent", "--"]


def strace_window(log, start, end, local_prefixes=()):
    """Paths touched between two wall-clock instants (strace -tt, local time).

    A path under a bound mount point is physically local even though strace
    prints the mount-point path; such prefixes count as "/bound".
    """
    import datetime
    lo = datetime.datetime.fromtimestamp(start).strftime("%H:%M:%S.%f")
    hi = datetime.datetime.fromtimestamp(end).strftime("%H:%M:%S.%f")
    counts, paths = {}, {}
    local_prefixes = [str(x).rstrip("/") for x in local_prefixes]
    quoted = re.compile(r'"((?:/)[^"]*)"')
    fdpath = re.compile(r"<(/[^>]*?)(?: \(deleted\))?>")
    with open(log, errors="replace") as handle:
        for line in handle:
            m = re.match(r"^\d+\s+(\d\d:\d\d:\d\d\.\d+)\s+(.*)$", line)
            if not m or not (lo <= m.group(1) <= hi):
                continue
            body = m.group(2)
            for found in set(quoted.findall(body)) | set(fdpath.findall(body)):
                top = "/" + found.split("/")[1] if found.count("/") else found
                if any(found == x or found.startswith(x + "/") for x in local_prefixes):
                    top = "/bound"
                elif found.startswith("/global/scratch"):
                    top = "/global/scratch"
                elif found.startswith("/global/home"):
                    top = "/global/home"
                counts[top] = counts.get(top, 0) + 1
                if top.startswith("/global"):
                    paths[found] = paths.get(found, 0) + 1
    return counts, paths


def bind(args, root, evidence):
    """CODEX_HOME on Lustre, sessions/ bound to local disk per process: Lustre operations per turn."""
    if not shutil.which("bwrap"):
        raise SystemExit("bwrap not installed")
    if not args.lustre_root:
        raise SystemExit("--lustre-root is required for the bind test")
    base = pathlib.Path(root) / "bind" / args.bind_set
    w = make_workspace(base / "w")
    h = Home(pathlib.Path(args.lustre_root) / f"bind-home-{args.bind_set}", trusted=[w], tmp=base / "tmp")  # shared-home stand-in on Lustre, TMPDIR local
    subdirs = ["sessions"] if args.bind_set == "sessions" else ["sessions", ".tmp", "tmp", "thread-writer-locks", "shell_snapshots"]
    for d in subdirs:
        (h.codex_home / d).mkdir(mode=0o700, exist_ok=True)
    local_sessions = base / "sessions-1"
    binds = {}
    for d in subdirs:
        local = local_sessions if d == "sessions" else base / f"local-{d.strip('.')}-1"
        local.mkdir(parents=True, mode=0o700, exist_ok=True)
        binds[local] = h.codex_home / d
    result = {"started_utc": utc(), "bind_set": args.bind_set, "bound": {str(k): str(v) for k, v in binds.items()},
              "codex_home": str(h.codex_home), "local_sessions": str(local_sessions), "tmpdir": str(h.tmp)}
    s = Server("bind", h, base / "sqlite-A", w, prefix=bwrap_prefix(binds), strace_log=base / "strace.log")
    tid, _ = s.start_thread()
    s.turn(tid, "SPKM_bind_0")
    t0 = time.time()
    turns = [s.turn(tid, f"SPKM_bind_{k}") for k in range(1, args.turns + 1)]
    t1 = time.time()
    result["turns_ok"] = sum(1 for t in turns if t["ok"])
    result["turn_p50"] = percentile([t["duration"] for t in turns if t["ok"]], 0.5)
    listed = s.rpc("thread/list", {"limit": 10, "useStateDbOnly": True})
    result["stored_path"] = [t["path"] for t in listed["data"] if t["id"] == tid][0]
    result["rollout_physical"] = [str(p.relative_to(local_sessions)) for p in local_sessions.rglob("*.jsonl")]
    result["shared_sessions_outside"] = [str(p.relative_to(h.codex_home / "sessions")) for p in (h.codex_home / "sessions").rglob("*")]
    result["fd_sqlite_like"] = sorted(x for proc in s.pids() for x in proc["paths"] if ".sqlite" in x or ".jsonl" in x or ".lock" in x)
    result["stop"] = s.stop()
    time.sleep(1)
    counts, paths = strace_window(base / "strace.log", t0, t1, local_prefixes=list(binds.values()))
    result["syscalls_per_toplevel_during_turns"] = counts
    result["global_paths_during_turns"] = paths
    result["lustre_ops_per_turn"] = (counts.get("/global/scratch", 0)) / max(1, args.turns)
    home = str(h.codex_home)
    groups = {}
    for pth, n in paths.items():
        if pth == home or home.startswith(pth + "/"):
            g = "ancestors of CODEX_HOME"
        elif pth.startswith(home + "/"):
            g = "CODEX_HOME/" + pth[len(home) + 1:].split("/")[0]
        else:
            g = "other"
        groups[g] = groups.get(g, 0) + n
    result["lustre_groups_during_turns"] = groups
    b = backup(base / "sqlite-A", base / "backup")
    result["backup"] = {k: b[k] for k in ("ok", "complete", "seconds", "bytes")}
    # "new allocation": restored databases and a copy of the local sessions directory bound at the same path
    binds2 = {}
    for local, target in binds.items():
        local2 = pathlib.Path(str(local)[:-2] + "-2")
        shutil.copytree(local, local2)
        shutil.rmtree(local)
        binds2[local2] = target
    result["restore"] = restore(base / "backup", base / "sqlite-B")
    s2 = Server("bind-restored", h, base / "sqlite-B", w, prefix=bwrap_prefix(binds2))
    try:
        r = s2.rpc("thread/resume", {"threadId": tid})
        result["resume_new_allocation"] = {"ok": True, "turns": len(r["thread"]["turns"])}
        t = s2.turn(tid, "SPKM_bindf_0")
        time.sleep(0.5)
        result["follow_up_ok"] = t["ok"]
        result["history_to_model"] = sorted(m for m in s2.model_markers() if m != "SPKM_bindf_0")
    except Exception as error:  # noqa: BLE001
        result["resume_new_allocation"] = {"ok": False, "error": str(error)[:300]}
    result["stop_2"] = s2.stop()
    result["shared_sessions_outside_after"] = [str(p.relative_to(h.codex_home / "sessions")) for p in (h.codex_home / "sessions").rglob("*")]
    print(f"  bind [{args.bind_set}]: stored_path under CODEX_HOME={str(h.codex_home) in result['stored_path']} rollout_physical={len(result['rollout_physical'])} "
          f"shared_sessions_outside={result['shared_sessions_outside']} lustre_ops_per_turn={result['lustre_ops_per_turn']:.1f} "
          f"groups={groups} per_toplevel={ {k: v for k, v in counts.items() if k in ('/bound', '/local', '/global/scratch', '/global/home', '/global')} } "
          f"resume_new_alloc={result['resume_new_allocation']} history={len(result.get('history_to_model', []))}/{args.turns + 1}", flush=True)
    print("  lustre paths during turns:", dict(sorted(((p, n) for p, n in paths.items() if p.startswith('/global/scratch')), key=lambda kv: -kv[1])[:8]), flush=True)
    return write_result(evidence, "bind", result)


# ---------------------------------------------------------------- outcome 7

def latency(args, root, evidence):
    # arm -> (sqlite placement root, CODEX_HOME placement root); "local" and "lustre"
    # keep the home on Lustre as production does and move only the databases;
    # "local_home" moves both, to show what the rollout traffic costs.
    local_root = pathlib.Path(root)
    lustre_root = pathlib.Path(args.lustre_root) if args.lustre_root else None
    arms = {}
    if "local" in args.placements:
        arms["local"] = (local_root / "lat", lustre_root / "lat-homes" if lustre_root else local_root / "lat-homes")
    if "lustre" in args.placements:
        if not lustre_root:
            raise SystemExit("--lustre-root is required for the lustre placement")
        arms["lustre"] = (lustre_root / "lat", lustre_root / "lat-homes")
    if "local_home" in args.placements:
        arms["local_home"] = (local_root / "lat-lh", local_root / "lat-homes")
    placements = {k: v[0] for k, v in arms.items()}
    turns_n = args.latency_turns
    result = {"started_utc": utc(), "turns": turns_n, "reps": args.reps, "arms": {k: {"sqlite": str(v[0]), "home": str(v[1])} for k, v in arms.items()}, "runs": []}

    def one_run(arm, rep, tag, target_dir):
        w = make_workspace(pathlib.Path(root) / "lat-ws" / f"{arm}-{rep}-{tag}")
        h = Home(arms[arm][1] / f"{arm}-{rep}-{tag}", trusted=[w])
        rec = {"arm": arm, "rep": rep, "tag": tag, "sqlite_home": str(target_dir), "codex_home": str(h.codex_home), "wall": utc()}
        t0 = now()
        s = Server(f"lat-{arm}-{rep}-{tag}", h, target_dir, w)
        rec["startup_seconds"] = s.ready_at - t0
        tid, rec["thread_start_seconds"] = s.start_thread()
        s.turn(tid, "SPKM_warm_0")
        durations = []
        for k in range(turns_n):
            t = s.turn(tid, f"SPKM_l{rep}_{k}", timeout=120)
            if t["ok"]:
                durations.append(t["duration"])
        rec["turn_seconds"] = stats(durations)
        rec["turn_failures"] = turns_n - len(durations)
        ts = now()
        s.stop()
        rec["stop_seconds"] = now() - ts
        b = backup(target_dir, pathlib.Path(root) / "lat-backups" / f"{arm}-{rep}-{tag}")
        rec["backup"] = {k: b[k] for k in ("ok", "seconds", "bytes")}
        restored_dir = pathlib.Path(target_dir).parent / (pathlib.Path(target_dir).name + "-restored")
        rec["restore"] = restore(pathlib.Path(root) / "lat-backups" / f"{arm}-{rep}-{tag}", restored_dir)
        t1 = now()
        s2 = Server(f"lat-{arm}-{rep}-{tag}-r", h, restored_dir, w)
        t2 = now()
        s2.rpc("thread/list", {"limit": 10})
        rec["restart_to_first_list_seconds"] = now() - t1
        t3 = now()
        s2.rpc("thread/resume", {"threadId": tid})
        rec["resume_seconds"] = now() - t3
        rec["restart_startup_seconds"] = t2 - t1
        s2.stop()
        return rec

    order = [a for a in ("local", "lustre", "local_home") if a in placements]
    for rep in range(args.reps):
        sequence = order if rep % 2 == 0 else list(reversed(order))
        for arm in sequence:
            rec = one_run(arm, rep, "single", placements[arm] / f"single-{rep}")
            result["runs"].append(rec)
            print(f"  latency {arm} rep{rep} single: turn p50={rec['turn_seconds']['p50']:.3f} p95={rec['turn_seconds']['p95']:.3f} "
                  f"start={rec['startup_seconds']:.2f} backup={rec['backup']['seconds']:.2f}s/{rec['backup']['bytes']}B "
                  f"restart+list={rec['restart_to_first_list_seconds']:.2f} resume={rec['resume_seconds']:.3f}", flush=True)
            write_result(evidence, "latency", result)
    if args.concurrency > 1:
        for arm in order:
            threads, recs = [], [None] * args.concurrency

            def run_one(j, arm=arm):
                recs[j] = one_run(arm, 0, f"conc{j}", placements[arm] / f"conc-{j}")
            for j in range(args.concurrency):
                t = threading.Thread(target=run_one, args=(j,))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            for rec in recs:
                if rec:
                    rec["concurrency"] = args.concurrency
                    result["runs"].append(rec)
                    print(f"  latency {arm} concurrent x{args.concurrency} {rec['tag']}: turn p50={rec['turn_seconds']['p50']:.3f} p95={rec['turn_seconds']['p95']:.3f}", flush=True)
            write_result(evidence, "latency", result)
    summary = {}
    for arm in placements:
        for conc in sorted({r.get("concurrency", 1) for r in result["runs"]}):
            vals = [r for r in result["runs"] if r["arm"] == arm and r.get("concurrency", 1) == conc]
            if vals:
                summary[f"{arm}_x{conc}"] = {"runs": len(vals), "turn_p50": [r["turn_seconds"]["p50"] for r in vals],
                                            "turn_p95": [r["turn_seconds"]["p95"] for r in vals],
                                            "startup": [r["startup_seconds"] for r in vals],
                                            "backup_seconds": [r["backup"]["seconds"] for r in vals],
                                            "restart_to_first_list": [r["restart_to_first_list_seconds"] for r in vals],
                                            "resume": [r["resume_seconds"] for r in vals]}
    result["summary"] = summary
    return write_result(evidence, "latency", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tests", nargs="+", choices=["restore", "interleave", "replay", "writers", "latency", "symlink", "bind", "all"])
    parser.add_argument("--root", default=os.environ.get("SPIKE_ROOT") or os.path.join(os.environ.get("SUCODER_LOCAL_ROOT", "/tmp"), "spike-codex-outcomes"))
    parser.add_argument("--evidence", required=True, help="durable directory for per-test result.json files")
    parser.add_argument("--lustre-root", default=os.environ.get("SUCODER_SHARED_STATE_TEST_ROOT"))
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--latency-turns", type=int, default=30)
    parser.add_argument("--runs", type=int, default=20, help="interleavings")
    parser.add_argument("--reps", type=int, default=3, help="latency repetitions per placement")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--placements", default="local,lustre")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--watch", type=int, default=30, help="seconds to watch for unprompted activity after resume")
    parser.add_argument("--keep", action="store_true", help="keep passing interleaving directories")
    parser.add_argument("--bind-set", choices=["sessions", "full"], default="full", help="which CODEX_HOME subdirectories the bind test binds to local disk")
    args = parser.parse_args()
    args.placements = [p for p in args.placements.split(",") if p]
    os.umask(0o077)
    root = pathlib.Path(args.root)
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    evidence = pathlib.Path(args.evidence)
    evidence.mkdir(parents=True, mode=0o700, exist_ok=True)
    (evidence / "environment.json").write_text(json.dumps({
        "host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID"), "utc": utc(), "codex": CODEX,
        "codex_version": subprocess.run([CODEX, "--version"], capture_output=True, text=True).stdout.strip(),
        "args": vars(args), "root": str(root)}, indent=1) + "\n")
    tests = ["restore", "interleave", "replay", "writers", "symlink", "latency"] if "all" in args.tests else args.tests
    for name in tests:
        print(f"== {name} {utc()}", flush=True)
        t0 = now()
        try:
            path = {"restore": restore_matrix, "interleave": interleavings, "replay": replay,
                    "writers": writers, "latency": latency, "symlink": sessions_symlink, "bind": bind}[name](args, root, evidence)
            print(f"== {name} done in {now() - t0:.0f}s -> {path}", flush=True)
        except Exception as error:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            write_result(evidence, name, {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()[-2000:]})
            print(f"== {name} FAILED: {error}", flush=True)


if __name__ == "__main__":
    main()
