#!/usr/bin/env python3
"""Lustre-episode tests for the rollout-local layout (docs/codex-local-state-spike.org).

The human's concern is Lustre's episodes of minute-long latency, so the
managed server must contain no synchronous Lustre operation in a turn.  These
tests measure the per-process bind-mount layout from codex_state_outcomes.py:
rollout skew between databases and rollout (the crash case), a per-turn tail
replication of the rollout to shared storage, the residue of Lustre
operations per turn, the TUI and remote-control modes under bwrap,
compaction, and whether the residual metadata calls are load-bearing.
Every launch uses an isolated home and the loopback fake model; nothing
touches the real account.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shlex
import shutil
import sqlite3
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import codex_state_outcomes as o  # noqa: E402
from codex_sqlite_placement_spike import fd_snapshot, descendants  # noqa: E402

FULL = ["sessions", ".tmp", "tmp", "thread-writer-locks", "shell_snapshots"]
STRACE = ["strace", "-f", "-tt", "-y", "-s", "256", "-e", "trace=%file,connect,bind,flock,fcntl"]


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def rows(sqlite_dir):
    out = {}
    for db, tables in (("thread_history_1.sqlite", ("thread_turns", "thread_items")), ("state_5.sqlite", ("threads",))):
        try:
            c = sqlite3.connect(f"{(pathlib.Path(sqlite_dir) / db).as_uri()}?mode=ro", uri=True, timeout=5)
            for t in tables:
                out[f"{db}:{t}"] = c.execute(f"select count(*) from {t}").fetchone()[0]
            c.close()
        except sqlite3.Error as error:
            out[db] = f"error: {error}"
    return out


def rollout_of(local_sessions, tid):
    found = sorted(pathlib.Path(local_sessions).rglob(f"rollout-*-{tid}.jsonl"))
    return found[0] if found else None


class Bound:
    """A shared-home stand-in on Lustre with local directories bound over its subdirectories."""

    def __init__(self, root, lustre_root, name, subdirs=FULL, extra_files=(), skills_copy=False):
        self.name = name
        self.local = pathlib.Path(root) / name
        self.local.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.workspace = o.make_workspace(self.local / "w")
        self.home = o.Home(pathlib.Path(lustre_root) / f"home-{name}", trusted=[self.workspace], tmp=self.local / "tmp")
        self.home.user_home = self.local / "user-home"            # production HOME is NFS, not Lustre
        self.home.user_home.mkdir(mode=0o700, exist_ok=True)
        self.binds = {}
        for d in subdirs:
            (self.home.codex_home / d).mkdir(mode=0o700, exist_ok=True)
            local = self.local / f"bound-{d.strip('.')}"
            local.mkdir(mode=0o700, exist_ok=True)
            self.binds[local] = self.home.codex_home / d
        if skills_copy:
            (self.home.codex_home / "skills").mkdir(mode=0o700, exist_ok=True)
            local = self.local / "bound-skills"
            if local.exists():
                shutil.rmtree(local)
            shutil.copytree(self.home.codex_home / "skills", local)
            self.binds[local] = self.home.codex_home / "skills"
        for f in extra_files:                                     # a file bound over a file in the home
            target = self.home.codex_home / f
            if not target.exists():
                target.write_text("")
            local = self.local / f"bound-file-{f}"
            if not local.exists():
                shutil.copyfile(target, local)
            self.binds[local] = target
        self.sessions = [k for k, v in self.binds.items() if v.name == "sessions"][0]

    def prefix(self):
        return o.bwrap_prefix(self.binds)

    def server(self, label, sqlite_dir, strace_log=None):
        return o.Server(label, self.home, sqlite_dir, self.workspace, prefix=self.prefix(), strace_log=strace_log)

    def rebind(self, sessions_dir):
        """A new Bound sharing the home but with another local sessions directory (a new allocation)."""
        other = Bound.__new__(Bound)
        other.__dict__.update(self.__dict__)
        other.binds = dict(self.binds)
        del other.binds[self.sessions]
        other.binds[pathlib.Path(sessions_dir)] = self.home.codex_home / "sessions"
        other.sessions = pathlib.Path(sessions_dir)
        return other


def lustre_residue(strace_log, t0, t1, binds, codex_home):
    counts, paths = o.strace_window(strace_log, t0, t1, local_prefixes=list(binds.values()))
    home = str(codex_home)
    scratch = {p: n for p, n in paths.items() if p.startswith("/global/scratch")}
    ancestors = {p: n for p, n in scratch.items() if p == home or home.startswith(p + "/")}
    inside = {p.replace(home, "CODEX_HOME"): n for p, n in scratch.items() if p.startswith(home + "/")}
    other = {p: n for p, n in scratch.items() if p not in ancestors and not p.startswith(home + "/")}
    return {"per_toplevel": {k: v for k, v in counts.items() if k in ("/bound", "/local", "/global/scratch", "/global/home", "/global/software", "/global")},
            "lustre_total": sum(scratch.values()), "ancestors_total": sum(ancestors.values()), "ancestor_depth": home.count("/"),
            "inside_home": dict(sorted(inside.items(), key=lambda kv: -kv[1])), "other": other}


def inspect_cell(bound, label, sqlite_dir, tid, result):
    s = bound.server(label, sqlite_dir)
    cell = {"rows_before": rows(sqlite_dir)}
    cell["inspect"] = {k: {kk: v.get(kk) for kk in ("ok", "error", "markers", "turns", "count", "found")} for k, v in s.inspect(tid).items()}
    try:
        t = s.turn(tid, f"SPKM_{label.replace('_', '')}_9", timeout=60)
        time.sleep(0.5)
        cell["follow_up"] = {"ok": t["ok"], "history_to_model": sorted(m for m in s.model_markers() if not m.endswith("_9"))}
    except Exception as error:  # noqa: BLE001
        cell["follow_up"] = {"ok": False, "error": str(error)[:300]}
    cell["stop"] = s.stop()
    cell["rows_after"] = rows(sqlite_dir)
    result[label] = cell
    ins = cell["inspect"]
    print(f"  {label}: resume={ins['resume']['ok']} read={ins['read']['ok']} items={ins['items']['ok']} turns={ins['turns']['ok']} "
          f"list={ins['list'].get('found')}/{ins['list_state_db_only'].get('found')} follow_up={cell['follow_up'].get('ok')} "
          f"history={cell['follow_up'].get('history_to_model')} rows {cell['rows_before']} -> {cell['rows_after']} "
          f"err={str(ins['resume'].get('error') or ins['items'].get('error') or '')[:90]!r}", flush=True)
    return cell


# ------------------------------------------------------------------ 4. rollout skew

def skew(args, root, lustre, evidence):
    b = Bound(root, lustre, "skew")
    result = {"started_utc": o.utc(), "turns": args.turns, "cells": {}}
    s = b.server("skew-src", b.local / "sqlite-A")
    tid, _ = s.start_thread()
    snaps = {}
    for k in range(1, args.turns + 1):
        s.turn(tid, f"SPKM_sk_{k}")
        time.sleep(0.5)
        r = rollout_of(b.sessions, tid)
        snap = b.local / f"rollout-after-{k}.jsonl"
        shutil.copyfile(r, snap)
        o.backup(b.local / "sqlite-A", b.local / f"backup-after-{k}")
        snaps[k] = {"rollout_bytes": snap.stat().st_size, "rollout_lines": sum(1 for _ in open(snap, errors="replace")), "rows": rows(b.local / f"backup-after-{k}")}
    result["thread"], result["snapshots"] = tid, snaps
    result["stop_source"] = s.stop()
    rel = rollout_of(b.sessions, tid).relative_to(b.sessions)
    N = args.turns
    cells = {"db_ahead_of_rollout": (N, N - 2, None), "rollout_truncated_mid_line": (N, N, "truncate"), "rollout_ahead_of_db": (2, N, None)}
    for label, (db_k, ro_k, mode) in cells.items():
        sessions = b.local / f"sessions-{label}"
        (sessions / rel).parent.mkdir(parents=True, mode=0o700)
        shutil.copyfile(b.local / f"rollout-after-{ro_k}.jsonl", sessions / rel)
        if mode == "truncate":
            data = (sessions / rel).read_bytes()
            last_nl = data.rstrip(b"\n").rfind(b"\n")
            cut = last_nl + 1 + (len(data) - last_nl - 1) // 2          # halfway through the last line
            (sessions / rel).write_bytes(data[:cut])
        before = {"rollout_bytes": (sessions / rel).stat().st_size, "rollout_sha": sha(sessions / rel)}
        o.restore(b.local / f"backup-after-{db_k}", b.local / f"sqlite-{label}")
        cell = inspect_cell(b.rebind(sessions), label, b.local / f"sqlite-{label}", tid, result["cells"])
        cell["db_from_turn"], cell["rollout_from_turn"] = db_k, ro_k
        cell["rollout_before"] = before
        cell["rollout_after"] = {"rollout_bytes": (sessions / rel).stat().st_size, "rollout_sha": sha(sessions / rel), "changed": sha(sessions / rel) != before["rollout_sha"]}
        cell["new_files_in_sessions"] = sorted(str(p.relative_to(sessions)) for p in sessions.rglob("*") if p.is_file() and p != sessions / rel)

    # database ahead, extended: does the projection ever catch up, and what does it hold afterwards?
    def projection(sqlite_dir):
        c = sqlite3.connect(f"{(pathlib.Path(sqlite_dir) / 'thread_history_1.sqlite').as_uri()}?mode=ro", uri=True, timeout=5)
        row = c.execute("select next_rollout_byte_offset, next_rollout_ordinal from thread_history_projection_state where thread_id = ?", (tid,)).fetchone()
        turns = c.execute("select count(*) from thread_turns").fetchone()[0]
        c.close()
        return {"next_rollout_byte_offset": row[0] if row else None, "next_rollout_ordinal": row[1] if row else None, "thread_turns": turns}
    sessions = b.local / "sessions-db_ahead_extended"
    (sessions / rel).parent.mkdir(parents=True, mode=0o700)
    shutil.copyfile(b.local / f"rollout-after-{N - 2}.jsonl", sessions / rel)
    o.restore(b.local / f"backup-after-{N}", b.local / "sqlite-db_ahead_extended")
    ext = {"rollout_bytes_restored": (sessions / rel).stat().st_size, "steps": []}
    bb = b.rebind(sessions)
    s = bb.server("db_ahead_extended", b.local / "sqlite-db_ahead_extended")
    s.rpc("thread/resume", {"threadId": tid})
    ext["steps"].append({"step": "after resume", **projection(b.local / "sqlite-db_ahead_extended"), "rollout_bytes": (sessions / rel).stat().st_size})
    for k in (10, 11, 12):
        t = s.turn(tid, f"SPKM_dbx_{k}")
        time.sleep(0.3)
        ext["steps"].append({"step": f"after turn {k}", "ok": t["ok"], "model_markers": sorted(s.model_seen[-1]["markers"]) if s.model_seen else None,
                             **projection(b.local / "sqlite-db_ahead_extended"), "rollout_bytes": (sessions / rel).stat().st_size})
    s.stop()
    s2 = bb.server("db_ahead_extended_restart", b.local / "sqlite-db_ahead_extended")
    s2.rpc("thread/resume", {"threadId": tid})
    t = s2.turn(tid, "SPKM_dbx_13")
    time.sleep(0.5)
    ins = s2.inspect(tid)
    ext["after_restart"] = {"turn_ok": t["ok"], "model_markers": sorted(s2.model_seen[-1]["markers"]) if s2.model_seen else None,
                            "read_paths": {k: sorted(v.get("markers", [])) for k, v in ins.items() if k in ("resume", "items", "turns", "read")},
                            **projection(b.local / "sqlite-db_ahead_extended")}
    s2.stop()
    result["db_ahead_extended"] = ext
    print("  db_ahead_extended:", [(x["step"], x["thread_turns"], x["next_rollout_byte_offset"], x["rollout_bytes"]) for x in ext["steps"]],
          "| after restart: model", ext["after_restart"]["model_markers"], "| items", ext["after_restart"]["read_paths"]["items"], flush=True)
    return o.write_result(evidence, "skew", result)


# ------------------------------------------------------------------ 6. tail replication

def replicate(args, root, lustre, evidence):
    b = Bound(root, lustre, "replicate")
    replica = pathlib.Path(lustre) / "replica" / "rollout.jsonl"
    replica.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    replica.write_bytes(b"")
    result = {"started_utc": o.utc(), "turns": args.turns, "replica": str(replica), "per_turn": []}
    s = b.server("replicate-src", b.local / "sqlite-A")
    tid, _ = s.start_thread()
    offset = 0
    for k in range(1, args.turns + 1):
        local = rollout_of(b.sessions, tid)
        same_at_start = sha(local) == sha(replica) if local else None       # the honest claim: identical at the next turn/start
        t = s.turn(tid, f"SPKM_rep_{k}")
        local = rollout_of(b.sessions, tid)
        t0 = o.now()
        with open(local, "rb") as src, open(replica, "ab") as dst:            # append only the new bytes
            src.seek(offset)
            chunk = src.read()
            dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        offset += len(chunk)
        rec = {"turn": k, "turn_seconds": t["duration"], "bytes_appended": len(chunk), "replicate_seconds": o.now() - t0,
               "identical_immediately": sha(local) == sha(replica), "identical_at_previous_turn_start": same_at_start}
        result["per_turn"].append(rec)
        if k in (2, 4):
            o.backup(b.local / "sqlite-A", b.local / f"gen-after-{k}")
    time.sleep(0.5)
    local = rollout_of(b.sessions, tid)
    with open(local, "rb") as src, open(replica, "ab") as dst:
        src.seek(offset)
        dst.write(src.read())
    result["final_identical"] = sha(local) == sha(replica)
    result["stop_source"] = s.stop()
    rel = local.relative_to(b.sessions)
    sessions2 = b.local / "sessions-from-replica"
    (sessions2 / rel).parent.mkdir(parents=True, mode=0o700)
    shutil.copyfile(replica, sessions2 / rel)
    o.restore(b.local / "gen-after-2", b.local / "sqlite-gen2")                 # a generation two intervals old
    result["cells"] = {}
    inspect_cell(b.rebind(sessions2), "replica_plus_gen_two_intervals_old", b.local / "sqlite-gen2", tid, result["cells"])
    stats = result["per_turn"]
    print(f"  replicate: identical immediately {sum(1 for x in stats if x['identical_immediately'])}/{len(stats)}, "
          f"at next turn/start {sum(1 for x in stats if x['identical_at_previous_turn_start'])}/{len(stats) - 1}, final {result['final_identical']}; "
          f"replicate seconds p50 {o.percentile([x['replicate_seconds'] for x in stats], 0.5):.4f} max {max(x['replicate_seconds'] for x in stats):.4f}; "
          f"bytes/turn p50 {o.percentile([x['bytes_appended'] for x in stats], 0.5)}", flush=True)
    return o.write_result(evidence, "replicate", result)


# ------------------------------------------------------------------ 1. residue to zero

def residue(args, root, lustre, evidence):
    result = {"started_utc": o.utc(), "turns": args.turns, "sets": {}}
    for name, subdirs, files, skills in (("full", FULL, (), False), ("zero", FULL, ("AGENTS.md", "AGENTS.override.md", "history.jsonl"), True)):
        b = Bound(root, lustre, f"residue-{name}", subdirs=subdirs, extra_files=files, skills_copy=skills)
        log = b.local / "strace.log"
        s = b.server(f"residue-{name}", b.local / "sqlite", strace_log=log)
        tid, _ = s.start_thread()
        s.turn(tid, "SPKM_rz_0")
        t0 = time.time()
        for k in range(1, args.turns + 1):
            s.turn(tid, f"SPKM_rz_{k}")
        t1 = time.time()
        s.stop()
        time.sleep(1)
        rec = lustre_residue(log, t0, t1, b.binds, b.home.codex_home)
        rec["binds"] = sorted(v.name for v in b.binds.values())
        rec["per_turn"] = rec["lustre_total"] / args.turns
        result["sets"][name] = rec
        print(f"  residue [{name}]: lustre ops/turn={rec['per_turn']:.1f} (ancestors {rec['ancestors_total']}, depth {rec['ancestor_depth']}) inside_home={rec['inside_home']} other={rec['other']}", flush=True)
    return o.write_result(evidence, "residue", result)


# ------------------------------------------------------------------ 2. TUI under bwrap

def tui(args, root, lustre, evidence):
    b = Bound(root, lustre, "tui", extra_files=("history.jsonl",))
    model = o.FakeModel()
    threading.Thread(target=model.serve_forever, daemon=True).start()
    seen = []
    threading.Thread(target=lambda: [seen.append(model.requests.get()) for _ in iter(int, 1)], daemon=True).start()
    options = o.base_options(model.server_port, str(b.local / "sqlite")) + o.QUIET
    log = b.local / "strace.log"
    marker = "SPKM_tui_1"
    cmd = [*b.prefix(), *STRACE, "-o", str(log), "--", o.CODEX, *options, "--cd", str(b.workspace), marker]
    env = b.home.env()
    shell = "exec env " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " " + shlex.join(cmd)
    sock = "spike-episode-tui"
    tmux = ["tmux", "-L", sock]
    subprocess.run([*tmux, "kill-server"], capture_output=True)
    subprocess.run([*tmux, "new-session", "-d", "-s", "c", "-x", "180", "-y", "45", "-c", str(b.workspace), shell], check=True)
    t0 = time.time()
    time.sleep(args.settle)
    pane_pid = int(subprocess.run([*tmux, "display", "-p", "-t", "c", "#{pane_pid}"], capture_output=True, text=True).stdout.strip() or 0)
    procs = fd_snapshot(descendants(pane_pid) + [pane_pid])
    result = {"started_utc": o.utc(), "pane": subprocess.run([*tmux, "capture-pane", "-p", "-t", "c"], capture_output=True, text=True).stdout,
              "procs": [(p["comm"], len([x for x in p["paths"] if ".sqlite" in x or ".jsonl" in x])) for p in procs],
              "fd_paths": sorted({x for p in procs for x in p["paths"] if ".sqlite" in x or ".jsonl" in x or ".lock" in x})}
    subprocess.run([*tmux, "send-keys", "-t", "c", "SPKM_tui_2"], capture_output=True)
    time.sleep(1)
    subprocess.run([*tmux, "send-keys", "-t", "c", "Enter"], capture_output=True)
    time.sleep(10)
    t1 = time.time()
    result["pane_after_second_prompt"] = subprocess.run([*tmux, "capture-pane", "-p", "-t", "c"], capture_output=True, text=True).stdout
    for keys, wait in ((["/quit", "Enter"], 5), (["C-c"], 2), (["C-c"], 2)):
        if subprocess.run([*tmux, "has-session", "-t", "c"], capture_output=True).returncode != 0:
            break
        subprocess.run([*tmux, "send-keys", "-t", "c", *keys], capture_output=True)
        time.sleep(wait)
    subprocess.run([*tmux, "kill-server"], capture_output=True)
    time.sleep(1)
    result["model_markers"] = sorted({m for r in seen for m in o.MARKER.findall(json.dumps(r))})
    result["rollout_local"] = [str(p.relative_to(b.sessions)) for p in b.sessions.rglob("*.jsonl")]
    result["history_jsonl_local"] = {str(k): (v.name, k.stat().st_size) for k, v in b.binds.items() if v.name == "history.jsonl"}
    result["shared_sessions_outside"] = [str(p) for p in (b.home.codex_home / "sessions").rglob("*")]
    result["shared_history_outside_bytes"] = (b.home.codex_home / "history.jsonl").stat().st_size
    result["residue_second_prompt_window"] = lustre_residue(log, t0 + args.settle, t1, b.binds, b.home.codex_home)
    model.shutdown()
    print(f"  tui: procs={result['procs']} model_markers={result['model_markers']} rollout_local={result['rollout_local']} "
          f"history_local={result['history_jsonl_local']} shared_outside={result['shared_sessions_outside']}/{result['shared_history_outside_bytes']}B "
          f"lustre ops in 2nd-prompt window={result['residue_second_prompt_window']['lustre_total']} inside={result['residue_second_prompt_window']['inside_home']}", flush=True)
    print("  pane:", " | ".join(l.strip() for l in result["pane_after_second_prompt"].splitlines() if l.strip())[:300], flush=True)
    return o.write_result(evidence, "tui", result)


# ------------------------------------------------------------------ 3. remote-control under bwrap

def remote_control(args, root, lustre, evidence):
    b = Bound(root, lustre, "rc")
    model = o.FakeModel()
    threading.Thread(target=model.serve_forever, daemon=True).start()
    options = o.base_options(model.server_port, str(b.local / "sqlite")) + o.QUIET
    cmd = [*b.prefix(), o.CODEX, *options, "remote-control", "--no-daemon"]
    out, err = open(b.local / "rc.stdout", "w"), open(b.local / "rc.stderr", "w")
    proc = subprocess.Popen(cmd, cwd=str(b.workspace), env=b.home.env(), stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
    snaps = []
    deadline = o.now() + 20
    while o.now() < deadline and proc.poll() is None:
        time.sleep(2)
        snaps.append(fd_snapshot(descendants(proc.pid)))
    exited = proc.poll()
    if exited is None:
        os.killpg(proc.pid, 15)
        proc.wait(timeout=10)
    out.close(); err.close()
    best = max(snaps, key=lambda s: sum(len(p["paths"]) for p in s)) if snaps else []
    result = {"started_utc": o.utc(), "exit": exited if exited is not None else f"killed, {proc.returncode}",
              "stderr": (b.local / "rc.stderr").read_text()[:400], "stdout": (b.local / "rc.stdout").read_text()[:200],
              "sqlite_files": sorted(p.name for p in (b.local / "sqlite").glob("*.sqlite")),
              "fd_paths": sorted({x for p in best for x in p["paths"] if ".sqlite" in x or "codex-daemon" in x or "rc.sock" in x or ".lock" in x}),
              "daemon_tmp_after": sorted(str(p) for p in pathlib.Path(f"/tmp/codex-daemon-{os.getuid()}").glob("*")) if pathlib.Path(f"/tmp/codex-daemon-{os.getuid()}").exists() else None,
              "app_server_control_in_home": sorted(str(p.name) for p in (b.home.codex_home / "app-server-control").glob("*")) if (b.home.codex_home / "app-server-control").exists() else None}
    model.shutdown()
    print(f"  remote-control: exit={result['exit']} sqlite={result['sqlite_files']} stderr={result['stderr'][:120]!r} control_in_home={result['app_server_control_in_home']}", flush=True)
    return o.write_result(evidence, "remote_control", result)


# ------------------------------------------------------------------ 5. compaction

def compaction(args, root, lustre, evidence):
    b = Bound(root, lustre, "compact")
    result = {"started_utc": o.utc()}
    s = b.server("compact-src", b.local / "sqlite-A")
    tid, _ = s.start_thread()
    for k in range(1, 5):
        s.turn(tid, f"SPKM_cp_{k}")
    time.sleep(0.5)
    r = rollout_of(b.sessions, tid)
    result["before"] = {"rollout_bytes": r.stat().st_size, "rollout_lines": sum(1 for _ in open(r, errors="replace")), "rows": rows(b.local / "sqlite-A"),
                        "files": sorted(str(p.relative_to(b.sessions)) for p in b.sessions.rglob("*") if p.is_file())}
    index = len(s.notifications)
    n_model = len(s.model_seen)
    t0 = o.now()
    try:
        result["compact_start"] = {"ok": True, "result": s.rpc("thread/compact/start", {"threadId": tid}, timeout=30)}
    except Exception as error:  # noqa: BLE001
        result["compact_start"] = {"ok": False, "error": str(error)[:300]}
    done = s.wait_for(lambda n: n.get("method") in ("thread/compacted", "turn/completed") or "error" in (n.get("method") or ""), 30, index)
    result["compact_seconds"] = o.now() - t0
    result["notifications"] = sorted({n.get("method") for n in s.notifications[index:]})
    result["compacted_notification"] = done.get("method") if done else None
    result["model_requests_during_compaction"] = len(s.model_seen) - n_model
    time.sleep(1)
    result["after"] = {"rollout_bytes": r.stat().st_size, "rollout_lines": sum(1 for _ in open(r, errors="replace")), "rows": rows(b.local / "sqlite-A"),
                       "files": sorted(str(p.relative_to(b.sessions)) for p in b.sessions.rglob("*") if p.is_file())}
    t = s.turn(tid, "SPKM_cp_5")
    time.sleep(0.5)
    result["turn_after_compaction"] = {"ok": t["ok"], "history_to_model_last_request": sorted(s.model_seen[-1]["markers"]) if s.model_seen else None}
    result["stop_source"] = s.stop()
    o.backup(b.local / "sqlite-A", b.local / "backup")
    sessions2 = b.local / "sessions-2"
    shutil.copytree(b.sessions, sessions2)
    o.restore(b.local / "backup", b.local / "sqlite-B")
    result["cells"] = {}
    inspect_cell(b.rebind(sessions2), "restore_after_compaction", b.local / "sqlite-B", tid, result["cells"])
    print(f"  compaction: start={result['compact_start'].get('ok')} notif={result['compacted_notification']} in {result['compact_seconds']:.1f}s "
          f"model_requests={result['model_requests_during_compaction']} rollout {result['before']['rollout_lines']}->{result['after']['rollout_lines']} lines "
          f"rows {result['before']['rows']} -> {result['after']['rows']} files_after={result['after']['files']} "
          f"turn_after={result['turn_after_compaction']}", flush=True)
    return o.write_result(evidence, "compaction", result)


# ------------------------------------------------------------------ 7. are the residual calls load-bearing?

def residual_failure(args, root, lustre, evidence):
    result = {"started_utc": o.utc(), "variants": {}}
    for name, skills_bound in (("skills_bound_local", True), ("skills_on_lustre", False)):
        b = Bound(root, lustre, f"resfail-{name}", skills_copy=skills_bound)
        skills_dir = [k for k, v in b.binds.items() if v.name == "skills"][0] if skills_bound else b.home.codex_home / "skills"
        s = b.server(name, b.local / "sqlite")
        tid, _ = s.start_thread()
        s.turn(tid, "SPKM_rf_0")
        rec = {"skills_dir": str(skills_dir)}
        os.chmod(skills_dir, 0)
        try:
            index = len(s.notifications)
            turns = [s.turn(tid, f"SPKM_rf_{k}", timeout=30) for k in (1, 2, 3)]
            rec["turns_ok_with_skills_unreadable"] = sum(1 for t in turns if t["ok"])
            rec["durations"] = [t["duration"] for t in turns]
            rec["notifications"] = sorted({n.get("method") for n in s.notifications[index:]})
            rec["warnings"] = [str(n.get("params"))[:200] for n in s.notifications[index:] if n.get("method") in ("warning", "error")][:3]
        except Exception as error:  # noqa: BLE001
            rec["error"] = str(error)[:300]
        finally:
            os.chmod(skills_dir, 0o700)
        rec["stop"] = s.stop()
        rec["stderr_tail"] = (b.home.root / f"{name}.stderr.log").read_text()[-400:]
        result["variants"][name] = rec
        print(f"  residual_failure [{name}]: turns ok={rec.get('turns_ok_with_skills_unreadable')} durations={[round(d, 3) for d in rec.get('durations', [])]} warnings={rec.get('warnings')} err={rec.get('error')}", flush=True)
    return o.write_result(evidence, "residual_failure", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tests", nargs="+", choices=["skew", "replicate", "residue", "tui", "rc", "compaction", "resfail", "all"])
    parser.add_argument("--root", default=os.environ.get("SPIKE_ROOT") or os.path.join(os.environ.get("SUCODER_LOCAL_ROOT", "/tmp"), "spike-codex-episode"))
    parser.add_argument("--lustre-root", default=os.environ.get("SUCODER_SHARED_STATE_TEST_ROOT"), required=False)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--settle", type=int, default=25, help="seconds to let the TUI start and run its first prompt")
    args = parser.parse_args()
    if not args.lustre_root:
        raise SystemExit("--lustre-root or SUCODER_SHARED_STATE_TEST_ROOT is required")
    if not shutil.which("bwrap"):
        raise SystemExit("bwrap is required")
    os.umask(0o077)
    root = pathlib.Path(args.root); root.mkdir(parents=True, mode=0o700, exist_ok=True)
    lustre = pathlib.Path(args.lustre_root); lustre.mkdir(parents=True, mode=0o700, exist_ok=True)
    evidence = pathlib.Path(args.evidence); evidence.mkdir(parents=True, mode=0o700, exist_ok=True)
    (evidence / "environment.json").write_text(json.dumps({"host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID"), "utc": o.utc(),
                                                           "codex": o.CODEX, "args": vars(args), "lustre_root": str(lustre)}, indent=1) + "\n")
    order = ["skew", "replicate", "residue", "tui", "rc", "compaction", "resfail"]
    tests = order if "all" in args.tests else [t for t in order if t in args.tests]
    fns = {"skew": skew, "replicate": replicate, "residue": residue, "tui": tui, "rc": remote_control, "compaction": compaction, "resfail": residual_failure}
    for name in tests:
        print(f"== {name} {o.utc()}", flush=True)
        t0 = o.now()
        try:
            path = fns[name](args, root, lustre, evidence)
            print(f"== {name} done in {o.now() - t0:.0f}s -> {path}", flush=True)
        except Exception as error:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            o.write_result(evidence, name, {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()[-2000:]})
            print(f"== {name} FAILED: {error}", flush=True)


if __name__ == "__main__":
    main()
