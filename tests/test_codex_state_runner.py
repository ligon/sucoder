"""Managed Codex state, Slice 2 (first half): rollout replication, database
generations and their publication, in the node-side runner.

Self-contained on purpose (the launcher-side module is edited separately):
the fake Codex here is scriptable over stdin -- ``turn N`` grows the rollout
by about N bytes and advances the history projection the way Codex does,
``goal <status>`` and ``queue add`` write the rows the restore half will
check, ``extra`` opens an unclassified store, ``quit`` exits 0, SIGTERM exits
37 -- and it runs under the oldest ``python3`` on the node too, so one case
exercises the runner's CLI backup path.  Everything needs ``bwrap``; the real
Codex case needs ``codex``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "sucoder" / "codex_state_runner.py"
PLACEHOLDER = "__SUCODER_CODEX_STATE_DB__"
PY36 = shutil.which("python3.6")
needs_bwrap = pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap not installed")
INTERPRETERS = [
    pytest.param(sys.executable, id="current"),
    pytest.param(PY36, id="py36", marks=pytest.mark.skipif(PY36 is None, reason="no python3.6 on this node")),
]

FAKE = r'''#!/usr/bin/env python3
"""A scriptable Codex: one rollout plus WAL-mode stores, driven over stdin."""
import json, os, re, signal, sqlite3, sys, uuid

home = os.environ["CODEX_HOME"]
sqlite_home = None
for arg in sys.argv[1:]:
    m = re.match(r'sqlite_home="([^"]+)"', arg)
    if m:
        sqlite_home = m.group(1)
tid = str(uuid.uuid4())
day = os.path.join(home, "sessions", "2026", "10", "15")
os.makedirs(day, exist_ok=True)
rollout_path = os.path.join(day, "rollout-2026-10-15T00-00-00-%s.jsonl" % tid)
rollout = open(rollout_path, "ab", buffering=0)
rollout.write((json.dumps({"type": "session_meta", "payload": {"id": tid}}) + "\n").encode())


def store(name):
    conn = sqlite3.connect(os.path.join(sqlite_home, name), isolation_level=None)
    conn.execute("pragma journal_mode=wal")
    return conn


history = store("thread_history_1.sqlite")
history.execute("create table if not exists thread_history_projection_state ("
                "thread_id TEXT PRIMARY KEY, next_rollout_byte_offset INTEGER NOT NULL, next_rollout_ordinal INTEGER NOT NULL)")
state = store("state_5.sqlite")
state.execute("create table if not exists threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL, cli_version TEXT NOT NULL DEFAULT '')")
state.execute("insert or replace into threads values (?, ?, ?)", (tid, rollout_path, "0.162.1"))
goals = store("goals_1.sqlite")
goals.execute("create table if not exists thread_goals (thread_id TEXT PRIMARY KEY NOT NULL, goal_id TEXT NOT NULL, "
              "objective TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('active','paused','blocked','usage_limited',"
              "'budget_limited','complete')), token_budget INTEGER, tokens_used INTEGER NOT NULL DEFAULT 0, "
              "time_used_seconds INTEGER NOT NULL DEFAULT 0, created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL)")
queue = store("queue_1.sqlite")
queue.execute("create table if not exists queued_items (id TEXT PRIMARY KEY NOT NULL, thread_id TEXT NOT NULL, "
              "payload_json TEXT NOT NULL, queue_order INTEGER NOT NULL, created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL)")
held = []
ordinal = 0
turn = 0
signal.signal(signal.SIGTERM, lambda s, f: os._exit(37))
sys.stdout.write("FAKE_READY %d %s\n" % (os.getpid(), tid))
sys.stdout.flush()
while True:
    line = sys.stdin.readline()
    if not line:
        break
    words = line.split()
    if not words:
        continue
    if words[0] == "quit":
        break
    if words[0] == "turn":
        turn += 1
        size = int(words[1])
        bare = json.dumps({"type": "response_item", "turn": turn, "pad": ""})
        item = json.dumps({"type": "response_item", "turn": turn, "pad": "x" * max(0, size - len(bare) - 1)})
        rollout.write((item + "\n").encode())           # the rollout grows first,
        ordinal += 1
        offset = os.path.getsize(rollout_path)
        history.execute("insert or replace into thread_history_projection_state values (?, ?, ?)", (tid, offset, ordinal))
        reply = "ok %d %d" % (offset, ordinal)          # then the projection points past it
    elif words[0] == "goal":
        goals.execute("insert or replace into thread_goals values (?, ?, ?, ?, NULL, 0, 0, 1, 1)",
                      (tid, "goal-" + tid[:8], "an objective", words[1]))
        reply = "ok"
    elif words[0] == "queue":
        n = queue.execute("select count(*) from queued_items").fetchone()[0]
        queue.execute("insert into queued_items values (?, ?, ?, ?, 1, 1)", (str(uuid.uuid4()), tid, '{"text": "queued"}', n + 1))
        reply = "ok"
    elif words[0] == "extra":
        conn = sqlite3.connect(os.path.join(sqlite_home, "extra.sqlite"))
        conn.execute("create table if not exists t(x)")
        conn.commit()
        held.append(conn)
        reply = "ok"
    else:
        reply = "error: unknown command " + words[0]
    sys.stdout.write(reply + "\n")
    sys.stdout.flush()
sys.exit(0)
'''


# ----------------------------------------------------------------- helpers

def _sha256(path, limit=None):
    data = Path(path).read_bytes()
    return hashlib.sha256(data if limit is None else data[:limit]).hexdigest()


@pytest.fixture
def home(tmp_path):
    codex_home = tmp_path / "home"
    for d in ("sessions", "thread-writer-locks", "skills"):
        (codex_home / d).mkdir(parents=True)
    (codex_home / "AGENTS.md").write_text("GLOBAL\n")
    (codex_home / "config.toml").write_text('model = "x"\n')
    local = tmp_path / "local"
    local.mkdir()
    durable = tmp_path / "durable"
    durable.mkdir()
    fake = tmp_path / "fake_codex.py"
    fake.write_text(FAKE)
    fake.chmod(0o700)
    return {"codex_home": codex_home, "local": local, "durable": durable, "fake": fake, "scope": "cd" * 32}


def _env(home, **extra):
    env = {k: v for k, v in os.environ.items() if k not in ("CODEX_SQLITE_HOME",)}
    env.update({"SUCODER_LOCAL_ROOT": str(home["local"]), "CODEX_HOME": str(home["codex_home"])})
    env.update(extra)
    return env


def _status(home):
    scope_dir = home["local"] / "state" / "codex" / home["scope"]
    run = (scope_dir / "latest-run").read_text().strip()
    return json.loads((scope_dir / run / "status.json").read_text()), scope_dir / run


def _wait(home, predicate, what, timeout=20):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            last, run_dir = _status(home)
            if predicate(last):
                return last, run_dir
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}; last status: {last}")


def _wait_idle(home, timeout=20):
    """The initial capture (after placement) has run and none is in flight."""
    return _wait(home, lambda st: st.get("checkpoint", {}).get("count", 0) >= 1
                 and st["checkpoint"]["state"] in ("idle", "failed"), "an idle checkpointer", timeout)[0]


def _request(home, timeout=20):
    """Touch checkpoint.request and wait for that capture to end, whatever its outcome."""
    before = _wait_idle(home)["checkpoint"]["count"]
    _, run_dir = _status(home)
    (run_dir / "checkpoint.request").touch()
    return _wait(home, lambda st: st["checkpoint"]["count"] > before and st["checkpoint"]["state"] in ("idle", "failed"),
                 "the requested capture", timeout)[0]


def _manifest(status):
    return json.loads((Path(status["last_generation"]["path"]) / "manifest.json").read_text())


class Managed:
    """The runner with the scriptable fake under it, placed and ready."""

    def __init__(self, home, python=None, timeout=30.0, extra=()):
        python = python or sys.executable
        self.home = home
        argv = [python, str(RUNNER), "--scope", home["scope"], "--durable-root", str(home["durable"]),
                "--checkpoint-timeout", str(timeout), *extra, "--",
                python, str(home["fake"]), "-c", f'sqlite_home="{PLACEHOLDER}"', "app-server"]
        self.proc = subprocess.Popen(argv, env=_env(home), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
        ready = self.proc.stdout.readline().split()
        assert ready[:1] == ["FAKE_READY"], (ready, self.proc.stderr.read() if self.proc.poll() is not None else "running")
        self.pid, self.tid = int(ready[1]), ready[2]
        _, self.run_dir = _wait(home, lambda st: st["state"] == "placed", "placement")
        self.durable_run = home["durable"] / "codex" / home["scope"] / "runs" / self.run_dir.name
        self.generations = self.durable_run / "generations"
        self.relative = f"2026/10/15/rollout-2026-10-15T00-00-00-{self.tid}.jsonl"
        self.local_rollout = self.run_dir / "home" / "sessions" / self.relative
        self.replica = self.durable_run / "rollouts" / f"{self.tid}.jsonl"

    def cmd(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        reply = self.proc.stdout.readline().strip()
        assert reply.startswith("ok"), reply
        return reply

    def status(self):
        return _status(self.home)[0]

    def wait_synced(self, timeout=20):
        def synced(st):
            entry = st.get("replica", {}).get("threads", {}).get(self.tid)
            size = self.local_rollout.stat().st_size
            return entry is not None and entry["local_bytes"] == entry["replicated_bytes"] == size and st["replica"]["lag_bytes"] == 0
        return _wait(self.home, synced, "the replica to catch up", timeout)[0]

    def quit(self, timeout=40):
        self.proc.stdin.write("quit\n")
        self.proc.stdin.flush()
        return self.proc.wait(timeout=timeout)

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)


# ----------------------------------------------------------------- replication and generations

@needs_bwrap
@pytest.mark.parametrize("python", INTERPRETERS)
def test_replication_and_a_requested_generation(home, python):
    """Cases 1, 2 and 8: the replica is byte-identical, a request publishes a
    warm generation with the pinned manifest, and the oldest python uses the CLI."""
    m = Managed(home, python)
    try:
        for n in (500, 1200, 300):
            m.cmd(f"turn {n}")
        m.wait_synced()
        assert m.replica.read_bytes() == m.local_rollout.read_bytes()
        assert oct(m.replica.stat().st_mode & 0o777) == "0o600"
        replica_json = json.loads((m.run_dir / "replica.json").read_text())
        entry = replica_json["threads"][m.tid]
        assert entry["replicated_bytes"] == entry["local_bytes"] == m.local_rollout.stat().st_size
        assert entry["relative_path"] == m.relative and entry["replica_path"] == str(m.replica)
        status = m.status()
        assert status["replica"]["lag_bytes"] == 0 and status["replica"]["age_seconds"] < 5
        assert status["replica"]["threads"][m.tid]["relative_path"] == m.relative
        assert status["replica"]["anomalies"] == {} and status["replica"]["unrecognized"] == []

        status = _request(home)
        last = status["last_generation"]
        assert status["checkpoint"]["state"] == "idle" and status["checkpoint"]["last_outcome"] == "published"
        assert status["checkpoint"]["count"] >= 2 and status["checkpoint"]["dropped"] == 0
        assert last["class"] == "warm" and last["trigger"] == "request"
        assert status["durable_age_seconds"] is not None and status["durable_age_seconds"] < 10
        gen_dir = Path(last["path"])
        assert gen_dir.parent == m.generations and gen_dir.name == last["name"]
        assert oct(gen_dir.stat().st_mode & 0o777) == "0o700"
        assert _sha256(gen_dir / "manifest.json") == last["manifest_sha256"]
        manifest = json.loads((gen_dir / "manifest.json").read_text())
        assert sorted(os.listdir(gen_dir / "db")) == ["goals_1.sqlite", "queue_1.sqlite", "state_5.sqlite", "thread_history_1.sqlite"]
        assert manifest["absent"] == ["logs_2.sqlite", "memories_1.sqlite"]
        assert manifest["backup_method"] == ("cli" if python == PY36 else "connection")
        assert (manifest["format"], manifest["class"], manifest["generation"], manifest["parent"]) == (1, "warm", last["name"], None)
        assert manifest["scope"] == home["scope"] and manifest["run"] == m.run_dir.name
        assert manifest["codex_version"] == "0.162.1" and manifest["codex_home"] == str(home["codex_home"])
        assert manifest["allocation"]["node"] == os.uname()[1] and manifest["captured_at_iso"].endswith("Z")
        for name, record in manifest["databases"].items():
            path = gen_dir / "db" / name
            assert _sha256(path) == record["sha256"] and path.stat().st_size == record["bytes"] and record["quick_check"] == "ok"
            assert oct(path.stat().st_mode & 0o777) == "0o600"
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            assert conn.execute("pragma journal_mode").fetchone()[0] == "delete"
            assert conn.execute("pragma quick_check").fetchone()[0] == "ok"
            conn.close()
        copy = sqlite3.connect(f"file:{gen_dir / 'db' / 'thread_history_1.sqlite'}?mode=ro", uri=True)
        offset, ordinal = copy.execute("select next_rollout_byte_offset, next_rollout_ordinal from thread_history_projection_state"
                                       " where thread_id = ?", (m.tid,)).fetchone()
        copy.close()
        thread = manifest["threads"][m.tid]
        assert (thread["next_rollout_byte_offset"], thread["next_rollout_ordinal"]) == (offset, ordinal) == (m.local_rollout.stat().st_size, 3)
        assert thread["relative_path"] == m.relative and thread["replica_bytes"] >= offset
        assert thread["rollout_sha256_prefix"] == _sha256(m.replica, thread["replica_bytes"])
        assert not list(m.durable_run.glob(".incoming-*"))
        run_json = json.loads((m.durable_run / "run.json").read_text())
        assert (run_json["format"], run_json["run"], run_json["parent"], run_json["scope"]) == (1, m.run_dir.name, None, home["scope"])
        assert run_json["codex_home"] == str(home["codex_home"]) and run_json["created_iso"].endswith("Z")
        # the local bundle is kept as well
        assert (m.run_dir / "snapshots" / last["name"] / "manifest.json").exists()

        assert m.quit() == 0
        status = m.status()
        assert status["state"] == "exited" and status["last_generation"]["class"] == "sealed"
        assert sorted(os.listdir(m.generations))[-1] == status["last_generation"]["name"]
        assert not list(home["codex_home"].rglob("*.sqlite")) and not list((home["codex_home"] / "sessions").rglob("*"))
    finally:
        m.close()


@needs_bwrap
def test_a_generation_waits_for_its_replica_and_stays_local_while_it_lags(home):
    """Case 3: the projection moves past what the replica holds; the candidate is
    not published and status says why; once the replica catches up, the next is."""
    m = Managed(home, timeout=3)
    try:
        m.cmd("turn 400")
        m.wait_synced()
        first = _wait_idle(home)["last_generation"]
        rollouts = m.durable_run / "rollouts"
        os.chmod(rollouts, 0)           # the durable side stalls; the conversation does not
        try:
            m.cmd("turn 800")
            status = _request(home, timeout=30)
        finally:
            os.chmod(rollouts, 0o700)
        checkpoint = status["checkpoint"]
        assert checkpoint["last_outcome"] == "not_durable" and checkpoint["state"] == "failed"
        assert m.tid in checkpoint["reason"] and "projection needs" in checkpoint["reason"]
        assert status["last_generation"] == first
        assert status["replica"]["lag_bytes"] > 0 and m.tid in status["replica"]["errors"]
        assert os.listdir(m.generations) == [first["name"]]
        assert not list(m.durable_run.glob(".incoming-*")), "nothing reached the durable side"
        local_only = sorted(os.listdir(m.run_dir / "snapshots"))[-1]
        kept = json.loads((m.run_dir / "snapshots" / local_only / "manifest.json").read_text())
        assert local_only != first["name"] and m.tid in kept["not_durable"]
        assert kept["threads"][m.tid]["replica_bytes"] < kept["threads"][m.tid]["next_rollout_byte_offset"]

        m.wait_synced()
        status = _request(home)
        assert status["checkpoint"]["last_outcome"] == "published" and status["checkpoint"]["state"] == "idle"
        assert status["last_generation"]["name"] != first["name"]
        thread = _manifest(status)["threads"][m.tid]
        assert thread["replica_bytes"] >= thread["next_rollout_byte_offset"] == m.local_rollout.stat().st_size
        assert "resumed" in (m.run_dir / "runner.log").read_text()
    finally:
        m.close()


@needs_bwrap
def test_a_failed_publication_leaves_incoming_and_the_last_generation_alone(home):
    """Case 4: generations/ unwritable -> no new generation, .incoming left,
    last_generation unchanged, checkpoint failed; then the next request publishes."""
    m = Managed(home)
    try:
        m.cmd("turn 300")
        first = _wait_idle(home)["last_generation"]
        os.chmod(m.generations, 0o500)
        try:
            status = _request(home)
        finally:
            os.chmod(m.generations, 0o700)
        checkpoint = status["checkpoint"]
        assert checkpoint["state"] == "failed" and checkpoint["last_outcome"] == "failed"
        assert "Permission denied" in checkpoint["reason"]
        assert status["last_generation"] == first
        incoming = list(m.durable_run.glob(".incoming-*"))
        assert len(incoming) == 1 and (incoming[0] / "manifest.json").exists() and (incoming[0] / "db" / "state_5.sqlite").exists()
        assert os.listdir(m.generations) == [first["name"]]

        status = _request(home)
        assert status["checkpoint"]["last_outcome"] == "published" and status["checkpoint"]["state"] == "idle"
        assert status["last_generation"]["name"] != first["name"]
        assert sorted(os.listdir(m.generations)) == sorted([first["name"], status["last_generation"]["name"]])
        assert list(m.durable_run.glob(".incoming-*")) == incoming, "an incomplete publication is left where it is, never a candidate"
        names = sorted(os.listdir(m.generations))
        assert int(names[1][:6]) > int(names[0][:6]), "sequence numbers strictly increase, consumed by the failure too"
    finally:
        m.close()


@needs_bwrap
def test_an_unclassified_store_rejects_the_candidate(home):
    """Case 5."""
    m = Managed(home)
    try:
        first = _wait_idle(home)["last_generation"]
        m.cmd("extra")
        status = _request(home)
        assert status["checkpoint"]["last_outcome"] == "failed"
        assert "unclassified store under db: extra.sqlite" in status["checkpoint"]["reason"]
        assert status["last_generation"] == first and os.listdir(m.generations) == [first["name"]]
        assert not list(m.durable_run.glob(".incoming-*"))
        assert status["state"] == "placed", "an extra store under db is placed correctly, only unclassified"
    finally:
        m.close()


@needs_bwrap
def test_a_required_store_that_disappears_fails_the_capture(home):
    """Case 6: thread_history_1 was seen, so its disappearance is a failure, not an absence."""
    m = Managed(home)
    try:
        m.cmd("turn 200")
        first = _request(home)["last_generation"]
        assert "thread_history_1.sqlite" in _manifest(_status(home)[0])["databases"]
        for path in (m.run_dir / "db").glob("thread_history_1.sqlite*"):
            path.unlink()
        status = _request(home)
        assert status["checkpoint"]["last_outcome"] == "failed"
        assert "required store disappeared: thread_history_1.sqlite" in status["checkpoint"]["reason"]
        assert status["last_generation"] == first
        assert sorted(os.listdir(m.generations))[-1] == first["name"]
    finally:
        m.close()


@needs_bwrap
@pytest.mark.parametrize("how", ["quit", "sigkill"])
def test_the_final_generation_is_sealed_after_a_clean_exit_and_crash_otherwise(home, how):
    """Case 7; also the accepted no-op flags and, on the clean path, the timer trigger."""
    extra = ("--fresh", "--unattended") + (("--checkpoint-minutes", "0.03") if how == "quit" else ())
    m = Managed(home, extra=extra)
    try:
        m.cmd("turn 256")
        m.wait_synced()
        _wait_idle(home)
        if how == "quit":
            def timer_fired(st):
                last = st.get("last_generation") or {}
                return last.get("trigger") == "interval"
            _wait(home, timer_fired, "a timer-driven generation", timeout=15)
            assert m.quit() == 0
        else:
            os.kill(m.pid, signal.SIGKILL)
            assert m.proc.wait(timeout=40) != 0
        status = m.status()
        last = status["last_generation"]
        assert last["class"] == ("sealed" if how == "quit" else "crash") and last["trigger"] == "final"
        assert status["state"] == "exited" and status["checkpoint"]["state"] == "idle"
        assert status["checkpoint"]["last_outcome"] == "published"
        manifest = json.loads((Path(last["path"]) / "manifest.json").read_text())
        assert manifest["class"] == last["class"]
        thread = manifest["threads"][m.tid]
        assert thread["next_rollout_byte_offset"] == m.local_rollout.stat().st_size == thread["replica_bytes"]
        assert m.replica.read_bytes() == m.local_rollout.read_bytes()
        log = (m.run_dir / "runner.log").read_text()
        assert f"generation {last['name']} published ({last['class']}, final" in log
        assert "starts empty and is separate from the shared history in " + str(home["codex_home"]) in log
        stderr = m.proc.stderr.read()
        assert "starts empty and is separate from the shared history" in stderr
        assert stderr.count("SUCODER codex-state") < 40, "one line per transition, no periodic chatter"
    finally:
        m.close()


@needs_bwrap
def test_restore_is_refused_today_and_the_durable_root_is_validated(home):
    """Case 10 (the bwrap check precedes the durable-root checks, so it needs bwrap too)."""
    head = [sys.executable, str(RUNNER), "--scope", home["scope"]]
    tail = ["--", sys.executable, str(home["fake"]), "-c", f'sqlite_home="{PLACEHOLDER}"', "app-server"]

    def run(*middle):
        return subprocess.run([*head, *middle, *tail], env=_env(home), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=60)

    proc = run("--durable-root", str(home["durable"]), "--restore", "a" * 32 + "/000001-deadbeef")
    assert proc.returncode == 70 and "restore is not implemented yet" in proc.stderr
    assert not (home["durable"] / "codex").exists() and not (home["local"] / "state").exists()

    proc = run("--durable-root", str(home["local"] / "durable"))
    assert proc.returncode == 70 and "allocation-local" in proc.stderr
    assert not (home["local"] / "durable").exists()

    proc = run("--durable-root", "relative/path")
    assert proc.returncode == 70 and "not an absolute path" in proc.stderr

    proc = run("--durable-root", "/tmp/with space")
    assert proc.returncode == 70 and "characters" in proc.stderr

    proc = run()
    assert proc.returncode == 2 and "--durable-root" in proc.stderr


# ----------------------------------------------------------------- the real Codex

class AppServer:
    """Minimal JSON-RPC over the app-server's stdio, with a reader thread."""

    def __init__(self, proc):
        self.proc = proc
        self.messages = []
        self.cond = threading.Condition()
        self.next_id = 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            with self.cond:
                self.messages.append(message)
                self.cond.notify_all()

    def send(self, payload):
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def rpc(self, method, params, timeout=60):
        self.next_id += 1
        rid = self.next_id
        self.send({"id": rid, "method": method, "params": params})
        reply = self.wait(lambda msg: msg.get("id") == rid, timeout, method)
        assert "error" not in reply, reply
        return reply["result"]

    def wait(self, predicate, timeout, what):
        deadline = time.time() + timeout
        with self.cond:
            while True:
                for message in self.messages:
                    if predicate(message):
                        return message
                remaining = deadline - time.time()
                assert remaining > 0 and self.proc.poll() is None, f"no {what} within {timeout}s (exit {self.proc.poll()})"
                self.cond.wait(min(remaining, 0.2))


@needs_bwrap
@pytest.mark.skipif(not shutil.which("codex"), reason="Codex CLI not installed")
def test_a_real_codex_turn_is_replicated_and_its_projection_offset_captured(home, tmp_path):
    """Case 9: thread/start, one turn through the loopback fake model, the
    replica byte-identical, and a requested generation whose manifest carries
    the thread's projection row from the copy and the replicated file's path."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            event = {"type": "response.completed", "response": {
                "id": "resp_test", "object": "response", "status": "completed", "output": [],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}}
            self.wfile.write(("event: response.completed\ndata: " + json.dumps(event) + "\n\n").encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    codex = shutil.which("codex")
    command = [codex, "-c", f'sqlite_home="{PLACEHOLDER}"',
               "-c", 'model_provider="offline"', "-c", 'model="fake"',
               "-c", "features.plugins=false", "-c", "features.remote_plugin=false", "-c", "features.apps=false",
               "-c", ('model_providers.offline={name="offline",'
                      f'base_url="http://127.0.0.1:{server.server_port}",wire_api="responses"}}'),
               "app-server"]
    (tmp_path / "user-home").mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "user-home"), "NO_PROXY": "127.0.0.1", "RUST_LOG": "error",
           "SUCODER_LOCAL_ROOT": str(home["local"]), "CODEX_HOME": str(home["codex_home"]), "TMPDIR": str(tmp_path)}
    proc = subprocess.Popen([sys.executable, str(RUNNER), "--scope", home["scope"], "--durable-root", str(home["durable"]),
                             "--", *command],
                            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    app = AppServer(proc)
    try:
        app.rpc("initialize", {"clientInfo": {"name": "sucoder_test", "version": "1"}, "capabilities": {"experimentalApi": True}})
        app.send({"method": "initialized", "params": {}})
        tid = app.rpc("thread/start", {"cwd": str(tmp_path)})["thread"]["id"]
        status, run_dir = _wait(home, lambda st: st["state"] in ("placed", "misplaced"), "placement", timeout=60)
        assert status["state"] == "placed", status
        app.rpc("turn/start", {"threadId": tid, "input": [{"type": "text", "text": "hello"}]})
        app.wait(lambda msg: msg.get("method") == "turn/completed" and (msg.get("params") or {}).get("threadId", tid) == tid,
                 60, "turn/completed")
        durable_run = home["durable"] / "codex" / home["scope"] / "runs" / run_dir.name
        replica = durable_run / "rollouts" / f"{tid}.jsonl"
        rollouts = list((run_dir / "home" / "sessions").rglob(f"rollout-*-{tid}.jsonl"))
        assert len(rollouts) == 1, "the thread id is the trailing uuid of Codex's rollout file name"
        local = rollouts[0]
        relative = str(local.relative_to(run_dir / "home" / "sessions"))

        def replicated(st):
            entry = st.get("replica", {}).get("threads", {}).get(tid)
            return entry is not None and entry["replicated_bytes"] == entry["local_bytes"] == local.stat().st_size > 0
        status, _ = _wait(home, replicated, "the real rollout's replica", timeout=30)
        assert replica.read_bytes() == local.read_bytes()
        assert status["replica"]["threads"][tid]["relative_path"] == relative

        deadline = time.time() + 60
        while True:                      # the projection row may land a moment after turn/completed
            status = _request(home, timeout=40)
            manifest = _manifest(status) if status["checkpoint"]["last_outcome"] == "published" else None
            thread = (manifest or {}).get("threads", {}).get(tid)
            if thread and "thread_history_1.sqlite" in manifest["databases"] and thread["next_rollout_ordinal"] > 0:
                break
            assert time.time() < deadline, (status["checkpoint"], manifest and manifest["threads"])
            time.sleep(1)
        gen_dir = Path(status["last_generation"]["path"])
        copy = sqlite3.connect(f"file:{gen_dir / 'db' / 'thread_history_1.sqlite'}?mode=ro", uri=True)
        row = copy.execute("select next_rollout_byte_offset, next_rollout_ordinal from thread_history_projection_state"
                           " where thread_id = ?", (tid,)).fetchone()
        copy.close()
        assert (thread["next_rollout_byte_offset"], thread["next_rollout_ordinal"]) == tuple(row)
        assert thread["relative_path"] == relative and (run_dir / "home" / "sessions" / thread["relative_path"]) == local
        assert thread["next_rollout_byte_offset"] <= thread["replica_bytes"] <= replica.stat().st_size
        assert thread["rollout_sha256_prefix"] == _sha256(replica, thread["replica_bytes"])
        assert manifest["codex_version"] and manifest["class"] == "warm" and manifest["absent"] == []
        assert sorted(manifest["databases"]) == ["goals_1.sqlite", "logs_2.sqlite", "memories_1.sqlite", "queue_1.sqlite",
                                                 "state_5.sqlite", "thread_history_1.sqlite"]
        assert not list(home["codex_home"].glob("*.sqlite")) and not list((home["codex_home"] / "sessions").rglob("*"))

        proc.stdin.close()
        assert proc.wait(timeout=30) == 0
        status, _ = _status(home)
        assert status["last_generation"]["class"] == "sealed" and status["checkpoint"]["last_outcome"] == "published"
        final = json.loads((Path(status["last_generation"]["path"]) / "manifest.json").read_text())
        assert final["threads"][tid]["next_rollout_byte_offset"] <= final["threads"][tid]["replica_bytes"] == local.stat().st_size
    finally:
        if proc.poll() is None:
            proc.kill()
        server.shutdown()
        server.server_close()
