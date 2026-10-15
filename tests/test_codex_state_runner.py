"""Managed Codex state, Slice 2: rollout replication, database generations
and their publication (first half), then restore from a generation, the
pending-work gate and same-allocation reuse in place (second half), in the
node-side runner.

Self-contained on purpose (the launcher-side module is edited separately):
the fake Codex here is scriptable over stdin -- ``turn N`` grows the rollout
by about N bytes and advances the history projection the way Codex does,
``goal <status>`` and ``queue add`` write the rows the gate checks, ``extra``
opens an unclassified store, ``quit`` exits 0, SIGTERM exits 37 -- and, like
Codex, it continues the thread it finds in the stores it is handed (the
recorded rollout path, the projection's ordinal), reporting what it found on
a ``FAKE_SEES`` line so a test can assert that restored state was read by the
child.  It runs under the oldest ``python3`` on the node too, so the
replication and restore cases exercise the runner's CLI backup path there.
Everything needs ``bwrap``; the real Codex cases need ``codex`` and use
``tests.codex_app_server``.
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


def store(name):
    conn = sqlite3.connect(os.path.join(sqlite_home, name), isolation_level=None)
    conn.execute("pragma journal_mode=wal")
    return conn


history = store("thread_history_1.sqlite")
history.execute("create table if not exists thread_history_projection_state ("
                "thread_id TEXT PRIMARY KEY, next_rollout_byte_offset INTEGER NOT NULL, next_rollout_ordinal INTEGER NOT NULL)")
state = store("state_5.sqlite")
state.execute("create table if not exists threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL, cli_version TEXT NOT NULL DEFAULT '')")
goals = store("goals_1.sqlite")
goals.execute("create table if not exists thread_goals (thread_id TEXT PRIMARY KEY NOT NULL, goal_id TEXT NOT NULL, "
              "objective TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('active','paused','blocked','usage_limited',"
              "'budget_limited','complete')), token_budget INTEGER, tokens_used INTEGER NOT NULL DEFAULT 0, "
              "time_used_seconds INTEGER NOT NULL DEFAULT 0, created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL)")
queue = store("queue_1.sqlite")
queue.execute("create table if not exists queued_items (id TEXT PRIMARY KEY NOT NULL, thread_id TEXT NOT NULL, "
              "payload_json TEXT NOT NULL, queue_order INTEGER NOT NULL, created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL)")
held = []
# What the child finds in the stores it was handed: the threads with their
# recorded rollout paths (and the file's size there), the projection rows,
# goals and queued items.  Then, like Codex, continue the thread on record
# at its recorded path, or start a new one.
sees = {"threads": [], "projection": {}, "goals": [], "queue": {}}
existing = state.execute("select id, rollout_path from threads order by rowid desc").fetchall()
for row_id, path in existing:
    try:
        size = os.path.getsize(path)
    except OSError:
        size = None
    sees["threads"].append({"id": row_id, "rollout_path": path, "rollout_bytes": size})
for row in history.execute("select thread_id, next_rollout_byte_offset, next_rollout_ordinal from thread_history_projection_state"):
    sees["projection"][row[0]] = [row[1], row[2]]
sees["goals"] = [list(row) for row in goals.execute("select thread_id, status from thread_goals order by thread_id")]
sees["queue"] = dict((row[0], row[1]) for row in queue.execute("select thread_id, count(*) from queued_items group by thread_id"))
if existing and not os.environ.get("FAKE_FRESH_THREAD"):
    tid, rollout_path = existing[0]
    rollout = open(rollout_path, "ab", buffering=0)
    row = history.execute("select next_rollout_ordinal from thread_history_projection_state where thread_id = ?", (tid,)).fetchone()
    ordinal = turn = row[0] if row else 0
else:
    tid = str(uuid.uuid4())
    day = os.path.join(home, "sessions", "2026", "10", "15")
    os.makedirs(day, exist_ok=True)
    rollout_path = os.path.join(day, "rollout-2026-10-15T00-00-00-%s.jsonl" % tid)
    rollout = open(rollout_path, "ab", buffering=0)
    rollout.write((json.dumps({"type": "session_meta", "payload": {"id": tid}}) + "\n").encode())
    state.execute("insert or replace into threads values (?, ?, ?)", (tid, rollout_path, "0.162.1"))
    ordinal = turn = 0
signal.signal(signal.SIGTERM, lambda s, f: os._exit(37))
sys.stdout.write("FAKE_SEES " + json.dumps(sees, sort_keys=True) + "\n")
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
    (codex_home / "history.jsonl").write_text("shared\n")
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
        self.sees = None            # what the fake found in the stores it was handed
        while True:
            line = self.proc.stdout.readline()
            words = line.split()
            if words[:1] == ["FAKE_SEES"]:
                self.sees = json.loads(line.split(" ", 1)[1])
            elif words[:1] == ["FAKE_READY"]:
                break
            else:
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                raise AssertionError((line, self.proc.stderr.read()))
        self.pid, self.tid = int(words[1]), words[2]
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
def test_the_durable_root_and_the_flags_are_validated_before_anything_is_created(home):
    """Case 10 (the bwrap check precedes the durable-root checks, so it needs bwrap too)."""
    head = [sys.executable, str(RUNNER), "--scope", home["scope"]]
    tail = ["--", sys.executable, str(home["fake"]), "-c", f'sqlite_home="{PLACEHOLDER}"', "app-server"]

    def run(*middle):
        return subprocess.run([*head, *middle, *tail], env=_env(home), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=60)

    proc = run("--durable-root", str(home["local"] / "durable"))
    assert proc.returncode == 70 and "allocation-local" in proc.stderr
    assert not (home["local"] / "durable").exists()

    proc = run("--durable-root", "relative/path")
    assert proc.returncode == 70 and "not an absolute path" in proc.stderr

    proc = run("--durable-root", "/tmp/with space")
    assert proc.returncode == 70 and "characters" in proc.stderr

    proc = run("--durable-root", str(home["durable"]), "--restore", "a" * 32 + "/000001-deadbeef", "--fresh")
    assert proc.returncode == 2 and "mutually exclusive" in proc.stderr

    proc = run()
    assert proc.returncode == 2 and "--durable-root" in proc.stderr
    assert not (home["durable"] / "codex").exists() and not (home["local"] / "state").exists()


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


# ----------------------------------------------------------------- restore, the gate, reuse in place

def _allocation(home, tmp_path, name, **changes):
    """Another allocation: a new local root (and, when asked, another scope or
    shared home) over the same durable root."""
    local = tmp_path / name
    local.mkdir()
    other = dict(home, local=local)
    other.update(changes)
    return other


def _attempt(home, *flags, timeout=60):
    """The runner to completion with the fake under it, for refusals."""
    argv = [sys.executable, str(RUNNER), "--scope", home["scope"], "--durable-root", str(home["durable"]), *flags, "--",
            sys.executable, str(home["fake"]), "-c", f'sqlite_home="{PLACEHOLDER}"', "app-server"]
    return subprocess.run(argv, env=_env(home), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=timeout)


def _assert_refused(proc, home, *needles):
    """Exit 70 with the message, Codex never started, nothing placed: no db/,
    no staging and no rollout under any run directory of the scope."""
    assert proc.returncode == 70, (proc.returncode, proc.stderr, proc.stdout)
    for needle in needles:
        assert needle in proc.stderr, (needle, proc.stderr)
    assert "FAKE_READY" not in proc.stdout and "refusing to launch" in proc.stderr
    scope_dir = home["local"] / "state" / "codex" / home["scope"]
    if scope_dir.exists():
        for run_dir in scope_dir.iterdir():
            if not run_dir.is_dir():
                continue
            assert not (run_dir / "db").exists() and not (run_dir / "restore-staging").exists(), run_dir
            assert not list((run_dir / "home" / "sessions").rglob("*")), run_dir
            status = json.loads((run_dir / "status.json").read_text())
            assert status["state"] == "refused" and status["reason"] in proc.stderr
        assert not (scope_dir / "latest-run").exists(), "a refused run never becomes the latest"


@needs_bwrap
@pytest.mark.parametrize("python", INTERPRETERS)
def test_restore_places_a_sealed_generation_in_another_allocation(home, tmp_path, python):
    """Case 1: run A seals a generation; run B restores it in another local
    root: the child reads the restored projection row and finds the rollout
    at its recorded path with the replica's bytes, the new run's replica is
    byte-identical, a further turn replicates from the right offset, and
    run.json and the next manifest carry the parent."""
    a = Managed(home, python)
    try:
        for n in (500, 1200, 300):
            a.cmd(f"turn {n}")
        a.wait_synced()
        assert a.quit() == 0
    finally:
        a.close()
    status_a = a.status()
    gen = status_a["last_generation"]
    assert gen["class"] == "sealed"
    manifest_a = _manifest(status_a)
    source = a.replica.read_bytes()
    assert source == a.local_rollout.read_bytes() and len(source) == manifest_a["threads"][a.tid]["next_rollout_byte_offset"]
    selector = f"{a.run_dir.name}/{gen['name']}"

    other = _allocation(home, tmp_path, "local-b")
    b = Managed(other, python, extra=("--restore", selector))
    try:
        assert b.run_dir.parents[3] == other["local"] and b.run_dir.name != a.run_dir.name
        assert b.tid == a.tid, "the child continued the restored thread"
        recorded = str(home["codex_home"] / "sessions" / a.relative)
        assert b.sees["threads"] == [{"id": a.tid, "rollout_path": recorded, "rollout_bytes": len(source)}]
        assert b.sees["projection"] == {a.tid: [len(source), 3]}
        assert b.sees["goals"] == [] and b.sees["queue"] == {}
        assert b.local_rollout.read_bytes() == source and oct(b.local_rollout.stat().st_mode & 0o777) == "0o600"
        assert oct(b.local_rollout.parent.stat().st_mode & 0o777) == "0o700"
        assert b.replica.read_bytes() == source and oct(b.replica.stat().st_mode & 0o777) == "0o600"
        for name in manifest_a["databases"]:
            placed = b.run_dir / "db" / name
            assert placed.exists() and oct(placed.stat().st_mode & 0o777) == "0o600"
        assert oct((b.run_dir / "db").stat().st_mode & 0o777) == "0o700" and not (b.run_dir / "restore-staging").exists()
        run_json = json.loads((b.durable_run / "run.json").read_text())
        parent = {"run": a.run_dir.name, "generation": gen["name"], "class": "sealed", "manifest_sha256": gen["manifest_sha256"]}
        assert run_json["parent"] == parent and run_json["run"] == b.run_dir.name
        assert json.loads((b.run_dir / "run.json").read_text()) == run_json
        status = b.status()
        restored = status["restored_from"]
        assert (restored["run"], restored["generation"], restored["class"], restored["manifest_sha256"]) == (
            a.run_dir.name, gen["name"], "sealed", gen["manifest_sha256"])
        assert restored["threads"] == 1 and restored["rollout_bytes"] == len(source)
        assert restored["database_bytes"] == sum(r["bytes"] for r in manifest_a["databases"].values())
        assert restored["bytes"] == restored["database_bytes"] + restored["rollout_bytes"]
        assert status["pending_work"] == {"active_goals": [], "queued": {}} and status["reused_run"] is False and status["mode"] == "restore"
        owner = json.loads((b.run_dir / "owner.json").read_text())
        assert owner["restored_from"] == restored and owner["reused_run"] is False

        before = b.local_rollout.stat().st_size
        b.cmd("turn 700")
        b.wait_synced()
        after = b.local_rollout.stat().st_size
        assert after > before and b.replica.stat().st_size == after
        assert b.replica.read_bytes() == b.local_rollout.read_bytes() and b.replica.read_bytes()[:len(source)] == source
        status = _request(other)
        manifest_b = _manifest(status)
        assert manifest_b["parent"] == parent and manifest_b["run"] == b.run_dir.name
        thread = manifest_b["threads"][a.tid]
        assert (thread["next_rollout_byte_offset"], thread["next_rollout_ordinal"], thread["relative_path"]) == (after, 4, a.relative)
        assert thread["replica_bytes"] == after and thread["rollout_sha256_prefix"] == _sha256(b.replica, after)
        log = (b.run_dir / "runner.log").read_text()
        assert f"restoring generation {selector} (sealed" in log and "placed 4 database(s)" in log
        assert f"restores generation {selector} (sealed" in log and "pending work: none" in log
        assert "starts empty" not in log
        assert b.quit() == 0
        assert f"restores generation {selector} (sealed" in b.proc.stderr.read()
    finally:
        b.close()
    assert a.replica.read_bytes() == source, "the source run's replica is untouched"
    assert sorted(os.listdir(a.generations))[-1] == gen["name"]


@needs_bwrap
def test_restore_refusals_place_nothing_and_never_start_codex(home, tmp_path):
    """Case 2: every refusal exits 70 with a specific message, no db/, no
    staging, no rollout placed, the child never started; the unattended gate
    refuses pending work that an attended restore reports and hands over."""
    a = Managed(home)
    try:
        a.cmd("turn 600")
        a.wait_synced()
        warm = _request(home)["last_generation"]
        assert warm["class"] == "warm"
        assert a.quit() == 0
    finally:
        a.close()
    sealed = a.status()["last_generation"]
    run_a = a.run_dir.name
    selector = f"{run_a}/{sealed['name']}"
    gen_dir = Path(sealed["path"])
    offset = json.loads((gen_dir / "manifest.json").read_text())["threads"][a.tid]["next_rollout_byte_offset"]
    original = a.replica.read_bytes()
    assert len(original) == offset
    numbers = iter(range(100))

    def attempt(*flags, **changes):
        alloc = _allocation(home, tmp_path, f"alloc-{next(numbers)}", **changes)
        return _attempt(alloc, *flags), alloc

    # the replica shorter than the recorded offset: the phantom-turn hazard
    a.replica.write_bytes(original[:offset - 1])
    try:
        proc, alloc = attempt("--restore", selector)
    finally:
        a.replica.write_bytes(original)
    _assert_refused(proc, alloc, f"thread {a.tid}: its replica", f"holds {offset - 1} bytes", f"next_rollout_byte_offset {offset}", "phantom")

    # one byte altered inside the hashed prefix
    altered = bytearray(original)
    altered[len(original) // 2] ^= 0x01
    a.replica.write_bytes(bytes(altered))
    try:
        proc, alloc = attempt("--restore", selector)
    finally:
        a.replica.write_bytes(original)
    _assert_refused(proc, alloc, f"thread {a.tid}: the first {offset} bytes of its replica", "rollout_sha256_prefix")

    # a database altered after publication, then truncated
    store = gen_dir / "db" / "goals_1.sqlite"
    pristine = store.read_bytes()
    damaged = bytearray(pristine)
    damaged[-1] ^= 0xFF
    store.write_bytes(bytes(damaged))
    try:
        proc, alloc = attempt("--restore", selector)
    finally:
        store.write_bytes(pristine)
    _assert_refused(proc, alloc, "goals_1.sqlite does not hash to the manifest's sha256")
    store.write_bytes(pristine[:-1])
    try:
        proc, alloc = attempt("--restore", selector)
    finally:
        store.write_bytes(pristine)
    _assert_refused(proc, alloc, f"goals_1.sqlite is {len(pristine) - 1} bytes, the manifest records {len(pristine)}")

    # the generation's scope: the same bundle copied under another scope's durable tree
    scope2 = "ef" * 32
    shutil.copytree(home["durable"] / "codex" / home["scope"] / "runs" / run_a, home["durable"] / "codex" / scope2 / "runs" / run_a)
    proc, alloc = attempt("--restore", selector, scope=scope2)
    _assert_refused(proc, alloc, f"belongs to scope {home['scope']}, not {scope2}")

    # another CODEX_HOME: the recorded rollout paths are valid only under the captured one
    home2 = tmp_path / "home2"
    (home2 / "sessions").mkdir(parents=True)
    (home2 / "config.toml").write_text('model = "x"\n')
    proc, alloc = attempt("--restore", selector, codex_home=home2)
    _assert_refused(proc, alloc, "was captured under CODEX_HOME " + str(home["codex_home"]), "not " + str(home2))

    # an incomplete publication, a generation or run that does not exist, a directory without a manifest
    proc, alloc = attempt("--restore", f"{run_a}/.incoming-{'0' * 32}")
    _assert_refused(proc, alloc, "incomplete publication")
    proc, alloc = attempt("--restore", f"{run_a}/999999-00000000")
    _assert_refused(proc, alloc, f"generation {run_a}/999999-00000000 does not exist")
    proc, alloc = attempt("--restore", f"{'0' * 32}/{sealed['name']}")
    _assert_refused(proc, alloc, "does not exist")
    (gen_dir.parent / "000099-0badf00d").mkdir()
    proc, alloc = attempt("--restore", f"{run_a}/000099-0badf00d")
    _assert_refused(proc, alloc, "has no manifest.json")

    # malformed selectors never touch the disk
    for bad in ("foo", run_a, f"{selector}/x", f"../{selector}", selector.upper(), f"/{selector}", str(gen_dir)):
        proc, alloc = attempt("--restore", bad)
        _assert_refused(proc, alloc, "is not RUN/GENERATION", "never a path")
        assert not (alloc["local"] / "state").exists()

    # --unattended accepts sealed generations only ...
    proc, alloc = attempt("--restore", f"{run_a}/{warm['name']}", "--unattended")
    _assert_refused(proc, alloc, "unattended continuation accepts sealed generations only", f"{run_a}/{warm['name']} is a warm generation")
    # ... but an operator may choose a warm one, and is told what it is
    w = Managed(_allocation(home, tmp_path, "alloc-warm"), extra=("--restore", f"{run_a}/{warm['name']}"))
    try:
        assert w.status()["restored_from"]["class"] == "warm" and w.tid == a.tid
        log = (w.run_dir / "runner.log").read_text()
        assert "(warm, captured" in log and "consistent per database only" in log
        assert w.quit() == 0
    finally:
        w.close()

    # pending work: an active goal and queued items in a sealed generation
    p = Managed(_allocation(home, tmp_path, "alloc-source-busy"))
    try:
        p.cmd("turn 300")
        p.cmd("goal active")
        p.cmd("queue add")
        p.cmd("queue add")
        p.wait_synced()
        assert p.quit() == 0
    finally:
        p.close()
    busy = f"{p.run_dir.name}/{p.status()['last_generation']['name']}"
    proc, alloc = attempt("--restore", busy, "--unattended")
    _assert_refused(proc, alloc, "unattended continuation refused: generation " + busy,
                    f"an active goal on thread(s) {p.tid}", f"2 queued item(s) on thread {p.tid}",
                    "pause it in Codex", "executes once on resume")
    # attended, the same generation restores; the pending work is reported and handed over untouched
    q = Managed(_allocation(home, tmp_path, "alloc-attended"), extra=("--restore", busy))
    try:
        assert q.tid == p.tid
        assert q.status()["pending_work"] == {"active_goals": [p.tid], "queued": {p.tid: 2}}
        assert q.sees["goals"] == [[p.tid, "active"]] and q.sees["queue"] == {p.tid: 2}
        assert f"pending work: an active goal on thread(s) {p.tid}; 2 queued item(s) on thread {p.tid}" in (q.run_dir / "runner.log").read_text()
        assert q.quit() == 0
    finally:
        q.close()
    # queued rows alone trip the gate too (there is no hold for the queue)
    v = Managed(_allocation(home, tmp_path, "alloc-source-queued"))
    try:
        v.cmd("turn 250")
        v.cmd("queue add")
        v.wait_synced()
        assert v.quit() == 0
    finally:
        v.close()
    queued = f"{v.run_dir.name}/{v.status()['last_generation']['name']}"
    proc, alloc = attempt("--restore", queued, "--unattended")
    _assert_refused(proc, alloc, f"generation {queued} holds 1 queued item(s) on thread {v.tid}. An active goal")
    assert "active goal on thread(s)" not in proc.stderr
    # a paused goal is the hold: unattended continuation proceeds
    r = Managed(_allocation(home, tmp_path, "alloc-source-paused"))
    try:
        r.cmd("turn 200")
        r.cmd("goal paused")
        r.wait_synced()
        assert r.quit() == 0
    finally:
        r.close()
    held = f"{r.run_dir.name}/{r.status()['last_generation']['name']}"
    s = Managed(_allocation(home, tmp_path, "alloc-unattended"), extra=("--restore", held, "--unattended"))
    try:
        assert s.tid == r.tid and s.status()["pending_work"] == {"active_goals": [], "queued": {}}
        assert s.sees["goals"] == [[r.tid, "paused"]]
        assert s.quit() == 0
    finally:
        s.close()


@needs_bwrap
def test_reuse_in_place_continues_the_run_and_refuses_a_live_owner(home, tmp_path):
    """Case 3: the same allocation launches again: same run, same stores, the
    managed history kept, replication and generation numbering continuing;
    --restore yields to local state; --fresh starts a new run; a live recorded
    pid refuses; a latest run without db/ is not reusable."""
    scope_dir = home["local"] / "state" / "codex" / home["scope"]
    a = Managed(home)
    try:
        a.cmd("turn 300")
        a.cmd("goal active")
        a.wait_synced()
        assert a.quit() == 0
    finally:
        a.close()
    first = a.status()
    sealed = first["last_generation"]
    assert sealed["class"] == "sealed"
    first_owner = json.loads((a.run_dir / "owner.json").read_text())
    history = a.run_dir / "home" / "history.jsonl"
    assert history.read_text() == "shared\n", "the bound copy of the shared file"
    history.write_text('shared\n{"marker": "between runs"}\n')
    size_a = a.local_rollout.stat().st_size
    assert a.replica.stat().st_size == size_a

    # the gate applies to the stores reused in place as well: unattended, the active goal refuses
    proc = _attempt(home, "--unattended")
    assert proc.returncode == 70 and "FAKE_READY" not in proc.stdout
    assert f"unattended continuation refused: this allocation's run {a.run_dir.name} holds an active goal on thread(s) {a.tid}" in proc.stderr
    assert (scope_dir / "latest-run").read_text().strip() == a.run_dir.name and (a.run_dir / "db" / "goals_1.sqlite").exists()
    assert json.loads((a.run_dir / "owner.json").read_text()) == first_owner, "a refusal rewrites no ownership"
    assert sorted(p.name for p in scope_dir.iterdir() if p.is_dir()) == [a.run_dir.name]

    b = Managed(home)
    try:
        assert b.run_dir == a.run_dir and b.tid == a.tid and b.durable_run == a.durable_run
        status = b.status()
        assert status["reused_run"] is True and status["mode"] == "reuse" and "restore_ignored" not in status
        assert status["transitions"][:len(first["transitions"])] == first["transitions"]
        assert [t["state"] for t in status["transitions"][len(first["transitions"]):]][:2] == ["refused", "starting"]
        assert status["pending_work"] == {"active_goals": [a.tid], "queued": {}} and status["restored_from"] is None
        assert b.sees["goals"] == [[a.tid, "active"]]
        assert "exit" not in status and "outcome" not in status, "the previous launch's transient keys are dropped"
        assert f"pending work: an active goal on thread(s) {a.tid}" in (b.run_dir / "runner.log").read_text()
        owner = json.loads((b.run_dir / "owner.json").read_text())
        assert owner["reused_run"] is True and owner["runner_pid"] == b.proc.pid != first_owner["runner_pid"]
        assert owner["previous"] == [{"runner_pid": first_owner["runner_pid"], "child_pid": first_owner["child_pid"], "started": first_owner["started"]}]
        assert owner["child_pid"] != first_owner["child_pid"] and owner["first_started"] == first_owner["started"]
        assert history.read_text() == 'shared\n{"marker": "between runs"}\n', "reuse never overwrites the managed history"
        assert (b.run_dir / "home" / "AGENTS.md").read_text() == "GLOBAL\n"
        assert b.sees["projection"] == {a.tid: [size_a, 1]}
        assert json.loads((b.run_dir / "run.json").read_text()) == json.loads((b.durable_run / "run.json").read_text())
        b.cmd("turn 450")
        b.wait_synced()
        assert b.local_rollout.stat().st_size > size_a and b.replica.read_bytes() == b.local_rollout.read_bytes()
        status = _request(home)
        assert int(status["last_generation"]["name"][:6]) > int(sealed["name"][:6])
        manifest = _manifest(status)
        assert manifest["run"] == a.run_dir.name and manifest["parent"] is None
        assert manifest["threads"][a.tid]["next_rollout_ordinal"] == 2
        assert "reusing this allocation's run with its local databases and 1 rollout(s)" in (b.run_dir / "runner.log").read_text()
        assert b.quit() == 0
        assert "reusing this allocation's run" in b.proc.stderr.read()
    finally:
        b.close()

    selector = f"{a.run_dir.name}/{sealed['name']}"
    c = Managed(home, extra=("--restore", selector))
    try:
        assert c.run_dir == a.run_dir and c.tid == a.tid
        status = c.status()
        assert status["restore_ignored"] == selector and status["reused_run"] is True and status["restored_from"] is None
        assert f"IGNORING --restore {selector}" in (c.run_dir / "runner.log").read_text()
        assert json.loads((c.run_dir / "owner.json").read_text())["restore_ignored"] == selector
        assert c.quit() == 0
    finally:
        c.close()

    d = Managed(home, extra=("--fresh",))
    try:
        assert d.run_dir != a.run_dir and d.tid != a.tid
        status = d.status()
        assert status["reused_run"] is False and status["mode"] == "fresh" and status["transitions"][0]["state"] == "starting"
        assert (scope_dir / "latest-run").read_text().strip() == d.run_dir.name
        assert "--fresh: starting a new run although" in (d.run_dir / "runner.log").read_text()
        assert d.quit() == 0
    finally:
        d.close()
    assert (a.run_dir / "db" / "state_5.sqlite").exists(), "the earlier run stays on disk"

    sleeper = subprocess.Popen(["sleep", "120"])
    try:
        owner_path = d.run_dir / "owner.json"
        owner = json.loads(owner_path.read_text())
        owner["child_pid"] = sleeper.pid
        owner_path.write_text(json.dumps(owner))
        proc = _attempt(home)
        assert proc.returncode == 70 and f"records a live child_pid {sleeper.pid}" in proc.stderr and "second writer" in proc.stderr
        assert "FAKE_READY" not in proc.stdout
        assert (scope_dir / "latest-run").read_text().strip() == d.run_dir.name and (d.run_dir / "db" / "state_5.sqlite").exists()
        assert sorted(p.name for p in scope_dir.iterdir() if p.is_dir()) == sorted([a.run_dir.name, d.run_dir.name]), "no run directory was created"
    finally:
        sleeper.kill()
        sleeper.wait()
    e = Managed(home)
    try:
        assert e.run_dir == d.run_dir and e.status()["reused_run"] is True
        assert e.quit() == 0
    finally:
        e.close()

    stub = scope_dir / ("f" * 32)
    stub.mkdir()
    (stub / "owner.json").write_text(json.dumps({"runner_pid": 1, "child_pid": 1}))
    (scope_dir / "latest-run").write_text("f" * 32 + "\n")
    f = Managed(home)
    try:
        assert f.run_dir.name not in (a.run_dir.name, d.run_dir.name, "f" * 32)
        assert f.status()["reused_run"] is False and f.status()["mode"] == "fresh"
        assert f.quit() == 0
    finally:
        f.close()


@needs_bwrap
@pytest.mark.skipif(not shutil.which("codex"), reason="Codex CLI not installed")
def test_a_real_codex_thread_resumes_from_a_restored_generation(tmp_path):
    """Case 4: run A (two turns through the loopback fake model, stopped by
    stdin EOF) seals a generation; run B restores it in another local root,
    resumes the thread and runs a third turn whose model request carries both
    earlier texts; the rollout it resumed is the one placed from the replica."""
    from sucoder.agent_mode import config_override
    from tests.codex_app_server import AppServer, FakeModel, Home, thread_row

    class ManagedHome(Home):
        """The harness's private home plus the local root the runner needs."""

        def __init__(self, root, local_root, **kwargs):
            super().__init__(root, **kwargs)
            self.local_root = local_root

        def env(self, **extra):
            extra.setdefault("SUCODER_LOCAL_ROOT", str(self.local_root))
            return super().env(**extra)

    root = tmp_path.resolve()
    workspace = root / "ws"
    workspace.mkdir()
    codex_home = root / "codex-home"
    durable = root / "durable"
    durable.mkdir()
    scope = "ef" * 32
    placeholder = ["-c", config_override("sqlite_home", PLACEHOLDER)]

    def allocation(name):
        local = root / name
        local.mkdir()
        return {"local": local, "scope": scope}

    def prefix(*flags):
        return [sys.executable, str(RUNNER), "--scope", scope, "--durable-root", str(durable), "--checkpoint-timeout", "40", *flags, "--"]

    def replicated(alloc, tid, local):
        def done(st):
            entry = st.get("replica", {}).get("threads", {}).get(tid)
            return entry is not None and entry["replicated_bytes"] == entry["local_bytes"] == local.stat().st_size > 0
        return _wait(alloc, done, "the replica", timeout=30)[0]

    def sealed(alloc):
        return _wait(alloc, lambda st: st["state"] == "exited" and st["checkpoint"]["state"] in ("idle", "failed")
                     and (st.get("last_generation") or {}).get("class") == "sealed", "the sealed generation", timeout=60)[0]

    alloc_a = allocation("local-a")
    home_a = ManagedHome(root, alloc_a["local"], codex_home=codex_home, trusted=[workspace])
    (codex_home / "sessions").mkdir()
    with FakeModel() as model:
        server = AppServer(home_a, model, workspace, prefix=prefix(), extra_options=placeholder, label="source")
        server.start()
        try:
            server.initialize()
            tid = server.thread_start()
            status, run_a = _wait(alloc_a, lambda st: st["state"] in ("placed", "misplaced"), "placement", timeout=60)
            assert status["state"] == "placed", status
            for i in range(2):
                assert server.turn(tid, f"MARKR_{i}").ok
            rollouts = list((run_a / "home" / "sessions").rglob(f"rollout-*-{tid}.jsonl"))
            assert len(rollouts) == 1
            local_a = rollouts[0]
            relative = str(local_a.relative_to(run_a / "home" / "sessions"))
            replicated(alloc_a, tid, local_a)
            stop = server.stop(timeout=60)
            assert stop.clean, (stop.steps, stop.exit_code, server.stderr_text())
        finally:
            if server.running():
                server.kill()
        status_a = sealed(alloc_a)
        gen = status_a["last_generation"]
        manifest_a = _manifest(status_a)
        thread_a = manifest_a["threads"][tid]
        assert thread_a["relative_path"] == relative
        assert thread_a["next_rollout_byte_offset"] <= thread_a["replica_bytes"] == local_a.stat().st_size
        recorded = thread_row(run_a / "db", tid)["rollout_path"]
        assert recorded == str(codex_home / "sessions" / relative), "canonical, under the shared home"
        source = local_a.read_bytes()
        selector = f"{run_a.name}/{gen['name']}"

        alloc_b = allocation("local-b")
        home_b = ManagedHome(root, alloc_b["local"], codex_home=codex_home, trusted=[workspace])
        server = AppServer(home_b, model, workspace, prefix=prefix("--restore", selector), extra_options=placeholder, label="restored")
        server.start()
        try:
            server.initialize()
            status_b, run_b = _wait(alloc_b, lambda st: st["state"] in ("starting", "launched", "placed"), "the restored run", timeout=60)
            assert run_b.name != run_a.name and status_b["restored_from"]["run"] == run_a.name
            assert status_b["restored_from"]["generation"] == gen["name"] and status_b["restored_from"]["threads"] == 1
            assert status_b["pending_work"] == {"active_goals": [], "queued": {}}
            placed = run_b / "home" / "sessions" / relative
            assert placed.read_bytes() == source
            replica_b = durable / "codex" / scope / "runs" / run_b.name / "rollouts" / f"{tid}.jsonl"
            assert replica_b.read_bytes() == source
            before = model.count()
            resumed = server.thread_resume(tid)
            assert len(resumed["thread"]["turns"]) == 2 and resumed["thread"]["path"] == recorded
            assert model.wait_for_requests(before, 1, timeout=1.0) == [], "thread/resume alone must not call the model"
            result = server.turn(tid, "MARKR_2")
            assert result.ok, result.completed
            requests = model.requests_since(result.requests_before)
            assert len(requests) == 1
            assert [t for t in FakeModel.user_texts(requests[0]) if t.startswith("MARKR_")] == ["MARKR_0", "MARKR_1", "MARKR_2"]
            assert placed.stat().st_size > len(source) and placed.read_bytes()[:len(source)] == source, "the resumed rollout is the placed one"
            assert not list((codex_home / "sessions").rglob("*")), "the shared sessions directory stays empty"
            replicated(alloc_b, tid, placed)
            assert replica_b.read_bytes() == placed.read_bytes()
            assert _status(alloc_b)[0]["state"] == "placed"
            stop = server.stop(timeout=60)
            assert stop.clean, (stop.steps, stop.exit_code, server.stderr_text())
        finally:
            if server.running():
                server.kill()
        status_b = sealed(alloc_b)
        manifest_b = _manifest(status_b)
        assert manifest_b["parent"] == {"run": run_a.name, "generation": gen["name"], "class": "sealed", "manifest_sha256": gen["manifest_sha256"]}
        thread_b = manifest_b["threads"][tid]
        assert thread_b["next_rollout_ordinal"] > thread_a["next_rollout_ordinal"]
        assert thread_b["next_rollout_byte_offset"] <= thread_b["replica_bytes"] == placed.stat().st_size
        assert json.loads((run_b / "run.json").read_text())["parent"] == manifest_b["parent"]
