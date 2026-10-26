"""Managed Codex state, Slice 2: rollout replication, database generations
and their publication (first half), then restore from a generation, the
pending-work gate and same-allocation reuse in place (second half), in the
node-side runner.

Self-contained on purpose (the launcher-side module is edited separately):
the fake Codex here is scriptable over stdin -- ``turn N`` grows the rollout
by about N bytes and advances the history projection the way Codex does,
``shrink N`` truncates the rollout (the append-only contract broken on
purpose), ``replace`` renames a copy of the rollout over it (the same bytes
from another inode), ``thread`` starts a second thread whose rollout holds only its
session_meta line, ``drop projection`` drops the projection table from the
history store, ``goal <status>`` and ``queue add`` write the rows the gate
checks, ``extra`` opens an unclassified store, ``quit`` exits 0, SIGTERM
exits 37 (``FAKE_SIGTERM_EXIT`` overrides the code, so a drained fake can
seal) -- and, like
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
import importlib.util
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import tty
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
state.execute("create table if not exists threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL, cli_version TEXT NOT NULL DEFAULT '', "
              "cwd TEXT NOT NULL DEFAULT '', archived INTEGER NOT NULL DEFAULT 0)")
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
sees = {"threads": [], "projection": {}, "goals": [], "queue": {}, "cwds": {}}
existing = state.execute("select id, rollout_path from threads order by rowid desc").fetchall()
for row_id, path in existing:
    try:
        size = os.path.getsize(path)
    except OSError:
        size = None
    sees["threads"].append({"id": row_id, "rollout_path": path, "rollout_bytes": size})
for row in history.execute("select thread_id, next_rollout_byte_offset, next_rollout_ordinal from thread_history_projection_state"):
    sees["projection"][row[0]] = [row[1], row[2]]
sees["cwds"] = dict(state.execute("select id, cwd from threads").fetchall())
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
    state.execute("insert or replace into threads (id, rollout_path, cli_version, cwd) values (?, ?, ?, ?)",
                  (tid, rollout_path, "0.162.1", os.getcwd()))
    ordinal = turn = 0
signal.signal(signal.SIGTERM, lambda s, f: os._exit(int(os.environ.get("FAKE_SIGTERM_EXIT", "37"))))
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
    elif words[0] == "shrink":
        os.ftruncate(rollout.fileno(), int(words[1]))
        reply = "ok %d" % os.path.getsize(rollout_path)
    elif words[0] == "replace":
        # another file with the same bytes renamed over the rollout: a new inode at the path
        with open(rollout_path, "rb") as current:
            data = current.read()
        with open(rollout_path + ".new", "wb") as fresh:
            fresh.write(data)
        os.rename(rollout_path + ".new", rollout_path)
        rollout.close()
        rollout = open(rollout_path, "ab", buffering=0)
        reply = "ok %d" % os.path.getsize(rollout_path)
    elif words[0] == "thread":
        other = str(uuid.uuid4())
        day = os.path.join(home, "sessions", "2026", "10", "15")
        os.makedirs(day, exist_ok=True)
        other_path = os.path.join(day, "rollout-2026-10-15T00-00-00-%s.jsonl" % other)
        other_rollout = open(other_path, "ab", buffering=0)
        other_rollout.write((json.dumps({"type": "session_meta", "payload": {"id": other}}) + "\n").encode())
        held.append(other_rollout)
        # ``thread [CWD [archived]]``: the cwd defaults to the child's own
        state.execute("insert or replace into threads (id, rollout_path, cli_version, cwd, archived) values (?, ?, ?, ?, ?)",
                      (other, other_path, "0.162.1", words[1] if len(words) > 1 else os.getcwd(),
                       1 if words[2:3] == ["archived"] else 0))
        reply = "ok %s %s" % (other, other_path)
    elif words[0] == "drop":
        history.execute("drop table thread_history_projection_state")
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
    """The runner with the scriptable fake under it, placed and ready.  With
    ``tty`` the runner's stdin and stdout are the slave of a pty in raw mode
    (no echo, no newline translation), so ``isatty()`` is true for it and for
    the fake, and the test speaks through the master; stderr stays a pipe so
    the runner's own lines can be read."""

    def __init__(self, home, python=None, timeout=30.0, extra=(), env=None, tty=False, cwd=None):
        python = python or sys.executable
        self.home = home
        argv = [python, str(RUNNER), "--scope", home["scope"], "--durable-root", str(home["durable"]),
                "--checkpoint-timeout", str(timeout), *extra, "--",
                python, str(home["fake"]), "-c", f'sqlite_home="{PLACEHOLDER}"', "app-server"]
        self.master = None
        self.buffer = b""
        if tty:
            self.master, slave = os.openpty()
            _setraw(slave)
            self.proc = subprocess.Popen(argv, env=_env(home, **(env or {})), stdin=slave, stdout=slave, stderr=subprocess.PIPE,
                                         text=True, cwd=cwd)
            os.close(slave)
        else:
            self.proc = subprocess.Popen(argv, env=_env(home, **(env or {})), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, cwd=cwd)
        self.sees = None            # what the fake found in the stores it was handed
        while True:
            line = self._readline()
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

    def _readline(self):
        if self.master is None:
            return self.proc.stdout.readline()
        while b"\n" not in self.buffer:
            try:
                chunk = os.read(self.master, 4096)
            except OSError:                 # EIO once every holder of the slave has gone
                chunk = b""
            if not chunk:
                line, self.buffer = self.buffer, b""
                return line.decode()
            self.buffer += chunk
        line, _, self.buffer = self.buffer.partition(b"\n")
        return line.decode() + "\n"

    def _write(self, text):
        if self.master is None:
            self.proc.stdin.write(text)
            self.proc.stdin.flush()
        else:
            os.write(self.master, text.encode())

    def cmd(self, line):
        self._write(line + "\n")
        reply = self._readline().strip()
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
        self._write("quit\n")
        return self.proc.wait(timeout=timeout)

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)
        if self.master is not None:
            os.close(self.master)
            self.master = None


def _setraw(fd):
    tty.setraw(fd)


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
        lineage = json.loads((m.durable_run.parent.parent / "lineages" / (m.run_dir.name + ".json")).read_text())
        assert lineage["child_run"] == m.run_dir.name and lineage["parent"] is None
        assert lineage["allocation"] == run_json["allocation"] and lineage["created_at"] > 0
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
    refused_runs = [d for d in (alloc["local"] / "state" / "codex" / home["scope"]).iterdir() if d.is_dir()]
    assert len(refused_runs) == 1
    refused_status = json.loads((refused_runs[0] / "status.json").read_text())
    assert refused_status["pending_work"] == {"active_goals": [p.tid], "queued": {p.tid: 2}}, "the pending work found is recorded by the refusal"
    assert [t["state"] for t in refused_status["transitions"]] == ["restoring", "refused"]
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
    refused = json.loads((a.run_dir / "status.json").read_text())
    assert refused["state"] == "refused" and refused["pending_work"] == {"active_goals": [a.tid], "queued": {}}, "the refusal records what it found"

    agents = a.run_dir / "home" / "AGENTS.md"
    assert agents.read_text() == "GLOBAL\n", "the bound copy of the shared file"
    agents.write_text("GLOBAL\nlocal marker\n")
    run_json_before = (a.run_dir / "run.json").read_text()
    assert run_json_before == (a.durable_run / "run.json").read_text()
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
        assert owner["runner_starttime"] == _starttime(b.proc.pid)
        assert owner["previous"] == [{"runner_pid": first_owner["runner_pid"], "child_pid": first_owner["child_pid"], "started": first_owner["started"]}]
        assert owner["child_pid"] != first_owner["child_pid"] and owner["first_started"] == first_owner["started"]
        assert history.read_text() == 'shared\n{"marker": "between runs"}\n', "reuse never overwrites the managed history"
        assert agents.read_text() == "GLOBAL\nlocal marker\n", "the local AGENTS copy is kept, not recopied from the shared home"
        assert b.sees["projection"] == {a.tid: [size_a, 1]}
        assert (b.run_dir / "run.json").read_text() == run_json_before == (b.durable_run / "run.json").read_text(), "provenance is never rewritten"
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


# ----------------------------------------------------------------- review follow-ups on the first half

TID = "0f0f0f0f-0f0f-0f0f-0f0f-0f0f0f0f0f0f"


@pytest.fixture
def runner(tmp_path):
    """The runner as a module of its own (it is a staged file, not a package
    module), fresh per test so monkeypatches and its log never leak."""
    spec = importlib.util.spec_from_file_location("codex_state_runner_under_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.LOG_FILE["path"] = str(tmp_path / "runner.log")
    module.LOG_FILE["quiet"] = True
    return module


def _log(tmp_path):
    return (tmp_path / "runner.log").read_text()


def test_installation_identity_is_bound_to_private_local_file(runner, tmp_path):
    canonical, local = tmp_path / "canonical", tmp_path / "local"
    (canonical / "sessions").mkdir(parents=True)
    local.mkdir()
    identity = "11111111-1111-4111-8111-111111111111"
    (canonical / "installation_id").write_text(identity)
    binds, skipped = runner.prepare_binds(str(canonical), str(local))
    assert (str(local / "installation_id"), str(canonical / "installation_id")) in binds
    assert (local / "installation_id").read_text() == identity
    assert (local / "installation_id").stat().st_mode & 0o077 == 0
    (local / "installation_id").write_text("local native update")
    assert (canonical / "installation_id").read_text() == identity


def test_reused_local_installation_identity_is_not_overwritten(runner, tmp_path):
    canonical, local = tmp_path / "canonical", tmp_path / "local"
    (canonical / "sessions").mkdir(parents=True)
    local.mkdir()
    (canonical / "installation_id").write_text("cold seed")
    runner.prepare_binds(str(canonical), str(local))
    (local / "installation_id").write_text("existing native identity")
    (canonical / "installation_id").write_text("changed cold seed")
    binds, skipped = runner.prepare_binds(str(canonical), str(local), keep_existing=True)
    assert (str(local / "installation_id"), str(canonical / "installation_id")) in binds
    assert (local / "installation_id").read_text() == "existing native identity"


def test_remote_owner_identity_binding_still_overrides_normal_identity(runner, tmp_path):
    canonical, local, database = tmp_path / "canonical", tmp_path / "local", tmp_path / "db"
    (canonical / "sessions").mkdir(parents=True)
    local.mkdir()
    database.mkdir()
    identity = "11111111-1111-4111-8111-111111111111"
    (canonical / "installation_id").write_text(identity)
    binds, skipped = runner.prepare_binds(str(canonical), str(local))
    remote_identity, remote_binding = runner.prepare_remote_identity(str(database), str(local), str(canonical))
    binds.append(remote_binding)  # run() preserves this ordering
    argv = runner.bwrap_argv(binds, ["true"])
    target = str(canonical / "installation_id")
    sources = [argv[index + 1] for index, value in enumerate(argv)
               if value == "--bind" and argv[index + 2] == target]
    assert sources == [str(local / "installation_id"), remote_binding[0]]
    assert Path(remote_binding[0]).read_text() == remote_identity
    assert remote_identity != identity
    assert (canonical / "installation_id").read_text() == identity


class InProcess:
    """A Checkpointer over real stores, a real rollout and a real Replicator
    whose passes the test drives by hand; nothing is started, a capture is
    begun and joined explicitly."""

    def __init__(self, runner, tmp_path, timeout=5.0, rollout=b'{"type": "session_meta"}\n'):
        self.runner = runner
        self.scope = "ab" * 32
        self.run_id = "1" * 32
        self.codex_home = tmp_path / "codex-home"
        (self.codex_home / "sessions").mkdir(parents=True)
        self.run_dir = tmp_path / "local" / "state" / "codex" / self.scope / self.run_id
        self.db_dir = self.run_dir / "db"
        self.sessions = self.run_dir / "home" / "sessions"
        for d in (self.db_dir, self.sessions, self.run_dir / "snapshots"):
            d.mkdir(parents=True)
        self.relative = f"2026/10/15/rollout-2026-10-15T00-00-00-{TID}.jsonl"
        self.local_rollout = self.sessions / self.relative
        self.local_rollout.parent.mkdir(parents=True)
        self.local_rollout.write_bytes(rollout)
        self.durable_base = tmp_path / "durable"
        self.durable_run = self.durable_base / "codex" / self.scope / "runs" / self.run_id
        runner.ensure_durable_run(str(self.durable_run), str(self.durable_base))
        self.rollouts = self.durable_run / "rollouts"
        self.generations = self.durable_run / "generations"
        self.replica = self.rollouts / f"{TID}.jsonl"
        state = sqlite3.connect(str(self.db_dir / "state_5.sqlite"))
        state.execute("create table threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL, cli_version TEXT NOT NULL DEFAULT '')")
        state.execute("insert into threads values (?, ?, ?)", (TID, str(self.codex_home / "sessions" / self.relative), "0.162.1"))
        state.commit()
        state.close()
        history = sqlite3.connect(str(self.db_dir / "thread_history_1.sqlite"))
        history.execute("create table thread_history_projection_state (thread_id TEXT PRIMARY KEY, "
                        "next_rollout_byte_offset INTEGER NOT NULL, next_rollout_ordinal INTEGER NOT NULL)")
        history.execute("insert into thread_history_projection_state values (?, ?, ?)", (TID, len(rollout), 1))
        history.commit()
        history.close()
        self.status = runner.Status(str(self.run_dir / "status.json"))
        self.replicator = runner.Replicator(str(self.sessions), str(self.rollouts), str(self.run_dir / "replica.json"), self.status)
        placed = threading.Event()
        placed.set()
        meta = {"scope": self.scope, "run": self.run_id, "parent": None, "allocation": {"job": None, "node": "test"},
                "codex_home": str(self.codex_home), "codex_home_real": str(self.codex_home)}
        self.cp = runner.Checkpointer(str(self.run_dir), str(self.db_dir), str(self.sessions), str(self.durable_run),
                                      self.replicator, self.status, meta, 10.0, timeout, placed)
        self.cp.ensure_seq()            # the scheduler thread's first act, done here since it is never started

    def capture(self, trigger="request", cls="warm", timeout=30):
        worker = self.cp.begin(trigger, cls)
        assert worker is not None
        worker.join(timeout)
        assert not worker.is_alive()
        return self.cp.last_outcome, self.cp.reason

    def status_json(self):
        return json.loads((self.run_dir / "status.json").read_text())


@needs_bwrap
def test_a_rollout_that_shrinks_while_replication_lags_is_an_anomaly_not_growth(home):
    """Finding 1: local 1200 -> 600 with 400 replicated.  The old check (local
    shorter than the replica) does not fire; the bytes beyond 400 are no longer
    the ones that followed the replicated prefix, so nothing may be appended."""
    m = Managed(home, timeout=3)
    try:
        m.cmd("turn 400")
        m.wait_synced()
        replicated = m.replica.stat().st_size
        rollouts = m.durable_run / "rollouts"
        os.chmod(rollouts, 0)           # replication stalls while the rollout grows
        try:
            m.cmd("turn 800")
            grown = m.local_rollout.stat().st_size
            _wait(home, lambda st: st["replica"]["threads"][m.tid]["local_bytes"] == grown and m.tid in st["replica"]["errors"],
                  "the pass that observed the growth")
            target = (replicated + grown) // 2
            assert replicated < target < grown
            assert m.cmd(f"shrink {target}") == f"ok {target}"
            status = _wait(home, lambda st: st["replica"]["anomalies"].get(m.tid) == "shrunk", "the anomaly")[0]
        finally:
            os.chmod(rollouts, 0o700)
        assert status["replica"]["threads"][m.tid]["local_bytes"] == target
        status = _request(home)         # a capture, and at least one more replication pass with the durable side back
        assert status["checkpoint"]["last_outcome"] == "failed" and "its rollout shrunk" in status["checkpoint"]["reason"]
        assert m.replica.stat().st_size == replicated, "nothing was appended after the shrink"
        assert m.replica.read_bytes() == m.local_rollout.read_bytes()[:replicated]
        assert status["replica"]["threads"][m.tid]["replicated_bytes"] == replicated
        assert status["replica"]["anomalies"] == {m.tid: "shrunk"} and status["replica"]["lag_bytes"] == 0
        log = (m.run_dir / "runner.log").read_text()
        assert (f"rollout of {m.tid} shrank to {target} bytes, shorter than it was at the previous pass ({grown} bytes; "
                f"{replicated} replicated); replication of this thread stops") in log
    finally:
        m.close()


@needs_bwrap
def test_an_unreadable_projection_table_fails_the_capture_instead_of_publishing_without_watermarks(home):
    """Finding 2: the copy exists and passes quick_check but has no projection
    table: a storage layout this runner does not know, never zero watermarks."""
    m = Managed(home)
    try:
        m.cmd("turn 200")
        first = _request(home)["last_generation"]
        m.cmd("drop projection")
        status = _request(home)
        checkpoint = status["checkpoint"]
        assert checkpoint["last_outcome"] == "failed" and checkpoint["state"] == "failed"
        assert "thread_history_1.sqlite: cannot read thread_history_projection_state" in checkpoint["reason"]
        assert "no such table" in checkpoint["reason"] and "storage layout is not the one this runner knows" in checkpoint["reason"]
        assert status["last_generation"] == first and sorted(os.listdir(m.generations))[-1] == first["name"]
        assert not list(m.durable_run.glob(".incoming-*"))
        assert "notes" not in _manifest(status)
    finally:
        m.close()


def test_new_replica_files_and_durable_directories_are_followed_by_a_directory_fsync(runner, tmp_path, monkeypatch):
    """Finding 3: a created directory's parent is synced, a new replica's
    directory is synced once after the file's first fsync, and a directory sync
    that fails is retried until it succeeds."""
    synced = []
    real = runner.fsync_path

    def recording(path):
        synced.append(path)
        real(path)
    monkeypatch.setattr(runner, "fsync_path", recording)
    base = tmp_path / "durable"
    run = base / "codex" / "ab" / "runs" / "11"
    runner.ensure_durable_run(str(run), str(base))
    created = [base, base / "codex", base / "codex" / "ab", base / "codex" / "ab" / "runs", run, run / "rollouts", run / "generations"]
    assert synced == [str(d.parent) for d in created], "each mkdir is followed by an fsync of the parent"
    assert all(oct(d.stat().st_mode & 0o777) == "0o700" for d in created)
    synced.clear()
    runner.ensure_durable_run(str(run), str(base))
    assert synced == [], "existing directories are checked, not re-synced"
    # the base's parent is the operator's filesystem: its sync is best effort; below the base it is not
    unreadable = {str(tmp_path / "other")}

    def refusing(path):
        synced.append(path)
        if path in unreadable:
            raise PermissionError(13, "Permission denied")
        real(path)
    monkeypatch.setattr(runner, "fsync_path", refusing)
    base2 = tmp_path / "other" / "durable"
    (tmp_path / "other").mkdir()
    runner.ensure_durable_run(str(base2 / "codex" / "ab" / "runs" / "11"), str(base2))
    assert (base2 / "codex" / "ab" / "runs" / "11" / "generations").is_dir()
    assert f"created {base2} but could not sync its parent directory (PermissionError" in _log(tmp_path)
    unreadable.add(str(base2 / "codex"))
    with pytest.raises(runner.Refused, match="cannot create the durable run directory .*Permission denied"):
        runner.ensure_durable_run(str(base2 / "codex" / "cd" / "runs" / "11"), str(base2))
    assert not (base2 / "codex" / "cd" / "runs").exists()
    monkeypatch.setattr(runner, "fsync_path", recording)
    synced.clear()

    sessions = tmp_path / "sessions"
    rollout = sessions / "2026" / f"rollout-2026-10-15T00-00-00-{TID}.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_bytes(b"one\n")
    status = runner.Status(str(tmp_path / "status.json"))
    rep = runner.Replicator(str(sessions), str(run / "rollouts"), str(tmp_path / "replica.json"), status)
    rep.pass_once()
    assert synced == [str(run / "rollouts")] and (run / "rollouts" / f"{TID}.jsonl").read_bytes() == b"one\n"
    rollout.write_bytes(b"one\ntwo\n")
    rep.pass_once()
    assert synced == [str(run / "rollouts")], "the directory is synced once per replica file"
    assert (run / "rollouts" / f"{TID}.jsonl").read_bytes() == b"one\ntwo\n"

    tid2 = TID.replace("0f", "1e")
    (sessions / "2026" / f"rollout-2026-10-15T00-00-00-{tid2}.jsonl").write_bytes(b"three\n")

    def failing(path):
        synced.append(path)
        raise OSError(5, "Input/output error")
    monkeypatch.setattr(runner, "fsync_path", failing)
    rep.pass_once()
    assert synced[-1] == str(run / "rollouts") and (run / "rollouts" / f"{tid2}.jsonl").read_bytes() == b"three\n"
    assert rep.threads[tid2]["replicated_bytes"] == 0, "the bytes are in the file but do not count until its name is durable"
    assert "Input/output error" in rep.view()["errors"][tid2] and tid2 not in rep.dir_synced
    monkeypatch.setattr(runner, "fsync_path", recording)
    rep.pass_once()
    assert synced[-1] == str(run / "rollouts") and tid2 in rep.dir_synced and rep.view()["errors"] == {}
    assert rep.threads[tid2]["replicated_bytes"] == 6
    assert f"replication of {tid2} resumed at 6 bytes" in _log(tmp_path)


def test_a_delayed_acknowledgement_never_overwrites_a_newer_generation(runner, tmp_path):
    """Finding 4: the first capture's status write is held back until a second
    capture has begun, completed and been acknowledged; the first write then
    records the second generation, not its own."""
    h = InProcess(runner, tmp_path)
    h.replicator.pass_once()
    hold, release = threading.Event(), threading.Event()
    real = h.status.note_latest
    calls = []

    def delayed(compute):
        calls.append(compute)
        if len(calls) == 1:
            hold.set()
            assert release.wait(30)
        real(compute)
    h.status.note_latest = delayed
    first = h.cp.begin("request", "warm")
    assert hold.wait(30)
    a = h.cp.last_generation["name"]
    assert h.cp.inflight is None and h.cp.last_outcome == "published", "acknowledged in memory before the status write"
    second = h.cp.begin("request", "warm")
    assert second is not None, "the first capture no longer blocks a second"
    second.join(30)
    b = h.cp.last_generation["name"]
    assert b != a and int(b[:6]) == int(a[:6]) + 1
    release.set()
    first.join(30)
    status = h.status_json()
    assert status["last_generation"]["name"] == b and h.cp.last_generation["name"] == b
    assert status["checkpoint"]["state"] == "idle" and status["checkpoint"]["count"] == 2 and status["checkpoint"]["last_outcome"] == "published"
    assert sorted(os.listdir(h.generations)) == [a, b]


def test_status_write_failures_are_logged_once_and_never_stop_a_capture(runner, tmp_path):
    """Finding 5: status.json is best effort; begin, the capture's finally and
    report all go through, the error is logged once per distinct text."""
    h = InProcess(runner, tmp_path)
    h.replicator.pass_once()
    real = runner.write_json

    def no_space(path, data):
        if path.endswith("status.json"):
            raise OSError(28, "No space left on device")
        real(path, data)
    runner.write_json = no_space
    assert h.capture() == ("published", None)
    assert h.capture() == ("published", None)
    log = _log(tmp_path)
    assert log.count("status.json not written") == 1 and "No space left on device" in log
    assert log.count("published (warm, request") == 2 and len(os.listdir(h.generations)) == 2
    assert "last_generation" not in h.status_json(), "the file is the replication pass's, stale"
    runner.write_json = real
    h.status.note(x=1)
    assert h.status_json()["last_generation"]["name"] == h.cp.last_generation["name"], "the acknowledgement lands with the next successful write"

    def denied(path, data):
        if path.endswith("status.json"):
            raise OSError(13, "Permission denied")
        real(path, data)
    runner.write_json = denied
    h.status.note(y=2)
    h.status.note(y=3)
    log = _log(tmp_path)
    assert log.count("status.json not written") == 2 and "Permission denied" in log


@needs_bwrap
def test_every_required_store_disappearing_is_a_failure_not_a_skip(home):
    """Finding 6: with db/ empty after the stores were seen, the capture fails
    naming them instead of reporting nothing to capture."""
    m = Managed(home)
    try:
        m.cmd("turn 200")
        first = _request(home)["last_generation"]
        for path in (m.run_dir / "db").iterdir():
            path.unlink()
        status = _request(home)
        assert status["checkpoint"]["last_outcome"] == "failed"
        assert "required store disappeared: state_5.sqlite, thread_history_1.sqlite" in status["checkpoint"]["reason"]
        assert status["last_generation"] == first and sorted(os.listdir(m.generations))[-1] == first["name"]
    finally:
        m.close()


@needs_bwrap
def test_a_thread_discovered_after_the_backup_waits_for_its_replica(home):
    """Finding 8: a thread whose rollout exists at capture but which the
    replicator has not discovered yet is not durable until its replica reaches
    the rollout's length at capture; then it publishes with that length."""
    m = Managed(home, timeout=2, env={"SUCODER_CODEX_STATE_REPLICATION_INTERVAL": "8"})
    try:
        m.cmd("turn 300")
        m.wait_synced(timeout=30)       # right after a pass: the next one is eight seconds away
        reply = m.cmd("thread")
        tid2 = reply.split()[1]
        local2 = m.run_dir / "home" / "sessions" / f"2026/10/15/rollout-2026-10-15T00-00-00-{tid2}.jsonl"
        size2 = local2.stat().st_size
        assert size2 > 0
        status = _request(home)
        checkpoint = status["checkpoint"]
        assert checkpoint["last_outcome"] == "not_durable", checkpoint
        assert f"{tid2} has 0 replicated bytes, the projection needs 0 and the rollout held {size2}" in checkpoint["reason"]
        assert tid2 not in status["replica"]["threads"], "not discovered yet when the capture ended"
        local_only = sorted(os.listdir(m.run_dir / "snapshots"))[-1]
        kept = json.loads((m.run_dir / "snapshots" / local_only / "manifest.json").read_text())
        assert kept["not_durable"] == {tid2: {"replica_bytes": 0, "next_rollout_byte_offset": 0, "local_bytes": size2}}
        assert kept["threads"][tid2]["replica_bytes"] == 0 and kept["threads"][tid2]["next_rollout_ordinal"] == 0
        assert kept["threads"][m.tid]["replica_bytes"] >= kept["threads"][m.tid]["next_rollout_byte_offset"] > 0

        def discovered(st):
            entry = st["replica"]["threads"].get(tid2)
            return entry is not None and entry["replicated_bytes"] == entry["local_bytes"] == size2
        _wait(home, discovered, "the second thread's replica", timeout=30)
        status = _request(home)
        assert status["checkpoint"]["last_outcome"] == "published"
        thread = _manifest(status)["threads"][tid2]
        assert thread["replica_bytes"] == size2 and thread["next_rollout_byte_offset"] == 0
        assert (m.durable_run / "rollouts" / f"{tid2}.jsonl").read_bytes() == local2.read_bytes()
        assert thread["rollout_sha256_prefix"] == _sha256(local2)
    finally:
        m.close()


def test_holders_of_reads_the_links_and_stats_nothing(runner, tmp_path, monkeypatch):
    """Finding 9: a stat of a descriptor follows the link onto its filesystem;
    holders are found by the link text alone, deleted stores included."""
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    store = db_dir / "state_5.sqlite"
    store.write_bytes(b"")

    def no_stat(*args, **kwargs):
        raise AssertionError("holders_of must not stat anything: %r" % (args,))
    monkeypatch.setattr(runner.os, "stat", no_stat)
    monkeypatch.setattr(runner.os, "lstat", no_stat)
    assert runner.holders_of(str(db_dir)) == []
    with open(str(store), "rb"):
        assert runner.holders_of(str(db_dir)) == [os.getpid()]
        os.unlink(str(store))
        assert runner.holders_of(str(db_dir)) == [os.getpid()], "a deleted store still held is still held"
    assert runner.holders_of(str(db_dir)) == []


def test_final_shares_one_shutdown_deadline_with_a_capture_in_flight(runner, tmp_path):
    """Finding 9: one budget of --checkpoint-timeout for the exit.  A capture in
    flight that overruns it skips the final capture within that budget; one
    that ends early leaves the remainder to the final capture."""
    h = InProcess(runner, tmp_path, timeout=1.0)
    begun = []
    h.cp.begin = lambda *args, **kwargs: begun.append((args, kwargs))
    h.cp.inflight = {"thread": None, "deadline": time.monotonic() + 1.0, "generation": "000007-deadbeef", "trigger": "request",
                     "class": "warm", "stuck": False, "started": time.time()}
    started = time.monotonic()
    h.cp.final("sealed")
    elapsed = time.monotonic() - started
    assert 0.9 <= elapsed < 2.0 and begun == [], elapsed
    assert "final sealed capture skipped: capture 000007-deadbeef is still in flight" in _log(tmp_path)

    h2 = InProcess(runner, tmp_path / "two", timeout=1.0)
    h2.cp.inflight = dict(h.cp.inflight)
    threading.Timer(0.4, lambda: setattr(h2.cp, "inflight", None)).start()
    deadlines = []
    h2.cp.capture = lambda generation, trigger, cls, deadline, final_pass: deadlines.append((time.monotonic(), deadline, final_pass))
    started = time.monotonic()
    h2.cp.final("sealed")
    elapsed = time.monotonic() - started
    assert len(deadlines) == 1 and deadlines[0][2] is True
    begun_at, deadline = deadlines[0][0] - started, deadlines[0][1] - started
    assert 0.4 <= begun_at <= 0.7, begun_at
    assert 0.95 <= deadline <= 1.05, "the final capture's deadline is the one shutdown deadline, not a second timeout: %.2f" % deadline
    assert elapsed < 1.5 and h2.cp.count == 1     # the stubbed capture ends at once; the real one clears in-flight in its finally

    # the scheduler never got its listing of generations/ back: the exit path does not ask the durable side itself
    h3 = InProcess(runner, tmp_path / "three", timeout=1.0)
    h3.cp.seq = None
    h3.cp.highest_seq = lambda: pytest.fail("final() must not list the durable generations directory")
    started = time.monotonic()
    h3.cp.final("sealed")
    assert time.monotonic() - started < 0.5 and h3.cp.count == 0
    assert "final sealed capture skipped: the durable generations listing never completed" in _log(tmp_path)


@needs_bwrap
def test_a_symlinked_or_writable_durable_component_refuses_before_anything_is_created(home, tmp_path):
    """Finding 10: an existing component under the durable base must be a real
    directory, ours, not group- or world-writable; 0755 passes."""
    scope_dir = home["durable"] / "codex" / home["scope"]
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    scope_dir.parent.mkdir(parents=True)
    scope_dir.symlink_to(elsewhere)
    alloc = _allocation(home, tmp_path, "alloc-symlink")
    _assert_refused(_attempt(alloc), alloc, f"{scope_dir} is a symbolic link", "real directories only")
    assert os.listdir(elsewhere) == [] and not (scope_dir / "runs").exists()
    scope_dir.unlink()
    scope_dir.mkdir()
    for mode in (0o777, 0o775):
        scope_dir.chmod(mode)
        alloc = _allocation(home, tmp_path, "alloc-%s" % oct(mode))
        _assert_refused(_attempt(alloc), alloc, f"{scope_dir} is writable by its group or by everyone (mode {oct(mode)})",
                        f"chmod g-w,o-w {scope_dir}")
        assert os.listdir(scope_dir) == []
    scope_dir.chmod(0o755)
    m = Managed(home)
    try:
        assert (scope_dir / "runs" / m.run_dir.name / "generations").is_dir()
        assert oct((scope_dir / "runs").stat().st_mode & 0o777) == "0o700"
        assert m.quit() == 0
    finally:
        m.close()


def test_a_failed_sync_after_the_rename_is_not_acknowledged_until_the_sync_succeeds(runner, tmp_path, monkeypatch):
    """Finding 11, amended by addendum C: the rename succeeded and the manifest
    is complete, but the directory entry is not known to be durable, so the
    generation is recorded as unsynced and is not the last generation until a
    retry of the directory sync (before a later capture, or at the exit)
    succeeds; a later publication's successful sync covers it as well."""
    h = InProcess(runner, tmp_path)
    h.replicator.pass_once()
    real = runner.fsync_path
    failing = {"on": True}

    def maybe_failing(path):
        if failing["on"] and path == str(h.generations):
            raise OSError(5, "Input/output error")
        real(path)
    monkeypatch.setattr(runner, "fsync_path", maybe_failing)
    assert h.capture() == ("published", None)
    a = h.cp.pending_sync
    assert a is not None and a["sync_warning"] == "OSError: [Errno 5] Input/output error"
    assert h.cp.last_generation is None and h.status.durable_at is None, "no durable acknowledgement"
    assert Path(a["path"]).parent == h.generations and (Path(a["path"]) / "manifest.json").exists()
    assert not list(h.durable_run.glob(".incoming-*"))
    status = h.status_json()
    assert status["unsynced_generation"] == a and "last_generation" not in status and status["durable_age_seconds"] is None
    assert status["checkpoint"]["last_outcome"] == "published"
    log = _log(tmp_path)
    assert f"generation {a['name']} is published but the sync of {h.generations} failed" in log
    assert f"generation {a['name']} published (warm, request" in log
    # a second capture while the sync still fails: the retry fails too, and the newer one is the unsynced one
    assert h.capture() == ("published", None)
    b = h.cp.pending_sync
    assert b["name"] != a["name"] and h.cp.last_generation is None
    assert f"generation {a['name']} is still not acknowledged: the sync of {h.generations} failed again (OSError" in _log(tmp_path)
    assert h.status_json()["unsynced_generation"] == b
    # the retry succeeds: acknowledged, without the warning
    failing["on"] = False
    h.cp.retry_sync()
    last = h.cp.last_generation
    assert last["name"] == b["name"] and "sync_warning" not in last and last["synced_at"] > 0 and h.cp.pending_sync is None
    assert h.status.durable_at is not None
    status = h.status_json()
    assert status["last_generation"] == last and status["unsynced_generation"] is None and status["durable_age_seconds"] is not None
    assert f"generation {b['name']} acknowledged: {h.generations} synced on retry" in _log(tmp_path)
    # a later publication whose own sync succeeds covers an unsynced one (same directory)
    failing["on"] = True
    assert h.capture() == ("published", None)
    c = h.cp.pending_sync
    assert c is not None and c["name"] != b["name"] and h.cp.last_generation == last
    failing["on"] = False
    assert h.capture() == ("published", None)
    d = h.cp.last_generation
    assert d["name"] != c["name"] and h.cp.pending_sync is None and h.status_json()["unsynced_generation"] is None
    assert sorted(os.listdir(h.generations)) == sorted([b["name"], c["name"], d["name"]])
    assert (h.durable_run / "retiring" / a["name"]).is_dir(), "the successful sync permits retention of the newest three warm"
    # the exit retries as well: inside the final capture's worker, never on the exit path itself
    failing["on"] = True
    assert h.capture() == ("published", None)
    e = h.cp.pending_sync
    assert e is not None
    failing["on"] = False
    h.cp.final("sealed")
    assert h.cp.pending_sync is None and h.cp.last_generation["class"] == "sealed" and h.cp.last_generation["name"] != e["name"]
    assert f"generation {e['name']} acknowledged: {h.generations} synced on retry" in _log(tmp_path)


# ----------------------------------------------------------------- review follow-ups on the second half

def _published(runner, tmp_path, active_goal=False):
    """A sealed generation published by the in-process harness, as a restore
    source: (harness, selector).  With ``active_goal`` the stores carry a
    goals_1 row with status active for the harness thread."""
    h = InProcess(runner, tmp_path.resolve())
    if active_goal:
        goals = sqlite3.connect(str(h.db_dir / "goals_1.sqlite"))
        goals.execute("create table thread_goals (thread_id TEXT PRIMARY KEY NOT NULL, goal_id TEXT NOT NULL, status TEXT NOT NULL)")
        goals.execute("insert into thread_goals values (?, ?, ?)", (TID, "goal-1", "active"))
        goals.commit()
        goals.close()
    h.replicator.pass_once()
    assert h.capture(trigger="final", cls="sealed") == ("published", None)
    return h, f"{h.run_id}/{h.cp.last_generation['name']}"


def _restore_in_process(runner, h, local_root, monkeypatch, *flags):
    """``runner.main`` in this process against the harness's durable root, so
    a step can be failed by monkeypatch; a refusal returns before anything is
    launched.  Returns (exit code, the run directories left under the scope)."""
    local_root.mkdir(exist_ok=True)
    local_root = local_root.resolve()
    monkeypatch.setenv("SUCODER_LOCAL_ROOT", str(local_root))
    monkeypatch.setenv("CODEX_HOME", str(h.codex_home))
    monkeypatch.delenv("CODEX_SQLITE_HOME", raising=False)
    runner.LOG_FILE["path"] = str(local_root.parent / "runner.log")        # until the attempt has a run directory
    code = runner.main(["--scope", h.scope, "--durable-root", str(h.durable_base), *flags,
                        "--", "codex", "-c", f'sqlite_home="{PLACEHOLDER}"'])
    scope_dir = local_root / "state" / "codex" / h.scope
    runs = sorted(p for p in scope_dir.iterdir() if p.is_dir()) if scope_dir.is_dir() else []
    return code, runs


def _tree(path):
    """A digest of every file under ``path``: relative names, modes and bytes."""
    digest = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(str(path)):
        dirnames.sort()
        for name in sorted(filenames):
            full = Path(dirpath) / name
            digest.update(str(full.relative_to(path)).encode())
            digest.update(oct(full.stat().st_mode).encode())
            digest.update(full.read_bytes())
    return digest.hexdigest()


def _assert_rolled_back(run_dir, h, *needles):
    """Exit 70's aftermath: status refused with the reason; no db/, no staging,
    no home/sessions, no run.json; no durable run directory for the new id."""
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "refused", status
    for needle in needles:
        assert needle in status["reason"], (needle, status["reason"])
    assert not (run_dir / "db").exists() and not (run_dir / "restore-staging").exists() and not (run_dir / "run.json").exists()
    assert not (run_dir / "home" / "sessions").exists() and not list((run_dir / "home").rglob("*"))
    assert os.listdir(h.durable_base / "codex" / h.scope / "runs") == [h.run_id], "no durable run directory for the refused run"


@needs_bwrap
def test_a_refused_restore_rolls_back_everything_the_attempt_created(runner, tmp_path, monkeypatch):
    """Findings 1 and 4: a failure while staging the replicas, while writing
    run.json, while checking the replicas (before any durable write) and in
    the middle of the activation each exits 70 with the step named, leaves no
    db/, no home/sessions, no run.json and no durable run directory for the
    new run id, and the source run (generation and replica) untouched."""
    h, selector = _published(runner, tmp_path)
    before = _tree(h.durable_run)
    real_copy_whole, real_sha256_file, real_write_json, real_rename = runner.copy_whole, runner.sha256_file, runner.write_json, os.rename

    # a torn replica copy: the targets exist, the read failed
    def failing_copy(source, targets, prefix=0):
        for target in targets:
            Path(target).write_bytes(b"torn")
        raise OSError(5, "Input/output error")
    monkeypatch.setattr(runner, "copy_whole", failing_copy)
    code, runs = _restore_in_process(runner, h, tmp_path / "local-1", monkeypatch, "--restore", selector)
    assert code == 70 and len(runs) == 1
    _assert_rolled_back(runs[0], h, f"generation {selector}: cannot stage the rollout of thread {TID}: [Errno 5] Input/output error")
    log = (runs[0] / "runner.log").read_text()
    assert "restore rolled back: removed" in log and str(runs[0] / "restore-staging") in log
    assert str(h.durable_base / "codex" / h.scope / "runs" / runs[0].name) in log, "the new run's durable directory was removed"
    assert _tree(h.durable_run) == before
    monkeypatch.setattr(runner, "copy_whole", real_copy_whole)

    # the durable run.json cannot be written, after the replicas were staged
    def no_space(path, data):
        if path.endswith("run.json") and path.startswith(str(h.durable_base)):
            raise OSError(28, "No space left on device")
        real_write_json(path, data)
    monkeypatch.setattr(runner, "write_json", no_space)
    code, runs = _restore_in_process(runner, h, tmp_path / "local-2", monkeypatch, "--restore", selector)
    assert code == 70 and len(runs) == 1
    new_durable = h.durable_base / "codex" / h.scope / "runs" / runs[0].name
    _assert_rolled_back(runs[0], h, f"cannot write {new_durable / 'run.json'}: [Errno 28] No space left on device")
    assert _tree(h.durable_run) == before
    monkeypatch.setattr(runner, "write_json", real_write_json)

    # an I/O error while hashing a source replica: before any durable write, so nothing to roll back
    def failing_hash(path, limit=None):
        if path.startswith(str(h.rollouts)):
            raise OSError(5, "Input/output error")
        return real_sha256_file(path, limit)
    monkeypatch.setattr(runner, "sha256_file", failing_hash)
    code, runs = _restore_in_process(runner, h, tmp_path / "local-3", monkeypatch, "--restore", selector)
    assert code == 70 and len(runs) == 1
    _assert_rolled_back(runs[0], h, f"restore of {selector} failed while checking the replicas: OSError: [Errno 5] Input/output error")
    assert "restore rolled back" not in (runs[0] / "runner.log").read_text(), "nothing had been created yet"
    monkeypatch.setattr(runner, "sha256_file", real_sha256_file)

    # the last rename (staging/db into db/) fails: the replicas renamed and the sessions tree installed are taken back
    def failing_rename(src, dst, *args, **kwargs):
        if str(src).endswith("restore-staging/db"):
            raise OSError(5, "Input/output error")
        return real_rename(src, dst, *args, **kwargs)
    monkeypatch.setattr(runner.os, "rename", failing_rename)
    try:
        code, runs = _restore_in_process(runner, h, tmp_path / "local-4", monkeypatch, "--restore", selector)
    finally:
        monkeypatch.setattr(runner.os, "rename", real_rename)
    assert code == 70 and len(runs) == 1
    _assert_rolled_back(runs[0], h, f"restore of {selector} failed while activating: OSError: [Errno 5] Input/output error")
    log = (runs[0] / "runner.log").read_text()
    assert str(runs[0] / "home" / "sessions") in log and "could not remove" not in log
    assert _tree(h.durable_run) == before
    assert h.replica.read_bytes() == h.local_rollout.read_bytes()

    # the readback of a staged replica copy fails (third pass, finding 8): the copies exist, the hash of the local one raises
    def failing_readback(path, limit=None):
        if "/restore-staging/sessions/" in path:
            raise OSError(5, "Input/output error")
        return real_sha256_file(path, limit)
    monkeypatch.setattr(runner, "sha256_file", failing_readback)
    code, runs = _restore_in_process(runner, h, tmp_path / "local-5", monkeypatch, "--restore", selector)
    assert code == 70 and len(runs) == 1
    _assert_rolled_back(runs[0], h, f"restore of {selector} failed while staging the replicas: OSError: [Errno 5] Input/output error")
    log = (runs[0] / "runner.log").read_text()
    assert "restore rolled back: removed" in log and str(h.durable_base / "codex" / h.scope / "runs" / runs[0].name) in log
    assert _tree(h.durable_run) == before

    # the durable copy reads back with another hash than the bytes that were written
    def differing_readback(path, limit=None):
        digest, size = real_sha256_file(path, limit)
        return ("00" * 32 if os.path.basename(path).startswith(".incoming-") else digest), size
    monkeypatch.setattr(runner, "sha256_file", differing_readback)
    code, runs = _restore_in_process(runner, h, tmp_path / "local-6", monkeypatch, "--restore", selector)
    assert code == 70 and len(runs) == 1
    incoming = h.durable_base / "codex" / h.scope / "runs" / runs[0].name / "rollouts" / f".incoming-{TID}.jsonl"
    _assert_rolled_back(runs[0], h, f"generation {selector}: the staged copy {incoming} does not hash to its source")
    assert _tree(h.durable_run) == before
    monkeypatch.setattr(runner, "sha256_file", real_sha256_file)


@needs_bwrap
def test_unknown_ownership_refuses_reuse(home):
    """Finding 3: a latest run with db/ whose owner.json carries no usable
    pids is of unknown ownership: refused, never treated as dead."""
    scope_dir = home["local"] / "state" / "codex" / home["scope"]
    run_id = "e" * 32
    run_dir = scope_dir / run_id
    (run_dir / "db").mkdir(parents=True)
    (scope_dir / "latest-run").write_text(run_id + "\n")
    for owner, needle in (({}, "runner_pid (None)"), ({"runner_pid": 1, "child_pid": "x"}, "child_pid ('x')"),
                          ({"runner_pid": True, "child_pid": 1}, "runner_pid (True)"), ({"runner_pid": 1, "child_pid": 0}, "child_pid (0)")):
        (run_dir / "owner.json").write_text(json.dumps(owner))
        proc = _attempt(home)
        assert proc.returncode == 70 and "FAKE_READY" not in proc.stdout, (proc.returncode, proc.stderr)
        assert f"this allocation's latest run {run_id} has local databases but its owner.json records no usable {needle}" in proc.stderr
        assert "ownership is unknown, so nothing is reused, deleted or started" in proc.stderr
        assert sorted(p.name for p in scope_dir.iterdir() if p.is_dir()) == [run_id] and (run_dir / "db").is_dir()
        assert (scope_dir / "latest-run").read_text().strip() == run_id


@needs_bwrap
def test_reuse_refuses_another_home_or_durable_root(home, tmp_path):
    """Finding 2: the same allocation relaunched under another CODEX_HOME or
    another --durable-root does not continue the run: its stored rollout
    paths are absolute under the recorded home, and its replicas and
    generations live under the recorded durable run."""
    scope_dir = home["local"] / "state" / "codex" / home["scope"]
    a = Managed(home)
    try:
        a.cmd("turn 200")
        a.wait_synced()
        assert a.quit() == 0
    finally:
        a.close()
    owner_before = (a.run_dir / "owner.json").read_text()
    home2 = tmp_path / "home2"
    for d in ("sessions", "thread-writer-locks", "skills"):
        (home2 / d).mkdir(parents=True)
    (home2 / "config.toml").write_text('model = "x"\n')
    proc = _attempt(dict(home, codex_home=home2))
    assert proc.returncode == 70 and "FAKE_READY" not in proc.stdout, (proc.returncode, proc.stderr)
    assert (f"run {a.run_dir.name} cannot be reused under CODEX_HOME {home2} ({os.path.realpath(home2)}): it was created under "
            f"{home['codex_home']} ({os.path.realpath(home['codex_home'])}), and its stored rollout paths are absolute under that home") in proc.stderr
    assert not list((home2 / "sessions").iterdir())
    durable2 = tmp_path / "durable2"
    durable2.mkdir()
    proc = _attempt(dict(home, durable=durable2))
    assert proc.returncode == 70 and "FAKE_READY" not in proc.stdout, (proc.returncode, proc.stderr)
    assert (f"run {a.run_dir.name} cannot be reused under --durable-root {durable2}: its replicas and generations are recorded under "
            f"{a.durable_run}, not {durable2 / 'codex' / home['scope'] / 'runs' / a.run_dir.name}") in proc.stderr
    assert not (durable2 / "codex").exists(), "nothing was created under the other durable root"
    assert (scope_dir / "latest-run").read_text().strip() == a.run_dir.name
    assert (a.run_dir / "owner.json").read_text() == owner_before and (a.run_dir / "db" / "state_5.sqlite").exists()
    assert sorted(p.name for p in scope_dir.iterdir() if p.is_dir()) == [a.run_dir.name], "no run directory was created"
    status = json.loads((a.run_dir / "status.json").read_text())
    assert status["state"] == "refused" and "cannot be reused under --durable-root" in status["reason"]


@needs_bwrap
def test_restore_reports_blocked_when_the_durable_side_does_not_answer(runner, tmp_path, monkeypatch):
    """Finding 5: status says restoring (selector and source) before the first
    durable read; a manifest read that never returns leaves the launch
    blocked within the restore deadline, exit 70, the step named and the
    operator told to retry or pass --fresh; the thread is abandoned."""
    assert runner.restore_timeout(30.0) == 120.0 and runner.restore_timeout(60.0) == 240.0
    assert runner.restore_timeout(1e30) == 86400.0, "the computed default is clamped to a day as well"
    for raw in ("inf", "abc", "0", "-5", "nan", "1e20", "86401"):
        monkeypatch.setenv("SUCODER_CODEX_STATE_RESTORE_TIMEOUT", raw)
        assert runner.restore_timeout(30.0) == 120.0, raw
    for raw, expected in (("86400", 86400.0), ("0.5", 0.5)):
        monkeypatch.setenv("SUCODER_CODEX_STATE_RESTORE_TIMEOUT", raw)
        assert runner.restore_timeout(30.0) == expected, raw
    h, selector = _published(runner, tmp_path)
    release, entered = threading.Event(), threading.Event()

    def blocking(gen_dir, name):
        entered.set()
        release.wait(30)
        raise runner.Refused("released")       # the abandoned thread ends without restoring anything
    monkeypatch.setattr(runner, "load_manifest", blocking)
    monkeypatch.setenv("SUCODER_CODEX_STATE_RESTORE_TIMEOUT", "1")
    started = time.monotonic()
    code, runs = _restore_in_process(runner, h, tmp_path / "local-b", monkeypatch, "--restore", selector)
    elapsed = time.monotonic() - started
    assert code == 70 and 1.0 <= elapsed < 5.0, elapsed
    assert entered.is_set() and len(runs) == 1
    status = json.loads((runs[0] / "status.json").read_text())
    assert [t["state"] for t in status["transitions"]] == ["restoring", "blocked"]
    assert status["state"] == "blocked" and status["restore"] == selector
    assert status["source"] == str(h.generations / selector.split("/")[1])
    assert "the durable side has not answered in 1s while reading the manifest" in status["reason"]
    assert "Retry when the durable filesystem answers, or pass --fresh" in status["reason"]
    log = (runs[0] / "runner.log").read_text()
    assert "restore blocked: the durable side has not answered" in log and "refusing to launch" not in log
    assert not (runs[0] / "db").exists() and not (runs[0] / "restore-staging").exists() and not (runs[0] / "home" / "sessions").exists()
    assert os.listdir(h.durable_base / "codex" / h.scope / "runs") == [h.run_id]
    workers = [t for t in threading.enumerate() if t.name == "codex-state-restore"]
    assert len(workers) == 1 and workers[0].daemon and workers[0].is_alive(), "abandoned, still blocked"
    release.set()
    workers[0].join(10)
    assert not workers[0].is_alive()


@needs_bwrap
def test_restore_validation_gaps_refuse(runner, tmp_path, monkeypatch):
    """Finding 6: a null threads entry is normalized once (check_replicas reads
    the same dict), a zero-length prefix is still hashed, a selector or scope
    with a trailing newline is not accepted, and a store present in the
    generation's db/ but unlisted refuses."""
    h, selector = _published(runner, tmp_path)
    generation = selector.split("/")[1]
    gen_dir = h.generations / generation
    manifest_path = gen_dir / "manifest.json"
    pristine = manifest_path.read_bytes()
    launch = (selector, h.run_id, generation, h.scope, str(h.codex_home), str(h.codex_home), False)

    nulled = dict(json.loads(pristine), threads=None)
    runner.check_manifest(nulled, *launch)
    assert nulled["threads"] == {}, "normalized in the manifest itself"
    assert runner.check_replicas(nulled, selector, str(h.durable_run)) == {}

    zero = json.loads(pristine)
    zero["threads"][TID].update({"replica_bytes": 0, "next_rollout_byte_offset": 0, "rollout_sha256_prefix": hashlib.sha256(b"").hexdigest()})
    runner.check_manifest(zero, *launch)
    assert TID in runner.check_replicas(zero, selector, str(h.durable_run)), "the empty prefix hashes to the hash of nothing"
    zero["threads"][TID]["rollout_sha256_prefix"] = "00" * 32
    manifest_path.write_bytes(json.dumps(zero).encode())
    code, runs = _restore_in_process(runner, h, tmp_path / "local-1", monkeypatch, "--restore", selector)
    assert code == 70 and len(runs) == 1
    _assert_rolled_back(runs[0], h, f"thread {TID}: the first 0 bytes of its replica", "do not hash to the manifest's rollout_sha256_prefix")
    manifest_path.write_bytes(pristine)

    code, runs = _restore_in_process(runner, h, tmp_path / "local-2", monkeypatch, "--restore", selector + "\n")
    assert code == 70 and runs == [] and not (tmp_path / "local-2" / "state").exists()
    assert "is not RUN/GENERATION" in _log(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        _restore_in_process(runner, h, tmp_path / "local-2", monkeypatch, "--scope", h.scope + "\n")
    assert exit_info.value.code == 2

    (gen_dir / "db" / "memories_1.sqlite").write_bytes(b"")
    code, runs = _restore_in_process(runner, h, tmp_path / "local-3", monkeypatch, "--restore", selector)
    assert code == 70 and len(runs) == 1
    _assert_rolled_back(runs[0], h, f"{gen_dir / 'db'} holds unlisted store(s) memories_1.sqlite", "altered after publication")


def test_load_manifest_hashes_the_bytes_it_parsed(runner, tmp_path, monkeypatch):
    """Finding 7: one read; the recorded sha256 is of the parsed bytes, with
    no second open of the file (which could see different bytes)."""
    gen_dir = tmp_path / "gen"
    gen_dir.mkdir()
    raw = b'{"format": 1, "threads": null}\n'
    (gen_dir / "manifest.json").write_bytes(raw)

    def no_second_read(path, limit=None):
        raise AssertionError("manifest.json must not be read a second time: %s" % path)
    monkeypatch.setattr(runner, "sha256_file", no_second_read)
    manifest, digest = runner.load_manifest(str(gen_dir), "r/g")
    assert manifest == {"format": 1, "threads": None} and digest == hashlib.sha256(raw).hexdigest()
    (gen_dir / "manifest.json").write_bytes(b"\xff\xfe not json")
    with pytest.raises(runner.Refused, match="manifest.json is unreadable"):
        runner.load_manifest(str(gen_dir), "r/g")


@needs_bwrap
def test_an_unattended_refusal_records_the_pending_work_it_found(runner, tmp_path, monkeypatch):
    """Finding 8: the pending work is written to status.json before the gate,
    so a refused unattended restore shows what it found, not only the reason."""
    h, selector = _published(runner, tmp_path, active_goal=True)
    code, runs = _restore_in_process(runner, h, tmp_path / "local-1", monkeypatch, "--restore", selector, "--unattended")
    assert code == 70 and len(runs) == 1
    _assert_rolled_back(runs[0], h, f"unattended continuation refused: generation {selector} holds an active goal on thread(s) {TID}")
    status = json.loads((runs[0] / "status.json").read_text())
    assert status["pending_work"] == {"active_goals": [TID], "queued": {}}
    assert [t["state"] for t in status["transitions"]] == ["restoring", "refused"]
    assert "restore rolled back: removed" in (runs[0] / "runner.log").read_text(), "the staged stores were removed"


@needs_bwrap
def test_reuse_keeps_a_restored_runs_provenance_and_local_edits(home, tmp_path):
    """Finding 9: a run that was itself restored is reused with its run.json
    (parent included) byte-identical to the pre-restart copies, its
    restored_from carried, and a local edit of the bound AGENTS copy kept."""
    a = Managed(home)
    try:
        a.cmd("turn 300")
        a.wait_synced()
        assert a.quit() == 0
    finally:
        a.close()
    gen = a.status()["last_generation"]
    selector = f"{a.run_dir.name}/{gen['name']}"
    other = _allocation(home, tmp_path, "local-b")
    b = Managed(other, extra=("--restore", selector))
    try:
        assert b.tid == a.tid
        assert b.quit() == 0
    finally:
        b.close()
    parent = {"run": a.run_dir.name, "generation": gen["name"], "class": "sealed", "manifest_sha256": gen["manifest_sha256"]}
    before = (b.run_dir / "run.json").read_text()
    assert json.loads(before)["parent"] == parent and before == (b.durable_run / "run.json").read_text()
    agents = b.run_dir / "home" / "AGENTS.md"
    agents.write_text("GLOBAL\nedited after the restore\n")
    restored_before = b.status()["restored_from"]
    size_b = b.local_rollout.stat().st_size
    c = Managed(other)
    try:
        assert c.run_dir == b.run_dir and c.tid == a.tid
        status = c.status()
        assert status["mode"] == "reuse" and status["restored_from"] == restored_before
        assert (c.run_dir / "run.json").read_text() == before == (c.durable_run / "run.json").read_text()
        assert agents.read_text() == "GLOBAL\nedited after the restore\n"
        assert c.sees["projection"] == {a.tid: [size_b, 1]}
        c.cmd("turn 100")
        c.wait_synced()
        assert _manifest(_request(other))["parent"] == parent
        assert c.quit() == 0
    finally:
        c.close()


# ----------------------------------------------------------------- addendum: confirmation pass over the first fix pass

def test_a_replica_counts_only_after_its_name_is_durable(runner, tmp_path, monkeypatch):
    """Addendum A: replicated_bytes advances only once the first sync of
    rollouts/ has succeeded; while it fails the bytes are in the file but not
    counted, a capture in between is not durable, and the retry on the next
    pass counts them without writing anything new."""
    h = InProcess(runner, tmp_path, timeout=1.0)
    real = runner.fsync_path
    failures = []

    def failing_once(path):
        if path == str(h.rollouts) and not failures:
            failures.append(path)
            raise OSError(5, "Input/output error")
        real(path)
    monkeypatch.setattr(runner, "fsync_path", failing_once)
    h.replicator.pass_once()
    entry = h.replicator.threads[TID]
    assert h.replica.read_bytes() == h.local_rollout.read_bytes(), "the bytes are in the file"
    assert entry["replicated_bytes"] == 0 and "Input/output error" in entry["error"] and TID not in h.replicator.dir_synced
    assert h.replicator.view()["lag_bytes"] == h.local_rollout.stat().st_size
    outcome, reason = h.capture()
    assert outcome == "not_durable" and f"{TID} has 0 replicated bytes" in reason
    mtime = h.replica.stat().st_mtime_ns
    h.replicator.pass_once()
    assert entry["replicated_bytes"] == h.replica.stat().st_size and TID in h.replicator.dir_synced and "error" not in entry
    assert h.replica.stat().st_mtime_ns == mtime and failures == [str(h.rollouts)], "nothing was written on the retry"
    assert h.capture() == ("published", None)


@needs_bwrap
def test_a_rollout_replaced_by_another_file_is_an_anomaly(home):
    """Addendum B: the same bytes at the same path from another inode: the
    anomaly "replaced", nothing appended after it, later captures of the
    thread fail."""
    m = Managed(home)
    try:
        m.cmd("turn 300")
        m.wait_synced()
        first = _wait_idle(home)["last_generation"]
        replicated = m.replica.stat().st_size
        m.cmd("replace")
        m.cmd("turn 200")
        status = _wait(home, lambda st: st["replica"]["anomalies"].get(m.tid) == "replaced", "the anomaly")[0]
        assert m.local_rollout.stat().st_size > replicated
        assert m.replica.stat().st_size == replicated and m.replica.read_bytes() == m.local_rollout.read_bytes()[:replicated]
        assert status["replica"]["threads"][m.tid]["replicated_bytes"] == replicated
        status = _request(home)
        assert status["checkpoint"]["last_outcome"] == "failed" and f"thread {m.tid}: its rollout replaced" in status["checkpoint"]["reason"]
        assert status["last_generation"] == first and m.replica.stat().st_size == replicated
        log = (m.run_dir / "runner.log").read_text()
        assert f"rollout of {m.tid} was replaced by another file (device:inode " in log and "replication of this thread stops" in log
    finally:
        m.close()


@needs_bwrap
def test_a_durable_root_below_absent_directories_is_created(home, tmp_path):
    """Addendum D: a --durable-root two levels below a directory that does not
    exist yet launches; the ancestors are created, the base and everything
    below it 0700."""
    durable = tmp_path / "absent" / "deeper" / "state"
    assert not (tmp_path / "absent").exists()
    m = Managed(dict(home, durable=durable))
    try:
        assert m.durable_run.is_dir() and (m.durable_run / "rollouts").is_dir()
        assert all(oct(p.stat().st_mode & 0o777) == "0o700" for p in (durable, durable / "codex", m.durable_run))
        m.cmd("turn 100")
        m.wait_synced()
        assert m.quit() == 0
    finally:
        m.close()
    assert m.status()["last_generation"]["class"] == "sealed"


@needs_bwrap
def test_an_unusable_replication_interval_falls_back_to_one_second(runner, home, monkeypatch):
    """Addendum E: inf, NaN, zero, negative, huge or non-numeric overrides are
    the 1.0 default (Event.wait overflows on inf and the loop would die); a
    sane value is taken; under inf replication still runs."""
    for raw, expected in (("inf", 1.0), ("-inf", 1.0), ("nan", 1.0), ("0", 1.0), ("-1", 1.0), ("3601", 1.0), ("1e9", 1.0),
                          ("abc", 1.0), ("", 1.0), ("0.5", 0.5), ("3600", 3600.0)):
        monkeypatch.setenv("SUCODER_CODEX_STATE_REPLICATION_INTERVAL", raw)
        assert runner.replication_interval() == expected, raw
    m = Managed(home, env={"SUCODER_CODEX_STATE_REPLICATION_INTERVAL": "inf"})
    try:
        m.cmd("turn 300")
        m.wait_synced(timeout=10)
        assert "replication pass failed" not in (m.run_dir / "runner.log").read_text()
        assert m.quit() == 0
    finally:
        m.close()


# ----------------------------------------------------------------- third review-fix pass

@needs_bwrap
def test_a_failure_while_preparing_the_binds_takes_the_launch_back(runner, tmp_path, monkeypatch):
    """Third pass, finding 1: prepare_binds runs after a restore has
    activated; an OSError there is exit 70 with the step named, and the
    transaction extends through it: the installed db/, the sessions tree,
    both run.json copies and the durable run directory are taken back.  A
    fresh launch takes back its empty db/ and its durable run directory as
    well, so the next launch starts clean; a reused run has nothing to take
    back and is refused with its stores, ownership and latest-run untouched."""
    h, selector = _published(runner, tmp_path)
    before = _tree(h.durable_run)
    activated = []
    real_activate = runner.activate_restore

    def recording_activate(name, attempt, *args):
        real_activate(name, attempt, *args)
        activated.append(attempt.db_installed)
    monkeypatch.setattr(runner, "activate_restore", recording_activate)

    def failing_binds(codex_home_real, home_local, keep_existing=False):
        raise OSError(13, "Permission denied", os.path.join(codex_home_real, "skills"))
    monkeypatch.setattr(runner, "prepare_binds", failing_binds)

    code, runs = _restore_in_process(runner, h, tmp_path / "local-1", monkeypatch, "--restore", selector)
    assert code == 70 and len(runs) == 1 and activated == [True], "the restore had activated (db/ installed) before the binds failed"
    _assert_rolled_back(runs[0], h, f"restore of {selector} failed while preparing the binds: PermissionError: [Errno 13] Permission denied: "
                                    f"'{h.codex_home / 'skills'}'")
    status = json.loads((runs[0] / "status.json").read_text())
    assert [t["state"] for t in status["transitions"]] == ["restoring", "refused"]
    log = (runs[0] / "runner.log").read_text()
    assert "restore rolled back: removed" in log and str(runs[0] / "db") in log and str(runs[0] / "home" / "sessions") in log
    assert str(h.durable_base / "codex" / h.scope / "runs" / runs[0].name) in log and "could not remove" not in log
    assert not (runs[0] / "owner.json").exists() and not (runs[0].parent / "latest-run").exists()
    assert _tree(h.durable_run) == before

    code, runs = _restore_in_process(runner, h, tmp_path / "local-2", monkeypatch)
    assert code == 70 and len(runs) == 1
    _assert_rolled_back(runs[0], h, f"the new run {runs[0].name[:8]} failed while preparing the binds: PermissionError: [Errno 13] Permission denied")
    log = (runs[0] / "runner.log").read_text()
    assert "fresh launch rolled back: removed" in log and str(runs[0] / "db") in log and "could not remove" not in log
    assert str(h.durable_base / "codex" / h.scope / "runs" / runs[0].name) in log, "the durable run directory this launch created"
    assert not (runs[0] / "owner.json").exists() and not (runs[0].parent / "latest-run").exists()
    assert runner.reusable_run(str(runs[0].parent)) == (None, None), "the next launch starts clean"

    # the harness's own run, made reusable: db/, dead pids, provenance under this home and durable run
    gone = subprocess.Popen(["true"])
    gone.wait()
    owner = {"runner_pid": gone.pid, "child_pid": gone.pid, "durable": str(h.durable_run)}
    (h.run_dir / "owner.json").write_text(json.dumps(owner))
    (h.run_dir / "run.json").write_text(json.dumps({"format": 1, "scope": h.scope, "run": h.run_id, "parent": None, "allocation": {},
                                                    "codex_home": str(h.codex_home), "codex_home_real": str(h.codex_home)}))
    (h.run_dir.parent / "latest-run").write_text(h.run_id + "\n")
    status_before = (h.run_dir / "status.json").read_text()
    code, runs = _restore_in_process(runner, h, tmp_path / "local", monkeypatch)
    assert code == 70 and runs == [h.run_dir]
    status = json.loads((h.run_dir / "status.json").read_text())
    assert status["state"] == "refused"
    assert f"reuse of run {h.run_id[:8]} failed while preparing the binds: PermissionError: [Errno 13] Permission denied" in status["reason"]
    assert status["transitions"][:-1] == json.loads(status_before)["transitions"], "the reused run's transitions accumulate"
    assert (h.run_dir / "db" / "state_5.sqlite").exists() and (h.run_dir / "run.json").exists() and h.local_rollout.exists()
    assert json.loads((h.run_dir / "owner.json").read_text()) == owner and (h.run_dir.parent / "latest-run").read_text().strip() == h.run_id
    assert "rolled back" not in (h.run_dir / "runner.log").read_text(), "nothing to take back from a reused run"
    assert _tree(h.durable_run) == before


def test_missing_ancestors_above_the_durable_base_are_created_one_by_one_and_synced(runner, tmp_path, monkeypatch):
    """Third pass, finding 2: a durable root below directories that do not
    exist yet: each missing ancestor is created on its own and its parent's
    entry synced before the next one, so a generation under a brand-new root
    depends on no name that was never synced.  A sync that fails above the
    base is logged and not fatal; an existing ancestor is neither synced nor
    judged (a world-writable one above the base passes)."""
    synced = []
    real = runner.fsync_path

    def recording(path):
        synced.append(path)
        real(path)
    monkeypatch.setattr(runner, "fsync_path", recording)
    base = tmp_path / "a" / "b" / "c" / "durable"
    run = base / "codex" / "ab" / "runs" / "11"
    runner.ensure_durable_run(str(run), str(base))
    created = [tmp_path / "a", tmp_path / "a" / "b", tmp_path / "a" / "b" / "c", base, base / "codex", base / "codex" / "ab",
               base / "codex" / "ab" / "runs", run, run / "rollouts", run / "generations"]
    assert synced == [str(d.parent) for d in created], "every created directory, the ancestors included, is followed by a sync of its parent"
    assert all(oct(d.stat().st_mode & 0o777) == "0o700" for d in created)
    synced.clear()
    runner.ensure_durable_run(str(run), str(base))
    assert synced == [], "existing directories, above the base too, are not re-synced"

    unreadable = {str(tmp_path / "x")}

    def refusing(path):
        synced.append(path)
        if path in unreadable:
            raise PermissionError(13, "Permission denied")
        real(path)
    monkeypatch.setattr(runner, "fsync_path", refusing)
    open_dir = tmp_path / "x"
    open_dir.mkdir()
    open_dir.chmod(0o777)
    base2 = open_dir / "y" / "z" / "durable"
    runner.ensure_durable_run(str(base2 / "codex" / "ab" / "runs" / "11"), str(base2))
    assert synced[:3] == [str(open_dir), str(open_dir / "y"), str(open_dir / "y" / "z")]
    assert f"created {open_dir / 'y'} but could not sync its parent directory (PermissionError" in _log(tmp_path)
    assert (base2 / "codex" / "ab" / "runs" / "11" / "generations").is_dir()
    assert oct(open_dir.stat().st_mode & 0o777) == "0o777", "the operator's directory above the base is not touched"


def test_the_exit_path_never_syncs_the_durable_side_itself(runner, tmp_path):
    """Third pass, finding 3: an unacknowledged generation is retried only by
    the final capture's worker, under the shutdown deadline.  final() itself
    never calls fsync on generations/: with that sync blocked, the exit still
    returns within the deadline and the generation stays unacknowledged.  When
    no final capture runs at all, the skip line says the retry did not happen."""
    h = InProcess(runner, tmp_path, timeout=1.0)
    h.replicator.pass_once()
    real = runner.fsync_path
    mode = {"on": "fail"}
    release = threading.Event()
    callers = []

    def controlled(path):
        if path == str(h.generations):
            callers.append(threading.current_thread().name)
            if mode["on"] == "fail":
                raise OSError(5, "Input/output error")
            if mode["on"] == "block":
                release.wait(30)
        real(path)
    h.runner.fsync_path = controlled
    assert h.capture() == ("published", None)
    pending = h.cp.pending_sync
    assert pending is not None
    callers.clear()
    mode["on"] = "block"
    started = time.monotonic()
    try:
        h.cp.final("sealed")
        elapsed = time.monotonic() - started
        assert 1.0 <= elapsed < 5.0, elapsed       # the deadline plus the 2 s join grace; the point is that it returns
        assert callers and "MainThread" not in callers, "the retry happened on the capture's worker, not on the exit path"
        assert h.cp.pending_sync is pending and h.cp.last_generation is None
        assert "final sealed capture still running after its" in _log(tmp_path)
        worker = h.cp.inflight["thread"]
    finally:
        mode["on"] = "ok"
        release.set()
    worker.join(30)
    assert not worker.is_alive()

    h2 = InProcess(runner, tmp_path / "two", timeout=1.0)
    h2.cp.pending_sync = dict(pending)
    h2.cp.seq = None
    h2.cp.final("sealed")
    assert ("final sealed capture skipped: the durable generations listing never completed, so nothing could be published; last durable "
            "generation: none; generation %s stays unacknowledged (its directory sync is retried only inside a capture)" % pending["name"]) in _log(tmp_path)
    assert h2.cp.pending_sync == pending
    h2.cp.inflight = {"thread": None, "deadline": time.monotonic(), "generation": "000007-deadbeef", "trigger": "request",
                      "class": "warm", "stuck": False, "started": time.time()}
    h2.cp.final("sealed")
    assert ("capture 000007-deadbeef is still in flight; last durable generation: none; generation %s stays unacknowledged"
            % pending["name"]) in _log(tmp_path)


@needs_bwrap
def test_a_blocked_restore_is_fenced_and_rolls_back_when_its_thread_wakes_up(runner, tmp_path, monkeypatch):
    """Third pass, finding 4: the durable side answers after the launch has
    reported the restore blocked.  The abandoned thread is fenced: its next
    durable write refuses, it rolls back what it staged, and nothing is
    activated (no db/, no home/sessions, no run.json, no durable run
    directory); the status stays blocked."""
    h, selector = _published(runner, tmp_path)
    before = _tree(h.durable_run)
    release, entered = threading.Event(), threading.Event()
    real_load = runner.load_manifest

    def late(gen_dir, name):
        entered.set()
        release.wait(30)
        return real_load(gen_dir, name)       # the durable side answers late, and the restore would go on
    monkeypatch.setattr(runner, "load_manifest", late)
    monkeypatch.setenv("SUCODER_CODEX_STATE_RESTORE_TIMEOUT", "1")
    try:
        code, runs = _restore_in_process(runner, h, tmp_path / "local-f", monkeypatch, "--restore", selector)
        assert code == 70 and entered.is_set() and len(runs) == 1
        workers = [t for t in threading.enumerate() if t.name == "codex-state-restore"]
        assert len(workers) == 1 and workers[0].is_alive()
    finally:
        release.set()
    workers[0].join(30)
    assert not workers[0].is_alive()
    status = json.loads((runs[0] / "status.json").read_text())
    assert status["state"] == "blocked" and [t["state"] for t in status["transitions"]] == ["restoring", "blocked"]
    assert status["pending_work"] == {"active_goals": [], "queued": {}}, "the late thread got as far as the gate"
    assert not (runs[0] / "db").exists() and not (runs[0] / "restore-staging").exists() and not (runs[0] / "run.json").exists()
    assert not (runs[0] / "home" / "sessions").exists() and not (runs[0] / "owner.json").exists()
    assert os.listdir(h.durable_base / "codex" / h.scope / "runs") == [h.run_id]
    log = (runs[0] / "runner.log").read_text()
    assert "restore rolled back: removed" in log and str(runs[0] / "restore-staging") in log
    assert "the abandoned restore thread ended: fenced: the launch already reported this restore blocked" in log
    assert "placed" not in log and _tree(h.durable_run) == before


@needs_bwrap
def test_a_huge_restore_timeout_is_clamped_and_the_restore_proceeds(runner, tmp_path, monkeypatch):
    """Third pass, finding 5: SUCODER_CODEX_STATE_RESTORE_TIMEOUT=1e20 no
    longer reaches Thread.join (which overflows); the restore runs to its
    normal outcome, here the unattended refusal of an active goal."""
    h, selector = _published(runner, tmp_path, active_goal=True)
    monkeypatch.setenv("SUCODER_CODEX_STATE_RESTORE_TIMEOUT", "1e20")
    assert runner.restore_timeout(30.0) == 120.0
    code, runs = _restore_in_process(runner, h, tmp_path / "local-1", monkeypatch, "--restore", selector, "--unattended")
    assert code == 70 and len(runs) == 1
    _assert_rolled_back(runs[0], h, f"unattended continuation refused: generation {selector} holds an active goal on thread(s) {TID}")


@needs_bwrap
def test_a_reused_run_takes_over_its_unsynced_generation(runner, tmp_path, home):
    """Third pass, finding 6: a run whose status.json records an unsynced
    generation is reused with that record as the pending sync, so the first
    capture's retry acknowledges it (in process: adopt, then retry); the
    wiring in run() is exercised by a relaunch under the subprocess harness.
    A record not shaped like this run's published record (its keys, a
    generation name, this run's generations/<name> as the path) is ignored;
    whether the generation is still there is the retry's business (see
    test_adopt_unsynced_judges_the_record_alone_and_the_retry_checks_the_generation)."""
    h = InProcess(runner, tmp_path)
    h.replicator.pass_once()
    assert h.capture() == ("published", None)
    published = h.cp.last_generation
    record = dict(published, sync_warning="OSError: [Errno 5] Input/output error")
    for unusable in (None, "x", {"name": published["name"]}, {"name": published["name"], "path": "/elsewhere/" + published["name"]},
                     {"name": "000009-00000000", "path": str(h.generations / "000009-00000000")}):
        assert h.cp.adopt_unsynced(unusable) is False and h.cp.pending_sync is None
    assert "status.json records an unusable unsynced_generation (None); it is not taken over" in _log(tmp_path)
    h.cp.last_generation = None
    assert h.cp.adopt_unsynced(record) is True and h.cp.pending_sync is record
    assert f"generation {published['name']} of the reused run is not acknowledged yet" in _log(tmp_path)
    h.cp.retry_sync()
    last = h.cp.last_generation
    assert last["name"] == published["name"] and "sync_warning" not in last and last["synced_at"] > 0 and h.cp.pending_sync is None
    assert h.status_json()["last_generation"] == last and h.status_json()["unsynced_generation"] is None

    a = Managed(home)
    try:
        a.cmd("turn 300")
        a.wait_synced()
        assert a.quit() == 0
    finally:
        a.close()
    status_path = a.run_dir / "status.json"
    status = json.loads(status_path.read_text())
    sealed = status.pop("last_generation")
    status["unsynced_generation"] = dict(sealed, sync_warning="OSError: [Errno 5] Input/output error")
    status_path.write_text(json.dumps(status))
    b = Managed(home)
    try:
        assert b.run_dir == a.run_dir
        # the loaded view still shows the earlier launch's captures, so wait for a generation newer than the seeded one
        idle = _wait(home, lambda st: st.get("last_generation", {}).get("name", sealed["name"]) != sealed["name"]
                     and st["checkpoint"]["state"] == "idle", "the initial capture")[0]
        log = (b.run_dir / "runner.log").read_text()
        assert f"generation {sealed['name']} of the reused run is not acknowledged yet (the sync of {b.generations} failed in the earlier launch)" in log
        assert f"generation {sealed['name']} acknowledged: {b.generations} synced on retry" in log
        assert idle["unsynced_generation"] is None and idle["durable_age_seconds"] is not None
        assert idle["last_generation"]["name"] != sealed["name"], "the initial capture published after the retry"
        assert int(idle["last_generation"]["name"][:6]) > int(sealed["name"][:6])
        assert b.quit() == 0
    finally:
        b.close()


def test_manifest_counts_must_be_integers(runner, tmp_path):
    """Third pass, finding 7: a byte offset, an ordinal, a replica length or
    a store size that is a float, a numeric string, a bool, negative or null
    is malformed; int() would have coerced 0.5 to 0, "7" to 7 and True to 1."""
    h, selector = _published(runner, tmp_path)
    generation = selector.split("/")[1]
    pristine = json.loads((h.generations / generation / "manifest.json").read_text())
    launch = (selector, h.run_id, generation, h.scope, str(h.codex_home), str(h.codex_home), False)
    runner.check_manifest(json.loads(json.dumps(pristine)), *launch)
    for key, value in (("next_rollout_byte_offset", 0.5), ("next_rollout_byte_offset", "7"), ("next_rollout_ordinal", True),
                       ("next_rollout_ordinal", 1.0), ("replica_bytes", -1), ("replica_bytes", None)):
        bad = json.loads(json.dumps(pristine))
        bad["threads"][TID][key] = value
        with pytest.raises(runner.Refused) as refused:
            runner.check_manifest(bad, *launch)
        assert f"manifest.json is malformed (TypeError: threads[{TID}].{key} is {value!r}, not a non-negative integer)" in str(refused.value)
    for value in ("7", 1.0, True, -1):
        bad = json.loads(json.dumps(pristine))
        bad["databases"]["state_5.sqlite"]["bytes"] = value
        with pytest.raises(runner.Refused, match="malformed \\(TypeError: databases\\[state_5.sqlite\\].bytes is"):
            runner.check_manifest(bad, *launch)
    assert runner.require_int(0, "x") == 0 and runner.require_int(7, "x") == 7


def _bounded_restore(runner, h, selector, local_root, run_id):
    """``restore_bounded``'s arguments for a new run under ``local_root``
    against the harness's durable root, as run() builds them: (local run
    directory, durable run directory, attempt, args)."""
    local = local_root / "state" / "codex" / h.scope / run_id
    home_local = local / "home"
    for directory in (local, home_local, local / "snapshots"):
        directory.mkdir(parents=True)
    durable = h.durable_base / "codex" / h.scope
    durable_run = durable / "runs" / run_id
    created = time.time()
    meta = {"format": 1, "scope": h.scope, "run": run_id, "parent": None, "allocation": {"job": None, "node": "test"},
            "codex_home": str(h.codex_home), "codex_home_real": str(h.codex_home), "created": created, "created_iso": runner.utc_iso(created)}
    status = runner.Status(str(local / "status.json"))
    attempt = runner.RestoreAttempt(str(local), str(home_local), str(durable_run))
    args = (runner.parse_selector(selector), str(durable), str(h.durable_base), h.scope, meta, False,
            str(local), str(home_local), str(durable_run), status, attempt)
    return local, durable_run, attempt, args


def test_the_fence_and_the_activation_exclude_each_other(runner, tmp_path, monkeypatch):
    """Residual 1: the restore deadline passes while the thread is activating.
    Fence first (the thread parked just before the activation's lock, at the
    rollouts sync): Blocked, and the thread, waking up, refuses at the lock
    and rolls back everything, db/ included.  Activation first (the thread
    parked inside the lock, mid-rename, until restore_bounded waits for the
    lock): the renames complete under the lock, restore_bounded finds db/
    installed, does not fence and returns the result; nothing is both
    reported blocked and installed."""
    h, selector = _published(runner, tmp_path)
    before = _tree(h.durable_run)
    generation = selector.split("/")[1]

    # -- fence first
    local, durable_run, attempt, args = _bounded_restore(runner, h, selector, tmp_path / "local-a", "a" * 32)
    release, parked = threading.Event(), threading.Event()
    real_fsync = runner.fsync_path

    def parking_fsync(path):
        if path == str(durable_run / "rollouts"):     # activate_restore, after the last pre-lock fence check
            parked.set()
            release.wait(30)
        real_fsync(path)
    monkeypatch.setattr(runner, "fsync_path", parking_fsync)
    started = time.monotonic()
    with pytest.raises(runner.Blocked) as blocked:
        runner.restore_bounded(1.0, attempt, *args)
    assert 1.0 <= time.monotonic() - started < 5.0
    assert "the durable side has not answered in 1s while activating" in str(blocked.value)
    assert parked.is_set() and attempt.fenced and not attempt.db_installed and not attempt.sessions_installed
    assert (durable_run / "rollouts" / f"{TID}.jsonl").exists(), "the replica had been renamed to its final name before the park"
    workers = [t for t in threading.enumerate() if t.name == "codex-state-restore"]
    assert len(workers) == 1 and workers[0].is_alive()
    release.set()
    workers[0].join(30)
    assert not workers[0].is_alive()
    monkeypatch.setattr(runner, "fsync_path", real_fsync)
    assert not (local / "db").exists() and not (local / "home" / "sessions").exists() and not (local / "run.json").exists()
    assert not (local / "restore-staging").exists() and not durable_run.exists()
    assert attempt.db_installed is False and attempt.sessions_installed is False
    log = _log(tmp_path)
    assert "the abandoned restore thread ended: fenced: the launch already reported this restore blocked" in log
    assert "restore rolled back: removed" in log and str(local / "restore-staging") in log and str(durable_run) in log
    assert "the launch takes the result" not in log and _tree(h.durable_run) == before

    # -- activation first
    local2, durable_run2, attempt2, args2 = _bounded_restore(runner, h, selector, tmp_path / "local-b", "b" * 32)

    class Instrumented:
        """The attempt's lock, telling the parked thread when restore_bounded is waiting for it."""

        def __init__(self, lock):
            self.lock = lock
            self.waiting = threading.Event()

        def __enter__(self):
            if threading.current_thread().name != "codex-state-restore":
                self.waiting.set()
            self.lock.acquire()

        def __exit__(self, *exc):
            self.lock.release()
    attempt2.lock = Instrumented(attempt2.lock)
    parked2 = threading.Event()
    real_rename = os.rename

    def parking_rename(src, dst, *more, **kwargs):
        if str(src).endswith("restore-staging/db"):       # inside the lock, past its fence check
            parked2.set()
            assert attempt2.lock.waiting.wait(30), "restore_bounded never came for the lock"
        return real_rename(src, dst, *more, **kwargs)
    monkeypatch.setattr(runner.os, "rename", parking_rename)
    started = time.monotonic()
    try:
        restored, pending = runner.restore_bounded(1.0, attempt2, *args2)
    finally:
        monkeypatch.setattr(runner.os, "rename", real_rename)
    assert 1.0 <= time.monotonic() - started < 5.0
    assert parked2.is_set() and attempt2.db_installed and attempt2.sessions_installed and not attempt2.fenced
    assert restored["generation"] == generation and restored["run"] == h.run_id and pending == {"active_goals": [], "queued": {}}
    assert (local2 / "db" / "state_5.sqlite").is_file() and (local2 / "home" / "sessions" / h.relative).is_file()
    assert (local2 / "run.json").is_file() and (durable_run2 / "run.json").is_file() and not (local2 / "restore-staging").exists()
    assert (durable_run2 / "rollouts" / f"{TID}.jsonl").read_bytes() == h.replica.read_bytes()
    assert not [t for t in threading.enumerate() if t.name == "codex-state-restore"], "joined: the result is complete"
    log = _log(tmp_path)
    assert "the restore deadline of 1s passed after the activation had landed; the launch takes the result" in log
    assert log.count("rolled back") == 1 and "fenced" not in log.split("the launch takes the result")[1]
    assert _tree(h.durable_run) == before


def test_a_created_ancestors_parent_must_sync(runner, tmp_path, monkeypatch):
    """Residual 2: while creating the missing ancestors above the durable
    base, a failing sync is tolerated only for a parent that existed before
    the call (the operator's directory); the sync of a parent this call
    created raises Refused naming it, for the ancestors and for the base
    alike."""
    real = runner.fsync_path
    failing = set()

    def refusing(path):
        if path in failing:
            raise OSError(5, "Input/output error")
        real(path)
    monkeypatch.setattr(runner, "fsync_path", refusing)

    # the parent of a created ancestor, created by this call as well
    base = tmp_path / "p" / "q" / "r" / "durable"
    failing.add(str(tmp_path / "p" / "q"))
    with pytest.raises(runner.Refused) as refused:
        runner.ensure_durable_run(str(base / "codex" / "ab" / "runs" / "11"), str(base))
    assert str(refused.value) == (f"created {tmp_path / 'p' / 'q' / 'r'} but could not sync its parent directory {tmp_path / 'p' / 'q'} "
                                  "(OSError: [Errno 5] Input/output error), which this launch created as well; a durable tree below it "
                                  "would depend on an unsynced name")
    assert (tmp_path / "p" / "q" / "r").is_dir() and not base.exists(), "stopped at the failed sync"

    # the base's own parent, created by this call
    failing.clear()
    failing.add(str(tmp_path / "s"))
    base2 = tmp_path / "s" / "durable"
    with pytest.raises(runner.Refused, match="cannot create the durable run directory .*: \\[Errno 5\\] Input/output error"):
        runner.ensure_durable_run(str(base2 / "codex" / "ab" / "runs" / "11"), str(base2))
    assert not (base2 / "codex").exists()

    # the pre-existing parent of the first created ancestor: tolerated, logged
    failing.clear()
    failing.add(str(tmp_path))
    base3 = tmp_path / "t" / "u" / "durable"
    runner.ensure_durable_run(str(base3 / "codex" / "ab" / "runs" / "11"), str(base3))
    assert (base3 / "codex" / "ab" / "runs" / "11" / "generations").is_dir()
    assert f"created {tmp_path / 't'} but could not sync its parent directory (OSError: [Errno 5] Input/output error); the parent is the operator's and the launch goes on" in _log(tmp_path)
    assert "created %s but could not sync" % (tmp_path / "t" / "u") not in _log(tmp_path), "t was created and synced fine"


def test_adopt_unsynced_judges_the_record_alone_and_the_retry_checks_the_generation(runner, tmp_path, monkeypatch):
    """Residual 3: adopt_unsynced runs in run() before child.wait(), so it
    touches no filesystem: a record is taken over when it has the published
    record's keys, a generation name and exactly this run's generations/<name>
    as its path ({"name": "/tmp", "path": "/tmp"} is not).  retry_sync, on the
    capture's worker, drops a record whose generation directory or
    manifest.json is gone instead of acknowledging it, and keeps one whose
    stat fails otherwise for the next retry."""
    h = InProcess(runner, tmp_path)
    h.replicator.pass_once()
    assert h.capture() == ("published", None)
    published = h.cp.last_generation
    record = dict(published, sync_warning="OSError: [Errno 5] Input/output error")
    gone = dict(record, name="000009-00000000", path=str(h.generations / "000009-00000000"))
    h.cp.last_generation = None

    def touched(*args, **kwargs):
        raise AssertionError("adopt_unsynced touched the filesystem: %r" % (args,))
    with monkeypatch.context() as m:
        for name in ("isdir", "isfile", "exists", "lexists", "realpath"):
            m.setattr(runner.os.path, name, touched)
        for name in ("stat", "lstat", "listdir"):
            m.setattr(runner.os, name, touched)
        for bad in ({"name": "/tmp", "path": "/tmp"},
                    dict(record, path="/tmp"),
                    dict(record, path=str(h.generations.parent / "elsewhere" / published["name"])),
                    dict(record, name="../" + published["name"], path=str(h.generations / ".." / published["name"])),
                    dict(record, name=published["name"] + "0", path=str(h.generations / (published["name"] + "0"))),
                    dict((k, v) for k, v in record.items() if k != "trigger"),
                    dict(record, name=["x"], path=str(h.generations / "x"))):
            assert h.cp.adopt_unsynced(bad) is False and h.cp.pending_sync is None, bad
        assert h.cp.adopt_unsynced(gone) is True and h.cp.pending_sync is gone, "well shaped, whether or not it still exists"
    assert "status.json records an unusable unsynced_generation ({'name': '/tmp', 'path': '/tmp'}); it is not taken over" in _log(tmp_path)
    assert "generation 000009-00000000 of the reused run is not acknowledged yet" in _log(tmp_path)

    # the retry finds no such generation directory: dropped, not acknowledged
    h.cp.retry_sync()
    assert h.cp.pending_sync is None and h.cp.last_generation is None
    assert f"generation 000009-00000000 is dropped, not acknowledged: {h.generations / '000009-00000000'} or its manifest.json is gone" in _log(tmp_path)
    assert h.status_json()["unsynced_generation"] is None and h.status_json()["last_generation"] == published, "not promoted"

    # a directory without its manifest.json is gone as well
    (h.generations / "000009-00000000").mkdir()
    assert h.cp.adopt_unsynced(gone) is True
    h.cp.retry_sync()
    assert h.cp.pending_sync is None and h.cp.last_generation is None
    assert _log(tmp_path).count("is dropped, not acknowledged") == 2

    # a stat that fails otherwise keeps the record for the next retry
    assert h.cp.adopt_unsynced(record) is True and h.cp.pending_sync is record
    real_stat = os.stat

    def failing_stat(path, *more, **kwargs):
        if str(path) == record["path"]:
            raise OSError(5, "Input/output error")
        return real_stat(path, *more, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(runner.os, "stat", failing_stat)
        h.cp.retry_sync()
    assert h.cp.pending_sync is record and h.cp.last_generation is None
    assert (f"generation {record['name']} is still not acknowledged: {record['path']} cannot be checked (OSError: [Errno 5] Input/output error); "
            "retried before the next capture") in _log(tmp_path)

    # and acknowledged once the generation is seen and generations/ syncs
    h.cp.retry_sync()
    assert h.cp.pending_sync is None and h.cp.last_generation["name"] == record["name"] and "sync_warning" not in h.cp.last_generation
    assert h.status_json()["last_generation"]["name"] == record["name"] and h.status_json()["unsynced_generation"] is None


# ----------------------------------------------------------------- drain requests (Slice 3c)

def _drain_request(run_dir, nonce, mode="release", **extra):
    """Written the way the launcher will: a temporary file renamed into place."""
    data = {"nonce": nonce, "mode": mode, "requested_at": time.time(), "requester": "test"}
    data.update(extra)
    tmp = run_dir / "drain.request.tmp"
    tmp.write_text(json.dumps(data))
    os.rename(tmp, run_dir / "drain.request")


def _ack(run_dir):
    return json.loads((run_dir / "drain.ack").read_text())


def _starttime(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def _wait_log(run_dir, needle, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        text = (run_dir / "runner.log").read_text()
        if needle in text:
            return text
        time.sleep(0.1)
    raise AssertionError(f"{needle!r} never appeared in runner.log:\n{(run_dir / 'runner.log').read_text()}")


def _fields(record):
    return {key: record[key] for key in ("name", "class", "manifest_sha256", "path")}


@needs_bwrap
@pytest.mark.parametrize("python", INTERPRETERS)
def test_a_release_drain_stops_codex_seals_and_acknowledges(home, python):
    """3c (a) and (e), under both interpreters: with no terminal (stdin is a
    pipe) the request stops the fake the way a forwarded SIGTERM does, the
    ordinary exit path seals, the acknowledgement names the generation and
    proves the stop with the pid and start time recorded at launch, and the
    runner exits with the child's code."""
    nonce = "a1" * 8
    m = Managed(home, python=python, env={"FAKE_SIGTERM_EXIT": "0"})
    try:
        m.cmd("turn 200")
        m.wait_synced()
        _wait_idle(home)
        owner = json.loads((m.run_dir / "owner.json").read_text())
        assert owner["terminal"] == "none" and owner["child_pid"] != m.pid, "bwrap's pid, with its start time"
        assert owner["child_starttime"] == _starttime(owner["child_pid"]) > 0
        assert m.status()["child_starttime"] == owner["child_starttime"] and m.status()["terminal"] == "none"
        _drain_request(m.run_dir, nonce, "release")
        assert m.proc.wait(timeout=40) == 0
        status = m.status()
        assert [t["state"] for t in status["transitions"]][-2:] == ["draining", "exited"]
        assert status["drain"]["nonce"] == nonce and status["drain"]["mode"] == "release" and status["drain"]["requester"] == "test"
        last = status["last_generation"]
        assert last["class"] == "sealed" and last["trigger"] == "final"
        ack = _ack(m.run_dir)
        assert ack == status["drain_ack"]
        assert ack["nonce"] == nonce and ack["mode"] == "release" and ack["outcome"] == "sealed" and "reason" not in ack
        assert ack["generation"] == _fields(last) and ack["exit"] == 0 and ack["stopped"] is True
        assert ack["child"] == {"pid": owner["child_pid"], "starttime": owner["child_starttime"]}
        assert ack["run"] == m.run_dir.name and ack["scope"] == home["scope"] and ack["terminal"] == "none"
        assert ack["requested_at"] == status["drain"]["requested_at"] and ack["acknowledged_at"] >= ack["requested_at"]
        assert not (m.run_dir / "drain.request").exists(), "consumed"
        log = (m.run_dir / "runner.log").read_text()
        assert f"drain request {nonce} (release, from test): stopping Codex gracefully" in log
        assert f"drain {nonce} acknowledged: sealed; generation {last['name']}; Codex stopped: yes" in log
        assert "Codex exited with 0" in log and "killing it" not in log
    finally:
        m.close()


@needs_bwrap
def test_a_renew_drain_holds_the_startup_lock_until_the_runner_is_released(home):
    """3c (b): after the acknowledgement the runner stays alive in state
    drained, the scope's startup.lock held so a second runner refuses; the
    same nonce again is idempotent, a new nonce is answered from the drained
    result, nothing durable changes while holding, and SIGTERM releases it
    with exit 0."""
    nonce = "b2" * 8
    m = Managed(home, env={"FAKE_SIGTERM_EXIT": "0"})
    try:
        m.cmd("turn 200")
        m.wait_synced()
        _wait_idle(home)
        _drain_request(m.run_dir, nonce, "renew")
        status, _ = _wait(home, lambda st: st["state"] == "drained", "the drained hold")
        assert m.proc.poll() is None
        assert [t["state"] for t in status["transitions"]][-3:] == ["draining", "exited", "drained"]
        assert status["hold"]["nonce"] == nonce and status["hold"]["runner_pid"] == m.proc.pid
        ack = _ack(m.run_dir)
        assert ack["outcome"] == "sealed" and ack["mode"] == "renew" and ack["stopped"] is True and ack["exit"] == 0
        assert ack["generation"] == _fields(status["last_generation"]) and status["last_generation"]["class"] == "sealed"
        durable_before = sorted(os.listdir(m.generations)), m.replica.read_bytes()

        proc = _attempt(home)
        assert proc.returncode == 70 and "another managed Codex server for this scope is running in this allocation" in proc.stderr

        _drain_request(m.run_dir, nonce, "renew")
        log = _wait_log(m.run_dir, f"drain request {nonce} (renew) is already acknowledged (sealed); the acknowledgement stands")
        assert _ack(m.run_dir) == ack and log.count(f"drain {nonce} acknowledged") == 1

        _drain_request(m.run_dir, "c3" * 8, "renew")
        _wait_log(m.run_dir, "drain c3c3c3c3c3c3c3c3 acknowledged: sealed")
        again = _ack(m.run_dir)
        assert again["nonce"] == "c3" * 8 and again["outcome"] == "sealed" and again["generation"] == ack["generation"]
        assert again["stopped"] is True and again["exit"] == 0 and again["acknowledged_at"] > ack["acknowledged_at"]
        assert m.proc.poll() is None and m.status()["state"] == "drained", "a renew request does not end the hold"
        assert (sorted(os.listdir(m.generations)), m.replica.read_bytes()) == durable_before

        os.kill(m.proc.pid, signal.SIGTERM)
        assert m.proc.wait(timeout=20) == 0
        status = m.status()
        assert status["state"] == "released" and status["released_by"] == "signal 15"
        assert [t["state"] for t in status["transitions"]][-2:] == ["drained", "released"]
        log = (m.run_dir / "runner.log").read_text()
        assert f"drained for request {nonce} (renew): holding startup.lock" in log and f"kill -TERM {m.proc.pid}" in log
        assert "released (signal 15)" in log
        assert (sorted(os.listdir(m.generations)), m.replica.read_bytes()) == durable_before, "nothing durable touched while holding"
    finally:
        m.close()


@needs_bwrap
def test_a_release_request_with_a_new_nonce_ends_the_hold(home):
    """3c (b), the other way out: a release-mode request while drained is
    acknowledged from the drained result and ends the hold with exit 0."""
    m = Managed(home)                       # the fake exits 37 on SIGTERM: a crash candidate, acknowledged as such
    try:
        m.cmd("turn 100")
        m.wait_synced()
        _wait_idle(home)
        _drain_request(m.run_dir, "d4" * 8, "renew")
        _wait(home, lambda st: st["state"] == "drained", "the drained hold")
        ack = _ack(m.run_dir)
        assert ack["outcome"] == "crash" and ack["exit"] == 37 and ack["stopped"] is True and ack["generation"]["class"] == "crash"
        _drain_request(m.run_dir, "e5" * 8, "release")
        assert m.proc.wait(timeout=20) == 0
        status = m.status()
        assert status["state"] == "released" and status["released_by"] == "drain request e5e5e5e5e5e5e5e5 (release)"
        assert status["drain_ack"]["nonce"] == "e5" * 8 and status["drain_ack"]["outcome"] == "crash"
        assert status["drain_ack"]["generation"] == ack["generation"]
    finally:
        m.close()


@needs_bwrap
def test_a_drain_at_a_tty_takes_a_warm_generation_and_seals_on_the_users_exit(home):
    """3c (c): under a pty the runner records terminal tty; a release request
    does not signal the fake, a warm generation with trigger drain is
    acknowledged warm_only, and the final capture on the user's own exit
    updates the acknowledgement to sealed under the same nonce."""
    nonce = "f6" * 8
    m = Managed(home, tty=True)
    try:
        m.cmd("turn 200")
        m.wait_synced()
        _wait_idle(home)
        owner = json.loads((m.run_dir / "owner.json").read_text())
        assert owner["terminal"] == "tty" and owner["child_starttime"] == _starttime(owner["child_pid"])
        before = m.status()["checkpoint"]["count"]
        _drain_request(m.run_dir, nonce, "release")
        status, _ = _wait(home, lambda st: (st.get("drain_ack") or {}).get("nonce") == nonce, "the warm_only acknowledgement")
        assert status["state"] == "drain_pending" and status["drain"]["nonce"] == nonce
        ack = _ack(m.run_dir)
        assert ack["outcome"] == "warm_only" and ack["stopped"] is False and ack["exit"] is None and "reason" not in ack
        warm = status["last_generation"]
        assert warm["trigger"] == "drain" and warm["class"] == "warm" and ack["generation"] == _fields(warm)
        assert status["checkpoint"]["count"] == before + 1 and status["checkpoint"]["state"] == "idle"
        os.kill(m.pid, 0)
        m.cmd("turn 100")                   # the fake was never signalled and still serves
        assert m.proc.poll() is None
        log = (m.run_dir / "runner.log").read_text()
        assert f"drain request {nonce} (release, from test): a person may be at the terminal, so Codex is not signalled" in log
        assert f"drain {nonce} acknowledged: warm_only; generation {warm['name']}; Codex stopped: no" in log

        assert m.quit() == 0
        status = m.status()
        final = status["last_generation"]
        assert final["class"] == "sealed" and final["trigger"] == "final" and status["state"] == "exited"
        ack = _ack(m.run_dir)
        assert ack == status["drain_ack"]
        assert ack["nonce"] == nonce and ack["mode"] == "release" and ack["outcome"] == "sealed" and ack["terminal"] == "tty"
        assert ack["generation"] == _fields(final) and ack["exit"] == 0 and ack["stopped"] is True
        assert ack["child"] == {"pid": owner["child_pid"], "starttime": owner["child_starttime"]}
        assert (m.run_dir / "runner.log").read_text().count(f"drain {nonce} acknowledged") == 2
    finally:
        m.close()


@needs_bwrap
def test_a_malformed_drain_request_is_rejected_and_stops_nothing(home):
    """3c (d): bad JSON, a bad nonce on a renew request and a bad mode are
    each renamed drain.request.rejected-<time> and logged once; the fake runs
    on and no acknowledgement exists."""
    m = Managed(home)
    try:
        m.cmd("turn 100")
        _wait_idle(home)
        (m.run_dir / "drain.request").write_text("{not json")
        _wait_log(m.run_dir, "drain request rejected (not JSON: ")
        rejected = sorted(m.run_dir.glob("drain.request.rejected-*"))
        assert len(rejected) == 1 and rejected[0].read_text() == "{not json"
        _drain_request(m.run_dir, "XYZ", "renew")
        _wait_log(m.run_dir, "drain request rejected (nonce must be 8-64 lowercase hex characters); kept as ")
        _drain_request(m.run_dir, "a7" * 8, "stop")
        _wait_log(m.run_dir, "drain request rejected (mode must be one of release, renew); kept as ")
        (m.run_dir / "drain.request").write_text("[1, 2]")
        log = _wait_log(m.run_dir, "drain request rejected (not a JSON object)")
        time.sleep(1.5)                     # another scheduler pass: nothing is logged twice
        log = (m.run_dir / "runner.log").read_text()
        assert log.count("drain request rejected") == 4 and log.count("nothing is stopped") == 4
        assert len(list(m.run_dir.glob("drain.request.rejected-*"))) == 4 and not (m.run_dir / "drain.request").exists()
        os.kill(m.pid, 0)
        m.cmd("turn 50")
        status = m.status()
        assert status["state"] == "placed" and "drain" not in status and "drain_ack" not in status
        assert not (m.run_dir / "drain.ack").exists()
        assert m.quit() == 0 and m.status()["state"] == "exited"
    finally:
        m.close()


def test_the_drain_state_machine_in_process(runner, tmp_path):
    """3c, driven by hand over a sleeping child: the ignored new nonce while a
    request is served, the warm_only acknowledgement, idempotency on the
    nonce, the final update under the same nonce, a failed final capture
    with its reason and last durable generation, and the SIGKILL bound on a
    child that ignores SIGTERM."""
    h = InProcess(runner, tmp_path, timeout=5.0)
    h.replicator.pass_once()                # the harness never starts the replicator: one pass makes the replica current
    child = subprocess.Popen(["sleep", "60"])
    try:
        info = {"pid": child.pid, "starttime": runner.read_starttime(child.pid)}
        assert info["starttime"] == _starttime(child.pid) and runner.process_gone(child.pid, info["starttime"]) is False
        assert runner.process_gone(child.pid, info["starttime"] + 1) is True, "the same pid with another start time is another process"
        d = runner.Drain(str(h.run_dir), child, info, "tty", h.status, h.cp, 5.0, h.scope, h.run_id)
        h.cp.drain = d
        d.poll()
        assert d.active is None and d.flight is None and not (h.run_dir / "drain.ack").exists()
        _drain_request(h.run_dir, "a" * 16, "renew", requester=None)
        d.poll()
        assert d.active["nonce"] == "a" * 16 and d.flight["trigger"] == "drain" and h.status.data["state"] == "drain_pending"
        assert not (h.run_dir / "drain.request").exists()
        _drain_request(h.run_dir, "b" * 16, "release")
        d.poll()
        assert "drain request bbbbbbbbbbbbbbbb (release) ignored: request aaaaaaaaaaaaaaaa is still being served" in _log(tmp_path)
        assert not (h.run_dir / "drain.request").exists() and d.active["nonce"] == "a" * 16
        d.flight["thread"].join(30)
        d.poll()
        ack = json.loads((h.run_dir / "drain.ack").read_text())
        assert ack["outcome"] == "warm_only" and ack["stopped"] is False and ack["exit"] is None and ack["requester"] is None
        assert ack["generation"]["class"] == "warm" and ack["generation"]["name"] == h.cp.last_generation["name"]
        assert d.active is None and d.pending_final["nonce"] == "a" * 16 and h.status.data["drain_ack"] == ack
        assert child.poll() is None, "never signalled"
        _drain_request(h.run_dir, "a" * 16, "renew")
        d.poll()
        assert "drain request aaaaaaaaaaaaaaaa (renew) is already acknowledged (warm_only); the acknowledgement stands" in _log(tmp_path)
        assert json.loads((h.run_dir / "drain.ack").read_text()) == ack

        child.kill()
        child.wait()
        skipped = h.cp.final("sealed")
        assert skipped is None and h.cp.last_generation["trigger"] == "final"
        request = d.settle(0, skipped)
        assert request["nonce"] == "a" * 16 and request["mode"] == "renew"
        ack = json.loads((h.run_dir / "drain.ack").read_text())
        assert ack["outcome"] == "sealed" and ack["stopped"] is True and ack["exit"] == 0 and "reason" not in ack
        assert ack["generation"] == _fields(h.cp.last_generation) and d.pending_final is None
        _drain_request(h.run_dir, "c" * 16, "renew")
        d.poll()                            # consumed by a scheduler that outlived settle: answered from the final result, never eaten
        late = json.loads((h.run_dir / "drain.ack").read_text())
        assert not (h.run_dir / "drain.request").exists() and late["nonce"] == "c" * 16 and late["outcome"] == "sealed"
        assert late["generation"] == ack["generation"] and late["stopped"] is True and d.active is None
    finally:
        child.kill()
        child.wait()

    # a failed final capture: the listing never completed, so final() skips; the acknowledgement says so
    h2 = InProcess(runner, tmp_path / "two", timeout=1.0)
    h2.replicator.pass_once()
    assert h2.capture("request", "warm")[0] == "published"
    warm = h2.cp.last_generation
    h2.cp.seq = None
    d2 = runner.Drain(str(h2.run_dir), child, info, "none", h2.status, h2.cp, 1.0, h2.scope, h2.run_id)
    _drain_request(h2.run_dir, "d" * 16, "release")
    assert d2.settle(3, h2.cp.final("crash"))["nonce"] == "d" * 16, "a request that arrived too late for the scheduler is answered on the exit path"
    ack = json.loads((h2.run_dir / "drain.ack").read_text())
    assert ack["outcome"] == "failed" and ack["generation"] is None and ack["exit"] == 3 and ack["stopped"] is True
    assert ack["reason"] == "the final capture was skipped: the durable generations listing never completed"
    assert ack["last_durable"] == _fields(warm)
    assert d2.settle(3, None) is None, "nothing left to answer"

    # terminal none: a child that ignores SIGTERM is killed after the bound
    stubborn = subprocess.Popen([sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
    try:
        h3 = InProcess(runner, tmp_path / "three", timeout=1.0)
        d3 = runner.Drain(str(h3.run_dir), stubborn, {"pid": stubborn.pid, "starttime": runner.read_starttime(stubborn.pid)},
                          "none", h3.status, h3.cp, 1.0, h3.scope, h3.run_id)
        time.sleep(0.5)                     # the interpreter must have installed its handler
        _drain_request(h3.run_dir, "e" * 16, "release")
        started = time.monotonic()
        d3.poll()
        assert h3.status.data["state"] == "draining" and h3.status.data["drain"]["nonce"] == "e" * 16
        assert stubborn.wait(timeout=10) == -signal.SIGKILL and 0.9 <= time.monotonic() - started < 5
        assert "Codex has not exited 1s after the drain's SIGTERM; killing it" in _log(tmp_path)
        assert d3.settle(137, None)["nonce"] == "e" * 16
        ack = json.loads((h3.run_dir / "drain.ack").read_text())
        assert ack["outcome"] == "failed" and ack["reason"] == "no final capture ran" and ack["stopped"] is True and ack["exit"] == 137
    finally:
        stubborn.kill()
        stubborn.wait()


# ----------------------------------------------------------------- scope metadata, parent references and retention


def _scope(h):
    return h.durable_run.parent.parent


def _pin_paths(h, child, generation=None):
    generation = generation or h.cp.last_generation["name"]
    return (_scope(h) / "pins" / h.run_id / generation / (child + ".json"),
            _scope(h) / "lineages" / (child + ".json"))


def _metadata_attempt(runner, h, tmp_path, child="2" * 32, parent=True):
    selector = h.run_id + "/" + h.cp.last_generation["name"]
    local, durable_run, attempt, args = _bounded_restore(runner, h, selector, tmp_path, child)
    meta, status = args[4], args[-2]
    if parent:
        meta["parent"] = {"run": h.run_id, "generation": h.cp.last_generation["name"],
                          "class": h.cp.last_generation["class"], "manifest_sha256": h.cp.last_generation["manifest_sha256"]}
    attempt.durable_created = True
    runner.ensure_durable_run(str(durable_run), str(h.durable_base))
    return local, durable_run, attempt, meta, status


@pytest.mark.parametrize("existing_directories", [False, True])
def test_metadata_pin_and_lineage_are_synced_under_the_lock_before_activation(runner, tmp_path, monkeypatch, existing_directories):
    h, selector = _published(runner, tmp_path)
    local, durable_run, attempt, args = _bounded_restore(runner, h, selector, tmp_path / "child", "2" * 32)
    pin, lineage = _pin_paths(h, local.name)
    lock = _scope(h) / "metadata.lock"
    real_write, real_activate, real_sync = runner.write_metadata, runner.activate_restore, runner.fsync_path
    writes, synced = [], []
    if existing_directories:
        # Visible directories left by an earlier failed sync are not proof
        # that their names are durable.  Sync them even without a mkdir.
        pin.parent.mkdir(parents=True)
        lineage.parent.mkdir()

    def write(path, record):
        if path in (str(pin), str(lineage)):
            owner = json.loads((lock / "owner.json").read_text())
            assert owner["operation"] == "parent pin" and owner["run"] == local.name
            assert owner["runner_pid"] == os.getpid() and owner["runner_starttime"] == runner.read_starttime(os.getpid())
            assert owner["allocation"] == args[4]["allocation"] and owner["acquired_at"] > 0
            assert not (local / "db").exists()
            writes.append(path)
        real_write(path, record)

    def sync(path):
        synced.append(path)
        real_sync(path)

    def activate(*values):
        assert not lock.exists(), "database activation is outside the critical section"
        assert writes == [str(pin), str(lineage)] and str(pin.parent) in synced and str(lineage.parent) in synced
        assert all(str(directory) in synced for directory in (pin.parent.parent, pin.parent.parent.parent, _scope(h)))
        assert json.loads(pin.read_text())["manifest_sha256"] == h.cp.last_generation["manifest_sha256"]
        record = json.loads(lineage.read_text())
        assert record["child_run"] == local.name and record["parent"] == args[4]["parent"]
        assert record["created_at"] > 0 and json.loads(pin.read_text())["pinned_at"] > 0
        assert json.loads((durable_run / "run.json").read_text())["parent"] == record["parent"]
        real_activate(*values)

    monkeypatch.setattr(runner, "write_metadata", write)
    monkeypatch.setattr(runner, "fsync_path", sync)
    monkeypatch.setattr(runner, "activate_restore", activate)
    runner.restore_generation(*args)
    assert (local / "db").is_dir() and not lock.exists()
    assert oct(pin.stat().st_mode & 0o777) == "0o600"
    # A later setup failure must undo only this child, not another child's reference.
    other_pin = pin.with_name("3" * 32 + ".json")
    other_pin.write_text('{"other": true}')
    other_lineage = lineage.with_name("3" * 32 + ".json")
    other_lineage.write_text('{"parent": null}')
    attempt.rollback()
    assert not pin.exists() and not lineage.exists() and not (local / "db").exists()
    assert other_pin.read_text() == '{"other": true}' and other_lineage.read_text() == '{"parent": null}'
    assert not lock.exists() and h.generations.is_dir()


@needs_bwrap
def test_metadata_records_are_rolled_back_after_bind_failure_and_fresh_lineage_has_no_parent(runner, tmp_path, monkeypatch):
    h, selector = _published(runner, tmp_path)
    seen = []

    def fail_binds(codex_home, home_local, keep_existing=False):
        child = Path(home_local).parent.name
        pin, lineage = _pin_paths(h, child)
        record = json.loads(lineage.read_text())
        seen.append(record["parent"])
        assert record["child_run"] == child and (record["parent"] is None or pin.exists())
        raise OSError(5, "bind failure")

    monkeypatch.setattr(runner, "prepare_binds", fail_binds)
    for name, flags in (("restore", ("--restore", selector)), ("fresh", ("--fresh",))):
        code, runs = _restore_in_process(runner, h, tmp_path / name, monkeypatch, *flags)
        assert code == 70
        _assert_rolled_back(runs[0], h, "bind failure")
        assert not any(p.exists() for p in _pin_paths(h, runs[0].name))
    assert seen[0]["generation"] == h.cp.last_generation["name"] and seen[1] is None


@needs_bwrap
def test_abandoned_metadata_lock_is_reported_never_stolen_and_restore_refuses(runner, tmp_path, monkeypatch):
    h, selector = _published(runner, tmp_path)
    lock = _scope(h) / "metadata.lock"
    lock.mkdir()
    gone = subprocess.Popen(["true"])
    gone.wait()
    owner = {"run": "f" * 32, "allocation": {"job": "old-job", "node": os.uname()[1]},
             "runner_pid": gone.pid, "runner_starttime": 1, "operation": "retention", "acquired_at": time.time() - 86400}
    (lock / "owner.json").write_text(json.dumps(owner))
    before = _tree(lock)
    monkeypatch.setattr(runner, "METADATA_LOCK_WAIT", 0.03)
    code, runs = _restore_in_process(runner, h, tmp_path / "child", monkeypatch, "--restore", selector)
    assert code == 70
    _assert_rolled_back(runs[0], h, "metadata lock unavailable")
    status = json.loads((runs[0] / "status.json").read_text())
    assert status["metadata_lock"]["owner"] == owner and status["metadata_lock"]["age_seconds"] >= 86400
    assert status["metadata_lock"]["state"] == "abandoned"
    assert _tree(lock) == before and not any(p.exists() for p in _pin_paths(h, runs[0].name))
    for _ in range(4):
        assert h.capture() == ("published", None), "metadata contention does not prevent publication"
    status = h.status_json()
    assert status["retention"]["outcome"] == "skipped" and "metadata lock unavailable" in status["retention"]["reason"]
    assert status["metadata_lock"]["owner"] == owner and _tree(lock) == before
    assert len(list(h.generations.iterdir())) == 5


@pytest.mark.parametrize("fault", ["mkdir_eio", "mkdir_timeout"])
def test_uncertain_lock_acquisition_grants_nothing_and_never_removes_the_directory(runner, tmp_path, monkeypatch, fault):
    h, _ = _published(runner, tmp_path)
    lock = _scope(h) / "metadata.lock"
    local, durable_run, attempt, meta, status = _metadata_attempt(runner, h, tmp_path / "child")
    real_mkdir = runner.os.mkdir
    monkeypatch.setattr(runner, "METADATA_LOCK_WAIT", 0.02)

    def mkdir(path, *args, **kwargs):
        result = real_mkdir(path, *args, **kwargs)
        if path == str(lock):
            if fault == "mkdir_eio":
                raise OSError(5, "uncertain mkdir")
            if fault == "mkdir_timeout":
                raise TimeoutError("mkdir timed out")
        return result

    monkeypatch.setattr(runner.os, "mkdir", mkdir)
    with pytest.raises(runner.MetadataUnavailable):
        runner.record_lineage(str(_scope(h)), meta, str(local), str(durable_run), status, attempt)
    assert lock.is_dir() and status.data["metadata_lock"]["state"] != "acquired"
    assert not any(p.exists() for p in _pin_paths(h, local.name)) and not (local / "db").exists()
    attempt.rollback()
    assert lock.is_dir(), "even rollback cannot remove an ambiguously acquired lock"


def test_a_slow_but_successful_lock_mkdir_grants_entry_and_is_released(runner, tmp_path, monkeypatch):
    # The wait bound is on a held lock.  A mkdir that returned success after
    # the bound is still this attempt's own directory: entering and releasing
    # it beats leaving a live runner's lock that it could never take again.
    h, _ = _published(runner, tmp_path)
    lock = _scope(h) / "metadata.lock"
    local, durable_run, attempt, meta, status = _metadata_attempt(runner, h, tmp_path / "child")
    real_mkdir = runner.os.mkdir
    monkeypatch.setattr(runner, "METADATA_LOCK_WAIT", 0.02)

    def slow(path, *args, **kwargs):
        result = real_mkdir(path, *args, **kwargs)
        if path == str(lock):
            time.sleep(0.05)
        return result

    monkeypatch.setattr(runner.os, "mkdir", slow)
    runner.record_lineage(str(_scope(h)), meta, str(local), str(durable_run), status, attempt)
    assert all(p.exists() for p in _pin_paths(h, local.name)) and (durable_run / "run.json").exists()
    assert not lock.exists() and status.data["metadata_lock"]["state"] == "released"
    with runner.MetadataLock(str(_scope(h)), h.cp.meta, h.status, "retention"):
        assert lock.is_dir()
    attempt.rollback()
    assert not any(p.exists() for p in _pin_paths(h, local.name)) and not lock.exists()


@pytest.mark.parametrize("fault", ["owner_tmp", "owner_sync", "durable_sync", "cleanup_fails"])
def test_a_failed_owner_record_after_a_successful_mkdir_removes_only_this_attempts_lock(runner, tmp_path, monkeypatch, fault):
    h, _ = _published(runner, tmp_path)
    lock = _scope(h) / "metadata.lock"
    owner_path = lock / "owner.json"
    local, durable_run, attempt, meta, status = _metadata_attempt(runner, h, tmp_path / "child")
    real_write, real_sync, real_rmdir = runner.write_metadata, runner.fsync_path, runner.os.rmdir
    failed = []

    def write(path, record):
        if path == str(owner_path):
            if fault == "owner_tmp":
                # interrupted before the replace: only owner.json.tmp exists
                Path(path + ".tmp").write_text(json.dumps(record))
                raise OSError(5, "owner write failed")
            real_write(path, record)
            if fault in ("owner_sync", "cleanup_fails"):
                raise OSError(5, "owner sync failed")
            return
        real_write(path, record)

    def sync(path):
        if fault == "durable_sync" and path == str(_scope(h)) and owner_path.exists() and not failed:
            failed.append(path)
            raise OSError(5, "durable sync failed")
        real_sync(path)

    def rmdir(path, *args, **kwargs):
        if fault == "cleanup_fails" and path == str(lock):
            raise OSError(5, "rmdir failed")
        return real_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(runner, "write_metadata", write)
    monkeypatch.setattr(runner, "fsync_path", sync)
    monkeypatch.setattr(runner.os, "rmdir", rmdir)
    with pytest.raises(runner.MetadataUnavailable, match="metadata lock unavailable"):
        runner.record_lineage(str(_scope(h)), meta, str(local), str(durable_run), status, attempt)
    assert not any(p.exists() for p in _pin_paths(h, local.name)) and not (durable_run / "run.json").exists()
    record = status.data["metadata_lock"]
    if fault == "cleanup_fails":
        # Only a failed removal leaves the directory, reported, never assumed gone.
        assert lock.is_dir() and record["state"] != "acquired" and "rmdir failed" in record["reason"]
        return
    assert not lock.exists() and record["state"] == "unavailable" and record["removed_own_attempt"] is True
    monkeypatch.setattr(runner, "write_metadata", real_write)
    monkeypatch.setattr(runner, "fsync_path", real_sync)
    with runner.MetadataLock(str(_scope(h)), h.cp.meta, h.status, "retention"):
        assert owner_path.exists()
    assert not lock.exists()


def test_a_failed_release_after_a_pin_keeps_the_restore_and_reports_it(runner, tmp_path, monkeypatch):
    # The pin, lineage and run.json are durable when the section ends; a
    # release failure must not refuse the restore, whose rollback could not
    # take the still-held lock and would orphan them.
    h, selector = _published(runner, tmp_path)
    local, durable_run, attempt, args = _bounded_restore(runner, h, selector, tmp_path / "child", "2" * 32)
    lock = _scope(h) / "metadata.lock"
    status = args[-2]
    real_rmdir = runner.os.rmdir

    def rmdir(path, *values, **kwargs):
        if path == str(lock):
            raise OSError(5, "release rmdir failed")
        return real_rmdir(path, *values, **kwargs)

    monkeypatch.setattr(runner.os, "rmdir", rmdir)
    runner.restore_generation(*args)
    assert (local / "db").is_dir() and all(p.exists() for p in _pin_paths(h, local.name))
    assert json.loads((durable_run / "run.json").read_text())["parent"]["generation"] == h.cp.last_generation["name"]
    record = status.data["metadata_lock"]
    assert record["state"] == "release_failed" and record["reason"].startswith("release failed: ")
    assert "release rmdir failed" in record["reason"] and lock.is_dir()
    assert "metadata_rollback" not in status.data


def test_a_fresh_lineage_is_written_without_the_lock(runner, tmp_path, monkeypatch):
    # A null-parent lineage protects nothing, so a stalled or abandoned lock
    # never blocks the explicit fresh-history choice.
    h, _ = _published(runner, tmp_path)
    lock = _scope(h) / "metadata.lock"
    lock.mkdir()
    (lock / "owner.json").write_text('{"unknown": true}')
    before = _tree(lock)
    local, durable_run, attempt, meta, status = _metadata_attempt(runner, h, tmp_path / "child", parent=False)
    monkeypatch.setattr(runner, "METADATA_LOCK_WAIT", 0.02)
    runner.record_lineage(str(_scope(h)), meta, str(local), str(durable_run), status, attempt)
    pin, lineage = _pin_paths(h, local.name)
    record = json.loads(lineage.read_text())
    assert record["child_run"] == local.name and record["parent"] is None and not pin.exists()
    assert json.loads((durable_run / "run.json").read_text())["parent"] is None
    assert (local / "run.json").exists() and _tree(lock) == before
    assert "metadata_lock" not in status.data


def test_metadata_lock_contention_can_succeed_after_the_exact_owner_releases(runner, tmp_path, monkeypatch):
    h, _ = _published(runner, tmp_path)
    monkeypatch.setattr(runner, "METADATA_LOCK_WAIT", 0.5)
    acquired, release = threading.Event(), threading.Event()
    def owner():
        with runner.MetadataLock(str(_scope(h)), h.cp.meta, h.status, "first"):
            acquired.set()
            assert release.wait(10)
    worker = threading.Thread(target=owner)
    worker.start()
    assert acquired.wait(10)
    timer = threading.Timer(0.05, release.set)
    timer.start()
    with runner.MetadataLock(str(_scope(h)), h.cp.meta, h.status, "second") as lock:
        assert lock.owner["operation"] == "second"
    worker.join(10)
    timer.join(10)
    assert not (_scope(h) / "metadata.lock").exists()


def _retention_generations(runner, h, classes):
    """Completed metadata plus a payload; retention never opens a database or hashes it."""
    template = json.loads((Path(h.cp.last_generation["path"]) / "manifest.json").read_text())
    names = []
    for index, cls in enumerate(classes, 10):
        name = "%06d-abcdef00" % index
        path = h.generations / name
        path.mkdir()
        manifest = dict(template, generation=name, **{"class": cls})
        (path / "manifest.json").write_text(json.dumps(manifest))
        (path / "payload").write_bytes(b"x" * index)
        names.append(name)
    return names


def test_retention_keeps_exactly_selected_generations_and_reports_storage(runner, tmp_path):
    h, _ = _published(runner, tmp_path)
    initial = h.cp.last_generation["name"]
    names = _retention_generations(runner, h, ["warm"] * 7 + ["sealed"] * 2 + ["crash"] * 2)
    pin = _scope(h) / "pins" / h.run_id / names[0]
    pin.mkdir(parents=True)  # an empty pin directory still keeps the generation
    lineages = _scope(h) / "lineages"
    lineages.mkdir()
    (lineages / ("9" * 32 + ".json")).write_text(json.dumps({"parent": {"run": h.run_id, "generation": names[1]}}))
    # Another allocation's run, replicas and partial publications are outside the pruning scope.
    other = _scope(h) / "runs" / ("8" * 32)
    (other / "generations" / "000001-abcdef00").mkdir(parents=True)
    (other / "payload").write_bytes(b"untouched")
    incoming = h.durable_run / ".incoming-unfinished"
    incoming.mkdir()
    (incoming / "payload").write_bytes(b"unfinished")
    incomplete = h.generations / "999999-abcdef00"
    incomplete.mkdir()
    before_other, before_replica = _tree(other), h.replica.read_bytes()
    keep = {names[0], names[1], names[2], *names[4:7], names[8], names[10], incomplete.name}
    candidates = {initial, *names} - keep
    freed = sum(p.stat().st_size for name in candidates for p in (h.generations / name).rglob("*") if p.is_file())
    h.cp.maintain(names[2])  # explicitly keep this publication even though it is older than the newest three
    assert set(os.listdir(h.generations)) == keep
    assert set(os.listdir(h.durable_run / "retiring")) == candidates
    assert all(not list(p.iterdir()) for p in (h.durable_run / "retiring").iterdir())
    status = h.status_json()
    assert status["retention"]["outcome"] == "completed" and status["retention"]["kept"] == len(keep)
    assert status["retention"]["retired"] == len(candidates) and status["retention"]["bytes_freed"] == freed
    assert status["durable_bytes"] == sum(p.stat().st_size for p in h.durable_run.rglob("*") if p.is_file())
    assert status["scope_storage"]["runs"] == 2 and status["scope_storage"]["complete"] is True
    assert status["scope_storage"]["bytes"] == sum(p.stat().st_size for p in _scope(h).rglob("*") if p.is_file())
    assert _tree(other) == before_other and h.replica.read_bytes() == before_replica and (incoming / "payload").exists()


@pytest.mark.parametrize("parent_state", ["retiring", "missing"])
def test_a_parent_retired_or_removed_during_staging_cannot_be_pinned_or_activated(runner, tmp_path, monkeypatch, parent_state):
    h, selector = _published(runner, tmp_path)
    local, durable_run, attempt, args = _bounded_restore(runner, h, selector, tmp_path / "child", "2" * 32)
    real_record = runner.record_lineage
    generation = h.cp.last_generation["name"]
    def retire(*values):
        if parent_state == "retiring":
            (h.durable_run / "retiring").mkdir()
            os.rename(h.generations / generation, h.durable_run / "retiring" / generation)
        else:
            shutil.rmtree(h.generations / generation)
        real_record(*values)
    monkeypatch.setattr(runner, "record_lineage", retire)
    with pytest.raises(runner.Refused, match="retiring" if parent_state == "retiring" else "does not exist"):
        runner.restore_generation(*args)
    assert not (local / "db").exists() and not durable_run.exists()
    assert not any(p.exists() for p in _pin_paths(h, local.name))


@needs_bwrap
def test_retiring_tombstone_refuses_restore_even_when_an_old_generation_directory_exists(runner, tmp_path, monkeypatch):
    h, selector = _published(runner, tmp_path)
    tombstone = h.durable_run / "retiring" / h.cp.last_generation["name"]
    tombstone.mkdir(parents=True)
    code, runs = _restore_in_process(runner, h, tmp_path / "child", monkeypatch, "--restore", selector)
    assert code == 70
    _assert_rolled_back(runs[0], h, "is retiring", "cannot be restored or pinned")


def test_interrupted_deletion_leaves_a_non_restorable_record_and_retries_payload_only(runner, tmp_path, monkeypatch):
    h, _ = _published(runner, tmp_path)
    names = _retention_generations(runner, h, ["warm"] * 4)
    target = h.durable_run / "retiring" / names[0]
    real_delete = runner.delete_retired
    def crash(path, progress):
        assert not (_scope(h) / "metadata.lock").exists(), "payload deletion is outside the lock"
        if path == str(target):
            size = (target / "payload").stat().st_size
            (target / "payload").unlink()
            progress["bytes_freed"] += size
            raise OSError(5, "interrupted deletion")
        real_delete(path, progress)
    monkeypatch.setattr(runner, "delete_retired", crash)
    h.cp.maintain(names[-1])
    assert h.status_json()["retention"]["outcome"] == "failed"
    assert target.is_dir() and (target / "manifest.json").exists() and not (h.generations / names[0]).exists()
    with pytest.raises(runner.Refused, match="retiring"):
        runner.load_manifest(str(h.generations / names[0]), h.run_id + "/" + names[0])
    monkeypatch.setattr(runner, "delete_retired", real_delete)
    h.cp.maintain(names[-1])
    assert h.status_json()["retention"]["outcome"] == "completed" and target.is_dir() and not list(target.iterdir())
    with pytest.raises(runner.Refused, match="retiring"):
        runner.load_manifest(str(h.generations / names[0]), h.run_id + "/" + names[0])


def test_retirement_sync_failure_keeps_payload_and_never_reselects_it(runner, tmp_path, monkeypatch):
    h, _ = _published(runner, tmp_path)
    names = _retention_generations(runner, h, ["warm"] * 4)
    real_sync = runner.fsync_path
    def fail(path):
        if path == str(h.generations):
            raise OSError(5, "retirement sync failed")
        real_sync(path)
    monkeypatch.setattr(runner, "fsync_path", fail)
    h.cp.maintain(names[-1])
    target = h.durable_run / "retiring" / names[0]
    assert target.is_dir() and (target / "payload").exists()
    assert h.status_json()["retention"]["retired"] == 1 and h.status_json()["retention"]["bytes_freed"] == 0
    with pytest.raises(runner.Refused, match="retiring"):
        runner.require_generation(str(h.generations / names[0]), names[0])


def test_an_emptied_tombstone_is_not_walked_again(runner, tmp_path, monkeypatch):
    h, _ = _published(runner, tmp_path)
    names = _retention_generations(runner, h, ["warm"] * 4)
    h.cp.maintain(names[-1])
    retiring = h.durable_run / "retiring"
    emptied = sorted(os.listdir(retiring))
    assert emptied and all(not list((retiring / name).iterdir()) for name in emptied)
    # A tombstone left with payload (an interrupted deletion) is still retried.
    leftover = retiring / "000099-abcdef00"
    leftover.mkdir(mode=0o700)
    (leftover / "payload").write_bytes(b"y" * 7)
    walked, synced = [], []
    real_delete, real_sync = runner.delete_retired, runner.fsync_path

    def delete(path, progress):
        walked.append(os.path.basename(path))
        real_delete(path, progress)

    def sync(path):
        synced.append(path)
        real_sync(path)

    monkeypatch.setattr(runner, "delete_retired", delete)
    monkeypatch.setattr(runner, "fsync_path", sync)
    h.cp.maintain(names[-1])
    assert walked == [leftover.name] and not list(leftover.iterdir())
    assert not any(path == str(retiring / name) for name in emptied for path in synced)
    assert h.status_json()["retention"]["bytes_freed"] == 7
    walked[:] = []
    h.cp.maintain(names[-1])
    assert walked == [], "nothing left to walk"
    assert sorted(os.listdir(retiring)) == sorted(emptied + [leftover.name]), "tombstones stay"


def test_housekeeping_skipped_while_an_earlier_one_runs_is_noted(runner, tmp_path):
    h, _ = _published(runner, tmp_path)
    h.cp.maintain(h.cp.last_generation["name"])
    assert h.status_json()["retention"]["outcome"] == "completed"
    assert h.cp.maintenance_lock.acquire(False)
    try:
        h.cp.maintain("000077-abcdef00")
    finally:
        h.cp.maintenance_lock.release()
    retention = h.status_json()["retention"]
    assert retention["outcome"] == "skipped" and retention["generation"] == "000077-abcdef00"
    assert "still running" in retention["reason"] and retention["retired"] == 0


def test_unreadable_lineage_fails_retention_closed(runner, tmp_path):
    h, _ = _published(runner, tmp_path)
    names = _retention_generations(runner, h, ["warm"] * 4)
    lineages = _scope(h) / "lineages"
    lineages.mkdir()
    (lineages / ("2" * 32 + ".json")).write_text('{"parent":')
    before = _tree(h.generations)
    h.cp.maintain(names[-1])
    assert _tree(h.generations) == before and h.status_json()["retention"]["outcome"] == "failed"
    assert not (_scope(h) / "metadata.lock").exists()


def test_housekeeping_follows_acknowledgement_and_never_delays_the_final_capture(runner, tmp_path, monkeypatch):
    h = InProcess(runner, tmp_path, timeout=0.5)
    h.replicator.pass_once()
    entered, release = threading.Event(), threading.Event()
    real_measure = runner.measure_storage
    def stalled(durable, durable_run):
        assert h.status_json()["last_generation"] == h.cp.last_generation
        assert h.cp.inflight is None
        entered.set()
        assert release.wait(10)
        return real_measure(durable, durable_run)
    monkeypatch.setattr(runner, "measure_storage", stalled)
    worker = h.cp.begin("request", "warm")
    try:
        assert entered.wait(10)
        assert worker.is_alive()
        assert h.capture() == ("published", None), "a second capture does not queue another size walk"
        started = time.monotonic()
        assert h.cp.final("sealed") is None
        assert time.monotonic() - started < 2
        assert h.cp.last_generation["class"] == "sealed"
        assert "exit takes priority" in h.status_json()["retention"]["reason"]
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()


def test_storage_measurement_is_bounded_and_does_not_follow_symlinks(runner, tmp_path, monkeypatch):
    h, _ = _published(runner, tmp_path)
    outside = tmp_path / "unrelated"
    outside.mkdir()
    (outside / "payload").write_bytes(b"x" * 1000)
    (_scope(h) / "link").symlink_to(outside, target_is_directory=True)
    run_bytes, storage = runner.measure_storage(str(_scope(h)), str(h.durable_run))
    assert storage["complete"] and storage["runs"] == 1
    assert run_bytes == storage["bytes"], "the linked payload is not counted"
    monkeypatch.setattr(runner, "STORAGE_WALK_ENTRIES", 2)
    _, storage = runner.measure_storage(str(_scope(h)), str(h.durable_run))
    assert storage["complete"] is False and storage["entries"] == 2 and "lower bounds" in storage["reason"]


def test_failed_lineage_write_rolls_back_only_this_attempt_and_preserves_existing_records(runner, tmp_path, monkeypatch):
    h, _ = _published(runner, tmp_path)
    local, durable_run, attempt, meta, status = _metadata_attempt(runner, h, tmp_path / "child")
    pin, lineage = _pin_paths(h, local.name)
    real_write = runner.write_metadata
    def torn(path, record):
        if path == str(lineage):
            Path(path + ".tmp").write_text("torn")
            raise OSError(28, "no space")
        real_write(path, record)
    monkeypatch.setattr(runner, "write_metadata", torn)
    with pytest.raises(OSError, match="no space"):
        runner.record_lineage(str(_scope(h)), meta, str(local), str(durable_run), status, attempt)
    assert pin.exists()
    attempt.rollback()
    assert not pin.exists() and not pin.parent.exists() and not lineage.exists() and not Path(str(lineage) + ".tmp").exists()
    # An existing child's record, even one with this same run id, is immutable.
    local, durable_run, attempt, meta, status = _metadata_attempt(runner, h, tmp_path / "retry")
    lineage.parent.mkdir(exist_ok=True)
    lineage.write_text('{"existing": true}')
    monkeypatch.setattr(runner, "write_metadata", real_write)
    with pytest.raises(runner.Refused, match="already exists"):
        runner.record_lineage(str(_scope(h)), meta, str(local), str(durable_run), status, attempt)
    attempt.rollback()
    assert lineage.read_text() == '{"existing": true}' and not pin.exists()


def test_a_pin_serializes_with_retention_and_survives_later_publications(runner, tmp_path, monkeypatch):
    h, _ = _published(runner, tmp_path)
    names = _retention_generations(runner, h, ["warm"] * 4)
    local, durable_run, attempt, meta, status = _metadata_attempt(runner, h, tmp_path / "child")
    meta["parent"].update(generation=names[0], manifest_sha256=_sha256(h.generations / names[0] / "manifest.json"))
    entered, release = threading.Event(), threading.Event()
    real_write = runner.write_run_meta
    errors = []
    def hold(*args):
        real_write(*args)
        entered.set()
        assert release.wait(10)
    def pin():
        try:
            runner.record_lineage(str(_scope(h)), meta, str(local), str(durable_run), status, attempt)
        except Exception as exc:
            errors.append(exc)
    monkeypatch.setattr(runner, "write_run_meta", hold)
    worker = threading.Thread(target=pin)
    worker.start()
    try:
        assert entered.wait(10)
        monkeypatch.setattr(runner, "METADATA_LOCK_WAIT", 0.03)
        h.cp.maintain(names[-1])
        assert h.status_json()["retention"]["outcome"] == "skipped"
        assert all((h.generations / name).exists() for name in names)
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive() and not errors
    h.cp.maintain(names[-1])
    assert (h.generations / names[0]).is_dir(), "the pin survives once retention can enter"
    # Lineage is independently protective if the pin file and its directory disappear.
    pin_path, lineage = _pin_paths(h, local.name, names[0])
    pin_path.unlink()
    pin_path.parent.rmdir()
    h.cp.maintain(names[-1])
    assert (h.generations / names[0]).is_dir() and lineage.exists()


def test_rollback_keeps_references_when_metadata_ownership_is_unavailable(runner, tmp_path, monkeypatch):
    h, _ = _published(runner, tmp_path)
    local, durable_run, attempt, meta, status = _metadata_attempt(runner, h, tmp_path / "child")
    runner.record_lineage(str(_scope(h)), meta, str(local), str(durable_run), status, attempt)
    lock = _scope(h) / "metadata.lock"
    lock.mkdir()
    (lock / "owner.json").write_text('{"unknown": true}')
    monkeypatch.setattr(runner, "METADATA_LOCK_WAIT", 0.01)
    attempt.rollback()
    assert not durable_run.exists() and all(p.exists() for p in _pin_paths(h, local.name))
    assert status.data["metadata_rollback"]["outcome"] == "kept" and lock.exists()
    # Reconciliation of that other owner permits the same attempt's cleanup.
    (lock / "owner.json").unlink()
    lock.rmdir()
    attempt.rollback()
    assert not any(p.exists() for p in _pin_paths(h, local.name))


def test_retired_payload_walk_errors_are_reported_and_leave_the_tombstone(runner, tmp_path, monkeypatch):
    h, _ = _published(runner, tmp_path)
    names = _retention_generations(runner, h, ["warm"] * 4)
    target = h.durable_run / "retiring" / names[0]
    real_scandir = runner.os.scandir
    def unreadable(path):
        if str(path) == str(target):
            raise PermissionError(13, "cannot list retired payload")
        return real_scandir(path)
    monkeypatch.setattr(runner.os, "scandir", unreadable)
    h.cp.maintain(names[-1])
    assert target.is_dir() and (target / "payload").exists()
    assert h.status_json()["retention"]["outcome"] == "failed"
    assert "cannot list retired payload" in h.status_json()["retention"]["reason"]
    assert h.status_json()["scope_storage"]["complete"] is False


def test_failed_pin_directory_sync_rolls_back_the_empty_reference(runner, tmp_path, monkeypatch):
    h, _ = _published(runner, tmp_path)
    local, durable_run, attempt, meta, status = _metadata_attempt(runner, h, tmp_path / "child")
    pin, lineage = _pin_paths(h, local.name)
    real_sync = runner.fsync_path

    def failed(path):
        if path == str(pin.parent.parent) and pin.parent.exists():
            raise OSError(5, "pin directory sync failed")
        real_sync(path)

    monkeypatch.setattr(runner, "fsync_path", failed)
    with pytest.raises(OSError, match="pin directory sync failed"):
        runner.record_lineage(str(_scope(h)), meta, str(local), str(durable_run), status, attempt)
    assert pin.parent.is_dir() and not pin.exists()
    monkeypatch.setattr(runner, "fsync_path", real_sync)
    attempt.rollback()
    assert not pin.parent.exists() and not lineage.exists()
    assert not (_scope(h) / "metadata.lock").exists()


def test_tty_drain_acknowledges_its_capture_while_housekeeping_is_stalled(runner, tmp_path, monkeypatch):
    h = InProcess(runner, tmp_path)
    h.replicator.pass_once()
    entered, release = threading.Event(), threading.Event()
    real_measure = runner.measure_storage

    def stalled(durable, durable_run):
        entered.set()
        assert release.wait(15)
        return real_measure(durable, durable_run)

    monkeypatch.setattr(runner, "measure_storage", stalled)
    child = subprocess.Popen(["sleep", "60"])
    worker = None
    try:
        info = {"pid": child.pid, "starttime": runner.read_starttime(child.pid)}
        drain = runner.Drain(str(h.run_dir), child, info, "tty", h.status, h.cp, 5, h.scope, h.run_id)
        _drain_request(h.run_dir, "a" * 16, "renew")
        drain.poll()
        flight = drain.flight
        worker = flight["thread"]
        assert entered.wait(10) and worker.is_alive() and flight["completed"].is_set()
        # Another capture can finish before the scheduler polls the drain.
        assert h.capture() == ("published", None)
        assert h.cp.last_generation["name"] != flight["generation"]
        drain.poll()
        ack = json.loads((h.run_dir / "drain.ack").read_text())
        assert ack["outcome"] == "warm_only" and ack["generation"]["name"] == flight["generation"]
        assert ack["stopped"] is False and child.poll() is None and worker.is_alive()
    finally:
        release.set()
        if worker is not None:
            worker.join(10)
        child.kill()
        child.wait()


def test_expired_lock_wait_does_not_make_one_more_mkdir_attempt(runner, tmp_path, monkeypatch):
    h, _ = _published(runner, tmp_path)
    lock = _scope(h) / "metadata.lock"
    lock.mkdir()
    (lock / "owner.json").write_text('{"unknown": true}')
    now = [0.0]
    monkeypatch.setattr(runner.time, "monotonic", lambda: now[0])

    def release_after_deadline(seconds):
        (lock / "owner.json").unlink()
        lock.rmdir()
        now[0] += runner.METADATA_LOCK_WAIT + 1

    monkeypatch.setattr(runner.time, "sleep", release_after_deadline)
    with pytest.raises(runner.MetadataUnavailable, match="timed out"):
        with runner.MetadataLock(str(_scope(h)), h.cp.meta, h.status, "retention"):
            pytest.fail("an expired attempt has no authority")
    assert not lock.exists(), "do not create an abandoned lock after the wait already expired"


# ----------------------------------------------------------------- issue 46: a restored thread's working directory

FAKE_APP = r'''#!/usr/bin/env python3
"""A scriptable ``codex app-server`` for the runner's pre-launch pass: it
answers ``initialize``, and ``thread/resume {threadId, cwd}`` rewrites the
thread's own row and appends one record to its rollout, as Codex 0.162.1
does (an unknown thread is an error); ``FAKE_APP_HANG`` makes thread/resume
never answer.  Its argv, cwd and every message go to ``FAKE_APP_LOG``."""
import json, os, re, sqlite3, sys, time

sqlite_home = None
for arg in sys.argv[1:]:
    m = re.match(r'sqlite_home="([^"]+)"', arg)
    if m:
        sqlite_home = m.group(1)
log = open(os.environ["FAKE_APP_LOG"], "a")
log.write(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd(), "sqlite_home": sqlite_home}) + "\n")
log.flush()
state = sqlite3.connect(os.path.join(sqlite_home, "state_5.sqlite"), isolation_level=None)


def reply(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    msg = json.loads(line)
    log.write(json.dumps(msg) + "\n")
    log.flush()
    if "id" not in msg:
        continue
    result = {}
    if msg["method"] == "thread/resume":
        if os.environ.get("FAKE_APP_HANG"):
            time.sleep(3600)
        params = msg["params"]
        row = state.execute("select rollout_path from threads where id = ?", (params["threadId"],)).fetchone()
        if row is None:
            reply({"id": msg["id"], "error": {"code": -32600, "message": "no rollout found for thread id"}})
            continue
        state.execute("update threads set cwd = ? where id = ?", (params["cwd"], params["threadId"]))
        with open(row[0], "ab") as rollout:
            rollout.write((json.dumps({"type": "event_msg", "payload": {"type": "thread_settings_applied", "cwd": params["cwd"]}}) + "\n").encode())
        result = {"thread": {"id": params["threadId"], "cwd": params["cwd"]}, "cwd": params["cwd"]}
    reply({"id": msg["id"], "result": result})
'''


def test_clone_layout_recognises_only_a_clone_under_a_job_root(runner):
    assert runner.clone_layout("/local/job222", "/local/job222/mirrors/tok") == ("/local", "job222", "mirrors/tok")
    assert runner.clone_layout("/local/job222/", "/local/job222/mirrors/tok/sub") == ("/local", "job222", "mirrors/tok/sub")
    for root, cwd, reason in [
        ("/local/run222", "/local/run222/mirrors/tok", "not shaped <disk>/job<ID>"),
        ("/local/job222", "/global/home/u/mirrors/tok", "not under the allocation-local root"),
        ("/local/job222", "/local/job222/tmp", "not a clone under /local/job222/mirrors"),
        ("/local/job222", "/local/job222", "not under the allocation-local root"),
    ]:
        disk, job, why = runner.clone_layout(root, cwd)
        assert disk is None and job is None and reason in why, (root, cwd, why)


def test_classify_cwds_moves_only_the_same_mirrors_clone_in_another_allocation(runner):
    here = "/local/job222/mirrors/tok"
    rows = [
        ("t-old", "/local/job111/mirrors/tok", False),
        ("t-sub", "/local/job111/mirrors/tok/src", False),
        ("t-gone", "/local/job111/mirrors/tok/build", False),
        ("t-older", "/local/job7/mirrors/tok/", False),            # any earlier allocation, trailing slash
        ("t-here", here, False),
        ("t-here-sub", here + "/src", False),
        ("t-archived", "/local/job111/mirrors/tok", True),
        ("t-goal", "/local/job111/mirrors/tok", False),
        ("t-queued", "/local/job111/mirrors/tok", False),
        ("t-tmp", "/local/job111/tmp", False),                     # another allocation, not the clone
        ("t-sibling", "/local/job111/mirrors/tokens", False),      # another mirror whose name extends ours
        ("t-other-mirror", "/local/job111/mirrors/other", False),
        ("t-ext", "/global/home/u/project", False),
        ("t-other-disk", "/scratch/job111/mirrors/tok", False),
        ("t-relative", "mirrors/tok", False),
    ]
    pending = {"active_goals": [{"thread": "t-goal", "goal": "g"}], "queued": {"t-queued": 1}}
    moves, kept, skipped = runner.classify_cwds(rows, "/local/job222", here, pending,
                                                isdir=lambda p: p == here + "/src")
    assert skipped is None
    assert moves == [
        {"thread": "t-old", "from": "/local/job111/mirrors/tok", "to": here, "fallback": False},
        {"thread": "t-sub", "from": "/local/job111/mirrors/tok/src", "to": here + "/src", "fallback": False},
        {"thread": "t-gone", "from": "/local/job111/mirrors/tok/build", "to": here, "fallback": True},
        {"thread": "t-older", "from": "/local/job7/mirrors/tok/", "to": here, "fallback": False},
    ]
    assert {k["thread"]: k["reason"] for k in kept} == {
        "t-here": "current", "t-here-sub": "current", "t-archived": "archived", "t-goal": "pending_work",
        "t-queued": "pending_work", "t-tmp": "other_allocation_path", "t-sibling": "other_allocation_path",
        "t-other-mirror": "other_allocation_path", "t-ext": "external", "t-other-disk": "external", "t-relative": "external"}
    assert all(k["cwd"] == dict((r[0], r[1]) for r in rows)[k["thread"]] for k in kept), "kept cwds are reported as recorded"
    assert runner.summarize_kept(kept) == "1 archived, 3 external, 3 other allocation path, 2 pending work"
    # a launch outside the layout moves nothing at all
    moves, kept, skipped = runner.classify_cwds(rows, "/local/job222", "/global/home/u/mirrors/tok", pending)
    assert (moves, kept) == ([], []) and "not under the allocation-local root" in skipped


def test_app_server_argv_must_carry_the_placeholder_exactly_once(runner):
    good = json.dumps(["bash", "-lc", f"exec codex -c 'sqlite_home=\"{PLACEHOLDER}\"' app-server"])
    assert runner.app_server_argv(good, PLACEHOLDER, "/local/job2/state/codex/ab/cd/db")[2] == (
        "exec codex -c 'sqlite_home=\"/local/job2/state/codex/ab/cd/db\"' app-server")
    for value, message in [
        ("not json", "not JSON"),
        ('{"a": 1}', "JSON list of strings"),
        ("[]", "JSON list of strings"),
        ('["codex", 1]', "JSON list of strings"),
        ('["codex", "app-server"]', "placeholder exactly once"),
        (json.dumps(["codex", "-c", f'sqlite_home="{PLACEHOLDER}"', "-c", f'x="{PLACEHOLDER}"', "app-server"]), "exactly once"),
        (json.dumps(["codex", "-c", f'other="{PLACEHOLDER}"', "app-server"]), "exactly once"),
    ]:
        with pytest.raises(runner.Refused, match=message):
            runner.app_server_argv(value, PLACEHOLDER, "/db")


def _state_store(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path / "state_5.sqlite"))
    conn.execute("create table threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL, cwd TEXT NOT NULL, archived INTEGER NOT NULL DEFAULT 0)")
    conn.executemany("insert into threads values (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


@needs_bwrap
def test_place_thread_cwds_without_an_app_server_reports_and_moves_nothing(runner, tmp_path, monkeypatch):
    clone = tmp_path / "disk" / "job2" / "mirrors" / "tok"
    clone.mkdir(parents=True)
    old = str(tmp_path / "disk" / "job1" / "mirrors" / "tok")
    db = tmp_path / "db"
    _state_store(db, [("t1", "/r1", old, 0), ("t2", "/r2", "/elsewhere", 0)])
    monkeypatch.chdir(clone)
    record = runner.place_thread_cwds(None, [], dict(os.environ), str(db), str(tmp_path / "disk" / "job2"),
                                      {}, 5, str(tmp_path / "helper.log"))
    assert record["moved"] == [] and "no --app-server command" in record["skipped"]
    assert [f["thread"] for f in record["failed"]] == ["t1"] and record["failed"][0]["error"] == "not attempted"
    assert record["kept"] == [{"thread": "t2", "cwd": "/elsewhere", "reason": "external"}]
    assert runner.thread_cwds(str(db)) == [("t1", old, False), ("t2", "/elsewhere", False)], "nothing was written"
    line = runner.describe_thread_cwds(record)
    assert "1 thread(s) NOT moved" in line and "codex resume -C %s" % clone in line and "1 external" in line


@needs_bwrap
def test_a_hanging_app_server_is_stopped_within_its_bound_and_reported(runner, tmp_path, monkeypatch):
    clone = tmp_path / "disk" / "job2" / "mirrors" / "tok"
    clone.mkdir(parents=True)
    old = str(tmp_path / "disk" / "job1" / "mirrors" / "tok")
    db = tmp_path / "db"
    rollout = tmp_path / "r1.jsonl"
    rollout.write_text("{}\n")
    _state_store(db, [("t1", str(rollout), old, 0)])
    fake_app = tmp_path / "fake_app.py"
    fake_app.write_text(FAKE_APP)
    monkeypatch.chdir(clone)
    env = dict(os.environ, FAKE_APP_LOG=str(tmp_path / "app.log"), FAKE_APP_HANG="1")
    helper = runner.app_server_argv(json.dumps([sys.executable, str(fake_app), "-c", f'sqlite_home="{PLACEHOLDER}"', "app-server"]),
                                    PLACEHOLDER, str(db))
    started = time.monotonic()
    record = runner.place_thread_cwds(helper, [], env, str(db), str(tmp_path / "disk" / "job2"), {}, 2.0, str(tmp_path / "helper.log"))
    assert time.monotonic() - started < 15
    assert record["moved"] == [] and record["failed"][0]["thread"] == "t1" and "no reply in time" in record["failed"][0]["error"]
    assert record["helper"]["exit"] is not None and record["helper"]["exit"] != 0, "the hung server was signalled"
    assert runner.thread_cwds(str(db)) == [("t1", old, False)]
    assert "NOT moved" in runner.describe_thread_cwds(record)


@needs_bwrap
@pytest.mark.parametrize("python", INTERPRETERS)
def test_a_restore_moves_threads_from_the_predecessors_clone_into_this_one(home, tmp_path, python):
    """Issue 46, hermetic: run A works in ``disk/job111/mirrors/tok`` with
    threads recorded there, in a subdirectory, outside the layout, in another
    allocation path and archived; its clone is removed (the successor's node
    never had it); run B restores in ``disk/job222/mirrors/tok`` with
    ``--app-server``: before the child starts, the app-server moved exactly
    the clone's threads (the child sees the new cwds), everything else is as
    recorded and reported, and the rollout only grew."""
    disk = (tmp_path / "disk").resolve()
    clone_a, clone_b = disk / "job111" / "mirrors" / "tok", disk / "job222" / "mirrors" / "tok"
    for d in (clone_a / "sub", clone_b / "sub"):
        d.mkdir(parents=True)
    external = (tmp_path / "elsewhere").resolve()
    external.mkdir()
    alloc_a, alloc_b = dict(home, local=disk / "job111"), dict(home, local=disk / "job222")
    a = Managed(alloc_a, python, cwd=clone_a)
    try:
        a.cmd("turn 400")
        sub_tid = a.cmd(f"thread {clone_a / 'sub'}").split()[1]
        gone_tid = a.cmd(f"thread {clone_a / 'gone'}").split()[1]
        ext_tid = a.cmd(f"thread {external}").split()[1]
        tmp_tid = a.cmd(f"thread {disk / 'job111' / 'tmp'}").split()[1]
        arch_tid = a.cmd(f"thread {clone_a} archived").split()[1]
        a.wait_synced()
        assert a.quit() == 0
    finally:
        a.close()
    status_a = a.status()
    gen = status_a["last_generation"]
    assert gen["class"] == "sealed" and status_a.get("thread_cwd") is None, "a fresh run has nothing to move"
    source = a.replica.read_bytes()
    shutil.rmtree(clone_a)
    fake_app = tmp_path / "fake_app.py"
    fake_app.write_text(FAKE_APP)
    app_log = tmp_path / "app.log"
    helper = json.dumps([python, str(fake_app), "-c", f'sqlite_home="{PLACEHOLDER}"', "app-server"])
    b = Managed(alloc_b, python, extra=("--restore", f"{a.run_dir.name}/{gen['name']}", "--app-server", helper),
                cwd=clone_b, env={"FAKE_APP_LOG": str(app_log)})
    try:
        assert b.sees["cwds"] == {a.tid: str(clone_b), sub_tid: str(clone_b / "sub"), gone_tid: str(clone_b),
                                  ext_tid: str(external), tmp_tid: str(disk / "job111" / "tmp"), arch_tid: str(clone_a)}
        record = b.status()["thread_cwd"]
        assert record["clone"] == str(clone_b) and record["skipped"] is None and record["failed"] == []
        assert record["moved_count"] == 3 and sorted(m["thread"] for m in record["moved"]) == sorted([a.tid, sub_tid, gone_tid])
        assert [m["fallback"] for m in record["moved"] if m["thread"] == gone_tid] == [True]
        assert {k["thread"]: k["reason"] for k in record["kept"]} == {
            ext_tid: "external", tmp_tid: "other_allocation_path", arch_tid: "archived"}
        assert record["helper"]["exit"] == 0 and record["helper"]["error"] is None
        log = [json.loads(line) for line in app_log.read_text().splitlines()]
        assert log[0]["cwd"] == str(clone_b) and log[0]["sqlite_home"] == str(b.run_dir / "db")
        assert [m.get("method") for m in log[1:]] == ["initialize", "initialized"] + ["thread/resume"] * 3
        assert sorted(m["params"]["threadId"] for m in log[3:]) == sorted([a.tid, sub_tid, gone_tid])
        placed = b.run_dir / "home" / "sessions" / a.relative
        replica = b.durable_run / "rollouts" / f"{a.tid}.jsonl"
        assert placed.read_bytes()[:len(source)] == source and placed.stat().st_size > len(source)

        def grown(st):
            entry = st.get("replica", {}).get("threads", {}).get(a.tid)
            return entry is not None and entry["local_bytes"] == entry["replicated_bytes"] == placed.stat().st_size
        _wait(alloc_b, grown, "the appended record to replicate")
        assert replica.read_bytes() == placed.read_bytes(), "the appended record replicates as growth"
        runner_log = (b.run_dir / "runner.log").read_text()
        assert f"thread cwd: 3 thread(s) moved to {clone_b} through Codex thread/resume" in runner_log
        assert "left as recorded: 1 archived, 1 external, 1 other allocation path" in runner_log
        assert b.quit() == 0
    finally:
        b.close()
