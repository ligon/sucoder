"""Smoke tests on the real Codex CLI for the managed-state (Slice 2) contracts.

Every case drives the installed ``codex app-server`` through
``tests.codex_app_server`` against a loopback fake model, in private
``HOME``/``CODEX_HOME``/``TMPDIR`` directories under ``tmp_path``; nothing
touches the real account.  Skipped when ``codex`` is not on PATH.  The whole
module takes a few seconds on this build (0.162.1); every wait is bounded.

What each test establishes:

1. rollout naming, the projection's byte watermark versus the rollout, and
   that ``threads.rollout_path`` is the *canonical* absolute path;
2. restore = backed-up databases + the rollout at the recorded absolute path,
   in a fresh home, then ``thread/resume`` and a turn that carries the history;
   plus the symlink variant of the same contract;
3. the hazard: a database generation ahead of the rollout silently freezes
   the SQLite projection while the model's context follows the rollout;
4. goal rows (``thread/goal/set``) and queue rows (``codex queue``) as they
   appear in the stores;
5. stdin EOF stops the server cleanly and every store passes ``quick_check``;
6. issue 46: a thread restored by the managed runner on another allocation
   works in the successor's clone (recall and a ``pwd`` through
   ``exec_command``), the runner having moved its cwd with ``thread/resume``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import pytest

from sucoder.agent_mode import config_override
from sucoder.codex_state import app_server_command
from tests.codex_app_server import (
    DATABASES,
    AppServer,
    FakeModel,
    Home,
    backup_databases,
    codex_available,
    goal_rows,
    model_options,
    projection,
    queue_rows,
    quick_check_all,
    rollout_of,
    settled_projection,
    thread_row,
)

pytestmark = pytest.mark.skipif(not codex_available(), reason="Codex CLI not installed")

ROLLOUT_NAME = re.compile(r"rollout-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\.jsonl")


@pytest.fixture
def root(tmp_path):
    # Resolved, so "the same absolute path" below means the canonical one.
    return tmp_path.resolve()


@pytest.fixture
def workspace(root):
    ws = root / "ws"
    ws.mkdir()
    return ws


@pytest.fixture
def model():
    with FakeModel() as fake:
        yield fake


def _markers(model, result, prefix):
    """The marker texts the model received in the requests of one turn."""
    requests = model.requests_since(result.requests_before)
    assert len(requests) == 1, [FakeModel.user_texts(r) for r in requests]
    return [t for t in FakeModel.user_texts(requests[0]) if t.startswith(prefix)]


# ----------------------------------------------------------------- 1


def test_rollout_name_and_projection_track_turns(root, workspace, model):
    real = root / "real"
    real.mkdir()
    link = root / "link"
    link.symlink_to(real)
    codex_home = link / "codex-home"              # CODEX_HOME with a symlink component
    home = Home(root, codex_home=codex_home, trusted=[workspace])
    sqlite_home = root / "sqlite"
    with AppServer(home, model, workspace, sqlite_home=sqlite_home) as server:
        thread_id = server.thread_start()
        for i in range(3):
            result = server.turn(thread_id, f"MARK1_turn_{i}")
            assert result.ok and result.turn_id, result.completed
            rollout = rollout_of(codex_home, thread_id)
            # At the moment turn/completed arrives the watermark never exceeds the file.
            at_completed = projection(sqlite_home, thread_id)
            size_at_completed = rollout.stat().st_size
            assert at_completed.next_rollout_byte_offset is not None
            assert at_completed.next_rollout_byte_offset <= size_at_completed
            # Observed on 0.162.1: equal, not lagging, once the projection settles.
            settled = settled_projection(sqlite_home, thread_id)
            assert settled.next_rollout_byte_offset == rollout.stat().st_size
            assert settled.thread_turns == i + 1
        assert _markers(model, result, "MARK1_") == ["MARK1_turn_0", "MARK1_turn_1", "MARK1_turn_2"]

        match = ROLLOUT_NAME.fullmatch(rollout.name)
        assert match and match.group(1) == thread_id, rollout.name
        assert rollout.name.endswith(f"-{thread_id}.jsonl")
        assert len(rollout.relative_to(codex_home / "sessions").parts) == 4   # YYYY/MM/DD/rollout-*.jsonl

        row = thread_row(sqlite_home, thread_id)
        assert row is not None
        # Codex canonicalizes: the row holds the realpath, not the symlinked path as written.
        assert row["rollout_path"] == os.path.realpath(rollout)
        assert row["rollout_path"].startswith(str(real / "codex-home" / "sessions"))
        assert row["rollout_path"] != str(rollout)
        assert row["cli_version"]
        assert server.stop().clean


# ----------------------------------------------------------------- 2


def _three_turns_then_stop(home, model, workspace, sqlite_home, prefix):
    with AppServer(home, model, workspace, sqlite_home=sqlite_home, label="source") as server:
        thread_id = server.thread_start()
        for i in range(3):
            assert server.turn(thread_id, f"{prefix}{i}").ok
        settled_projection(sqlite_home, thread_id)
        rel = rollout_of(home.codex_home, thread_id).relative_to(home.codex_home)
        assert server.stop().clean
    return thread_id, rel


def test_resume_from_backup_plus_rollout_in_a_fresh_home(root, workspace, model):
    """The restore contract: backed-up stores + the rollout at its recorded absolute path.

    ``threads.rollout_path`` is absolute (and canonical), so the fresh
    ``CODEX_HOME`` is created at the *same* path: the old home is renamed
    aside and the directory name re-used; only ``config.toml`` and the one
    rollout (at the same ``sessions/YYYY/MM/DD/`` relative path) are placed
    in it.  Codex creates ``thread-writer-locks``, ``skills``, ``tmp`` and the
    rest itself.
    """
    codex_home = root / "codex-home"
    home = Home(root, codex_home=codex_home, trusted=[workspace])
    source_stores, restored_stores = root / "sqlite-A", root / "sqlite-B"
    thread_id, rel = _three_turns_then_stop(home, model, workspace, source_stores, "MARK2_")
    recorded = thread_row(source_stores, thread_id)["rollout_path"]
    assert recorded == str(codex_home / rel)

    assert set(backup_databases(source_stores, restored_stores).values()) == {"ok"}
    assert set(DATABASES) <= {p.name for p in restored_stores.iterdir()}

    moved_aside = root / "codex-home-old"
    codex_home.rename(moved_aside)
    fresh = Home(root, codex_home=codex_home, trusted=[workspace])
    (codex_home / rel).parent.mkdir(parents=True)
    shutil.copyfile(moved_aside / rel, codex_home / rel)
    assert sorted(p.name for p in codex_home.iterdir()) == ["config.toml", "sessions"]

    with AppServer(fresh, model, workspace, sqlite_home=restored_stores, label="restored") as server:
        before_resume = model.count()
        resumed = server.thread_resume(thread_id)
        assert len(resumed["thread"]["turns"]) == 3
        assert resumed["thread"]["path"] == recorded
        # A goal- or queue-driven turn would arrive asynchronously; give it a second.
        assert model.wait_for_requests(before_resume, 1, timeout=1.0) == [], "thread/resume alone must not call the model"
        result = server.turn(thread_id, "MARK2_3")
        assert result.ok
        assert _markers(model, result, "MARK2_") == ["MARK2_0", "MARK2_1", "MARK2_2", "MARK2_3"]
        assert len(server.turns_list(thread_id)) == 4 and len(server.items_list(thread_id)) >= 4
        after = settled_projection(restored_stores, thread_id)
        assert after.thread_turns == 4
        assert after.next_rollout_byte_offset == (codex_home / rel).stat().st_size
        assert server.stop().clean
    assert (codex_home / "thread-writer-locks").is_dir()


def test_resume_follows_a_symlink_at_the_recorded_path(root, workspace, model):
    """Variant: the fresh home lives elsewhere and a symlink sits at the recorded path.

    Codex opens the recorded path, follows the link, and on resume rewrites
    ``threads.rollout_path`` to the new canonical location.
    """
    codex_home = root / "codex-home"
    home = Home(root, codex_home=codex_home, trusted=[workspace])
    source_stores, restored_stores = root / "sqlite-A", root / "sqlite-B"
    thread_id, rel = _three_turns_then_stop(home, model, workspace, source_stores, "MARK2s_")
    backup_databases(source_stores, restored_stores)
    codex_home.rename(root / "codex-home-old")
    elsewhere = root / "elsewhere" / "codex-home"
    fresh = Home(root / "elsewhere", codex_home=elsewhere, trusted=[workspace])
    (elsewhere / rel).parent.mkdir(parents=True)
    shutil.copyfile(root / "codex-home-old" / rel, elsewhere / rel)
    codex_home.symlink_to(elsewhere)

    with AppServer(fresh, model, workspace, sqlite_home=restored_stores, label="restored") as server:
        assert len(server.thread_resume(thread_id)["thread"]["turns"]) == 3
        result = server.turn(thread_id, "MARK2s_3")
        assert _markers(model, result, "MARK2s_") == ["MARK2s_0", "MARK2s_1", "MARK2s_2", "MARK2s_3"]
        assert server.stop().clean
    assert thread_row(restored_stores, thread_id)["rollout_path"] == str(elsewhere / rel)


# ----------------------------------------------------------------- 3


def test_database_ahead_of_rollout_is_the_hazard(root, workspace, model):
    """A generation whose watermark lies beyond the rollout freezes the projection.

    Five turns, the databases backed up after the fifth, the rollout cut back
    to its size after the second.  The gap (three turns) exceeds what one
    resume plus one turn appends, so the new bytes stay below the stale
    ``next_rollout_byte_offset`` and are never projected: the read paths keep
    showing the three phantom turns, ``thread_turns`` does not grow, while
    the model's context follows the rollout (turns 0, 1 and the new one).
    With a one-turn gap the append crosses the watermark and the projection
    resumes mid-record instead, which is the other face of the same hazard.
    """
    home = Home(root, trusted=[workspace])
    source_stores, restored_stores = root / "sqlite-A", root / "sqlite-B"
    sizes = []
    with AppServer(home, model, workspace, sqlite_home=source_stores, label="source") as server:
        thread_id = server.thread_start()
        for i in range(5):
            assert server.turn(thread_id, f"MARK3_{i}").ok
            settled_projection(source_stores, thread_id)
            sizes.append(rollout_of(home.codex_home, thread_id).stat().st_size)
        assert server.stop().clean
    assert sizes == sorted(sizes) and len(set(sizes)) == 5
    backup_databases(source_stores, restored_stores)
    rollout = rollout_of(home.codex_home, thread_id)
    os.truncate(rollout, sizes[1])
    stale = projection(restored_stores, thread_id)
    assert stale.next_rollout_byte_offset == sizes[4] and stale.thread_turns == 5

    with AppServer(home, model, workspace, sqlite_home=restored_stores, label="restored") as server:
        resumed = server.thread_resume(thread_id)
        assert len(resumed["thread"]["turns"]) == 5, "the read path shows the database's phantom turns"
        result = server.turn(thread_id, "MARK3_5")
        assert result.ok, "the turn completes anyway"
        assert _markers(model, result, "MARK3_") == ["MARK3_0", "MARK3_1", "MARK3_5"]
        after = settled_projection(restored_stores, thread_id)
        assert rollout.stat().st_size < stale.next_rollout_byte_offset, "the append stayed below the watermark"
        assert after.as_tuple() == stale.as_tuple(), "nothing was projected"
        assert len(server.turns_list(thread_id)) == 5
        assert server.stop().clean


# ----------------------------------------------------------------- 4


def test_goal_and_queue_rows_are_visible_in_the_stores(root, workspace, model):
    """Goal rows via ``thread/goal/set``; queue rows via the ``codex queue`` CLI.

    The protocol of this build has no client request that queues a message
    (only the ``thread/queue/changed`` notification), so the row is written
    by ``codex -c sqlite_home=... queue --thread ID --message TEXT`` with the
    server stopped.  The goal is paused before the stop and the thread is
    never resumed here: an active goal drives unprompted turns.
    """
    home = Home(root, trusted=[workspace])
    stores = root / "sqlite"
    with AppServer(home, model, workspace, sqlite_home=stores) as server:
        thread_id = server.thread_start()
        assert server.turn(thread_id, "MARK4_0").ok
        active = server.goal_set(thread_id, "MARK4_objective")
        assert active["goal"]["status"] == "active"
        rows = goal_rows(stores)
        assert [(r["thread_id"], r["objective"], r["status"]) for r in rows] == [(thread_id, "MARK4_objective", "active")]
        assert rows[0]["goal_id"]
        paused = server.goal_pause(thread_id)
        assert paused["goal"]["status"] == "paused" and paused["goal"]["objective"] == "MARK4_objective"
        assert [r["status"] for r in goal_rows(stores)] == ["paused"]
        got = server.goal_get(thread_id)
        assert got is not None and got["status"] == "paused"
        assert queue_rows(stores) == []
        assert server.stop().clean

        queued = server.queue_message(thread_id, "MARK4_queued_text")
        assert queued.returncode == 0, queued.stderr
        assert "Queued message" in queued.stdout and thread_id in queued.stdout

    rows = queue_rows(stores)
    assert len(rows) == 1 and rows[0]["thread_id"] == thread_id and rows[0]["queue_order"] == 0
    payload = json.loads(rows[0]["payload_json"])
    assert "MARK4_queued_text" in rows[0]["payload_json"]
    assert payload["UserInput"]["content"][0] == {"type": "text", "text": "MARK4_queued_text", "text_elements": []}
    assert [r["status"] for r in goal_rows(stores)] == ["paused"]


# ----------------------------------------------------------------- 5


def test_stdin_eof_stops_cleanly_and_quick_check_passes(root, workspace, model):
    home = Home(root, trusted=[workspace])
    stores = root / "sqlite"
    with AppServer(home, model, workspace, sqlite_home=stores) as server:
        thread_id = server.thread_start()
        assert server.turn(thread_id, "MARK5_0").ok
        assert server.pids(), "the server is a live codex process"
        stopped = server.stop()
        assert stopped.steps == ["stdin-close"] and stopped.exit_code == 0, stopped.__dict__
        assert server.pids() == []
    names = {p.name for p in stores.iterdir()}
    assert set(DATABASES) <= names
    # WAL/SHM sidecars may remain after a clean stop; the backup API reads through them.
    verdicts = quick_check_all(stores)
    assert set(verdicts) == set(DATABASES) and set(verdicts.values()) == {"ok"}, verdicts
    assert set(backup_databases(stores, root / "backup").values()) == {"ok"}
    assert not any(p.name.endswith(("-wal", "-shm")) for p in (root / "backup").iterdir())
    assert not list(home.codex_home.glob("*.sqlite")), "no store in CODEX_HOME when sqlite_home is set"


# ----------------------------------------------------------------- 7


needs_bwrap = pytest.mark.skipif(not codex_available(need_bwrap=True), reason="bwrap not installed")


def _bwrap(binds):
    """The runner's layout (``codex_state_runner.bwrap_argv``), spelled out so
    this test does not import a module that is still moving."""
    argv = ["bwrap", "--dev-bind", "/", "/"]
    for local, target in binds:
        argv += ["--bind", str(local), str(target)]
    return argv + ["--unshare-user", "--uid", str(os.getuid()), "--gid", str(os.getgid()), "--die-with-parent", "--"]


def _listed(server, thread_id, **params):
    """``thread/list`` hits for ``thread_id`` -> that entry's ``path`` (or None)."""
    for entry in server.rpc("thread/list", {"limit": 100, **params}).get("data", []):
        if entry.get("id") == thread_id:
            return entry.get("path")
    return None


@pytest.mark.parametrize("layout", [
    pytest.param("control", id="control"),
    pytest.param("sessions", id="sessions-bound", marks=needs_bwrap),
    pytest.param("both", id="sessions-and-archived-bound", marks=needs_bwrap),
    pytest.param("whole-home", id="whole-home-bound", marks=needs_bwrap),
])
def test_archive_renames_the_rollout_and_fails_with_exdev_across_bind_mounts(root, workspace, model, layout):
    """``thread/archive`` renames the rollout into ``CODEX_HOME/archived_sessions``.

    Control and a single bind covering the whole home: the file is renamed
    (same inode) from ``sessions/YYYY/MM/DD/`` to a flat ``archived_sessions/``,
    ``threads.rollout_path`` follows, ``archived`` is set, the writer lock is
    removed, the file is immutable while archived, ``thread/resume`` and
    ``turn/start`` are refused, default ``thread/list`` hides it and
    ``archived: true`` shows it, and ``thread/unarchive`` renames it back and
    resets the row.  With ``sessions`` bound alone *or* with
    ``archived_sessions`` bound as a second mount, the rename crosses a
    vfsmount boundary: Codex reports EXDEV, does not fall back to copy+delete,
    leaves the rollout and the row untouched (but has already dropped the
    writer lock), and the thread stays usable.
    """
    codex_home = root / "codex-home"
    home = Home(root, codex_home=codex_home, trusted=[workspace])
    local = root / "local"
    stores = root / "sqlite"
    if layout == "control":
        prefix, where = [], codex_home
    elif layout == "whole-home":
        where = local / "home"
        for d in ("sessions", "archived_sessions"):
            (where / d).mkdir(parents=True)
        shutil.copyfile(codex_home / "config.toml", where / "config.toml")
        prefix = _bwrap([(where, codex_home)])
    else:
        where = local
        binds = ["sessions"] if layout == "sessions" else ["sessions", "archived_sessions"]
        for d in binds:
            (local / d).mkdir(parents=True)
            (codex_home / d).mkdir()
        prefix = _bwrap([(local / d, codex_home / d) for d in binds])
    archive_works = layout in ("control", "whole-home")

    with AppServer(home, model, workspace, sqlite_home=stores, prefix=prefix) as server:
        thread_id = server.thread_start()
        for i in range(2):
            assert server.turn(thread_id, f"MARK7_{i}").ok
        settled_projection(stores, thread_id)
        rollout = rollout_of(where, thread_id)
        inode, size, before = rollout.stat().st_ino, rollout.stat().st_size, rollout.read_bytes()
        recorded = thread_row(stores, thread_id)["rollout_path"]
        assert recorded == str(codex_home / "sessions" / rollout.relative_to(where / "sessions"))
        lock = where / "thread-writer-locks" / f"{thread_id}.lock" if layout == "whole-home" else codex_home / "thread-writer-locks" / f"{thread_id}.lock"
        assert lock.exists()

        if not archive_works:
            with pytest.raises(RuntimeError, match="Cross-device link \\(os error 18\\)"):
                server.rpc("thread/archive", {"threadId": thread_id})
            assert rollout.stat().st_ino == inode and rollout.read_bytes() == before
            assert not list((where / "archived_sessions").glob("*")) and not list(codex_home.glob("archived_sessions/*"))
            row = thread_row(stores, thread_id)
            assert (row["rollout_path"], row["archived"], row["archived_at"]) == (recorded, 0, None)
            assert not lock.exists(), "the failed archive has already released the writer lock"
            assert _listed(server, thread_id) == recorded and _listed(server, thread_id, archived=True) is None
            assert len(server.thread_resume(thread_id)["thread"]["turns"]) == 2
            assert lock.exists()
        else:
            assert server.rpc("thread/archive", {"threadId": thread_id}) == {}
            archived = where / "archived_sessions" / rollout.name          # flat: no YYYY/MM/DD
            assert archived.is_file() and not rollout.exists()
            assert archived.stat().st_ino == inode and archived.read_bytes() == before, "a rename, not a copy"
            assert not lock.exists()
            recorded_archived = str(codex_home / "archived_sessions" / rollout.name)
            row = thread_row(stores, thread_id)
            assert (row["rollout_path"], row["archived"]) == (recorded_archived, 1) and row["archived_at"]
            assert _listed(server, thread_id) is None and _listed(server, thread_id, archived=True) == recorded_archived
            assert _listed(server, thread_id, useStateDbOnly=True) is None
            assert len(server.thread_read(thread_id)["thread"]["turns"]) == 2
            assert len(server.items_list(thread_id)) == 2 and len(server.turns_list(thread_id)) == 2
            with pytest.raises(RuntimeError, match="is archived"):
                server.thread_resume(thread_id)
            with pytest.raises(RuntimeError, match="thread not found"):
                server.rpc("turn/start", {"threadId": thread_id, "input": [{"type": "text", "text": "MARK7_refused"}]})
            assert (archived.stat().st_ino, archived.stat().st_size, archived.read_bytes()) == (inode, size, before), "immutable while archived"
            restored = server.rpc("thread/unarchive", {"threadId": thread_id})
            assert restored["thread"]["path"] == recorded and rollout.is_file() and not archived.exists()
            assert rollout.stat().st_ino == inode and rollout.read_bytes() == before
            row = thread_row(stores, thread_id)
            assert (row["rollout_path"], row["archived"], row["archived_at"]) == (recorded, 0, None)
            assert _listed(server, thread_id) == recorded and _listed(server, thread_id, archived=True) is None
            assert len(server.thread_resume(thread_id)["thread"]["turns"]) == 2
        result = server.turn(thread_id, "MARK7_after")
        assert _markers(model, result, "MARK7_") == ["MARK7_0", "MARK7_1", "MARK7_after"]
        assert server.stop().clean


# ----------------------------------------------------------------- 6


RUNNER = Path(__file__).resolve().parents[1] / "sucoder" / "codex_state_runner.py"
PLACEHOLDER = "__SUCODER_CODEX_STATE_DB__"


class _AllocationHome(Home):
    """The private home, plus the allocation-local root the runner resolves."""

    def __init__(self, root: Path, local_root: Path, **kwargs) -> None:
        super().__init__(root, **kwargs)
        self.local_root = local_root

    def env(self, **extra):
        extra.setdefault("SUCODER_LOCAL_ROOT", str(self.local_root))
        return super().env(**extra)


def _run_status(local_root, scope, predicate, what, timeout=60):
    scope_dir = Path(local_root) / "state" / "codex" / scope
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            run_dir = scope_dir / (scope_dir / "latest-run").read_text().strip()
            last = json.loads((run_dir / "status.json").read_text())
            if predicate(last):
                return last, run_dir
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}; last status: {last}")


@pytest.mark.skipif(not codex_available(need_bwrap=True), reason="needs codex and bwrap")
def test_a_restored_thread_works_in_the_successors_clone(root):
    """Issue 46 on the real build, as the pilot ran it: run A (the managed
    runner, allocation ``disk/job111``) holds a two-turn thread started in its
    clone ``disk/job111/mirrors/tok`` and seals; that allocation's disk is
    gone; run B restores the generation in ``disk/job222`` from its own clone
    with ``--app-server``.  Before any client touches the thread its stored
    cwd is B's clone; a plain ``thread/resume`` (what the TUI's "Use session
    directory" does) and one turn then recall both earlier messages and run
    ``pwd; ls`` in B's clone, and the restored rollout bytes are a prefix of
    the resumed one."""
    disk = root / "disk"
    clone_a, clone_b = disk / "job111" / "mirrors" / "tok", disk / "job222" / "mirrors" / "tok"
    for d in (clone_a, clone_b):
        d.mkdir(parents=True)
    (clone_b / "SUCCESSOR_CLONE").write_text("here\n")
    codex_home = root / "codex-home"
    durable = root / "durable"
    durable.mkdir()
    scope = "4e" * 32
    placeholder = ["-c", config_override("sqlite_home", PLACEHOLDER)]

    def prefix(*flags):
        return [sys.executable, str(RUNNER), "--scope", scope, "--durable-root", str(durable),
                "--checkpoint-timeout", "40", *flags, "--"]

    def sealed(local_root):
        return _run_status(local_root, scope, lambda st: st["state"] == "exited"
                           and st["checkpoint"]["state"] in ("idle", "failed")
                           and (st.get("last_generation") or {}).get("class") == "sealed", "the sealed generation")

    home_a = _AllocationHome(root, disk / "job111", codex_home=codex_home, trusted=[clone_a, clone_b])
    (codex_home / "sessions").mkdir()
    with FakeModel(reply_text="noted", exec_on="RUN_PWD", exec_cmd="pwd; ls") as model:
        server = AppServer(home_a, model, clone_a, prefix=prefix(), extra_options=placeholder, label="predecessor")
        server.start()
        try:
            server.initialize()
            tid = server.thread_start()
            status, run_a = _run_status(disk / "job111", scope, lambda st: st["state"] in ("placed", "misplaced"), "placement")
            assert status["state"] == "placed", status
            assert server.turn(tid, "MARKR_0 remember the word heron").ok
            assert server.turn(tid, "MARKR_1 and the number 46").ok
            local_a = rollout_of(run_a / "home", tid)
            assert thread_row(run_a / "db", tid)["cwd"] == str(clone_a)
            stop = server.stop(timeout=60)
            assert stop.clean, (stop.steps, stop.exit_code, server.stderr_text())
        finally:
            if server.running():
                server.kill()
        status_a, _ = sealed(disk / "job111")
        source = local_a.read_bytes()
        selector = f"{run_a.name}/{status_a['last_generation']['name']}"
        shutil.rmtree(disk / "job111")      # the successor's node never had the predecessor's disk

        # What the launcher derives from the launch's own argv: the executable,
        # the placeholder and the config overrides (here the fake provider).
        helper = json.dumps(app_server_command(["codex", *placeholder, *model_options(model.port), "--yolo"]))
        assert json.loads(helper)[-1] == "app-server" and "--yolo" not in helper
        (disk / "job222").mkdir(exist_ok=True)
        home_b = _AllocationHome(root, disk / "job222", codex_home=codex_home, trusted=[clone_a, clone_b])
        server = AppServer(home_b, model, clone_b, prefix=prefix("--restore", selector, "--app-server", helper),
                           extra_options=placeholder, label="successor")
        server.start()
        try:
            server.initialize()
            status_b, run_b = _run_status(disk / "job222", scope, lambda st: st["state"] in ("launched", "placed")
                                          and "thread_cwd" in st, "the restored run")
            record = status_b["thread_cwd"]
            assert record["moved_count"] == 1 and record["failed"] == [] and record["skipped"] is None, record
            assert record["moved"][0] == {"thread": tid, "from": str(clone_a), "to": str(clone_b), "fallback": False}
            assert thread_row(run_b / "db", tid)["cwd"] == str(clone_b), "moved before any client resumed it"
            before = model.count()
            resumed = server.thread_resume(tid)
            assert (resumed.get("cwd") or resumed["thread"].get("cwd")) == str(clone_b)
            assert len(resumed["thread"]["turns"]) == 2
            assert model.wait_for_requests(before, 1, timeout=1.0) == [], "neither the pass nor the resume called the model"
            result = server.turn(tid, "RUN_PWD what did I ask you to remember?", timeout=90)
            assert result.ok, result.completed
            requests = model.requests_since(result.requests_before)
            assert len(requests) == 2, [FakeModel.user_texts(r) for r in requests]
            recalled = [t for t in FakeModel.user_texts(requests[0]) if t.startswith("MARKR_")]
            assert recalled == ["MARKR_0 remember the word heron", "MARKR_1 and the number 46"]
            assert {FakeModel.environment_cwd(r) for r in requests} == {str(clone_b)}
            outputs = FakeModel.tool_outputs(requests[1])
            assert len(outputs) == 1 and "exited with code 0" in outputs[0], outputs
            assert str(clone_b) in outputs[0] and "SUCCESSOR_CLONE" in outputs[0], outputs
            placed = rollout_of(run_b / "home", tid)
            assert placed.read_bytes()[:len(source)] == source and placed.stat().st_size > len(source)
            assert not list((codex_home / "sessions").rglob("*.jsonl")), "the shared sessions directory stays empty"
            stop = server.stop(timeout=60)
            assert stop.clean, (stop.steps, stop.exit_code, server.stderr_text())
        finally:
            if server.running():
                server.kill()
        status_b, _ = sealed(disk / "job222")
        assert status_b["thread_cwd"]["moved_count"] == 1
