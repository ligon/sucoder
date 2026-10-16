"""Drive the real ``codex app-server`` against a loopback fake model.

A reusable harness for tests that need to establish, on the installed Codex
CLI rather than a mock, the contracts that SuCoder's managed Codex state
relies on: the rollout file under ``CODEX_HOME/sessions`` is the authoritative
history, the SQLite stores under ``sqlite_home`` are a projection of it, and
restore means "databases from a backup plus the rollout at the absolute path
the database recorded".

Everything here runs under private ``HOME``, ``CODEX_HOME`` and ``TMPDIR``
directories; nothing touches the real account.  The launch options are the
ones Slice 1's inline test used (``model_provider="offline"`` pointing at the
fake model's ``responses`` wire API), with the spike's request-compression
switch added so the fake model can parse every request body.

The module is a helper, not a test: pytest collects ``test_*.py`` only, so it
is importable as ``from tests.codex_app_server import ...``.  It is never
staged to a node, so it may use anything the repository's Python offers.
"""

from __future__ import annotations

import gzip
import json
import os
import queue
import shutil
import signal
import sqlite3
import subprocess
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import IO, Any, Callable, Dict, List, Optional, Sequence, Tuple

from sucoder.agent_mode import config_override

#: The SQLite stores Codex 0.162.1 keeps under ``sqlite_home``.
DATABASES = ("state_5.sqlite", "thread_history_1.sqlite", "goals_1.sqlite",
             "queue_1.sqlite", "memories_1.sqlite", "logs_2.sqlite")

#: Feature switches that keep the server from reaching for the network or
#: plugin stores; the same set Slice 1's inline test used, plus the spike's
#: request-compression switch so the fake model receives plain JSON bodies.
QUIET_OPTIONS = ["-c", "features.plugins=false", "-c", "features.remote_plugin=false",
                 "-c", "features.apps=false", "-c", "features.enable_request_compression=false"]

JsonDict = Dict[str, Any]


def codex_path() -> Optional[str]:
    return shutil.which("codex")


def codex_available(need_bwrap: bool = False) -> bool:
    """A predicate for ``skipif``: codex on PATH, and bwrap too when asked."""
    if codex_path() is None:
        return False
    return not need_bwrap or os.access("/usr/bin/bwrap", os.X_OK)


def model_options(port: int, sqlite_home: Optional[Path] = None) -> List[str]:
    """The ``-c`` overrides that point Codex at the fake model (and sqlite_home)."""
    options = ["-c", 'model_provider="offline"', "-c", 'model="fake"', *QUIET_OPTIONS,
               "-c", ('model_providers.offline={name="offline",'
                      f'base_url="http://127.0.0.1:{port}",wire_api="responses"}}')]
    if sqlite_home is not None:
        options += ["-c", config_override("sqlite_home", str(sqlite_home))]
    return options


# ----------------------------------------------------------------- fake model


class FakeModel:
    """A Responses-wire endpoint on 127.0.0.1 that records every request.

    Each ``POST`` body is parsed as JSON and appended to ``requests`` (under
    ``lock``).  The reply is a ``response.completed`` event whose output is
    empty by default, or one assistant message carrying ``reply_text``.
    """

    def __init__(self, reply_text: str = "", exec_on: Optional[str] = None, exec_cmd: str = "pwd") -> None:
        self.reply_text = reply_text
        #: When the newest input item is a user message containing this text,
        #: the reply is one ``exec_command`` call running ``exec_cmd`` instead;
        #: Codex runs it in the turn's cwd and sends its output back in the
        #: next request (``tool_outputs``), which gets the plain reply.
        self.exec_on = exec_on
        self.exec_cmd = exec_cmd
        self.requests: List[JsonDict] = []
        self.lock = threading.Lock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("FakeModel is not started")
        return int(self._server.server_port)

    def start(self) -> "FakeModel":
        model = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                model._record(raw, self.headers.get("Content-Encoding", ""), self.path)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(model._sse().encode())

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None

    def __enter__(self) -> "FakeModel":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def _record(self, raw: bytes, encoding: str, path: str) -> None:
        body: JsonDict
        try:
            if encoding == "gzip":
                raw = gzip.decompress(raw)
            elif encoding == "deflate":
                raw = zlib.decompress(raw)
            body = json.loads(raw)
            if not isinstance(body, dict):
                body = {"_value": body}
        except (ValueError, OSError, zlib.error):
            body = {"_raw": raw[:200].decode("utf-8", "replace"), "_encoding": encoding}
        body["_path"] = path
        body["_t"] = time.monotonic()
        with self.lock:
            self.requests.append(body)

    def _wants_exec(self) -> bool:
        if not self.exec_on:
            return False
        with self.lock:
            last = (self.requests[-1].get("input") or [None])[-1] if self.requests else None
        if not isinstance(last, dict) or last.get("role") != "user":
            return False
        content = last.get("content")
        texts = [content] if isinstance(content, str) else [str(p.get("text", "")) for p in content or [] if isinstance(p, dict)]
        return any(self.exec_on in t for t in texts)

    def _sse(self) -> str:
        output: List[JsonDict] = []
        events: List[JsonDict] = []
        if self._wants_exec():
            n = self.count()
            item = {"type": "function_call", "id": f"fc_{n}", "call_id": f"call_{n}", "name": "exec_command",
                    "arguments": json.dumps({"cmd": self.exec_cmd, "login": False}), "status": "completed"}
            output.append(item)
            events.append({"type": "response.output_item.done", "output_index": 0, "item": item})
        elif self.reply_text:
            item = {"type": "message", "id": "msg_fake", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": self.reply_text, "annotations": []}]}
            output.append(item)
            events.append({"type": "response.output_item.done", "output_index": 0, "item": item})
        events.append({"type": "response.completed", "response": {
            "id": "resp_fake", "object": "response", "status": "completed", "output": output,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}})
        return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)

    # -- inspection

    def count(self) -> int:
        with self.lock:
            return len(self.requests)

    def requests_since(self, index: int) -> List[JsonDict]:
        with self.lock:
            return list(self.requests[index:])

    def wait_for_requests(self, index: int, n: int = 1, timeout: float = 10) -> List[JsonDict]:
        """Bounded wait until at least ``n`` requests arrived after ``index``."""
        deadline = time.monotonic() + timeout
        while True:
            seen = self.requests_since(index)
            if len(seen) >= n or time.monotonic() >= deadline:
                return seen
            time.sleep(0.05)

    @staticmethod
    def tool_outputs(request: JsonDict) -> List[str]:
        """The ``function_call_output`` texts a request carries back to the model."""
        return [str(item.get("output")) for item in request.get("input") or []
                if isinstance(item, dict) and item.get("type") == "function_call_output"]

    @staticmethod
    def environment_cwd(request: JsonDict) -> Optional[str]:
        """The ``<cwd>`` of the newest ``<environment_context>`` in a request."""
        found = None
        for text in FakeModel.user_texts(request):
            if "<environment_context>" in text and "<cwd>" in text:
                found = text.split("<cwd>", 1)[1].split("</cwd>", 1)[0]
        return found

    @staticmethod
    def user_texts(request: JsonDict) -> List[str]:
        """The user-visible text of every user message in a request's input.

        Codex adds its own user-role items (environment context and the
        like), so callers assert containment of distinctive markers rather
        than equality.
        """
        texts: List[str] = []
        for item in request.get("input") or []:
            if not isinstance(item, dict) or item.get("role") != "user":
                continue
            content = item.get("content")
            if isinstance(content, str):
                texts.append(content)
                continue
            for part in content or []:
                if isinstance(part, dict) and part.get("type") == "input_text":
                    texts.append(str(part.get("text", "")))
        return texts


# ----------------------------------------------------------------- homes


class Home:
    """Private ``CODEX_HOME``, ``HOME`` and ``TMPDIR`` under one root.

    ``codex_home`` may be given explicitly so a test can place it at a path of
    its choosing (one with a symlink component, or the exact absolute path an
    earlier home used).
    """

    def __init__(self, root: Path, codex_home: Optional[Path] = None, trusted: Sequence[Path] = ()) -> None:
        self.root = Path(root)
        self.codex_home = Path(codex_home) if codex_home is not None else self.root / "codex-home"
        self.user_home = self.root / "user-home"
        self.tmp = self.root / "tmp"
        for d in (self.root, self.codex_home, self.user_home, self.tmp):
            d.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.trusted = [Path(p) for p in trusted]
        self.write_config()

    def write_config(self, extra: Sequence[str] = ()) -> None:
        lines = list(extra)
        for w in self.trusted:
            lines += [f'[projects."{w}"]', 'trust_level = "trusted"']
        (self.codex_home / "config.toml").write_text("\n".join(lines) + "\n")

    def env(self, **extra: str) -> Dict[str, str]:
        env = {"PATH": os.environ["PATH"], "HOME": str(self.user_home), "CODEX_HOME": str(self.codex_home),
               "TMPDIR": str(self.tmp), "NO_PROXY": "127.0.0.1", "RUST_LOG": "error",
               "TERM": "xterm-256color", "LANG": "C.UTF-8"}
        env.update(extra)
        return env

    def rollouts(self, thread_id: Optional[str] = None) -> List[Path]:
        sessions = self.codex_home / "sessions"
        pattern = f"rollout-*-{thread_id}.jsonl" if thread_id else "rollout-*.jsonl"
        return sorted(sessions.rglob(pattern)) if sessions.exists() else []


# ----------------------------------------------------------------- the server


class TurnResult:
    def __init__(self, turn_id: Optional[str], completed: Optional[JsonDict], seconds: float, requests_before: int) -> None:
        self.turn_id = turn_id
        self.completed = completed
        self.seconds = seconds
        #: ``FakeModel.count()`` just before ``turn/start``; the model requests
        #: this turn caused are ``model.requests_since(requests_before)``.
        self.requests_before = requests_before

    @property
    def ok(self) -> bool:
        return self.completed is not None


class StopResult:
    def __init__(self, steps: List[str], exit_code: Optional[int], seconds: float) -> None:
        self.steps = steps
        self.exit_code = exit_code
        self.seconds = seconds

    @property
    def clean(self) -> bool:
        return self.steps == ["stdin-close"] and self.exit_code == 0


class AppServer:
    """One ``codex app-server`` on stdio, in a private home, against ``model``.

    ``argv = [*prefix, codex, *options, "app-server"]`` where ``options`` points
    at the fake model and, when ``sqlite_home`` is given, carries
    ``-c sqlite_home="<abs>"`` (encoded with ``config_override``).  A reader
    thread parses stdout: replies go to a queue keyed by id, everything else
    is appended to ``notifications`` under a condition variable so no
    notification is lost between waits.  Every wait is bounded.
    """

    def __init__(self, home: Home, model: FakeModel, cwd: Path, sqlite_home: Optional[Path] = None,
                 prefix: Sequence[str] = (), extra_options: Sequence[str] = (), codex: Optional[str] = None,
                 label: str = "app-server") -> None:
        self.home = home
        self.model = model
        self.cwd = Path(cwd)
        self.sqlite_home = Path(sqlite_home) if sqlite_home is not None else None
        if self.sqlite_home is not None:
            self.sqlite_home.mkdir(parents=True, mode=0o700, exist_ok=True)
        binary = codex or codex_path()
        if binary is None:
            raise RuntimeError("codex is not on PATH")
        self.options = model_options(model.port, self.sqlite_home) + list(extra_options)
        self.argv = [*prefix, binary, *self.options, "app-server"]
        self.codex = binary
        self.label = label
        self.proc: Optional[subprocess.Popen[str]] = None
        self.stdin: Optional[IO[str]] = None
        self.notifications: List[JsonDict] = []
        self.cond = threading.Condition()
        self.replies: "queue.Queue[JsonDict]" = queue.Queue()
        self.next_id = 100
        self.stderr_path = home.root / f"{label}.stderr.log"
        self._stderr: Optional[IO[str]] = None
        self._reader: Optional[threading.Thread] = None

    # -- lifecycle

    @property
    def store_dir(self) -> Path:
        """Where the SQLite stores live: ``sqlite_home`` or ``CODEX_HOME``."""
        return self.sqlite_home if self.sqlite_home is not None else self.home.codex_home

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> "AppServer":
        if self.proc is not None:
            raise RuntimeError("already started")
        self._stderr = open(self.stderr_path, "a")
        self.proc = subprocess.Popen(self.argv, cwd=str(self.cwd), env=self.home.env(), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self._stderr, text=True, start_new_session=True)
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.stdin = self.proc.stdin
        self._reader = threading.Thread(target=self._read, args=(self.proc.stdout,), daemon=True)
        self._reader.start()
        return self

    def initialize(self, timeout: float = 60) -> JsonDict:
        result = self.rpc("initialize", {"clientInfo": {"name": "sucoder_test", "version": "1"},
                                         "capabilities": {"experimentalApi": True}}, timeout=timeout)
        self._send({"method": "initialized", "params": {}})
        return result

    def __enter__(self) -> "AppServer":
        self.start()
        self.initialize()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.running():
            result = self.stop()
            if result.exit_code is None:
                self.kill()
        self._close()

    def _read(self, stdout: IO[str]) -> None:
        for line in stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue
            if "id" in msg and ("result" in msg or "error" in msg):
                self.replies.put(msg)
            else:
                msg["_t"] = time.monotonic()
                with self.cond:
                    self.notifications.append(msg)
                    self.cond.notify_all()
        with self.cond:
            self.cond.notify_all()

    def _send(self, obj: JsonDict) -> None:
        if self.stdin is None:
            raise RuntimeError("server is not started")
        self.stdin.write(json.dumps(obj) + "\n")
        self.stdin.flush()

    def rpc(self, method: str, params: JsonDict, timeout: float = 60) -> JsonDict:
        """One request; its ``result`` or ``RuntimeError`` on ``error``/exit."""
        if self.proc is None:
            raise RuntimeError("server is not started")
        self.next_id += 1
        rid = self.next_id
        self._send({"id": rid, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
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
                result = msg.get("result")
                return result if isinstance(result, dict) else {"_value": result}

    def wait_for(self, predicate: Callable[[JsonDict], bool], timeout: float, start_index: int = 0) -> Optional[JsonDict]:
        """The first notification at or after ``start_index`` matching ``predicate``, or None."""
        deadline = time.monotonic() + timeout
        with self.cond:
            while True:
                for n in self.notifications[start_index:]:
                    if predicate(n):
                        return n
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self.proc is None or self.proc.poll() is not None:
                    return None
                self.cond.wait(min(remaining, 0.2))

    def methods_since(self, index: int) -> List[str]:
        with self.cond:
            return [str(n.get("method")) for n in self.notifications[index:]]

    # -- protocol

    def thread_start(self, cwd: Optional[Path] = None, timeout: float = 60) -> str:
        result = self.rpc("thread/start", {"cwd": str(cwd or self.cwd)}, timeout=timeout)
        return str(result["thread"]["id"])

    def turn(self, thread_id: str, text: str, timeout: float = 60) -> TurnResult:
        """Send one user message and wait for its ``turn/completed``."""
        index = len(self.notifications)
        before = self.model.count()
        t0 = time.monotonic()
        started = self.rpc("turn/start", {"threadId": thread_id, "input": [{"type": "text", "text": text}]}, timeout=timeout)
        turn_id = (started.get("turn") or {}).get("id")

        def done(n: JsonDict) -> bool:
            if n.get("method") != "turn/completed":
                return False
            params = n.get("params") or {}
            if params.get("threadId", thread_id) != thread_id:
                return False
            seen = (params.get("turn") or {}).get("id", params.get("turnId"))
            return turn_id is None or seen is None or seen == turn_id

        completed = self.wait_for(done, timeout, index)
        return TurnResult(turn_id, completed, time.monotonic() - t0, before)

    def thread_resume(self, thread_id: str, cwd: Optional[Path] = None, timeout: float = 60) -> JsonDict:
        params: JsonDict = {"threadId": thread_id}
        if cwd is not None:
            params["cwd"] = str(cwd)
        return self.rpc("thread/resume", params, timeout=timeout)

    def thread_read(self, thread_id: str, include_turns: bool = True, timeout: float = 60) -> JsonDict:
        return self.rpc("thread/read", {"threadId": thread_id, "includeTurns": include_turns}, timeout=timeout)

    def items_list(self, thread_id: str, limit: int = 500, timeout: float = 60) -> List[JsonDict]:
        return list(self.rpc("thread/items/list", {"threadId": thread_id, "limit": limit}, timeout=timeout).get("data", []))

    def turns_list(self, thread_id: str, limit: int = 500, timeout: float = 60) -> List[JsonDict]:
        return list(self.rpc("thread/turns/list", {"threadId": thread_id, "limit": limit}, timeout=timeout).get("data", []))

    def goal_set(self, thread_id: str, objective: str, status: str = "active", timeout: float = 60) -> JsonDict:
        """``thread/goal/set`` with an objective; the spike's exact parameters."""
        return self.rpc("thread/goal/set", {"threadId": thread_id, "objective": objective, "status": status,
                                            "origin": "user"}, timeout=timeout)

    def goal_pause(self, thread_id: str, timeout: float = 60) -> JsonDict:
        """``thread/goal/set`` with ``status="paused"`` and no objective: the hold."""
        return self.rpc("thread/goal/set", {"threadId": thread_id, "status": "paused", "origin": "user"}, timeout=timeout)

    def goal_get(self, thread_id: str, timeout: float = 60) -> Optional[JsonDict]:
        goal = self.rpc("thread/goal/get", {"threadId": thread_id}, timeout=timeout).get("goal")
        return goal if isinstance(goal, dict) else None

    def queue_message(self, thread_id: str, text: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
        """Queue a message for ``thread_id`` with the ``codex queue`` CLI.

        The app-server protocol of this build has no client request that
        queues (only the ``thread/queue/changed`` notification), so this runs
        ``codex -c sqlite_home=... queue --thread ID --message TEXT`` against
        the same stores, and requires the server to be stopped first: a
        running server would hold the thread and may consume the row at once.
        """
        if self.running():
            raise RuntimeError("stop the server before queueing; the CLI writes the stores directly")
        argv = [self.codex, *self.options, "queue", "--thread", thread_id, "--message", text]
        return subprocess.run(argv, cwd=str(self.cwd), env=self.home.env(), capture_output=True, text=True,
                              timeout=timeout, check=False)

    # -- stopping

    def pids(self) -> List[int]:
        """Live descendant PIDs whose comm is ``codex`` (the server and any helpers)."""
        if self.proc is None:
            return []
        found: List[int] = []
        for pid in [self.proc.pid, *_descendants(self.proc.pid)]:
            try:
                comm = Path(f"/proc/{pid}/comm").read_text().strip()
            except OSError:
                continue
            if comm == "codex":
                found.append(pid)
        return found

    def stop(self, timeout: float = 15) -> StopResult:
        """Close stdin and wait; SIGTERM then SIGKILL the group on timeout."""
        if self.proc is None:
            return StopResult([], None, 0.0)
        t0 = time.monotonic()
        steps = ["stdin-close"]
        try:
            if self.stdin is not None:
                self.stdin.close()
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
        return StopResult(steps, self.proc.returncode, time.monotonic() - t0)

    def kill(self) -> StopResult:
        if self.proc is None:
            return StopResult([], None, 0.0)
        t0 = time.monotonic()
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        self._close()
        return StopResult(["SIGKILL"], self.proc.returncode, time.monotonic() - t0)

    def _close(self) -> None:
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    def stderr_text(self) -> str:
        try:
            return self.stderr_path.read_text()
        except OSError:
            return ""


def _descendants(root_pid: int) -> List[int]:
    children: Dict[int, List[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        ppid = int(stat.rsplit(")", 1)[1].split()[1])
        children.setdefault(ppid, []).append(int(entry.name))
    found: List[int] = []
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        for child in children.get(pid, []):
            found.append(child)
            stack.append(child)
    return found


# ----------------------------------------------------------------- files and stores


def rollout_of(codex_home: Path, thread_id: str) -> Path:
    """The one ``sessions/**/rollout-*-<thread_id>.jsonl`` under ``codex_home``."""
    matches = sorted((Path(codex_home) / "sessions").rglob(f"rollout-*-{thread_id}.jsonl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one rollout for {thread_id} under {codex_home}, found {matches}")
    return matches[0]


def _connect_ro(path: Path) -> sqlite3.Connection:
    """Read-only when the WAL sidecars allow it; a plain open otherwise."""
    try:
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=5)
        conn.execute("select count(*) from sqlite_master").fetchone()
        return conn
    except sqlite3.Error:
        return sqlite3.connect(str(path), timeout=5)


def _rows(path: Path, sql: str, params: Tuple[Any, ...] = ()) -> List[JsonDict]:
    conn = _connect_ro(Path(path))
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


class Projection:
    """``thread_history_projection_state`` plus the thread's ``thread_turns`` count."""

    def __init__(self, next_rollout_byte_offset: Optional[int], next_rollout_ordinal: Optional[int], thread_turns: int) -> None:
        self.next_rollout_byte_offset = next_rollout_byte_offset
        self.next_rollout_ordinal = next_rollout_ordinal
        self.thread_turns = thread_turns

    def as_tuple(self) -> Tuple[Optional[int], Optional[int], int]:
        return (self.next_rollout_byte_offset, self.next_rollout_ordinal, self.thread_turns)

    def __repr__(self) -> str:
        return f"Projection(offset={self.next_rollout_byte_offset}, ordinal={self.next_rollout_ordinal}, turns={self.thread_turns})"


def projection(sqlite_home: Path, thread_id: str) -> Projection:
    path = Path(sqlite_home) / "thread_history_1.sqlite"
    state = _rows(path, "select next_rollout_byte_offset, next_rollout_ordinal from thread_history_projection_state "
                        "where thread_id = ?", (thread_id,))
    turns = _rows(path, "select count(*) as n from thread_turns where thread_id = ?", (thread_id,))
    if not state:
        return Projection(None, None, int(turns[0]["n"]))
    return Projection(int(state[0]["next_rollout_byte_offset"]), int(state[0]["next_rollout_ordinal"]), int(turns[0]["n"]))


def settled_projection(sqlite_home: Path, thread_id: str, timeout: float = 3.0, quiet: float = 0.3) -> Projection:
    """The projection once it has stopped changing for ``quiet`` seconds (bounded)."""
    deadline = time.monotonic() + timeout
    last = projection(sqlite_home, thread_id)
    stable_since = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(0.05)
        current = projection(sqlite_home, thread_id)
        if current.as_tuple() != last.as_tuple():
            last, stable_since = current, time.monotonic()
        elif time.monotonic() - stable_since >= quiet:
            break
    return last


def thread_row(sqlite_home: Path, thread_id: str) -> Optional[JsonDict]:
    rows = _rows(Path(sqlite_home) / "state_5.sqlite", "select * from threads where id = ?", (thread_id,))
    return rows[0] if rows else None


def goal_rows(sqlite_home: Path) -> List[JsonDict]:
    return _rows(Path(sqlite_home) / "goals_1.sqlite", "select * from thread_goals order by thread_id")


def queue_rows(sqlite_home: Path) -> List[JsonDict]:
    return _rows(Path(sqlite_home) / "queue_1.sqlite", "select * from queued_items order by thread_id, queue_order")


def quick_check_all(directory: Path) -> Dict[str, str]:
    """``pragma quick_check`` of every ``*.sqlite`` in ``directory``, opened read-only."""
    out: Dict[str, str] = {}
    for path in sorted(Path(directory).glob("*.sqlite")):
        try:
            conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=5)
            try:
                out[path.name] = str(conn.execute("pragma quick_check").fetchall()[0][0])
            finally:
                conn.close()
        except sqlite3.Error as error:
            out[path.name] = f"error: {error}"
    return out


def backup_databases(src_dir: Path, dst_dir: Path, pages: int = 64, sleep: float = 0.005) -> Dict[str, str]:
    """Online-backup every ``*.sqlite`` of ``src_dir`` into ``dst_dir``.

    Uses ``Connection.backup`` (reads through the WAL, never copies live
    files), switches each destination to ``journal_mode=DELETE`` after the
    copy (the backup carries the WAL header with page 1), runs ``quick_check``
    and returns its verdict per database.  Leaves no ``-wal``/``-shm``
    sidecars in ``dst_dir``.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    verdicts: Dict[str, str] = {}
    for path in sorted(src_dir.glob("*.sqlite")):
        target = dst_dir / path.name
        src = _connect_ro(path)
        dst = sqlite3.connect(str(target), timeout=5)
        try:
            src.backup(dst, pages=pages, sleep=sleep)
            dst.execute("pragma journal_mode=DELETE").fetchone()
            verdicts[path.name] = str(dst.execute("pragma quick_check").fetchall()[0][0])
        finally:
            dst.close()
            src.close()
        os.chmod(target, 0o600)
    sidecars = [p.name for p in dst_dir.iterdir() if p.name.endswith(("-wal", "-shm", "-journal"))]
    if sidecars:
        raise RuntimeError(f"backup left sidecars in {dst_dir}: {sidecars}")
    return verdicts
