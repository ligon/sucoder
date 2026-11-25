"""Bounded, isolated native role transport; the coordinator owns all artifacts.

Implementation ledger sections 2, 4 and 6: dispatch bypass modes are unsuitable.
This inline transport has no evidence file tools and no access to the real home
or repository. Linux Landlock confines the entire native process tree.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import platform
import selectors
import shutil
import signal
import stat
import subprocess
import tempfile
import time

from .dream import DreamError, _json, canonical, digest


MAX_TRACE = 4 * 1024 * 1024
READ = (1 << 0) | (1 << 2) | (1 << 3)
WRITE = sum(1 << i for i in (1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))


class _Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class FilesystemBoundary:
    """Pre-open a Landlock ruleset, inherited only by the owned subprocess.

    Unlike namespace setup this needs no privileged service or mount changes.
    Unsupported kernels fail before any provider request.
    """
    def __init__(self, scratch):
        if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
            raise DreamError("Native dream execution requires Linux Landlock on this build", 3)
        self.libc = ctypes.CDLL(None, use_errno=True)
        abi = self.libc.syscall(444, 0, 0, 1)
        if abi < 3:
            raise DreamError("Native dream execution requires Landlock ABI 3 or newer", 3)
        ruleset = _Ruleset(READ | WRITE)
        self.fd = self.libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
        if self.fd < 0:
            raise DreamError("Unable to create native filesystem boundary", 3)
        try:
            for path in ("/usr", "/lib", "/lib64", "/etc/ssl", "/etc/resolv.conf", "/etc/hosts",
                         "/etc/nsswitch.conf", "/etc/gai.conf", "/etc/localtime", "/dev/null", "/dev/urandom"):
                if Path(path).exists():
                    self.allow(Path(path), READ)
            self.allow(Path(scratch), READ | WRITE)
        except BaseException:
            self.close()
            raise

    def allow(self, path, access):
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            if not stat.S_ISDIR(os.fstat(fd).st_mode):
                access &= ~((1 << 3) | sum(1 << i for i in range(4, 14)))
            rule = _PathRule(access, fd)
            if self.libc.syscall(445, self.fd, 1, ctypes.byref(rule), 0):
                raise DreamError("Unable to define native filesystem boundary", 3)
        finally:
            os.close(fd)

    def enter(self):
        if self.libc.prctl(38, 1, 0, 0, 0) or self.libc.syscall(446, self.fd, 0):
            os._exit(126)

    def close(self):
        os.close(self.fd)


def capture(argv, payload, cwd, env, deadline, boundary=None, limit=MAX_TRACE):
    """Byte-preserving stdin/stdout transport with deadlines and pipe limits.

    Reuses observation_process's selector/process-group pattern, adding the
    immutable stdin and filesystem restriction that observation capture lacks.
    """
    if time.monotonic() >= deadline:
        raise DreamError("Native role deadline exhausted", 3)
    with tempfile.TemporaryFile() as input_stream:
        input_stream.write(payload)
        input_stream.seek(0)
        try:
            child = subprocess.Popen(argv, cwd=cwd, env=env, stdin=input_stream,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                preexec_fn=boundary.enter if boundary else None,
                pass_fds=(boundary.fd,) if boundary else ())
        except (OSError, subprocess.SubprocessError):
            raise DreamError("Native harness could not start inside its boundary", 3) from None
        streams = [bytearray(), bytearray()]
        selector = selectors.DefaultSelector()
        try:
            for i, pipe in enumerate((child.stdout, child.stderr)):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, i)
            while selector.get_map() or child.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DreamError("Native role deadline exhausted", 3)
                for key, _ in selector.select(min(.05, remaining)):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                    else:
                        streams[key.data].extend(block)
                        if sum(map(len, streams)) > limit:
                            raise DreamError("Native role trace exceeds its byte allowance", 3)
            return child.returncode, bytes(streams[0]), bytes(streams[1])
        except DreamError as exc:
            setattr(exc, "partial_stdout", bytes(streams[0]))
            setattr(exc, "partial_stderr", bytes(streams[1]))
            raise
        finally:
            selector.close()
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.stdout.close()
            child.stderr.close()
            try:
                child.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                pass


def environment(scratch, harness):
    """Minimal authentication handoff, never copy settings, hooks or plugins."""
    scratch = Path(scratch)
    result = dict(PATH=os.environ.get("PATH", os.defpath), HOME=str(scratch),
                  LANG="C.UTF-8", TMPDIR=str(scratch))
    # Native authentication is the only inherited secret. No parent session,
    # relay/socket location or arbitrary environment reaches this subprocess.
    for name in ("ANTHROPIC_API_KEY",) if harness == "claude" else ("OPENAI_API_KEY",):
        if name in os.environ:
            result[name] = os.environ[name]
    original_home = Path.home()
    native_home = scratch / (".codex" if harness == "codex" else ".claude")
    native_home.mkdir(mode=0o700)
    original = Path(os.environ.get("CODEX_HOME", str(original_home / ".codex"))) if harness == "codex" else Path(os.environ.get("CLAUDE_CONFIG_DIR", str(original_home / ".claude")))
    filename = "auth.json" if harness == "codex" else ".credentials.json"
    try:
        fd = os.open(original / filename, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except FileNotFoundError:
        pass  # Environment authentication or a native actionable auth error.
    else:
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise DreamError("Native credential seed must be a regular file")
            data = stream.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise DreamError("Native credential seed exceeds its allowance")
        target = os.open(native_home / filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(target, "wb") as stream:
            stream.write(data)
    result["CODEX_HOME" if harness == "codex" else "CLAUDE_CONFIG_DIR"] = str(native_home)
    return result


def command(harness, executable, model, schema):
    if harness == "claude":
        return [executable, "--print", "--restricted", "--safe-mode", "--tools", "",
                "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--setting-sources", "", "--settings", '{"disableAllHooks":true}',
                "--permission-mode", "dontAsk", "--permission-prompts", "none",
                "--disable-slash-commands", "--no-chrome", "--no-session-persistence",
                "--output-format", "stream-json", "--verbose", "--model", model,
                "--json-schema", canonical(schema).decode()]
    return [executable, "exec", "--ignore-user-config", "--ignore-rules", "--strict-config",
            "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "--json",
            "--model", model, "--output-schema", "result.schema.json",
            "-c", 'approval_policy="never"', "-c", 'web_search="disabled"',
            "-c", "project_doc_max_bytes=0", "-c", 'developer_instructions="Use only the inline assignment. Do not call tools."',
            "-c", "agents.enabled=false", "-c", "mcp_servers={}",
            *[item for feature in ("shell_tool", "view_image", "apps", "plugins", "hooks", "multi_agent",
                                  "browser_use", "computer_use", "image_generation", "code_mode", "shell_snapshot")
              for item in ("--disable", feature)], "-"]


def parse_trace(raw, harness):
    events = [_json(line) for line in raw.split(b"\n") if line.strip()]
    if any(not isinstance(event, dict) for event in events):
        raise DreamError("Unrecognized native trace record")
    final, usage, completed = None, None, False
    for event in events:
        if harness == "claude":
            if event.get("type") not in ("system", "assistant", "user", "result", "rate_limit_event"):
                raise DreamError("Unknown Claude native trace event")
            if event.get("type") == "system" and event.get("subtype") not in ("init", "ui_invalidate"):
                raise DreamError("Unexpected Claude system event, hook or compaction")
            if event.get("type") == "assistant":
                message = event.get("message")
                if not isinstance(message, dict) or not isinstance(message.get("content"), list):
                    raise DreamError("Invalid Claude assistant trace")
                blocks = message["content"]
                if any(not isinstance(block, dict) for block in blocks):
                    raise DreamError("Invalid Claude content block")
                if any(block.get("type") == "tool_use" and block.get("name") != "StructuredOutput" for block in blocks):
                    raise DreamError("Native role attempted a tool; inline execution forbids external tools")
            if event.get("type") == "system" and event.get("subtype") == "init":
                if set(event.get("tools", [])) - {"StructuredOutput"} or event.get("mcp_servers"):
                    raise DreamError("Native role exposed unexpected external tools")
            if event.get("type") == "result":
                if event.get("is_error") or event.get("subtype") != "success":
                    raise DreamError("Claude role failed; inspect its private native trace", 3)
                final = event.get("structured_output")
                usage = dict(native=event.get("usage"), currency_usd=event.get("total_cost_usd"))
                completed = True
        else:
            if event.get("type") not in ("thread.started", "turn.started", "item.started", "item.completed",
                                         "item.updated", "turn.completed", "error", "turn.failed"):
                raise DreamError("Unknown Codex native trace event")
            if event.get("type") in ("item.started", "item.completed", "item.updated"):
                item = event.get("item", {})
                if not isinstance(item, dict):
                    raise DreamError("Invalid Codex item trace")
                if item.get("type") not in ("agent_message", "reasoning", "todo_list"):
                    raise DreamError("Native role attempted external tools or returned an unknown item")
                if item.get("type") == "agent_message":
                    final = item.get("text", "")
            if event.get("type") in ("error", "turn.failed"):
                raise DreamError("Codex role failed; inspect its private native trace", 3)
            if event.get("type") == "turn.completed":
                usage = dict(native=event.get("usage"), currency_usd=None)
                completed = True
    if completed and harness == "codex" and isinstance(final, str):
        final = _json(final.encode())
    if not completed or not isinstance(final, dict):
        raise DreamError("Native trace has no completed structured role result", 3)
    return final, usage


def invoke(harness, model, assignment, schema, deadline, *, expected_version=None):
    if harness not in ("codex", "claude"):
        raise DreamError("Unsupported native dream harness")
    executable = shutil.which(harness)
    if executable is None:
        raise DreamError("Requested native harness is not installed: " + harness, 3)
    with tempfile.TemporaryDirectory(prefix="sucoder-dream-role-") as scratch:
        env = environment(scratch, harness)
        boundary = FilesystemBoundary(scratch)
        try:
            boundary.allow(Path(executable).resolve(), READ)
            code, help_text, _ = capture([executable, *( ["exec", "--help"] if harness == "codex" else ["--help"])],
                                         b"", scratch, env, min(deadline, time.monotonic() + 10), boundary)
            required = (["--ignore-user-config", "--output-schema", "--ephemeral", "--ignore-rules"] if harness == "codex" else
                        ["--restricted", "--safe-mode", "--json-schema", "--tools", "--permission-prompts"])
            if code or any(flag.encode() not in help_text for flag in required):
                raise DreamError("Installed native harness lacks required isolation/schema capabilities", 3)
            code, version, _ = capture([executable, "--version"], b"", scratch, env,
                                       min(deadline, time.monotonic() + 10), boundary)
            if code:
                raise DreamError("Native harness version probe failed", 3)
            if expected_version is not None and version.decode().strip() != expected_version:
                raise DreamError("Native harness version changed since the run was frozen", 3)
            Path(scratch, "result.schema.json").write_bytes(canonical(schema))
            payload = canonical(assignment)
            failure = None
            try:
                code, stdout, stderr = capture(command(harness, executable, model, schema), payload,
                                                scratch, env, deadline, boundary)
            except DreamError as exc:
                code = None
                stdout = getattr(exc, "partial_stdout", b"")
                stderr = getattr(exc, "partial_stderr", b"")
                failure = str(exc)
            trace = dict(harness=harness, model=model, version=version.decode().strip(),
                         input_sha256=digest(payload), stdout=stdout.decode("utf-8"),
                         stderr=stderr.decode("utf-8"), returncode=code,
                         boundary="linux-landlock-inline-v1")
            if failure is not None:
                trace["validation_error"] = failure
                return None, None, trace
            if code:
                # Return failure traces to the coordinator for private storage.
                return None, None, trace
            try:
                result, usage = parse_trace(stdout, harness)
            except DreamError as exc:
                trace["validation_error"] = str(exc)
                return None, None, trace
            return result, usage, trace
        finally:
            boundary.close()


class Transport:
    """Freeze the actual native build before any semantic role assignment."""
    def __init__(self, harness, deadline):
        self.harness = harness
        executable = shutil.which(harness)
        if executable is None:
            raise DreamError("Requested native harness is not installed: " + harness, 3)
        with tempfile.TemporaryDirectory(prefix="sucoder-dream-probe-") as scratch:
            boundary = FilesystemBoundary(scratch)
            try:
                boundary.allow(Path(executable).resolve(), READ)
                env = dict(PATH=os.environ.get("PATH", os.defpath), HOME=scratch, TMPDIR=scratch,
                           CODEX_HOME=str(Path(scratch) / ".codex"), CLAUDE_CONFIG_DIR=str(Path(scratch) / ".claude"))
                code, version, _ = capture([executable, "--version"], b"", scratch, env,
                                           min(deadline, time.monotonic() + 10), boundary)
                if code or not version.strip():
                    raise DreamError("Native harness build probe failed", 3)
                self.version = version.decode().strip()
            finally:
                boundary.close()

    def __call__(self, harness, model, assignment, schema, deadline):
        if harness != self.harness:
            raise DreamError("Frozen native transport cannot change harness")
        return invoke(harness, model, assignment, schema, deadline, expected_version=self.version)
