"""Offline native request certification; synthetic credentials and local HTTP only."""
from __future__ import annotations

import ctypes
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import struct
import tempfile
import threading
import time

from . import dream_native
from .dream import DreamError, canonical, digest


class _NetworkRuleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]


class _PortRule(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("port", ctypes.c_uint64)]


class Boundary(dream_native.FilesystemBoundary):
    """Keep normal file confinement and deny TCP connects except fixture port."""
    def __init__(self, scratch, port):
        super().__init__(scratch)
        self.network_fd = None
        try:
            if self.libc.syscall(444, 0, 0, 1) < 4:
                raise DreamError("Offline native certification requires Landlock TCP restrictions", 3)
            rules = _NetworkRuleset(0, 2)  # LANDLOCK_ACCESS_NET_CONNECT_TCP
            fd = self.libc.syscall(444, ctypes.byref(rules), ctypes.sizeof(rules), 0)
            if fd < 0:
                raise DreamError("Unable to create offline native network boundary", 3)
            self.network_fd = fd
            rule = _PortRule(2, port)
            if self.libc.syscall(445, fd, 2, ctypes.byref(rule), 0):
                raise DreamError("Unable to bind offline native network fixture port", 3)
        except BaseException:
            self.close()
            raise

    def enter(self):
        super().enter()
        if self.libc.syscall(446, self.network_fd, 0):
            os._exit(125)

    def close(self):
        if self.network_fd is not None:
            os.close(self.network_fd)
            self.network_fd = None
        super().close()


class RecordingServer:
    """Minimal bounded Responses/Messages fixtures, including Responses WS."""
    def __init__(self):
        self.requests = []
        self.result = dict(answer="fixture")
        self.maximum_requests = 8
        self.request_bytes = 0
        self.lock = threading.Lock()
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                self.connection.settimeout(5)

            def log_message(self, *_args):
                pass

            def do_GET(self):
                if self.headers.get("Upgrade", "").lower() != "websocket":
                    self.send_error(404)
                    return
                key = self.headers.get("Sec-WebSocket-Key", "")
                accept = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
                self.send_response(101)
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept)
                self.end_headers()
                self.connection.settimeout(5)
                chunks = bytearray()
                try:
                    while True:
                        header = self.rfile.read(2)
                        if len(header) != 2:
                            return
                        opcode, size = header[0] & 15, header[1] & 127
                        if opcode == 8:
                            return
                        if size == 126:
                            size = struct.unpack("!H", self.rfile.read(2))[0]
                        elif size == 127:
                            size = struct.unpack("!Q", self.rfile.read(8))[0]
                        if size > 2097152 or len(chunks) + size > 2097152:
                            return
                        mask = self.rfile.read(4) if header[1] & 128 else b""
                        raw = self.rfile.read(size)
                        if len(raw) != size:
                            return
                        if mask:
                            raw = bytes(byte ^ mask[index % 4] for index, byte in enumerate(raw))
                        if opcode in (1, 0):
                            chunks.extend(raw)
                            if header[0] & 128:
                                body = fixture.record(bytes(chunks), "", "websocket")
                                for event in fixture.responses(body):
                                    data = json.dumps(event).encode()
                                    self.wfile.write(bytes((129, len(data))) if len(data) < 126 else b"\x81\x7e" + struct.pack("!H", len(data)))
                                    self.wfile.write(data)
                                self.wfile.flush()
                                return
                except (OSError, ValueError, struct.error):
                    return

            def do_POST(self):
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2097152:
                    self.send_error(413)
                    return
                body = fixture.record(self.rfile.read(size), self.headers.get("Content-Encoding", ""), "http")
                if "count_tokens" in self.path:
                    data = b'{"input_tokens":100}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                events = fixture.messages(body) if "messages" in self.path else fixture.responses(body)
                data = "".join("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n" for event in events).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
                self.close_connection = True

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    @property
    def port(self):
        return self.server.server_port

    def record(self, raw, encoding, transport):
        if encoding:
            raise ValueError("Compressed requests are unsupported by bounded offline certification")
        if len(raw) > 2097152:
            raise ValueError("Expanded fixture request exceeds allowance")
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("Fixture request must be an object")
        with self.lock:
            if len(self.requests) >= self.maximum_requests or self.request_bytes + len(raw) > 4194304:
                raise ValueError("Offline native request ceiling exhausted")
            self.request_bytes += len(raw)
            self.requests.append(dict(body=body, transport=transport))
        return body

    def responses(self, body):
        item = dict(type="message", id="msg_fixture", role="assistant", status="completed",
            content=[dict(type="output_text", text=canonical(self.result).decode(), annotations=[])])
        return [dict(type="response.output_item.done", output_index=0, item=item),
            dict(type="response.completed", response=dict(id="resp_fixture", object="response", status="completed",
                output=[item], usage=dict(input_tokens=100, input_tokens_details=dict(cached_tokens=30, cache_write_tokens=20),
                    output_tokens=5, total_tokens=105)))]

    def messages(self, body):
        structured = any(tool.get("name") == "StructuredOutput" for tool in body.get("tools", []))
        block = (dict(type="tool_use", id="tool_fixture", name="StructuredOutput", input={}) if structured else
                 dict(type="text", text=""))
        delta = (dict(type="input_json_delta", partial_json=canonical(self.result).decode()) if structured else
                 dict(type="text_delta", text=canonical(self.result).decode()))
        return [dict(type="message_start", message=dict(id="msg_fixture", type="message", role="assistant", content=[],
                    model=body.get("model", "fixture"), usage=dict(input_tokens=100, cache_read_input_tokens=30,
                        cache_creation_input_tokens=20, output_tokens=0,
                        cache_creation=dict(ephemeral_5m_input_tokens=20, ephemeral_1h_input_tokens=0)))),
            dict(type="content_block_start", index=0, content_block=block),
            dict(type="content_block_delta", index=0, delta=delta), dict(type="content_block_stop", index=0),
            dict(type="message_delta", delta=dict(stop_reason="tool_use" if structured else "end_turn", stop_sequence=None),
                usage=dict(output_tokens=5)), dict(type="message_stop")]


SCHEMA = dict(type="object", properties=dict(answer=dict(type="string")), required=["answer"], additionalProperties=False)


def fixture_result(schema):
    """A deterministic schema-valid synthetic result, with no semantic claim."""
    if "enum" in schema:
        return schema["enum"][0]
    if schema["type"] == "object":
        return {key: fixture_result(schema["properties"][key]) for key in schema["required"]}
    if schema["type"] == "array":
        return []
    if schema["type"] == "integer":
        return schema.get("minimum", 0)
    return "fixture"


def _jwt(payload):
    def part(value):
        return base64.urlsafe_b64encode(canonical(value)).rstrip(b"=").decode()
    return part(dict(alg="none")) + "." + part(payload) + ".fixture"


def fixture_environment(scratch, harness, route, port):
    """Never read/copy real authentication or parent configuration."""
    env = dict(PATH=os.environ.get("PATH", os.defpath), HOME=scratch, TMPDIR=scratch, LANG="C.UTF-8")
    home = Path(scratch) / (".codex" if harness == "codex" else ".claude")
    home.mkdir(mode=0o700)
    if harness == "codex":
        env["CODEX_HOME"] = str(home)
        if route == "environment-api-key":
            env["OPENAI_API_KEY"] = "offline-fixture-only"
            auth = dict(OPENAI_API_KEY="offline-fixture-only")
        else:
            token = _jwt(dict(exp=4102444800, **{"https://api.openai.com/auth": dict(
                chatgpt_account_id="fixture-account", chatgpt_plan_type="plus")}))
            auth = dict(auth_mode="chatgpt", tokens=dict(access_token=token, id_token=token,
                refresh_token="offline-fixture-only", account_id="fixture-account"))
        path = home / "auth.json"
    else:
        env.update(CLAUDE_CONFIG_DIR=str(home), ANTHROPIC_BASE_URL="http://127.0.0.1:" + str(port),
                   CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
        if route == "environment-api-key":
            env["ANTHROPIC_API_KEY"] = "offline-fixture-only"
            auth = {}
        else:
            auth = dict(claudeAiOauth=dict(accessToken="sk-ant-oat01-offline-fixture-only",
                refreshToken="offline-fixture-only", expiresAt=4102444800000, refreshTokenExpiresAt=4102444800000,
                scopes=["user:inference", "user:profile"], subscriptionType="max", rateLimitTier="default_claude_max_5x"))
        path = home / ".credentials.json"
    path.write_bytes(canonical(auth))
    path.chmod(0o600)
    return env


def _prefix(body, harness, policy):
    try:
        return _checked_prefix(body, harness, policy)
    except DreamError:
        raise
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise DreamError("Native request has an unsupported prefix shape", 3) from exc


def _checked_prefix(body, harness, policy):
    """Only explicitly cache-bearing surfaces; never normalize changing text."""
    if harness == "claude":
        tools = body.get("tools", [])
        if any(tool.get("name") != "StructuredOutput" for tool in tools):
            raise DreamError("Native request exposed an unexpected external tool", 3)
        blocks = body.get("system", [])
        if not isinstance(blocks, list):
            raise DreamError("Native request did not preserve a system block boundary", 3)
        end = next((index + 1 for index, block in enumerate(blocks)
                    if block.get("text") == policy and block.get("cache_control", {}).get("type") == "ephemeral"), None)
        if end is None:
            raise DreamError("Native policy has no observed cache boundary", 3)
        return dict(model=body.get("model"), system=blocks[:end], tools=tools,
            thinking=body.get("thinking"), output_config=body.get("output_config"))
    if body.get("tools") not in (None, []):
        raise DreamError("Native request exposed external tools despite the tool prohibition", 3)
    items = body.get("input", [])
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise DreamError("Native request did not preserve message boundaries", 3)
    for item in items:
        if item.get("type") == "additional_tools" and item.get("tools"):
            raise DreamError("Native request exposed additional tools despite the tool prohibition", 3)
    initial = []
    for item in items:
        if item.get("role") == "user":
            break
        # Response item IDs and transport metadata are not message text.
        initial.append({key: value for key, value in item.items()
                        if key not in ("id", "internal_chat_message_metadata_passthrough")})
    if not re.match(r"gpt-(?:5\.(?:[6-9]|[1-9][0-9])|[6-9](?:[.-]|$))", str(body.get("model", ""))):
        raise DreamError("Native model has no verified instruction-prefix cache eligibility rule", 3)
    options = body.get("prompt_cache_options") or {}
    if not isinstance(options, dict) or options.get("mode", "implicit") != "implicit" or options.get("prewarm"):
        raise DreamError("Native cache mode has no verified implicit instruction breakpoint", 3)
    if not initial or any(item.get("type", "message") != "message" or item.get("role") != "developer" for item in initial):
        raise DreamError("Native policy is outside the initial consecutive developer block", 3)
    if not any(policy == part.get("text") for item in initial
               for part in item.get("content", []) if isinstance(part, dict)):
        raise DreamError("Native request did not preserve the policy as an initial developer message", 3)
    return dict(model=body.get("model"), initial=initial, tools=body.get("tools"),
        instructions=body.get("instructions"), tool_choice=body.get("tool_choice"),
        parallel_tool_calls=body.get("parallel_tool_calls"), reasoning=body.get("reasoning"),
        text=body.get("text"), prompt_cache_key=body.get("prompt_cache_key"),
        prompt_cache_options=options, service_tier=body.get("service_tier"))


def certify(config, harness, model, *, seconds=120, route=None):
    """Two fresh invocations per variant, no real credentials or paid endpoint."""
    from . import dream, dream_prompt_native as native, dream_run, dream_usage
    if harness not in ("codex", "claude") or not isinstance(model, str) or not model.strip():
        raise DreamError("Certification requires explicit codex/claude and model", 3)
    if type(seconds) is not int or not 1 <= seconds <= 120:
        raise DreamError("Offline certification seconds must be between 1 and 120", 3)
    deadline = time.monotonic() + seconds
    build = native.resolve(harness, deadline)
    auth = native.authentication(harness)
    route = route or auth["route"]
    if route not in ("environment-api-key", "native-credentials"):
        raise DreamError("Select environment-api-key or native-credentials for synthetic certification", 3)
    cases = native.variants()
    observations, prefixes, failures, captures = [], [], [], []
    certified = []
    invocations = 0
    version = None
    with RecordingServer() as server:
        server.maximum_requests = len(cases) * 8
        for index in range(2 * len(cases)):
            case, step = cases[index // 2], index % 2
            policy, schema = case["policy"], case["schema"]
            server.result = fixture_result(schema)
            if time.monotonic() >= deadline:
                failures.append("Offline native certification deadline exhausted")
                break
            with tempfile.TemporaryDirectory(prefix="sucoder-cache-certify-") as scratch:
                env = fixture_environment(scratch, harness, route, server.port)
                env.update(build["environment"])
                boundary = Boundary(scratch, server.port)
                schema_fd = None
                before = len(server.requests)
                try:
                    boundary.allow(Path(build["native"]["path"]), dream_native.READ)
                    code, raw_version, _ = dream_native.capture([build["entry"]["path"], "--version"], b"", scratch,
                        env, min(deadline, time.monotonic() + 5), boundary, extra_fds=(boundary.network_fd,))
                    if code:
                        raise DreamError("Native certification version probe failed", 3)
                    current_version = raw_version.decode().strip()
                    if version is not None and current_version != version:
                        raise DreamError("Native build changed during certification", 3)
                    version = current_version
                    candidate = native.draft_profile(build, model, version,
                        dict(route=route, principal=None, account_continuity="unverified"))
                    schema_path = "unused"
                    if harness == "codex":
                        if step == 0 and build["launcher"] == "direct-vendor-native":
                            path = Path(scratch) / "reference.schema.json"
                            path.write_bytes(canonical(schema))
                            schema_path = str(path)
                        else:
                            schema_fd = native.sealed_schema(schema)
                            schema_path = "/proc/self/fd/" + str(schema_fd)
                    arguments = native.argv(candidate, policy, schema, schema_path)
                    if step == 0:
                        arguments[0] = build["entry"]["path"]
                    if harness == "codex":
                        overrides = ["-c", "openai_base_url=" + json.dumps("http://127.0.0.1:" + str(server.port)),
                            "-c", "chatgpt_base_url=" + json.dumps("http://127.0.0.1:" + str(server.port)),
                            "-c", "features.enable_request_compression=false"]
                        arguments = arguments[:-1] + overrides + arguments[-1:]
                    native.check_bounds(policy, arguments, env)
                    invocations += 1
                    code, stdout, stderr = dream_native.capture(arguments, canonical(dict(current=dict(
                        assignment_id=str(step), remaining_calls=2 - step), data="synthetic-data-" + str(step))),
                        scratch, env, min(deadline, time.monotonic() + 25), boundary,
                        extra_fds=(boundary.network_fd,) + (() if schema_fd is None else (schema_fd,)))
                    calls = server.requests[before:]
                    substantive = [row["body"] for row in calls if row["body"].get("generate") is not False and
                        ("input" in row["body"] if harness == "codex" else "messages" in row["body"])]
                    if code or not substantive:
                        raise DreamError("Native synthetic request failed: " + stderr.decode(errors="replace")[-500:], 3)
                    if len(substantive) != 1:
                        raise DreamError("Native certification requires exactly one substantive request per invocation", 3)
                    prefixes.append(_prefix(substantive[0], harness, policy))
                    result, usage = dream_native.parse_trace(stdout, harness)
                    if result != server.result:
                        raise DreamError("Native fixture output did not match its required schema", 3)
                    mapping = candidate["usage_mapping"]
                    normalized = dream_usage.normalize(usage, mapping)
                    if normalized["input_total"] != (100 if harness == "codex" else 150):
                        raise DreamError("Native usage mapping did not preserve the synthetic aggregate", 3)
                    if normalized["input_cache_read"] != 30 or normalized["input_cache_write"] != 20:
                        raise DreamError("Native cache usage fields did not preserve nonzero fixture counters", 3)
                    observations.append(dict(variant=case["name"], step=step, native_requests=len(calls), mapping=normalized,
                        raw_usage=usage,
                        transport=[row["transport"] for row in calls]))
                except (DreamError, OSError, ValueError) as exc:
                    failures.append(str(exc))
                finally:
                    captures.extend(server.requests[before:])
                    if schema_fd is not None:
                        os.close(schema_fd)
                    boundary.close()
            if step == 1:
                if len(prefixes) != 2 or canonical(prefixes[0]) != canonical(prefixes[1]):
                    failures.append("Two fresh native invocations did not expose an identical eligible prefix and cache scope")
                if failures:
                    break
                certified.append(dict(name=case["name"], policy_sha256=digest(policy.encode()),
                    schema_sha256=digest(canonical(schema)), prefix_sha256=digest(canonical(prefixes[0]))))
                prefixes = []
    if len(certified) != len(cases) and not failures:
        failures.append("Offline certification did not cover every production policy/schema variant")
    report = dict(contract="offline-native-cache-certification-v1", harness=harness, model=model,
        build=build, authentication_route=route, native_invocations=invocations, observations=observations,
        captures=captures,
        limits=dict(native_invocations=2 * len(cases), seconds=seconds), variants=certified,
        prefix_sha256=digest(canonical(certified)) if not failures else None,
        failures=failures, paid_inference=False, result="uncertified" if failures else "certified")
    if len(canonical(report)) + 1 > native.MAX_CERTIFICATION_BYTES:
        raise DreamError("Offline native certification report exceeds its private byte allowance", 3)
    output = native.directory(config)
    fd = dream._open_private_directory(output)
    try:
        name = "probe-" + digest(canonical(report)) + ".json"
        dream_run._store(fd, name, canonical(report) + b"\n")
        if failures:
            return dict(schema_version=1, operation="dream_cache_certify", state="uncertified", report_path=str(output / name),
                        failures=failures, paid_inference=False)
        candidate = native.draft_profile(build, model, version,
            dict(route=route, principal=None, account_continuity="unverified"))
        candidate["capabilities"]["instruction_prefix"] = "certified"
        candidate["variants"] = certified
        candidate["certification"] = dict(contract=report["contract"], report_sha256=digest(canonical(report)),
            native_invocations=invocations, prefix_sha256=digest(canonical(certified)))
        candidate.pop("sha256")
        candidate["sha256"] = digest(canonical(candidate))
        if len(canonical(candidate)) + 1 > native.MAX_PROFILE_BYTES:
            raise DreamError("Native cache profile exceeds its private byte allowance", 3)
        name = native.profile_key(build, model, route) + ".json"
        dream_run._store(fd, name, canonical(candidate) + b"\n")
        return dict(schema_version=1, operation="dream_cache_certify", state="certified", profile_path=str(output / name),
            capabilities=candidate["capabilities"], paid_inference=False)
    finally:
        os.close(fd)
