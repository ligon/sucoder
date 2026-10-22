"""Owned-socket diagnostics and pairing, staged standalone on the target.

Stdlib only, Python 3.6 compatible like codex_state_runner. This client never
starts a daemon, resumes a thread, or performs inference (ledger 3--5).
"""
import argparse
import base64
import hashlib
import json
import os
import math
import stat
import socket
import struct
import sys
import time


MAX_MESSAGE = 8 * 1024 * 1024


class RemoteError(Exception):
    pass


class RPC(object):
    """Bounded RFC 6455 text messages over the owner's Unix socket."""

    def __init__(self, path, timeout=10):
        self.deadline = time.monotonic() + timeout
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sequence = 0
        try:
            self.socket.settimeout(timeout)
            self.socket.connect(path)
            key = base64.b64encode(os.urandom(16)).decode("ascii")
            request = ("GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                       "Sec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n" % key)
            self.socket.sendall(request.encode("ascii"))
            header = bytearray()
            while not header.endswith(b"\r\n\r\n"):
                if len(header) >= 16384:
                    raise RemoteError("WebSocket response headers exceed 16 KiB")
                header.extend(self.exact(1))
            lines = header.decode("ascii").split("\r\n")
            headers = {key.lower(): value.strip() for key, value in
                       (line.split(":", 1) for line in lines[1:] if ":" in line)}
            accept = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()).decode("ascii")
            if (not lines[0].startswith("HTTP/1.1 101 ")
                    or headers.get("sec-websocket-accept", "") != accept):
                raise RemoteError("owned socket did not complete a WebSocket handshake")
            self.call("initialize", {"clientInfo": {"name": "sucoder_remote_control", "version": "1"},
                                     "capabilities": {"experimentalApi": True}})
            self.send({"method": "initialized"})
        except Exception:
            self.close()
            raise

    def close(self):
        self.socket.close()

    def exact(self, length):
        data = bytearray()
        while len(data) < length:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise RemoteError("owned-socket request timed out")
            self.socket.settimeout(remaining)
            part = self.socket.recv(length - len(data))
            if not part:
                raise RemoteError("owned socket closed during a request")
            data.extend(part)
        return bytes(data)

    def frame(self, opcode, payload):
        mask = os.urandom(4)
        length = len(payload)
        if length < 126:
            header = bytes([0x80 | opcode, 0x80 | length])
        elif length < 65536:
            header = bytes([0x80 | opcode, 0x80 | 126]) + struct.pack("!H", length)
        else:
            header = bytes([0x80 | opcode, 0x80 | 127]) + struct.pack("!Q", length)
        self.socket.sendall(header + mask + bytes(c ^ mask[i % 4] for i, c in enumerate(payload)))

    def send(self, value):
        self.frame(1, json.dumps(value, separators=(",", ":")).encode("utf-8"))

    def receive(self):
        chunks = bytearray()
        started = False
        while True:
            first, second = self.exact(2)
            opcode, final = first & 15, bool(first & 128)
            if first & 112 or second & 128:
                raise RemoteError("unsupported WebSocket frame flags")
            length = second & 127
            if length == 126:
                length = struct.unpack("!H", self.exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self.exact(8))[0]
            if length > MAX_MESSAGE or len(chunks) + length > MAX_MESSAGE:
                raise RemoteError("owned socket message exceeds 8 MiB")
            if opcode >= 8 and (not final or length > 125):
                raise RemoteError("invalid WebSocket control frame")
            payload = self.exact(length)
            if opcode == 8:
                raise RemoteError("owned socket closed the connection")
            if opcode == 9:
                self.frame(10, payload)
                continue
            if opcode == 10:
                continue
            if opcode not in (0, 1) or (opcode == 0) != started:
                raise RemoteError("unsupported WebSocket message sequence")
            chunks.extend(payload)
            started = True
            if final:
                value = json.loads(chunks.decode("utf-8"))
                if not isinstance(value, dict):
                    raise RemoteError("owned socket returned a non-object JSON message")
                return value

    def call(self, method, params=None):
        self.sequence += 1
        request = {"id": self.sequence, "method": method}
        if params is not None:
            request["params"] = params
        self.send(request)
        for _ in range(256):
            reply = self.receive()
            if reply.get("id") == self.sequence:
                if "error" in reply:
                    error = reply["error"]
                    raise RemoteError("%s: %s" % (method, error.get("message", "request failed") if isinstance(error, dict) else "invalid error response"))
                result = reply.get("result")
                if not isinstance(result, dict):
                    raise RemoteError("%s returned no result object" % method)
                return result
        raise RemoteError("too many notifications while waiting for %s" % method)


def process_start(pid):
    try:
        with open("/proc/%d/stat" % pid) as handle:
            return handle.read().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError, ValueError, TypeError):
        return None


def register(runtime, pid):
    run = os.environ["SUCODER_CODEX_STATE_RUN"]
    start = process_start(pid)
    if start is None:
        raise RemoteError("owned server exited before registration")
    receipt = {"run": os.path.realpath(run), "node": socket.gethostname(), "pid": pid, "start": start,
               "socket": os.path.join(runtime, "server.sock"),
               "installation_id": os.environ["SUCODER_CODEX_REMOTE_ID"]}
    path = os.path.join(run, "remote.json")
    temporary = path + ".tmp-%d" % os.getpid()
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(receipt, handle)
    os.replace(temporary, path)
    return receipt


def load_receipt(run):
    path = os.path.join(run, "remote.json")
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_size > 16384:
        raise RemoteError("remote-control receipt is not an owned regular file")
    with open(path) as handle:
        receipt = json.load(handle)
    if not isinstance(receipt, dict):
        raise RemoteError("remote-control receipt is not an object")
    if receipt.get("run") != os.path.realpath(run) or receipt.get("node") != socket.gethostname():
        raise RemoteError("remote-control receipt belongs to another run or node")
    if not isinstance(receipt.get("pid"), int) or isinstance(receipt["pid"], bool):
        raise RemoteError("remote-control receipt has no server process identity")
    if not isinstance(receipt.get("start"), str) or not receipt["start"].isdigit():
        raise RemoteError("remote-control receipt has no server start identity")
    if not isinstance(receipt.get("socket"), str) or not os.path.isabs(receipt["socket"]):
        raise RemoteError("remote-control receipt has no absolute socket path")
    if not isinstance(receipt.get("installation_id"), str) or not receipt["installation_id"]:
        raise RemoteError("remote-control receipt has no installation identity")
    return receipt


def request(run, action, timeout=10):
    if action not in ("status", "pair"):
        raise RemoteError("remote-control action must be status or pair")
    receipt = load_receipt(run)
    if process_start(receipt["pid"]) != receipt.get("start"):
        if action == "status":
            return dict(receipt, status="stopped", local_ready=False)
        raise RemoteError("the recorded remote-control owner has stopped; start/select a live launch")
    client = RPC(receipt["socket"], timeout=timeout)
    try:
        status = client.call("remoteControl/status/read")
        if status.get("installationId") != receipt["installation_id"]:
            raise RemoteError("owned socket returned a different installation identity")
        if action == "status":
            return dict(receipt, **status, local_ready=True)
        if status.get("status") != "connected" or not status.get("environmentId"):
            raise RemoteError("remote relay is %s; pairing requires connected status (a local socket is insufficient)" % status.get("status"))
        pairing = client.call("remoteControl/pairing/start", {"manualCode": True})
        after = client.call("remoteControl/status/read")
        if (pairing.get("environmentId") != status["environmentId"]
                or after.get("environmentId") != status["environmentId"]
                or after.get("installationId") != receipt["installation_id"]
                or after.get("status") != "connected"):
            raise RemoteError("remote-control environment changed while pairing; retry after checking status")
        return dict(pairing, node=receipt["node"], installationId=receipt["installation_id"])
    finally:
        client.close()


def wait_connected(run, seconds):
    deadline = time.monotonic() + seconds
    last = "initializing"
    while time.monotonic() < deadline:
        try:
            value = request(run, "status", timeout=min(2, max(0.1, deadline - time.monotonic())))
            last = value.get("status", "unknown")
            if last == "connected":
                print("SuCoder: Codex remote relay connected on %s (installation %s)." %
                      (value["node"], value["installation_id"]), file=sys.stderr)
                return 0
            if last == "stopped":
                break
        except (RemoteError, OSError, ValueError) as exc:
            last = str(exc)
        time.sleep(min(0.25, max(0, deadline - time.monotonic())))
    print("SuCoder: remote access is unavailable (%s); local socket readiness does not prove relay connectivity. "
          "Use sucoder codex-remote-status to check this launch." % last, file=sys.stderr)
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "wait", "status", "pair"))
    parser.add_argument("--run", default=os.environ.get("SUCODER_CODEX_STATE_RUN"))
    parser.add_argument("--runtime")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args(argv)
    try:
        if args.action == "register":
            if not args.runtime or not args.pid:
                parser.error("register requires --runtime and --pid")
            register(args.runtime, args.pid)
            return 0
        if not args.run:
            parser.error("--run is required")
        if not math.isfinite(args.timeout) or args.timeout <= 0 or args.timeout > 60:
            parser.error("--timeout must be positive and at most 60 seconds")
        if args.action == "wait":
            return wait_connected(args.run, args.timeout)
        print(json.dumps(request(args.run, args.action, timeout=args.timeout)))
        return 0
    except (RemoteError, OSError, ValueError, KeyError) as exc:
        print("SuCoder remote control: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
