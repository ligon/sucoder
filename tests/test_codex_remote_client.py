"""Owned-socket pairing has no daemon, inference or live-account dependency."""
import base64
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import tempfile
import threading
import time
import uuid

import pytest

from sucoder import codex_remote_client as client


def exact(sock, size):
    result = bytearray()
    while len(result) < size:
        part = sock.recv(size - len(result))
        if not part:
            raise EOFError()
        result.extend(part)
    return bytes(result)


def read_frame(sock):
    first, second = exact(sock, 2)
    size = second & 127
    if size == 126:
        size = struct.unpack("!H", exact(sock, 2))[0]
    elif size == 127:
        size = struct.unpack("!Q", exact(sock, 8))[0]
    mask = exact(sock, 4) if second & 128 else None
    data = exact(sock, size)
    if mask:
        data = bytes(c ^ mask[i % 4] for i, c in enumerate(data))
    return first & 15, data


def frame(data, opcode=1, final=True):
    size = len(data)
    header = bytes([(128 if final else 0) | opcode])
    if size < 126:
        header += bytes([size])
    elif size < 65536:
        header += bytes([126]) + struct.pack("!H", size)
    else:
        header += bytes([127]) + struct.pack("!Q", size)
    return header + data


class FakeServer:
    def __init__(self, runtime, installation, *, status="connected", pair_environment="env_test", fragmented=False,
                 oversized=False, stall=False):
        self.runtime = runtime
        self.installation = installation
        self.status = status
        self.pair_environment = pair_environment
        self.fragmented = fragmented
        self.oversized = oversized
        self.stall = stall
        self.methods = []
        self.errors = []
        self.stop = threading.Event()
        self.socket = socket.socket(socket.AF_UNIX)
        self.socket.bind(str(runtime / "server.sock"))
        self.socket.listen()
        self.socket.settimeout(0.1)
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        while not self.stop.is_set():
            try:
                sock, _ = self.socket.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with sock:
                sock.settimeout(2)
                try:
                    self.connection(sock)
                except (EOFError, OSError):
                    pass
                except Exception as exc:
                    self.errors.append(exc)

    def connection(self, sock):
        header = bytearray()
        while not header.endswith(b"\r\n\r\n"):
            header.extend(exact(sock, 1))
        headers = dict(line.split(": ", 1) for line in header.decode().split("\r\n")[1:] if ": " in line)
        accept = base64.b64encode(hashlib.sha1((headers["Sec-WebSocket-Key"] +
                                  "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
        sock.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                      "Sec-WebSocket-Accept: " + accept + "\r\n\r\n").encode())
        while not self.stop.is_set():
            opcode, data = read_frame(sock)
            if opcode != 1:
                continue
            message = json.loads(data)
            method = message["method"]
            self.methods.append(method)
            if "id" not in message:
                continue
            if self.stall:
                self.stop.wait(0.5)
                return
            if method == "initialize":
                result = {"userAgent": "fake_codex"}
            elif method == "remoteControl/status/read":
                result = {"status": self.status, "installationId": self.installation,
                          "environmentId": "env_test", "serverName": "test-node"}
            elif method == "remoteControl/pairing/start":
                result = {"pairingCode": "1234567890", "manualPairingCode": "ABCD-EFGH",
                          "environmentId": self.pair_environment, "expiresAt": int(time.time()) + 300}
            else:
                raise AssertionError("unexpected RPC " + method)
            output = json.dumps({"id": message["id"], "result": result}).encode()
            if self.oversized:
                sock.sendall(bytes([0x81, 127]) + struct.pack("!Q", client.MAX_MESSAGE + 1))
                return
            if self.fragmented:
                sock.sendall(frame(b"ping", opcode=9) + frame(output[:7], final=False) + frame(output[7:], opcode=0))
            else:
                sock.sendall(frame(output))

    def close(self):
        self.stop.set()
        self.socket.close()
        self.thread.join(timeout=3)
        assert not self.thread.is_alive()
        assert not self.errors


@contextmanager
def owner(tmp_path, monkeypatch, **options):
    runtime = Path(tempfile.mkdtemp(prefix="srpc-"))
    installation = str(uuid.uuid4())
    run = tmp_path / uuid.uuid4().hex
    run.mkdir(mode=0o700)
    server = FakeServer(runtime, installation, **options)
    monkeypatch.setenv("SUCODER_CODEX_STATE_RUN", str(run))
    monkeypatch.setenv("SUCODER_CODEX_REMOTE_ID", installation)
    client.register(str(runtime), os.getpid())
    try:
        yield run, server
    finally:
        server.close()
        shutil.rmtree(runtime)


def test_status_and_pair_address_the_same_owner_without_storing_pairing_secret(tmp_path, monkeypatch):
    with owner(tmp_path, monkeypatch) as (run, server):
        before = (run / "remote.json").read_bytes()
        status = client.request(str(run), "status")
        assert status["status"] == "connected" and status["local_ready"] is True
        pairing = client.request(str(run), "pair")
        assert pairing["manualPairingCode"] == "ABCD-EFGH"
        assert pairing["installationId"] == status["installationId"]
        assert (run / "remote.json").read_bytes() == before
        assert not any("thread" in method or "turn" in method for method in server.methods)
        assert (run / "remote.json").stat().st_mode & 0o777 == 0o600


def test_pairing_refuses_disconnected_relay_even_when_socket_is_ready(tmp_path, monkeypatch):
    with owner(tmp_path, monkeypatch, status="errored") as (run, server):
        assert client.request(str(run), "status")["local_ready"]
        with pytest.raises(client.RemoteError, match="requires connected"):
            client.request(str(run), "pair")
        assert "remoteControl/pairing/start" not in server.methods


def test_pairing_checks_environment_and_installation_identity(tmp_path, monkeypatch):
    with owner(tmp_path, monkeypatch, pair_environment="another_owner") as (run, server):
        with pytest.raises(client.RemoteError, match="environment changed"):
            client.request(str(run), "pair")
        server.installation = str(uuid.uuid4())
        with pytest.raises(client.RemoteError, match="different installation"):
            client.request(str(run), "status")


def test_stale_process_incarnation_never_uses_the_socket(tmp_path, monkeypatch):
    with owner(tmp_path, monkeypatch) as (run, server):
        path = run / "remote.json"
        receipt = json.loads(path.read_text())
        receipt["start"] = str(int(receipt["start"]) + 1)
        path.write_text(json.dumps(receipt))
        assert client.request(str(run), "status")["status"] == "stopped"
        with pytest.raises(client.RemoteError, match="owner has stopped"):
            client.request(str(run), "pair")
        assert not server.methods


def test_another_node_receipt_is_not_followed(tmp_path, monkeypatch):
    with owner(tmp_path, monkeypatch) as (run, server):
        path = run / "remote.json"
        receipt = json.loads(path.read_text())
        receipt["node"] = "n0041.savio4"
        path.write_text(json.dumps(receipt))
        with pytest.raises(client.RemoteError, match="another run or node"):
            client.request(str(run), "status")
        assert not server.methods


def test_control_protocol_accepts_fragmented_messages_and_answers_pings(tmp_path, monkeypatch):
    with owner(tmp_path, monkeypatch, fragmented=True) as (run, _server):
        assert client.request(str(run), "status")["status"] == "connected"


def test_control_protocol_rejects_oversized_messages_before_reading_payload(tmp_path, monkeypatch):
    with owner(tmp_path, monkeypatch, oversized=True) as (run, _server):
        with pytest.raises(client.RemoteError, match="exceeds 8 MiB"):
            client.request(str(run), "status")


def test_control_request_timeout_is_bounded(tmp_path, monkeypatch):
    with owner(tmp_path, monkeypatch, stall=True) as (run, _server):
        started = time.monotonic()
        with pytest.raises((client.RemoteError, socket.timeout)):
            client.request(str(run), "status", timeout=0.1)
        assert time.monotonic() - started < 1
