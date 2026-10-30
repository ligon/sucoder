"""Contract tests use disposable workspaces and fake transports, never SSH."""
import copy
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

import pytest
from typer.testing import CliRunner

from sucoder import cli
from sucoder.config import Config, RemoteConfig, SlurmConfig
from sucoder.observation_process import Budget, Capture, ObservationFailure, capture
from sucoder.observations import (WorkspaceBinding, list_sessions, source_id, Transport,
                                  LOGIN_PROBE, pane_class, MAX_BYTES)
from sucoder.observation_cli import validate_response, dispatch, relay_sessions


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return Config(human_user="test", mirror_root=tmp_path / "mirrors")


def remote(host="cluster", user="account", partition="p", **kwargs):
    return RemoteConfig(gateway=host, remote_user=user,
                        slurm=SlurmConfig(partition=partition, account="acct"), **kwargs)


class FakeTransport:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def run(self, remote, host, script):
        self.calls.append((host, remote.remote_user, script))
        response = self.responses.get((host, remote.remote_user), "")
        if isinstance(response, Exception):
            raise response
        return response


def job(number=42, token="sample", partition="p"):
    return f"{number}|sucoder-{token}|{partition}|acct|normal|RUNNING|1:00|node01\n"


def mirror(config, name="sample"):
    from sucoder.config import MirrorSettings, BranchPrefixes
    config.mirrors[name] = MirrorSettings(name, Path("/canonical"), name, BranchPrefixes())


def record(config, name="sample--a", **values):
    import yaml
    path = Path.home() / ".sucoder/sessions"
    path.mkdir(parents=True, exist_ok=True)
    (path / (name + ".yaml")).write_text(yaml.safe_dump(values))


def checked(value):
    validate_response(value, value["operation"])
    return value


def test_empty_no_targets(config):
    value = checked(list_sessions(config, Budget(3)))
    assert value["status"] == "complete"
    assert value["data"]["sessions"] == []


def test_same_job_different_sources_and_unknown_not_absent(config):
    mirror(config)
    config.targets = {"a": remote("one"), "b": remote("two")}
    record(config, slurm_job_id=42)
    record(config, "sample--b", slurm_job_id=42)
    fake = FakeTransport({("one", "account"): job(), ("two", "account"): ObservationFailure("TIMEOUT")})
    value = checked(list_sessions(config, Budget(3), fast=True, transport=fake))
    assert value["status"] == "partial"
    records = {r["key"]: r for r in value["data"]["records"]}
    assert records["sample--a"]["state"] == "matched"
    assert records["sample--b"]["state"] == "unknown"
    assert value["data"]["sessions"][0]["record_keys"] == ["sample--a"]
    assert len(fake.calls) == 2


def test_account_dedup_and_unknown_accounts(config):
    config.targets = {"a": remote(user="alice"), "b": remote(user="bob"),
                      "c": remote(user="alice"), "d": remote(user=None), "e": remote(user=None)}
    fake = FakeTransport({})
    value = checked(list_sessions(config, Budget(3), fast=True, transport=fake))
    assert len(fake.calls) == 4
    assert value["status"] == "complete"
    assert len({s["source_id"] for s in value["sources"] if s["kind"] == "scheduler"}) == 4


def test_narrowing_shared_source_retains_ambiguous_rows(config):
    config.targets = {"a": remote(partition="p"), "b": remote(partition="q")}
    fake = FakeTransport({("cluster", "account"): job(1, partition="q") + job(2, partition="other")})
    value = checked(list_sessions(config, Budget(3), target="a", fast=True, transport=fake))
    rows = value["data"]["sessions"]
    assert [r["key"]["job_id"] for r in rows] == ["2"]
    assert rows[0]["targets"] == ["a", "b"]
    assert value["status"] == "partial"


def test_replacement_mixed_never_absent(config):
    mirror(config)
    config.targets = {"a": remote()}
    record(config, slurm_job_id=42, replacement={"state": "submitted", "new_job": 43})
    value = checked(list_sessions(config, Budget(3), fast=True,
                     transport=FakeTransport({("cluster", "account"): job()})))
    rec = value["data"]["records"][0]
    assert rec["state"] == "unknown"
    assert [j["state"] for j in rec["jobs"]] == ["matched", "absent"]
    assert value["status"] == "partial"


@pytest.mark.parametrize("raw", ["[invalid]", "false", "0", "slurm_job_id: bad",
    "slurm_job_id: 42\nreplacement: {state: submitted, new_job: bad}"])
def test_corrupt_record_never_absent(config, raw):
    mirror(config)
    config.targets = {"a": remote()}
    record(config)
    (Path.home() / ".sucoder/sessions/sample--a.yaml").write_text(raw)
    value = checked(list_sessions(config, Budget(3), fast=True, transport=FakeTransport({})))
    assert value["data"]["records"][0]["state"] == "corrupt"
    assert value["status"] == "partial"


def test_output_limit_remains_valid_partial(config):
    config.targets = {"a": remote()}
    output = "".join(job(i) for i in range(1, 520))
    value = checked(list_sessions(config, Budget(3), fast=True,
                                  transport=FakeTransport({("cluster", "account"): output})))
    assert value["status"] == "partial" and value["truncated"]
    assert len(json.dumps(value).encode()) < MAX_BYTES
    assert any(e["code"] == "OUTPUT_LIMIT" for e in value["errors"])


def git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True)


def test_context_unborn_detached_and_changed(config, tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    git(root, "init", "-b", "main")
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    binding = WorkspaceBinding(link)
    value = checked(binding.inspect(config, Budget(3)))
    assert value["status"] == "complete"
    assert value["data"]["head"] == dict(value=None, state="observed", source="local")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--allow-empty", "-m", "fixture")
    git(root, "checkout", "--detach")
    value = checked(binding.inspect(config, Budget(3)))
    assert value["data"]["branch"]["value"] is None
    other = tmp_path / "other"
    other.mkdir()
    git(other, "init")
    link.unlink()
    link.symlink_to(other, target_is_directory=True)
    value = checked(binding.inspect(config, Budget(3)))
    assert value["data"]["binding"] == "changed"


def test_timeout_output_and_next_request():
    for script, expected in [("import time; time.sleep(20)", "TIMEOUT"),
                              ("print('x' * 20000)", "OUTPUT_LIMIT")]:
        start = time.monotonic()
        with pytest.raises(ObservationFailure) as exc:
            capture([sys.executable, "-c", script], Budget(.3), limit=1000)
        assert exc.value.code == expected
        assert time.monotonic() - start < 1.5
    assert capture([sys.executable, "-c", "print('ok')"], Budget(2)).stdout == "ok\n"


@pytest.mark.parametrize("stderr,code,success", [
    ("no server running on /tmp/tmux-123/default", 1, True),
    ("error connecting to /tmp/tmux-123/default (No such file or directory)", 1, True),
    ("error connecting to /tmp/tmux-123/default (Permission denied)", 1, False),
    ("tmux: command not found", 127, False)])
def test_tmux_empty_distinguished(tmp_path, stderr, code, success):
    tmux = tmp_path / "tmux"
    import shlex
    tmux.write_text("#!/bin/sh\nprintf '%s\\n' " + shlex.quote(stderr) + " >&2\nexit " + str(code) + "\n")
    tmux.chmod(0o700)
    result = subprocess.run(["/bin/sh", "-c", LOGIN_PROBE], env={"PATH": str(tmp_path)}, capture_output=True)
    assert (result.returncode == 0) == success


def test_transport_never_ensure_and_account_options(config):
    calls = []
    def runner(argv, budget, **kwargs):
        calls.append(argv)
        return Capture(0, "")
    t = Transport(Budget(3), runner)
    t.run(remote(user=None, ssh_options={"User": "bob", "BatchMode": "no"}), "cluster", "true")
    argv = calls[0]
    assert "User=bob" in argv
    assert argv.index("BatchMode=yes") < argv.index("BatchMode=no")
    assert "ControlMaster=no" in argv and "ClearAllForwardings=yes" in argv
    assert "--" in argv


def test_cli_json_skips_mutation_startup(config, tmp_path, monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("interactive startup was invoked")
    monkeypatch.setattr(cli, "run_startup_checks", forbidden)
    path = tmp_path / "config.yaml"
    path.write_text("human_user: test\nmirror_root: /tmp/mirrors\n")
    # Invalid config is also a safe machine-readable domain result.
    result = CliRunner().invoke(cli.app, ["--config", str(path) + ".missing", "sessions", "--json"])
    assert result.exit_code == 2, result.output
    value = checked(json.loads(result.stdout))
    assert value["errors"][0]["code"] == "CONFIG_INVALID"
    assert result.stdout.count("\n") == 1


def test_old_relay_no_fallback(config):
    from sucoder.config import SessionRelayConfig
    config.session_relay = SessionRelayConfig(user="human", command=["/approved/sucoder"])
    calls = []
    def runner(argv, budget, **kwargs):
        calls.append(argv)
        return Capture(0, "") if "-l" in argv else Capture(2, "old human error")
    with pytest.raises(ObservationFailure) as exc:
        relay_sessions(config, {}, Budget(3), runner)
    assert exc.value.code == "RELAY_UNSUPPORTED"
    assert len(calls) == 2
    assert all("-n" in call for call in calls)


def test_contract_fixtures_and_unknown_version():
    directory = Path(__file__).resolve().parents[1] / "docs/agent-interface"
    assert (directory / "response.schema.json").read_bytes() == (directory.parents[1] / "sucoder/observation-response.schema.json").read_bytes()
    for path in (directory / "examples").glob("*.json"):
        value = json.loads(path.read_text())
        checked(value)
        value["schema_version"] = "1.1"
        with pytest.raises(ObservationFailure, match="SCHEMA_UNSUPPORTED"):
            checked(value)


def test_options_identity_and_pinned_account(config):
    config.targets = {"a": remote(ssh_options={"HostName": "one"}),
                      "b": remote(ssh_options={"HostName": "two"})}
    fake = FakeTransport({})
    checked(list_sessions(config, Budget(3), fast=True, transport=fake))
    assert len(fake.calls) == 2
    opts, _ = Transport(Budget(3)).options(remote(user=None, ssh_options={"User": "bob", "ControlPath": "/unsafe"}), "pin")
    assert opts[opts.index("-l") + 1] == "bob"
    assert "ControlPath=none" in opts
    opts, _ = Transport(Budget(3)).options(remote(ssh_options={"ControlPath": "/unsafe"}), "cluster")
    assert opts.index("ControlPath=none") < opts.index("ControlPath=/unsafe")


def test_batched_panes_use_allocation_route(config):
    mirror(config)
    config.targets = {"a": remote()}
    config.targets["a"].slurm.confined = True
    class Batch:
        def __init__(self):
            self.calls = []
        def run(self, remote, host, script):
            self.calls.append((host, script))
            if script.startswith("squeue"):
                return job(42) + job(43)
            assert host == "cluster"
            assert "--jobid=42" in script and "--jobid=43" in script
            assert "sucoder-sample" in script
            return "42\tcodex\tterminal\n43\tbash\tterminal\n"
    batch = Batch()
    value = checked(list_sessions(config, Budget(3), login_nodes=False, transport=batch))
    assert len(batch.calls) == 2
    assert [r["pane"] for r in value["data"]["sessions"]] == ["recognized_agent_command", "shell_command"]
    assert value["status"] == "complete"


def test_login_fused_and_unknown_command(config):
    mirror(config)
    config.targets = {"a": RemoteConfig(host="direct", remote_user="account")}
    fake = FakeTransport({("direct", "account"): "sucoder-sample\tother\tterminal\n"})
    value = checked(list_sessions(config, Budget(3), transport=fake))
    assert len(fake.calls) == 1
    assert value["data"]["sessions"][0]["pane"] == "unknown"
    assert value["status"] == "partial"


def test_node_local_mapping_needs_config_and_origin(config, tmp_path):
    from sucoder.local_tier import work_path
    mirror(config)
    r = remote()
    r.mirror_root = config.mirror_root
    r.slurm.local_disk = str(tmp_path / "local")
    r.slurm.confined = True
    config.targets = {"a": r}
    root = Path(work_path(r.slurm.local_disk, "sample", 42))
    root.mkdir(parents=True)
    git(root, "init")
    git(root, "remote", "add", "origin", str(config.mirror_root / "sample"))
    value = checked(WorkspaceBinding(root).inspect(config, Budget(3)))
    assert value["data"]["mirror"]["value"] == "sample"
    assert value["data"]["confinement"]["value"] == "true"
    assert value["data"]["durable_mirror"]["value"] == str(config.mirror_root / "sample")
    config.targets = {}
    value = checked(WorkspaceBinding(root).inspect(config, Budget(3)))
    assert value["data"]["durable_mirror"]["state"] == "unknown"


def test_contended_lock_timeout(config):
    import threading
    from sucoder.tunnel import session_lock
    lock = session_lock(("cluster", "account"))
    held, release = threading.Event(), threading.Event()
    def holder():
        with lock:
            held.set()
            release.wait(2)
    thread = threading.Thread(target=holder)
    thread.start()
    assert held.wait(1)
    try:
        with pytest.raises(ObservationFailure, match="TIMEOUT"):
            Transport(Budget(.2), lambda *a, **k: pytest.fail("must not spawn")).run(remote(), "cluster", "true")
    finally:
        release.set()
        thread.join(1)


def test_relay_validated_and_caller_identity(config):
    from sucoder.config import SessionRelayConfig
    config.session_relay = SessionRelayConfig(user="human", command=["/approved/sucoder"])
    value = list_sessions(config, Budget(3), fast=True, transport=FakeTransport({}))
    value["observer"].update(host="relay-host", account="human", configuration_authority="human@relay-host")
    calls = []
    def runner(argv, budget, **kwargs):
        calls.append(argv)
        if "-l" in argv:
            return Capture(0, "")
        if "capabilities" in argv:
            return Capture(0, json.dumps(dict(schema_version="1.0", operations=["list_sessions"])))
        return Capture(0, json.dumps(value))
    result = relay_sessions(config, {"target": "a", "fast": True}, Budget(5), runner)
    assert result["observer"]["account"] != "human"
    assert result["observer"]["configuration_authority"] == "human@relay-host"
    assert "--target" in calls[-1]
    assert int(calls[-1][calls[-1].index("--timeout") + 1]) <= 4


def test_cli_context_actual_cwd_no_relay(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    git(root, "init")
    config = tmp_path / "config.yaml"
    config.write_text("human_user: test\nmirror_root: /tmp/mirrors\n")
    monkeypatch.chdir(root)
    result = CliRunner().invoke(cli.app, ["--config", str(config), "context", "--json"])
    assert result.exit_code == 0, result.output
    value = checked(json.loads(result.stdout))
    assert value["data"]["git_root"]["value"] == str(root)


def process_running(pid):
    try:
        # Zombies have exited and cannot perform work; the init process owns
        # reaping after a parent is terminated.
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


@pytest.mark.parametrize('error', [FileNotFoundError, ProcessLookupError])
def test_process_running_handles_disappearance_during_proc_read(monkeypatch, error):
    def disappeared(path):
        raise error()
    monkeypatch.setattr(Path, 'read_text', disappeared)
    assert not process_running(123)


def test_capture_cleans_owned_descendants(tmp_path):
    pidfile = tmp_path / "pid"
    code = ("import subprocess,sys,time; from pathlib import Path; "
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)']); "
            "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(20)")
    with pytest.raises(ObservationFailure, match="TIMEOUT"):
        capture([sys.executable, "-c", code, str(pidfile)], Budget(.5))
    pid = int(pidfile.read_text())
    end = time.monotonic() + .5
    while process_running(pid) and time.monotonic() < end:
        time.sleep(.01)
    assert not process_running(pid)


def test_supervisor_deadline_survives_caller_disconnect(tmp_path):
    """Same supervisor used by relay; no sudo/other-account authority needed."""
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    pidfile = tmp_path / "probe-pid"
    ssh = fakebin / "ssh"
    ssh.write_text("#!" + sys.executable + "\nimport os,time\nfrom pathlib import Path\n"
                   "Path(" + repr(str(pidfile)) + ").write_text(str(os.getpid()))\ntime.sleep(20)\n")
    ssh.chmod(0o700)
    path = tmp_path / "config.yaml"
    path.write_text("human_user: test\nmirror_root: /tmp/mirrors\ntargets:\n"
                    "  a:\n    gateway: fixture\n    transfer_host: fixture\n    remote_user: test\n"
                    "    slurm: {partition: p, account: a}\n")
    options = dict(operation="list_sessions", config=str(path), timeout=2, fast=True)
    package_root = str(Path(__file__).resolve().parents[1])
    code = ("import sys,json; sys.path.insert(0,sys.argv[1]); "
            "from sucoder.observation_cli import dispatch; "
            "print(json.dumps(dispatch(json.loads(sys.argv[2]))))")
    proc = subprocess.Popen([sys.executable, "-c", code, package_root, json.dumps(options)],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            env={"HOME": str(tmp_path), "PATH": str(fakebin) + ":/usr/bin:/bin"})
    proc.stdout.close()  # Caller disappears; supervisor must still enforce its budget.
    try:
        proc.wait(timeout=4)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=1)
    assert pidfile.exists(), "fixture SSH probe was never reached"
    assert not process_running(int(pidfile.read_text()))


def test_malformed_relay_response_rejected(config):
    value = list_sessions(config, Budget(2))
    value["observer"]["secret"] = "must not leak"
    with pytest.raises(ObservationFailure, match="INVALID_RESPONSE"):
        validate_response(value, "list_sessions")
    value["observer"].pop("secret")
    value["coverage"].append(dict(source_id="missing", scope="scheduler", state="complete", count=0, reason=None))
    with pytest.raises(ObservationFailure, match="INVALID_RESPONSE"):
        validate_response(value, "list_sessions")


def test_gateway_socket_cannot_impersonate_override_or_pin(config, monkeypatch):
    monkeypatch.setattr(Path, "is_socket", lambda self: True)
    transport = Transport(Budget(3), lambda *a, **k: Capture(0, "hostname direct\nuser test\n"))
    opts, _ = transport.options(remote(user=None, ssh_options={"Port": "2222"}), "cluster")
    assert "ControlPath=none" in opts
    r = RemoteConfig(host="direct", remote_user="test")
    direct, _ = transport.options(r, "direct")
    pin, _ = transport.options(r, "pinned")
    assert any(s.startswith("ControlPath=") and "direct-" in s for s in direct)
    assert "ControlPath=none" in pin
    assert any(s.startswith("ProxyCommand=") for s in pin)


def test_interrupt_reaps_only_owned_child_and_next_request(tmp_path):
    pidfile = tmp_path / "interrupt-pid"
    class InterruptBudget(Budget):
        def check(self):
            super().check()
            if pidfile.exists():
                raise KeyboardInterrupt()
    script = "import os,time,sys; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(20)"
    with pytest.raises(KeyboardInterrupt):
        capture([sys.executable, "-c", script, str(pidfile)], InterruptBudget(2))
    assert not process_running(int(pidfile.read_text()))
    assert capture([sys.executable, "-c", "print('next')"], Budget(2)).stdout == "next\n"


def test_json_help_never_runs_startup(monkeypatch):
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **k: pytest.fail("startup in help"))
    for command in ("context", "sessions"):
        result = CliRunner().invoke(cli.app, [command, "--help"])
        assert result.exit_code == 0
        # Rich colours help under CI (GITHUB_ACTIONS), splitting "--json" with escapes.
        assert "--json" in re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
