#!/usr/bin/env python3
"""Bounded, opt-in live relay probe; starts two owners and sends no inference.

Run from the SuCoder checkout. Existing services and shared installation /
enrollment state are not stopped or reset. All evidence stays in report-root.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sucoder.codex_remote import build_remote_command
from sucoder.codex_remote_client import request
from sucoder.codex_state import PLACEHOLDER, app_server_command, runner_command, with_sqlite_home_placeholder


def enrollment_fingerprint(home):
    path = home / "state_5.sqlite"
    if not path.exists():
        return None
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=3) as database:
        if not database.execute("SELECT 1 FROM sqlite_master WHERE name='remote_control_enrollments'").fetchone():
            return None
        rows = database.execute("SELECT * FROM remote_control_enrollments ORDER BY websocket_url,account_id,app_server_client_name").fetchall()
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="authorize this probe's two temporary authenticated relay connections")
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--codex-home", type=Path, default=Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser())
    parser.add_argument("--wait-seconds", type=float, default=45)
    parser.add_argument("--verify-pairing", action="store_true", help="Verify native pairing responses without saving or displaying their secrets")
    parser.add_argument("--owners", type=int, choices=(1, 2), default=2)
    parser.add_argument("--scope", help="Shared 64-hex probe scope for simultaneous probes on distinct nodes")
    parser.add_argument("--hold-seconds", type=float, default=0, help="Keep connected owners alive for cross-node overlap (maximum 60)")
    args = parser.parse_args()
    if not 0 <= args.hold_seconds <= 60 or (args.scope and not re.fullmatch("[0-9a-f]{64}", args.scope)):
        parser.error("invalid hold duration or probe scope")
    if not args.live or not 0 < args.wait_seconds <= 120:
        parser.error("--live is required and --wait-seconds must be positive and at most 120")
    os.umask(0o077)
    report = args.report_root.expanduser().resolve()
    report.mkdir(parents=True, exist_ok=False, mode=0o700)
    home = args.codex_home.expanduser().resolve()
    source_id = (home / "installation_id").read_bytes()
    source_enrollment = enrollment_fingerprint(home)
    root = Path(__file__).resolve().parents[1]
    scope = args.scope or hashlib.sha256(("sucoder-remote-probe-" + uuid.uuid4().hex).encode()).hexdigest()
    codex = shutil.which("codex")
    if not codex:
        parser.error("codex is not installed")
    base = with_sqlite_home_placeholder([codex])
    helper = str(root / "sucoder/codex_remote_client.py")
    command = runner_command(str(root / "sucoder/codex_state_runner.py"), scope,
                             build_remote_command(base, "server"),
                             durable_root=str(report / "durable"), checkpoint_minutes=10,
                             checkpoint_timeout_seconds=20, app_server=app_server_command(base),
                             temporary_local_root=True, remote_helper=helper)
    processes = []
    handles = []
    results = {}
    pairing_checks = {}
    env = dict(os.environ, CODEX_HOME=str(home), PYTHONUNBUFFERED="1")
    env.pop("TMUX", None)
    env.pop("TMUX_PANE", None)
    try:
        for index in range(args.owners):
            handle = (report / ("owner-%d.log" % index)).open("wb")
            handles.append(handle)
            processes.append(subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                               env=env, start_new_session=True))
        deadline = time.monotonic() + args.wait_seconds
        while time.monotonic() < deadline:
            receipts = list(Path("/tmp").glob("sucoder-codex-state-*/state/codex/%s/*/remote.json" % scope))
            results = {}
            for receipt in receipts:
                try:
                    data = request(str(receipt.parent), "status", timeout=2)
                    results[str(receipt.parent)] = data
                except Exception as exc:
                    results[str(receipt.parent)] = {"error": str(exc)}
            if len(results) == args.owners and all(value.get("status") == "connected" for value in results.values()):
                break
            if any(process.poll() is not None for process in processes):
                break
            time.sleep(0.5)
        if len(results) == args.owners and all(value.get("status") == "connected" for value in results.values()):
            if args.verify_pairing:
                for run, status in results.items():
                    pairing = request(run, "pair", timeout=10)
                    pairing_checks[run] = (bool(pairing.get("pairingCode")) and bool(pairing.get("manualPairingCode"))
                                           and pairing.get("environmentId") == status.get("environmentId")
                                           and pairing.get("installationId") == status.get("installationId"))
            time.sleep(args.hold_seconds)
    finally:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=35)
            except subprocess.TimeoutExpired:
                # Only this opt-in probe's isolated process group; state and
                # all evidence are retained even after a forced stop.
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        for handle in handles:
            handle.close()
    identities = {value.get("installationId") for value in results.values()}
    environments = {value.get("environmentId") for value in results.values()}
    passed = (len(results) == args.owners and len(identities) == args.owners and None not in identities
              and len(environments) == args.owners and None not in environments
              and all(value.get("status") == "connected" for value in results.values()))
    unchanged = ((home / "installation_id").read_bytes() == source_id
                 and enrollment_fingerprint(home) == source_enrollment)
    final = {run: json.loads((Path(run) / "status.json").read_text()) for run in results}
    clean = (all(process.returncode == 0 for process in processes)
             and all(value.get("last_generation", {}).get("class") == "sealed" for value in final.values()))
    output = {"passed": passed and unchanged and clean and (not args.verify_pairing or len(pairing_checks) == args.owners and all(pairing_checks.values())),
              "pairing_verified": pairing_checks, "final_state": final, "scope": scope, "source_identity_and_enrollment_unchanged": unchanged,
              "codex_version": subprocess.check_output([codex, "--version"], text=True).strip(),
              "owners": results, "exit_codes": [process.returncode for process in processes],
              "report_root": str(report)}
    (report / "result.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
