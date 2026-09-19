#!/usr/bin/env python3
"""Clear stale SLURM state from ~/.sucoder/sessions/*.yaml.

A stale record names a job the scheduler no longer has.  ``sucoder release``
is the supported way to clear one, but it resolves the mirror through
``config.mirrors`` and the target through ``-T``, so it cannot reach a record
whose mirror has since been dropped from the config -- which is most of them
on a long-lived launcher host.

This does what ``release``'s ``_forget_allocation`` does and nothing more:
sets ``slurm_job_id`` and ``compute_node`` to null, leaving ``login_node``
and every other key untouched so later commands can still resolve the
gateway path.  The files are not deleted.

Safety: a record is only touched when its job is absent from a *successful*
``squeue`` on every configured cluster.  If any cluster cannot be queried the
run aborts, because "squeue did not answer" must never be read as "the job is
gone".  Dry-run unless --apply, which first writes a timestamped tar backup.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

import yaml

SESSIONS = Path.home() / ".sucoder" / "sessions"
CONFIG = Path.home() / ".sucoder" / "config.yaml"
SOCKETS = Path.home() / ".sucoder" / "ssh"


def gateways() -> list[str]:
    cfg = yaml.safe_load(CONFIG.read_text()) or {}
    found = {
        t["gateway"]
        for t in (cfg.get("targets") or {}).values()
        if isinstance(t, dict) and t.get("gateway") and t.get("slurm")
    }
    return sorted(found)


def live_job_ids() -> set[int]:
    """Every job id squeue reports, across all scheduler gateways."""
    live: set[int] = set()
    for gw in gateways():
        sock = SOCKETS / f"{gw}.sock"
        if not sock.exists():
            sys.exit(
                f"abort: no warm ControlMaster for {gw} ({sock}).\n"
                f"Bring one up first (e.g. `sucoder -T <target> tunnel`), so a\n"
                f"failed query is never mistaken for an empty queue."
            )
        cmd = ["ssh", "-S", str(sock), gw, "squeue --me --noheader -o %i"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(
                f"abort: squeue failed on {gw} (exit {proc.returncode}): "
                f"{(proc.stderr or proc.stdout).strip()}"
            )
        for line in proc.stdout.split():
            # Array elements ("39025067_3") hold the parent allocation too.
            head = line.split("_", 1)[0]
            if head.isdigit():
                live.add(int(head))
        print(f"  {gw}: {len(proc.stdout.split())} job(s) in queue")
    return live


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="Actually write (default: show what would change).")
    args = ap.parse_args()

    if not SESSIONS.is_dir():
        sys.exit(f"no session directory at {SESSIONS}")

    print("querying the scheduler:")
    live = live_job_ids()
    print(f"  live job ids: {sorted(live) or 'none'}\n")

    stale: list[tuple[Path, int]] = []
    for path in sorted(SESSIONS.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except (yaml.YAMLError, OSError) as exc:
            print(f"  ?? {path.name}: unreadable ({exc}); skipped")
            continue
        job = data.get("slurm_job_id") if isinstance(data, dict) else None
        if job and int(job) not in live:
            stale.append((path, int(job)))

    if not stale:
        print("No stale records. Nothing to do.")
        return 0

    width = max(len(p.name) for p, _ in stale)
    for path, job in stale:
        print(f"  {path.name.ljust(width)}  job {job}  -> slurm_job_id/compute_node = null")
    print(f"\n{len(stale)} record(s) stale.")

    if not args.apply:
        print("\nDry run. Re-run with --apply to write.")
        return 0

    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%SZ")
    backup = SESSIONS.parent / f"sessions-backup-{stamp}.tar.gz"
    with tarfile.open(backup, "w:gz") as tar:
        tar.add(SESSIONS, arcname="sessions")
    print(f"\nbackup: {backup}")

    for path, job in stale:
        data = yaml.safe_load(path.read_text()) or {}
        data["slurm_job_id"] = None
        data["compute_node"] = None
        # Atomic replace, the same discipline RemoteSession.save uses: a
        # truncated session file loads as blank, which for a SLURM target
        # would drop a live job id and make the next collaborate resubmit.
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, default_flow_style=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        print(f"  cleared {path.name} (was job {job})")

    print(f"\nCleared {len(stale)} record(s). Verify with `sucoder sessions --fast`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
