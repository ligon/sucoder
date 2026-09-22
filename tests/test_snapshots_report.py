"""``sucoder snapshots``: the pure half, and the listing script run for real.

The precedence under test is the one issue 14 settled on: a job still going
keeps its snapshot whatever the ref's age; an ended job's snapshot goes once
its *end* is past the window; a job accounting cannot place falls back to
ref age against the cluster's longest allocation plus the window; and where
even that has no finite bound, nothing is deleted.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from sucoder.snapshots_report import (
    DAY, SNAPSHOT_LIST_SH, ClusterSnapshots, Decision, JobRecord, Listing,
    SnapshotRef, decide, parse_listing, parse_retire_output, render,
    retire_script,
)

NOW = 1_790_067_930          # what the cluster reported on 2026-09-22
HOME = "/global/home/users/x/mirrors"

# Verbatim from a live run on 2026-09-22 (job ids and hashes as they were),
# so the parser is tested against what a login node actually prints.
LIVE = (
    f"NOW\t{NOW}\t-0700\n"
    f"REF\t{HOME}/LSMS_Library\trefs/sucoder/wip-job/LSMS_Library/39123514\t8d27481882bf4752e96ffdba954efea49d74f4fd\t1790067766\tWIP snapshot 2026-09-22T02:02:46-07:00 job 39123514\n"
    f"REF\t{HOME}/LSMS_Library\trefs/sucoder/wip/LSMS_Library\t378e07c6c42e10590c5db842e65b64d52dac7dda\t1789772602\tWIP snapshot 2026-09-18T16:03:22-07:00 job 39002769\n"
    f"REF\t{HOME}/SuCoder\trefs/sucoder/wip/SuCoder\t341ca242d7fc208ab243a46c5f01f685e8821393\t1789772911\tWIP snapshot 2026-09-18T16:08:31-07:00 job 39025067\n"
    "SACCT-AVAILABLE\t1\n"
    "SACCT-RC\t0\n"
    "SACCT\t39002769|CANCELLED by 0|1789779242|savio4_htc|co_carleton|carleton_htc4_normal\n"
    "SACCT\t39025067|CANCELLED by 0|1789779242|savio4_htc|co_carleton|carleton_htc4_normal\n"
    "SACCT\t39123514|RUNNING|Unknown|savio4_htc|co_carleton|carleton_htc4_normal\n"
)


def _snap(ref, committed, subject="", path=f"{HOME}/M", sha="a" * 40):
    return SnapshotRef(path, ref, sha, committed, subject)


# -- parsing ---------------------------------------------------------------------

def test_parses_what_a_login_node_prints():
    listing = parse_listing(LIVE)
    assert listing.now == NOW and listing.tz_offset == "-0700"
    assert listing.sacct_available is True and listing.sacct_rc == 0
    assert [r.mirror for r in listing.refs] == ["LSMS_Library", "LSMS_Library", "SuCoder"]
    assert listing.refs[0].job_id == 39123514           # from the ref name
    assert listing.refs[1].job_id == 39002769           # from a legacy subject
    assert listing.jobs[39002769].state == "CANCELLED"  # "CANCELLED by 0" -> first word
    assert listing.jobs[39002769].end == 1789779242
    assert listing.jobs[39123514].end is None and not listing.jobs[39123514].ended
    assert listing.errors == []


def test_an_iso_end_is_placed_with_the_clusters_offset():
    text = (
        f"NOW\t{NOW}\t-0700\n"
        "SACCT-AVAILABLE\t1\nSACCT-RC\t0\n"
        "SACCT\t1|TIMEOUT|2026-09-18T17:54:02|p|a|q\n"
    )
    listing = parse_listing(text)
    # 2026-09-18T17:54:02-07:00
    assert listing.jobs[1].end == 1789779242


def test_malformed_rows_are_reported_not_dropped():
    listing = parse_listing(f"NOW\t{NOW}\t-0700\nREF\tonly\ttwo\nSACCT\tgarbage\nWHAT\n")
    assert listing.refs == [] and listing.jobs == {}
    assert len(listing.errors) == 3


def test_a_failed_sacct_keeps_its_complaint_and_says_it_did_not_answer():
    listing = parse_listing(
        f"NOW\t{NOW}\t-0700\nSACCT-AVAILABLE\t1\nSACCT-RC\t1\n"
        "SACCT\tsacct: error: slurm_persist_conn_open_without_init: failed\n"
    )
    assert not listing.sacct_answered
    assert listing.errors == ["sacct: sacct: error: slurm_persist_conn_open_without_init: failed"]


def test_a_legacy_ref_without_a_job_in_its_subject_has_no_job_id():
    assert _snap("refs/sucoder/wip/M", NOW, "WIP snapshot").job_id is None
    assert _snap("refs/sucoder/wip-job/M/not-a-job", NOW).job_id is None


# -- the precedence ------------------------------------------------------------

FALLBACK = 12 * DAY + 7 * DAY      # carleton-htc's 12 days plus the window


def _listing(refs, jobs=(), *, sacct=True, rc=0, now=NOW):
    return Listing(
        now=now, sacct_available=sacct, sacct_rc=rc, refs=list(refs),
        jobs={j.job_id: j for j in jobs},
    )


def test_a_running_jobs_snapshot_is_kept_whatever_its_age():
    stale = _snap("refs/sucoder/wip-job/M/7", NOW - 30 * DAY)   # older than any bound
    [d] = decide(_listing([stale], [JobRecord(7, "RUNNING", None)]), fallback_seconds=FALLBACK)
    assert not d.retire and d.reason == "kept: job still going"


def test_an_ended_job_goes_once_its_end_is_past_the_window():
    snap = _snap("refs/sucoder/wip-job/M/7", NOW - 20 * DAY)
    fresh = JobRecord(7, "TIMEOUT", NOW - 6 * DAY)
    old = JobRecord(7, "TIMEOUT", NOW - 8 * DAY)
    [kept] = decide(_listing([snap], [fresh]), fallback_seconds=FALLBACK)
    [gone] = decide(_listing([snap], [old]), fallback_seconds=FALLBACK)
    assert not kept.retire and "within the 7d window" in kept.reason
    assert gone.retire and gone.reason == "job ended 8d ago, past the 7d window"
    assert gone.state == "timeout 8d ago"


def test_the_window_is_measured_from_the_end_not_the_ref():
    # The ref froze 20 days ago; the job only ended yesterday.  Keep.
    snap = _snap("refs/sucoder/wip-job/M/7", NOW - 20 * DAY)
    [d] = decide(_listing([snap], [JobRecord(7, "CANCELLED", NOW - DAY)]), fallback_seconds=FALLBACK)
    assert not d.retire


def test_a_job_accounting_cannot_place_falls_back_to_ref_age():
    young = _snap("refs/sucoder/wip-job/M/7", NOW - 10 * DAY)
    old = _snap("refs/sucoder/wip-job/M/8", NOW - 25 * DAY)
    kept, gone = decide(_listing([young, old]), fallback_seconds=FALLBACK)
    assert not kept.retire and "no accounting record" in kept.reason
    assert gone.retire and gone.reason == "no accounting record; snapshot 25d old, past the 19d fallback"
    assert gone.state == "unknown"


def test_no_job_id_at_all_falls_back_to_ref_age():
    snap = _snap("refs/sucoder/wip/M", NOW - 25 * DAY, "WIP snapshot")
    [d] = decide(_listing([snap]), fallback_seconds=FALLBACK)
    assert d.retire and d.reason.startswith("no job recorded")


def test_sacct_unavailable_means_ref_age_for_everyone_and_says_so():
    snap = _snap("refs/sucoder/wip-job/M/7", NOW - 25 * DAY)
    [d] = decide(_listing([snap], sacct=False), fallback_seconds=FALLBACK)
    assert d.retire and d.reason.startswith("sacct unavailable")


def test_a_failed_sacct_is_not_read_as_an_answer():
    # rc != 0 with a row that happens to parse must not count: the rows are
    # whatever sacct printed on its way out.
    snap = _snap("refs/sucoder/wip-job/M/7", NOW - DAY)
    [d] = decide(
        _listing([snap], [JobRecord(7, "COMPLETED", NOW - 30 * DAY)], rc=1),
        fallback_seconds=FALLBACK,
    )
    assert not d.retire and "sacct unavailable" in d.reason


def test_no_safe_bound_deletes_nothing():
    snap = _snap("refs/sucoder/wip-job/M/7", NOW - 400 * DAY)
    [d] = decide(
        _listing([snap]), fallback_seconds=None,
        fallback_reason="target x has no finite slurm.time",
    )
    assert not d.retire
    assert d.reason == "kept: no accounting record; target x has no finite slurm.time"


def test_no_clock_from_the_cluster_deletes_nothing():
    snap = _snap("refs/sucoder/wip-job/M/7", NOW - 400 * DAY)
    [d] = decide(
        _listing([snap], [JobRecord(7, "FAILED", NOW - 300 * DAY)], now=None),
        fallback_seconds=FALLBACK,
    )
    assert not d.retire and "no clock" in d.reason


def test_an_ended_job_with_an_unknown_end_falls_back_to_ref_age():
    snap = _snap("refs/sucoder/wip-job/M/7", NOW - 25 * DAY)
    [d] = decide(_listing([snap], [JobRecord(7, "NODE_FAIL", None)]), fallback_seconds=FALLBACK)
    assert d.retire and d.reason.startswith("job ended at an unknown time")


def test_the_window_is_configurable():
    snap = _snap("refs/sucoder/wip-job/M/7", NOW - 5 * DAY)
    [d] = decide(
        _listing([snap], [JobRecord(7, "COMPLETED", NOW - 2 * DAY)]),
        window_days=1, fallback_seconds=FALLBACK,
    )
    assert d.retire and "past the 1d window" in d.reason


# -- rendering and the retire script ------------------------------------------

def test_render_shows_the_decision_in_words_and_counts_them():
    listing = parse_listing(LIVE)
    decisions = decide(listing, fallback_seconds=FALLBACK)
    out = render([ClusterSnapshots("hpc.brc", ["carleton-htc"], decisions)])
    assert "hpc.brc   (carleton-htc)" in out
    assert "wip-job/LSMS_Library/39123514" in out and "kept: job still going" in out
    assert "3 snapshot(s); nothing past retention." in out
    # The prefix is dropped: the namespace is the same on every row.
    assert "refs/sucoder/" not in out


def test_render_names_what_retire_would_do():
    snap = _snap("refs/sucoder/wip-job/M/7", NOW - 25 * DAY)
    d = Decision(snap, True, "unknown", "no accounting record; snapshot 25d old, past the 19d fallback", 25 * DAY)
    assert "1 past retention.  --retire deletes them." in render([ClusterSnapshots("c", ["t"], [d])])
    assert "retiring 1." in render([ClusterSnapshots("c", ["t"], [d])], retiring=True)


def test_render_says_when_a_cluster_has_nothing_and_surfaces_errors():
    out = render([ClusterSnapshots("c", ["t"], [], notes=["a note"], errors=["boom"])])
    assert "no WIP snapshots" in out and "(a note)" in out and "! c: boom" in out


def test_retire_script_guards_each_delete_with_the_hash_the_listing_saw():
    snap = _snap("refs/sucoder/wip-job/M/7", NOW, path=f"{HOME}/K Agg", sha="b" * 40)
    keep = Decision(_snap("refs/sucoder/wip-job/M/8", NOW), False, "running", "kept", 0)
    script = retire_script([Decision(snap, True, "unknown", "old", 0), keep])
    assert f"update-ref -d refs/sucoder/wip-job/M/7 {'b' * 40}" in script
    assert "'/global/home/users/x/mirrors/K Agg'" in script
    assert "wip-job/M/8" not in script


def test_parse_retire_output_splits_retired_from_failed():
    retired, failed = parse_retire_output("RETIRED\t/m/A\trefs/x\nFAILED\t/m/B\trefs/y\njunk\n")
    assert retired == [("/m/A", "refs/x")] and failed == [("/m/B", "refs/y")]


# -- the script, run for real ------------------------------------------------------

pytestmark_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(cwd, *argv):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *argv],
        cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytestmark_git
def test_listing_script_finds_every_snapshot_once_and_asks_sacct_once(tmp_path):
    root = tmp_path / "mirrors"
    a = root / "A"
    a.mkdir(parents=True)
    _git(a, "init", "-q")
    _git(a, "commit", "-q", "--allow-empty", "-m", "base")
    head = _git(a, "rev-parse", "HEAD")
    tree = _git(a, "rev-parse", "HEAD^{tree}")
    snap = _git(a, "commit-tree", tree, "-p", head, "-m", "WIP snapshot 2026-09-01T00:00:00 job 11")
    _git(a, "update-ref", "refs/sucoder/wip-job/A/11", snap)
    _git(a, "update-ref", "refs/sucoder/wip/A", snap)        # legacy: id from the subject
    _git(a, "update-ref", "refs/heads/not-a-snapshot", snap)  # never listed
    # A linked worktree beside the mirror shares its refs: listed once.
    _git(a, "worktree", "add", "-q", "--detach", str(root / "A.wt"), head)
    # A directory that is not a repository is skipped.
    (root / "notes").mkdir()
    # A second mirror with no snapshots contributes nothing.
    b = root / "B"
    b.mkdir()
    _git(b, "init", "-q")

    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "sacct").write_text(
        "#!/bin/sh\n"
        "echo \"$@\" > \"$SACCT_ARGS\"; echo \"$SLURM_TIME_FORMAT\" >> \"$SACCT_ARGS\"\n"
        "printf '11|TIMEOUT|1790000000|p|a|q\\n'\n"
    )
    (stub / "sacct").chmod(0o755)
    env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}", SACCT_ARGS=str(tmp_path / "args"))

    result = subprocess.run(
        ["bash", "-c", SNAPSHOT_LIST_SH, "_", str(root)],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    listing = parse_listing(result.stdout)
    assert listing.errors == [], result.stdout
    assert sorted((r.mirror, r.short_ref) for r in listing.refs) == [
        ("A", "wip-job/A/11"), ("A", "wip/A"),
    ]
    assert all(r.job_id == 11 for r in listing.refs)
    assert listing.sacct_available and listing.sacct_rc == 0
    assert listing.jobs[11].state == "TIMEOUT" and listing.jobs[11].end == 1790000000
    # One sacct call, for the de-duplicated id list, asking for epoch times.
    args, fmt = (tmp_path / "args").read_text().splitlines()
    assert "-j 11 " in args and fmt == "%s"


@pytestmark_git
def test_listing_script_without_sacct_says_so(tmp_path):
    root = tmp_path / "mirrors"
    root.mkdir()
    stub = tmp_path / "bin"
    stub.mkdir()
    # A PATH holding only git and the shell basics, no sacct.
    for tool in ("git", "bash", "sh", "cut", "sed", "grep", "sort", "paste", "rm", "date", "printf"):
        path = shutil.which(tool)
        if path:
            os.symlink(path, stub / tool)
    result = subprocess.run(
        ["bash", "-c", SNAPSHOT_LIST_SH, "_", str(root)],
        env={"PATH": str(stub), "HOME": str(tmp_path)}, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert parse_listing(result.stdout).sacct_available is False


@pytestmark_git
def test_retire_script_deletes_only_the_unmoved_ref(tmp_path):
    m = tmp_path / "M"
    m.mkdir()
    _git(m, "init", "-q")
    _git(m, "commit", "-q", "--allow-empty", "-m", "base")
    head = _git(m, "rev-parse", "HEAD")
    tree = _git(m, "rev-parse", "HEAD^{tree}")
    first = _git(m, "commit-tree", tree, "-p", head, "-m", "WIP snapshot 1 job 7")
    second = _git(m, "commit-tree", tree, "-p", head, "-m", "WIP snapshot 2 job 8")
    _git(m, "update-ref", "refs/sucoder/wip-job/M/7", first)
    _git(m, "update-ref", "refs/sucoder/wip-job/M/8", first)

    seen7 = SnapshotRef(str(m), "refs/sucoder/wip-job/M/7", first, NOW, "")
    seen8 = SnapshotRef(str(m), "refs/sucoder/wip-job/M/8", first, NOW, "")
    # Between the listing and the retire, job 8 snapshotted again.
    _git(m, "update-ref", "refs/sucoder/wip-job/M/8", second)

    script = retire_script([
        Decision(seen7, True, "unknown", "old", 0),
        Decision(seen8, True, "unknown", "old", 0),
    ])
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    retired, failed = parse_retire_output(result.stdout)
    assert retired == [(str(m), "refs/sucoder/wip-job/M/7")]
    assert failed == [(str(m), "refs/sucoder/wip-job/M/8")]
    assert _git(m, "rev-parse", "refs/sucoder/wip-job/M/8") == second
    assert subprocess.run(
        ["git", "rev-parse", "-q", "--verify", "refs/sucoder/wip-job/M/7"],
        cwd=m, capture_output=True,
    ).returncode != 0
