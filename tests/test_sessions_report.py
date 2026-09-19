"""``sucoder sessions``: the pure half (sucoder.sessions_report).

No SSH, no scheduler: the caller does the I/O and hands this module text.
What is pinned here is the reasoning the listing exists for -- that it is
enumerated from squeue rather than from the session files, so it can show
the jobs those files have lost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sucoder.sessions_report import (
    JobRow,
    SessionEntry,
    build_report,
    cluster_key,
    group_targets_by_cluster,
    match_target,
    parse_squeue,
    render_report,
    target_signature,
    token_to_mirror,
)


# -- stand-ins for config objects (only the attributes the module reads) -------

@dataclass
class _Slurm:
    partition: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    confined: bool = False


@dataclass
class _Remote:
    gateway: Optional[str] = None
    host: Optional[str] = None
    slurm: Optional[_Slurm] = None


# The real config, reduced: four targets on one cluster plus a direct host.
TARGETS = {
    "savio": _Remote(gateway="hpc.brc", slurm=_Slurm()),
    "savio-node": _Remote(gateway="hpc.brc", slurm=_Slurm("savio3", "fc_jevons")),
    "savio-htc": _Remote(gateway="hpc.brc", slurm=_Slurm("savio4_htc", "fc_jevons", "savio_normal")),
    "carleton-htc": _Remote(gateway="hpc.brc", slurm=_Slurm("savio4_htc", "co_carleton", "carleton_htc4_normal", True)),
    "hhsurveys": _Remote(host="hhsurveys.example.org"),
}
MIRROR_TOKENS = {"SuCoder": "SuCoder", "LSMS_Library": "LSMS_Library", "K-Aggregators": "K-Aggregators"}


def _row(job_id, token, partition="savio4_htc", account="co_carleton",
         qos="carleton_htc4_normal", state="RUNNING", left="11-22:30", node="n0029.savio4"):
    return f"{job_id}|sucoder-{token}|{partition}|{account}|{qos}|{state}|{left}|{node}"


# -- grouping ------------------------------------------------------------------

def test_targets_sharing_a_gateway_are_one_cluster():
    """One squeue answers for all of them: same gateway, same $HOME, same
    scheduler.  Querying per target would be four identical round trips."""
    clusters, schedulerless = group_targets_by_cluster(TARGETS)
    assert clusters == {"hpc.brc": ["carleton-htc", "savio", "savio-htc", "savio-node"]}
    assert schedulerless == ["hhsurveys"]


def test_a_target_without_a_scheduler_has_no_cluster():
    assert cluster_key(TARGETS["hhsurveys"]) is None
    assert cluster_key(TARGETS["carleton-htc"]) == "hpc.brc"


def test_signature_omits_unset_parts():
    assert target_signature(TARGETS["carleton-htc"]) == "savio4_htc / co_carleton / carleton_htc4_normal"
    assert target_signature(TARGETS["savio-node"]) == "savio3 / fc_jevons"
    assert target_signature(TARGETS["savio"]) == "(scheduler defaults)"
    assert target_signature(TARGETS["hhsurveys"]) == "no scheduler"


# -- parsing -------------------------------------------------------------------

def test_parse_keeps_sucoder_jobs_and_ignores_other_peoples():
    jobs, bad = parse_squeue("\n".join([
        _row(39025067, "SuCoder"),
        "39030000|my-unrelated-batch|savio3|fc_jevons|normal|RUNNING|1:00:00|n0100.savio3",
        _row(39002769, "LSMS_Library", node="n0043.savio4", left="11-00:18"),
    ]))
    assert [j.job_id for j in jobs] == [39025067, 39002769]
    assert [j.token for j in jobs] == ["SuCoder", "LSMS_Library"]
    assert bad == []


def test_malformed_lines_are_reported_not_dropped():
    """This listing exists to find what local state has lost; swallowing a
    row it could not parse would defeat the point."""
    jobs, bad = parse_squeue("garbage\n" + _row(1, "SuCoder") + "\n39025067_3|sucoder-X|p|a|q|R|1:00|n1")
    assert [j.job_id for j in jobs] == [1]
    assert len(bad) == 2 and any("39025067_3" in b for b in bad)


def test_blank_output_is_not_an_error():
    assert parse_squeue("\n  \n") == ([], [])


# -- target matching -----------------------------------------------------------

def test_job_maps_to_its_target_by_partition_account_qos():
    jobs, _ = parse_squeue(_row(1, "SuCoder"))
    assert match_target(jobs[0], TARGETS, list(TARGETS)) == "carleton-htc"
    jobs, _ = parse_squeue(_row(2, "K-Aggregators", partition="savio3", account="fc_jevons", qos="normal"))
    assert match_target(jobs[0], TARGETS, list(TARGETS)) == "savio-node"


def test_ambiguous_signature_matches_nothing():
    """Slurm records no target name.  Two targets that cannot be told apart
    must not have a job filed under one of them arbitrarily."""
    targets = {"a": _Remote(gateway="g", slurm=_Slurm("p", "acct")),
               "b": _Remote(gateway="g", slurm=_Slurm("p", "acct"))}
    jobs, _ = parse_squeue(_row(1, "X", partition="p", account="acct", qos="q"))
    assert match_target(jobs[0], targets, ["a", "b"]) is None


def test_token_maps_back_to_a_configured_mirror_or_not_at_all():
    assert token_to_mirror("SuCoder", MIRROR_TOKENS) == "SuCoder"
    # An ephemeral mirror is named from a git root and is never in the config.
    assert token_to_mirror("some_scratch_repo", MIRROR_TOKENS) is None


# -- the report ----------------------------------------------------------------

def _report(rows, holders=None, recorded=None, **kw):
    jobs, _ = parse_squeue("\n".join(rows))
    clusters, schedulerless = group_targets_by_cluster(TARGETS)
    return build_report(
        jobs_by_cluster={"hpc.brc": jobs}, clusters=clusters, targets=TARGETS,
        mirror_tokens=MIRROR_TOKENS, holders=holders or {}, recorded=recorded or {},
        schedulerless=schedulerless, **kw,
    )


def test_a_job_no_session_record_names_is_an_orphan():
    """The case the listing exists for: nothing local points at this job, so
    attach/release/renew cannot reach it and only scancel can."""
    report = _report([_row(39025067, "SuCoder")], holders={})
    entry = report.groups[0].entries[0]
    assert entry.orphaned
    assert "no session record" in render_report(report)


def test_a_job_a_session_record_names_is_not_an_orphan():
    report = _report([_row(39025067, "SuCoder")], holders={39025067: ["SuCoder--carleton-htc"]})
    assert not report.groups[0].entries[0].orphaned


def test_a_record_pointing_at_a_vanished_job_is_stale():
    """The reverse sweep.  Neither direction alone is enough: orphans need
    squeue-to-files, stale records need files-to-squeue."""
    report = _report(
        [_row(39025067, "SuCoder")],
        holders={39025067: ["SuCoder--carleton-htc"]},
        recorded={"SuCoder--carleton-htc": 39025067,
                  "MetricsMiscellany--carleton-htc": 38999103},
    )
    assert [(s.key, s.job_id) for s in report.stale] == [
        ("MetricsMiscellany--carleton-htc", 38999103)
    ]


def test_a_stale_record_names_the_command_that_clears_it():
    """`release` resolves the mirror through config.mirrors and the target
    through -T, so the advice has to carry both or it exits 1."""
    report = _report([], recorded={"SuCoder--carleton-htc": 39025067})
    stale = report.stale[0]
    assert (stale.mirror, stale.target, stale.clearable) == (
        "SuCoder", "carleton-htc", True
    )
    assert "sucoder -T carleton-htc release SuCoder" in render_report(report)


def test_a_stale_record_for_an_unconfigured_mirror_says_so():
    """The common case on a long-lived laptop: the job is gone, the mirror
    has been dropped from the config, and no `release` invocation can reach
    the record.  Printing `sucoder release` at it would only exit 1."""
    report = _report([], recorded={"CerealDemand--carleton-htc": 35768982})
    stale = report.stale[0]
    assert not stale.clearable
    assert stale.release_command == ""
    out = render_report(report)
    assert "sucoder release CerealDemand" not in out
    assert "mirror not configured" in out


def test_a_record_with_no_target_suffix_still_splits():
    """Older launches wrote `<mirror>.yaml` with no target half."""
    report = _report([], recorded={"SuCoder": 32922079})
    stale = report.stale[0]
    assert (stale.mirror, stale.target) == ("SuCoder", "")
    assert stale.release_command == "sucoder release SuCoder"


def test_a_shell_in_the_pane_means_the_agent_exited():
    """`exec bash -l` keeps the window alive past the agent, and the keeper
    polls has-session, so the job holds its slice with nobody home.  Session
    existence cannot distinguish this; the pane command can."""
    report = _report([_row(39025067, "SuCoder")], holders={39025067: ["SuCoder--carleton-htc"]})
    entry = report.groups[0].entries[0]
    entry.pane = "bash"
    assert entry.agent_exited
    assert "agent exited" in render_report(report)
    entry.pane = "claude"
    assert not entry.agent_exited


def test_an_unprobed_pane_is_never_reported_as_a_dead_agent():
    """A failed probe is unknown, not evidence.  Same rule as the scheduler
    queries: absence of an answer is not an answer."""
    entry = SessionEntry(job=JobRow(1, "sucoder-X", "p", "a", "q", "RUNNING", "1:00", "n1"))
    assert entry.pane is None and not entry.agent_exited


def test_fast_mode_says_the_pane_was_not_probed():
    report = _report([_row(39025067, "SuCoder")], holders={39025067: ["SuCoder--carleton-htc"]})
    assert "not probed" in render_report(report, probed=False)
    assert "not probed" not in render_report(report, probed=True)


def test_output_groups_by_target_and_names_schedulerless_ones():
    report = _report([
        _row(39025067, "SuCoder"),
        _row(38991234, "K-Aggregators", partition="savio3", account="fc_jevons",
             qos="normal", left="3-04:11", node="n0142.savio3"),
    ])
    out = render_report(report)
    assert "carleton-htc   savio4_htc / co_carleton / carleton_htc4_normal" in out
    assert "savio-node   savio3 / fc_jevons" in out
    assert "hhsurveys   no scheduler" in out
    assert out.index("carleton-htc") < out.index("savio-node")
    assert "SuCoder" in out and "39025067" in out and "11-22:30" in out


def test_a_job_matching_no_target_is_still_listed():
    """Never silently drop a job.  An unrecognised signature is exactly the
    kind of thing someone needs to see."""
    report = _report([_row(7, "SuCoder", partition="other", account="x", qos="y")])
    assert report.groups == []
    assert [e.job.job_id for e in report.unmatched] == [7]
    assert "matching no configured target" in render_report(report)


def test_no_jobs_says_so_rather_than_printing_nothing():
    report = _report([])
    assert "No SuCoder jobs found." in render_report(report)


def test_errors_are_surfaced_not_swallowed():
    report = _report([], errors=["squeue on hpc.brc failed: connection closed"])
    assert "! squeue on hpc.brc failed" in render_report(report)


# -- regressions for two bugs these tests found --------------------------------

def test_a_target_pinning_nothing_does_not_claim_every_job():
    """`savio` in the real config sets no partition, account or qos.  Matching
    on "every configured field agrees" made it a wildcard that swallowed every
    job on the cluster, so nothing was ever filed under its real target."""
    jobs, _ = parse_squeue(_row(1, "SuCoder"))
    assert match_target(jobs[0], TARGETS, ["savio"]) is None
    assert match_target(jobs[0], TARGETS, list(TARGETS)) == "carleton-htc"


def test_most_specific_target_wins():
    """savio-htc and carleton-htc share a partition and differ on account and
    qos; a target pinning only the partition must not tie with them."""
    targets = {
        "broad": _Remote(gateway="g", slurm=_Slurm("savio4_htc")),
        "exact": _Remote(gateway="g", slurm=_Slurm("savio4_htc", "co_carleton", "carleton_htc4_normal")),
    }
    jobs, _ = parse_squeue(_row(1, "SuCoder"))
    assert match_target(jobs[0], targets, ["broad", "exact"]) == "exact"


def test_a_slurm_array_element_is_not_read_as_a_job_id():
    """int() accepts underscores as digit separators, so int("39025067_3") is
    390250673 -- an array element would become a plausible id belonging to
    nobody, and the listing would report a job that does not exist."""
    jobs, bad = parse_squeue("39025067_3|sucoder-X|p|a|q|RUNNING|1:00|n1")
    assert jobs == []
    assert bad == ["39025067_3|sucoder-X|p|a|q|RUNNING|1:00|n1"]


# -- the pane probe, driven under bash ------------------------------------------
#
# This is the piece that was wrong first time.  `#{pane_current_command}` alone
# reports the pane's SHELL: measured on a live session, a running `claude`
# showed as `bash`, because the window command is `bash -lc '<agent>; exec
# bash -l'` and the agent is its child.  A probe built on that flags every
# working agent as exited -- and the flag's whole purpose is to say a slice is
# safe to release, so a false positive there costs somebody their session.

import os
import shutil
import subprocess

import pytest

from sucoder.sessions_report import PANE_PROBE_SH

_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _probe(tmp_path, panes, children, session="sucoder-K", socket=""):
    """Drive PANE_PROBE_SH with stubbed tmux and ps.

    *panes* is the ``pane_pid pane_current_command`` table tmux reports (an
    empty list makes tmux fail); *children* maps a pane pid to its child
    commands -- a single string, or a list in the order ``ps`` prints them.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("head",):
        found = shutil.which(name)
        if found:
            (bin_dir / name).symlink_to(found)
    if panes:
        rows = "\n".join(panes)
        (bin_dir / "tmux").write_text(f"#!/bin/sh\ncat <<'ROWS'\n{rows}\nROWS\n")
    else:
        (bin_dir / "tmux").write_text("#!/bin/sh\nexit 1\n")
    # The real call is `ps -o comm= --ppid <pid>`: flag and value are
    # separate arguments, which the first version of this stub got wrong.
    cases = ""
    for pid, comm in children.items():
        kids = [comm] if isinstance(comm, str) else list(comm)
        body = "; ".join(f'echo "{k}"' for k in kids)
        cases += f"  {pid}) {body} ;;\n"
    (bin_dir / "ps").write_text(
        "#!/bin/sh\np=\nwhile [ $# -gt 0 ]; do\n"
        '  case "$1" in --ppid) shift; p="$1" ;; --ppid=*) p="${1#--ppid=}" ;; esac\n'
        "  shift\ndone\n"
        'case "$p" in\n' + cases + "  *) : ;;\nesac\n"
    )
    for f in bin_dir.iterdir():
        if not f.is_symlink():
            f.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    return subprocess.run(
        ["bash", "-c", PANE_PROBE_SH, "_", session, socket],
        capture_output=True, text=True, env=env,
    ).stdout.strip()


@_bash
def test_probe_reports_the_agent_not_the_shell_hosting_it(tmp_path):
    """The regression that matters: tmux says `bash`, the child is the agent."""
    assert _probe(tmp_path, ["316429 bash"], {316429: "claude"}) == "claude"


@_bash
def test_probe_reports_a_bare_shell_as_a_shell(tmp_path):
    """`exec bash -l` after a clean /exit: no child, so the pane's own command
    stands and the entry flags as `agent exited`."""
    assert _probe(tmp_path, ["316429 bash"], {}) == "bash"


@_bash
def test_probe_prefers_a_non_shell_pane_over_a_shell_one(tmp_path):
    """A session with a spare shell window beside the agent's is still live."""
    assert _probe(tmp_path, ["316429 bash", "317637 bash"], {317637: "claude"}) == "claude"


@_bash
def test_probe_looks_past_a_shell_child_to_the_agent(tmp_path):
    """The regression this replaced: `ps ... | head -n 1` took only the
    FIRST child, so a shell-named one listed ahead of the agent made a live
    session read as bare.  Measured against a real tmux server, a pane whose
    children were (sh, sleep) reported `sh` -- an `agent exited` flag on a
    working session, which is the false positive that costs somebody their
    slice.  Every child is considered now, non-shells first.
    """
    assert _probe(tmp_path, ["316429 bash"], {316429: ["sh", "claude"]}) == "claude"


@_bash
def test_probe_reports_a_shell_child_when_that_is_all_there_is(tmp_path):
    """A shell child is still a shell: the human ran `bash` at the prompt
    after the agent exited.  Nothing non-shell is running, so the flag
    stands."""
    assert _probe(tmp_path, ["316429 bash"], {316429: ["sh", "bash"]}) == "sh"


@_bash
def test_probe_says_nothing_when_tmux_fails(tmp_path):
    """Unknown must stay unknown: SessionEntry.agent_exited is False for
    pane=None, so a failed probe never reads as a dead agent."""
    assert _probe(tmp_path, [], {}) == ""
