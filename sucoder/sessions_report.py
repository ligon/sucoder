"""``sucoder sessions``: every SuCoder job on a cluster, in one listing.

Enumerated from ``squeue``, deliberately, not from the session files.

``~/.sucoder/sessions/<mirror>--<target>.yaml`` holds exactly one
``slurm_job_id``, so a file-first listing inherits that one-slot limit and
shows precisely the jobs that are *not* the problem.  The interesting ones
are those the files have lost: a record overwritten by a later launch,
corrupted (``RemoteSession.load`` swallows parse errors and returns a blank
session), or written under a different ``-T`` spelling.  Those jobs are
unreachable by every other SuCoder command -- ``attach``, ``release`` and
``renew`` all resolve through the same file -- and can only be ``scancel``ed
by hand.  Slurm knows about them, because every confined launch submits with
``--job-name=sucoder-<token>`` (``mirror._build_sbatch_command``).

Two consequences shape the module:

- *One query per cluster, not per target.*  Several targets routinely share
  one gateway, one ``$HOME`` and one scheduler, differing only in
  partition/account/qos, so they are grouped by :func:`cluster_key` and one
  ``squeue --me`` answers for all of them.  The partition/account/qos triple
  then maps each job back to the target that launched it.
- *A job outliving its agent is the normal case, not an error.*  The window
  command ends in ``exec bash -l`` (``mirror._build_remote_agent_cmd_str``)
  so the tmux window survives a clean ``/exit``; the batch body's keeper
  polls ``has-session``, so the allocation then runs to its full ``--time``.
  Nothing else reports that, which is why :class:`JobRow` carries what the
  pane is actually running rather than merely whether a session exists.

Everything here is pure: the caller does the SSH and passes text in.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

JOB_NAME_PREFIX = "sucoder-"

# What the caller asks squeue for.  Pipe-delimited because a job name is
# user-controlled and node lists contain commas; '|' is rejected by Slurm in
# a job name, and our names are sanitized anyway.
SQUEUE_FORMAT = "%i|%j|%P|%a|%q|%T|%L|%N"


# Resolve what a tmux session is really running, for one job.
#
# NOT ``#{pane_current_command}`` on its own.  The window command is
# ``bash -lc '<agent>; exec bash -l'``, and tmux reports the pane's shell:
# measured on a live session, a running ``claude`` showed as ``bash``.  A
# probe built on that would flag every working agent as exited -- and the
# whole point of the flag is to tell you a slice is safe to release, so a
# false positive there costs somebody their session.
#
# The shell's *child* is the real signal: ``bash -l`` idle has none, ``bash``
# running the agent has one.  Report the first non-shell child across the
# session's panes, else the pane's own command (so a genuinely bare shell
# reads as ``bash``), else nothing at all -- an unanswered probe must stay
# unknown rather than become evidence.
#
# *All* of a pane shell's children are considered, not just the first.
# Taking ``ps ... | head -n 1`` meant that a shell-named child listed ahead
# of the agent (a backgrounded helper, anything the agent spawns beside
# itself) made a live session read as a bare shell -- the same
# false-positive that ``#{pane_current_command}`` produced, and just as
# costly, since the flag is what tells somebody a slice is safe to release.
#
# Takes ``<session> [socket]``; an empty socket means the default tmux
# server (unconfined launches do not use a dedicated one).
PANE_PROBE_SH = (
    'sess="$1"; sock="$2"; '
    'if [ -n "$sock" ]; then set -- -L "$sock"; else set --; fi; '
    'tmux "$@" list-panes -s -t "$sess" '
    '-F "#{pane_pid} #{pane_current_command}" 2>/dev/null | '
    '{ first=""; pick=""; '
    'while read -r p cmd; do '
    'kids=$(ps -o comm= --ppid "$p" 2>/dev/null); '
    'c=$(printf "%s\n" "$kids" | '
    'grep -vxE "bash|sh|zsh|dash|ksh|fish|-bash" | head -n 1); '
    '[ -z "$c" ] && c=$(printf "%s\n" "$kids" | head -n 1); '
    'r="${c:-$cmd}"; '
    '[ -z "$first" ] && first="$r"; '
    'case "$r" in bash|sh|zsh|dash|ksh|fish|-bash) ;; '
    '*) [ -z "$pick" ] && pick="$r" ;; esac; '
    'done; echo "${pick:-$first}"; }'
)


# Enumerate SuCoder tmux sessions on one host.  This is the login-node
# analogue of ``squeue --me --name=sucoder-<token>``: the tmux server is
# the registry, and the ``sucoder-`` prefix is the same filter the job
# names carry, so a session whose local record was lost is still found.
# Unconfined launches use the DEFAULT tmux socket, so no ``-L`` here.
LOGIN_SESSION_SH = (
    'tmux list-sessions -F "#{session_name}" 2>/dev/null || true'
)


# The same enumeration, but each name followed by a TAB and the command its
# pane runs -- the two queries fused into ONE remote shell.
#
# Splitting them cost a full extra ssh *session open* per host, and on a BRC
# login node a session open is ~10s of remote setup before the command is
# even exec'd (measured 2026-09-19), so listing two sessions paid ~20s to
# answer what one round trip answers.  The loop runs remote-side; the
# ``sucoder-`` filter deliberately stays local, in `parse_login_sessions`,
# so the prefix rule lives in exactly one place.
LOGIN_SESSION_PANES_SH = (
    'tmux list-sessions -F "#{session_name}" 2>/dev/null | '
    'while read -r s; do '
    'printf "%s\t" "$s"; '
    'bash -c ' + shlex.quote(PANE_PROBE_SH) + ' _ "$s" "" 2>/dev/null; '
    'echo; '
    'done || true'
)


# Two scripts, one session.
#
# A gateway is both "the cluster's scheduler host" and "a login host worth
# sweeping for tmux sessions", so the listing asked it two questions -- and
# on a ``MaxSessions 1`` master the second request is refused outright,
# which then costs a serial reconnect and a re-query.  Ask once instead:
# the answers are separated by a marker no shell output of ours contains.
#
# The FIRST script's exit status is what survives, because that is the
# scheduler query whose failure the caller must report; the sweep is
# advisory and ends in ``|| true`` anyway.
FUSED_SECTION_MARKER = "===sucoder-section==="


def fuse_scripts(first: str, second: str) -> str:
    """Return one remote script running *first* then *second*, in that order."""
    return (
        f"{first}\n"
        "__sucoder_rc=$?\n"
        f"printf '%s\\n' {shlex.quote(FUSED_SECTION_MARKER)}\n"
        f"{second}\n"
        "exit $__sucoder_rc\n"
    )


def split_fused(text: str) -> Tuple[str, Optional[str]]:
    """Split fused output into its two halves.

    The second half is ``None`` when the marker never printed -- the remote
    shell died before reaching it, or the transport did.  ``None`` means
    *unanswered*, and the caller must ask that host again rather than
    record an absence the query never established.
    """
    head, sep, tail = text.partition(FUSED_SECTION_MARKER)
    if not sep:
        return text, None
    return head, tail.lstrip("\n")


@dataclass(frozen=True)
class LoginSession:
    """A SuCoder tmux session found on a login (or direct-SSH) host.

    There is no scheduler behind these, and therefore no walltime: a
    confined job's allocation ends at its ``--time`` and takes the tmux
    server with it, but a login-node session outlives everything until
    the node reboots.  The window command ends in ``exec bash -l`` just
    as the confined one does, so the same clean ``/exit`` leaves the same
    orphan -- with nothing to ever reap it.  That is why these are worth
    listing even though they hold no allocation.
    """

    host: str
    name: str                      # sucoder-<token>
    target: str
    pane: Optional[str] = None     # command running in the pane

    @property
    def token(self) -> str:
        return self.name[len(JOB_NAME_PREFIX):]

    @property
    def agent_exited(self) -> bool:
        """Same rule as a job entry: unknown is not evidence."""
        return self.pane is not None and self.pane in _SHELLS


def parse_tmux_sessions(text: str) -> List[str]:
    """SuCoder session names from ``tmux list-sessions -F '#{session_name}'``.

    Everything not carrying the ``sucoder-`` prefix is somebody else's
    tmux and is dropped, exactly as ``parse_squeue`` drops jobs whose name
    lacks it.
    """
    names: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        # `list-sessions` prints "name: N windows (...)" without -F, and a
        # stray banner line can reach us through a login shell; take only
        # bare names carrying the prefix.
        if line.startswith(JOB_NAME_PREFIX) and " " not in line:
            names.append(line.rstrip(":"))
    return names


def parse_login_sessions(text: str) -> Tuple[List[str], Dict[str, str]]:
    """Split ``LOGIN_SESSION_PANES_SH`` output into (names, panes).

    Each line is ``<session name>\t<pane command>``; the pane half is empty
    when the probe found nothing, which stays absent from the mapping so an
    unanswered probe reads as unknown rather than as a dead agent -- the
    same rule the job-entry probe follows.

    Filtering is delegated to :func:`parse_tmux_sessions` so the
    ``sucoder-`` prefix rule is not restated here.
    """
    rows = [line.partition("\t") for line in text.splitlines()]
    names = parse_tmux_sessions("\n".join(name for name, _, _ in rows))
    wanted = set(names)
    panes: Dict[str, str] = {}
    for name, _, pane in rows:
        key = name.strip().rstrip(":")
        if key in wanted and pane.strip():
            panes[key] = pane.strip()
    return names, panes


@dataclass(frozen=True)
class JobRow:
    """One Slurm job that SuCoder launched."""

    job_id: int
    name: str                      # sucoder-<token>
    partition: str
    account: str
    qos: str
    state: str
    time_left: str
    node: str

    @property
    def token(self) -> str:
        """The sanitized mirror token embedded in the job name."""
        return self.name[len(JOB_NAME_PREFIX):]


@dataclass
class SessionEntry:
    """A job as reported, joined to whatever local state knows about it."""

    job: JobRow
    mirror: Optional[str] = None       # configured mirror name, if known
    target: Optional[str] = None       # configured target name, if known
    session_keys: Sequence[str] = ()   # <mirror>--<target> records naming it
    pane: Optional[str] = None         # command running in the tmux pane
    wip: Optional[str] = None          # age of the last WIP snapshot

    @property
    def orphaned(self) -> bool:
        """No session record names this job, so no command can reach it."""
        return not self.session_keys

    @property
    def agent_exited(self) -> bool:
        """The job is alive but its pane is a shell: nobody is home.

        Unknown (``pane is None``) is deliberately not this: a failed probe
        must not be reported as a dead agent.
        """
        return self.pane is not None and self.pane in _SHELLS


_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh", "fish", "-bash", "login"})


@dataclass
class StaleRecord:
    """A session file pointing at a job the scheduler no longer has.

    *mirror* and *target* are the halves of ``key``, and *clearable* says
    whether ``release`` can actually reach this record: it resolves the
    mirror through ``config.mirrors`` and the target through ``-T``, so a
    record naming a mirror that is no longer configured cannot be cleared
    by any invocation.  Saying so beats printing advice that exits 1.
    """

    key: str
    job_id: int
    mirror: str = ""
    target: str = ""
    clearable: bool = False

    @property
    def release_command(self) -> str:
        """The exact invocation that clears this record, if one exists."""
        if not self.clearable:
            return ""
        if self.target:
            return f"sucoder -T {self.target} release {self.mirror}"
        return f"sucoder release {self.mirror}"


@dataclass
class TargetGroup:
    name: str
    signature: str                 # "partition / account / qos", for the header
    entries: List[SessionEntry] = field(default_factory=list)


@dataclass
class Report:
    groups: List[TargetGroup] = field(default_factory=list)
    schedulerless: List[str] = field(default_factory=list)
    stale: List[StaleRecord] = field(default_factory=list)
    unmatched: List[SessionEntry] = field(default_factory=list)
    logins: List[LoginSession] = field(default_factory=list)
    login_probed: bool = False
    errors: List[str] = field(default_factory=list)


def cluster_key(remote) -> Optional[str]:
    """What distinguishes one cluster from another, for grouping.

    The gateway: targets sharing it share a ``$HOME`` and a scheduler, so
    one ``squeue --me`` covers them all.  A direct-SSH target has no
    gateway and no scheduler, and returns ``None``.
    """
    if getattr(remote, "slurm", None) is None:
        return None
    return getattr(remote, "gateway", None) or getattr(remote, "host", None)


def group_targets_by_cluster(targets: Mapping[str, object]) -> Tuple[Dict[str, List[str]], List[str]]:
    """Split configured targets into ``{cluster: [target, ...]}`` and the rest.

    The second element is the targets with no scheduler to query; they are
    listed so their absence from the report is explicit rather than silent.
    """
    clusters: Dict[str, List[str]] = {}
    schedulerless: List[str] = []
    for name in sorted(targets):
        key = cluster_key(targets[name])
        if key is None:
            schedulerless.append(name)
        else:
            clusters.setdefault(key, []).append(name)
    return clusters, schedulerless


def login_hosts_for(target_name: str, remote, recorded_nodes: Mapping[str, str]) -> List[str]:
    """Hosts to ask for tmux sessions belonging to *target_name*.

    A gateway round-robins across several login nodes, and at this site the
    node you land on depends on the account class -- condo and FCA accounts
    reach different ones -- so reconnecting to the gateway can never see
    every node, and a hardcoded list would rot.  Take the nodes the session
    records actually pin (``RemoteSession.login_nodes_for_target``) and add
    the gateway itself, which resolves to whichever node this account gets.
    A direct-SSH target has no gateway and is simply its own host.

    De-duplicated, order stable, so a listing costs one connection per
    distinct host rather than one per record.
    """
    hosts: List[str] = []
    for node in recorded_nodes.values():
        if node and node not in hosts:
            hosts.append(node)
    own = getattr(remote, "gateway", None) or getattr(remote, "host", None)
    if own and own not in hosts:
        hosts.append(own)
    return hosts


def target_signature(remote) -> str:
    """``partition / account / qos``, omitting the parts a target leaves unset."""
    slurm = getattr(remote, "slurm", None)
    if slurm is None:
        return "no scheduler"
    parts = [getattr(slurm, attr, None) for attr in ("partition", "account", "qos")]
    shown = [p for p in parts if p]
    return " / ".join(shown) if shown else "(scheduler defaults)"


def parse_squeue(text: str) -> Tuple[List[JobRow], List[str]]:
    """Parse ``squeue --noheader -o SQUEUE_FORMAT`` output.

    Returns SuCoder's jobs and the lines that could not be parsed.  A
    malformed line is reported rather than dropped: this listing exists to
    find things the local state has lost, so silently discarding a row
    would defeat it.
    """
    jobs: List[JobRow] = []
    bad: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split("|")
        if len(fields) != 8:
            bad.append(line)
            continue
        job_id, name, partition, account, qos, state, time_left, node = (
            f.strip() for f in fields
        )
        if not name.startswith(JOB_NAME_PREFIX):
            continue          # someone else's job in the same account
        # ``isdigit`` rather than a bare ``int()``: Python accepts
        # underscores as digit separators, so ``int("39025067_3")`` is
        # 390250673 -- a Slurm array element would silently become a
        # plausible-looking job id that belongs to nobody.
        if not job_id.isdigit():
            bad.append(line)
            continue
        numeric = int(job_id)
        jobs.append(JobRow(
            job_id=numeric, name=name, partition=partition, account=account,
            qos=qos, state=state, time_left=time_left, node=node,
        ))
    return jobs, bad


def match_target(job: JobRow, targets: Mapping[str, object], candidates: Sequence[str]) -> Optional[str]:
    """Which configured target launched *job*, by partition/account/qos.

    Slurm records no target name, so the submission parameters are the
    signature.  The most specific match wins: a target that pins
    partition, account and qos beats one that pins only a partition, and a
    target that pins *nothing* (every field left to the cluster default)
    matches nothing at all rather than claiming every job on the cluster.

    Returns ``None`` when nothing matches, and when two targets are equally
    specific -- guessing between them would file a job under a heading that
    is simply wrong, and a job shown with no target is more useful than a
    job shown under the wrong one.
    """
    best: List[Tuple[int, str]] = []
    for name in candidates:
        slurm = getattr(targets[name], "slurm", None)
        if slurm is None:
            continue
        score = 0
        for attr, value in (
            ("partition", job.partition), ("account", job.account), ("qos", job.qos),
        ):
            configured = getattr(slurm, attr, None)
            if not configured:
                continue
            if configured != value:
                break
            score += 1
        else:
            if score:
                best.append((score, name))
    if not best:
        return None
    top = max(score for score, _ in best)
    winners = [name for score, name in best if score == top]
    return winners[0] if len(winners) == 1 else None


def token_to_mirror(token: str, mirror_tokens: Mapping[str, str]) -> Optional[str]:
    """Reverse ``sanitize_session_token`` via the configured mirrors.

    *mirror_tokens* maps mirror name -> token.  Sanitizing is lossy, so this
    is a lookup rather than an inversion; an ephemeral mirror (named from a
    git root directory and never in the config) simply will not be found,
    and the caller shows the token itself.
    """
    for mirror, tok in mirror_tokens.items():
        if tok == token:
            return mirror
    return None


def build_report(
    *,
    jobs_by_cluster: Mapping[str, Sequence[JobRow]],
    clusters: Mapping[str, Sequence[str]],
    targets: Mapping[str, object],
    mirror_tokens: Mapping[str, str],
    holders: Mapping[int, Sequence[str]],
    recorded: Mapping[str, int],
    schedulerless: Sequence[str] = (),
    errors: Sequence[str] = (),
) -> Report:
    """Join scheduler truth to local state.

    *holders* maps job id -> the ``<mirror>--<target>`` session keys naming
    it (``RemoteSession.holders_of_job``); an empty list is what makes a job
    an orphan.  *recorded* is the reverse -- every session key and the job it
    claims -- which yields the stale records: files pointing at jobs the
    scheduler no longer has.  Both directions matter, and neither alone is
    enough.
    """
    report = Report(schedulerless=list(schedulerless), errors=list(errors))
    live: set = set()
    by_target: Dict[str, List[SessionEntry]] = {}

    for cluster, names in clusters.items():
        for job in jobs_by_cluster.get(cluster, ()):
            live.add(job.job_id)
            target = match_target(job, targets, names)
            entry = SessionEntry(
                job=job,
                mirror=token_to_mirror(job.token, mirror_tokens),
                target=target,
                session_keys=list(holders.get(job.job_id, ())),
            )
            if target is None:
                report.unmatched.append(entry)
            else:
                by_target.setdefault(target, []).append(entry)

    for name in sorted(by_target):
        report.groups.append(TargetGroup(
            name=name,
            signature=target_signature(targets[name]),
            entries=sorted(by_target[name], key=lambda e: e.job.job_id),
        ))

    for key, job_id in sorted(recorded.items()):
        if job_id not in live:
            # ``<mirror>--<target>``, or a bare ``<mirror>`` from an older
            # launch that wrote no target suffix.  ``rsplit`` so a mirror
            # whose own name contains ``--`` still splits at the target.
            mirror, sep, target = key.rpartition("--")
            if not sep:
                mirror, target = key, ""
            report.stale.append(StaleRecord(
                key=key, job_id=job_id, mirror=mirror, target=target,
                # ``release`` looks the mirror up in ``config.mirrors``;
                # *mirror_tokens* is keyed by exactly that mapping.
                clearable=mirror in mirror_tokens,
            ))

    return report


def _entry_row(entry: SessionEntry) -> Tuple[str, ...]:
    job = entry.job
    flags = []
    if entry.orphaned:
        flags.append("no session record")
    if entry.agent_exited:
        flags.append("agent exited")
    return (
        entry.mirror or job.token,
        str(job.job_id),
        job.state,
        job.time_left or "-",
        job.node or "-",
        entry.pane or "-",
        entry.wip or "-",
        ("! " + ", ".join(flags)) if flags else "",
    )


def render_report(report: Report, *, probed: bool = True) -> str:
    """One block per target, then the things that fit under no target."""
    rows: List[Tuple[str, ...]] = []
    for group in report.groups:
        rows.extend(_entry_row(e) for e in group.entries)
    rows.extend(_entry_row(e) for e in report.unmatched)
    widths = [0] * 7
    for row in rows:
        for i in range(7):
            widths[i] = max(widths[i], len(row[i]))

    def line(row: Tuple[str, ...]) -> str:
        cells = [row[i].ljust(widths[i]) for i in range(7)]
        return ("  " + "  ".join(cells) + ("  " + row[7] if row[7] else "")).rstrip()

    out: List[str] = []
    for group in report.groups:
        out.append(f"{group.name}   {group.signature}")
        for entry in group.entries:
            out.append(line(_entry_row(entry)))
        out.append("")

    if report.unmatched:
        out.append("jobs matching no configured target")
        for entry in report.unmatched:
            out.append(line(_entry_row(entry)))
        out.append("")

    for name in report.schedulerless:
        sessions = [s for s in report.logins if s.target == name]
        if sessions:
            out.append(f"{name}   no scheduler (login-node sessions)")
            width = max(len(s.name) for s in sessions)
            for sess in sessions:
                flag = "  ! agent exited" if sess.agent_exited else ""
                out.append(
                    f"  {sess.name.ljust(width)}  {sess.host}  "
                    f"{sess.pane or '-'}{flag}"
                )
        elif report.login_probed:
            out.append(f"{name}   no scheduler; no login-node sessions")
        else:
            out.append(f"{name}   no scheduler (login-node sessions not inspected)")
    if report.schedulerless:
        out.append("")
        if any(s.agent_exited for s in report.logins):
            out.append(
                "  A login-node session has no walltime: nothing reaps it when the\n"
                "  agent exits, unlike a job, whose allocation ends at its --time."
            )
            out.append("")

    if not report.groups and not report.unmatched:
        out.append("No SuCoder jobs found.")
        out.append("")

    if report.stale:
        out.append("stale session records (the job is gone; the record is not):")
        width = max(len(r.key) for r in report.stale)
        for record in report.stale:
            hint = record.release_command or "mirror not configured; edit the record by hand"
            out.append(f"  {record.key.ljust(width)}  -> {record.job_id}   {hint}")
        out.append("")
        if any(not r.clearable for r in report.stale):
            out.append(
                "  (`release` resolves the mirror through config.mirrors and the "
                "target through -T,\n   so a record whose mirror is no longer "
                "configured cannot be cleared by it.)"
            )
            out.append("")

    if not probed:
        out.append(
            "(--fast: the pane column is not probed, so 'agent exited' is not reported)"
        )
        out.append("")

    for message in report.errors:
        out.append(f"! {message}")

    return "\n".join(out).rstrip("\n") + "\n"
