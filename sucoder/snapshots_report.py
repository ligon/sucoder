"""``sucoder snapshots``: every WIP snapshot on a cluster's mirrors, and
whether its job is over.

The launch-independent half of issue 14.  Retention inside the prepare
script retires ended jobs' snapshots, but it runs only when a job is
launched against that mirror: stop using a mirror and its refs are
permanent, and each one pins its snapshot tree reachable forever.  This
module is what sweeps them from the laptop, on whatever schedule the human
keeps, with no allocation involved.

What decides a snapshot's fate is *when its job ended*, not how old the ref
is.  Ref age is unsafe on its own: the snapshotter skips the ref update
when the tree is unchanged, and a confined window ends in ``exec bash -l``,
so the usual case is an agent that exited early while the keeper holds the
allocation for its full ``--time`` -- the ref then sits frozen at the last
change while the job is still alive, and any age threshold under the
allocation length deletes a live job's only snapshot.  ``sacct`` gives the
better clock and answers for dead jobs, so the order of precedence is:

1. ``sacct`` says the job is still going: keep, whatever the ref's age.
2. ``sacct`` says it ended, and when: retire once that is more than the
   recovery window ago (default seven days -- a fact about how long a
   human takes to notice, which does not scale with job length).
3. ``sacct`` cannot say (aged out of accounting, no accounting at all):
   fall back to ref age, against a threshold that *must* exceed the
   cluster's longest allocation: the longest configured ``slurm.time``
   plus the window.  Derived, not hand-set, so the common case needs no
   new configuration.
4. Nothing can answer -- a target without a finite ``--time`` means the
   fallback has no safe bound: keep, and say why.  Deleting recovery data
   on the strength of a failed query is the one outcome worse than
   keeping too much, the same rule the pane probe follows.

Everything here is pure: the caller does the SSH and passes text in.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

DAY = 86400
DEFAULT_RECOVERY_WINDOW_DAYS = 7

# Slurm's terminal job states, the ones after which ``End`` means something.
# Anything else -- RUNNING, PENDING, COMPLETING, SUSPENDED, REQUEUED and
# the rarer transitional ones -- is treated as still going.  Being wrong in
# that direction keeps a ref; being wrong in the other deletes somebody's
# work, so the list names the ended states and not the live ones.
TERMINAL_STATES = frozenset({
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
    "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT",
})

WIP_PREFIX = "refs/sucoder/"

# Enumerate every snapshot ref on every mirror under the roots given as
# arguments, then ask sacct about the jobs they name -- one remote shell,
# because a login-node session open costs ~10s before the command runs.
#
# Mirrors come from the filesystem, not the config: a mirror dropped from
# the config is exactly the one nothing else can reach (``release``
# resolves through config.mirrors), and its refs are the ones that would
# otherwise be permanent.  A linked worktree is skipped (its git dir is
# not its common dir): its refs are the main checkout's, and listing it
# would show the same snapshot under a second name.  Two directory
# entries for one repository (a symlink beside its target) are listed
# once, by the real git dir, under whichever name sorts first -- the glob
# runs without a trailing slash and under LC_ALL=C so that a name that is
# a prefix of another always comes first, on every runner.
#
# ``SLURM_TIME_FORMAT=%s`` asks sacct for epoch seconds so no time-zone
# arithmetic is done on this side; the cluster's offset is sent anyway, for
# a Slurm that ignores the variable and prints ISO local time.  A job id is
# taken from a per-job ref's name, or from a legacy shared ref's subject
# ("WIP snapshot <date> job <ID>"), the same two rules the prepare script
# applies.
SNAPSHOT_LIST_SH = r'''printf 'NOW\t%s\t%s\n' "$(date +%s)" "$(date +%z)"
export LC_ALL=C
seen=" "; jobs=""
for root in "$@"; do
  for m in "$root"/*; do
    [ -d "$m" ] || continue
    gd=$(git -C "$m" rev-parse --git-dir 2>/dev/null) || continue
    cd_=$(git -C "$m" rev-parse --git-common-dir 2>/dev/null) || continue
    [ "$gd" = "$cd_" ] || continue
    gd=$(cd "$m" && cd "$gd" && pwd -P) || continue
    case "$seen" in *" $gd "*) continue ;; esac
    seen="$seen$gd "
    git -C "$m" for-each-ref \
        --format="REF%09$m%09%(refname)%09%(objectname)%09%(committerdate:unix)%09%(subject)" \
        refs/sucoder/wip-job/ refs/sucoder/wip/ 2>/dev/null |
    while IFS= read -r line; do
      printf '%s\n' "$line"
      ref=$(printf '%s' "$line" | cut -f3); subj=$(printf '%s' "$line" | cut -f6-)
      case "$ref" in
        refs/sucoder/wip-job/*) id=${ref##*/} ;;
        *) id=$(printf '%s\n' "$subj" | sed -n 's/^.*[[:space:]]job[[:space:]]\{1,\}\([0-9][0-9]*\)[[:space:]]*$/\1/p') ;;
      esac
      case "$id" in ''|*[!0-9]*) ;; *) printf 'JOB\t%s\n' "$id" ;; esac
    done
  done
done > "${TMPDIR:-/tmp}/sucoder-snapshots.$$"
grep '^REF' "${TMPDIR:-/tmp}/sucoder-snapshots.$$"
jobs=$(grep '^JOB' "${TMPDIR:-/tmp}/sucoder-snapshots.$$" | cut -f2 | sort -u | paste -sd, -)
rm -f "${TMPDIR:-/tmp}/sucoder-snapshots.$$"
if command -v sacct >/dev/null 2>&1; then
  printf 'SACCT-AVAILABLE\t1\n'
  if [ -n "$jobs" ]; then
    out=$(SLURM_TIME_FORMAT=%s sacct -j "$jobs" -X -n -P -o JobID,State,End,Partition,Account,QOS 2>&1); rc=$?
    printf 'SACCT-RC\t%s\n' "$rc"
    printf '%s\n' "$out" | while IFS= read -r l; do [ -n "$l" ] && printf 'SACCT\t%s\n' "$l"; done
  fi
else
  printf 'SACCT-AVAILABLE\t0\n'
fi
'''


@dataclass(frozen=True)
class SnapshotRef:
    path: str                      # the mirror's directory on the cluster
    ref: str
    sha: str
    committed: int                 # unix seconds, the snapshot's committer date
    subject: str

    @property
    def mirror(self) -> str:
        return self.path.rstrip("/").rsplit("/", 1)[-1]

    @property
    def short_ref(self) -> str:
        return self.ref[len(WIP_PREFIX):] if self.ref.startswith(WIP_PREFIX) else self.ref

    @property
    def job_id(self) -> Optional[int]:
        """From the ref name for a per-job ref, from the subject for a legacy one."""
        if self.ref.startswith(WIP_PREFIX + "wip-job/"):
            tail = self.ref.rsplit("/", 1)[-1]
            return int(tail) if tail.isdigit() else None
        words = self.subject.split()
        if len(words) >= 2 and words[-2] == "job" and words[-1].isdigit():
            return int(words[-1])
        return None


@dataclass(frozen=True)
class JobRecord:
    job_id: int
    state: str                     # first word: "CANCELLED by 123" -> "CANCELLED"
    end: Optional[int]             # unix seconds; None when Slurm says Unknown
    partition: str = ""
    account: str = ""
    qos: str = ""

    @property
    def ended(self) -> bool:
        return self.state in TERMINAL_STATES


@dataclass
class Listing:
    now: Optional[int] = None
    tz_offset: str = ""
    sacct_available: Optional[bool] = None    # None: the script never said
    sacct_rc: Optional[int] = None
    refs: List[SnapshotRef] = field(default_factory=list)
    jobs: Dict[int, JobRecord] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def sacct_answered(self) -> bool:
        """Whether the accounting answer can be read at all."""
        return bool(self.sacct_available) and (self.sacct_rc in (None, 0))


def _parse_end(value: str, tz_offset: str) -> Optional[int]:
    """sacct's End: epoch seconds when SLURM_TIME_FORMAT was honoured, else
    ISO local time (``2026-09-18T17:54:02``) placed with the cluster's own
    offset.  ``Unknown``/``None``/empty mean the job has not ended."""
    v = value.strip()
    if not v or v in ("Unknown", "None", "N/A"):
        return None
    if v.isdigit():
        return int(v)
    from datetime import datetime, timedelta, timezone
    try:
        naive = datetime.fromisoformat(v)
    except ValueError:
        return None
    if naive.tzinfo is not None:
        return int(naive.timestamp())
    off = tz_offset.strip()
    if len(off) == 5 and off[0] in "+-" and off[1:].isdigit():
        sign = 1 if off[0] == "+" else -1
        delta = timedelta(hours=int(off[1:3]), minutes=int(off[3:5])) * sign
        return int(naive.replace(tzinfo=timezone(delta)).timestamp())
    return None


def parse_listing(text: str) -> Listing:
    """Read what :data:`SNAPSHOT_LIST_SH` printed.  Malformed lines are
    reported, never silently dropped."""
    listing = Listing()
    for raw in text.splitlines():
        if not raw.strip():
            continue
        kind, _, rest = raw.partition("\t")
        if kind == "NOW":
            now, _, tz = rest.partition("\t")
            listing.now = int(now) if now.strip().isdigit() else None
            listing.tz_offset = tz.strip()
        elif kind == "REF":
            parts = rest.split("\t", 4)
            if len(parts) < 4 or not parts[3].strip().isdigit():
                listing.errors.append(f"could not parse snapshot row: {raw}")
                continue
            path, ref, sha, when = parts[:4]
            subject = parts[4] if len(parts) == 5 else ""
            listing.refs.append(SnapshotRef(path, ref, sha, int(when), subject))
        elif kind == "SACCT-AVAILABLE":
            listing.sacct_available = rest.strip() == "1"
        elif kind == "SACCT-RC":
            listing.sacct_rc = int(rest) if rest.strip().isdigit() else 1
        elif kind == "SACCT":
            cols = rest.split("|")
            if len(cols) < 3 or not cols[0].strip().isdigit():
                # A non-row here is sacct's own complaint (rc != 0); it is
                # surfaced through the rc, not as a parse error.
                if listing.sacct_rc not in (None, 0):
                    listing.errors.append(f"sacct: {rest.strip()}")
                else:
                    listing.errors.append(f"could not parse sacct row: {rest}")
                continue
            job = int(cols[0])
            state = cols[1].strip().split()[0] if cols[1].strip() else ""
            listing.jobs[job] = JobRecord(
                job_id=job, state=state, end=_parse_end(cols[2], listing.tz_offset),
                partition=cols[3].strip() if len(cols) > 3 else "",
                account=cols[4].strip() if len(cols) > 4 else "",
                qos=cols[5].strip() if len(cols) > 5 else "",
            )
        else:
            listing.errors.append(f"unexpected line from the cluster: {raw}")
    return listing


@dataclass(frozen=True)
class Decision:
    snapshot: SnapshotRef
    retire: bool
    state: str                     # what is known about the job, for the listing
    reason: str                    # why keep / why retire
    age: Optional[int] = None      # seconds since the snapshot was taken


def _days(seconds: int) -> str:
    if seconds < DAY:
        hours = max(seconds // 3600, 0)
        return f"{hours}h"
    return f"{seconds // DAY}d"


def decide(
    listing: Listing, *,
    window_days: float = DEFAULT_RECOVERY_WINDOW_DAYS,
    fallback_seconds: Optional[int],
    fallback_reason: str = "",
) -> List[Decision]:
    """Apply the precedence in the module docstring to every ref.

    *fallback_seconds* is the ref-age threshold for a job accounting cannot
    place (the cluster's longest ``slurm.time`` plus the window), or
    ``None`` when no safe bound exists; *fallback_reason* then says why, and
    those refs are kept.
    """
    window = int(window_days * DAY)
    now = listing.now
    out: List[Decision] = []
    for snap in listing.refs:
        age = (now - snap.committed) if now is not None else None
        job = snap.job_id
        record = listing.jobs.get(job) if (job is not None and listing.sacct_answered) else None

        def by_age(state: str, why_fallback: str) -> Decision:
            if age is None:
                return Decision(snap, False, state, "kept: the cluster sent no clock", age)
            if fallback_seconds is None:
                return Decision(
                    snap, False, state,
                    f"kept: {why_fallback}; {fallback_reason or 'no safe age bound'}",
                    age,
                )
            if age > fallback_seconds:
                return Decision(
                    snap, True, state,
                    f"{why_fallback}; snapshot {_days(age)} old, past the "
                    f"{_days(fallback_seconds)} fallback",
                    age,
                )
            return Decision(
                snap, False, state,
                f"kept: {why_fallback}; snapshot {_days(age)} old, within the "
                f"{_days(fallback_seconds)} fallback",
                age,
            )

        if job is None:
            out.append(by_age("no job id", "no job recorded"))
        elif record is None:
            why = "no accounting record" if listing.sacct_answered else "sacct unavailable"
            out.append(by_age("unknown", why))
        elif not record.ended:
            out.append(Decision(snap, False, record.state.lower() or "live", "kept: job still going", age))
        elif record.end is None:
            out.append(by_age(f"{record.state.lower()}, end unknown", "job ended at an unknown time"))
        else:
            since = (now - record.end) if now is not None else None
            state = f"{record.state.lower()} {_days(since)} ago" if since is not None else record.state.lower()
            if since is None:
                out.append(Decision(snap, False, state, "kept: the cluster sent no clock", age))
            elif since > window:
                out.append(Decision(
                    snap, True, state,
                    f"job ended {_days(since)} ago, past the {_days(window)} window", age,
                ))
            else:
                out.append(Decision(snap, False, state, f"kept: within the {_days(window)} window", age))
    return out


def retire_script(decisions: List[Decision]) -> str:
    """Delete the refs marked for retirement, each guarded by the sha the
    listing saw: a snapshot rewritten since is left alone (``update-ref -d``
    with an old value refuses when the ref has moved)."""
    lines = []
    for d in decisions:
        if not d.retire:
            continue
        path = shlex.quote(d.snapshot.path)
        ref = shlex.quote(d.snapshot.ref)
        sha = shlex.quote(d.snapshot.sha)
        lines.append(
            f"if git -C {path} update-ref -d {ref} {sha} 2>/dev/null; then "
            f"printf 'RETIRED\\t%s\\t%s\\n' {path} {ref}; else "
            f"printf 'FAILED\\t%s\\t%s\\n' {path} {ref}; fi"
        )
    return "\n".join(lines)


def parse_retire_output(text: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """``(retired, failed)`` as ``(mirror path, ref)`` pairs."""
    retired, failed = [], []
    for raw in text.splitlines():
        kind, _, rest = raw.partition("\t")
        path, _, ref = rest.partition("\t")
        if kind == "RETIRED":
            retired.append((path, ref))
        elif kind == "FAILED":
            failed.append((path, ref))
    return retired, failed


@dataclass
class ClusterSnapshots:
    cluster: str
    targets: List[str]
    decisions: List[Decision] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)      # per-cluster caveats
    errors: List[str] = field(default_factory=list)


def render(clusters: List[ClusterSnapshots], *, retiring: bool = False) -> str:
    """One block per cluster; a row per snapshot; the decision in words."""
    out: List[str] = []
    total = to_retire = 0
    for c in clusters:
        out.append(f"{c.cluster}   ({', '.join(c.targets)})")
        if c.decisions:
            rows = [
                (
                    d.snapshot.mirror,
                    d.snapshot.short_ref,
                    _days(d.age) if d.age is not None else "?",
                    str(d.snapshot.job_id) if d.snapshot.job_id is not None else "-",
                    d.state,
                    ("retire: " if d.retire else "") + d.reason,
                )
                for d in c.decisions
            ]
            widths = [max(len(r[i]) for r in rows) for i in range(5)]
            for r in rows:
                cells = "  ".join(r[i].ljust(widths[i]) for i in range(5))
                out.append(f"  {cells}  {r[5]}")
            total += len(c.decisions)
            to_retire += sum(1 for d in c.decisions if d.retire)
        else:
            out.append("  no WIP snapshots")
        for note in c.notes:
            out.append(f"  ({note})")
        out.append("")
    if not clusters:
        out.append("No clusters configured; nothing to list.")
        out.append("")
    if total:
        if to_retire and not retiring:
            out.append(f"{total} snapshot(s); {to_retire} past retention.  --retire deletes them.")
        elif to_retire:
            out.append(f"{total} snapshot(s); retiring {to_retire}.")
        else:
            out.append(f"{total} snapshot(s); nothing past retention.")
        out.append("")
    for c in clusters:
        for message in c.errors:
            out.append(f"! {c.cluster}: {message}")
    return "\n".join(out).rstrip("\n") + "\n"
