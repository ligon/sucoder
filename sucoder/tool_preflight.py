"""Launch-time tool-version preflight: record what the target ships, warn
when it is old, and never stop a session starting.

The shipped prompts and skills tell an agent to use ``gh``, ``git``,
``jq``, ``rg`` and ``tmux``, but sucoder hands a session whatever happens
to be in the target's ``$HOME`` and says nothing about how old it is.

*The incident (GH #20).*  A target's ``gh`` was 2.67.0 (February 2025).
Against a server-side change fixed upstream in October 2025
(``cli/cli#11983``), every ``gh pr edit`` and ``gh issue view`` failed
with ``GraphQL: Projects (classic) is being deprecated ...
(repository.pullRequest.projectCards)`` -- an error naming a GitHub
product sunset, emitted for a command carrying no project flags.  An
agent concluded *"gh pr edit is broken on this repo"* and wrote that up
as a repo-specific gotcha.  The misattribution, not the outage, is what
this module exists to prevent: the version is in the log before the
agent starts, so "my tool is stale" is available as an explanation.

Four constraints, each load-bearing:

- **Report, never gate.**  A stale tool must not stop a session.
  Nothing here raises; :func:`evaluate` returns findings and the caller
  logs them.  (Contrast ``startup_checks``, which raises ``StartupError``
  by design -- that is the gating model, and it is the wrong one here.)
- **No network.**  The floor is a constant in config, never a lookup
  against a release API: a preflight that needs the internet fails on
  exactly the constrained targets it is meant to help.
- **Best effort, like ``slurm_timer.snapshot_wip``.**  A missing binary
  or an unparseable ``--version`` is *recorded*, not raised.  A missing
  ``jq`` must produce ``jq: not found``, not a traceback.
- **Cheap.**  One round trip, a handful of ``--version`` calls.

The probe is a ``@PLACEHOLDER@`` bash template (repo convention -- the
script is dense with ``$`` and ``{}``); it only *gathers*.  Parsing and
comparison are Python, which keeps them unit-testable without a shell.

Versions are not all clean semver: ``tmux 3.3a``, ``jq-1.6``, ``gh
version 2.67.0 (2025-02-11)``, ``ripgrep 14.1.0``, ``git version
2.39.3``.  :func:`parse_version` takes the first dotted-numeric run and
drops the rest; anything it cannot read is ``unknown`` -- recorded,
neither a pass nor a failure.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

#: Tools the shipped prompts and skills assume.  The mapping is
#: ``name -> floor``; a ``None`` floor means "record the version, do not
#: judge it".  Users extend or override it through ``tool_preflight.floors``
#: in the sucoder config.
#:
#: Only ``gh`` is incident-derived.  ``cli/cli#11983`` was closed
#: 2025-10-21 and v2.83.0 (2025-11-04) is the first release *published
#: after* that close -- v2.82.1 landed one day after and is ambiguous, so
#: the floor is deliberately the conservative side of the boundary.  The
#: other four are modest "old enough to surprise someone" defaults, not
#: findings; raise them locally if your prompts need more.
DEFAULT_TOOL_FLOORS: Dict[str, Optional[str]] = {
    "gh": "2.83.0",
    "git": "2.34.0",
    "jq": "1.6",
    "rg": "13.0.0",
    "tmux": "3.0",
}

#: How each tool spells "print your version".  ``tmux`` is the reminder
#: that this table is necessary: it takes ``-V`` and treats ``--version``
#: as a usage error, so a hardcoded ``--version`` would report every host's
#: tmux as unparseable.
VERSION_FLAGS: Dict[str, List[str]] = {
    "gh": ["--version"],
    "git": ["--version"],
    "jq": ["--version"],
    "rg": ["--version"],
    "tmux": ["-V"],
}

DEFAULT_VERSION_FLAG = "--version"

#: Every probe line carries this tag.  The probe runs under a *login*
#: shell (so it sees the PATH the agent will see), and login shells print
#: banners and MOTDs; untagged output is ignored rather than parsed.
LINE_TAG = "SUCODER_TOOL"
HOST_TAG = "SUCODER_HOST"

# Status vocabulary.  ``UNKNOWN`` is not a pass and not a failure: it
# means the version string was recorded but could not be compared.
OK = "ok"
BELOW = "below"
UNKNOWN = "unknown"
MISSING = "missing"

_VERSION_RE = re.compile(r"\d+(?:\.\d+)+")


def parse_version(raw: Optional[str]) -> Optional[Tuple[int, ...]]:
    """First dotted-numeric run of *raw* as an int tuple, or ``None``.

    ``gh version 2.67.0 (2025-02-11)`` -> ``(2, 67, 0)`` (the date's
    separators are hyphens, so it cannot match first), ``tmux 3.3a`` ->
    ``(3, 3)``, ``jq-1.6`` -> ``(1, 6)``, ``ripgrep 14.1.0`` ->
    ``(14, 1, 0)``, ``git version 2.39.3`` -> ``(2, 39, 3)``.

    A bare integer (``jq-1``) has no dot and reads as ``None``: a single
    number is as likely to be a build id as a version, and ``unknown`` is
    the honest answer.
    """
    if not raw:
        return None
    match = _VERSION_RE.search(raw)
    if match is None:
        return None
    try:
        return tuple(int(part) for part in match.group(0).split("."))
    except ValueError:                              # pragma: no cover - regex forbids
        return None


def _pad(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    width = max(len(a), len(b))
    return a + (0,) * (width - len(a)), b + (0,) * (width - len(b))


def version_at_least(found: Tuple[int, ...], floor: Tuple[int, ...]) -> bool:
    """``found >= floor`` with the shorter tuple zero-padded (3.3 == 3.3.0)."""
    left, right = _pad(found, floor)
    return left >= right


@dataclass(frozen=True)
class ToolReport:
    """One tool's finding.  Data only; the caller decides what to do."""

    name: str
    status: str                                  # OK | BELOW | UNKNOWN | MISSING
    raw: str = ""                                # first line of --version output
    path: str = ""                               # resolved binary (`command -v`)
    version: Optional[Tuple[int, ...]] = None
    floor: Optional[str] = None                  # floor as configured, verbatim

    @property
    def is_warning(self) -> bool:
        """True for the two states worth a WARNING line.

        ``UNKNOWN`` deliberately is not one: an unreadable version string
        is a gap in this module's parser far more often than it is a
        problem with the target, and a warning nobody can act on trains
        readers to ignore the ones they can.
        """
        return self.status in (BELOW, MISSING)

    def describe(self) -> str:
        where = f" [{self.path}]" if self.path else ""
        floor = f", floor {self.floor}" if self.floor else ""
        if self.status == MISSING:
            return f"{self.name}: not found{floor}"
        if self.status == BELOW:
            return f"{self.name}: {self.raw}{floor} -- BELOW FLOOR{where}"
        if self.status == UNKNOWN:
            return f"{self.name}: {self.raw or '(no output)'} -- version unreadable{where}"
        return f"{self.name}: {self.raw}{floor}{where}"


# The probe only gathers.  It is written so that nothing in it can fail
# the caller: every command is guarded, and the script exits 0
# unconditionally.  ``timeout`` is used when the target has it so a
# wedged binary cannot hold the launch (and is simply skipped when it
# does not, rather than making the whole probe unavailable).
_PROBE_TEMPLATE = r'''#!/bin/bash
# sucoder tool-version preflight probe (generated; do not edit).
# Gathers only: reports, never judges, never fails.
TIMEOUT=""
if command -v timeout >/dev/null 2>&1; then TIMEOUT="timeout 5"; fi
printf '@HOST_TAG@\t%s\n' "$(hostname 2>/dev/null || echo unknown)"
probe() {
    name="$1"
    shift
    path=$(command -v "$1" 2>/dev/null) || path=""
    if [ -z "$path" ]; then
        printf '@LINE_TAG@\t%s\t\tnot found\n' "$name"
        return 0
    fi
    raw=$($TIMEOUT "$@" 2>&1 | head -n 1)
    printf '@LINE_TAG@\t%s\t%s\t%s\n' "$name" "$path" "$raw"
}
@PROBES@
exit 0
'''


def build_probe_script(tools: Sequence[str]) -> str:
    """Render the probe for *tools* (order preserved).

    Tool names are shell-quoted; the version flag comes from
    :data:`VERSION_FLAGS`, defaulting to ``--version`` for a tool the
    table does not know.
    """
    probes = "\n".join(
        "probe {} {} {}".format(
            shlex.quote(name),
            shlex.quote(name),
            " ".join(shlex.quote(f) for f in VERSION_FLAGS.get(name, [DEFAULT_VERSION_FLAG])),
        )
        for name in tools
    )
    return (
        _PROBE_TEMPLATE
        .replace("@HOST_TAG@", HOST_TAG)
        .replace("@LINE_TAG@", LINE_TAG)
        .replace("@PROBES@", probes)
    )


def evaluate(
    stdout: str,
    floors: Mapping[str, Optional[str]],
) -> Tuple[Optional[str], List[ToolReport]]:
    """Turn probe output into ``(hostname, reports)``.

    Untagged lines (login-shell banners, MOTD) are ignored.  A tool named
    in *floors* but absent from the output gets no report at all --
    silence about a tool we did not manage to probe is better than an
    invented finding.
    """
    host: Optional[str] = None
    reports: List[ToolReport] = []
    for line in (stdout or "").splitlines():
        fields = line.rstrip("\n").split("\t")
        if fields[0] == HOST_TAG and len(fields) >= 2:
            host = fields[1].strip() or None
            continue
        if fields[0] != LINE_TAG or len(fields) < 4:
            continue
        name, path, raw = fields[1], fields[2].strip(), "\t".join(fields[3:]).strip()
        floor_text = floors.get(name)
        if not path:
            reports.append(
                ToolReport(name=name, status=MISSING, raw=raw, floor=floor_text)
            )
            continue
        found = parse_version(raw)
        floor = parse_version(floor_text)
        if found is None:
            status = UNKNOWN
        elif floor is not None and not version_at_least(found, floor):
            status = BELOW
        else:
            status = OK
        reports.append(
            ToolReport(
                name=name, status=status, raw=raw, path=path,
                version=found, floor=floor_text,
            )
        )
    return host, reports


def format_report(host: Optional[str], reports: Sequence[ToolReport]) -> str:
    """One-line-per-tool summary, headed by the host that was probed.

    The hostname is in the report because it is load-bearing: at launch
    time a *confined* (sbatch) target is probed on the login node while
    the agent runs on a compute node.  Tools under a shared ``$HOME``
    match; a system ``git`` or ``tmux`` need not.
    """
    head = f"Tool versions on {host or 'the target'}:"
    return "\n".join([head] + [f"  {r.describe()}" for r in reports])
