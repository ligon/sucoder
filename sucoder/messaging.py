"""``sucoder message``: put a line in front of a running agent.

Issue 26.  Two agents worked one mirror on 2026-09-21 -- one in a confined
job, one on a laptop -- with no way to reach each other, and the diagnosis
that turned out wrong was overturned only because a human copied a
contradicting observation from one terminal into the other.  A file drop
is a dead drop: nothing tells the peer to read it.  For a terminal agent
*stdin is the API*, and tmux owns it, so a message is ``tmux send-keys``
into the session's pane -- through ``srun --overlap`` for a job, since the
tmux server lives inside the allocation's cgroup, exactly as the pane probe
already reaches it.

Three rules shape what is pure here:

- *Say who it is from.*  Injected text carries the same authority as the
  human typing, so every message is framed with its sender and the time.
  The system prompt's bulletin convention tells agents to treat such text
  as a peer's report, to be checked, not as an instruction from the human.
- *Never type into a shell.*  A confined window ends in ``exec bash -l``,
  so after the agent exits the pane is a login shell, and ``send-keys``
  followed by Enter there *runs the message as a command*.  A recipient
  whose pane is a shell is refused, always; one whose pane could not be
  probed is refused unless forced, because unknown is not evidence of an
  agent either.
- *One line.*  A newline sent literally is an Enter to the pty and would
  submit a half-typed message, so the text is flattened onto one line.

Everything here is pure: the caller does the SSH and passes text in.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .sessions_report import LoginSession, Report, SessionEntry

# Between the text and the Enter that submits it.  A burst of typed
# characters followed by an immediate Enter can reach a TUI's input loop
# before it has drained the burst, and the Enter is then lost or lands
# early; a short pause costs nothing.
_SUBMIT_DELAY = "0.3"


def new_message_id() -> str:
    """Four hex characters: enough to tell today's messages apart, short
    enough to type in a reply."""
    import secrets
    return secrets.token_hex(2)


def frame(text: str, sender: str, when: str, msg_id: str = "") -> str:
    """The line that is typed: an id, who, when, and the text on one line.

    The id is what a reply names (``REPLY <id>: ...``), so the sender can
    find that reply in the pane among everything else the agent prints.
    """
    flat = " ".join(text.split())
    tag = f"message {msg_id} from" if msg_id else "message from"
    return f"[{tag} {sender} via sucoder, {when}] {flat}"


REPLY_PREFIX = "REPLY"

# What the remote agent is asked to do, in the default system prompt: answer
# on its own screen with a line that names the message.  The sender then
# reads the pane back through the same route it typed into.  There is no
# path from a cluster to a laptop behind NAT, so a reply cannot be pushed;
# it can only be pulled, and this line is what the pull looks for.


def capture_pane_command(recipient: Recipient, lines: int = 200) -> str:
    """The shell that prints the last *lines* rows of the recipient's pane,
    scrollback included, through the same route ``send_keys_command`` types
    into."""
    sock = f"-L {shlex.quote(recipient.socket)} " if recipient.socket else ""
    sess = shlex.quote(recipient.session)
    tmux = f"tmux {sock}capture-pane -p -S -{int(lines)} -t {sess}"
    if recipient.job_id is not None:
        tmux = (
            f"TMPDIR=/tmp srun --jobid={recipient.job_id} --overlap --quiet --chdir=/tmp "
            f"bash -c {shlex.quote(tmux)}"
        )
    return tmux


def pane_tail(text: str, lines: int) -> str:
    """The last *lines* non-blank rows of a captured pane."""
    rows = [r.rstrip() for r in text.splitlines() if r.strip()]
    return "\n".join(rows[-lines:]) if lines > 0 else ""


def extract_reply(pane_text: str, msg_id: str) -> Optional[str]:
    """The reply naming *msg_id*, as the agent typed it, or ``None``.

    A TUI reflows long lines into several rows, indented, so the reply is
    taken from the row holding ``REPLY <id>`` down to the next blank row or
    the next row that starts a box border (the input prompt).  The last
    such reply wins, since an agent may correct itself.
    """
    needle = f"{REPLY_PREFIX} {msg_id}"
    rows = pane_text.splitlines()
    start = None
    for i, row in enumerate(rows):
        if needle in row and not row.lstrip().startswith("[message "):
            start = i
    if start is None:
        return None
    first = rows[start]
    first = first[first.index(needle) + len(needle):].lstrip(":").strip()
    collected = [first] if first else []
    for row in rows[start + 1:]:
        if not row.strip() or row.lstrip().startswith(("╭", "│", "╰", "┌", "└", "> ")):
            break
        collected.append(row.strip())
    return " ".join(collected).strip() or None


@dataclass(frozen=True)
class Recipient:
    """One session a message can be typed into."""

    label: str                     # what the human sees: mirror + where
    host: str                      # where the send runs: cluster host or login host
    session: str                   # tmux session name
    socket: str = ""               # dedicated tmux socket, or "" for the default server
    job_id: Optional[int] = None   # reached through srun --overlap when set
    target: Optional[str] = None
    pane: Optional[str] = None     # what the probe saw running there


def send_keys_command(recipient: Recipient, line: str) -> str:
    """The shell that types *line* into *recipient*'s pane and submits it.

    ``send-keys -l`` sends the text literally, so nothing in it is read as
    a key name.  For a job the whole thing runs under ``srun --overlap`` in
    the allocation, with the node-local ``TMPDIR`` and cwd neutralised the
    way the pane probe does.  Prints ``SENT`` or ``FAILED`` with the
    session name so a batch of sends reports per recipient.
    """
    sock = f"-L {shlex.quote(recipient.socket)} " if recipient.socket else ""
    sess = shlex.quote(recipient.session)
    tmux = (
        f"tmux {sock}send-keys -t {sess} -l {shlex.quote(line)} && "
        f"sleep {_SUBMIT_DELAY} && tmux {sock}send-keys -t {sess} Enter"
    )
    if recipient.job_id is not None:
        tmux = (
            f"TMPDIR=/tmp srun --jobid={recipient.job_id} --overlap --quiet --chdir=/tmp "
            f"bash -c {shlex.quote(tmux)}"
        )
    return (
        f"if {tmux} 2>/dev/null; then printf 'SENT\\t%s\\n' {sess}; "
        f"else printf 'FAILED\\t%s\\n' {sess}; fi"
    )


def parse_send_output(text: str) -> Tuple[List[str], List[str]]:
    """``(sent, failed)`` session names from a batch of sends."""
    sent, failed = [], []
    for raw in text.splitlines():
        kind, _, name = raw.partition("\t")
        if kind == "SENT":
            sent.append(name.strip())
        elif kind == "FAILED":
            failed.append(name.strip())
    return sent, failed


def _session_names(mirror_or_token: str) -> Tuple[str, str]:
    from .mirror import confined_tmux_target
    return confined_tmux_target(mirror_or_token)


def plan_recipients(
    report: Report, *,
    cluster_hosts: dict,           # {cluster: (host, control)} as `sessions` resolved them
    target_cluster: dict,          # {target name: cluster}
    confined_targets: set,         # target names whose jobs use a dedicated tmux socket
    mirror: Optional[str] = None,
    target: Optional[str] = None,
    everyone: bool = False,
    force: bool = False,
    gateway_hosts: Optional[set] = None,   # hosts that are round-robin aliases
    for_reading: bool = False,             # a peek: a shell pane is fine to read
) -> Tuple[List[Recipient], List[str]]:
    """Who gets the message, and who was passed over and why.

    *mirror* selects by configured mirror name (or by the sanitized token a
    job or session carries, for one no config names); *target* narrows to
    one target; *everyone* takes every live session the report found.
    """
    from .config import sanitize_session_token

    wanted_token = sanitize_session_token(mirror) if mirror else None
    gateway_hosts = gateway_hosts or set()
    chosen: List[Recipient] = []
    skipped: List[str] = []

    def consider(label: str, pane: Optional[str], exited: bool, make) -> None:
        if for_reading:
            chosen.append(make())
            return
        if exited:
            skipped.append(f"{label}: agent exited; its pane is a shell, and a message "
                           "typed there would run as a command")
            return
        if pane is None and not force:
            skipped.append(f"{label}: pane not probed, so it may be a shell; --force sends anyway")
            return
        chosen.append(make())

    def entry_selected(entry: SessionEntry, group_target: Optional[str]) -> bool:
        if not everyone:
            if wanted_token is None:
                return False
            if entry.mirror != mirror and entry.job.token != wanted_token:
                return False
        return target is None or group_target == target

    for group in report.groups:
        for entry in group.entries:
            if not entry_selected(entry, group.name):
                continue
            cluster = target_cluster.get(group.name)
            if cluster is None or cluster not in cluster_hosts:
                skipped.append(f"{entry.mirror or entry.job.token} (job {entry.job.job_id}): "
                               "its cluster did not answer, so it cannot be reached")
                continue
            host = cluster_hosts[cluster][0]
            name = entry.mirror or entry.job.token
            session, socket = _session_names(name)
            confined = group.name in confined_targets
            label = f"{name} (job {entry.job.job_id} on {entry.job.node or '?'}, {group.name})"
            consider(
                label, entry.pane, entry.agent_exited,
                lambda: Recipient(
                    label=label, host=host, session=session,
                    socket=socket if confined else "", job_id=entry.job.job_id,
                    target=group.name, pane=entry.pane,
                ),
            )
    for entry in report.unmatched:
        if entry_selected(entry, None) and (everyone or target is None):
            skipped.append(f"{entry.job.token} (job {entry.job.job_id}): matches no configured "
                           "target, so whether its tmux uses a dedicated socket is unknown")
    # A gateway is a round-robin alias for the login nodes, so the sweep
    # can find ONE tmux session under two host names: the gateway's and
    # the node's own (measured 2026-09-22: hpc.brc.berkeley.edu resolved to
    # ln003, which was also pinned).  Typing into both delivers twice.  When
    # a session with the same name and target is reported on a named login
    # node, the gateway's copy is the same session and is dropped.
    named_hosts = {
        (sess.name, sess.target) for sess in report.logins
        if sess.host not in gateway_hosts
    }
    for sess in report.logins:
        if not everyone:
            if wanted_token is None or sess.token != wanted_token:
                continue
        if target is not None and sess.target != target:
            continue
        if sess.host in gateway_hosts and (sess.name, sess.target) in named_hosts:
            continue
        label = f"{sess.token} (login session on {sess.host}, {sess.target})"
        consider(
            label, sess.pane, sess.agent_exited,
            lambda: Recipient(
                label=label, host=sess.host, session=sess.name, socket="",
                job_id=None, target=sess.target, pane=sess.pane,
            ),
        )
    return chosen, skipped
