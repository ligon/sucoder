"""Where this process sits relative to tmux, and what a launch should do about it.

Every launch ends in a tmux one way or another.  A remote launch always
does (``tmux new-session -A`` over ``ssh -t``, or ``srun --pty tmux
attach`` for a confined job): the remote tmux is what survives an SSH
drop and what ``message`` / ``peek`` / ``renew`` talk to.  A local launch
never did: the agent ran straight in the caller's terminal, so it had a
survivable, re-attachable session only when the caller happened to be
inside tmux already.

Two annoyances follow, mirror images of each other:

- local, *outside* tmux: the agent dies with the terminal, and nothing
  can attach to it;
- remote, *inside* tmux: ``$TMUX`` does not cross ssh, so the remote tmux
  nests silently inside the local one.  The caller finds out only after
  the allocation came through, when the outer tmux starts eating the
  prefix key.

This module settles both before anything expensive runs.  :func:`detect`
reads the local tmux state.  For a local launch outside tmux, the CLI
re-runs its own command line inside a local session named like the
remote ones (``sucoder-<mirror>``, :func:`new_session_argv`); the inner
``sucoder`` sees ``$TMUX`` and launches exactly as before, so the agent's
``sudo`` and TTY handling are untouched and the session belongs to the
human's own tmux server.  For a remote launch inside tmux the nesting is
structural, so :func:`nested_remote_warning` says what the keys will do.

Every probe here addresses the *default* tmux server, the one a launch
outside tmux creates on: ``$TMUX`` is dropped from the probe's
environment, so being inside some other server does not change the
answer.  ``has-session`` targets carry the ``=`` exact-match prefix:
``sucoder-Foo`` must not answer for ``sucoder-FooBar``.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Sequence

from .config import sanitize_session_token

DEFAULT_PREFIX = "C-b"
_PROBE_TIMEOUT = 5

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


@dataclass(frozen=True)
class TmuxContext:
    """The tmux this process is inside, if any.

    ``socket`` is the outer server's socket path, from ``$TMUX``;
    ``session`` and ``prefix`` come from asking that server and fall back
    to ``None`` / :data:`DEFAULT_PREFIX` when it does not answer.
    """

    inside: bool
    socket: Optional[str] = None
    session: Optional[str] = None
    prefix: str = DEFAULT_PREFIX

    @property
    def on_default_server(self) -> bool:
        """Whether the outer tmux is the default server (``tmux`` with no ``-L``/``-S``).

        That is the server a local launch creates its session on, so it is
        the one ``switch-client`` can reach without nesting.
        """
        return self.inside and os.path.basename(self.socket or "") == "default"


def _probe_env(environ: Mapping[str, str]) -> dict:
    return {k: v for k, v in environ.items() if k != "TMUX"}


def _ask(argv: Sequence[str], run: Runner, env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Run a tmux query; ``None`` when tmux is missing, errors, or says nothing."""
    try:
        result = run(
            list(argv), capture_output=True, text=True, check=False,
            timeout=_PROBE_TIMEOUT, env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    text = (result.stdout or "").strip()
    return text or None


def detect(
    environ: Mapping[str, str] = os.environ, *, run: Runner = subprocess.run,
) -> TmuxContext:
    """What tmux, if any, this process is inside.

    ``$TMUX`` is ``<socket>,<server pid>,<session index>``; the session
    *name* and the prefix key are asked of the outer server so a warning
    can name them.  A server that does not answer still counts as being
    inside: the environment variable is the fact, the rest is decoration.
    """
    raw = environ.get("TMUX")
    if not raw:
        return TmuxContext(inside=False)
    socket = raw.split(",", 1)[0] or None
    session = _ask(["tmux", "display-message", "-p", "#S"], run)
    prefix = _ask(["tmux", "show-options", "-gv", "prefix"], run) or DEFAULT_PREFIX
    return TmuxContext(inside=True, socket=socket, session=session, prefix=prefix)


def local_session_name(mirror: str) -> str:
    """The local session for *mirror*: the same ``sucoder-<token>`` the remote paths use."""
    return f"sucoder-{sanitize_session_token(mirror)}"


def tmux_available() -> bool:
    return shutil.which("tmux") is not None


def interactive_terminal(
    environ: Mapping[str, str] = os.environ, *, stdin=None, stdout=None,
) -> bool:
    """A terminal tmux can take over: a tty on both ends, and not Emacs's dumb one.

    ``M-x shell`` and eshell hand out a pty with ``TERM=dumb``; tmux
    refuses it ("missing or unsuitable terminal"), so a re-exec there would
    fail after the fact instead of launching in place.
    """
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    try:
        if not (stdin.isatty() and stdout.isatty()):
            return False
    except (AttributeError, ValueError):
        return False
    return environ.get("TERM", "") not in ("", "dumb")


def session_exists(
    session: str, *, environ: Mapping[str, str] = os.environ, run: Runner = subprocess.run,
) -> bool:
    """Whether *session* exists on the default server (exact name match)."""
    try:
        result = run(
            ["tmux", "has-session", "-t", f"={session}"],
            capture_output=True, text=True, check=False,
            timeout=_PROBE_TIMEOUT, env=_probe_env(environ),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def command_argv(argv: Optional[Sequence[str]] = None) -> List[str]:
    """The command line to run again inside the new session.

    The console script is an executable path, or a bare name the shell
    found on ``PATH``; either re-runs as it is.  A path that is not
    executable was run through an interpreter, so it keeps this one.
    """
    argv = list(sys.argv if argv is None else argv)
    head = argv[0] if argv else ""
    if os.sep in head and not os.access(head, os.X_OK):
        return [sys.executable, *argv]
    return argv


def new_session_argv(
    session: str, command: Sequence[str], cwd: str, window_name: Optional[str] = None,
) -> List[str]:
    """``tmux new-session`` running *command* in *cwd* as session *session*.

    The command is one shell-quoted string, as the remote launch builds
    it; ``-c`` keeps the working directory, which zero-config mode reads
    the canonical repository from.  ``window_name`` (the mirror name, not
    a path basename, which differs between local and remote checkouts)
    labels the window with ``-n``; tmux then turns ``automatic-rename``
    off for it, so the agent's command does not replace the label.
    """
    window_flags = ["-n", window_name] if window_name else []
    return [
        "tmux", "new-session", "-s", session, "-c", cwd, *window_flags,
        shlex.join(list(command)),
    ]


def attach_argv(session: str) -> List[str]:
    return ["tmux", "attach-session", "-t", f"={session}"]


def switch_argv(session: str) -> List[str]:
    """``switch-client``: reach *session* from inside the same server without nesting."""
    return ["tmux", "switch-client", "-t", f"={session}"]


def nested_remote_warning(state: TmuxContext, *, what: str = "the remote agent") -> str:
    """Say, before the ssh, that the remote tmux will nest and what the keys then do.

    The outer tmux reads every key first.  Its prefix pressed twice is
    passed through (the default ``send-prefix`` binding), so with the
    remote prefix at its default ``C-b`` the sequence ``<p> <p> d``
    detaches the remote session, while ``<p> d`` detaches the local one
    and leaves the ssh running in its window.
    """
    where = f'tmux session "{state.session}"' if state.session else "a tmux session"
    p = state.prefix
    return (
        f"note: this shell is inside {where}, so {what}'s own tmux will nest in this pane.\n"
        f"      The outer tmux reads keys first: `{p} {p} d` detaches the remote session "
        f"(prefix twice passes it through); `{p} d` detaches this one and leaves the "
        f"ssh open in its window."
    )


def unwrapped_local_warning(session: str, reason: str) -> str:
    """Say that a local launch is staying bound to this terminal, and why."""
    return (
        f"note: {reason}, so the agent runs bound to this terminal: closing it ends "
        f"the agent, and `sucoder attach` has nothing to reach.  Run `tmux new -s "
        f"{session}` first for a detachable session."
    )
