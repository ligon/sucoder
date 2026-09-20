"""SSH connection and tunnel lifecycle management for remote execution.

Manages a ControlMaster connection to avoid repeated authentication
(critical for OTP-based logins like university HPC clusters), and
a local port forward for git transport through a data transfer node.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_LOG = logging.getLogger(__name__)

# Tuning for SshControl.is_active()'s end-to-end liveness probe.  The probe
# execs a remote command, which spawns a login shell on the far side; on a
# hammered HPC login node that spawn can be slow enough to blow a single
# tight timeout and make a live master read as dead --- a spurious re-auth.
# Retrying with a generous per-attempt budget lets a merely-slow master
# answer, while a genuine zombie (mux alive, TCP dead) still fails fast on
# every attempt and so can never be masked by the retry.
_LIVENESS_PROBE_ATTEMPTS = 3
# Budget per attempt.  This must clear the cost of a *session open*, not of
# the `true` it runs: on a BRC login node the remote session setup alone is
# ~10s wall-clock before the command is even exec'd (measured 2026-09-19 --
# `true`, `echo`, `squeue` and `tmux list-sessions` all cost 9.4-10.0s over
# an already-warm master).  At the old 12s a merely-loaded node blew the
# budget, all three attempts timed out, and a live master was declared
# expired -- forcing a full re-auth, which on this gateway can cost an OTP.
# Raising it is one-sided: a genuine zombie returns non-zero *immediately*
# (see the rc branch below) rather than timing out, so detection of a dead
# master is unaffected and only the slow-but-alive case waits longer.
_LIVENESS_PROBE_TIMEOUT = 30    # seconds, wall-clock per attempt
_LIVENESS_PROBE_BACKOFF = 1.0   # seconds between attempts
# How long a *successful* end-to-end probe is trusted for, per socket.  The
# probe costs a full session open (~8s here), and a command that needs the
# tunnel three times in a minute paid it three times over.  What the probe
# rules out is a socket that died between calls -- a suspend, a network
# change -- which cannot happen twice inside one command run without the
# work itself failing and taking the reconnect path.  So: verify once, then
# trust it for a couple of minutes; a stale verdict costs one failed command
# (``_capture_over_tunnel`` re-authenticates and retries), never a wrong
# answer.  Process-local by design: another process's probe proves nothing
# about this one's socket handle.
_LIVENESS_CACHE_TTL = 120       # seconds
_VERIFIED: Dict[str, float] = {}
_VERIFIED_GUARD = threading.Lock()


class TunnelError(RuntimeError):
    """Raised when SSH connection or tunnel operations fail.

    ``stderr`` carries the captured ssh stderr (when available) so that
    callers can *classify* the failure --- e.g. distinguish a transient
    ``kex_exchange_identification`` closure worth retrying from a hard
    authentication or host error that should surface immediately.  See
    :func:`is_transient_ssh_error`.
    """

    def __init__(self, *args, stderr: str = "") -> None:
        super().__init__(*args)
        self.stderr = stderr


# Substrings that mark a *transient* SSH transport fault --- one that
# typically clears on its own within seconds and is therefore worth a
# bounded retry.  Two common sources on an HPC cluster:
#   * a just-allocated SLURM compute node refusing SSH while ``sshd`` /
#     ``pam_slurm_adopt`` register the job, and
#   * a busy login node shedding connections during the protocol-banner
#     exchange (``MaxStartups`` / fail2ban),
# both of which surface as ``kex_exchange_identification: Connection
# closed by remote host``.
#
# Deliberately excludes ``could not resolve hostname``: that does not
# self-heal by waiting, and for a jump-only login node it signals the
# *wrong* (local) resolution path rather than a transient blip.
TRANSIENT_SSH_MARKERS = (
    "session open refused",          # mux refused a new session
    "the remote end hung up",        # peer died mid-stream
    "connection closed",
    "connection refused",
    "connection timed out",
    "connection reset",
    "broken pipe",
    "no route to host",
    "kex_exchange_identification",   # sshd dropped us before the banner
)


# A mux that answers "Session open refused by peer" is ALIVE: the refusal
# came from the remote sshd, which means the transport carried the request
# and brought back the answer.  It means *busy*, not *broken* --- see
# :func:`session_lock` for why this host hands them out one at a time.
SESSION_BUSY_MARKERS = (
    "session open refused",          # the mux relaying the peer's refusal
    "session request failed",        # mux_client_request_session's own wording
)


def is_session_busy_error(text: str) -> bool:
    """True if *text* says the master is carrying its one session already."""
    low = (text or "").lower()
    return any(marker in low for marker in SESSION_BUSY_MARKERS)


# ------------------------------------------------------------------
# One session channel at a time, per ControlMaster
# ------------------------------------------------------------------
#
# BRC's sshd sets ``MaxSessions 1``: a ControlMaster carries exactly ONE
# session channel at a time (measured 2026-09-20 on hpc.brc.berkeley.edu
# and ln002.brc --- of N simultaneous ``ssh <host> true`` over one warm
# master, exactly one succeeds for every N tried).  The extra requests are
# not queued but REFUSED, and on a refusal ``ControlMaster=auto`` falls back
# to dialling the host directly: a fresh authentication, which on this
# gateway earned ``Too many authentication failures`` and, with fail2ban in
# front, can cost more than the connection ever saved.
#
# Port-forward channels are exempt from ``MaxSessions``, which is why
# fanning out ACROSS hosts through one gateway (each hop a ``-W`` channel)
# works, while running two commands ON one host does not.  So concurrency
# stays where it pays and is serialised where it cannot.
#
# In-process only.  It cannot know about another sucoder, an attached
# session, or a renew loop holding the slot --- those still collide, which
# is what :func:`is_session_busy_error` and the caller's retry are for.
_SESSION_LOCKS: Dict[str, threading.RLock] = {}
_SESSION_LOCKS_GUARD = threading.Lock()


def session_lock(key) -> threading.RLock:
    """Return the lock guarding session channels on one ControlMaster.

    Keyed by socket path (or host, when there is no socket), so every
    caller reaching one master shares one lock.  Reentrant: a probe nested
    inside a command path on the same host must not deadlock itself.
    """
    name = str(key)
    with _SESSION_LOCKS_GUARD:
        lock = _SESSION_LOCKS.get(name)
        if lock is None:
            lock = threading.RLock()
            _SESSION_LOCKS[name] = lock
        return lock


def is_transient_ssh_error(text: str) -> bool:
    """True if *text* (ssh/git stderr or an error message) looks transient.

    Matches against :data:`TRANSIENT_SSH_MARKERS` case-insensitively.
    Used to decide whether a failed ControlMaster bring-up is worth a
    bounded retry rather than failing the launch outright.
    """
    low = (text or "").lower()
    return any(marker in low for marker in TRANSIENT_SSH_MARKERS)


def _find_free_port() -> int:
    """Bind to port 0 and return the OS-assigned ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _control_socket_dir() -> Path:
    """Return (and create) the directory for SSH control sockets."""
    d = Path("~/.sucoder/ssh").expanduser()
    d.mkdir(parents=True, exist_ok=True)
    d.chmod(0o700)
    return d


def _control_socket_path(gateway: str) -> Path:
    """Return the ControlPath for a given gateway host."""
    return _control_socket_dir() / f"{gateway}.sock"


# ------------------------------------------------------------------
# ControlMaster management
# ------------------------------------------------------------------


@dataclass
class SshControl:
    """Manages a persistent SSH ControlMaster connection.

    Authenticate once (interactively --- pin + OTP etc.) and all
    subsequent ``ssh`` commands to the same host reuse the connection
    through a Unix domain socket.

    Supports an optional *jump_host* for two-hop connections (e.g.,
    gateway -> login node).  When a jump host is provided, the
    ControlMaster for the jump host is used to reach the target.

    If the socket expires (``control_persist`` elapsed, network drop,
    etc.), :meth:`ensure` will detect the dead socket and
    re-establish, prompting for credentials again.
    """

    gateway: str
    control_persist: str = "7d"
    keepalive_interval: int = 30
    keepalive_count_max: int = 120
    jump_host: Optional[str] = None
    jump_control: Optional["SshControl"] = field(default=None, repr=False)
    extra_options: List[str] = field(default_factory=list)
    debug: bool = False
    user: Optional[str] = None
    # Local SSH certificate (private-key path) to present when authenticating
    # *directly* to the gateway.  Its ``<cert_file>-cert.pub`` sibling is
    # offered as the CertificateFile.  Applied ONLY on the direct/gateway hop
    # (``jump_host is None``); login/DTN/compute authenticate by publickey
    # through the gateway mux and must not be forced onto the gateway cert.
    cert_file: Optional[str] = None
    socket_name: Optional[str] = None  # Optional connection identity for direct targets
    # Set True by establish() when *this* process authenticated the master.
    # A master we just brought up cannot be a post-suspend zombie, so
    # is_active() can trust the cheap structural check and skip the remote
    # shell round-trip for it (see is_active()).
    _established_this_session: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # ``gateway`` is typed non-Optional, but it is routinely fed from
        # session state (``session.compute_node``, ``session.login_node``)
        # that CAN be None.  Left unchecked, the None survives all the way
        # into ``subprocess.run(["ssh", ..., None])`` and surfaces as a
        # 40-line ``TypeError: expected str, bytes or os.PathLike object,
        # not NoneType`` from deep inside Popen -- naming neither the host
        # nor the caller.  Fail here instead, where we still know which hop
        # is unresolved.
        if not self.gateway:
            raise TunnelError(
                "SshControl was given no host to connect to "
                f"(gateway={self.gateway!r}).  This means the caller's "
                "session state is missing a node name; it is a bug in the "
                "caller, not a network failure."
            )

    @property
    def socket_path(self) -> Path:
        return _control_socket_path(self.socket_name or self.gateway)

    def _format_host(self, host: str) -> str:
        if self.user:
            return f"{self.user}@{host}"
        return host

    def is_active(self, *, deep: Optional[bool] = None, sleep=time.sleep) -> bool:
        """Return True if the ControlMaster connection is usable.

        Two layers:

        * a **structural** ``ssh -O check`` that asks the local mux daemon
          whether the master process is alive.  Cheap, and involves no
          remote shell.
        * an **end-to-end** probe (``ssh ... true``) that opens a real
          session through the mux.  This is what catches a *zombie* socket
          --- mux process alive but the underlying TCP dead after a suspend
          or network change --- which ``-O check`` alone reports as a
          misleading ``ACTIVE``.

        The end-to-end probe execs a remote command, so it pays for a login
        shell on the far side; on a hammered HPC login node that spawn is
        slow and occasionally exceeds a single timeout, which used to make a
        perfectly live master read as dead and trigger a spurious re-auth
        (the "re-authenticate on every hop" bug).  Two guards fix that:

        * **skip** the probe entirely for a master *this* process just
          established (:attr:`_established_this_session`) or just probed
          successfully (:data:`_LIVENESS_CACHE_TTL`): neither can have
          become a zombie in that window, so the structural check suffices
          and no remote shell is spawned on the hot path.  This is what
          keeps a command that reaches one host several times from paying
          a session open each time --- and, on a ``MaxSessions 1`` host,
          from spending the one slot it has on a probe.
        * **retry** the probe otherwise (:data:`_LIVENESS_PROBE_ATTEMPTS`)
          with a generous per-attempt budget; a true zombie fails fast on
          every attempt, so the retry cannot hide it, but a merely-slow
          master gets to answer.

        ``deep`` forces the choice (``True`` = always probe end-to-end,
        ``False`` = structural only); the default auto-selects per the rule
        above.  ``sleep`` is injectable for tests.  All probes use
        ``BatchMode=yes`` + a wall-clock timeout so a wedged mux can never
        fall through to interactive ``/dev/tty`` auth (an invisible hang
        inside a spinner block).
        """
        if not self.socket_path.exists():
            return False
        if not self._mux_alive():
            _LOG.debug("is_active(%s): structural -O check failed", self.gateway)
            return False
        if deep is None:
            deep = not (self._established_this_session or self._recently_verified())
        if not deep:
            return True
        return self._probe_end_to_end(sleep=sleep)

    # -- what the end-to-end probe remembers ----------------------------

    def _recently_verified(self, *, now=time.monotonic) -> bool:
        """True if this socket carried a session for us just now."""
        with _VERIFIED_GUARD:
            when = _VERIFIED.get(str(self.socket_path))
        return when is not None and (now() - when) < _LIVENESS_CACHE_TTL

    def _note_verified(self, *, now=time.monotonic) -> None:
        """Record that this socket carried a session for us."""
        with _VERIFIED_GUARD:
            _VERIFIED[str(self.socket_path)] = now()

    def _forget_verified(self) -> None:
        """Drop the remembered verdict: the socket is gone or suspect."""
        with _VERIFIED_GUARD:
            _VERIFIED.pop(str(self.socket_path), None)

    def _mux_alive(self) -> bool:
        """Structural liveness: is the local mux daemon running? (no remote shell)."""
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o", "BatchMode=yes",
                    "-o", f"ControlPath={self.socket_path}",
                    "-O", "check",
                    self._format_host(self.gateway),
                ],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                check=False,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            return False
        return result.returncode == 0

    def _probe_end_to_end(self, *, sleep=time.sleep) -> bool:
        """End-to-end liveness: open a real session through the mux.

        Retried to tell a *slow* master (busy login node, high shell-spawn
        latency) apart from a *dead* one (zombie mux).  Logs the reason on
        every failure so a wild recurrence is diagnosable without
        ``--debug-ssh`` --- which perturbs the socket state and masks the
        bug.  ``BatchMode=yes`` keeps a bad mux from falling through to
        interactive ``/dev/tty`` auth.

        A refusal (``Session open refused by peer``) is NOT a failure here.
        It is the remote sshd answering that the master's one session slot
        is taken, and only a working transport could have carried that
        answer back; reading it as a dead master is how a merely busy
        connection earned itself a full re-authentication, which on this
        gateway can cost an OTP.
        """
        last_rc: Optional[int] = None
        last_err = ""
        lock = session_lock(self.socket_path)
        for attempt in range(1, _LIVENESS_PROBE_ATTEMPTS + 1):
            try:
                with lock:
                    result = subprocess.run(
                        [
                            "ssh",
                            "-o", "BatchMode=yes",
                            "-o", "ControlMaster=auto",
                            "-o", f"ControlPath={self.socket_path}",
                            "-o", "ConnectTimeout=5",
                            *self.extra_options,
                            self._format_host(self.gateway),
                            "true",
                        ],
                        capture_output=True,
                        stdin=subprocess.DEVNULL,
                        text=True,
                        check=False,
                        timeout=_LIVENESS_PROBE_TIMEOUT,
                    )
            except subprocess.TimeoutExpired:
                last_rc, last_err = None, "timeout"
                _LOG.debug(
                    "is_active(%s): probe %d/%d timed out after %ss",
                    self.gateway, attempt, _LIVENESS_PROBE_ATTEMPTS,
                    _LIVENESS_PROBE_TIMEOUT,
                )
            else:
                if result.returncode == 0:
                    self._note_verified()
                    return True
                last_rc = result.returncode
                err = (result.stderr or "").strip()
                last_err = err.splitlines()[-1] if err else ""
                if is_session_busy_error(err):
                    _LOG.debug(
                        "is_active(%s): master busy (one session at a time); "
                        "treating as alive", self.gateway,
                    )
                    self._note_verified()
                    return True
                _LOG.debug(
                    "is_active(%s): probe %d/%d rc=%s%s",
                    self.gateway, attempt, _LIVENESS_PROBE_ATTEMPTS, last_rc,
                    f": {last_err}" if last_err else "",
                )
            if attempt < _LIVENESS_PROBE_ATTEMPTS:
                sleep(_LIVENESS_PROBE_BACKOFF)
        _LOG.info(
            "is_active(%s): end-to-end probe failed after %d attempts "
            "(last rc=%s%s); treating connection as expired",
            self.gateway, _LIVENESS_PROBE_ATTEMPTS, last_rc,
            f": {last_err}" if last_err else "",
        )
        return False

    def establish(self, logger: logging.Logger) -> None:
        """Open a ControlMaster connection (may prompt for credentials).

        If a live socket already exists this is a no-op.  If a stale
        socket file remains from a previous session it is removed
        first.  When ``jump_host`` is set, the jump host's
        ControlMaster is used for the first hop.
        """
        if self.is_active():
            logger.debug("ControlMaster to %s already active", self.gateway)
            return

        # Ensure the jump host ControlMaster is alive first.
        if self.jump_control is not None:
            self.jump_control.ensure(logger)

        # Clean up stale socket if present.
        if self.socket_path.exists():
            logger.debug("Removing stale control socket %s", self.socket_path)
            try:
                self.socket_path.unlink()
            except OSError:
                pass

        logger.info(
            "Establishing SSH connection to %s (authentication may be required)",
            self.gateway,
        )
        cmd = [
            "ssh",
            "-o", "ControlMaster=yes",
            "-o", f"ControlPath={self.socket_path}",
            "-o", f"ControlPersist={self.control_persist}",
            "-o", f"ServerAliveInterval={self.keepalive_interval}",
            "-o", f"ServerAliveCountMax={self.keepalive_count_max}",
        ]
        if self.user:
            cmd.extend(["-o", f"User={self.user}"])
        # Route through jump host's ControlMaster if available.
        if self.jump_host:
            if self.jump_control and self.jump_control.is_active():
                cmd.extend([
                    "-J", self._format_host(self.jump_host),
                    "-o", f"ProxyJump={self._format_host(self.jump_host)}",
                ])
                # Make the ProxyJump itself use the gateway ControlMaster.
                # SSH respects ControlPath for ProxyJump targets.
                cmd[:0] = []  # placeholder; options added below
                # Reconstruct: we need ProxyJump to use the gateway socket.
                # Accept host keys for internal nodes (compute nodes are
                # dynamically assigned and may not be in known_hosts yet).
                cmd = [
                    "ssh",
                    "-o", "ControlMaster=yes",
                    "-o", f"ControlPath={self.socket_path}",
                    "-o", f"ControlPersist={self.control_persist}",
                    "-o", f"ServerAliveInterval={self.keepalive_interval}",
                    "-o", f"ServerAliveCountMax={self.keepalive_count_max}",
                    "-o", "StrictHostKeyChecking=accept-new",
                    "-o", f"ProxyCommand=ssh -o ControlMaster=auto "
                          f"-o ControlPath={self.jump_control.socket_path} "
                          f"-W %h:%p {self._format_host(self.jump_host)}",
                ]
            else:
                cmd.extend(["-J", self._format_host(self.jump_host)])

        # Present a gateway SSH certificate (direct hop only --- see
        # _identity_options).
        cmd += self._identity_options()
        cmd.extend(self.extra_options)
        if self.debug:
            cmd.append("-vvv")
        cmd.extend(["-fN", self._format_host(self.gateway)])
        logger.debug("ControlMaster command: %s", cmd)

        # Capture stderr to a temp file (NOT subprocess.PIPE) so callers
        # can classify the failure --- e.g. a transient
        # ``kex_exchange_identification`` worth retrying --- without
        # breaking two things:
        #   1. The interactive auth prompt: ssh reads passwords/OTP from
        #      ``/dev/tty``, not stderr, so redirecting stderr does not
        #      suppress the prompt (stdin/stdout stay inherited too).
        #   2. ``-fN`` backgrounding: the master forks and holds its
        #      stderr open for the life of the connection.  A PIPE would
        #      never see EOF and ``run`` would hang on success; a regular
        #      file has no such dependency --- ``run`` returns as soon as
        #      the foreground process exits.
        with tempfile.TemporaryFile() as errfile:
            try:
                subprocess.run(cmd, check=True, stderr=errfile)
            except subprocess.CalledProcessError as exc:
                errfile.seek(0)
                stderr = errfile.read().decode("utf-8", "replace")
                # Keep the failure reason visible exactly as before: the
                # bare ``kex_exchange_identification`` / ``Connection
                # closed`` line in plain mode, or the full ``-vvv`` trace
                # under --debug-ssh.
                if stderr:
                    sys.stderr.write(
                        stderr if stderr.endswith("\n") else stderr + "\n"
                    )
                raise TunnelError(
                    f"Failed to establish SSH connection to {self.gateway}",
                    stderr=stderr,
                ) from exc
            # Under --debug-ssh the negotiation trace lands on the captured
            # stderr even on success; re-emit it so the trace stays visible.
            if self.debug:
                errfile.seek(0)
                trace = errfile.read().decode("utf-8", "replace")
                if trace:
                    sys.stderr.write(trace)

        self._record_debug_mode()
        # We authenticated this master ourselves this run, so is_active()
        # may trust the cheap structural check for it (no remote shell) and
        # not re-authenticate a connection it just brought up.  Recorded on
        # the socket as well as on self, because a later command builds its
        # own SshControl for the same socket and should inherit the verdict
        # rather than re-prove it.
        self._established_this_session = True
        self._note_verified()

    def _identity_options(self) -> List[str]:
        """The ``-o`` flags that present our gateway certificate.

        Applied on the *direct* hop only (``jump_host is None``): login,
        DTN and compute hops ride the gateway mux and authenticate by
        publickey, and forcing ``IdentitiesOnly`` plus the gateway cert on
        them would break that.  A missing or expired cert degrades
        cleanly --- ssh offers no identity and falls back to the
        interactive prompt.
        """
        if self.jump_host is not None or not self.cert_file:
            return []
        return [
            "-o", f"IdentityFile={self.cert_file}",
            "-o", f"CertificateFile={self.cert_file}-cert.pub",
            "-o", "IdentitiesOnly=yes",
        ]

    @property
    def _debug_marker(self) -> Path:
        """Sidecar written when the socket WAS created with ``-vvv``."""
        return self.socket_path.with_suffix(".sock.debug")

    @property
    def _plain_marker(self) -> Path:
        """Sidecar written when the socket was NOT created with ``-vvv``.

        The counterpart matters: with only a debug marker, "no marker"
        meant both "created without -vvv" (the overwhelmingly common case)
        and "created before markers existed", and telling those apart cost
        a remote session on every single call --- see
        :meth:`_socket_debug_mode`.
        """
        return self.socket_path.with_suffix(".sock.plain")

    def _record_debug_mode(self) -> None:
        """Record which verbosity this socket was created with."""
        keep, drop = (
            (self._debug_marker, self._plain_marker) if self.debug
            else (self._plain_marker, self._debug_marker)
        )
        try:
            keep.touch()
        except OSError:  # a marker we cannot write is not worth failing over
            pass
        try:
            drop.unlink()
        except (FileNotFoundError, OSError):
            pass

    def _clear_debug_markers(self) -> None:
        """Remove both sidecars (the socket they describe is gone)."""
        for marker in (self._debug_marker, self._plain_marker):
            try:
                marker.unlink()
            except (FileNotFoundError, OSError):
                pass

    def _socket_debug_mode(self) -> Optional[bool]:
        """Was the live socket created with ``-vvv``?  ``None`` if unknowable.

        Answered from the sidecars, which :meth:`_record_debug_mode` writes
        for BOTH states, so an ordinary socket answers from a ``stat`` and
        costs nothing.

        Only a socket from before the sidecars existed needs the remote
        probe below, and its answer is *recorded*, so a legacy socket pays
        once in its life rather than once per command.  It used to pay
        every time: the old encoding was "marker present == debug", which
        made the normal non-debug case indistinguishable from a legacy one,
        so every ``ensure()`` opened a session to run ``true`` and look for
        ``debug1:`` on its stderr.  At ~8s per session open that was the
        single largest fixed cost in any remote command, and on a
        ``MaxSessions 1`` host it also spent the only session slot.

        An unanswered probe records nothing and returns ``None``: the
        caller must not tear down a working master over a guess.
        """
        if self._debug_marker.exists():
            return True
        if self._plain_marker.exists():
            return False
        if not self.socket_path.exists():
            return None
        # Legacy socket: ask it once.  A socket started with ``-vvv`` emits
        # ``debug1:`` lines on stderr even for a bare ``true``.  BatchMode
        # keeps an unattachable socket from blocking on /dev/tty.
        try:
            with session_lock(self.socket_path):
                result = subprocess.run(
                    [
                        "ssh",
                        "-o", "BatchMode=yes",
                        "-o", "ControlMaster=auto",
                        "-o", f"ControlPath={self.socket_path}",
                        "-o", "ConnectTimeout=5",
                        self._format_host(self.gateway),
                        "true",
                    ],
                    capture_output=True,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    check=False,
                    timeout=_LIVENESS_PROBE_TIMEOUT,
                )
        except subprocess.TimeoutExpired:
            return None
        if result.returncode != 0:
            # Busy or broken; either way the stderr is not evidence about
            # verbosity.  Record nothing and try again another time.
            return None
        found = "debug1:" in (result.stderr or "")
        marker = self._debug_marker if found else self._plain_marker
        try:
            marker.touch()
        except OSError:
            pass
        return found

    def _debug_mode_mismatch(self) -> bool:
        """True if the live socket was created with a different ``debug``."""
        socket_debug = self._socket_debug_mode()
        if socket_debug is None:
            return False
        return socket_debug != self.debug

    def ensure(self, logger: logging.Logger) -> None:
        """Ensure the ControlMaster is active, re-establishing if needed.

        Call this before any operation that needs the connection.  If
        the socket has expired, the user will be prompted to
        authenticate again.

        If the socket was created with a different ``debug`` setting
        (e.g. previous run used ``--debug-ssh`` but this one doesn't),
        the socket is closed and re-established so that SSH verbosity
        matches the current session.
        """
        if self.is_active():
            if self._debug_mode_mismatch():
                logger.info(
                    "SSH debug mode changed for %s, re-establishing connection",
                    self.gateway,
                )
                self.close(logger)
            else:
                return
        logger.info("SSH connection to %s expired, re-authenticating", self.gateway)
        self.establish(logger)

    def close(self, logger: logging.Logger) -> None:
        """Request a clean shutdown of the ControlMaster."""
        # We no longer own a live master; future is_active() calls must run
        # the full end-to-end probe rather than trusting the structural check.
        self._established_this_session = False
        self._forget_verified()
        if not self.socket_path.exists():
            self._clear_debug_markers()
            return
        subprocess.run(
            [
                "ssh",
                "-o", f"ControlPath={self.socket_path}",
                "-O", "exit",
                self._format_host(self.gateway),
            ],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        self._clear_debug_markers()
        logger.debug("ControlMaster to %s closed", self.gateway)

    def ssh_options(self, *, with_fallback: bool = False) -> List[str]:
        """Return the -o flags needed to reuse this ControlMaster.

        With ``ControlMaster=auto`` ssh reuses the live mux when it can.
        But if the mux refuses a new session (``mux_client_request_session:
        ... Session open refused by peer``), ssh falls back to opening a
        *fresh* connection to ``self.gateway`` directly.  For a jump-only
        host such as a pinned login node (``ln003.brc``), that direct dial
        fails with ``Could not resolve hostname`` because the name only
        resolves *inside* the gateway.

        ``with_fallback=True`` makes that fresh connection route through
        the jump host instead.  When a ``jump_control`` is set, the
        fallback reuses the jump host's own ControlMaster socket (no
        re-auth); otherwise it emits a plain ``ProxyJump``.  This mirrors
        :meth:`establish`'s jump handling so one-off commands survive a
        wedged mux without trying to resolve a jump-only hostname locally.

        For the *gateway itself* there is no jump to route through, and
        that fallback used to dial out with no identity options at all:
        ssh then offered every key the agent holds and the gateway
        answered ``Too many authentication failures`` (observed
        2026-09-20, on the refusal a concurrent sweep provoked).  Give it
        the same certificate :meth:`establish` presents, so a fallback
        authenticates the way the master did instead of hammering the
        host --- which, with fail2ban in front of it, is worse than the
        failure it was papering over.
        """
        opts = [
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={self.socket_path}",
        ]
        if self.user:
            opts.extend(["-o", f"User={self.user}"])
        if with_fallback:
            if self.jump_host and self.jump_control is not None:
                opts.extend([
                    "-o",
                    "ProxyCommand=ssh -o ControlMaster=auto "
                    f"-o ControlPath={self.jump_control.socket_path} "
                    f"-W %h:%p {self._format_host(self.jump_host)}",
                ])
            elif self.jump_host:
                opts.extend(["-o", f"ProxyJump={self._format_host(self.jump_host)}"])
            else:
                opts.extend(self._identity_options())
        return opts


# ------------------------------------------------------------------
# Port-forward tunnel (goes through the ControlMaster)
# ------------------------------------------------------------------


@dataclass
class SshTunnel:
    """Manages a local port forward through an SSH gateway.

    The tunnel forwards ``localhost:<local_port>`` to
    ``<target_host>:<target_port>`` via the gateway.  When a
    :class:`SshControl` is provided, the tunnel reuses the existing
    ControlMaster connection (no re-authentication).
    """

    gateway: str
    target_host: str
    target_port: int = 22
    local_port: Optional[int] = None
    keepalive_interval: int = 30
    keepalive_count_max: int = 120
    control: Optional[SshControl] = field(default=None, repr=False)
    _pid: Optional[int] = field(default=None, repr=False)

    def open(self, logger: logging.Logger) -> int:
        """Open the tunnel and return the local port.

        If ``local_port`` is ``None``, an ephemeral port is selected
        automatically.  The SSH process runs in the background (``-f``).
        """
        if self.local_port is None:
            self.local_port = _find_free_port()

        forward_spec = f"{self.local_port}:{self.target_host}:{self.target_port}"

        cmd = [
            "ssh",
            "-f",                               # background after auth
            "-N",                               # no remote command
            "-L", forward_spec,                 # local forward
            "-o", "ExitOnForwardFailure=yes",
            "-o", f"ServerAliveInterval={self.keepalive_interval}",
            "-o", f"ServerAliveCountMax={self.keepalive_count_max}",
        ]
        # Reuse ControlMaster if available — no re-authentication.
        if self.control is not None:
            cmd.extend(self.control.ssh_options())

        gateway = self.gateway
        if self.control and self.control.user:
            gateway = f"{self.control.user}@{self.gateway}"
        cmd.append(gateway)

        logger.info(
            "Opening SSH tunnel localhost:%d -> %s:%d via %s",
            self.local_port,
            self.target_host,
            self.target_port,
            self.gateway,
        )
        logger.debug("Tunnel command: %s", cmd)

        try:
            proc = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise TunnelError(
                f"Failed to open SSH tunnel: {exc.stderr.strip()}",
                stderr=exc.stderr or "",
            ) from exc

        # ssh -f backgrounds itself; find the PID by scanning for our port.
        self._pid = self._find_tunnel_pid()
        logger.debug("Tunnel PID: %s", self._pid)

        return self.local_port

    def is_alive(self) -> bool:
        """Check whether the tunnel process is still running."""
        if self._pid is None:
            return False
        try:
            os.kill(self._pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def close(self) -> None:
        """Terminate the tunnel process if it is running."""
        if self._pid is not None:
            try:
                os.kill(self._pid, 15)  # SIGTERM
            except (OSError, ProcessLookupError):
                pass
            self._pid = None

    @classmethod
    def from_session(
        cls,
        gateway: str,
        target_host: str,
        tunnel_port: Optional[int] = None,
        tunnel_pid: Optional[int] = None,
        target_port: int = 22,
        control: Optional[SshControl] = None,
    ) -> "SshTunnel":
        """Reconstruct a tunnel handle from saved session state."""
        tunnel = cls(
            gateway=gateway,
            target_host=target_host,
            target_port=target_port,
            local_port=tunnel_port,
            control=control,
        )
        tunnel._pid = tunnel_pid
        return tunnel

    def _find_tunnel_pid(self) -> Optional[int]:
        """Best-effort PID discovery for the backgrounded ssh process."""
        if self.local_port is None:
            return None
        try:
            result = subprocess.run(
                ["pgrep", "-f", f"-L {self.local_port}:{self.target_host}:{self.target_port}"],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if line.isdigit():
                    return int(line)
        except FileNotFoundError:
            pass  # pgrep not available
        return None
