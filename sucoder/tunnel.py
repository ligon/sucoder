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
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


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

    @property
    def socket_path(self) -> Path:
        return _control_socket_path(self.gateway)

    def is_active(self) -> bool:
        """Return True if the ControlMaster connection actually works.

        ``ssh -O check`` only verifies the local mux daemon is alive.
        A zombie socket (mux running, TCP dead) passes that check but
        fails when a real session is requested.  We follow up with a
        lightweight ``true`` command to confirm end-to-end connectivity.

        Both probes use ``BatchMode=yes`` and a wall-clock timeout so
        that a stale or wedged mux can never fall through to interactive
        ``/dev/tty`` auth (which would hang silently inside a spinner
        block); if the socket is bad the probe fails fast and
        :meth:`ensure` will surface a visible re-auth prompt instead.
        """
        if not self.socket_path.exists():
            return False
        # Quick structural check --- is the mux daemon running?
        # Timeout guards against a wedged mux daemon; BatchMode keeps
        # ssh from prompting if -O check is somehow misrouted.
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o", "BatchMode=yes",
                    "-o", f"ControlPath={self.socket_path}",
                    "-O", "check",
                    self.gateway,
                ],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                check=False,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            return False
        if result.returncode != 0:
            return False
        # End-to-end check --- can we actually open a session?
        # BatchMode=yes is critical: without it, an unattachable mux
        # causes ssh to fall back to interactive auth on /dev/tty,
        # which bypasses stdin=DEVNULL and capture_output.
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o", "BatchMode=yes",
                    "-o", "ControlMaster=auto",
                    "-o", f"ControlPath={self.socket_path}",
                    "-o", "ConnectTimeout=5",
                    self.gateway,
                    "true",
                ],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                check=False,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            return False
        return result.returncode == 0

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
        # Route through jump host's ControlMaster if available.
        if self.jump_host:
            if self.jump_control and self.jump_control.is_active():
                cmd.extend([
                    "-J", self.jump_host,
                    "-o", f"ProxyJump={self.jump_host}",
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
                          f"-W %h:%p {self.jump_host}",
                ]
            else:
                cmd.extend(["-J", self.jump_host])

        cmd.extend(self.extra_options)
        if self.debug:
            cmd.append("-vvv")
        cmd.extend(["-fN", self.gateway])
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

    @property
    def _debug_marker(self) -> Path:
        """Sidecar file that records whether the socket was created with -vvv."""
        return self.socket_path.with_suffix(".sock.debug")

    def _record_debug_mode(self) -> None:
        """Write or remove the debug marker to match current ``self.debug``."""
        if self.debug:
            self._debug_marker.touch()
        else:
            try:
                self._debug_marker.unlink()
            except FileNotFoundError:
                pass

    def _debug_mode_mismatch(self) -> bool:
        """True if the live socket was created with a different debug setting.

        Handles legacy sockets (created before the marker feature) by
        probing the socket for debug output.  A socket started with
        ``-vvv`` emits ``debug1:`` lines on stderr even for a simple
        ``true`` command.
        """
        marker_exists = self._debug_marker.exists()
        if marker_exists != self.debug:
            return True
        # If no marker and not requesting debug, probe for legacy
        # debug sockets (created before the marker feature existed).
        # BatchMode=yes prevents the probe from blocking on /dev/tty if
        # the socket is somehow unattachable.
        if not self.debug and not marker_exists and self.socket_path.exists():
            try:
                result = subprocess.run(
                    [
                        "ssh",
                        "-o", "BatchMode=yes",
                        "-o", "ControlMaster=auto",
                        "-o", f"ControlPath={self.socket_path}",
                        "-o", "ConnectTimeout=5",
                        self.gateway,
                        "true",
                    ],
                    capture_output=True,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    check=False,
                    timeout=10,
                )
                if "debug1:" in (result.stderr or ""):
                    return True
            except subprocess.TimeoutExpired:
                pass
        return False

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
        if not self.socket_path.exists():
            return
        subprocess.run(
            [
                "ssh",
                "-o", f"ControlPath={self.socket_path}",
                "-O", "exit",
                self.gateway,
            ],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        # Clean up the debug marker.
        try:
            self._debug_marker.unlink()
        except FileNotFoundError:
            pass
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
        """
        opts = [
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={self.socket_path}",
        ]
        if with_fallback and self.jump_host:
            if self.jump_control is not None:
                opts.extend([
                    "-o",
                    "ProxyCommand=ssh -o ControlMaster=auto "
                    f"-o ControlPath={self.jump_control.socket_path} "
                    f"-W %h:%p {self.jump_host}",
                ])
            else:
                opts.extend(["-o", f"ProxyJump={self.jump_host}"])
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

        cmd.append(self.gateway)

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
