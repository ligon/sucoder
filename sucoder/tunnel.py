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
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


class TunnelError(RuntimeError):
    """Raised when SSH connection or tunnel operations fail."""


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
    control_persist: str = "12h"
    jump_host: Optional[str] = None
    jump_control: Optional["SshControl"] = field(default=None, repr=False)

    @property
    def socket_path(self) -> Path:
        return _control_socket_path(self.gateway)

    def is_active(self) -> bool:
        """Return True if a ControlMaster socket exists and is live."""
        if not self.socket_path.exists():
            return False
        result = subprocess.run(
            [
                "ssh",
                "-o", f"ControlPath={self.socket_path}",
                "-O", "check",
                self.gateway,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
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
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
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
                cmd = [
                    "ssh",
                    "-o", "ControlMaster=yes",
                    "-o", f"ControlPath={self.socket_path}",
                    "-o", f"ControlPersist={self.control_persist}",
                    "-o", "ServerAliveInterval=30",
                    "-o", "ServerAliveCountMax=3",
                    "-o", f"ProxyCommand=ssh -o ControlMaster=auto "
                          f"-o ControlPath={self.jump_control.socket_path} "
                          f"-W %h:%p {self.jump_host}",
                ]
            else:
                cmd.extend(["-J", self.jump_host])

        cmd.extend(["-fN", self.gateway])
        logger.debug("ControlMaster command: %s", cmd)

        try:
            # Not capturing output --- the auth prompt needs the terminal.
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            raise TunnelError(
                f"Failed to establish SSH connection to {self.gateway}"
            ) from exc

    def ensure(self, logger: logging.Logger) -> None:
        """Ensure the ControlMaster is active, re-establishing if needed.

        Call this before any operation that needs the connection.  If
        the socket has expired, the user will be prompted to
        authenticate again.
        """
        if self.is_active():
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
            text=True,
            check=False,
        )
        logger.debug("ControlMaster to %s closed", self.gateway)

    def ssh_options(self) -> List[str]:
        """Return the -o flags needed to reuse this ControlMaster."""
        return [
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={self.socket_path}",
        ]


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
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
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
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise TunnelError(
                f"Failed to open SSH tunnel: {exc.stderr.strip()}"
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
