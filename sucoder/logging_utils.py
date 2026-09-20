"""Logging helpers for the sucoder."""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Loggers that belong to the machinery rather than to a command, and whose
# records would otherwise land nowhere.  ``sucoder.tunnel`` reports why a
# master was declared dead, why a probe timed out, whether a refusal was
# read as "busy" -- exactly what you want when a command "just hangs", and
# exactly what `-v` used to fail to produce, because it configured only the
# command's own logger and this one propagated to a root with no handlers.
_SHARED_LOGGERS = ("sucoder.tunnel", "sucoder.remote")


def setup_logger(name: str, log_dir: Optional[Path], verbose: bool) -> logging.Logger:
    """Configure and return a logger with console and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to respect latest configuration.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(log_dir, 0o700)
        except PermissionError:
            logger.warning(
                "Unable to restrict permissions on log directory %s — "
                "logs may contain sensitive output (commands, paths, environment details). "
                "Ensure only the owning user can read this directory.",
                log_dir,
            )

        safe_name = re.sub(r"[^\w.-]", "_", name)
        log_path = log_dir / f"{safe_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        try:
            os.chmod(log_path, 0o600)
        except PermissionError:
            logger.warning("Unable to set permissions on log file %s", log_path)

    for shared in _SHARED_LOGGERS:
        _attach_handlers(logging.getLogger(shared), logger.handlers)

    return logger


def _attach_handlers(target: logging.Logger, handlers: Sequence[logging.Handler]) -> None:
    """Give *target* the same handlers, replacing any it already had.

    Replacing rather than adding: ``setup_logger`` may run more than once in
    a process (tests, a command that reconfigures), and a handler added
    twice prints twice.
    """
    target.setLevel(logging.DEBUG)
    for handler in list(target.handlers):
        target.removeHandler(handler)
    for handler in handlers:
        target.addHandler(handler)


# ------------------------------------------------------------------
# Progress: what a remote command is waiting on, while it waits
# ------------------------------------------------------------------
#
# A session open on a BRC login node costs ~8s before the remote command is
# even exec'd, and a command that makes a dozen round trips therefore sits
# silent for minutes.  Users read that as a hang and kill it.  One line per
# round trip, on *stderr* (so pipelines and `$(...)` are unaffected), named
# by host, printed BEFORE the trip: when something really is stuck, the
# last line names the host and the command it is stuck on.

_PROGRESS_ENABLED = False
_PROGRESS_LOCK = threading.Lock()

# ssh options that consume the following argument.  Needed to find the host
# in an ssh argv: the first bare token AFTER these pairs is the host.
_SSH_VALUE_FLAGS = frozenset("bcDEeFIiJLlmOoPpQRSWw")


def set_progress(enabled: bool) -> None:
    """Turn the per-round-trip progress lines on or off (process-wide)."""
    global _PROGRESS_ENABLED
    _PROGRESS_ENABLED = enabled


def progress_enabled() -> bool:
    return _PROGRESS_ENABLED


def progress(host: str, what: str) -> None:
    """Announce one remote round trip, before it is made."""
    if not _PROGRESS_ENABLED:
        return
    with _PROGRESS_LOCK:
        print(f"· {host}  {summarize_command(what)}", file=sys.stderr, flush=True)


def summarize_command(command) -> str:
    """One short line naming what a remote command does.

    Remote "commands" here are often whole shell scripts; the first
    meaningful line is what identifies them, and 60 characters is enough to
    tell `squeue` from `tmux list-sessions` without wrapping a narrow
    terminal.
    """
    if not isinstance(command, str):
        command = " ".join(str(part) for part in command)
    line = next(
        (part.strip() for part in command.splitlines() if part.strip()), "",
    )
    line = " ".join(line.split())
    return line if len(line) <= 60 else line[:57] + "..."


def describe_remote(argv: Sequence[str]) -> Optional[Tuple[str, str]]:
    """Return ``(host, what)`` if *argv* runs something on another machine.

    ``None`` for a local command, so a caller can hand it every command it
    runs and get progress only for the ones that cost a round trip.
    """
    if not argv:
        return None
    parts: List[str] = [str(a) for a in argv]
    program = Path(parts[0]).name
    if program == "ssh":
        rest = parts[1:]
        index = 0
        while index < len(rest):
            token = rest[index]
            if token.startswith("-"):
                # -oFoo=bar / -p22 carry their value in the same token;
                # -o Foo=bar takes the next one.
                if len(token) == 2 and token[1] in _SSH_VALUE_FLAGS:
                    index += 2
                else:
                    index += 1
                continue
            host = token.split("@")[-1]
            command = " ".join(rest[index + 1:]) or "(interactive session)"
            return host, command
        return None
    if program in ("git", "scp", "rsync"):
        for token in parts[1:]:
            if token.startswith("-") or "://" in token:
                continue
            head, sep, tail = token.partition(":")
            if sep and head and "/" not in head and " " not in head:
                verb = " ".join(parts[:2]) if len(parts) > 1 else program
                return head.split("@")[-1], f"{verb} {tail}"
    return None
