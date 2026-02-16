"""Logging helpers for the sucoder."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional


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

    return logger
