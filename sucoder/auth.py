"""Subscription <-> API-key auth switching for agent harnesses.

The API key lives only in the human user's ``pass`` store; it is never put in a
process environment variable.  Each harness authenticates differently:

- **codex** has no per-process key channel.  Switching to ``apikey`` pipes
  ``pass show <entry> | codex login --with-api-key`` (stdin) into the shared
  ``~/.codex/auth.json``, after backing the current (subscription) login up to a
  ``pass`` blob or a local ``.bak`` so ``subscription`` can restore it.
- **claude** reads ``apiKeyHelper`` from ``--settings`` JSON: an executable whose
  stdout is the key.  SuCoder stages a tiny helper that reads the key from
  ``pass``; OAuth stays on disk untouched, so the switch is per-launch and
  non-destructive.

The swap is host-global for codex (one shared ``auth.json``); two concurrent
codex sessions on one host cannot run mixed auth modes.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .config import Config, HarnessAuthConfig
from .executor import CommandExecutor


class AuthError(RuntimeError):
    pass


# Harness name -> (native CLI, default apikey credential, default subscription credential)
_HARNESSES = {
    "codex": ("codex", "openai", "codex-chatgpt-auth"),
    "claude": ("claude", "anthropic", None),
}


@dataclass
class AuthStatus:
    harness: str
    mode: str  # "subscription" | "apikey" | "unknown"
    detail: str


def _codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))


def _auth_config(config: Config, harness: str) -> HarnessAuthConfig:
    """Resolve the effective auth config for a harness (global launcher)."""
    launcher = config.agent_launcher
    if launcher is not None and harness in launcher.auth:
        return launcher.auth[harness]
    return HarnessAuthConfig()


def _credential_entry(config: Config, name: str) -> str:
    if name not in config.credentials:
        raise AuthError(
            f"Credential {name!r} is not defined under `credentials:` in the config."
        )
    return config.credentials[name].pass_entry


def _pass_show(executor: CommandExecutor, pass_entry: str) -> str:
    try:
        result = executor.run_human(
            ["pass", "show", pass_entry], check=False, capture_output=True,
        )
    except FileNotFoundError as exc:
        raise AuthError("`pass` is not installed or not on PATH.") from exc
    if result.returncode != 0:
        raise AuthError(
            f"Could not read pass entry {pass_entry!r} (exit {result.returncode})."
        )
    lines = result.stdout.splitlines()
    if not lines or not lines[0]:
        raise AuthError(f"Pass entry {pass_entry!r} has an empty first line.")
    return lines[0]


def _run_pass_insert(executor: CommandExecutor, pass_entry: str, value: str) -> None:
    argv = ["pass", "insert", "-m", "-f", pass_entry]
    if executor.human_user != os.environ.get("USER"):
        argv = ["sudo", "-n", "-u", executor.human_user, *argv]
    try:
        proc = subprocess.run(argv, input=value + "\n", capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AuthError("`pass` is not installed or not on PATH.") from exc
    if proc.returncode != 0:
        raise AuthError(
            f"Could not store pass entry {pass_entry!r}: {proc.stderr.strip()}"
        )


# ---------------------------------------------------------------------- codex

def _codex_backup_subscription(
    config: Config, executor: CommandExecutor, cfg: HarnessAuthConfig, logger: logging.Logger,
) -> None:
    auth_json = _codex_home() / "auth.json"
    if not auth_json.exists():
        return
    blob = auth_json.read_text(encoding="utf-8")
    sub_cred = cfg.subscription_credential or _HARNESSES["codex"][2]
    if sub_cred and sub_cred in config.credentials:
        entry = _credential_entry(config, sub_cred)
        _run_pass_insert(executor, entry, blob)
        logger.info("Backed up codex subscription auth.json to pass entry %r.", entry)
    else:
        bak = auth_json.with_suffix(".json.sucoder-bak")
        bak.write_text(blob, encoding="utf-8")
        os.chmod(bak, stat.S_IRUSR | stat.S_IWUSR)
        logger.info("Backed up codex subscription auth.json to %s.", bak)


def codex_to_apikey(
    config: Config, executor: CommandExecutor, cfg: HarnessAuthConfig, logger: logging.Logger,
) -> None:
    _codex_backup_subscription(config, executor, cfg, logger)
    cred = cfg.apikey_credential or _HARNESSES["codex"][1]
    entry = _credential_entry(config, cred)
    key = _pass_show(executor, entry)
    proc = subprocess.run(
        ["codex", "login", "--with-api-key"],
        input=key + "\n",
        capture_output=True,
        text=True,
        env={**os.environ, "CODEX_HOME": str(_codex_home())},
    )
    if proc.returncode != 0:
        raise AuthError(f"`codex login --with-api-key` failed: {proc.stderr.strip()}")
    logger.info("codex now authenticates with the API key from pass entry %r.", entry)


def codex_to_subscription(
    config: Config, executor: CommandExecutor, cfg: HarnessAuthConfig, logger: logging.Logger,
) -> None:
    auth_json = _codex_home() / "auth.json"
    sub_cred = cfg.subscription_credential or _HARNESSES["codex"][2]
    blob: Optional[str] = None
    if sub_cred and sub_cred in config.credentials:
        blob = _pass_show(executor, _credential_entry(config, sub_cred))
    else:
        bak = auth_json.with_suffix(".json.sucoder-bak")
        if bak.exists():
            blob = bak.read_text(encoding="utf-8")
    if not blob:
        raise AuthError(
            "No codex subscription backup found. Run `codex login` to restore "
            "ChatGPT authentication."
        )
    auth_json.write_text(blob, encoding="utf-8")
    os.chmod(auth_json, stat.S_IRUSR | stat.S_IWUSR)
    logger.info("Restored codex subscription auth.json.")


def codex_status(executor: CommandExecutor) -> AuthStatus:
    try:
        proc = subprocess.run(
            ["codex", "login", "status"], capture_output=True, text=True,
            env={**os.environ, "CODEX_HOME": str(_codex_home())},
        )
    except FileNotFoundError:
        return AuthStatus("codex", "unknown", "codex not installed")
    text = (proc.stdout + proc.stderr).strip()
    lowered = text.lower()
    if "chatgpt" in lowered:
        mode = "subscription"
    elif "api" in lowered and "key" in lowered:
        mode = "apikey"
    else:
        mode = "unknown"
    return AuthStatus("codex", mode, text)


# --------------------------------------------------------------------- claude

def claude_status(executor: CommandExecutor) -> AuthStatus:
    try:
        proc = subprocess.run(
            ["claude", "auth", "status"], capture_output=True, text=True,
        )
    except FileNotFoundError:
        return AuthStatus("claude", "unknown", "claude not installed")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return AuthStatus("claude", "unknown", proc.stdout.strip())
    method = data.get("authMethod", "")
    if not data.get("loggedIn"):
        return AuthStatus("claude", "unknown", "not logged in")
    if method == "api_key":
        mode = "apikey"
    elif method:
        mode = "subscription"
    else:
        mode = "unknown"
    sub = data.get("subscriptionType") or ""
    return AuthStatus("claude", mode, f"{method} {sub}".strip())


# --------------------------------------------------------------------- facade

def status(config: Config, executor: CommandExecutor, harness: str) -> AuthStatus:
    if harness == "codex":
        return codex_status(executor)
    if harness == "claude":
        return claude_status(executor)
    raise AuthError(f"Unknown harness {harness!r}.")


def to_apikey(
    config: Config, executor: CommandExecutor, harness: str, logger: logging.Logger,
) -> None:
    cfg = _auth_config(config, harness)
    if harness == "codex":
        codex_to_apikey(config, executor, cfg, logger)
        return
    if harness == "claude":
        # Per-launch via apiKeyHelper; nothing persistent to switch.  Surface the
        # credential that will be used so a missing one fails here, not at launch.
        cred = cfg.apikey_credential or _HARNESSES["claude"][1]
        _credential_entry(config, cred)
        logger.info(
            "claude uses the API key from credential %r per launch "
            "(set agent_launcher.auth.claude.mode=apikey or pass --auth apikey).", cred,
        )
        return
    raise AuthError(f"Unknown harness {harness!r}.")


def to_subscription(
    config: Config, executor: CommandExecutor, harness: str, logger: logging.Logger,
) -> None:
    cfg = _auth_config(config, harness)
    if harness == "codex":
        codex_to_subscription(config, executor, cfg, logger)
        return
    if harness == "claude":
        logger.info(
            "claude subscription login is untouched; launches use it unless "
            "agent_launcher.auth.claude.mode=apikey or --auth apikey is passed."
        )
        return
    raise AuthError(f"Unknown harness {harness!r}.")


def harnesses() -> Sequence[str]:
    return tuple(_HARNESSES)
