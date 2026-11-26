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


def _agent_codex_home(config: Config) -> str:
    """The agent account's codex home (where its auth.json lives)."""
    return f"/home/{config.agent_user}/.codex"


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

def _read_agent_auth_json(executor: CommandExecutor, agent_codex_home: str) -> Optional[str]:
    """Read the agent's auth.json as the agent (coder), or None if absent."""
    result = executor.run_agent(
        ["cat", f"{agent_codex_home}/auth.json"], check=False, capture_output=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout


def _write_agent_auth_json(executor: CommandExecutor, agent_codex_home: str, blob: str) -> None:
    """Write the agent's auth.json (mode 600) via stdin, never argv."""
    executor.run_agent(
        [
            "sh", "-c",
            'umask 077 && mkdir -p "$(dirname "$1")" && cat > "$1" && chmod 600 "$1"',
            "sucoder-codex-auth", f"{agent_codex_home}/auth.json",
        ],
        input=blob,
        check=True,
        capture_output=True,
    )


def _is_codex_subscription(blob: str) -> bool:
    """Whether an auth.json blob holds a subscription login, not an API key.

    Backing up whatever is present would let a second switch to apikey replace
    the only copy of the subscription login with the API key itself.
    """
    try:
        data = json.loads(blob)
    except ValueError:
        return False
    return (
        isinstance(data, dict)
        and data.get("auth_mode", "chatgpt") != "apikey"
        and bool(data.get("tokens"))
    )


def _codex_backup_subscription(
    config: Config, executor: CommandExecutor, cfg: HarnessAuthConfig, logger: logging.Logger,
) -> None:
    agent_home = _agent_codex_home(config)
    blob = _read_agent_auth_json(executor, agent_home)
    if blob is None:
        return
    if not _is_codex_subscription(blob):
        logger.info(
            "codex is not logged in to a subscription; keeping any existing backup."
        )
        return
    sub_cred = cfg.subscription_credential or _HARNESSES["codex"][2]
    if sub_cred and sub_cred in config.credentials:
        entry = _credential_entry(config, sub_cred)
        _run_pass_insert(executor, entry, blob)
        logger.info("Backed up codex subscription auth.json to pass entry %r.", entry)
    else:
        executor.run_agent(
            [
                "sh", "-c",
                'umask 077 && cat > "$1" && chmod 600 "$1"',
                "sucoder-codex-auth-bak", f"{agent_home}/auth.json.sucoder-bak",
            ],
            input=blob,
            check=True,
            capture_output=True,
        )
        logger.info("Backed up codex subscription auth.json for the agent account.")


def codex_to_apikey(
    config: Config, executor: CommandExecutor, cfg: HarnessAuthConfig, logger: logging.Logger,
) -> None:
    """Swap codex to API-key auth.

    The key is read from the human's pass store (as ligon) and written to the
    agent account's ~/.codex/auth.json (as coder) via stdin -- the key crosses
    the account boundary once, in a pipe, and persists only in the agent's
    mode-600 auth.json (unavoidable: codex's only API-key mechanism stores the
    key in its own auth file).
    """
    _codex_backup_subscription(config, executor, cfg, logger)
    cred = cfg.apikey_credential or _HARNESSES["codex"][1]
    entry = _credential_entry(config, cred)
    key = _pass_show(executor, entry)
    blob = json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": key}, indent=2) + "\n"
    _write_agent_auth_json(executor, _agent_codex_home(config), blob)
    logger.info(
        "codex now authenticates with the API key from pass entry %r "
        "(written to the agent account's auth.json).", entry,
    )


def codex_to_subscription(
    config: Config, executor: CommandExecutor, cfg: HarnessAuthConfig, logger: logging.Logger,
) -> None:
    agent_home = _agent_codex_home(config)
    sub_cred = cfg.subscription_credential or _HARNESSES["codex"][2]
    blob: Optional[str] = None
    if sub_cred and sub_cred in config.credentials:
        blob = _pass_show(executor, _credential_entry(config, sub_cred))
    else:
        result = executor.run_agent(
            ["cat", f"{agent_home}/auth.json.sucoder-bak"], check=False, capture_output=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            blob = result.stdout
    if not blob:
        raise AuthError(
            "No codex subscription backup found. Run `codex login` as the agent "
            "account to restore ChatGPT authentication."
        )
    if not _is_codex_subscription(blob):
        raise AuthError(
            "The codex subscription backup holds no subscription login (it may be "
            "an API key); nothing was changed. Run `codex login` as the agent "
            "account to restore ChatGPT authentication."
        )
    _write_agent_auth_json(executor, agent_home, blob)
    logger.info("Restored the agent account's codex subscription auth.json.")


def _is_codex_apikey(blob: str) -> bool:
    """Whether an auth.json blob authenticates with an API key."""
    try:
        data = json.loads(blob)
    except ValueError:
        return False
    if not isinstance(data, dict) or _is_codex_subscription(blob):
        return False
    return data.get("auth_mode") == "apikey" or bool(data.get("OPENAI_API_KEY"))


def codex_ensure_subscription(
    config: Config, executor: CommandExecutor, cfg: HarnessAuthConfig, logger: logging.Logger,
) -> bool:
    """Undo an earlier API-key switch; return whether a login was restored.

    The switch rewrites the shared auth.json and outlives the launch that made
    it, so a later subscription launch would otherwise keep billing the key.
    Anything but an API key (a login, no file, an unknown form) is left alone.
    Raises AuthError when a key is in place and no login can be restored.
    """
    blob = _read_agent_auth_json(executor, _agent_codex_home(config))
    if blob is None or not _is_codex_apikey(blob):
        return False
    codex_to_subscription(config, executor, cfg, logger)
    return True


def codex_status(executor: CommandExecutor, config: Config) -> AuthStatus:
    result = executor.run_agent(
        ["codex", "login", "status"],
        check=False, capture_output=True,
        env={"CODEX_HOME": _agent_codex_home(config)},
    )
    if result.returncode != 0 and not result.stdout.strip():
        return AuthStatus("codex", "unknown", "codex not installed or not logged in")
    text = (result.stdout + result.stderr).strip()
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
        return codex_status(executor, config)
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
