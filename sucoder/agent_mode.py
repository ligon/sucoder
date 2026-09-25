"""Codex service invocations and their tmux metadata (ledger sections 2-5).

AgentType identifies the harness. A subcommand also determines whether its
stdin is a conversation; remote control is an app server, not a terminal UI.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


TERMINAL = "terminal"
CODEX_REMOTE_CONTROL = "codex-remote-control"
MODE_OPTION = "@sucoder-agent-mode"
LAUNCH_OPTION = "@sucoder-service-launch"

# Skip option *values* before looking for a subcommand: a model or prompt
# named remote-control must not turn an ordinary Codex launch into a service.
_CODEX_VALUE_OPTIONS = {
    "-c", "--config", "--enable", "--disable", "-m", "--model",
    "-p", "--profile", "-s", "--sandbox", "-a", "--ask-for-approval",
    "-C", "--cd", "-i", "--image", "--local-provider", "--add-dir",
    "--permission-profile", "-P",
}
_REMOTE_VALUE_OPTIONS = {"-c", "--config", "--enable", "--disable"}


def _first_positional(command: Sequence[str], start: int, options: set) -> Optional[int]:
    index = start
    while index < len(command):
        token = command[index]
        if token == "--":
            return None
        if token in options:
            index += 2
        elif token.startswith("-"):
            index += 1
        else:
            return index
    return None


def codex_remote_command(command: Sequence[str]) -> Optional[List[str]]:
    """Recognize a service launch and normalize ``start`` to foreground.

    SuCoder already supervises the process through tmux/Slurm (ledger 4).
    Management operations such as stop and pair belong outside agent launch.
    """
    if not command or Path(command[0]).name != "codex":
        return None
    index = _first_positional(command, 1, _CODEX_VALUE_OPTIONS)
    if index is None or command[index] != "remote-control":
        return None
    action = _first_positional(command, index + 1, _REMOTE_VALUE_OPTIONS)
    result = list(command)
    if action is not None:
        if command[action] != "start":
            raise ValueError(
                "Use 'codex remote-control' or 'codex remote-control start' "
                "to launch a service; run remote-control management commands separately."
            )
        del result[action]
        if _first_positional(result, index + 1, _REMOTE_VALUE_OPTIONS) is not None:
            raise ValueError("Codex remote-control does not accept a positional prompt.")
    return result


def config_override(key: str, value: str) -> str:
    """A single Codex -c argument, encoded as a TOML basic string.

    Preserve Unicode directly: JSON's ASCII encoding uses surrogate pairs for
    non-BMP characters, which TOML disallows. Escape DEL explicitly as well as
    JSON's usual control characters. Never shlex-split the resulting value.
    """
    encoded = json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
    return f"{key}={encoded}"


def service_launch(value: Any) -> Optional[Dict[str, Any]]:
    """Validate a recorded launch without trusting it as arbitrary code."""
    if isinstance(value, str):
        value = json.loads(value) if value.strip() else None
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Invalid remote-control launch record.")
    command, model = value.get("command"), value.get("model")
    if (not isinstance(command, list) or not command
            or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in command)
            or (model is not None and (not isinstance(model, str) or not model.strip()))):
        raise ValueError("Invalid remote-control command/model in session record.")
    normalized = codex_remote_command(command)
    if normalized is None:
        raise ValueError("Recorded service command is not Codex remote-control.")
    return {"command": normalized, "model": model}


def service_marker(launch: Dict[str, Any]) -> str:
    """Stamp only the pane that actually starts the service.

    Reusing an existing window does not execute this or relabel somebody
    else's process (ledger 4).
    """
    value = shlex.quote(json.dumps(launch, ensure_ascii=True))
    return (
        f'tmux set-option -p -t "$TMUX_PANE" {MODE_OPTION} {CODEX_REMOTE_CONTROL} && '
        f'tmux set-option -p -t "$TMUX_PANE" {LAUNCH_OPTION} {value} && '
    )


# These probes run inside the job for confined sessions. Both take session and
# socket arguments, like the existing pane probe; no launcher record is needed.
_PROBE_PREFIX = (
    'sess="$1"; sock="$2"; '
    'if [ -n "$sock" ]; then set -- -L "$sock"; else set --; fi; '
)
MODE_PROBE_SH = (
    _PROBE_PREFIX
    + 'modes=$(tmux "$@" list-panes -s -t "$sess" '
    + f'-F "#{{{MODE_OPTION}}}" 2>/dev/null) || exit 1; '
    + f'case "$modes" in *{CODEX_REMOTE_CONTROL}*) echo {CODEX_REMOTE_CONTROL};; '
    + f'*) echo {TERMINAL};; esac'
)
SERVICE_LAUNCH_PROBE_SH = (
    _PROBE_PREFIX
    + 'launches=$(tmux "$@" list-panes -s -t "$sess" '
    + f'-F "#{{{LAUNCH_OPTION}}}" 2>/dev/null) || exit 1; '
    + 'printf "%s\\n" "$launches" | '
    + 'while IFS= read -r launch; do '
    + '[ -z "$launch" ] || { printf "%s\\n" "$launch"; break; }; done'
)
