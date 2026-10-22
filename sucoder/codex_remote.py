"""Build a launch-owned Codex app server and optional terminal client.

The shell runs on the target, inside the existing tmux/SLURM boundary.  It
never starts, stops, or reconfigures the user's shared Codex daemon.
"""

from __future__ import annotations

import json
from typing import List, Optional, Sequence

from .agent_mode import _CODEX_VALUE_OPTIONS, _first_positional


# Keep this self-contained: SuCoder need not be installed on the target.
_SUPERVISOR = r'''set -u
mode=$1; count=$2; shift 2
server=("${@:1:count}"); shift "$count"
client=("$@")
umask 077
runtime=$(mktemp -d /tmp/sucoder-codex-XXXXXXXX) || exit 1
socket="unix://$runtime/server.sock"
server_pid=
client_pid=
cleanup() {
    final_status=$?
    trap - EXIT
    trap '' INT TERM HUP
    for pid in "$client_pid" "$server_pid"; do
        [ -z "$pid" ] || kill -TERM "$pid" 2>/dev/null || true
    done
    # Bound cleanup even if a child is stuck during shutdown.
    for attempt in {1..20}; do
        alive=false
        for pid in "$client_pid" "$server_pid"; do
            [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null || alive=true
        done
        "$alive" || break
        sleep 0.1
    done
    for pid in "$client_pid" "$server_pid"; do
        if [ -n "$pid" ]; then
            if kill -0 "$pid" 2>/dev/null; then
                kill -KILL "$pid" 2>/dev/null || true
                [ "$final_status" -ne 0 ] || final_status=137
            fi
            wait "$pid" 2>/dev/null; child_status=$?
            # A clean TUI exit cannot seal a crashed/forced server.
            if [ "$final_status" -eq 0 ] && [ "$child_status" -ne 0 ]; then
                final_status=$child_status
            fi
        fi
    done
    rm -rf -- "$runtime"
    exit "$final_status"
}
stop_server() {
    # cleanup waits for a graceful native exit, bounds it, and preserves
    # the native status. Never report a forced stop as a sealed owner.
    if [ "$mode" = server ]; then exit 0; fi
    exit 143
}
trap cleanup EXIT
if [ "$mode" = terminal ]; then trap '' INT; else trap stop_server INT; fi
trap stop_server TERM
trap 'exit 129' HUP

# Resolve to a native executable/shim in the same login/nvm environment as
# the normal launch. Probe the experimental flag before creating a server.
if ! "${server[0]}" app-server --remote-control --help >"$runtime/probe.log" 2>&1; then
    cat "$runtime/probe.log" >&2
    echo 'SuCoder: this Codex version does not support app-server remote control.' >&2
    exit 1
fi
echo "SuCoder: starting Codex remote control in $PWD" >&2
"${server[@]}" --listen "$socket" >"$runtime/server.log" 2>&1 &
server_pid=$!
ready=false
for attempt in {1..300}; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        wait "$server_pid"; status=$?
        server_pid=
        cat "$runtime/server.log" >&2
        echo 'SuCoder: Codex app-server exited before its local socket was ready.' >&2
        [ "$status" -ne 0 ] || status=1
        exit "$status"
    fi
    if [ -S "$runtime/server.sock" ]; then ready=true; break; fi
    sleep 0.1
done
if ! "$ready"; then
    cat "$runtime/server.log" >&2
    echo 'SuCoder: Codex app-server did not create its local socket within 30 seconds.' >&2
    exit 1
fi
echo "SuCoder: owned Codex server; local socket: $socket; server log: $runtime/server.log" >&2
if [ -n "${SUCODER_CODEX_REMOTE_HELPER:-}" ]; then
    python3 "$SUCODER_CODEX_REMOTE_HELPER" register --runtime "$runtime" --pid "$server_pid" || exit 1
    python3 "$SUCODER_CODEX_REMOTE_HELPER" wait --timeout 20 || true
fi
if [ "$mode" = server ]; then
    echo 'SuCoder: server only. Check relay status before pairing. Ctrl-C stops this server.' >&2
    wait "$server_pid"; status=$?
    server_pid=
    [ "$status" -eq 0 ] || cat "$runtime/server.log" >&2
    exit "$status"
fi
# Explicit stdin keeps the background client attached to the terminal while
# allowing traps to clean up both children on signals to the supervisor.
"${client[@]}" --remote "$socket" <&0 &
client_pid=$!
wait "$client_pid"; status=$?
client_pid=
if [ "$status" -ne 0 ]; then cat "$runtime/server.log" >&2; fi
exit "$status"
'''


def build_remote_command(
    command: Sequence[str], mode: str, *, instructions: Optional[str] = None,
    legacy_service: bool = False,
) -> List[str]:
    """Translate normal Codex settings, retaining terminal argv verbatim.

``instructions`` is one complete TOML override, or SuCoder's externalized
prelude sentinel. Passing it as its own argument lets remote launches stage
large prompts over SSH stdin using the existing prelude machinery.
    """
    if mode not in ("terminal", "server"):
        raise ValueError(f"Unknown Codex remote-control mode: {mode!r}")
    if legacy_service:
        # Detection already validated this service. Keep option values
        # atomic: a config value called remote-control is not a subcommand.
        at = _first_positional(command, 1, _CODEX_VALUE_OPTIONS)
        if at is None or command[at] != "remote-control":
            raise ValueError("Expected a Codex remote-control service command.")
        values = {"-c", "--config", "--enable", "--disable"}
        normalized = list(command[:at])
        pending_value = False
        for arg in command[at + 1:]:
            if pending_value:
                normalized.append(arg)
                pending_value = False
            elif arg in values:
                normalized.append(arg)
                pending_value = True
            elif arg != "--no-daemon":
                normalized.append(arg)
        command = normalized
    server = [command[0], "app-server", "--remote-control"]
    value_keys = {
        "--model": "model", "-m": "model",
        "--sandbox": "sandbox_mode", "-s": "sandbox_mode",
        "--ask-for-approval": "approval_policy", "-a": "approval_policy",
    }
    # Unknown arguments must not silently become different server defaults.
    passthrough = {"--config", "-c", "--enable", "--disable"}
    args = iter(command[1:])
    for arg in args:
        name, equals, inline = arg.partition("=")
        if name in value_keys or name in passthrough:
            value = inline if equals else next(args, None)
            if value is None:
                raise ValueError(f"Codex option {name} requires a value.")
            if name in value_keys:
                server.extend(["-c", value_keys[name] + "=" + json.dumps(value)])
            else:
                server.extend([name, value])
        elif arg == "--search":
            server.extend(["-c", 'web_search="live"'])
        elif arg == "--dangerously-bypass-approvals-and-sandbox":
            server.extend(["-c", 'sandbox_mode="danger-full-access"',
                           "-c", 'approval_policy="never"'])
        elif arg == "--no-alt-screen" and mode == "terminal":
            pass
        else:
            raise ValueError(
                f"Codex remote control does not support launch argument {arg!r}; "
                "use model/sandbox/approval options, -c, --enable, --disable, "
                "or --no-remote-control for the ordinary CLI."
            )
    if instructions:
        server.extend(["-c", instructions])
    return ["bash", "-c", _SUPERVISOR, "sucoder-codex-remote", mode,
            str(len(server)), *server, *command]
