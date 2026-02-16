#!/usr/bin/env bash
# setup-agent-user.sh — Create the sandboxed agent user and configure group
# membership so the human and agent can share files through group permissions.
#
# Usage:
#   sudo ./scripts/setup-agent-user.sh                # defaults: agent=coder, human=<caller>
#   sudo ./scripts/setup-agent-user.sh --agent mybot   # custom agent username
#   sudo ./scripts/setup-agent-user.sh --human alice    # explicit human username
#
# What this script does:
#   1. Creates the agent group (default: coder)
#   2. Creates the agent user with that group, locked password, no login
#   3. Adds the human user to the agent group (so both can read shared files)
#   4. Sets the agent home directory permissions
#   5. Creates ~/.sucoder config directory with correct ownership
#
# This is the Unix-permissions sandboxing model: the agent is just another
# user on the box.  You share what it needs (mirrors, project files) via
# group permissions and don't share what it doesn't need (your keys, tokens).

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
AGENT_USER="coder"
AGENT_GROUP=""           # defaults to AGENT_USER if unset
HUMAN_USER="${SUDO_USER:-$(logname 2>/dev/null || echo "")}"
CONFIG_DIR=""            # defaults to ~HUMAN_USER/.sucoder
MIRROR_ROOT="/var/tmp/coder-mirrors"
DRY_RUN=false

# ── Parse arguments ───────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: sudo $0 [OPTIONS]

Options:
  --agent USER      Agent username (default: coder)
  --group GROUP     Agent group (default: same as agent username)
  --human USER      Human username (default: detected from SUDO_USER)
  --config-dir DIR  Config directory (default: ~HUMAN/.sucoder)
  --mirror-root DIR Mirror root directory (default: /var/tmp/coder-mirrors)
  --dry-run         Show what would be done without making changes
  -h, --help        Show this help message
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)      AGENT_USER="$2";   shift 2 ;;
        --group)      AGENT_GROUP="$2";  shift 2 ;;
        --human)      HUMAN_USER="$2";   shift 2 ;;
        --config-dir) CONFIG_DIR="$2";   shift 2 ;;
        --mirror-root) MIRROR_ROOT="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true;      shift   ;;
        -h|--help)    usage ;;
        *)            echo "Unknown option: $1" >&2; usage ;;
    esac
done

AGENT_GROUP="${AGENT_GROUP:-$AGENT_USER}"

if [[ -z "$HUMAN_USER" ]]; then
    echo "Error: Could not detect human username.  Pass --human <user>." >&2
    exit 1
fi

if [[ -z "$CONFIG_DIR" ]]; then
    HUMAN_HOME=$(eval echo "~$HUMAN_USER")
    CONFIG_DIR="$HUMAN_HOME/.sucoder"
fi

# ── Require root ──────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == false && $EUID -ne 0 ]]; then
    echo "Error: This script must be run with sudo (or --dry-run to preview)." >&2
    exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────────
run() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [dry-run] $*"
    else
        echo "  $*"
        "$@"
    fi
}

info() { echo "→ $*"; }
ok()   { echo "  ✓ $*"; }

# ── 1. Create agent group ────────────────────────────────────────────────
info "Creating group '$AGENT_GROUP' (if it doesn't exist)..."
if getent group "$AGENT_GROUP" >/dev/null 2>&1; then
    ok "Group '$AGENT_GROUP' already exists"
else
    run groupadd "$AGENT_GROUP"
    ok "Created group '$AGENT_GROUP'"
fi

# ── 2. Create agent user ─────────────────────────────────────────────────
info "Creating user '$AGENT_USER' (if it doesn't exist)..."
if id -u "$AGENT_USER" >/dev/null 2>&1; then
    ok "User '$AGENT_USER' already exists"
else
    run useradd -m -s /bin/bash -g "$AGENT_GROUP" "$AGENT_USER"
    run passwd -l "$AGENT_USER"
    ok "Created user '$AGENT_USER' with locked password"
fi

# ── 2b. Set restrictive umask for agent ──────────────────────────────────
AGENT_HOME=$(eval echo "~$AGENT_USER")
BASHRC="$AGENT_HOME/.bashrc"
info "Ensuring restrictive umask (0077) in $BASHRC..."
if [[ -f "$BASHRC" ]] && grep -q 'umask 0077' "$BASHRC" 2>/dev/null; then
    ok "umask 0077 already set in $BASHRC"
else
    run bash -c "echo 'umask 0077' >> $BASHRC"
    ok "Added 'umask 0077' to $BASHRC (agent files default to owner-only)"
fi

# ── 3. Add human user to agent group ─────────────────────────────────────
info "Adding '$HUMAN_USER' to group '$AGENT_GROUP'..."
if id -nG "$HUMAN_USER" 2>/dev/null | grep -qw "$AGENT_GROUP"; then
    ok "'$HUMAN_USER' is already in group '$AGENT_GROUP'"
else
    run usermod -aG "$AGENT_GROUP" "$HUMAN_USER"
    ok "Added '$HUMAN_USER' to group '$AGENT_GROUP'"
fi

# ── 4. Set agent home directory permissions ───────────────────────────────
info "Setting permissions on $AGENT_HOME..."
run chmod 755 "$AGENT_HOME"
ok "$AGENT_HOME is world-readable (agent needs access from sudo)"

# ── 5. Create config directory ────────────────────────────────────────────
info "Creating config directory $CONFIG_DIR..."
if [[ -d "$CONFIG_DIR" ]]; then
    ok "$CONFIG_DIR already exists"
else
    run install -d -m 750 -o "$HUMAN_USER" -g "$AGENT_GROUP" "$CONFIG_DIR"
    ok "Created $CONFIG_DIR (owner=$HUMAN_USER, group=$AGENT_GROUP)"
fi

# ── 6. Create mirror root ────────────────────────────────────────────────
info "Creating mirror root $MIRROR_ROOT..."
if [[ -d "$MIRROR_ROOT" ]]; then
    ok "$MIRROR_ROOT already exists"
else
    run install -d -m 2770 -o "$HUMAN_USER" -g "$AGENT_GROUP" "$MIRROR_ROOT"
    ok "Created $MIRROR_ROOT with setgid (owner=$HUMAN_USER, group=$AGENT_GROUP)"
fi

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "Done.  Summary:"
echo "  Agent user:    $AGENT_USER (group: $AGENT_GROUP)"
echo "  Human user:    $HUMAN_USER (added to group: $AGENT_GROUP)"
echo "  Config dir:    $CONFIG_DIR"
echo "  Mirror root:   $MIRROR_ROOT"
echo ""
echo "Next steps:"
echo "  1. Log out and back in (or run 'newgrp $AGENT_GROUP') for group membership to take effect."
echo "  2. Copy config.example.yaml to $CONFIG_DIR/config.yaml and edit it."
echo "  3. Copy default_system_prompt.org to $CONFIG_DIR/system_prompt.org and customize."
echo "  4. Run 'sucoder mirror-init <name>' to create your first mirror."
