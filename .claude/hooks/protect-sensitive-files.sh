#!/usr/bin/env bash
# Hook: PreToolUse / Edit|Write
# Blocks writes to sensitive files (secrets, live config).
# Exit 2 = block the tool call; anything else = allow.

set -euo pipefail

input=$(cat)
file_path=$(printf '%s' "$input" | python3 -c "import sys,json; print(json.load(sys.stdin).get('input',{}).get('file_path',''))" 2>/dev/null || echo "")

# Basename for simple pattern matching
basename=$(basename "$file_path" 2>/dev/null || echo "")

# Block list: filenames and path fragments that should not be written by the agent.
blocked_names=(
    ".env"
    ".env.local"
    ".env.production"
    "credentials.json"
    "secrets.yaml"
    "secrets.yml"
    "id_rsa"
    "id_ed25519"
)

blocked_paths=(
    "/.sucoder/config.yaml"
)

for name in "${blocked_names[@]}"; do
    if [[ "$basename" == "$name" ]]; then
        echo "BLOCKED: write to sensitive file: $basename" >&2
        exit 2
    fi
done

for fragment in "${blocked_paths[@]}"; do
    if [[ "$file_path" == *"$fragment" ]]; then
        echo "BLOCKED: write to protected path: $file_path" >&2
        exit 2
    fi
done

exit 0
