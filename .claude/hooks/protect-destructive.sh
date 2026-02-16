#!/usr/bin/env bash
# Hook: PreToolUse / Bash
# Blocks destructive shell commands that could cause data loss.
# Exit 2 = block the tool call; anything else = allow.

set -euo pipefail

# Read the tool input JSON from stdin
input=$(cat)
command=$(printf '%s' "$input" | python3 -c "import sys,json; print(json.load(sys.stdin).get('input',{}).get('command',''))" 2>/dev/null || echo "")

# Patterns that should never run without explicit human request.
# The agent's system prompt says "avoid destructive commands unless explicitly
# requested", but instructions are advisory.  This hook enforces it.
destructive_patterns=(
    'rm[[:space:]]+-[[:alnum:]]*r[[:alnum:]]*f'   # rm -rf (any flag order)
    'rm[[:space:]]+-[[:alnum:]]*f[[:alnum:]]*r'   # rm -fr
    'git[[:space:]]+reset[[:space:]]+--hard'
    'git[[:space:]]+push[[:space:]]+.*--force'
    'git[[:space:]]+push[[:space:]]+-f[[:space:]]'
    'git[[:space:]]+clean[[:space:]]+-[[:alnum:]]*f'
    'git[[:space:]]+checkout[[:space:]]+\.'
    'git[[:space:]]+restore[[:space:]]+\.'
)

for pattern in "${destructive_patterns[@]}"; do
    if [[ "$command" =~ $pattern ]]; then
        echo "BLOCKED: destructive command matched pattern: $pattern" >&2
        exit 2
    fi
done

exit 0
