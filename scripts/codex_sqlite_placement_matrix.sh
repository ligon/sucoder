#!/bin/bash
# Slice 0 read-only inventory matrix: one isolated Codex launch per case.
# See docs/codex-local-state-spike.org.  Run inside a Slurm allocation with
# local-disk tiering; every case uses an isolated HOME and CODEX_HOME.
set -u
umask 077
SP=$(dirname "$(readlink -f "$0")")
export SUCODER_REPO=${SUCODER_REPO:-$(dirname "$SP")}
export SPIKE_ROOT=${SPIKE_ROOT:-${SUCODER_LOCAL_ROOT:-/tmp}/spike-codex-state}
mkdir -p -m 700 "$SPIKE_ROOT/logs"
R=$SPIKE_ROOT/cases
LOG=$SPIKE_ROOT/logs/matrix.log
run() { name=$1; shift; echo "### $(date -u +%FT%TZ) $name: $*" >> "$LOG"; timeout 300 python3 -I "$SP/codex_sqlite_placement_spike.py" --case "$name" "$@" >> "$LOG" 2>&1; echo "### exit=$? $(date -u +%FT%TZ)" >> "$LOG"; }
run c0-default
run c2-env            --sqlite-env    $R/c2-env/sqlite-env
run c3-env-vs-cfg     --sqlite-env    $R/c3-env-vs-cfg/sqlite-env    --sqlite-config $R/c3-env-vs-cfg/sqlite-cfg
run c4-toml-vs-cfg    --sqlite-toml   $R/c4-toml-vs-cfg/sqlite-toml  --sqlite-config $R/c4-toml-vs-cfg/sqlite-cfg
run c10-env-vs-toml   --sqlite-env    $R/c10-env-vs-toml/sqlite-env  --sqlite-toml   $R/c10-env-vs-toml/sqlite-toml
run c5-relative       --sqlite-config rel-sqlite
run c6-missing        --sqlite-config $R/c6-missing/does-not-exist/sqlite --no-create
run c7-unwritable     --sqlite-config $R/c7-unwritable/sqlite-ro --unwritable
run c8-rc-nodaemon    --mode remote-control --sqlite-config $R/c8-rc-nodaemon/sqlite-cfg
run c9-tui            --mode tui            --sqlite-config $R/c9-tui/sqlite-cfg
echo "### MATRIX DONE $(date -u +%FT%TZ)" >> "$LOG"
grep -E '^(### |== )' "$LOG"
