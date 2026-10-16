#!/bin/bash
#SBATCH --job-name=codex-state-spike
#SBATCH --account=co_carleton
#SBATCH --partition=savio4_htc
#SBATCH --qos=carleton_htc4_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --no-requeue
#SBATCH --output=/global/scratch/fsa/fc_jevons/ligon/.sucoder-state-spike/job39969504/outcomes-slurm-%j.out
# Slice 0 outcomes 3-7 burst (docs/codex-local-state-spike.org): a disposable
# co_carleton allocation that clones the shared mirror into its own /local and
# runs scripts/codex_state_outcomes.py against synthetic Codex homes only.
# --no-requeue: a preempted run must not silently start over.
set -u
umask 077
export PATH=/global/home/users/ligon/.local/bin:/global/home/users/ligon/.nvm/versions/node/v20.20.2/bin:/usr/bin:/bin
export SUCODER_LOCAL_ROOT=/local/job$SLURM_JOB_ID
if ! { [ -d "$SUCODER_LOCAL_ROOT" ] && [ -w "$SUCODER_LOCAL_ROOT" ]; }; then
  SUCODER_LOCAL_ROOT=${TMPDIR:-/tmp}/job$SLURM_JOB_ID
fi
mkdir -p -m 700 "$SUCODER_LOCAL_ROOT/mirrors" "$SUCODER_LOCAL_ROOT/tmp" || exit 1
export TMPDIR=$SUCODER_LOCAL_ROOT/tmp
TEST_ROOT=/global/scratch/fsa/fc_jevons/ligon/.sucoder-state-spike/job39969504
export SUCODER_SHARED_STATE_TEST_ROOT=$TEST_ROOT/outcomes-job$SLURM_JOB_ID
EV=$TEST_ROOT/evidence-outcomes/job$SLURM_JOB_ID
mkdir -p -m 700 "$SUCODER_SHARED_STATE_TEST_ROOT" "$EV" || exit 1
MIRROR=/global/home/users/ligon/mirrors/SuCoder
echo "host=$(hostname) job=$SLURM_JOB_ID utc=$(date -u +%FT%TZ) nproc=$(nproc) local=$SUCODER_LOCAL_ROOT"
cat /proc/self/cgroup
git clone -q --branch "${BRANCH:-main}" "$MIRROR" "$SUCODER_LOCAL_ROOT/mirrors/SuCoder" || exit 1
cd "$SUCODER_LOCAL_ROOT/mirrors/SuCoder" || exit 1
git log -1 --format='commit=%h %s'
codex --version
# Node images differ: ~/.local/bin/python3 -> /usr/bin/python3.12 dangles on
# some nodes and PATH then falls through to the system 3.6.  Resolve a usable
# interpreter explicitly; the global-software build is reachable everywhere.
PY=
for cand in /usr/bin/python3.12 /usr/bin/python3.11 \
    /global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/python-3.11.6-jitbcnrgutxjywtsrijww5lahxpwak2c/bin/python3; do
  if [ -x "$cand" ] && "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
    PY=$cand
    break
  fi
done
if [ -z "$PY" ]; then echo "no python >= 3.8 found"; exit 1; fi
echo "python=$PY $("$PY" --version)"
export SUCODER_REPO=$PWD SPIKE_ROOT=$SUCODER_LOCAL_ROOT/spike-codex-outcomes
"$PY" -I scripts/codex_state_outcomes.py all --evidence "$EV" \
  --turns "${TURNS:-6}" --runs "${RUNS:-20}" --reps "${REPS:-3}" --concurrency 2 \
  --latency-turns "${LATENCY_TURNS:-30}" --placements local,lustre,local_home \
  --watch "${WATCH:-30}" --seed 20261015 --lustre-root "$SUCODER_SHARED_STATE_TEST_ROOT" 2>&1 | tee "$EV/driver.log"
echo "done utc=$(date -u +%FT%TZ) local_disk_used=$(du -sh "$SPIKE_ROOT" 2>/dev/null | cut -f1)"
