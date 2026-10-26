#!/bin/bash
# Profile-aware adaptation of the existing LSMS burst runner. Resource options
# are examples and can be overridden at sbatch. Review before installing over
# the shared burst/burst.sbatch; existing submitted jobs remain independent.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
set -euo pipefail
umask 077
stage=${1:?usage: sbatch burst.sbatch FROZEN_VERIFICATION_DIRECTORY}
# All source/profile/helper/command bytes were frozen by verify-stage BEFORE
# sbatch. The standalone driver discards inherited coordinator runtime state.
exec python3 "$stage/dispatch_verify.py" "$stage"
