#!/bin/bash
# Bias-correct all GCM files on Levante: one array task per file.
#
# 1. Set the paths in run_one_file.sh (or export them before sbatch;
#    sbatch passes the environment on by default).
# 2. Set --array=0-<number of files - 1> and the account below.
# 3. sbatch examples/slurm_array.sh
#
#SBATCH --job-name=bcsd
#SBATCH --partition=shared
#SBATCH --account=bm0146
#SBATCH --array=0-24
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=bcsd_%A_%a.log
set -euo pipefail

# Under sbatch the script runs from a spool copy, so locate the repo via the submit directory.
REPO=${REPO:-$SLURM_SUBMIT_DIR}

# Memory per worker ~ tile_points x (file days + 2 x reference days) x 12 bytes:
# 32x32 tiles with a 30-year reference period need ~340 MB, so 8 workers fit in 8G.
export DEC=$SLURM_ARRAY_TASK_ID
export WORKERS=$SLURM_CPUS_PER_TASK
export CHUNK=${CHUNK:-32}

"$REPO/examples/run_one_file.sh" "$@"
