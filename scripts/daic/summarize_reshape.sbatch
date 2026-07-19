#!/bin/bash
#SBATCH --job-name=rsh-sum
# tiny no-GPU dependency job: runs AFTER all dataset arrays finish (afterany) and writes
# OUTPUT_ROOT/EXP/SUMMARY.md — the automatic end-of-run table. submitted by submit_reshape.sh.
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"
: "${EXP:?}" ; : "${OUTPUT_ROOT:?}"
echo "===== summarizing ${OUTPUT_ROOT}/${EXP} ====="
"${DAIC_PYTHON}" scripts/analysis/summarize_reshape.py --exp "${OUTPUT_ROOT}/${EXP}"
