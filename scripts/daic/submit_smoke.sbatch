#!/bin/bash
#SBATCH --job-name=daic-smoke
#SBATCH --time=00:05:00
# partition / qos / gres are injected per-cluster by submit.sh smoke.
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/smoke/%x-%j.out
#SBATCH --error=logs/smoke/%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# GPU smoke test — 5 minutes on an A40. Confirms CUDA works, project
# imports succeed, residuals + prior yaml load, the SDE forward marginal
# evaluates correctly, and wandb logs.
#
# Submit:
#   sbatch --export=DATASET=metrla scripts/daic/submit_smoke.sbatch

set -euo pipefail
mkdir -p logs/smoke
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${DATASET:=metrla}"

export STAGE="smoke"
RUN_NAME="smoke_${DATASET}"
scontrol update job="${SLURM_JOB_ID}" JobName="${RUN_NAME}" 2>/dev/null || true

log_run_header "${DAIC_PYTHON} scripts/daic/smoke_test.py --dataset ${DATASET}"

srun --export=ALL "${DAIC_PYTHON}" scripts/daic/smoke_test.py --dataset "${DATASET}"
