#!/bin/bash
#SBATCH --job-name=residuals
#SBATCH --time=00:30:00
# partition / qos / gres are injected per-cluster by submit.sh.
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
# residual extraction holds a whole split's predictions in memory before saving;
# pems-bay peaks ~9g, so 16g leaves headroom (8g OOM-killed there). any gpu is
# fine - this is short inference, not training.
#SBATCH --output=logs/residuals/%x-%j.out
#SBATCH --error=logs/residuals/%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# run inference with the best backbone checkpoint and dump train/val/test residuals.
#   sbatch --export=DATASET=metrla,SEED=0 scripts/daic/submit_extract_residuals.sbatch

set -euo pipefail
mkdir -p logs/residuals
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${DATASET:?Set DATASET=metrla|pemsbay|pems08f}"
: "${SEED:?Set SEED=<int>}"

export STAGE="residuals"
RUN_NAME="residuals_${DATASET}_seed${SEED}"
scontrol update job="${SLURM_JOB_ID}" JobName="${RUN_NAME}" 2>/dev/null || true

log_run_header "${DAIC_PYTHON} scripts/extract_residuals.py --dataset ${DATASET} --seed ${SEED}"

srun --export=ALL "${DAIC_PYTHON}" scripts/extract_residuals.py \
    --dataset "${DATASET}" \
    --seed "${SEED}"
