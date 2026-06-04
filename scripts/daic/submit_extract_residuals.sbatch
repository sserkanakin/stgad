#!/bin/bash
#SBATCH --job-name=gwn-extract
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:5:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
# 24G was chosen so PEMS-BAY (36k windows × 325 sensors × 12 horizons) fits
# comfortably; 8G OOM-killed there. PEMS08F runs in ~3G.
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
