#!/bin/bash
#SBATCH --job-name=gwn-extract
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/extract-residuals/%x-%j.out
#SBATCH --error=logs/extract-residuals/%x-%j.err
#
# Run inference with the best checkpoint and dump train/val/test residuals.
# Submit:
#   source scripts/daic-exp/env.sh
#   sbatch --export=DATASET=metrla,SEED=0 scripts/daic-exp/submit_extract_residuals.sbatch

set -euo pipefail
mkdir -p logs/extract-residuals
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic-exp/_common.sh"

: "${DATASET:?Set DATASET=metrla or DATASET=pemsbay}"
: "${SEED:?Set SEED=<int>}"

scontrol update job="${SLURM_JOB_ID}" JobName="extr-${DATASET}-s${SEED}" 2>/dev/null || true

srun --export=ALL "${DAIC_PYTHON}" scripts/extract_residuals.py \
    --dataset "${DATASET}" \
    --seed "${SEED}"
