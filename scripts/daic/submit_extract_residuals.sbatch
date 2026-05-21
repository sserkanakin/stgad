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
#SBATCH --output=logs/extract-residuals/%x-%j.out
#SBATCH --error=logs/extract-residuals/%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# Run inference with the best checkpoint and dump train/val/test residuals.
# Submit:
#   sbatch --export=DATASET=metrla,SEED=0 scripts/daic/submit_extract_residuals.sbatch

set -euo pipefail
mkdir -p logs/extract-residuals
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${DATASET:?Set DATASET=metrla|pemsbay|pems08f}"
: "${SEED:?Set SEED=<int>}"

scontrol update job="${SLURM_JOB_ID}" JobName="extr-${DATASET}-s${SEED}" 2>/dev/null || true

srun --export=ALL "${DAIC_PYTHON}" scripts/extract_residuals.py \
    --dataset "${DATASET}" \
    --seed "${SEED}"
