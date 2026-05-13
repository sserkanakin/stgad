#!/bin/bash
#SBATCH --job-name=gwn-extract
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# Run inference with the best checkpoint and dump train/val/test residuals.
# Submit:
#   sbatch --export=DATASET=metrla,SEED=0 scripts/daic/submit_extract_residuals.sbatch

set -euo pipefail
mkdir -p logs
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${DATASET:?Set DATASET=metrla or DATASET=pemsbay}"
: "${SEED:?Set SEED=<int>}"

srun --export=ALL "${DAIC_PYTHON}" scripts/extract_residuals.py \
    --dataset "${DATASET}" \
    --seed "${SEED}"
