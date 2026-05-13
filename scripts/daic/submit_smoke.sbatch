#!/bin/bash
#SBATCH --job-name=daic-smoke
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# GPU smoke test — 5 minutes on an A40. Confirms CUDA works, project
# imports succeed, residuals + prior yaml load, the SDE forward marginal
# evaluates correctly, and wandb logs.
#
# Submit:
#   sbatch --export=DATASET=metrla scripts/daic/submit_smoke.sbatch

set -euo pipefail
mkdir -p logs
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${DATASET:=metrla}"

srun --export=ALL "${DAIC_PYTHON}" scripts/daic/smoke_test.py --dataset "${DATASET}"
