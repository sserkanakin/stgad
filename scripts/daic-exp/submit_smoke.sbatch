#!/bin/bash
#SBATCH --job-name=daic-exp-smoke
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/smoke/%x-%j.out
#SBATCH --error=logs/smoke/%x-%j.err
#
# GPU smoke test on experimental DAIC. 5 minutes on an A40. Confirms CUDA
# works, imports succeed, residuals + prior yaml load, the SDE forward
# marginal evaluates correctly, and wandb logs.
#
# Submit:
#   source scripts/daic-exp/env.sh        # one-time per shell
#   sbatch --export=DATASET=metrla scripts/daic-exp/submit_smoke.sbatch
#
# To pick a different GPU:
#   sbatch --gres=gpu:nvidia_l40:1 --export=DATASET=metrla scripts/daic-exp/submit_smoke.sbatch

set -euo pipefail
mkdir -p logs/smoke
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic-exp/_common.sh"

: "${DATASET:=metrla}"

srun --export=ALL "${DAIC_PYTHON}" scripts/daic/smoke_test.py --dataset "${DATASET}"
