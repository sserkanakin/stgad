#!/bin/bash
#SBATCH --job-name=eval-preflight
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/eval-diffusion/preflight-%j.out
#SBATCH --error=logs/eval-diffusion/preflight-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# Validate the eval pipeline end-to-end on a real A40 before submitting the
# 6 full-eval jobs. Runs:
#   A. File-presence checks across all 6 (dataset, prior) checkpoints
#   B. End-to-end smoke per dataset (B=16, K=50, n_steps=100 on 1 batch
#      of real test data) — same dimensions as the real eval
#   C. Prints recommended --time= per dataset for the real eval jobs
#
# Wall-clock: ~3-5 minutes on an A40, regardless of which qos.
#
# Submit:
#   sbatch scripts/daic/submit_preflight.sbatch

set -euo pipefail
mkdir -p logs/eval-diffusion
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

echo "Running eval preflight on $(hostname) at $(date)"
srun --export=ALL "${DAIC_PYTHON}" scripts/eval_preflight.py
