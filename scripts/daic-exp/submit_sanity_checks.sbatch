#!/bin/bash
#SBATCH --job-name=sde-sanity
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/sde-sanity/%x-%j.out
#SBATCH --error=logs/sde-sanity/%x-%j.err
#
# CPU-only: SDE sanity checks (Check 1 stationary + Check 2 oracle round-trip).
# NumPy/float64 throughout — no GPU. Submit:
#   source scripts/daic-exp/env.sh
#   sbatch scripts/daic-exp/submit_sanity_checks.sbatch

set -euo pipefail
mkdir -p logs/sde-sanity
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic-exp/_common.sh"

srun --export=ALL "${DAIC_PYTHON}" experiments/sde-sanity/run_sanity_checks.py
