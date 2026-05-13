#!/bin/bash
#SBATCH --job-name=sde-sanity
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# CPU-only: re-run exp-004 sanity checks (Check 1 stationary + Check 2 oracle
# round-trip). NumPy/float64 throughout — no GPU. Submit:
#   sbatch scripts/daic/submit_sanity_checks.sbatch

set -euo pipefail
mkdir -p logs
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

srun --export=ALL "${DAIC_PYTHON}" experiments/exp-004/run_sanity_checks.py
