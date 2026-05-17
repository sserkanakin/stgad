#!/bin/bash
#SBATCH --job-name=fit-prior
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/fit-prior/%x-%j.out
#SBATCH --error=logs/fit-prior/%x-%j.err
#
# CPU-only: fit the Kronecker prior (NumPy + LAPACK; no GPU needed). Submit:
#   source scripts/daic-exp/env.sh
#   sbatch --export=DATASET=metrla scripts/daic-exp/submit_fit_prior.sbatch

set -euo pipefail
mkdir -p logs/fit-prior
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic-exp/_common.sh"

: "${DATASET:?Set DATASET=metrla or DATASET=pemsbay}"
SEED="${SEED:-0}"

scontrol update job="${SLURM_JOB_ID}" JobName="fit-${DATASET}" 2>/dev/null || true

srun --export=ALL "${DAIC_PYTHON}" scripts/fit_prior.py \
    --dataset "${DATASET}" \
    --seed "${SEED}"
