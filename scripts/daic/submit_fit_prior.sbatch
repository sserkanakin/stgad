#!/bin/bash
#SBATCH --job-name=fit-prior
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/fit-prior/%x-%j.out
#SBATCH --error=logs/fit-prior/%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# CPU-only: fit the Kronecker prior (NumPy + LAPACK; no GPU needed). Submit:
#   sbatch --export=DATASET=metrla scripts/daic/submit_fit_prior.sbatch

set -euo pipefail
mkdir -p logs/fit-prior
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${DATASET:?Set DATASET=metrla or DATASET=pemsbay}"
SEED="${SEED:-0}"

scontrol update job="${SLURM_JOB_ID}" JobName="fit-${DATASET}" 2>/dev/null || true

srun --export=ALL "${DAIC_PYTHON}" scripts/fit_prior.py \
    --dataset "${DATASET}" \
    --seed "${SEED}"
