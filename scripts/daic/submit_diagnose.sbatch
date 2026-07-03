#!/bin/bash
#SBATCH --job-name=diagnose_samples
# sample-correctness diagnostic (Wasserstein / KL / covariance). the reverse check
# generates K samples per window with the EM sampler, so it needs a GPU; 40min covers
# k=100 × 256 windows × ~1000 steps on an A40.
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/diagnose/%x-%j.out
#SBATCH --error=logs/diagnose/%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# reads a trained best.pt from the array's OUTPUT_ROOT (windowshome/research-tud-out)
# and the residuals from the data share. pass the diagnostic flags via DIAG_ARGS.
#   DIAG_ARGS="--check reverse --dataset metrla --prior kronecker --backbone gwn --exp dev --seed 42" \
#     scripts/daic/submit.sh diagnose metrla 42
set -euo pipefail
mkdir -p logs/diagnose
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export STAGE="diagnose"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

# same roots as submit_array.sh so best.pt (OUTPUT_ROOT) + residuals (RESULTS_ROOT) resolve.
export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"

DIAG_ARGS="${DIAG_ARGS:---check forward --dataset metrla}"
echo "inputs (residuals): ${RESULTS_ROOT}"
echo "OUTPUTS (best.pt):  ${OUTPUT_ROOT}"
log_run_header "${DAIC_PYTHON} scripts/diagnose_samples.py ${DIAG_ARGS}"

# shellcheck disable=SC2086
srun --unbuffered --export=ALL "${DAIC_PYTHON}" -u scripts/diagnose_samples.py ${DIAG_ARGS}
