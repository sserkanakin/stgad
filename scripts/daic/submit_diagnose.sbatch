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
# and the residuals from the data share. args are built from clean single-value env vars
# (CHECK/PRIOR/BACKBONE/DIAG_*), never a space-containing DIAG_ARGS blob (slurm --export
# mangles those). submit via:
#   CHECK=reverse PRIOR=kronecker BACKBONE=gwn scripts/daic/submit.sh diagnose metrla 42
set -euo pipefail
mkdir -p logs/diagnose
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export STAGE="diagnose"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${DATASET:?Set DATASET via --export}"
: "${SEED:?Set SEED via --export}"
CHECK="${CHECK:-reverse}"
export PRIOR="${PRIOR:-kronecker}"
BACKBONE="${BACKBONE:-gwn}"
EXP="${EXP:-dev}"
DIAG_K="${DIAG_K:-50}"
DIAG_WINDOWS="${DIAG_WINDOWS:-128}"
DIAG_STEPS="${DIAG_STEPS:-1000}"
DIAG_GENBATCH="${DIAG_GENBATCH:-256}"

# same roots as submit_array.sh so best.pt (OUTPUT_ROOT) + residuals (RESULTS_ROOT) resolve.
export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"

RUN_NAME="diagnose_${CHECK}_${DATASET}_${PRIOR}_${BACKBONE}_seed${SEED}"
scontrol update job="${SLURM_JOB_ID}" JobName="${RUN_NAME}" 2>/dev/null || true
echo "inputs (residuals): ${RESULTS_ROOT}"
echo "OUTPUTS (best.pt):  ${OUTPUT_ROOT}"
log_run_header "${DAIC_PYTHON} scripts/diagnose_samples.py --check ${CHECK} --dataset ${DATASET} \
--prior ${PRIOR} --backbone ${BACKBONE} --exp ${EXP} --seed ${SEED} --k ${DIAG_K} \
--windows ${DIAG_WINDOWS} --n-steps ${DIAG_STEPS} --gen-batch ${DIAG_GENBATCH}"

srun --unbuffered --export=ALL "${DAIC_PYTHON}" -u scripts/diagnose_samples.py \
    --check "${CHECK}" --dataset "${DATASET}" --prior "${PRIOR}" --backbone "${BACKBONE}" \
    --exp "${EXP}" --seed "${SEED}" --k "${DIAG_K}" --windows "${DIAG_WINDOWS}" \
    --n-steps "${DIAG_STEPS}" --gen-batch "${DIAG_GENBATCH}"
