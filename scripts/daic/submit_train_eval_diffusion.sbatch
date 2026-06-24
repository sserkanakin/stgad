#!/bin/bash
#SBATCH --job-name=traineval_diffusion
# one combined job: train the denoiser (best.pt selected by val-CRPS), then run the
# locked test eval in-process and write test_results.json — the Vimal-style lifecycle.
# 14h covers train (~1-3h, early-stops) + the test eval (pemsbay is the slow one).
#SBATCH --time=14:00:00
# partition / qos / gres injected per-cluster by submit.sh.
#SBATCH --cpus-per-task=4
# 64G host ram for the eval gen-batch staged per reverse_sample (matches submit_eval).
#SBATCH --mem=64G
#SBATCH --output=logs/traineval/%x-%j.out
#SBATCH --error=logs/traineval/%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# required: DATASET, SEED. optional: PRIOR (default kronecker), EXP, EPOCHS,
# BATCH_SIZE, LR, EVAL_STEPS, EXTRA_FLAGS (verbatim).
#   sbatch --export=DATASET=metrla,SEED=42,PRIOR=kronecker scripts/daic/submit_train_eval_diffusion.sbatch

set -euo pipefail
mkdir -p logs/traineval
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${DATASET:?Set DATASET via --export}"
: "${SEED:?Set SEED via --export}"
export PRIOR="${PRIOR:-kronecker}"
BACKBONE="${BACKBONE:-gwn}"
EXP="${EXP:-dev}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
EVAL_STEPS="${EVAL_STEPS:-50}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

export STAGE="traineval"
RUN_NAME="traineval_${BACKBONE}_${DATASET}_${PRIOR}_seed${SEED}"
scontrol update job="${SLURM_JOB_ID}" JobName="${RUN_NAME}" 2>/dev/null || true

log_run_header "${DAIC_PYTHON} scripts/train_diffusion.py --dataset ${DATASET} --prior ${PRIOR} \
--backbone ${BACKBONE} --exp ${EXP} --seed ${SEED} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} \
--lr ${LR} --eval-after --eval-steps ${EVAL_STEPS} ${EXTRA_FLAGS}"

# shellcheck disable=SC2086
srun --unbuffered --export=ALL "${DAIC_PYTHON}" -u scripts/train_diffusion.py \
    --dataset "${DATASET}" \
    --prior "${PRIOR}" \
    --backbone "${BACKBONE}" \
    --exp "${EXP}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --eval-after \
    --eval-steps "${EVAL_STEPS}" \
    ${EXTRA_FLAGS}
