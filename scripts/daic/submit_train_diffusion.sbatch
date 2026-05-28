#!/bin/bash
#SBATCH --job-name=diff-train
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/train-diffusion/%x-%j.out
#SBATCH --error=logs/train-diffusion/%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# Train the diffusion denoiser (R0Denoiser) on one dataset / seed. Submit with:
#   sbatch --job-name="diff-metrla-s0" \
#          --export=DATASET=metrla,SEED=0 \
#          scripts/daic/submit_train_diffusion.sbatch
#
# Required env vars: DATASET (metrla|pemsbay|pems08f), SEED (int).
# Optional:           EPOCHS (default 200), BATCH_SIZE (default 32), LR, D_H,
#                     C_0, ALPHA, EXTRA_FLAGS (passed verbatim to the python script).

set -euo pipefail
mkdir -p logs/train-diffusion
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${DATASET:?Set DATASET=metrla|pemsbay|pems08f via --export}"
: "${SEED:?Set SEED=<int> via --export}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
D_H="${D_H:-32}"
C_0="${C_0:-20000}"
ALPHA="${ALPHA:-7.0}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# Rename for readability in squeue.
scontrol update job="${SLURM_JOB_ID}" JobName="diff-${DATASET}-s${SEED}" 2>/dev/null || true

echo "Training diffusion denoiser on ${DATASET} (seed ${SEED}, ${EPOCHS} epochs)"
echo "  d_h=${D_H}  batch=${BATCH_SIZE}  lr=${LR}  c_0=${C_0}  alpha=${ALPHA}"
echo "  extra flags: ${EXTRA_FLAGS}"

# shellcheck disable=SC2086
srun --export=ALL "${DAIC_PYTHON}" scripts/train_diffusion.py \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --d-h "${D_H}" \
    --c-0 "${C_0}" \
    --alpha "${ALPHA}" \
    ${EXTRA_FLAGS}
