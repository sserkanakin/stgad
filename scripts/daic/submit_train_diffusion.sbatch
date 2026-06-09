#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --time=24:00:00
# partition / qos / gres are injected per-cluster by submit.sh.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/diffusion/%x-%j.out
#SBATCH --error=logs/diffusion/%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# train the diffusion denoiser on one dataset / seed.
#   sbatch --export=DATASET=metrla,SEED=0 scripts/daic/submit_train_diffusion.sbatch
#
# required: DATASET, SEED. optional: PRIOR (default kronecker, used for naming),
# EPOCHS, BATCH_SIZE, LR, D_H, C_0, ALPHA, EXTRA_FLAGS (passed verbatim).
# for a run-named log file also pass --job-name=diffusion_<dataset>_<prior>_seed<n>.

set -euo pipefail
mkdir -p logs/diffusion
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${DATASET:?Set DATASET=metrla|pemsbay|pems08f via --export}"
: "${SEED:?Set SEED=<int> via --export}"
export PRIOR="${PRIOR:-kronecker}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
D_H="${D_H:-32}"
C_0="${C_0:-20000}"
ALPHA="${ALPHA:-7.0}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

export STAGE="diffusion"
RUN_NAME="diffusion_${DATASET}_${PRIOR}_seed${SEED}"
scontrol update job="${SLURM_JOB_ID}" JobName="${RUN_NAME}" 2>/dev/null || true

log_run_header "${DAIC_PYTHON} scripts/train_diffusion.py --dataset ${DATASET} --prior ${PRIOR} --seed ${SEED} \
--epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --lr ${LR} --d-h ${D_H} --c-0 ${C_0} --alpha ${ALPHA} ${EXTRA_FLAGS}"

# shellcheck disable=SC2086
srun --export=ALL "${DAIC_PYTHON}" scripts/train_diffusion.py \
    --dataset "${DATASET}" \
    --prior "${PRIOR}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --d-h "${D_H}" \
    --c-0 "${C_0}" \
    --alpha "${ALPHA}" \
    ${EXTRA_FLAGS}
