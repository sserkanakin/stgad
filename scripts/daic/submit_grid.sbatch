#!/bin/bash
#SBATCH --job-name=grid_diffusion
# one array TASK = one (dataset, backbone, prior) combined train+eval. 16h covers the
# slowest cell (pemsbay train + full test eval). best.pt is checkpointed on every CRPS
# improvement, so a walltime kill still keeps the best model.
#SBATCH --time=16:00:00
# partition / qos / gres + --array + --output are injected by submit_array.sh.
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# reads the SLURM_ARRAY_TASK_ID-th row of GRID_FILE (idx<TAB>dataset<TAB>backbone<TAB>prior),
# then runs the combined train+eval for that cell. submit via scripts/daic/submit_array.sh.

set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${GRID_FILE:?Set GRID_FILE via --export (the dataset-major combo list)}"
: "${EXP:?Set EXP via --export}"
: "${SEED:?Set SEED via --export}"
EVAL_STEPS="${EVAL_STEPS:-100,50,25}"
EPOCHS="${EPOCHS:-200}"            # max epochs (early-stops on val-CRPS plateau, patience 15)

# keep wandb's local staging cache off the (quota'd) linuxhome too — park it under
# OUTPUT_ROOT (windowshome) next to the checkpoints/eval/logs.
if [ -n "${OUTPUT_ROOT:-}" ]; then
  export WANDB_DIR="${OUTPUT_ROOT}/wandb"
  export WANDB_CACHE_DIR="${OUTPUT_ROOT}/wandb/cache"
  mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"
fi

# pick this task's row (1-indexed line = task id + 1).
ROW="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${GRID_FILE}")"
read -r IDX DATASET BACKBONE PRIOR <<< "${ROW}"
: "${DATASET:?empty row ${SLURM_ARRAY_TASK_ID} in ${GRID_FILE}}"

export STAGE="grid"
RUN_NAME="grid_${DATASET}_${BACKBONE}_${PRIOR}_seed${SEED}"
scontrol update job="${SLURM_JOB_ID}" JobName="${RUN_NAME}" 2>/dev/null || true
echo "===== array task ${SLURM_ARRAY_TASK_ID}: ${DATASET} / ${BACKBONE} / ${PRIOR} (EXP=${EXP}) ====="

log_run_header "${DAIC_PYTHON} scripts/train_diffusion.py --dataset ${DATASET} --backbone ${BACKBONE} \
--prior ${PRIOR} --exp ${EXP} --seed ${SEED} --epochs ${EPOCHS} --eval-after --eval-steps ${EVAL_STEPS}"

# shellcheck disable=SC2086
srun --unbuffered --export=ALL "${DAIC_PYTHON}" -u scripts/train_diffusion.py \
    --dataset "${DATASET}" \
    --backbone "${BACKBONE}" \
    --prior "${PRIOR}" \
    --exp "${EXP}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --eval-after \
    --eval-steps "${EVAL_STEPS}"
