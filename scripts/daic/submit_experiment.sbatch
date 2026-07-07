#!/bin/bash
#SBATCH --job-name=experiment
# one array TASK = one (dataset, prior) full lifecycle: train (val-loss selection,
# EM step-sweep diagnostic every 10 epochs) → auto-pick reverse steps on best.pt →
# FULL val-set eval at that count. EM sampling is the dominant cost (~0.9ms/traj-step
# fp32 at N=207, measured); 3 days is a generous ceiling so even pemsbay finishes
# with margin. best.pt is checkpointed every improvement, so a walltime kill keeps
# the model (re-run only the final-val part).
#SBATCH --time=72:00:00
# partition / qos / gres + --array + --output are injected by submit_experiment.sh.
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# reads the SLURM_ARRAY_TASK_ID-th row of GRID_FILE (idx<TAB>dataset<TAB>prior<TAB>run_dir).
# submit via scripts/daic/submit_experiment.sh (never sbatch this file directly).

set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${GRID_FILE:?Set GRID_FILE via --export}"
: "${EXP:?Set EXP via --export}"
: "${SEED:?Set SEED via --export}"
BACKBONE="${BACKBONE:-gwn}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-1e-3}"
LOSS="${LOSS:-mse}"

# wandb's local staging cache off the (quota'd) linuxhome — park it on windowshome.
if [ -n "${OUTPUT_ROOT:-}" ]; then
  export WANDB_DIR="${OUTPUT_ROOT}/wandb"
  export WANDB_CACHE_DIR="${OUTPUT_ROOT}/wandb/cache"
  mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"
fi

# pick this task's row (1-indexed line = task id + 1).
ROW="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${GRID_FILE}")"
read -r IDX DATASET PRIOR RUN_DIR <<< "${ROW}"
: "${DATASET:?empty row ${SLURM_ARRAY_TASK_ID} in ${GRID_FILE}}"

# short prior for readable squeue names: kronecker→kron, isotropic→iso.
PRIOR_SHORT="$([ "${PRIOR}" = "kronecker" ] && echo kron || echo iso)"
export STAGE="diffusion"
RUN_NAME="${EXP}_${DATASET}_${PRIOR_SHORT}_s${SEED}"
# rename THIS array element (jobid_taskid). renaming by SLURM_JOB_ID is wrong here:
# one task's raw id equals the array master id, and renaming that renames ALL tasks.
scontrol update job="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
  JobName="${RUN_NAME}" 2>/dev/null || true
echo "===== array task ${SLURM_ARRAY_TASK_ID}: ${DATASET} / ${PRIOR} (EXP=${EXP}) ====="
echo "run dir: ${RUN_DIR}"

# self-contained run folder: copy this task's slurm log next to train.log at exit
# (also on failure/timeout), keyed by the real job id for traceability.
LOG_SRC="${OUTPUT_ROOT}/logs/${EXP}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
copy_logs() {
  mkdir -p "${RUN_DIR}/train" 2>/dev/null || true
  cp -f "${LOG_SRC}.out" "${RUN_DIR}/train/slurm_${SLURM_JOB_ID}.out" 2>/dev/null || true
  cp -f "${LOG_SRC}.err" "${RUN_DIR}/train/slurm_${SLURM_JOB_ID}.err" 2>/dev/null || true
}
trap copy_logs EXIT

log_run_header "${DAIC_PYTHON} scripts/train_diffusion.py --dataset ${DATASET} --prior ${PRIOR} \
--backbone ${BACKBONE} --exp ${EXP} --seed ${SEED} --epochs ${EPOCHS} --lr ${LR} --loss ${LOSS} --final-val"

srun --unbuffered --export=ALL "${DAIC_PYTHON}" -u scripts/train_diffusion.py \
    --dataset "${DATASET}" \
    --prior "${PRIOR}" \
    --backbone "${BACKBONE}" \
    --exp "${EXP}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --loss "${LOSS}" \
    --final-val
