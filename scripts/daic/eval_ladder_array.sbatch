#!/bin/bash
#SBATCH --job-name=ladder-eval
# one array TASK = one test-split eval cell (dataset, backbone, arm, seed). eval-only:
# reuses the val-CRPS-selected best_crps.pt (no retraining, no test leakage). ancestral
# sampler, k=100 posterior samples/window, y==0 masked (DiffSTG/SpecSTG). writes, per cell,
# MAE/RMSE/normalized-CRPS + calibration (Winkler/coverage/PIT/reliability) + W1/energy under
# OUTPUT_ROOT/<exp>/<arm>/<ds>/<bb>/<prior>/seed<S>/eval/ (paths.eval_dir).
#SBATCH --time=6:00:00
# partition / qos / gres + --array + --output are injected by submit_ladder_eval_array.sh.
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# reads the SLURM_ARRAY_TASK_ID-th row of GRID_FILE:  idx  dataset  backbone  name  prior  seed
# submit via scripts/daic/submit_ladder_eval_array.sh (never sbatch this file directly).
set -uo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${GRID_FILE:?Set GRID_FILE via --export}"
: "${EXP:?Set EXP via --export}"
export WANDB_MODE="${WANDB_MODE:-disabled}"    # eval writes json/npz to disk; no dashboard
# eval knobs (headline defaults; overridable from the submitter's --export).
SPLIT="${SPLIT:-test}"
SAMPLER="${SAMPLER:-ancestral}"                # closed-form OU posterior (exact for the linear SDE)
NSTEPS="${NSTEPS:-50}"                          # ancestral reverse steps (run_plan phase 2a)
K="${K:-100}"                                   # posterior samples/window (CSDI/DiffSTG/SpecSTG)
CKPT="${CKPT:-best_crps.pt}"                    # val-CRPS-selected weights -> no test leakage

ROW="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${GRID_FILE}")"
read -r IDX DATASET BACKBONE NAME PRIOR SEED <<< "${ROW}"
: "${DATASET:?empty row ${SLURM_ARRAY_TASK_ID} in ${GRID_FILE}}"

# bigger-N datasets get a smaller gen-batch so the k=100 generation batch stays under gpu mem.
case "${DATASET}" in pemsbay) GB=2048;; metrla) GB=3072;; *) GB=4096;; esac

# the run tree is namespaced by arm name (iso, emp-past, ...); --exp carries that so eval_dir
# resolves to the exact trained cell. the diffusion seed names the ckpt/eval dir + rng; the
# stage-1 residual seed is read from the run's config.json inside eval_diffusion.py.
ARM_EXP="${EXP}/${NAME}"
RUN_EVAL_DIR="${OUTPUT_ROOT:-${PROJECT_DIR}/results}/${ARM_EXP}/${DATASET}/${BACKBONE}/${PRIOR}/seed${SEED}/eval"

case "${DATASET}" in
  metrla) DSS=mt;; pemsbay) DSS=bay;; pems04f) DSS=04f;; pems04s) DSS=04s;;
  pems08f) DSS=08f;; pems08s) DSS=08s;; *) DSS="${DATASET}";; esac
RUN_NAME="ev-${DSS}-${BACKBONE}-${NAME}-s${SEED}"
scontrol update job="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="${RUN_NAME}" 2>/dev/null || true
echo "===== task ${SLURM_ARRAY_TASK_ID}: eval ${DATASET}/${BACKBONE}/${NAME} seed${SEED} "\
"(prior=${PRIOR} ${SAMPLER}@${NSTEPS} k=${K} ckpt=${CKPT}) ====="

# mirror the slurm .out/.err into the cell's eval dir at exit, next to the metrics json.
LOG_SRC="${OUTPUT_ROOT}/logs/${EXP}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
copy_logs() {
  mkdir -p "${RUN_EVAL_DIR}" 2>/dev/null || true
  cp -f "${LOG_SRC}.out" "${RUN_EVAL_DIR}/slurm_${SLURM_JOB_ID}.out" 2>/dev/null || true
  cp -f "${LOG_SRC}.err" "${RUN_EVAL_DIR}/slurm_${SLURM_JOB_ID}.err" 2>/dev/null || true
}
trap copy_logs EXIT

log_run_header "${DAIC_PYTHON} scripts/eval_diffusion.py --dataset ${DATASET} --prior ${PRIOR} \
--backbone ${BACKBONE} --exp ${ARM_EXP} --seed ${SEED} --split ${SPLIT} --sampler ${SAMPLER} \
--n-steps ${NSTEPS} --k ${K} --ckpt ${CKPT}"

srun --unbuffered --export=ALL "${DAIC_PYTHON}" -u scripts/eval_diffusion.py \
    --dataset "${DATASET}" --prior "${PRIOR}" --backbone "${BACKBONE}" \
    --exp "${ARM_EXP}" --seed "${SEED}" --split "${SPLIT}" \
    --sampler "${SAMPLER}" --n-steps "${NSTEPS}" --k "${K}" --ckpt "${CKPT}" \
    --w1-samples 4 --save-diagnostics --gen-batch "${GB}"
rc=$?
[ ${rc} -ne 0 ] && echo "===== task ${SLURM_ARRAY_TASK_ID} FAILED rc=${rc} (${DATASET}/${BACKBONE}/${NAME}/seed${SEED}) ====="
exit ${rc}
