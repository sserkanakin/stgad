#!/bin/bash
#SBATCH --job-name=ladder-eval
# one array TASK = one (dataset, backbone, arm), evaluated across ALL seeds SEQUENTIALLY.
# batching seeds keeps the task count under the account's MaxSubmit=100 limit (per-cell
# tasks would be 360). eval-only: reuses the val-CRPS-selected best_crps.pt (no retraining,
# no test leakage). ancestral sampler, k=100, y==0 masked (DiffSTG/SpecSTG). writes per cell
# MAE/RMSE/normalized-CRPS + calibration + W1/energy under
# OUTPUT_ROOT/<exp>/<arm>/<ds>/<bb>/<prior>/seed<S>/eval/ (paths.eval_dir).
#SBATCH --time=10:00:00
# partition / qos / gres + --array + --output are injected by submit_ladder_eval_array.sh.
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# reads the SLURM_ARRAY_TASK_ID-th row of GRID_FILE:  idx  dataset  backbone  name  prior
# seeds come from SEEDS (underscore-joined via --export). submit via
# scripts/daic/submit_ladder_eval_array.sh (never sbatch this file directly).
set -uo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${GRID_FILE:?Set GRID_FILE via --export}"
: "${EXP:?Set EXP via --export}"
: "${SEEDS:?Set SEEDS via --export (underscore-joined, e.g. 42_1_126_2022_3407)}"
export WANDB_MODE="${WANDB_MODE:-disabled}"    # eval writes json/npz to disk; no dashboard
SEEDS="${SEEDS//_/ }"                           # sbatch --export forbids spaces; decode here
SPLIT="${SPLIT:-test}"
SAMPLER="${SAMPLER:-ancestral}"                # closed-form OU posterior (exact for the linear SDE)
# ancestral reverse steps (run_plan phase 2a). accepts a LIST (underscore-joined via
# --export, e.g. 5_10_25_50_100) for the few-step/NFE curve — each step count is a separate
# eval; the metrics filename embeds sampler+steps so they never clobber.
NSTEPS="${NSTEPS:-50}"; NSTEPS="${NSTEPS//_/ }"
K="${K:-100}"                                   # posterior samples/window (CSDI/DiffSTG/SpecSTG)
CKPT="${CKPT:-best_crps.pt}"                    # val-CRPS-selected weights -> no test leakage

ROW="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${GRID_FILE}")"
read -r IDX DATASET BACKBONE NAME PRIOR <<< "${ROW}"
: "${DATASET:?empty row ${SLURM_ARRAY_TASK_ID} in ${GRID_FILE}}"

# bigger-N datasets get a smaller gen-batch so the k=100 generation batch stays under gpu mem.
case "${DATASET}" in pemsbay) GB=2048;; metrla) GB=3072;; *) GB=4096;; esac
# the run tree is namespaced by arm name (iso, emp-past, ...); --exp carries that so eval_dir
# resolves to the exact trained cell. the diffusion seed names the ckpt/eval dir + rng; the
# stage-1 residual seed is read from each run's config.json inside eval_diffusion.py.
ARM_EXP="${EXP}/${NAME}"
OUT="${OUTPUT_ROOT:-${PROJECT_DIR}/results}"

case "${DATASET}" in
  metrla) DSS=mt;; pemsbay) DSS=bay;; pems04f) DSS=04f;; pems04s) DSS=04s;;
  pems08f) DSS=08f;; pems08s) DSS=08s;; *) DSS="${DATASET}";; esac
RUN_NAME="ev-${DSS}-${BACKBONE}-${NAME}"
scontrol update job="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="${RUN_NAME}" 2>/dev/null || true
echo "===== task ${SLURM_ARRAY_TASK_ID}: eval ${DATASET}/${BACKBONE}/${NAME} "\
"(prior=${PRIOR} ${SAMPLER}@${NSTEPS} k=${K} ckpt=${CKPT}) seeds: ${SEEDS} ====="

# mirror the slurm .out/.err into the arm's tree at exit (seed-agnostic location).
LOG_SRC="${OUT}/logs/${EXP}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
copy_logs() {
  local d="${OUT}/${ARM_EXP}/${DATASET}/${BACKBONE}/${PRIOR}"
  mkdir -p "${d}" 2>/dev/null || true
  cp -f "${LOG_SRC}.out" "${d}/eval_slurm_${SLURM_JOB_ID}.out" 2>/dev/null || true
  cp -f "${LOG_SRC}.err" "${d}/eval_slurm_${SLURM_JOB_ID}.err" 2>/dev/null || true
}
trap copy_logs EXIT

# one seed failing (or a missing ckpt) must not abandon the task's other seeds. record and
# exit non-zero at the end so sacct shows a partial task rather than a false COMPLETED.
FAILED=""
for S in ${SEEDS}; do
  ckpt="${OUT}/${ARM_EXP}/${DATASET}/${BACKBONE}/${PRIOR}/seed${S}/train/${CKPT}"
  if [ ! -f "${ckpt}" ]; then
    echo "  !! skip seed ${S}: no ${CKPT} at ${ckpt}"; FAILED="${FAILED} ${S}(no-ckpt)"; continue
  fi
  # AUTOSPREAD (comma grid, e.g. "1.0,1.2,1.4,1.6,1.8"): fit the ensemble-spread scalar on
  # VAL, then evaluate SPLIT at the best s (recalibrated protocol; uniform across arms).
  EXTRA=()
  [ -n "${AUTOSPREAD:-}" ] && EXTRA+=(--auto-spread "${AUTOSPREAD}")
  for NS in ${NSTEPS}; do
    echo "───── seed ${S}  (${DATASET}/${BACKBONE}/${NAME})  ${SAMPLER}@${NS} ─────"
    srun --unbuffered --export=ALL "${DAIC_PYTHON}" -u scripts/eval_diffusion.py \
        --dataset "${DATASET}" --prior "${PRIOR}" --backbone "${BACKBONE}" \
        --exp "${ARM_EXP}" --seed "${S}" --split "${SPLIT}" \
        --sampler "${SAMPLER}" --n-steps "${NS}" --k "${K}" --ckpt "${CKPT}" \
        --w1-samples 4 --save-diagnostics --gen-batch "${GB}" "${EXTRA[@]}"
    rc=$?
    [ ${rc} -ne 0 ] && { echo "  !! seed ${S} steps ${NS} FAILED rc=${rc}"; FAILED="${FAILED} ${S}@${NS}"; }
  done
done
[ -n "${FAILED}" ] && { echo "===== task ${SLURM_ARRAY_TASK_ID} failed:${FAILED} ====="; exit 1; }
echo "===== task ${SLURM_ARRAY_TASK_ID} complete: all seeds (${SEEDS}) ====="
