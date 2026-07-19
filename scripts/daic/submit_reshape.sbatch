#!/bin/bash
#SBATCH --job-name=reshape
# one array TASK = one (dataset, arm) diffusion run: train with per-epoch sampled ancestral
# CRPS selection (best_crps.pt). NO --final-val -> no EM step-sweep (the reshaped/empirical
# priors are stiff; EM is inappropriate, ancestral is exact). the comparison metric is the
# best per-epoch sampled CRPS on --val-windows clean windows.
#SBATCH --time=24:00:00
# partition / qos / gres + --array + --output are injected by submit_reshape.sh.
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# reads the SLURM_ARRAY_TASK_ID-th row of GRID_FILE:
#   idx  dataset  name  prior  temporal  sampler  cond  cap  run_dir
# submit via scripts/daic/submit_reshape.sh (never sbatch this file directly).
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${GRID_FILE:?Set GRID_FILE via --export}"
: "${EXP:?Set EXP via --export}"
: "${SEED:?Set SEED via --export}"
# BACKBONE is read per-row from the grid (below), not from the env.
EPOCHS="${EPOCHS:-200}"
LR="${LR:-1e-4}"
VAL_WINDOWS="${VAL_WINDOWS:-512}"

if [ -n "${OUTPUT_ROOT:-}" ]; then
  export WANDB_DIR="${OUTPUT_ROOT}/wandb"; export WANDB_CACHE_DIR="${OUTPUT_ROOT}/wandb/cache"
  mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"
fi

ROW="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${GRID_FILE}")"
read -r IDX DATASET BACKBONE NAME PRIOR TEMPORAL CAP LOSS SCHED RUN_DIR <<< "${ROW}"
: "${DATASET:?empty row ${SLURM_ARRAY_TASK_ID} in ${GRID_FILE}}"

# small GCNN denoiser, per-node norm, residual-history conditioning (validated locally).
# sampler: loguniform for our FCPS arms; powerlaw for the vanilla DDPM baseline (its VP
# schedule has no log-uniform floor, and a textbook DDPM uses standard sampling). schedule
# + loss are per-arm; ancestral eval; NO final-val (EM sweep skipped for the stiff priors).
SAMPLER=loguniform; [ "${PRIOR}" = "vp" ] && SAMPLER=powerlaw
FLAGS=(--denoiser gcnn --norm pernode --noise-sampler "${SAMPLER}" --cond-past
       --loss "${LOSS}" --schedule "${SCHED}"
       --val-every 0 --val-windows "${VAL_WINDOWS}" --fast-val-steps 50 --val-anc-steps 50)
[ "${TEMPORAL}" != "-" ] && FLAGS+=(--temporal "${TEMPORAL}")
[ "${CAP}" != "-" ]      && FLAGS+=(--stiffness-cap "${CAP}")

ARM_EXP="${EXP}/${NAME}"
export STAGE="diffusion"
case "${DATASET}" in
  metrla) DSS=mt;; pemsbay) DSS=bay;; pems04f) DSS=04f;; pems04s) DSS=04s;;
  pems08f) DSS=08f;; pems08s) DSS=08s;; *) DSS="${DATASET}";; esac
RUN_NAME="${DSS}-${BACKBONE}-${NAME}-s${SEED}-${EXP}"
scontrol update job="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="${RUN_NAME}" 2>/dev/null || true
echo "===== task ${SLURM_ARRAY_TASK_ID}: ${DATASET}/${BACKBONE} / ${NAME} (prior=${PRIOR} temporal=${TEMPORAL} sampler=${SAMPLER} loss=${LOSS} cond=past) ====="

LOG_SRC="${OUTPUT_ROOT}/logs/${EXP}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
copy_logs() {
  mkdir -p "${RUN_DIR}/train" 2>/dev/null || true
  cp -f "${LOG_SRC}.out" "${RUN_DIR}/train/slurm_${SLURM_JOB_ID}.out" 2>/dev/null || true
  cp -f "${LOG_SRC}.err" "${RUN_DIR}/train/slurm_${SLURM_JOB_ID}.err" 2>/dev/null || true
}
trap copy_logs EXIT

log_run_header "${DAIC_PYTHON} scripts/train_diffusion.py --dataset ${DATASET} --prior ${PRIOR} \
--backbone ${BACKBONE} --exp ${ARM_EXP} --seed ${SEED} --epochs ${EPOCHS} --lr ${LR} ${FLAGS[*]}"

srun --unbuffered --export=ALL "${DAIC_PYTHON}" -u scripts/train_diffusion.py \
    --dataset "${DATASET}" --prior "${PRIOR}" --backbone "${BACKBONE}" \
    --exp "${ARM_EXP}" --seed "${SEED}" --epochs "${EPOCHS}" --lr "${LR}" --loss snr \
    "${FLAGS[@]}"
