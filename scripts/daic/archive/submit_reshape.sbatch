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
: "${SEEDS:?Set SEEDS via --export (underscore-separated, e.g. 42_126_2022)}"
# BACKBONE is read per-row from the grid (below), not from the env.
EPOCHS="${EPOCHS:-200}"
LR="${LR:-1e-4}"
VAL_WINDOWS="${VAL_WINDOWS:-512}"
# patience 15 (the default) truncated the whitened arms unfairly: on pems08f/mlp emp_w hit a
# spurious sampled-CRPS minimum at epoch 3 and early-stopped at 18 while its val_loss was
# still improving monotonically — then got compared against iso, which ran the full 200.
# that is a comparison of early-stopping luck, not of losses. 40 gives every arm a fair run.
PATIENCE="${PATIENCE:-40}"
# stage-1 seed (which residuals to train on), pinned by the submitter; stage-2 seeds are SEEDS.
RESIDUAL_SEED="${RESIDUAL_SEED:-42}"
# ALL stage-2 seeds run SEQUENTIALLY inside one task. previously each seed was its own array
# submission, so a 5-seed grid meant 5x18 = 90 separate ~25-minute jobs. that is a lot of
# scheduler churn for a shared cluster and looks like slot-farming. one task now does one
# (backbone, arm) across every seed: ~2h of real work per job, 5x fewer jobs.
# sbatch --export cannot carry spaces in a value, so the submitter sends "42_126_2022";
# turn it back into a space-separated list here.
SEEDS="${SEEDS:-42}"; SEEDS="${SEEDS//_/ }"
# conditioning: fo = forecast only (C = Yhat, (B,N,H)); past = + previous window's residual
# (C = [R_past | Yhat], (B,N,2H)). gcnn takes both since it splits them into channels.
COND="${COND:-fo}"
# denoiser: gcnn = GAD low-capacity graph-filter cascade (~17k); ugnet = expressive (~1M).
# the capacity axis of the mechanism test — does a bigger denoiser learn what the prior
# supplies, shrinking its advantage?
DENOISER="${DENOISER:-gcnn}"

if [ -n "${OUTPUT_ROOT:-}" ]; then
  export WANDB_DIR="${OUTPUT_ROOT}/wandb"; export WANDB_CACHE_DIR="${OUTPUT_ROOT}/wandb/cache"
  mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"
fi

ROW="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${GRID_FILE}")"
read -r IDX DATASET BACKBONE NAME PRIOR TEMPORAL CAP LOSS SCHED RUN_DIR <<< "${ROW}"
: "${DATASET:?empty row ${SLURM_ARRAY_TASK_ID} in ${GRID_FILE}}"

# small GCNN denoiser, per-node norm. sampler: loguniform for FCPS arms; powerlaw for the
# vanilla DDPM baseline (its VP schedule has no log-uniform floor). schedule + loss per-arm;
# ancestral eval; NO final-val.
SAMPLER=loguniform; [ "${PRIOR}" = "vp" ] && SAMPLER=powerlaw
FLAGS=(--denoiser "${DENOISER}" --norm pernode --noise-sampler "${SAMPLER}"
       --loss "${LOSS}" --schedule "${SCHED}" --patience "${PATIENCE}"
       --residual-seed "${RESIDUAL_SEED}"
       --val-every 0 --val-windows "${VAL_WINDOWS}" --fast-val-steps 50)
[ "${TEMPORAL}" != "-" ] && FLAGS+=(--temporal "${TEMPORAL}")
[ "${CAP}" != "-" ]      && FLAGS+=(--stiffness-cap "${CAP}")
[ "${COND}" = "past" ]   && FLAGS+=(--cond-past)

# the run dir is keyed on (dataset, prior, seed, exp, backbone) — NOT on conditioning. so a
# cond=past run would silently overwrite the cond=fo results for the same arm. namespace the
# arm by conditioning whenever it is not the default.
ARM_EXP="${EXP}/${NAME}"
[ "${COND}" != "fo" ]        && ARM_EXP="${ARM_EXP}-${COND}"
# same collision hazard as COND: the run dir has no denoiser component, so a ugnet run
# would silently overwrite the gcnn results for the same arm.
[ "${DENOISER}" != "gcnn" ]  && ARM_EXP="${ARM_EXP}-${DENOISER}"
export STAGE="diffusion"
case "${DATASET}" in
  metrla) DSS=mt;; pemsbay) DSS=bay;; pems04f) DSS=04f;; pems04s) DSS=04s;;
  pems08f) DSS=08f;; pems08s) DSS=08s;; *) DSS="${DATASET}";; esac
RUN_NAME="${DSS}-${BACKBONE}-${NAME}-${COND}-${DENOISER}-${EXP}"
scontrol update job="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="${RUN_NAME}" 2>/dev/null || true
echo "===== task ${SLURM_ARRAY_TASK_ID}: ${DATASET}/${BACKBONE} / ${NAME} (prior=${PRIOR} temporal=${TEMPORAL} sampler=${SAMPLER} loss=${LOSS} cond=${COND} denoiser=${DENOISER}) ====="
echo "===== seeds (sequential in this one job): ${SEEDS} ====="

LOG_SRC="${OUTPUT_ROOT}/logs/${EXP}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
copy_logs() {
  mkdir -p "${RUN_DIR}/train" 2>/dev/null || true
  cp -f "${LOG_SRC}.out" "${RUN_DIR}/train/slurm_${SLURM_JOB_ID}.out" 2>/dev/null || true
  cp -f "${LOG_SRC}.err" "${RUN_DIR}/train/slurm_${SLURM_JOB_ID}.err" 2>/dev/null || true
}
trap copy_logs EXIT

SEED="${SEEDS}"   # log_run_header prints "seed:"; show the whole list, not just the first
log_run_header "${DAIC_PYTHON} scripts/train_diffusion.py --dataset ${DATASET} --prior ${PRIOR} \
--backbone ${BACKBONE} --exp ${ARM_EXP} --seeds '${SEEDS}' --epochs ${EPOCHS} --lr ${LR} ${FLAGS[*]}"

# NB: the loss is per-arm and lives in FLAGS (--loss "${LOSS}"). do NOT add a second
# --loss here: argparse takes the last one, so a stray literal would silently decide the
# whitened-vs-snr arm depending on flag order.
#
# one seed failing must not abandon the rest of the task's work, so drop `set -e` around the
# loop and record failures instead. the job still exits non-zero if any seed failed, so a
# partial task is visible in sacct rather than silently reported as COMPLETED.
FAILED=""
for S in ${SEEDS}; do
  echo "───── seed ${S}  (${DATASET}/${BACKBONE}/${NAME}/${COND}) ─────"
  set +e
  srun --unbuffered --export=ALL "${DAIC_PYTHON}" -u scripts/train_diffusion.py \
      --dataset "${DATASET}" --prior "${PRIOR}" --backbone "${BACKBONE}" \
      --exp "${ARM_EXP}" --seed "${S}" --epochs "${EPOCHS}" --lr "${LR}" \
      "${FLAGS[@]}"
  rc=$?
  set -e
  [ ${rc} -ne 0 ] && { echo "  !! seed ${S} FAILED rc=${rc}"; FAILED="${FAILED} ${S}"; }
done
[ -n "${FAILED}" ] && { echo "===== task ${SLURM_ARRAY_TASK_ID} failed seeds:${FAILED} ====="; exit 1; }
echo "===== task ${SLURM_ARRAY_TASK_ID} complete: all seeds (${SEEDS}) ====="
