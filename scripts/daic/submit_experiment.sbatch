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
LR="${LR:-1e-4}"
LOSS="${LOSS:-snr}"
# final-val step sweep. empty = train_diffusion's full default sweep (EM
# 1000..3500 x anc 25/50/100/200, per-checkpoint) as used in round 2. set a SINGLE
# value each for a light final-val (no sweep) — e.g. FINAL_STEPS_LIST=1000
# FINAL_ANC_STEPS=25 for the iso_base control (isotropic prior is less stiff, so it
# needs no per-cell EM tuning). the chosen count still lands in val_results.json,
# which the test-eval launcher reads.
FINAL_STEPS_LIST="${FINAL_STEPS_LIST:-}"
FINAL_ANC_STEPS="${FINAL_ANC_STEPS:-}"
FINAL_FLAGS=()
[ -n "${FINAL_STEPS_LIST}" ] && FINAL_FLAGS+=(--final-steps-list "${FINAL_STEPS_LIST}")
[ -n "${FINAL_ANC_STEPS}" ]  && FINAL_FLAGS+=(--final-anc-steps "${FINAL_ANC_STEPS}")

# wandb's local staging cache off the (quota'd) linuxhome — park it on windowshome.
if [ -n "${OUTPUT_ROOT:-}" ]; then
  export WANDB_DIR="${OUTPUT_ROOT}/wandb"
  export WANDB_CACHE_DIR="${OUTPUT_ROOT}/wandb/cache"
  mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"
fi

# pick this task's row (1-indexed line = task id + 1).
ROW="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${GRID_FILE}")"
read -r IDX DATASET ARM PRIOR COND NORM RUN_DIR <<< "${ROW}"
: "${DATASET:?empty row ${SLURM_ARRAY_TASK_ID} in ${GRID_FILE}}"

# per-arm flags + nested exp tag (each arm gets its own tree/wandb namespace).
ARM_FLAGS=(--norm "${NORM}")
[ "${COND}" = "past" ] && ARM_FLAGS+=(--cond-past)
# GAD-style small γ for the raw-signal arm (Σ_S=(L+γI)^-1); unset -> load_prior default 0.8.
[ -n "${GAMMA:-}" ] && ARM_FLAGS+=(--gamma "${GAMMA}")
# temporal ridge ρ for the raw arm (tames the stiff empirical Σ_T); unset -> default 0.1.
[ -n "${SHRINK_T:-}" ] && ARM_FLAGS+=(--shrink-t "${SHRINK_T}")
# temporal factor: 'laplacian' -> clean product graph (raw arm); unset -> empirical Σ_T.
[ -n "${TEMPORAL:-}" ] && ARM_FLAGS+=(--temporal "${TEMPORAL}")
[ -n "${GAMMA_T:-}" ]  && ARM_FLAGS+=(--gamma-t "${GAMMA_T}")
# spatial graph (road/correlation/hybrid) + winsor for the predict-refine robust-prior arm.
[ -n "${SPATIAL_GRAPH:-}" ] && ARM_FLAGS+=(--spatial-graph "${SPATIAL_GRAPH}")
[ -n "${WINSOR:-}" ]        && ARM_FLAGS+=(--winsor "${WINSOR}")
# low-capacity gcnn denoiser (prior-isolation arm); unset -> ugnet.
[ -n "${DENOISER:-}" ]      && ARM_FLAGS+=(--denoiser "${DENOISER}")
ARM_EXP="${EXP}/${ARM}"
export STAGE="diffusion"

# squeue-friendly name: the distinguishing part FIRST (short dataset + short arm),
# the exp tag last so column truncation only ever hides the date.
case "${DATASET}" in
  metrla)  DS_SHORT="mt"  ;;
  pemsbay) DS_SHORT="bay" ;;
  pems04f) DS_SHORT="04f" ;;
  pems04s) DS_SHORT="04s" ;;
  pems08f) DS_SHORT="08f" ;;
  pems08s) DS_SHORT="08s" ;;
  *)       DS_SHORT="${DATASET}" ;;
esac
case "${ARM}" in
  kron_pastG)  ARM_SHORT="krP"  ;;
  kron_pastPN) ARM_SHORT="krPN" ;;
  kron_base)   ARM_SHORT="kr0"  ;;
  iso_pastG)   ARM_SHORT="isoP" ;;
  *)           ARM_SHORT="${ARM}" ;;
esac
RUN_NAME="${DS_SHORT}-${ARM_SHORT}-s${SEED}-${EXP}"
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
--backbone ${BACKBONE} --exp ${ARM_EXP} --seed ${SEED} --epochs ${EPOCHS} --lr ${LR} \
--loss ${LOSS} ${ARM_FLAGS[*]} --final-val"

srun --unbuffered --export=ALL "${DAIC_PYTHON}" -u scripts/train_diffusion.py \
    --dataset "${DATASET}" \
    --prior "${PRIOR}" \
    --backbone "${BACKBONE}" \
    --exp "${ARM_EXP}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --loss "${LOSS}" \
    "${ARM_FLAGS[@]}" \
    "${FINAL_FLAGS[@]}" \
    --final-val
