#!/usr/bin/env bash
# PRIOR-RESHAPE experiment: submit ONE slurm array PER DATASET (datasets run in parallel).
# each array's tasks are the arms below — a prior × conditioning grid plus a sampler ablation,
# all on the small GCNN denoiser + per-node norm + FCPS (no schedule confound), ancestral eval.
#
#   EXP=20260719_reshape scripts/daic/submit_reshape.sh 42
#   DATASETS="pems08f pems08s pems04s" EXP=... scripts/daic/submit_reshape.sh 42   # subset
#   anything after the seed passes straight through to sbatch (e.g. --time=12:00:00)
#
# arms answer: (a) does the prior help? iso vs empirical vs reshaped; (b) does it interact with
# conditioning? forecast-only (fo) vs +residual-history (past); (c) sampler? loguniform vs powerlaw
# on the two stiff priors. metric = best per-epoch sampled ancestral CRPS (best_crps.pt).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/env.sh"

SEED="${1:?usage: EXP=<tag> submit_reshape.sh <seed> [extra sbatch args]}"; shift
: "${EXP:?set EXP=<dated tag>, e.g. EXP=20260719_reshape}"
DATASETS="${DATASETS:-metrla pemsbay pems04f pems04s pems08f pems08s}"
BACKBONE="${BACKBONE:-gwn}"
EPOCHS="${EPOCHS:-200}"
THROTTLE="${THROTTLE:-8}"

# arm := name:prior:temporal:cap:loss:schedule   ('-' = omitted). sampler=loguniform and
# --cond-past are FIXED in the sbatch (validated locally). tests: (a) prior iso/emp/reshaped,
# (b) loss snr vs whitened — does whitening make the structured prior actually help?, and
# (c) a vanilla DDPM baseline (plain white noise + standard schedule — the textbook control).
ARMS=(
  "ddpm:vp:-:-:snr:auto"                        # vanilla DDPM baseline (prior=vp auto-resolves VP schedule)
  "iso:isotropic:-:-:snr:fcps"                  # iso control (white noise, our machinery)
  "emp:kronecker:empirical:-:snr:fcps"          # empirical prior, GAD snr loss
  "rsh:kronecker:reshaped:50:snr:fcps"          # reshaped prior, GAD snr loss
  "iso_w:isotropic:-:-:whitened:fcps"           # iso + whitened (sanity: should ≈ iso)
  "emp_w:kronecker:empirical:-:whitened:fcps"   # empirical + whitened  (KEY)
  "rsh_w:kronecker:reshaped:50:whitened:fcps"   # reshaped + whitened   (KEY)
)

export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
LOG_DIR="${OUTPUT_ROOT}/logs/${EXP}"; mkdir -p "${LOG_DIR}"

source "${HERE}/_gpu.sh"
GRES="$(gpu_blackwell_gres || true)"; GRES="${GRES:-${DAIC_GRES_GPU}}"
RES=(--partition="${DAIC_PARTITION}" --gres="${GRES}")
[ -n "${DAIC_QOS_MEDIUM:-}" ] && RES+=(--qos="${DAIC_QOS_MEDIUM}")
[ -n "${DAIC_GPU_EXCLUDE:-}" ] && RES+=(--exclude="${DAIC_GPU_EXCLUDE}")

ln -sfn "${OUTPUT_ROOT}" "$(cd "${HERE}/../.." && pwd)/out" 2>/dev/null || true
echo "===== reshape experiment ${EXP}  seed=${SEED}  backbone=${BACKBONE}  ${#ARMS[@]} arms/dataset ====="
echo "datasets: ${DATASETS}   (one array each — parallel)"

for DS in ${DATASETS}; do
  GRID_FILE="${LOG_DIR}/grid_${DS}_seed${SEED}.tsv"; : > "${GRID_FILE}"
  idx=0
  for ARM in "${ARMS[@]}"; do
    IFS=':' read -r NAME PR TMP CAP LOSS SCHED <<< "${ARM}"
    RUN_DIR="${OUTPUT_ROOT}/${EXP}/${NAME}/${DS}/${BACKBONE}/${PR}/seed${SEED}"
    printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${idx}" "${DS}" "${NAME}" "${PR}" "${TMP}" "${CAP}" "${LOSS}" "${SCHED}" "${RUN_DIR}" >> "${GRID_FILE}"
    idx=$((idx + 1))
  done
  N="${idx}"
  EXPORTS="ALL,EXP=${EXP},SEED=${SEED},BACKBONE=${BACKBONE},EPOCHS=${EPOCHS},GRID_FILE=${GRID_FILE}"
  EXPORTS="${EXPORTS},DATA_ROOT=${DATA_ROOT},RESULTS_ROOT=${RESULTS_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}"
  EXPORTS="${EXPORTS},WANDB_MODE=disabled"   # storage: all needed info is in validation_history.json
  JOBID="$(sbatch --parsable \
         --array="0-$((N - 1))%${THROTTLE}" \
         --export="${EXPORTS}" \
         --job-name="${DS}-${EXP}" \
         --output="${LOG_DIR}/%A_%a.out" \
         --error="${LOG_DIR}/%A_%a.err" \
         "${RES[@]}" "$@" "${HERE}/submit_reshape.sbatch")"
  echo "  ${DS}: array ${JOBID} (${N} arms)  grid=${GRID_FILE}"
  ARRAY_IDS="${ARRAY_IDS:+${ARRAY_IDS}:}${JOBID}"
done

# automatic end-of-run summary: a tiny job that waits for ALL dataset arrays (afterany),
# then writes OUTPUT_ROOT/EXP/SUMMARY.md.
SUM_ID="$(sbatch --parsable \
       --dependency="afterany:${ARRAY_IDS}" \
       --export="ALL,EXP=${EXP},OUTPUT_ROOT=${OUTPUT_ROOT}" \
       --job-name="sum-${EXP}" \
       --output="${LOG_DIR}/summary.out" --error="${LOG_DIR}/summary.err" \
       "${RES[@]}" "${HERE}/summarize_reshape.sbatch")"
echo "  summary job ${SUM_ID} (runs after all arrays → ${OUTPUT_ROOT}/${EXP}/SUMMARY.md)"
echo
echo "watch:   squeue --me   |   logs: ${LOG_DIR}/"
echo "summary: cat ${OUTPUT_ROOT}/${EXP}/SUMMARY.md   (auto-written when everything finishes)"
