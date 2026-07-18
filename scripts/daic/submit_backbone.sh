#!/usr/bin/env bash
# Submit the backbone -> residual grid on DAIC. One array task = one (VARIANT,
# BACKBONE) cell: train_backbone.py (best.pt by val MAE) then extract_residuals.py.
# Organized exactly like submit_experiment.sh:
#   residuals/backbones -> RESULTS_ROOT (the data share, where the diffusion stage reads them)
#   logs + grid manifest -> OUTPUT_ROOT/logs/<EXP>/   (windowshome, quota-free)
#
# usage:
#   EXP=20260718_backbones scripts/daic/submit_backbone.sh
#   EXP=20260718_backbones VARIANTS="metrla pems08f" BACKBONES="mlp gcn gwn" scripts/daic/submit_backbone.sh
#   # split across accounts: run half the VARIANTS on each daic02 account, same EXP.
# anything after the env vars passes straight to sbatch (e.g. --time=18:00:00).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/clusters.sh"

: "${EXP:?set EXP=<dated tag>, e.g. EXP=20260718_backbones (names the log tree)}"
SEED="${SEED:-42}"
VARIANTS="${VARIANTS:-metrla pemsbay pems04f pems04s pems08f pems08s}"   # molene has no backbone
BACKBONES="${BACKBONES:-mlp gcn gwn}"
EPOCHS="${EPOCHS:-100}"
THROTTLE="${THROTTLE:-8}"

# roots: reads/writes of residuals+backbones on the data share; logs on windowshome.
export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
LOG_DIR="${OUTPUT_ROOT}/logs/${EXP}"
mkdir -p "${LOG_DIR}"

short_ds() { case "$1" in metrla) echo mla;; pemsbay) echo pby;; *) echo "${1#pems}";; esac; }
DS_TAG="$(for d in ${VARIANTS}; do short_ds "$d"; done | paste -sd- -)"
GRID_FILE="${LOG_DIR}/grid_backbone_seed${SEED}_${DS_TAG}.tsv"
: > "${GRID_FILE}"
idx=0
for V in ${VARIANTS}; do
  for B in ${BACKBONES}; do
    printf '%d\t%s\t%s\n' "${idx}" "${V}" "${B}" >> "${GRID_FILE}"
    idx=$((idx + 1))
  done
done
N="${idx}"

echo "===== backbone grid  EXP=${EXP}  seed=${SEED}  (${N} tasks, throttle ${THROTTLE}) ====="
echo "variants:  ${VARIANTS}"
echo "backbones: ${BACKBONES}   epochs=${EPOCHS}"
echo "residuals: ${RESULTS_ROOT}/residuals/<ds>/<backbone>/seed${SEED}/"
echo "logs:      ${LOG_DIR}/"
column -t "${GRID_FILE}" | sed 's/^/  /'
echo

# browsable ./out symlink to the output tree (same as submit_experiment.sh)
ln -sfn "${OUTPUT_ROOT}" "$(cd "${HERE}/../.." && pwd)/out" 2>/dev/null || true

GRES="$(gpu_blackwell_gres || true)"; GRES="${GRES:-${DAIC_GRES_GPU}}"
RES=(--partition="${DAIC_PARTITION}" --gres="${GRES}")
[ -n "${DAIC_QOS_MEDIUM:-}" ] && RES+=(--qos="${DAIC_QOS_MEDIUM}")

EXPORTS="ALL,EXP=${EXP},SEED=${SEED},EPOCHS=${EPOCHS},GRID_FILE=${GRID_FILE}"
EXPORTS="${EXPORTS},DATA_ROOT=${DATA_ROOT},RESULTS_ROOT=${RESULTS_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}"

JOBID="$(sbatch --parsable \
       --array="0-$((N - 1))%${THROTTLE}" \
       --export="${EXPORTS}" \
       --job-name="bb-${DS_TAG}-${EXP}" \
       --output="${LOG_DIR}/%A_%a.out" \
       --error="${LOG_DIR}/%A_%a.err" \
       "${RES[@]}" "$@" "${HERE}/submit_backbone.sbatch")"

echo "submitted ArrayJobID=${JOBID}"
awk -F'\t' -v jid="${JOBID}" -v ld="${LOG_DIR}" \
  '{printf "  task %s  %-9s %-4s  log=%s/%s_%s.out\n", $1, $2, $3, ld, jid, $1}' "${GRID_FILE}"
echo "watch: squeue --me   |   summary: python scripts/summarize_backbones.py --seed ${SEED}"
