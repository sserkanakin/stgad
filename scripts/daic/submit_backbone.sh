#!/usr/bin/env bash
# Submit the backbone -> residual grid: each (VARIANT, BACKBONE) cell trains the
# point forecaster and extracts its residuals. Cluster-aware (partition/qos/gres
# from clusters.sh). The two daic02 accounts split the grid via VARIANTS/BACKBONES.
#
# usage:
#   SEED=42 scripts/daic/submit_backbone.sh
#   VARIANTS="metrla pems08f" BACKBONES="mlp gcn gwn" SEED=42 scripts/daic/submit_backbone.sh
#   # split across accounts: run the headline half on one, the rest on the other.
#
# anything after the (optional) env vars still passes through to sbatch.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/clusters.sh"

SEED="${SEED:-42}"
# 6 traffic variants (molene has no backbone -- it's the unconditional arm).
VARIANTS="${VARIANTS:-metrla pemsbay pems04f pems04s pems08f pems08s}"
BACKBONES="${BACKBONES:-mlp gcn gwn}"
EPOCHS="${EPOCHS:-100}"
THROTTLE="${THROTTLE:-8}"

LOG_DIR="${PROJECT_DIR}/logs/backbone"
mkdir -p "${LOG_DIR}"

short_ds() { case "$1" in metrla) echo mla;; pemsbay) echo pby;; *) echo "${1#pems}";; esac; }
DS_TAG="$(for d in ${VARIANTS}; do short_ds "$d"; done | paste -sd- -)"
GRID_FILE="${LOG_DIR}/grid_seed${SEED}_${DS_TAG}.tsv"
: > "${GRID_FILE}"

idx=0
for V in ${VARIANTS}; do
  for B in ${BACKBONES}; do
    printf '%d\t%s\t%s\n' "${idx}" "${V}" "${B}" >> "${GRID_FILE}"
    idx=$((idx + 1))
  done
done
N="${idx}"

echo "===== backbone grid  seed=${SEED}  (${N} tasks, throttle ${THROTTLE}) ====="
echo "variants:  ${VARIANTS}"
echo "backbones: ${BACKBONES}   epochs=${EPOCHS}"
column -t "${GRID_FILE}" | sed 's/^/  /'
echo

GRES="$(gpu_blackwell_gres || true)"; GRES="${GRES:-${DAIC_GRES_GPU}}"
RES=(--partition="${DAIC_PARTITION}" --gres="${GRES}")
[ -n "${DAIC_QOS_MEDIUM:-}" ] && RES+=(--qos="${DAIC_QOS_MEDIUM}")

JOBID="$(sbatch --parsable \
       --array="0-$((N - 1))%${THROTTLE}" \
       --export="ALL,SEED=${SEED},EPOCHS=${EPOCHS},GRID_FILE=${GRID_FILE}" \
       --job-name="bb-${DS_TAG}" \
       --output="${LOG_DIR}/%A_%a.out" \
       --error="${LOG_DIR}/%A_%a.err" \
       "${RES[@]}" "$@" "${HERE}/submit_backbone.sbatch")"

echo "submitted ArrayJobID=${JOBID}"
awk -F'\t' -v jid="${JOBID}" -v ld="${LOG_DIR}" \
  '{printf "  task %s  %-9s %-4s  log=%s/%s_%s.out\n", $1, $2, $3, ld, jid, $1}' "${GRID_FILE}"
echo "watch: squeue --me   |   tail: tail -f ${LOG_DIR}/${JOBID}_<task>.out"
