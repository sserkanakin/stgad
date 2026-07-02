#!/usr/bin/env bash
# launch the FULL diffusion grid as a SINGLE slurm array, dataset-major so a dataset's
# every method runs before the next dataset (not one method across all datasets first).
#
# one array task = one (dataset, backbone, prior) combined train+eval (best.pt by val-CRPS
# → locked test eval → test_results.json). the array is ordered:
#   metrla×{gwn,gcn,mlp,raw}×{kron,iso}, then pemsbay×{...}, then pems08f×{...}, ...
# with a concurrency throttle so the first wave is all metrla cells.
#
#   EXP=20260624 scripts/daic/submit_array.sh 42
#   EXP=20260624 THROTTLE=8 BACKBONES="gwn raw" scripts/daic/submit_array.sh 42
#
# backbone residuals (gwn/gcn/mlp) and the raw signal cache must already exist for the
# (dataset, seed).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/env.sh"

SEED="${1:?usage: submit_array.sh <seed>}"
export EXP="${EXP:-dev}"
# dataset-major × method order. raw = pure-signal arm (history-conditioned).
DATASETS="${DATASETS:-metrla pemsbay pems08f pems04f pems04s pems08s}"
BACKBONES="${BACKBONES:-gwn gcn mlp raw}"
PRIORS="${PRIORS:-kronecker isotropic}"
THROTTLE="${THROTTLE:-8}"
EVAL_STEPS="${EVAL_STEPS:-100,50,25}"
# INPUTS (residuals/processed) read from the data share (DAIC_DATA_DIR, where sync put
# them); OUTPUTS (checkpoints/eval/logs/wandb) written to windowshome in one clean tree.
# override OUTPUT_ROOT to write elsewhere. ARRAY_SPEC: re-run only specific failed cells.
export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
ARRAY_SPEC="${ARRAY_SPEC:-}"
: "${RESULTS_ROOT:?set DAIC_DATA_DIR in env.sh (or pass RESULTS_ROOT)}"
: "${OUTPUT_ROOT:?set DAIC_OUT_DIR in env.sh (or pass OUTPUT_ROOT=<windowshome path>)}"

# build the dataset-major combo list (idx \t dataset \t backbone \t prior). logs + the
# grid file live under OUTPUT_ROOT (windowshome) — one organized place for all writes.
LOG_BASE="${OUTPUT_ROOT}"
GRID_DIR="${LOG_BASE}/logs/${EXP}/grid"
mkdir -p "${GRID_DIR}"
GRID_FILE="${GRID_DIR}/grid_seed${SEED}.tsv"
: > "${GRID_FILE}"
idx=0
for DS in ${DATASETS}; do
  for BB in ${BACKBONES}; do
    for PR in ${PRIORS}; do
      printf '%d\t%s\t%s\t%s\n' "${idx}" "${DS}" "${BB}" "${PR}" >> "${GRID_FILE}"
      idx=$((idx + 1))
    done
  done
done
N="${idx}"

echo "===== diffusion grid  EXP=${EXP}  seed=${SEED} ====="
echo "datasets : ${DATASETS}"
echo "backbones: ${BACKBONES}   priors: ${PRIORS}"
echo "${N} cells (dataset-major), throttle ${THROTTLE} concurrent. config: ${GRID_FILE}"
echo "inputs (residuals): ${RESULTS_ROOT}"
echo "OUTPUTS (ckpt/eval/logs): ${OUTPUT_ROOT}"
column -t "${GRID_FILE}" | sed 's/^/  /'
echo

# per-cluster slurm resources (partition / qos / gres / gpu-exclude), same as submit.sh.
RES=(--partition="${DAIC_PARTITION}" --gres="${DAIC_GRES_GPU}")
[ -n "${DAIC_QOS_MEDIUM:-}" ] && RES+=(--qos="${DAIC_QOS_MEDIUM}")
[ -n "${DAIC_GPU_EXCLUDE:-}" ] && RES+=(--exclude="${DAIC_GPU_EXCLUDE}")

ARRAY="${ARRAY_SPEC:-0-$((N - 1))%${THROTTLE}}"     # full grid, or a failed-index subset
EXPORTS="ALL,EXP=${EXP},SEED=${SEED},GRID_FILE=${GRID_FILE},EVAL_STEPS=${EVAL_STEPS}"
EXPORTS="${EXPORTS},DATA_ROOT=${DATA_ROOT},RESULTS_ROOT=${RESULTS_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}"

set -x
sbatch --array="${ARRAY}" \
       --export="${EXPORTS}" \
       --output="${GRID_DIR}/%A_%a.out" \
       --error="${GRID_DIR}/%A_%a.err" \
       "${RES[@]}" "${HERE}/submit_grid.sbatch"
set +x

echo
echo "submitted array of ${N} cells. watch: squeue --me   |   logs: ${GRID_DIR}/"
