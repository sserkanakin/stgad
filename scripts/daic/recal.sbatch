#!/bin/bash
#SBATCH --job-name=recal
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# serial adaptive-recal runner: RUNS = semicolon-separated arg strings for
# scripts/analysis/adaptive_recal.py. results land in ${RESULTS_ROOT}/analysis/adaptive_recal.
#   sbatch --partition=... --gres=gpu:1 --export=ALL,RUNS="--dataset metrla --backbone gwn --cond past" scripts/daic/recal.sbatch
set -uo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"
export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
export WANDB_MODE=disabled
IFS=';' read -r -a RUN_ARR <<< "${RUNS:?set RUNS}"
# the script may be parked outside the repo (linuxhome quota) — RECAL_SCRIPT overrides.
SCRIPT="${RECAL_SCRIPT:-${PROJECT_DIR}/scripts/analysis/adaptive_recal.py}"
fail=0
for args in "${RUN_ARR[@]}"; do
  echo "───── adaptive_recal.py ${args} ─────"
  # shellcheck disable=SC2086
  "${DAIC_PYTHON}" "${SCRIPT}" ${args} || fail=$((fail+1))
done
echo "done, ${fail} failed"; [ "${fail}" -eq 0 ]
