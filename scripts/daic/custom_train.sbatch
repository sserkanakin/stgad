#!/bin/bash
#SBATCH --job-name=custom-train
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# generic one-off training runner: RUNS env = semicolon-separated argument strings for
# train_diffusion.py, executed serially in one job. submit with partition/gres injected,
# e.g.  sbatch --partition=... --gres=gpu:1 --export=ALL,RUNS="..." custom_train.sbatch
set -uo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"
export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
export WANDB_MODE=disabled
IFS=';' read -r -a RUN_ARR <<< "${RUNS:?set RUNS}"
fail=0
for args in "${RUN_ARR[@]}"; do
  echo "───── train_diffusion.py ${args} ─────"
  # shellcheck disable=SC2086
  "${DAIC_PYTHON}" scripts/train_diffusion.py ${args} || fail=$((fail+1))
done
echo "done, ${fail} failed"; [ "${fail}" -eq 0 ]
