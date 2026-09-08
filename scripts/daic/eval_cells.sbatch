#!/bin/bash
#SBATCH --job-name=evalcells
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# TEST eval only, for cells that were trained but need (re-)evaluation -- e.g. the five
# 20260907_matched cells evaluated before the joint-CRPS fix (b452482). Same protocol as
# histcond_cell.sbatch: best_crps.pt, corrected ancestral at the DERIVED floor and step
# count (no --n-steps), fair CRPS, k=100, y==0 masked, per-point + joint metrics.
#
# CELLS = semicolon-separated "exp|dataset|prior|seed", prior = isotropic | kronecker, e.g.
#   sbatch --export=ALL,CELLS='20260907_matched/kron-hist|pems08f|kronecker|42' scripts/daic/eval_cells.sbatch
set -uo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"
export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
export WANDB_MODE=disabled
K="${K:-100}"
fail=0
IFS=';' read -r -a CELL_ARR <<< "${CELLS:?set CELLS}"
for cell in "${CELL_ARR[@]}"; do
  IFS='|' read -r exp ds prior seed <<< "${cell}"
  echo "═════ eval ${exp} · ${ds} · ${prior} · seed${seed} ═════"
  "${DAIC_PYTHON}" scripts/eval_diffusion.py \
      --dataset "${ds}" --backbone gwn --prior "${prior}" \
      --exp "${exp}" --seed "${seed}" --split test \
      --sampler ancestral --k "${K}" --ckpt best_crps.pt \
    || { echo "EVAL FAILED ${cell}"; fail=$((fail+1)); }
done
echo "done, ${fail} failed"; [ "${fail}" -eq 0 ]
