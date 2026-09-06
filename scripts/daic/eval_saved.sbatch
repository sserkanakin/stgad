#!/bin/bash
#SBATCH --job-name=evalsaved
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# Locked-protocol TEST eval for checkpoints that were TRAINED but never test-evaluated.
#
# WHY. Two axes the thesis reports are validation-only, and both say so in the text:
#   * denoiser capacity -- the ~996k UGnet against the shared 17,305-param GCNN
#     (arms  emp-past-ugnet / iso-past-ugnet / rsh-past-ugnet)
#   * the corruption family -- a standard variance-preserving DDPM in place of the
#     structured OU process  (arms  ddpm / ddpm-past, prior dir 'vp')
# Their magnitudes are large enough that the ordering is not in doubt, but the exact
# values are not test-grade and cannot be mixed into a test comparison. That caveat
# appears three times in the thesis. These checkpoints already exist; only the eval
# was never run.
#
# CELLS = semicolon-separated "exp|dataset|backbone|prior|seed", where exp is the
# arm path under the campaign root and prior is the DIRECTORY name:
#   isotropic | kronecker
#
#   sbatch --export=ALL,CELLS='20260728_histcond/kron-hist|pems08f|mlp|kronecker|42' \
#          scripts/daic/eval_saved.sbatch
#
# Protocol is the locked one and is NOT configurable here on purpose: best_crps.pt,
# ancestral@50, k=100, y==0 masked, test split. That is what makes these numbers
# comparable to every other table in the project.
set -uo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
export STAGE=eval
source "${PROJECT_DIR}/scripts/daic/_common.sh"
export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

K="${K:-100}"
STEPS="${STEPS:-50}"

fail=0; ran=0; skipped=0
IFS=';' read -r -a CELL_ARR <<< "${CELLS:?set CELLS}"
for cell in "${CELL_ARR[@]}"; do
  IFS='|' read -r exp ds bb prior seed <<< "${cell}"
  base="${OUTPUT_ROOT}/${exp}/${ds}/${bb}/${prior}/seed${seed}"

  if [ ! -s "${base}/train/best_crps.pt" ]; then
    echo "MISSING CHECKPOINT: ${base}/train/best_crps.pt"; fail=$((fail+1)); continue
  fi
  out="${base}/eval/test_metrics_bestcrps_ancestral${STEPS}.json"
  if [ -s "${out}" ]; then
    echo "skip (exists): ${cell}"; skipped=$((skipped+1)); continue
  fi

  echo "═════ TEST EVAL  ${exp} · ${ds} · ${bb} · ${prior} · seed${seed} ═════"
  "${DAIC_PYTHON}" scripts/eval_diffusion.py \
      --dataset "${ds}" --backbone "${bb}" --prior "${prior}" \
      --exp "${exp}" --seed "${seed}" --split test \
      --sampler ancestral --n-steps "${STEPS}" --k "${K}" \
      --ckpt best_crps.pt \
    && ran=$((ran+1)) \
    || { echo "EVAL FAILED ${cell}"; fail=$((fail+1)); }
done
echo "done: ${ran} ran, ${skipped} skipped, ${fail} failed"
[ "${fail}" -eq 0 ]
