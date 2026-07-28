#!/bin/bash
#SBATCH --job-name=pgraph
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# GTCNN product-graph campaign runner. CELLS env = semicolon-separated "arm|dataset"
# pairs, executed serially in one job. each cell is:
#   1. train              raw signal, history-only conditioning, GCNN 17,305-param denoiser
#   2. --final-val        val step sweep (ancestral grid + EM grid) for the step choice
#   3. --eval-after       TEST eval over the (sampler, steps) grid at k=EVAL_K  -> the
#                         few-step question: does the prior buy anything at low steps
#   4. headline eval      TEST ancestral@50 at k=100, the locked protocol number that is
#                         comparable to every other campaign in the project
#
# submit via scripts/daic/pgraph.sh (which gates on the verification script first).
set -uo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"
export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

EXP="${EXP:?set EXP}"
SEED="${SEED:-42}"
GAMMA="${GAMMA:-0.1}"
C0="${C0:-100}"
EPOCHS="${EPOCHS:-200}"
EVAL_K="${EVAL_K:-8}"                       # k for the (sampler, steps) GRID
HEAD_K="${HEAD_K:-100}"                     # k for the locked-protocol headline point
ANC_STEPS="${ANC_STEPS:-5,10,25,50,100}"    # ancestral: no stability floor -> few-step story
EM_STEPS="${EM_STEPS:-250}"                 # EM: floored at max(200, theta_max*c_0)=210
HEAD_STEPS="${HEAD_STEPS:-50}"              # headline = ancestral@50, the locked protocol
# SCHEDULE=fcps for EVERY arm, deliberately. --schedule auto would put the isotropic arm
# on ULS and the product arms on FCPS; since this campaign's headline question is "does
# the prior shape help at FEW reverse steps", a schedule that differs across arms would
# confound exactly the comparison being made (the step grid is placed by the schedule).
# With c_0 shared, all arms see the same total heat AND the same c-bar curve, so the only
# difference left is the shape of theta over modes. Set SCHEDULE=auto to restore the
# project's legacy per-prior choice (iso->ULS) if cross-campaign iso comparability is
# wanted instead.
SCHEDULE="${SCHEDULE:-fcps}"

fail=0
IFS=';' read -r -a CELL_ARR <<< "${CELLS:?set CELLS}"
for cell in "${CELL_ARR[@]}"; do
  arm="${cell%%|*}"; ds="${cell##*|}"
  # arm -> (prior, temporal). iso is the variance-matched white-noise control.
  if [ "${arm}" = "iso" ]; then
    prior_args="--prior isotropic"
  else
    prior_args="--prior kronecker --temporal ${arm} --gamma ${GAMMA}"
  fi
  echo "═════ cell ${arm} · ${ds} (exp ${EXP}_${arm}) ═════"
  exp="${EXP}_${arm}"

  # 1-3: train + val step sweep + test grid
  "${DAIC_PYTHON}" scripts/train_diffusion.py \
      --dataset "${ds}" --backbone raw ${prior_args} \
      --denoiser gcnn --c-0 "${C0}" --schedule "${SCHEDULE}" --noise-sampler powerlaw \
      --epochs "${EPOCHS}" --exp "${exp}" --seed "${SEED}" \
      --final-val --final-anc-steps "${ANC_STEPS}" --final-steps-list "${EM_STEPS}" \
      --eval-after --eval-ckpt best_crps.pt --eval-k "${EVAL_K}" \
      --eval-anc-steps "${ANC_STEPS}" --eval-steps "${EM_STEPS}" \
    || { echo "TRAIN/EVAL FAILED ${arm} ${ds}"; fail=$((fail+1)); continue; }

  # 4: locked-protocol headline (ancestral@50, k=100, y==0 masked, best_crps.pt).
  # separate pass because the grid runs at a cheaper k -- this is the number that is
  # comparable to the heat/kron campaigns.
  "${DAIC_PYTHON}" scripts/eval_diffusion.py \
      --dataset "${ds}" --backbone raw \
      --prior "$([ "${arm}" = iso ] && echo isotropic || echo kronecker)" \
      --exp "${exp}" --seed "${SEED}" --split test \
      --sampler ancestral --n-steps "${HEAD_STEPS}" --k "${HEAD_K}" \
      --ckpt best_crps.pt \
    || { echo "HEADLINE EVAL FAILED ${arm} ${ds}"; fail=$((fail+1)); }
done
echo "done, ${fail} failed"; [ "${fail}" -eq 0 ]
