#!/bin/bash
#SBATCH --job-name=histcond
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# Conditioning-axis campaign runner. CELLS env = semicolon-separated
# "arm|dataset|backbone|seed|cond" tuples, executed serially in one job.
#
#   cond=hist  --cond-past  -> C = [X_z | Yhat_z]   (B,N,P+H), observed history window
#   cond=fo    (no flag)    -> C = Yhat_z           (B,N,H),   forecast only
#
# Each cell is:
#   1. train  200 epochs, patience 40, 17,305-param GCNN denoiser
#   2. locked-protocol TEST eval: best_crps.pt, ancestral@50, k=100, y==0 masked
#
# EVERY hyperparameter below is pinned to the 20260721_ladder values so the new
# history arm is directly comparable to that campaign's forecast-only cells.
# Flags that differ from the script defaults and so MUST be passed explicitly:
#   --patience 40         (default 15)
#   --noise-sampler loguniform (default powerlaw)
#   --denoiser gcnn       (default ugnet -- the ~1M UGnet is a different rung)
#   --norm pernode        (default global)
#   --schedule fcps       (default 'auto', which would silently put the ISOTROPIC arm
#                          on ULS while the kronecker arm gets FCPS -- that would
#                          confound the prior comparison with a schedule change, and
#                          the ladder's iso cells are all fcps)
#   --val-windows 512     (default 64) NOT a diagnostic knob: _fast_val scores this
#                          same clean-window subset, and that signal drives
#                          best_crps.pt, early stopping and the LR plateau. At 64 the
#                          checkpoint would be selected on an 8x smaller val set than
#                          the ladder's forecast-only cells we compare against.
#   --val-every 0         (default 10) disables the periodic EM step-sweep diagnostic,
#                          as the ladder did. It does not select anything, but it costs
#                          GPU time every 10 epochs for a number we do not use here.
# submit via scripts/daic/histcond.sh.
set -uo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"
export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

EXP="${EXP:?set EXP}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-40}"
C0="${C0:-7.0}"
SCHEDULE="${SCHEDULE:-fcps}"
NORM="${NORM:-pernode}"
LOSS="${LOSS:-snr}"
NOISE_SAMPLER="${NOISE_SAMPLER:-loguniform}"
RESIDUAL_SEED="${RESIDUAL_SEED:-42}"     # backbone pinned; only the diffusion seed varies
VAL_WINDOWS="${VAL_WINDOWS:-512}"        # sizes the checkpoint-SELECTION signal, see above
VAL_EVERY="${VAL_EVERY:-0}"              # EM step-sweep diagnostic off, as the ladder had
HEAD_K="${HEAD_K:-100}"
HEAD_STEPS="${HEAD_STEPS:-50}"

fail=0
IFS=';' read -r -a CELL_ARR <<< "${CELLS:?set CELLS}"
for cell in "${CELL_ARR[@]}"; do
  IFS='|' read -r arm ds bb seed cond <<< "${cell}"

  # arm -> prior flags. emp = Sigma_T(empirical) (x) Sigma_S=(L_road + gamma I)^-1,
  # gamma left unset so prior.py uses GAMMA_DEFAULT=0.8 (GAD's fixed shift), which is
  # what every 20260721_ladder emp cell resolved to.
  if [ "${arm}" = "iso" ]; then
    prior_args=(--prior isotropic)
    prior_name="isotropic"
  else
    prior_args=(--prior kronecker --temporal empirical --spatial structured --spatial-graph road)
    prior_name="kronecker"
  fi
  cond_args=()
  [ "${cond}" = "hist" ] && cond_args=(--cond-past)

  exp="${EXP}/${arm}-${cond}"
  echo "═════ cell ${arm} · ${ds} · ${bb} · seed${seed} · cond=${cond} (exp ${exp}) ═════"

  "${DAIC_PYTHON}" scripts/train_diffusion.py \
      --dataset "${ds}" --backbone "${bb}" \
      "${prior_args[@]}" "${cond_args[@]}" \
      --denoiser gcnn \
      --c-0 "${C0}" --schedule "${SCHEDULE}" --noise-sampler "${NOISE_SAMPLER}" \
      --loss "${LOSS}" --norm "${NORM}" \
      --epochs "${EPOCHS}" --patience "${PATIENCE}" \
      --val-windows "${VAL_WINDOWS}" --val-every "${VAL_EVERY}" \
      --exp "${exp}" --seed "${seed}" --residual-seed "${RESIDUAL_SEED}" \
    || { echo "TRAIN FAILED ${cell}"; fail=$((fail+1)); continue; }

  # locked protocol -- the number comparable to every other campaign in the project.
  "${DAIC_PYTHON}" scripts/eval_diffusion.py \
      --dataset "${ds}" --backbone "${bb}" --prior "${prior_name}" \
      --exp "${exp}" --seed "${seed}" --split test \
      --sampler ancestral --n-steps "${HEAD_STEPS}" --k "${HEAD_K}" \
      --ckpt best_crps.pt \
    || { echo "EVAL FAILED ${cell}"; fail=$((fail+1)); }
done
echo "done, ${fail} failed"; [ "${fail}" -eq 0 ]
