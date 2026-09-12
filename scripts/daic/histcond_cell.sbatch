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
#   1. train  200 epochs, patience 40, 17,305-param GCNN denoiser. the kron arm uses
#      --global-edge match: (beta, gamma) chosen on the train residuals so the prior's
#      spread on the NETWORK TOTAL is right. the noise floor and the ancestral step
#      count are derived from the prior's spectrum and recorded in config.json.
#   2. TEST eval: best_crps.pt, corrected ancestral at the derived floor and step
#      count, fair CRPS, k=100, y==0 masked, per-point AND joint (total/cumulative).
#
# EVERY hyperparameter below is pinned to the 20260721_ladder values so the new
# history arm is directly comparable to that campaign's forecast-only cells.
# Flags that differ from the script defaults and so MUST be passed explicitly:
#   --patience 40         (default 15)
#   --denoiser gcnn       (default ugnet -- the ~1M UGnet is a different rung)
#   --norm pernode        (default global)
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
NORM="${NORM:-pernode}"
RESIDUAL_SEED="${RESIDUAL_SEED:-42}"     # backbone pinned; only the diffusion seed varies
VAL_WINDOWS="${VAL_WINDOWS:-512}"        # sizes the checkpoint-SELECTION signal, see above
VAL_EVERY="${VAL_EVERY:-0}"              # EM step-sweep diagnostic off, as the ladder had
HEAD_K="${HEAD_K:-100}"
HEAD_STEPS="${HEAD_STEPS:-}"             # empty = the derived count in config.json
GLOBAL_EDGE="${GLOBAL_EDGE:-match}"      # kron arm only
# 17,305-param GCNN by default. DENOISER=ugnet gives the ~996k UGnet of DiffSTG --
# the capacity axis. It is a different rung, so it must be given its own EXP tag
# (e.g. EXP=20260812_histcond_ugnet) or it will collide with the 17k cells.
DENOISER="${DENOISER:-gcnn}"

fail=0
IFS=';' read -r -a CELL_ARR <<< "${CELLS:?set CELLS}"
for cell in "${CELL_ARR[@]}"; do
  IFS='|' read -r arm ds bb seed cond <<< "${cell}"

  # arm -> prior. iso = white noise sigma^2 I. kron = Sigma_T (x) Sigma_S, Sigma_T fit
  # from the residuals, Sigma_S = (L_road + gamma I)^-1; gamma left unset so prior.py
  # uses GAMMA_DEFAULT.
  prior_extra=()
  case "${arm}" in
    iso)  prior_name="isotropic" ;;
    kron) prior_name="kronecker"; prior_extra=(--global-edge "${GLOBAL_EDGE}") ;;
    # product-graph priors (the ablation of the separable form): kronecker-sum
    # precision on road graph x horizon path, (beta, gamma) matched with the exact rule
    cart)   prior_name="cartesian"; prior_extra=(--global-edge "${GLOBAL_EDGE}") ;;
    strong) prior_name="strong";    prior_extra=(--global-edge "${GLOBAL_EDGE}") ;;
    *) echo "unknown arm '${arm}' (use iso|kron|cart|strong)"; fail=$((fail+1)); continue ;;
  esac
  cond_args=()
  [ "${cond}" = "hist" ] && cond_args=(--cond-past)

  exp="${EXP}/${arm}-${cond}"
  echo "═════ cell ${arm} · ${ds} · ${bb} · seed${seed} · cond=${cond} (exp ${exp}) ═════"

  "${DAIC_PYTHON}" scripts/train_diffusion.py \
      --dataset "${ds}" --backbone "${bb}" \
      --prior "${prior_name}" "${prior_extra[@]}" "${cond_args[@]}" \
      --denoiser "${DENOISER}" \
      --c-0 "${C0}" --norm "${NORM}" \
      --epochs "${EPOCHS}" --patience "${PATIENCE}" \
      --val-windows "${VAL_WINDOWS}" --val-every "${VAL_EVERY}" \
      --exp "${exp}" --seed "${seed}" --residual-seed "${RESIDUAL_SEED}" \
    || { echo "TRAIN FAILED ${cell}"; fail=$((fail+1)); continue; }

  # locked protocol -- the number comparable to every other campaign in the project.
  "${DAIC_PYTHON}" scripts/eval_diffusion.py \
      --dataset "${ds}" --backbone "${bb}" --prior "${prior_name}" \
      --exp "${exp}" --seed "${seed}" --split test \
      --sampler ancestral ${HEAD_STEPS:+--n-steps "${HEAD_STEPS}"} --k "${HEAD_K}" \
      --ckpt best_crps.pt \
    || { echo "EVAL FAILED ${cell}"; fail=$((fail+1)); }
done
echo "done, ${fail} failed"; [ "${fail}" -eq 0 ]
