#!/bin/bash
#SBATCH --job-name=rshl-pilot
# rsh-λ PILOT: joint-spectrum shrinkage for the reshaped prior, v_emp ← (1-λ)v_emp + λ·v̄
# (Ledoit-Wolf-style; the ridge the other components already have — γ spatial, ρ temporal).
# Motivated by the test reversal: the loud residual modes DRIFT train→test (energy-weighted
# ratio 1.3–3.8×) and the unregularized per-mode fit under-disperses exactly there.
# λ=0 = raw rsh (current), λ→1 = iso. Select λ on val, then roll out the winner.
# ONE job loops (λ × backbone × seed) serially — one queue slot.
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
# partition / qos / gres injected by submit_shrink_pilot.sh (never sbatch directly).
set -uo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

EXP="${EXP:-20260722_rshshrink}"
DATASET="${DATASET:-pems04f}"
LAMBDAS="${LAMBDAS:-0.2 0.5}"; LAMBDAS="${LAMBDAS//_/ }"
BACKBONES="${BACKBONES:-mlp gwn}"; BACKBONES="${BACKBONES//_/ }"
SEEDS="${SEEDS:-42}"; SEEDS="${SEEDS//_/ }"
EPOCHS="${EPOCHS:-200}"; LR="${LR:-1e-4}"; PATIENCE="${PATIENCE:-40}"
RESIDUAL_SEED="${RESIDUAL_SEED:-42}"; VAL_WINDOWS="${VAL_WINDOWS:-512}"

echo "===== rsh-λ pilot ${EXP}  ${DATASET}  λ=[${LAMBDAS}] × bb=[${BACKBONES}] × seeds=[${SEEDS}] ====="

n=0; fail=0
for L in ${LAMBDAS}; do
  TAG="l${L//./p}"                 # 0.2 → l0p2
  ARM_EXP="${EXP}/rsh_${TAG}"
  FLAGS=(--denoiser gcnn --norm pernode --noise-sampler loguniform
         --loss snr --schedule fcps --patience "${PATIENCE}"
         --residual-seed "${RESIDUAL_SEED}"
         --val-every 0 --val-windows "${VAL_WINDOWS}" --fast-val-steps 50
         --temporal reshaped --shrink-s "${L}")
  for BB in ${BACKBONES}; do
    for S in ${SEEDS}; do
      n=$((n + 1))
      echo "───── [${n}] ${DATASET}/${BB}  reshaped λ=${L}  seed${S}  (exp=${ARM_EXP}) ─────"
      "${DAIC_PYTHON}" scripts/train_diffusion.py \
        --dataset "${DATASET}" --prior kronecker --backbone "${BB}" \
        --exp "${ARM_EXP}" --seed "${S}" --epochs "${EPOCHS}" --lr "${LR}" \
        "${FLAGS[@]}" \
        || { echo "  !! FAILED λ=${L}/${BB}/seed${S}"; fail=$((fail + 1)); }
    done
  done
done
echo "===== rsh-λ pilot done: ${n} runs, ${fail} failed ====="
echo "compare val CRPS vs 20260721_ladder/{iso,emp,rsh} — pick λ, then roll out 5 seeds/4 ds."
[ "${fail}" -eq 0 ]
