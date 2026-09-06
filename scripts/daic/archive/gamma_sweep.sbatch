#!/bin/bash
#SBATCH --job-name=gamma-sweep
# γ-sweep for the EMPIRICAL prior only (γ sets the spatial ridge Σ_S=(L+γI)^{-1}; it does
# NOT touch iso or reshaped). Tests whether γ=0.8 under-commits: as γ↓ the empirical spatial
# spread steepens toward the measured residual spectrum (24.6× on pems04f) and toward the
# reshaped arm. If empirical plateaus ABOVE reshaped, the reshape novelty is necessary; if it
# reaches reshaped, γ-tuning would suffice. ONE job loops every (γ, backbone, seed) serially,
# so it occupies exactly one slot. ancestral per-epoch CRPS selection → best_crps.pt.
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
# partition / qos / gres are injected by submit_gamma_sweep.sh (never sbatch directly).
set -uo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

EXP="${EXP:-20260721_gammasweep}"
DATASET="${DATASET:-pems08f}"
# multi-word values arrive underscore-joined over --export (no spaces allowed); decode here.
GAMMAS="${GAMMAS:-0.04 0.1 0.2 0.4}"; GAMMAS="${GAMMAS//_/ }"
BACKBONES="${BACKBONES:-mlp gwn}";    BACKBONES="${BACKBONES//_/ }"
SEEDS="${SEEDS:-42}";                 SEEDS="${SEEDS//_/ }"
EPOCHS="${EPOCHS:-200}"; LR="${LR:-1e-4}"; PATIENCE="${PATIENCE:-40}"
RESIDUAL_SEED="${RESIDUAL_SEED:-42}"; VAL_WINDOWS="${VAL_WINDOWS:-512}"

echo "===== γ-sweep ${EXP}  dataset=${DATASET}  γ=[${GAMMAS}] × bb=[${BACKBONES}] × seeds=[${SEEDS}] ====="

n=0; fail=0
for GAMMA in ${GAMMAS}; do
  TAG="g${GAMMA//./p}"                 # 0.04 → g0p04, 0.1 → g0p1
  ARM_EXP="${EXP}/emp_${TAG}"
  # same recipe as the reshape empirical arm — only --gamma differs.
  FLAGS=(--denoiser gcnn --norm pernode --noise-sampler loguniform
         --loss snr --schedule fcps --patience "${PATIENCE}"
         --residual-seed "${RESIDUAL_SEED}"
         --val-every 0 --val-windows "${VAL_WINDOWS}" --fast-val-steps 50
         --temporal empirical --gamma "${GAMMA}")
  for BB in ${BACKBONES}; do
    for S in ${SEEDS}; do
      n=$((n + 1))
      echo "───── [${n}] ${DATASET}/${BB}  empirical γ=${GAMMA}  seed${S}  (exp=${ARM_EXP}) ─────"
      "${DAIC_PYTHON}" scripts/train_diffusion.py \
        --dataset "${DATASET}" --prior kronecker --backbone "${BB}" \
        --exp "${ARM_EXP}" --seed "${S}" --epochs "${EPOCHS}" --lr "${LR}" \
        "${FLAGS[@]}" \
        || { echo "  !! FAILED γ=${GAMMA}/${BB}/seed${S}"; fail=$((fail + 1)); }
    done
  done
done
echo "===== γ-sweep done: ${n} runs attempted, ${fail} failed ====="
echo "compare: ${OUTPUT_ROOT}/${EXP}/emp_g*/  vs  20260721_ladder/{emp (γ=0.8), rsh}"
[ "${fail}" -eq 0 ]
