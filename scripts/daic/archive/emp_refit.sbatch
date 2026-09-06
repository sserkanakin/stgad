#!/bin/bash
#SBATCH --job-name=emp-refit
# Empirical arm REFIT at the tuned γ (the γ-sweep showed GAD's γ=0.8 under-commits by ~21% on
# weak backbones; γ≈0.2 is the plateau optimum). Re-runs ONLY empirical so the baseline in the
# main tables is fair — iso/reshaped/ddpm/whitened are γ-free and stay. One array TASK = one
# (dataset, backbone) full training run, all seeds sequential, ancestral per-epoch CRPS → best_crps.pt.
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
# partition / qos / gres + --array are injected by submit_emp_refit.sh (never sbatch directly).
set -uo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

EXP="${EXP:-20260721_emprefit}"
GAMMA="${GAMMA:-0.2}"
# multi-word values arrive underscore-joined over --export; decode.
DATASETS="${DATASETS:-pems08f pems08s pems04f pems04s}"; DATASETS="${DATASETS//_/ }"
BACKBONES="${BACKBONES:-mlp gwn}";                       BACKBONES="${BACKBONES//_/ }"
SEEDS="${SEEDS:-42 126 2022 3407 1}";                    SEEDS="${SEEDS//_/ }"
EPOCHS="${EPOCHS:-200}"; LR="${LR:-1e-4}"; PATIENCE="${PATIENCE:-40}"
RESIDUAL_SEED="${RESIDUAL_SEED:-42}"; VAL_WINDOWS="${VAL_WINDOWS:-512}"

# task id → (dataset, backbone): backbone varies fastest.
read -r -a DS_ARR <<< "${DATASETS}"; read -r -a BB_ARR <<< "${BACKBONES}"
NB=${#BB_ARR[@]}
di=$(( SLURM_ARRAY_TASK_ID / NB )); bi=$(( SLURM_ARRAY_TASK_ID % NB ))
DATASET="${DS_ARR[$di]}"; BACKBONE="${BB_ARR[$bi]}"
ARM_EXP="${EXP}/emp"      # empirical arm; γ recorded in config.json

echo "===== emp-refit task ${SLURM_ARRAY_TASK_ID}: ${DATASET}/${BACKBONE}  empirical γ=${GAMMA}  (exp=${ARM_EXP}) ====="
echo "===== seeds (sequential): ${SEEDS} ====="

FLAGS=(--denoiser gcnn --norm pernode --noise-sampler loguniform
       --loss snr --schedule fcps --patience "${PATIENCE}"
       --residual-seed "${RESIDUAL_SEED}"
       --val-every 0 --val-windows "${VAL_WINDOWS}" --fast-val-steps 50
       --temporal empirical --gamma "${GAMMA}")

fail=0
for S in ${SEEDS}; do
  echo "───── seed ${S}  (${DATASET}/${BACKBONE}/emp/γ=${GAMMA}) ─────"
  "${DAIC_PYTHON}" scripts/train_diffusion.py \
    --dataset "${DATASET}" --prior kronecker --backbone "${BACKBONE}" \
    --exp "${ARM_EXP}" --seed "${S}" --epochs "${EPOCHS}" --lr "${LR}" \
    "${FLAGS[@]}" || { echo "  !! FAILED seed ${S}"; fail=$((fail + 1)); }
done
echo "===== ${DATASET}/${BACKBONE} done: ${fail} seed(s) failed ====="
[ "${fail}" -eq 0 ]
