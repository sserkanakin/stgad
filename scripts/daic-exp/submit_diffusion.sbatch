#!/bin/bash
#SBATCH --job-name=dsm-train
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=logs/train-diffusion/%x-%j.out
#SBATCH --error=logs/train-diffusion/%x-%j.err
#
# Score-network training on experimental DAIC.
#
# Submit one variant:
#   source scripts/daic-exp/env.sh
#   sbatch --export=DATASET=metrla,SEED=0,PRIOR=structured \
#          scripts/daic-exp/submit_diffusion.sbatch
#
# Full 20-job sweep (5 seeds × 2 priors × 2 datasets):
#   for ds in metrla pemsbay; do
#     for prior in structured isotropic; do
#       for seed in 0 1 2 3 4; do
#         sbatch --job-name="dsm-${ds}-${prior}-s${seed}" \
#                --export=DATASET=$ds,SEED=$seed,PRIOR=$prior \
#                scripts/daic-exp/submit_diffusion.sbatch
#       done
#     done
#   done

set -euo pipefail
mkdir -p logs/train-diffusion
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic-exp/_common.sh"

: "${DATASET:?Set DATASET=metrla or DATASET=pemsbay}"
: "${SEED:?Set SEED=<int>}"
: "${PRIOR:=structured}"
EPOCHS="${EPOCHS:-100}"

scontrol update job="${SLURM_JOB_ID}" JobName="dsm-${DATASET}-${PRIOR}-s${SEED}" 2>/dev/null || true

echo "Training score net: dataset=${DATASET} seed=${SEED} prior=${PRIOR} epochs=${EPOCHS}"

if [ ! -f scripts/train_diffusion.py ]; then
  echo "ERROR: scripts/train_diffusion.py not present yet." >&2
  exit 1
fi

srun --export=ALL "${DAIC_PYTHON}" scripts/train_diffusion.py \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --prior "${PRIOR}" \
    --epochs "${EPOCHS}"
