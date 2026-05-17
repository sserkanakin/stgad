#!/bin/bash
#SBATCH --job-name=gwn-train
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/train-backbone/%x-%j.out
#SBATCH --error=logs/train-backbone/%x-%j.err
#
# Train the GWN backbone on one dataset / seed. Submit with:
#   source scripts/daic-exp/env.sh
#   sbatch --export=DATASET=metrla,SEED=0 scripts/daic-exp/submit_train_backbone.sbatch

set -euo pipefail
mkdir -p logs/train-backbone
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic-exp/_common.sh"

: "${DATASET:?Set DATASET=metrla or DATASET=pemsbay via --export}"
: "${SEED:?Set SEED=<int> via --export}"
EPOCHS="${EPOCHS:-100}"

scontrol update job="${SLURM_JOB_ID}" JobName="gwn-${DATASET}-s${SEED}" 2>/dev/null || true

echo "Training GWN on ${DATASET} (seed ${SEED}, ${EPOCHS} epochs)"
srun --export=ALL "${DAIC_PYTHON}" scripts/train_backbone.py \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}"
