#!/bin/bash
#SBATCH --job-name=gwn-train
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# Train the GWN backbone on one dataset / seed. Submit with:
#   sbatch --export=DATASET=metrla,SEED=0 scripts/daic/submit_train_backbone.sbatch
#
# Required env vars: DATASET (metrla|pemsbay), SEED (int).

set -euo pipefail
mkdir -p logs
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${DATASET:?Set DATASET=metrla or DATASET=pemsbay via --export}"
: "${SEED:?Set SEED=<int> via --export}"
EPOCHS="${EPOCHS:-100}"

echo "Training GWN on ${DATASET} (seed ${SEED}, ${EPOCHS} epochs)"
srun --export=ALL "${DAIC_PYTHON}" scripts/train_backbone.py \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}"
