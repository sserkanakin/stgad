#!/bin/bash
#SBATCH --job-name=exp003-dsm
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# exp-003 score-network training. PLACEHOLDER — script
# scripts/train_diffusion.py is written in step §14.4 of the PLAN.
#
# Submit one variant:
#   sbatch --export=DATASET=metrla,SEED=0,PRIOR=structured scripts/daic/submit_diffusion.sbatch
#
# Or sweep with a job array (5 seeds × 2 priors × 2 datasets = 20 jobs):
#   for ds in metrla pemsbay; do
#     for prior in structured isotropic; do
#       for seed in 0 1 2 3 4; do
#         sbatch --export=DATASET=$ds,SEED=$seed,PRIOR=$prior \
#                scripts/daic/submit_diffusion.sbatch
#       done
#     done
#   done

set -euo pipefail
mkdir -p logs
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${DATASET:?Set DATASET=metrla or DATASET=pemsbay}"
: "${SEED:?Set SEED=<int>}"
: "${PRIOR:=structured}"   # structured | isotropic
EPOCHS="${EPOCHS:-100}"

echo "Training score net: dataset=${DATASET} seed=${SEED} prior=${PRIOR} epochs=${EPOCHS}"

if [ ! -f scripts/train_diffusion.py ]; then
  echo "ERROR: scripts/train_diffusion.py not present yet — exp-003 step §14.4 pending." >&2
  exit 1
fi

srun "${DAIC_PYTHON}" scripts/train_diffusion.py \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --prior "${PRIOR}" \
    --epochs "${EPOCHS}"
