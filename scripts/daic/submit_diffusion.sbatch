#!/bin/bash
#SBATCH --job-name=dsm-train
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=logs/train-diffusion/%x-%j.out
#SBATCH --error=logs/train-diffusion/%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# score-network score-network training. PLACEHOLDER — script
# scripts/train_diffusion.py is written in step §14.4 of the PLAN.
#
# Submit one variant:
#   sbatch --export=DATASET=metrla,SEED=0,PRIOR=structured scripts/daic/submit_diffusion.sbatch
#
# Or sweep with a job array (5 seeds × 2 priors × 3 datasets = 30 jobs):
#   for ds in metrla pemsbay pems08f; do
#     for prior in structured isotropic; do
#       for seed in 0 1 2 3 4; do
#         sbatch --export=DATASET=$ds,SEED=$seed,PRIOR=$prior \
#                scripts/daic/submit_diffusion.sbatch
#       done
#     done
#   done

set -euo pipefail
mkdir -p logs/train-diffusion
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${DATASET:?Set DATASET=metrla|pemsbay|pems08f}"
: "${SEED:?Set SEED=<int>}"
: "${PRIOR:=structured}"   # structured | isotropic
EPOCHS="${EPOCHS:-100}"

# Rename for readability in squeue.
scontrol update job="${SLURM_JOB_ID}" JobName="dsm-${DATASET}-${PRIOR}-s${SEED}" 2>/dev/null || true

echo "Training score net: dataset=${DATASET} seed=${SEED} prior=${PRIOR} epochs=${EPOCHS}"

if [ ! -f scripts/train_diffusion.py ]; then
  echo "ERROR: scripts/train_diffusion.py not present yet — score-network step §14.4 pending." >&2
  exit 1
fi

srun --export=ALL "${DAIC_PYTHON}" scripts/train_diffusion.py \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --prior "${PRIOR}" \
    --epochs "${EPOCHS}"
