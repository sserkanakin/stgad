#!/bin/bash
#SBATCH --job-name=backbone
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/backbone/%x-%j.out
#SBATCH --error=logs/backbone/%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# train the gwn backbone on one dataset / seed.
#   sbatch --job-name=backbone_metrla_seed0 --export=DATASET=metrla,SEED=0 \
#          scripts/daic/submit_train_backbone.sbatch
#
# resources are sized for this small model: any gpu (gpu:1), 2 cpus, 8g, and a
# short 10-min walltime, so smokes backfill into the queue fast. cli flags
# override these directives, so for the full 100-epoch run give it more time:
#   sbatch --time=01:00:00 --job-name=backbone_metrla_seed0 \
#          --export=DATASET=metrla,SEED=0 scripts/daic/submit_train_backbone.sbatch
# pin a specific gpu only if you need to: --gres=gpu:a40:1
#
# --job-name makes the log file run-named; squeue shows the run name either way.

set -euo pipefail
mkdir -p logs/backbone
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${DATASET:?Set DATASET=metrla|pemsbay|pems08f via --export}"
: "${SEED:?Set SEED=<int> via --export}"
EPOCHS="${EPOCHS:-100}"

export STAGE="backbone"
RUN_NAME="backbone_${DATASET}_seed${SEED}"
scontrol update job="${SLURM_JOB_ID}" JobName="${RUN_NAME}" 2>/dev/null || true

log_run_header "${DAIC_PYTHON} scripts/train_backbone.py --dataset ${DATASET} --seed ${SEED} --epochs ${EPOCHS}"

srun --export=ALL "${DAIC_PYTHON}" scripts/train_backbone.py \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}"
