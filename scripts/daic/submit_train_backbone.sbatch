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
# the --time below (2h) is the full-run default. a quick smoke needs minutes, so
# request a short walltime to backfill into the queue much faster (cli --time
# overrides the directive):
#   sbatch --time=00:20:00 --job-name=backbone_metrla_seed0 \
#          --export=DATASET=metrla,SEED=0,EPOCHS=2 \
#          scripts/daic/submit_train_backbone.sbatch
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
