#!/bin/bash
#SBATCH --job-name=phase0-checks
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:08:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/phase0-checks/%x-%j.out
#SBATCH --error=logs/phase0-checks/%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# Phase 0 verification checks. Confirms Steps 1 + 2 of the Stage 0 sequence:
#   A. Val determinism — val_epoch produces identical loss across two runs.
#   B. Normalised CRPS — eval_diffusion JSON gains the new normalised keys
#                        and Variant2[h] = Variant1[h] / mean|y|[h] holds.
#
# Submit:
#   # Stage A only (no checkpoint needed)
#   sbatch --export=DATASET=metrla scripts/daic/submit_phase0_checks.sbatch
#
#   # Stages A + B (requires an existing trained checkpoint)
#   sbatch --export=DATASET=metrla,RUN_ID=dsm_structured_metrla_seed0_<hash> \
#          scripts/daic/submit_phase0_checks.sbatch

set -euo pipefail
mkdir -p logs/phase0-checks
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${DATASET:=metrla}"

scontrol update job="${SLURM_JOB_ID}" \
    JobName="phase0-checks-${DATASET}" 2>/dev/null || true

echo "Phase 0 verification: dataset=${DATASET}  run_id=${RUN_ID:-(skipped)}"

if [ -n "${RUN_ID:-}" ]; then
  srun --export=ALL "${DAIC_PYTHON}" scripts/phase0_checks.py \
      --dataset "${DATASET}" \
      --run-id "${RUN_ID}"
else
  srun --export=ALL "${DAIC_PYTHON}" scripts/phase0_checks.py \
      --dataset "${DATASET}"
fi
