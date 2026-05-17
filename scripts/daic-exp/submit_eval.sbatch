#!/bin/bash
#SBATCH --job-name=dsm-eval
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/eval-diffusion/%x-%j.out
#SBATCH --error=logs/eval-diffusion/%x-%j.err
#
# Sample + compute CRPS / PIT / pinball for one trained checkpoint.
#
# Submit:
#   source scripts/daic-exp/env.sh
#   sbatch --export=RUN_ID=dsm_structured_metrla_seed0_<hash> \
#          scripts/daic-exp/submit_eval.sbatch

set -euo pipefail
mkdir -p logs/eval-diffusion
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic-exp/_common.sh"

: "${RUN_ID:?Set RUN_ID=<checkpoint folder name> via --export}"
N_SAMPLES="${N_SAMPLES:-50}"
N_STEPS="${N_STEPS:-200}"
OUT_DIR="${OUT_DIR:-experiments/score-network}"

PRIOR_SHORT=$(echo "${RUN_ID}" | awk -F_ '{print $2}')
DS_SHORT=$(echo "${RUN_ID}" | awk -F_ '{print $3}')
scontrol update job="${SLURM_JOB_ID}" JobName="eval-${DS_SHORT}-${PRIOR_SHORT}" 2>/dev/null || true

echo "Evaluating ${RUN_ID}  (K=${N_SAMPLES}, steps=${N_STEPS})  →  ${OUT_DIR}"
srun --export=ALL "${DAIC_PYTHON}" scripts/eval_diffusion.py \
    --run_id "${RUN_ID}" \
    --n_samples "${N_SAMPLES}" \
    --n_steps "${N_STEPS}" \
    --out_dir "${OUT_DIR}"
