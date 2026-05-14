#!/bin/bash
#SBATCH --job-name=dsm-eval
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# Sample + compute CRPS / PIT / pinball for one trained checkpoint.
#
# Submit:
#   sbatch --export=RUN_ID=dsm_structured_metrla_seed0_f07d4ad3 \
#          scripts/daic/submit_eval.sbatch
#
# Or the four-checkpoint sweep:
#   for run in dsm_structured_metrla_seed0_f07d4ad3 \
#              dsm_isotropic_metrla_seed0_e2c03b5f \
#              dsm_structured_pemsbay_seed0_09034d3d \
#              dsm_isotropic_pemsbay_seed0_a15ca06b; do
#     sbatch --export=RUN_ID=$run scripts/daic/submit_eval.sbatch
#   done

set -euo pipefail
mkdir -p logs
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${RUN_ID:?Set RUN_ID=<checkpoint folder name> via --export}"
N_SAMPLES="${N_SAMPLES:-50}"
N_STEPS="${N_STEPS:-200}"

echo "Evaluating ${RUN_ID}  (K=${N_SAMPLES}, steps=${N_STEPS})"
srun --export=ALL "${DAIC_PYTHON}" scripts/eval_diffusion.py \
    --run_id "${RUN_ID}" \
    --n_samples "${N_SAMPLES}" \
    --n_steps "${N_STEPS}"
