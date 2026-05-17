#!/bin/bash
#SBATCH --job-name=dsm-eval
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/eval-diffusion/%x-%j.out
#SBATCH --error=logs/eval-diffusion/%x-%j.err
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
mkdir -p logs/eval-diffusion
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${RUN_ID:?Set RUN_ID=<checkpoint folder name> via --export}"
N_SAMPLES="${N_SAMPLES:-50}"
# n_steps=100 is the default. Empirical check showed CRPS within 0.06 mph of
# n_steps=200 — negligible at our scale, and halving steps halves wall time.
# Override per-job with --export=N_STEPS=200 if you need the tighter setting.
N_STEPS="${N_STEPS:-100}"
OUT_DIR="${OUT_DIR:-experiments/score-network}"
# Optional: only evaluate the first MAX_WINDOWS test windows. Used by the
# throughput-probe pattern to measure win/s before sizing real --time=.
MAX_WINDOWS="${MAX_WINDOWS:-}"
# batch_size for the eval DataLoader. Bigger B → fewer batches → less Python
# overhead per window. A40 has 46 GB; model is small so B=64 or 128 is fine.
# Optimal value per dataset comes from `submit_preflight.sbatch`; default 16
# matches the historical setting for backward-compatibility.
BATCH_SIZE="${BATCH_SIZE:-16}"

# Derive a readable job name from RUN_ID so squeue shows what's running.
# Pattern: dsm_<prior>_<dataset>_seed<S>_<hash>  →  eval-<dataset>-<prior>
PRIOR_SHORT=$(echo "${RUN_ID}" | awk -F_ '{print $2}')
DS_SHORT=$(echo "${RUN_ID}" | awk -F_ '{print $3}')
scontrol update job="${SLURM_JOB_ID}" JobName="eval-${DS_SHORT}-${PRIOR_SHORT}" 2>/dev/null || true

MAX_WIN_ARG=""
if [ -n "${MAX_WINDOWS}" ]; then
  MAX_WIN_ARG="--max_windows ${MAX_WINDOWS}"
  echo "Probe mode: --max_windows ${MAX_WINDOWS}"
fi

echo "Evaluating ${RUN_ID}  (K=${N_SAMPLES}, steps=${N_STEPS}, batch_size=${BATCH_SIZE})  →  ${OUT_DIR}"
# MAX_WIN_ARG is intentionally unquoted so word-splitting expands it correctly
# when set (and to nothing when empty). The string form works under `set -u`
# where empty bash arrays trip "unbound variable".
srun --export=ALL "${DAIC_PYTHON}" scripts/eval_diffusion.py \
    --run_id "${RUN_ID}" \
    --n_samples "${N_SAMPLES}" \
    --n_steps "${N_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --out_dir "${OUT_DIR}" \
    ${MAX_WIN_ARG}
