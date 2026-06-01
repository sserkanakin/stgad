#!/bin/bash
#SBATCH --job-name=diff-eval
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/eval-samples/%x-%j.out
#SBATCH --error=logs/eval-samples/%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# Generate diffusion forecast samples, score them, and build the comparison
# tables + figures for one dataset / seed. Submit with:
#   sbatch --job-name="eval-metrla-s0" \
#          --export=DATASET=metrla,SEED=0,SPLIT=test \
#          scripts/daic/submit_generate_samples.sbatch
#
# Required env vars: DATASET (metrla|pemsbay|pems08f), SEED (int).
# Optional: SPLIT (default test), K (default 100), N_STEPS (default 100),
#           ETA (default 1.0), CKPT_DIR (default: auto-discover best.pt under
#           results/checkpoints/diffusion/diff_gwn_${DATASET}_seed${SEED}_*),
#           WHICH (best|last, default best), RUN_EVAL (1|0, default 1 — also run
#           the CPU evaluate/diagnostics/figures chain), EXTRA_FLAGS.
#
# NB: the log dir is created below; if submitting from a shell, this exists
# before the first run. Mirrors submit_train_diffusion.sbatch conventions.

set -euo pipefail
mkdir -p logs/eval-samples
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${DATASET:?Set DATASET=metrla|pemsbay|pems08f via --export}"
: "${SEED:?Set SEED=<int> via --export}"
SPLIT="${SPLIT:-test}"
K="${K:-100}"
N_STEPS="${N_STEPS:-100}"
ETA="${ETA:-1.0}"
WHICH="${WHICH:-best}"
RUN_EVAL="${RUN_EVAL:-1}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# Resolve the checkpoint dir. If CKPT_DIR is unset, pick the single run dir for
# this dataset/seed that contains a best.pt (fail loudly if 0 or >1 match).
if [ -z "${CKPT_DIR:-}" ]; then
  shopt -s nullglob
  matches=(results/checkpoints/diffusion/diff_gwn_"${DATASET}"_seed"${SEED}"_*)
  resolved=()
  for d in "${matches[@]}"; do
    [ -f "${d}/best.pt" ] && resolved+=("${d}")
  done
  if [ "${#resolved[@]}" -eq 0 ]; then
    echo "ERROR: no checkpoint dir with best.pt for ${DATASET} seed ${SEED}." >&2
    echo "       Train first, or pass CKPT_DIR=<dir> via --export." >&2
    exit 1
  fi
  if [ "${#resolved[@]}" -gt 1 ]; then
    echo "ERROR: multiple checkpoint dirs match; set CKPT_DIR explicitly:" >&2
    printf '       %s\n' "${resolved[@]}" >&2
    exit 1
  fi
  CKPT_DIR="${resolved[0]}"
fi

scontrol update job="${SLURM_JOB_ID}" JobName="eval-${DATASET}-s${SEED}" 2>/dev/null || true

echo "Generating samples for ${DATASET} (seed ${SEED}, split ${SPLIT})"
echo "  ckpt=${CKPT_DIR} (${WHICH})  K=${K}  n_steps=${N_STEPS}  eta=${ETA}"
echo "  extra flags: ${EXTRA_FLAGS}"

# shellcheck disable=SC2086
srun --export=ALL "${DAIC_PYTHON}" scripts/eval/generate_samples.py \
    --ckpt-dir "${CKPT_DIR}" \
    --which "${WHICH}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --n-samples "${K}" \
    --n-steps "${N_STEPS}" \
    --eta "${ETA}" \
    ${EXTRA_FLAGS}

if [ "${RUN_EVAL}" = "1" ]; then
  echo "Scoring baselines + assembling comparison tables"
  "${DAIC_PYTHON}" scripts/eval/evaluate_forecasts.py --dataset "${DATASET}" --split "${SPLIT}"
  echo "Running diffusion correctness diagnostics"
  "${DAIC_PYTHON}" scripts/eval/diffusion_diagnostics.py --dataset "${DATASET}" \
      --ckpt-dir "${CKPT_DIR}" --split val
  echo "Rendering figures"
  "${DAIC_PYTHON}" scripts/eval/make_figures.py --dataset "${DATASET}" --split "${SPLIT}"
fi

echo "Done. Results under results/eval/${DATASET}/"
