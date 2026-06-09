#!/bin/bash
#SBATCH --job-name=eval_diffusion
#SBATCH --time=01:30:00
# partition / gres are passed on the sbatch command line (pin a supported GPU).
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/eval/%x-%j.out
#SBATCH --error=logs/eval/%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# evaluate trained diffusion checkpoints on the test split.
#   sbatch --partition=all --gres=gpu:nvidia_l40:1 scripts/daic/submit_eval.sbatch
# pass eval flags via EXTRA_FLAGS, e.g. EXTRA_FLAGS="--k 50 --n-steps 100".

set -euo pipefail
mkdir -p logs/eval
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export STAGE="eval"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

EXTRA_FLAGS="${EXTRA_FLAGS:-}"
log_run_header "${DAIC_PYTHON} scripts/eval_diffusion.py --all ${EXTRA_FLAGS}"

# shellcheck disable=SC2086
srun --export=ALL "${DAIC_PYTHON}" scripts/eval_diffusion.py --all ${EXTRA_FLAGS}
