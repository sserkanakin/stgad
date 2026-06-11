#!/bin/bash
#SBATCH --job-name=ablate_sampling
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/ablation/%x-%j.out
#SBATCH --error=logs/ablation/%x-%j.err
#
# ground the reverse-sampler (n_steps, K) on one checkpoint via a subset sweep.
#   ABLATE_ARGS="--dataset metrla --prior kronecker --seed 42 --exp 20260612" \
#     sbatch --partition=all --gres=gpu:1 scripts/daic/submit_ablation.sbatch
set -euo pipefail
mkdir -p logs/ablation
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export STAGE="ablation"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

ABLATE_ARGS="${ABLATE_ARGS:?set ABLATE_ARGS}"
log_run_header "${DAIC_PYTHON} scripts/analysis/ablate_sampling.py ${ABLATE_ARGS}"
# shellcheck disable=SC2086
srun --unbuffered --export=ALL "${DAIC_PYTHON}" -u scripts/analysis/ablate_sampling.py ${ABLATE_ARGS}
