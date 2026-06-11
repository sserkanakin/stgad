#!/bin/bash
#SBATCH --job-name=eval_diffusion
# 10h cap: the point phase samples k draws over ALL test windows; pemsbay (~10k
# windows, N=325) needs ~8h, metrla/pems08f finish well under 6h. resume_pipeline
# passes a per-dataset --time that overrides this (pemsbay 10h, others 6h); the
# 10h default here just keeps a bare --all run (which includes pemsbay) safe.
#SBATCH --time=10:00:00
# partition / gres are passed on the sbatch command line (pin a supported GPU).
#SBATCH --cpus-per-task=4
# 64G host ram to match the larger gen-batch (4096) staged per reverse_sample.
#SBATCH --mem=64G
#SBATCH --output=logs/eval/%x-%j.out
#SBATCH --error=logs/eval/%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# evaluate trained diffusion checkpoints on the test split.
#   # all 6 sequentially in one job:
#   sbatch --partition=all --gres=gpu:nvidia_a40:1 scripts/daic/submit_eval.sbatch
#   # one checkpoint per job (fan out 6 in parallel):
#   EVAL_ARGS="--dataset metrla --prior kronecker" sbatch ... scripts/daic/submit_eval.sbatch
# EVAL_ARGS defaults to --all; add extra eval flags (e.g. --k 50) right in it.

set -euo pipefail
mkdir -p logs/eval
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export STAGE="eval"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

EVAL_ARGS="${EVAL_ARGS:---all}"

# if ABL_JSON is set, read the grounded (n_steps, K) from the ablation at RUNTIME
# and append them (argparse: last value wins, so this overrides any earlier flag).
# lets us submit the evals with a dependency on the ablation, before it finishes;
# pick_knee falls back to sane defaults if the json is missing/partial.
if [ -n "${ABL_JSON:-}" ]; then
  KNEE="$("${DAIC_PYTHON}" "${PROJECT_DIR}/scripts/analysis/pick_knee.py" "${ABL_JSON}" 2>/dev/null || echo "100 30")"
  set -- ${KNEE:-100 30}; _NS="${1:-100}"; _K="${2:-30}"
  EVAL_ARGS="${EVAL_ARGS} --n-steps ${_NS} --k ${_K}"
  echo "[eval] ablation knee from ${ABL_JSON}: n_steps=${_NS} k=${_K}" >&2
fi

log_run_header "${DAIC_PYTHON} scripts/eval_diffusion.py ${EVAL_ARGS}"

# --unbuffered + python -u so the per-chunk progress is written to the log live
# (otherwise srun buffers job stdout and the log looks frozen mid-run).
# shellcheck disable=SC2086
srun --unbuffered --export=ALL "${DAIC_PYTHON}" -u scripts/eval_diffusion.py ${EVAL_ARGS}
