# Common preamble for every sbatch script. Source this from inside the job:
#   source "${PROJECT_DIR}/scripts/daic/_common.sh"
#
# It loads modules, activates the project venv, cd's into the project dir,
# prints diagnostics, and prepares wandb env vars.

set -euo pipefail

# Allow sbatch scripts to set PROJECT_DIR (canonical), or fall back to
# walking up from this file. Sbatch runs with $PWD at submission, so we
# resolve relative to the script's own location instead.
if [ -z "${PROJECT_DIR:-}" ]; then
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
fi
export PROJECT_DIR

ENV_FILE="${PROJECT_DIR}/scripts/daic/env.sh"
if [ ! -f "${ENV_FILE}" ]; then
  echo "ERROR: ${ENV_FILE} not found. Copy env.example.sh → env.sh on DAIC." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${ENV_FILE}"

module use /opt/insy/modulefiles
module load "${DAIC_CUDA_MODULE}"
# cuDNN is bundled in PyTorch's cu121 wheels, so the system module isn't
# required. Load it only if env.sh pinned a version that exists.
if [ -n "${DAIC_CUDNN_MODULE:-}" ]; then
  module load "${DAIC_CUDNN_MODULE}" 2>/dev/null || echo \
    "  [warn] could not load ${DAIC_CUDNN_MODULE}; relying on PyTorch's bundled cuDNN"
fi
# shellcheck source=/dev/null
source "${DAIC_VENV}/bin/activate"

cd "${PROJECT_DIR}"

# Symlink large artefact directories on first run (idempotent).
mkdir -p data results
[ -e data/raw          ] || ln -sfn "${DAIC_DATA_DIR}/raw"          data/raw
[ -e data/processed    ] || ln -sfn "${DAIC_DATA_DIR}/processed"    data/processed
[ -e results/checkpoints  ] || ln -sfn "${DAIC_DATA_DIR}/checkpoints"  results/checkpoints
[ -e results/residuals    ] || ln -sfn "${DAIC_DATA_DIR}/residuals"    results/residuals
[ -e results/prior_params ] || ln -sfn "${DAIC_DATA_DIR}/prior_params" results/prior_params

# Diagnostics so the slurm log starts with reproducibility info.
echo "─── job env ─────────────────────────────────────────────────────"
echo "host:     $(hostname)"
echo "job:      ${SLURM_JOB_ID:-?}  ${SLURM_JOB_NAME:-?}"
echo "node:     ${SLURMD_NODENAME:-?}  cuda=${CUDA_VISIBLE_DEVICES:-?}"
echo "project:  ${PROJECT_DIR}"
echo "data:     ${DAIC_DATA_DIR}"
echo "python:   $(which python)  $(python --version 2>&1)"
echo "git:      $(cd "${PROJECT_DIR}" && git rev-parse --short HEAD 2>/dev/null || echo 'no-git')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
echo "─────────────────────────────────────────────────────────────────"
