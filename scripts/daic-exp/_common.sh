# Common preamble for every sbatch script on experimental DAIC. Source from
# inside the job:
#   source "${PROJECT_DIR}/scripts/daic-exp/_common.sh"
#
# It loads modules, activates the project conda env, cd's into the project
# dir, prints diagnostics, and prepares wandb env vars.

set -euo pipefail

# Resolve PROJECT_DIR. Sbatch runs with $PWD at submission, but we want this
# to also work when sourced from a script that was cwd'd elsewhere.
if [ -z "${PROJECT_DIR:-}" ]; then
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
fi
export PROJECT_DIR

ENV_FILE="${PROJECT_DIR}/scripts/daic-exp/env.sh"
if [ ! -f "${ENV_FILE}" ]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  echo "       Copy env.example.sh → env.sh on the cluster, edit, source." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${ENV_FILE}"

# Sanity-check that the user actually filled the placeholders.
if [ "${DAIC_EXP_ACCOUNT:-CHANGE_ME}" = "CHANGE_ME" ]; then
  echo "ERROR: DAIC_EXP_ACCOUNT is still CHANGE_ME. Edit env.sh." >&2
  echo "       Find your account with:" >&2
  echo "         sacctmgr show user \$USER withassoc format=account%30" >&2
  exit 1
fi

# ── modules ────────────────────────────────────────────────────────────────
# Best practice from the DAIC docs: start with a clean environment.
module purge 2>/dev/null || true
module load "${DAIC_EXP_BASE_MODULE}"
module load "${DAIC_EXP_CUDA_MODULE}"
# cuDNN is bundled in PyTorch's cu121 wheels; the system module is optional.
if [ -n "${DAIC_EXP_CUDNN_MODULE:-}" ]; then
  module load "${DAIC_EXP_CUDNN_MODULE}" 2>/dev/null || echo \
    "  [warn] could not load ${DAIC_EXP_CUDNN_MODULE}; relying on PyTorch's bundled cuDNN"
fi

# ── conda ──────────────────────────────────────────────────────────────────
# Experimental DAIC has no miniconda module — setup_env.sh installed
# Miniforge into ${DAIC_EXP_MINIFORGE_ROOT}. Activate from there.
if [ ! -x "${DAIC_EXP_MINIFORGE_ROOT}/bin/conda" ]; then
  echo "ERROR: Miniforge not found at ${DAIC_EXP_MINIFORGE_ROOT}/bin/conda." >&2
  echo "       Run scripts/daic-exp/setup_env.sh on the login node first." >&2
  exit 1
fi

# conda.sh references $PS1, which is unset under `set -u` in sbatch jobs.
: "${PS1:=}"
# shellcheck source=/dev/null
source "${DAIC_EXP_MINIFORGE_ROOT}/etc/profile.d/conda.sh"

CONDA_ENV_PATH="${DAIC_EXP_MINIFORGE_ROOT}/envs/${DAIC_EXP_CONDA_ENV}"
if [ ! -x "${CONDA_ENV_PATH}/bin/python" ]; then
  echo "ERROR: conda env not found at ${CONDA_ENV_PATH}." >&2
  echo "       Run scripts/daic-exp/setup_env.sh on the login node." >&2
  exit 1
fi
conda activate "${CONDA_ENV_PATH}"

# srun on some Slurm configs strips PATH; use absolute python in sbatch.
export DAIC_PYTHON="$(which python)"

# Caches under the data dir (large; cluster home is 5 MB).
export PIP_CACHE_DIR="${DAIC_EXP_DATA_DIR}/.cache/pip"
export WANDB_CACHE_DIR="${DAIC_EXP_DATA_DIR}/.cache/wandb"

# Unbuffered Python stdout so progress prints appear in the slurm log in
# real time instead of waiting for the 4 KB block buffer to fill.
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"

# ── artefact symlinks ──────────────────────────────────────────────────────
# Same shape as scripts/daic/_common.sh: keep data/ and results/{ckpt,resid}
# pointed at the large-storage tier, while keeping results/prior_params in
# git (small YAMLs).
mkdir -p data results
[ -e data/raw          ] || ln -sfn "${DAIC_EXP_DATA_DIR}/raw"          data/raw
[ -e data/processed    ] || ln -sfn "${DAIC_EXP_DATA_DIR}/processed"    data/processed
[ -e results/checkpoints  ] || ln -sfn "${DAIC_EXP_DATA_DIR}/checkpoints"  results/checkpoints
[ -e results/residuals    ] || ln -sfn "${DAIC_EXP_DATA_DIR}/residuals"    results/residuals

# ── diagnostics ────────────────────────────────────────────────────────────
echo "─── job env (experimental DAIC) ─────────────────────────────────"
echo "host:     $(hostname)"
echo "job:      ${SLURM_JOB_ID:-?}  ${SLURM_JOB_NAME:-?}"
echo "account:  ${DAIC_EXP_ACCOUNT}"
echo "node:     ${SLURMD_NODENAME:-?}  cuda=${CUDA_VISIBLE_DEVICES:-?}"
echo "project:  ${PROJECT_DIR}"
echo "data:     ${DAIC_EXP_DATA_DIR}"
echo "conda:    ${CONDA_ENV_PATH}"
echo "python:   $(which python)  $(python --version 2>&1)"
echo "git:      $(cd "${PROJECT_DIR}" && git rev-parse --short HEAD 2>/dev/null || echo 'no-git')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
echo "─────────────────────────────────────────────────────────────────"
