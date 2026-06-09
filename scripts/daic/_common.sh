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

# load modules + activate the project python, branching on the cluster.
# clusters.sh (sourced via env.sh) set DAIC_PYTHON_KIND + the module names.
if [ "${DAIC_PYTHON_KIND}" = "conda" ]; then
  # stable DAIC: insy modulefiles + a conda env on NFS home.
  module use "${DAIC_MODULE_USE}"
  module load "${DAIC_CUDA_MODULE}"
  if [ -n "${DAIC_CUDNN_MODULE:-}" ]; then
    module load "${DAIC_CUDNN_MODULE}" 2>/dev/null || echo \
      "  [warn] could not load ${DAIC_CUDNN_MODULE}; relying on PyTorch's bundled cuDNN"
  fi
  module load "${DAIC_CONDA_MODULE}"
  # conda.sh references $PS1, which is unset under `set -u` in sbatch jobs.
  : "${PS1:=}"
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${DAIC_CONDA_PREFIX}"
else
  # experimental DAIC: 2025 Lmod tree (python 3.11 + cuda torch) + a venv
  # overlay for our pure-python deps. the venv is built with
  # --system-site-packages so it reuses the module's torch.
  module load "${DAIC_MODULE_TREE}"
  module load "${DAIC_PYTHON_MODULE}"
  module load "${DAIC_TORCH_MODULE}"
  [ -n "${DAIC_CUDNN_MODULE:-}" ] && module load "${DAIC_CUDNN_MODULE}" 2>/dev/null || true
  if [ ! -x "${DAIC_VENV_DIR}/bin/python" ]; then
    echo "ERROR: venv not found at ${DAIC_VENV_DIR} — run setup_env.sh first." >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${DAIC_VENV_DIR}/bin/activate"
  # the 2025 py-* modules put their site-packages on PYTHONPATH, which would
  # otherwise shadow the newer versions we pip-installed into the venv (e.g.
  # wandb needs a newer typing_extensions than the module ships). put the venv
  # site-packages first so our deps win; module-only packages (torch) still
  # resolve further down PYTHONPATH.
  export PYTHONPATH="${DAIC_VENV_DIR}/lib/python${DAIC_PYTHON_VERSION}/site-packages:${PYTHONPATH:-}"
fi

# Absolute python path. srun on DAIC's Slurm strips PATH, so `srun python`
# fails with "No such file or directory". sbatch scripts use $DAIC_PYTHON
# explicitly instead.
export DAIC_PYTHON="$(which python)"

# Pip cache stays on the data dir (large; /winhome handles read-mostly).
export PIP_CACHE_DIR="${DAIC_DATA_DIR}/.cache/pip"

# Unbuffered Python stdout so progress prints appear in the slurm log in
# real time instead of waiting for the 4 KB block buffer to fill.
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"

# symlink the big artefact dirs to the data dir on first run (idempotent).
# create the targets first so the links aren't dangling.
mkdir -p data results
mkdir -p "${DAIC_DATA_DIR}"/{raw,processed,backbone,diffusion,residuals,eval}
[ -e data/raw          ] || ln -sfn "${DAIC_DATA_DIR}/raw"          data/raw
[ -e data/processed    ] || ln -sfn "${DAIC_DATA_DIR}/processed"    data/processed
[ -e results/backbone  ] || ln -sfn "${DAIC_DATA_DIR}/backbone"     results/backbone
[ -e results/diffusion ] || ln -sfn "${DAIC_DATA_DIR}/diffusion"    results/diffusion
[ -e results/residuals ] || ln -sfn "${DAIC_DATA_DIR}/residuals"    results/residuals
[ -e results/eval      ] || ln -sfn "${DAIC_DATA_DIR}/eval"         results/eval

# wandb writes its run buffer here; make sure it exists.
mkdir -p "${WANDB_DIR:-${PROJECT_DIR}/wandb}"

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


# print a run-identity block. an sbatch calls this after STAGE / DATASET / SEED /
# PRIOR / RUN_NAME are set, passing the command it will run. makes every slurm
# log self-identifying: which stage, dataset, seed, prior, wandb group, command.
log_run_header() {
  local group="${STAGE:-?}-${DATASET:-?}"
  [ -n "${PRIOR:-}" ] && group="${group}-${PRIOR}"
  echo "─── run ─────────────────────────────────────────────────────────"
  echo "stage:    ${STAGE:-?}"
  echo "dataset:  ${DATASET:-?}    seed: ${SEED:-?}    prior: ${PRIOR:-n/a}"
  echo "run name: ${RUN_NAME:-?}"
  echo "wandb:    project=thesis-diffusion  group=${group}  name=${RUN_NAME:-?}"
  echo "command:  $*"
  echo "─────────────────────────────────────────────────────────────────"
}
