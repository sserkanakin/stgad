#!/bin/bash
# One-time DAIC environment bootstrap. Run from the login node, inside the
# project directory, AFTER copying env.example.sh → env.sh and editing it.
#
#   ssh daic
#   cd $DAIC_PROJECT_DIR
#   source scripts/daic/env.sh
#   bash scripts/daic/setup_env.sh
#
# Creates a Python 3.11 conda env named ${DAIC_CONDA_ENV} (default
# "thesis-diffusion"), installs the project + CUDA torch wheels, and
# symlinks the data directory tree into the project. Idempotent: re-run
# to repair or update.
#
# Why conda instead of venv: DAIC's miniconda module ships Python 3.9,
# but newer versions of pandas, wandb, h5py, matplotlib etc. have dropped
# 3.9 wheels and try to compile from source — which fails on DAIC's
# GCC 4.8. Using `conda create -n … python=3.11` side-steps all of this.

set -euo pipefail

if [ -z "${DAIC_CONDA_ENV:-}" ]; then
  echo "ERROR: env.sh not sourced or out of date (DAIC_CONDA_ENV unset)." >&2
  echo "       cp scripts/daic/env.example.sh scripts/daic/env.sh and edit," >&2
  echo "       then source it." >&2
  exit 1
fi

PYVER="${DAIC_PYTHON_VERSION:-3.11}"

echo "[1/5] Loading modules ..."
module use /opt/insy/modulefiles
module load "${DAIC_CUDA_MODULE}"
# cuDNN is bundled in PyTorch's cu121 wheels; load only if version exists.
if [ -n "${DAIC_CUDNN_MODULE:-}" ]; then
  module load "${DAIC_CUDNN_MODULE}" 2>/dev/null || echo \
    "      [warn] ${DAIC_CUDNN_MODULE} not found; PyTorch ships its own cuDNN"
fi
module load miniconda/3

# Initialise conda for `conda activate` inside this script. Without this,
# only `conda` commands work, not the env-activation machinery.
source "$(conda info --base)/etc/profile.d/conda.sh"

# Storage split (DAIC: 8 GB Linux home quota, plenty of room on /winhome
# or staff-umbrella):
#   - conda ENV itself: $HOME/.conda/envs/<name> (NFS, full POSIX —
#     pip's rename operations fail on /winhome's SMB-backed mount).
#   - conda pkg cache  : ${DAIC_DATA_DIR}/.conda/pkgs (read-mostly, fine
#     on /winhome).
#   - pip wheel cache  : ${DAIC_DATA_DIR}/.cache/pip (same).
# Net home usage after install: ~3.7 GB env. Fits 8 GB quota.
DAIC_CONDA_PATH="${HOME}/.conda/envs/${DAIC_CONDA_ENV}"
mkdir -p "${HOME}/.conda/envs" \
         "${DAIC_DATA_DIR}/.conda/pkgs" \
         "${DAIC_DATA_DIR}/.cache/pip"
# Pkg cache on data dir (idempotent; ignores if already in .condarc).
conda config --prepend pkgs_dirs "${DAIC_DATA_DIR}/.conda/pkgs" 2>/dev/null || true
# Clean any prior envs_dirs entry pointing at /winhome from earlier runs.
conda config --remove envs_dirs "${DAIC_DATA_DIR}/.conda/envs" 2>/dev/null || true
export PIP_CACHE_DIR="${DAIC_DATA_DIR}/.cache/pip"

echo "[2/5] Creating conda env at ${DAIC_CONDA_PATH} (python ${PYVER}) ..."
echo "      pkgs cache: ${DAIC_DATA_DIR}/.conda/pkgs"
echo "      pip cache:  ${PIP_CACHE_DIR}"
if [ -d "${DAIC_CONDA_PATH}" ] && [ -x "${DAIC_CONDA_PATH}/bin/python" ]; then
  echo "      env already exists — reusing"
else
  conda create --prefix "${DAIC_CONDA_PATH}" "python=${PYVER}" pip -y
fi

conda activate "${DAIC_CONDA_PATH}"
echo "      python: $(which python) ($(python --version))"
python -m pip install --upgrade pip setuptools wheel

echo "[3/5] Installing PyTorch with CUDA 12.1 wheels ..."
pip install --prefer-binary --index-url https://download.pytorch.org/whl/cu121 \
    "torch>=2.2,<2.7"

echo "[4/5] Installing the project (editable) and its deps ..."
# --prefer-binary: pick the latest version *with a wheel for this platform*
# rather than the absolute latest. DAIC's compute nodes are CentOS 7
# (glibc 2.17, GCC 4.8.5); recent numpy/pandas/scipy releases have moved
# to manylinux_2_28 wheels (glibc 2.28+), and without --prefer-binary
# pip falls back to source builds and fails on the old toolchain.
pip install --prefer-binary -e ".[dev]"

echo "[5/5] Sanity check ..."
python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device 0: {torch.cuda.get_device_name(0)}")
    print(f"  capability: {torch.cuda.get_device_capability(0)}")
PY

# Set up the artefact directories and symlinks.
mkdir -p "${DAIC_DATA_DIR}"/{raw,processed,checkpoints,residuals,prior_params,wandb}
mkdir -p "${DAIC_PROJECT_DIR}/logs"

cd "${DAIC_PROJECT_DIR}"
mkdir -p data results
ln -sfn "${DAIC_DATA_DIR}/raw"          data/raw
ln -sfn "${DAIC_DATA_DIR}/processed"    data/processed
ln -sfn "${DAIC_DATA_DIR}/checkpoints"  results/checkpoints
ln -sfn "${DAIC_DATA_DIR}/residuals"    results/residuals
ln -sfn "${DAIC_DATA_DIR}/prior_params" results/prior_params

echo
echo "[bonus] wandb login (one-time; only needed for live dashboards):"
if python -c "import wandb, sys; sys.exit(0 if wandb.api.api_key else 1)" 2>/dev/null; then
  echo "      already logged in — skipping"
else
  echo "      Run \`wandb login\` now and paste your API key from"
  echo "      https://wandb.ai/authorize  (skip if WANDB_MODE=offline)."
fi

echo
echo "Setup complete."
echo "Project:    ${DAIC_PROJECT_DIR}"
echo "Data:       ${DAIC_DATA_DIR}"
echo "Conda env:  ${DAIC_CONDA_PATH}  (python ${PYVER})"
echo
echo "To use this env interactively on the login node:"
echo "    module load miniconda/3 && conda activate ${DAIC_CONDA_PATH}"
echo
echo "Next:  bash scripts/daic/verify_setup.sh         # pre-flight"
echo "Then:  sbatch scripts/daic/submit_smoke.sbatch   # 5-min GPU sanity"
