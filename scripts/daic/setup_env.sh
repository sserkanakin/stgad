#!/bin/bash
# One-time DAIC environment bootstrap. Run from the login node, inside the
# project directory, AFTER copying env.example.sh → env.sh and editing it.
#
#   ssh daic           # or: ssh daic-exp
#   cd <project dir>
#   source scripts/daic/env.sh
#   bash scripts/daic/setup_env.sh
#
# Builds the project's python environment for the CURRENT cluster (clusters.sh
# decides which), installs the project + deps, and symlinks the data dir tree
# into the project. Idempotent: re-run to repair or update.
#
#   stable DAIC -> conda env on NFS home (miniconda module ships py3.9, too old,
#                  so we create a py3.11 conda env and pip-install cu121 torch).
#   exp DAIC    -> the 2025 Lmod tree already provides py3.11 + a cuda torch, so
#                  we make a venv (--system-site-packages) over it and pip-install
#                  only our remaining pure-python deps. no conda involved.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ALWAYS re-source env.sh (not just when DAIC_CLUSTER is unset): a shell that
# sourced an OLDER env.sh keeps stale DAIC_* vars, so newly-added config (e.g.
# DAIC_PIP_EXTRAS) would be silently ignored and setup would use wrong defaults.
# shellcheck source=/dev/null
source "${HERE}/env.sh"

PYVER="${DAIC_PYTHON_VERSION:-3.11}"

# ── stable DAIC: conda ──────────────────────────────────────────────────────
setup_conda() {
  echo "[1/6] Loading modules (stable DAIC) ..."
  module use "${DAIC_MODULE_USE}"
  module load "${DAIC_CUDA_MODULE}"
  if [ -n "${DAIC_CUDNN_MODULE:-}" ]; then
    module load "${DAIC_CUDNN_MODULE}" 2>/dev/null || echo \
      "      [warn] ${DAIC_CUDNN_MODULE} not found; PyTorch ships its own cuDNN"
  fi
  module load "${DAIC_CONDA_MODULE}"
  : "${PS1:=}"
  source "$(conda info --base)/etc/profile.d/conda.sh"

  # storage split: env on NFS home (POSIX, pip renames), caches on data dir.
  mkdir -p "${HOME}/.conda/envs" "${DAIC_DATA_DIR}/.conda/pkgs" "${DAIC_DATA_DIR}/.cache/pip"
  conda config --prepend pkgs_dirs "${DAIC_DATA_DIR}/.conda/pkgs" 2>/dev/null || true
  export PIP_CACHE_DIR="${DAIC_DATA_DIR}/.cache/pip"

  echo "[2/6] Creating conda env at ${DAIC_CONDA_PREFIX} (python ${PYVER}) ..."
  if [ -x "${DAIC_CONDA_PREFIX}/bin/python" ]; then
    echo "      env already exists — reusing"
  else
    conda create --prefix "${DAIC_CONDA_PREFIX}" "python=${PYVER}" pip -y
  fi
  conda activate "${DAIC_CONDA_PREFIX}"
  python -m pip install --upgrade pip setuptools wheel

  echo "[3/6] C-heavy deps from conda-forge (h5py) ..."
  conda install -c conda-forge -y h5py

  echo "[4/6] PyTorch (CUDA 12.1 wheels) ..."
  pip install --prefer-binary --index-url https://download.pytorch.org/whl/cu121 "torch>=2.2,<2.7"

  echo "[5/6] Project (editable) + remaining deps ..."
  pip install --prefer-binary -e ".${DAIC_PIP_EXTRAS-[dev]}"
}

# ── experimental DAIC: module tree + venv overlay ───────────────────────────
setup_venv() {
  echo "[1/4] Loading modules (experimental DAIC) ..."
  module load "${DAIC_MODULE_TREE}"
  module load "${DAIC_PYTHON_MODULE}"
  # torch: either the module (built for sm_80/sm_90 only) or our own cu128 wheel
  # (>=2.7, adds sm_120 / Blackwell). when pip-cu128 we deliberately do NOT load
  # py-torch so its bundled cuda libs don't clash with the wheel's — see clusters.sh.
  if [ "${DAIC_TORCH_KIND:-module}" = "module" ]; then
    module load "${DAIC_TORCH_MODULE}"   # provides cuda torch + numpy
  fi
  [ -n "${DAIC_CUDNN_MODULE:-}" ] && module load "${DAIC_CUDNN_MODULE}" 2>/dev/null || true

  export PIP_CACHE_DIR="${DAIC_DATA_DIR}/.cache/pip"
  mkdir -p "${PIP_CACHE_DIR}"

  echo "[2/4] Creating venv (--system-site-packages) at ${DAIC_VENV_DIR} ..."
  if [ -x "${DAIC_VENV_DIR}/bin/python" ]; then
    echo "      venv already exists — reusing"
  else
    python3 -m venv --system-site-packages "${DAIC_VENV_DIR}"
  fi
  # shellcheck source=/dev/null
  source "${DAIC_VENV_DIR}/bin/activate"
  # venv deps must win over the module py-* packages on PYTHONPATH.
  export PYTHONPATH="${DAIC_VENV_DIR}/lib/python${DAIC_PYTHON_VERSION}/site-packages:${PYTHONPATH:-}"
  python -m pip install --upgrade pip setuptools wheel

  if [ "${DAIC_TORCH_KIND:-module}" = "pip-cu128" ]; then
    echo "[3/4] PyTorch (${DAIC_TORCH_SPEC}, cu128 wheel — Blackwell/sm_120 capable) + project ..."
    # install torch from the cu128 index INTO the venv so it overrides any module
    # torch on PYTHONPATH. the wheel pulls its own nvidia-* cuda libs + numpy.
    pip install --prefer-binary --index-url "${DAIC_TORCH_PIP_INDEX}" "${DAIC_TORCH_SPEC}"
    # remaining pure-python deps from PyPI (torch already satisfied above).
    pip install --prefer-binary -e ".${DAIC_PIP_EXTRAS-[dev]}"
  else
    echo "[3/4] Project (editable) + remaining deps ..."
    # torch/numpy come from the module via --system-site-packages, so pip sees
    # them satisfied and installs only the rest (wandb, easydict, einops, ...).
    pip install --prefer-binary -e ".${DAIC_PIP_EXTRAS-[dev]}"
  fi
}

if [ "${DAIC_PYTHON_KIND}" = "conda" ]; then setup_conda; STEP="[6/6]"; else setup_venv; STEP="[4/4]"; fi

echo "${STEP} Sanity check ..."
python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device 0: {torch.cuda.get_device_name(0)}")
    print(f"  capability: {torch.cuda.get_device_capability(0)}")
PY

# artefact directories + symlinks (match _common.sh).
mkdir -p "${DAIC_DATA_DIR}"/{raw,processed,backbone,diffusion,residuals,eval,wandb,logs}
cd "${DAIC_PROJECT_DIR}"
mkdir -p data results
ln -sfn "${DAIC_DATA_DIR}/raw"        data/raw
ln -sfn "${DAIC_DATA_DIR}/processed"  data/processed
ln -sfn "${DAIC_DATA_DIR}/backbone"   results/backbone
ln -sfn "${DAIC_DATA_DIR}/diffusion"  results/diffusion
ln -sfn "${DAIC_DATA_DIR}/residuals"  results/residuals
ln -sfn "${DAIC_DATA_DIR}/eval"       results/eval
# logs live on the data share too, so slurm .out/.err don't eat the tight
# linuxhome quota that now holds the cu128 torch venv. only replace logs/ with a
# symlink when it's safe (a symlink, missing, or empty) — never clobber real logs.
if [ -L logs ] || [ ! -e logs ] || [ -z "$(ls -A logs 2>/dev/null)" ]; then
  rm -rf logs 2>/dev/null || true
  ln -sfn "${DAIC_DATA_DIR}/logs" logs
else
  echo "      [warn] logs/ is a non-empty real dir — leaving it on linuxhome"
fi

echo
echo "[bonus] wandb login (one-time; only needed for live dashboards):"
if python -c "import wandb, sys; sys.exit(0 if wandb.api.api_key else 1)" 2>/dev/null; then
  echo "      already logged in — skipping"
else
  echo "      Run \`wandb login\` and paste your key from https://wandb.ai/authorize"
fi

echo
echo "Setup complete."
echo "Cluster:  ${DAIC_CLUSTER}"
echo "Project:  ${DAIC_PROJECT_DIR}"
echo "Data:     ${DAIC_DATA_DIR}"
if [ "${DAIC_PYTHON_KIND}" = "conda" ]; then
  echo "Env:      ${DAIC_CONDA_PREFIX}  (conda, python ${PYVER})"
elif [ "${DAIC_TORCH_KIND:-module}" = "pip-cu128" ]; then
  echo "Env:      ${DAIC_VENV_DIR}  (venv + pip ${DAIC_TORCH_SPEC} cu128)"
else
  echo "Env:      ${DAIC_VENV_DIR}  (venv over ${DAIC_TORCH_MODULE})"
fi
echo
echo "Next:  bash scripts/daic/verify_setup.sh          # pre-flight"
echo "Then:  bash scripts/daic/submit.sh smoke metrla   # 5-min GPU sanity"
