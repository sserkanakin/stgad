#!/bin/bash
# One-time DAIC environment bootstrap. Run from the login node, inside the
# project directory, AFTER copying env.example.sh → env.sh and editing it.
#
#   ssh daic
#   cd $DAIC_PROJECT_DIR     # wherever you cloned the repo
#   source scripts/daic/env.sh
#   bash scripts/daic/setup_env.sh
#
# Creates a venv under $DAIC_VENV, installs the project + CUDA torch wheels,
# and prints a `module load` snippet that every sbatch script reuses.

set -euo pipefail

if [ -z "${DAIC_VENV:-}" ]; then
  echo "ERROR: env.sh not sourced. Run: source scripts/daic/env.sh" >&2
  exit 1
fi

echo "[1/5] Loading modules ..."
module use /opt/insy/modulefiles
module load "${DAIC_CUDA_MODULE}"
# cuDNN is bundled in PyTorch's cu121 wheels; load only if version exists.
if [ -n "${DAIC_CUDNN_MODULE:-}" ]; then
  module load "${DAIC_CUDNN_MODULE}" 2>/dev/null || echo \
    "      [warn] ${DAIC_CUDNN_MODULE} not found; PyTorch ships its own cuDNN"
fi
module load miniconda/3
echo "      python3: $(which python3) ($(python3 --version))"

# DAIC's miniconda is just for the base python. We use a plain venv so the
# environment lives on the project FS (kinder to home quotas + portable
# across nodes).
if [ ! -d "${DAIC_VENV}" ]; then
  echo "[2/5] Creating venv at ${DAIC_VENV} ..."
  python3 -m venv "${DAIC_VENV}"
else
  echo "[2/5] venv already exists at ${DAIC_VENV} — reusing"
fi

# shellcheck source=/dev/null
source "${DAIC_VENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "[3/5] Installing PyTorch with CUDA 12.1 wheels ..."
# Pin a known-good torch; cu121 wheels match the cuda/12.1 module.
pip install --index-url https://download.pytorch.org/whl/cu121 \
    "torch>=2.2,<2.7"

echo "[4/5] Installing the project (editable) and its deps ..."
# pyproject.toml already lists torch; with --no-deps for torch via the
# previous step's wheel, `-e .` will pull the rest from PyPI.
pip install -e ".[dev]"

echo "[5/5] Sanity check ..."
python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device 0: {torch.cuda.get_device_name(0)}")
    print(f"  capability: {torch.cuda.get_device_capability(0)}")
PY

mkdir -p "${DAIC_DATA_DIR}"/{raw,processed,checkpoints,residuals,prior_params,wandb}
mkdir -p "${DAIC_PROJECT_DIR}/logs"

# Symlink artefact dirs into the project tree (idempotent).
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
echo "Project:   ${DAIC_PROJECT_DIR}"
echo "Data:      ${DAIC_DATA_DIR}"
echo "Venv:      ${DAIC_VENV}"
echo
echo "Next:  bash scripts/daic/verify_setup.sh         # pre-flight"
echo "Then:  sbatch scripts/daic/submit_smoke.sbatch   # 5-min GPU sanity"
