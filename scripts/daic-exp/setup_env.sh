#!/bin/bash
# One-time experimental-DAIC bootstrap. Run from the login node, inside the
# project directory, AFTER copying env.example.sh → env.sh and editing it.
#
#   ssh <netid>@daic01.hpc.tudelft.nl     # kinit triggered automatically
#   cd ~/linuxhome/research-tud
#   cp scripts/daic-exp/env.example.sh scripts/daic-exp/env.sh
#   $EDITOR scripts/daic-exp/env.sh       # set NetID + Slurm account
#   source scripts/daic-exp/env.sh
#   bash scripts/daic-exp/setup_env.sh
#
# Differences vs scripts/daic/setup_env.sh:
#   - Installs Miniforge3 because experimental DAIC has no miniconda module.
#   - Uses module load 2025/gpu + cuda/12.9 (no /opt/insy/modulefiles).
#   - All paths default to ~/linuxhome (30 GB) — /trinity/home is 5 MB.
#
# Idempotent: re-run to repair or update.

set -euo pipefail

# ── pre-flight ─────────────────────────────────────────────────────────────
for v in DAIC_EXP_NETID DAIC_EXP_ACCOUNT DAIC_EXP_PROJECT_DIR \
         DAIC_EXP_DATA_DIR DAIC_EXP_CONDA_ENV DAIC_EXP_MINIFORGE_ROOT \
         DAIC_EXP_BASE_MODULE DAIC_EXP_CUDA_MODULE; do
  if [ -z "${!v:-}" ] || [ "${!v:-}" = "CHANGE_ME" ]; then
    echo "ERROR: ${v} is unset or CHANGE_ME. Edit scripts/daic-exp/env.sh." >&2
    exit 1
  fi
done

PYVER="${DAIC_EXP_PYTHON_VERSION:-3.11}"

# ── modules ────────────────────────────────────────────────────────────────
echo "[1/6] Loading modules ..."
module purge 2>/dev/null || true
module load "${DAIC_EXP_BASE_MODULE}"
module load "${DAIC_EXP_CUDA_MODULE}"
if [ -n "${DAIC_EXP_CUDNN_MODULE:-}" ]; then
  module load "${DAIC_EXP_CUDNN_MODULE}" 2>/dev/null || \
    echo "      [warn] could not load ${DAIC_EXP_CUDNN_MODULE} (optional)"
fi

# ── miniforge install ──────────────────────────────────────────────────────
echo "[2/6] Installing Miniforge into ${DAIC_EXP_MINIFORGE_ROOT} ..."
if [ -x "${DAIC_EXP_MINIFORGE_ROOT}/bin/conda" ]; then
  echo "      already installed — skipping"
else
  mkdir -p "$(dirname "${DAIC_EXP_MINIFORGE_ROOT}")"
  TMP_INSTALLER="$(mktemp -t miniforge.XXXXXX.sh)"
  trap 'rm -f "${TMP_INSTALLER}"' EXIT
  echo "      downloading installer ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "${TMP_INSTALLER}" \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  else
    wget -q -O "${TMP_INSTALLER}" \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  fi
  bash "${TMP_INSTALLER}" -b -p "${DAIC_EXP_MINIFORGE_ROOT}"
  rm -f "${TMP_INSTALLER}"
  trap - EXIT
fi

# Initialise conda for `conda activate` inside this script.
: "${PS1:=}"
# shellcheck source=/dev/null
source "${DAIC_EXP_MINIFORGE_ROOT}/etc/profile.d/conda.sh"

# Caches on the data dir so they survive cluster home quotas.
mkdir -p "${DAIC_EXP_DATA_DIR}/.cache/pip" \
         "${DAIC_EXP_DATA_DIR}/.conda/pkgs"
conda config --prepend pkgs_dirs "${DAIC_EXP_DATA_DIR}/.conda/pkgs" 2>/dev/null || true
export PIP_CACHE_DIR="${DAIC_EXP_DATA_DIR}/.cache/pip"

# ── create env ─────────────────────────────────────────────────────────────
CONDA_ENV_PATH="${DAIC_EXP_MINIFORGE_ROOT}/envs/${DAIC_EXP_CONDA_ENV}"
echo "[3/6] Creating conda env at ${CONDA_ENV_PATH} (python ${PYVER}) ..."
if [ -d "${CONDA_ENV_PATH}" ] && [ -x "${CONDA_ENV_PATH}/bin/python" ]; then
  echo "      env already exists — reusing"
else
  conda create --prefix "${CONDA_ENV_PATH}" "python=${PYVER}" pip -y
fi
conda activate "${CONDA_ENV_PATH}"
echo "      python: $(which python) ($(python --version))"
python -m pip install --upgrade pip setuptools wheel

# ── C-heavy deps via conda-forge ───────────────────────────────────────────
echo "[4/6] Installing C-heavy deps from conda-forge (h5py) ..."
conda install -c conda-forge -y h5py

# ── torch + project ────────────────────────────────────────────────────────
echo "[5/6] Installing PyTorch with CUDA 12.1 wheels ..."
# We pin to cu121 wheels even though the system module is cuda/12.9.
# PyTorch's wheels ship the CUDA runtime bundled, so the system CUDA only
# needs to match the driver (which it does — 12.9 driver supports 12.1
# runtime libs via forward compatibility).
pip install --prefer-binary --index-url https://download.pytorch.org/whl/cu121 \
    "torch>=2.2,<2.7"

echo "[6/6] Installing the project (editable) + dev deps ..."
cd "${DAIC_EXP_PROJECT_DIR}"
pip install --prefer-binary -e ".[dev]"

# ── sanity check ───────────────────────────────────────────────────────────
echo
echo "── torch sanity ──────────────────────────────────────────────────"
python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device 0: {torch.cuda.get_device_name(0)}")
    print(f"  capability: {torch.cuda.get_device_capability(0)}")
else:
    print("  (login node has no GPU — expected. Real check runs in submit_smoke.)")
PY

# ── artefact dirs and symlinks ─────────────────────────────────────────────
mkdir -p "${DAIC_EXP_DATA_DIR}"/{raw,processed,checkpoints,residuals,prior_params,wandb}
mkdir -p "${DAIC_EXP_PROJECT_DIR}/logs"

cd "${DAIC_EXP_PROJECT_DIR}"
mkdir -p data results
ln -sfn "${DAIC_EXP_DATA_DIR}/raw"          data/raw
ln -sfn "${DAIC_EXP_DATA_DIR}/processed"    data/processed
ln -sfn "${DAIC_EXP_DATA_DIR}/checkpoints"  results/checkpoints
ln -sfn "${DAIC_EXP_DATA_DIR}/residuals"    results/residuals
# results/prior_params stays in git (small YAMLs), do not symlink.

echo
echo "── wandb ─────────────────────────────────────────────────────────"
if python -c "import wandb, sys; sys.exit(0 if wandb.api.api_key else 1)" 2>/dev/null; then
  echo "  already logged in"
else
  echo "  Run \`wandb login\` now and paste the key from"
  echo "  https://wandb.ai/authorize  (skip if WANDB_MODE=offline)"
fi

echo
echo "Setup complete."
echo "Project:    ${DAIC_EXP_PROJECT_DIR}"
echo "Data:       ${DAIC_EXP_DATA_DIR}"
echo "Conda env:  ${CONDA_ENV_PATH}  (python ${PYVER})"
echo
echo "Activate interactively on the login node:"
echo "    source ${DAIC_EXP_MINIFORGE_ROOT}/etc/profile.d/conda.sh"
echo "    conda activate ${CONDA_ENV_PATH}"
echo
echo "Next: bash scripts/daic-exp/verify_setup.sh    # pre-flight"
echo "Then: sbatch scripts/daic-exp/submit_smoke.sbatch"
