#!/bin/bash
# Login-node pre-flight check. Run after setup_env.sh and after rsync'ing
# data up. No Slurm, no GPU — confirms the environment is wired up before
# wasting a job submission.
#
#   source scripts/daic/env.sh
#   bash scripts/daic/verify_setup.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${PROJECT_DIR}"

if [ -z "${DAIC_CONDA_ENV:-}" ]; then
  echo "FAIL: env.sh not sourced or out of date (DAIC_CONDA_ENV unset)." >&2
  echo "      Regenerate from env.example.sh and edit your NetID + DATA_DIR." >&2
  exit 1
fi

OK="\033[32m✓\033[0m"
FAIL="\033[31m✗\033[0m"
WARN="\033[33m!\033[0m"
PASS=true
warn() { echo -e "  ${WARN} $1"; }
ok()   { echo -e "  ${OK} $1"; }
fail() { echo -e "  ${FAIL} $1"; PASS=false; }

echo
echo "── env vars ────────────────────────────────────────────────────"
for v in DAIC_NETID DAIC_PROJECT_DIR DAIC_DATA_DIR DAIC_CONDA_ENV DAIC_CUDA_MODULE DAIC_GPU_TYPE; do
  if [ -n "${!v:-}" ] && [ "${!v}" != "CHANGE_ME" ]; then
    ok "${v}=${!v}"
  else
    fail "${v} is unset or CHANGE_ME"
  fi
done

echo
echo "── modules ─────────────────────────────────────────────────────"
module use /opt/insy/modulefiles 2>/dev/null || true
if module load "${DAIC_CUDA_MODULE}" 2>/dev/null; then
  ok "loaded ${DAIC_CUDA_MODULE}"
else
  fail "could not load ${DAIC_CUDA_MODULE} (run \`module avail cuda\` to list)"
fi
# cuDNN is optional — PyTorch wheels bundle it.
if [ -n "${DAIC_CUDNN_MODULE:-}" ]; then
  if module load "${DAIC_CUDNN_MODULE}" 2>/dev/null; then
    ok "loaded ${DAIC_CUDNN_MODULE} (optional)"
  else
    warn "${DAIC_CUDNN_MODULE} not loadable — PyTorch's bundled cuDNN will be used"
  fi
fi

echo
echo "── conda env ───────────────────────────────────────────────────"
module load miniconda/3 2>/dev/null || true
# conda.sh references $PS1, which is unset in non-interactive shells
# under `set -u`. Define it as empty to silence the unbound-variable error.
: "${PS1:=}"
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
DAIC_CONDA_PATH="${HOME}/.conda/envs/${DAIC_CONDA_ENV}"
if [ -x "${DAIC_CONDA_PATH}/bin/python" ]; then
  conda activate "${DAIC_CONDA_PATH}"
  ok "conda env activated: ${DAIC_CONDA_PATH}"
  ok "python: $(python --version 2>&1)  ($(which python))"
else
  fail "conda env not found at ${DAIC_CONDA_PATH} — run setup_env.sh"
  exit 1
fi

echo
echo "── python imports ──────────────────────────────────────────────"
python - <<'PY' || PASS=false
import sys
status = 0
def check(name, statement):
    global status
    try:
        exec(statement)
        print(f"  \033[32m✓\033[0m import {name}")
    except Exception as e:
        print(f"  \033[31m✗\033[0m import {name}: {e}")
        status = 1
check("torch", "import torch")
check("numpy", "import numpy")
check("scipy", "import scipy")
check("yaml", "import yaml")
check("h5py", "import h5py")
check("wandb", "import wandb")
check("src", "import src; import src.diffusion.sde as sde; import src.diffusion.prior_fit as pf")
sys.exit(status)
PY

echo
echo "── data symlinks ───────────────────────────────────────────────"
# prior_params is tracked in git (small YAMLs), not symlinked.
for sub in data/raw data/processed results/checkpoints results/residuals; do
  if [ -L "$sub" ] || [ -d "$sub" ]; then
    target="$(readlink -f "$sub" 2>/dev/null || echo "$sub")"
    ok "$sub → $target"
  else
    warn "$sub missing (will be created on first sbatch via _common.sh)"
  fi
done

echo
echo "── data files ──────────────────────────────────────────────────"
declare -a CHECK_FILES=(
  "data/raw/metr-la.h5"
  "data/raw/pems-bay.h5"
  "data/processed/metrla_train.npz"
  "data/processed/pemsbay_train.npz"
  "data/processed/metrla_scaler.yaml"
  "results/checkpoints"  # directory; per-run subdirs created during training
)
for f in "${CHECK_FILES[@]}"; do
  if [ -e "$f" ]; then
    sz=$(du -sh "$f" 2>/dev/null | cut -f1)
    ok "$f  (${sz})"
  else
    warn "$f missing (push from local Mac with sync_to_daic.sh)"
  fi
done

echo
echo "── wandb ───────────────────────────────────────────────────────"
if [ "${WANDB_MODE:-online}" = "online" ]; then
  if python -c "import wandb,sys; sys.exit(0 if wandb.api.api_key else 1)" 2>/dev/null; then
    ok "wandb online: API key present"
  else
    warn "wandb online but no API key — run \`wandb login\` or set WANDB_MODE=offline"
  fi
else
  ok "wandb offline (events buffered to disk)"
fi

echo
echo "── unit tests (subset) ─────────────────────────────────────────"
if pytest -q tests/test_sde.py tests/test_prior_fit.py 2>&1 | tail -5; then
  ok "test_sde + test_prior_fit pass"
else
  fail "unit tests failed"
fi

echo
if $PASS; then
  echo -e "${OK} Pre-flight OK. Next: \`sbatch scripts/daic/submit_smoke.sbatch\`"
else
  echo -e "${FAIL} Some checks failed. Fix them before submitting jobs."
  exit 1
fi
