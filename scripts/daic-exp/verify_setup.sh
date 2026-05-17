#!/bin/bash
# Login-node pre-flight for experimental DAIC. Run after setup_env.sh
# and after rsync'ing data up. No Slurm, no GPU — confirms the env is wired.
#
#   source scripts/daic-exp/env.sh
#   bash scripts/daic-exp/verify_setup.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${PROJECT_DIR}"

if [ -z "${DAIC_EXP_CONDA_ENV:-}" ]; then
  echo "FAIL: env.sh not sourced (DAIC_EXP_CONDA_ENV unset)." >&2
  echo "      source scripts/daic-exp/env.sh" >&2
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
for v in DAIC_EXP_NETID DAIC_EXP_ACCOUNT DAIC_EXP_PROJECT_DIR \
         DAIC_EXP_DATA_DIR DAIC_EXP_CONDA_ENV DAIC_EXP_MINIFORGE_ROOT \
         DAIC_EXP_BASE_MODULE DAIC_EXP_CUDA_MODULE DAIC_EXP_GPU_TYPE \
         DAIC_EXP_PARTITION; do
  val="${!v:-}"
  if [ -n "${val}" ] && [ "${val}" != "CHANGE_ME" ]; then
    ok "${v}=${val}"
  else
    fail "${v} is unset or CHANGE_ME"
  fi
done

echo
echo "── slurm account ───────────────────────────────────────────────"
if command -v sacctmgr >/dev/null 2>&1; then
  if sacctmgr -n show user "${USER}" withassoc format=account%30 2>/dev/null \
       | tr -d ' ' | grep -qx "${DAIC_EXP_ACCOUNT}"; then
    ok "account '${DAIC_EXP_ACCOUNT}' is associated to your user"
  else
    warn "account '${DAIC_EXP_ACCOUNT}' not visible via sacctmgr — sbatch may reject jobs"
    echo "       Try:  sacctmgr show user \$USER withassoc format=account%30"
  fi
else
  warn "sacctmgr not in PATH (are we on a Slurm host?)"
fi

echo
echo "── modules ─────────────────────────────────────────────────────"
module purge 2>/dev/null || true
if module load "${DAIC_EXP_BASE_MODULE}" 2>/dev/null; then
  ok "loaded ${DAIC_EXP_BASE_MODULE}"
else
  fail "could not load ${DAIC_EXP_BASE_MODULE} (run \`module avail\`)"
fi
if module load "${DAIC_EXP_CUDA_MODULE}" 2>/dev/null; then
  ok "loaded ${DAIC_EXP_CUDA_MODULE}"
else
  fail "could not load ${DAIC_EXP_CUDA_MODULE}"
fi
if [ -n "${DAIC_EXP_CUDNN_MODULE:-}" ]; then
  if module load "${DAIC_EXP_CUDNN_MODULE}" 2>/dev/null; then
    ok "loaded ${DAIC_EXP_CUDNN_MODULE} (optional)"
  else
    warn "${DAIC_EXP_CUDNN_MODULE} unloadable — PyTorch's bundled cuDNN will be used"
  fi
fi

echo
echo "── conda env ───────────────────────────────────────────────────"
if [ ! -x "${DAIC_EXP_MINIFORGE_ROOT}/bin/conda" ]; then
  fail "Miniforge not at ${DAIC_EXP_MINIFORGE_ROOT} — run setup_env.sh"
  exit 1
fi
: "${PS1:=}"
# shellcheck source=/dev/null
source "${DAIC_EXP_MINIFORGE_ROOT}/etc/profile.d/conda.sh"
CONDA_ENV_PATH="${DAIC_EXP_MINIFORGE_ROOT}/envs/${DAIC_EXP_CONDA_ENV}"
if [ -x "${CONDA_ENV_PATH}/bin/python" ]; then
  conda activate "${CONDA_ENV_PATH}"
  ok "conda env activated: ${CONDA_ENV_PATH}"
  ok "python: $(python --version 2>&1)  ($(which python))"
else
  fail "conda env not at ${CONDA_ENV_PATH} — run setup_env.sh"
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
for sub in data/raw data/processed results/checkpoints results/residuals; do
  if [ -L "$sub" ] || [ -d "$sub" ]; then
    target="$(readlink -f "$sub" 2>/dev/null || echo "$sub")"
    ok "$sub → $target"
  else
    warn "$sub missing (created on first sbatch via _common.sh)"
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
  "results/checkpoints"
)
for f in "${CHECK_FILES[@]}"; do
  if [ -e "$f" ]; then
    sz=$(du -sh "$f" 2>/dev/null | cut -f1)
    ok "$f  (${sz})"
  else
    warn "$f missing (push from your local machine with sync_to_daic_exp.sh)"
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
  ok "wandb offline (events buffered to disk; run \`wandb sync\` afterwards)"
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
  echo -e "${OK} Pre-flight OK. Next: sbatch scripts/daic-exp/submit_smoke.sbatch"
else
  echo -e "${FAIL} Some checks failed. Fix them before submitting jobs."
  exit 1
fi
