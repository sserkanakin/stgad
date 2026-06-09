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

# slurm opens job logs into these dirs *before* the job body runs and won't
# create them, so make sure they exist (they're gitignored and easy to wipe).
mkdir -p logs/{backbone,residuals,diffusion,smoke,eval}

if [ -z "${DAIC_CLUSTER:-}" ]; then
  echo "FAIL: env.sh not sourced or out of date (DAIC_CLUSTER unset)." >&2
  echo "      Run: source scripts/daic/env.sh   (it sources clusters.sh)." >&2
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
for v in DAIC_NETID DAIC_CLUSTER DAIC_LOGIN_HOST DAIC_PROJECT_DIR DAIC_DATA_DIR DAIC_CUDA_MODULE DAIC_PARTITION; do
  if [ -n "${!v:-}" ] && [ "${!v}" != "CHANGE_ME" ]; then
    ok "${v}=${!v}"
  else
    fail "${v} is unset or CHANGE_ME"
  fi
done

echo
echo "── modules + python (cluster: ${DAIC_CLUSTER}) ─────────────────"
if [ "${DAIC_PYTHON_KIND}" = "conda" ]; then
  # stable DAIC: insy modulefiles + conda.
  module use "${DAIC_MODULE_USE}" 2>/dev/null || true
  if module load "${DAIC_CUDA_MODULE}" 2>/dev/null; then
    ok "loaded ${DAIC_CUDA_MODULE}"
  else
    fail "could not load ${DAIC_CUDA_MODULE} (run \`module avail cuda\` to list)"
  fi
  if [ -n "${DAIC_CUDNN_MODULE:-}" ]; then
    module load "${DAIC_CUDNN_MODULE}" 2>/dev/null \
      && ok "loaded ${DAIC_CUDNN_MODULE} (optional)" \
      || warn "${DAIC_CUDNN_MODULE} not loadable — PyTorch's bundled cuDNN will be used"
  fi
  module load "${DAIC_CONDA_MODULE}" 2>/dev/null || true
  : "${PS1:=}"
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [ -x "${DAIC_CONDA_PREFIX}/bin/python" ]; then
    conda activate "${DAIC_CONDA_PREFIX}"
    ok "conda env activated: ${DAIC_CONDA_PREFIX}"
    ok "python: $(python --version 2>&1)  ($(which python))"
  else
    fail "conda env not found at ${DAIC_CONDA_PREFIX} — run setup_env.sh"
    exit 1
  fi
else
  # experimental DAIC: 2025 Lmod tree + venv overlay.
  for m in "${DAIC_MODULE_TREE}" "${DAIC_PYTHON_MODULE}" "${DAIC_TORCH_MODULE}"; do
    module load "$m" 2>/dev/null && ok "loaded $m" || fail "could not load $m"
  done
  [ -n "${DAIC_CUDNN_MODULE:-}" ] && { module load "${DAIC_CUDNN_MODULE}" 2>/dev/null \
    && ok "loaded ${DAIC_CUDNN_MODULE} (optional)" \
    || warn "${DAIC_CUDNN_MODULE} not loadable — torch ships its own cuDNN"; }
  if [ -x "${DAIC_VENV_DIR}/bin/python" ]; then
    # shellcheck source=/dev/null
    source "${DAIC_VENV_DIR}/bin/activate"
    # venv deps must win over the module py-* packages on PYTHONPATH.
    export PYTHONPATH="${DAIC_VENV_DIR}/lib/python${DAIC_PYTHON_VERSION}/site-packages:${PYTHONPATH:-}"
    ok "venv activated: ${DAIC_VENV_DIR}"
    ok "python: $(python --version 2>&1)  ($(which python))"
  else
    fail "venv not found at ${DAIC_VENV_DIR} — run setup_env.sh"
    exit 1
  fi
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
check("easydict", "import easydict")
check("src.diffusion", "import src.diffusion.forward_fcps, src.diffusion.prior, src.diffusion.reverse, src.diffusion.denoiser")
check("src.data + paths", "import src.data.adjacency, src.data.dataset; from src.utils import paths")
sys.exit(status)
PY

echo
echo "── data symlinks ───────────────────────────────────────────────"
for sub in data/raw data/processed results/backbone results/diffusion results/residuals results/eval; do
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
  "data/raw/metrla/metr-la.h5"
  "data/raw/pemsbay/pems-bay.h5"
  "data/processed/metrla/train.npz"
  "data/processed/pemsbay/train.npz"
  "data/processed/metrla/scaler.yaml"
  "results/backbone"  # directory; per-run subdirs created during training
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
echo "── unit tests (collect only) ───────────────────────────────────"
# the login node kills heavy processes, so here we only confirm the suite
# imports + collects cleanly. run the full suite on a compute node, e.g.:
#   srun --mem=8G --cpus-per-task=2 --time=00:10:00 pytest -q tests/
if pytest -q --collect-only tests/ >/dev/null 2>&1; then
  ok "tests collect cleanly (run the full suite on a compute node)"
else
  fail "test collection failed — run: pytest --collect-only tests/"
fi

echo
if $PASS; then
  echo -e "${OK} Pre-flight OK. Next: \`sbatch scripts/daic/submit_smoke.sbatch\`"
else
  echo -e "${FAIL} Some checks failed. Fix them before submitting jobs."
  exit 1
fi
