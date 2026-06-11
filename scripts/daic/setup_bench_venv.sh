#!/usr/bin/env bash
# one-time bootstrap of a SEPARATE, throwaway venv carrying a cu128 PyTorch so we
# can run on the RTX PRO 6000 Blackwell nodes (sm_120), which the production
# py-torch/2.5.1 module crashes on ("no kernel image available"). this venv is
# deliberately NOT the production one (scripts/daic/setup_env.sh builds that) —
# we never want a cu128 torch shadowing the validated a40/l40 runs.
#
# run on the EXPERIMENTAL login node, from the project dir:
#   ssh daic-exp
#   cd ~/linuxhome/research-tud
#   source scripts/daic/env.sh
#   bash scripts/daic/setup_bench_venv.sh
#
# QUOTA WARNING: linuxhome is ~8 GB. a cu128 torch wheel + bundled cuda libs is
# several GB. this script checks free space first and `du`s the result; if it's
# tight, `rm -rf "$BENCH_VENV_DIR"` after benchmarking (it's disposable).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
[ -n "${DAIC_CLUSTER:-}" ] || source "${HERE}/env.sh"

if [ "${DAIC_CLUSTER}" != "exp" ]; then
  echo "ERROR: the Blackwell cards are only on the experimental cluster (exp)." >&2
  echo "       got DAIC_CLUSTER=${DAIC_CLUSTER}. ssh daic-exp and retry." >&2
  exit 1
fi

# sibling of the production venv, clearly marked as the cu128 bench env.
BENCH_VENV_DIR="${BENCH_VENV_DIR:-${DAIC_VENV_DIR}-cu128}"
CU128_INDEX="https://download.pytorch.org/whl/cu128"

echo "─── cu128 bench venv ────────────────────────────────────────────"
echo "target:   ${BENCH_VENV_DIR}"
echo "index:    ${CU128_INDEX}"
df -h "$(dirname "${BENCH_VENV_DIR}")" 2>/dev/null | sed 's/^/free:     /' || true
echo "─────────────────────────────────────────────────────────────────"

# load ONLY the module tree + python. crucially NOT py-torch: we want the venv's
# cu128 torch to be the one and only torch, with no module torch on PYTHONPATH or
# its cuda libs on LD_LIBRARY_PATH to fight the wheel's bundled runtime.
export LMOD_IGNORE_CACHE=1
module purge 2>/dev/null || true
module load "${DAIC_MODULE_TREE}"
module load "${DAIC_PYTHON_MODULE}"

export PIP_CACHE_DIR="${DAIC_DATA_DIR}/.cache/pip"
mkdir -p "${PIP_CACHE_DIR}"

echo "[1/4] Creating venv (--system-site-packages reuses module numpy etc) ..."
if [ -x "${BENCH_VENV_DIR}/bin/python" ]; then
  echo "      venv already exists — reusing (delete it to rebuild clean)"
else
  python3 -m venv --system-site-packages "${BENCH_VENV_DIR}"
fi
# shellcheck source=/dev/null
source "${BENCH_VENV_DIR}/bin/activate"
# the venv's packages must win over anything the module tree left on PYTHONPATH.
export PYTHONPATH="${BENCH_VENV_DIR}/lib/python${DAIC_PYTHON_VERSION}/site-packages:${PYTHONPATH:-}"
python -m pip install --upgrade pip setuptools wheel

echo "[2/4] cu128 PyTorch (needs >=2.7 for sm_120 / Blackwell) ..."
# no version pin: take the latest stable cu128 wheel, which carries sm_120 kernels.
pip install --prefer-binary --index-url "${CU128_INDEX}" "torch>=2.7"

echo "[3/4] Project (editable) + remaining pure-python deps ..."
# torch now comes from the venv (cu128); numpy from system-site. pip fills the
# rest (wandb, easydict, einops, h5py). --no-build-isolation keeps it using the
# torch we just installed rather than pulling another.
pip install --prefer-binary -e ".[dev]"

echo "[4/4] Sanity check ..."
python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda_build={torch.version.cuda}  "
      f"cuda_available={torch.cuda.is_available()}")
# on the login node there is no gpu, so capability prints only inside a job.
if torch.cuda.is_available():
    print(f"  device 0: {torch.cuda.get_device_name(0)}  "
          f"cap={torch.cuda.get_device_capability(0)}")
PY

echo
echo "venv size:"; du -sh "${BENCH_VENV_DIR}" 2>/dev/null || true
df -h "$(dirname "${BENCH_VENV_DIR}")" 2>/dev/null | sed 's/^/free:     /' || true
echo
echo "Done. Next: bash scripts/daic/submit_bench.sh   # fires the side-by-side cells"
echo "Cleanup when finished:  rm -rf ${BENCH_VENV_DIR}"
