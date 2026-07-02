# SOURCE this for an interactive DAIC python env that matches the sbatch jobs:
#
#   source scripts/daic/activate.sh
#
# it loads the same module stack + project venv (or conda env) that _common.sh
# sets up inside jobs — the missing piece when you `activate` the venv by hand and
# hit "ModuleNotFoundError: numpy" (numpy comes from the py-torch module, which the
# --system-site-packages venv reuses; without `module load` it isn't on the path).
#
# unlike _common.sh this is interactive-SAFE: no `set -e` (a failed command won't
# kill your login shell) and no data-dir relinking / diagnostics (those are job-only).
# use it for one-off CPU chores on the login node (e.g. build_signal_cache) or an
# `srun --pty` shell on a compute node.

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=/dev/null
source "${_HERE}/env.sh"     # sets DAIC_* (cluster, module names, venv dir) via clusters.sh

if [ "${DAIC_PYTHON_KIND}" = "conda" ]; then
  # stable DAIC: insy modulefiles + conda env on NFS home.
  module use "${DAIC_MODULE_USE}"
  module load "${DAIC_CUDA_MODULE}"
  [ -n "${DAIC_CUDNN_MODULE:-}" ] && module load "${DAIC_CUDNN_MODULE}" 2>/dev/null || true
  module load "${DAIC_CONDA_MODULE}"
  : "${PS1:=}"
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${DAIC_CONDA_PREFIX}"
else
  # experimental DAIC: 2025 Lmod tree (python 3.11 + cuda torch, which ships numpy)
  # + the project venv overlay. ignore the stale lmod cache (see _common.sh).
  export LMOD_IGNORE_CACHE=1
  module purge 2>/dev/null || true
  module load "${DAIC_MODULE_TREE}"
  module load "${DAIC_PYTHON_MODULE}"
  module load "${DAIC_TORCH_MODULE}"           # provides cuda torch + numpy
  [ -n "${DAIC_CUDNN_MODULE:-}" ] && module load "${DAIC_CUDNN_MODULE}" 2>/dev/null || true
  if [ ! -x "${DAIC_VENV_DIR}/bin/python" ]; then
    echo "ERROR: venv not found at ${DAIC_VENV_DIR} — run scripts/daic/setup_env.sh first." >&2
    return 1 2>/dev/null || exit 1
  fi
  # shellcheck source=/dev/null
  source "${DAIC_VENV_DIR}/bin/activate"
  # venv site-packages first so our pip deps win over the module versions.
  export PYTHONPATH="${DAIC_VENV_DIR}/lib/python${DAIC_PYTHON_VERSION}/site-packages:${PYTHONPATH:-}"
fi

export DAIC_PYTHON="$(which python)"
export PYTHONUNBUFFERED=1
cd "${PROJECT_DIR:-${DAIC_PROJECT_DIR:-$PWD}}" 2>/dev/null || true
echo "[activate] python: $(which python)  $(python --version 2>&1)"
python -c "import numpy, torch; print('[activate] numpy', numpy.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())" 2>/dev/null \
  || echo "[activate] WARN: numpy/torch import failed — check the module stack"
