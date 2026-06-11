#!/bin/bash
#SBATCH --job-name=bench_gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/bench/%x-%j.out
#SBATCH --error=logs/bench/%x-%j.err
#
# one benchmark cell: time the eval (sampling) or train (optimiser) hot loop on
# the gpu this job lands on, with the chosen torch stack. partition / gres are
# passed on the command line by submit_bench.sh (which pins the gpu type).
#
# this is intentionally self-contained and does NOT source _common.sh, because
# the cu128 stack must load python WITHOUT the py-torch module (so the bench
# venv's cu128 torch is the only torch — no module torch on PYTHONPATH, no
# module cuda libs on LD_LIBRARY_PATH fighting the wheel's bundled runtime).
#
# driven entirely by env vars (submit_bench.sh sets them via --export):
#   BENCH_STACK   module | cu128        (which torch)
#   BENCH_MODE    eval | train
#   BENCH_DS, BENCH_PRIOR, BENCH_SEED, BENCH_EXP
#   BENCH_GEN_BATCH, BENCH_NSTEPS       (eval)
#   BENCH_BATCH_SIZE                    (train)
#   BENCH_WARMUP, BENCH_ITERS
set -euo pipefail
mkdir -p logs/bench
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
# shellcheck source=/dev/null
source "${PROJECT_DIR}/scripts/daic/env.sh"
cd "${PROJECT_DIR}"

BENCH_STACK="${BENCH_STACK:-module}"
BENCH_MODE="${BENCH_MODE:-eval}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

export LMOD_IGNORE_CACHE=1
module purge 2>/dev/null || true
module load "${DAIC_MODULE_TREE}"
module load "${DAIC_PYTHON_MODULE}"

if [ "${BENCH_STACK}" = "cu128" ]; then
  VENV_DIR="${BENCH_VENV_DIR:-${DAIC_VENV_DIR}-cu128}"
  if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "ERROR: cu128 venv missing at ${VENV_DIR} — run setup_bench_venv.sh first." >&2
    exit 1
  fi
  # NB: deliberately NOT loading py-torch here.
else
  VENV_DIR="${DAIC_VENV_DIR}"
  module load "${DAIC_TORCH_MODULE}"
  [ -n "${DAIC_CUDNN_MODULE:-}" ] && module load "${DAIC_CUDNN_MODULE}" 2>/dev/null || true
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${VENV_DIR}/lib/python${DAIC_PYTHON_VERSION}/site-packages:${PYTHONPATH:-}"
BENCH_PY="${VENV_DIR}/bin/python"

echo "─── bench env ───────────────────────────────────────────────────"
echo "host:     $(hostname)"
echo "node:     ${SLURMD_NODENAME:-?}  cuda=${CUDA_VISIBLE_DEVICES:-?}"
echo "stack:    ${BENCH_STACK}   mode: ${BENCH_MODE}"
echo "venv:     ${VENV_DIR}"
echo "python:   ${BENCH_PY}  $(${BENCH_PY} --version 2>&1)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
echo "─────────────────────────────────────────────────────────────────"

COMMON=(--mode "${BENCH_MODE}" --dataset "${BENCH_DS:-metrla}"
        --prior "${BENCH_PRIOR:-kronecker}" --seed "${BENCH_SEED:-42}"
        --exp "${BENCH_EXP:-dev}"
        --warmup "${BENCH_WARMUP:-2}" --iters "${BENCH_ITERS:-5}")
if [ "${BENCH_MODE}" = "eval" ]; then
  COMMON+=(--gen-batch "${BENCH_GEN_BATCH:-2048}" --n-steps "${BENCH_NSTEPS:-50}")
else
  COMMON+=(--batch-size "${BENCH_BATCH_SIZE:-32}")
fi

# pass BENCH_STACK through so the BENCH_RESULT line is self-labelling.
srun --unbuffered --export=ALL,BENCH_STACK="${BENCH_STACK}" \
     "${BENCH_PY}" -u scripts/bench_diffusion.py "${COMMON[@]}"
