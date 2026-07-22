#!/usr/bin/env bash
# Submit the rsh-λ (joint-spectrum shrinkage) pilot as ONE job. See shrink_pilot.sbatch.
#   scripts/daic/submit_shrink_pilot.sh                       # pems04f, λ∈{0.2,0.5} × {mlp,gwn}, seed42
#   LAMBDAS="0.1 0.3" DATASET=pems08f scripts/daic/submit_shrink_pilot.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/env.sh"
# shellcheck source=/dev/null
source "${HERE}/_gpu.sh"

EXP="${EXP:-20260722_rshshrink}"
DATASET="${DATASET:-pems04f}"
LAMBDAS="${LAMBDAS:-0.2 0.5}"
BACKBONES="${BACKBONES:-mlp gwn}"
SEEDS="${SEEDS:-42}"

export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
LOG_DIR="${OUTPUT_ROOT}/logs/${EXP}"; mkdir -p "${LOG_DIR}"

# PREFLIGHT: training reads seed42 train+val residuals for every backbone.
missing=""
for BB in ${BACKBONES}; do for SP in train val; do
  f="${RESULTS_ROOT}/residuals/${DATASET}/${BB}/seed42/${SP}.npz"
  [ -s "${f}" ] || missing="${missing}  ${f}"$'\n'
done; done
[ -n "${missing}" ] && { echo "ABORT: missing residuals:" >&2; printf '%s' "${missing}" >&2; exit 1; }

GRES="$(gpu_blackwell_gres || true)"; GRES="${GRES:-${DAIC_GRES_GPU}}"
RES=(--partition="${DAIC_PARTITION}" --gres="${GRES}")
[ -n "${DAIC_QOS_MEDIUM:-}" ] && RES+=(--qos="${DAIC_QOS_MEDIUM}")
[ -n "${DAIC_GPU_EXCLUDE:-}" ] && RES+=(--exclude="${DAIC_GPU_EXCLUDE}")

EXPORTS="ALL,EXP=${EXP},DATASET=${DATASET},DATA_ROOT=${DATA_ROOT},RESULTS_ROOT=${RESULTS_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}"
EXPORTS="${EXPORTS},LAMBDAS=${LAMBDAS// /_},BACKBONES=${BACKBONES// /_},SEEDS=${SEEDS// /_}"

N=$(( $(set -- ${LAMBDAS}; echo $#) * $(set -- ${BACKBONES}; echo $#) * $(set -- ${SEEDS}; echo $#) ))
echo "===== rsh-λ pilot ${EXP}  ${DATASET}  λ=[${LAMBDAS}] × [${BACKBONES}]  = ${N} runs in ONE job ====="
JOBID="$(sbatch --parsable \
  --export="${EXPORTS}" \
  --job-name="rshl-${DATASET}" \
  --output="${LOG_DIR}/%j.out" --error="${LOG_DIR}/%j.err" \
  "${RES[@]}" "$@" "${HERE}/shrink_pilot.sbatch")"
echo "  job ${JOBID}  logs: ${LOG_DIR}/${JOBID}.out"
echo "  results: ${OUTPUT_ROOT}/${EXP}/rsh_l<λ>/${DATASET}/<bb>/kronecker/seed*/train/"
