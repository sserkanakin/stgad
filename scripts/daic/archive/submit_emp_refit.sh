#!/usr/bin/env bash
# Submit the EMPIRICAL-refit at the tuned γ as a slurm ARRAY (one task per dataset×backbone,
# all seeds sequential). Throttled to the free-slot count so it fills sakin without hogging.
# Only empirical is re-run (γ-tuned); iso/reshaped/ddpm/whitened are γ-free and untouched.
#
#   scripts/daic/submit_emp_refit.sh                          # 4 ds × {mlp,gwn} × 5 seeds, γ=0.2
#   GAMMA=0.1 scripts/daic/submit_emp_refit.sh                # different γ
#   DATASETS="pems08f pems08s" scripts/daic/submit_emp_refit.sh
#   THROTTLE=5 scripts/daic/submit_emp_refit.sh               # match free slots
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/env.sh"
# shellcheck source=/dev/null
source "${HERE}/_gpu.sh"

EXP="${EXP:-20260721_emprefit}"
GAMMA="${GAMMA:-0.2}"
DATASETS="${DATASETS:-pems08f pems08s pems04f pems04s}"
BACKBONES="${BACKBONES:-mlp gwn}"
SEEDS="${SEEDS:-42 126 2022 3407 1}"
THROTTLE="${THROTTLE:-5}"

export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
LOG_DIR="${OUTPUT_ROOT}/logs/${EXP}"; mkdir -p "${LOG_DIR}"

# PREFLIGHT: empirical reads seed42 train+val residuals for every (ds, bb).
missing=""
for DS in ${DATASETS}; do for BB in ${BACKBONES}; do for SP in train val; do
  f="${RESULTS_ROOT}/residuals/${DS}/${BB}/seed42/${SP}.npz"
  [ -s "${f}" ] || missing="${missing}  ${f}"$'\n'
done; done; done
[ -n "${missing}" ] && { echo "ABORT: missing residual inputs (copy them to sakin first):" >&2; printf '%s' "${missing}" >&2; exit 1; }

ND=$(set -- ${DATASETS}; echo $#); NB=$(set -- ${BACKBONES}; echo $#); N=$((ND * NB))

GRES="$(gpu_blackwell_gres || true)"; GRES="${GRES:-${DAIC_GRES_GPU}}"
RES=(--partition="${DAIC_PARTITION}" --gres="${GRES}")
[ -n "${DAIC_QOS_MEDIUM:-}" ] && RES+=(--qos="${DAIC_QOS_MEDIUM}")
[ -n "${DAIC_GPU_EXCLUDE:-}" ] && RES+=(--exclude="${DAIC_GPU_EXCLUDE}")

EXPORTS="ALL,EXP=${EXP},GAMMA=${GAMMA},DATA_ROOT=${DATA_ROOT},RESULTS_ROOT=${RESULTS_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}"
EXPORTS="${EXPORTS},DATASETS=${DATASETS// /_},BACKBONES=${BACKBONES// /_},SEEDS=${SEEDS// /_}"

echo "===== emp-refit ${EXP}  γ=${GAMMA}  [${DATASETS}] × [${BACKBONES}] × seeds[${SEEDS}]  = ${N} tasks (throttle ${THROTTLE}) ====="
JOBID="$(sbatch --parsable \
  --array="0-$((N - 1))%${THROTTLE}" \
  --export="${EXPORTS}" \
  --job-name="emprefit" \
  --output="${LOG_DIR}/%A_%a.out" --error="${LOG_DIR}/%A_%a.err" \
  "${RES[@]}" "$@" "${HERE}/emp_refit.sbatch")"
echo "  array ${JOBID}  logs: ${LOG_DIR}/"
echo "  results: ${OUTPUT_ROOT}/${EXP}/emp/<ds>/<bb>/kronecker/seed*/train/best_crps.pt"
echo "  watch: squeue --me"
