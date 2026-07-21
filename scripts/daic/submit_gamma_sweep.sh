#!/usr/bin/env bash
# γ-sweep for the EMPIRICAL prior (γ only touches empirical; iso/reshaped are γ-free).
# Submits ONE array — one task per (γ, backbone) — so it drains through whatever slots are
# free (1 now, more as the ugnet jobs finish). γ=0.8 already exists as the `emp` ladder arm,
# so we only train the NEW values. Aggregate afterwards: CRPS vs γ should fall toward the
# reshaped arm; if it plateaus above reshaped, the reshape novelty is necessary.
#
#   scripts/daic/submit_gamma_sweep.sh                          # γ∈{.04,.1,.2,.4} × {mlp,gwn}, pems08f
#   GAMMAS="0.02 0.04 0.1 0.2 0.4" scripts/daic/submit_gamma_sweep.sh
#   DATASET=pems08s BACKBONES="mlp" scripts/daic/submit_gamma_sweep.sh
# anything after the flags passes straight to sbatch (e.g. --time=8:00:00).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/env.sh"
# shellcheck source=/dev/null
source "${HERE}/_gpu.sh"

EXP="${EXP:-20260721_gammasweep}"
DATASET="${DATASET:-pems08f}"                  # sakin holds pems08f/pems08s
GAMMAS="${GAMMAS:-0.04 0.1 0.2 0.4}"           # 0.8 already trained (the `emp` arm)
BACKBONES="${BACKBONES:-mlp gwn}"              # weak + strong; γ matters most on mlp
SEEDS="${SEEDS:-42}"

export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
LOG_DIR="${OUTPUT_ROOT}/logs/${EXP}"; mkdir -p "${LOG_DIR}"

# PREFLIGHT: residuals (INPUTS) must exist for every backbone we sweep (read from RESULTS_ROOT).
missing=""
for BB in ${BACKBONES}; do for SP in train val; do
  f="${RESULTS_ROOT}/residuals/${DATASET}/${BB}/seed${SEEDS%% *}/${SP}.npz"
  [ -s "${f}" ] || missing="${missing}  ${f}"$'\n'
done; done
[ -n "${missing}" ] && { echo "ABORT: missing residual inputs:" >&2; printf '%s' "${missing}" >&2; exit 1; }

N=$(( $(set -- ${GAMMAS}; echo $#) * $(set -- ${BACKBONES}; echo $#) * $(set -- ${SEEDS}; echo $#) ))

GRES="$(gpu_blackwell_gres || true)"; GRES="${GRES:-${DAIC_GRES_GPU}}"
RES=(--partition="${DAIC_PARTITION}" --gres="${GRES}")
[ -n "${DAIC_QOS_MEDIUM:-}" ] && RES+=(--qos="${DAIC_QOS_MEDIUM}")
[ -n "${DAIC_GPU_EXCLUDE:-}" ] && RES+=(--exclude="${DAIC_GPU_EXCLUDE}")

# multi-word values cannot carry spaces over --export: underscore-join (sbatch decodes).
EXPORTS="ALL,EXP=${EXP},DATASET=${DATASET},DATA_ROOT=${DATA_ROOT},RESULTS_ROOT=${RESULTS_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}"
EXPORTS="${EXPORTS},GAMMAS=${GAMMAS// /_},BACKBONES=${BACKBONES// /_},SEEDS=${SEEDS// /_}"

echo "===== γ-sweep ${EXP}  dataset=${DATASET}  γ=[${GAMMAS}] × bb=[${BACKBONES}]  = ${N} runs in ONE job ====="
JOBID="$(sbatch --parsable \
  --export="${EXPORTS}" \
  --job-name="gsweep-${DATASET}" \
  --output="${LOG_DIR}/%j.out" --error="${LOG_DIR}/%j.err" \
  "${RES[@]}" "$@" "${HERE}/gamma_sweep.sbatch")"
echo "  job ${JOBID}  (loops ${N} runs serially)  logs: ${LOG_DIR}/${JOBID}.out"
echo "  results: ${OUTPUT_ROOT}/${EXP}/emp_g<γ>/${DATASET}/<bb>/kronecker/seed*/train/best_crps.pt"
echo "  compare vs existing: 20260721_ladder/{emp (γ=0.8), rsh}"
echo "  watch: squeue --me"
