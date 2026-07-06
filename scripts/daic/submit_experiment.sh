#!/usr/bin/env bash
# ONE command per experiment round: submit a slurm ARRAY where each task is the
# complete lifecycle for one (dataset, prior) cell —
#   train (per-epoch val-loss selection, EM step-sweep diagnostic every 10 epochs)
#   → auto-pick the reverse step count on best.pt (fewest within 1% of min CRPS)
#   → FULL val-set eval at that count (val_results.json + val_summary.txt).
# the test split is NOT touched (separate decision later).
#
#   # pilot (2 tasks):
#   DATASETS=metrla EXP=20260706_gadmse scripts/daic/submit_experiment.sh 42
#   # fan out the rest:
#   DATASETS="pems04s pems04f pems08s pems08f pemsbay" EXP=20260706_gadmse \
#     scripts/daic/submit_experiment.sh 42
#   # anything after the seed passes straight to sbatch (e.g. --time=30:00:00)
#
# outputs land in ONE dated tree: ${OUTPUT_ROOT}/${EXP}/<ds>/gwn/<prior>/seed<seed>/{train,eval}
# logs:      ${OUTPUT_ROOT}/logs/${EXP}/<ArrayJobID>_<task>.out (+ copied into the run dir)
# manifest:  ${OUTPUT_ROOT}/logs/${EXP}/grid_seed<seed>.tsv  (task -> dataset/prior/run_dir)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/env.sh"

SEED="${1:?usage: EXP=<tag> submit_experiment.sh <seed> [extra sbatch args]}"
shift
: "${EXP:?set EXP=<dated tag>, e.g. EXP=20260706_gadmse (names the results tree + wandb group)}"

# priority order from the user; backbone fixed to gwn for this protocol round.
DATASETS="${DATASETS:-metrla pems04s pems04f pems08s pems08f pemsbay}"
PRIORS="${PRIORS:-kronecker isotropic}"
BACKBONE="${BACKBONE:-gwn}"
THROTTLE="${THROTTLE:-6}"
EPOCHS="${EPOCHS:-200}"

# reads from the data share; ALL writes into one dated tree on windowshome.
export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"

LOG_DIR="${OUTPUT_ROOT}/logs/${EXP}"
mkdir -p "${LOG_DIR}"
GRID_FILE="${LOG_DIR}/grid_seed${SEED}.tsv"
: > "${GRID_FILE}"
idx=0
for DS in ${DATASETS}; do
  for PR in ${PRIORS}; do
    RUN_DIR="${OUTPUT_ROOT}/${EXP}/${DS}/${BACKBONE}/${PR}/seed${SEED}"
    printf '%d\t%s\t%s\t%s\n' "${idx}" "${DS}" "${PR}" "${RUN_DIR}" >> "${GRID_FILE}"
    idx=$((idx + 1))
  done
done
N="${idx}"

echo "===== experiment ${EXP}  seed=${SEED}  (${N} tasks, throttle ${THROTTLE}) ====="
echo "datasets: ${DATASETS}   priors: ${PRIORS}   backbone: ${BACKBONE}"
echo "tree:     ${OUTPUT_ROOT}/${EXP}/"
echo "logs:     ${LOG_DIR}/"
column -t "${GRID_FILE}" | sed 's/^/  /'

# browsable from the project root as ./out
ln -sfn "${OUTPUT_ROOT}" "$(cd "${HERE}/../.." && pwd)/out" 2>/dev/null || true

# per-cluster slurm resources (partition / qos / gres / gpu-exclude), as submit.sh.
RES=(--partition="${DAIC_PARTITION}" --gres="${DAIC_GRES_GPU}")
[ -n "${DAIC_QOS_MEDIUM:-}" ] && RES+=(--qos="${DAIC_QOS_MEDIUM}")
[ -n "${DAIC_GPU_EXCLUDE:-}" ] && RES+=(--exclude="${DAIC_GPU_EXCLUDE}")

EXPORTS="ALL,EXP=${EXP},SEED=${SEED},BACKBONE=${BACKBONE},GRID_FILE=${GRID_FILE}"
EXPORTS="${EXPORTS},EPOCHS=${EPOCHS}"
EXPORTS="${EXPORTS},DATA_ROOT=${DATA_ROOT},RESULTS_ROOT=${RESULTS_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}"

JOBID="$(sbatch --parsable \
       --array="0-$((N - 1))%${THROTTLE}" \
       --export="${EXPORTS}" \
       --job-name="${EXP}" \
       --output="${LOG_DIR}/%A_%a.out" \
       --error="${LOG_DIR}/%A_%a.err" \
       "${RES[@]}" "$@" "${HERE}/submit_experiment.sbatch")"

echo
echo "submitted ArrayJobID=${JOBID}"
awk -F'\t' -v jid="${JOBID}" -v ld="${LOG_DIR}" \
  '{printf "  task %s  %-9s %-10s  log=%s/%s_%s.out\n", $1, $2, $3, ld, jid, $1}' "${GRID_FILE}"
echo "watch:  squeue --me   |   bash scripts/daic/watch_jobs.sh --follow"
echo "tail:   tail -f ${LOG_DIR}/${JOBID}_<task>.out"
