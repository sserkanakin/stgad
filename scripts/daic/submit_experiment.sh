#!/usr/bin/env bash
# ONE command per experiment round: submit a slurm ARRAY where each task is the
# complete lifecycle for one (dataset, prior) cell —
#   train (per-epoch val-loss selection, EM step-sweep diagnostic every 10 epochs)
#   → auto-pick the reverse step count on best.pt (fewest within 1% of min CRPS)
#   → FULL val-set eval at that count (val_results.json + val_summary.txt).
# the test split is NOT touched (separate decision later).
#
#   # pilot (4 arms on metrla):
#   DATASETS=metrla EXP=20260708_r2 scripts/daic/submit_experiment.sh 42
#   # everything (4 arms x 6 datasets = 24 tasks):
#   EXP=20260708_r2 scripts/daic/submit_experiment.sh 42
#   # anything after the seed passes straight to sbatch (e.g. --time=30:00:00)
#
# outputs land per arm: ${OUTPUT_ROOT}/${EXP}/<arm>/<ds>/gwn/<prior>/seed<seed>/{train,eval}
# logs:      ${OUTPUT_ROOT}/logs/${EXP}/<ArrayJobID>_<task>.out (+ copied into the run dir)
# manifest:  ${OUTPUT_ROOT}/logs/${EXP}/grid_seed<seed>.tsv  (task -> ds/arm/prior/cond/norm/run_dir)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/env.sh"

SEED="${1:?usage: EXP=<tag> submit_experiment.sh <seed> [extra sbatch args]}"
shift
: "${EXP:?set EXP=<dated tag>, e.g. EXP=20260706_gadmse (names the results tree + wandb group)}"

# priority order from the user; backbone fixed to gwn for this protocol round.
DATASETS="${DATASETS:-metrla pems04s pems04f pems08s pems08f pemsbay}"
BACKBONE="${BACKBONE:-gwn}"
THROTTLE="${THROTTLE:-8}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-1e-4}"
LOSS="${LOSS:-snr}"          # snr (GAD's rate-weighted mse, default) or mse
# arms: <name>:<prior>:<cond>:<norm>. each arm gets its own nested exp tree
# (${EXP}/<name>/...) and wandb runs, so kron arms never collide on a run dir.
#   kron_pastG   kronecker + past-residual conditioning + global norm
#   kron_pastPN  kronecker + past-residual conditioning + per-node norm
#   kron_base    kronecker, forecast-only conditioning (continuity w/ 20260707)
#   iso_pastG    isotropic + past-residual conditioning (prior ablation)
ARMS="${ARMS:-kron_pastG:kronecker:past:global kron_pastPN:kronecker:past:pernode kron_base:kronecker:base:global iso_pastG:isotropic:past:global}"

# reads from the data share; ALL writes into one dated tree on windowshome.
export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"

LOG_DIR="${OUTPUT_ROOT}/logs/${EXP}"
mkdir -p "${LOG_DIR}"
# one manifest PER SUBMISSION (datasets in the name): tasks read the grid at
# runtime, so a shared filename would be clobbered by the next submission
# (e.g. one array per dataset) while earlier tasks still sit in the queue.
short_ds() {  # squeue-friendly dataset codes (same map as the sbatch rename)
  case "$1" in
    metrla) echo mt ;; pemsbay) echo bay ;;
    pems04f) echo 04f ;; pems04s) echo 04s ;;
    pems08f) echo 08f ;; pems08s) echo 08s ;;
    *) echo "$1" ;;
  esac
}
DS_TAG="$(for d in ${DATASETS}; do short_ds "$d"; done | paste -sd- -)"
GRID_FILE="${LOG_DIR}/grid_seed${SEED}_${DS_TAG}.tsv"
: > "${GRID_FILE}"
idx=0
for DS in ${DATASETS}; do
  for ARM in ${ARMS}; do
    IFS=':' read -r NAME PR COND NORM <<< "${ARM}"
    RUN_DIR="${OUTPUT_ROOT}/${EXP}/${NAME}/${DS}/${BACKBONE}/${PR}/seed${SEED}"
    printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${idx}" "${DS}" "${NAME}" "${PR}" "${COND}" "${NORM}" "${RUN_DIR}" >> "${GRID_FILE}"
    idx=$((idx + 1))
  done
done
N="${idx}"

echo "===== experiment ${EXP}  seed=${SEED}  (${N} tasks, throttle ${THROTTLE}) ====="
echo "datasets: ${DATASETS}   backbone: ${BACKBONE}   lr=${LR}   loss=${LOSS}"
echo "arms:     ${ARMS}"
echo "tree:     ${OUTPUT_ROOT}/${EXP}/<arm>/"
echo "logs:     ${LOG_DIR}/"
column -t "${GRID_FILE}" | sed 's/^/  /'

# browsable from the project root as ./out
ln -sfn "${OUTPUT_ROOT}" "$(cd "${HERE}/../.." && pwd)/out" 2>/dev/null || true

# per-cluster slurm resources. hard-pin Blackwell (GPU_PREFER=blackwell, the default) via the
# rtx_pro_6000 gres TYPE; else the cluster default any-gpu gres. (an explicit --gres after the
# seed still overrides this, since it comes later on the sbatch line.)
source "${HERE}/_gpu.sh"
GRES="$(gpu_blackwell_gres || true)"; GRES="${GRES:-${DAIC_GRES_GPU}}"
RES=(--partition="${DAIC_PARTITION}" --gres="${GRES}")
[ -n "${DAIC_QOS_MEDIUM:-}" ] && RES+=(--qos="${DAIC_QOS_MEDIUM}")
[ -n "${DAIC_GPU_EXCLUDE:-}" ] && RES+=(--exclude="${DAIC_GPU_EXCLUDE}")

EXPORTS="ALL,EXP=${EXP},SEED=${SEED},BACKBONE=${BACKBONE},GRID_FILE=${GRID_FILE}"
EXPORTS="${EXPORTS},EPOCHS=${EPOCHS},LR=${LR},LOSS=${LOSS}"
EXPORTS="${EXPORTS},DATA_ROOT=${DATA_ROOT},RESULTS_ROOT=${RESULTS_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}"
# optional light final-val: single EM/anc step values instead of the full sweep.
[ -n "${FINAL_STEPS_LIST:-}" ] && EXPORTS="${EXPORTS},FINAL_STEPS_LIST=${FINAL_STEPS_LIST}"
[ -n "${FINAL_ANC_STEPS:-}" ]  && EXPORTS="${EXPORTS},FINAL_ANC_STEPS=${FINAL_ANC_STEPS}"
# GAD-style γ for the raw-signal arm (Σ_S=(L+γI)^-1); unset -> load_prior default (0.8).
[ -n "${GAMMA:-}" ] && EXPORTS="${EXPORTS},GAMMA=${GAMMA}"
# temporal ridge ρ for the raw arm (tames stiff empirical Σ_T); unset -> default (0.1).
[ -n "${SHRINK_T:-}" ] && EXPORTS="${EXPORTS},SHRINK_T=${SHRINK_T}"
# temporal factor 'laplacian' (product graph) + its γ_t for the raw arm.
[ -n "${TEMPORAL:-}" ] && EXPORTS="${EXPORTS},TEMPORAL=${TEMPORAL}"
[ -n "${GAMMA_T:-}" ]  && EXPORTS="${EXPORTS},GAMMA_T=${GAMMA_T}"
# spatial graph (road/correlation/hybrid) + winsor for the predict-refine robust-prior arm.
[ -n "${SPATIAL_GRAPH:-}" ] && EXPORTS="${EXPORTS},SPATIAL_GRAPH=${SPATIAL_GRAPH}"
[ -n "${WINSOR:-}" ]        && EXPORTS="${EXPORTS},WINSOR=${WINSOR}"

JOBID="$(sbatch --parsable \
       --array="0-$((N - 1))%${THROTTLE}" \
       --export="${EXPORTS}" \
       --job-name="${DS_TAG}-${EXP}" \
       --output="${LOG_DIR}/%A_%a.out" \
       --error="${LOG_DIR}/%A_%a.err" \
       "${RES[@]}" "$@" "${HERE}/submit_experiment.sbatch")"

echo
echo "submitted ArrayJobID=${JOBID}"
awk -F'\t' -v jid="${JOBID}" -v ld="${LOG_DIR}" \
  '{printf "  task %s  %-9s %-12s  log=%s/%s_%s.out\n", $1, $2, $3, ld, jid, $1}' "${GRID_FILE}"
echo "watch:  squeue --me   |   bash scripts/daic/watch_jobs.sh --follow"
echo "tail:   tail -f ${LOG_DIR}/${JOBID}_<task>.out"
