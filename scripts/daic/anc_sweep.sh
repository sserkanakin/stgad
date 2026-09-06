#!/usr/bin/env bash
# Ancestral-sampler resolution sweep: submit driver.
#
# THE QUESTION. Is the structured prior's under-dispersion a property of the prior,
# or of where the sampler stops?
#
#   reverse_sample_ancestral walks a geometric grid c_0 -> c_lo and returns the
#   denoiser's R0_hat at c_lo, discarding the noise still present. The
#   variance-weighted fraction still un-denoised there,
#
#       f(c_lo) = sum_m var_m (1 - e^{-2 theta_m c_lo}) / sum_m var_m
#
#   at the locked c_lo = 0.02 is 3.15% for the isotropic control in every cell
#   (theta = 0.8 flat) and 19.0-28.2% for the structured prior, also in every
#   cell. Training draws go to 8e-4, where the figures are 0.13% and 0.9-2.4%.
#   So the locked protocol resolves the two arms to fidelities differing by 6-9x.
#
#   Local probe (pems08f, 256 windows, k=24, seed 42): lowering c_lo to 8e-4
#   moves iso's coverage <1 point on both gcn and gwn and emp's by +4.8/+4.9,
#   taking emp past iso on coverage, Winkler and energy.
#
# WHAT THIS SUBMITS. Eval only, on saved best_crps.pt checkpoints -- no training.
# Grid = C_LOS x STEPS. "default" reproduces the locked protocol exactly and is
# skipped when its json already exists, so headline numbers can never be clobbered.
#
# USAGE (run ON the cluster)
#   DRY=1 scripts/daic/anc_sweep.sh                    # preview
#   scripts/daic/anc_sweep.sh                          # the pilot (default CELLS)
#   SHARD=a scripts/daic/anc_sweep.sh                  # half the jobs (two accounts)
#   CELLS_SPEC=full scripts/daic/anc_sweep.sh          # all 18 histcond cells
#   anything after the script name passes straight through to sbatch.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
export WANDB_MODE=disabled

C_LOS="${C_LOS:-default 8e-4}"
STEPS="${STEPS:-50 200}"
K="${K:-100}"
SEEDS="${SEEDS:-42}"
CELLS_PER_JOB="${CELLS_PER_JOB:-2}"
SHARD="${SHARD:-all}"
DRY="${DRY:-0}"
CELLS_SPEC="${CELLS_SPEC:-pilot}"

# pilot: the stiffness ladder on one flow benchmark plus one cross-check. These four
# span 1/theta_max from 0.005 (mlp, deep below the floor) to 0.062 (gwn, above it),
# so if the effect is a floor artefact the pilot must show it varying monotonically
# with stiffness -- and if it does not, the prior explanation stands.
PILOT_CELLS="pems08f/mlp pems08f/gcn pems08f/gwn pems04f/gcn"
FULL_DATASETS="${FULL_DATASETS:-pems08f pems08s pems04f pems04s metrla pemsbay}"
FULL_BACKBONES="${FULL_BACKBONES:-mlp gcn gwn}"

case "${CELLS_SPEC}" in
  pilot) pairs="${PILOT_CELLS}" ;;
  full)  pairs=""
         for ds in ${FULL_DATASETS}; do for bb in ${FULL_BACKBONES}; do
           pairs="${pairs} ${ds}/${bb}"
         done; done ;;
  *)     pairs="${CELLS_SPEC}" ;;      # explicit "ds/bb ds/bb ..." list
esac

# both arms, always: the whole point is the iso-vs-kron comparison at matched fidelity.
ARMS="${ARMS:-iso kron}"
COND="${COND:-hist}"
cells=()
for p in ${pairs}; do
  ds="${p%%/*}"; bb="${p##*/}"
  for sd in ${SEEDS}; do for arm in ${ARMS}; do
    cells+=("${arm}|${ds}|${bb}|${sd}|${COND}")
  done; done
done

n=${#cells[@]}
jobs=()
for ((i=0; i<n; i+=CELLS_PER_JOB)); do
  chunk=("${cells[@]:i:CELLS_PER_JOB}")
  jobs+=("$(IFS=';'; echo "${chunk[*]}")")
done
sel=()
for ((j=0; j<${#jobs[@]}; j++)); do
  case "${SHARD}" in
    a)   [ $((j % 2)) -eq 0 ] && sel+=("${jobs[j]}") ;;
    b)   [ $((j % 2)) -eq 1 ] && sel+=("${jobs[j]}") ;;
    all) sel+=("${jobs[j]}") ;;
    *)   echo "SHARD must be a, b or all"; exit 1 ;;
  esac
done
echo "=== ${n} cells (${CELLS_SPEC}) -> ${#jobs[@]} jobs; shard '${SHARD}' takes ${#sel[@]} ==="
echo "    grid: c_lo in {${C_LOS}} x steps in {${STEPS}}, k=${K}"

source "${HERE}/_gpu.sh" 2>/dev/null || true
GRES="$(gpu_blackwell_gres 2>/dev/null || true)"; GRES="${GRES:-${DAIC_GRES_GPU}}"
QOS=""; [ -n "${DAIC_QOS_SHORT:-${DAIC_QOS_MEDIUM:-}}" ] && QOS="--qos=${DAIC_QOS_SHORT:-${DAIC_QOS_MEDIUM}}"
LOG="${OUTPUT_ROOT}/logs/20260812_ancsweep"; mkdir -p "${LOG}"

njob=0
for spec in "${sel[@]}"; do
  first="${spec%%;*}"
  IFS='|' read -r a d b s c <<< "${first}"
  tag="${a}-${d}-${b}-s${s}"
  njob=$((njob+1))
  if [ "${DRY}" = "1" ]; then
    echo "  [dry] job ${njob}: CELLS='${spec}'"
    continue
  fi
  jid=$(sbatch --parsable --partition="${DAIC_PARTITION}" ${QOS} --gres="${GRES}" \
        ${DAIC_GPU_EXCLUDE:+--exclude=$DAIC_GPU_EXCLUDE} \
        --job-name="anc-${tag}" \
        --output="${LOG}/%j_${tag}.out" \
        --export=ALL,CELLS="${spec}",C_LOS="${C_LOS}",STEPS="${STEPS}",K="${K}" \
        "$@" "${HERE}/anc_sweep.sbatch")
  echo "  job ${njob} (${jid}): ${spec}"
done
if [ "${DRY}" = "1" ]; then
  echo "=== DRY RUN — NOTHING SUBMITTED. ${njob} jobs would go to shard '${SHARD}'."
else
  echo "=== submitted ${njob} jobs; logs in ${LOG} ==="
  echo "    aggregate with: python scripts/analysis/anc_sweep_report.py"
fi
