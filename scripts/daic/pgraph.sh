#!/usr/bin/env bash
# GTCNN product-graph diffusion campaign orchestrator.
#
# WHAT THIS RUNS
#   5 arms x 6 datasets = 30 cells, on the RAW SIGNAL (no predict-refine, no forecast
#   anywhere): the conditioning is the z-scored history window only. Denoiser is fixed at
#   the 17,305-param GCNN for every cell, so the ONLY thing that varies is the shape of
#   the diffusion forward operator across graph-time modes.
#
#     iso              variance-matched white noise (control)
#     prod_cartesian   GTCNN Cartesian product graph   lam_S + lam_T
#     prod_strong      GTCNN strong product graph      (1+lam_S)(1+lam_T) - 1
#     prod_tensor      GTCNN Kronecker product graph   lam_S * lam_T   (negative control:
#                      N+H-1 modes collapse to theta_min, and the Laplacian's nonzero
#                      diagonal means its GSO is not the tensor product graph's operator)
#     prod_param       GTCNN eq.10 parametric product graph, 4 coefficients fitted offline
#
#   Every product form is rescaled to lam in [0,2], so theta = lam + gamma gives the SAME
#   theta_min, stiffness and GAD mixing number in every cell -- only the shape differs.
#
#   gamma = 0.1 (the low value the heat cascade needs, NOT the residual arm's 0.8).
#   c_0   = 100, DERIVED not guessed: the raw signal's graph-DC x time-DC mode carries
#           ~90-400x its stationary variance, so GAD's 2*c_0*theta_min >= 5 rule leaves
#           the forward far from stationary (measured: 61% of the largest target variance
#           still un-erased at c_0=25). The requirement is c_0 >= 5*ln(80*r); measured
#           r gives 46-52 across all cells, and 100 clears it with room.
#
# USAGE (run ON the cluster, once per account -- 16 jobs total across two accounts):
#   SHARD=a scripts/daic/pgraph.sh          # half the jobs   (e.g. sakin)
#   SHARD=b scripts/daic/pgraph.sh          # the other half  (e.g. dkrylov)
#   SHARD=all DRY=1 scripts/daic/pgraph.sh  # preview everything, submit nothing
# anything after the script name passes straight through to sbatch, same as
# submit_experiment.sh:   SHARD=a scripts/daic/pgraph.sh --nodelist=gpu[36-53]
# DATASETS=... restricts the run to the datasets whose caches are on this machine.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
export WANDB_MODE=disabled

EXP="${EXP:-20260728_pgraph}"
SEED="${SEED:-42}"
GAMMA="${GAMMA:-0.1}"
C0="${C0:-100}"
# matched schedule across ALL arms (see pgraph_cell.sbatch): 'auto' would put iso on ULS
# and the product arms on FCPS, confounding the few-step comparison this campaign exists
# to make. SCHEDULE=auto restores the legacy per-prior choice.
SCHEDULE="${SCHEDULE:-fcps}"
ARMS="${ARMS:-iso prod_cartesian prod_strong prod_tensor prod_param}"
DATASETS="${DATASETS:-metrla pemsbay pems04f pems04s pems08f pems08s}"
CELLS_PER_JOB="${CELLS_PER_JOB:-2}"
SHARD="${SHARD:-all}"
DRY="${DRY:-0}"
# DAIC_PYTHON is exported by activate.sh / _common.sh (the module stack + project venv).
# rawsig.sh requires you to have sourced activate.sh first and dies with a bare
# "parameter null or not set" if you forget; do it here instead so the orchestrator
# works whether or not the caller pre-sourced it.
if [ -z "${DAIC_PYTHON:-}" ]; then
  echo "  [env] DAIC_PYTHON unset — sourcing activate.sh"
  # shellcheck source=/dev/null
  source "${HERE}/activate.sh"
fi
PY="${DAIC_PYTHON:?activate.sh did not set DAIC_PYTHON — check the module stack}"

# ── 1. raw signal caches ─────────────────────────────────────────────────────
echo "=== 1. raw signal caches ==="
# DAIC rotates datasets through the storage quota, so a missing cache usually means the
# dataset simply isn't on this machine — not that it should be rebuilt. Report the split
# clearly and let the caller decide (DATASETS=... to run what is here, or rsync first);
# building blind can blow the quota mid-campaign and strand every job.
missing_ds=""; present_ds=""
for ds in ${DATASETS}; do
  n=0
  for sp in train val test; do
    [ -s "${RESULTS_ROOT}/residuals/${ds}/raw/seed${SEED}/${sp}.npz" ] && n=$((n+1))
  done
  if [ "${n}" = "3" ]; then
    echo "  ${ds}: present (3/3 splits)"; present_ds="${present_ds}${ds} "
  else
    echo "  ${ds}: MISSING (${n}/3 splits)"; missing_ds="${missing_ds}${ds} "
  fi
done
if [ -n "${missing_ds}" ]; then
  echo
  echo "  raw caches absent for: ${missing_ds}"
  echo "  either rsync them from the Mac, or restrict the run to what is here:"
  echo "    DATASETS=\"${present_ds% }\" SHARD=${SHARD} scripts/daic/pgraph.sh"
  [ "${BUILD_MISSING:-0}" = "1" ] || { echo "  (set BUILD_MISSING=1 to build them here — watch the quota)"; exit 1; }
  for ds in ${missing_ds}; do
    echo "  building ${ds}"
    "${PY}" "${HERE}/../build_signal_cache.py" --dataset "${ds}" \
      || { echo "cache build FAILED ${ds}"; exit 1; }
  done
fi

# ── 2. parametric product-graph coefficients (needed before prod_param loads) ─
if echo "${ARMS}" | grep -q prod_param; then
  echo "=== 2. fit parametric product-graph coefficients ==="
  missing=0
  for ds in ${DATASETS}; do
    [ -s "${RESULTS_ROOT}/analysis/product_graph/s_ij_${ds}_raw_seed${SEED}.npy" ] || missing=1
  done
  if [ "${missing}" = "1" ]; then
    # NB --datasets, not --all: --all hardcodes all six and would look for caches this
    # machine may not have (DAIC rotates datasets through the storage quota).
    "${PY}" "${HERE}/../analysis/fit_product_graph.py" --datasets ${DATASETS} \
        --backbone raw --seed "${SEED}" \
      || { echo "parametric fit FAILED"; exit 1; }
  else
    echo "  all coefficients present"
  fi
fi

# ── 3. VERIFICATION GATE — nothing is submitted unless every check passes ─────
echo "=== 3. verification gate (operator / schedule / forward / stationary law) ==="
VLOG="${OUTPUT_ROOT}/logs/${EXP}/verify.txt"
mkdir -p "$(dirname "${VLOG}")"
# the gate takes ~1 min/cell, so a 15-cell run costs ~18 min EVERY invocation — including
# a real submit that directly follows a green DRY=1 preview of the same config, and any
# later resubmit of a failed cell. SKIP_GATE=1 reuses the existing verify.txt when it
# already covers this run's cells and passed. It is a REUSE, not a bypass: a stale or
# narrower log is rejected, so it cannot let an unverified config through.
gate_ok=0
if [ "${SKIP_GATE:-0}" = "1" ] && [ -s "${VLOG}" ] && grep -q "ALL CHECKS PASS" "${VLOG}"; then
  gate_ok=1
  for ds in ${DATASETS}; do for arm in ${ARMS}; do
    grep -q "^=== ${ds} · ${arm} === ALL PASS" "${VLOG}" || { gate_ok=0; miss="${ds}/${arm}"; }
  done; done
  if [ "${gate_ok}" = "1" ]; then
    echo "  SKIP_GATE=1: reusing ${VLOG} (covers every cell in this run, ALL CHECKS PASS)"
  else
    echo "  SKIP_GATE=1 requested but ${VLOG} does not cover ${miss} — re-running the gate"
  fi
fi
if [ "${gate_ok}" != "1" ]; then
  if ! "${PY}" "${HERE}/../verify_product_graph.py" \
        --datasets ${DATASETS} --forms ${ARMS} --seed "${SEED}" 2>&1 | tee "${VLOG}" | tail -30; then
    echo "VERIFICATION ERRORED — not submitting"; exit 1
  fi
  grep -q "ALL CHECKS PASS" "${VLOG}" || { echo "VERIFICATION FAILED — not submitting (see ${VLOG})"; exit 1; }
fi

# ── 4. build the cell list, chunk it into jobs, then shard the JOBS ──────────
# DATASET-major ordering with iso first, so the first job of every dataset is the
# (iso, prod_cartesian) headline pair: any single job that finishes is already a usable
# within-dataset comparison. jobs are then dealt ALTERNATELY to the two shards, so a
# whole account failing still leaves every dataset and every arm represented.
cells=()
for ds in ${DATASETS}; do for arm in ${ARMS}; do cells+=("${arm}|${ds}"); done; done
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
echo "=== 4. ${n} cells -> ${#jobs[@]} jobs (${CELLS_PER_JOB}/job); shard '${SHARD}' takes ${#sel[@]} ==="

# ── 5. submit ────────────────────────────────────────────────────────────────
source "${HERE}/_gpu.sh" 2>/dev/null || true
GRES="$(gpu_blackwell_gres 2>/dev/null || true)"; GRES="${GRES:-${DAIC_GRES_GPU}}"
QOS=""; [ -n "${DAIC_QOS_MEDIUM:-}" ] && QOS="--qos=${DAIC_QOS_MEDIUM}"
LOG="${OUTPUT_ROOT}/logs/${EXP}"; mkdir -p "${LOG}"

njob=0
for spec in "${sel[@]}"; do
  tag="$(echo "${spec%%;*}" | tr '|' '-')"
  njob=$((njob+1))
  if [ "${DRY}" = "1" ]; then
    echo "  [dry] job ${njob}: CELLS='${spec}'"
    continue
  fi
  jid=$(sbatch --parsable --partition="${DAIC_PARTITION}" ${QOS} --gres="${GRES}" \
        ${DAIC_GPU_EXCLUDE:+--exclude=$DAIC_GPU_EXCLUDE} \
        --job-name="pg-${tag}" \
        --output="${LOG}/%j_${tag}.out" \
        --export=ALL,CELLS="${spec}",EXP="${EXP}",SEED="${SEED}",GAMMA="${GAMMA}",C0="${C0}",SCHEDULE="${SCHEDULE}" \
        "$@" "${HERE}/pgraph_cell.sbatch")
  echo "  job ${njob} (${jid}): ${spec}"
done
echo "=== submitted ${njob} jobs for shard '${SHARD}' (exp ${EXP}_<arm>) ==="
