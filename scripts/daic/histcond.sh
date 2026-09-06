#!/usr/bin/env bash
# Conditioning-axis campaign orchestrator.
#
# WHAT THIS RUNS
#   The question: what does the diffusion gain from being told the observed history
#   on top of the point forecast?
#
#     cond=fo    C = Yhat_z            forecast only
#     cond=hist  C = [X_z | Yhat_z]    observed history window + forecast
#
#   Both conditionings are (B,N,12) and (B,N,24) respectively, so the GCNN denoiser
#   takes them as 1 vs 2 conditioning CHANNELS -- 17,305 vs 17,449 params. Nothing
#   else changes: same denoiser, loss, schedule, sampler, normalization, backbone
#   residuals (residual seed pinned to 42) and locked test protocol.
#
#   Two prior arms:
#     iso   scale-matched white noise (the matched control)
#     kron  Sigma = Sigma_T (x) Sigma_S: Sigma_T fit from the residuals (per-horizon
#           variance on the diagonal, pearson correlation off it, ridge rho),
#           Sigma_S = unit-diagonal (L_road + beta*all-to-all + gamma I)^-1 with
#           (beta, gamma) matched on the train residuals (--global-edge match).
#
#   IMPORTANT (2026-07-28): the 20260721_ladder "-past" arm is NOT this experiment.
#   --cond-past used to fall back silently to the previous window's RESIDUAL field
#   when a cache had no history channel, and no cache ever had one, so that whole
#   arm trained C = [R_prev | Yhat]. It is retired. See commit 1e253cc.
#
# STORAGE
#   The residual caches are ~10G against an 8G windowshome quota, so DAIC rotates
#   datasets. This script never builds or deletes caches -- it reports what is
#   present and refuses to run cells whose cache is absent. Run it once per phase:
#     DATASETS="pems08f pems08s pemsbay"        scripts/daic/histcond.sh   # phase A
#     DATASETS="metrla pems04f pems04s"         scripts/daic/histcond.sh   # phase B
#
# USAGE (run ON the cluster)
#   DRY=1 scripts/daic/histcond.sh                    # preview, submit nothing
#   scripts/daic/histcond.sh                          # submit this shard
#   SHARD=a scripts/daic/histcond.sh                  # half the jobs (two accounts)
#   CONDS=fo BACKBONES=gcn DATASETS="metrla pemsbay" scripts/daic/histcond.sh
#                                                     # the forecast-only gap-fill
#   anything after the script name passes straight through to sbatch.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
export WANDB_MODE=disabled

EXP="${EXP:-20260728_histcond}"
ARMS="${ARMS:-iso kron}"
CONDS="${CONDS:-hist}"
DATASETS="${DATASETS:-metrla pemsbay pems04f pems04s pems08f pems08s}"
BACKBONES="${BACKBONES:-mlp gcn gwn}"
SEEDS="${SEEDS:-1 42}"
RESIDUAL_SEED="${RESIDUAL_SEED:-42}"
CELLS_PER_JOB="${CELLS_PER_JOB:-2}"
SHARD="${SHARD:-all}"
DRY="${DRY:-0}"

if [ -z "${DAIC_PYTHON:-}" ]; then
  echo "  [env] DAIC_PYTHON unset — sourcing activate.sh"
  # shellcheck source=/dev/null
  source "${HERE}/activate.sh"
fi
PY="${DAIC_PYTHON:?activate.sh did not set DAIC_PYTHON — check the module stack}"

# ── 1. inputs: residual caches (per ds/bb) + processed splits (the history) ───
echo "=== 1. inputs ==="
present_cells=""; missing_cells=""
for ds in ${DATASETS}; do
  # the history channel is read from the processed split, not the residual cache,
  # so that file must be there too (it always is -- it is what built everything).
  for sp in train val test; do
    [ -s "${DATA_ROOT}/processed/${ds}/${sp}.npz" ] || {
      echo "  MISSING processed split ${ds}/${sp} — cannot build the history conditioning"; exit 1; }
  done
  for bb in ${BACKBONES}; do
    n=0
    for sp in train val test; do
      [ -s "${RESULTS_ROOT}/residuals/${ds}/${bb}/seed${RESIDUAL_SEED}/${sp}.npz" ] && n=$((n+1))
    done
    if [ "${n}" = "3" ]; then present_cells="${present_cells}${ds}/${bb} "
    else missing_cells="${missing_cells}${ds}/${bb}(${n}/3) "; fi
  done
done
echo "  residual caches present: ${present_cells:-none}"
if [ -n "${missing_cells}" ]; then
  echo "  residual caches ABSENT:  ${missing_cells}"
  echo
  echo "  DAIC rotates datasets through the 8G quota, so this usually means the dataset"
  echo "  is simply not on this machine. rsync it from the Mac, or restrict the run:"
  echo "    DATASETS=\"<what is here>\" scripts/daic/histcond.sh"
  exit 1
fi

# ── 2. preflight: the conditioning actually resolves, and is what we think ────
# cheap and worth it: this campaign exists because a conditioning field was wrong
# for a week without anyone noticing. Fail before burning 80 GPU-hours.
echo "=== 2. preflight: conditioning shape + content ==="
for ds in ${DATASETS}; do
  for bb in ${BACKBONES}; do
    "${PY}" - "$ds" "$bb" "${RESIDUAL_SEED}" <<'PYEOF' || exit 1
import sys, numpy as np, torch
from src.data.residual_dataset import make_dataset, train_residual_sigma
ds, bb, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
sig = train_residual_sigma(ds, seed, bb, norm="pernode")
fo = make_dataset(ds, "val", seed, bb, sig, cond_past=False, zero_masked=True)
hi = make_dataset(ds, "val", seed, bb, sig, cond_past=True, zero_masked=True)
H = fo.Y_hat.shape[-1]
assert hi.Y_hat.shape[-1] == 2 * H, f"{ds}/{bb}: cond width {hi.Y_hat.shape[-1]} != 2*{H}"
assert torch.equal(hi.Y_hat[..., H:], fo.Y_hat), f"{ds}/{bb}: forecast half differs from the fo arm"
assert torch.equal(hi.R0, fo.R0), f"{ds}/{bb}: target differs between arms"
# the history half must be the observed input window, NOT any residual field
from src.utils import paths
x = np.load(paths.split_npz(ds, "val"))["x"]
d = np.load(paths.residual_npz(ds, "val", seed, bb))
want = (np.ascontiguousarray(x[..., 0].transpose(0, 2, 1)) - float(d["mean"])) / float(d["std"])
assert np.allclose(hi.Y_hat[..., :H].numpy(), want, atol=1e-5), f"{ds}/{bb}: history half is not the observed window"
print(f"  {ds}/{bb}: OK  fo={tuple(fo.Y_hat.shape)}  hist={tuple(hi.Y_hat.shape)}")
PYEOF
  done
done

# ── 3. cell list -> jobs -> shards ───────────────────────────────────────────
# DATASET-major with iso first so any single finished job is already a usable
# within-cell iso/emp comparison.
cells=()
for ds in ${DATASETS}; do for bb in ${BACKBONES}; do for cond in ${CONDS}; do
  for arm in ${ARMS}; do for sd in ${SEEDS}; do
    cells+=("${arm}|${ds}|${bb}|${sd}|${cond}")
  done; done
done; done; done
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
echo "=== 3. ${n} cells -> ${#jobs[@]} jobs (${CELLS_PER_JOB}/job); shard '${SHARD}' takes ${#sel[@]} ==="

# ── 4. submit ────────────────────────────────────────────────────────────────
source "${HERE}/_gpu.sh" 2>/dev/null || true
GRES="$(gpu_blackwell_gres 2>/dev/null || true)"; GRES="${GRES:-${DAIC_GRES_GPU}}"
QOS=""; [ -n "${DAIC_QOS_MEDIUM:-}" ] && QOS="--qos=${DAIC_QOS_MEDIUM}"
LOG="${OUTPUT_ROOT}/logs/${EXP}"; mkdir -p "${LOG}"

njob=0
for spec in "${sel[@]}"; do
  first="${spec%%;*}"
  IFS='|' read -r a d b s c <<< "${first}"
  tag="${a}-${d}-${b}-s${s}-${c}"
  njob=$((njob+1))
  if [ "${DRY}" = "1" ]; then
    echo "  [dry] job ${njob}: CELLS='${spec}'"
    continue
  fi
  jid=$(sbatch --parsable --partition="${DAIC_PARTITION}" ${QOS} --gres="${GRES}" \
        ${DAIC_GPU_EXCLUDE:+--exclude=$DAIC_GPU_EXCLUDE} \
        --job-name="hc-${tag}" \
        --output="${LOG}/%j_${tag}.out" \
        --export=ALL,CELLS="${spec}",EXP="${EXP}",RESIDUAL_SEED="${RESIDUAL_SEED}" \
        "$@" "${HERE}/histcond_cell.sbatch")
  echo "  job ${njob} (${jid}): ${spec}"
done
if [ "${DRY}" = "1" ]; then
  echo "=== DRY RUN — NOTHING WAS SUBMITTED. ${njob} jobs would go to shard '${SHARD}'."
  echo "    Re-run without DRY=1 to actually submit."
else
  echo "=== submitted ${njob} jobs for shard '${SHARD}' (exp ${EXP}/<arm>-<cond>) ==="
  echo "    verify with: squeue -u \$USER"
fi
