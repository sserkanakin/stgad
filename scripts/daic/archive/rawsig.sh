#!/usr/bin/env bash
# raw-signal diffusion pilot orchestrator. builds the centered signal cache, VALIDATES the
# heat prior (aborts on any FAIL), then submits one GPU job per (dataset,prior,d_h) cell.
# heat-faithful prior = --temporal cartesian (additive graph Laplacian, theta=lambda+gamma,
# spectrum in [0,2]); history-conditioned SignalDenoiser (DiffSTG-style); ancestral eval.
#
#   DATASETS="pems08f metrla" DHS="32 8" scripts/daic/rawsig.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
export WANDB_MODE=disabled

DATASETS="${DATASETS:-pems08f metrla}"
DHS="${DHS:-32 8}"                 # denoiser capacities: 32 = UGnet-scale, 8 = simple
PRIORS="${PRIORS:-kronecker isotropic}"
SEED="${SEED:-42}"
GAMMA="${GAMMA:-0.04}"
EPOCHS="${EPOCHS:-200}"
EXP="${EXP:-20260723_rawsig}"
PY="${DAIC_PYTHON:?}"

echo "=== 1. build centered signal caches ==="
for ds in ${DATASETS}; do
  [ -s "${RESULTS_ROOT}/residuals/${ds}/raw/seed42/train.npz" ] && { echo "  cache ${ds} exists"; continue; }
  "${PY}" "${HERE}/../build_signal_cache.py" --dataset "${ds}" || { echo "cache build FAILED ${ds}"; exit 1; }
done

echo "=== 2. validate heat prior (cartesian, gamma=${GAMMA}) — abort on FAIL ==="
if ! "${PY}" "${HERE}/../validate_signal_prior.py" --datasets ${DATASETS} --gamma "${GAMMA}" 2>&1 | tee /tmp/rawsig_val.txt | tail -20; then
  echo "VALIDATION script errored"; exit 1
fi
grep -q "ALL CHECKS PASS" /tmp/rawsig_val.txt || { echo "PRIOR VALIDATION FAILED — not firing"; exit 1; }

echo "=== 3. submit one GPU job per (dataset,prior,d_h) ==="
source "${HERE}/_gpu.sh" 2>/dev/null || true
GRES="$(gpu_blackwell_gres 2>/dev/null || true)"; GRES="${GRES:-${DAIC_GRES_GPU}}"
QOS=""; [ -n "${DAIC_QOS_MEDIUM:-}" ] && QOS="--qos=${DAIC_QOS_MEDIUM}"
LOG="${OUTPUT_ROOT}/logs/${EXP}"; mkdir -p "${LOG}"
n=0
for ds in ${DATASETS}; do for dh in ${DHS}; do for pr in ${PRIORS}; do
  exp="${EXP}_dh${dh}"
  args="--dataset ${ds} --backbone raw --prior ${pr} --temporal cartesian --gamma ${GAMMA}"
  args="${args} --denoiser ugnet --d-h ${dh} --noise-sampler powerlaw --schedule auto"
  args="${args} --epochs ${EPOCHS} --exp ${exp} --seed ${SEED}"
  jid=$(sbatch --parsable --partition="${DAIC_PARTITION}" ${QOS} --gres="${GRES}" \
        ${DAIC_GPU_EXCLUDE:+--exclude=$DAIC_GPU_EXCLUDE} \
        --job-name="rawsig-${ds}-${pr:0:3}-dh${dh}" \
        --output="${LOG}/%j_${ds}_${pr}_dh${dh}.out" \
        --export=ALL,RUNS="${args}",RECAL_SCRIPT="" \
        "${HERE}/custom_train.sbatch")
  echo "  ${ds} ${pr} d_h=${dh} -> job ${jid}"
  n=$((n+1))
done; done; done
echo "=== submitted ${n} raw-signal cells (exp ${EXP}_dh*) ==="
