#!/usr/bin/env bash
# Submit the faithful DiffSTG baseline (their code, their protocol) for one or
# more of our datasets. Mirrors submit.sh's per-cluster partition/qos/gres
# injection and pre-creates the slurm log dir, but is self-contained (the DiffSTG
# baseline is a separate codebase from the main pipeline).
#
#   scripts/daic/submit_diffstg.sh                     # all six, seed 43
#   scripts/daic/submit_diffstg.sh pems08f metrla      # a subset
#   SEED=43 EXP=diffstg_ref scripts/daic/submit_diffstg.sh
#   TIME=23:59:59 scripts/daic/submit_diffstg.sh pemsbay   # heavy one, more walltime
#
# env: SEED (43), EXP (diffstg_ref), TIME (sbatch --time override),
#      BATCH_SIZE/N/HIDDEN/... (forwarded to the sbatch as faithful-default overrides).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/env.sh"
# shellcheck source=/dev/null
source "${HERE}/_gpu.sh"

DATASETS=("$@")
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(metrla pemsbay pems04f pems04s pems08f pems08s)
SEED="${SEED:-43}"
EXP="${EXP:-diffstg_ref}"
# longer-running than a smoke: prefer the medium qos if the cluster has one.
QOS="${DAIC_QOS_MEDIUM:-${DAIC_QOS_SHORT:-}}"

# per-cluster slurm resources (same logic as submit.sh).
GRES="$(gpu_blackwell_gres || true)"; GRES="${GRES:-${DAIC_GRES_GPU}}"
RES=(--partition="${DAIC_PARTITION}" --gres="${GRES}")
[ -n "${QOS}" ] && RES+=(--qos="${QOS}")
[ -n "${DAIC_GPU_EXCLUDE:-}" ] && RES+=(--exclude="${DAIC_GPU_EXCLUDE}")
[ -n "${TIME:-}" ] && RES+=(--time="${TIME}")

# forward faithful-default hyperparam overrides to the job env if set.
EXPORTS="ALL,SEED=${SEED},EXP=${EXP}"
for k in N SAMPLE_STEPS HIDDEN NSAMPLES BATCH_SIZE LR BETA_END; do
  v="${!k:-}"; [ -n "${v}" ] && EXPORTS="${EXPORTS},${k}=${v}"
done

LOGDIR="logs/${EXP}/diffstg"
mkdir -p "${LOGDIR}"

for DS in "${DATASETS[@]}"; do
  RUN_NAME="diffstg_${DS}_seed${SEED}"
  echo "─── submitting ${RUN_NAME} ─────────────────────────────────────"
  echo "slurm log: ${LOGDIR}/${RUN_NAME}_<jobid>.{out,err}"
  echo "results:   \${DAIC_DATA_DIR}/diffstg/${EXP}/DiffSTG.csv  (+ forecast pkl)"
  echo "──────────────────────────────────────────────────────────────────"
  set -x
  sbatch --job-name="${RUN_NAME}" \
         --export="${EXPORTS},DATASET=${DS}" \
         --output="${LOGDIR}/${RUN_NAME}_%j.out" \
         --error="${LOGDIR}/${RUN_NAME}_%j.err" \
         "${RES[@]}" scripts/daic/submit_diffstg_baseline.sbatch
  set +x
done
