#!/usr/bin/env bash
# submit a pipeline stage to slurm, making sure the log directory exists first
# and injecting the per-cluster partition / qos / gres.
#
# slurm opens the --output file BEFORE the job body runs and does NOT create its
# parent dir, so if logs/<stage>/ is missing the job fails instantly with no log
# (the mkdir inside the sbatch is too late). this wrapper mkdirs it, names the
# job so logs + squeue + wandb all share the run name, passes DATASET / SEED
# (and EPOCHS / PRIOR if set) through, and adds the partition/qos/gres that the
# current cluster needs (stable and experimental DAIC differ — see clusters.sh).
#
#   scripts/daic/submit.sh smoke metrla
#   scripts/daic/submit.sh backbone metrla 0
#   EPOCHS=2 scripts/daic/submit.sh backbone metrla 0 --time=00:15:00
#   scripts/daic/submit.sh residuals metrla 0
#   PRIOR=kronecker scripts/daic/submit.sh diffusion metrla 0 --time=12:00:00
#
# anything after the positional args is passed straight to sbatch and OVERRIDES
# the injected defaults (e.g. --time, or --gres to pin a specific gpu type).
set -euo pipefail

# resolve the cluster config (partition/qos/gres) from env.sh.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/env.sh"

STAGE="${1:?usage: submit.sh <smoke|backbone|residuals|diffusion|eval> <dataset> [seed] [sbatch args]}"
DATASET="${2:?dataset}"

# qos tier per stage (empty string on clusters without a qos, e.g. exp).
QOS="${DAIC_QOS_SHORT}"
case "$STAGE" in
  smoke)
    SBATCH=submit_smoke.sbatch;            RUN_NAME="smoke_${DATASET}"
    shift 2 ;;
  backbone)
    SEED="${3:?seed}"; SBATCH=submit_train_backbone.sbatch
    RUN_NAME="backbone_${DATASET}_seed${SEED}"; shift 3 ;;
  residuals)
    SEED="${3:?seed}"; SBATCH=submit_extract_residuals.sbatch
    RUN_NAME="residuals_${DATASET}_seed${SEED}"; shift 3 ;;
  diffusion)
    SEED="${3:?seed}"; SBATCH=submit_train_diffusion.sbatch
    RUN_NAME="diffusion_${DATASET}_${PRIOR:-kronecker}_seed${SEED}"
    QOS="${DAIC_QOS_MEDIUM}"; shift 3 ;;
  eval)
    SEED="${3:?seed}"; SBATCH=submit_eval.sbatch
    RUN_NAME="eval_${DATASET}_${PRIOR:-kronecker}_seed${SEED}"; shift 3 ;;
  *) echo "stage must be: smoke | backbone | residuals | diffusion | eval" >&2; exit 2 ;;
esac

# per-run log subdir so resubmits of the same run group together instead of
# piling up flat in logs/<stage>/. slurm opens the --output file before the job
# body, so the dir must exist now (the mkdir inside the sbatch is too late).
mkdir -p "logs/${STAGE}/${RUN_NAME}"

EXPORTS="ALL,DATASET=${DATASET}"
[ "${STAGE}" != "smoke" ] && EXPORTS="${EXPORTS},SEED=${SEED}"
[ -n "${EPOCHS:-}" ] && EXPORTS="${EXPORTS},EPOCHS=${EPOCHS}"
[ -n "${PRIOR:-}" ]  && EXPORTS="${EXPORTS},PRIOR=${PRIOR}"

# per-cluster slurm resources. partition always; qos only if the cluster has
# one; gres = any gpu by default (pass --gres after the args to pin a type).
RES=(--partition="${DAIC_PARTITION}" --gres="${DAIC_GRES_GPU}")
[ -n "${QOS}" ] && RES+=(--qos="${QOS}")
# DAIC has no OR-constraint for gpu type, so to use "any working gpu" we ask for
# a generic gpu and exclude the nodes whose card is incompatible (rtx pro 6000 /
# blackwell sm_120 crashes our py-torch). DAIC_GPU_EXCLUDE is set in clusters.sh.
[ -n "${DAIC_GPU_EXCLUDE:-}" ] && RES+=(--exclude="${DAIC_GPU_EXCLUDE}")

set -x
sbatch --job-name="${RUN_NAME}" --export="${EXPORTS}" \
       --output="logs/${STAGE}/${RUN_NAME}/%j.out" \
       --error="logs/${STAGE}/${RUN_NAME}/%j.err" \
       "${RES[@]}" "$@" "scripts/daic/${SBATCH}"
