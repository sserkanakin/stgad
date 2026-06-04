#!/usr/bin/env bash
# submit a pipeline stage to slurm, making sure the log directory exists first.
#
# slurm opens the --output file BEFORE the job body runs and does NOT create its
# parent dir, so if logs/<stage>/ is missing the job fails instantly with no log
# (the mkdir inside the sbatch is too late). this wrapper mkdirs it, names the
# job so logs + squeue + wandb all share the run name, and passes DATASET / SEED
# (and EPOCHS / PRIOR if set) through.
#
#   scripts/daic/submit.sh backbone metrla 0
#   EPOCHS=2 scripts/daic/submit.sh backbone metrla 0 --time=00:15:00
#   scripts/daic/submit.sh residuals metrla 0
#   PRIOR=kronecker scripts/daic/submit.sh diffusion metrla 0 --time=12:00:00
#
# anything after the seed is passed straight to sbatch (e.g. --time, --gres).
set -euo pipefail

STAGE="${1:?usage: submit.sh <backbone|residuals|diffusion> <dataset> <seed> [sbatch args]}"
DATASET="${2:?dataset}"
SEED="${3:?seed}"
shift 3

case "$STAGE" in
  backbone)  SBATCH=submit_train_backbone.sbatch;    RUN_NAME="backbone_${DATASET}_seed${SEED}" ;;
  residuals) SBATCH=submit_extract_residuals.sbatch; RUN_NAME="residuals_${DATASET}_seed${SEED}" ;;
  diffusion) SBATCH=submit_train_diffusion.sbatch;   RUN_NAME="diffusion_${DATASET}_${PRIOR:-kronecker}_seed${SEED}" ;;
  *) echo "stage must be: backbone | residuals | diffusion" >&2; exit 2 ;;
esac

mkdir -p "logs/${STAGE}"   # the fix: dir exists before slurm opens the log file

EXPORTS="ALL,DATASET=${DATASET},SEED=${SEED}"
[ -n "${EPOCHS:-}" ] && EXPORTS="${EXPORTS},EPOCHS=${EPOCHS}"
[ -n "${PRIOR:-}" ]  && EXPORTS="${EXPORTS},PRIOR=${PRIOR}"

set -x
sbatch --job-name="${RUN_NAME}" --export="${EXPORTS}" "$@" "scripts/daic/${SBATCH}"
