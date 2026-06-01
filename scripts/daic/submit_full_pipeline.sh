#!/bin/bash
# Full train→eval chain for one or more datasets, one seed each. Training uses
# submit_train_diffusion.sbatch's #SBATCH defaults (200 epochs, medium QoS, 24h);
# eval uses submit_generate_samples.sbatch's defaults (K=100, 100 reverse steps,
# test split) and runs the full evaluate/diagnostics/figures chain. Each eval is
# chained --dependency=afterok: its training job. Submits with the project's
# standard `sbatch --export=KEY=VAL,...` convention.
#
# Run on the DAIC login node, AFTER the smoke pipeline has passed:
#   cd ~/research-tud && git pull
#   bash scripts/daic/submit_full_pipeline.sh "metrla pems08f pemsbay"
#
# Eval auto-discovers the real checkpoint and ignores any leftover smoke one
# (see submit_generate_samples.sbatch), so no manual cleanup is needed.
set -euo pipefail

DATASETS="${1:?usage: submit_full_pipeline.sh \"ds1 ds2 ...\" [seed]}"
SEED="${2:-0}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

# Slurm opens each job's --output file BEFORE the job script runs, so the log
# dirs must already exist at submission time (the mkdir inside the .sbatch body
# is too late for the first-ever run). Create them here on the login node.
mkdir -p logs/train-diffusion logs/eval-samples

for ds in ${DATASETS}; do
  train_id=$(sbatch --parsable \
    --job-name="diff-${ds}-s${SEED}" \
    --export="DATASET=${ds},SEED=${SEED}" \
    scripts/daic/submit_train_diffusion.sbatch)
  echo "[${ds}] train : ${train_id}"

  eval_id=$(sbatch --parsable --dependency=afterok:"${train_id}" \
    --job-name="eval-${ds}-s${SEED}" \
    --export="DATASET=${ds},SEED=${SEED},SPLIT=test" \
    scripts/daic/submit_generate_samples.sbatch)
  echo "[${ds}] eval  : ${eval_id}  (starts after ${train_id} succeeds)"
done

echo
echo "Submitted. Results land in results/eval/<dataset>/ as each eval finishes."
echo "Pull them to your Mac with:  bash scripts/daic/sync_from_daic.sh"
