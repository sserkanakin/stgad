#!/bin/bash
# Submit the FULL train→eval chain for one or more datasets, one seed each.
# Training uses submit_train_diffusion.sbatch defaults (200 epochs, medium QoS,
# 24h); eval uses submit_generate_samples.sbatch defaults (K=100, 100 reverse
# steps, test split, short QoS) and runs the full evaluate/diagnostics/figures
# chain. Each eval is chained afterok: its training job.
#
# Run on the DAIC login node, AFTER the smoke pipeline has passed:
#   cd ~/research-tud && git pull
#   bash scripts/daic/submit_full_pipeline.sh "metrla pems08f pemsbay"
#
# The eval auto-discovers the real checkpoint and ignores any leftover smoke one
# (see submit_generate_samples.sbatch), so no manual cleanup is needed.
set -euo pipefail

DATASETS="${1:?usage: submit_full_pipeline.sh \"ds1 ds2 ...\" [seed]}"
SEED="${2:-0}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

for ds in ${DATASETS}; do
  train_id=$(DATASET="${ds}" SEED="${SEED}" \
    sbatch --parsable --export=ALL \
      --job-name="diff-${ds}-s${SEED}" \
      scripts/daic/submit_train_diffusion.sbatch)
  echo "[${ds}] train : ${train_id}"

  eval_id=$(DATASET="${ds}" SEED="${SEED}" SPLIT=test \
    sbatch --parsable --export=ALL --dependency=afterok:"${train_id}" \
      --job-name="eval-${ds}-s${SEED}" \
      scripts/daic/submit_generate_samples.sbatch)
  echo "[${ds}] eval  : ${eval_id}  (starts after ${train_id} succeeds)"
done

echo
echo "Submitted. Results land in results/eval/<dataset>/ as each eval finishes."
echo "Pull them to your Mac with:  bash scripts/daic/sync_from_daic.sh"
