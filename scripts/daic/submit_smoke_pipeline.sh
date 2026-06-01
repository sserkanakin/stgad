#!/bin/bash
# Submit a fast, low-resource SMOKE train→eval chain for one dataset, to confirm
# both stages run on DAIC before committing GPU hours to full runs. The work
# itself is tiny (1 epoch / few batches; K=8, 20 reverse steps), and `short` QoS
# + short walltime schedule it at high priority.
#
# Run on the DAIC login node:
#   cd ~/research-tud && git fetch && git checkout eval-suite && git pull
#   bash scripts/daic/submit_smoke_pipeline.sh metrla
#
# The eval job is chained with afterok: so it starts only if training succeeds.
# Watch with:  bash scripts/daic/watch_jobs.sh
set -euo pipefail

DATASET="${1:?usage: submit_smoke_pipeline.sh <dataset> [seed]}"
SEED="${2:-0}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

train_id=$(DATASET="${DATASET}" SEED="${SEED}" EXTRA_FLAGS="--smoke --fresh" \
  sbatch --parsable --export=ALL \
    --job-name="smoke-train-${DATASET}" --qos=short --time=00:15:00 \
    --cpus-per-task=2 --mem=16G \
    scripts/daic/submit_train_diffusion.sbatch)
echo "[${DATASET}] smoke train  : ${train_id}"

eval_id=$(DATASET="${DATASET}" SEED="${SEED}" SPLIT=val K=8 N_STEPS=20 EXTRA_FLAGS="--smoke" \
  sbatch --parsable --export=ALL --dependency=afterok:"${train_id}" \
    --job-name="smoke-eval-${DATASET}" --qos=short --time=00:20:00 \
    --cpus-per-task=2 --mem=16G \
    scripts/daic/submit_generate_samples.sbatch)
echo "[${DATASET}] smoke eval   : ${eval_id}  (starts after ${train_id} succeeds)"

echo
echo "Both submitted. When they finish, check:"
echo "  results/eval/${DATASET}/comparison_val.md      (table renders)"
echo "  results/eval/${DATASET}/diagnostics.json       (oracle + forward marginal PASS)"
echo "  results/eval/${DATASET}/figures/               (8 PNGs)"
