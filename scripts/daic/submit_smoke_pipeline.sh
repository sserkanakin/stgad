#!/bin/bash
# Smoke train→eval chain for ONE dataset — fast and low-resource, to confirm both
# stages run on DAIC before committing GPU hours to full runs. Submits with the
# project's standard `sbatch --export=KEY=VAL,...` convention; `short` QoS + short
# walltime schedule it at high priority, and EXTRA_FLAGS=--smoke makes the work
# tiny (1 epoch / few batches; eval K=8, 20 reverse steps).
#
# Run on the DAIC login node:
#   cd ~/research-tud && git pull
#   bash scripts/daic/submit_smoke_pipeline.sh metrla
#
# The eval is chained with --dependency=afterok: so it starts only if training
# succeeds. Watch with:  bash scripts/daic/watch_jobs.sh
set -euo pipefail

DATASET="${1:?usage: submit_smoke_pipeline.sh <dataset> [seed]}"
SEED="${2:-0}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

train_id=$(sbatch --parsable \
  --job-name="smoke-train-${DATASET}" --qos=short --time=00:15:00 --cpus-per-task=2 --mem=16G \
  --export="DATASET=${DATASET},SEED=${SEED},EXTRA_FLAGS=--smoke" \
  scripts/daic/submit_train_diffusion.sbatch)
echo "[${DATASET}] smoke train : ${train_id}"

eval_id=$(sbatch --parsable --dependency=afterok:"${train_id}" \
  --job-name="smoke-eval-${DATASET}" --qos=short --time=00:20:00 --cpus-per-task=2 --mem=16G \
  --export="DATASET=${DATASET},SEED=${SEED},SPLIT=val,K=8,N_STEPS=20,EXTRA_FLAGS=--smoke" \
  scripts/daic/submit_generate_samples.sbatch)
echo "[${DATASET}] smoke eval  : ${eval_id}  (starts after ${train_id} succeeds)"

echo
echo "When they finish, check:"
echo "  results/eval/${DATASET}/comparison_val.md      (table renders, 4 method rows)"
echo "  results/eval/${DATASET}/diagnostics.json       (oracle + forward marginal PASS)"
echo "  results/eval/${DATASET}/figures/               (8 PNGs)"
