#!/usr/bin/env bash
# Submit the backbone -> residuals grid to slurm: every (dataset x backbone x seed)
# cell, chained so residuals start only after its backbone succeeds (afterok).
# NO diffusion/eval (that stage is separate). Backbone training has no wandb.
#
#   source scripts/daic/env.sh
#   scripts/daic/submit_residual_grid.sh                 # full grid (54 cells)
#   DATASETS="pems08f pems08s" scripts/daic/submit_residual_grid.sh   # subset
#   EXP=grid EPOCHS=100 scripts/daic/submit_residual_grid.sh
#
# override DATASETS / BACKBONES / SEEDS / EXP / EPOCHS via env. Each cell submits
# 2 jobs (backbone + dependent residuals), so the full grid is 108 jobs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/env.sh"

DATASETS="${DATASETS:-metrla pemsbay pems04f pems04s pems08f pems08s}"
BACKBONES="${BACKBONES:-gwn gcn mlp}"
SEEDS="${SEEDS:-42 43 44}"
export EXP="${EXP:-grid}"

jobid_of() { awk '/Submitted batch job/{print $NF}'; }

# backbone walltime per dataset (pemsbay N=325 is heaviest).
bb_time() { case "$1" in pemsbay) echo 03:00:00 ;; metrla) echo 01:30:00 ;; *) echo 01:00:00 ;; esac; }

n=0
for ds in ${DATASETS}; do
  for bb in ${BACKBONES}; do
    for s in ${SEEDS}; do
      n=$((n+1))
      echo "== [$n] ${ds} / ${bb} / seed${s} =="
      BB=$(BACKBONE="$bb" bash "${HERE}/submit.sh" backbone "$ds" "$s" \
             --time="$(bb_time "$ds")" | tee /dev/stderr | jobid_of)
      : "${BB:?failed to submit backbone for ${ds}/${bb}/seed${s}}"
      BACKBONE="$bb" bash "${HERE}/submit.sh" residuals "$ds" "$s" \
        --dependency=afterok:"$BB" | tee /dev/stderr | jobid_of >/dev/null
    done
  done
done
echo
echo "submitted ${n} backbone->residual chains under EXP=${EXP}. watch: squeue --me"
echo "residual jobs sit PENDING (Dependency) until their backbone finishes."
