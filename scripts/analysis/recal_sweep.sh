#!/bin/bash
# adaptive-recal sweep, LOCAL = pems08 cells only (their residuals live only on this Mac
# after the DAIC space rotation); metrla/pemsbay/pems04 run on the DAIC GPUs instead.
# serial, resumable (skips existing jsons), log next to the results.
cd "$(dirname "$0")/../.." || exit 1
LOG=results/analysis/adaptive_recal/sweep.log
mkdir -p results/analysis/adaptive_recal
run() {
  ds=$1; bb=$2; seed=$3; cond=$4
  tag=""; [ "$cond" = past ] && tag="_past"
  out=results/analysis/adaptive_recal/${ds}_${bb}${tag}_seed${seed}.json
  [ -s "$out" ] && { echo "SKIP $ds/$bb/$cond/$seed" >>"$LOG"; return; }
  echo "=== $(date +%H:%M) start $ds/$bb cond=$cond seed$seed" >>"$LOG"
  .venv/bin/python scripts/analysis/adaptive_recal.py --dataset "$ds" --backbone "$bb" \
    --seed "$seed" --cond "$cond" --val-windows 400 --test-windows 600 --k 32 --n-steps 25 \
    >>"$LOG" 2>&1 || echo "FAIL $ds/$bb/$cond/$seed" >>"$LOG"
}
run pems08s gwn 42 past      # redo (smoke json was deleted)
run pems08s gcn 42 past
run pems08f gcn 42 past
run pems08s gwn 1  past
run pems08f gwn 1  past
run pems08s gcn 42 fo
run pems08f gcn 42 fo
echo "=== $(date +%H:%M) local sweep complete" >>"$LOG"
