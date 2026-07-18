#!/usr/bin/env bash
# Run the full backbone -> residual grid locally on MPS, fastest-first so residuals
# start landing immediately. Each cell: train_backbone (best.pt by val MAE) then
# extract_residuals. Logs per cell to logs/backbone_local/.
#
# usage:  bash scripts/run_backbone_grid_local.sh
#         SEED=42 DEVICE=mps EPOCHS=100 bash scripts/run_backbone_grid_local.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=.venv/bin/python
SEED="${SEED:-42}"
DEVICE="${DEVICE:-mps}"
EPOCHS="${EPOCHS:-100}"
LOG="logs/backbone_local"
mkdir -p "$LOG"

# smallest/fastest first so a full dataset (all 3 backbones) completes early
VARIANTS="${VARIANTS:-pems08s pems08f pems04s pems04f metrla pemsbay}"
BACKBONES="${BACKBONES:-mlp gcn gwn}"

echo "grid: [$VARIANTS] x [$BACKBONES]  seed=$SEED device=$DEVICE"
for V in $VARIANTS; do
  for B in $BACKBONES; do
    tag="${V}_${B}_seed${SEED}"
    echo "=== $(date '+%H:%M:%S')  START $tag ==="
    if $PY scripts/train_backbone.py --variant "$V" --backbone "$B" --seed "$SEED" \
         --epochs "$EPOCHS" --device "$DEVICE" > "$LOG/${tag}.train.log" 2>&1; then
      $PY scripts/extract_residuals.py --variant "$V" --backbone "$B" --seed "$SEED" \
         --device "$DEVICE" > "$LOG/${tag}.extract.log" 2>&1 \
        && echo "=== $(date '+%H:%M:%S')  DONE  $tag  ($(grep -h 'best val_mae' "$LOG/${tag}.train.log" | tail -1))" \
        || echo "=== $(date '+%H:%M:%S')  EXTRACT FAILED $tag (see $LOG/${tag}.extract.log)"
    else
      echo "=== $(date '+%H:%M:%S')  TRAIN FAILED $tag (see $LOG/${tag}.train.log)"
    fi
  done
done
echo "=== $(date '+%H:%M:%S')  GRID COMPLETE ==="
ls -R results/residuals 2>/dev/null | head -40
