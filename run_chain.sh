#!/bin/bash
cd /workspace/research-tud
. .venv/bin/activate
export WANDB_MODE=online
for ds in metrla pems08f pemsbay; do
  echo "=== TRAIN $ds start $(date -u +%H:%M:%S) ===" >> logs/diff_${ds}_s0.log
  python scripts/train_diffusion.py --dataset "$ds" --seed 0 --fresh >> logs/diff_${ds}_s0.log 2>&1
  echo "=== TRAIN $ds done $(date -u +%H:%M:%S) ===" >> logs/diff_${ds}_s0.log
done
echo "ALLDONE $(date -u +%H:%M:%S)" > logs/chain_done.log
