#!/bin/bash
set -euo pipefail
cd /workspace/research-tud
. .venv/bin/activate
export WANDB_MODE=online
DATASET="${1:-metrla}"; SEED="${2:-0}"
K="${K:-50}"; STEPS="${STEPS:-50}"; MAXW="${MAXW:-512}"; BS="${BS:-32}"
shopt -s nullglob
mapfile -t real < <(
  for d in results/checkpoints/diffusion/diff_gwn_${DATASET}_seed${SEED}_*; do
    [ -f "$d/best.pt" ] || continue
    grep -q "\"smoke\": *true" "$d/config.json" 2>/dev/null && continue
    printf "%s\t%s\n" "$(stat -c %Y "$d/best.pt")" "$d"
  done | sort -rn | cut -f2-
)
CKPT_DIR="${real[0]}"
echo "=== FAST EVAL $DATASET ckpt=$CKPT_DIR  K=$K steps=$STEPS maxw=$MAXW ==="
python scripts/eval/generate_samples.py --ckpt-dir "$CKPT_DIR" --which best --split test --seed "$SEED" \
    --n-samples "$K" --n-steps "$STEPS" --eta 1.0 --max-windows "$MAXW" --batch-size "$BS"
python scripts/eval/evaluate_forecasts.py --dataset "$DATASET" --split test
python scripts/eval/make_figures.py --dataset "$DATASET" --split test || true
echo "FAST_EVAL_DONE $DATASET"
