#!/bin/bash
set -euo pipefail
cd /workspace/research-tud
. .venv/bin/activate
export WANDB_MODE=online
DATASET="${1:?usage: run_eval.sh DATASET [SEED]}"
SEED="${2:-0}"
# Collect non-smoke checkpoint dirs that have a best.pt, newest first by mtime.
shopt -s nullglob
mapfile -t real < <(
  for d in results/checkpoints/diffusion/diff_gwn_${DATASET}_seed${SEED}_*; do
    [ -f "$d/best.pt" ] || continue
    grep -q "\"smoke\": *true" "$d/config.json" 2>/dev/null && continue
    printf "%s\t%s\n" "$(stat -c %Y "$d/best.pt")" "$d"
  done | sort -rn | cut -f2-
)
if [ "${#real[@]}" -eq 0 ]; then echo "ERROR: no real best.pt for $DATASET seed $SEED"; exit 1; fi
if [ "${#real[@]}" -gt 1 ]; then
  echo "NOTE: ${#real[@]} real checkpoints for $DATASET; using newest:"
  printf "      %s\n" "${real[@]}"
fi
CKPT_DIR="${real[0]}"
echo "=== EVAL $DATASET  ckpt=$CKPT_DIR ==="
python scripts/eval/generate_samples.py --ckpt-dir "$CKPT_DIR" --which best --split test --seed "$SEED" --n-samples 100 --n-steps 100 --eta 1.0
python scripts/eval/evaluate_forecasts.py --dataset "$DATASET" --split test
python scripts/eval/diffusion_diagnostics.py --dataset "$DATASET" --ckpt-dir "$CKPT_DIR" --split val
python scripts/eval/make_figures.py --dataset "$DATASET" --split test
echo "=== EVAL $DATASET DONE ==="
