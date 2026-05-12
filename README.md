# Structured Residual Diffusion for Probabilistic Spatiotemporal Traffic Forecasting

Master's thesis project. The single entry point for results, methodology, and figures is [`experiments/README.md`](experiments/README.md). Each phase has a `SUMMARY.md` with embedded numbers and figures.

## Setup

```bash
pip install -e ".[dev]"
```

## Pipeline

```bash
bash scripts/download_data.sh
python scripts/preprocess.py     --dataset metrla
python scripts/preprocess.py     --dataset pemsbay
python scripts/train_backbone.py --dataset metrla  --seed 0 --epochs 100
python scripts/train_backbone.py --dataset pemsbay --seed 0 --epochs 100
python scripts/extract_residuals.py --dataset metrla  --seed 0
python scripts/extract_residuals.py --dataset pemsbay --seed 0
python scripts/fit_prior.py      --dataset metrla
python scripts/fit_prior.py      --dataset pemsbay
python experiments/exp-004/run_sanity_checks.py
```

See `experiments/README.md` for full reproduction notes.
