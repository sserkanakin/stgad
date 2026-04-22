# Structured Residual Diffusion for Probabilistic Spatiotemporal Traffic Forecasting

Master's thesis project. See `docs/PROJECT_SPEC.md` for the full specification.

## Setup

```bash
pip install -e ".[dev]"
```

## Data

```bash
bash scripts/download_data.sh
python scripts/preprocess.py --dataset metrla
python scripts/preprocess.py --dataset pemsbay
```

## Running

```bash
# Smoke test (1 epoch, seed 0, Mac)
python scripts/train_backbone.py --dataset metrla --seed 0 --epochs 1 --smoke
```
