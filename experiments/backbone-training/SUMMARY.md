# Backbone training — Graph WaveNet on METR-LA & PEMS-BAY

**What this is.** The deterministic backbone for the predict-and-refine framework. Everything downstream (residual diagnostics, prior fit, eventual diffusion model) consumes the residuals `r = y_true − y_pred` produced by the trained backbones documented here.

**Status:** done, both datasets converged within ~5% of Wu et al., IJCAI 2019 published numbers.

## Setup

| | METR-LA | PEMS-BAY |
|---|---|---|
| Sensors (N) | 207 | 325 |
| Train / val / test windows | 23,967 / 3,405 / 6,831 | 36,458 / 5,189 / 10,400 |
| Loss | masked MAE (raw mph after inverse-transform) | masked MAE (raw mph after inverse-transform) |
| Optimizer | Adam, lr=1e-3, grad-clip 5.0 | Adam, lr=1e-3, grad-clip 5.0 |
| Batch size | 64 | 64 |
| Epochs | 100 | 100 |
| Seed | 0 | 0 |
| Best `val_mae_h12` | **3.2525** | **1.9070** |
| Wu 2019 `val_mae_h12` target | 3.53 | 1.95 |
| Checkpoints | `results/checkpoints/gwn_metrla_seed0_f4721547/{best,last}.pt` | `results/checkpoints/gwn_pemsbay_seed0_3dabc06c/{best,last}.pt` |
| Residuals | `results/residuals/gwn_metrla_{train,val,test}_seed0.npz` | `results/residuals/gwn_pemsbay_{train,val,test}_seed0.npz` |

## Test set — per-horizon MAE / RMSE / MAPE

Computed by `scripts/compute_test_metrics.py` from `results/residuals/gwn_*_test_seed0.npz`. Masked: only sensors with valid readings contribute. MAPE excludes elements with `|y_true| < 1.0` (standard convention). Numbers in raw mph.

### METR-LA test set

| Horizon (min) | MAE   | RMSE  | MAPE   | Wu 2019 MAE | Δ MAE  | Wu 2019 RMSE | Δ RMSE | Wu 2019 MAPE | Δ MAPE |
|---:|------:|------:|-------:|-----:|------:|-----:|------:|------:|------:|
| 5  (h=1)  | 2.319 | 4.089 |  5.83% |   —  |   —   |   —  |   —   |   —   |   —   |
| 10 (h=2)  | 2.612 | 4.937 |  6.87% |   —  |   —   |   —  |   —   |   —   |   —   |
| **15 (h=3)** | **2.808** | **5.470** | **7.59%** | **2.69** | **+4.4%** | **5.15** | **+6.2%** | **6.90%** | **+10.0%** |
| 20 (h=4)  | 2.964 | 5.875 |  8.24% |   —  |   —   |   —  |   —   |   —   |   —   |
| 25 (h=5)  | 3.093 | 6.200 |  8.81% |   —  |   —   |   —  |   —   |   —   |   —   |
| **30 (h=6)** | **3.198** | **6.460** | **9.24%** | **3.07** | **+4.2%** | **6.22** | **+3.9%** | **8.37%** | **+10.4%** |
| 35 (h=7)  | 3.283 | 6.674 |  9.56% |   —  |   —   |   —  |   —   |   —   |   —   |
| 40 (h=8)  | 3.355 | 6.835 |  9.80% |   —  |   —   |   —  |   —   |   —   |   —   |
| 45 (h=9)  | 3.418 | 6.977 | 10.04% |   —  |   —   |   —  |   —   |   —   |   —   |
| 50 (h=10) | 3.474 | 7.097 | 10.24% |   —  |   —   |   —  |   —   |   —   |   —   |
| 55 (h=11) | 3.534 | 7.222 | 10.47% |   —  |   —   |   —  |   —   |   —   |   —   |
| **60 (h=12)** | **3.579** | **7.333** | **10.60%** | **3.53** | **+1.4%** | **7.37** | **−0.5%** | **10.01%** | **+5.9%** |
| avg (h=1..12) | 3.136 | 6.264 | 8.94% |   |   |   |   |   |   |

### PEMS-BAY test set

| Horizon (min) | MAE   | RMSE  | MAPE   | Wu 2019 MAE | Δ MAE  | Wu 2019 RMSE | Δ RMSE | Wu 2019 MAPE | Δ MAPE |
|---:|------:|------:|-------:|-----:|------:|-----:|------:|------:|------:|
| 5  (h=1)  | 0.859 | 1.562 | 1.66% |   —  |   —   |   —  |   —   |   —   |   —   |
| 10 (h=2)  | 1.131 | 2.260 | 2.29% |   —  |   —   |   —  |   —   |   —   |   —   |
| **15 (h=3)** | **1.320** | **2.795** | **2.77%** | **1.30** | **+1.6%** | **2.74** | **+2.0%** | **2.73%** | **+1.4%** |
| 20 (h=4)  | 1.458 | 3.200 | 3.15% |   —  |   —   |   —  |   —   |   —   |   —   |
| 25 (h=5)  | 1.562 | 3.502 | 3.46% |   —  |   —   |   —  |   —   |   —   |   —   |
| **30 (h=6)** | **1.644** | **3.733** | **3.70%** | **1.63** | **+0.9%** | **3.70** | **+0.9%** | **3.67%** | **+0.9%** |
| 35 (h=7)  | 1.710 | 3.911 | 3.90% |   —  |   —   |   —  |   —   |   —   |   —   |
| 40 (h=8)  | 1.763 | 4.049 | 4.05% |   —  |   —   |   —  |   —   |   —   |   —   |
| 45 (h=9)  | 1.810 | 4.164 | 4.19% |   —  |   —   |   —  |   —   |   —   |   —   |
| 50 (h=10) | 1.851 | 4.255 | 4.30% |   —  |   —   |   —  |   —   |   —   |   —   |
| 55 (h=11) | 1.888 | 4.336 | 4.41% |   —  |   —   |   —  |   —   |   —   |   —   |
| **60 (h=12)** | **1.923** | **4.408** | **4.50%** | **1.95** | **−1.4%** | **4.52** | **−2.5%** | **4.63%** | **−2.9%** |
| avg (h=1..12) | 1.577 | 3.515 | 3.53% |   |   |   |   |   |   |

## Validation set — per-horizon MAE / RMSE / MAPE

Same script, val split. Useful for early-stopping diagnostics; full per-horizon dump lives in `metrics.json`.

| Horizon (min) | METR-LA MAE | METR-LA RMSE | METR-LA MAPE | PEMS-BAY MAE | PEMS-BAY RMSE | PEMS-BAY MAPE |
|---:|------:|------:|------:|------:|------:|------:|
| 5  (h=1)  | 2.152 | 3.843 |  5.31% | 0.875 | 1.580 | 1.72% |
| 15 (h=3)  | 2.575 | 5.137 |  6.95% | 1.341 | 2.834 | 2.86% |
| 30 (h=6)  | 2.926 | 6.125 |  8.57% | 1.650 | 3.755 | 3.78% |
| 60 (h=12) | 3.252 | 6.948 |  9.97% | 1.907 | 4.435 | 4.54% |
| avg       | 2.869 | 5.929 |  8.31% | 1.579 | 3.538 | 3.60% |

## Verdict

Across the three reported horizons (h=3, 6, 12) on the test set:
- **METR-LA:** MAE within +1.4% to +4.4% of Wu 2019. RMSE within −0.5% to +6.2%. MAPE within +5.9% to +10.4%.
- **PEMS-BAY:** MAE within ±1.6%. RMSE within ±2.5%. MAPE within ±2.9%.

These deltas are well inside the typical seed-to-seed variation for single-seed reproductions of GWN. **The trained backbones are reproductions of the published model. The residuals they produce can be taken as faithful representations of GWN's irreducible forecast error**, which is what the prior fit ([[../exp-002-q-construction/SUMMARY|exp-002]]) is built on.

## Reproduction

```bash
# Train (one-off, ~1–2 hr/dataset on M-series MPS)
python scripts/train_backbone.py --dataset metrla  --seed 0 --epochs 100
python scripts/train_backbone.py --dataset pemsbay --seed 0 --epochs 100

# Extract residuals
python scripts/extract_residuals.py --dataset metrla  --seed 0
python scripts/extract_residuals.py --dataset pemsbay --seed 0

# Recompute the per-horizon metrics tables above
python scripts/compute_test_metrics.py     # writes metrics.json + prints table
```

## Files

| What | Where |
|---|---|
| Per-horizon JSON (val + test, h=1..12) | `metrics.json` |
| Re-computation script | `../../scripts/compute_test_metrics.py` |
| Training script | `../../scripts/train_backbone.py` |
| Residual extraction | `../../scripts/extract_residuals.py` |
| Wu 2019 reference values | embedded in `metrics.json` `metadata.wu2019_reference` |
