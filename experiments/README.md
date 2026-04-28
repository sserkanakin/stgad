# Experiments — index

> **Single entry point.** Every experiment-level artifact lives below this directory. Each subfolder has a `SUMMARY.md` with embedded figures and analysis. Code that *produces* the artifacts lives in `../scripts/` and `../src/`; cross-experiment shared outputs (checkpoints, residual `.npz` files, fitted prior YAMLs) live in `../results/`.

## Map (read top-to-bottom)

| # | What | Where | Status |
|---|---|---|---|
| 0 | Data & residuals — dataset shapes, missingness, scaler values, residual scale + bias, key EDA figures | [`data-and-residuals-overview.md`](data-and-residuals-overview.md) | done |
| 1 | Backbone training (GWN) — per-horizon MAE / RMSE / MAPE for val + test, comparison to Wu 2019 | [`backbone-training/SUMMARY.md`](backbone-training/SUMMARY.md), [`metrics.json`](backbone-training/metrics.json) | done |
| 2 | Residual diagnostics — variance vs horizon, sensor×sensor covariance, horizon×horizon correlation, regime stratification, Kronecker separability | [`exp-001-residual-diagnostics/SUMMARY.md`](exp-001-residual-diagnostics/SUMMARY.md) | done |
| 3 | Prior construction & sanity + AR(1)-vs-two-exp comparison | [`exp-002-q-construction/SUMMARY.md`](exp-002-q-construction/SUMMARY.md), [`results_ar1_vs_2exp.json`](exp-002-q-construction/results_ar1_vs_2exp.json) | done |
| 4 | SDE implementation + sanity checks | [`exp-004/SUMMARY.md`](exp-004/SUMMARY.md), [`sanity_results.json`](exp-004/sanity_results.json) | ⚠ Check 2 PASS / Check 1 FAIL (schedule undersized — see SUMMARY §3) |
| 5 | Structured-vs-isotropic CRPS comparison | _planned, blocked on #4 schedule resolution_ | future |
| - | Cross-experiment summary with everything embedded inline | [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) | done |

EDA figures (weekly cycles, missingness, ACF, Dirichlet energy, Laplacian spectrum, regime comparison, etc.) live in [`eda/figures/`](eda/figures/) and are referenced from the data-and-residuals overview.

---

## At a glance — the numbers

### GWN test MAE (mph), vs Wu et al., IJCAI 2019

| Dataset  | h=3 (15min)        | h=6 (30min)        | h=12 (60min)       |
|----------|-------------------:|-------------------:|-------------------:|
| METR-LA  | 2.808 (Wu: 2.69, **+4.4%**) | 3.198 (Wu: 3.07, **+4.2%**) | 3.579 (Wu: 3.53, **+1.4%**) |
| PEMS-BAY | 1.320 (Wu: 1.30, **+1.6%**) | 1.644 (Wu: 1.63, **+0.9%**) | 1.923 (Wu: 1.95, **−1.4%**) |

**Verdict:** within the typical seed-to-seed range for single-seed reproductions. The trained backbones are faithful to the published model and the residuals can be trusted.

For full per-horizon (h=1..12), val + test, MAE / RMSE / MAPE, see [`backbone-training/SUMMARY.md`](backbone-training/SUMMARY.md).

### Residual scale (training set, mph)

| | METR-LA | PEMS-BAY |
|---|---:|---:|
| Global mean residual | −0.66 (slight over-prediction) | −0.13 (negligible) |
| h=1 std | 3.82 | 1.48 |
| h=12 std | 6.44 | 3.67 |
| h=12 / h=1 growth | 1.69× | 2.48× |

### Fitted Kronecker prior (`Σ_prior = Q_H ⊗ Σ_S`)

| Parameter | METR-LA | PEMS-BAY |
|---|---:|---:|
| α (slow-component weight) | 0.7995 | 0.5486 (irrelevant — degenerate) |
| ρ₁ (slow decay) | 0.8321 | 0.81870 |
| ρ₂ (fast decay) | 0.1015 | 0.81870 (= ρ₁ — collapses to AR(1)) |
| β (spatial smoothing) | 0.0252 (weak) | 0.7839 (strong) |
| σ (VP-SDE coefficient) | 3.138 | 3.808 |
| Covariance check rel. Frob | 0.121 [PASS] | 0.086 [PASS] |
| Tweedie MSE ratios (τ=0.5/1.0/2.0) | 1.000/0.999/1.000 [PASS] | 1.000/1.000/1.001 [PASS] |

Full YAMLs: [`../results/prior_params/gwn_metrla.yaml`](../results/prior_params/gwn_metrla.yaml), [`../results/prior_params/gwn_pemsbay.yaml`](../results/prior_params/gwn_pemsbay.yaml).

### AR(1) vs two-exp temporal kernel — verdict

| Dataset | RMSE drop | AIC drop | Verdict |
|---|---:|---:|---|
| METR-LA | 0.0432 → 0.0029 (**15×**) | −73 → −135 (**+61 better**) | two-exp wins decisively |
| PEMS-BAY | 0.01636 → 0.01636 (**identical**) | −96.7 → −92.7 (AR(1) wins by 4) | two-exp degenerates to AR(1) |

Adopted: two-exp uniformly. It's a strict superset of AR(1); on data with one timescale (PEMS-BAY) the optimiser collapses to AR(1) automatically; on data with two (METR-LA) it separates them. See [`exp-002-q-construction/SUMMARY.md`](exp-002-q-construction/SUMMARY.md) §6 for the full analysis.

---

## Reproduction (full pipeline)

```bash
# 0. Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. Data
bash scripts/download_data.sh
python scripts/preprocess.py --dataset metrla
python scripts/preprocess.py --dataset pemsbay
python scripts/diagnose_temporal_acf.py                    # EDA ACF figures

# 2. Backbone (~1–2 hr each on M-series MPS)
python scripts/train_backbone.py    --dataset metrla  --seed 0 --epochs 100
python scripts/train_backbone.py    --dataset pemsbay --seed 0 --epochs 100
python scripts/extract_residuals.py --dataset metrla  --seed 0
python scripts/extract_residuals.py --dataset pemsbay --seed 0
python scripts/compute_test_metrics.py                     # writes backbone-training/metrics.json

# 3. Prior fit + sanity
python scripts/fit_prior.py            --dataset metrla
python scripts/fit_prior.py            --dataset pemsbay
python scripts/compare_ar1_vs_2exp.py                      # AR(1) vs two-exp comparison artifact
```

Each step's outputs go to a deterministic location described in the corresponding `SUMMARY.md`.

---

## Vault cross-references

The vault (`a-thesis-vault/`, gitignored) contains the planning side: experiment specs (the `03-experiments/exp-NNN-*.md` files), code-level notes (`04-code/`), the runbook (`04-code/runbook.md`), and the methodology mirror (`01-thesis/`). The vault and these `SUMMARY.md` files mostly overlap in content; the vault is where chat-Claude / Serkan plan, this directory is where Claude Code reports.

If you ever see a contradiction between a vault note and a `SUMMARY.md` here, **the SUMMARY.md is the truth** — it's regenerated alongside the figures and metrics it describes.
