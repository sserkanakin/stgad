# Structured Residual Diffusion for Probabilistic Spatiotemporal Traffic Forecasting

Master's thesis project. A deterministic Graph WaveNet (GWN) backbone produces point forecasts; a score-based diffusion model then generates a probability distribution over the forecast residuals, conditioned on the backbone's prediction. The key contribution is the use of a **Kronecker-structured prior** `Σ_prior = Q_H ⊗ Σ_S` (temporal × spatial) for the diffusion's noise covariance — providing data-aware inductive bias that an isotropic prior cannot.

---

## Documentation map

| Where | What |
|---|---|
| **[`THESIS_OVERVIEW.md`](THESIS_OVERVIEW.md)** | **Comprehensive walkthrough of the whole approach** — formulas, design choices, why each piece is the way it is. Start here. |
| [`DIVERGENCES.md`](DIVERGENCES.md) | Where we deviate from standard score-based diffusion practice and why. |
| [`CODE_OVERVIEW.md`](CODE_OVERVIEW.md) | Module-by-module tour of every Python file. |
| [`experiments/README.md`](experiments/README.md) | Per-phase experiment index. Each has its own `SUMMARY.md` with figures and numbers. |
| [`experiments/score-network/PLAN.md`](experiments/score-network/PLAN.md) | Score-net design + training plan. |
| [`scripts/daic/README.md`](scripts/daic/README.md) | Running on TU Delft DAIC GPU cluster. |

---

## Setup

```bash
pip install -e ".[dev]"
```

For HDF5-reading utility scripts (preprocess.py, diagnose_temporal_acf.py):
```bash
pip install -e ".[dev,hdf]"
```

For DAIC: see [`scripts/daic/README.md`](scripts/daic/README.md). Python 3.11 conda env with cu121 PyTorch wheels.

---

## End-to-end pipeline

```bash
# 1. Data
bash scripts/download_data.sh
python scripts/preprocess.py     --dataset metrla
python scripts/preprocess.py     --dataset pemsbay

# 2. Deterministic backbone (GWN)
python scripts/train_backbone.py    --dataset metrla  --seed 0 --epochs 100
python scripts/train_backbone.py    --dataset pemsbay --seed 0 --epochs 100
python scripts/extract_residuals.py --dataset metrla  --seed 0
python scripts/extract_residuals.py --dataset pemsbay --seed 0

# 3. Fit the Kronecker-structured residual prior
python scripts/fit_prior.py --dataset metrla
python scripts/fit_prior.py --dataset pemsbay

# 4. Validate the SDE machinery
python experiments/sde-sanity/run_sanity_checks.py

# 5. Train the score network for residual diffusion
python scripts/train_diffusion.py --dataset metrla  --seed 0 --prior structured
python scripts/train_diffusion.py --dataset metrla  --seed 0 --prior isotropic   # ablation
python scripts/train_diffusion.py --dataset pemsbay --seed 0 --prior structured
python scripts/train_diffusion.py --dataset pemsbay --seed 0 --prior isotropic

# 6. Probabilistic evaluation (CRPS, PIT, pinball)
python scripts/eval_diffusion.py --run_id dsm_structured_metrla_seed0_<hash>
# (one invocation per trained run)
```

For DAIC, the same pipeline runs under Slurm — see [`scripts/daic/README.md`](scripts/daic/README.md). The 5+6 sweep across both datasets and priors takes ~3 hours of cluster wall-time.

---

## Status (score-network in progress)

| Phase | What | Status |
|---|---|---|
| 0–2 | Preprocessing, EDA, backbone training | done |
| residual-diagnostics | Residual diagnostics | done |
| prior-construction | Prior fit (Kronecker, standardised, two-exp temporal kernel) | done |
| sde-sanity | SDE forward + reverse + sanity checks | done |
| **score-network** §14.1–§14.5 | Score net + DSM training + sampler + CRPS evaluator | done |
| **score-network** evaluation | 4-run sweep on DAIC + CRPS/PIT/pinball comparison | running |

---

## Quick test

```bash
pytest tests/ -q
```

162 tests covering SDE math (NumPy + PyTorch), DSM loss, score network, reverse sampler, CRPS / PIT / pinball, and preprocessing.
