# Project Specification: Structured Residual Diffusion for Probabilistic Spatiotemporal Traffic Forecasting

This document defines the concrete plan, conventions, and deliverables for the implementation phase of the thesis project. It is the authoritative reference for Claude Code and for the author. Any deviation should be documented.

## 1. High-level goal

Build and evaluate a probabilistic spatiotemporal traffic forecasting pipeline combining:

1. A frozen deterministic backbone (STID) producing point forecasts.
2. A conditional diffusion model on the residuals, with a structured Kronecker-product prior (graph-Laplacian spatial factor × horizon-dependent AR(1) temporal factor).

The thesis contribution is the structured prior in the residual-diffusion framework. Everything else (backbone, data protocol, evaluation, baselines) follows community standards.

The entire pipeline must first run end-to-end on a MacBook (one epoch, one dataset, one seed) before any Delft (HPC) submission. Full-scale experiments run on Delft afterwards.

## 2. Datasets

Two datasets, both standard in ST forecasting:

- **METR-LA**: 207 sensors, Los Angeles freeways, 5-min resolution, March 2012 – June 2012 (34,272 timestamps). ~8.11% missing values encoded as 0.0.
- **PEMS-BAY**: 325 sensors, San Francisco Bay Area, 5-min resolution, Jan 2017 – May 2017 (52,116 timestamps). ~0.003% missing.

Source: DCRNN repo (`https://github.com/liyaguang/DCRNN`), `data/METR-LA.h5` and `data/PEMS-BAY.h5`. Adjacency matrices from the same repo (`data/sensor_graph/adj_mx.pkl` and `adj_mx_bay.pkl`) computed via thresholded Gaussian kernel of road-network distances with `normalized_k=0.1`.

Use these files exactly. Do not regenerate adjacencies or splits.

## 3. Data protocol (FIXED — do not deviate)

These values are the community standard. Matching them is how our numbers are comparable to published baselines.

| Aspect | Value |
|---|---|
| Split ratio | 70% train / 10% val / 20% test, chronological order (no shuffling across boundary) |
| Input window length (T_in) | 12 steps (1 hour of history) |
| Forecast horizon (T_out) | 12 steps (1 hour ahead) |
| Sliding window stride | 1 step |
| Normalization | Z-score, computed on training set only, applied to train/val/test |
| Normalization scope | Global (one mean, one std across all sensors and all timestamps) |
| Missing value encoding | 0.0 in the raw .h5 files |
| Missing value handling | Mask in loss and all metrics — never included in numerator/denominator |
| Time features | time-of-day ∈ [0, 287] (288 slots per day) and day-of-week ∈ [0, 6], both as integer indices fed to learnable embeddings |
| Batch size | 64 |

Concrete mask definition: `mask = (raw_value != 0.0)`. Compute *before* normalization. Apply to loss as `loss = mean(mask * per_element_loss) / (mean(mask) + 1e-8)`.

All metrics (MAE, RMSE, CRPS, energy score, variogram score) must use this mask. Using published reference implementations of CRPS etc. often needs a wrapper that respects the mask — do not skip this.

## 4. Evaluation protocol (FIXED)

### 4.1 Deterministic metrics

Reported for: STID alone, the mean of diffusion model samples, and all baselines.

- **MAE** per horizon step 3/6/12 (= 15/30/60 minutes ahead) and averaged over all 12 steps.
- **RMSE** per horizon step 3/6/12 and averaged.
- **MAPE** (%) per horizon step 3/6/12 and averaged. Exclude entries where `|y_true| < 1.0` from MAPE denominator (standard convention for traffic speed).

### 4.2 Probabilistic metrics

Reported for: our method and all probabilistic baselines.

- **CRPS** per horizon and averaged over horizons and sensors. Use empirical CRPS from samples via the `properscoring.crps_ensemble` function, with masking applied.
- **Energy score** (multivariate generalization of CRPS). Implementation: for each test window, compute `ES = mean_i ||s_i - y|| - 0.5 * mean_{i,j} ||s_i - s_j||` where s_i are samples and y is ground truth, with l2 norm over the (horizon × sensor) dimensions. One number per test window, averaged. Masked.
- **Variogram score** of order p = 0.5. Implementation: standard formula over pairs (i,j), weighting by inverse distance if graph distances available. See Scheuerer & Hamill (2015) or `scoringRules` R package reference. Port carefully and unit-test against a known reference case.
- **PIT histograms** per horizon (3/6/12). For each (test sample, horizon, sensor) compute PIT = F_pred(y_true) where F_pred is the empirical CDF of samples. Plot histogram per horizon; a well-calibrated model gives a flat histogram.
- **Prediction interval coverage probability (PICP)** at nominal levels 50%, 90%, 95%. Report per horizon.
- **Sharpness**: mean interval width at 90% coverage.

### 4.3 Sampling protocol at test time

100 samples per test input. Reuse the same 100 samples for all probabilistic metrics computed on that test input.

### 4.4 Regime-stratified evaluation

Stratify test samples by traffic regime based on the *mean speed of the input window*:
- Free-flow: mean speed in top 1/3 of test set
- Transition: middle 1/3
- Congested: bottom 1/3

Report CRPS and MAE per regime per horizon. This is how we show the structured prior helps most in congested regimes (our hypothesis).

### 4.5 Statistical significance

For main results table: 5 seeds per configuration. Report `mean ± std` over seeds. Use paired t-test or Wilcoxon signed-rank across seeds when claiming significance. Threshold p < 0.05.

For ablation tables: 3 seeds per configuration (same format).

## 5. Target numbers for STID (reproduction check)

After training STID on each dataset, these are the reference values. Implementation is correct if matched within ±5%.

**METR-LA** (STID official / CIKM'22 table):
- MAE @ 15/30/60 min: 2.81 / 3.19 / 3.55
- RMSE @ 15/30/60 min: 5.54 / 6.57 / 7.55
- MAPE @ 15/30/60 min: 7.40% / 9.09% / 10.77%

**PEMS-BAY** (STID official):
- MAE @ 15/30/60 min: 1.31 / 1.62 / 1.89
- RMSE @ 15/30/60 min: 2.79 / 3.68 / 4.40
- MAPE @ 15/30/60 min: 2.78% / 3.68% / 4.42%

If numbers are more than 5% off, stop and debug data pipeline / normalization / mask before proceeding.

## 6. Seeds (FIXED)

Seed values for all experiments: **0, 1, 2, 3, 4**.

- Main results table: all five seeds.
- Ablation tables: seeds 0, 1, 2.
- Smoke tests on Mac: seed 0 only.

Seed controls: Python `random`, NumPy, PyTorch CPU, PyTorch MPS (Mac) / CUDA (Delft). Set via single `set_seed(s)` function called at the start of every script.

## 7. Experiment tracking

Use **Weights & Biases** (`wandb`). Project name: `struct-residual-diffusion-traffic`.

Each run logs:
- Full config YAML (as wandb config).
- Commit hash (`git rev-parse HEAD`).
- Per-epoch: train_loss, val_loss, val_MAE per horizon, val_RMSE per horizon.
- Every 10 epochs: val_CRPS per horizon, val_energy_score (on a subset of 200 val windows to keep it cheap).
- End of training: all final metrics on test set (100 samples per input).
- System: GPU type, memory peak, wall-clock time.

Tags per run: `dataset={metrla,pemsbay}`, `backbone=stid`, `prior={iso,spatial,temporal,kronecker,kronecker_const_rho}`, `seed={0..4}`, `phase={smoke,main,ablation,generality}`.

Run naming convention: `{phase}_{dataset}_{prior}_seed{seed}`.

## 8. Reuse vs. write-new policy

**Reuse from public repos (cite commit hash):**
- Data loading and preprocessing from DCRNN repo (the `scripts/generate_training_data.py` pattern). This is the canonical preprocessing and every comparable paper uses it.
- STID: use the official repo at `github.com/GestaltCogTeam/STID`. Clone into `third_party/STID`, pin the commit hash. Wrap with a thin `src/backbones/__init__.py::load_stid()` function. Do not copy or modify STID's source.
- Baselines for comparison (DiffSTG, CSDI, PriSTI, STGNF, CoST): use official repos. Cite commit hash. Run on our splits. Do not reimplement.
- Standard diffusion utilities (EDM preconditioning, Heun sampler): port from Karras et al. reference implementation with citation in comments.
- Metrics: `properscoring` package for CRPS. Energy score and variogram score — port from `scoringRules` R reference or a clean Python reference, unit-test against a known input.

**Write new (the thesis contribution):**
- Forward SDE with Kronecker-structured prior (`src/diffusion/sde.py`).
- Reverse SDE / Tweedie-based sampler for structured noise (`src/diffusion/sampler.py`).
- Denoiser interfacing with structured noise (`src/diffusion/denoiser.py`).
- Prior parameter fitting from empirical residuals (`src/diffusion/prior_fit.py`).
- Regime stratification and stratified evaluation (`src/eval/stratified.py`).

Every reused file has a header comment:
```python
# Adapted from <repo URL> at commit <hash>
# Original license: <LICENSE>
# Modifications: <short description>
```

## 9. Repository layout

```
thesis_diffusion/
├── README.md
├── pyproject.toml                   # pinned versions
├── .gitignore
├── configs/
│   ├── data/metrla.yaml
│   ├── data/pemsbay.yaml
│   ├── backbone/stid.yaml
│   └── experiment/
│       ├── smoke_mac.yaml
│       ├── main_structured.yaml
│       ├── ablation_iso.yaml
│       ├── ablation_spatial_only.yaml
│       ├── ablation_temporal_only.yaml
│       └── ablation_const_rho.yaml
├── data/
│   ├── raw/                         # .h5 files, adjacencies (gitignored)
│   └── processed/                   # train/val/test .npz (gitignored)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py          # sliding window, split, scaler
│   │   ├── dataset.py                # PyTorch Dataset
│   │   └── adjacency.py              # load adjacency + compute Laplacian
│   ├── backbones/
│   │   └── stid.py                   # STID reimplementation
│   ├── diffusion/
│   │   ├── sde.py                    # forward SDE, Kronecker prior ops
│   │   ├── prior_fit.py              # fit delta_h, rho, beta from residuals
│   │   ├── denoiser.py               # graph-MLP + AdaLN-Zero
│   │   ├── precond.py                # EDM preconditioning
│   │   └── sampler.py                # Heun with structured noise
│   ├── eval/
│   │   ├── metrics_det.py            # MAE, RMSE, MAPE, masked
│   │   ├── metrics_prob.py           # CRPS, energy, variogram, PIT
│   │   └── stratified.py             # regime stratification
│   ├── training/
│   │   ├── backbone_train.py
│   │   └── diffusion_train.py
│   └── utils/
│       ├── seed.py
│       ├── logging.py                # wandb wrapper
│       └── config.py                 # YAML loader
├── scripts/
│   ├── download_data.sh              # pull from DCRNN repo
│   ├── preprocess.py                 # raw .h5 → processed .npz
│   ├── train_backbone.py             # entry: train STID, save residuals
│   ├── fit_prior.py                  # entry: fit prior params from residuals
│   ├── train_diffusion.py            # entry: train diffusion model
│   ├── evaluate.py                   # entry: full eval on test set
│   └── submit_delft.sh               # SLURM submission template
├── notebooks/
│   ├── 01_data_eda.ipynb
│   ├── 02_residual_analysis.ipynb
│   └── 03_results_analysis.ipynb
├── tests/
│   ├── test_preprocessing.py
│   ├── test_stid_shapes.py
│   ├── test_metrics.py
│   └── test_sde.py
└── results/
    └── (gitignored, populated by runs)
```

## 10. Environment

Python 3.11. Use `uv` or `pip` with a pinned `pyproject.toml`. Minimum packages:

```
torch>=2.2               # supports MPS backend on Mac
numpy
scipy
pandas
h5py
pyyaml
wandb
properscoring
tqdm
matplotlib
jupyter
pytest
```

No CUDA-only packages. The same code must run on Mac MPS and Delft CUDA.

Device selection:
```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
```

## 11. Phased execution plan

### Phase 0 — Scaffolding (½ day, Mac)

Set up repo, environment, wandb account, DCRNN data files downloaded to `data/raw/`. Run `pytest tests/` (empty but passing) to confirm scaffolding.

**Deliverable**: repo with structure above, `pip install -e .` succeeds, data files present, wandb login works.

### Phase 1 — Data preprocessing + EDA (1 day, Mac)

Write `scripts/preprocess.py` that reads raw .h5, builds sliding windows with the protocol in §3, saves `data/processed/{metrla,pemsbay}_{train,val,test}.npz` with arrays `{x, y, x_time, y_time, mask_x, mask_y}`.

Unit test: shapes match expected. Test-set size matches DCRNN (e.g., METR-LA test has ~6,850 windows).

Write `notebooks/01_data_eda.ipynb` producing five figures as described:
- `fig_weekly_cycles.png`
- `fig_missingness_heatmap.png`
- `fig_acf_raw_vs_demeaned.png`
- `fig_dirichlet_energy_distribution.png`
- `regimes.npz` (regime labels per test timestamp)

**Go/no-go**: visual sanity — weekly cycles clear, METR-LA missingness concentrated at a few sensors, ACF decays roughly exponentially, Dirichlet energy distribution is right-skewed (low values = smooth on graph).

### Phase 2 — STID backbone (official code) (1 day, Mac)

Clone the official STID repo into `third_party/STID` (gitignored), pin the commit hash. Write a thin wrapper `src/backbones/__init__.py::load_stid(dataset, device)` that instantiates STID with the correct config. Train via `scripts/train_backbone.py` on METR-LA and PEMS-BAY.

**Go/no-go**: MAE within ±5% of §5 targets on both datasets. If off, debug: (1) mask application, (2) normalization direction, (3) time-feature encoding, (4) loss sign.

Save checkpoints to `results/checkpoints/stid_metrla_seed0.pt` and `stid_pemsbay_seed0.pt`.

Compute residuals on **train set** (for fitting prior) and **val+test sets** (for diffusion training) by running frozen backbone, storing as `results/residuals/{dataset}_{split}.npz`.

### Phase 3 — Residual analysis (1 day, Mac)

Write `notebooks/02_residual_analysis.ipynb` loading `results/residuals/metrla_train.npz`. Produce:
- `fig_variance_vs_horizon.png` — Var(R_h) as function of h.
- `fig_sigma_Q.png` — H×H temporal residual covariance heatmap.
- `fig_sigma_R_spectral.png` — spatial covariance projected on Laplacian eigenbasis.
- Scalar: separability Frobenius relative error.
- `fig_variance_by_regime.png` — variance curves stratified by regime.

Repeat for PEMS-BAY.

**Go/no-go**:
- Variance grows with horizon (monotonic-ish). If flat, horizon-dependent prior has no motivation — stop and rethink.
- Separability error < 40%. If higher, document as limitation and consider whether Kronecker is still defensible.
- Spatial residual covariance concentrates in low-frequency Laplacian modes. If energy is uniform across eigenvalues, Laplacian prior is badly matched — stop and rethink.

### Phase 4 — Prior parameter fitting (½ day, Mac)

Write `scripts/fit_prior.py`:
- Load train residuals.
- Fit δ_h = empirical Var(R_h) per horizon.
- Fit ρ = empirical lag-1 autocorrelation (one value, or per-horizon ρ_h).
- Fit β (spatial scaling) via `scipy.optimize.minimize` minimizing Frobenius distance between `(β L + ε I)^{-1}` and empirical Σ_R.
- Derive σ (SDE diffusion coefficient) from eigenvalue analysis such that forward process reaches the target GMRF in finite τ.
- Save to `results/prior_params/{dataset}.yaml`.

Sanity checks in the same script:
- Draw 1000 samples from the prior, compute empirical covariance, compare to target. Save `fig_prior_samples_covariance_check.png`.
- Verify forward process marginals match closed-form Tweedie predictions at τ ∈ {0, 0.5, 1.0, 2.0}.

**Go/no-go**: both sanity checks pass. If not, the SDE implementation has a bug.

### Phase 5 — Diffusion model implementation (3–4 days, Mac)

Write `src/diffusion/sde.py`, `precond.py`, `denoiser.py`, `sampler.py`.

Denoiser specification:
- Input: noisy residual tensor shape (batch, H=12, N, 1).
- Spatial identity embedding + horizon index embedding added to tokens.
- 4 blocks of: AdaLN-Zero → graph attention (Chebyshev-K=3 on precomputed normalized Laplacian) → AdaLN-Zero → MLP with GELU.
- AdaLN conditioning vector = MLP(τ_embed ⊕ history_summary), where history_summary comes from the frozen STID's penultimate features pooled.
- Output: noise (or v-parameterization) prediction, same shape as input.
- Total parameters: ≤ 1M.

**Mandatory test: overfit one batch.** Take a single batch of 32 samples, train the denoiser (no backbone update) for 500 steps on this batch only. Loss must go to near-zero and a sampled trajectory must match the batch. This test lives in `tests/test_overfit.py`. If this fails, no larger run will succeed — debug before continuing.

### Phase 6 — End-to-end smoke test on Mac (2 days, Mac)

Train diffusion model for **5 epochs** on METR-LA with full-structured Kronecker prior, seed 0. Run full evaluation on val set with 20 samples (reduced from 100 to fit in time budget).

Expected runtime: 4–8 hours overnight on MacBook (MPS or CPU). If it takes more than 24h, reduce denoiser size further.

Then run the same 5-epoch test with isotropic prior (Σ = I), same seed, for comparison.

**Go/no-go**:
- Pipeline runs end-to-end without crashing.
- Samples are not pure noise (visually, by plotting overlaid samples vs. ground truth for 3 test sensors).
- Structured prior beats isotropic on *some* probabilistic metric, even if the absolute numbers are poor. If structured is worse than or equal to isotropic even at 5 epochs, the prior implementation likely has a bug — investigate before Delft.

Checkpoint this result as the "Mac smoke test complete" milestone.

### Phase 7 — Delft: full experiments (1–2 weeks wall-clock)

Only after Phase 6 passes. Submit with `scripts/submit_delft.sh` as a SLURM array job.

**Main experiment matrix**: 6 prior configurations × 2 datasets × 5 seeds = 60 runs.
- Iso, Spatial-only, Temporal-only, Full Kronecker, Kronecker-with-constant-ρ, plus STID-alone baseline (already computed in Phase 2).

**Training**: 200 epochs, full val evaluation every 10 epochs, full test evaluation once at end with 100 samples.

**Framework-generality**: skip for first pass; add in a second batch if time allows.

**Baselines**: run DiffSTG, CSDI, STGNF, CoST on both datasets with 3 seeds each, using official code. Each baseline gets its own submission script calling the reference repo.

### Phase 8 — Analysis and writing (2–3 weeks, Mac)

Load all wandb results, generate final tables and figures. `notebooks/03_results_analysis.ipynb` produces:
- Main results table (LaTeX).
- Ablation table (LaTeX).
- Per-regime CRPS figure.
- PIT calibration figure.
- Qualitative sample plots (3 interesting test windows).

## 12. Failure criteria (commit to these now)

For each phase, stop and rethink if the condition is hit:

| Phase | Failure condition |
|---|---|
| 2 | STID MAE > 5% off published numbers — data pipeline bug |
| 3 | Residual variance flat across horizons — horizon-dep. prior unmotivated |
| 3 | Separability Frobenius error > 40% — Kronecker approximation too crude |
| 4 | Prior sample covariance check fails — SDE bug |
| 5 | Overfit-one-batch test fails — denoiser implementation bug |
| 6 | Structured ≤ isotropic at 5 epochs — prior implementation bug or wrong parameters |

If any of these hit, the correct action is to diagnose and fix before moving forward, not to push ahead hoping it'll resolve.

## 13. Claude Code task instructions

This specification is the authoritative source of truth. When implementing any component, follow it exactly. When uncertain about a detail not covered here:

1. Look at how comparable papers (STID, DCRNN, CoST) handle it in their official repos.
2. Default to the most community-standard convention.
3. Flag the decision in a code comment with `# DESIGN DECISION: ...` and note it in a `DECISIONS.md` log.

Never deviate silently from the protocol in §3 or §4. These are the numbers that make our results comparable to the literature, and drift here invalidates the comparison.

Write tests as you go — every module in `src/` gets a corresponding test in `tests/`. A module without tests should not be considered complete.

Write clean git commits with messages of the form `phase<N>: <what was done>`. Commit after each logical unit of work (not after each file).

Keep `DECISIONS.md` updated with any non-obvious choice and its rationale.

## 14. Reference links

- STID paper: arXiv:2208.05233, code at `github.com/GestaltCogTeam/STID`.
- DCRNN repo (data source): `github.com/liyaguang/DCRNN`.
- BasicTS benchmark (reference implementations): `github.com/GestaltCogTeam/BasicTS`.
- CoST paper: arXiv:2502.11013.
- FALDA paper: arXiv:2505.11306.
- GAD paper: arXiv:2510.05036 (supervisor's work).
- EDM paper: Karras et al., NeurIPS 2022, arXiv:2206.00364.
- DiffSTG: official code at `github.com/wenhaomin/DiffSTG`.
- CSDI: `github.com/ermongroup/CSDI`.
- PriSTI: `github.com/LMZZML/PriSTI`.
- STGNF (spatio-temporal graph normalizing flow): CIKM 2024, search for code.

End of specification.
