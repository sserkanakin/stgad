# EDA Analysis: METR-LA vs PEMS-BAY

Interpretation of `notebooks/01_data_eda.ipynb`. All numbers come from executed notebook outputs.
Figures are in `results/figures/eda/`.

---

## 1. Dataset overview

| Property | METR-LA | PEMS-BAY |
|---|---|---|
| Sensors (N) | 207 | 325 |
| Timestamps (T) | 34,272 | 52,116 |
| Date range | Mar–Jun 2012 | Jan–Jun 2017 |
| Resolution | 5 min | 5 min |
| Train windows | 23,967 | 36,458 |
| Val windows | 3,405 | 5,189 |
| Test windows | 6,831 | 10,400 |
| Speed mean (train) | 54.41 mph | 62.73 mph |
| Speed std (train) | 19.49 mph | 9.44 mph |

**PEMS-BAY is a larger dataset** by both sensor count (57% more sensors) and duration (52% more
timestamps). Its longer recording period gives more seasonal variation to train on.

**Speed distributions differ substantially.** METR-LA (Los Angeles urban/freeway mix) has a
lower mean (54 mph) and much higher variance (std=19 mph, CV=36%). PEMS-BAY (San Francisco Bay
Area freeway) is faster and tighter (std=9 mph, CV=15%). This matters for:
- **Scaler**: both datasets normalised separately; do not mix scalers across datasets.
- **Expected residual scale**: backbone residuals on METR-LA will have larger absolute magnitude
  than on PEMS-BAY, so CRPS and MAE targets are not directly comparable across datasets.
- **Prior width**: the Kronecker prior's marginal variance should be tuned per dataset.

---

## 2. Data quality — missingness

| Property | METR-LA | PEMS-BAY |
|---|---|---|
| Mean missing rate | 8.11% | 0.003% |
| Max missing rate (any sensor) | 20.1% (sensor 105) | 0.0% |
| Sensors >10% missing | 34 of 207 | 0 of 325 |

**METR-LA has significant, non-uniform missing data.** 34 sensors are missing more than 10% of
readings; one sensor (105) is missing 20% of the time. The missing pattern is not purely random —
sensors along certain corridors tend to drop out together, suggesting infrastructure-level failures
rather than independent measurement noise.

**PEMS-BAY is effectively complete.** The 0.003% missing rate (≈ 5 readings across the entire
dataset) can be treated as negligible; the masked-MAE loss on PEMS-BAY degenerates to standard
MAE.

**Implications for training:**
- The masked-MAE loss (`loss = |pred - y| * mask / (mask.sum() + 1e-8)`) is load-bearing for
  METR-LA but a no-op for PEMS-BAY. This asymmetry is expected.
- METR-LA evaluation metrics must exclude missing timesteps; PEMS-BAY metrics are unaffected.
- When fitting the Kronecker prior to backbone residuals, the ~531 all-missing windows in the
  METR-LA test split are assigned the global mean speed for regime labelling (handled in notebook).

---

## 3. Temporal structure — ACF

**Figures**: `fig_acf_raw_vs_demeaned.png`, `fig_acf_raw_vs_demeaned_pemsbay.png`,
`fig_acf_comparison.png`

The ACF is computed on the training portion only, averaged across sensors with <50% missing data.
Two series are shown: raw speed, and speed after subtracting the per-sensor per-time-slot mean
("de-meaned" — diurnal pattern removed).

**Raw speed ACF**: Both datasets show strong, slowly-decaying autocorrelation (ACF > 0.8 at lag
12, > 0.5 at lag 72). This reflects the dominant diurnal cycle: morning/evening rush hours recur
daily. The raw ACF is not representative of the residual structure the backbone must model, since
the backbone is trained to predict the diurnal mean.

**De-meaned ACF**: After removing the diurnal mean, the ACF decays much faster. At lag 12 (60
min, the full prediction horizon) the de-meaned ACF is:
- **METR-LA**: approximately 0.3–0.4 — moderate positive autocorrelation remains
- **PEMS-BAY**: approximately 0.2–0.3 — slightly less autocorrelated

Both decay profiles are roughly exponential over the 0–72 lag window, which motivates the
**AR(1) temporal prior** for the residuals. The faster decay in PEMS-BAY implies its residuals
are less temporally persistent — a smaller AR(1) coefficient ρ is expected.

**Key thesis observation**: The de-meaned ACF is a proxy for what the backbone residuals will
look like. If the backbone captures the diurnal mean well, its residuals should resemble the
de-meaned speed series. The Kronecker prior's AR(1) temporal factor is designed to capture
exactly this residual structure; the shape of this ACF directly motivates choosing AR(1) over
a flat (IID) or full-rank temporal covariance.

---

## 4. Spatial structure — Dirichlet energy

**Figures**: `fig_dirichlet_energy_distribution.png`, `fig_dirichlet_energy_pemsbay.png`,
`fig_dirichlet_comparison.png`

The normalised Dirichlet energy `E(t) = x_t^T L x_t / (x_t^T x_t)` measures how smooth the
speed field is with respect to the road network graph at each timestamp. A value near 0 means
adjacent sensors have very similar speeds (spatially smooth); a value near 1 means adjacent
sensors differ as much as random vectors would.

| Metric | METR-LA | PEMS-BAY |
|---|---|---|
| Mean Dirichlet energy | 0.0681 | 0.0484 |
| Median | 0.060 | — |
| Distribution shape | Right-skewed | Right-skewed |

Both values are far below 1.0, confirming that **traffic speed is a spatially smooth signal on
both road graphs** — the key empirical premise behind the graph-Laplacian spatial prior.

**PEMS-BAY (0.048) is smoother than METR-LA (0.068).** This is consistent with the datasets'
characters: PEMS-BAY covers Bay Area freeways where traffic flows more uniformly along
corridors; METR-LA includes a denser urban network with more abrupt speed transitions at
intersections and on-ramps. Concretely, the prior's spatial bandwidth parameter (the lengthscale
or decay rate in the eigenbasis) should be narrower for PEMS-BAY than for METR-LA.

**Both distributions are right-skewed**, with a spike near zero (congested periods where all
sensors are slow) and a long tail (peak hours with mixed speeds). This means the spatial
smoothness varies significantly over time — free-flow conditions are smoother than transitional
conditions, which is the motivation for regime-stratified evaluation.

---

## 5. Spatial structure — Laplacian spectrum

**Figures**: `fig_laplacian_spectrum_comparison.png`

| Property | METR-LA | PEMS-BAY |
|---|---|---|
| λ_min | 0.0000 | 0.0000 |
| λ_max | 1.7062 | 1.8083 |
| λ_mean | 1.0000 | 1.0000 |

**Both spectra are PSD** (λ_min = 0, all eigenvalues non-negative), confirming that the
symmetrize+strip-self-loops preprocessing fix is correct. Without this fix, the DCRNN adjacency
matrices (directed, with unit self-loops) produce 13 negative eigenvalues — invalidating the
Kronecker prior and ChebConv (see `DECISIONS.md`).

**The mean eigenvalue of a normalised Laplacian is always 1.0** (it is the algebraic mean of
the degree-normalised spectrum). This is a sanity check, not a coincidence.

**PEMS-BAY has a slightly wider spectrum** (λ_max = 1.808 vs 1.706). The maximum eigenvalue of
a normalised Laplacian is related to the graph's irregularity — a path graph has λ_max < 2, a
star graph approaches 2. The slightly higher λ_max for PEMS-BAY suggests its graph has a few
hub nodes with notably higher connectivity than their neighbours, which is plausible for freeway
interchange sensors.

**For ChebConv** (`compute_scaled_laplacian`), the eigenvalues are rescaled to [-1, 1] as
`2L/λ_max - I`. Since λ_max differs between datasets (1.706 vs 1.808), the scaling factor is
also different, but this is handled automatically by the function.

**For the Kronecker prior**, the spatial factor uses the eigenvectors of L as the basis. The
spectrum shape (concentrated at low frequencies, right-skewed) is typical for road network
graphs and validates that a low-pass spatial prior will concentrate prior mass on smooth
spatial patterns — appropriate for traffic.

---

## 6. Traffic regimes

**Figure**: `fig_regime_comparison.png`

Regimes are tercile splits on mean input-window speed, computed separately per dataset.

| Regime | METR-LA threshold | PEMS-BAY threshold |
|---|---|---|
| Congested | < 57.2 mph | < 61.9 mph |
| Transition | 57.2 – 61.0 mph | 61.9 – 66.3 mph |
| Free-flow | > 61.0 mph | > 66.3 mph |
| Windows per regime (test) | 2,277 of 6,831 | ~3,467 of 10,400 |

By construction, each regime contains exactly one-third of test windows. The regime labels are
used in Phase 3 for **stratified evaluation**: reporting CRPS and MAE separately for each
traffic regime to answer the question "does the structured prior help more in congestion than in
free-flow?"

**The speed thresholds differ markedly between datasets** (METR-LA "free-flow" > 61 mph,
PEMS-BAY "free-flow" > 66 mph), reflecting their different mean speeds. This means "congested
on PEMS-BAY" is still higher speed than "free-flow on METR-LA" in absolute terms — the regimes
are relative to each dataset's own distribution, not an absolute speed standard.

**Why this matters**: The backbone (DCRNN/GWN) is trained on all regimes jointly, but its
point-forecast error typically increases with congestion (non-linear dynamics, sudden speed
drops). The structured prior should provide calibrated uncertainty estimates in all regimes,
with the key test being whether it over/under-covers in congested windows where uncertainty is
inherently highest.

---

## 7. Summary and thesis implications

| Dimension | METR-LA | PEMS-BAY | Thesis implication |
|---|---|---|---|
| Scale | Smaller (207 N) | Larger (325 N) | Results on PEMS-BAY test generality at scale |
| Speed variance | High (std=19 mph) | Low (std=9 mph) | Different prior width needed |
| Missingness | Severe (8%) | Negligible (0%) | Masked loss critical for METR-LA only |
| Temporal persistence | Moderate ACF decay | Faster ACF decay | AR(1) ρ will be smaller for PEMS-BAY |
| Spatial smoothness | Moderate (D=0.068) | Higher (D=0.048) | Prior bandwidth: PEMS-BAY tighter |
| Laplacian PSD | ✓ (fixed) | ✓ (fixed) | Prior valid on both graphs |
| Regime span | 57–61 mph | 62–66 mph | Regimes are dataset-relative, not absolute |

**The two datasets are complementary.** METR-LA is a harder forecasting problem (more missing,
more variable, more congestion dynamics); PEMS-BAY is a cleaner benchmark with more data. Showing
that the structured prior improves calibration on both datasets — with different prior parameters
fitted to each — is the "framework generality" argument of the thesis.

**Prior fitting priorities** (for Phase 4):
1. Fit AR(1) ρ from the horizon-to-horizon covariance of backbone residuals (not from the raw
   ACF shown here, which includes diurnal signal).
2. Fit the spatial prior bandwidth from the spectral profile of residual covariance in the
   Laplacian eigenbasis (Phase 3, `fig_sigma_R_spectral.png`).
3. Both will differ between datasets; the prior is fitted separately per (backbone, dataset)
   combination, not shared.
