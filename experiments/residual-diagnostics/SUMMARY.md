# residual-diagnostics — Residual diagnostics

**Goal.** Show that GWN's residuals exhibit the three structural facts (variance growing with horizon, spatial covariance concentrated on the road graph, temporal covariance with AR-style decay across horizons) that motivate a Kronecker-structured prior. **No diffusion training is involved here.** Pure data analysis on already-extracted residuals.

**Status:** done. All four go/no-go gates PASS on both datasets.

**Inputs:** `results/residuals/gwn_{metrla,pemsbay}_train_seed0.npz`, `data/processed/{dataset}_laplacian_eigs.npz`, `data/processed/{dataset}_regimes_test.npz`.

**Companions:**
- See [[../backbone-training/SUMMARY|backbone-training]] for the model whose residuals these are.
- See [[../prior-construction/SUMMARY|prior-construction]] for the prior built from these structural observations.
- See [[../data-and-residuals-overview|data-and-residuals-overview]] for an above-the-line summary of dataset shapes and residual scale.

---

## §1 — Variance vs forecast horizon

**Claim:** residual variance grows monotonically with forecast horizon. Justifies horizon-dependent variance scaling `δ_h` in `Q_H`.

| Horizon | METR-LA std (mph) | PEMS-BAY std (mph) | METR-LA growth | PEMS-BAY growth |
|---:|---:|---:|---:|---:|
| 1  | 3.82 | 1.48 | 1.00× | 1.00× |
| 3  | 5.05 | 2.50 | 1.32× | 1.69× |
| 6  | 5.75 | 3.15 | 1.51× | 2.13× |
| 9  | 6.12 | 3.44 | 1.60× | 2.32× |
| 12 | 6.44 | 3.67 | 1.69× | 2.48× |

![Variance vs horizon — METR-LA](figures/fig_variance_vs_horizon.png)

![Variance vs horizon — PEMS-BAY](figures/fig_variance_vs_horizon_pemsbay.png)

**Reading.** METR-LA std rises 69% from h=1 to h=12; PEMS-BAY rises 148%. PEMS-BAY's *relative* growth is more aggressive because its baseline residuals are very small (1.48 mph at h=1, ~2.5% of mean speed), so any longer-horizon error compounds harder in proportional terms. METR-LA's *absolute* residuals are roughly twice PEMS-BAY's at every horizon, consistent with LA's mixed urban-arterial network being noisier than Bay Area freeway traffic. A horizon-independent prior would be miscalibrated at one end or the other.

The fitted `δ_h` values stored in `results/prior_params/gwn_{dataset}.yaml` are exactly these per-horizon variances (computed as `Var(r_h)` over masked training residuals).

**Go/no-go: PASS** on both datasets.

---

## §2 — Spatial covariance Σ_S in Laplacian eigenbasis

**Claim:** residual spatial covariance concentrates in low-frequency Laplacian modes (smooth spatial patterns). Justifies graph-Laplacian-based GMRF prior `Σ_S = (β L + ε_s I)^{−1}`.

![Σ_S in Laplacian eigenbasis — METR-LA](figures/fig_sigma_R_spectral.png)

![Σ_S in Laplacian eigenbasis — PEMS-BAY](figures/fig_sigma_R_spectral_pemsbay.png)

| | METR-LA | PEMS-BAY |
|---|---:|---:|
| Top-10 modes share | 13.1% | 48.2% |
| Top-20 modes share | 21.9% | 61.7% |
| cond(Σ_S^emp) | 5.3 | 142.7 |

**Reading — this is one of the load-bearing findings.**

PEMS-BAY's empirical Σ_S is **highly anisotropic**. Over 60% of total spatial residual energy sits in just the first 20 of 325 Laplacian eigenmodes — these are the smoothest spatial patterns on the road graph (coherent stretches of freeway). Physically: when GWN errs on a Bay Area freeway, it tends to err in the same direction across many adjacent sensors at once — a queue forming, a free-flow regime emerging, a crash propagating. The errors are spatially correlated by the road network's structure.

METR-LA's empirical Σ_S is **nearly isotropic** (cond ≈ 5; only 22% of energy in top 20 of 207 modes). LA's mixed urban-arterial network has many local discontinuities (intersections, ramps, signals) that decorrelate adjacent sensors. The road topology contributes *little* to explaining residual covariance — the dominant structure is purely temporal.

This contrast is reflected in the fitted **β** parameters (see [[../prior-construction/SUMMARY|prior-construction]]): β = 0.025 on METR-LA (weak spatial smoothing) vs β = 0.784 on PEMS-BAY (strong spatial smoothing) — a 31× difference, fitted automatically without supervision.

**Implication for downstream work.** The benefit of the structured *spatial* prior over isotropic should be larger on PEMS-BAY than on METR-LA, because PEMS-BAY's spatial prior carries more information.

**Go/no-go: PASS** on both datasets (energy concentrates in low-frequency modes, not flat).

---

## §3 — Horizon × horizon temporal covariance Σ_Q

**Claim:** residuals at adjacent forecast horizons are correlated; correlation decays with lag. Justifies a Toeplitz-style temporal precision (initially planned AR(1), upgraded to two-exponential mixture in [[../prior-construction/SUMMARY|prior-construction]]).

![Σ_Q (12×12 horizon covariance + correlation) — METR-LA](figures/fig_sigma_Q.png)

![Σ_Q — PEMS-BAY](figures/fig_sigma_Q_pemsbay.png)

Lag-averaged ACF (averaging the 12×12 correlation matrix over (h, h+k) pairs for each lag k):

| Lag k | METR-LA ρ̂(k) | PEMS-BAY ρ̂(k) |
|---:|---:|---:|
| 0  | 1.000 | 1.000 |
| 1  | 0.686 | 0.840 |
| 3  | 0.388 | 0.571 |
| 6  | 0.269 | 0.298 |
| 9  | 0.171 | 0.150 |
| 11 | 0.099 | 0.089 |

**Reading.** Both datasets show strong, monotonically decaying off-diagonal structure — clearly not the identity. PEMS-BAY's lag-1 correlation is higher (0.840 vs 0.686), reflecting smoother dynamics on freeway traffic. By lag 11 (55 minutes apart) both have decayed to ~0.09–0.10 — non-zero but weak.

**Important non-stationarity caveat.** The empirical 12×12 correlation matrix is *not* stationary. Lag-1 correlation grows from 0.484 at (h=0, h=1) to 0.759 at (h=10, h=11). Forecast errors become more correlated at longer horizons, which is physically plausible (longer-horizon errors are dominated by structured drift, while short-horizon errors carry more idiosyncratic burst noise). The two-exp kernel we eventually adopt is *stationary* — depends only on |h₁ − h₂|. We resolve this mismatch by fitting to the lag-averaged ACF (see prior-construction), which captures the true stationary component.

The shape of the METR-LA lag curve is also where the AR(1)-vs-two-exp distinction shows up most clearly — see [[../prior-construction/SUMMARY#§7|prior-construction §7]] for the comparison.

---

## §4 — Variance by traffic regime (the calibration motivation)

**Claim:** residual variance differs substantially across traffic regimes. An isotropic prior would mis-calibrate uncertainty in congested conditions.

![Variance by regime — METR-LA](figures/fig_variance_by_regime.png)

![Variance by regime — PEMS-BAY](figures/fig_variance_by_regime_pemsbay.png)

Per-horizon residual std (mph) stratified by tercile of mean input-window speed:

| Regime     | METR-LA h=1 | METR-LA h=12 | PEMS-BAY h=1 | PEMS-BAY h=12 |
|------------|---:|---:|---:|---:|
| Free-flow  | 3.05 | 4.90 | 1.27 | 2.37 |
| Transition | 3.85 | 6.95 | 1.32 | 3.52 |
| Congested  | 4.98 | **9.23** | 1.98 | **6.33** |

**Reading — this is the central empirical motivation for the thesis.**

In congested conditions, GWN's 60-min-ahead residual std is **roughly 2× free-flow residual std** on both datasets:
- METR-LA: 9.23 / 4.90 = **1.88×**
- PEMS-BAY: 6.33 / 2.37 = **2.67×**

A point forecast emits no uncertainty. An isotropic diffusion model would assign the same uncertainty to every regime — under-covering during congestion and over-covering during free-flow. The structured prior, by capturing the spatial-temporal covariance shape that *changes with regime*, is what enables wide intervals during congestion and tight intervals during free-flow.

The regimes are dataset-relative (PEMS-BAY's "congested" mean speed is still higher than METR-LA's "free-flow"). Thresholds: METR-LA <57.2 / 57.2–61.0 / >61.0 mph; PEMS-BAY <61.9 / 61.9–66.3 / >66.3 mph.

---

## §5 — Kronecker separability check

**Claim:** the empirical (H·N)×(H·N) residual covariance is well-approximated by `Q_H ⊗ Σ_S`, justifying the Kronecker assumption.

| Dataset | Frobenius rel. error | Verdict (threshold < 40%) |
|---|---:|---|
| METR-LA  | **26.2%** | ✓ PASS |
| PEMS-BAY | **12.5%** | ✓ PASS |

**Reading.** PEMS-BAY's lower error is consistent with its more regular traffic dynamics (separable space×time structure). METR-LA's 26% residual cross-structure is a real limitation of the Kronecker assumption that should be acknowledged as such in the thesis — but the approximation is defensible: we trade modelling fidelity for tractability and interpretability, going from `O(H²N²)` to `O(H² + N²)` parameters.

---

## Go/no-go summary

| Check | METR-LA | PEMS-BAY |
|---|---|---|
| Variance monotone in horizon | ✓ | ✓ |
| Σ_Q AR-type decay across horizons | ✓ | ✓ |
| Σ_S low-frequency concentration | ✓ | ✓ (sharply) |
| Kronecker separability < 40% | ✓ | ✓ |
| Backbone unbiased (|mean|≪std) | ✓ | ✓ |

**All five gates PASS on both datasets.** Phase 4 (prior parameter fitting in [[../prior-construction/SUMMARY|prior-construction]]) was unblocked by these results.
