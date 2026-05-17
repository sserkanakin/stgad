"""Generate notebooks/05_stage2_analysis.ipynb — cross-dataset analysis
for Stage 2: per-mode decay curves, joint eigenvalue heatmaps, ACF fit
quality, fitted-parameter comparison table, and Dirichlet-energy diagnostic.

Five deliverables from existing artifacts only (priors, residuals,
eigendecompositions). No DAIC dependency.

Build:
    .venv/bin/python notebooks/_build_05_stage2_analysis.py
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "05_stage2_analysis.ipynb"
cells: list[tuple[str, str]] = []


def md(s: str) -> None:
    cells.append(("md", s.strip("\n")))


def code(s: str) -> None:
    cells.append(("code", s.strip("\n")))


# ─────────────────────────────────────────────────────────────────────────────
# Title + framing
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
# 05 — Stage 2 Cross-Dataset Analysis

This notebook produces the cross-dataset comparison plots that the
walkthrough notebook (`04_walkthrough.ipynb`) shows only for METR-LA.
Output is the analysis material for the meeting plot deck while the
six Stage 2 DSM training jobs run on DAIC.

**Five deliverables, all from existing artifacts on disk:**

1. **Per-mode decay curves** for all three datasets — confirms the
   forward SDE schedule exercises every joint mode on every dataset.
2. **Joint eigenvalue `s[i,j]` heatmaps** for all three datasets — the
   visual signature of each dataset's Kronecker prior.
3. **ACF fit quality** for all three datasets — shows whether the
   two-exponential temporal kernel actually fits, or degenerates (as on
   PEMS-BAY) or has a dominant slow component (PEMS08F).
4. **Cross-dataset fitted-parameter table** — α, ρ₁, ρ₂, β, σ, residual
   std, condition number of Σ_prior, plus Dirichlet ratio (item 5).
5. **Dirichlet energy diagnostic** — `mean(rᵀLr) / mean(rᵀr)` on
   standardised residuals. Low values → graph captures residual smoothness;
   high values → residuals look "random on the graph".

Inputs (no downloads or jobs needed):
- `results/prior_params/gwn_{metrla,pemsbay,pems08f}.yaml`
- `results/residuals/gwn_{ds}_train_seed0.npz`
- `data/processed/{ds}_laplacian_eigs.npz`
""")

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 1. Setup
""")

code(r"""
import sys, os
from pathlib import Path

# Locate project root
ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 9})

DATASETS = ("metrla", "pemsbay", "pems08f")
UNITS    = {"metrla": "mph", "pemsbay": "mph", "pems08f": "vehicles/5min"}
print(f"Project root: {ROOT}")
print(f"Datasets:     {DATASETS}")
""")

code(r"""
# Load everything once. All three priors + Laplacian eigs + residuals.
from src.diffusion.prior_fit import build_Q_H, build_Sigma_S
from src.diffusion.sde import eigendecompose_prior

ARTIFACTS: dict[str, dict] = {}

for ds in DATASETS:
    yaml_path = f"results/prior_params/gwn_{ds}.yaml"
    if not Path(yaml_path).exists():
        print(f"  [warn] {yaml_path} missing — skip {ds}")
        continue
    with open(yaml_path) as f:
        p = yaml.safe_load(f)

    eigs = np.load(f"data/processed/{ds}_laplacian_eigs.npz")
    ev_L = eigs["eigenvalues"].astype(np.float64)
    V_L  = eigs["eigenvectors"].astype(np.float64)

    Q_H = build_Q_H(
        delta_h=np.array(p["delta_h"]),
        alpha=p["alpha"], rho1=p["rho1"], rho2=p["rho2"],
    )
    Sigma_S = build_Sigma_S(p["beta"], p["epsilon_s"], ev_L, V_L)
    eig_np  = eigendecompose_prior(Q_H, Sigma_S)

    resid_path = f"results/residuals/gwn_{ds}_train_seed0.npz"
    if Path(resid_path).exists():
        rnpz = np.load(resid_path)
        resid = rnpz["residuals"].squeeze(-1).astype(np.float64)   # (N_win, 12, N)
        mask  = rnpz["mask"].squeeze(-1).astype(bool)
    else:
        resid = mask = None
        print(f"  [warn] {resid_path} missing — Dirichlet diag will skip {ds}")

    ARTIFACTS[ds] = {
        "params":  p,
        "Q_H":     Q_H,
        "Sigma_S": Sigma_S,
        "eig":     eig_np,
        "ev_L":    ev_L,
        "V_L":     V_L,
        "resid":   resid,
        "mask":    mask,
    }
    print(f"  {ds:>8}: N={Sigma_S.shape[0]}, s range "
          f"[{eig_np['s'].min():.3f}, {eig_np['s'].max():.3f}],  "
          f"σ={p['sigma']:.3f}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# Deliverable 1 — per-mode forward decay curves, all 3 datasets
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 2. Per-mode forward decay curves

The VP-SDE forward marginal on each joint mode `(i, j)` evolves as:

$$
x_\tau \mid x_0 \;\sim\; \mathcal{N}\!\left(\, e^{-\frac{1}{2}\bar\beta(\tau)/s_{ij}} \cdot \tilde x_0, \; s_{ij} (1 - e^{-\bar\beta(\tau)/s_{ij}}) \,\right)
$$

The decay multiplier `exp(-½ β̄(τ)/s)` controls how fast each mode's signal is
erased; the conditional variance `s · (1 - exp(-β̄/s))` saturates at `s`
when the mode reaches stationarity.

**What to look for in the plot below:**

- Each panel shows decay curves for 5 modes spanning the range of `s` values
  for that dataset: smallest, 25%-ile, median, 75%-ile, largest.
- Curves with **small `s`** drop fast (modes erased early). Curves with
  **large `s`** drop slow (modes retain signal until τ→1).
- All curves should approach 0 at τ=1 for the schedule to be doing its job.
  If a curve plateaus well above 0 at τ=1, that mode is *under-noised* —
  the score net never sees it in a near-stationary state, which hurts
  reverse-process learning.
- Compare across datasets: METR-LA and PEMS-BAY should look similar
  (broadly Gaussian-graph-conditioned spectra); PEMS08F's binary graph
  may give a more clustered distribution of `s` values.
""")

code(r"""
beta_min, beta_max = 0.01, 50.0
taus = np.linspace(0.0, 1.0, 200)
bb_curve = beta_min * taus + 0.5 * taus**2 * (beta_max - beta_min)

fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharey="row")
percentile_picks = [0, 25, 50, 75, 100]
percentile_labels = ["smallest", "25%", "median", "75%", "largest"]

for col, ds in enumerate(DATASETS):
    if ds not in ARTIFACTS:
        for r in (0, 1): axes[r, col].set_visible(False)
        continue
    s = ARTIFACTS[ds]["eig"]["s"].ravel()
    s_sorted = np.sort(s)
    n = len(s_sorted)
    picks = [s_sorted[min(max(int(p * (n - 1) / 100), 0), n - 1)] for p in percentile_picks]

    ax_d = axes[0, col]
    ax_v = axes[1, col]
    for s_val, lbl in zip(picks, percentile_labels):
        decay = np.exp(-0.5 * bb_curve / s_val)
        var   = s_val * (1 - np.exp(-bb_curve / s_val))
        ax_d.plot(taus, decay, label=f"{lbl}  (s={s_val:.2g})", lw=1.2)
        ax_v.plot(taus, var,   label=f"{lbl}  (s={s_val:.2g})", lw=1.2)
    ax_d.set_title(f"{ds} — decay  exp(−½ β̄(τ)/s)", fontsize=10)
    ax_d.set_xlabel("τ");  ax_d.grid(alpha=0.3)
    ax_v.set_title(f"{ds} — variance  s·(1 − exp(−β̄/s))", fontsize=10)
    ax_v.set_xlabel("τ");  ax_v.grid(alpha=0.3)
    ax_d.legend(fontsize=7, loc="best")
    ax_v.legend(fontsize=7, loc="best")
axes[0, 0].set_ylabel("decay multiplier")
axes[1, 0].set_ylabel("conditional variance")
fig.suptitle("Per-mode forward SDE: decay + variance across τ ∈ [0,1] — all three datasets", fontsize=11)
plt.tight_layout()

out_path = Path("experiments/score-network/comparison_seed0/fig_decay_curves_all3.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=110, bbox_inches="tight")
plt.show()
print(f"saved → {out_path}")
""")

md(r"""
**Reading the panels:**

- The top row shows how quickly each mode's signal multiplier drops. The
  largest-`s` mode (red, top of each curve) takes the longest to decay —
  on PEMS08F that's especially slow because its `s` distribution has a
  large head value.
- The bottom row shows the conditional variance growing toward each mode's
  stationary level. The lines stratify by `s` — modes with small `s`
  saturate near zero (their stationary variance is small), modes with
  large `s` saturate near a large value.
- **If any curve doesn't visibly approach saturation by τ=1**, that mode is
  under-noised. The VP-SDE forward marginal check (Phase 0 Step 5) already
  confirmed all modes reach near-stationarity for these β-schedule
  parameters, but this plot is the visual proof.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Deliverable 2 — joint eigenvalue heatmaps
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 3. Joint eigenvalue `s[i,j]` heatmaps

`s[i,j] = c_T[i] · c_S[j]` is the stationary variance under the prior of
joint mode `(i, j)`. The heatmap is the literal *picture* of each dataset's
Kronecker prior — bright cells say "the prior expects a lot of energy
here", dark cells say "expect near-zero".

Same colour scale + log normalisation across panels so the three datasets
are directly comparable.
""")

code(r"""
# 3-panel heatmap of joint s[i,j] with a shared log-norm colorbar
from matplotlib.colors import LogNorm

s_max = max(ARTIFACTS[ds]["eig"]["s"].max() for ds in ARTIFACTS)
s_min = max(min(ARTIFACTS[ds]["eig"]["s"].min() for ds in ARTIFACTS), 1e-6)

fig = plt.figure(figsize=(15, 4))
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.04])
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
cax  = fig.add_subplot(gs[0, 3])

for ax, ds in zip(axes, DATASETS):
    if ds not in ARTIFACTS:
        ax.set_visible(False)
        continue
    s = ARTIFACTS[ds]["eig"]["s"]                        # (12, N)
    im = ax.imshow(s, aspect="auto", cmap="viridis",
                   norm=LogNorm(vmin=s_min, vmax=s_max))
    ax.set_title(f"{ds}  N={s.shape[1]}\nrange [{s.min():.3g}, {s.max():.3g}]", fontsize=9)
    ax.set_xlabel("spatial mode index j")
    ax.set_ylabel("temporal mode index i")
plt.colorbar(im, cax=cax, label="s[i, j] (shared log scale)")
fig.suptitle("Joint Kronecker eigenvalues s[i, j] = c_T[i] · c_S[j]", fontsize=11)
plt.tight_layout()

out_path = Path("experiments/score-network/comparison_seed0/fig_s_heatmaps_all3.png")
fig.savefig(out_path, dpi=110, bbox_inches="tight")
plt.show()
print(f"saved → {out_path}")
""")

md(r"""
**Reading the heatmaps:**

- **Top-left corner** of each panel is the largest `s` — low-frequency
  temporal × low-frequency spatial. This is the broad-and-persistent
  prior energy (e.g. "central LA freeways under-forecast during evening
  rush").
- **Bottom-right corner** is high-frequency × high-frequency — the prior
  says: don't expect residual energy here.
- The *aspect ratio* and *gradient direction* differ across datasets
  because their `c_T` and `c_S` spectra are different. METR-LA's wider
  Σ_S spectrum gives more horizontal variation; PEMS08F's clustered
  spectrum gives a tighter column structure.
- All three heatmaps should be **smoothly graded** (no random hot pixels
  in the middle). Bright outliers in interior cells would suggest a
  numerical issue with the eigendecomposition.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Deliverable 3 — ACF fit quality
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 4. Temporal kernel fit quality — empirical ACF vs biexponential

The two-exponential temporal kernel `C_H[i,j] = α·ρ₁^|i-j| + (1-α)·ρ₂^|i-j|`
is fitted to the **lag-averaged empirical autocorrelation function** of the
training residuals. This panel shows whether that fit actually tracks the
empirical ACF, and how the two components contribute.

**Expected story across datasets:**

- **METR-LA**: a clean two-timescale fit — α around 0.5, ρ₁ around 0.85,
  ρ₂ around 0.1. Both components contribute meaningfully.
- **PEMS-BAY**: tends to *degenerate* to `ρ₁ ≈ ρ₂` (effectively an AR(1)).
  This is fine — the model has the capacity to collapse the second
  component when not needed.
- **PEMS08F**: tends to have **dominant slow** component (α ≈ 0.45,
  ρ₁ ≈ 0.87, ρ₂ ≈ 0.02). The "fast" component is almost a delta — most
  of the temporal correlation is captured by the slow ρ₁.

The three panels also report the per-dataset RMSE between empirical and
fitted ACF; a small RMSE (< 0.02) means the biexponential is doing its job.
""")

code(r"""
from src.diffusion.prior_fit import (
    empirical_temporal_cov, empirical_temporal_corr, lag_averaged_acf,
)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
for ax, ds in zip(axes, DATASETS):
    if ds not in ARTIFACTS:
        ax.set_visible(False); continue
    A = ARTIFACTS[ds]
    if A["resid"] is None:
        ax.set_visible(False); continue

    # Re-derive the empirical ACF on standardised residuals (the same scale
    # the prior was fitted on, so fit and empirical are directly comparable).
    ts = np.array(A["params"]["target_scale"])
    r_std = A["resid"] / ts[None, :, None]
    Q_emp = empirical_temporal_cov(r_std, A["mask"])
    Q_corr = empirical_temporal_corr(Q_emp)
    acf_emp = lag_averaged_acf(Q_corr)               # (11,) for lags k=1..11

    alpha, rho1, rho2 = A["params"]["alpha"], A["params"]["rho1"], A["params"]["rho2"]
    lags = np.arange(1, 12)
    acf_fit  = alpha * rho1 ** lags + (1.0 - alpha) * rho2 ** lags
    acf_slow = rho1 ** lags
    acf_fast = rho2 ** lags

    rmse = float(np.sqrt(np.mean((acf_emp - acf_fit) ** 2)))

    ax.plot(lags, acf_emp,  "o-",  color="black",      lw=1.5,  label="empirical ACF")
    ax.plot(lags, acf_fit,  "x--", color="tomato",     lw=1.3,  label=f"fit (α={alpha:.3f})")
    ax.plot(lags, acf_slow, ":",   color="tab:blue",   lw=1.0,  alpha=0.7, label=f"slow ρ₁={rho1:.3f}")
    ax.plot(lags, acf_fast, ":",   color="tab:purple", lw=1.0,  alpha=0.7, label=f"fast ρ₂={rho2:.3f}")
    ax.set_title(f"{ds}  RMSE={rmse:.4f}", fontsize=10)
    ax.set_xlabel("lag k")
    ax.set_ylabel("correlation")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
fig.suptitle("Temporal kernel fit: empirical lag-averaged ACF vs biexponential fit", fontsize=11)
plt.tight_layout()

out_path = Path("experiments/score-network/comparison_seed0/fig_acf_fit_all3.png")
fig.savefig(out_path, dpi=110, bbox_inches="tight")
plt.show()
print(f"saved → {out_path}")
""")

md(r"""
**Reading the ACF panels:**

- The black "empirical ACF" curve should track the dashed red "fit" curve
  closely. Gaps would mean the biexponential family doesn't have the right
  shape for this dataset (e.g. heavy-tailed ACF or oscillating).
- The two thin dotted curves show what each component contributes
  *individually*. Their sum, weighted by `α` and `1-α`, gives the fit curve.
- The RMSE in the title is your one-number quality metric. < 0.02 is
  "biexponential is appropriate". > 0.05 would warrant investigating
  whether a different kernel family (e.g. Matérn) fits better.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Deliverable 5 — Dirichlet energy diagnostic
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 5. Dirichlet energy diagnostic

For each (window, horizon) the residual is a vector `r ∈ ℝ^N` over sensors.
The Dirichlet energy on the graph is

$$
\text{Dir}(r) \;=\; r^\top L\, r \;=\; \tfrac{1}{2}\sum_{(i,j)\in E} A_{ij}\, (r_i - r_j)^2
$$

where `L` is the symmetric normalised Laplacian. It measures *non-smoothness
on the graph*: large when neighbouring sensors have very different residuals,
small when they agree.

We report the **mean ratio**

$$
\rho_{\text{graph}}(\text{dataset}) \;=\; \frac{\mathbb{E}[r^\top L r]}{\mathbb{E}[r^\top r]} \;\in\; [0, \lambda_{\max}(L)] = [0, 2]
$$

For a *random* signal independent of the graph, `ρ_graph` ≈ mean of `L`'s
eigenvalues ≈ 1. For a signal *smooth on the graph* (e.g. residuals that
respect the road-network neighbourhood structure), `ρ_graph` < 1. For a
signal *anti-correlated across edges* (oscillating), `ρ_graph` > 1.

**Interpretation:** smaller `ρ_graph` → graph captures real residual
structure → spatial prior `Σ_S = (β·L + ε_s·I)⁻¹` is well-motivated. Larger
`ρ_graph` → graph doesn't help much → spatial smoothing in the prior is
mostly inert.

We compute on **standardised residuals** (so the ratio is directly
comparable across datasets with different units). Masked positions are set
to zero — they contribute neither to numerator nor denominator.
""")

code(r"""
from src.data.adjacency import load_adjacency, compute_normalized_laplacian

def dirichlet_ratio(r_std: np.ndarray, mask: np.ndarray, L: np.ndarray) -> float:
    # ρ_graph = mean(rᵀLr) / mean(rᵀr) on standardised residuals.
    # r_std: (N_win, 12, N) standardised residuals.
    # mask:  (N_win, 12, N) bool; masked positions zeroed before computing.
    # L:     (N, N) symmetric normalised Laplacian.
    # Returns scalar ratio in [0, 2].
    R = (r_std * mask).reshape(-1, r_std.shape[-1])    # (N_win * 12, N)
    LR = R @ L                                          # (N_win * 12, N)
    num = (R * LR).sum(axis=1)                          # (N_win * 12,) — rᵀLr per row
    den = (R * R).sum(axis=1)
    # Weighted-mean (= overall) ratio.
    return float(num.sum() / max(den.sum(), 1e-30))


rows = []
for ds in DATASETS:
    if ds not in ARTIFACTS or ARTIFACTS[ds]["resid"] is None:
        continue
    A = ARTIFACTS[ds]
    ts = np.array(A["params"]["target_scale"])
    r_std = A["resid"] / ts[None, :, None]
    L = compute_normalized_laplacian(load_adjacency(ds))    # (N, N)
    rho = dirichlet_ratio(r_std, A["mask"], L)
    rows.append({
        "dataset": ds,
        "N sensors": L.shape[0],
        "mean λ(L)": float(A["ev_L"].mean()),
        "ρ_graph (standardised)": rho,
        "interpretation": (
            "smooth on graph" if rho < 0.8
            else "≈ random on graph" if rho < 1.2
            else "anti-correlated on graph"
        ),
    })

dirichlet_df = pd.DataFrame(rows).set_index("dataset")
dirichlet_df
""")

md(r"""
**How to read this:**

- `mean λ(L)` is the average Laplacian eigenvalue — a graph property, not
  data-dependent. It's the "baseline" Dirichlet ratio for a uniformly
  random signal.
- `ρ_graph (standardised)` is the actual ratio measured on training
  residuals.
- If `ρ_graph < mean λ(L)`, the residuals are *smoother* on the graph than
  random noise — confirming that the road-network adjacency is encoding
  real structure in the residuals.
- If `ρ_graph ≈ mean λ(L)`, the graph is *no better than random* —
  spatial prior would be redundant.
- If `ρ_graph > mean λ(L)`, the graph is *anti-correlated* with the
  residuals — wrong graph structure. This would warrant investigating
  alternative graph constructions.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Deliverable 4 — cross-dataset comparison table
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 6. Cross-dataset fitted-parameter comparison table

Single table summarising the fitted Kronecker prior across all three
datasets, plus diagnostic numbers:

- **α, ρ₁, ρ₂, β, ε_s, σ**: the fitted parameters (from each YAML).
- **mean(target_scale)**: the per-horizon scale we de-standardise by;
  rough proxy for "typical residual magnitude in raw units".
- **cond(Σ_prior)**: condition number of the joint covariance — the
  dynamic range the VP-SDE schedule must accommodate.
- **mean(residual)**: bias of the backbone forecaster.
- **std(residual)**: spread of residuals in raw units.
- **ρ_graph**: Dirichlet ratio from Section 5 — graph-smoothness of
  the standardised residuals.

This is the table for the thesis data section.
""")

code(r"""
rows = []
for ds in DATASETS:
    if ds not in ARTIFACTS:
        continue
    A = ARTIFACTS[ds]
    p = A["params"]
    s = A["eig"]["s"]
    cond_prior = float(s.max() / max(s.min(), 1e-30))

    # Residual stats over unmasked positions (raw units)
    if A["resid"] is not None and A["mask"] is not None:
        rs = A["resid"][A["mask"]]
        mean_resid = float(rs.mean())
        std_resid  = float(rs.std())
    else:
        mean_resid = std_resid = float("nan")

    rho_row = next((r for r in rows if False), None)  # placeholder lookup
    rho_graph = float(dirichlet_df.loc[ds, "ρ_graph (standardised)"]) if ds in dirichlet_df.index else float("nan")

    rows.append({
        "dataset":   ds,
        "N sensors": A["Sigma_S"].shape[0],
        "graph":     "Gaussian-kernel" if ds in ("metrla","pemsbay") else "binary",
        "α":         p["alpha"],
        "ρ₁":        p["rho1"],
        "ρ₂":        p["rho2"],
        "β":         p["beta"],
        "ε_s":       p["epsilon_s"],
        "σ":         p["sigma"],
        "mean ts[h]":float(np.mean(p["target_scale"])),
        "cond Σ_pr": cond_prior,
        "mean(r)":   mean_resid,
        "std(r)":    std_resid,
        "ρ_graph":   rho_graph,
    })
fit_df = pd.DataFrame(rows).set_index("dataset")
fit_df
""")

code(r"""
# Also save the comparison table as CSV alongside the figures.
out_path = Path("experiments/score-network/comparison_seed0/table_fitted_priors_compare.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
fit_df.to_csv(out_path)
print(f"saved → {out_path}")
""")

md(r"""
**Reading the table:**

- The **α and ρ-pair** tell you whether the temporal kernel is genuinely
  two-timescale (α far from 0 or 1, ρ₁ ≠ ρ₂) or has degenerated. PEMS-BAY
  often degenerates to AR(1); PEMS08F has a dominant ρ₁ and near-zero ρ₂.
- **β** is the spatial precision. Higher β → more spatial smoothing.
  Compare across datasets: METR-LA's denser Gaussian graph supports
  higher β; PEMS08F's sparser binary graph fits a smaller β.
- **σ** is the VP-SDE diffusion coefficient. Same β_max=50 schedule on
  all datasets, but σ adjusts to ensure all modes reach near-stationary
  by τ=1.
- **cond(Σ_prior)** above ~100 would mean the prior is very anisotropic
  and the SDE schedule struggles to accommodate it; after Phase 0's
  per-horizon standardisation these all sit in the O(10) range.
- **mean(r)** confirms the backbone is roughly unbiased on each dataset.
  std(r) is the typical residual magnitude.
- **ρ_graph** is the Dirichlet ratio — your evidence that the graph
  encodes useful spatial structure in the residuals. Lower is better.

## What to do with these plots

These four figures + two tables comprise the data-section material for
the meeting plot deck:

- `fig_decay_curves_all3.png`
- `fig_s_heatmaps_all3.png`
- `fig_acf_fit_all3.png`
- `table_fitted_priors_compare.csv`
- The Dirichlet diagnostic numbers (also in the comparison table)

The remaining Stage 2 output — *what the trained score nets actually
produce* — will land when the 6 DSM jobs finish, get evaluated, and
`scripts/compare_dsm_runs.py` produces the four CRPS-comparison tables
listed in its docstring.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Build + save
# ─────────────────────────────────────────────────────────────────────────────
nb = nbf.v4.new_notebook()
nb_cells = []
for kind, src in cells:
    if kind == "md":
        nb_cells.append(nbf.v4.new_markdown_cell(src))
    else:
        nb_cells.append(nbf.v4.new_code_cell(src))
nb["cells"] = nb_cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

OUT.write_text(nbf.writes(nb))
print(f"Wrote {OUT}  ({len(nb_cells)} cells)")
