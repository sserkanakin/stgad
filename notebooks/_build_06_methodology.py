"""Generate notebooks/06_methodology_overview.ipynb — a clear, code-grounded
walkthrough of the WHOLE methodology so a new reader can read it top-to-bottom
and understand every choice + every result.

Run from the repo root:
    .venv/bin/python notebooks/_build_06_methodology.py

Writes notebooks/06_methodology_overview.ipynb (not pre-executed; open and Run-All).
"""
from __future__ import annotations
import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).parent / "06_methodology_overview.ipynb"

cells: list[tuple[str, str]] = []


def md(s: str) -> None:
    cells.append(("md", s.strip("\n")))


def code(s: str) -> None:
    cells.append(("code", s.strip("\n")))


# ─────────────────────────────────────────────────────────────────────────────
# 0. Title + reader contract + table of contents
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
# 06 — Methodology Overview (code-grounded)

This notebook is a single linear read-through of **the whole pipeline** as it
currently stands in this repo. It is meant to be the document you open when
you ask *"how does this method actually work, end to end, and why is it built
that way?"*

It is **not** a results report. The numbers you see here come from this repo
and are explained, but the goal is understanding — not running every step.

For each piece of the system we cover four things:

1. **What it is** — the object and its math.
2. **Why this choice** — what alternatives exist and why we picked this one.
3. **Where it lives in the code** — file and function pointer.
4. **A small executable demo** — short code cell that prints a number you can
   poke at.

The last two sections (§11–§12) discuss **results and what is going wrong** —
specifically the catastrophic PEMS-BAY structured-prior eval, the marginal
METR-LA structured eval, and what each failure tells us.

---

## Table of contents

1. The problem and the four datasets
2. Notation
3. Data pipeline: windows, splits, masks, normalisation
4. Backbone (Graph WaveNet) — what residuals we model
5. The residual signal — why structured diffusion at all
6. The Kronecker prior   `Σ_prior = Q_H ⊗ Σ_S`
   - 6a. Spatial factor   `Σ_S = (β L + ε_s I)⁻¹`
   - 6b. Temporal factor  `C_H = α·ρ₁^k + (1−α)·ρ₂^k`
   - 6c. Per-horizon standardisation `target_scale`
   - 6d. Fitted parameters table
7. The forward SDE — closed-form marginal in the eigenbasis
8. The score network — FiLM-conditioned GWN block
9. The DSM loss — Tweedie / x₀-prediction with eigenbasis weighting
10. Reverse sampling — Euler-Maruyama and its stability condition
11. Evaluation — CRPS, PIT, pinball
12. **What the current results say**  (Stage-2, seed 0)
13. **What is going wrong and why** (the two failure modes)
14. Pointers — where each artefact lives on disk
""")

# ─────────────────────────────────────────────────────────────────────────────
# Setup cell
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## Setup

Run this once. It adds the repo root to `sys.path` so `import src.*` works,
and pre-loads NumPy / Matplotlib / YAML.
""")

code(r"""
import sys, os, json, yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
print("repo root:", ROOT)
""")

# ─────────────────────────────────────────────────────────────────────────────
# §1. Problem + datasets
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 1. The problem and the four datasets

We forecast traffic speeds on a sensor network at horizons of 5 to 60 minutes
ahead. Concretely, given the last 12 readings (60 minutes of history at 5-min
cadence) on every sensor, predict the next 12 readings on every sensor.

For each input window:

- input  `x ∈ ℝ^{12 × N × 2}`  — last 12 frames of (speed, time-of-day) per sensor
- target `y ∈ ℝ^{12 × N}`     — next 12 frames of speed per sensor

The novelty is not point forecasting — Graph WaveNet (GWN) handles that well —
but **probabilistic** forecasting: we want a full predictive distribution
`p(y | x)` so we can report uncertainty intervals. The thesis claim is that
making that distribution **structured** (correlated across sensors via the
road graph, correlated across horizons via a temporal kernel) gives sharper,
better-calibrated forecasts than an isotropic Gaussian.

Three datasets, picked to span easy → medium → hard:

| Dataset    | Sensors N | Cadence | Target variable | Region                  | Notes |
|------------|----------:|---------|-----------------|-------------------------|-------|
| METR-LA    | 207       | 5 min   | speed (mph)     | LA highways + arterials | Urban, mixed dynamics, ~7% missing |
| PEMS-BAY   | 325       | 5 min   | speed (mph)     | Bay Area freeways       | Smooth dynamics, ~0% missing |
| PEMS08F    | 170       | 5 min   | **flow** (veh/h)| San Bernardino corridor | Different *variable*, single corridor, ~0% missing |

The first two datasets are the standard DCRNN / GWN benchmarks. PEMS08F is a
deliberate **out-of-distribution** stress test: it is not a speed dataset, it
is not a sensor *network* (just a long corridor), and the prior fitted on it
ends up qualitatively different from the other two (see §5b). We carry it
through every step so that "does structured residual diffusion help?" is
answered on three datasets, not two.
""")

code(r"""
import numpy as np
for ds in ["metrla", "pemsbay", "pems08f"]:
    Z = np.load(f"data/processed/{ds}_train.npz")
    print(f"{ds:10s}  x {Z['x'].shape}  y {Z['y'].shape}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# §2. Notation
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 2. Notation (one place to look it up)

| Symbol | Meaning |
|---|---|
| `N` | number of sensors |
| `H = 12` | forecast horizons (5,10,…,60 min) |
| `y, ŷ ∈ ℝ^{H × N}` | true / backbone-predicted speed |
| `r = y − ŷ ∈ ℝ^{H × N}` | residual (what the diffusion models) |
| `target_scale[h]` | per-horizon std of raw residuals; used to standardise |
| `r̃[h,n] = r[h,n] / target_scale[h]` | **standardised** residual (unit per-horizon variance) |
| `Σ_S ∈ ℝ^{N × N}` | spatial covariance (GMRF) |
| `C_H, Q_H ∈ ℝ^{H × H}` | temporal correlation, temporal covariance |
| `Σ_prior = Q_H ⊗ Σ_S` | full Kronecker prior covariance |
| `Q = Σ_prior⁻¹` | precision matrix |
| `τ ∈ [τ_eps, 1]` | diffusion time (forward: 0 → 1; reverse: 1 → τ_eps) |
| `β(τ)` | noise-rate schedule, `g(τ)² = β(τ)` |
| `β̄(τ) = ∫₀^τ β(s) ds` | cumulative schedule (closed-form: see §7) |
| `r_τ` | residual at diffusion time τ |
| `s[i,j] = λᵢ(Q_H) · λⱼ(Σ_S)` | eigenvalues of `Σ_prior` in the joint basis |
| `q = 1/s` | eigenvalues of `Q` in the joint basis |
""")

# ─────────────────────────────────────────────────────────────────────────────
# §3. Data pipeline
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 3. Data pipeline

`scripts/preprocess.py` produces three `.npz` files per dataset (`train`,
`val`, `test`) plus a Laplacian eigendecomposition `*_laplacian_eigs.npz`.

Conventions (locked, from the project memory):

- **Windowing**: 12-frame history → 12-frame target, stride 1.
- **Splits**: standard DCRNN/GWN split (chronological 70/10/20 for METR-LA &
  PEMS-BAY; PEMS08F uses the published train/val/test indices).
- **`y` is never normalised on disk.** Only the speed channel of `x` is
  z-scored at load time (so backbone training matches Wu 2019).
- **Mask**: `(x != 0)` for the speed channel — missing readings in METR-LA
  appear as exact zeros. Loss and metrics are computed over masked positions.
- **No data augmentation.** Fixed seeds 0–4 for the multi-seed runs.

Below: load one window and show shapes.
""")

code(r"""
import numpy as np
Z = np.load("data/processed/metrla_train.npz")
x, y = Z["x"], Z["y"]
print("x shape (W, 12, N, 2):", x.shape)
print("y shape (W, 12, N):   ", y.shape)
print("speed range x[..., 0]:", x[..., 0].min(), x[..., 0].max())
print("time-of-day x[..., 1]:", x[..., 1].min(), x[..., 1].max())
""")

# ─────────────────────────────────────────────────────────────────────────────
# §4. Backbone — what residuals we model
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 4. The backbone (Graph WaveNet)

We use **Graph WaveNet** (Wu et al., IJCAI 2019) as a deterministic baseline.
GWN is a stack of dilated temporal convolutions + adaptive-graph convolutions
that maps `x → ŷ`. It is not modified — we use the upstream architecture
exactly (`third_party/Graph-WaveNet` @ commit `6b162e8`) with a small Conv1d→
Conv2d patch on load so the 4-D residual shape stays consistent.

Why GWN and not DCRNN or STID?

- We need a competent backbone — the diffusion model corrects its *residuals*,
  so a weak backbone would make the residual signal dominated by trivial
  prediction error.
- GWN is well-known, easy to reproduce, and has a clean adaptive-adjacency
  formulation that's interesting to reuse in the score network (see §8).
- DCRNN is kept around as an alternative (with a TensorFlow shim) but the
  current production runs use GWN only.

**Training**: Adam, lr=1e-3, masked MAE loss, gradient clip 5.0, 100 epochs,
seed 0. Validation MAE@h=12 reaches 3.25 mph on METR-LA (Wu 2019: 3.53) and
1.91 mph on PEMS-BAY (Wu 2019: 1.95). The trained checkpoint is then used to
extract residuals.

**Residual extraction** (`scripts/extract_residuals.py`):

```
r[w, h, n] = y[w, h, n] − GWN(x[w])[h, n]
```

The residuals + masks for each split are saved in
`results/residuals/gwn_{dataset}_{split}_seed0.npz`. This is the only input
to everything downstream — the diffusion model never sees raw speeds again.
""")

code(r"""
import numpy as np
R = np.load("results/residuals/gwn_metrla_train_seed0.npz")
# Residual file ships y_pred / y_true / residuals / mask, with a trailing
# singleton "channel" axis kept for shape symmetry with the loader.
r = R["residuals"][..., 0]    # (W, 12, 207)
m = R["mask"][..., 0]         # (W, 12, 207)
print("residuals shape:", r.shape, " mask shape:", m.shape)

# Per-horizon std on masked positions — should rise monotonically with horizon.
for h in [0, 5, 11]:
    vals = r[:, h, :][m[:, h, :]]
    print(f"  h={h+1:2d}  std(r) = {vals.std():.3f} mph  mean = {vals.mean():+.3f}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# §5. Why structured residual diffusion
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 5. Why structured residual diffusion?

Three empirical facts about GWN's residuals justify the choice. Numbers come
from `experiments/residual-diagnostics/` (METR-LA + PEMS-BAY).

**Fact 1 — variance grows with horizon.**

Per-horizon std rises 69% (METR-LA) and 148% (PEMS-BAY) from h=1 to h=12. An
isotropic prior with a single σ would be miscalibrated at one end. Hence the
per-horizon scaling `target_scale[h]`.

**Fact 2 — spatial errors are graph-correlated, but the *amount* depends on
the dataset.**

Projecting the empirical sensor×sensor covariance onto the road graph's
Laplacian eigenbasis, **PEMS-BAY puts 62% of the residual energy in the top
20 (of 325) smoothest modes**; **METR-LA puts only 22% in the top 20 (of
207)**. PEMS-BAY's freeway network produces coherent forecast errors along
entire stretches; LA's mixed grid does not. This justifies a *graph-Laplacian
GMRF* spatial prior with a fitted strength parameter β.

**Fact 3 — residuals at adjacent horizons are correlated.**

Lag-averaged ACF at lag k=1 is 0.686 on METR-LA, 0.840 on PEMS-BAY. By
lag 11 it has decayed to ~0.09. A Toeplitz-style temporal correlation matrix
is justified, and on METR-LA the lag curve has **two timescales** (a slow
component that persists, a fast burst component) — hence the two-exponential
kernel.

**Fact 4 (the calibration argument) — variance is regime-dependent.**

Stratifying residuals by tercile of input speed (free-flow / transition /
congested), congested-regime h=12 residual std is ≈2× free-flow std on both
datasets. An isotropic forecast would assign the same uncertainty everywhere
and be wrong in both regimes. Structured forecasting can be tight in
free-flow, wide in congestion.

These are the four ingredients the prior in §6 has to capture.

> **A note on the three numbers above being only METR-LA / PEMS-BAY.** The
> residual diagnostics in `experiments/residual-diagnostics/` were run on the
> two speed datasets where the empirical motivation for the GMRF + two-exp
> prior was developed. PEMS08F was added later (different *variable*, see
> §1) and the same pipeline was fitted to it without rerunning the
> diagnostics from scratch. The fitted parameters for it (§6d) tell us
> directly whether the same prior class is appropriate — see §5b.
""")

# ─────────────────────────────────────────────────────────────────────────────
# §5b. PEMS08F — what it tells us, and why it's in the lineup
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 5b. PEMS08F — the negative control

PEMS08F is **not** a speed dataset; the target is **traffic flow** in
vehicles/hour. The numbers look very different from the other two:
""")

code(r"""
import numpy as np
Z = np.load("data/processed/pems08f_test.npz")
y = Z["y"][..., 0]
print(f"pems08f y range: {y.min():.0f} – {y.max():.0f}  veh/h")
print(f"pems08f y mean : {y.mean():.1f}  std: {y.std():.1f}")

R = np.load("results/residuals/gwn_pems08f_train_seed0.npz")
r = R["residuals"][..., 0]; m = R["mask"][..., 0]
for h in [0, 5, 11]:
    v = r[:, h, :][m[:, h, :]]
    print(f"  h={h+1:2d}  std(r) = {v.std():6.2f} veh/h   ratio vs h=1 = {v.std()/r[:, 0, :][m[:, 0, :]].std():.2f}×")
""")

md(r"""
Three things make PEMS08F qualitatively different from METR-LA / PEMS-BAY,
and together they make it the cleanest **negative control** in the lineup:

**1. The residual scale is much larger and grows *less* with horizon.**

PEMS08F residual std goes from ≈20.6 to ≈25.6 veh/h between h=1 and h=12 —
only a 1.24× growth, vs 1.68× on METR-LA and 2.48× on PEMS-BAY. Why?
Flow is much less volatile minute-to-minute than speed — the *level* drifts
slowly, but instantaneous noise is high. So most of the residual energy is
already present at h=1 and additional horizon doesn't add much.

This means the *per-horizon-standardisation* step (§6c) buys us less on
PEMS08F: `target_scale[h]` only ranges over ≈21→26, so the standardised
residuals are almost in unit-variance shape already. The thesis claim "the
structured prior matters when the horizon-dependent variance is strong"
already predicts that PEMS08F should benefit less than the speed datasets.

**2. The temporal prior is well-separated two-exp.**

Fitted parameters on PEMS08F (§6d): `α=0.455, ρ₁=0.869, ρ₂=0.022`. ρ₂ is
nearly zero — the fast component basically encodes "lag-1 burst noise
decorrelates immediately", while the slow component captures persistent
drift. This is the *cleanest* two-exp fit of the three datasets: more
separated than METR-LA's (0.832, 0.102) and unambiguous (vs PEMS-BAY's
degeneracy to single AR(1), §6d). The two-exp parameterisation is
**earning its complexity here**.

**3. The graph is a single corridor, and `β` reflects that.**

PEMS08F's road graph is one long-and-thin corridor on I-08 in San Bernardino,
not a sensor network. The fitted `β = 0.120` is the smallest of the three —
spatial smoothing is weak, because there's no rich graph structure for the
GMRF to exploit. Errors propagate along a line, not across a topology.

**4. The Kronecker prior is well-conditioned here, so the sampler is safe.**

`q_max = 2.23` on PEMS08F vs 5.6 (METR-LA) and 14.1 (PEMS-BAY). The reverse
Euler-Maruyama is comfortably stable at `n_steps = 100` (§10). So whatever
PEMS08F's Stage-2 result says, it isn't an integrator artefact.

### What the Stage-2 result on PEMS08F means

CRPS_norm averaged across horizons (from `table3_struct_vs_iso.csv`):

| Prior      | h=3   | h=6   | h=12  | avg   |
|------------|------:|------:|------:|------:|
| structured | 0.0455 | 0.0488 | 0.0536 | 0.0487 |
| isotropic  | 0.0456 | 0.0488 | 0.0537 | 0.0488 |
| **Δ (iso − struct)** | +0.0001 | +0.0001 | +0.0000 | **+0.0001** |

Tie to four decimal places. Coverage and pinball-loss tables tell the same
story. **On PEMS08F the structured prior neither helps nor hurts.**

This is *useful information*, not a null result to apologise for. It says:

- The structured-prior pipeline does **not** make things worse on a dataset
  where its assumptions don't pay off (the prior shape it imposes is mild
  enough that the score net can learn around it without paying a
  calibration tax).
- The pipeline's *complexity overhead* — fitting Σ_S, Q_H, doing
  eigenbasis-weighted DSM, running a structured reverse SDE — is therefore
  defensible: on a dataset where graph structure is weak, you fall back
  cleanly to isotropic-equivalent behaviour.
- If a future iteration of the methodology *did* hurt PEMS08F, that would
  be the canary for "the structured prior is now too restrictive". Right
  now it's not.

This is the negative-control role PEMS08F plays in the thesis. The
*positive* claim has to be carried by METR-LA (and, once the sampler bug
in §13 is fixed, hopefully PEMS-BAY).
""")

# ─────────────────────────────────────────────────────────────────────────────
# §6. The Kronecker prior
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 6. The Kronecker prior

The prior is

$$
\Sigma_{\text{prior}} \;=\; Q_H \,\otimes\, \Sigma_S,\qquad
r \in \mathbb{R}^{H \times N}.
$$

Conceptually: temporal structure and spatial structure are assumed
**separable**. This is a strong assumption — it forces every spatial mode to
decay in time the same way and vice versa. We pay a known accuracy cost for
that (see [project-thesis-diffusion] §3: per-mode variance ratios span 0.32×
to 2,506× on METR-LA). The trade is computational: every operation in §7–§10
costs `O(HN)` per sample because we work in the joint eigenbasis where
`Σ_prior` is **diagonal**.

The full prior covariance has size `(H·N) × (H·N) ≈ 2484×2484` for METR-LA,
never materialised explicitly.
""")

md(r"""
### 6a. Spatial factor `Σ_S` — graph-Laplacian GMRF

$$
\Sigma_S \;=\; (\beta L + \varepsilon_s I)^{-1}
$$

where `L` is the symmetric-normalised Laplacian of the road-network adjacency
matrix (eigenvalues in `[0, 2]`). The eigenvectors of `L` are smoothness
basis functions on the graph: the first one (eigenvalue 0) is constant, the
next ones are increasingly "wavy" patterns.

The inverse `(βL + ε_s I)^{-1}` has the same eigenvectors, with eigenvalues
`1 / (β·λᵢ(L) + ε_s)`. Large `β` means smooth patterns (small Laplacian
eigenvalues) get **large** prior variance, encouraging samples that vary
slowly along the road graph. Small `β` means flat / nearly isotropic.

**Why a GMRF and not full empirical Σ_S?**

- Identifiability: full `Σ_S` has `N²/2 ≈ 53k` free parameters on PEMS-BAY;
  we have a few million training residuals but they are highly correlated
  across time. A full `Σ_S` overfits.
- Interpretability: `β` is one number that says "how graph-smooth are
  forecast errors on this dataset?".

**Why `ε_s` is fixed at 1.0 (not 0.01)?**

This is the 2026-05-03 standardisation upgrade. After we standardise the
residuals (§6c), the constant Laplacian mode (eigenvalue 0) has empirical
variance ≈ 1, so the prior's variance for that mode is `1/(β·0 + ε_s) = 1/ε_s`,
which we set to 1. Earlier, on raw mph² residuals, the same mode had variance
~25 mph², which forced `ε_s ≈ 0.04`. Decoupling scale (via `target_scale`)
from structure (via `β`) gave us `ε_s = 1.0` for free.

Code:  `src/diffusion/prior_fit.py::fit_spatial_beta`,
       `src/diffusion/prior_fit.py::build_Sigma_S`.
""")

code(r"""
import numpy as np
import yaml

ds = "metrla"
prior = yaml.safe_load(open(f"results/prior_params/gwn_{ds}.yaml"))
eig_L = np.load(f"data/processed/{ds}_laplacian_eigs.npz")
from src.diffusion.prior_fit import build_Sigma_S
Sigma_S = build_Sigma_S(prior["beta"], prior["epsilon_s"],
                        eig_L["eigenvalues"], eig_L["eigenvectors"])
print(f"{ds}: β = {prior['beta']:.4f}  ε_s = {prior['epsilon_s']}")
print(f"Σ_S shape {Sigma_S.shape}")
print(f"Σ_S eigval range: [{Sigma_S.shape and np.linalg.eigvalsh(Sigma_S).min():.4f}, "
      f"{np.linalg.eigvalsh(Sigma_S).max():.4f}]")
print(f"cond(Σ_S) = {np.linalg.cond(Sigma_S):.2f}")
""")

md(r"""
### 6b. Temporal factor `C_H` — two-exponential mixture

Empirical lag-averaged autocorrelation `ρ̂(k)` of residuals across horizons
**is not** a single exponential on METR-LA — the lag curve has a sharp drop
in the first two steps and then flattens for 4–11 step gaps. A single AR(1)
`ρ^k` cannot fit both behaviours.

$$
C_H[h_1, h_2] \;=\; \alpha\,\rho_1^{|h_1 - h_2|} \;+\; (1-\alpha)\,\rho_2^{|h_1-h_2|},
\qquad \rho_1 \geq \rho_2 \in (0, 1).
$$

- `ρ₁` (slow, large): captures persistent forecast drift (a sustained
  systematic error across many horizons — e.g. a missed regime change).
- `ρ₂` (fast, small): captures burst errors that decorrelate within ~2 steps.
- `α`: relative weight of the slow component.

**Why two exponentials and not three / Matern / state-space?**

- AR(1) (`ρ^k`) misfits METR-LA: lag-curve RMSE 0.043 vs 0.003 for two-exp,
  ΔAIC = 61, ΔBIC = 60. **A 15× RMSE drop justifies the upgrade on METR-LA.**
- A two-exp is a strict superset of AR(1). On PEMS-BAY it **degenerates**
  cleanly to `ρ₁ ≈ ρ₂ ≈ 0.819` (single timescale), giving the same fit at
  the cost of a 4-point AIC penalty. Letting the model decide is fine.
- Three exponentials and Matern kernels were not pursued — the residual
  energy at lag > 2 is too low to identify a third component reliably.

The fit is Nelder-Mead least-squares against the **lag-averaged** empirical
ACF (lags 1..11), sigmoid-parameterised, 8 deterministic + 4 random restarts
to avoid the degenerate ρ₁=ρ₂ local minimum.

`C_H` is a correlation matrix. The covariance `Q_H` is

$$
Q_H[h_1, h_2] = \sqrt{\delta_{h_1}\,\delta_{h_2}}\cdot C_H[h_1, h_2]
$$

After standardisation (§6c), `δ_h ≈ 1` for all h, so `Q_H ≈ C_H`.

Code:  `src/diffusion/prior_fit.py::fit_temporal_kernel`,
       `src/diffusion/prior_fit.py::build_Q_H`.
""")

code(r"""
import numpy as np
import yaml
from src.diffusion.prior_fit import build_Q_H

ds = "metrla"
prior = yaml.safe_load(open(f"results/prior_params/gwn_{ds}.yaml"))
delta_h = np.array(prior["delta_h"], dtype=float)
Q_H = build_Q_H(delta_h, prior["alpha"], prior["rho1"], prior["rho2"])
print(f"{ds}:  α = {prior['alpha']:.3f}   ρ₁ = {prior['rho1']:.4f}   ρ₂ = {prior['rho2']:.4f}")
print("Q_H[0:5, 0:5] =\n", np.round(Q_H[:5, :5], 3))
print(f"cond(Q_H) = {np.linalg.cond(Q_H):.2f}")
print(f"Q_H eigval range: [{np.linalg.eigvalsh(Q_H).min():.4f}, {np.linalg.eigvalsh(Q_H).max():.4f}]")
""")

md(r"""
### 6c. Per-horizon standardisation `target_scale`

The diffusion model operates on **standardised** residuals:

$$
\tilde r[h, n] = r[h, n] / \texttt{target\_scale}[h],\qquad
\texttt{target\_scale}[h] = \sqrt{\operatorname{Var}(r_h)} \;\;(\text{mph}).
$$

After training, samples are de-standardised by multiplying by
`target_scale[h]` per horizon, then added back to the backbone forecast:

$$
y_{\text{sample}} = \hat y_{\text{backbone}} + \tilde r \cdot \texttt{target\_scale}.
$$

Why standardise?

1. **The SDE schedule needs unit-scale data.** With raw mph² residuals,
   `Σ_prior` eigenvalues are `O(10⁴)`, the forward process barely diffuses by
   `τ=1` with any reasonable `β_max`, and the score values explode. With
   standardised residuals, eigenvalues are `O(1)–O(10)` and `β_max = 50`
   reaches the stationary distribution at `τ = 1` in 99%+ of joint modes.
2. **Per-horizon variance growth is absorbed by `target_scale`.** The
   structural part of the prior (Q_H, Σ_S) then only has to model
   *correlation*, not *scale*. This is what makes the AR(1)-vs-two-exp
   comparison clean (correlations are scale-invariant).
3. **A unit-variance variable is what FiLM/Adam expect.** Network weights
   trained on `O(1)` activations are stable; on `O(10²)` activations they
   are not.

Code:  `src/diffusion/prior_fit.py::fit_prior` (the `standardize=True` branch).
""")

code(r"""
import numpy as np, yaml
for ds in ["metrla", "pemsbay", "pems08f"]:
    p = yaml.safe_load(open(f"results/prior_params/gwn_{ds}.yaml"))
    ts = np.array(p["target_scale"], dtype=float)
    print(f"{ds:10s}  target_scale  h1={ts[0]:.3f}  h6={ts[5]:.3f}  h12={ts[11]:.3f}   ratio h12/h1 = {ts[11]/ts[0]:.2f}×")
""")

md(r"""
### 6d. Fitted parameters — side-by-side

These come from `results/prior_params/gwn_*.yaml`. The fits are fully
unsupervised: just minimise the lag-curve / spectral objectives on the
training-split residuals. **Nothing here is hand-tuned**.

| Parameter      | METR-LA  | PEMS-BAY | PEMS08F | What it does |
|----------------|---------:|---------:|--------:|--------------|
| `α`            | 0.799    | **0.005**| 0.455   | weight on slow temporal component |
| `ρ₁` (slow)    | 0.832    | 0.81870  | 0.869   | slow-component decay |
| `ρ₂` (fast)    | 0.102    | **0.81870** | 0.022| fast-component decay |
| `β`            | 0.202    | 0.237    | 0.120   | graph-Laplacian strength |
| `ε_s`          | 1.0      | 1.0      | 1.0     | Laplacian floor (fixed) |
| `σ` (derived)  | 2.84     | 3.02     | 2.61    | VP-SDE diffusion coeff |
| `δ_h` (mean)   | ≈1.0     | ≈1.0     | ≈1.0    | per-horizon var (≈1 by construction) |
| `target_scale[h=12]` | 6.44 mph | 3.67 mph | 25.6 mph | raw residual scale at h=12 |

**Things to flag**:

- PEMS-BAY's α=0.005 with ρ₁=ρ₂ means **two-exp has degenerated to single-
  exponential**. Mathematically this is a single AR(1) with ρ=0.819. The
  fit is correct; it's just that PEMS-BAY's lag curve doesn't have two
  timescales.
- PEMS08F's target_scale[h=12]=25.6 mph is much larger than METR-LA's
  (6.4) and PEMS-BAY's (3.7). Different sensor regime entirely; the
  residual magnitudes live on a different scale.
""")

# ─────────────────────────────────────────────────────────────────────────────
# §7. Forward SDE
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 7. The forward SDE

We use the **VP-SDE** (variance-preserving) with the Kronecker prior as the
stationary distribution:

$$
\mathrm d r_\tau \;=\; -\tfrac{1}{2}\,g(\tau)^2\,Q\,r_\tau\,\mathrm d\tau \;+\; g(\tau)\,\mathrm d W_\tau,
\qquad Q = \Sigma_{\text{prior}}^{-1}.
$$

**Schedule** (linear `β`, locked):

$$
\beta(\tau) = \beta_{\min} + \tau(\beta_{\max} - \beta_{\min}),\qquad
g(\tau)^2 = \beta(\tau),\qquad
\bar\beta(\tau) = \int_0^\tau \beta(s)\,\mathrm d s = \beta_{\min}\tau + \tfrac{1}{2}\tau^2(\beta_{\max}-\beta_{\min}).
$$

with `β_min = 0.01`, `β_max = 50`. Conventional `β_max ≈ 20` from DDPM does
not fully diffuse this prior; the `Σ_prior` worst-case condition number
(see §6d) requires `σ ≈ 2.8–3.0` (which corresponds to `β_max` in this range).

**The forward marginal is closed form.** Because the SDE is linear and
`Σ_prior = Q_H ⊗ Σ_S` is diagonal in the joint eigenbasis `(U_T ⊗ U_S)`:

$$
r_\tau \mid r_0 \;\sim\; \mathcal{N}\!\left(\mu_\tau(r_0),\,\Sigma_\tau\right),
$$
$$
\mu_\tau \;=\; e^{-\tfrac{1}{2}\bar\beta(\tau) Q}\,r_0,\qquad
\Sigma_\tau \;=\; Q^{-1}\!\left(I - e^{-\bar\beta(\tau) Q}\right).
$$

In the joint eigenbasis these become **elementwise**:

- decay per mode `(i,j)`:  `exp(-½ · β̄(τ) / s[i,j])`
- variance per mode `(i,j)`: `s[i,j] · (1 − exp(-β̄(τ) / s[i,j]))`

Sampling `r_τ | r_0` is then: rotate `r_0` to the eigenbasis, multiply by
decay, add `N(0, var)` noise, rotate back. **`O(HN)` per sample.** No
matrix exponentials.

**Why a linear / VP SDE and not e.g. a sub-VP or EDM-style?**

- Closed-form forward marginal → DSM target is exact (no need for noise-
  conditional score parameterisations).
- Stationary distribution = `N(0, Σ_prior)` is exactly what we want at τ=1.
- Linear-in-`r` drift means we can integrate in the eigenbasis directly.

Code: `src/diffusion/sde.py` (NumPy), `src/diffusion/sde_torch.py` (PyTorch
mirror used by training/sampling).
""")

code(r"""
import numpy as np, yaml
from src.diffusion.prior_fit import build_Q_H, build_Sigma_S
from src.diffusion.sde import eigendecompose_prior, beta_bar, forward_sample

ds = "metrla"
p = yaml.safe_load(open(f"results/prior_params/gwn_{ds}.yaml"))
eig_L = np.load(f"data/processed/{ds}_laplacian_eigs.npz")
Q_H = build_Q_H(np.array(p["delta_h"]), p["alpha"], p["rho1"], p["rho2"])
Sigma_S = build_Sigma_S(p["beta"], p["epsilon_s"], eig_L["eigenvalues"], eig_L["eigenvectors"])
eig = eigendecompose_prior(Q_H, Sigma_S)

beta_min, beta_max = 0.01, 50.0
for tau in [0.01, 0.1, 0.5, 1.0]:
    bb = beta_bar(tau, beta_min, beta_max)
    # fraction of joint modes whose decay is < 1% (i.e. reached stationary)
    decay = np.exp(-0.5 * bb / eig["s"])
    frac_stat = (decay < 0.01).mean()
    print(f"τ={tau:.2f}  β̄={bb:7.3f}   fraction-of-modes-near-stationary = {frac_stat*100:5.1f}%")
""")

# ─────────────────────────────────────────────────────────────────────────────
# §8. Score net
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 8. The score network

The model output is `r̂₀` — a prediction of the **clean** residual given the
noisy `r_τ` and conditioning. This is the **Tweedie / x₀-prediction**
parameterisation (preferred over predicting the score directly because the
target stays on a fixed unit scale across all τ — much more stable to train).

Architecture (4 stacked blocks; ~150k–250k parameters):

```
inputs:
  r_τ              (B, H, N)        — noisy standardised residual
  τ                (B,)             — diffusion time
  x_history        (B, 2, H, N)     — normalised speed + time-of-day
  y_pred_backbone  (B, H, N)        — standardised backbone forecast

  concat along channel dim    → (B, 4, H, N)
  1×1 Conv2d → hidden (64 ch) → (B, 64, H, N)

  ┌─────────────────────────────────────────────────────┐
  │ x4:                                                 │
  │   gated TCN  (3×1 conv, tanh ⊙ sigmoid)             │
  │   GraphConv  ([self ; Â · self] · 1×1 conv)         │
  │   FiLM(τ)    (per-channel γ(τ)·x + β(τ))            │
  │   GroupNorm                                         │
  │   + residual                                        │
  └─────────────────────────────────────────────────────┘

  1×1 Conv2d → 1 channel       → (B, H, N)  =  r̂₀
```

**Why this architecture?**

- **GWN-style block**: matches the backbone's inductive biases for traffic
  data (TCN for temporal, graph conv for spatial). Reusing the block type
  makes ablations more interpretable.
- **FiLM conditioning on τ**: per-block `γ(τ)·x + β(τ)` lets each block
  adapt its behaviour at different noise levels. Initialised to identity
  so the network starts as a τ-independent denoiser and *learns* the τ
  dependence — more stable than additive τ-embeddings.
- **Sinusoidal τ embedding**: standard DDPM trick, `scale=1000` spreads
  the frequencies over `τ ∈ [0,1]`.
- **Static normalised adjacency, K=1 hops**: stacking 4 blocks gives
  4-hop spatial receptive field. Adaptive adjacency (the GWN backbone's
  feature) is deferred — it's an open lever.
- **Weights NOT shared with the backbone**: the backbone forecast is
  a conditioning *input*, not a starting point. Fresh weights so the
  optimiser can specialise on the denoising objective.

Code: `src/diffusion/score_net.py::ScoreNet`.
""")

code(r"""
import torch
from src.diffusion.score_net import ScoreNet

net = ScoreNet(num_horizons=12, num_sensors=207, hidden_dim=64, num_blocks=4, tau_embed_dim=128)
print(f"trainable params: {net.num_parameters:,}")

# Smoke test: random input shapes
B = 2
r_tau    = torch.randn(B, 12, 207)
tau      = torch.rand(B)
x_hist   = torch.randn(B, 2, 12, 207)
y_pred   = torch.randn(B, 12, 207)
out = net(r_tau, tau, x_hist, y_pred)
print("output:", tuple(out.shape))
""")

# ─────────────────────────────────────────────────────────────────────────────
# §9. DSM loss
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 9. The DSM loss — eigenbasis-weighted x₀ prediction

For each batch element with clean `r₀` and a freshly sampled τ:

1. Draw `τ ∼ LogUniform[τ_eps, 1]` with `τ_eps = 10⁻³`.
   - Why log-uniform? Equal density per *decade* of τ; the small-τ regime
     (`τ < 0.1`) is where the network does most of the denoising work, so
     log-uniform allocates ~1/3 of training compute there.
2. Sample `r_τ ∼ N(μ_τ(r₀), Σ_τ)` using §7's closed-form (spectral domain;
   no sampling from an explicit covariance).
3. Predict `r̂₀ = f_θ(r_τ, τ, x_history, y_pred)`.
4. Compute the loss in the joint eigenbasis:

$$
\mathcal{L}_{\text{DSM}} \;=\; \mathbb{E}_{\tau, r_0, \varepsilon}\Big[\;\sum_{i,j}\;\frac{[\widetilde{r̂_0 - r_0}]_{i,j}^2}{\sigma^2_{(i,j)}(\tau)}\;\Big]
$$

with `σ²_{(i,j)}(τ) = s[i,j]·(1 − exp(−β̄(τ)/s[i,j]))` (per-mode forward
variance) and `~` meaning "rotated into the joint eigenbasis".

**Why the eigenbasis 1/var weighting?**

- For a Gaussian conditional `p(r_τ | r₀)` with covariance `Σ_τ`, the
  natural negative-log-likelihood weight is `Σ_τ⁻¹`. In the joint eigenbasis
  `Σ_τ` is diagonal so `Σ_τ⁻¹` becomes elementwise `1/var`.
- This makes every mode contribute *comparable* gradient magnitude at every
  τ — modes with small variance (low-noise, well-resolved at this τ) carry
  more weight; modes with large variance carry less. Without it, the
  largest-variance modes drown out everything.
- The orthogonal rotation `to_spectral` preserves the sum of squares
  (Plancherel), so the masked-position trick still works: zero the raw-space
  error at missing readings, rotate, weight, sum — masked energy stays zero.

The DSM-loss code also supports `weighting='uniform'` (plain MSE) as a
sanity ablation. For all production runs we use `'eigenbasis'`.

Code: `src/diffusion/dsm.py::dsm_x0_loss`, `src/diffusion/dsm.py::sample_tau`.
""")

code(r"""
import numpy as np, torch
from src.diffusion.sde import eigendecompose_prior
from src.diffusion.sde_torch import eig_to_torch, beta_bar
from src.diffusion.dsm import sample_tau, dsm_x0_loss
from src.diffusion.prior_fit import build_Q_H, build_Sigma_S
import yaml

ds = "metrla"
p = yaml.safe_load(open(f"results/prior_params/gwn_{ds}.yaml"))
eig_L = np.load(f"data/processed/{ds}_laplacian_eigs.npz")
Q_H = build_Q_H(np.array(p["delta_h"]), p["alpha"], p["rho1"], p["rho2"])
Sigma_S = build_Sigma_S(p["beta"], p["epsilon_s"], eig_L["eigenvalues"], eig_L["eigenvectors"])
eig_np = eigendecompose_prior(Q_H, Sigma_S)
eig = eig_to_torch(eig_np)

# Oracle test: if r̂_0 = r_0, the loss is zero.
B, H, N = 4, 12, 207
torch.manual_seed(0)
r_0 = torch.randn(B, H, N)
tau = sample_tau(B)
loss_zero = dsm_x0_loss(r_hat_0=r_0, r_0=r_0, tau=tau, eig=eig, beta_min=0.01, beta_max=50.0)
print(f"oracle (perfect prediction) loss = {loss_zero.item():.3e}  (should be 0)")

# Random prediction: loss > 0.
r_hat = torch.randn_like(r_0)
print(f"random prediction loss          = {dsm_x0_loss(r_hat, r_0, tau, eig, 0.01, 50.0).item():.3e}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# §10. Reverse sampling
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 10. Reverse sampling — Euler-Maruyama

At inference time we run the **reverse-time** SDE (Anderson 1982):

$$
\mathrm d\bar r_\tau \;=\; \big[\,-\tfrac12 g^2 Q\,\bar r_\tau \;-\; g^2 \,\nabla_{r_\tau}\log p_\tau(\bar r_\tau)\,\big]\mathrm d\tau \;+\; g(\tau)\,\mathrm d\bar W_\tau.
$$

The score is recovered from the network's `r̂₀` via **Tweedie**:

$$
\nabla_{r_\tau}\log p_\tau(r_\tau) \;\approx\; -\Sigma_\tau^{-1}\big(r_\tau - \mu_\tau(\hat r_0(r_\tau, \tau, c))\big).
$$

**Algorithm** (`src/diffusion/sampling.py::reverse_learned`):

```
1.  r_τ  ←  draw from N(0, Σ_prior)            # sample_stationary
2.  for k in 0..n_steps-1:
3.      τ_k, Δτ from linspace(τ_start, τ_eps, n_steps+1)
4.      r̂_0   ←  network(r_τ, τ_k, x_hist, y_pred)
5.      score ←  -(r_τ - μ_{τ_k}(r̂_0)) / var_{τ_k}    (in eigenbasis)
6.      drift ←  -½ g²(τ_k)·Q·r_τ  -  g²(τ_k)·score
7.      r_τ   ←  r_τ  -  drift·Δτ  +  g(τ_k)·√Δτ·ε     ε ∼ N(0, I)
8.  return r̂_0 at τ = τ_eps
```

`Q·r_τ` is computed by rotating to the eigenbasis, multiplying elementwise
by `q = 1/s`, rotating back. Same for the score. No dense matrix-vector
products on `(HN) × (HN)` matrices, ever.

**The stability condition — why `n_steps` matters**

For each joint eigenmode `(i, j)` with eigenvalue `s = s[i,j]` of `Σ_prior`,
the linear part of the reverse drift in that mode contributes an
amplification factor of approximately

$$
\big|\,1 - \tfrac12\,g^2(\tau)\,q\,\Delta\tau\,\big|,\qquad q = 1/s.
$$

For numerical stability this must stay `< 1`, i.e.

$$
\boxed{\;\Delta\tau \;<\; \frac{2}{g^2_{\max}\,q_{\max}} \;=\; \frac{2}{\beta_{\max}\,/\,s_{\min}}\;}
$$

with `g²_max = β_max = 50`. The smallest joint eigenvalue `s_min = 1/q_max`
is the bottleneck — it's set by the **most poorly conditioned** mode of the
prior.

Per dataset (see [project-dsm-em-stability]):

| Dataset    | `q_max` | min `n_steps` for stability | n_steps = 100 verdict |
|------------|--------:|----------------------------:|------------------------|
| PEMS08F    | 2.2     | > 55                        | safe |
| METR-LA    | 5.6     | > 140                       | **marginal** — mild over-coverage |
| PEMS-BAY   | 14.1    | > 360                       | **catastrophic** — std blows up |

The current eval used **n_steps = 100** for every dataset. This is the proximate
cause of the PEMS-BAY structured failure (§13). The longer-term fix is an
exponential integrator on the linear drift (unconditionally stable) — see §13.

Code: `src/diffusion/sampling.py::reverse_learned`,
      `src/diffusion/sampling.py::sample_per_window`.
""")

code(r"""
import numpy as np, yaml
from src.diffusion.prior_fit import build_Q_H, build_Sigma_S
from src.diffusion.sde import eigendecompose_prior

for ds in ["metrla", "pemsbay", "pems08f"]:
    p = yaml.safe_load(open(f"results/prior_params/gwn_{ds}.yaml"))
    eig_L = np.load(f"data/processed/{ds}_laplacian_eigs.npz")
    Q_H = build_Q_H(np.array(p["delta_h"]), p["alpha"], p["rho1"], p["rho2"])
    Sigma_S = build_Sigma_S(p["beta"], p["epsilon_s"], eig_L["eigenvalues"], eig_L["eigenvectors"])
    eig = eigendecompose_prior(Q_H, Sigma_S)
    q_max = eig["q"].max()
    n_min = int(np.ceil(50.0 * q_max / 2.0))   # β_max=50; need Δτ < 2/(β_max·q_max) → n > β_max·q_max/2
    print(f"{ds:10s}  q_max = {q_max:6.2f}   minimum n_steps for stability ≈ {n_min}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# §11. Eval metrics
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 11. Evaluation — CRPS, PIT, pinball

After sampling `K = 50` reverse trajectories per test window we have an
**ensemble** of forecast samples per `(window, horizon, sensor)`. We score it
with three classic probabilistic metrics.

**CRPS (Continuous Ranked Probability Score)** — sharpness + calibration combined:

$$
\mathrm{CRPS} = \frac{1}{K}\sum_i |x_i - y| \;-\; \frac{1}{2K^2}\sum_{i,j}|x_i - x_j|.
$$

The first term penalises **distance from truth** (calibration), the second
rewards **concentrated samples** (sharpness). Lower is better. CRPS reduces
to MAE when K=1 (degenerate point forecast).

Two normalisations we report:

- **CRPS_norm = CRPS / |y|** — per-horizon CRPS divided by absolute mean speed
  at that horizon. Comparable across datasets.
- **CRPS_norm_pooled** — pooled mean of all per-element CRPS_norm. The same
  thing aggregated differently.

**PIT (Probability Integral Transform)**:

$$
\mathrm{PIT}(y) = \frac{1}{K}\,\big|\{i:\;x_i \le y\}\big|.
$$

For a perfectly calibrated forecast PIT is `Uniform(0, 1)`. Deviations
diagnose miscalibration:
- **U-shaped** histogram → ensemble too narrow (under-dispersed, samples
  cluster, truth often falls in the tails)
- **dome-shaped** → ensemble too wide (over-dispersed)
- **skew** → systematic bias

**Pinball loss** at `q ∈ {0.1, 0.5, 0.9}`:

$$
\rho_q(y, \hat y_q) = \max(q(y - \hat y_q),\,(1-q)(\hat y_q - y)).
$$

with `ŷ_q` the empirical `q`-quantile of the ensemble. Pinball is minimised
by the true `q`-quantile, so it is a quantile-specific sharpness/accuracy
check. We report q=0.1 (lower tail) and q=0.9 (upper tail) to detect
asymmetric miscalibration.

Code: `src/eval/probabilistic.py`.
""")

# ─────────────────────────────────────────────────────────────────────────────
# §12. What current results say
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 12. Current results (Stage-2, seed 0)

These come from `experiments/score-network/eval_dsm_*_seed0_*.json`
(K=50 samples, n_steps=100, β_max=50, τ_eps=1e-3) and are summarised in
`experiments/score-network/comparison_seed0/table*.csv`.

**Normalised CRPS, structured vs isotropic** (lower is better):

| Dataset  | Struct h3 | Iso h3 | Struct h6 | Iso h6 | Struct h12 | Iso h12 | Struct avg | Iso avg |
|----------|----------:|-------:|----------:|-------:|-----------:|--------:|-----------:|--------:|
| METR-LA  | 0.044     | 0.039  | 0.051     | 0.044  | 0.059      | 0.050   | **0.050**  | **0.044** |
| PEMS-BAY | 0.448     | 0.017  | 0.624     | 0.021  | 0.526      | 0.025   | **0.534**  | **0.020** |
| PEMS08F  | 0.045     | 0.046  | 0.049     | 0.049  | 0.054      | 0.054   | **0.049**  | **0.049** |

**Coverage at the 50% and 80% nominal levels** (closer to {0.50, 0.80} = better):

| Dataset  | Prior      | cov50 avg | cov80 avg | sample std avg |
|----------|------------|----------:|----------:|---------------:|
| METR-LA  | structured | 0.77      | 0.90      | 5.77           |
| METR-LA  | isotropic  | 0.56      | 0.79      | 4.17           |
| PEMS-BAY | structured | 0.88      | 0.99      | **212.3** (!!) |
| PEMS-BAY | isotropic  | 0.53      | 0.78      | 1.99           |
| PEMS08F  | structured | 0.49      | 0.75      | 18.7           |
| PEMS08F  | isotropic  | 0.51      | 0.77      | 21.6           |

**Point-forecast MAE lift over the bare backbone** (negative = the diffusion
model is *worse* than just using the deterministic backbone, since MAE on the
ensemble mean):

| Dataset  | Prior      | MAE_backbone avg | MAE_dsm avg | lift avg |
|----------|------------|-----------------:|------------:|---------:|
| METR-LA  | structured | 3.14             | 3.32        | -0.19    |
| METR-LA  | isotropic  | 3.14             | 3.27        | -0.13    |
| PEMS-BAY | structured | 1.58             | **25.20**   | **-23.6** |
| PEMS-BAY | isotropic  | 1.58             | 1.64        | -0.06    |
| PEMS08F  | structured | 14.75            | 14.94       | -0.19    |
| PEMS08F  | isotropic  | 14.75            | 15.01       | -0.26    |

Read the next section for what these numbers actually mean and what is
broken.
""")

# ─────────────────────────────────────────────────────────────────────────────
# §13. What is going wrong, and why
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 13. What is going wrong, and why

Three things, in order of severity.

### 13a. PEMS-BAY structured: **catastrophic sampler failure (not a model failure)**

The numbers (MAE = 25 mph, sample std = 212 mph, cov80 = 0.99) are not the
score network mis-learning the prior. They are the **reverse-SDE integrator
blowing up**.

Recall (§10): the Euler-Maruyama step is stable per mode only if
`Δτ < 2/(β_max · q_max)`. For PEMS-BAY structured, `q_max ≈ 14.1`. With
`β_max = 50` and `n_steps = 100`:

$$
\Delta\tau = 1/100 = 0.01,\qquad
\text{required: } \Delta\tau < \frac{2}{50 \cdot 14.1} \approx 0.0028.
$$

We are **3.5×** over the stability boundary in the worst mode. Mid-trajectory
the per-mode std explodes by a factor of 100+ before contracting back, but it
never recovers (the noise term cannot un-do an explosion that already pushed
the state outside the linear regime). The diagnostic script
`scripts/diag_pemsbay_structured.py` reproduces this directly: setting
`DIAG_N_STEPS=100` shows per-step std reaching 10⁷ then coming back to ~200;
`DIAG_N_STEPS=400` is well-behaved.

**Why is `q_max` so big on PEMS-BAY structured?**

It is the product of two things:
1. The temporal prior degenerated to a single AR(1) with `ρ ≈ 0.819` (§6d).
   `Q_H` for an AR(1) is **ill-conditioned**: the smallest eigenvalue is the
   coefficient of the highest-frequency Chebyshev mode, which is small
   relative to the largest. Even after standardisation, `cond(Q_H) ≈ 10`.
2. The spatial `Σ_S` puts strong mass on the smoothest Laplacian modes
   (β = 0.24, 325 sensors), so `c_S_min` (smallest spatial eigenvalue) is
   ~0.07.

The joint smallest is `s_min = c_T_min · c_S_min ≈ 0.10 · 0.71 ≈ 0.071`,
giving `q_max = 14.1`. The problem is genuine to PEMS-BAY's prior class —
this is **not** a bug in the fit.

**Two fixes, both clean.**

1. **Quick fix**: re-run eval with `n_steps ≥ 400`. The model isn't broken,
   the integrator is. This is one line in `submit_eval.sbatch`. Throws away
   ~4× wall-clock but is honest.
2. **Long fix**: in `src/diffusion/sampling.py::_reverse_step`, replace the
   explicit Euler step on the linear drift with an exponential integrator
   per mode: in the joint eigenbasis, the linear part is `r ← r · exp(½ g² q Δτ)`
   (closed form, **unconditionally stable** for any `Δτ > 0`). About 10
   lines. Then `n_steps = 100` is fine for every prior.

### 13b. METR-LA structured: **mildly over-dispersed, also a sampler signal**

cov80 = 0.90 vs nominal 0.80 (10 percentage points over-covering); sample std
5.77 vs the empirical 4.17 from the isotropic baseline. This is the same
stability bug at a much milder magnitude: `q_max = 5.6`, so `n_steps = 100`
gives `Δτ = 0.01` vs the boundary `Δτ < 0.0071` — only 1.4× over.
Manifests as **inflated** sample std, not explosion.

This also explains the CRPS_norm gap (structured 0.050 vs isotropic 0.044):
the structured samples are too wide, so the `|x_i − y|` term in CRPS punishes
them. The model is **not** the problem; the integrator step size is.

### 13c. The METR-LA Kronecker prior is mis-specified per-mode

Independent of the sampler bug, the structured prior on METR-LA has a known
limitation noted in `experiments/sde-sanity/SUMMARY.md` §3: direct empirical-
vs-prior **per-mode** variance ratios span 0.32 to 2,506 (median 1.69). The
Kronecker factorisation forces every spatial mode to share the same temporal
decay, which is a strong assumption the data does not satisfy.

The score network is *meant* to absorb that mismatch during training — and on
PEMS08F it does (CRPS_norm 0.049 structured ≈ 0.049 isotropic). On METR-LA
it apparently can't make up for both the prior mis-specification AND the
extra dispersion from the sampler bug.

**Followup question for the thesis**: once the sampler is fixed (13a/b), is
the METR-LA structured prior actually better than isotropic, or worse? The
current numbers cannot answer that — they're confounded by the integrator.

### 13d. Smaller signals worth noting

- **PEMS-BAY temporal prior degeneracy** (`α ≈ 0.005`, `ρ₁ = ρ₂`): the
  two-exp kernel collapses to a single AR(1). This is the **correct**
  behaviour of a flexible model on data that doesn't need the extra
  capacity (§6d) — but it does *amplify* the conditioning problem in 13a.
  A worthwhile follow-up is to clamp the optimiser away from α-collapse and
  refit with a single-exponential prior on PEMS-BAY; this should reduce
  `cond(Q_H)` and hence `q_max`.
- **PEMS08F structured ≈ isotropic** across all metrics. PEMS08F has the
  smallest condition number (q_max=2.2) and a fairly isotropic Σ_S, so the
  *additional structure* the Kronecker prior provides is small. If we ever
  redesign the prior class (non-separable, low-rank-plus-diagonal, etc.),
  PEMS08F is the cleanest dataset on which to test that we have not lost
  ground there.
- **All three datasets show negative MAE lift** vs the deterministic
  backbone — i.e., the *mean* of the diffusion ensemble is a worse point
  forecast than the deterministic backbone. This is expected: the ensemble
  mean is a denoised draw from a probabilistic model that doesn't see the
  backbone's smoothing prior, while the backbone is L1-optimised on the
  point forecast directly. Reporting probabilistic metrics, not MAE, is
  the right framing.
""")

# ─────────────────────────────────────────────────────────────────────────────
# §14. Where things live
# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 14. Where each artefact lives

| What | Path |
|---|---|
| Raw HDF5 / NPZ data | `data/raw/`, `data/processed/` |
| Backbone checkpoints | `results/checkpoints/gwn_{ds}_seed0_*/best.pt` |
| Residuals | `results/residuals/gwn_{ds}_{split}_seed0.npz` |
| Fitted prior YAMLs | `results/prior_params/gwn_{ds}.yaml` |
| Diffusion checkpoints | `results/checkpoints/dsm_{prior}_{ds}_seed0_*/best.pt` |
| Eval JSONs | `experiments/score-network/eval_dsm_*.json` |
| Stage-2 comparison tables/figures | `experiments/score-network/comparison_seed0/` |
| **SDE math** (NumPy) | `src/diffusion/sde.py` |
| **SDE math** (PyTorch) | `src/diffusion/sde_torch.py` |
| **Prior fit** | `src/diffusion/prior_fit.py` |
| **DSM loss** | `src/diffusion/dsm.py` |
| **Score network** | `src/diffusion/score_net.py` |
| **Sampling** | `src/diffusion/sampling.py` |
| **Probabilistic metrics** | `src/eval/probabilistic.py` |
| Training script | `scripts/train_diffusion.py` |
| Eval script | `scripts/eval_diffusion.py` |
| Stability diagnostic (PEMS-BAY) | `scripts/diag_pemsbay_structured.py` |
| DAIC cluster sbatch + runbook | `scripts/daic/`, `scripts/daic/README.md` |

That is the whole methodology and the current state of results in one read.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Serialise
# ─────────────────────────────────────────────────────────────────────────────
nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(s) if t == "md" else nbf.v4.new_code_cell(s)
    for (t, s) in cells
]
# Strip default execution metadata so the file is small and clean.
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
OUT.write_text(nbf.writes(nb))
print(f"wrote {OUT}  ({len(cells)} cells)")
