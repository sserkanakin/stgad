# Empirical vs. Reshaped prior — explained with a worked example

This note explains, from scratch and with concrete numbers, **exactly what our diffusion
prior is**, and **exactly how the two variants (`empirical` and `reshaped`) differ**. The
reshaped prior is the thesis novelty; by the end this should be obvious rather than abstract.

---

## 1. What are we even modelling?

Our forecasting backbone (MLP / GCN / GWN) predicts traffic, but it is never perfect. The
leftover error is the **residual**:

```
residual  R  =  (true value)  −  (backbone forecast)
```

`R` has one number for every **sensor** and every **forecast horizon**. So for one time
window it is a grid of shape `(N sensors) × (H horizons)`.

We want **probabilistic** forecasts — not just "speed = 42" but "speed = 42 ± 6, and the
error at sensor A is correlated with sensor B." To get that, we learn the *distribution*
of these residuals with a diffusion model. The **prior** is the shape of the noise the
diffusion model starts from. A prior shaped like real residuals means the denoiser has
almost no work to do; a badly shaped prior means it has to fix everything itself.

**The whole game is: give the diffusion a prior whose "noise" already looks like real
residual noise.**

---

## 2. A Gaussian prior is just "how much wiggle in each direction"

A Gaussian is fully described by its covariance. Instead of storing a giant `NH × NH`
covariance matrix, we pick a smart set of **directions (modes)** and store just **one
variance number per mode**. Two families of directions:

- **spatial (graph) modes** — patterns *across sensors*, from the road-network graph.
- **temporal modes** — patterns *across horizons*.

A joint mode `(k, h)` is "spatial pattern `k` combined with temporal pattern `h`", and the
prior is the list of variances **σ²[k,h] = how much the residual wiggles in that mode**.

### The spatial modes, concretely

Take a tiny network of 3 sensors on a road: **A — B — C**. The graph Laplacian gives 3
spatial modes (this is a real eigen-decomposition, eigenvalues λ = 0, 1, 2):

| mode | shape across (A, B, C) | picture | λ (roughness) |
|---|---|---|---|
| **smooth** | (+, +, +) — all move together | `▁▁▁` all up | 0 (smoothest) |
| **gradient** | (+, 0, −) — one end up, other down | `╱` tilt | 1 |
| **zigzag** | (+, −, +) — neighbours disagree | `＾v＾` | 2 (roughest) |

"Smooth" means neighbouring sensors behave alike (low `λ`); "zigzag" means neighbours
disagree (high `λ`). **Every residual pattern is a mix of these three.** The prior says how
strong each ingredient typically is.

---

## 3. The one question that separates `empirical` from `reshaped`

> **How much variance does each mode get?**

Both variants use the **same** spatial graph modes and the **same** temporal modes. They
disagree only on **where the variance numbers come from.**

### `empirical` — uses a *formula* (GAD's heat equation)

GAD says: noise on a graph should be **smooth**, following the heat equation. That fixes the
variance of each spatial mode by a formula — no looking at the data:

```
variance of mode k   =   1 / (λ_k + γ)          with γ = 0.8   (GAD's value)
```

Plug in our three λ's:

| mode | λ | formula 1/(λ+0.8) | share of total |
|---|---|---|---|
| smooth   | 0 | 1 / 0.8 = **1.25** | 58% |
| gradient | 1 | 1 / 1.8 = **0.56** | 26% |
| zigzag   | 2 | 1 / 2.8 = **0.36** | 16% |

So `empirical` **assumes** most noise is smooth, some is gradient, a bit is zigzag — purely
from the graph, before seeing a single residual. This is exactly GAD's smoothness prior
`Σ_S = (L + γI)⁻¹`. (The temporal part *is* fit from data; the spatial shape is the formula.)

### `reshaped` — **measures** the variance from the data

`reshaped` keeps the same three modes, but **throws the formula away**. It takes the actual
training residuals, projects them onto each mode, and computes how much they *really* wiggle:

```
project residuals onto each mode, then
variance of mode k   =   average of (residual's coordinate in mode k)²
```

Suppose the measured numbers come out like this:

| mode | measured variance | share of total |
|---|---|---|
| smooth   | **5.0** | 68% |
| gradient | **2.0** | 27% |
| zigzag   | **0.3** | 4% |

### Put them side by side (normalised to shares, since overall scale is matched anyway)

| mode | `empirical` (formula) | `reshaped` (measured) |
|---|---|---|
| smooth   | 58% | **68%** |
| gradient | 26% | 27% |
| zigzag   | 16% | **4%** |

**Read this table — it is the entire novelty.** Both agree on the *ordering*
(smooth > gradient > zigzag), which is why the heat-equation formula already works
reasonably. But the formula **over-estimates the zigzag noise** (16% vs the true 4%): real
residuals are *even smoother* than the heat equation predicts. `empirical` therefore tells
the denoiser "expect a fair bit of zigzag noise," and the denoiser wastes effort chasing
noise that isn't there. `reshaped` tells it the truth — almost no zigzag — so it spends its
capacity where the real noise is.

> **`empirical` guesses the variance profile from a graph formula.
> `reshaped` measures the real one.**
> The formula is close but not exact, and closing that gap is the contribution.

---

## 4. The second difference: `reshaped` isn't forced to "factorise"

So far we looked at spatial modes only. With **temporal** modes too, there's a second, subtler
gap. Say 2 temporal modes: `t0` = overall level, `t1` = trend over the horizon.

**`empirical` forces the grid to be a product** (spatial profile × temporal profile). If the
temporal split is `t0:t1 = 70:30`, then *every* spatial mode must use that same 70/30 split:

```
empirical variance[k,h] = spatial[k] × temporal[h]     (a "rank-1" grid)

              t0 (0.70)   t1 (0.30)
  smooth        0.48        0.21          ← every row is the same 70/30 shape
  gradient      0.19        0.08
  zigzag        0.03        0.01
```

**`reshaped` measures each cell independently**, so it can capture "smooth errors happen
early, gradient errors grow late" — which a product cannot represent:

```
reshaped variance[k,h] = measured directly, cell by cell

              t0          t1
  smooth        0.50        0.15          ← smooth noise concentrated early
  gradient      0.15        0.12          ← gradient noise grows toward late horizons
  zigzag        0.02        0.01
```

Analogy: `empirical` assumes a spreadsheet is "row-pattern × column-pattern" (one row shape,
one column shape). `reshaped` reads every cell. If reality isn't a clean product, only
`reshaped` can express it.

| | spatial modes | variance per mode | grid shape |
|---|---|---|---|
| **empirical** | graph Laplacian | **formula** `1/(λ+γ)` × empirical time | forced **product** (`Σ_T ⊗ Σ_S`) |
| **reshaped** | graph Laplacian | **measured** from data | **free** grid (no product constraint) |

---

## 5. Where the prior fits in the whole diffusion process

The prior is fit **once, offline**, from training residuals. Then:

1. **Fit prior** — from the training residuals, get the variance of every mode
   (by formula → `empirical`, or by measurement → `reshaped`).
2. **Forward (add noise)** — take a clean residual and gradually corrupt it until it becomes
   a pure sample from the prior. Each mode is corrupted **at its own rate**: low-variance
   (zigzag) modes get wiped out fast, high-variance (smooth) modes persist. This
   per-mode-rate corruption is exactly GAD's anisotropic graph diffusion.
3. **Train denoiser** — a small (17k-param) graph network learns to undo one noising step,
   conditioned on the point forecast `Ŷ`.
4. **Sample (reverse)** — start from prior noise, run the denoiser backwards step by step to
   get a plausible residual `R̂`, then

   ```
   probabilistic forecast  =  Ŷ (point forecast)  +  R̂ (sampled residual)
   ```

   Draw many `R̂`'s → many forecasts → an uncertainty band and correlated errors.

The prior's job in step 2 is to make the "pure noise" endpoint already look like real
residual noise. The closer it is, the less the denoiser has to fix — which is why a
better-shaped prior helps **most when the backbone/denoiser is weak**, and barely matters
when the backbone is already strong (it has left almost nothing for the prior to shape).

---

## 6. Are these correct with respect to GAD (smoothness / heat equation)?

- **`empirical` is literally GAD.** `Σ_S = (L + γI)⁻¹` with the normalised symmetric graph
  Laplacian, no self-loops, `γ = 0.8`. The quadratic form `xᵀ(L+γI)x` penalises the graph
  Dirichlet energy `Σ_edges (x_i − x_j)²`, i.e. it **penalises non-smoothness** — the exact
  Gaussian smoothness prior of the graph heat equation. Our forward/reverse were checked to
  reproduce GAD's figures, so the process is faithful.

- **`reshaped` is GAD's *Remark 1* ("adapt the noise per eigenvalue") taken to its
  data-driven limit.** It keeps GAD's **geometry** — the graph eigenmodes, the coordinate
  system that makes the diffusion anisotropic — but replaces the smoothness **formula**
  `1/(λ+γ)` with the **measured** per-mode variance. So, precisely:

  > `reshaped` is **not** a smoothness prior. It is a **spectrum-matched** prior *expressed in
  > the graph basis.* Its justification is "match the residual's real spectrum," not
  > "assume smoothness." And the only reason it beats `empirical` is that **real residuals
  > are not perfectly graph-smooth**, so the heat-equation formula is slightly wrong.

This is the honest one-line positioning for the thesis:

- **iso → empirical** tests *GAD's* claim: does the graph heat-equation geometry help at all?
- **empirical → reshaped** tests *our* claim: does measuring the true spectrum, instead of
  trusting the heat-equation formula, help on top of that?

`reshaped` (spectrum-matched) is the novelty; `empirical` (GAD-faithful) is the comparison
rung that makes the novelty visible.

---

# Part 2 — the exact formulas, the γ question, and how to write it up

## 7. The exact formulas (as implemented)

Notation: residual window `R ∈ ℝ^{N×H}`. Prior is Gaussian, diagonal in `V_S ⊗ V_T`, with
per-mode **precision** `θ[k,h]` and **stationary variance** `Σ[k,h] = σ² / θ[k,h]`.

- `V_S, λ_S` — eigenvectors/eigenvalues of the normalised symmetric graph Laplacian `L`
  (no self-loops), `λ_S ∈ [0, 2]`.
- `V_T` — eigenvectors of the empirical temporal covariance `Σ_T`.

**Isotropic (baseline).**
```
θ[k,h] = 1 / σ²        (constant),   V_S = I,  V_T = I,   σ² = Var(residuals)
Σ[k,h] = σ²            (flat — same variance in every direction)
```

**Empirical (GAD-faithful).** Spatial precision from the heat-kernel formula; temporal from data:
```
spatial:   q_S[k] = λ_S[k] + γ                 # Σ_S = (L + γ I)^{-1},  γ = 0.8
temporal:  Σ_T   = D^{1/2} R_H D^{1/2},         # D = diag(δ_h),  δ_h = mean(r_h²),  R_H = corr
           μ_T, V_T = eig(Σ_T),  μ_T ← (1-ρ)μ_T + ρ·mean(μ_T)   # mild temporal ridge, ρ = 0.1
joint:     θ[k,h] = q_S[k] / (τ² · μ_T[h])      # SEPARABLE  ⇒  Σ = Σ_T ⊗ Σ_S
```

**Reshaped (the novelty).** Same bases; per-mode variance **measured**, not formula:
```
project:   C[b,k,h] = (V_Sᵀ R_b V_T)[k,h]       # residual coordinates in the joint basis
measure:   v_emp[k,h] = mean_b ( C[b,k,h]² )     # real variance of each mode
joint:     θ[k,h] = 1 / v_emp[k,h]               # NON-separable full diagonal
           Σ[k,h] = σ² · v_emp[k,h]              # matches the measured spectrum exactly
```

**The one-line contrast:**
```
empirical:  θ[k,h] = (λ_S[k] + γ) / (τ² μ_T[h])     ← spatial shape from a formula (tunable by γ)
reshaped:   θ[k,h] = 1 / v_emp[k,h]                  ← every mode's variance read off the data
```

## 8. How different are they, *really*?

Two separate questions — the priors, and the forecasts they produce.

**The priors are very different objects** (real numbers, pems04f/gwn residuals):

| | empirical (γ=0.8) | reshaped (measured) |
|---|--:|--:|
| spatial spread (smoothest ÷ roughest mode) | **3.5×** | **24.6×** |
| joint dynamic range (max ÷ min mode variance) | 21× | **258×** |
| how much they differ | rel. L1 = **28.7%**, correlation = **0.64** | |

**But the forecasts they produce differ only when the backbone is weak** (seed-averaged CRPS):

| backbone | empirical | reshaped | gap |
|---|--:|--:|--:|
| mlp (weak) | 0.239 | **0.191** | −20% (real) |
| gcn | 0.122 | 0.120 | ~1% |
| gwn (strong) | 0.0770 | 0.0776 | tie |

**This *is* the thesis in one sentence:** the two priors are genuinely different, but a strong
backbone leaves so little residual structure that even a 7×-different prior barely moves the
forecast. Structure matters when the model is otherwise weak.

## 9. Is γ = 0.8 too high? (yes — and it's grounded)

γ is the single knob that sets empirical's spatial spread, via `variance ∝ 1/(λ+γ)`:

| γ | spatial spread `(λ_max+γ)/(λ_min+γ)` | comment |
|---|--:|---|
| 0.04 | 51× | GAD's "small-γ" regime; EM-unstable (stiffness 51) |
| 0.2 | 11× | |
| **0.8** | **3.5×** | our default (GAD's EM-stable value) — **nearly isotropic** |
| ≈0.085 | ≈24.6× | the γ that would **match the measured spread** |

So at γ=0.8, empirical commits to only a 3.5× spread while the data shows 24.6× — it is
**~7× too flat**, sitting much closer to iso (1×) than to reality. This directly explains why
`iso → empirical` is a small step, and why `empirical → reshaped` recovers the rest.

**Why 0.8 was chosen, and why it doesn't have to be.** GAD picks a large γ for **EM-sampler
stability**: stiffness = (2+γ)/γ, so γ=0.8 → 3.5 (gentle), γ=0.04 → 51 (a violent, EM-unfriendly
collapse). GAD's own Fig. 1 (attached) shows this — each mode `e^{-c̄ λ_i}` decays at its own
rate, and a too-stiff spectrum makes the fast modes collapse before the reverse SDE can track
them. **But we do not use EM — we use the ancestral sampler, which is exact for the linear OU
forward and has no stiffness floor.** So the reason for γ=0.8 does not bind us; we are free to
lower it.

**And this is exactly what motivates `reshaped`.** Lowering γ steepens empirical toward the data;
in the limit you stop trusting the formula altogether and just use the measured spectrum — which
*is* `reshaped`. Reshaped is "empirical with the best possible per-mode γ, read from the data
instead of tuned." That reframes the novelty cleanly:

> A single scalar γ can only slide empirical along one flat-to-steep line. The residual spectrum
> is not a `1/(λ+γ)` curve — it has its own per-mode shape. `reshaped` fits that shape directly.

## 10. Grounding / citations for each choice

- **iso** — scale-matched white noise `Σ = σ²I`. The standard diffusion prior (DDPM/EDM). Its
  role is a *control*: identical machinery, structure removed, so any gap is attributable to the
  prior's shape alone. (Not a novelty — a baseline.)
- **empirical** — `Σ_S = (L+γI)^{-1}`, the graph heat-equation / Gaussian-Markov-random-field
  smoothness prior. Grounded in **GAD (Rozada et al. 2025, arXiv:2510.05036)** and the classical
  graph-signal smoothness literature. Faithful re-use, not new.
- **reshaped** — the **data-driven limit of GAD's Remark 1** ("adapt the noise per eigenvalue"):
  keep GAD's eigen-geometry, replace the `1/(λ+γ)` variance law with the measured per-mode
  spectrum. This is the contribution.

## 11. How to write the thesis without drowning in configurations

Collapse everything onto **one axis: where does the noise model's structure come from?**

```
   no structure   →   structure from a formula   →   structure from the data
      (iso)              (empirical, GAD)                  (reshaped, ours)
        1× spread            3.5× (γ=0.8) … 51× (γ↓)          24.6× (measured)
```

That is the **spine** — three rungs, one question. Everything else is a *one-knob ablation*
hanging off it, and each answers a single sub-question:

| axis | question it answers | verdict |
|---|---|---|
| **γ sweep** (empirical) | does the formula improve if tuned toward the data? | steepens toward reshaped — motivates it |
| **backbone** mlp→gcn→gwn | when does structure matter? | most when weak |
| **whitened loss** | does aligning the loss to the prior help? | no (negative result) |
| **past conditioning** | does history substitute for the prior? | yes — shrinks the gap |
| **denoiser capacity** 17k→1M | does a bigger denoiser substitute? | only when backbone weak |
| **sampler / few-step** | is the prior's edge largest at few steps (GAD)? | to test, on mlp |

Write the methodology around the **spine** (Sections: residual → forward/heat-equation →
prior: iso/empirical/reshaped → denoiser → sampler). Write the experiments around the **one
question** ("when does structure help") and present the ablations as evidence for it. You are
not presenting 20 configurations — you are presenting **one axis and its controls.**

## 12. The γ-sweep — RUN (2026-07-21/22), and it confirmed the prediction

γ ∈ {0.04, 0.1, 0.2, 0.4} × {mlp, gwn} on pems08f, seed 42 (`results/20260721_gammasweep`):

| bb | γ=0.04 | γ=0.1 | γ=0.2 | γ=0.4 | γ=0.8 (ladder) | reshaped | iso |
|---|--|--|--|--|--|--|--|
| mlp | 0.2060 | 0.2081 | **0.2055** | 0.2284 | 0.2385* | **0.1908*** | 0.2761* |
| gwn | 0.0573 | 0.0571 | 0.0572 | 0.0573 | 0.0574* | 0.0573* | 0.0580* |

(*5-seed means; sweep cells seed-42.) Three findings, exactly as predicted:
1. **γ=0.8 under-commits**: tuning to γ≈0.2 improves empirical ~14–21% on mlp; the curve
   plateaus for γ≤0.2 (the ancestral sampler handles the stiff prior EM could not).
2. **Reshaped still wins**: tuned empirical (0.206) closes ~87% of the iso→reshaped gap;
   reshaped (0.191) takes the rest — a single scalar cannot match a measured spectrum.
3. **On gwn γ is irrelevant** (flat within noise) — consistent with the whole thesis.

Consequence: the **empirical arm was retrained at γ=0.2** ("emp-refit",
`20260721_emprefit`, 4 datasets × mlp/gwn × 5 seeds) so the main tables compare reshaped
against a FAIR formula baseline. iso/reshaped/ddpm/whitened are γ-free and unchanged.
