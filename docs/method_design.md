# The method, decision by decision

*Working document. For every step of the pipeline: what we build, what else could go
there, what the choice asserts about the data, whether the math needs to change, and
how the constant it introduces will be decided. Nothing here is settled by citation
alone — every constant gets a criterion.*

Companion analysis: `scripts/analysis/prior_design.py` (CPU only, no training). It
scores design choices by **held-out log-likelihood per mode**: fit on train, evaluate
the Gaussian NLL of the test residual field in the prior's own eigenbasis, reported as
a delta against the scale-matched isotropic prior, in nats. Positive = better than
white noise. Same estimator as the attribution table in the results chapter.

---

## 0. The pipeline, and where the decisions are

```
 raw series
     │
 (1) backbone f_φ, frozen                     → point forecast Ŷ
     │
 (2) residual  R₀ = Y − Ŷ                     → the object we model
     │
 (3) normalise                                → per-sensor σ_i   ← DECISION A
     │
 (4) fit the prior Σ = Σ_T ⊗ Σ_S
     │      ├── spatial  Σ_S = (L + γI)⁻¹     ← DECISIONS B (γ), C (normalisation)
     │      └── temporal Σ_T = D^½ R_H D^½    ← DECISIONS D (centring), E (ridge ρ)
     │
 (5) joint eigenbasis, calibrate τ², κ        ← DECISION F
     │
 (6) forward OU SDE with schedule c(t)        ← DECISION G (c₀, α, c_min)
     │
 (7) train: noise distribution + loss weight  ← DECISIONS H, I
     │
 (8) sample: sampler, steps, terminal floor   ← DECISIONS J, K, L
     │
 forecast ensemble
```

Twelve decisions. The register in §10 lists all of them with their current value, its
provenance, and the criterion that will replace it.

---

## 1. Stage one — the backbone and the residual

**What we build.** A deterministic multi-horizon forecaster `f_φ: R^{N×P} → R^{N×H}`,
trained on masked MAE in raw units, then frozen. The residual `R₀ = Y − Ŷ` is the
diffusion's target.

**Why freezing.** It fixes the residual distribution *before* the generative stage, so
that distribution can be measured and a prior fitted to it. Joint training would make
the target a moving object and the whole prior-fitting step incoherent.

**Alternatives.**
- *Joint training* (one network, one loss). Rules out a fitted prior.
- *Model the signal directly* rather than the residual. This is what DiffSTG and
  SpecSTG do. It spends generative capacity on the conditional mean, which a
  deterministic network already estimates well — and it is why they need ~1M-parameter
  denoisers where 17k suffices here.
- *Different backbone*. Deliberately varied: linear (312 params) / GCN (17k) / Graph
  WaveNet (309k). This is not a tuning axis but the explanatory variable — it controls
  how much structure is left in the residual.

**What the choice asserts.** That the interesting stochasticity is in what a good point
forecaster misses, not in the signal as a whole.

---

## 2. Decision A — normalisation, and what it does to the math

**What we build.** Each sensor's residuals are divided by that sensor's training
standard deviation, `R_{i,h} ← r_{i,h} / σ_i`, and multiplied back after sampling.

**Why.** *Which sensor* an entry belongs to explains more of the raw residual's spread
than anything the prior is meant to describe: per-sensor residual σ spans 3–12× across
the network between its 5th and 95th percentiles, against at most 2.5× across the
twelve lead times. Without it the prior spends its per-mode budget reproducing a static
per-sensor scale that is known in advance.

**The consequence for the math, and it matters.** After this step the field has, by
construction, **unit variance at every node**. So a correctly specified prior must also
assign unit variance to every node. This constrains the spatial factor in a way the
current construction does not respect — see Decision C.

It also means the graph-smoothness assumption is being made about *standardised*
residuals, i.e. about the correlation structure rather than the covariance. That is the
right object (a busy detector and a quiet ramp should count equally toward "adjacent
sensors co-err"), but it should be stated, because `Σ_S = (L+γI)⁻¹` is normally
motivated as a covariance.

**Alternatives.**
- *Global scalar* (one σ for the whole tensor). Leaves the per-sensor scale in the
  data, which the prior then has to model.
- *Per-sensor per-horizon* `σ_{i,h}`. Removes the horizon profile too — but reproducing
  that profile is exactly the temporal factor's job, so this would gut `D_H`. Rejected
  for that reason, and it should be said explicitly rather than left implicit.
- *Rank/quantile transform to Gaussian marginals.* Would address the heavy tails
  (excess kurtosis 0.8–87.6) that the Gaussian prior cannot represent, at the cost of a
  non-linear, non-invertible-in-distribution map between the sampled and the reported
  space. Worth naming as the untaken option, since tails are the one calibration lever
  a Gaussian prior cannot touch.

**How other work does it.** HySRD-Net report per-sensor normalisation as the single
largest preprocessing gain in their residual-diffusion ablation. PriorGrad's diagonal
prior is exactly a per-coordinate variance, i.e. the same idea folded into the prior
instead of the preprocessing — which is a useful framing: *normalisation and a diagonal
prior are the same intervention applied at different points.*

---

## 3. What we are modelling

Three assertions, in order of strength.

**(i) Second-order.** A Gaussian prior fixes scale and correlation and nothing else.
Measured: excess kurtosis 0.8–87.6, heaviest exactly where the backbone is strongest.
So the prior is knowingly misspecified in the tails; that bounds achievable calibration
and is why a correction step exists at all.

**(ii) Zero-centred.** The forward process transports toward `N(0, Σ)`. The backbone's
masked-MAE objective targets the conditional median, so residual means are small but
not exactly zero. This is why `D_H` is a **second moment** rather than a variance about
the sample mean — the quantity the prior must reproduce at each lead time is `E[r²]`.

**(iii) Separable.** `Cov(R_{i,h}, R_{j,h'}) = [Σ_T]_{h,h'} · [Σ_S]_{i,j}`. Measured
separability (Van Loan–Pitsianis) is 0.82–0.99 across all eighteen cells, so a single
Kronecker product represents almost all of the covariance energy. What it cannot
represent is a spatial coupling whose *shape* depends on lead time — propagation, where
detectors ten minutes apart co-err most strongly at a ten-minute lead. That remainder is
left to the denoiser.

---

## 4. The prior is a product graph — the unification

This is the frame the rest of the document hangs on, and it corrects a claim in the
current methodology chapter.

The GTCNN parametric product graph over factor GSOs `S_S` (sensors) and `S_T`
(horizons) is

```
    S_◇ = Σ_{i,j∈{0,1}} s_ij (S_T^i ⊗ S_S^j)
        = s₀₀ I + s₀₁ (I ⊗ S_S) + s₁₀ (S_T ⊗ I) + s₁₁ (S_T ⊗ S_S)
```

and every member shares the eigenvectors `V_T ⊗ V_S`. Setting the joint precision `Q`
to a member of this family and reading off `Θ[k,h]`:

| member | s₀₀,s₀₁,s₁₀,s₁₁ | Θ[k,h] | what it asserts |
|---|---|---|---|
| isotropic | 1,0,0,0 | const | nothing |
| spatial only | 0,1,0,0 | λ_k + γ | smooth over the graph, no horizon structure |
| temporal only | 0,0,1,0 | 1/μ_h | horizon structure, spatially white |
| **tensor — ours** | 0,0,0,1 | **(λ_k + γ)/μ_h** | the two multiply |
| Cartesian (heat) | 0,1,1,0 | (λ_k+γ) + (ω_h+γ) | a literal space⊕time heat equation |
| strong | 0,1,1,1 | (1+λ_k+γ)(1+ω_h+γ) − 1 | both plus their product |

**So the thesis's prior is not outside the product-graph family — it is its tensor
member**, with a *fitted* temporal GSO (`Q_T = Σ_T^{-1}`) in place of a horizon path
Laplacian. The current methodology says the construction "is not the Laplacian of any
product graph". That is true and beside the point: it is the **GSO** of the tensor
product graph, which is what the family is defined over. The honest statement is:

> The prior is the tensor (Kronecker) member of the product-graph family, taking a
> shifted graph Laplacian as the spatial factor and an estimated precision as the
> temporal one. The Cartesian member is a genuine heat equation and is the natural
> competitor; the difference is whether the two factors' eigenvalues add or multiply.

**Measured** — held-out ΔNLL per mode vs isotropic, all 18 cells, γ=0.8, ρ=0.1;
positive = better than white noise:

| member | mean ΔNLL | cells better than iso |
|---|---:|---:|
| spatial only | +0.044 | 14 / 18 |
| temporal only | +0.722 | 18 / 18 |
| **tensor — ours** | **+0.740** | **18 / 18** |
| Cartesian fitted (add, not multiply) | +0.367 | 18 / 18 |
| Cartesian heat | −0.146 | **0 / 18** |
| strong heat | −0.233 | **0 / 18** |
| tensor heat | −0.395 | **0 / 18** |

Per-cell extremes: tensor ranges +0.13 (pems04f/gwn) to +1.16 (pemsbay/mlp).

Four things fall out:

1. **The tensor member is the best of the family**, on the mean and on 15 of 18 cells
   individually. The construction is the right member, decided by likelihood rather
   than preference.
2. **Every heat variant is worse than white noise on residuals — 0 of 18, without
   exception.** A heat prior asserts smoothness *along the horizon*; the residual is
   strongly structured along the horizon but not smooth, so the assertion is actively
   wrong. This is the principled reason the heat construction belongs on the raw signal
   (where it does well) and not on the residual, and it settles that question without a
   single GPU hour.
3. **Multiplication beats addition.** "Cartesian fitted" — the same two *fitted* factors
   combined additively instead of multiplicatively — captures only 40–55% of the
   tensor's gain while being positive everywhere. So the Kronecker structure is doing
   real work; the loss in the heat rows is the horizon *Laplacian*, not additivity as
   such. Both controls are needed to separate those, and both are here.
4. **The spatial factor alone is worth ≈0** (+0.044 mean, negative on 4 cells), and on
   three Graph WaveNet cells temporal-only actually edges out the full tensor. The
   prior's own likelihood says drop the graph factor exactly where the end-to-end
   results say it hurts.

---

## 5. Decisions B & C — the spatial factor

**What we build.** `Σ_S = (L + γI)^{-1}`, the shifted-Laplacian GMRF on the symmetrised,
component-repaired sensor graph. Its quadratic form is the graph Dirichlet energy plus a
ridge, so draws are smooth over the graph with high probability.

### Decision C — how Σ_S is normalised. **This is a specification bug, not a tuning knob.**

The code normalises `Σ_S` to **unit mean eigenvalue** (`τ² = N / Σ_k (λ_k+γ)^{-1}`), so
that the total scale is carried by `D_H` alone. But the *diagonal* of the resulting
`τ²Σ_S` is not 1 — it varies across nodes:

| dataset | N | γ=0.8: min / max / ratio | γ=0.2 ratio | γ=0.05 ratio |
|---|---:|---|---:|---:|
| pems08f/s | 170 | 0.926 / 1.147 / **1.24** | 1.97 | 3.61 |
| pems04f | 307 | 0.892 / 1.107 / **1.24** | 1.98 | 3.68 |
| pems04s | 307 | 0.892 / 1.116 / **1.25** | 1.99 | 3.69 |
| metrla | 207 | 0.966 / 1.133 / **1.17** | 1.71 | 3.51 |
| pemsbay | 325 | 0.944 / 1.236 / **1.31** | 2.15 | 3.34 |

Decision A standardised the residual to **unit variance at every node**. The prior then
asserts node variances differing by 1.17–1.31× at γ=0.8 — and by **3.3–3.7× at γ=0.05**.
The prior is misspecified on its own diagonal, and the misspecification grows as γ falls.

**The fix.** Normalise `Σ_S` to **unit diagonal** — the GMRF's *correlation* matrix,
`Σ_S ← D^{-1/2} Σ_S D^{-1/2}` with `D = diag(Σ_S)`. Then every node has unit variance
regardless of γ, and γ controls correlation range and nothing else. Standard practice
for GMRFs used as correlation models; one extra eigendecomposition per dataset.

**Measured effect: negligible.** Scored both ways, the difference in held-out ΔNLL is in
the fourth decimal (e.g. pems08f γ=0.8: 0.6545 vs 0.6548; the largest gap anywhere is
pems08s at γ=0.2, 0.8898 vs 0.9046, +1.7%). So this is a **coherence fix, not a
performance fix** — worth making because the specification should say what it means, and
because it decouples γ, but it changes no result.

> *Correction.* An earlier draft of this document proposed that this misspecification
> explained the γ=0.2 campaign losing 23 of 24 cells. The sweep below falsifies that:
> γ=0.2 is simply worse than γ=0.8 by held-out likelihood on two of the three backbones,
> which is a complete explanation on its own, and the diagonal effect is an order of
> magnitude too small to contribute.

### Decision B — γ. **The inherited value survives contact with the data.**

Currently 0.8, inherited from GAD, fixed across datasets and backbones. Swept by
held-out ΔNLL per mode (18 cells, ρ=0.1, tensor member):

| γ | 0.02 | 0.05 | 0.1 | 0.2 | 0.4 | **0.8** | 1.6 | 3.2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean ΔNLL | +0.453 | +0.609 | +0.677 | +0.716 | +0.734 | **+0.740** | +0.738 | +0.733 |
| argmax on | 1 | 3 | 0 | 2 | 4 | 1 | 3 | 4 |

**γ = 0.8 is the best shared value**, and a per-cell oracle would buy only +3.1% over it
(+0.763 vs +0.740). Per backbone the optima are mlp 0.1 (+2.2% over 0.8), gcn 0.8
(+0.0%), gwn 1.6 (+1.2%) — a monotone ladder in exactly the expected direction (weaker
backbone → more graph structure wanted), but the gains are small enough that a single
shared value costs almost nothing.

Two consequences worth carrying into the thesis:

- The inherited constant is now **derived rather than borrowed**, and it happens to
  agree. That is a stronger statement than "we followed GAD".
- **The γ axis is nearly flat above 0.4** (0.734 → 0.740 → 0.738 → 0.733). The spatial
  factor barely matters over the whole plausible range, which is the same conclusion the
  attribution table, the mode-slope diagnostic and the end-to-end results all reach —
  now reached a fourth way, before any training. It also explains the γ=0.2 campaign:
  0.2 scores +0.716 against 0.8's +0.740, i.e. slightly but consistently worse.
- A wrongly *small* γ is genuinely damaging on capable backbones — pems04f/gwn falls to
  **−0.84** at γ=0.02, far below white noise. The failure mode is asymmetric.

---

## 6. Decisions D & E — the temporal factor

**What we build.** `Σ_T = D_H^{1/2} R_H D_H^{1/2}` — the variance–correlation
decomposition (Barnard et al. 2000).

- `D_H = diag(δ_1..δ_H)`, `δ_h = mean over (window, sensor) of r_{s,i,h}²` — the
  **per-horizon second moment**. Answers *how large is the error at this lead time*.
- `R_H` — the **Pearson correlation** between horizons over all trajectories observed at
  all H horizons. Answers *how does an error at one lead time relate to another*.

**What each represents, and why this decomposition.** Two properties are measurable and
must both be carried, and no parametric kernel carries both:

1. *Scale grows with lead time* — up to 2.45× from h=1 to h=12. A shared per-horizon
   variance is misspecified before any correlation is considered.
2. *Correlation is not a function of separation alone* — lag-1 correlation is up to 0.31
   higher at the end of the window than at the start. Any **stationary** kernel (AR
   included) must assign one correlation per lag and cannot reproduce that ordering.

Splitting scale from correlation is what lets both be read off independently, and it is
why no stationarity has to be assumed: every entry of `R_H` is a free number.

**The estimate is cheap and well-determined**: millions of complete (window, sensor)
trajectories for a 12×12 matrix. This is the opposite of the spatial side, where a free
`N×N` covariance would be hopeless and the graph supplies the structure instead. *The
two factors draw on different sources of knowledge — a structural assumption for space,
an empirical fit for time — and meet in the same form.*

### Decision D — centring

`D_H` is uncentred (second moment, deliberately, per §3-ii) while `R_H` is centred
(Pearson subtracts the per-horizon mean). The product is PSD either way, so the prior is
well posed — but the two halves are not centred alike, and the fitted factor therefore
reproduces `E[r_h r_{h'}]` only up to the residual mean.

**Options:** (a) keep as is and state it; (b) use the uncentred correlation
`E[r_h r_{h'}]/√(δ_h δ_{h'})`, making the whole factor consistently uncentred and
exactly equal to the second-moment matrix; (c) centre both.
**(b) is the coherent choice** and is a two-line change. Its effect is bounded by the
residual mean, which is small — worth measuring once and then fixing.

### Decision E — the ridge ρ

`μ_h ← (1−ρ)μ_h + ρ·μ̄`, currently **ρ = 0.1**, chosen to tame the worst conditioning
number (PEMS-BAY 116 → 44). That is a numerical-stability criterion, not a statistical
one, and the user's instinct that it may not be needed is worth taking seriously:

- The estimand is 12-dimensional and the sample is ~10⁶ trajectories. Classical
  shrinkage (Ledoit–Wolf) targets the `p/n → const` regime; here `p/n ≈ 10⁻⁵`. **There
  is no estimation-error argument for shrinking at all.**
- The real argument is about the *inverse*: the smallest `μ_h` becomes the largest
  precision and therefore the fastest decay rate in the forward process, which sets the
  sampler's step budget. That is a computational argument, and it should be made as one
  — or dropped, since the ancestral sampler has no stability floor.
- It has a cost: correlations are damped by up to (1−ρ), so the prior commits to a more
  conservative dependence than the data shows.

**Measured, and the ridge should go.** Swept by held-out ΔNLL per mode over 18 cells:

| ρ | **0.0** | **0.01** | 0.03 | 0.1 (current) | 0.2 | 0.4 | 0.8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean ΔNLL | +0.806 | **+0.807** | +0.796 | +0.740 | +0.666 | +0.549 | +0.331 |
| argmax on | **8** | 5 | 4 | **0** | 1 | 0 | 0 |

**ρ = 0.1 is the argmax on none of the eighteen cells** and costs **9.1%** of the
prior's mean held-out fit against ρ=0.01. The worst individual cases are the
weak-backbone, high-conditioning cells the ridge was introduced to protect:
PEMS-BAY/mlp 1.439 → 1.156 (−20%), PEMS08F/mlp 1.216 → 1.050 (−14%). The ridge is
damaging exactly where it was supposed to help.

**Decision: ρ = 0.01, or 0.** The two are within 0.001 of each other on the mean; ρ=0
is the argmax on more cells, ρ=0.01 on the mean. Either removes a constant that was
chosen by a numerical criterion the sampler does not have. Unregularised conditioning
numbers run 6.7–815, which the ancestral sampler tolerates because it has no stability
floor — the very reason the ridge existed dissolves once the EM integrator is not the
headline sampler. Recommend **ρ = 0** for one fewer constant, with the sweep reported.

**Alternatives to the whole factor:** stationary kernel (ruled out by measurement,
above); horizon path-graph Laplacian (the heat option — §4 shows it scores *worse than
white noise*); full unstructured `H×H` covariance (that *is* what this is, since `R_H`
is a free matrix — the decomposition is a reparameterisation, not a restriction).

---

## 7. Decision F — joint calibration

Two scalars, both bookkeeping:

- **τ²** fixes the non-uniqueness `(Σ_T, Σ_S) → (cΣ_T, Σ_S/c)`. Under Decision C this
  becomes the unit-diagonal normalisation instead, which is a better answer to the same
  question.
- **κ** anchors `Θ_min` to a fixed value so one noise schedule serves every dataset;
  absorbed into `σ²`, leaves the prior unchanged.

Both belong in the boxed `Θ` equation, which currently omits them.

**Total-variance matching between arms.** The structured prior's mean stationary
variance is `(1/H)Σ_h δ_h`; the isotropic control's is the pooled training variance.
These agree to within 0.2% on the four PEMS benchmarks but differ by up to **4.8%** on
METR-LA and PEMS-BAY, because the pooled variance runs over every observed entry while
`D_H` is restricted to trajectories observed at all H horizons — and those sets diverge
where missingness is highest. **Fix: compute both over the same complete-trajectory
subset.** One line, removes a scale difference between arms that is currently reported
as if it were zero.

---

## 8. Decisions G–L — the diffusion

**Forward.** `dr = −c_t Q r dt + √(2c_t) σ dw`, stationary covariance `σ²Q^{-1}` by
Lyapunov. Verified: eigenbasis exact to 1e-14, per-mode variance error 2.1% on real
residuals, sliced-W₁ to the sampling floor. **No change needed.**

| # | decision | current | provenance | criterion to replace it |
|---|---|---|---|---|
| G | schedule `c(t)` = floor-constrained polynomial; `c₀=7, α=4, c_min=0.1` | GAD's exact values | inherited | `c₀` should follow from the mixing condition `2Θ_min c₀ ≥ 5` given the κ anchor, and be *derived* per dataset rather than reproduced. α and c_min: sweep or justify as insensitive. |
| H | training noise distribution: log-uniform in `[8e-4, c₀]` | ours (GAD uses `c₀·u^0.65`) | changed, undocumented reason | Justify: log-uniform covers every mode's transition `c̄θ ~ O(1)` evenly, which a stiff spectrum needs. Measure both. |
| I | loss weight `w = 1/(1+2v̄)` | ours, generalising GAD's SNR weight | derived | Alternatives already measured: unweighted MSE (GAD) and per-mode whitened (PriorGrad, costs 52% on weak backbones). Keep, report the comparison. |
| J | sampler: per-mode ancestral | ours | exact for the linear SDE, no stability floor | Keep; EM retained for GAD parity. |
| K | reverse steps: 50 | chosen | not derived | Sweep 5/25/50/100/200 and report the accuracy–cost curve. This is also a *result* (few-step behaviour), not just a setting. |
| L | **terminal floor `c_lo = 0.02`** | sampler default, never passed | **defect** | Must equal the training floor `8e-4`. At 0.02 the sampler discards 3.15% of the isotropic prior's variance but **19–28% of the structured prior's**, in every cell. Measured: matching it moves iso <1pt of coverage and the structured arm +4.8pt, past iso on coverage, Winkler and energy. |

---

## 9. How other work makes these choices

| choice | GAD | PriorGrad | DiffSTG / SpecSTG | HySRD | here |
|---|---|---|---|---|---|
| what is corrupted | static graph signal | signal | signal | reshaped signal | backbone residual |
| prior | `(L+γI)^{-1}`, γ=0.8 fixed | data-dependent **diagonal** | isotropic | isotropic (data reshaped instead) | `Σ_T ⊗ Σ_S`, tensor product graph |
| how the prior's constant is set | fixed by fiat | from data statistics | n/a | n/a | **to be derived** |
| loss | unweighted MSE | per-mode whitened | standard | standard | SNR-weighted |
| sampler | Euler–Maruyama, ~5000 steps | DDPM | DDPM-200 / DDIM-40 | — | exact per-mode ancestral, 50 |
| normalisation | — | folded into the prior | global | **per-sensor, reported as their largest gain** | per-sensor |

Two observations worth carrying into the thesis. **PriorGrad's diagonal prior and our
per-sensor normalisation are the same intervention at different points in the
pipeline** — which is a clean way to explain why the spatial factor has so little left
to do by the time the prior sees the field. And **GAD fixes γ by fiat**, which we
adopted for parity; deriving it independently returns the same value and shows the axis
to be nearly flat above 0.4, so the agreement is not a coincidence but a statement about
how little the spatial factor can do at any shift.

---

## 10. Hyperparameter register

| # | constant | current | provenance | status |
|---|---|---|---|---|
| A | residual normalisation | per-sensor σ_i | measured (3–12× spread) | **settled** |
| B | spatial shift γ | 0.8 | GAD, inherited | **DERIVED — keep 0.8.** Best shared value by held-out NLL; oracle worth +3.1%; axis flat above 0.4 |
| C | Σ_S normalisation | unit mean eigenvalue | implementation default | **fix** — unit diagonal. Coherence only; effect in the 4th decimal |
| D | R_H centring | Pearson (centred) vs uncentred D_H | inconsistent | **fix** — uncentred throughout |
| E | temporal ridge ρ | 0.1 | conditioning number | **DERIVED — set to 0.** ρ=0.1 is argmax on 0/18 cells and costs 9.1% of held-out fit |
| F | τ², κ | as implemented | bookkeeping | settled; put in the equation |
| F′ | arm variance matching | different subsets | oversight | **fix** — common subset |
| G | c₀, α, c_min | 7, 4, 0.1 | GAD | **derive c₀** from mixing |
| H | noise draw | log-uniform [8e-4, c₀] | ours | justify, measure |
| I | loss weight | SNR | ours | settled, comparison already run |
| J | sampler | ancestral | ours | settled |
| K | reverse steps | 50 | chosen | **sweep and report** |
| L | terminal floor c_lo | 0.02 | default, never passed | **fix** — 8e-4 |
| M | ensemble size K | 100 | protocol | settled |
| N | denoiser | 17,305-param GCNN | GAD-sized | settled (capacity is a reported axis) |

### Settled so far

| decision | outcome | changes the runs? |
|---|---|---|
| B — γ | **0.8 confirmed** by held-out likelihood; now derived, not inherited | no |
| E — ρ | **0.1 → 0**; the current value is optimal on no cell and costs 9.1% | **yes — refit the prior** |
| C — Σ_S normalisation | unit diagonal; coherence fix, immaterial numerically | no (but do it) |
| D — R_H centring | uncentred throughout | measure, then decide |
| F′ — arm variance matching | common complete-trajectory subset | small, do it |
| L — sampler floor | **0.02 → 8e-4**; the arms are currently resolved to fidelities differing 6–9× | **yes — re-evaluate** |
| §4 family | tensor member confirmed best; all heat variants lose 18/18 | no — becomes an ablation table |

Only **two** changes actually require re-running anything: ρ (retrain, because the prior
changes) and the sampler floor (re-evaluate only, checkpoints unaffected). Everything
else is either confirmed or free.

Still open: G (c₀ derivation), H (noise draw justification), K (step sweep).

---

## 11. What the final run has to produce

Once §10 is settled, one campaign, one protocol:

1. **Main comparison** — structured vs isotropic, 6 datasets × 3 backbones × ≥5 seeds,
   history conditioning, under the corrected sampler.
2. **Backbone ladder** — the same grid *is* this analysis; it needs no extra runs.
3. **Product-graph family ablation** — §4, held-out likelihood; already CPU-only. Add
   trained cells only for the members that score competitively (on current evidence:
   tensor and Cartesian-fitted).
4. **Conditioning** — forecast-only vs history, already complete.
5. **Sampler steps** — 5/25/50/100/200 accuracy and cost, eval-only on saved
   checkpoints. Doubles as the few-step result.
6. **Sampling speed** — vs DiffSTG at matched hardware, already measured.
7. **Capacity** — 17k vs 996k, needs test evals and re-running under history
   conditioning.
8. **SOTA table** — already complete and robust to every change above (GWN numbers move
   by ≤0.0002 CRPS).
