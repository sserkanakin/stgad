# Building the spatio-temporal diffusion prior: what is correct, and what the literature + data say

Date 2026-07-26. This report answers, from the math and from the GAD/GTCNN/diffusion literature,
**how the Σ = Σ_T ⊗ Σ_S prior should be built, whether our construction is correct, and what to do
next.** It combines (a) a literature deep-dive, (b) a from-scratch structural analysis of the raw
signal vs the backbone residual, and (c) a machine-precision verification that running the forward
SDE in the joint eigenspace is exact.

Scripts / data: `scripts/analysis/raw_vs_residual_prior.py` and `kronecker_audit.py` →
`results/analysis/paper_results/{raw_vs_residual_prior,kronecker_audit}.md` + `figs/`. Companion:
`docs/kronecker_audit_memo.md`.

---

## 0. TL;DR

1. **The code is correct.** The forward runs per-mode in the (V_S⊗V_T) basis; we verified numerically
   that this basis exactly diagonalises the full precision Q=Σ_T⁻¹⊗Σ_S⁻¹ and reproduces the full
   `NH×NH` covariance marginal to **machine precision (~1e-15)**. It is an exact reparameterisation,
   not an approximation. This matches the "whitened score" line (Alido et al. 2025) — the standard
   way to make an anisotropic Gaussian forward tractable.
2. **The form (separable Kronecker) is correct** — the residual covariance is 82–93% Kronecker-
   separable, the raw signal 99–100%. A matrix-normal prior is the right family.
3. **Our construction is a hybrid, and its two factors have different pedigrees.** Spatial =
   graph-Laplacian GMRF `(L+γI)⁻¹` (GAD-correct). Temporal = *empirical* covariance (a matrix-normal /
   PriorGrad-style data prior, **not** a graph). The GAD/GTCNN-canonical object is the *Cartesian
   product-graph* (both axes Laplacians); ours trades that purity for temporal data-fit.
4. **Structure lives where the signal is concentrated.** On the raw signal the graph factor genuinely
   helps on top of time (graph|T = +0.06…+0.07 nats/mode); on the frozen-backbone residual it is ~0.
   The backbone already ate the spatial structure.
5. **Two literature-backed reasons structure doesn't win end-to-end**, and one fix to try:
   (i) the training loss is GAD's per-*sample* SNR weight, **not** the per-*mode* `Σ⁻¹`/precision
   weight PriorGrad/SpecGrad use — and the θ-weighted `whitened_mse` is implemented but **not wired
   into `compute_loss`**; (ii) the Wiener/SNR-per-mode bound: matched noise sits in the signal
   subspace and can't beat isotropic on a concentrated signal. The untested cell the literature points
   at is **raw signal + structured prior + whitened (Σ⁻¹) loss**.

---

## 1. Three ways to build it (math)

Notation: residual/signal field `R ∈ ℝ^{N×H}`, `vec(R)` with index `h·N+i`.

| | Level 0 — GAD (space only) | Level 1 — GTCNN/GAD product graph | Level 2 — **ours** |
|---|---|---|---|
| Object | GMRF on the graph | Cartesian product-graph GMRF | matrix-normal (Kronecker of covariances) |
| Precision `Q` | `L_S+γI` | `L_□+γI = L_T⊗I + I⊗L_S + γI` | `Σ_T⁻¹ ⊗ (L_S+γI)` |
| Joint spectrum | `λ_S+γ` | `λ_S+λ_T+γ` (**additive**) | `q_S · (1/μ_T)` (**multiplicative**) |
| Temporal factor | — | path-graph Laplacian (imposed smoothness) | **empirical covariance** (data-fit) |
| Drift meaning | graph heat equation | heat equation on the product graph | variance-matched Ornstein–Uhlenbeck |
| Stiffness | bounded `[γ, λ_max+γ]` | bounded `[γ, 2+γ]` | **unbounded** (product) — 18–139 here |

The forward SDE (all three) is `dr = −c(t)·Q·r dt + √(2cσ²)dW`, white `dW`; the structure is entirely
in the drift operator `Q`, the injected noise is isotropic. Any zero-mean Gaussian `N(0,Σ)` is the
stationary law of this SDE with `Q=Σ⁻¹` — GAD's heat equation is the special case `Q=L+γI`; ours is
the case where `Q` is a fitted precision.

**How the temporal matrix is built** (`prior._fit_sigma_T`): pool every `(node, window)` H-vector of
errors, take `δ_h = mean(r_h²)` (per-horizon variance) and `R_H = corrcoef` across horizons, set
`Σ_T = D^½ R_H D^½` (the empirical temporal covariance), ridge-shrink toward scaled-identity
(`μ←(1-ρ)μ+ρμ̄`, ρ=0.1), eigendecompose. It pools over nodes (⇒ one shared temporal covariance = the
separability assumption), and it is an MLE (⇒ estimation noise, train→test drift; hence the shrinkage).

---

## 2. Literature deep-dive

### 2a. Is shaping the forward-noise covariance a good idea? (general diffusion)

The consistent finding: **structured/colored forward noise helps only when it matches the data's
per-mode SNR structure AND is paired with the right loss/parameterisation; otherwise it is neutral or
hurts.** Evidence:

- **EDM** (Karras et al. 2022, arXiv:2206.00364) — noise is isotropic; what they shape is the *scalar*
  noise-level distribution `p(σ)` and per-level loss weight. First-order lesson: "how much noise, aimed
  at which SNR levels" dominates — any Σ-shaping induces a per-mode SNR schedule that must be *tuned*.
- **Variational Diffusion Models** (Kingma et al. 2021, arXiv:2107.00630) — the diffusion loss is an
  integral over SNR, invariant to *how* you traverse it. ⟹ the payoff of covariance shaping lives
  entirely in *which modes get which SNR when*, plus the endpoints.
- **Blurring / heat-dissipation** (Rissanen et al. IHDM, arXiv:2206.13397; Hoogeboom & Salimans,
  arXiv:2209.05557) — non-isotropic noise in the Fourier/DCT basis (per-frequency schedule). Useful
  *inductive bias* (coarse-to-fine), but on its own does **not** beat isotropic; needs an isotropic
  noise floor for a well-defined score.
- **Soft / Cold diffusion** (Daras et al. arXiv:2209.05442; Bansal et al. arXiv:2208.09392) — general
  linear / deterministic corruptions. Work only with retained stochasticity; deterministic degradation
  is neutral-to-worse. Stochasticity/SNR structure, not the specific operator, carries the modelling
  power.
- **Critically-damped Langevin (CLD)** (Dockhorn et al. arXiv:2112.07068) — anisotropic noise injected
  through an *auxiliary velocity*; **helps** (smoother score). Positive result for deliberately
  anisotropic forward noise — via coupling, not colouring the data axis.
- **Non-isotropic Gaussian noise** (Voleti et al. arXiv:2210.12254) — general non-diagonal Σ in data
  space **did not improve** over isotropic. **Direct negative result** for arbitrary covariance shaping —
  our residual finding is consistent with it.
- **Spectral matching** — Shortest-Path Diffusion (Das et al. arXiv:2306.00501) argues the optimal
  forward is spectrally shaped to the data power spectrum; **EqualSNR / Fourier perspective** (Falck et
  al. 2025, arXiv:2505.11278) shows equalising per-frequency SNR **helps on high-frequency-dominated
  data and gives little on ordinary images** — i.e. the benefit is *data-spectrum dependent*.
- **Whitened Score diffusion** (Alido et al. 2025, arXiv:2505.10311) — learns the Σ-whitened score so
  you **never invert Σ**; enables arbitrary anisotropic Gaussian forwards, and structured priors
  **help when matched** to structured measurement noise. *This is our eigenspace method* (diagonalise
  once, per-mode) stated as a general recipe.
- **Blue noise** (Huang et al. 2024, arXiv:2402.04930) — correlated *temporal* noise helps video
  coherence: matched *temporal* structure is the lower-risk win.

**Unifying view (Wiener / SNR-per-mode).** In Σ's eigenbasis each mode is an independent scalar SNR
trajectory; the optimal denoiser is the Wiener filter shrinking by `SNR/(SNR+1)`. Structured Σ helps
iff it *raises* SNR on hard-but-informative modes and lowers it where denoising is already trivial; it
hurts/neutral if it merely re-noises the signal subspace. This is precisely why "matched" priors can
fail on concentrated signals.

### 2b. Data-dependent diffusion priors — the direct precedent

Almost all ST forecasting diffusions use an **isotropic** prior and put structure only in the
*denoiser/conditioning*: DiffSTG (UGnet + graph conv, arXiv:2301.13629), SpecSTG (diffuses the graph-
Fourier signal, arXiv:2401.08119), CSDI (arXiv:2107.03502), TimeGrad (arXiv:2101.12072), ScoreGrad
(arXiv:2106.10121), ProGen (arXiv:2411.01267, unverified authors), GCRDD (2023, unverified), DYffusion
(arXiv:2306.01984). None structure the prior covariance.

The two that **do** are our real precedent:

- **PriorGrad** (Lee et al., ICLR 2022, arXiv:2106.06406). Replaces `N(0,I)` with an instance-specific
  **diagonal** `N(μ,Σ_c)` computed from the conditioning (per-frame energy / per-phoneme stats).
  Forward `x_t=√ᾱ(x_0−μ)+√(1−ᾱ)ε, ε∼N(0,Σ)`; **loss is Σ⁻¹-weighted**: `‖ε−ε_θ‖²_{Σ⁻¹}`.
  Benefit: prior closer to data ⇒ shorter denoising distance ⇒ faster convergence, better quality,
  **robust under smaller network capacity**.
- **SpecGrad** (Koizumi et al., Interspeech 2022, arXiv:2203.16749). Generalises PriorGrad to a **full
  correlated** covariance via `Σ = LLᵀ`, `L=G⁺MG` (STFT × diagonal spectral envelope × iSTFT); colored
  noise `ε=L ẽ`; **whitened loss** `‖L⁻¹(ε−F_θ)‖²`. Cheap via FFT. Benefit: higher fidelity, esp. high
  frequencies, at ≈ isotropic cost. **A Kronecker Σ_T⊗Σ_S is the graph×time analogue of SpecGrad's
  `L=G⁺MG`.**

Takeaway: the precedent that a structured, data-derived prior *helps* comes bundled with a **whitened
(Σ⁻¹-weighted) loss** and a regime (per-instance prior, limited capacity, matched structure).

### 2c. GAD / GTCNN / Kronecker-GP math

- **GTCNN** (Isufi & Mazzola 2021 arXiv:2103.01730; Sabbaqi & Isufi 2022 arXiv:2206.15174) — product
  graph (Kronecker `λ_Sλ_T`, Cartesian `λ_S+λ_T`, strong = both) as the *convolution support*.
- **GAD** (Rozada, Vimal K.B., Cavallo, Marques, Jamali-Rad, Isufi 2025, arXiv:2510.05036) — OU forward
  with Laplacian drift `−c_t L_γ x`, FCPS schedule, **GMRF stationary `σ²(L+γI)⁻¹`**, Tweedie backward
  = graph denoising. Our spatial factor *is* GAD's; our forward machinery is GAD's.
- **Matrix-normal / Kronecker GP** (Saatçi 2011; Rakitsch/Stegle et al. NeurIPS 2013; Wilson & Nickisch
  ICML 2015) — separable Σ=Σ_S⊗Σ_T ⟹ per-factor eigendecomposition, `O(N³+H³)` not `O((NH)³)`, and the
  vec-trick. **Van Loan–Pitsianis** (1993) nearest-Kronecker rearrangement gives a **separability index**
  `σ₁²/‖Σ‖_F²`. A **sum of a few Kronecker terms** (Rakitsch/Stegle) is the tractable escape if one term
  is insufficient.
- **Separability in geostatistics** (Cressie & Huang 1999; Gneiting 2002; tests: Fuentes 2006;
  Mitchell–Genton–Gumpertz 2005) — separable ⟺ *no space-time interaction* (spatial decay independent of
  time lag; no advection/transport). It is the strongest, most fragile assumption; non-separable classes
  exist for when it fails.
- **Empirical vs Laplacian temporal factor** — a path-graph Laplacian gives a *sparse tridiagonal*,
  high-bias/low-variance, local-smoothness precision (a heat drift in time); an empirical `Σ_T` gives a
  *dense*, low-bias/high-variance, data-matched precision (drift toward the observed temporal spectrum)
  but needs shrinkage and is stiffer. AR(1) (tridiagonal GMRF) is the natural regularised bridge.

---

## 3. Analysis: raw signal vs backbone residual

Fields from the same npz (`y_true = y_pred + residual`): **raw** `(y_true−mean)/std`, **resid**
`residual/σ`. Backbone gwn, γ=0.8, held-out test.

### 3.1 Separability (VLP top-1 / top-3; 1.0 = perfectly Kronecker)

| dataset | raw | resid |
|---|--:|--:|
| metrla  | 0.995 / 1.000 | 0.934 / 0.970 |
| pems08f | 0.999 / 1.000 | 0.900 / 0.950 |
| pems04f | 0.998 / 1.000 | 0.818 / 0.914 |

Both are well-approximated by a single Kronecker product; the **raw signal is essentially perfectly
separable**, the residual slightly less (gwn's space-time mixing leaves a little non-separable
structure). A matrix-normal prior is justified for both.

### 3.2 Spatial structure (graph spectrum; Dirichlet white=1, lower=graph-smooth)

| dataset | field | Dirichlet | DC-mode share | smooth-10% (white .10) | align w/ 1/(λ+γ) |
|---|---|--:|--:|--:|--:|
| metrla  | raw | 0.478 | 0.242 | 0.525 | 0.495 |
| metrla  | resid | 0.773 | 0.040 | 0.253 | 0.792 |
| pems08f | raw | 0.516 | 0.360 | 0.482 | 0.240 |
| pems08f | resid | 0.861 | 0.027 | 0.149 | 0.549 |
| pems04f | raw | 0.310 | 0.449 | 0.634 | 0.169 |
| pems04f | resid | 0.871 | 0.030 | 0.155 | 0.306 |

The **raw signal is strongly graph-smooth** (Dirichlet 0.31–0.52; 24–45% of energy in the single
DC/global mode) — exactly the low-frequency regime GAD's `(L+γI)⁻¹` prior is built for. The **residual
is nearly white** (Dirichlet 0.77–0.87; DC 3–4%) — the backbone removed the smooth part. (Note: raw
energy is so DC-concentrated that its per-mode `align` with the graded heat kernel is *lower* than the
residual's — raw is a DC spike + rest, not a smooth `1/(λ+γ)` decay; a heat kernel is an imperfect fit
to raw too.)

### 3.3 Temporal structure (Σ_T)

| dataset | field | var growth (H/1) | lag-1 ρ | common-mode share | off-band precision (path=0) |
|---|---|--:|--:|--:|--:|
| metrla  | raw | 1.00 | 0.94 | 0.86 | 0.21 |
| metrla  | resid | 1.72 | 0.66 | 0.44 | 0.19 |
| pems08f | raw | 1.00 | 0.98 | 0.97 | 0.46 |
| pems08f | resid | 1.24 | 0.40 | 0.33 | 0.52 |
| pems04f | raw | 1.00 | 0.98 | 0.97 | 0.53 |
| pems04f | resid | 1.19 | 0.35 | 0.30 | 0.54 |

**Raw** temporal structure is extreme smoothness (lag-1 ρ 0.94–0.98, common-mode 0.86–0.97, no variance
growth) — a slowly varying level. **Residual** is variance-growth-with-horizon (1.2–1.7×) + a weaker
common mode. **Neither is a path graph** (off-band precision mass 0.19–0.54 ≫ 0): the empirical Σ_T is a
dense, common-mode-dominated covariance, not a tridiagonal local-smoothing operator.

### 3.4 Attribution — separate temporal, separate spatial, combined (held-out per-mode NLL)

ΔT = iso−T-only (time buys); ΔS = iso−S-only (graph buys); **graph|T** = T-only−full (graph *on top of*
time). nats/mode, higher = better.

| dataset | field | iso | T-only | S-only | full | ΔT | ΔS | **graph\|T** |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| metrla  | raw | 0.108 | −0.868 | −0.054 | −0.931 | 0.976 | 0.162 | **+0.063** |
| metrla  | resid | 0.567 | 0.198 | 0.506 | 0.170 | 0.369 | 0.062 | **+0.028** |
| pems08f | raw | 0.504 | −1.038 | 0.377 | −1.111 | 1.542 | 0.128 | **+0.073** |
| pems08f | resid | 0.500 | 0.343 | 0.486 | 0.344 | 0.157 | 0.014 | **−0.000** |
| pems04f | raw | 0.514 | −0.900 | 0.314 | −0.963 | 1.413 | 0.200 | **+0.063** |
| pems04f | resid | 0.542 | 0.433 | 0.546 | 0.453 | 0.109 | −0.004 | **−0.020** |

**Temporal dominates in both worlds** (ΔT 0.1–1.5). **The graph factor earns its keep on the raw signal**
(graph|T +0.06…+0.07, always positive) but is **inert/negative on the residual** (graph|T ≈ 0). This is
the crisp, quantified answer to "does working in raw signals help the graph prior": *yes* — because the
raw signal carries the graph-smooth structure the backbone strips from the residual. Note even on raw,
ΔT ≫ graph|T (≈15–20×): time is still the overwhelming factor, because the raw signal is *extremely*
temporally smooth.

### 3.5 Eigenspace-forward correctness (verification)

Building the full `NH×NH` precision `Q=Σ_T⁻¹⊗Σ_S⁻¹` and checking the per-mode eigenspace forward against
it:

| dataset | N·H | orthonormality | Q diagonalised by V_S⊗V_T | per-mode vs full marginal | empirical cov (20k draws) |
|---|--:|--:|--:|--:|--:|
| metrla  | 2484 | 3.1e-14 | 2.3e-15 | 7.3e-15 | 0.323 |
| pems08f | 2040 | 2.7e-14 | 2.4e-15 | 6.9e-15 | 0.307 |
| pems04f | 3684 | 4.5e-14 | 2.7e-15 | 8.4e-15 | 0.414 |

**The per-mode eigenspace forward is exact** — `V_S⊗V_T` diagonalises the full precision and reproduces
the full-covariance marginal to machine precision (~1e-15). The forward-in-eigenspace is **not wrong**
and **not an approximation**; it is an exact reparameterisation (the standard whitened-score trick). The
`emp_err` ≈ 0.3–0.4 is only finite-sample noise from estimating an `NH×NH` covariance with 20k draws
(`NH²` ≫ 20k), not a code error — the analytic 1e-15 is the proof.

---

## 4. A correctness gap in the training loss

`losses.py`: the default training loss is **`snr`** = plain MSE weighted per *sample* by GAD's SNR
weight `w(c̄)=1/(1+2v̄)` (every noise level contributes comparably). This is GAD-faithful, but it is
**not** the per-*mode* precision (`Σ⁻¹`/θ) weighting that PriorGrad and SpecGrad use for a structured
prior. Under plain-over-modes MSE, the loud (high-variance) modes — exactly the ones a structured prior
loads with noise — dominate the gradient, so the model is *not* incentivised to exploit the structure.
A θ-weighted **`whitened_mse` is implemented** in `losses.py` (its docstring states this precisely) but
is **not wired into `compute_loss`** (dispatch = `snr`/`mse`/`mahalanobis` only). So the PriorGrad/
SpecGrad-principled loss for our structured prior exists in the codebase but has never trained a model.

This is likely a *co-cause* (with the Wiener bound and the whitened-out residual structure) of the
recurring "iso ties/beats structured" result.

---

## 5. Synthesis — what is correct, and what to do

**Correct as built:** the eigenspace forward (exact), the separable form (data is 82–100% Kronecker),
and the spatial `(L+γI)⁻¹` factor (GAD). **Framing to fix:** the temporal factor is an empirical
matrix-normal / PriorGrad-style prior, *not* a graph — so the honest name is a *separable (matrix-normal)
diffusion prior: graph heat-kernel on space × empirical covariance on time*, of which GAD's product-graph
is the Laplacian-temporal special case. **Honest result:** structure matches the raw signal (graph|T>0)
but not the strong-backbone residual (graph|T≈0), and even where matched, the Wiener/SNR-per-mode bound
plus a non-whitened loss keep it from beating isotropic end-to-end.

### What can be done (ranked, literature-backed)

1. **Wire in and test the whitened (Σ⁻¹/θ) loss** (PriorGrad/SpecGrad; already coded as `whitened_mse`).
   This is the *principled* loss for a structured prior and the missing ingredient vs the precedent that
   reports structure helping. Lowest-effort, highest-information.
2. **Run the untested cell: raw signal + structured prior + whitened loss.** The raw signal is where the
   graph factor is meaningful (graph|T>0) *and* nearly perfectly separable; the earlier raw campaign used
   the default (non-whitened) loss, so this combination has never actually been tried. Frame as a test,
   not a promised win — the Wiener bound says it may still tie isotropic.
3. **Match, don't impose, the spatial graph** (correlation graph for flow, road for speed; cf.
   `kronecker_audit_memo.md`), or reshape the prior's spectrum to the measured one (SpecGrad / GAD
   Remark-1). Only adopt a single Kronecker term where the VLP index is high (it is); otherwise use a
   *sum of two Kronecker terms* (Rakitsch/Stegle).
4. **Present the GAD/GTCNN Cartesian product graph as the principled baseline** and the empirical Σ_T as
   the measured alternative — keeps the math honest in the GSP framework and makes the *characterisation*
   (when/why structure helps, via Wiener) the contribution.
5. **Keep an isotropic noise floor** and **report against an honest control** (same denoiser capacity and
   scalar SNR schedule for iso vs structured) — the recurring confound in this literature.

### The one-line verdict
The prior is *correctly implemented* and the *right family*; the graph half is only meaningful on
signals whose structure the model hasn't already removed (raw, not residual); and whether structure
*wins* hinges on the loss (whitened vs plain) and the Wiener bound — which is a cleaner, more defensible
thesis than "our graph prior wins."

---

## 6. References

Verified via web this session unless marked (unverified). GAD arXiv id from project records.

**Structured-noise diffusion**
- Karras, Aittala, Aila, Laine. *Elucidating the Design Space of Diffusion-Based Generative Models* (EDM). NeurIPS 2022. arXiv:2206.00364.
- Kingma, Salimans, Poole, Ho. *Variational Diffusion Models.* NeurIPS 2021. arXiv:2107.00630.
- Rissanen, Heinonen, Solin. *Generative Modelling with Inverse Heat Dissipation.* ICLR 2023. arXiv:2206.13397.
- Hoogeboom, Salimans. *Blurring Diffusion Models.* ICLR 2023. arXiv:2209.05557.
- Daras, Delbracio, Talebi, Dimakis, Milanfar. *Soft Diffusion: Score Matching for General Corruptions.* 2022. arXiv:2209.05442.
- Bansal et al. *Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise.* NeurIPS 2023. arXiv:2208.09392.
- Dockhorn, Vahdat, Kreis. *Score-Based Generative Modeling with Critically-Damped Langevin Diffusion.* ICLR 2022. arXiv:2112.07068.
- Voleti, Pal, Oberman. *Score-based Denoising Diffusion with Non-Isotropic Gaussian Noise Models.* 2022. arXiv:2210.12254.
- Das et al. *Image Generation with Shortest Path Diffusion.* ICML 2023. arXiv:2306.00501. (authors unverified)
- Falck et al. *A Fourier Space Perspective on Diffusion Models.* 2025. arXiv:2505.11278. (authors unverified)
- Alido, Li, Sun, Tian. *Whitened Score Diffusion: A Structured Prior for Imaging Inverse Problems.* 2025. arXiv:2505.10311.
- Huang et al. *Blue Noise for Diffusion Models.* 2024. arXiv:2402.04930. (authors unverified)
- Dieleman. *Diffusion is Spectral Autoregression* (blog, 2024, informal).

**Data-dependent priors + spatio-temporal / graph diffusion**
- Lee, Kim, Shin, Tan, Liu, Meng, Qin, Chen, Yoon, Liu. *PriorGrad.* ICLR 2022. arXiv:2106.06406.
- Koizumi, Zen, Yatabe, Chen, Bacchiani. *SpecGrad.* Interspeech 2022. arXiv:2203.16749.
- Wen, Lin, Xia, Wan, Wen, Zimmermann, Liang. *DiffSTG.* ACM SIGSPATIAL 2023. arXiv:2301.13629.
- Lin, Shi, Han, Gao. *SpecSTG.* 2024. arXiv:2401.08119.
- Tashiro, Song, Song, Ermon. *CSDI.* NeurIPS 2021. arXiv:2107.03502.
- Rasul, Seward, Schuster, Vollgraf. *TimeGrad.* ICML 2021. arXiv:2101.12072.
- Yan, Zhang, Zhou, Zhan, Xia. *ScoreGrad.* 2021. arXiv:2106.10121.
- *ProGen.* 2024. arXiv:2411.01267. (authors/venue unverified)
- Li et al. *GCRDD.* 2023. (venue/authors unverified)
- Rühling Cachay, Zhao, Joren, Yu. *DYffusion.* NeurIPS 2023. arXiv:2306.01984.

**Graph-time SP + Kronecker/GMRF math**
- Isufi, Mazzola. *Graph-Time Convolutional Neural Networks.* 2021. arXiv:2103.01730. Sabbaqi, Isufi. 2022. arXiv:2206.15174.
- Rozada, Vimal K.B., Cavallo, Marques, Jamali-Rad, Isufi. *Graph-Aware Diffusion for Signal Generation (GAD).* 2025. arXiv:2510.05036.
- Saatçi. *Scalable Inference for Structured Gaussian Process Models.* PhD thesis, Cambridge, 2011.
- Rakitsch, Lippert, Borgwardt, Stegle. *It is all in the noise.* NeurIPS 2013.
- Wilson, Nickisch. *Kernel Interpolation for Scalable Structured GPs (KISS-GP).* ICML 2015.
- Van Loan, Pitsianis. *Approximation with Kronecker Products.* 1993.
- Cressie, Huang. *Classes of Nonseparable, Spatio-Temporal Stationary Covariance Functions.* JASA 1999. Gneiting. *Nonseparable, Stationary Covariance Functions for Space-Time Data.* JASA 2002.
- Fuentes. *Testing for separability of spatial–temporal covariance functions.* JSPI 2006. (details unverified) Mitchell, Genton, Gumpertz. *Testing for separability of space–time covariances.* Technometrics 2005. (details unverified)
- Rue, Held. *Gaussian Markov Random Fields: Theory and Applications.* 2005. Lindgren, Rue, Lindström. *SPDE approach.* JRSS-B 2011.
- Ledoit, Wolf. *A well-conditioned estimator for large-dimensional covariance matrices.* JMVA 2004.
