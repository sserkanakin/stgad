# Decision log

One entry per locked decision; newest first. Methodology decisions reference the
evidence that forced them.

## 2026-07-28 — GTCNN product-graph campaign: γ=0.1 with c₀=100, DERIVED not assumed

**Decision.** The raw-signal product-graph campaign (`20260728_pgraph`, 5 arms × 6
datasets, GCNN 17,305-param denoiser, history-only conditioning) runs at **γ=0.1, c₀=100**,
with every product spectrum rescaled to **λ ∈ [0,2]**. The rescale is an exact
reparameterization (σ² absorbs it), and it buys the controlled comparison: θ = λ+γ gives
**identical θ_min, stiffness (21) and mixing number (20) in every cell**, so the only thing
that differs across arms is the *shape* of θ over graph×time modes.

**Why c₀=100 and not GAD's rule.** GAD's mixing condition `2·c₀·θ_min ≥ 5` implicitly
assumes per-mode data variance is comparable to σ². On the **raw signal it is not**: the
graph-DC × time-DC mode is the mean traffic level and carries **49–670× its own stationary
variance** (measured, all 24 product cells). At the previous campaign's c₀=25 the forward
still leaves **61% of the largest target variance un-erased at t=T** — i.e.
`results/20260724_rawsig_*` ran an **under-mixed forward**, an independent reason those
numbers are unreliable on top of the run-health defects in
[[project_rawsignal_verdict_20260723]]. The requirement is `c₀ ≥ 5·ln(80·r)`; measured r
gives 42–55 across every cell, so c₀=100 carries a 1.8× margin. Gate check B3 recomputes
this per cell rather than trusting the rule.

**Product forms follow GTCNN, not the re-normalized product graph.** GTCNN (arXiv:2206.15174)
defines the product on the FACTOR GSOs (their Sec. 2.1) and Prop. 3 gives eigenvectors
exactly `V_S ⊗ V_T` — so the separable form is the paper-faithful one and the verified 2-D
rotation is reused unchanged. Verified numerically: the Cartesian and strong GSOs reproduce
the product graph's **edge set exactly**, and all three products are **single-component on
all 6 datasets**. The re-normalized product Laplacian is a *different* object (measured
1–8% spectral difference for Cartesian, 32–36% strong, 70–99.8% tensor) and is NOT used.

**prod_tensor is a negative control, on two independent structural grounds.** (1) Its GSO
is not the tensor product graph's operator — the normalized Laplacian's nonzero diagonal
leaks 10,316 edges the graph lacks (pems08f). (2) It pins **N+H−1 modes at exactly θ_min**
(181/2040 = 8.9% on pems08f), which the prior cannot distinguish and gives maximum variance.
Run anyway, to convert a prediction into a measurement.

**prod_param lands on temporal-only, unanimously.** Fitting GTCNN eq.10's four coefficients
to held-out per-mode NLL (`scripts/analysis/fit_product_graph.py`) drives the **spatial
coupling to exactly 0 on all 6 datasets** (s01=0, s10=1, s11=0; resulting θ is spatially
flat to 1e-15). It beats every fixed form on fit (+0.74 to +0.88 nats vs iso; cartesian
+0.37 to +0.62). This is the constructive answer to the ambiguous mode-slope test above:
given four degrees of freedom spanning every product graph, the data chooses no spatial
structure. **Caveat carried forward: fit ≠ samples** — the heat campaign's better-fitting
arm lost at test.

**Gate.** `scripts/verify_product_graph.py` must print ALL CHECKS PASS or
`scripts/daic/pgraph.sh` refuses to submit. It checks the operator (A0–A4), the schedule
(B1, B3) and the forward/stationary law (C1 stationary variance, C2 forward reaches it from
real data, C3 terminal modes are uncorrelated — C3 catches a wrong eigenbasis, which a
variance match alone cannot). Permanent unit guards in `tests/test_product_graph_prior.py`.

## 2026-07-27 — product-vs-sum: NOT locked to product; spatial form is empirically moot

**Decision.** The pre-registered mode-variance slope test (rotate residuals to
`V_Sᵀ R V_T`, regress `log v[·,h]` on `log(λ_k+γ)` per temporal mode; product ⟺ slope
−1 every h, sum ⟺ shallow curved) **does not license rewriting the Cartesian-sum text
to a "it's a product" claim.** Rule was: ≥9/12 slopes in [−1.2, −0.8] AND product SSE <
sum SSE → lock product. **Met in 0 of 12 cells** (gwn + mlp × 6 datasets). Speed
residuals (metrla/pemsbay) are *steeper* than −1 (DC-dominated, up to −2.3); flow
residuals (pems04f/08f/04s) are far *shallower* (−0.1 to −0.6, some positive at rough
temporal modes → residual energy grows with graph frequency, anti-smooth). Both
parametric fits carry huge SSE (300–1830) and neither wins consistently. → **AMBIGUOUS,
supervisor discussion before any redesign** (the pre-registered branch).

**Why this is the right read, not a null result.** The diagnostic holds the temporal
factor fixed (free per-h intercept in BOTH product and sum fits), isolating the pure
*spatial* functional form. So it explains the audit memo's "empirical Σ_T⊗Σ_S beats
Cartesian in held-out NLL": that win is entirely the **empirical temporal factor** (the
per-horizon variances), which both fits get for free — NOT evidence that the *spatial*
precision is multiplicative. Once the temporal factor is controlled, the residual's
spatial variance barely tracks graph frequency at all (flow slopes ≈ 0), so the
product-vs-sum *precision* distinction is empirically moot on the residual. This sharpens
[[project_kronecker_audit_20260726]] Verdict 3 (temporal carries everything, spatial
near-inert) to the per-mode level. Write-up: do NOT sell a multiplicative spatio-temporal
precision; frame as empirical-temporal re-timing with a near-inert spatial term.
Figure: `results/analysis/paper_results/figs/mode_slope_diagnostic.png`; script
`scripts/analysis/mode_slope_diagnostic.py`; table `…/mode_slope_diagnostic.md`.

**Σ_S scale audit (rider).** `Σ_S=(L+γI)⁻¹` at the locked **γ=0.8** (NOT the 0.158/0.213/
0.924 the plan assumed — those don't exist in the code): `tr(Σ_S)/N` ≈ **0.59–0.66** on
every dataset, and `diag(Σ_S)` is nearly flat (**0.57–0.73**, max/min ≈ 1.2–1.3). At γ=0.8
the imposed spatial factor is within ~15–25% of a *scaled identity* — a second, independent
quantification of "near-inert". Adopt `Σ_S ← Σ_S/(tr Σ_S/N)` so σ owns the average per-mode
scale and `diag(Σ_S)` carries only (small) node heterogeneity; math unchanged (σ² already
absorbs the ~0.6). State this in Section 5.

**Form-mixing guard.** New permanent pytest `tests/test_prior_form_mixing.py`: the
rotate-scale-rotate sampler draws `N(0, σ²·Σ_T⊗Σ_S)` to **1%** relative Frobenius (2e5
samples, N=6/H=4), while the inversion-order variant `Q_T⊗Σ_S` misses by **76%**. The
inversion-order bug can no longer return silently.

## 2026-07-08 — training protocol: sample-based selection, EMA, loss demoted to dashboard

**Decision.** Selection, early stopping and the LR plateau key off **sampled
forecast CRPS** (fixed clean val windows, fixed noise seed, k=4, ancestral
reverse at 50 steps, computed every epoch on **EMA(0.999) weights**). The raw
denoising validation loss is logged as a dashboard only and drives nothing.
`best_crps.pt` (EMA weights) is the shipped model; `best.pt` (min val loss, raw
weights) is kept as an ablation/dashboard artifact. Patience 15 on sampled CRPS,
ceiling 200 epochs.

**Why.** Two independent lines of evidence:
1. *Analytic floor*: the denoising val loss is ~93% irreducible — the observed
   metrla plateaus (kron 1.023, iso 0.894) match the per-eigenmode Gaussian
   Wiener floor (1.020 / 0.927) computed from the prior spectra and the c̄
   sampler. A monitor that saturates at the floor cannot indicate convergence.
2. *Lineage audit* (repos pulled 2026-07-08): DiffSTG early-stops on sampled
   val MAE with a frozen seed (denoising loss never consulted); CSDI has no
   early stop at all (fixed 200 epochs, all-t deterministic val loss for
   checkpointing only); HySRD-Net uses OneCycleLR (fixed budget by construction);
   the EDM/CorrDiff/GenCast lineage uses fixed budgets with EMA weights as the
   model. **No reference early-stops on the raw denoising loss.** Our 2026-07-06/07
   runs stopped at ~25-30 epochs because GWN's early-stopping protocol was pointed
   at a saturated signal — the monitor stopped, not the model.

**Enabler.** The closed-form ancestral sampler (exact per-eigenmode OU posterior,
no EM stability floor) makes per-epoch sampled validation affordable (~10-25 s
vs ~7 min for an EM sweep). Sampled CRPS in raw units is also prior-agnostic,
resolving the kron-vs-iso loss incomparability at the selection level.

**Methodology untouched:** forward SDE, FCPS/ULS schedules, γ=0.8, ρ=0.1, S=7,
x0-prediction, UGnet, EM as the GAD-parity headline sampler for reported evals.

## 2026-07-08 — round-2 recipe additions (submitted as EXP=20260708_r2)

- **SNR loss default** (GAD's rate-weighted MSE): only arm whose sampled CRPS
  improved through training; best metrla full-val CRPS 0.0451 vs 0.0483 (mse).
- **Past-residual conditioning** (`--cond-past`): conditioning = [R_prev | Ŷ_z];
  the previous window's residuals are fully observed at forecast time (no
  leakage) and carry the volatility signal (HySRD-Net's only conditioning source).
- **Masked-input zeroing**: metrla holds residuals of ≈ −11σ at offline-sensor
  coordinates (7%); they entered the network input and graph convs smeared them
  into scored neighbors. Zeroed at input; loss mask unchanged; prior fit was
  already masked.
- **Per-node normalization arm** (`--norm pernode`): per-sensor train std
  (window-level residual std varies ~3×); HySRD's ablation shows global-vs-node
  normalization costs 5-7% CRPS. Run as an arm, not a default (changes the prior).
- **Ancestral sampler** (`--sampler ancestral`): DDPM-ancestral analogue, exact
  for the linear per-mode OU SDE (bridge identity verified numerically); no
  θ_max·S stability floor → ~50-100 steps replace 1500-3500 EM steps. Final-val
  cross-validates it against EM on every run (sweep + full-val, both checkpoints).
