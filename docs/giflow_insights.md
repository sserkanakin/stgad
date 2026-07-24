# GiFlow (ICLR 2026 under review) vs our thesis — what it means for our drift

**GiFlow = Graph-Informed Flow Matching for SpatioTemporal Imputation.** Same Cartesian
space⊕time product-graph heat operator as us, same PeMS04/08 benchmarks — but it uses the
heat operator on the OTHER side of the generative process, and that difference is the most
useful thing in the paper for us.

## The exact correspondence (and the one crucial difference)

| | GiFlow | Our thesis |
|---|---|---|
| ST operator | Kronecker sum `L_{ηξ} = τ_ξ L_ξ ⊕ τ_η L_η` (their Eq. after 174) | `L_ST = ½(L_T ⊕ L_S)` — same object, equal weights |
| where heat enters | the **source distribution / mean**: filtered observation `X_τ = e^{-τ_η L_η} X_1^M e^{-τ_ξ L_ξ}` is the FM start point | the **noise covariance**: `Σ = σ²(L_ST+γI)⁻¹` is the diffusion stationary law |
| generative process | flow matching (deterministic ODE, few-step, direct regression) | diffusion SDE (ancestral/EM) |
| factor weighting | learned `τ_η, τ_ξ` (their Eq. 6, SGD) | `γ` by criterion; we already split `γ=0.05` vs `γ_t=0.6` (anisotropic — GiFlow validates this) |
| their key result | **Theorem 1: graph-informed source REDUCES transport cost** vs Gaussian → structure HELPS | **Wiener bound: graph-informed noise covariance RAISES optimal-denoiser error** → structure HURTS at capacity |
| task | imputation (mean intact) | forecasting residuals (mean removed) |

## The insight that unifies our whole result

Heat structure on the same Cartesian graph gives **opposite outcomes depending on WHERE in
the generative process it lives**:

- **Heat in the source/mean (warm start)** → shortens the generative path, aligns start to
  target → **helps**. Proven by GiFlow's transport theorem; also Double-Diffusion (heat as
  mean anchor) and every warm-start method.
- **Heat in the corruption noise covariance** → puts the noise in the signal's own subspace,
  making the denoising problem intrinsically harder → **hurts at capacity**. Proven by our
  Wiener bound; measured by our raw-signal control.

This is not two unrelated facts — it is one dichotomy, and GiFlow is independent, concurrent,
top-venue confirmation of the *helping* side, exactly complementary to our proof of the
*hurting* side. **Our modest/negative margins stop being a weakness and become half of a
complete characterization: heat belongs in the mean, not in the corruption covariance.**

## Why our design lands on the hard side (the deep point for the drift)

We **residualize**: the frozen backbone removes the conditional mean optimally, so by the time
the diffusion runs, the object GiFlow's source-filtering would help with (the signal mean) is
already gone. What remains — the residual's second-order structure — is exactly the covariance,
i.e. the Wiener-bad side. So our two headline design choices are in tension:

> residualization strips the mean (the part heat helps most), leaving only the covariance
> (the part heat provably struggles with).

That is why the road-heat prior (γ=0.8) barely moves CRPS, why even the raw-signal best-case
loses, and why only the aggressive learned-heat covariance recovers small gains. The
characterization is airtight and now doubly-sourced.

## Concrete recommendations

1. **Reframe the discussion around the source-vs-covariance dichotomy** (edit made to
   `6_discussion.tex` §mechanism). Cite GiFlow as concurrent confirmation of the source side.
   This converts the result from "structured noise mostly ties" to "we locate structure's
   value precisely, and concurrent SOTA agrees." Strong defense.
2. **Related work**: add GiFlow to the heat-lineage / structured-forward paragraph, positioned
   as *heat-as-source-for-flow-matching* — the dual of our *heat-as-covariance-for-diffusion*
   on the identical operator. Sharpens our novelty boundary (nobody uses the Cartesian ST heat
   operator as a diffusion forward covariance on measured residuals).
3. **Anisotropic weighting**: we already use `γ≠γ_t` — say so explicitly and cite GiFlow's
   learned `τ_η,τ_ξ` as the same principle. Cheap credibility.
4. **Future work (the real payoff of the drift question)**: a **flow-matching refiner whose
   heat-informed source is the (heat-filtered) backbone forecast** — puts heat on the *helping*
   side, sidesteps the EM-stiffness problem (deterministic few-step), and unifies our result
   with GiFlow/Double-Diffusion. This is the principled "make the drift work" answer.

## Citation caveat
GiFlow is an anonymous double-blind ICLR 2026 submission — no public author list yet. Cite as
concurrent/under-review work, or hold until it de-anonymizes. The framing edits are written to
stand on our existing Wiener bound + Double-Diffusion even if GiFlow is not formally cited.
