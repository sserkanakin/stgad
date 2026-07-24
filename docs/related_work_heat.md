# Related work map — heat-equation priors, product graphs, learned priors (researched 2026-07-24)

## NOVELTY VERDICT (composite claim: UNCLAIMED)
"Cartesian space-time product Laplacian as diffusion forward operator, with the spatial
factor LEARNED from forecast-residual correlations, γ by spectral matching, applied to
predict-refine residual diffusion in traffic forecasting."
Every ingredient is precedented; the COMPOSITION is not. Claim the composition. Three
defensible axes: (1) learned-from-residuals graph as forward operator (graph learning and
diffusion priors have never been composed); (2) spatial→ST extension of GAD via the
Cartesian product (CITRUS uses the same operator discriminatively; nobody generatively);
(3) γ selection by spectral matching to the residual spectrum (no precedent found).

## Mandatory differentiations
- GAD (2510.05036): given graph, spatial-only, raw signals. We: learned graph, space-time, residuals.
- Double-Diffusion (2506.23053): graph heat as MEAN prior/warm-start, iso noise. We: heat as noise COVARIANCE.
- CoST (2502.11013) / NPDiff (2501.13794): predict-refine residual diffusion with scale/dynamics noise, no correlated covariance.
- DR / Zheng-Sun (2301.06650): learned residual Kronecker covariance but linear-Gaussian AR, no diffusion.
- CITRUS (2405.18877): heat kernels on Cartesian product graphs — discriminative GNN, not generative.
- Kadambari-Chepuri (2012.08090) / Shi-Mishne (2402.08105): LEARN Cartesian product Laplacians from data — never used generatively.
- SpecSTG (2401.08119): GFT-domain diffusion = implicit vertex-correlated noise (subtle; note it).
- Whitened Score (2505.10311): general anisotropic-Gaussian diffusion training theory — cite for correctness.
- PriorGrad→RestoreGrad→NsDiff/TSFlow: adaptive prior lineage (diagonal/GP/jointly-learned; none graph-structured).
- IHDM (2206.13397) + Blurring Diffusion (2209.05557): grid-Laplacian heat forward; Blurring = "heat = anisotropic Gaussian in operator eigenbasis" — our exact lens; IHDM's σ²I floor = citable justification for γI, and their "pick dissipation time by matching a data statistic" = precedent for our spectral-matching methodology.
- Jia-Benson (2002.08274): GNN residuals are graph-correlated GMRFs (non-generative) — motivates the residual graph.
- Smola-Kondor 2003: (I+σ²L)⁻¹ as principled graph GP covariance — classical justification of our Σ.
- Grassi et al. time-vertex framework (1705.02307) + Sandryhaila-Moura 2014: canonical product-graph justification.

## Cheap tricks worth adopting (in priority order)
1. CITRUS separability: exp(−tL_⊞)=exp(−tL_T)⊗exp(−tL_S) — per-factor computation, never materialize NT×NT.
2. Double-Diffusion warm start: begin reverse at an intermediate step (residual prior close to target) — free speedup.
3. Blurring-Diffusion per-eigenvalue schedule framing (free knob; clean closed forms).
4. Whitened Score objective if stiff-prior training destabilizes.
5. NsDiff-style diagonal×structure factorization (matches our adaptive-variance finding).
6. RestoreGrad joint prior fine-tune as an ablation.

(Full tables with links in the research agent report; baseline tier: DiffSTG 2301.13629,
SpecSTG 2401.08119, ProGen 2411.01267; survey to sweep at write-up: 2507.14507.)
