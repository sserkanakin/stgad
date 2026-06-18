# Implementation Log — Structured Residual Diffusion

What was built and **why**, for the validation + training pipeline. The plan is
`plan.md` (v2): a sequential spine of gated phases **A → B → C → D → E**, iso-vs-kron
built into each. This log covers Phases **A** and **B** (no training) plus the training
machinery they certified. Deep-dive companion: `notebooks/stage0_explained.md`.

The model: a frozen GraphWaveNet backbone produces forecasts; we model the *distribution
of its residuals* `R ∈ ℝ^{N×H}` with a graph-aware OU diffusion whose stationary prior is
the Kronecker covariance `Σ = Σ_T ⊗ Σ_S`. Everything runs in **unit-variance space**
(residuals ÷ σ_r, one global train scalar) and de-normalizes to raw units (× σ_r) for metrics.

---

## Phase A — Prior validation (no training)

**Goal:** prove `Σ = Σ_T ⊗ Σ_S` is valid, numerically sane, and *more informative than
isotropic* before any network is trained.

**New code**
- `src/eval/signal_features.py` — GAD's scalar feature reduction of a field `R`: spatial
  TV, spectral centroid, degree correlation, lag-1 autocorr, horizon centroid, plus
  `compare_feature_sets` (1D-W1 / normalized-W1 / RBF-MMD / KS). *Why:* you cannot compare
  two distributions over 2484 dims directly; reduce each field to interpretable scalars and
  compare 1-D clouds (GAD's approach). Reused in Phase D.
- `src/eval/prior_diagnostics.py` — `psd_checks` (6-check validity gate) and
  `conditioning_table` (κ, derived `c₀=5/θ_min`). *Why:* separate "is it a valid covariance"
  (hard gate) from "how stiff is it" (explains the schedule).
- `src/data/archive_import.py` — `ensure_residuals` bootstraps the pre-rewrite GWN residuals
  into the current layout. *Why:* the residual *statistics* the prior is tested against are
  robust to the exact backbone checkpoint (verified: identical windowing, structure present),
  so we don't retrain the backbone just to validate the prior.
- `notebooks/stage0_prior_{metrla,pems08f,pemsbay}.ipynb` (+ `stage0_lib.py`, generator
  `_build_stage0_notebooks.py`).

**Result:** all three datasets **PASS**. kron beats iso on feature-distribution distance
(mean KS metrla 0.51 vs 0.68, pems08f 0.34 vs 0.66, pemsbay 0.42 vs 0.75), dominating the
spatial features. Spatial structure is real but weak (1-hop residual corr ≈ 0.16, decays in
2 hops — GWN already absorbed most of it); temporal structure is strong (`R_H` lag-1
0.41–0.84). Both verified empirically.

**The stiffness finding (`A2`).** The Kronecker precision is *stiff*: `κ(Q) = θ_max/θ_min ≈
311` (metrla), because we stack a temporal factor (κ_T≈26) on a small-γ spatial factor
(κ_S≈12). This forces a large noising budget `c₀ ≈ 170`, vs GAD's `S=7` (GAD is spatial-only,
κ≈3.5). `scripts/analysis/conditioning_tradeoff.py` shows the budget is *tunable*: a
`stiffness_cap ≤ 13` reaches GAD's `c₀≈7` while **still beating iso** (the win lives in the
eigenvectors + bulk spectrum shape, the stiffness in a few extreme eigenvalues — the cap hits
one and not the other). Cap deferred; it has a small calibration cost (it tightens an already
under-covered dominant mode) that only Phase E can price.

---

## Phase B — Forward / reverse machinery (no training)

**Goal:** prove every moving part is correct before spending GPU hours.

**New code:** `notebooks/stage1_machinery_{ds}.ipynb` (+ `stage1_lib.py`, generator),
`src/data/adjacency.py::connectivity`, `scripts/analysis/structure_and_connectivity.py`.

**Verified (all 3 datasets):**
- **Forward** marginal `H(c̄)=e^{−c̄θ}`, `σ²(c̄)=(1−e^{−2c̄θ})/θ` matches empirical to machine
  precision.
- **Reverse** *exact*: an oracle denoiser (returns true R0) traces the analytic forward
  trajectory to ~1e-6. The only gap to clean R0 is the deliberate ~4% noise floor at `c̄_lo`.
- **Normalization** exact: the prior's σ_r equals the trainer's σ_r, unit-variance = 1.000,
  raw↔unit round-trip lossless.
- **Connectivity** (new check — the old code only repaired degree-0 nodes, which can't rule
  out a multi-node island): a BFS/DFS graph search + Laplacian spectrum confirm one connected
  component (pemsbay's *raw* graph splits into 7 pieces — all singletons, so the repair
  sufficed, but only the check proves it). With one component + γ>0 every mode mixes; the
  slowest retains <1% signal at `c̄_hi`.

---

## The training fix — noise-level (c̄) parameterization

**The bug (Phase B, training-readiness):** training sampled `t ~ U(0,T)` and the reverse used
a uniform-in-`t` grid. With the steep `α=8` schedule, that crushes the noise level near 0:
**53% of training samples had `c̄ < 1`** (≈ clean, nothing to denoise), and the loose modes —
including the dominant one the prior starts from — were in their learnable transition only
~18% of the time. The reverse took giant `c̄` jumps at the high-noise end. The denoiser would
barely learn exactly the modes generation starts from → under-dispersion.

**The fix (EDM-standard):** parameterize everything by the **noise level `c̄`**, not `t`.
- **Training:** sample `c̄` **log-uniform** in `[c̄_lo, c̄_hi]` — even coverage of *every*
  mode's transition (uniform-t misses the middle; uniform-`c̄` over-corrects for our stiff
  prior, dropping stiff-mode coverage to 0.3%; **log-uniform** gives balanced ~32% across all
  modes). `c̄_hi = 5/θ_min` (mix the slowest mode), `c̄_lo` set for a clean ~4% final noise floor.
- **Reverse:** step a **geometric `c̄` grid** — even in log-noise, no giant jumps.
- **Conditioning:** the denoiser conditions on normalized `log c̄` (tau ∈ [0,1]) instead of raw
  `t` — even resolution across the noise range, *zero* architecture change.

**Code:** `src/diffusion/forward_fcps.py` (`noise_level_bounds`, `sample_log_c_bar_torch`,
`c_bar_to_tau`, `sample_R_t_from_c_bar(_torch)`, `reverse_c_bar_grid`), `reverse.py` (geometric
grid + tau), `train_diffusion.py` (train_step / validate / val diagnostic).
*Consequence:* checkpoints trained on the old `t`-conditioning are invalid — retrain from scratch.

---

## Losses

`src/diffusion/losses.py` (GAD-style functional). Three ways to score `(R̂₀ − R₀)`:
- **`snr` (training default)** — SNR-weighted MSE, `w(c̄)=1/(1+2·v̄)` (GAD's form, `v̄` = mean
  per-mode noise variance). *Why:* plain MSE is dominated by high-noise samples where the best
  possible error ≈ the full signal variance and there's nothing to learn. The weight (≈1 at low
  noise, ~0.34 at full noise) rebalances so every level contributes comparably. The downweight
  is deliberately *gentle* — an aggressive →0 weight would gut the loose modes' high-`c̄` training.
- **`mahalanobis` (comparison metric)** — error standardized per mode by its noise variance `σ²(c̄)`.
  *Why:* kron and iso have different noise geometries, so their plain-MSE losses aren't comparable;
  standardizing puts them in the same units. **This resolves the "iso wins val / kron wins test"
  paradox** (the val loss was never cross-prior comparable). It is a logged val metric, not the
  training objective (its `1/σ²` weight blows up at low noise); available as a training ablation via
  `--loss mahalanobis`.
- **`mse`** — the plain baseline.

`train_diffusion.py` gets `--loss {snr,mse,mahalanobis}` (default `snr`).

---

## Validation diagnostic

Consolidated four overlapping eval functions (+10 flags) into **one** `val_diagnostic` (+4
`--val-*` flags). Every N epochs, on fully-observed val windows: noise them **close to full
(`c̄_hi`)**, run the reverse SDE back, and report **recon MSE/MAE (raw units, per horizon) +
Wasserstein (sliced-W1 + energy) + the Mahalanobis metric**. Fully-observed windows only, so
the node vectors and the eigenbasis rotation carry no masked garbage. *Why this shape:* it is
Phase C2 — does the trained denoiser+sampler invert a near-full perturbation — plus a
distributional check, in one pass, with far fewer knobs.

---

## Running it (DAIC)

Single entry `scripts/run.sh` (`profile` / `train` / `all`) hides the env-var soup; the
diffusion sbatch now **profiles GPU/CPU/RAM** (peak GPU mem inline + `seff`/`sacct` hints) so a
short production-config run sizes the production resource requests. Full runbook:
`scripts/daic/README.md`.

---

## Status

| Phase | state |
|---|---|
| A — Prior | ✅ PASS (notebooks) |
| B — Forward/reverse machinery | ✅ PASS (notebooks); forward & reverse proven exact |
| training pipeline (loss, noise sampling, normalization) | ✅ verified, smoke-trains, loss drops |
| C — Denoiser / D — Generative / E — Forecast skill | ⬜ need training runs |
| Branch T (Student-t tails) | queued — the headline contribution, after the Gaussian spine |

**Open methodology items:** the stiffness cap (optional, Phase-E-priced) and the Mahalanobis
*training* ablation. Everything else from `plan.md`'s training settings is in (snr loss,
log-uniform-c̄ sampling, no TV regulariser).
