# Hyperparameter audit — everything that touches the prior, the heat process, and training

A complete inventory so nothing surprises us late. For each constant: its value, where it
lives, who chose it (GAD-literal / our choice / derived-from-data), **what it affects**,
**which arms it touches**, and whether it could be questioned. Verified against the code
(`src/diffusion/prior.py`, `src/diffusion/forward_fcps.py`, `scripts/train_diffusion.py`).

The one structural fact that makes the comparison clean: **every knob is either (a) GAD-literal,
(b) derived from the data, or (c) shared identically across all arms.** The only two knobs that
change a prior's *shape* — γ and ρ — touch **empirical only**; `iso` and `reshaped` are free of
both. So no arm gets an unfair schedule or training advantage.

---

## Group 1 — Prior shape (the heat operator Σ = Σ_T ⊗ Σ_S)

| knob | value | source | affects | scope | risk |
|---|---|---|---|---|---|
| **γ** spatial ridge `Σ_S=(L+γI)⁻¹` | **0.8 → 0.2** | GAD → swept | spatial variance profile | **empirical only** | **RESOLVED — sweep confirmed 0.8 under-commits (~21% on mlp), plateau γ≤0.2; empirical retrained at γ=0.2 (20260721_emprefit); reshaped still wins** |
| **ρ = SHRINK_T** temporal ridge `μ_T←(1-ρ)μ_T+ρμ̄` | **0.1** | **our choice** | temporal variance profile | **empirical only** | **MEDIUM — the temporal twin of γ, never questioned** |
| gamma_t temporal path-graph | 0.6 | our choice | temporal Laplacian | only `temporal=laplacian/cartesian` — **NOT emp/rsh** | LOW (unused by our arms) |
| spatial | "structured" | our design | graph vs flat Σ_S | kronecker | it *is* the contribution (ablated by "flat") |
| spatial_graph | "road" | data-fit | which graph | kronecker | LOW–MED (flow sets might prefer kNN-corr) |
| knn_k | 15 | our choice | only if `spatial_graph=correlation` | n/a for road | LOW |
| repair | True | our choice | connect graph components | kronecker | LOW (γ already lifts zero modes) |
| stiffness_cap | **OFF** | deliberate | floors θ_min | none | LOW (off on purpose — capping truncates the loud modes rsh matches) |
| winsor | **OFF** | — | robust correlation fit | off in reshape exp | LOW |
| **norm** | **pernode** | our choice | the residual *space* the prior is fit in | **all arms (data + prior)** | **MEDIUM — load-bearing, but shared** |
| τ², D_H, R_H, v_emp | derived | data | — | — | none (computed, not chosen) |

**Key point:** `reshaped` uses **neither** γ nor ρ — it measures the per-mode variance directly
(`θ=1/v_emp`), so both formula-knobs are bypassed. `empirical` carries both. That means the
γ-sweep should really be a **(γ, ρ) question for empirical** — but it does not threaten the
`reshaped` headline at all.

---

## Group 2 — Forward / heat schedule (how noise is added: `c(t)`, `c̄`, and GAD's `S`)

Forward SDE: `dr = -c(t) Q r dt + √(2c(t)) dW`. Per mode, signal `H=e^{-c̄θ}`, noise
`σ²=(1-e^{-2c̄θ})/θ`. The "`c`" and "`S`" you asked about live here:

| knob | value | source | affects | scope | risk |
|---|---|---|---|---|---|
| **c_0 (= GAD's S)** total heat `c̄(T)` | **7.0** | **GAD-literal** (exp_gasde.py) | slowest-mode mixing | all (via θ_min→0.8) | LOW — reproduces GAD's mixing `2·S·θ_min≈11.2` |
| **c_min** FCPS floor | **0.1** | GAD-literal | schedule floor | kronecker (fcps) | LOW |
| **α** FCPS polynomial order | **4.0** | GAD-literal | schedule curvature | kronecker (fcps) | LOW |
| **_SNR_HI** mixing target | **5.0** | GAD criterion | `c̄_hi = 5/θ_min` | all | LOW (GAD's `2c̄θ_min≳5`) |
| **_NOISE_FLOOR** | **0.04** | our choice | `c̄_lo = 0.04²/2 ≈ 8e-4` (final sample ~4% noise) | all | LOW–MED |
| θ_min→0.8 normalize anchor | 0.8 | = GAMMA_DEFAULT | schedule anchor `c_0≈5/0.8` | **all arms** | LOW (the "other 0.8"; a scale convention, not a shape) |
| schedule | **fcps** | our choice | schedule family | **all reshape arms incl. iso** | LOW — matched across arms, **no confound** |

**Note on iso's schedule:** the module *default* would give iso a linear ULS schedule (slope=14
= 2·c_0, heat-matched). But the reshape experiment forces `--schedule fcps` on **every** arm
including iso, so iso and the kronecker arms share the identical schedule — the schedule is not a
confound in our numbers. (The `ddpm` arm alone uses the VP/DDPM schedule + power-law sampler — it
is the textbook control, deliberately not a matched ablation.)

---

## Group 2b — Do GAD's (spatial-only) schedule constants cover OUR joint spectrum?

GAD tuned `γ, S, c_min, α` for a **single spatial graph** whose Laplacian spans ~3.5×. Our prior
is the **Kronecker product** of a spatial and a temporal spectrum — far stiffer (measured joint
range up to 258×). So: does GAD's schedule still *cover* it? A mode is trained only if its
transition `c̄ ≈ 1/θ` lies inside `[c̄_lo, c̄_hi]`. Checked on pems04f/gwn:

| arm | stiffness θ_max/θ_min | % modes below floor | % above ceiling |
|---|--:|--:|--:|
| iso | 1× | 0.0 | 0.0 |
| empirical γ=0.8 | 21× | 0.0 | 0.0 |
| empirical γ=0.1 | 124× | 0.0 | 0.0 |
| empirical γ=0.04 | 300× | 0.0 | 0.0 |
| reshaped | 258× | 0.0 | 0.0 |

**Verdict: the schedule covers 100% of the joint spectrum, even at 300× stiffness.** GAD's
`S/c_min/α` + our `θ_min→0.8` normalization + the fixed `0.04` floor are wide enough. No mode is
left un-trained. Reproduce with `scripts/analysis/params.py`.

Why it transfers despite being "spatial-only":
- **`S = c_0 = 7` self-corrects.** `θ_min` is pinned to 0.8 (= GAD's spatial `θ_min`), so
  `2·S·θ_min = 11.2` reproduces GAD's mixing regardless of our joint stiffness.
- **The noise sampler was already adapted** — log-uniform-in-`c̄` (not GAD's power-law) exactly so
  a wide joint spectrum is covered evenly.
- **`ρ = 0.1` was already added** as the temporal ridge GAD never had.
- **Still at GAD-spatial values, un-re-derived:** `γ` (→ being swept) and the FCPS curvature `α=4`
  (covers fine, but not verified *optimal* for a 258×-wide spectrum — a secondary sensitivity
  check, not a correctness issue).

---

## Group 3 — Training loss & noise sampling

| knob | value | source | affects | scope | risk |
|---|---|---|---|---|---|
| loss | **snr** = GAD rate-weighted MSE, `w = 1/(1+2σ²c(t))` | GAD | gradient weighting | all (shared) | LOW (tested vs whitened=worse, mse) |
| **noise-sampler** | **loguniform** | **our choice — deviates from GAD's power-law** | which `c̄` training sees | all kronecker | **MEDIUM — a deliberate GAD deviation** (even coverage for stiff priors); shared, so fair |
| _GAD_SAMPLE_POWER | 0.65 | GAD | only if power-law | `ddpm` arm | LOW |

---

## Group 4 — Optimization & denoiser

| knob | value | source | risk |
|---|---|---|---|
| epochs | 200 | our | LOW |
| lr | 1e-4 | our | LOW–MED (an lr-find exists but unused) |
| batch_size | 32 | our | LOW |
| weight_decay | 1e-4 | our | LOW |
| patience | 40 | our (raised from 15 so early-stop luck doesn't truncate the whitened arms) | LOW |
| ema | 0.999 | our (best_crps.pt = EMA weights, the shipped model) | LOW |
| lr-factor / lr-patience | 0.5 / 5 | our (plateau scheduler) | LOW |
| denoiser hidden / K / n_layers | 24 / 5 / 3 → **17,305 params** | our | LOW (the capacity axis tests exactly this) |
| val selection | best_crps.pt via ancestral CRPS, k=4, 50 steps | our | LOW |
| residual-seed | 42 (frozen backbone for all stage-2 seeds) | our design | GOOD — cancels backbone variance in the paired comparison |

---

## What could actually bite us — the short list

1. **ρ = SHRINK_T = 0.1 is the un-examined temporal twin of γ.** It shrinks the empirical
   *temporal* covariance toward isotropic by 10%, exactly as γ shrinks the spatial one. It was our
   own invention (no GAD basis — tuned to tame pems-bay's stiffest temporal mode). If we sweep γ
   for empirical, we should at least sanity-check ρ too, or state clearly that empirical is
   reported at (γ=0.8, ρ=0.1) and reshaped bypasses both. **This is the most likely "why is
   empirical configured that way?" question.**

2. **noise-sampler = loguniform deviates from GAD's power-law.** Well-motivated (stiff priors need
   even noise-level coverage) and shared across arms, so it's fair — but a GAD-literalist reviewer
   will notice. Worth one sentence in the methodology.

3. **norm = pernode** sets the entire space the prior is fit in. Shared across arms (not a
   confound), but it is load-bearing — the 3.5×/24.6× spreads and every CRPS number are in
   per-node-normalized units.

4. **c_0 = S = 7, c_min = 0.1, α = 4 are GAD-literal and fixed.** Defensible and heat-matched
   across arms — not confounds. State they're taken verbatim from GAD (exp_gasde.py), not tuned.

**Everything else** (τ², D_H, R_H, v_emp) is derived from data, and the optimization knobs are
standard and shared. The comparison iso vs empirical vs reshaped is clean: no arm gets a
different schedule, loss, optimizer, or data space — only the prior changes, and only empirical
carries the (γ, ρ) formula-knobs.
