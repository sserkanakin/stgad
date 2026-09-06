# Complete results inventory — 2026-08-13

Everything trained, everything measured, what it establishes, and what is usable.

**Totals on disk:** 1,177 trained diffusion runs · 993 saved checkpoints ·
694 locked-protocol test evaluations (+69 recalibrated / EM / non-default variants) ·
7 processed datasets (6 used; `molene` built but never run).

---

## 1. Trained campaigns

| # | campaign | runs | test evals | what it is | usable? |
|---|---|---:|---:|---|---|
| 1 | `20260721_ladder` | 728 | 525 | The main body of work. 5 prior arms (`iso`, `emp`, `rsh`, `emp_w`, `rsh_w`) × 6 datasets × 3 backbones × up to 5 seeds, under **forecast-only** and **prev-residual** conditioning; plus `ddpm` (VP-SDE) and `*-ugnet` (996k denoiser) sub-arms. | **fo: yes.** `-past`: retired conditioning, appendix only. `ddpm`/`ugnet`: never test-evaluated. |
| 2 | `20260728_histcond` | 80 | 80 | The **correct history-conditioning grid**: `iso`/`emp` × 6 datasets × 3 backbones × 2 seeds, config-parity verified against the ladder. | **Yes — this is now the thesis's main grid.** |
| 3 | `20260724_heat` | 136 | 80 | Cartesian product-graph **heat** priors — the only construction that stays a literal space⊕time heat equation. 8 arms: road-graph `cart`, learned-correlation-graph `cartL`, and refinements. | Partly. `cart-fo` valid; `cart-past` on the retired arm. |
| 4 | `20260721_emprefit` | 44 | 48 | Empirical prior refit at tuned γ, 4 datasets × mlp/gwn × 5 seeds. | Yes (fo). |
| 5 | `20260727_kron` | 48 | 0 | γ=0.2 + per-mode whitened loss across all 6 datasets, 3 backbones, + raw signal. | Val-only, single seed. Bounds, does not measure. |
| 6 | `20260724_rawsig_dh{4,8,32,6s}` | 41 | 0 | Raw-signal (not residual) arm at four denoiser capacities — the "best case for structure" control. | Val-only. |
| 7 | `20260728_pgraph_*` | 30 | 30 | GTCNN product graphs (Cartesian / strong / tensor / fitted-parametric) on the raw signal. | **No** — see §4. |
| 8 | `20260721_gammasweep`, `20260722_{gammaflat,rshshrink,tiny}` | 20 | 0 | γ ∈ {0.04, 0.1, 0.2, 0.4, 1.5, 3.0}, reshaped shrinkage, tiny-denoiser probes. | Val-only sweeps. |
| 9 | `model-ugnet-notcomplete` | — | — | DiffSTG re-run from its own released code, its split/sampler/DDIM-40/seed 43. | Yes, 5 of 6 datasets (PEMS-BAY died on quota). |

## 2. Offline analyses (no GPU)

| analysis | what it establishes | in thesis? |
|---|---|---|
| Residual structure table | 18 cells × 10 statistics: horizon growth up to 2.45×, non-stationary lag-1 correlation, graph roughness 0.42–0.94, DC share, spectral alignment, **Van Loan–Pitsianis separability 0.82–0.99**, excess kurtosis 0.8–87.6, joint stiffness 18–308 | ✅ Table 5.1 |
| Prior attribution (held-out NLL) | Nested family iso / Σ_T / Σ_S / Σ_T⊗Σ_S scored on test. **Temporal factor 0.11–1.43 nats/mode; graph factor ≤0.10 on top, negative in 8/18 cells** | ✅ Table 5.2 + Fig |
| Forward-process verification | GAD Fig. 1 & Fig. 2 reproduced on our Θ; eigenbasis exact to 1e-14; per-mode variance error 29%→2.1%; sliced-W1 → sampling floor; mixing condition 2Θ_min·c₀ = 11.2 ≥ 5 on all 6 datasets | ✅ Fig 5.2 |
| Kronecker separability audit | Covariance is separable (VLP 0.84–0.99); **temporal factor carries nearly all the value; imposed road-graph spatial factor near-inert** (mean eig 0.59–0.66, node-variance ratio ≤1.31 — within ¼ of a scaled identity) | ✅ §5.3.2 |
| Sampling cost | 1.7–3.4 ms per (window, sample) vs DiffSTG's 29–55 ms; prior overhead <1.1% | ✅ §5.6 |
| Spread recalibration | One validation-fitted scalar; coverage 0.53→0.78–0.90; fitted scale orders the arms by fit tightness (1.24 / 1.32 / 1.76) | ✅ §5.7 |
| Backbone accuracy | Point MAE/RMSE for all 3 backbones × 6 datasets | ✅ App. B |
| **Product-graph parametric fit** | GTCNN Eq. 10's four coefficients fitted on all 6 datasets and 2 clusters → **spatial coupling exactly 0 every time** (s₀₁=0, s₁₀=1, s₁₁=0). The data selects a temporal-only prior out of the full product-graph family. | ❌ **not in thesis** |
| **HySRD residual check** | Their iso-Gaussian residual claim refuted using their own recipe on PEMS04-flow: 73× spectral spread, kurtosis 21 — worse than our GWN residual | ❌ not in thesis |
| **Tails / distribution** | Per-dataset residual distribution + PIT + tail diagnostics | ❌ not in thesis |
| **Error localization** | Where in the network/horizon each arm's error concentrates | ❌ not in thesis |
| **Adaptive recalibration** | Time-of-day and per-mode recalibration variants; beat iso-best 9/12 | ❌ deliberately dropped (post-processing, not novelty) |
| Mode-slope diagnostic | 0/12 — spatial prior form empirically moot | ❌ not in thesis |
| Sampler-floor diagnostic (new, 2026-08-13) | Variance-weighted un-denoised fraction at the terminal floor: **3.15% iso vs 19.0–28.2% structured, grid-wide** | ⏳ pending R2 |

## 3. What the evidence establishes, ranked by strength

**Tier 1 — solid, multiply verified, test-grade.**

1. **The system beats published SOTA.** GWN + 17k denoiser vs DeepVAR / TransNVP /
   TimeGrad / GCRDD / DiffSTG / PriSTI / SpecSTG / HySRD-Net: ours takes 47 of 48
   columns across 4 benchmarks, 8–15% CRPS over the strongest. Corroborated by an
   independent re-run of DiffSTG from its own code on 5 datasets.
2. **At 1/58 of the generative parameters and 1/17 of the sampling cost**, taking
   *more* reverse steps.
3. **The construction is mathematically correct and numerically exact.** OU
   derivation, Lyapunov stationarity, Tweedie, Anderson reversal, ancestral
   posterior — all verified; eigenbasis exact to 1e-14; forward process reaches
   its designed stationary law to ~2% on real residuals.
4. **The residual field has the structure the prior assumes** — separability
   0.82–0.99, horizon growth, non-stationary correlation, graph smoothness — and
   every one of those properties **decays by an order of magnitude as the backbone
   improves**. Measured before any generative model is trained.
5. **The attribution is lopsided**: the estimated temporal factor carries the fit;
   the imposed graph factor is near-inert and sometimes negative.
6. **The conditioning ladder**: history conditioning worth 43% / 18% / 0.2% of CRPS
   down the three backbone rungs, unanimous across 6 datasets.

**Tier 2 — real but thin.**

7. **The structured prior beats isotropic on CRPS where the residual is
   structured** — every linear and every GCN cell of the four PEMS benchmarks,
   16/24 paired seeds, **p = 0.08**. Two seeds. Suggestive, not significant.
8. **It sharpens the conditional mean everywhere** — MAE 33/36, p = 1e-7. This one
   *is* significant.
9. **Capacity is a substitute for the prior** (+27–35% weak / +1% strong) — but
   validation-only and on the retired conditioning arm.
10. **The heat prior ≥ the covariance prior nearly everywhere**, with 6
    locked-protocol wins over iso — but mostly on the retired arm.

**Tier 3 — measured, unusable as evidence.**

11. Under-dispersion, Winkler and joint-dependence losses — **all three now
    suspected sampler artefacts** (§5 below).
12. Product graphs on the raw signal — budget skew 4–6× across arms, coverage 0.108.
13. Everything val-only: γ sweeps, raw-signal capacity ladder, DDPM, kron γ=0.2.

## 4. What is broken and why

- **The retired conditioning arm.** ~150 test-evaluated runs in `20260721_ladder/*-past`
  and most of `20260724_heat` trained `C = [R_prev|Ŷ]` while the thesis describes
  `C = [X|Ŷ]`. Legitimate as a design, but not the one documented. Fixed for the
  main grid; the heat campaign's headline wins are still on it.
- **The product-graph campaign.** Ran on the raw signal rather than the residual, so
  it does not test the object the thesis is about; training budgets differed 4–6×
  across arms and the two datasets where structure "wins" are exactly the two where
  it trained longest; ancestral coverage 0.108 means CRPS was effectively scoring a
  point forecast. Only its parametric fit survives.
- **The sampler terminal floor.** `c_lo = 0.02` leaves 3.15% of the isotropic
  prior's variance un-denoised and 19–28% of the structured prior's, in *every*
  cell. Measured effect of matching it to the training floor (8e-4) on pems08f:
  isotropic moves <1 point of coverage, structured gains 4.8–4.9 and overtakes the
  control on coverage, Winkler **and** energy while keeping its CRPS win.
- **Two seeds** on the main grid, against five on the retired one.

## 5. Is it enough?

**In volume, comfortably.** 1,177 trained models across 6 benchmarks, 3 backbone
capacities, 3 conditioning regimes, 8+ prior families, 2 denoiser capacities, 2
samplers, with an offline diagnostic suite that predicts the end-to-end outcome on
5 of 6 benchmarks. That is well beyond an MSc thesis's normal evidentiary base, and
the SOTA result is unambiguous and independently corroborated.

**In claim support, three gaps are load-bearing.**

1. **The sampler question decides three of four findings in §5.4.** Until the R2
   sweep lands, the thesis cannot say whether the structured prior under-disperses
   or whether the protocol under-resolves it. This is not a caveat to add — it is a
   result to determine. *Eval-only, ~30 GPU-h.*
2. **p = 0.08 is not a result.** Three more seeds moves the central claim from
   "suggestive" to conclusive. *~100 GPU-h.*
3. **The product-graph question has no experiment.** §4.2.3 argues in prose why the
   Kronecker product of precisions is used rather than a Cartesian product graph's
   Laplacian, and nothing tests it. The held-out NLL comparison answers it for free.
   *0 GPU.*

**Two strategic questions beyond the gaps.**

4. **The heat campaign is 136 trained models and the only structured prior with
   locked-protocol wins over isotropic — and it is entirely absent from the thesis.**
   On the valid forecast-only arm it beats the covariance prior on 9 of 12 cells and
   beats isotropic on 4 (best: pems08f/GWN +1.6%). It is also the more principled
   object: a genuine space⊕time heat equation, where the current prior is a Kronecker
   product of precisions that is *not* any product graph's Laplacian — a distinction
   the methodology already spends a paragraph on. Including it would turn that
   paragraph into a chapter section with data behind it.
5. **METR-LA is unexplained.** It is where the offline diagnostics are most
   optimistic and the end-to-end result is worst, on all three backbones. Two
   candidate explanations tested and killed. One unexplained benchmark out of six is
   acceptable in a thesis provided it is stated; it is currently stated.

**What would NOT improve the thesis:** more γ sweeps (24 cells already say the
verdict does not flip), re-running product graphs on the residual arm (~200 runs for
what the offline NLL answers), or a seventh dataset (`molene` is built and unused —
adding a non-traffic benchmark would broaden but not deepen, and the graph-prior
story is already told across 6).
