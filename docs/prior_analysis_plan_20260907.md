# The prior chapter — analysis plan (2026-09-07)

Goal: every claim about the prior is backed by a measured quantity on the fresh residuals,
shown as a figure, with the obvious objection answered next to it. Configuration and derived
numbers go to the appendix. Five questions, in the order the chapter argues them:

1. What does the residual field actually look like?           (§A)
2. How close is the prior to it — and where is it off?         (§B)
3. What does the prior change, concretely?                     (§C)
4. Does the forward process behave like GAD says it should?    (§D)
5. Does the structure survive the trained denoiser?            (§E)

All of §A–§D need only the fresh residual caches + graphs → Mac, CPU, now.
§E needs saved checkpoints (17k params, pulled as cells finish) → Mac, CPU, subset of windows.
Everything writes JSON under `results/analysis/prior/` and the figures are built from those.

---

## §A — the residual field (facts, no model)

| # | question | computation (per dataset, unit space, train & test) | figure | claim it supports |
|---|---|---|---|---|
| A1 | how is error spread over horizons? | δ_h = mean r²_h; R_H | horizon variance profile + R_H heatmap | error grows 1.7–2.9× over the hour and adjacent horizons correlate ≈0.9 → a temporal factor is needed |
| A2 | do neighbours err together? | mean corr. by graph hop distance 1,2,3,…,non-neighbour; scatter corr vs road distance | correlation-vs-hops curve | spatial correlation is real but weak and short-ranged (0.14 at 1 hop, 0.05 beyond) |
| A3 | is there a shared component? | share of variance on the constant eigenvector u=1/√N; first spatial PC of the residual and its overlap with u | bar: share data vs 1/N; PC1 vs constant | 3–21 % of the variance is "everyone together", 5–35× what independence allows |
| A4 | the network total | sd(Σ_i R_ih) per horizon vs √Σσ_i² | total-sd curve with the independence line | the total's uncertainty is 2–3× what independent errors give — the quantity iso cannot get right |
| A5 | tails | kurtosis per sensor, QQ vs Gaussian | QQ plot, kurtosis histogram | the marginals are heavy-tailed: the Gaussian prior is a *noise process*, not a model of the residual; the denoiser has to supply the tails (checked in §E4) |
| A6 | stationarity | train/val/test variance ratio per horizon, per sensor | ratio plot | the fit on train transfers (or where it does not — pems-bay) |

## §B — closeness of the prior to the field

| # | question | computation | figure | claim / defence |
|---|---|---|---|---|
| B1 | graph-spectral fit (the GAD view) | variance of the residual per graph frequency: v_k = mean over windows,h of (V_Sᵀ r)²_kh, against the prior's σ²/θ̄_k (h-averaged) — for iso (flat), road-only γ=0.8, matched | **variance vs λ_k, log scale, data vs the three priors** | the prior's spectrum follows the data's over the low and mid frequencies; the road-only prior misses the λ=0 point by 5–10×; iso misses the whole slope. *Objection "fitted on train" → same plot on test.* |
| B2 | joint mode fit | data variance per (k,h) mode vs prior σ²/θ_kh; Spearman rank correlation across all NH modes; ratio distribution | scatter (prior var vs data var) + rank-corr per dataset | the prior orders the modes correctly (rank corr ≈ 0.9); ratios within ×2 for the modes carrying 95 % of the variance |
| B3 | separability | nearest Kronecker product of the empirical covariance (Van Loan–Pitsianis): fraction of ‖C‖_F explained by the best rank-1 Kronecker term | bar per dataset | Σ_T⊗Σ_S loses ≤ X % of the covariance — the Kronecker restriction is justified. *Objection "learn the full Σ" → NH=3684, 6.8 M entries from 10k windows, and it is not a graph.* |
| B4 | multi-scale variance | ratio prior-implied / measured sd of sums over: 1 sensor, 1-hop neighbourhoods, spectral clusters of the graph (k = 4, 16, 64), the whole network; for iso, road-only, matched | **ratio vs aggregation scale, one line per prior** | iso is right at scale 1 and collapses to 0.4 at the network; road-only holds to the neighbourhood and collapses at the network; matched stays ≈1 at every scale. This is the mechanism of the joint win, at the prior level. *Objection "you matched the total" → the neighbourhood and cluster scales were never matched.* |
| B5 | held-out likelihood | Gaussian NLL per point on test under iso / temporal-only / spatial-only / road-only / matched | table + bar | the structured prior is closer in likelihood on every dataset; the global edge costs nothing in NLL (likelihood is blind to one mode) |
| B6 | whitening test | Σ^{-1/2} r: variance per mode after whitening should be flat=1 | flatness (spectral entropy) before/after, per prior | the matched prior whitens the residual best; what remains is the non-Gaussian part |
| B7 | robustness of (β, γ) | the match surface ratio(β, γ) with the neighbour-correlation contour; ratio on TEST at the train-chosen pair | heatmap + test ratio | one pair, chosen on train by one moment, generalises (test ratio within 5 %); shows why two knobs |

## §C — what the prior changes (before any training)

| # | question | computation | figure | claim |
|---|---|---|---|---|
| C1 | what does a draw look like? | one draw from N(0,Σ) for iso / road-only / matched, and a real residual window, on the sensor map (sensor colour = error) | 4 maps per dataset | kron draws have the patches and the common shift of real errors; iso draws are salt-and-pepper |
| C2 | implied distributions of derived quantities | prior vs data histograms of: the network total, neighbour differences r_i − r_j, horizon increments r_{h+1} − r_h | 3 panels | the prior gets the *scales* of these right (or over-smooths neighbour differences — say so) |
| C3 | the θ spectrum | histogram of θ_kh (kron) vs the single θ=0.8 (iso); which modes transition where (c̄·θ ≈ 1) | θ histogram with the c̄ grid overlaid | why the floor and the step count are prior-dependent; why the old c_lo=0.02 left 43–62 % un-denoised |

## §D — the forward process, GAD-style

| # | question | computation | figure | claim |
|---|---|---|---|---|
| D1 | frequencies going down | a real window forward-noised at c̄ ∈ {1e-3, 1e-2, 0.1, 1, 7}: energy per graph-frequency band (and per temporal mode), normalised, iso vs kron | **heatmap (λ band × c̄)** for both arms | kron erases high frequencies first and the smooth/common modes last (the GAD picture); iso erases everything at once |
| D2 | convergence to the prior | GAD's signal metrics (total variation, spectral content, smoothness) of the noised field vs c̄, with the prior's own value as the target (`src/eval/forward_analysis.py`) | metric-vs-c̄ curves | the forward SDE reaches the stationary prior by c̄ = 7 on all six datasets |
| D3 | fidelity at the floor | f(c̄) = un-denoised fraction, both arms, with c_lo(iso)=8e-4, c_lo(kron), and the old 0.02 marked | f(c̄) curves | both arms are trained/sampled to the same 0.13 %; the old protocol was 15–450× less faithful for kron |
| D4 | a window through the whole process | one window: forward to c̄=7 (kron), then reverse with the trained denoiser, shown at the same c̄ levels — the field, its graph spectrum, the network total | filmstrip | the reverse creates the common mode first and detail last, mirroring D1 |

## §E — does the structure survive the trained denoiser? (checkpoints, CPU subset)

| # | question | computation | figure | claim / defence |
|---|---|---|---|---|
| E1 | spectrum of the samples | B1's plot for the *generated samples* (k×B windows): data vs prior vs iso-samples vs kron-samples | variance vs λ_k, four lines | closes the loop: prior right at low frequencies → samples right there; iso samples flat |
| E2 | multi-scale variance of the samples | B4 on the samples | ratio vs scale, iso vs kron vs prior | the joint win in the results chapter is exactly this curve; where kron falls short of the prior line (0.6 at the network scale) is the denoiser's doing |
| E3 | where along the chain the spread is lost | `chain_common_mode.py`: total spread per noise level; learned / oracle / hybrid denoiser | spread-vs-c̄ curves | sampler is exact (oracle → 1.0); the loss is the net's treatment of the common mode (hybrid recovers it) — or not; either answer is reportable |
| E4 | tails of the samples | kurtosis / QQ of sample residuals vs data | QQ | the denoiser supplies the tails the Gaussian prior lacks (A5's answer) |
| E5 | example fields | one real window: truth residual, iso sample, kron sample on the map | 3 maps | qualitative twin of C1 after training |

## Ablations of the prior's ingredients

Prior level (free, CPU): iso · temporal-only (Σ_T⊗I) · spatial-only (I⊗Σ_S) · road-only γ=0.8 · matched → B1, B4, B5 for each.
Sample level (needs training): if the chapter needs it, a small pass on pems08f + pems08s only:
{temporal-only, road-only γ=0.8} × 2 seeds = 8 cells (≈2 h each on the 8-slot queue). Decision for later.

## Where it goes in the thesis

- Methodology: the construction (already there) — math only, with pointers to the analysis.
- **Chapter "The prior and the residual field"** (analysis chapter): §A → §B → §C → §D, each figure with its claim and the objection answered in the caption or text; conclusion paragraph per section (template rule).
- Results chapter: the joint win (campaign) + §E1/E2 as the mechanism, E3 as the honest limit.
- Appendix: derived per-cell configuration table, the (β, γ) grid, sampler/step counts, run times.

## Execution order (Mac, CPU)

1. `scripts/analysis/prior_fidelity.py` → §A1–A6, §B1–B7, §C2–C3 for all six datasets → JSON.
2. `scripts/analysis/forward_spectrum.py` → §D1, D3 (D2 via the existing forward_analysis) → JSON.
3. figure page from the JSONs (one page, one section per §).
4. `scripts/analysis/sample_spectrum.py` → §E1, E2, E4, E5 on the pems08 checkpoints now, the rest as they land.
5. §D4 / §C1 / §E5 maps last (need the sensor coordinates from the raw csv).
