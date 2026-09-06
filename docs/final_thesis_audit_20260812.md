# Final-thesis audit — 2026-08-12

Scope: everything in `tudelft_thesis_draft/` against everything on disk in `results/`
and `src/`. Post-greenlight pass, aimed at submission.

Build status: `latexmk` clean, **77 pages**, 0 undefined refs/citations, 4 underfull
hboxes (all in the nomenclature). Nothing is broken typographically.

---

## Part 1 — Correctness defects

### D1. The main result table is a different experiment from the one the thesis describes. **BLOCKING**

`Chapter 4, Eq. (4.2)` defines the conditioning as

```
C = [X , Ŷ]     X = the observed history window
```

and the whole of §4.1 argues for it (three paragraphs on why the history is not
redundant with the forecast). Every headline number in Chapter 5 —
`tab:exp_main`, `tab:exp_dist`, `tab:exp_paired`, the "history" column of
`tab:exp_cond`, `tab:exp_sota`, `tab:exp_extra`, and the appendix seed-spread
table — is aggregated from `results/20260721_ladder/{iso,emp}-past`.

That arm did **not** condition on `X`. `--cond-past` fell back silently to the
previous window's *residual* field, `C = [R_prev , Ŷ]`, because no residual cache
had a `history` key until `7eb0a49`. Fixed in `1e253cc`; documented in
`docs/conditioning_report.md`; the arm was formally retired on 2026-07-28. The
thesis text was never re-aggregated.

Verified numerically — every value in `tab:exp_main` reproduces exactly from
`iso-past`/`emp-past`, and none of them reproduces from the real history arm:

| PEMS08 flow | thesis `tab:exp_main` | ladder `-past` (R_prev) | histcond `-hist` (real X) |
|---|---|---|---|
| linear, iso / ours | 0.0829 / 0.0841 | 0.0829 / 0.0841 | 0.0939 / 0.0920 |
| GCN, iso / ours    | 0.0854 / 0.0843 | 0.0854 / 0.0843 | 0.0742 / 0.0732 |
| GWN, iso / ours    | 0.0496 / 0.0501 | 0.0496 / 0.0501 | 0.0497 / 0.0502 |

The correct campaign is complete and unused: `results/20260728_histcond`,
6 datasets × 3 backbones × 2 priors × 2 seeds = **72 test evals, zero failures**,
config-parity verified field-by-field against the ladder.

**Fixing this improves the thesis's own result.** Paired-seed CRPS, structured vs
isotropic:

| arm | CRPS pairs won | MAE | cells won |
|---|---|---|---|
| what the thesis reports (`-past`) | 17 / 50 (p = 0.99 *against*) | 41 / 50 | 5 of 16 |
| the real history arm (`-hist`), PEMS | **16 / 24** (p = 0.076) | 22 / 24 | **8 of 12** |
| the real history arm, all 6 datasets | 21 / 36 (p = 0.20) | 33 / 36 | 11 of 18 |

and the cell pattern is exactly the ladder law the thesis argues for, which the
`-past` numbers currently contradict:

```
             mlp      gcn      gwn        (test CRPS, structured minus isotropic, %)
pems08f    +2.00    +1.28    -1.02
pems08s    +3.22    +0.38    +0.94
pems04f    +1.86    +1.23    -0.77
pems04s    +0.85    +1.04    -0.79
pemsbay    +1.75    +0.42    -1.30
metrla     -0.98    -0.29    -2.01
```

Structured wins **every mlp and every gcn cell on the four PEMS benchmarks**, and
loses on GWN — i.e. it pays in proportion to the structure the backbone leaves
behind, which is the thesis's central claim. Under the `-past` numbers the mlp
row loses everywhere and §5.4 has to argue around it.

The SOTA claim is unaffected: the GWN numbers move by ≤0.0002 CRPS, so
`tab:exp_sota` still beats HySRD-Net by 8–16% on every dataset.

Also note `metrla` is the one dataset where structure loses on all three
backbones — consistent with §5.3.2's `align` diagnostic and worth saying so.

---

### D2. The reported sampler systematically truncates the structured arm. **HIGH**

`reverse_sample_ancestral` stops at a terminal noise floor `c_lo = 0.02` and
returns the denoiser's `R̂₀` there, discarding whatever noise is still present.
How much that is, per arm, is the variance-weighted

    f(c_lo) = Σ_m var_m (1 − e^{−2 θ_m c_lo}) / Σ_m var_m

At the locked floor:

| | isotropic | structured (range over the 18 cells) |
|---|---|---|
| f(0.02) — locked | **3.15%**, identical in every cell | **19.0 – 28.2%** |
| f(8e-4) — training floor | 0.13% | 0.9 – 2.4% |

The isotropic figure is a constant because θ ≡ 0.8 everywhere. The structured arm
discards **6–9× more of its variance** than the control when sampling stops, and
it does so in *every* cell of the grid — including Graph WaveNet, whose spectrum
is the mildest (θ_max = 16) and still leaves 19% un-denoised.

(An earlier version of this note used `1/θ_max` against the floor as the summary.
That is the wrong statistic: it is a property of one mode rather than of the
prior's variance budget, and it wrongly predicts that Graph WaveNet cells are
unaffected. The probe below shows they are affected as much as the stiff ones,
which is what the variance-weighted measure predicts.)

So the two arms are not sampled to the same fidelity, and the gap is present
grid-wide. That directly manufactures the thesis's most emphatic negative finding
— "structure buys under-dispersion, without exception, in every paired seed".

**Measured, pems08f, 256 test windows, k=24, seed 42:**

| cell | arm | c_lo | coverage | CRPS | Winkler | energy |
|---|---|---|---:|---:|---:|---:|
| gcn (f = 26.1%) | iso | 0.02 | 0.751 | 0.1085 | 221.4 | 11.18 |
| | iso | 8e-4 | 0.742 | 0.1087 | 223.7 | 11.24 |
| | emp | 0.02 | 0.713 | 0.1047 | 222.1 | 11.12 |
| | emp | **8e-4** | **0.761** | **0.1041** | **210.1** | **10.62** |
| gwn (f = 19.2%) | iso | 0.02 | 0.772 | 0.0514 | 105.2 | 3.13 |
| | iso | 8e-4 | 0.770 | 0.0515 | 105.9 | 3.30 |
| | emp | 0.02 | 0.712 | 0.0518 | 112.4 | 3.82 |
| | emp | **8e-4** | **0.761** | **0.0516** | **106.7** | **3.53** |

The isotropic arm moves by 0.2–0.9 points of coverage; the structured arm gains
4.8–4.9 and overtakes the control on coverage, Winkler and energy on both cells
while keeping its CRPS advantage. On these two cells, three of the thesis's four
negative findings are artefacts of where the sampler stops.

Corroborated by the one matched integrator sweep already on disk
(pems08f/gwn/seed42, `docs/conditioning_report.md` §Calibration):

| sampler | iso CRPS | emp CRPS | gap | iso cov | emp cov | rsh cov |
|---|---|---|---|---|---|---|
| ancestral@50 | 0.04966 | 0.05010 | +0.9% | 0.822 | 0.764 | 0.646 |
| EM@1500      | 0.04953 | 0.04957 | **+0.1%** | 0.856 | **0.855** | **0.878** |

Push the integrator to convergence and the accuracy gap becomes a tie and the
calibration gap closes completely. Euler–Maruyama integrates to `c̄ = 0` rather
than stopping at a floor, which is the same intervention by a much more expensive
route (EM@1500 is 29× the wall-clock of ancestral@50).

Note the training floor is `8e-4` (log-uniform draws span `[8e-4, c_0]`), so the
sampler stops **25× above** the noise level the denoiser was trained down to.
This is a bug, not a design choice: `c_lo` is a `reverse_sample_ancestral`
kwarg that `eval_diffusion.py` never passes.

---

### D3. Smaller correctness items

| # | item | severity |
|---|---|---|
| D3a | `5_experiments.tex:175` — stray `t` after `\caption{...}`; renders a literal "t" above Table 5.1 | typo, visible |
| D3b | The boxed `eq:Theta`, `Θ=(λ_k+γ)/μ_h`, omits the two scalars the code applies: `τ² = N/Σ(1/q_S)` and the anchor `θ_min → 0.8`. Both are described in prose two paragraphs later but the equation a reader copies is not the implemented one. | accuracy |
| D3c | `Σ_T = D_H^{1/2} R_H D_H^{1/2}` mixes an **uncentred** second moment (`D_H`) with a **centred Pearson** correlation (`R_H = np.corrcoef`). The result is still PSD and the construction is sound, but §4.2.2 argues at length that `D_H` is a mean-square *on purpose* and then never says `R_H` is not. One sentence + one footnote fixes it. | rigour |
| D3d | Arm variance matching is approximate, not exact: mean stationary variance is 1.000 for kron in all cells but 0.942–1.000 for iso (worst on METR-LA, a 4.7% deficit). Cause: iso uses `R[mask].var()` over all masked entries, kron uses `E[r²]` over rows with all 12 horizons observed. §5.4 calls the arms "scaled to the same total residual variance by construction". | rigour |
| D3e | Seed-count claims ("five per cell on PEMS08") become wrong after D1 — the histcond grid is 2 seeds uniformly. Every caption and the paired-count table need re-deriving. | follows from D1 |
| D3f | `\listoffigures` / `\listoftables` are commented out in `report.tex`. TU Delft submissions normally carry both. | formatting |

**What checks out, verified:** the OU derivation and the appendix proof; the
Lyapunov stationary covariance; Tweedie; the Anderson reverse drift; the
ancestral posterior algebra (`1/v = a_d²/s_d + 1/s'` — exact); the Kronecker
eigenvalue rule; the `τ²` unit-mean-eigenvalue normalisation matches the prose;
the appendix hyperparameter table matches the actual `config.json` on disk
(patience 40, log-uniform noise, `best_crps.pt`, ancestral@50, k=100, per-sensor
normalisation, γ=0.8, ρ=0.1) — all correct.

---

## Part 2 — Presentation

### Figures: 4 in 77 pages, and 12 already-rendered ones sitting unused

Referenced: `f_attribution`, `f_forward`, `f_channels`, `f_reliability`.
On disk and never included: `f1_mechanism`, `f2_fanchart`, `f3_reliability`,
`f4_fewstep`, `f5_speed_pareto`, `f7_horizons`, `f8_backbone_improvement`,
`f9_convergence`, `f10_residual_stats`, `f_law`, `fig_theta_spectrum`,
`fig_forward_decay`, `fig_forward_convergence`.

Gaps that matter for a reader:

1. **No picture of the prior itself.** The thesis's whole subject is `Σ_T ⊗ Σ_S`
   and there is no figure of `Σ_T`, of `R_H`, or of the `Θ` spectrum over the
   `(N × H)` mode grid. This is the single most useful missing figure and it is
   what makes the "Kronecker product / Pearson correlation + per-horizon
   variance" construction legible at a glance.
2. **No forecast picture.** No fan chart, no sample trajectories. Every
   probabilistic-forecasting thesis has one; it is also the only place a reader
   sees that the output is a distribution.
3. **No per-horizon comparison against baselines** — `tab:exp_sota` has 156
   numbers and no figure. One CRPS-vs-lead-time panel replaces a third of it.
4. **No cost figure** despite §5.6 being one of the strongest results
   (17k params, 17× cheaper sampling).
5. `f_reliability` is a single small panel; it should carry both conditioning
   regimes or be merged into the calibration figure.

### Structural overlaps to prune

The RQ answers are stated four times in near-identical language: §5.3.4
(Summary), §5.8 (chapter Conclusion), §6.1–6.2 (Discussion), §7.1 (Answers).
Roughly 2.5 pages of pure repetition. The intended division is:
Ch5 = what was measured, Ch6 = why, Ch7 = the answers. §5.3.4 and §5.8 should
compress to 3–4 sentences each and stop restating attribution numbers that
Ch6 §6.2 then restates again.

Other redundancies: the `γ=0.8` near-identity argument appears in §5.3.2, §5.8,
§6.2 and §7.1 with the same two numbers; the "heavy tails cap calibration"
sentence appears in §5.3.1, §5.7, §6.4 and §7.3.

### The product-graph question is absent from the thesis

§4.2.3 argues correctly *in prose* why the Kronecker **product** of precisions
is used rather than the Cartesian product graph's Laplacian (a Kronecker
**sum**, whose inverse does not factorise, so the factors cannot be estimated
separately and the stationary law cannot be matched to the measured
`Σ_T ⊗ Σ_S`). There is no experiment behind it.

`results/20260728_pgraph_*` ran cartesian / strong / tensor / parametric, but
(a) on the **raw signal**, not the residual, so it does not test the object the
thesis is about, and (b) `docs/pgraph_campaign_report.md` shows it is
uninterpretable — training budgets differ 4–6× between arms and the two
datasets where structure "wins" are exactly the two where it trained longest,
and the sampler produced coverage 0.108. It cannot go in as-is.

One result from it *is* solid and needs no training: fitting GTCNN Eq. 10's four
free coefficients puts **spatial coupling at exactly zero on all six datasets**
(`s₀₁=0, s₁₀=1, s₁₁=0`), i.e. the data selects a temporal-only prior out of the
full product-graph family. That is a strong, cheap ablation.

---

## Part 3 — What to run

Ranked by (value to the thesis) ÷ (GPU cost). Nothing here is training from
scratch except R4.

| # | run | cost | what it buys |
|---|---|---|---|
| **R1** | Re-aggregate Ch5/6/7 on `20260728_histcond`. | **0 GPU** | Fixes D1. Turns the headline from "17/50, evidence for the control" into "wins every mlp and gcn cell on PEMS". |
| **R2** | Ancestral sampler sweep on saved checkpoints: `c_lo ∈ {0.02, 8e-4}` × `steps ∈ {50, 200}`, eval-only. Pilot on 4 cells (pems08f × {mlp,gcn,gwn} + pems04f/gcn), both priors, 1 seed. Needs a 3-line `--anc-c-lo` flag in `eval_diffusion.py`. | ~2 GPU-h pilot; ~30 GPU-h for the full 18-cell grid | Fixes D2. Either the under-dispersion is real (thesis stands, now defensibly) or it is a grid artefact (the calibration story changes completely and in our favour). Currently unfalsified either way. |
| **R3** | Held-out NLL of the whole prior family on saved residuals: iso / Σ_T-only / Σ_S-only / Kron / Cartesian / strong / tensor / GTCNN-parametric, 6 datasets × 3 backbones. | **0 GPU** (CPU minutes) | Gives §4.2.3's prose argument an experiment, answers "why not a vanilla product graph" with data, and lands the spatial-coupling-is-zero result. |
| **R4** | 3 more diffusion seeds on the histcond grid (18 cells × 2 priors × 3 seeds = 108 runs). | ~100 GPU-h | Takes PEMS CRPS from p=0.076 to conclusive. This is the difference between "suggestive" and "significant" for the thesis's main claim. |
| **R5** | Test evals for the two axes currently reported as validation-only: the 996k-UGnet capacity ladder and the DDPM corruption-family ablation. | ~15 GPU-h | Removes the "these are validation numbers" caveat from §5.5.2, §5.7 and both Limitations sections. |
| **R6** | Reshaped arm under history conditioning (currently only fo/`-past`). | ~20 GPU-h | Keeps the "tightest fit" extreme point on the same axis as the main table; otherwise §5.7's last ablation is stranded on a retired arm. |

R1 + R3 are free and should happen regardless. R2 is the highest-information
run in the list. R4 is the one that costs real time.

Two things NOT recommended: re-running the product-graph campaign on the
residual arm (3 forms × 18 cells ≈ 200 runs for an ablation R3 answers offline),
and any new γ sweep (24 cells already say the verdict does not flip).
