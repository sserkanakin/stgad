# Thesis notes — decisions, one step at a time

*Running log of every discussion about the thesis: what we decided to say, which figure carries it,
what we leave out, and why. One step per conversation; the step is closed only when the user says so.
Method reference: `docs/method.md`. Results: `results/analysis/*.json`, figures `results/figures/`.*

## How we work through it

| step | topic | outcome we need before moving on |
|---|---|---|
| 0 | scope and claim | one paragraph everyone agrees with: problem, claim, what we do NOT claim, the research questions |
| 1 | Ch. 1 Introduction | motivation story, RQs, contribution list, outline |
| 2 | Ch. 2 Background | the concepts the reader needs and at what depth (Laplacian/GSP, Kronecker, diffusion, GAD, the metrics) |
| 3 | Ch. 3 Literature | positioning: what others do, the gap, one sentence per neighbour paper |
| 4 | Ch. 4 Methodology | section list, formulas kept in the body vs appendix, the two figures (toy Kronecker; Σ_S maps) |
| 5a | Ch. 5.1–5.2 setup + residual field + prior | figures 1–6, the claims and the objections answered |
| 5b | Ch. 5.3 main result | the joint figure/table, the per-point tie, how to phrase the win |
| 5c | Ch. 5.4 ladder | MLP/GCN/GWN figure, the "when does structure help" sentence |
| 5d | Ch. 5.5–5.7 baselines, calibration, ablations | SOTA table, old-vs-new, chain diagnostic, DC rule, two-moment prior |
| 6 | Ch. 6 Discussion | mechanism, limitations, practice |
| 7 | Ch. 7 Conclusion + appendices | answers to RQs, future work, App. B contents |

For each step: we talk → decisions are written here → the LaTeX section is drafted from the decisions →
the user reads it → the step is closed.

---

## Step 0 — scope and claim (open)

**Proposed scope, in plain words (to be corrected by the user):**

A strong deterministic traffic forecaster (Graph WaveNet) is wrong by a structured amount. We model
the distribution of that error with a diffusion model. The one thing we change, and the thesis is
about, is the *noise* the diffusion model corrupts with and starts from: instead of white noise we use
a Gaussian whose covariance is built from the road graph (heat kernel + one weak global edge) and the
error's own horizon structure (Kronecker product), with its two free scalars set so the prior's
uncertainty of the network-wide total matches the data. Everything else — data, backbone, denoiser
(17k parameters), training, sampler, evaluation — is identical between the white-noise arm and ours.

**What we claim (measured):**
1. The residual field is structured (neighbour correlation, horizon correlation, a network-wide
   component); a Kronecker graph prior captures it at every aggregation scale; white noise and the
   road-only prior do not (they collapse at the network scale).
2. With the prior, per-sensor accuracy is unchanged (a tie; both arms below published methods) and
   every network-level uncertainty measure improves on all six datasets (CRPS_sum −3…−6 %, coverage
   of the total +6–8 points, spread +21–53 %), 4 seeds.
3. The weaker the backbone, the larger the gain — with an MLP backbone the prior also wins per
   sensor (−1.5…−6 % CRPS) and by −4…−24 % CRPS_sum.
4. The remaining under-dispersion of the total is the denoiser's treatment of one mode (the
   network-wide shift); the sampler is exact; the prior's filter for that mode restores coverage
   (an ablation, not the final fix).

**What we do NOT claim:** better point accuracy than the backbone; a learned prior; heavy-tailed
modelling; that the prior is right under distribution shift (PEMS08 speed / PEMS-BAY test drift).

**Research questions (draft):**
- RQ1 What structure does the forecast residual have, and can a graph-based Gaussian prior capture it?
- RQ2 Does diffusing with that prior, instead of white noise, change what the forecaster gets right?
- RQ3 Where does the remaining error come from — prior, sampler or network — and what fixes it?

**Decisions (2026-09-10, step 0 CLOSED):**
- **Scope is general:** probabilistic forecasting of *graph signals over time* (any sensor network on a
  graph whose point forecaster leaves structured errors). Traffic is the case study where everything is
  measured (six datasets); claims stay within what is measured. No "traffic" in the title.
- **Name / title:** method = **Spatio-Temporal Graph-Aware Diffusion (STGAD)**; title
  *"Spatio-Temporal Graph-Aware Diffusion for Probabilistic Forecasting of Graph Signals"*.
  The text says once that it builds on GAD (graph-aware diffusion, Rozada et al. 2025).
- **"Uncertainty" means the forecast distribution, scored by CRPS** — per sensor (a tie between the
  arms; both below published methods) and **CRPS_sum on the network total** (+ coverage, spread),
  the standard multivariate score (TimeGrad/CSDI) and the one we defend; with the MLP backbone the
  prior also wins per sensor.
- **GSP is the language of the thesis:** the prior is a GMRF with Laplacian precision (a graph
  filter); the forward process is graph heat diffusion with a time-warped rate; denoising is a
  sequence of graph-filtering problems (GAD Prop. 1); the residual analysis is graph-spectral
  (variance per graph frequency, the λ=0 mode, aggregation over graph clusters); the global edge is
  a graph modification read off the spectrum; the one failure found sits at one graph frequency.
- The sampler/floor material is step 5d (ablations), explained with an example there.

---

## Step 1 — Introduction (drafted 2026-09-10, awaiting the user's read)

**Live draft:** `tudelft_thesis_only/` in the repo (gitignored, like the old draft). Preamble updated:
`\method` = STGAD, `\title` = the step-0 title. The frontmatter Summary still tells the OLD story
(prior degrades joint dependence, sharper mean only) — rewrite it LAST, from the final results.

**Decisions:**
- Five-paragraph story: graph signals and point forecasts → one number is not enough (intervals and
  sums; the 5 %-correlation example) → diffusion on residuals and the unquestioned choice of white noise
  (wrong on sums by construction) → GSP alternative in GAD's own words (heat equation, GMRF with the
  inverse shifted Laplacian, "smooth with respect to the graph", coarse-first reverse), extended to
  space–time as a Kronecker prior; **no global edge in the intro**, no "variance on frequencies"
  wording (that language is for the analysis chapter where it is measured) → what we did and found.
- RQ1 structure of the residual and whether the prior captures it; RQ2 does the prior change what the
  distribution gets right, per node and on aggregates; RQ3 when does structure matter most and where
  the remaining error comes from (prior / sampler / denoiser).
- **Three contributions only:** the prior; the graph-spectral residual analysis; the controlled
  comparison (six datasets, three backbones). Sampler and DC-mode diagnosis are NOT headline
  contributions (methodology detail / ablation).
- Kept from the old draft: the opening examples, the calibration/proper-scoring citations, the
  chapter labels (`ch:background`, `ch:literature_review`, `chap:method`, `ch:experiments`,
  `ch:discussion`, `chapter:conclusion`). Dropped: the old contribution 4 ("improves on all published
  baselines on every metric") and the old RQ framing around capacity/conditioning.
- Citations available in report.bib for later chapters: `timegrad`, `tashiro2021csdi` (CRPS_sum).
- **User read (2026-09-10): good.** Changes applied: the DC-mode sentence removed from the intro
  (discussed later, Ch. 5/6); each RQ now has a short description of how it is answered; the chapter
  has sections (Motivation · Diffusion on graph signals and the question of the noise · This thesis ·
  Research questions · Contributions · Outline). **Step 1 CLOSED.**

---

## Step 2 — Background (drafted 2026-09-10, awaiting the user's read)

**Decisions:** keep the existing chapter (7 sections, 4 figures; CRPS, calibration, GFT, filters, heat
kernel, GMRF, Kronecker algebra, DDPM/SDE/Tweedie, GAD with the FCPS rate and the optimal graph-filter
denoiser were all already there). Four additions, no rewrite:
1. **2.1 "Scoring a forecast over a graph"** (new subsection): per-node CRPS is blind to dependence;
   CRPS_sum on the network total and on the cumulative per node (TimeGrad, CSDI); energy score;
   PICP + interval score (Gneiting & Raftery; "Winkler"); spread–error ratio. Every Ch. 5 metric defined
   once, standard names, citations.
2. **2.2 GFT:** the constant mode v₁ = 1/√N (λ₁ = 0) named as "the network-wide shift", the rough modes
   near λ = 2; "a prior that gives the constant mode too little variance cannot represent a network-wide
   shift".
3. **2.2 GMRF:** variance of a sum over a node set under the GMRF, Eq. (bg_gmrf_sum_variance):
   Σ_k (v_kᵀ1_S)²/(λ_k+γ); the whole network = the constant mode alone (N/γ); under white noise N·σ² —
   the calculation behind the multi-scale analysis.
4. **2.3 "Product graphs: the other route to a spatio-temporal graph"** (new subsection, the user's
   "vanilla spatio-temporal graphs"): Cartesian / Kronecker / strong products (Sandryhaila & Moura;
   GTCNN), the Cartesian Laplacian is a Kronecker SUM; a GMRF on it is NOT separable (penalties add) vs
   the separable covariance (precision = Kronecker product; penalties multiply). **To be compared as an
   ablation in Ch. 5 (prior level: NLL + multi-scale; add a `cartesian` prior to prior_fidelity.py).**
Deferred (optional trims): 2.1.3 deterministic limits, 2.1.6 autoregressive vs joint, 2.4 neural
forecasting. `\citet` → `\textcite` (biblatex). Builds: 79 pages, no warnings.

**User read (2026-09-10):** (a) the product-graph text must not call product graphs "natural" or prejudge
the comparison — reworded to neutral ("could equally be defined on them"; "neither assumption is right a
priori; Chapter 5 measures how well each fits"). (b) **All background figures redrawn to paper quality**
with one shared visual language (preamble: `fig`, `sensor`, `road`, `tedge`, `block`, `flow`,
`condarrow`, `lab` tikz styles; palette figink/figmuted/figrule/figaccent/figwarm/figamber/figgreen/
figviolet; `\sensorgraph` macro): 2.1 sensor graph at t* with value-shaded nodes + identity-coloured
trajectories with t* marked; 2.2 fan chart with observed history, forecast origin, nested one-hue bands,
ticks, brackets, two-row legend; 2.3 predict-and-refine with stage bands, history + forecast conditioning
(dashed), stacked-card ensembles; 2.4 (new) product graphs — factors, Cartesian (Kronecker sum, penalties
add), Kronecker product, separable covariance (penalties multiply); 2.5 diffusion chain drawn ON a graph
signal (smooth → noise). Rendered and checked twice (pymupdf); builds 80 pages.
**Rule from now on: every new figure in the thesis uses these styles.** Step 2 still open for the user's read.

**User read 2 (2026-09-10):** overflowing text in 2.3 → cards sized to their text; 2.2 redone so the
ensemble members are visibly many random paths and the bands are their pointwise quantiles (caption
explains why bands are smooth while paths wiggle; the realised future is one more path); strong product
added to the text (A_⊠ = A_□ + A_⊗) and to the figure (2.4 now: top row factors + separable covariance,
bottom row Cartesian / Kronecker / strong); 2.5 redone as "one sample" (four-node graph states) over
"the distribution" (scatter of two neighbouring nodes over many samples with 2σ ellipse: narrow diagonal
→ round isotropic cloud). All five rendered and checked; builds 81 pages.

**User read 3 (2026-09-10):** product graphs redrawn with lead time running left to right — one copy of
G_S per column, h = 1,2,3 labelled, a "lead time" axis under each product; solid = spatial edges, dashed =
temporal edges (Cartesian: same sensor at h±1; Kronecker: neighbouring sensors at neighbouring h;
strong: both). Diffusion figure redrawn as a signal on a road of seven sensors, diverging colours
(blue −, red +) plus a value profile under each state, smooth wave → white noise; distribution row kept
(scatter + 2σ ellipse); fitted to the text width. Both rendered and checked.

**User read 4 (2026-09-10) — claims audit + figures, final round:**
- *Citation audit of Ch. 1–2:* every paragraph checked (`scratchpad/bg_audit.txt` method: paragraphs > 40 words
  without `\cite`). Added: data examples (li2018dcrnn, guo2019astgcn), two dependences (capone2025stgnn_survey),
  windowing convention (li2018dcrnn, wu2019graphwavenet), margins-blind scoring (gneiting2007strictly),
  spread–error ratio (**new bib entry `fortin2014spread`**), DiffCast/CorrDiff regimes (diffcast2024,
  mardani2025corrdiff), autoregressive (salinas2019deepar, timegrad) vs joint-horizon (tashiro2021csdi,
  wen2023diffstg), normalised-Laplacian identity + connected-graph zero mode (chung1997spectral), heat kernel
  (shuman2013emerging, hammond2011wavelets), matrix normal (dutilleul1999mle), graph products
  (hammack2011handbook), conditioning by context (tashiro2021csdi, wen2023diffstg), isotropic-vs-structured
  corruption (graph-aware-diffusion, rissanen2023ihd, hoogeboom2023blurring), Tweedie (efron2011tweedie).
  Intro: white-noise claim now cited (timegrad, tashiro2021csdi, wen2023diffstg, lin2024specstg,
  liu2023pristi, hysrdnet2026); cross-refs to §2.1.5 (scores), §2.2.5 (GMRF sums), §2.6 (GAD), Ch. 5.
- *Figures, after checking the canonical depictions online:* product graphs drawn as **lattices** as in
  Sandryhaila & Moura (2014, IEEE SPM): sensors down the side, lead times along the bottom, Cartesian = grid,
  Kronecker = diagonals, strong = both; separable covariance shown apart. Diffusion figure reverted to the
  **classic DDPM chain** (Ho et al. 2020 Fig. 2 layout): four panels of the same road signal, smooth → noise,
  forward arrow above, learned reverse below; the distribution row was dropped as too much.
- Sources consulted: Sandryhaila & Moura 2014 (hajim.rochester.edu PDF), DDPM figure descriptions (arXiv
  2405.13540 and reviews).
- **Figure rules (permanent):** shared preamble styles only; no text may overflow its box; render and inspect
  every figure (pymupdf at 120 dpi) before declaring it done; one idea per figure.
