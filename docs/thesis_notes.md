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

---

## Results dossier (2026-09-11) — how every result is presented

Web view, built from `results/analysis/results_dossier.json` (collector inline in the session; figures
from `results/figures/*.png` embedded) → artifact **"STGAD Results Dossier"**
https://claude.ai/code/artifact/7c8c2379-7f14-49ae-9b7d-87672c6bfe75 . Companion to the story page
("Noise With a Shape", method). Each result = figure/table + the claim sentence + where it goes + what it
supports + what it does not show:
- Setup table (datasets, backbone MAE, (β, γ), match ratios, steps) → §5.1 / App. B.
- R1 residual field (corr vs hops, common-mode share, total sd vs independence, horizon growth) → §5.2.
- R2 prior fit at every scale (multiscale, spectrum, NLL + separability) → §5.2, key figure.
- R3 per-node tie + SOTA table with the Gaussian floor → §5.3 / §5.5.
- R4 network level 6/6 (joint total + cumulative figures, per-dataset table with DC column) → §5.3.
- R5 ladder (prior level + result level; GCN partial) → §5.4.
- R6 old protocol vs new → §5.7 + discussion.
- R7 chain diagnostic + DC rule on all cells → §5.7 / discussion.
- R8 prior-level ablations (five priors, NLL + network scale) + two-moment rule → §5.7.
- R9 cost (steps, s/window) → §4.5 / App. B.
- "Not in yet": GCN wave, Cartesian prior row, aMMD + steps curve, calibration curves, sampler ablation, DiffSTG.

---

## Evidence for Step 3 (literature) — who reports network-level scores (checked 2026-09-11 in the PDFs)

| paper | metrics reported | network-level? |
|---|---|---|
| DiffSTG (SIGSPATIAL 2023) | MAE, RMSE, per-node CRPS ("CRPS is the primary metric") | no |
| HySRD-Net (ESWA 2026) | RMSE, MAE, CRPS; "coverage" only qualitatively in a figure | no |
| SpecSTG (2024) | RMSE, MAE, CRPS; **explicitly: "We do not adopt CRPSsum [TimeGrad; CSDI] as a metric because sensors are not regarded as features in our experiments"** | no, by choice |
| USTD (2023) | MAE, RMSE, CRPS | no |
| TSGDiff, Double-Diffusion | MAE, RMSE, (MAPE), CRPS | no |
| GCRDD | MAE, MAPE, RMSE only | no |
| TimeGrad (ICML 2021) | CRPS per dimension **and CRPS_sum** ("on the sum of all time series dimensions"; proper, de Bézenac et al.) | yes — the standard |
| CSDI (NeurIPS 2021) | CRPS, **CRPS-sum** "CRPS for the distribution of the sum of all time series across K features and accounts for joint effect" | yes |
| ProbTS benchmark (2023) | NMAE, CRPS; notes many methods "primarily reveal aggregated distributional metrics ... namely CRPS-sum" | acknowledges it |
| SSTVM (IEEE TSIPN 2025, spatio-temporal, PeMS08/AIR-BJ) | MAE, RMSE, CRPS, **CRPS_sum** (Eq. 32: sum samples across dimensions) | yes — first ST-graph paper we have that does |
| GenCast (Nature 2025) | CRPS, spread/skill, rank histograms, **spatially pooled CRPS** and **regional wind-power sums** "to evaluate how well a model captures spatial dependency structure" | yes — same idea, different field |

Use in Ch. 3 positioning: the traffic-diffusion line (DiffSTG → SpecSTG → HySRD) evaluates per node only and
SpecSTG says why; the general multivariate line (TimeGrad, CSDI) and the newest ST-graph work (SSTVM) and
weather (GenCast pooled CRPS) treat the sum as the test of joint structure. Our CRPS_sum + cumulative +
PICP + spread/skill = that standard, applied where the field had not.

**Comparable network-level numbers in the literature (checked 2026-09-11):**
- SSTVM (TSIPN 2025) Table IV, *normalised* CRPS_sum on PeMS08: TimeVAE 0.163, TimeGrad 0.090, CSDI 0.110,
  ScoreGrad 0.044, TSDiff 0.097, SSTVM 0.074 (their MAE on PeMS08 = 13.3 vs DiffSTG's 17.7 → a different
  protocol/normalisation; not directly comparable, quote with that caveat).
- Ours in the same convention (CRPS of the sum ÷ mean |sum|, test, GWN): pems08f iso 0.0115 → kron 0.0112;
  pems08s 0.0046 → 0.0044; pems04f 0.0123 → 0.0117; pems04s 0.0041 → 0.0040; metrla 0.0103 → 0.0099;
  pemsbay 0.0065 → 0.0061 (−3 to −6 %). An order of magnitude below SSTVM's table → protocols differ
  (residual on GWN with history conditioning vs full-signal generation); present side by side only with the
  caveat, or not at all.
- Pal et al. (ICML 2021, RNN + particle flow): PeMSD3/4/7/8 **speed**, 70/10/20 split, P = Q = 12, raw-unit
  per-node CRPS (PeMSD8 60 min: 1.28; PeMSD4: 1.56) — per node only, and our PEMS split is 60/20/20
  (ASTGNN), so not comparable without a re-split.
- AIR-BJ / AIR-GZ (air quality, station graph): USTD reports MAE/RMSE/CRPS, SSTVM CRPS_sum 0.055 on AIR-BJ;
  processed data in the USTD repo (Google Drive) → the natural non-traffic dataset if we extend the scope.
- GenCast's device for spatial dependence = **pooled CRPS at several spatial scales**; our prior-level
  multi-scale check is the same idea; the sample-level version (CRPS of sums over spectral clusters of the
  road graph, per model) is an eval-only addition — proposed as "R4b".

**Decisions (2026-09-11):**
- **AIR-BJ / non-traffic dataset: NOT added now.** Goes in as a side note (future work / "the method needs only
  a graph, regular sampling and a point forecaster; USTD/SSTVM's AIR-BJ is the natural next case"). Work
  with the six traffic datasets.
- Chapter 5 order: network-level results first (GenCast-style: CRPS_sum, cumulative, PICP, spread/skill),
  then the per-node SOTA table as "nothing lost per node, both arms below published methods".
- SSTVM's PeMS08 CRPS_sum: mention in the discussion with the protocol caveat, not in the main table.
- Pooled-CRPS at several graph scales (sample level, eval-only) — proposed, not yet approved.

**Decisions (2026-09-11, later):**
- **Pooled-CRPS evaluation APPROVED** ("if you say it is important lets do it"): CRPS/PICP/spread of sums over
  spectral clusters (64/16/4) + network, on the 48 GWN checkpoints, eval-only, files tagged `_pooled`
  (`--tag`), clusters shared with the prior-level analysis (`src/data/adjacency.spectral_clusters`).
  → Result 4b in the dossier and §5.3 figure "advantage vs aggregation scale" (GenCast's pooled CRPS).
- **Order of work:** finish Ch. 2 (user reading) → Step 3 Literature review → then a dedicated
  **results-overview session**: walk through every result and every number (dossier R0–R9) until the user
  can explain each one, BEFORE writing Ch. 4–5. The user said they currently do not follow the numbers.
- Status snapshot: GWN campaign 48/48 + DC ablation 48/48; MLP wave 24/24; GCN wave ~10/24 running;
  prior analyses 18/18 (all backbones × datasets) + two-moment; figures 13; story page + dossier live.

---

## Step 3 — Literature review (proposal 2026-09-11, awaiting the user)

Existing chapter: 5,061 words, 5 sections (ST graph forecasting · diffusion-based probabilistic forecasting ·
predict-and-refine residual diffusion · structured forward processes · synthesis/gap/positioning), a
"where structure enters" figure, a gap-map figure and a design-axes table, ~40 citations — solid, keep the
structure. Proposed changes:
1. **New subsection "How probabilistic spatio-temporal forecasts are evaluated"** (in §3.2): the evidence
   table — DiffSTG/HySRD/USTD/TSGDiff per node only, SpecSTG's explicit refusal of CRPS_sum, TimeGrad/CSDI
   CRPS_sum as the multivariate standard, SSTVM (TSIPN 2025) on PeMS08, GenCast's pooled CRPS + spread/skill.
   Ends with: "the evaluation this thesis uses is that standard, applied where the traffic line skipped it".
2. §3.5 positioning rewritten to the step-0 claims: the gap is (i) the corruption (isotropic) AND (ii) the
   evaluation (per node); STGAD fills (i), the chapter-5 protocol fills (ii). No old-result claims.
3. §3.4: keep GAD/heat/SPDE; add one sentence on product-graph GMRFs as the alternative structure (cf. §2.3.1).
4. Pooled eval pass running (jobs 286582–286593) → will feed the new subsection's "our protocol" sentence.
5. Figures (structure-entry, gap map) redrawn in the shared style; citation audit as for Ch. 2.

**User read 5 (2026-09-11) — product graphs + diffusion figure, final form:** product-graph figure now in
the GTCNN style the user pointed to (G_S ◇ G_T = one product drawing with three edge colours: spatial red,
temporal green arrows, diagonal grey; legend + the three adjacency formulas and edge counts). Diffusion
figure now a two-row comparison in the GiffCF style the user pointed to, but with OUR content: the same
residual field (7 road sensors × 8 lead times, a congestion wave) destroyed (a) by white noise → speckle and
(b) by the graph-aware Kronecker OU process → smooth patches; both rows are REAL simulations on a path graph
(`figures/generated/diffusion_rows_cells.tex`, generator inline in the session; seed 3). Arrows:
perturb/denoise vs diffuse-on-the-graph/graph-filter.

**User read 6 (2026-09-11) — final:** the two-row diffusion figure was too confusing → replaced by the
single-row DDPM figure (top row of the GiffCF-style example): four simulated panels of one residual field
(7 sensors × 8 lead times) under white-noise corruption, row-long arrows "forward process: perturb with
q(z_n|z_{n−1}) — fixed and known" above and "reverse process: denoise with p_θ(z_{n−1}|z_n) — learned"
below; caption states the DDPM forward marginal (Eq. bg_ddpm_marginal). Product-graph figure: larger nodes,
the three products as an aligned equation block (adjacency + edge counts) with the colour legend at left.
Both confirmed on rendered screenshots (pages 23, 25). The graph-aware two-row version is kept in git history
for a possible Ch. 4 figure.

**User read 7 (2026-09-11) — edges off the nodes, row centred:** user: "the kronecker is not bad, but can be
better the edges are overlapping with the nodes, make them look better with a 3d effect maybe. diffusion is
good just make sure it is all centered."
- Product-graph figure (2.4): the copies of the sensor graph are now drawn as thin slabs in an oblique view
  (front face + extruded top/right faces along the horizon direction), offset d = (3.0, 1.2) cm per lead time.
  The four-node tree was replaced by four sensors along a road (a path 1–2–3–4, gentle wiggle) because a
  geometric search over layouts showed the tree could never get more than ~0.35 cm between an edge and a
  node it does not touch (node radius 0.22 cm), while the path gets 0.66 cm: no temporal or diagonal arrow
  passes over a node any more. Draw order: slabs → nodes → diagonal (grey) → temporal (green) → spatial (red)
  → nodes again on top. Legend moved under the factors; the aligned equation block (Cartesian/Kronecker/
  strong adjacency + edge counts) is centred under the whole drawing; the picture now fits the text width
  (before it ran 20 pt into the right margin). Caption: "Four sensors along a road", "given below the drawing".
- Diffusion figure (2.5): the row of four panels sat 24 pt right of the text centre and the picture overflowed
  the margin by 8 pt. Fix: picture scaled 0.88, both end labels ("data: residual field" / "noise: N(0,I) −/+")
  given the same width (1.3 cm) and the same distance to the row, moved clear of the rotated "sensor i" axis
  label. Measured on the PDF: panel row centre 297.6 pt vs text-block centre 297.5 pt.
- Both confirmed on rendered screenshots (pages 23 and 25). Rule kept: measure figure bounding boxes against
  the text block with pymupdf after every figure change (drawings + words), not by eye.

**Final pass on Ch. 1–2 (2026-09-11):** user: "do a final run on the intro and background. check all figures with
screenshots. confirm every claim is supported. no unnecessary info. nothing is mentioned before explaining.
and cited. and normal words are used, nothing too fancy so that it doesnt look like AI too."
- Claims corrected (each checked against the data or the source):
  - Intro: "diffusion models are the strongest tool" → plain description; "applied to the residual … reach
    SOTA [DiffSTG, PriSTI]" was wrong (DiffSTG diffuses the signal, PriSTI is imputation) → "hold the published
    per-node records [DiffSTG, SpecSTG, HySRD]; the most recent diffuse the residual [HySRD]".
  - Intro: "wrong by construction … however good the denoiser is" overclaimed (a perfect score model would
    recover the correlations from any prior) → "a poor starting point; the small denoisers used here do not
    put the shared component back" (what we measured).
  - Intro: "both sit below the published methods" → qualified to the four PEMS benchmarks where diffusion
    forecasters publish per-node CRPS (ours 0.066/0.021/0.050/0.017 vs best published 0.072/0.024/0.055/0.020;
    METR-LA and PEMS-BAY have no such published numbers). "per-node linear model" for the smallest backbone
    verified: DLinear has no nonlinearity.
  - Intro: the 5 %-correlation example made explicit as a hypothetical with the arithmetic
    ((N−1)ρ = 169·0.05 ≈ 8.5, "more than eight times").
  - Background: CRPS_sum credited to Salinas et al. 2019 (GP-Copula), not TimeGrad; energy-score weakness
    now cited (Pinson & Tastu 2013, new bib entry); PriSTI removed from "forecasters we compare against";
    the unverifiable "same left-factor convention as the product-graph literature" sentence removed; the
    "EM sampler kept as parity comparison" sentence removed (no EM evaluations exist in the matched campaign,
    only ancestral); a garbled double "Tweedie's formula" sentence fixed; worked-example numbers re-verified
    (S₂ = 7 / 124, energies 19 / 28, Rayleigh 0.18 / 1.92, path-graph eigenvalues {0,1,2}, 13.6 M entries).
- Order: forward pointers added where a term is used before its section (diffusion in §2.1.4 and §2.1.6);
  GMRF glossed in the intro before its name is used.
- Unnecessary info removed: the three adjacency + edge-count formulas under Figure 2.4 (they duplicate §2.3.1)
  → one line "edges kept: Cartesian = spatial + temporal; Kronecker = diagonal; strong = all three"; the
  GNN-variant enumeration in §2.4.1; the sampler-literature sentence shortened.
- Wording toned down in ~25 places ("common currency", "natural headline metric", "paradigm", "fingerprint",
  "signature", "the rest of this thesis lives in it", "precisely", "cosmetic", "division of labor", …).
- Background conclusion now also carries the scoring thread (six things, not five).
- All five Chapter-2 figures re-rendered and checked (pages 13, 15, 17, 23, 25 of the PDF): no overflow,
  no edge over a node, diffusion row centred. Build clean, zero undefined references or citations.
- Decision recorded: product-graph prior (Kronecker sum, same eigenbasis V_T⊗V_S, precision adds instead
  of multiplies) is a one-line ablation for later; Kronecker/strong products would need a full NH×NH
  eigendecomposition and stay text-only.

**Campaign status (2026-09-11):**
- **GCN wave done 24/24** (jobs 286347–286370). Ladder now complete: GWN 48/48, MLP 24/24, GCN 24/24.
  GCN per-node CRPS kron vs iso: −1.0 % (pems04f) … −2.0 % (metrla), 6/6 datasets; every joint metric 6/6.
  Metrics JSON synced to `results/20260909_ladder/`; `aggregate_matched.py` + `make_figures.py` rerun →
  `ladder_results` / `ladder_prior` figures now show all three backbones (per-node gain MLP > GCN > GWN ≈ 0;
  CRPS_sum gain −5…−24 % MLP, −3…−7 % GCN, −3…−6 % GWN; total-coverage gain shrinks the same way).
- **Pooled-CRPS pass failed 0/48 — cause was NOT Kerberos:** the preflight wanted
  `residuals/<ds>/gwn/seed42/train.npz`, which had been rotated off DAIC to fit the MLP/GCN caches (8 GB
  windowshome quota). Fix: GCN caches removed from DAIC after md5 check of all 18 npz + 18 model files
  against the Mac copies (`results/residuals/<ds>/gcn/seed42/`); GWN train+test npz (≈2.9 GB) uploaded
  back. `rsync -a` fails on windowshome ("mkstemp … Operation not permitted") → use
  `rsync -r --inplace --no-perms --no-owner --no-group --no-times --size-only`.
  Submission script `scripts/daic/submit_pooled.sh` (12 jobs: 8×5 cells + 4×2 pems-bay cells,
  `EVAL_EXTRA=--tag pooled`), dry-run verified on DAIC; submitted once the upload is md5-verified.

**Pooled pass resubmitted (2026-09-11 12:1x):** GWN train+test caches md5-verified on DAIC (12/12 files) →
`submit_pooled.sh` → jobs 286645–286656 (8 "pooled" × 5 cells, 4 "pooled-bay" × 2 cells), monitor bf8810ej1.

## Step 3 — Literature review: DONE (2026-09-11), awaiting the user's read

Followed the approved proposal; the chapter went from 5,061 to ~5,600 words (one new subsection + one table).
1. **New §3.2.1 "How these forecasters are evaluated"** + Table 3.1 (model × per-node CRPS × CRPS of sums ×
   coverage/spread–error): DiffSTG/USTD/TSGDiff per node only, HySRD coverage only in a figure, SpecSTG's
   stated refusal of CRPS_sum, TimeGrad/CSDI CRPS_sum (score introduced by GP-Copula), SSTVM CRPS_sum on
   PeMS08/air quality, GenCast pooled CRPS + spread–error. Closes with: a change of the corruption is
   invisible in every published table of the graph line; Ch. 5 keeps per-node scores and adds the standard
   of the other lines (total, cumulative, pooled; coverage; spread–error).
2. **§3.5 positioning** rewritten to the step-0 claims: the gap is the corruption (isotropic) AND the
   evaluation (per node); the last paragraph names the three claims Ch. 5 tests (nothing lost per node,
   network-level scores improve, gain grows as the backbone leaves more structure). No result numbers in Ch. 3.
3. **§3.4** + one sentence: product-graph GMRF (Kronecker sum) as the other structure, "no diffusion
   forecaster uses it either, Ch. 5 compares the two at the level of the prior" → **commitment: the
   Cartesian-product prior row (prior level, Mac CPU) must be added to the prior-fidelity analysis.**
4. Figures 3.1 (structure entry) and 3.2 (gap map) redrawn in the shared tikz style; confirmed on renders
   (pages 30, 36). Table 3.2 untouched.
5. Claims verified online (the PDF folder ~/Desktop/tudelft/thesis/papers is not readable from this
   session: macOS blocks it): **DR paper** (arXiv 2301.06650) wraps STGCN, Graph WaveNet, AGCRN, ASTGCN,
   D2STGNN on PEMSD7(M) speed + PEMS08 flow; error covariance Σ_Q⊗Σ_N + σ²I; bilinear AR on the error
   field one hour (Δ=12) or one week (Δ=2016) earlier; trained jointly; reports CRPS + ρ-risk → the old
   TODO resolved, "one seasonal period" → "an hour or a week", "proved Kronecker-separable" → "showed a
   Kronecker-structured error model improves CRPS". **SpecSTG** (arXiv 2401.08119): autoregressive ✓,
   "We do not adopt CRPS_sum because sensors are not regarded as features" ✓, PEMS04/08 flow+speed, 60 min.
   **Not re-verifiable now:** HySRD-Net's ablation percentages (~10 % CRPS from per-sensor normalisation,
   16–17 % from stage-one error) — taken from the PDF in an earlier session; ScienceDirect blocks fetching.
   → ask the user to copy the literature PDFs into a readable folder (e.g. `papers/` in the repo, ignored).
6. Removed: the "UGnet as capacity reference in the experiments" claim (no UGnet run in the matched
   campaign — old data), the DiT-theory aside (fu2025ditgp), "deliberately", "of course", "orthogonal",
   "consequential", "mature field", "cleanest".

**User read 1 of Ch. 3 (2026-09-11):** "we need to make this chapter significantly better … too simple … the
first diagram doesn't need any equation-like symbols and is confusing … the two tables don't tell much … not
sure the evaluation literature belongs in a literature review … figure 3.2 and table 3.2 unnecessary, maybe a
tree diagram dividing into categories … make the whole section connected, only information relevant to our
contribution." Changes:
- **New Figure 3.1 = the chapter's map:** a category tree (root: probabilistic forecasting of a signal on a
  sensor graph → *where does the uncertainty come from?* (perturb the model | generative model) → *what is
  generated?* (the signal | the residual) → *how is it corrupted?* (white | structured)), leaves list the
  systems, the marker ●/○ after each name = scored on sums / per sensor only; highlighted leaf = this thesis;
  each branch labelled with the section that covers it. Replaces the gap-map figure AND the design-axes
  table AND the evaluation table. Chapter intro rewritten to walk the tree; §3.5 opens with "the path
  through Figure 3.1 summarises the review" and closes on its empty leaf.
- **Figure 3.2 (where domain knowledge enters) redrawn in words only** — noise field → denoiser network →
  one forecast sample → ensemble by repeating; past observations + road graph → encoder → denoiser; labels
  (i) what the network is told, (ii) how it is built, (iii) the noise it starts from and is trained to
  remove, "this thesis works here" — moved from §3.1 to the opening of §3.4 where corruption is discussed.
- **Evaluation literature kept, but as one paragraph** ("What these systems report", end of §3.2) and as
  the ●/○ marker in the tree; the table is gone. Why it stays: a literature review positions the work, and
  the position here is that the graph line's published tables cannot see the choice we make (per-node
  CRPS is blind to dependence, §2.1.5); we add no new metric, we adopt the multivariate standard of
  TimeGrad/CSDI/SSTVM/GenCast. That is the justification of Ch. 5's protocol and belongs here.
- No result numbers in Ch. 3 (unchanged rule). Words 5,624 → 5,250. Chapter overview no longer forces a
  page break, so the tree sits with the intro and §3.1 follows on the same page.
- Rendered and checked: tree (page 30 of the PDF, fits 78.8–516.3 pt), mechanism figure (page 34, 91.8–509.7 pt).

**User read 2 of Ch. 3 (2026-09-11):** "remove figure 3.2 … the tree looks too over-populated and confusing …
remove the scored-by things … just important things for us … content much better written, good connection to
the background and later … confirm that each paper is used correctly, confirm them by reading." Done:
- **Chapter rewritten from scratch (5,250 → 3,290 words), five sections, one figure.** Every section now
  opens from a Chapter-2 section and closes towards Chapter 4/5: §3.1 (backbones, §2.4.2/§2.1.3; non-generative
  UQ gives one error bar per entry, §2.1.5) → §3.2 (diffusion forecasters; horizon handling §2.1.7; denoiser
  fixed §2.6; "what these systems report" paragraph → Ch. 5 protocol) → §3.3 (residual, §2.1.6; DR's
  Kronecker covariance = §2.3's separable form, placed in the stationary law in Ch. 4; ResCAL → history
  conditioning in Ch. 4) → §3.4 (white noise's cost §2.6.1; horizon: schedules vs covariance → Σ_T of Ch. 4;
  graph: IHD/blurring ↔ our eigenbasis §2.3, GiffCF, GAD §2.6, SPDE ↔ GMRF §2.2.5, product-graph GMRF §2.3.1 →
  Ch. 5 prior-level comparison; Double-Diffusion = heat as mean, ours = heat as covariance) → §3.5 (the empty
  leaf; the three claims Ch. 5 tests).
- **Figure 3.1 (tree) simplified:** markers and legend removed, ≤5 names per leaf, the path to the thesis leaf
  drawn in the accent colour; questions carry the section numbers. Figure 3.2 (mechanism) removed; its one idea
  (three places where knowledge can enter: what the network is told / how it is built / the noise) is one
  sentence in §3.4. No tables.
- **Papers dropped as not important for the contribution:** PriSTI, GCRDD, TSGDiff, LaDCast, PSTDT, DiT,
  fu2025 (DiT theory), DiffSTOCK, TSDiff, STVNN, DoubleDiffusion-on-meshes (wang2025meshdd), STAEformer kept
  only as "attention line". 26 citations remain.
- **Verification by reading (sources fetched online; Desktop PDF folder is blocked by macOS for this session):**
  DiffSTG (arXiv 2301.13629: first DDPM for STG, non-autoregressive, UGnet, CRPS+RMSE) ✓; USTD (2310.17360:
  shared pre-trained encoder + attention decoders for forecasting and kriging, one-shot, isotropic, MAE/RMSE/CRPS) ✓;
  SpecSTG (2401.08119: autoregressive, spectral, "do not adopt CRPS_sum because sensors are not regarded as
  features") ✓; DR (2301.06650: STGCN/GWN/AGCRN/ASTGCN/D2STGNN, Σ_Q⊗Σ_N+σ²I, bilinear AR on errors 1 h or 1 week
  earlier, joint training, CRPS) ✓; CorrDiff (2309.15214: UNet mean trained first, diffusion on r = x − μ̂) ✓;
  DiffCast (2312.06734: end-to-end joint, residual in segments, isotropic) ✓; GenCast (2312.15796: residual
  w.r.t. last state, graph transformer on icosahedral mesh, 12 h steps, CRPS/spread-skill/rank histograms/
  spatially pooled CRPS because per-cell scores "do not measure the accuracy of larger-scale spatial structures") ✓;
  Double-Diffusion (2506.23053: heat prior = residual target + conditioning, Resfusion warm start, corruption
  standard, 4 air-quality/traffic networks) ✓; Resfusion (2311.14900: reverse starts from the degraded input) ✓;
  ResCAL (2209.05406: future errors estimated from previous errors + graph signals, real-time correction) ✓;
  GiffCF (2311.08744: heat-equation smoothing on the item graph as forward process) ✓; ERDM (2506.20024: noise
  increases with lead time, initialisation for the first window, weather + Navier–Stokes) ✓; TSGDiff (TRC 2025
  abstract: graph-attention U-Net, horizon as input, point-to-point) ✓ but dropped.
  From earlier PDF reading (recorded in the evidence table): HySRD-Net coverage-only-in-a-figure, SSTVM CRPS_sum
  on PeMS08/AIR-BJ, TimeGrad/CSDI CRPS_sum. **Removed because not re-verifiable now:** HySRD-Net's ablation
  percentages (per-sensor normalisation, stage-one error) and the "oracle trend" sentence. → still ask the user
  to copy the literature PDFs into a readable folder.
- Build clean, 0 undefined references; tree confirmed on render (page 30, fits 75.7–519.4 pt).

**User read 3 of Ch. 3 (2026-09-11):** "better but some of the boxes are glued together … make the overall thing
look better. Also: if we mention the other residual forecasters like DiffCast, CorrDiff, why aren't we using them
as comparisons in our experiments?"
- Tree re-laid out: the non-generative branch is one taller box at level 1 (method names + the "one error bar
  per entry, no joint samples" note inside it), so levels 2–3 have the full width; leaves 2.55 cm wide with
  0.5 cm gaps, equal heights; the two level-2 boxes moved apart (x = 5.6 / 12.3); the first question sits to
  the right of the root's vertical path. Confirmed on render (page 30, 83–512 pt, no overlaps).
- **Why the residual forecasters are not baselines — answer written into §3.5:** the comparison is controlled,
  not broad. (i) The white-noise arm IS the two-stage residual diffusion of CorrDiff/HySRD-Net rebuilt on our
  backbone, denoiser and training — the two arms differ in nothing but the noise. (ii) CorrDiff and DiffCast are
  image models (UNets) for weather fields; no graph version, no sensor-network results; porting them would
  reproduce arm (i). (iii) HySRD-Net has no public code (checked earlier) → compared through its published
  per-node numbers (SOTA table). (iv) DR is linear-Gaussian with a Kronecker covariance → our prior used
  WITHOUT a denoiser is that kind of model → **commitment: add a "prior alone (no denoiser)" row to Ch. 5**
  (Mac CPU: sample R ~ N(0, Σ_kron) per cell on the test set, score with the same metrics; extend
  `gaussian_no_diffusion_baseline.py`). Running DR's own code would be a new GPU campaign — not planned.
- Folder rename by the user: `tudelft_thesis_only/` → `tudelft_thesis_paper/`; added to .gitignore (still the
  live, uncommitted draft). Build path updated in this session.

## Step 4 — Methodology (proposal 2026-09-11, awaiting the user)

Source of truth: `docs/method.md` (what the 20260907_matched code does). The existing chapter (7,139 words) describes
the OLD pipeline (γ = 0.8 fixed, no global edge, no matching, iso "by flattening the spectrum", no derived floor,
no DC rule) → rewrite section by section, reusing only what still holds. Workflow per section: (1) plain-words
explanation of what we do and why, (2) the section written with its figure/algorithm, rendered and checked,
(3) one line "what Chapter 5 reports about this", (4) notes. User reads each section before the next.

| § | content | figure | algorithm | what Ch. 5 must report |
|---|---|---|---|---|
| 4.0 | overview: the pipeline; principle "only the noise changes" (two arms) | pipeline figure (redrawn, shared style) | — | the two arms, three backbones, six datasets |
| 4.1 | the residual field: backbone, R = Y − Ŷ, mask, per-sensor scale r = R/σ_i; what structure it carries (horizon growth, neighbour correlation, network-wide share) | "three faces of the residual" from the prior-analysis JSONs (GWN) | — | R1 residual structure (multi-scale variance ratios), separability 91–99 % |
| 4.2 | the prior (the contribution): white noise control; Kronecker meaning + toy; temporal factor from training residuals; spatial factor = heat-kernel GMRF on road graph + global edge; unit diagonal; (β,γ) by matching the spread of the network total; joint eigenbasis, θ_{k,h}, energy match to iso; what it does not claim | (a) Σ_S correlation from one sensor on the road graph, road-only vs road+global edge; (b) Σ_T (variance growth + horizon correlation); (c) mode-variance map over (graph frequency × horizon), kron vs flat iso | Alg. 1 BuildPrior | R2 prior fidelity (spread ratio at every scale), NLL, two-moment variant, Cartesian-product prior (prior level), (β,γ) per dataset (appendix) |
| 4.3 | forward process: per-mode OU, closed form, FCPS schedule, log-uniform noise levels, derived floor f(c_lo) = 1.3e-3 | erasure order per mode (DC last, checkerboard first) + f(c̄) with both floors | Alg. 2 Corrupt(r, c̄) | derived floors/steps per dataset (appendix); "43–62 % un-denoised" under the old floor as the reason |
| 4.4 | denoiser + training: 17,305-param GCNN (graph taps + temporal conv + FiLM), conditioning [X_z \| Ŷ_z], SNR-weighted masked MSE, AdamW, EMA, checkpoint = best val CRPS (fair, K = 16) | denoiser block diagram (words) | Alg. 3 Train | training curves, checkpoint selection; identical denoiser in every cell |
| 4.5 | sampling: exact per-mode ancestral step with the x₀-posterior variance; step grid 16/decade; DC rule (ablation only); why not Euler–Maruyama | one real reverse chain on pems08f (Mac CPU, local checkpoint): network-wide pattern first, detail last | Alg. 4 Sample | chain diagnostic (oracle / learned / hybrid), DC-rule ablation, steps sweep |
| 4.6 | evaluation protocol: per node (CRPS, MAE, RMSE, PICP, Winkler, energy), sums (network total, cumulative, pooled clusters 64/16/4: CRPS_sum, PICP, spread–error), fair CRPS, K = 100, y = 0 masked; comparisons: iso arm, prior-alone (no denoiser), published per-node numbers; ladder MLP/GCN/GWN | "what is compared" grid (arms × backbones × datasets) | — | the whole of Ch. 5 |
| 4.7 | cost (short) + conclusion | — | — | — |

Feasible now: 8 pems08f checkpoints (17k params) are on the Mac → real reverse-chain figure on CPU; prior JSONs
(18 files) → prior figures; `cell_config_preview.json` → derived numbers. Commitments carried in: prior-alone row,
Cartesian-product prior at prior level (both Mac CPU).

**User feedback on the Step 4 plan (2026-09-11):** "do not report the global edge (keep in code) · what is the
94 % captured, is it relevant · explain why we deviate from vanilla spatio-temporal (product graphs), I am not
sure myself · much less on choosing γ, β — remember GAD: say why γ matters, too small / too large · no tails ·
no repetition of background · focus on the Kronecker part · the derived floor is confusing, just how we choose
numbers, does not belong · better explanations and equations in the style of Mazzola's thesis and the GAD paper
· build it in the draft, one by one."

Decisions taken:
- **Global edge:** cannot be dropped from the description because it is in the model that produced every
  result (β = 0.003–0.03 on all six datasets; without it the prior's spread on the network total is ~0.5 of
  the truth, which is the very quantity the network-level wins rest on). Minimal honest treatment: ONE sentence
  in §4.3.2 ("the road graph is regularised with a weak link between every pair of sensors before the
  Laplacian is formed, which lets the prior carry the shared network-wide part of the error; weights and the
  selection of (β, γ) in Appendix B"), no β in the main text otherwise. Flagged to the user.
- **"91–94 % captured":** the nearest-Kronecker-product test — how much of the residual's empirical
  covariance one Σ_T ⊗ Σ_S can reproduce. It is the data-side justification of the separable form (RQ1),
  belongs in Ch. 5 (R1) as one number; in Ch. 4 only as a pointer.
- **Why separable (Kronecker product) and not a product-graph GMRF (Kronecker sum):** written into §4.3.1:
  (i) the temporal structure of forecast errors (variance growing with lead time, correlation between
  horizons) is a property of the forecaster, not of any graph — no temporal path Laplacian gives it, the
  training residuals give it directly; (ii) DR showed a Kronecker product describes traffic residuals;
  (iii) the eigenbasis is the product of the two factor bases, so GAD's per-mode machinery carries over
  unchanged; (iv) the product-graph GMRF ties both axes to one γ — Ch. 5 compares the two at prior level.
- γ, β selection, derived floor, step counts, tails: OUT of the main text → Appendix B (one line each in Ch. 4).
- Style model: Mazzola (TU Delft 2020, Isufi): Overview → components as subsections → architecture →
  numerical insight → conclusion; GAD §3–4: forward SDE (4) → closed-form marginal (5) → FCPS → γ
  (too small: low modes decay too slowly; too large: geometry lost) → stationary GMRF → Remark 1 (per-mode
  decay = future work, which is exactly what our Kronecker prior does) → reverse SDE (6) → Tweedie (7) →
  MMSE denoiser (8) → GSP filter (9–12) → GCNN (16) → loss (17). Copies fetched to the scratchpad.

Final Ch. 4 structure: 4.1 Overview (pipeline figure) · 4.2 The residual as a graph-time signal · 4.3 A
space-time prior for the residual (4.3.1 from GAD's prior to a space-time prior: the design choice; 4.3.2
spatial factor; 4.3.3 temporal factor; 4.3.4 the prior in its eigenbasis + Alg. 1 + 3-panel figure) · 4.4
Diffusion on the space-time prior (4.4.1 forward + Alg. 2 + erasure figure; 4.4.2 reverse process and denoiser
+ Alg. 3; 4.4.3 sampling + Alg. 4 + real reverse-chain figure) · 4.5 Evaluation protocol · 4.6 Conclusion ·
Appendix B derived numbers and selection procedures.

**Round 1 written (2026-09-11): §4.1 Overview + §4.2 The residual as a graph-time signal** (1,224 words,
new file; old chapter kept as `4_methodology_old.tex`, not compiled). Pipeline figure redrawn in the shared
style: top band "once per dataset: build the prior" (road graph → Σ_S, training residuals → Σ_T → prior, with
the white-noise control beside it), bottom band "per forecast window" (history → frozen backbone → point
forecast → + → forecast samples; residual diffusion starting from a draw of N(0, Σ) → residual samples;
conditioning dashed), the prior feeding the diffusion box. §4.2: residual (Eq. 4.1), mask, per-sensor
scale (Eq. 4.2), vec order, and the three structures with PEMS08-flow numbers computed on the Mac from the
rebuilt prior/residuals (variance 0.78→1.18 over the hour; adjacent-horizon correlation 0.27→0.48 — NOTE:
method.md's "adjacent ≈ 0.9" is wrong for pems08f/GWN, corrected here; neighbour correlation 0.14 measured;
constant-mode share 2.5–6.5 % vs 1/N). The 33 undefined references in the build are Ch. 5–7 pointing at the
old chapter's labels; they resolve when those chapters are rewritten.

**Round 2 written (2026-09-11): §4.3–4.6 + Appendix B — the whole chapter is drafted (4,850 words, 5 figures, 4 algorithms).**
User asked for: good figures, a connected story with references instead of repetition, the denoiser's job
in graph-filter terms, everything checked against the code, figures confirmed on screenshots, creativity.
- **Checked against the code** (`prior.py`, `forward_fcps.py`, `denoiser.py`, `losses.py`, `reverse.py`,
  `train_diffusion.py`, `eval_diffusion.py`, `crps.py`, `intervals.py`, the pems08f train config):
  conditioning = [X_z | Ŷ_z] split into channels; denoiser = GCNN, K = 5 hops on S = D̃^{-1/2}(A+I)D̃^{-1/2}
  of the plain repaired road graph (**same graph in both arms — method.md's "S of A_β for kron" was wrong,
  corrected**), 3 layers, 24 channels, depthwise temporal conv (1×3), FiLM on a sinusoidal τ embedding
  (t×10), SiLU, residual output r_c̄ + head; 17,305 params; loss = masked MSE with w = 1/(1+2c(t));
  c̄ ~ logU[c_lo, c_0]; AdamW 1e-4/1e-4, batch 32, clip 1.0, EMA 0.999, val every epoch (512 windows,
  K = 16, ancestral at the derived steps), best_crps.pt, patience 40, ≤ 200 epochs; sampler = exact per-mode
  ancestral step with the ς² (x₀-posterior) term, 16 steps/decade (63 iso, 87–96 kron), DC rule optional;
  eval: α = 0.1 (90 % intervals), normalised quantile CRPS (19 quantiles), MAE/RMSE of the ensemble mean,
  Winkler, energy distance, sums scored with the fair CRPS (K(K−1)), PICP, spread/error; K = 100; y = 0 masked.
  θ rescale to θ_min = 0.8 = GAD's γ; σ² = mean r²; ridge ρ = 0.1; derived floor f(c_lo) = 1.3e-3; β,γ grid
  {0.003…0.3} × 40 log-spaced γ ∈ [1e-3, 3]; match on mean_h ratio of the total's sd.
- **GSP reading written into §4.4.2:** for the prior alone, E[C₀ | C_c̄] = e^{−θ c̄} C_c̄ per mode (stationary
  OU reversibility) = GAD's optimal filter for the space–time prior: graph low-pass at large c̄ → all-pass
  at small c̄; for white noise a scalar gain. The learned denoiser = this filter + what the conditioning adds.
- **Figures (all real PEMS08-flow data, computed on the Mac):** 4.2 the prior (correlation vs hops:
  data 0.14→floor 0.045 vs prior 0.33→0.022; Σ_T; mode-variance map); 4.3 forward corruption of test window
  3189 under both priors at c̄ ∈ {0, 0.01, 0.1, 1, 7} + survival curves (white noise = the θ = 0.8 curve for
  every mode); 4.4 the optimal filter vs graph frequency at five c̄; 4.5 one real reverse chain of the trained
  kron model (seed 42, epoch 49, 87 steps; state and estimate at 6 levels vs the true residual; verified
  equal to `reverse_sample_ancestral` with the same generator). All confirmed on renders; 4.2 needed a
  constrained layout (labels clipped), 4.3 a wider spacer.
- **Algorithms:** 1 BuildPrior, 2 Corrupt (one-shot), 3 Train, 4 Sample — each line points at its equation.
- **Appendix B (`app:derived`) written:** graph repair + regularising link, (β,γ) matching rule, floor
  rule f(c_lo) = 1.3e-3 and n = ⌈16 log10(c_0/c_lo)⌉, Table B.1 from the six train configs + rebuilt priors
  (identical to method.md §9).
- Honest sentences kept: the prior over-states neighbour correlation (0.33 vs 0.14, "the price of matching
  one scale"); the regularising link in one sentence; "43–62 % un-denoised at a fixed floor" in the appendix.
- Remaining undefined refs (26) are Ch. 5–7 → old labels; resolved when those chapters are rewritten.

**User read 1 of Ch. 4 (2026-09-11):** "no analysis figures in the methodology (that is for the experiments)
· is the sampler basically DDPM or something different? make sure nothing is mathematically wrong · Fig 4.1
has overlapping boxes/text/lines · better visuals of our actual method, with the graph · some equations wrong
or under-explained, e.g. the variance stuff · maybe things not needed, maybe important info skipped."
Round 3 (2026-09-11):
- **All data-analysis figures removed from Ch. 4** (the four real-PEMS08 figures stay in
  `figures/generated/m_*_pems08f.pdf` for Chapter 5: prior fit, forward corruption of a real window,
  optimal filter on the real θ, one real reverse chain of the trained model). Replaced by **schematic figures
  on a seven-sensor toy graph** (same layout as Fig. 2.1; node shades = signed values computed from a toy
  prior with γ = 0.3, H = 3, θ ∈ [0.8, 177]; shades written by a script to `figures/generated/toy/*.tex`):
  4.2 a draw of the Kronecker prior vs a draw of white noise at three lead times + how the prior is assembled;
  4.3 forward corruption of one toy field at c̄ = 0, 0.05, 0.5, 5 under both priors; 4.4 analytic curves
  (survival e^{−θc̄} for θ = 0.8/8/80; optimal-filter gain vs graph frequency at c̄ = 0.3, 3, solid vs
  dashed white noise); 4.5 the denoiser as a block diagram; 4.6 the reverse process on the toy graph with the
  prior's own filter as denoiser (state and estimate at c̄ = 7, 1, 0.1, 0.001).
- **Sampler section rewritten and verified.** Plain statement: it IS DDPM ancestral sampling, per mode, on
  the continuous OU process (DDPM's chain is the discretisation of its own VP process). Eq. 4.13 defines the
  three (shrinkage, variance) pairs, Eq. 4.14 the DDPM posterior q(x_{c'} | x_c, x_0) (mean and v), then the
  variance question: DDPM's two standard choices (β̃ = v: x̂₀ treated as known; β = s_d: x₀ as uncertain as
  under the prior) bracket the truth; ours carries the prior-posterior variance ς² of x₀ | x_c through the
  mean → Eq. 4.15. **Verified numerically (random θ ∈ [0.8, 3000], c, c'):** the corrected variance equals
  the exact reverse transition σ²(1−a_d²)/θ to 1e-6, and with x̂₀ = the prior's filter the mean equals
  a_d x_c; DDPM's β̃ is up to 27 % too small for slow modes at small θc. Euler–Maruyama would need steps ∝
  θ_max. The optional DC rule kept as one sentence (ablation).
- **Fig 4.1 cleaned**: wider bands, the prior→diffusion line routed around the right end of the forecast
  row (no crossings), labels moved off the lines.
- **Equations tightened:** forward marginal now read line by line (mean shrinks by e^{−θc̄}, variance grows
  from 0 to σ²/θ); the σ² convention stated as energy matching; θ_min = 0.8 explained as a choice of units
  for c̄ (GAD's schedule constants transfer) that does not change the prior; dataset numbers removed from
  §4.2/§4.3 (qualitative + "Chapter 5 measures"); the optimal filter derivation stated (stationarity ⇒
  E[C₀ | C_c̄] = e^{−θc̄} C_c̄).
- Build note: the renamed draft folder corrupts latexmk's aux/out files (NUL bytes; sync-drive artefact?)
  → the thesis is now built in the scratchpad (`scratchpad/thesis_build`, rsync of the folder) and the PDF
  copied back to `tudelft_thesis_paper/build/report.pdf`.
- **Pooled-CRPS pass DONE 48/48** (jobs 286645–286656); `_pooled.json` files synced to
  `results/20260907_matched/`. Example pems08f kron s42: CRPS on cluster sums c64/c16/c4 = 19.4/50.0/127.4,
  coverage 0.85/0.85/0.85, spread ratio 0.88/0.85/0.69 (total: 437, 0.71, 0.62). → to aggregate iso vs
  kron over 4 seeds and add "advantage vs aggregation scale" (R4b) in the results session.

**User read 2 of Ch. 4 (2026-09-11):** "the graph examples are a nice idea, but colours and details need to be
better; two figures are not clear how they work; words and boxes must not overflow." Round 4:
- Toy prior made visibly smooth (γ = 0.15, θ ∈ [0.8, 333]); node colours now scaled per figure to the largest
  value (saturated), and a colour-scale bar (negative → 0 → positive) added to every graph figure.
- Fig 4.2: the shown Kronecker draw is a real draw chosen among 60 seeds at the 90th percentile of
  smooth-mode energy (seed 49), so the features named in the caption are visible; assembly column reworded
  ("how lead times are related", "how sensors are related", "one variance per space–time pattern").
- Fig 4.3 redesigned: instead of random corrupted draws it shows **what survives** of a constructed field
  (left-to-right trend + alternating fine detail) after heat c̄ = 0.05, 0.5, 5 — the MEAN of the forward
  marginal — under both priors, with a note under every panel (fine detail gone / trend kept / trend fading /
  pure prior noise vs. everything fades at the same rate); caption says the corrupted field is this plus
  prior-shaped noise (Fig 4.2).
- Fig 4.6: notes under every column (nothing decided → network-wide lean → neighbourhoods → every detail =
  the sample); chain seed chosen the same way (seed 39).
- Overflow: Fig 4.2 was 11 pt over the text edge → graph spacing and column tightened (now 89–519 pt);
  all six chapter figures measured inside 74–521 pt. Build in the scratchpad, PDF copied back.

