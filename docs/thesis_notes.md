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
