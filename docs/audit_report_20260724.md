# Thesis audit — findings (2026-07-24)

Scope: full draft `tudelft_thesis_draft/`, title → conclusion. Every number below
was recomputed from the eval JSONs on disk (`results/`) or read directly from the
draft. Severity: **H** wrong fact/number/math/citation · **M** unsupported/misleading/missing ·
**L** style/language.

Status of the automated multi-agent audit: the 16-auditor parallel workflow was
launched but **all agents hit the account session limit (resets 11:30 Amsterdam)**
and returned nothing. Everything below is my own direct verification. The
exhaustive per-citation / background-math / symbol / figure-content sweep still
wants the parallel re-run once the limit clears; the number-provenance findings
(the load-bearing ones) are fully verified here.

---

## A. Load-bearing problem: experiments-chapter numbers do not trace to one protocol

The flagship heat results are clean. The **older main-grid and appendix tables are not.**

### A1 — Appendix B "Uncalibrated Five-Seed **Test** CRPS" is actually the **validation** table  · **H**
`appendix/appendix-b-results.tex` §"Uncalibrated Five-Seed Results" reports e.g.
PEMS08 flow/linear iso `0.2761±0.0177`, structured `0.2385±0.0202`;
GWN iso `0.0580`, structured `0.0574`. These are **byte-identical to the VAL split,
k=4 numbers** in `docs/results_bank.md` Table A. The real **test** 5-seed means are:

| cell | appendix says (=val) | actual test 5-seed mean |
|---|---|---|
| pems08f/mlp iso | 0.2761 | **0.2295** |
| pems08f/mlp emp | 0.2385 | **0.1978** |
| pems08f/gwn iso | 0.0580 | **0.0497** |
| pems08f/gwn emp | 0.0574 | **0.0502** |

Consequence: the caption "The structured prior has the lower mean in every cell" is
**false on the real test split** — on pems08f/gwn (and the other strong-backbone gwn
cells) the **isotropic** control has the lower test CRPS (0.0497 < 0.0502). Fix: replace
the table with true-test 5-seed means (below), or relabel it honestly as validation and
delete the "lower in every cell" claim. The `^{4}` footnote also says the lost seed was a
"node failure"; the record says it was a training miss on pems04s.

### A2 — Main comparison table `tab:exp_main` mixes arms, seeds, and experiments  · **H**
The "structured (STARDiff)" column is not one method. Reconciling each cell against disk
(recalibrated, seed 42 unless noted):

- pems08f/**mlp**: table `0.1865`. Not the covariance/empirical arm (recal `0.2141`),
  not `emprefit` recal (`0.1698`). `0.1865` is `emprefit` **raw seed 126**. The `+22.9%`
  headline margin therefore compares iso seed 126 (`0.2419`) against a *different* arm at a
  *different* seed, uncalibrated. With one consistent arm (empirical, recal s42) the margin
  is **≈ +11%**, not 22.9%.
- pems08f/gcn `0.1015`, pems04f/gcn `0.1149`, pems04s/gcn `0.0284`, pems08s/gcn `0.0243`:
  these match **ladder empirical recal s42**.
- pems04f/mlp `0.2504`: matches **ladder empirical recal s42** (not emprefit) — so mlp is
  sourced inconsistently between datasets.
- iso column mixes seeds: pems08f/mlp iso `0.2419` = seed 126; other cells = seed 42.

The methodology (`4_methodology.tex`, and `5_experiments.tex:296`) declares the main-grid
structured arm to be the **covariance/empirical variant**. The reshaped ("rsh") arm — which
is what actually produces the big weak-backbone margins — is a *different* prior and is not
introduced in the methodology at all. So the table both (a) draws from ≥3 sources and
(b) silently swaps in the strongest arm per cell.

**Single-arm reconciliation (empirical/covariance variant, one protocol):**

| dataset | bb | iso (recal s42) | empirical (recal s42) | honest margin |
|---|---|---|---|---|
| pems08f | mlp | 0.2411 | 0.2141 | +11.2% |
| pems08f | gcn | 0.1041 | 0.1008 | +3.2% |
| pems08f | gwn | 0.0495 | 0.0494 | ≈ |
| pems08s | mlp | 0.0429 | 0.0427 | +0.5% |
| pems08s | gcn | 0.0247 | 0.0243 | +1.6% |
| pems08s | gwn | 0.0173 | 0.0173 | ≈ |
| pems04f | mlp | 0.2569 | 0.2504 | +2.5% |
| pems04f | gcn | 0.1146 | 0.1149 | −0.3% (iso wins) |
| pems04f | gwn | 0.0661 | 0.0660 | ≈ |
| pems04s | mlp | 0.0508 | 0.0506 | +0.4% |
| pems04s | gcn | 0.0289 | 0.0284 | +1.7% |
| pems04s | gwn | 0.0203 | 0.0202 | ≈ |

Raw 5-seed test means (uncalibrated, all seeds) for the appendix replacement:

| dataset | bb | iso | empirical | reshaped |
|---|---|---|---|---|
| pems08f | mlp | 0.2295 | 0.1978 | 0.1642 |
| pems08f | gcn | 0.1057 | 0.1050 | 0.1060 |
| pems08f | gwn | 0.0497 | 0.0502 | 0.0523 |
| pems08s | mlp | 0.0430 | 0.0428 | 0.0426 |
| pems08s | gcn | 0.0248 | 0.0246 | 0.0249 |
| pems08s | gwn | 0.0174 | 0.0175 | 0.0177 |
| pems04f | mlp | 0.2790 | 0.2606 | 0.2473 |
| pems04f | gcn | 0.1158 | 0.1169 | 0.1211 |
| pems04f | gwn | 0.0663 | 0.0668 | 0.0714 |
| pems04s | mlp | 0.0506 | 0.0508 | 0.0528 |
| pems04s | gcn | 0.0289 | 0.0288 | 0.0298 |
| pems04s | gwn | 0.0205 | 0.0206 | 0.0211 |

Note the uncalibrated raw picture is *less* favourable than the recalibrated one:
on GWN, iso ≤ empirical on all four datasets. The thesis's claim that the structured
prior wins/ties everywhere holds under **recalibration**, not raw — the text must say which.

### A3 — "STARDiff" name appears only in ch5 tables/captions, defined nowhere  · **M**
`\newcommand{\method}{STARDiff}` sits at `5_experiments.tex:1`; the name is used only in
that chapter and never introduced in prose, the intro, or the title (which is "Graph-Aware
Residual Diffusion"). Either name the method properly at first use (intro/method) or drop
the name and write "our model".

### A4 — Verified-clean (no change needed)
- Heat table `tab:exp_heat`: every cell matches disk (pems08s/gwn cart-past 0.0170,
  cartL3 0.0171, cartLg2 0.0171; pems04f/gcn 0.1026; pems04s/gcn 0.0274; pems08f/gwn 0.0497;
  pems04f/gwn 0.0662; pems04s/gwn 0.0201; metrla 0.0433; pemsbay 0.0208 — thesis says 0.0209,
  rounding). ✓
- SOTA table "ours" rows = recalibrated seed-42 empirical run; RMSE/MAE/CRPS trace
  (pems04f 30.52/19.02/0.0660 ✓, etc.). ✓
- Coverage story (uncal 0.73–0.83 → recal 0.85–0.90) matches the diag fields. ✓

---

## B. Claims to tighten (verified against the draft)

- **B1 · M** "improves on all published baselines on all four benchmarks and all three
  metrics" (intro 49, conclusion 60, summary): true for the **hour-averaged** RMSE/MAE/CRPS,
  but the SOTA table's own per-horizon breakdown shows HySRD-Net keeps the bold **30-min
  RMSE on PEMS08 speed** (3.40 vs our 3.42). Add "in the hour-averaged scores" or note the
  single per-horizon exception.
- **B2 · M (provenance exists)** "training several times cheaper" = the `4–9× epochs-to-
  converge` result (`tab:exp_convergence`). The `validation_history.json` files carry
  `best_crps_epoch` + full epoch history, so this IS reconstructable — just recompute the
  table from them to lock the exact epoch counts (72.6/89.2/39.4/94.2 vs 10.6/10.0/9.8/11.2).
- **B3 · ✓ verified** Timing traces: `gen_seconds=611.0` for 3548 windows, `s_per_window=
  172 ms` at K=100 → **1.7 ms/sample-window** and ~10 min/test-split — matches the thesis.
  Only the DiffSTG "29 ms / 17×" side rests on the DiffSTG native run (`model-ugnet-
  notcomplete/.../DiffSTG.csv`); confirm that one number there.
- **B4 · M** Tiny-denoiser claim "4,233 parameters retains full-size accuracy"
  (`5_experiments.tex:426`): `results/20260722_tiny` exists but I did not find eval JSONs
  under it — confirm the 4,233 count and the 0.0572/0.0574/0.0581 numbers, or cut the claim.

## C. Appendix / methodology consistency

- **C1 · H** Appendix training-config table says "spatial shift γ: grid {0.04…0.8},
  val-selected". Chapter 4 says γ is **fixed by criterion** (0.8 cited on the road arm,
  0.05 by spectral matching on the learned arm); only the covariance-variant *sweep* used a
  grid. The appendix presents grid-selection as the shared config — contradicts ch4. Decide
  which is true per arm and make them agree.
- **C2 · L** Appendix config lists `c0=7` as the shared schedule constant, but ch4 says the
  learned arm forces `c0=50` and the covariance variant more. "Shared across all variants"
  is wrong for `c0` and `γ`.

## D. Citations / bib (spot-checked; full sweep pending parallel re-run)

- **D1 · ok** GAD entry is correct: Rozada, Vimal, Cavallo, Marques, Jamali-Rad, Isufi
  (2025), arXiv:2510.05036. Committee member Isufi is a co-author — good to keep exact.
  Verify the exact title string ("Graph-Aware Diffusion for Signal Generation") against arXiv.
- **D2 · pending** Every "[work] does/lacks X" attribution (DiffSTG ~1M UGnet, SpecSTG
  isotropic-despite-spectral, HySRD reshape-to-white, PriorGrad Prop. 2 entropy-matched
  control, Double-Diffusion heat-as-mean, DR Kronecker covariance) needs the paper-level
  check the parallel auditors were assigned.
- **D3 · pending** Baseline numbers in `tab:exp_sota` for DeepVAR/TransNVP/TimeGrad/GCRDD/
  PriSTI (the rows we did NOT run ourselves) need a source per row — confirm each is quoted
  from its paper or a common benchmark, not filled in.

## E. Language / structure (from my read; the prose is generally strong)

- The writing already reads like a well-edited paper — few AI-tells. Main items:
  heavy em-dash density in ch4–ch5 (stylistic, not wrong); "STARDiff (ours)" naming (A3);
  ch3 literature review is 137 lines and carries a lot — assess depth vs the reference-thesis
  template once numbers are settled.
- All 7 chapters have a Conclusion section ✓. `xchapter-2.tex` (template "About the
  Template") is present in `mainmatter/` but NOT `\input` in report.tex ✓. Chapter-label
  styles are inconsistent (`ch:`, `chap:`, `chapter:`) but all resolve.

---

## Decision that gates the experiments chapter (REFRAMED 2026-07-24)

The real problem is deeper than "which arm". The **methodology's method is the Cartesian
heat prior** (road + learned graph, GAD extension); the heat report says the
empirical/covariance arm "can be dropped at zero cost". Yet `tab:exp_main` (the main
comparison) uses that covariance/empirical arm, because the ladder data (`20260721_ladder`)
predates the heat campaign. So the chapter is two half-stories.

**Key numbers finding (verified):** the GAD-literal **road-heat** prior (γ=0.8) does NOT
produce the ladder law — on the weak backbone (pems08f/mlp) it *loses* to iso on test
(0.2314 vs 0.2295); on gcn it ties (0.1054 vs 0.1057). The big weak-backbone margins
(+22.9%) belong to the empirical/reshaped arms only. The **learned-heat** prior (cartL,
γ=0.05 — the stiff, aggressive, principled cousin of empirical) is the one that could
reproduce them AND be the actual method.

**And the learned-heat ladder is trained but unevaluated:** `cartL-fo` (gcn+gwn, 2 seeds)
checkpoints exist; nobody ran the eval. See `docs/heat_ladder_run_plan.md`:
- Step 1: eval the existing `cartL-fo` gcn+gwn checkpoints (fast on DAIC).
- Step 2: train+eval the missing `cartL-fo` **mlp** rung (never trained — the ladder-law cell).

**Plan once numbers land:** rebuild `tab:exp_main` as learned-heat (and/or road-heat) vs
isotropic across the backbone ladder — one prior, one protocol — and demote the
empirical/covariance variant to the appendix ablation. Then the whole chapter is one
GAD-connected heat story: ladder law + deployment wins + Wiener mechanism + raw-signal
control. This is what makes the methodology and experiments finally agree.

Do NOT apply the number/table fixes in §A until the learned-heat ladder is evaluated —
the correct table depends on it.
