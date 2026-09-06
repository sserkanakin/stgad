# GTCNN product-graph campaign — report (2026-07-28)

**Verdict: inconclusive on the science question, and the reason is diagnosable. The
operator machinery is verified correct; the training/sampling recipe on the raw-signal
arm is not fit for purpose, and two independent defects make the arm comparison
untrustworthy as it stands.**

30 cells = 5 priors × 6 datasets, raw signal, history-only conditioning, GCNN denoiser
fixed at 17,305 params in every cell, γ=0.1, c₀=100, FCPS for all arms, seed 42.
16 jobs, 0 failures. Gate passed 30/30 on both clusters.
Aggregate with `scripts/analysis/pgraph_aggregate.py`.

## 1. What is solid

- **The operator is what we claim.** Cartesian and strong GSOs reproduce the product
  graph's edge set exactly; all products single-component on all 6 datasets; θ ∈ [0.1,2.1]
  with identical θ_min, stiffness 21 and mixing 20 in every cell.
- **The forward process reaches its designed stationary law** (C1/C2 within 5%, C3
  confirms the eigenbasis). c₀=100 was derived, not guessed: the raw signal's
  graph-DC × time-DC mode carries 49–670× its stationary variance, and the previous
  campaign's c₀=25 left 61% of it un-erased.
- **The parametric fit is unanimous.** Given GTCNN eq.10's four free coefficients on all
  six datasets and two independent clusters, the fit puts **spatial coupling at exactly
  zero** every time (s₀₁=0, s₁₀=1, s₁₁=0). `prod_param` is therefore a *temporal-only heat
  prior selected by the data*, not a spatio-temporal one.

## 2. Defect A — early stopping gave arms unequal training budgets

Best checkpoints land at **epoch 2–7** on most cells while train loss is still falling, and
early stop fires 15 epochs later. Worse, the budget is **not matched across arms**:

| dataset | iso | cart | strong | tensor | param | ratio |
|---|--:|--:|--:|--:|--:|--:|
| pems04f | 19 | 19 | 19 | **126** | **49** | **6.6×** |
| pems08f | 27 | **88** | **114** | 27 | 27 | **4.2×** |
| metrla / pemsbay / pems04s / pems08s | 18–24 | 17–31 | 17–23 | 17–21 | 18–23 | ≤1.6× |

**The two datasets where structured priors "win" are exactly the two where they trained
3–6× longer.** Excluding them, at the protocol setting (ancestral@50):

| arm | all 6 | budget-matched (4) |
|---|--:|--:|
| cart | 2/6 | **1/4** |
| strong | 2/6 | **1/4** |
| tensor | 2/6 | **0/4** |
| param | 2/6 | **0/4** |

The single matched win is cart on pems08s at +2.5% — one seed, sub-3%.

## 3. Defect B — the two samplers disagree, and neither is trustworthy

Mean coverage against a nominal 0.90: **ancestral@50 = 0.108**, **EM@250 = 0.907**.
The headline sampler is producing near-deterministic samples, so its CRPS is effectively a
point-accuracy score. Under the *calibrated* sampler the ranking largely inverts:

| arm | ancestral@50 (cov 0.11) | EM@250 (cov 0.91) |
|---|--:|--:|
| cart | 1/4 | 1/4 |
| strong | 1/4 | 2/4 |
| tensor | 0/4 | **3/4** |
| param | 0/4 | **4/4** (+3.2 to +4.6%) |

So: under ancestral, prior fit **anti**-predicts sample quality (param fits best on 6/6 and
samples worst); under EM, fit **predicts** it (param wins 4/4). Same checkpoints, same data.

Do **not** read this as "EM is right". EM@250 runs at only 1.2× its stability floor (210)
and its CRPS is ~3× worse than ancestral's — consistent with over-dispersion that happens
to land near nominal coverage. The honest statement is that **the sampler, not the prior,
determines the conclusion**, which means neither ranking is evidence about priors yet.

## 4. Few-step behaviour

Every arm gets **worse** with more steps (CRPS best at 5, degrading to 100) — backwards for
a diffusion sampler and a direct consequence of the under-dispersion. iso degrades most
(−7.5% from 5→100), structured arms are flatter (−1.5 to −4.6%). The E3b "structured
degrades less at few steps" pattern is visible but is contaminated: it reflects iso being
unusually good at 5 steps rather than structured priors being good there.

## 5. What to fix before this means anything

1. **Match the training budget across arms.** Fixed epoch count, or early stop on a
   common schedule. Without this no arm comparison is interpretable.
2. **Fix the under-dispersion.** Candidates, in order of suspicion: (a) the model is
   barely trained (epoch 2–7) and collapses to the conditional mean; (b) the LR plateau
   drops 1e-4 → 2.5e-5 within ~18 epochs, freezing it there; (c) the ancestral sampler's
   `c_lo=0.02` terminal floor interacting with c₀=100 — a 5000× geometric span versus the
   ~350× the default was tuned for.
3. **Then re-run.** The operator/gate work needs no changes; only the recipe does.

## 6. What is publishable as-is

The **methodology**: the derived-c₀ result (GAD's mixing rule is insufficient when a data
mode dominates the stationary variance — measured 49–670×, and the previous campaign was
under-mixed by it), the verified product-graph construction, and the unanimous parametric
fit collapsing spatial coupling to zero on 6/6 datasets. Those are measurements, not
artefacts of the broken recipe.
