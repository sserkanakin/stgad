# Run plan — everything still to execute

Written 2026-07-20, after the full code + paper + thesis audit. Supersedes ad-hoc launches.

**The claim we are now testing** (changed by the 20260720b_reshape result): the structured
Kronecker prior substitutes for denoiser capacity — its advantage grows as the backbone
weakens. Pre-registered in `prior_design_note.md:115,125` *before* the runs; first evidence
on pems08f is a monotone ladder **−11.3% (mlp) → −3.5% (gcn) → −1.5% (gwn)**.

Everything below exists to (a) put error bars on that ladder, (b) produce metrics that are
directly comparable to published papers, and (c) measure the two axes nobody else reports.

---

## Phase 0 — code fixes. BLOCKING. No GPU.

The audits found five things that change how results must be interpreted. Fix before any
further training, or the numbers are not defensible.

| # | Fix | Why it blocks |
|---|---|---|
| **F1** | **Drop `--stiffness-cap 50` from the `rsh` arm** | The cap clips the *loud* modes — i.e. exactly the ones the reshaped prior exists to match. Energy clipped: pems08f 2.8%, pems04f 3.4%, **metrla 35%, pems04s 61%, pems08s 65%, pemsbay 91%**. So `rsh` is a *different treatment on every dataset*, and the claim "Σ_stat = v_emp exactly" (`prior.py:383`) is false wherever the cap bites. The cap exists only for EM stability; **ancestral sampling has no such floor** (`reverse.py:11-13`), and this experiment is ancestral-only. Removing it is free. |
| **F2** | Fix the rescale gate `prior.py:462` to include `stiffness_cap is not None` | Latent: applying any cap to `emp` silently breaks variance matching (−12% on pemsbay). Would bite the moment we cap `emp` to equalise stiffness. |
| **F3** | **Delete the `iso_w` arm** | Proven a bit-for-bit duplicate of `iso` (loss diff 2.3e-12, grad diff 3.3e-15) — with `V_S=V_T=I` and constant θ the whitening cancels algebraically. It is a tautology, not a control. Frees 3 jobs per dataset-seed. |
| **F4** | Equalise the epoch budget across arms | `emp_w` early-stopped at epoch 18 vs `iso`'s 200 (patience-15 on a spurious epoch-3 CRPS minimum) while its loss was still improving. Comparing a truncated arm to a full one is not a loss comparison. Raise patience to ≥40 or run a fixed 200 for all arms. |
| **F5** | `whitened_mse` mask denominator | **DONE** — was scaling the loss by the observed fraction, silently shrinking the effective LR on metrla/pemsbay only. |
| F6 | Guard `--noise-sampler loguniform` on UL/VP schedules | `AttributeError` on step 1 (`c_lo` undefined). Current grid dodges it; a stray `--schedule auto` would hit it. |
| F7 | Docstrings: `prior.py:19-20` (`V_S⊗V_T`→`V_T⊗V_S`), `:244` (stale winsor claim), `:383` (add "when uncapped") | Prose wrong, code right. |

**Verified correct — no action:** Kronecker/vec convention (exact to 3.8e-15, transposed
variant excluded), forward OU closed form, ancestral posterior (2.3e-16), Parseval in the
reshaped transform, Laplacian self-loop stripping + cache integrity on all 6 graphs,
per-node normalization and its inversion, prior fit in the same space the diffusion runs in,
and — importantly — **the old schedule confound is properly closed** (iso is pinned to FCPS,
not `auto`).

---

## Phase 1 — headline ladder with error bars

The core claim. Two datasets, both PEMS08 (flow + speed), both already on DAIC.

- arms: `iso`, `emp`, `rsh`, `emp_w`, `rsh_w` (5) — `ddpm` once per cell at seed 42 as a
  reference point only. **`iso_w` deleted** (algebraic duplicate of `iso`, see F3).
- backbones: `mlp`, `gcn`, `gwn`
- seeds: **42, 43, 44**
- **48 runs per dataset** (45 + 3 ddpm) ≈ 2.3 h; **96 runs** for the pair ≈ **4.7 h**

The whitened arms stay in the grid. The 20260720b result shows the loss **interacts with
backbone capacity in the opposite direction to the prior**, which is a second mechanism on
the same axis, not noise:

| backbone | best snr | best whitened | whitening |
|---|--:|--:|:--|
| mlp | emp **0.26076** | rsh_w 0.29213 | hurts |
| gcn | emp **0.11934** | rsh_w 0.12382 | hurts |
| gwn | rsh 0.05718 | **rsh_w 0.05706** | helps |

Reading: a weak denoiser needs the prior to carry the variance profile, and equalising every
mode on top wastes its capacity on quiet modes; a strong denoiser has capacity to spare and
benefits from the equalisation. `rsh_w` ran 200/144 epochs on mlp/gcn so its loss there is
real; only `emp_w` (19/22 epochs) was truncated, which F4 fixes.

Submit one dataset at a time (21–27 jobs each) — the account submit cap rejects ~105+.

**Gate:** if the ladder's monotonicity survives 3 seeds on both datasets, the thesis has its
mechanism result and Phase 3/4 are breadth. If it collapses into seed noise, stop and
re-scope to the bounded-result framing before spending more GPU.

---

## Phase 2 — full metrics. No training; runs on Phase-1 checkpoints.

`eval_diffusion.py` already produces everything below from a saved `best_crps.pt`.

### 2a — the paper-comparable table (TEST split, S=100)

This is the table that goes in Ch. 5 next to HySRD/DiffSTG.

- `--split test --sampler ancestral --n-steps 50 --k 100 --ckpt best_crps.pt`
- outputs **MAE / RMSE / CRPS**, per-horizon (h=3,6,12) **and** averaged
- masked with PEMS `null_val=0` (required to match SpecSTG/DiffSTG)
- 2 ds × 3 bb × 3 arms × 3 seeds = **54 evals**

**Comparability is confirmed:** all four PEMS datasets use 60/20/20 on the time axis —
identical to HySRD — and our CRPS is the same normalized 19-quantile (CSDI/DiffSTG/SpecSTG)
score they report. Targets to beat//land near:

| | RMSE | MAE | CRPS |
|---|--:|--:|--:|
| HySRD PEMS08F | 25.49 | 16.30 | 0.0547 |
| SpecSTG PEMS08F | 25.59 | 17.06 | 0.0572 |
| DiffSTG PEMS08F | 28.26 | 18.99 | 0.0692 |
| HySRD PEMS08S | 3.39 | 1.52 | 0.0204 |
| HySRD PEMS04F | 31.78 | 20.20 | 0.0723 |
| HySRD PEMS04S | 3.82 | 1.79 | 0.0241 |

**NB** every number we have so far is **val** with **k=4** samples — a coarse estimator
(19 quantiles from 4 order statistics). Rankings are fair (identical CRN across arms) but
absolute values will move at S=100. There are still **zero test numbers on disk**.

### 2b — the two axes nobody else reports

Both come free from the same evaluator and are what the thesis text actually claims
("not that this model wins every point metric, but... calibration and the joint
spatio-temporal dependence structure").

- **Calibration**: PIT, PICP@90%, Winkler@90%
- **Joint**: sliced Wasserstein-1, energy distance

Neither DiffSTG nor HySRD reports *any* of these — both evaluate only marginal per-node
metrics, which are close to the worst possible probe for a change in the generative
*covariance*. Plausibly the effect is larger here than in CRPS.

### 2c — few-step curve (GAD parity)

GAD's own headline is that the graph prior's advantage is *largest at small step counts*.
We have tested this once, at 25/60/120 steps, and got a flat result — **but only on gwn**,
the backbone where the prior contributes ~1%. The theory says the few-step advantage should
track how much work the prior is doing, i.e. it should appear on **mlp**.

- `--n-steps ∈ {5, 10, 25, 50, 100}`, val split, seed 42
- 2 ds × 3 bb × 3 arms × 5 = **90 evals** (cheap)
- deliverable: CRPS-vs-NFE curve per arm; efficiency claim vs DiffSTG (13.6 s/window) and
  HySRD (3.8 s/window)

---

## Phase 3 — breadth: the PEMS04 pair

Same grid as Phase 1 on `pems04f`, `pems04s`. **Requires storage rotation** (below).
Seeds 42/43/44 if Phase 1's gate passed, else seed 42 only.

- **54 runs** ≈ 3 h, plus 54 test evals

## Phase 4 — METR-LA and PEMS-BAY

- `metrla` is a headline dataset; `pemsbay` is the **limitation case study** (train→val
  distribution shift), not a win condition.
- **Blocker:** `pemsbay/gwn` on DAIC is missing `val.npz` (training needs it) and `test.npz`.
  Push a complete copy from the Mac during rotation.
- These are the two gappy datasets, so they are the first place F5 (mask denominator) and
  the metrla `v_emp` shape bias (per-mode ratios 0.50–3.15 from zero-fill) actually matter.
- **54 runs** + evals

## Phase 5 — external baseline anchor

Run DiffSTG natively on our exact split via `submit_diffstg.sh` (recipe already validated,
seed 43). Everything else (HySRD, SpecSTG, PriSTI, GCRDD, TimeGrad) is cited from HySRD
Table 1 — same split, same metric, no run needed.

---

## Storage rotation (windowshome is 8 G hard; residuals are 7.9 G for all six)

Only ~2 datasets fit at once. The Mac holds the complete 9.0 G copy, so this is lossless.

| round | resident on DAIC | size | phase |
|---|---|--:|---|
| 1 (now) | pems08f, pems08s | 1.4 G | Phase 1 + 2 |
| 2 | pems04f, pems04s | 2.5 G | Phase 3 |
| 3 | metrla, pemsbay | 4.2 G | Phase 4 |

Rotate: delete the outgoing pair's residuals on DAIC, rsync the incoming pair from the Mac.
**Never delete run outputs** — they are ~150 MB total for everything and are the results.
Detecting a full quota needs a real 32 MB `dd`; `quota -s` reports 0 K and `touch` succeeds
even when full. The submitter now preflights both disk and residual inputs.

---

## Totals

Per dataset: 5 arms × 3 backbones × 3 seeds = 45, plus 3 `ddpm` = **48 runs**, ~2.3 h.
48 jobs per submission sits safely under the ~105 account submit cap — one dataset at a time.

| | training runs | evals | wall-clock |
|---|--:|--:|--:|
| Phase 1 (pems08f, pems08s) | 96 | — | ~4.7 h |
| Phase 2 (no training) | 0 | 234 | ~3 h |
| Phase 3 (pems04f, pems04s) | 96 | 90 | ~5 h |
| Phase 4 (metrla, pemsbay) | 96 | 90 | ~5 h |
| Phase 5 (DiffSTG anchor) | 2 | — | ~24 h |
| **total** | **290** | **414** | **~2 days** |

## What is deliberately NOT in this plan

- **`iso_w`.** Algebraically identical to `iso` — a duplicate run, not a control.
- **`ddpm/vp` as a controlled ablation.** It differs in prior *and* schedule *and* noise
  sampler, and generates its final sample at a noise level seen in 0.015% of training draws.
  Reference point only.
- **More method development.** The mechanism results are the finding; the gap is evaluation.
- **More method development.** The mechanism result is the finding; the gap is evaluation,
  not method.

## Confounds to state in the thesis (not fixable, must be disclosed)

1. **Corruption budget is not matched.** `_normalize_scale` pins θ_min=0.8 for every arm, so
   structured priors carry extra fast modes and are strictly *more* corrupted at any c̄ —
   mean signal retained 0.75 (iso) vs 0.51 (emp) vs 0.45 (rsh). They win anyway, so the
   effect is conservative; but pinning θ_mean instead would flip the asymmetry, and no choice
   matches both. **This is the first thing a GAD co-author will look for.**
2. **Ancestral discretisation.** A stiffer prior packs more mode transitions into the same
   50 steps, so "50 steps" is not equal difficulty across arms. Phase 2c measures this.
3. `emp` matches `var_full` by coincidence (≤3%), not by construction — its rescale is gated
   off. Optionally close this (audit item 5).
