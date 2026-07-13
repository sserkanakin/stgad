# Evaluation protocol

Decision record for how the thesis evaluates the structured (Kronecker) graph-prior
diffusion model: what we compare, against what, with which sampler, on which numbers.
This seeds `mainmatter/5_experiments.tex`. Written 2026-07-13.

Grounded in: the eval suite (`src/eval/`, `scripts/eval_diffusion.py`), the round-2
results (`results/pulled/20260708_r2/`), the draft's commitments
(`1_introduction.tex` RQs, `4_methodology.tex` metric statement), and the published
diffusion-STG protocol (DiffSTG, SpecSTG, ProGen, CSDI).

---

## 1. Sampler: ancestral (exponential-integrator DDPM) is primary; EM is an ablation

We have two reverse samplers (`src/diffusion/reverse.py`):

- **Ancestral** (`reverse_sample_ancestral`, line 157): exact closed-form OU posterior
  per eigenmode via an exponential integrator, geometric heat grid. No discretization
  error in the linear (prior) part, no stability floor. Converges at **25 steps**.
- **EM** (`reverse_sample_em`, line 77): explicit Euler–Maruyama on the reverse SDE
  (GAD parity). Conditionally stable → needs a step floor `max(200, ceil(θ_max·c_0))`
  and in practice **1000–3500 steps**.

**Decision — headline = ancestral @ 25 steps.** In round 2 the best CRPS on every
dataset came from `ancestral` at `n_steps=25`, at ~40–140× lower cost than EM. The
ancestral integrator is *exact* for the linear OU dynamics of our forward process, so
extra steps buy nothing; EM's many steps only recover accuracy it loses to Euler
discretization. Ancestral is also what the field uses (DiffSTG/SpecSTG/CSDI are all
DDPM-ancestral), so it keeps us protocol-comparable.

**EM stays as a reported ablation**, not the headline. It (a) gives GAD/continuous-SDE
parity — the connection to `reverse_sample_em` and the ProGen-style SDE framing — and
(b) demonstrates sampler-robustness: the ranking (kron > iso) must hold under both
samplers. Report one sampler-comparison table (CRPS & MAE vs step count for both
samplers, headline datasets). This supersedes the earlier "EM-only for GAD parity"
working rule now that the exact ancestral integrator exists and wins.

Selection rule (already implemented): "fewest steps within 1% of min CRPS." Report the
selected step count.

---

## 2. Metrics

Everything is scored on the **reconstructed raw-unit forecast**
`fc = (y_pred + residual_sample).clamp(min=0)` (backbone point term cancels because all
scores are translation-invariant), masked with the PEMS `null_val=0` convention
(`mask_zero`, matches SpecSTG/DiffSTG).

### Tier 1 — leaderboard trio (external comparability). Report always.
| Metric | Code | Convention |
|---|---|---|
| **CRPS** (headline) | `crps.py:28` normalized 19-quantile (0.05–0.95) pinball / (Σ\|y\|·19) | CSDI/DiffSTG/SpecSTG — directly comparable to their published CRPS |
| **MAE** | ensemble-mean, masked | field standard |
| **RMSE** | ensemble-mean, masked | field standard |

Report MAE/RMSE/CRPS **per horizon at h = 3, 6, 12 and averaged**.

### Tier 2 — calibration (this is the thesis's core claim: structure improves calibration)
| Metric | Code |
|---|---|
| **PIT histograms** | per sensor/horizon; visual + a flatness statistic |
| **Coverage / PICP @ 90%** | `eval_diffusion.py:213`, target 0.90 (α=0.1) |
| **Winkler / interval score @ 90%** | `intervals.py:19` |

Round-2 val coverage is well below nominal (0.30–0.46 across datasets) — the
under-coverage story (esp. PEMS-BAY) is a headline finding, not a bug to hide.

### Tier 3 — joint spatial dependence (the second thesis claim; few competitors report this)
| Metric | Code |
|---|---|
| **Sliced Wasserstein-1** (cloud vs cloud in R^N) | `distribution.py:19` |
| **Energy distance** (two-sample) | `distribution.py:44` |

These are the differentiator: they show the structured prior reproduces the *joint*
cross-sensor law that a scale-matched isotropic prior cannot. Report on headline
datasets, structured vs isotropic.

### Prior-validation diagnostics (RQ1, separate from forecast eval — already done, Stage 0)
NKP captured-variance fraction (`tab:nkp`), per-entry NLL gap (kron vs iso), per-mode
2-Wasserstein (`prior_closeness.py`). Keep in the experiments chapter as the RQ1 answer.

**Secondary CRPS:** also log the energy/"fair" CRPS (`crps.py:7`) as a robustness check,
but the normalized quantile CRPS is *the* reported number for comparability.

---

## 3. What we compare against

### Tier A — controlled internal comparison (primary; answers RQ2). We own every knob.
Denoiser (UGnet, from DiffSTG) **held fixed**; only the forward process / prior changes.

**HEADLINE PAIR (one-variable comparison):**
1. **kron_base** — kronecker Σ_S=(L+γI)⁻¹, base conditioning, loss=snr, lr=1e-4. Trained (r2).
2. **iso_base** — isotropic σ²I, **matched**: base conditioning, loss=snr, lr=1e-4, uls schedule.
   Differs from kron_base ONLY in the prior → clean RQ2. **Must be trained** (see below).

> **Control-mismatch found 2026-07-13 & fixed.** r2 shipped four arms
> (kron_pastG, kron_pastPN, kron_base, **iso_pastG**) but no iso_base. The only
> isotropic arm, `iso_pastG`, has `cond_past=True` while the best model `kron_base` has
> `cond_past=False` — so kron_base-vs-iso_pastG confounds prior *and* conditioning. The
> older `20260706_gadmse` has a matched base pair but at the stale recipe (loss=mse,
> lr=1e-3, no best_crps.pt). Fix: train the missing `iso_base` at the r2 recipe:
> `ARMS="iso_base:isotropic:base:global" EXP=20260708_r2 scripts/daic/submit_experiment.sh 42`.
> (schedule fcps vs uls is NOT a confound — an isotropic prior correctly uses ULS.)

**SECONDARY (conditioning ablation, existing r2 checkpoints):** kron_base vs kron_pastG
vs kron_pastPN (round-2 shows base wins), and the matched kron_pastG vs iso_pastG pair.
Report to justify the base choice; not the headline.

Other ablations (denoiser fixed): spatial **structured vs flat**; loss **snr vs mse**;
sampler **ancestral vs EM** (§1); checkpoint **best_loss vs best_crps**.

### Tier B — external leaderboard (positioning). Contextualizes Tier A against the field.
Published probabilistic diffusion-STG methods on the shared PEMS04/PEMS08 flow+speed
protocol (6:2:2, 12→12, S=100, normalized CRPS):

**DiffSTG, SpecSTG, PriSTI, GCRDD, TimeGrad, CSDI, HySRD-Net**, plus the strong recent
SDE/diffusion entrants (see §6): **ProGen**, and — where numbers exist — DiffSTWave /
Double-Diffusion. Numbers already transcribed in
`results/pulled/20260708_r2/comparison.md`.

Phased (decided 2026-07-13):
- **Phase 1 (initial draft — now): cited numbers only.** Broad table
  (DiffSTG/SpecSTG/PriSTI/GCRDD/TimeGrad/ProGen/HySRD-Net) transcribed from papers with a
  clear "same 6:2:2 / 12→12 / S=100 / normalized-CRPS protocol" caveat. Fastest path to a
  full table; already scaffolded in `comparison.md`. Depends on §4 split-verification.
- **Phase 2 (post-draft): self-run DiffSTG anchor.** Vimal's `DiffSTG+MAS-SNR` fork
  (`~/Projects/_vimal_daic_run`) uses the *same UGnet denoiser we do*, so a self-run
  DiffSTG on our exact split is the cleanest external comparison — same data, same
  denoiser, only the framework differs. Add on headline datasets when we do the full run.

---

## 4. Fixed experimental protocol (commit these; the draft currently leaves them open)

| Choice | Value | Rationale |
|---|---|---|
| Task | 12 → 12 (P=12 history, H=12 horizon) | field standard; draft leaves P open → fix to 12 |
| Split | **speed = 70/10/20 (DCRNN/GWN); flow = 60/20/20 (SpecSTG)** | verified in `build_datasets.py:201,215`. Per-family convention: flow matches SpecSTG/DiffSTG/ProGen exactly; speed matches DCRNN/GWN. External leaderboard is on PEMS04/08 flow+speed only; METR-LA/PEMS-BAY are internal (kron-vs-iso + limitation). |
| Posterior samples | **S = 100** for all final test metrics | field standard: SpecSTG "S=100 for all models", CSDI 100, and the HySRD Table 1 baselines are all S=100. Round-2 used k=8 → the quantile-CRPS estimator is biased at low S, so k=8 is NOT comparable to their 100; must rerun at S=100. |
| Seeds | **Phase 1: seed 42 only** (fast, no error bars). **Phase 2: 3 seeds (42/43/44), mean ± std** on headline datasets. | decided 2026-07-13: get the initial-draft table fast, add variance later. |
| Report split | **TEST** | everything on disk is val-only — no diffusion test numbers exist yet. This is the main outstanding run. |
| Interval level | 90% (α=0.1) | matches Winkler/coverage defaults |
| CRPS convention | normalized 19-quantile | matches all cited numbers |

### Datasets and their role
- **Headline (full treatment, 3 seeds, Tier A+B):** METR-LA (speed), PEMS08F (flow),
  PEMS08S (speed). These carry the main story.
- **Supporting (Tier A+B, published-number comparison):** PEMS04F, PEMS04S.
- **Limitation case study:** PEMS-BAY — the train→val distribution shift / under-coverage
  finding (`1_introduction.tex:36`, `7_conclusion.tex:25`). Reported as a diagnosis, not
  a win.

---

## 5. What must still be run

### Phase 1 — initial draft (do now, seed 42, cited baselines). Two clusters, 8 slots each.
1. **Train iso_base** (the missing matched control), all 6 datasets, seed 42, r2 recipe,
   with a **light final-val** (single step value each — no sweep; isotropic prior is less
   stiff, needs no per-cell EM tuning; anc=25 is uniform across all r2 arms):
   `ARMS="iso_base:isotropic:base:global" EXP=20260708_r2 FINAL_STEPS_LIST=1000 FINAL_ANC_STEPS=25 scripts/daic/submit_experiment.sh 42`
   6 array tasks; leaves 2 slots free on that cluster. (Do NOT pass `--nodelist=gpu[36-53]`
   — a node list requests ALL 18 nodes → AssocMaxNodePerJobLimit. To prefer Blackwell pin
   by GRES TYPE: `--gres=gpu:<rtx-token>:1`, token from `sinfo -h -o '%G'|tr ',' '\n'|grep -i rtx`.)
2. **kron_base test evals** start immediately in parallel (checkpoints exist), EM first,
   split across clusters via the `DATASETS` env so nothing double-submits:
   `ARMS="kron_base" DATASETS="metrla pemsbay pems04f pems04s pems08s" scripts/daic/submit_test_evals.sh` (cluster 1, 10 jobs)
   + `ARMS="kron_base" DATASETS="pems08f" scripts/daic/submit_test_evals.sh` (cluster 2's 2 spare slots).
3. **iso_base test evals** once its training finishes: `ARMS="iso_base" scripts/daic/submit_test_evals.sh` (12 jobs, EM first).
   All test evals: S=100 (`--k 100`), best_crps.pt, ancestral@25 + EM@val-selected. Eval-only.
   *Nothing on disk is test split yet* — this is the biggest gap. 30 jobs total.
3. **Verify the split** matches the cited papers per family (§4) before putting cited
   numbers next to ours. Do it first — validates the whole comparison. Speed=70/10/20
   (DCRNN), flow=60/20/20 (SpecSTG); confirmed in `scripts/build_datasets.py:201,215`.
4. Regenerate `comparison.md` from the new *test/S=100* rows via
   `scripts/summarize_valsweep.py --exp 20260708_r2` (reads the `*_metrics_*.json` the
   standalone eval writes; same aggregator that made the r2 table).

### Phase 2 — full experiments (post-draft)
5. **Seeds 43, 44** on headline datasets → mean ± std.
6. **Self-run DiffSTG** (Vimal fork) on headline datasets, same split, as a controlled
   external anchor.

Everything else (metrics, both samplers, prior arms, aggregation) is already implemented.

> Note: Phase-1 test/S=100 CRPS will differ from the round-2 val/k=8 numbers in §6 —
> more samples usually *improve* CRPS, and test≠val. Treat §6 as indicative only until
> Phase 1 completes.

---

## 6. Positioning vs the field (round-2 VAL numbers, k=8 — indicative only)

From `comparison.md`. Ours is currently **val / k=8**; cited are **test / S=100** — so
these are indicative, not final, but the standing is genuinely competitive:

- **PEMS08F (flow):** kron_base MAE **15.73** / RMSE 25.66 / CRPS 0.0565 — MAE beats
  DiffSTG (18.99), SpecSTG (17.06), HySRD-Net (16.3), and is within ~5% of ProGen's
  best-in-class 14.99. CRPS ≈ SpecSTG, ~18% better than DiffSTG.
- **PEMS08S (speed):** kron_base CRPS **0.0188** — best on the board (beats HySRD-Net
  0.0204, SpecSTG 0.0211), −7.8% vs HySRD-Net.
- **PEMS04F/S:** kron_base ≈ HySRD-Net (+3.3% CRPS), beats DiffSTG/TimeGrad/GCRDD.
- In every dataset **kron_base > isotropic** on CRPS — the RQ2 result holds.

### Recent work to cite / compare (2024–2026)
| Method | Venue/year | Why relevant |
|---|---|---|
| **ProGen** (arXiv 2411.01267) | 2024/25 | SDE-based continuous generative STG; current SOTA on PEMS flow (PEMS08 MAE 14.99 / CRPS 0.05). Direct, strong probabilistic competitor. |
| **HySRD-Net** | 2026 | "closest system to this thesis" (`3_literature-review.tex:62`); already in comparison table. |
| **SpecSTG** (arXiv 2401.08119) | 2024 | spectral diffusion; our spectral cousin; per-dataset table already transcribed. |
| **DiffSTG** (arXiv 2301.13629) | 2023 | denoiser (UGnet) provenance + self-run anchor. |
| **DiffSTWave** | 2025 | reports 7–14% CRPS gains over 2nd-best on PEMS08; check for a citable table. |
| **Double-Diffusion** (arXiv 2506.23053) | 2025 | speed/accuracy/uncertainty trade-off on urban sensor nets; recent probabilistic point. |
| **NPDiff / Disentangled-diffusion** (Springer 2025) | 2025 | recent probabilistic STG baselines for the related-work leaderboard breadth. |
| CSDI, TimeGrad, PriSTI, GCRDD, DeepAR, MC-Dropout | 2021–23 | standard probabilistic baseline set (cited numbers). |
