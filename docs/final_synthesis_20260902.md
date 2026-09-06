# Final synthesis — 2026-09-02

Where everything is, what is actually wrong with the pipeline, what the evidence
supports, and the shortest path to a finished thesis.

Supersedes the planning half of `final_thesis_audit_20260812.md` and
`final_run_plan_20260812.md`; the inventory in `results_inventory_20260813.md`
still stands.

---

## 1. State of the machines

**Nothing has run since 2026-07-29.** Both the Aug-12 run plan (R2/R4/R5/R6) and
the code it needs were written but never submitted.

| | state |
|---|---|
| `daic-exp` (`sakin@daic02`) | reachable, **queue empty**, repo at `ee3f29d` — **6 commits behind** the Mac |
| `daic-exp-denis` (`dkrylov@daic01`) | **password expired**, no TTY over ssh — needs a manual reset before the second shard can be used |
| campaigns on DAIC | `20260728_histcond`, `20260728_pgraph_*`, `20260727_kron` — all already mirrored to the Mac |
| what is on DAIC and *not* on the Mac | the **checkpoints** (`best_crps.pt` / `best.pt` / `last.pt`, 575 MB for histcond alone) plus `train.log`, `history.csv`, `validation_history.json`. Metrics and configs are fully synced. |
| what is on the Mac and not on DAIC | every commit since `ee3f29d`, the uncommitted `--anc-c-lo` / `rsh` / `DENOISER` code, `scripts/analysis/*`, `scripts/daic/anc_sweep*`, `scripts/daic/eval_saved.sbatch`, and the whole `tudelft_thesis_draft/` (gitignored) |

Consequence: the checkpoints needed for every eval-only run below are on the
cluster and intact. Nothing has to be retrained to fix the open defect.

---

## 2. The audit: the three things you suspected

### 2.1 Conditioning — **was broken, is fixed, now verified**

The bug was real and it was the worst one in the project: `--cond-past` fell back
silently to the previous window's *residual* field, so `20260721_ladder/*-past`
trained `C = [R_prev | Ŷ]` while Chapter 4 defines `C = [X | Ŷ]`. About 150
test-evaluated runs are on that arm.

It is fixed (`1e253cc`), the replacement campaign `20260728_histcond` is complete
(80/80, zero failures, config-parity verified), and the thesis draft has already
been re-aggregated onto it.

Newly verified here, because the history is read from the processed split by row
index rather than from a key inside the residual cache:

| dataset / backbone | `\|y_split − (ŷ + r)\|` mean | same, rows shifted by 1 |
|---|---:|---:|
| pems08f / gcn | 1.4e-12 | 16.7 |
| metrla / gcn | 4.6e-09 | 2.92 |
| pemsbay / gwn | 7.6e-10 | 0.95 |

Row *i* of the residual cache is row *i* of the processed split to float32
precision, and a one-row misalignment would show up at 3–17 raw units. **The
history conditioning is aligned. This is closed.**

### 2.2 Normalisation — **two real defects, both small, neither explains anything**

Per-sensor normalisation itself is right and well motivated: per-sensor residual σ
spans **3.1–12.3×** (p95/p5) across the network, against ≤2.5× across the twelve
lead times, so without it the prior spends its budget reproducing a static scale
that is known in advance.

Two genuine defects:

1. **The arms are not exactly variance-matched.** The isotropic prior takes
   `R[mask].var()` over *all* masked entries; the Kronecker arm's `D_H` is a second
   moment over *fully observed windows only*. Measured mean stationary variance:

   | | iso | kron |
   |---|---|---|
   | best case | 1.000 | 1.000 |
   | worst case | **0.942** (metrla/mlp) | 0.960 (metrla/gwn) |

   So §5.4's "scaled to the same total residual variance by construction" is
   accurate to ~1% typically and 4.7% at worst, on METR-LA. It should say so.

2. **`Σ_S` is normalised to unit *mean eigenvalue*, not unit *diagonal*.** After
   per-sensor normalisation the field has unit variance at every node, so a
   correctly specified prior must too; this one asserts node variances differing by
   1.17–1.31× at γ=0.8 (and 3.3–3.7× at γ=0.05). Already documented in
   `method_design.md` §5, already measured: scoring both ways moves held-out ΔNLL
   in the fourth decimal. **A coherence fix, not a performance fix.**

Neither of these is why the graph channel does nothing.

### 2.3 Graphs — **not a bug. The graph really is worth almost nothing, and that is now proven six ways**

Held-out NLL per mode, gain over isotropic (fit on train, scored on val, one free
scale each, all 18 cells, `results/analysis/prior_family_nll.json`):

| family | range over 18 cells | reading |
|---|---|---|
| temporal only, `Σ_T ⊗ I` | **+0.13 … +1.14** | carries the entire gain |
| spatial only, `I ⊗ Σ_S` (γ=0.8) | **−0.05 … +0.18** | ≈ nothing |
| `Σ_T ⊗ Σ_S`, γ=0.8 — *the shipped prior* | +0.11 … +1.19 | temporal + 0.00–0.05 |
| same, γ fitted per cell | +0.13 … +1.33 | ≤ +0.14 over shipped, usually +0.01 |
| same, γ **and** a free exponent | no further gain | the shape is not the problem |
| `Σ_T ⊗ Σ_S^{emp}` (data-fit spatial) | +0.14 … +1.37 | beats the Laplacian in 17/18, still small |
| **Cartesian product graph** | **−0.12 … −0.67** | **worse than white noise, 18/18** |
| strong product | −0.12 … −0.67 | same |
| reshaped (empirical per-mode) | +0.15 … +1.49 | the ceiling; all remaining headroom is here |

Add the four measurements already on disk — attribution table (graph ≤0.10 nats,
negative in 8/18), γ-sweep flat above 0.4, mode-slope 0/12, GTCNN parametric fit
returning spatial coupling **exactly 0** on all six datasets — and the conclusion
is not a suspicion any more.

One nuance worth keeping: the spatial term is consistently positive on METR-LA and
PEMS-BAY (+0.05 … +0.18), the two **distance-weighted DCRNN** graphs, and ≈0 or
negative on the PEMS quartet's **binary connectivity** graphs. Graph quality
matters; it just does not matter *enough*.

The residual is not spatially white — the log per-graph-mode variance falls with λ
in every cell (slope −0.21 to −2.05, prior shape correlation +0.35 to +0.85). It is
that a smooth `(L+γI)^{-1}` captures almost none of that fall, and no amount of
retuning γ changes it.

### 2.4 The defect that *is* still open, and it is the big one

`reverse_sample_ancestral` stops at a terminal floor `c_lo = 0.02` and returns the
denoiser's `R̂₀` there. `eval_diffusion.py` never passes the kwarg. Training draws
go down to **8e-4**, so the sampler stops **25× above** the noise level the
denoiser was trained to, and the returned ensemble is a posterior *mean* — it
under-disperses by exactly the conditional variance still present.

How much is still present differs by arm, because it depends on the spread of θ
(`results/analysis/sampler_floor.json`):

| arm | un-denoised variance at `c_lo = 0.02` | at 8e-4 |
|---|---|---|
| isotropic | **3.15 %**, identical in all 18 cells (θ ≡ 0.8) | 0.13 % |
| Kronecker | **19.0 – 28.2 %** | 0.9 – 2.4 % |
| reshaped | **38.9 – 63.6 %** | 5.4 – 18.1 % |

That ordering — iso ≪ kron ≪ rsh — is *exactly* the ordering of the thesis's
under-dispersion finding, and exactly the ordering of the coverage numbers
(0.822 / 0.764 / 0.646 on the one matched-integrator cell on disk). Push the
integrator to convergence with EM@1500 and the ranking **reverses**: 0.856 /
0.855 / **0.878**, and the CRPS gap collapses from +0.9 % to +0.1 %.

The local probe on pems08f (256 windows, k=24) says the same for the cheap fix:
isotropic moves <1 point of coverage, structured gains **4.8–4.9** and overtakes
the control on coverage, Winkler **and** energy while keeping its CRPS win.

**So the thesis's most emphatic negative finding — "structure buys sharpness at the
cost of dispersion, without exception" — is very probably an artefact of where the
sampler stops.** Lowering `c_lo` costs nothing: it is the endpoint of the grid, not
the number of steps. The same 50 steps, re-spaced.

This is the one thing standing between the current draft and a clean positive
result, and it is eval-only on checkpoints that already exist.

---

## 3. The clean story

Three claims, in the order they should be argued. Everything else is support,
appendix, or cut.

### Claim 1 — the positive result

**Matching the diffusion's corruption process to the second-order structure of the
forecast residual improves probabilistic forecasts, and the gain is proportional to
the structure the backbone leaves behind.**

- Test CRPS, structured − isotropic, history conditioning, seed-mean over 2 seeds:
  wins **every linear and every GCN cell on all four PEMS benchmarks**, loses on
  Graph WaveNet. 16/24 paired seeds, p = 0.076.
- MAE 33/36 cells, p = 1e-7. This one *is* significant.
- The ladder law is measured **before any diffusion is trained**: residual
  separability, horizon growth, graph roughness and joint stiffness all decay by an
  order of magnitude from linear → GCN → GWN, and the offline held-out ΔNLL
  (+1.04 → +0.75 → +0.16 on pems08f) predicts the end-to-end ordering.

That is a real, mechanism-backed, honestly-sized result.

### Claim 2 — the negative result, which is the actual novelty

**The graph half of the structure is worth almost nothing, and a product-graph
Laplacian is worse than white noise.**

The entire graph-diffusion literature assumes the sensor graph belongs in the
generative process. On forecast residuals it does not:

- temporal factor +0.13…+1.14 nats/mode; graph factor on top **≤ +0.05** typical,
  negative in 8/18 cells;
- γ is flat above 0.4 — a per-cell oracle buys +3.1 %;
- fitting GTCNN Eq. 10's four coefficients returns **spatial coupling exactly 0**
  on all six datasets: given the whole product-graph family, the data selects the
  temporal-only member;
- Cartesian, strong and tensor product graphs all score **below isotropic** in
  18/18 cells.

Stated positively, this is the useful finding: **the structure worth putting in the
noise is temporal, not spatial** — and it is worth putting in, which is Claim 1.
This closes §4.2.3's prose argument with data and costs zero GPU.

### Claim 3 — the systems result

**At 1/58 of the generative parameters and 1/17 of the sampling cost, the system
takes 47 of 48 columns against DeepVAR / TransNVP / TimeGrad / GCRDD / DiffSTG /
PriSTI / SpecSTG / HySRD-Net**, 8–15 % CRPS over the strongest, corroborated by an
independent re-run of DiffSTG from its own code. Unaffected by everything above —
the GWN numbers move by ≤0.0002 CRPS.

### The methodological finding (pending R2)

**A stiff prior must be sampled to a lower terminal noise floor than a flat one, or
the comparison measures the sampler.** Concrete rule: sample to the floor the
denoiser was trained at. This is a contribution in its own right and it is what
turns the calibration section from a caveat into a result.

### What the story is *not*

- Not "a graph prior for spatio-temporal diffusion". The graph does not pay.
- Not "the prior substitutes for denoiser capacity" — the denoiser is fixed at
  17,305 parameters in every cell; the ladder varies the **data**.
- Not built on the heat/product-graph campaigns. `20260724_heat` is 136 models
  whose headline wins sit on the retired conditioning arm; `20260728_pgraph_*` ran
  on the raw signal with 4–6× budget skew and coverage 0.108. Both stay out; only
  the parametric fit survives, and it survives into Claim 2.

---

## 4. What to run, in order

The queue is empty and both the code and the checkpoints exist. Everything except
R-new is already written.

### Step 0 — unblock (minutes)

```bash
git add -A && git commit && git push                # the --anc-c-lo / rsh / DENOISER code
ssh daic-exp 'cd $PROJECT_DIR && rm -f .git/index.lock && git pull'
```
Reset the `dkrylov` password if the second shard is wanted; without it everything
below runs at half rate on `sakin` alone.

### Step 1 — R2, the sampler floor. **Eval only. Do this first.** (~30 GPU-h)

```bash
DRY=1 scripts/daic/anc_sweep.sh          # pilot: 4 cells x 2 arms
SHARD=a scripts/daic/anc_sweep.sh
CELLS_SPEC=full SHARD=a scripts/daic/anc_sweep.sh   # 18 cells, once the pilot holds
python scripts/analysis/anc_sweep_report.py
```
Grid `c_lo ∈ {0.02, 8e-4} × steps ∈ {50, 200}`. **Add the `rsh` arm** — it is the
sharpest test (38.9–63.6 % un-denoised) and the biggest potential upside, since the
reshaped prior is the best of the family by held-out NLL and currently looks the
worst end-to-end.

Nothing in Chapters 5–7 is final until this reports.

### Step 2 — the temporal-only arm. **The missing end-to-end ablation.** (~70 GPU-h)

No `--spatial flat` cell has ever been trained. Claim 2 currently rests entirely on
offline likelihood; one arm makes it end-to-end.

```bash
ARMS=flat SEEDS=42 scripts/daic/histcond.sh    # 18 cells, needs a 3-line arm in histcond_cell.sbatch
```
If `Σ_T ⊗ I ≈ Σ_T ⊗ Σ_S` on test CRPS, Claim 2 is proven at the level the thesis
actually reports, and the thesis has a genuinely interesting negative result rather
than an attribution table.

### Step 3 — R4, three more seeds on the history grid (~100 GPU-h)

Takes Claim 1 from p = 0.076 to conclusive. Phase A (`pems08f pems08s pemsbay`)
first; phase B only after A reports clean, because of the 8 GB cache rotation.

### Step 4 — free, no GPU, do while the queue runs

- Land the held-out-NLL family table (§2.3) as the experiment behind §4.2.3, and
  the product-graph row as the ablation that answers "why not a product graph".
- D3a stray `t`; D3b write the two missing scalars into `eq:Theta`; D3c one
  footnote on `D_H` uncentred vs `R_H` centred; D3d state the 0.942–1.000 variance
  match instead of claiming exactness; D3e re-derive the seed counts for histcond's
  2 seeds; D3f uncomment `\listoffigures` / `\listoftables`.
- Figures: **the prior itself** (`Σ_T`, `R_H`, the `Θ` spectrum over the N×H mode
  grid) — the single most useful missing figure; a fan chart; a CRPS-vs-lead-time
  panel against the baselines; the cost figure. Twelve rendered figures are already
  on disk and unused.
- Compress the four near-identical RQ restatements (§5.3.4, §5.8, §6.1–6.2, §7.1) —
  about 2.5 pages of pure repetition.

### Explicitly not doing

R5 (ugnet/DDPM test evals) and R6 (rsh under history conditioning) unless Step 1
finishes early — both remove caveats rather than add results. No new γ sweep, no
product-graph retrain on the residual arm, no seventh dataset.

---

## 5. The one-paragraph version

The conditioning bug is fixed and verified, and the thesis is already re-aggregated
onto the correct arm — that change alone turned the headline from 17/50 against us
into wins on every linear and GCN cell across the four PEMS benchmarks.
Normalisation has two small real defects that should be *stated*, not fixed. The
graph channel is not broken; it genuinely carries ~2 % of what the temporal channel
carries, product graphs score below white noise in 18/18 cells, and that negative
result — established six independent ways — is the most novel thing in the project.
The one live bug is the sampler's terminal floor: it resolves the isotropic control
to 3.15 % and the structured prior to 19–28 % (the reshaped prior to 39–64 %),
which manufactures the thesis's under-dispersion finding out of nothing. Fixing it
is an eval-only pass over checkpoints that already exist, and on the two cells
probed so far it flips coverage, Winkler and energy in our favour while keeping the
CRPS win. Run that, add the temporal-only arm to make the negative result
end-to-end, add three seeds, and the thesis is done.
