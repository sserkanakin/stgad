# Rerun plan — 2026-09-02

What to fix, what to tune, then what to run. Everything below assumes the clean
slate: `results/` is empty, the archive is at `archive/20260902/`, DAIC holds only
code, and there is **one account** (8 concurrent jobs, no `SHARD=`).

---

## 1. The thing to tune first, and why it is not a knob

The training noise range is `c̄ ~ logU[c_lo, c_0]`. Its **top** is already
prior-aware: `_normalize_scale` puts `θ_min → 0.8` so that `2·c₀·θ_min ≈ 11.2` for
every prior — the slowest mode is guaranteed to mix. Its **bottom** is a constant,
`c_lo = NOISE_FLOOR²/2 = 8e-4`, justified in the code by "σ(c̄) ≈ √(2c̄), so every
dataset ends equally clean (~4%)".

That identity is the `θ → 0` limit. What actually remains un-erased at `c̄` is
variance-weighted over the modes:

```
f(c̄) = Σ_m var_m (1 − e^{−2 θ_m c̄}) / Σ_m var_m
```

and it depends on the θ *spread*, which is exactly what distinguishes the arms. At
the shared floor, measured on all 18 cells (`scripts/analysis/sampler_floor.py`):

| arm | θ_max | f at the current sampler stop (0.02) | f at the training floor (8e-4) | floor needed to match iso |
|---|---:|---:|---:|---|
| isotropic | 0.8 | 3.15 % | **0.13 %** | 8.1e-4 — already right |
| Kronecker | 16 – 232 | 19 – 28 % | 0.9 – 2.4 % | 4e-5 – 1.2e-4 (**7–20× lower**) |
| reshaped | 151 – 842 145 | 39 – 76 % | 5 – 18 % | 1.1e-6 – 1.9e-5 (**37–721× lower**) |

So there are two separate defects, and only the first is a simple bug:

**Defect 1 — the sampler stops above where the denoiser was trained.**
`reverse_sample_ancestral` hardcodes `c_lo = 0.02`; training goes to `8e-4`.
25× too high, and `eval_diffusion.py` never passes the kwarg. One line.

**Defect 2 — the training floor is prior-independent while the ceiling is not.**
Even after fixing defect 1, the arms are resolved to 0.13 % / 0.9–2.4 % / 5–18 %.
The comparison still measures the sampler as much as the prior.

**The fix, and it is a training-time decision — hence "tune first":** set *both* ends
of the noise range from the prior's own spectrum. Top: the slowest mode must mix
(`2·c₀·θ_min ≈ 11.2`, already done). Bottom: no more than a fixed fraction of the
variance may remain un-erased (`f(c_lo) = 1.3e-3`, what isotropic already gets).
Then every arm is trained and sampled to the same fidelity by construction, and a
prior comparison is a prior comparison.

Cost of doing it: the noise range widens from 3.9 decades (iso) to ~5.2 (Kronecker)
and ~6.2–6.8 (reshaped). Log-uniform draws spread thinner per decade, and the
geometric sampler grid gets coarser at fixed step count. **Both are pilot questions**
(§4), not things to guess.

Fallback if the widened range proves untrainable: `--stiffness-cap` bounds
`θ_max/θ_min` instead. It works, but it changes the *prior* rather than the protocol,
and it blunts precisely what the reshaped arm exists to test. Protocol first.

---

## 2. Code fixes before anything is submitted

| # | fix | where | why |
|---|---|---|---|
| **A** | `reverse_sample_ancestral(c_lo=None)` → default to `schedule.c_lo` instead of `0.02` | `src/diffusion/reverse.py` | defect 1. Keep `--anc-c-lo` for sweeps. |
| **B** | prior-aware floor: solve `f(c) = 1.3e-3` per prior, feed it to the schedule, **record it in `config.json`** | `src/diffusion/forward_fcps.py`, `scripts/train_diffusion.py` | defect 2. Eval must read it back so sampling matches training. |
| **C** | match the arms exactly: isotropic uses `R[mask].var()` (centred, all masked entries), Kronecker uses `E[r²]` over fully observed windows | `src/diffusion/prior.py` | arms currently differ by up to 4.7 % in total variance (worst METR-LA). §5.4 claims they are matched "by construction". |
| **D** | normalise `Σ_S` to unit **diagonal**, not unit mean eigenvalue | `src/diffusion/prior.py` | per-sensor normalisation makes node variance 1 everywhere; the prior currently asserts 1.17–1.31× spread. Measured effect is 4th-decimal — a coherence fix that also decouples γ from the diagonal. |
| **E** | write `history` into the residual `.npz` | `scripts/extract_residuals.py` | conditioning currently reads the history from the processed split **by row index**. It is verified correct, but re-extracting is free and removes the failure mode permanently. |
| **F** | `flat` arm (`Σ_T ⊗ I`) | `scripts/daic/histcond_cell.sbatch` | ✅ already added |

Then, free on the Mac, before any GPU time:

```bash
RESULTS_ROOT=$PWD/archive/20260902/results .venv/bin/python scripts/analysis/prior_family_nll.py
RESULTS_ROOT=$PWD/archive/20260902/results .venv/bin/python scripts/analysis/sampler_floor.py
```

Re-run against the archived residuals under fixes C+D and confirm two things: γ=0.8
is still the best shared value (it was, by a flat margin), and `f(c_lo)` is now equal
across arms. If either moves, stop and look before spending GPU time.

---

## 3. Stage 1 — backbones and residuals (needed regardless)

18 cells = 6 datasets × {mlp, gcn, gwn}, seed 42. `submit_backbone.sh` already does
train + extract in one array task.

```bash
EXP=20260903_backbones scripts/daic/submit_backbone.sh
```

One array of 18, throttle 8, 12 h walltime each (generous — GWN is the only heavy
one). **Everything downstream depends on this, so it goes first.**

Verify before proceeding: 18 × 3 `.npz` files, 18 `model/best.pt`, and the point
MAE/RMSE within noise of the archived run (`archive/20260902/results/residuals/`)
— the backbones are the one part of the pipeline that was never in doubt, so a large
divergence means something else broke.

---

## 4. Stage 2 — the protocol pilot. Do not skip this.

One dataset, everything else varied. **pems08f**: mid-sized, 3548 test windows,
fastest full cell in the grid.

```
3 backbones × 4 arms (iso, emp, flat, rsh) × 1 seed = 12 training runs
each evaluated at n_steps ∈ {50, 100, 200} at the new floor
```

≈ 18 GPU-h, ~3 h wall at throttle 8. It answers four questions, all of which change
what the main grid costs:

1. **Does the prior-aware floor close the calibration gap?** Expect isotropic to move
   <1 point of coverage and the structured arms to gain ~5. If it does not, defect 2
   is not the explanation and the under-dispersion is real — which is also a result,
   but a different thesis.
2. **How many ancestral steps?** The grid spans 3.9 decades for iso and up to 6.8 for
   rsh. If resolution per decade is what matters, n_steps should scale with the range
   (≈50 / 65 / 85). If 50 suffices everywhere, the main grid's eval cost stays flat;
   if 200 is needed, eval cost goes up 4× and that decides seed count.
3. **Is the reshaped arm rescued?** It is the best prior in the family by held-out NLL
   (+0.15…+1.49 nats/mode vs +0.11…+1.19 for the shipped Kronecker) and currently the
   worst end-to-end, with 39–76 % of its variance un-denoised. If the floor fix
   rescues it, `rsh` — not `emp` — is the headline method.
4. **Does `flat` match `emp`?** Held-out NLL says the graph adds ≤0.05 nats on top of
   the temporal factor's 0.13–1.14. If test CRPS agrees, the negative result is
   established end-to-end instead of by offline likelihood alone.

**Lock the protocol on these answers, write it down, and do not touch it again.**

---

## 5. Stage 3 — the main grid

```
6 datasets × 3 backbones × 4 arms × 4 seeds = 288 runs
```

From the archived timings: ~30 min train + 18–59 min eval per cell (pemsbay is the
slow one at 10,419 test windows). So ≈ 290 GPU-h at n_steps=50, ~36 h wall at
throttle 8 — call it **2–3 days** with headroom. At n_steps=200 the eval side
triples; drop to 3 seeds if the pilot forces that.

Four seeds is the point of this stage. The archived campaign had 2, which gave
16/24 paired cells and **p = 0.076** — suggestive, not a result. Four takes the
central claim to significance, and it is the cheapest significance available.

Arms, and what each is for:

| arm | prior | what it establishes |
|---|---|---|
| `iso` | σ²I | the matched control |
| `emp` | Σ_T ⊗ Σ_S | the thesis's prior |
| `flat` | Σ_T ⊗ I | **the graph contributes nothing** — the missing ablation |
| `rsh` | empirical per-mode | the ceiling of the family; possibly the winner once resolved |

Storage: 288 runs × ~7 MB ≈ 2 GB of checkpoints on windowshome, which is now empty.
Residual caches are ~10 GB against an 8 GB quota, so **the dataset rotation is still
required** — run `pems08f pems08s pemsbay` first, then `metrla pems04f pems04s`, and
do not delete a cache until its cells report.

---

## 6. Stage 4 — analysis, free

Already written and re-runnable: `thesis_tables.py` (every number in Ch 5–7),
`prior_family_nll.py`, `sampler_floor.py`, `anc_sweep_report.py`, `config_parity.py`.

New things this rerun makes possible that the old one could not support:
- the `flat` arm as a paired end-to-end ablation of the graph;
- calibration numbers that mean something, because the arms are resolved equally;
- a seed count that supports a significance claim;
- one conditioning regime throughout, so no table sits on a retired arm.

---

## 7. Ordering, with the reason for each dependency

```
0. code fixes A–F + offline re-validation        Mac, free        ← blocks everything
1. backbones + residuals (18 cells)              ~1 wave, ~40 GPU-h
2. protocol pilot on pems08f (12 runs × 3 evals) ~3 h wall, ~18 GPU-h  ← sets n_steps + floor target
3. main grid (288 runs)                          ~2-3 days, ~290 GPU-h
4. tables, figures, offline ablations            Mac, free
```

Do not run stage 3 before stage 2 reports. The entire reason the last campaign
produced 1,177 models and an unfalsifiable calibration story is that the protocol
was never pinned down first.
