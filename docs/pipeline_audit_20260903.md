# Pipeline audit — 2026-09-03

Every hyperparameter, resolved from the code as it actually runs; what is wrong;
what is *not* wrong; and what it takes to beat the isotropic control.

Nothing below is quoted from a docstring. Every number was produced by loading the
real priors and running the real sampler against the archived residuals.

---

## 1. Every parameter, and where its value comes from

### 1.1 Global constants — identical for all six datasets

| symbol | value | where | is it defensible? |
|---|---|---|---|
| γ (spatial shift, `Σ_S = (L+γI)^{-1}`) | **0.8** | `prior.GAMMA_DEFAULT` | yes — a held-out-NLL sweep over 18 cells picks 0.8 as the best shared value; a per-cell oracle buys +3.1 % |
| ρ (temporal ridge, `μ_T ← (1-ρ)μ_T + ρμ̄`) | **0.1** | `prior.SHRINK_T` | yes — tames pemsbay's cond 116→44, barely touches the mild sets |
| `c_min` | 0.1 | `--c-min` | GAD-literal |
| `c_0` (= GAD's S, total heat) | **7.0** | `--c-0` | yes, **and it is per-dataset-correct by construction** — see §1.3 |
| α (schedule exponent) | 4.0 | `--alpha` | GAD-literal |
| `c_lo` (training noise floor) | **8e-4** | `FCPSSchedule.c_lo` = `0.04²/2` | **no — this is defect D2** |
| `NOISE_FLOOR` | 0.04 | `forward_fcps._NOISE_FLOOR` | the reasoning behind it (`σ ≈ √(2c̄)`) is the θ→0 limit only |
| loss | `snr` | `--loss` | yes — the alternative was measured and is far worse (§3.1) |
| noise sampler | `loguniform` | `--noise-sampler` | yes |
| normalisation | `pernode` | `--norm` | yes — per-sensor σ spans 3.1–12.3× vs ≤2.5× across horizons |
| denoiser | GCNN, 17,305 params | `--denoiser gcnn` | fixed by design; the ladder varies the data, not capacity |
| sampler | ancestral, `n_steps=50`, `k=100` | eval flags | **no — defects D1/D2** |
| terminal floor at sampling | **0.02**, hardcoded | `reverse.reverse_sample_ancestral` | **no — defect D2** |
| selection signal | val CRPS, `k=4`, 512 windows, ancestral@50 | `--val-k`, `--fast-val-steps` | **no — defect D3** |

### 1.2 Per-dataset derived quantities (these *are* adapted, correctly)

| quantity | how | range over the 18 cells |
|---|---|---|
| σ_r (per-sensor residual scale) | train-split per-node std, floored at 0.1·global | p95/p5 = 3.07 – 12.34 |
| τ² (`= N/Σ(1/q_S)`) | trace calibration of `Σ_S` | 1.5168 (pems04) – 1.7001 (metrla) |
| σ² (`sigma2_scale`) | absorbs the θ rescale, then a variance match | 0.75 – 583 |
| `D_H`, `R_H` | per-horizon second moment + Pearson correlation | fitted per cell |
| λ_S, V_S | normalised Laplacian of the repaired road graph | N = 170 / 207 / 307 / 325 |

### 1.3 The one thing that *is* matched across datasets, and it matters

`_normalize_scale` rescales the precision so `θ_min → γ = 0.8` for **every** dataset,
backbone and arm. Measured, all 72 (dataset × backbone × arm) combinations:

```
2·c₀·θ_min = 11.20     — identical everywhere, GAD's target ≈ 11.2
```

So the **top** of the noise range is prior-adaptive by construction: the slowest mode
is guaranteed to mix on every dataset, which is why a single `c_0 = 7` is legitimate.

The **bottom** is not. `θ_max` is left free and spans six orders of magnitude:

| arm | θ_min | θ_max | condition number |
|---|---|---|---|
| iso | 0.8 | 0.8 | 1 |
| flat (`Σ_T ⊗ I`) | 0.8 | 4.6 – 69.2 | 6 – 87 |
| emp (`Σ_T ⊗ Σ_S`) | 0.8 | 16.1 – 232.3 | 20 – 290 |
| rsh (empirical per-mode) | 0.8 | 151 – 842,145 | 189 – 1,052,681 |

A fixed `c_lo` therefore resolves the arms to wildly different fidelities. That is the
root of defects D1–D2.

> Side result worth keeping: `flat` → `emp` triples `θ_max` (4.6→16.1, 27→94, 58→200).
> **The graph factor triples the stiffness and buys ≤0.05 nats/mode of fit.** That is
> the cost/benefit of the graph in one line.

---

## 2. What is actually wrong — four defects, all pushing the same way

Every one of them penalises a high-θ-spread prior more than a flat one, so all four
are biased *against* the thesis's own method.

### D1. The ancestral step discards the posterior variance of `x₀`. **This is the big one.**

`reverse_sample_ancestral` draws `x_{c'} ~ N(mean, v_post)` with

```
v_post = s_d·s_p / (a_d²·s_p + s_d)
```

which is the variance of `q(x_{c'} | x_c, x₀)` — i.e. it treats `x̂₀` as **known**. It
is not: given `x_c`, the posterior over `x₀` still has variance
`σ_p² = 1/(a_c²/s_c + 1/v)`, and that uncertainty propagates through the mean.

This is exactly DDPM's `σ_t` ambiguity (Ho et al., `β̃_t` vs `β_t`). The code takes the
`β̃` **lower bound**. For isotropic noise the gap is small — which is why nobody
notices it on images. For a stiff spectrum it is not:

| θ·Δc | coded / exact step variance |
|---|---|
| large (≥ 8) | 1.0000 |
| 0.04 | 0.979 |
| 0.008 | **0.520** |

Under the model the exact reverse transition is `N(a_d·x_c, v(1-a_d²))`. The mean
coefficient the code computes is **already exact**; only the variance is wrong. The
correction

```
var = v_post + (v_post·a_p/s_p)² · σ_p²,     σ_p² = 1/(a_c²/s_c + 1/v)
```

reproduces `v(1-a_d²)` **to 6 decimal places** across θ ∈ [0.8, 5000] and
Δc ∈ [1e-3, 0.5]. It uses only quantities the sampler already holds.

### D2. The sampler stops 25× above the noise floor the denoiser was trained to

`c_lo = 0.02` is hardcoded; training draws go to `8e-4`; `eval_diffusion.py` never
passes the kwarg. The variance-weighted fraction left un-erased,
`f(c) = Σ var_m(1-e^{-2θ_m c})/Σ var_m`, caps what any sampler can deliver:

| arm | f(0.02) | ceiling 1−f | f(8e-4) | ceiling |
|---|---|---|---|---|
| iso | 3.15 % | 0.9685 | 0.13 % | 0.9987 |
| flat | 10.6 – 18.3 % | ~0.85 | 0.45 – 1.25 % | ~0.99 |
| emp | 19.0 – 28.2 % | ~0.74 | 0.89 – 2.4 % | ~0.98 |
| rsh | 38.9 – 76.4 % | ~0.53 | 4.4 – 18.2 % | ~0.90 |

Even at `8e-4` the reshaped arm is only at 0.90, because the floor is
prior-independent while the ceiling (`c_0`) is not. To equalise fidelity the floor
must be solved per prior: `f(c_lo) = 1.3e-3` gives 8.1e-4 (iso), 4e-5–1.2e-4 (emp,
**7–20× lower**), 1.1e-6–1.9e-5 (rsh, **37–721× lower**).

### D1 + D2, measured end to end

Oracle test: draw the data **from the arm's own stationary law** (model perfectly
specified) and use the **Bayes-optimal denoiser**, so every deviation is the sampler's.
Dispersion ratio = Var(sampled)/Var(target), pems08f/gcn:

| arm | ceiling | coded @50 | **corrected @50** | coded @200 | coded @800 |
|---|---|---|---|---|---|
| iso, c_lo=0.02 | 0.9685 | 0.898 | **0.968** | 0.950 | 0.961 |
| iso, c_lo=8e-4 | 0.9987 | 0.889 | **0.999** | 0.971 | 0.992 |
| flat, c_lo=8e-4 | 0.9895 | 0.876 | **0.992** | 0.958 | 0.984 |
| emp, c_lo=8e-4 | 0.9799 | 0.864 | **0.981** | 0.952 | 0.979 |
| rsh, c_lo=8e-4 | 0.8950 | 0.795 | **0.896** | 0.869 | 0.888 |

**The corrected sampler hits the theoretical ceiling at 50 steps — better than the
coded one reaches at 800.** A 16× compute saving *and* it is exact.

Delivered spread under the locked protocol vs fully fixed, 50 steps:

| arm | locked (c_lo=0.02, coded) | fixed (c_lo=8e-4, corrected) | gain |
|---|---|---|---|
| iso | 0.898 | 0.999 | **+11 %** |
| emp | **0.690** | **0.981** | **+42 %** |
| rsh | 0.495 | 0.896 | +81 % |

The structured arm gains roughly **four times** what the control gains. That is the
whole calibration story, and it was never a property of the prior.

### D3. The CRPS estimator is the biased one, and it drives checkpoint selection

`empirical_crps` uses `w = (2i−K−1)/K²`, i.e. `ΣΣ|x_i−x_j|/(2K²)` — the **naive**
estimator, not the fair `1/(2K(K−1))` its docstring claims. Its expectation is
`CRPS_true + E|X−X'|/(2K)`: a positive bias **proportional to ensemble spread**, so at
equal true skill the *narrower* ensemble scores better.

Simulated on a calibrated unit-normal forecast (20,000 points):

| K | coded | fair | coded bias |
|---|---|---|---|
| **4** | 0.70600 | 0.56464 | **+25.1 %** |
| 8 | 0.63658 | 0.56613 | +12.8 % |
| 100 | 0.57164 | 0.56600 | +1.3 % |

And the decisive case — two arms with equal true skill, one under-dispersed ×0.87
(what the oracle test measures for `emp` under the locked protocol):

| K | coded verdict | fair verdict |
|---|---|---|
| 4 | under-dispersed arm looks **+2.27 % better** | it is 0.53 % **worse** |
| 100 | −0.45 % | −0.59 % |

`best_crps.pt`, **early stopping** and the **LR plateau** are all keyed off
`_fast_val` → `periodic_val_eval(..., k=4, anc_list=[50])`, which calls
`reverse_sample_ancestral` **without `c_lo`** → 0.02. So the structured arm's
checkpoint is chosen by a k=4 CRPS with a +25 % spread-proportional bias, computed
with a sampler that has already discarded 19–28 % of its variance. The selection stage
actively rewards under-dispersion, by more than the entire ±1–2 % effect being measured.

At k=100 the test numbers are only mildly affected (+1.3 %); **the damage is
concentrated in selection, not scoring.**

### D4. The arms are not variance-matched, despite §5.4 claiming they are

`_build_isotropic_prior` uses `R[mask].var()` — centred, over *all* masked entries.
The Kronecker path's `D_H` uses `E[r²]` — uncentred, over *fully observed windows only*.
Measured `Estat_emp/Estat_iso − 1`:

| | worst | typical |
|---|---|---|
| over-matched | **+4.75 %** (metrla/mlp), +4.67 % (pemsbay/mlp) | +0.1 – 0.5 % |
| under-matched | **−2.86 %** (metrla/gwn), −1.72 % (metrla/gcn) | — |

Correlation with the end-to-end CRPS gap is only r = +0.23 (n=18), so this is **not**
the reason the structured arm loses. But METR-LA is the one dataset with a *negative*
mismatch on two backbones and it is the one dataset where structure loses on all three
— worth fixing, and it is a stated claim in the text.

---

## 3. What is *not* wrong — hypotheses tested and killed

### 3.1 The whitened / PriorGrad loss is not the answer. It is much worse.

`whitened_mse` weights each mode by θ, and is wired into `_loss` (the old "coded but
not wired" note is stale). It ran on 178 cells in `20260721_ladder` (`emp_w`, `rsh_w`)
and 24 in `20260727_kron`. Paired against iso on the forecast-only arm, locked test
protocol:

| comparison | n | wins | mean Δ CRPS |
|---|---|---|---|
| iso vs emp (snr) | 66 | 27 | **+1.51 %** |
| iso vs rsh (snr) | 66 | 16 | +0.50 % |
| iso vs **emp_w** (whitened) | 58 | **1** | **−8.50 %** |
| iso vs **rsh_w** (whitened) | 57 | **0** | **−13.92 %** |
| emp → emp_w, same cells | 58 | 1 | **−10.94 %** |

Why: weighting by θ *up*-weights the high-θ modes, which carry the *least* variance
(`var = σ²/θ`). The loss ends up dominated by modes that contribute almost nothing to
a raw-unit CRPS — worst for `rsh`, whose θ spans 6 decades. **The whitened loss
optimises a metric orthogonal to the one we report.** Keep `snr`.

### 3.2 The time embedding resolves the whole noise range

`_tau_embed` uses `exp(linspace(0,6))` = 1…403, finest resolvable Δt ≈ 0.0025. The
training range `c̄ ∈ [8e-4, 7]` maps to `t ∈ [0.008, 1]`, and **0.0 %** of log-uniform
draws fall below the embedding's finest scale. Not a defect.

(It does show D2 again in another coordinate: the locked sampler grid spans only
`t ∈ [0.185, 1]`, so it never visits the bottom 18 % of the schedule time the denoiser
was trained on.)

### 3.3 γ = 0.8 is already the right shared value

Held-out ΔNLL over 18 cells: 0.02→+0.453, 0.2→+0.716, **0.8→+0.740**, 3.2→+0.733.
Flat above 0.4; a per-cell oracle buys +3.1 %. Fitting γ *and* a free exponent adds
nothing. Not a tuning opportunity.

### 3.4 Conditioning is correct

`C = [X|Ŷ]` verified by row alignment: `|y_split − (ŷ+r)|` mean 1e-12…5e-9 against a
shift-1 control of 0.95–17 raw units.

### 3.5 The prior itself is *not* the problem

Held-out NLL per mode vs isotropic, fit on train, scored on val, one free scale each:
temporal **+0.13…+1.14 nats/mode**, `Σ_T ⊗ Σ_S` +0.11…+1.19, reshaped **+0.15…+1.49**.
The structured prior is a decisively better density than isotropic. It is the
machinery around it that throws the advantage away.

---

## 4. Why we do not currently beat iso, and what it takes

The prior is better by 0.11–1.49 nats/mode, and the pipeline converts that into
+1.5 % CRPS on the mlp rung and −1 % on GWN. The four defects explain the gap:

1. selection picks the narrowest-ensemble epoch (D3, +25 % bias at k=4)
2. …using a sampler that has already thrown away 19–28 % of the structured arm's
   variance and only 3 % of the control's (D2)
3. …and loses a further 11 % to the missing posterior variance (D1)
4. …between two arms whose total variance differs by up to 4.75 % (D4)

Fixing D1+D2 alone raises the structured arm's delivered spread from 0.690 to 0.981
while the control moves 0.898 → 0.999. **The structured arm recovers 4× more.** Since
CRPS at a fixed conditional mean is minimised at correct dispersion, and the structured
prior already wins on the conditional mean (MAE 33/36 cells, p = 1e-7), that is where
the win comes from.

### The novelty this produces

Three claims, each now backed by a measurement rather than an argument:

1. **A corruption prior matched to the residual's temporal second-order structure
   beats white noise**, in proportion to the structure the backbone leaves behind.
2. **The graph half is worth almost nothing** — ≤0.05 nats/mode on top of the temporal
   factor while *tripling* the condition number, and Cartesian/strong product graphs
   score *below* white noise in 18/18 cells.
3. **DDPM's `σ_t` ambiguity is benign for isotropic noise and severe for a stiff
   structured prior — and for the linear-OU/graph-heat forward there is an exact
   closed form that removes it.** Both ends of the training noise range must be set
   from the prior's own spectrum, or a prior comparison silently measures the sampler.

Claim 3 is a genuine methodological contribution, it is cheap to demonstrate, and it
generalises beyond this thesis: *anyone* comparing structured against isotropic
diffusion priors under a shared sampler budget has the same confound.

---

## 5. Fixes, in dependency order

| # | fix | file | effort |
|---|---|---|---|
| **F1** | posterior-variance correction in the ancestral step | `src/diffusion/reverse.py` | ~4 lines |
| **F2** | `c_lo` defaults to `schedule.c_lo`; add a prior-aware floor solving `f(c_lo)=1.3e-3`, recorded in `config.json` | `reverse.py`, `forward_fcps.py`, `train_diffusion.py` | small |
| **F3** | fair CRPS: `w = (2i−K−1)/(K(K−1))` | `src/eval/crps.py` | 1 line |
| **F4** | raise `--val-k` from 4 (with F3 the bias is mostly gone; k=8–16 removes the rest) | `histcond_cell.sbatch` | flag |
| **F5** | match the arms exactly — one estimator, one window subset | `src/diffusion/prior.py` | small |
| **F6** | `Σ_S` to unit diagonal | `src/diffusion/prior.py` | small |
| **F7** | write `history` into the residual npz | `extract_residuals.py` | small |

**Verification before any GPU time** (all free, all already written): re-run the oracle
sampler test and confirm every arm reaches its ceiling at 50 steps; re-run
`prior_family_nll.py` under F5+F6 and confirm γ=0.8 still holds.

Then the staged run from `docs/rerun_plan_20260902.md` — backbones → protocol pilot →
grid — with one change: the pilot now also has to confirm that F1's correction, which
is exact under the Gaussian model, behaves as expected with a *learned* denoiser,
where the true posterior over `x₀` is neither Gaussian nor exactly `σ_p²`.
