# Diffusion Methodology — Full Specification

_Last updated 2026-06-24. Every value below is verified against the code; file:line
references let you re-check. Scope: the **residual diffusion stage**. The backbone forecast
and residual extraction are upstream and treated as fixed inputs._

---

## 0. Overview

We forecast a spatiotemporal signal `Y ∈ ℝ^{N×H}` (N sensors, H horizon steps) in two stages:

1. **Backbone** (deterministic): Graph WaveNet (default) produces a point forecast `Ŷ`.
2. **Residual diffusion** (probabilistic): we model the **residual** `R = Y − Ŷ` with a
   continuous-time, graph-aware diffusion whose stationary prior is a structured
   Kronecker covariance `Σ = Σ_T ⊗ Σ_S`. A U-Net denoiser learns to reconstruct the clean
   residual; at test time we draw an ensemble of residuals and add them back to `Ŷ`.

The predictive distribution of `Y` is therefore `Ŷ + (residual samples)`. The **scientific
claim** lives entirely in the prior: a graph-structured, temporally-correlated prior
(`kronecker`/`structured`) should produce better-calibrated, sharper forecasts than an
isotropic `σ²I` prior (`isotropic`). Every experiment is a **matched kron-vs-iso pair**
sharing the same backbone, residuals, denoiser architecture, schedule, and seed.

Pipeline: `backbone → residuals → {kron diffusion, iso diffusion} → eval`.

### Methods / arms
The same diffusion machinery runs over several **methods** (the `--backbone` axis), each × {kron, iso}:
- **`gwn` / `gcn` / `mlp`** — diffuse the residual of a trained backbone (forecast-conditioned).
- **`raw`** — **pure-signal diffusion** (no trained backbone): diffuse the future signal `Y` directly,
  **conditioned on the history window** (DiffSTG-style), via `SignalDenoiser` (UGnet with `T_h>0`,
  input `[history, Y_t]` / `[history, zeros]`). Cache built by `scripts/build_signal_cache.py` from
  `split.npz`; the future signal is stored under the `residuals` key so the prior/`σ_r` reuse the same
  path, with `history` as the conditioning and `y_pred=0`. The signal prior is much stiffer than a
  residual prior (strong spatiotemporal correlation, `θ_max`~260) but DDIM stays stable.

---

## 1. Data representation

| Item | Value / choice | Where |
|---|---|---|
| Signal shape | `(B, N, H)` — batch, sensors, horizon | — |
| Modeled quantity | residual `R₀ = Y − Ŷ` (raw units) | residual extraction |
| **Training space** | **unit-variance**: model `R₀ / σ_r` | `train_diffusion.py:403` |
| `σ_r` | single scalar = std of **train** residuals; frozen, reused for val/test | `residual_dataset.py:28–36` |
| Conditioning input | **z-scored backbone forecast** `Ŷ` | `residual_dataset.py` |
| Mask | per-element validity `(B,N,H)`; missing/zero entries excluded from loss & metrics | `losses.py:32–35` |
| PEMS masking | PEMS residuals use `null_val=nan`; eval **must mask `y_true==0`** (SpecSTG/DiffSTG convention) | memory: pems-nullval |
| De-normalization | eval multiplies samples by `σ_r` to score in **raw units** | `eval_diffusion.py:190` |

Why unit space: keeps the loss scale dataset-independent and the prior's `θ` well-conditioned.
`σ_r` is a single scalar (not per-node/horizon) so the prior's structure carries all anisotropy.

---

## 2. The prior `Σ = Σ_T ⊗ Σ_S`

**File:** `src/diffusion/prior.py`. The prior is stored in its **joint eigenbasis**
`(V_S, V_T, θ)` so every forward/reverse op is a cheap per-mode scalar.

### 2.1 Temporal factor `Σ_T` (H×H)
- `delta_h = mean over valid of R₀²` per horizon → true per-horizon variance.
- `R_H = corrcoef(R_valid)` → empirical horizon-to-horizon correlation matrix.
- `Σ_T = diag(√delta) · R_H · diag(√delta)`.
- Eigendecomp `Σ_T = V_T diag(mu_T) V_Tᵀ`; clip `mu_T ≥ 1e-12`.
- **Ridge shift toward scaled-identity** `SHRINK_T = ρ = 0.1` (`prior.py`): `mu_T ← (1-ρ)·mu_T + ρ·μ̄`.
  - This is the **exact temporal twin of the spatial γ**: γ→∞ drives `Σ_S`→isotropic, ρ→1 drives
    `Σ_T`→isotropic. Applied as a smooth, **trace-preserving** shift, not the old hard
    condition-number clamp (`KAPPA_CAP`, removed).
  - Why the spread exists: `Σ_T` is near-singular because horizon residuals are strongly
    **collinear** (mean cross-horizon corr ≈0.36 metrla / 0.50 pemsbay; >56% of variance in one
    common mode), NOT because of heteroscedasticity (only 1.2–2.6× across horizons). So the
    condition number is driven by *tiny* eigenvalues, not a giant one; heavy tails (excess kurtosis
    8–52) make those tiny eigenvalues noisy.
  - Effect of ρ=0.1: pems-bay cond 116→44 (~matching the old κ=40), metr-la 23→16, pems08f 8→7 —
    keeps the real horizon correlation; the log-uniform-c̄ schedule handles any residual stiffness.
    Override per call via `load_prior(shrink_t=...)`; `0` = faithful empirical `Σ_T`.

### 2.2 Spatial factor `Σ_S` (N×N) — **the arm switch**
| `--spatial` | `Σ_S` | spatial precision `q_S` | meaning |
|---|---|---|---|
| `structured` (default) | `(L + γI)⁻¹` | `λ_S + γ` | spatio-temporal prior — **the thesis contribution** |
| `flat` | `I` | `1` | temporal-only ablation (`γ` forced to 0) |

- `L` = **symmetric normalized** graph Laplacian (eigenvalues in [0,2]); a symmetric operator is
  required so the Kronecker forward/reverse can rotate into an orthonormal eigenbasis. (UGnet's own
  spatial conv uses asymmetric random-walk supports `asym_adj(A), asym_adj(Aᵀ)` — we can't and don't
  reuse those in the prior; `gcn_self` is unnecessary since our loss has no per-graph-mode term.)
- **No graph surgery** (`REPAIR_ISOLATED = False`): the γ shift lifts every Laplacian zero mode, so
  isolated sensors and **multi-node disconnected components** (pems04f/s each have **12 components**:
  sizes 237/32/9/9/4/3/3/2/2/2/2/2) all get a finite, well-conditioned prior with no invented edges —
  the GAD/Vimal convention (rely on the shift, not reconnection). Degenerate zero-modes are
  basis-independent (all share precision γ), so the arbitrary null-space eigenvectors cancel.
- `λ_S, V_S` from cached Laplacian eigendecomposition.
- **`γ` is a FIXED hyperparameter = 0.8 for all datasets** (`GAMMA_DEFAULT`, `prior.py`), following
  **GAD (Rozada et al. 2025)**. This is deliberately *not* data-fit: a single principled,
  paper-aligned shift, the same convention Vimal uses (his `mas_gamma`, also a fixed `(L+γI)` shift,
  default `1e-3`). It supersedes the previous per-dataset fitted values (0.158/0.213/0.924).
  Overridable per call via `load_prior(gamma=...)` for sweeps. With normalized `λ_S∈[0,2]`,
  `q_S = λ_S + 0.8 ∈ [0.8, 2.8]` — a well-conditioned spatial precision. Resulting joint stiffness
  `θ_max/θ_min`: metrla ≈72×, pems04f ≈22× (both trainable; κ-cap on `Σ_T` still applies).

### 2.3 Joint precision `θ` (N×H)
- **Trace calibration** `τ² = N / Σ(1/q_S)` — rescales the spatial factor so its average
  eigenvalue does not shift total variance (`τ²=1` exactly when flat).
- Per-mode precision `θ[k,h] = q_S[k] / (τ² · mu_T[h])` (`prior.py:225–233`).
- **Optional joint-stiffness cap** `--stiffness-cap C` (default 0 = off): floor `θ` so
  `θ_max/θ_min ≤ C`. Trainability fallback for stiff structured priors; off by default to
  keep the prior faithful.

### 2.4 Isotropic baseline (`--prior isotropic`)
Scale-matched white noise `Σ = σ²·I` with the **same total variance** as the structured prior,
so only the *structure* differs, not the energy. Built by `_build_isotropic_prior`.

### 2.5 Reproducibility invariant
The prior is **rebuilt identically at eval** from `config.json` (`spatial`, `stiffness_cap`,
`backbone`, `seed`) — `eval_diffusion.py:91–98`. Train and test see the exact same `Σ`.

---

## 3. Forward (noising) process

**File:** `src/diffusion/forward_fcps.py`. SDE from GAD (Rozada et al. 2025).

### 3.1 SDE and closed-form marginal
`dr = −c(t)·Q·r dt + √(2c(t)) dW`, `Q = Σ⁻¹`, `W ~ N(0, I_{NH} dt)`. In the joint eigenbasis,
each mode `k` is independent with **closed-form** marginal:
```
c_k(t) | c_k(0) ~ N( e^{−c̄(t)·θ_k} · c_k(0),   (1 − e^{−2 c̄(t) θ_k}) / θ_k )
H(c̄)  = e^{−c̄ θ}          # signal retained, per mode
σ²(c̄) = (1 − e^{−2 c̄ θ}) / θ   # noise variance, per mode
```
`c̄(t) = ∫₀ᵗ c(s)ds` is the **cumulative noise level** — the axis everything is parameterized by.
The slowest mode mixes once `2 c̄(T) θ_min ≳ 5`.

### 3.2 Noise-level (c̄) parameterization — EDM-style (the 2026-06-18 fix)
Training samples the noise level **log-uniformly in c̄**, not uniformly in `t`, because the
stiff prior makes uniform-in-t spend nearly all steps in a degenerate regime. Constants
(`forward_fcps.py:190–202`):

| const | value | meaning |
|---|---|---|
| `_SNR_HI` | `5.0` | `c̄_hi·θ_min`: slowest mode fully mixed (`H = e⁻⁵ ≈ 0.007`) |
| `_NOISE_FLOOR` | `0.04` | target std of residual noise in final sample (~4%, unit space) |
| `c̄_lo` | `_NOISE_FLOOR²/2 = 8e-4` | barely-noised floor; fixed (not tied to θ_max) so every dataset's final sample is equally clean |
| `c̄_hi` | `_SNR_HI / θ_min` | upper noise level; mixes the slowest mode (per-dataset, depends on prior) |

- **Train sampling:** `c̄ ~ logU[c̄_lo, c̄_hi]` per batch (`sample_log_c_bar_torch`, l.216).
- **Conditioning variable** `tau = (log c̄ − log c̄_lo)/(log c̄_hi − log c̄_lo) ∈ [0,1]`
  (`c_bar_to_tau`, l.205). **The denoiser conditions on `tau`, not raw `t` or `c̄`.**
- **Forward sampler used in training:** `sample_R_t_from_c_bar_torch` (l.231) — rotate `R₀` to
  eigenbasis, scale by `H`, add `√σ²·Z`, rotate back. All float32 on device.

### 3.3 Schedules (needed only for the reverse grid + diagnostics, not for c̄ training sampling)
| schedule | `c(t)` | `c̄(t)` | params |
|---|---|---|---|
| **FCPSSchedule** (default) | `c_min + k·uᵅ`, `u=t/T` | `c_min·t + (c_0 − c_min·T)·u^{α+1}`, `c̄(T)=c_0` | `c_min=0.05`, `alpha=7.0`, `T=1.0` |
| ULSchedule (baseline) | `slope·t/T` | `slope·t²/(2T)` | `slope=200` |

- `FCPSSchedule.for_prior` **auto-derives `c_0`** from the fitted prior:
  `c_0 = max(c_min·T+1, 0.5·mix_target·mix_margin/θ_min)` with `mix_target=5`, `mix_margin=2`
  (`forward_fcps.py:65–79`). **`--c-0` on the CLI is IGNORED** — the auto value is computed,
  printed, and written back to `config.json` (`train_diffusion.py:424–432`). This was a footgun
  fix: a human-typed `c_0` misled eval.
- Validation `c_min ∈ (0,1)`, `alpha > 1`, `c_0 > c_min·T` enforced in `__post_init__`.

---

## 4. Denoiser network

**File:** `src/diffusion/denoiser.py` (`R0Denoiser`). Wraps the vendored **UGnet** from DiffSTG
(`third_party/DiffSTG/algorithm/diffstg/ugnet.py`).

### 4.1 Signature & I/O
```python
R0_hat = denoiser(R_t, tau, Y_hat)
#  R_t   (B, N, H)  noised residual
#  tau   (B,)       normalized noise level in [0,1]   (NOT raw t)
#  Y_hat (B, N, H)  backbone forecast (conditioning)
#  ->    (B, N, H)  predicted CLEAN residual R̂₀     (predicts R₀, not ε)
```
We predict `R₀` (clean target), not the noise `ε`; the network is otherwise unchanged from DiffSTG.

### 4.2 UGnet config (`make_ugnet_config`, denoiser.py:41)
| key | value | note |
|---|---|---|
| `d_h` | `32` (`--d-h`) | hidden channels |
| `T_p` | `H` | we feed only the H forecast steps |
| `T_h` | `0` | no history fed to denoiser |
| `F` | `1` | single feature channel |
| `V` | `N` | sensors |
| `A` | adjacency (float32) | same graph as the GWN backbone |
| `channel_multipliers` | `[1, 2]` (`--channel-multipliers`) | per-resolution expansion |
| `n_blocks` | `2` (`--n-blocks`) | residual blocks per resolution |
| `supports_len` | `2` | two asymmetric graph supports (`A`, `Aᵀ`) — handles directed graphs |

### 4.3 Internals
- **Noise embedding:** `tau` is scaled by `T_EMBED_SCALE = 1000.0` (denoiser.py:38) into UGnet's
  integer-step sinusoidal embedding range, then injected into every residual block via a learned
  linear map added to the feature channels.
- **Conditioning:** `Y_hat` enters as UGnet's `x_masked`; UGnet concatenates `x` and `x_masked`
  along time (→ 2H) and projects back to H. `pos_w/pos_d` are passed as dummy zeros (ignored).
- **Residual block:** temporal conv (`TcnBlock`, kernel 3×k) → spatial graph conv over the 2
  supports → LayerNorm → + time conditioning.
- **U-Net:** downsample over resolutions (`channel_multipliers`), middle = 2 residual blocks,
  symmetric upsample with skip connections.

---

## 5. Training

**Entrypoint:** `scripts/train_diffusion.py`. One `(dataset, prior, seed)` per invocation.

### 5.1 Per-step loop (`train_diffusion.py:521–540`)
1. draw `c̄ ~ logU[c̄_lo, c̄_hi]` (per batch);
2. forward-noise `R₀ → R_t` via `sample_R_t_from_c_bar_torch`;
3. `tau = c_bar_to_tau(c̄)`;
4. `R̂₀ = denoiser(R_t, tau, Ŷ)`;
5. `loss = compute_loss(...)`;
6. `loss.backward()`; **grad clip `‖g‖ ≤ 5.0`**; `optim.step()`.

### 5.2 Loss — GAD `gad_snr_weighted_mse` (`src/diffusion/losses.py`)
Training loss is **GAD's**: masked MSE on `R̂₀` weighted per sample by `w = 1/(1 + 2σ²·c(t))`, where
`c(t)` is the schedule **rate** (not cumulative `c̄`) — matches GAD's `β = 2σ²c(t)` exactly. Optional
`+ tv_weight·graph_tv(R̂₀)` (GAD's `5e-3·TV`, default 0). The legacy `snr_weighted_mse`/`compute_loss`
(weight on cumulative `v̄`) and `mahalanobis_mse` remain for the diagnostic/ablation but are **not** the
training path.

### 5.3 Optimizer & schedule
| | value |
|---|---|
| Optimizer | **AdamW** (GAD) |
| `lr` | `1e-3` (`--lr`); between DiffSTG's `2e-3` and GAD's `1e-4` |
| `weight_decay` | `1e-4` |
| LR scheduler | **`ReduceLROnPlateau`** on val-CRPS (`factor=0.5`, `patience=5`) — Vimal/DiffSTG style |
| Grad clip | `1.0` (max-norm, GAD) |

### 5.4 Training config / defaults
| flag | default | meaning |
|---|---|---|
| `--epochs` | `200` | max epochs |
| `--batch-size` | `32` | |
| `--patience` | `15` | **early stop** after N epochs without val improvement |
| `--backbone` | `gwn` | residual source (also `gcn`, `mlp`) |
| `--seed` | `0` (pipeline passes 42) | |
| `--c-min` | `0.05` | schedule floor |
| `--alpha` | `7.0` | schedule steepness |
| `--c-0` | ignored (auto-derived) | — |
| `--t-eps` | `1e-3` | lower t bound to avoid σ²=0 |
| `--stiffness-cap` | `0` (off) | joint-spectrum cap |

### 5.5 **Checkpoint selection** — val-CRPS (Vimal's protocol)
- Each epoch `val_crps` draws `--crps-k` (8) samples per clean val window via the reverse process
  (`--crps-steps` 40, eta=1) and computes the **fair ensemble CRPS** (raw units). `best.pt` saved when
  val-CRPS improves; `last.pt` every epoch. Field `best_val_mae` now stores best val-CRPS (legacy name).
- **`ReduceLROnPlateau`** on val-CRPS (`--lr-factor` 0.5, `--lr-patience` 5) — replaced the old
  `sched=None` flat-lr; Vimal/DiffSTG use a plateau scheduler.
- Early stop when `bad_epochs ≥ patience(15)` on val-CRPS. `val_loss` (GAD objective, no sampling)
  and the periodic diagnostic are logged but **don't drive selection**.

### 5.6 Validation diagnostic (every `--val-every` epochs, default 5)
On a fixed subsample of **fully-observed** val windows (`--val-windows`, default 512; clean so the
eigenbasis rotation is valid), noise close to full, reverse-sample (`--val-steps`, default 100),
and score: **recon MAE/MSE** (raw units), **sliced Wasserstein-1** (`--val-proj`, default 128
projections), **energy distance**, **mahalanobis**. Logged to wandb. **Does not drive selection.**
A dedicated RNG (`seed+999`) keeps it comparable across epochs (falls back to CPU gen on MPS).

### 5.7 Utilities
- `--lr-find N`: Smith LR range test over `[1e-6, 1.0]` for N steps, writes `lr_find.json`, exits.
- `--smoke`: 1 epoch, batch 8, ≤5 train batches, tiny val diagnostic, wandb disabled.
- `--fresh`: delete checkpoint dir and restart. Otherwise resumes from `last.pt` (model, optim,
  epoch, best_val, RNG states all restored).

### 5.8 Logging & artifacts (Vimal-style)
- **`train.log`** — readable per-epoch line: `epoch | train val | val_CRPS val_MAE | lr | time`,
  with `[saved best]` and `[early stop]` markers.
- **`validation_history.json`** — `{select_metric, best_epoch, history:[{epoch, train_loss, val_loss,
  val_crps, val_mae, lr, time, (recon_mae/w1/mahalanobis on diagnostic epochs)}]}`, rewritten each epoch.
- **`test_results.json`** (from `--eval-after`) — `{run_id, dataset, prior, …, best_epoch,
  test_results:[{n_steps, mae, rmse, mape, crps, winkler, coverage, w1, energy, …}]}` per `--eval-steps`.
- **wandb** still logged in parallel (`train/loss`, `val/loss`, `val/crps`, `val/mae`, `lr`).
- `config.json` written at start + re-written after `c_0` is derived (for exact eval rebuild).

### 5.9 Output layout (`src/utils/paths.py`)
```
results/diffusion/<exp>/<ds>/<backbone>/<prior>/seed<s>/   best.pt last.pt config.json train.log validation_history.json
results/eval/<exp>/<ds>/<backbone>/<prior>/seed<s>/        test_metrics.json  test_results.json
results/eval/<exp>/summary.csv  summary.md                 (from scripts/summarize_runs.py)
```
`run_id = diff_<backbone>_<ds>_seed<s>_<cfg_hash>`. The 8-char `cfg_hash` excludes
`{seed, epochs, batch_size, lr, weight_decay, patience, exp, lr_find*}` so the *method* hash is stable.

---

## 6. Reverse (sampling) process

**File:** `src/diffusion/reverse.py` (`reverse_sample`). DDIM-style sweep over a **geometric grid
in c̄** from `c̄_hi` (≈ prior) down to `c̄_lo` (≈ clean): `geomspace(c̄_hi, c̄_lo, n_steps+1)` —
even in log-c̄, so all modes are well-spaced.

Per step `i` (current `c̄`, next `c̄`):
1. `H, σ²` at both noise levels (per mode);
2. `tau = c_bar_to_tau(c̄_cur)`, `R̂₀ = denoiser(R_t, tau, Ŷ)`;
3. Tweedie noise estimate `ε̂ = (C_t − H_cur·Ĉ₀)/√σ²_cur` (σ² floored at 1e-20);
4. DDIM update `C_next = H_next·Ĉ₀ + √drift_var·ε̂ + η·√σ̃²·ζ`, where
   `σ̃² = η²·σ²_next·(1 − (σ²_next/σ²_cur)(H_cur/H_next)²)`, `drift_var = σ²_next − σ̃²`.

| param | train-diagnostic | eval | meaning |
|---|---|---|---|
| `n_steps` | 100 | 100 (pipeline uses 50) | reverse steps |
| `eta` | (diagnostic uses 0) | **1.0** | 0 = deterministic DDIM; **1.0 = full ancestral / proper ensemble** |
| `R_T` | drawn from prior | drawn from prior | start state (can be supplied) |

Eval uses `eta=1.0` so the predictive intervals are properly dispersed (the during-training
`eta=0` Winkler under-covered).

---

## 7. Evaluation

**Entrypoint:** `scripts/eval_diffusion.py` (`evaluate`). Loads `best.pt`, rebuilds prior+schedule
identically, de-normalizes samples to raw units.

### 7.1 Phases & metrics
**Point + interval** (`k` samples/window, `eta=1.0`), per horizon over masked-valid positions:
| metric | def | file |
|---|---|---|
| MAE | `mean|mean(samples) − R₀|` (forecast error = residual error) | eval_diffusion.py |
| RMSE | `√mean((mean − R₀)²)` | eval_diffusion.py |
| **MAPE** | `mean(|mean − R₀| / |y_true|)`, `y_true = y_pred + R₀` (reconstructed raw GT) | eval_diffusion.py |
| Winkler score | `(hi−lo) + (2/α)(max(lo−y,0)+max(y−hi,0))` at central (1−α) interval (= Vimal's MIS) | `intervals.py` |
| Coverage | empirical fraction inside the interval (target `1−α`) | eval_diffusion.py |
| **CRPS (normalized)** | **CSDI/DiffSTG/SpecSTG convention** — 19-quantile pinball / `Σ|y|`, `~0.06–0.10`. **This is the headline number comparable to Vimal/papers** (`quantile_crps_num`, `src/eval/crps.py`) | eval_diffusion.py |
| CRPS (energy) | raw-unit fair ensemble `E|X−y| − ½E|X−X'|` (also reported as `crps_energy`) | eval_diffusion.py |

**All metrics now computed in raw FORECAST space**: the forecast is reconstructed as
`y_pred + residual_sample` and **clipped to [0,∞)** (non-negative flow/speed, matches DiffSTG), then
scored against `y_true`. **Metric set matches Vimal's** (MAE/RMSE/MAPE/**normalized-CRPS**/MIS) plus
coverage, energy-CRPS, W1, energy. Reported at each `--eval-steps` (e.g. `100,50,25`) so `CRPS@100` is
directly comparable. `y_true = y_pred + residual` (raw) is reconstructed in the dataset.

**PEMS `y==0` masking** (`--no-mask-zero` to disable; on by default): drops `y_true==0` points from
all metrics to match SpecSTG/DiffSTG. Speed datasets already exclude zeros via the stored mask
(`null_val=0`); flow datasets (`null_val=nan`, mask all-true) is where this bites.

**Distribution fidelity** (complete-case windows, `w1_samples` per window), per horizon:
| metric | def | file |
|---|---|---|
| Sliced Wasserstein-1 | `w1_proj` random projections, 1D W1 via quantiles | `distribution.py:19` |
| Energy distance | `√(2E‖g−t‖ − E‖g−g‖ − E‖t−t‖)` | `distribution.py:44` |

### 7.2 Eval parameters
| flag | default | pipeline value | meaning |
|---|---|---|---|
| `--k` | `50` | `30` | samples/window for MAE/MSE/Winkler/CRPS |
| `--n-steps` | `100` | `50` | reverse steps |
| `--eta` | `1.0` | `1.0` | full stochastic reverse SDE |
| `--alpha` | `0.1` | `0.1` | → central **90%** interval |
| `--w1-windows` | `512` | — | complete-case windows for W1/energy |
| `--w1-samples` | `4` | `4` | samples/window for W1/energy |
| `--w1-proj` | `128` | — | projections for sliced W1 |
| `--gen-batch` | `1024` | `1024` | OOM-guard batch for reverse_sample |
| `--max-windows` | `0` (all) | all | cap test windows |

### 7.3 Output
`test_metrics.json` — overall + per-horizon `_h` arrays for every metric, plus run metadata
(`ckpt_epoch`, `k_samples`, `n_steps`, `eta`, `alpha`, `n_test_windows`). `--all` evaluates all
datasets × both priors and prints a kron-vs-iso comparison table sorted by metric.

---

## 8. Full run lifecycle (Vimal-style, combined train+eval)

**One combined slurm task per `(dataset, backbone, prior)`** trains and evaluates in-process:
```
[grid task]  train (best.pt by val-CRPS) → locked test eval at 100,50,25 → test_results.json
```
- **`scripts/daic/submit_array.sh <seed>`** launches the whole grid as **one SLURM array**,
  **dataset-major**: `metrla×{gwn,gcn,mlp,raw}×{kron,iso}`, then `pemsbay×…`, … (48 cells), with a
  concurrency throttle so a dataset's every method runs before the next dataset. The combo list is
  `logs/<EXP>/grid/grid_seed<s>.tsv` (one row per task; `submit_grid.sbatch` reads `SLURM_ARRAY_TASK_ID`).
- `submit_all.sh <seed>` (per-job fan-out) and `submit.sh traineval …` (single cell) remain for one-offs.
- Each job: `train_diffusion.py … --eval-after` → after training, reloads `best.pt` and runs the
  locked test eval at `--eval-steps` (e.g. `100,50,25`), writing `test_results.json`.
- **Per-run artifacts** (all under `results/diffusion/<exp>/<ds>/<bb>/<prior>/seed<s>/`):
  `best.pt`, `last.pt`, `config.json`, **`train.log`** (readable per-epoch line), and
  **`validation_history.json`** (epoch-by-epoch `train_loss/val_loss/val_crps/val_mae/lr`, + `best_epoch`).
  `test_results.json` lands under `results/eval/<exp>/<ds>/<bb>/<prior>/seed<s>/`.
- **`train.log`** line: `epoch | train val | val_CRPS val_MAE | lr | time`, plus `[saved best]` markers.
- **Leaderboard:** `python scripts/summarize_runs.py --exp <EXP>` → CRPS-sorted `summary.csv`/`.md`
  + per-dataset kron-vs-iso Δ%.
- wandb is still logged in parallel. Backbone + residuals shared per `(dataset, seed)`.

---

## 9. Parity with Vimal — all aligned

The **math (prior, forward, reverse, loss, denoiser) follows GAD**; the **run harness follows Vimal**.

| Aspect | Vimal | Us (now) |
|---|---|---|
| Granularity | one seed/run | ✅ one seed/run |
| Train + test eval | **one job** | ✅ one job (`--eval-after`) |
| Best ckpt by | **val-CRPS** | ✅ val-CRPS (`val_crps`, k=8, 40 steps) |
| LR schedule | plateau scheduler | ✅ `ReduceLROnPlateau` on val-CRPS |
| Per-epoch log | plain `{exp}.log` | ✅ `train.log` readable line |
| `validation_history.json` | yes | ✅ emitted each epoch |
| `test_results.json` | multi `sample_steps` | ✅ multi `--eval-steps` |
| Test metrics | MAE RMSE MAPE CRPS MIS | ✅ + coverage, W1, energy (superset) |
| `y==0` masking | yes (PEMS) | ✅ default on (`--no-mask-zero` to disable) |
| Leaderboard | CRPS-sorted CSV/MD | ✅ `summarize_runs.py` |
| Two-phase / hybrid loss | MAS + author-eps blend, warm-start | ✗ **not needed** — our GAD x₀ loss is the single principled objective |

**On Vimal's "two phases":** it is *not* pretrain→finetune — it's a hybrid loss
`λ_mas·MAS + λ_author_eps·DDPM_eps + λ_x0_future·x0_recon` (+ optional warm-start). He blends DiffSTG's
original ε-loss back in because his MAS spectral loss alone underperforms. We don't have that problem:
our loss **is** the GAD x₀ objective, and `λ_x0_future` (direct x₀ reconstruction) is exactly what we
already optimize. So we keep a single clean objective — no hybrid, no warm-start.

### GAD alignment (2026-06-24) — "change only Q, keep the rest GAD"

After grounding against the GAD reference (`github.com/vimalkb7/gad`: `sde.py`, `exp_gasde.py`),
the method was aligned to GAD on everything except the prior **Q** (our Kronecker contribution):

| Aspect | Now (GAD-aligned) | was |
|---|---|---|
| Schedule | `alpha=4`, `c_min=0.1` (GAD) | `alpha=7`, `c_min=0.05` |
| **`c_0` (= GAD's `S`)** | **uniform ≈6.25** all datasets | per-dataset 37–74 |
| Prior scale | normalized so `θ_min=γ=0.8` (GAD's smooth-mode precision); `σ²` (`sigma2_scale`) absorbs the rescale → `Σ_stat=Σ_data` preserved. **Exact reparam** of the old large-`c_0` prior. | `θ_min≈0.07–0.15`, `σ=1` |
| Train noise sampling | **power-law `c̄ = c_0·u^0.65`** (`sample_c_bar_powerlaw_torch`) | log-uniform-in-c̄ |
| Net conditioning | **diffusion time `t`** (`t = schedule.t_of_c_bar(c̄)`), GAD's convention | normalized `tau` |
| Loss | **GAD `gad_snr_weighted_mse`**: `w = 1/(1+2σ²·c(t))` (rate `c(t)`) `+ tv_weight·graph_tv` | SNR weight on `v̄` (cumulative) |
| TV penalty | `--tv-weight` (GAD uses `5e-3`); **default 0** (residuals are rough → may be misspecified) | none |
| Optimizer | **AdamW**, grad-clip **1.0** (GAD) | Adam, clip 5.0 |
| Reverse | **DDIM** = stable integrator of GAD's reverse SDE; `reverse_sample_em` (true Euler–Maruyama) kept for the appendix/ablation | DDIM, tau-conditioned |
| Predict / forward / marginal | x₀ / graph-OU / closed-form — already identical to GAD | (unchanged) |

**Why DDIM not GAD's EM:** EM is only conditionally stable; on our stiffer Q it diverges below
~3000 steps (pemsbay `θ_max≈114` after normalization, mechanism `c(T)·θ_max·dt ≫ 1`). DDIM
integrates the same reverse SDE exactly (stable at 100 steps). The integrator is forced by the Q
change; the model/score/prior are GAD's.

**Run scale = Vimal, not GAD's 5000-epoch/batch-128** (GAD's is for tiny graph-signal data); PEMS-scale
uses Vimal's reference (≈300 epochs, early stop, CRPS selection, `sample_steps≈40` for val).

### γ resolved (no longer a blocker)
`γ` is now a **fixed 0.8** for all 6 datasets (GAD convention) — see §2.2. The structured
Kronecker arm builds for **all** datasets including the 3 new ones (pems04f/pems04s/pems08s).
Verified: priors construct with healthy, trainable stiffness for every dataset.
