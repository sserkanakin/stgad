# Thesis overview — structured residual diffusion for traffic forecasting

A self-contained walkthrough of the entire approach. Reads top-to-bottom; every formula has the *why* alongside the math.

---

## 1. The problem

**Probabilistic spatiotemporal traffic speed forecasting.** Given the past hour of traffic-speed observations at 207 (METR-LA) or 325 (PEMS-BAY) road sensors — 12 time steps at 5-minute resolution — predict the *distribution* of speeds at the next 12 steps. Not just the most likely speed, but the entire predictive density, so downstream consumers can compute confidence intervals, expected delays, or risk-aware routes.

State of the art on these datasets focuses on **point forecasting** — predict a single number per (sensor, horizon) and report MAE/RMSE. Graph WaveNet (Wu et al., 2019) is the canonical baseline: a graph convolutional network with dilated temporal convolutions, achieving MAE of ~3.5 mph at 60 minutes on METR-LA.

The gap: **point forecasts can't quantify uncertainty.** They tell you the mean prediction but not the spread, the bias, or the calibration. For decisions under uncertainty (routing, dispatch, scheduling) that's a real loss.

---

## 2. The contribution

A **diffusion model over residuals**, conditioned on a deterministic backbone's forecast. Specifically:

1. Train a deterministic GWN backbone to produce point forecasts `ŷ`.
2. Define the residual `r = y_true − ŷ` (per sensor per horizon).
3. Train a denoising score network to generate samples from `p(r | history, ŷ)`.
4. At inference: predict K=50 residual samples, add each to `ŷ`, get K predictive samples in mph.

The key design choice is the **diffusion's noise covariance**. Standard score-based diffusion uses isotropic noise `Σ = I`. We use a **Kronecker-structured prior**:

$$
\Sigma_{\text{prior}} = Q_H \otimes \Sigma_S
$$

where `Q_H` (12×12) captures temporal correlation across forecast horizons and `Σ_S` (N×N) captures spatial correlation across sensors. This structure provides data-aware inductive bias: the diffusion process knows that "nearby in time" and "nearby in space" residuals are correlated, so it doesn't have to learn this structure from scratch.

The thesis claim: this structured prior gives sharper, better-calibrated probabilistic forecasts than an isotropic-prior baseline at matched compute.

---

## 3. The pipeline

```
                      ┌───────────────────────────────────────┐
                      │  Deterministic backbone: GWN          │
                      │  trained once, frozen afterwards      │
                      └──────────┬────────────────────────────┘
                                 │ ŷ = f_back(x_history)
                                 ▼
   x_history ─┐         ┌─────────────────────┐
              ├────────►│   Score network     │ ──► r̂₀ (predicted clean residual)
   ŷ          │         │   s_θ(r_τ, τ, c)    │
   r_τ ──────┘         └─────────────────────┘
   τ                                                  │ K times
                                                      ▼
                                            r̂₀_1, …, r̂₀_K samples
                                                      │
                                                      ▼
                                            y_sample = ŷ + r̂₀
```

The blue diagram is one **reverse-SDE sample**: start from a fresh `r_τ ~ N(0, Σ_prior)` at τ=1, iteratively denoise via the score net until τ→0, end with a clean residual sample. Repeat K times per test window to build an empirical predictive distribution.

---

## 4. Diffusion machinery — the forward SDE

We define a linear OU forward process on the residual:

$$
\boxed{\; dr_\tau = -\tfrac{1}{2}\, g(\tau)^2\, Q\, r_\tau\, d\tau \;+\; g(\tau)\, dW_\tau \;}
$$

with `Q = Σ_prior⁻¹` (the precision matrix). The drift `-½g²Qr` pulls trajectories toward zero; the noise `g·dW` injects fluctuations. As τ → 1 (full diffusion), the process equilibrates to its **stationary distribution**:

$$
\lim_{\tau \to 1} p(r_\tau) \;=\; \mathcal{N}\!\left(0,\; Q^{-1}\right) \;=\; \mathcal{N}(0,\, \Sigma_{\text{prior}}).
$$

Why this form: the **drift uses Q, not g²·I**. This is the key non-standard choice. With isotropic drift `-½g²·I·r` you'd get a stationary distribution `N(0, σ²·I)` — equal variance in every direction. With the Q drift, you get `N(0, Σ_prior)` — equal variance per *Kronecker-structured eigenmode*, with smaller modes diffused faster and bigger modes diffused slower. The diffusion respects the data's correlation structure.

### 4.1 The β schedule

`g(τ)² = β(τ)` is linear in τ:

$$
\beta(\tau) = \beta_{\min} + \tau(\beta_{\max} - \beta_{\min}), \qquad \bar\beta(\tau) = \int_0^\tau \beta(s)\, ds = \beta_{\min}\tau + \tfrac{1}{2}\tau^2(\beta_{\max} - \beta_{\min}).
$$

We use `β_min=0.01`, `β_max=50`. The integrated quantity `β̄(τ)` controls **how diffused** the process is at time τ:
- At τ=0: `β̄=0` → no diffusion, `r_τ ≡ r_0`.
- At τ=1: `β̄(1) = 25.005` → for any prior eigenmode `s_{ij}`, mode is "stationary" when `β̄/s_{ij} ≫ 1`.

For our priors, max `s ≈ 5-6`. `β̄(1)/max(s) ≈ 4-5`, so `exp(-β̄/s) ≈ 0.01` — meaning 99% of modes have reached stationarity at τ=1.

Why β_max=50 specifically: chosen so that `β̄(1) > max(s)` *after* the prior was standardised (max(s) was 6.4k in raw mph; 5.5 after standardisation). This is documented in exp-004 — the original `β_max=10` was 1000× too small and the SDE didn't reach stationarity at τ=1.

### 4.2 Closed-form forward marginal

The OU SDE admits a closed-form Gaussian marginal at every τ:

$$
\boxed{\; r_\tau \mid r_0 \;\sim\; \mathcal{N}\!\left(\mu_\tau(r_0),\; \Sigma_\tau\right) \;}
$$

with

$$
\mu_\tau = e^{-\tfrac{1}{2}\bar\beta Q}\, r_0, \qquad \Sigma_\tau = Q^{-1}\!\left(I - e^{-\bar\beta Q}\right).
$$

The matrix exponentials are intractable for the full 12·N × 12·N matrix — but **diagonal in the joint eigenbasis**. Eigendecompose once:

$$
Q_H = U_T \operatorname{diag}(c_T) U_T^\top, \quad \Sigma_S = U_S \operatorname{diag}(c_S) U_S^\top
$$

Then in the joint basis the matrices are diagonal with eigenvalues `s[i,j] = c_T[i] · c_S[j]` and `q[i,j] = 1/s[i,j]`. The matrix exponentials become elementwise scalar operations on the per-mode eigenvalues. This is why we use Kronecker structure: it makes the SDE tractable.

**At training time**, this gives us a one-shot sample of `r_τ`:
- Rotate `r_0` to the joint eigenbasis.
- Multiply elementwise by `exp(-½β̄/s)` (the per-mode decay).
- Add Gaussian noise with elementwise std `sqrt(s·(1 - exp(-β̄/s)))` (the per-mode variance).
- Rotate back.

This is `forward_sample` in `src/diffusion/sde_torch.py`. Cost: two einsum rotations + two elementwise ops. Differentiable, runs on MPS/CUDA.

---

## 5. The Kronecker-structured prior

The whole approach pivots on the choice of `Σ_prior`. The deliberate choice: **Kronecker product of a temporal kernel and a spatial GMRF.**

### 5.1 Spatial GMRF — `Σ_S`

A Gaussian Markov Random Field on the road graph:

$$
\Sigma_S = (\beta\, L + \varepsilon_s\, I)^{-1}
$$

where `L` is the symmetric normalised graph Laplacian `L = I - D^{-1/2} A D^{-1/2}` (eigenvalues in [0, 2]).

**Interpretation:** `β` controls how much spatial smoothing the prior expects (larger β → stronger smoothing toward graph neighbours); `ε_s` is a regularisation term that keeps Σ_S well-conditioned and prevents the constant eigenvector (eigval = 0) from blowing up. We fit `β` by maximum likelihood to the empirical spatial covariance of residuals (computed across windows and horizons).

### 5.2 Temporal kernel — `Q_H`

Two-exponential mixture on the horizon lag:

$$
C_H[h_1, h_2] = \alpha \rho_1^{|h_1 - h_2|} + (1 - \alpha) \rho_2^{|h_1 - h_2|}
$$

with `0 ≤ α ≤ 1`, `0 < ρ_1, ρ_2 < 1`. Then:

$$
Q_H[h_1, h_2] = \sqrt{\delta_{h_1}\, \delta_{h_2}} \cdot C_H[h_1, h_2]
$$

where `δ_h = Var(R_h)` is the per-horizon residual variance.

**Why two exponentials, not AR(1)?** We initially planned a single AR(1), but the empirical autocorrelation showed two clearly separated timescales — a fast decay (~5 min) and a slower decay (~30 min). Single AR(1) RMSE was 0.043 against the empirical ACF; two-exp dropped it to 0.003 on METR-LA. Whole story in `experiments/exp-002-q-construction/SUMMARY.md`.

**Why δ_h scaling?** Without it, `Q_H` is a pure correlation matrix and the diagonal is `1`. With `δ_h`, the diagonal is the per-horizon variance — encoding that residual magnitude grows with the forecast horizon (variance at h=12 is ~2.5× that at h=1, because longer-horizon forecasts are harder).

### 5.3 The standardisation upgrade

After exp-004 showed the original `β_max=10` was too small for raw-mph residuals, we standardised the residual per horizon:

$$
\tilde r_h = r_h \,/\, \text{target\_scale}[h], \qquad \text{target\_scale}[h] = \sqrt{\text{Var}(r_h)}
$$

With standardised residuals, `δ_h ≈ 1` for all h, `Q_H ≈ C_H`, and `Σ_prior` eigenvalues drop from `~10⁴` to `~10`. The schedule `β_max=50` then reaches stationarity at τ=1 in 99%+ of modes. Documented in `experiments/exp-004/SUMMARY.md` §3.

At inference time, the score net outputs a standardised residual `r̂₀_std`; we de-standardise via `r̂₀_mph = r̂₀_std × target_scale[h]` and add to the backbone forecast.

### 5.4 Limitations of Kronecker structure

The Kronecker form **forces** the covariance to factorise into "temporal × spatial." Real residual covariance has cross-coupling that doesn't factorise cleanly. We measured this in exp-004 §3:

| Dataset | % of joint eigenmodes within ±10% of empirical | Worst-mode ratio (emp/prior) |
|---|---:|---:|
| METR-LA  | 99.2% | **30.4×** under-modelled |
| PEMS-BAY | 100% | 1.20× |

So **99% of eigenmodes are dead-on**, but on METR-LA a handful of modes carry 30× more energy than the Kronecker prior allocates them. This is the central approximation limit. The score network is designed to **correct** this gap during training (see §7).

PEMS-BAY's two-exponential kernel **degenerated** to AR(1) during fitting: `ρ₁ = ρ₂ = 0.819`. The "structure" of the structured prior on PEMS-BAY is therefore essentially "AR(1) × spatial GMRF" — far weaker structural prior than METR-LA's. This is a documented limitation and we expect the structured-vs-isotropic gap to be smaller on PEMS-BAY.

---

## 6. The reverse SDE and Tweedie

Anderson (1982) gives the reverse-time SDE corresponding to our forward process:

$$
\boxed{\; d\bar r_\tau = \left[-\tfrac{1}{2} g^2 Q \bar r_\tau \;-\; g^2 \,\nabla_{r_\tau} \log p_\tau(\bar r_\tau)\right] d\tau \;+\; g(\tau)\, d\bar W_\tau \;}
$$

(No Jacobian correction because `g(τ)` is a scalar.) The catch: `∇log p_τ` — the **score** — is unknown because we don't know the data distribution. The score network's job is to learn it.

### 6.1 Tweedie / x₀-prediction parametrisation

Instead of having the network predict `∇log p_τ` directly (which blows up as τ → 0 because `Σ_τ → 0`), we use **Tweedie's formula** to parametrise the score via an `x₀-prediction`:

$$
\hat r_0(r_\tau, \tau) \approx \mathbb{E}[r_0 \mid r_\tau, \tau]
$$

The conditional score then has the clean form:

$$
\nabla_{r_\tau} \log p(r_\tau \mid r_0) = -\Sigma_\tau^{-1}\!\left(r_\tau - \mu_\tau(r_0)\right) = -\Sigma_\tau^{-1}\!\left(r_\tau - e^{-\frac{1}{2}\bar\beta Q}\hat r_0\right)
$$

Why x₀-prediction:
- The target `r_0` is well-defined and bounded; ε-prediction's target has variance that grows with τ.
- Numerically stable — no division by `Σ_τ` inside the network output.
- Loss is naturally MSE-like in the original variable space.
- Standard practice in EDM and recent score-based diffusion (Karras et al. 2022).

### 6.2 Reverse-time Euler-Maruyama

We discretise the reverse SDE with **uniform-in-τ Euler-Maruyama**:

```
for k = 0 .. n_steps:
    τ_k    = linspace(τ_start, τ_eps, n_steps + 1)[k]
    Δτ     = τ_k - τ_{k+1}                     # positive
    r̂₀     = network(r_τ, τ_k, conditioning)
    drift  = -½ g² Q r_τ - g² · score(r_τ, r̂₀, τ_k)
    r_τ   ← r_τ - drift · Δτ + g · sqrt(Δτ) · ε    # ε ~ N(0, I)
```

We stop at `τ_eps = 10⁻³` (not τ=0) because `Σ_τ → 0` as τ → 0 and the score `1/var` blows up. At τ_eps, `β̄(τ_eps) ≈ 10⁻⁵`, so the residual gap between `μ_{τ_eps}(r_0)` and `r_0` is well below 1% in every eigenmode — effectively done.

**100 steps** is the exp-004 default for sanity checks. For final inference we use **200 steps**, giving better numerical accuracy in the reverse Euler step.

### 6.3 Stationary initialisation

At inference, we start `r_τ=1 ~ N(0, Σ_prior)` via `sample_stationary` (rotate i.i.d. Gaussian by `sqrt(s)` in eigenbasis, rotate back). This is a closed-form draw — no MCMC needed.

---

## 7. Training the score network — DSM

### 7.1 The DSM loss

Denoising score matching with the x₀-prediction parametrisation:

$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}_{r_0,\,\tau,\,\epsilon}\!\left[ \;\left\| \hat r_0(r_\tau, \tau, c) - r_0 \right\|^2_W \;\right]
$$

where `r_τ ~ N(μ_τ(r_0), Σ_τ)` is the closed-form forward sample, `c` is the conditioning bundle `(x_history, ŷ_back)`, and `‖·‖²_W` is a **weighted** squared error.

**Our specific weighting** is in the **joint eigenbasis** with per-mode variance:

$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}\!\left[ \;\frac{1}{HN} \sum_{i,j} \frac{(\tilde{\hat r}_{0,ij} - \tilde r_{0,ij})^2}{\operatorname{var}_{ij}(\tau)} \;\right]
$$

where `~` denotes "in the eigenbasis" and `var_{ij}(τ) = s_{ij}·(1 - exp(-β̄/s_{ij}))` is the per-mode forward variance at time τ.

**Why this weighting:** it's the Gaussian likelihood weighting. Each mode contributes in proportion to its precision at the current noise level. Modes that haven't diffused much yet (`var` small → `1/var` large) are weighted heavily, because they require accurate prediction; modes already at stationarity (`var ≈ s`, large) contribute less. This is the natural-parameter weighting and the formally correct DSM objective for Gaussian transitions.

`weighting="uniform"` is also supported as an ablation — falls back to plain MSE in the original basis. We expect eigenbasis weighting to converge faster and give better calibration.

### 7.2 τ sampling — log-uniform

We sample `τ ~ LogUniform[τ_eps, 1]` for training:

$$
\log \tau \sim \mathcal{U}[\log \tau_{\text{eps}},\; 0]
$$

Why log-uniform: it gives equal weight to every order of magnitude in τ. Linear sampling would over-represent τ near 1 (where the data is fully diffused and the network has little to learn); log-uniform spreads samples more uniformly across the *difficulty spectrum* of denoising.

### 7.3 Training loop

1. Per batch (size 32-64):
   - Sample `τ ~ LogUniform[τ_eps, 1]` per batch element.
   - Sample `r_τ ~ p(r_τ | r_0)` via the closed-form forward marginal.
   - Predict `r̂_0 = f_θ(r_τ, τ, x_history, ŷ_back)`.
   - Compute eigenbasis-weighted loss.
2. Adam optimiser, lr=1e-4, grad clipping at 1.0.
3. Validate every epoch on the val split; save `best.pt` by val DSM loss; early-stop after 20 epochs without improvement.
4. wandb logging of train/val loss and epoch-time.

Wall-clock on A40: ~16s/epoch on METR-LA, ~38s/epoch on PEMS-BAY. 100 epochs in 30-60 min per run.

---

## 8. The score network — GWN-as-score-net

### 8.1 Architecture

The score net `f_θ` takes four inputs and outputs the x₀-prediction:

```
inputs:  r_τ          (B, 12, N)   noisy residual at time τ, standardised
         τ             (B,)         diffusion time in [τ_eps, 1]
         x_history     (B, 2, 12, N) backbone input (speed + time-of-day)
         ŷ_back        (B, 12, N)   backbone forecast in standardised units

output:  r̂_0          (B, 12, N)   predicted clean residual, standardised
```

The four input tensors are concatenated along the channel dim (1 + 2 + 1 = 4 channels), projected to `hidden_dim=64` via a 1×1 conv, then passed through 4 **score-net blocks**, then a final 1×1 conv to 1 output channel.

Each score-net block:

```
       ┌─ TCN_filter ─tanh ─┐
   x ──┤                     × ─── GCN ── FiLM(τ) ── GroupNorm ──+── x
       └─ TCN_gate ─ sigmoid ┘                                   │
                                                                 │
                                                          residual
```

- **TCN gated unit**: two parallel `Conv2d(kernel=(3,1), padding=(1,0))` over the temporal dim, multiplied as `tanh × sigmoid`. Standard in GWN; preserves H=12.
- **GraphConv**: `y = Linear([x, x @ Â])`, single-support K=1 graph mixing with `Â = I - L` (symmetric normalised adjacency with eigenvalues in [-1, 1]).
- **FiLM**: `x ← γ(τ) ⊙ x + β(τ)`, per-channel affine modulation conditioning on the τ embedding (see §8.2).
- **GroupNorm + residual**: standard tricks for stable deep networks.

Total parameters: ~232k (same for METR-LA and PEMS-BAY because the adjacency is a buffer, not a parameter).

### 8.2 τ conditioning via FiLM

The τ embedding flow:

```
τ ──► SinusoidalEmbedding(d=128) ──► Linear → SiLU → Linear ──► cond (B, 128)
       (DDPM-style, scale=1000)

cond ─► FiLM_block_i: Linear(128 → 2·64) → (γ_i, β_i)   one per block
                                              │
                                              ▼
                                      x ← γ_i ⊙ x + β_i
```

**FiLM** (Feature-wise Linear Modulation, Perez et al. 2018) is a per-channel affine transform whose parameters depend on the conditioning input. Each block has its own `Linear(128 → 2·64)` that produces its own `(γ, β)` from the shared τ embedding.

**Identity init**: γ initialised to 1, β to 0, so every FiLM starts as the identity (the network's output doesn't depend on τ at iteration 0). The network learns τ-dependence from gradient signal. After training, early blocks tend to learn "high τ → coarse smoothing" and later blocks specialise on "low τ → fine detail" — the standard diffusion specialisation pattern.

Why FiLM over alternatives:
- Per-channel scalar parameters — cheap (128 numbers per block).
- Plays nicely with conv layers (broadcasts cleanly over spatial dims).
- Well-tested in score-based diffusion UNets.

---

## 9. Probabilistic evaluation

After training, we sample K=50 reverse trajectories per test window:

```
for each test window w:
    sample r_τ=1 ~ N(0, Σ_prior)          # initial draws, K of them
    for step = 0..200:
        r̂_0 = f_θ(r_τ, τ, x_history[w], ŷ_back[w])
        update r_τ via reverse-SDE step
    r̂_0_std[1..K]                          # K samples in standardised units
    r̂_0_mph = r̂_0_std × target_scale[h]
    sample[1..K] = ŷ_back[w] + r̂_0_mph    # K samples in raw mph
```

This builds an empirical predictive distribution `{y_1, ..., y_K}` per `(window, horizon, sensor)`. We compute three metrics against the ground truth `y_true`:

### 9.1 CRPS — Continuous Ranked Probability Score

$$
\text{CRPS}(F, y) = \int_{-\infty}^{\infty} \left(F(z) - \mathbb{1}[z \geq y]\right)^2 dz
$$

For an empirical ensemble:

$$
\text{CRPS} = \frac{1}{K}\sum_i |x_i - y| \;-\; \frac{1}{2K^2}\sum_{i,j}|x_i - x_j|
$$

The first term is **accuracy** (mean absolute error of samples to truth); the second is **sharpness** (how concentrated the samples are). Lower CRPS = sharper *and* more accurate. CRPS is a proper scoring rule — minimised in expectation by the true predictive distribution.

For a deterministic forecast (K=1), CRPS reduces to MAE. So we can compare CRPS directly to the backbone's MAE: if our diffusion sample distribution gives CRPS < backbone MAE, the probabilistic forecast is *strictly better* than the point forecast.

### 9.2 PIT — Probability Integral Transform

$$
\text{PIT}(y) = \frac{1}{K}\left|\{i : x_i \leq y\}\right| \in [0, 1]
$$

If the forecast distribution is well-calibrated, PIT values aggregated over many test points are **uniform on [0, 1]**. Deviations diagnose miscalibration:
- U-shaped histogram: distribution too narrow (under-dispersion — forecast over-confident).
- Dome-shaped histogram: distribution too wide (over-dispersion — forecast under-confident).
- Skewed: systematic bias.

We compute PIT per `(window, horizon, sensor)`, then aggregate into per-horizon histograms (20 bins).

### 9.3 Pinball loss

$$
\rho_q(y, \hat y_q) = \max\!\left(q(y - \hat y_q),\; (1-q)(\hat y_q - y)\right)
$$

where `ŷ_q` is the empirical q-th quantile of the samples. Computed at `q ∈ {0.1, 0.5, 0.9}`.

Pinball is minimised in expectation by the true q-th quantile, so it's a quantile-specific sharpness check. The set q ∈ {0.1, 0.5, 0.9} gives us the median plus 80% prediction-interval edges.

---

## 10. Implementation conventions

| Where | Convention |
|---|---|
| Residual tensor shape | `(B, H, N)` for scalar fields, `(B, C, H, N)` for multi-channel (channel-first, GWN style). |
| Standardisation | All scripts operate on **per-horizon standardised** residuals: `r̃ = r / target_scale[h]`. De-standardised only at inference end. |
| Eigendecomposition | NumPy/float64 on CPU at startup; converted to torch tensors via `eig_to_torch`. Kept as buffer in the score net. |
| Device | CUDA on DAIC, MPS on Mac, CPU fallback. Same code path everywhere. |
| Seeds | Fixed `--seed 0` for all residual extraction and prior fitting. Seeds 0-4 reserved for future per-seed sweeps. |
| Run IDs | `{model}_{prior}_{dataset}_seed{seed}_{cfg_hash}` (deterministic by hyperparameters). The hash includes `platform.system()` so Mac and DAIC runs of the same config get separate run IDs. |
| Checkpoints | `results/checkpoints/{run_id}/{best,last}.pt` plus optimizer state and config snapshot. |
| Tracking | wandb in online mode (publishes to `tud-thesis/thesis-diffusion`). Authentication via `~/.netrc` (set via `wandb login`). |

---

## 11. Choices made and why — quick reference

| Choice | Alternatives considered | Why this one |
|---|---|---|
| Diffuse residuals (not raw forecasts) | Diffuse `y` directly | Backbone gives a strong point forecast; diffusion has less to learn (just the residual), trains faster, lower variance. |
| Kronecker `Q_H ⊗ Σ_S` prior | Full unstructured Σ; Kronecker-sum; low-rank | Tractable matrix exponentials; captures dominant temporal-vs-spatial structure; small param count to fit. |
| Two-exponential temporal kernel | Single AR(1); GP with RBF | Empirical ACF shows two timescales; AR(1) RMSE 0.043 vs two-exp RMSE 0.003 on METR-LA. |
| Spatial GMRF `(βL + ε_s I)⁻¹` | Diffusion-distance kernel; ChebConv-based | Closed-form inverse; eigenstructure tied to graph Laplacian (interpretable); fits one parameter β well. |
| VP-SDE (linear OU) | VE-SDE; sub-VP | Closed-form Gaussian marginal at every τ; standard in score-based literature; reaches stationary distribution. |
| Linear β schedule | Cosine; sigmoid; quadratic | Closed-form β̄; sufficient for our setup once β_max is right; easy to reason about. |
| β_max = 50 | 10 (too small after standardisation); 100+ (overkill) | Reaches >99% of modes stationary at τ=1 with the standardised prior. |
| Standardised residuals | Raw mph; per-(h, sensor) standardised | Per-horizon makes target_scale a single 12-vector; per-(h,sensor) needs an extra N-vector and over-parameterises. |
| Tweedie x₀-prediction | ε-prediction; v-prediction | Cleaner numerics near τ=0; stable target magnitude; standard in EDM. |
| Eigenbasis-weighted DSM loss | Plain MSE; ε-prediction MSE | Natural-parameter / likelihood-weighted; balances mode contributions across τ. |
| Log-uniform τ sampling | Uniform; log-normal | Equal attention to each order of magnitude in τ. |
| GWN-as-score-net | UNet; transformer; small CNN | Familiar architecture; spatial+temporal structure built-in; reuses backbone style. |
| FiLM conditioning | Concat; cross-attention; AdaLN | Cheap (~128 params/block); standard in diffusion UNets; plays well with conv. |
| K=50 inference samples | K=20 (faster, noisier); K=200 (slower) | Trade-off between MC error and wall-clock; CRPS converges around K=50 for our scale. |
| 200 reverse steps | 100 (too noisy); 500+ (overkill) | exp-004 showed 100 marginal on PEMS-BAY; 200 gives PASS-strict on both. |

---

## 12. Current results (as of exp-003 in progress)

**Backbone (Phase 3)** — deterministic GWN, matches Wu et al. 2019:

| Dataset | h=3 MAE | h=6 MAE | h=12 MAE | Wu2019 reference |
|---|---:|---:|---:|---|
| METR-LA  | 2.7-2.8 | 3.1-3.2 | 3.5-3.6 | 2.69 / 3.07 / 3.53 |
| PEMS-BAY | 1.3 | 1.6 | 1.95 | 1.30 / 1.63 / 1.95 |

**exp-002 — Prior fit** (standardised, two-exp temporal):
- METR-LA: `α=0.80, ρ₁=0.83, ρ₂=0.10, β=0.20, ε_s=1.0`. Frobenius covariance check 0.12 (PASS).
- PEMS-BAY: `α=0.99, ρ₁=ρ₂=0.819` (degenerate; collapses to AR(1)). Frobenius 0.09.

**exp-004 — SDE sanity** (standardised prior, β_max=50):
- Check 1 (stationary covariance match): rel.Frob 0.78 METR-LA (prior inadequacy on a few modes — see §5.4), 0.107 PEMS-BAY (just outside the 0.10 strict threshold).
- Check 2 (oracle round-trip): rel.recovery mean 0.069 METR-LA (PASS-strict), 0.158 PEMS-BAY (PASS-acceptable).
- **The SDE machinery is verified correct.** The Check 1 partial fail is a known property of the Kronecker low-rank prior, not a bug.

**exp-003 — Score net training** (4 runs on DAIC, 1 on Mac):

| Dataset | Structured DSM val loss | Isotropic DSM val loss | Best |
|---|---:|---:|---|
| METR-LA  | **1.155** | 1.248 | structured |
| PEMS-BAY | 1.033 | **0.831** | isotropic |

Suggestive but inconclusive on its own. The DSM loss is computed in different eigenbases for the two priors so absolute values aren't directly comparable. CRPS evaluation (in progress) gives the conclusive answer.

**Expected CRPS pattern** (the thesis claim):
- METR-LA: structured beats isotropic by a meaningful margin (the prior captures real structure → sharper, calibrated forecasts).
- PEMS-BAY: gap is smaller or null (the prior degenerated to AR(1) → less structural advantage).

---

## 13. Limitations + open questions

1. **Kronecker prior under-models a few METR-LA modes by up to 30×.** The score net is expected to correct this during training, but the per-mode gap is real. A richer prior class (Kronecker-sum, low-rank correction, or learned residual covariance) could help. Out of scope for the thesis but a clear follow-up.

2. **PEMS-BAY's two-exp kernel degenerates to AR(1).** Two timescales aren't visible in the empirical ACF — only one. The "structured" prior on PEMS-BAY is therefore essentially "AR(1) × GMRF" — barely structured. Whether to report this as "structure helps on METR-LA, neutral on PEMS-BAY" or to try a richer temporal kernel on PEMS-BAY is a thesis-narrative call.

3. **Single seed (seed=0) for residual extraction.** A more thorough study would extract residuals from multiple backbone seeds and average over them. We have residuals from a single seed-0 backbone; scaling to 5 seeds is straightforward (rerun extract_residuals and the training sweep) but adds 4× wall-time.

4. **Test-set CRPS only; no real-world deployment metrics.** We don't measure decision-impact metrics (route choice, dispatch decisions). CRPS / PIT / pinball are proper scoring rules and standard in the probabilistic-forecasting literature; the thesis claim is well-defined in those terms.

5. **Score net is small (232k params).** Could go larger. We chose this size for fast iteration; a publication-quality model would explore hidden_dim ∈ {64, 128, 256} and num_blocks ∈ {4, 6, 8}. Current numbers are a strong starting point but not necessarily optimal.

---

## 14. What's left

| Step | Status |
|---|---|
| Run CRPS evaluation on the 4 trained DAIC checkpoints | running |
| Compare structured vs isotropic CRPS / PIT / pinball | pending eval output |
| Write `experiments/exp-003-score-network/SUMMARY.md` with results + figures | next |
| Multi-seed extension (5 seeds × 2 priors × 2 datasets) | optional, if results warrant |
| LaTeX writeup | final |

---

## 15. References

The math + design are standard with two non-standard choices (Kronecker prior, eigenbasis-weighted DSM):

- Ho, Jain, Abbeel, 2020. *Denoising Diffusion Probabilistic Models*. NeurIPS.
- Song et al., 2021. *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR.
- Karras et al., 2022. *Elucidating the Design Space of Diffusion-Based Generative Models* (EDM). NeurIPS.
- Anderson, 1982. *Reverse-Time Diffusion Equation Models*. Stochastic Proc. Appl.
- Wu et al., 2019. *Graph WaveNet for Deep Spatial-Temporal Graph Modeling*. IJCAI.
- Perez et al., 2018. *FiLM: Visual Reasoning with a General Conditioning Layer*. AAAI.
- Gneiting & Raftery, 2007. *Strictly Proper Scoring Rules, Prediction and Estimation*. J. Am. Statist. Assoc.
