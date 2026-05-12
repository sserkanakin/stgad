# exp-003 — Score-network training on standardised residuals (PLAN)

**Status:** plan only; no code written yet. Awaiting review before scaffolding.
**Unblocked by:** exp-004 v2 (5762320) — schedule fix complete, SDE machinery verified.

---

## §1 — Goal

Train a denoising-score-matching (DSM) network on standardised GWN residuals using the Kronecker-structured forward SDE from `src/diffusion/sde.py`. Compare against an isotropic-prior baseline (same architecture, same training budget) on horizon-stratified probabilistic metrics computed on the held-out test split.

**Thesis claim being tested:** the Kronecker-structured prior produces sharper, better-calibrated probabilistic forecasts than an isotropic-prior baseline at matched compute. If true, it justifies the entire exp-002 prior-fit pipeline.

---

## §2 — Locked design decisions (from review)

| Decision | Choice |
|---|---|
| Architecture | Reuse GWN-style graph-WaveNet block as the score net (not the trained backbone — new weights) |
| Loss | Tweedie / x₀-prediction: net predicts r̂₀ given r_τ, τ, conditioning |
| Baseline | Isotropic-prior (`Q = I` → `Σ_τ ∝ I`), same architecture, same training budget |
| Comparison | Structured vs isotropic on horizon-stratified CRPS + PIT + pinball loss, test split, 5 seeds |

---

## §3 — Forward / reverse SDE (re-using exp-004)

Forward marginal (closed-form):
$$ r_\tau \mid r_0 \sim \mathcal{N}(\mu_\tau(r_0), \Sigma_\tau), \quad
   \mu_\tau = e^{-\tfrac{1}{2}\bar\beta(\tau)Q} r_0, \quad
   \Sigma_\tau = Q^{-1}\big(I - e^{-\bar\beta(\tau) Q}\big). $$

Both $\mu_\tau$ and $\Sigma_\tau$ are diagonal in the joint Q_H ⊗ Σ_S eigenbasis — sampling r_τ at any τ is O(HN) elementwise multiplication plus two basis rotations.

Schedule (locked): β_min = 0.01, β_max = 50, τ ∈ [τ_eps, 1] with τ_eps = 10⁻³. β̄(1) = 25.005.

The score (for use in the reverse SDE at inference time) is recovered from the network output via Tweedie:
$$ \nabla_{r_\tau} \log p_\tau(r_\tau) \approx -\Sigma_\tau^{-1}(r_\tau - \mu_\tau(\hat r_0(r_\tau, \tau, c))) $$
where $c$ is the conditioning bundle (history + backbone forecast).

**Isotropic baseline:** same code path with Q_H = I_H, Σ_S = I_N. Joint eigenvalues s[i,j] = 1 → standard variance-preserving SDE with isotropic Gaussian noise. Same network, same loss, same training budget.

---

## §4 — Variable conventions

The score net operates on **standardised** residuals throughout. Concretely:
- Training data: `r̃_0 = r_0 / target_scale[h]` (per-horizon, from `gwn_{dataset}_{train}_seed0.npz` and `target_scale` in the prior YAML).
- All forward/reverse sampling is in standardised space.
- Inference output `r̂_0_std` is de-standardised by multiplying by `target_scale[h]` per horizon.
- Final forecast: `y_sample = y_pred_backbone + r̂_0` in raw mph.

This matches exp-004 v2 — keeps the network operating on a unit-variance variable, which is what makes β_max = 50 the right reach.

---

## §5 — Architecture

GWN-style block, weight-fresh (not the trained backbone). Inputs:

| Stream | Shape | Source |
|---|---|---|
| `x_history` | (B, 2, 12, N) | normalised speed + time-of-day (the GWN backbone's input) |
| `y_pred_backbone` | (B, 12, N, 1) | deterministic forecast from the trained GWN, in standardised units (divided by target_scale) |
| `r_τ` | (B, 12, N) | noisy residual at level τ, standardised units |
| `τ_embedding` | (B, d_τ) | sinusoidal embedding of τ, broadcast through layers |

Output:
- `r̂_0` (B, 12, N) — standardised x₀ prediction.

Implementation plan:
- Stack `x_history`, `y_pred_backbone`, and `r_τ` along the channel dimension → (B, C_in, 12, N).
- Pass through GWN's diffusion-graph-conv + temporal-conv blocks (re-use `src/backbones/gwn.py` building blocks, not the wrapper — strip the multi-step output head).
- Inject τ via FiLM (per-block scale/shift modulation): `h ← γ(τ) · h + β(τ)`.
- Output head: 1×1 conv → (B, 12, N).

**Decision:** weights NOT shared with the trained backbone. The backbone forecast is conditioning input, not a starting point for the score net.

---

## §6 — Loss

Tweedie / x₀-prediction. For each batch element:

1. Sample τ ~ log-uniform[τ_eps, 1]. (Log-uniform is the simplest defensible choice; EDM uses log-normal but log-uniform is fine for v1.)
2. Compute closed-form `μ_τ, Σ_τ` in the joint eigenbasis (precomputed eigendecomp of the prior — done once at startup).
3. Sample `r_τ ~ N(μ_τ(r̃_0), Σ_τ)` via spectral-domain sampling (rotate r̃_0 to eigenbasis, multiply by decay, add eigen-scaled Gaussian noise, rotate back).
4. Network predicts `r̂_0 = f_θ(r_τ, τ, x_history, y_pred_backbone)`.
5. Loss: `||(r̂_0 - r̃_0)||²` weighted by `1 / Σ_τ` *in the joint eigenbasis* (EDM-style: weight modes by inverse variance so all modes contribute comparable gradient magnitude across τ).

The eigenbasis-weighted loss is equivalent to the natural-parameter loss for the Gaussian transition and is the formally correct DSM target for this SDE.

---

## §7 — Sampling at inference

For each test window:
1. Sample `r_τ=1 ~ N(0, Σ_prior)` via `src/diffusion/sde.py:sample_stationary`.
2. Run reverse Euler-Maruyama for n_reverse_steps (locked: 200; exp-004 showed 100 is marginal on PEMS-BAY):
   - At step k: compute `r̂_0 = f_θ(r_τ_k, τ_k, conditioning)`, recover score via Tweedie, apply reverse-SDE update from `src/diffusion/sde.py:reverse_oracle` structure (replace oracle score with network score).
3. After τ → τ_eps: take `r̂_0(τ_eps)` as the final standardised residual sample.
4. De-standardise: `r̂_0_mph = r̂_0_std * target_scale[h]`.
5. Final sample: `y_sample = y_pred_backbone + r̂_0_mph`.
6. Repeat steps 1–5 K times per window (K = 100 for the final eval) to build an empirical predictive distribution.

---

## §8 — Hyperparameters (initial)

| Hyperparameter | Value | Source |
|---|---|---|
| β_min, β_max | 0.01, 50 | exp-004 v2 locked |
| τ range | [1e-3, 1] | exp-004 v2 locked |
| τ sampling | log-uniform | this experiment |
| n_reverse_steps | 200 (eval) | exp-004 Check 2 PEMS-BAY needed > 100 |
| Optimizer | Adam, lr 1e-4 | matches GWN backbone training |
| Batch size | 32 windows | MPS memory budget; tune in first run |
| Epochs | 100 max, early-stop on val DSM loss with patience 20 | matches backbone protocol |
| τ embedding dim | 128 | standard |
| Inference samples K | 100 | for CRPS estimation |
| Seeds | {0, 1, 2, 3, 4} for the final reported runs | matches project protocol |

First run: seed 0 only, 1 epoch smoke-test, then scale up.

---

## §9 — Compute

Two execution targets, same code path:

**Local Mac (development / iteration):** PyTorch on MPS (float32). Forward/reverse rotations of `r̃_0` to/from the joint eigenbasis are matrix-multiplies on (B, 12, N) tensors — cheap on MPS. Use for fast iteration on shape bugs, loss scaling, etc.

**DAIC cluster (full runs):** PyTorch on CUDA (A40, 46 GB; A100/L40 also fine if available). Slurm-managed via `scripts/daic/submit_diffusion.sbatch`. The full 20-job sweep (5 seeds × 2 priors × 2 datasets) is one `for` loop in `scripts/daic/README.md` and runs in parallel modulo cluster availability.

Common to both:
- Precompute U_T, U_S, c_T, c_S, s, q on CPU/NumPy/float64 at startup, transfer to the accelerator as float32 constants.
- **CRPS / PIT:** NumPy on CPU on the K test samples (post-sampling, single pass).

The prior eigendecomposition is a stable bridge between `src/diffusion/sde.py` (NumPy) and the new training/sampling code (PyTorch). The adapter `src/diffusion/sde_torch.py` (step §14.1) wraps `eigendecompose_prior` and exposes torch versions of `to_spectral`, `from_spectral`, `forward_marginal`, `sample_stationary`.

---

## §10 — Evaluation

Computed on the **test split** residuals (`gwn_{dataset}_test_seed0.npz`), once, after training is locked.

| Metric | What it measures |
|---|---|
| Per-horizon CRPS (h = 1..12) | sharpness + calibration combined |
| PIT histogram (per horizon, binned over all sensors × windows) | calibration; flat histogram = perfectly calibrated |
| Reliability diagram | calibration curve at quantile levels |
| Per-horizon pinball loss at q ∈ {0.1, 0.5, 0.9} | sharpness at specific quantiles |
| Sample-spread vs error scatter | sanity: high spread should track high MAE |

Reported: structured vs isotropic, on both datasets, averaged over 5 seeds with ±1 std. Final numbers go in `experiments/exp-003-score-network/SUMMARY.md` with figures in `figures/`.

---

## §11 — Deliverables

| File | Purpose |
|---|---|
| `src/diffusion/score_net.py` | GWN-as-score-net architecture (FiLM-conditioned on τ) |
| `src/diffusion/dsm.py` | DSM loss (eigenbasis-weighted MSE, closed-form forward) |
| `src/diffusion/sampling.py` | reverse-SDE sampler using a learned network |
| `src/diffusion/sde_torch.py` | small PyTorch adapter (eig → MPS tensors, basis rotations) |
| `scripts/train_diffusion.py` | training entry point, dataset + isotropic flag |
| `scripts/sample_diffusion.py` | generate K samples per test window |
| `scripts/compute_probabilistic_metrics.py` | CRPS / PIT / pinball, JSON output |
| `experiments/exp-003-score-network/SUMMARY.md` | results, tables, embedded figures |
| `experiments/exp-003-score-network/figures/` | CRPS bars, PIT histograms, reliability diagrams |
| `tests/test_score_net.py` | shape checks, FiLM works |
| `tests/test_dsm.py` | oracle x₀ → near-zero loss; isotropic == structured when Q = I; gradient finite at τ → τ_eps |
| `tests/test_sampling.py` | sampler returns correct shape; stationary init = `sample_stationary` |

---

## §12 — Acceptance criteria

Before declaring exp-003 complete:
1. All unit tests pass.
2. Both runs (structured + isotropic) train to convergence (val DSM loss plateaus) on both datasets, 5 seeds each.
3. CRPS, PIT, pinball reported per horizon for all four design points (2 datasets × 2 priors), with 5-seed mean ± std.
4. `SUMMARY.md` includes the comparison table, the figures, and a per-horizon analysis matching the depth of exp-002 / exp-004 SUMMARYs.

---

## §13 — Risks & open questions

- **Per-mode prior mismatch on METR-LA** (Check 1 §3 in exp-004): the Kronecker prior under-models a few modes by up to 30×. The score net is *supposed* to correct this during training. If it doesn't (val DSM loss plateaus high; PIT badly miscalibrated on METR-LA), we revisit the prior class in a follow-up experiment — not in exp-003 itself.
- **Computation budget on MPS:** 200 reverse steps × K = 100 samples × N_test test windows × 5 seeds × 2 datasets × 2 priors is a lot. If wall-clock becomes a problem, drop K to 50 first (still defensible for CRPS), then reduce seeds to 3.
- **PEMS-BAY temporal kernel is degenerate** (ρ₁ ≈ ρ₂): the two-exp mixture collapses to AR(1) on PEMS-BAY. This is fine for the score net but means PEMS-BAY's structured prior is genuinely less "structured" than METR-LA's — the gap to isotropic may be smaller.
- **τ sampling distribution:** log-uniform is the v1 default. If training is unstable at small τ, switch to log-normal centred away from 0 (EDM convention).

---

## §14 — Sequence of next moves (post-review)

1. Implement `src/diffusion/sde_torch.py` adapter + smoke-test that `to_spectral` / `from_spectral` agree with the NumPy version to 1e-5.
2. Implement `src/diffusion/score_net.py` (GWN-as-score-net with FiLM) + shape tests.
3. Implement `src/diffusion/dsm.py` + tests (oracle x₀ → zero loss; Q = I path correctness).
4. Implement `scripts/train_diffusion.py` + smoke test (1 epoch, seed 0, METR-LA, structured).
5. Run full structured training: METR-LA seed 0, all epochs, validate convergence.
6. Run isotropic baseline: METR-LA seed 0.
7. Implement sampler + probabilistic metrics, compare structured vs isotropic on METR-LA seed 0.
8. If the comparison goes the right way: scale to PEMS-BAY + 5 seeds × 2 priors × 2 datasets.
9. Write `SUMMARY.md` with figures, commit.
