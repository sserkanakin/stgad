# Where we diverge from standard score-based diffusion practice

This document is the honest list of "we did it differently than the textbook." Each entry says **what's standard**, **what we do**, and **why**. Read alongside [`THESIS_OVERVIEW.md`](THESIS_OVERVIEW.md) for the full mathematical context.

---

## 1. The forward SDE uses a **non-identity precision matrix** in the drift

**Standard.** Variance-Preserving SDE (Ho et al. 2020, Song et al. 2021):
$$
dr_\tau = -\tfrac{1}{2} \beta(\tau) \, r_\tau \, d\tau + \sqrt{\beta(\tau)} \, dW_\tau
$$
Stationary distribution: `N(0, I)` — isotropic. Every direction in residual-space diffuses at the same rate.

**Us.** Same form but with `Q = Σ_prior⁻¹` in the drift:
$$
dr_\tau = -\tfrac{1}{2} \beta(\tau) \, Q \, r_\tau \, d\tau + \sqrt{\beta(\tau)} \, dW_\tau
$$
Stationary distribution: `N(0, Σ_prior) = N(0, Q_H ⊗ Σ_S)` — Kronecker-structured.

**Why.** The data has known correlation structure (temporal × spatial). Building that structure into the forward process means:
- The score net only has to learn the per-mode *deviations* from the prior, not the full covariance.
- The reverse SDE respects the data's correlation structure for free.
- The closed-form forward marginal `Σ_τ = Q⁻¹(I - exp(-β̄Q))` makes diagonal-eigenbasis arithmetic possible (cheap) — without Kronecker structure this would be intractable for 12·N × 12·N matrices.

**Cost.** The Kronecker structure is **approximate**. Real residual covariance has cross-coupling that doesn't factorise. We measured the worst joint eigenmode is off by 30× on METR-LA (see THESIS_OVERVIEW §5.4). The score net is expected to absorb this gap during training.

---

## 2. We diffuse **residuals**, not the forecast itself

**Standard.** Score-based diffusion for time series typically diffuses the target variable directly: `y_t → y_t + noise → ε prediction`.

**Us.** Train a deterministic GWN backbone first → extract residuals `r = y_true - ŷ_back` → diffuse the residuals.

**Why.**
- The backbone provides a strong deterministic prior (MAE ~3 mph on METR-LA). The diffusion only has to model the *residual distribution*, which has much smaller variance and clearer structure than `y` itself.
- Residuals are approximately zero-mean (the backbone is unbiased), making the symmetric forward OU process well-fit to their distribution.
- Training the diffusion model on residuals converges 5-10× faster than training it on `y` directly (informal observation from preliminary experiments).
- At inference, we sample `r̂_0` and add to the backbone's `ŷ` — recovering a full predictive distribution for `y` with the deterministic forecast as anchor.

**Cost.** We depend on the backbone's quality. If the backbone is systematically biased, the residual mean isn't zero, and the diffusion has to learn a non-trivial mean function as well as the spread. Empirically not a problem on METR-LA / PEMS-BAY; the backbone is well-calibrated.

---

## 3. **Tweedie / x₀-prediction** instead of ε-prediction

**Standard.** DDPM (Ho et al. 2020) trains the network to predict the **noise** ε added during forward diffusion:
$$
\mathcal{L}_\epsilon = \mathbb{E}\!\left[\|\epsilon - \epsilon_\theta(r_\tau, \tau)\|^2\right]
$$

**Us.** x₀-prediction (Tweedie's formula): network predicts the **clean residual** `r̂₀`:
$$
\mathcal{L}_{x_0} = \mathbb{E}\!\left[\|r_0 - r_0^θ(r_\tau, \tau)\|_W^2\right]
$$
with appropriate weighting `W` (see #4).

**Why.**
- Target magnitude `r_0` is bounded by the data scale (~1 in standardised units). ε-prediction has target std `1` only when `Σ_τ = I`; for our `Σ_τ` that varies wildly per mode, ε's effective target depends on τ in awkward ways.
- Numerically stable: predicting `r̂₀` avoids the `1/Σ_τ → ∞` blow-up that ε-parametrisation has near τ=0.
- Standard in EDM (Karras et al. 2022) and recent diffusion literature for low-data regimes.
- Loss is naturally MSE-like in the original variable space, which is more intuitive for hyperparameter tuning.

**Cost.** ε-prediction is slightly easier to plug into off-the-shelf diffusion samplers; x₀-prediction requires deriving the reverse-SDE drift via Tweedie:
$$
\text{score} \;=\; -\Sigma_\tau^{-1}(r_\tau - \mu_\tau(\hat r_0)).
$$

This adds a tiny amount of code complexity for substantial numerical stability gain.

---

## 4. **Eigenbasis-weighted** DSM loss (not unweighted MSE)

**Standard.** Plain MSE on x₀ or ε:
$$
\mathcal{L}_{\text{plain}} = \mathbb{E}\!\left[\|\hat r_0 - r_0\|^2\right]
$$

**Us.** Per-mode weighted MSE in the joint Q_H ⊗ Σ_S eigenbasis:
$$
\mathcal{L}_{\text{eig}} = \mathbb{E}\!\left[\frac{1}{HN}\sum_{i,j}\frac{(\tilde{\hat r}_{0,ij} - \tilde r_{0,ij})^2}{\operatorname{var}_{ij}(\tau)}\right]
$$
where `~` is "projected to eigenbasis" and `var_{ij}(τ) = s_{ij}(1 - exp(-β̄/s_{ij}))`.

**Why.** This is the **Gaussian likelihood weighting**: each mode contributes in proportion to its precision at the current noise level. Modes with small variance (not yet diffused) get heavy weight — accurate prediction there is informative. Modes near stationarity get small weight — they're indistinguishable from the prior. This is the formally correct DSM target for our forward process and gives the right balance between training across τ values.

**Cost.** Loss magnitude is platform-dependent (depends on the prior's eigenvalue spectrum) and not directly interpretable. Two priors with different `s` distributions produce non-comparable absolute losses. CRPS evaluation in original units is what matters for the thesis claim. We provide `weighting="uniform"` as a fallback for ablations.

---

## 5. **Per-horizon standardisation** of residuals

**Standard.** Score-based diffusion for time series typically works on the raw target variable. The diffusion's noise variance is scaled to match the data's natural scale.

**Us.** Residuals are standardised per horizon **before** any diffusion step:
$$
\tilde r_h = r_h / \text{target\_scale}[h], \quad \text{target\_scale}[h] = \sqrt{\text{Var}(r_h)}
$$
All training, sampling, and SDE math operates on standardised residuals. De-standardisation happens only at the final inference output.

**Why.** Residual variance grows with the forecast horizon (h=12 variance is ~2.5× h=1's). Without standardisation:
- The Kronecker prior's eigenvalue range spans 1000s (raw mph²), requiring β_max → 10⁴+.
- Per-horizon variance has to be absorbed into `δ_h` in the temporal kernel `Q_H`.
- The score net's output magnitude varies wildly with h.

With per-horizon standardisation:
- Eigenvalue range drops to O(1) — O(10).
- `β_max = 50` reaches stationarity at τ=1 in 99%+ of modes.
- The score net sees roughly unit-variance signals at every horizon, easier optimisation.
- `δ_h ≈ 1` for all h; `Q_H ≈ C_H` becomes a pure correlation matrix.

**Cost.** A 12-vector `target_scale` is now part of the trained model's configuration. Inference output must be de-standardised correctly (multiply by `target_scale[h]` per horizon, then add backbone forecast). One extra line of code; easy to get right.

**The story for the writeup.** "We initially planned to diffuse raw residuals; experimental evidence (sde-sanity §3) showed the un-standardised Kronecker prior produces eigenvalues spanning 4 orders of magnitude, requiring β_max ≫ 10. Standardising per horizon brings the prior into the regime where standard score-based diffusion practice (β_max ∈ [10, 100]) is sensible."

---

## 6. **Conditioning** via FiLM, not concat or cross-attention

**Standard.** DDPM uses additive time embeddings (added to feature maps). Diffusion transformers (DiT) use AdaLN. Some UNets use concat-along-channel.

**Us.** FiLM modulation per block: each block has its own tiny `Linear(τ_embed → 2·C)` that produces `(γ, β)` and applies `x ← γ ⊙ x + β`. Identity initialisation (`γ=1`, `β=0` at start).

**Why.**
- Cheap: 128 parameters per block for τ injection, vs concat which adds C channels.
- Per-channel scalars play cleanly with conv layers (broadcast over H, N).
- Standard in modern score-based diffusion UNets (EDM, etc.).
- Identity init lets the network start as if there's no τ conditioning and learn τ-dependence from data — much more stable than starting with random τ injection.

**Cost.** Slightly more code than concat. No real downside.

---

## 7. **Sampling τ log-uniformly**, not uniformly

**Standard.** Many implementations sample τ uniformly in [0, 1] or discrete-uniformly over T=1000 steps.

**Us.** `τ ~ LogUniform[τ_eps, 1]`:
$$
\log \tau \sim \mathcal{U}[\log \tau_{\text{eps}},\, 0]
$$

**Why.** Equal weight to each order of magnitude in τ. Uniform sampling would over-represent τ near 1 (where the data is fully diffused and the network has nothing meaningful to learn). Log-uniform spreads the training signal more uniformly across the *difficulty spectrum* of denoising.

EDM uses log-normal sampling centred away from 0; log-uniform is simpler and sufficient for our setup. We may revisit if training is unstable at small τ.

**Cost.** Sampling distribution affects the effective loss weighting. Should be chosen jointly with the loss weighting; ours are consistent (log-uniform τ + eigenbasis 1/var loss).

---

## 8. **CPU + NumPy** for SDE sanity checks; **CUDA/MPS + PyTorch** for training

**Standard.** Everything in PyTorch on GPU.

**Us.**
- `src/diffusion/sde.py` — pure NumPy / float64 / CPU. Used by `experiments/sde-sanity/run_sanity_checks.py` for closed-form validation of the SDE machinery.
- `src/diffusion/sde_torch.py` — PyTorch / float32 / device-aware. Used by training + inference.

**Why.**
- LAPACK on CPU is faster than `torch.linalg.eigh` for our 12×12 and 207/325×207/325 eigendecompositions — sizes too small for GPU to win.
- Float64 NumPy gives bit-deterministic sanity-check results we can compare against the torch implementation (12 of the 30 SDE tests do exactly this).
- The two implementations share API shape so converting between them is a one-line call (`eig_to_torch`).
- Sanity checks (sde-sanity) don't need autograd; training (score-network) does. Right tool for each.

**Cost.** Two implementations of the same math, kept in sync. Mitigated by unit tests asserting numerical equivalence to ≤1e-10 (float64) or ≤1e-5 (float32).

---

## 9. **K=50** inference samples (not K=100 or K=1000)

**Standard.** Probabilistic forecasting papers often report K=200 to K=1000 samples.

**Us.** K=50 per test window for the main evaluation.

**Why.**
- CRPS Monte Carlo error scales as `1/sqrt(K)`. At K=50, MC error is ~14% of CRPS std — small compared to the structured-vs-isotropic gap we expect.
- Wall-clock matters: 50× K vs 200× K is 4× more compute. K=50 fits comfortably in our DAIC budget; K=200 would push training+eval to overnight × multiple nights.
- We can re-evaluate at K=100 or K=200 post-hoc if the K=50 results are borderline.

**Cost.** Slightly noisier CRPS estimates. Acceptable for the relative comparison; would want K=100+ for absolute publication numbers.

---

## 10. **Single static adjacency**, K=1 hop graph conv

**Standard.** Graph WaveNet uses three supports (forward transition, backward transition, learned adaptive). K=2 hops per layer with Chebyshev-style expansion.

**Us.** Single support: `A_norm = I - L` (symmetric normalised adjacency, eigvals in [-1, 1]). K=1 hop per layer (just `[x, x·Â]` concat).

**Why.**
- Score net is meant to model *residual structure*, not the full forecasting problem. The backbone already captured the dominant graph-conv patterns. The score net needs less expressivity.
- Stacking 4 blocks gives an effective 4-hop receptive field — plenty for the per-step denoising task.
- Smaller parameter count (~232k vs GWN's ~290k) → faster training, less risk of overfitting on the smaller residual signal.
- Simpler code, fewer hyperparameters to tune.

**Cost.** Possibly less expressive than full GWN's diffusion-conv. If results show under-fitting, we can re-introduce the adaptive adjacency and multi-hop expansion.

---

## 11. **Hash-includes-platform** in run_id

**Standard.** Run IDs hash the hyperparameters of the experiment.

**Us.** Plus `"platform": platform.system().lower()` — so a Mac run and a DAIC run of the same hyperparameters get different `cfg_hash` values.

**Why.** wandb's deterministic-run-ID feature (`resume="allow"`) would otherwise treat the Mac and DAIC runs as resumptions of each other, interleaving their logs into one wandb run. Including platform in the hash makes them separate runs, which is what we want for cross-platform reproducibility checks.

**Cost.** The same configuration produces two run IDs across machines, so resumption-after-migration doesn't work. We don't migrate runs across machines, so this isn't a real cost.

---

## 12. **Code/data on different filesystems** on DAIC

**Standard.** Single project directory.

**Us.** Code in `~/research-tud` (NFS Linux home, 8 GB quota); data + conda env caches in `/winhome/$USER/research-tud-data` (1.1 TB Windows-side share). The two are linked via symlinks in the project tree.

**Why.** DAIC's home quota is too small for the diffusion env + checkpoints + residuals (8 GB vs ~15 GB total). /winhome is the only generally-writable large filesystem we have access to as students. The conda env itself stays on /home (NFS, full POSIX) because pip's in-place rename ops fail on /winhome's SMB-backed mount.

**Cost.** Two filesystems means two backup policies. We document the split in [`scripts/daic/README.md`](scripts/daic/README.md). Once staff-umbrella access is granted, we can collapse to one location with a single line edit.

---

## Common-but-non-default choices (worth mentioning for completeness)

- **VP-SDE with linear β** (not cosine, not sub-VP) — simplest closed-form; sufficient.
- **GroupNorm not BatchNorm** — standard in diffusion UNets for batch-independent normalisation.
- **Adam (not AdamW)** with `lr=1e-4` — DDPM default.
- **Gradient clipping at 1.0** — stability around small τ where loss can spike.
- **Early stopping with patience=20** — saves compute on multi-seed sweeps.

None of these are unusual but worth listing for the writeup.

---

## What we do NOT do (and won't)

For the reader-of-the-thesis who's seen everything in score-based diffusion:

- **No EDM-style preconditioning** (`c_in, c_out, c_skip, c_noise`). EDM's design space is for general image diffusion; our prior structure plays a similar role. We adopt EDM's `x₀`-target choice but not its preconditioning machinery.
- **No CFG or guidance**. Conditioning is via FiLM on τ and concat for x_history/y_pred. No classifier-free guidance because there's no class label.
- **No latent-space diffusion** (LDM). Residuals are small enough (12 × 207-325 floats) that working in pixel-space is fine.
- **No DDIM-style deterministic sampler**. We use stochastic Euler-Maruyama. DDIM gives equivalent results in expectation; sticking with the SDE form keeps the math transparent.
- **No multi-step Heun or RK4**. Standard Euler-Maruyama is sufficient with 200 steps. Higher-order would buy maybe 2× wall-clock reduction for a touch of accuracy; not worth the implementation complexity.
- **No DDP / FSDP**. Single-GPU training; the model is small enough.
