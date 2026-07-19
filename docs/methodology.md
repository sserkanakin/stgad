# Methodology & math, start to end (predict → refine diffusion)

_2026-07-19, branch `clean-rebuild`. The full pipeline for the spatio-temporal graph
residual-diffusion forecaster. Notation: `N` sensors, `H=12` horizons, `B` windows, `⊗`
Kronecker product._

## Stage 1 — Predict (backbone → residual)

1. **Normalize the raw signal.** One global scalar z-score on the value channel, train-only:
   `x̄ = (x − μ)/σ`. Time-of-day kept raw in [0,1). (GWN/DCRNN/DiffSTG convention.)
2. **Backbone forecast.** Deterministic net `f_φ` (MLP / GCN / GWN): `ŷ_z = f_φ(x̄_past)`,
   shape `(B,H,N)`; trained with masked MAE in raw units (`ŷ = ŷ_z·σ + μ`).
3. **Residual.** `r = y − ŷ`, shape `(B,N,H)`, raw units — the stochastic part diffusion models.

## Stage 2 — Normalize residual for diffusion
Per-node std from train residuals (masked): `σ_node ∈ ℝ^N`; `R₀ = r / σ_node`. Each sensor's
residual becomes ~unit variance. `σ_node` is multiplied back at the very end.

## Stage 3 — Build the prior (noise geometry)

**Graph operator.** `L = I − D^{−1/2} A D^{−1/2} = V_S diag(λ_S) V_Sᵀ`, `λ_S ∈ [0,2]`.
**Temporal factor.** `Σ_T = V_T diag(μ_T) V_Tᵀ` (`H×H`); empirical/reshaped fit it from the
residual (`Σ_T = D_H^{1/2} R_H D_H^{1/2}`).
**Joint prior** `Σ = Σ_T ⊗ Σ_S`: eigenvectors `V_S ⊗ V_T`, per-mode precision `θ[k,h]`,
stationary variance `σ²/θ[k,h]`. The three arms differ ONLY in `θ`:

| prior | `θ[k,h]` | meaning |
|---|---|---|
| iso | `1/σ²` const | white noise |
| empirical | `(λ_S[k]+γ)/(τ²·μ_T[h])` | graph heat width × data-fit temporal width |
| reshaped | `1/v_emp[k,h]` | width = residual's MEASURED variance `v_emp=mean_b([V_Sᵀ R₀ V_T]_{k,h})²` |

**Scale normalization.** Rescale `θ ← θ·(0.8/θ_min)`, `σ² ← σ²·(0.8/θ_min)` (leaves `σ²/θ`
unchanged) so `θ_min = 0.8` → one fixed schedule `c₀=7` mixes every dataset. `σ²` then matched
to total residual variance. `reshaped` uses `--stiffness-cap` (~50) since raw `θ_max/θ_min`
reaches ~5891 (near-zero-variance noise modes).

## Stage 4 — Forward process (heat equation)

`dR = −c(t)·Q·R dt + √(2c(t))·dW`, `Q=Σ⁻¹`. Per independent mode `u_k=(V_Sᵀ R V_T)_k`, with
accumulated noise `c̄(t)=∫₀ᵗc`:

```
E[u_k] = e^(−θ_k c̄)·u_k(0)                 (signal retained)
Var[u_k] = (σ²/θ_k)(1 − e^(−2θ_k c̄))        (noise injected)
c̄→∞ ⇒ u_k ~ N(0, σ²/θ_k) = prior          (rough modes die first, smooth last)
```

**FCPS schedule:** `c̄(t)=c_min·t+(c₀−c_min·T)·u^(α+1)`, `u=t/T`, `c̄(T)=c₀=7`, `α=4`,
`c_min=0.1`. Mixing guarantee `2·θ_min·c₀ = 11.2 ≥ 5`.

## Stage 5 — Training (denoising)
Draw a noise level, forward-noise `R₀`, recover it:
- **Noise-level draw:** powerlaw `c̄=c₀·u^0.65` (GAD, high-noise biased) OR log-uniform
  `c̄=c_lo·(c₀/c_lo)^u` (even across decades, safer for stiff priors). `u~U(0,1)`.
- **Denoiser** `R̂₀ = g_ψ(R_{c̄}, t(c̄), COND)`, small graph-CNN (~17k params):
  `xˡ⁺¹=SiLU(FiLM_τ(Σ_{k≤5} Sᵏ xˡ W_k + depthwise_conv_H(x)))`, `S=D^{−1/2}(A+I)D^{−1/2}`,
  input channels `[R_{c̄} | COND]`, output `R_{c̄}+head`.
- **Conditioning `COND`:** `ŷ` (backbone forecast, z-scored) — a learned summary of the
  history; with `--cond-past`, `COND=[R_prev | ŷ]` adds the previous window's residuals
  (residual history, no leakage; HySRD-style). `--uncond` zeros it (pure generation).
- **Loss** (SNR-weighted masked x₀-MSE, GAD parity):
  `L = Σ w(c̄)·mask·(R̂₀−R₀)² / Σ w·mask`, `w(c̄)=1/(1+2·v̄(c̄))`.

## Stage 6 — Reverse (sampling)
Start `R_T ~ N(0,Σ)`, descend a geometric `c̄` grid `c₀→c_lo`. **Ancestral** exact per-mode
posterior (no stability floor, works ~25–100 steps even for stiff priors):

```
R̂₀ = g_ψ(R_c, t(c), COND)
a_d=e^(−θ(c−c')), s_d=(σ²/θ)(1−e^(−2θ(c−c')))     # transition c'→c
a' =e^(−θc'),     s' =(σ²/θ)(1−e^(−2θc'))          # marginal at c'
v = s_d·s'/(a_d²·s'+s_d) ;  mean = v(a_d·u_c/s_d + a'·R̂₀/s') ;  u_{c'}=mean+√v·z
```

Output `R̂₀` at `c_lo`. **De-normalize:** `r̂ = R̂₀·σ_node`, forecast `ŷ_final = ŷ + r̂`.

## Stage 7 — Evaluation
`k` samples → `k` forecasts/window. Point: MAE/RMSE of the sample mean (masked, raw).
Probabilistic: 19-quantile CRPS (leaderboard), energy CRPS, coverage/Winkler.

## Where the contribution lives
Everything is GAD-style diffusion EXCEPT (1) Stage-3 `θ` for the **reshaped** arm (mode
variance = residual's measured value, not the heat-kernel `1/(λ+γ)`), and (2) Stage-5's
**noise-level draw** (log-uniform vs powerlaw). Everything else is held fixed so any CRPS
difference is attributable to exactly those choices. See [design note](prior_design_note.md)
for why the prior is a bounded (15–42% diagonal) lever and the graph belongs in the denoiser.
