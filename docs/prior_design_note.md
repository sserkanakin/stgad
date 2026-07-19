# Design note: what the diffusion prior can and cannot do for predict→refine

_2026-07-19, branch `clean-rebuild`. Grounded in `scripts/analysis/{verify_process, prior_residual_match, params, explain}.py`, GAD (Rozada et al. 2025), HySRD-Net (Chen et al. 2026), and Vimal's code._

## 0. The one-paragraph version

A diffusion model turns data into a simple noise prior and back. It works best when the
data already *looks like* the prior. Our data is a **forecaster residual** `r = y − ŷ`.
The whole design question is how to close the gap between the residual's distribution and
the prior. There are two dual levers: **reshape the data to fit white noise** (HySRD) or
**reshape the prior to fit the data** (GAD / us). We measured the gap and found: (a) the
prior is a *diagonal* object — a per-mode variance profile — and matching it to the
residual's actual profile removes ~7× of the shape mismatch; but (b) only **15–42%** of
the residual's covariance is diagonal in the graph eigenbasis, so the bulk (cross-mode
correlation) is unreachable by *any* prior and must be learned by the **denoiser**.
Conclusion: the prior's honest job is to get the per-mode variance scale right (cheap,
closed-form), which mainly helps a **small** denoiser; the graph structure belongs in the
denoiser, not the noise.

## 1. What a prior actually is — basis × variances

Every Gaussian prior we use factors into **two independent choices**:

1. a **basis** (the "directions" / modes): eigenvectors `V_S ⊗ V_T`.
2. a **per-mode variance** (the "width" of the noise in each direction): `σ²/θ[k,h]`.

A prior is fully described by (basis, variance-profile). That decomposition is the key to
naming the variants — and answers "does it use the Laplacian?":

| variant | spatial basis `V_S` | spatial variance | temporal basis `V_T` | temporal variance | uses graph? |
|---|---|---|---|---|---|
| **iso** (white) | identity | flat | identity | flat | no |
| **kronC** (Cartesian) | **graph Laplacian** eigvecs | `1/(λ_S+γ)` (heat formula) | path-graph Laplacian | `1/(ω_T+γ)` | fully (basis **and** widths) |
| **kronE** (empirical) | **graph Laplacian** eigvecs | `1/(λ_S+γ)` (heat formula) | data covariance | data eigenvalues | **spatially yes**; temporally data-fit |
| **reshaped** | **graph Laplacian** eigvecs | **measured from residual** | data covariance | **measured** | basis only (widths from data) |
| **full-empirical** | residual's own eigvecs | measured | residual's own eigvecs | measured | **no graph at all** |

Reading of the two names the question was about:

- **"empirical" (kronE)** keeps the Laplacian **fully on the spatial side** — spatial
  directions *and* spatial widths still come from `(L+γI)`. Only the **temporal** factor is
  data-fit (a fitted covariance instead of a path graph). So it is *not* "no Laplacian."
- **"reshaped"** (as measured in `explain.py`) keeps the graph Laplacian eigenvectors as the
  spatial **basis**, but replaces the heat-kernel width formula `1/(λ+γ)` with the residual's
  **actually measured** per-mode variance. The graph still defines the modes; we simply stop
  trusting the heat-equation formula for how wide each mode's noise should be and measure it.
- The genuinely graph-free option is **full-empirical**: use the residual's own covariance
  (its own directions and widths). Then the graph plays no role — it is just "fit a Gaussian
  to the residual," and the graph-aware story is gone. We do **not** propose this as the
  headline; it is the ablation that shows what the graph is (or isn't) worth.

## 2. The two dual levers (why "colored beats white" is not, by itself, novel)

The gap between residual and prior can be closed from either side:

- **HySRD** reshapes the **data**: trend removal + residual amplification + **per-node**
  normalization + graph-Fourier transform, until the residual is ~stationary/white, then a
  standard white-noise DDPM with a **large** denoiser (sLSTM + ChebyKAN + WaveNet).
- **GAD / us** reshape the **prior**: give the noise a covariance that matches the data.

These are the **same move** (PriorGrad equivalence: a data-covariance prior ≡ whitening the
data then using a white prior). So a colored/reshaped prior is not automatically better than
white — at the optimum they coincide. The novelty has to be about *which side is cheaper* and
*what it buys a small model*, not "colored wins."

## 3. What we measured

All offline/CPU, every value derived (no module defaults). See `results/analysis/`.

**Forward + reverse are correct** (`verify_process.py`): stationary variance = σ²/θ, forward
reaches it, FCPS mixes the slowest mode (`2·θ_min·c₀ = 11.2 ≥ 5`), heat ordering holds — all
pass on both variants. Reverse (oracle denoiser): **EM diverges below the `θ_max·c₀` step
floor and is exact at it; ancestral converges with no floor.** Reproduces GAD's Fig. 1.

**Prior-vs-residual shape mismatch** (`prior_residual_match.md`), TV = fraction of prior
variance on the wrong modes; per-mode KL(residual‖prior) in nats (pems08f/gwn):

| prior | KL nats/mode | TV beats iso? |
|---|--:|:--|
| white (iso) | 0.288 | — |
| Cartesian Laplacian (γ=.8) | 0.085 | on 2/18 cells |
| empirical | 0.041 | **14/18 cells** |
| reshaped | 0.000 | by construction |

- The **empirical** prior matches best on **strong** backbones (gwn/gcn); iso wins only on
  weak-backbone (MLP) cells, where the empirical prior becomes pathologically stiff (one giant
  temporal common mode, stiffness up to 308) and over-concentrates.
- The **Cartesian/Laplacian** prior — the GAD-faithful one — is the **worst** match almost
  everywhere: the heat-kernel width `1/(λ+γ)` is wrong for residuals (they are mid-frequency,
  not graph-smooth).

**The ceiling** (`diag_dominance`): the residual covariance is only **15–42% diagonal** in the
graph×temporal eigenbasis. A diagonal prior — iso, Laplacian, empirical, *or* reshaped — can
only ever address that diagonal fraction. The remaining **58–85%** is cross-mode correlation
that the **denoiser** must create during the reverse process. So reshaping the prior fixes the
diagonal exactly (real, but bounded), and cannot substitute for a capable denoiser.

**Noise samplers** (`explain.py`, `noise_samplers.png`): GAD's `c̄=c₀·u^0.65` biases training
to high noise → trains smooth modes fully, under-trains rough modes; log-uniform covers all
modes evenly. For our gentle Cartesian prior (stiffness 3) GAD's sampler is fine; for the stiff
empirical prior (stiffness up to 300) **log-uniform is the safer default**.

## 4. What we are missing (from HySRD, helps regardless of prior)

- **Per-node normalization** — HySRD ablation: global norm costs **+7% CRPS**. We default to
  global; a `pernode` path exists but is unused.
- **Residual amplification** (`α_amp≈100` before normalizing) — **+4.5% CRPS**. Not implemented.
- HySRD's Appendix B: a better Stage-1 predictor → better refine (16–17% ceiling from Stage-1
  error). Our strong GWN backbone is an asset here.

## 5. The honest contribution and the experiment

The prior cannot carry the graph structure (most of it is off-diagonal). Its defensible job is
to get the **per-mode variance profile** right in closed form, so the network does not have to
learn the scale — which should matter **most for a small denoiser**. That gives a sharp,
testable thesis:

> **Can a residual-spectrum-matched (reshaped) prior let a small GCNN denoiser match a large
> denoiser that uses white noise?** — i.e. move the whitening burden from the network to the
> (free) prior, and put the graph where it actually helps: inside the denoiser.

**Proposed run (high-headroom cells only):** PEMS-flow {pems08f, pems08s, pems04s} × backbone
{mlp, gwn} × prior {iso, kronE, reshaped} with a **small GCNN denoiser**, **per-node norm +
amplification**, one shared schedule-law, **ancestral** sampler, metric **CRPS** (+ MAE/RMSE,
coverage). Expectation, stated up front: a **single-digit-% CRPS** effect, largest with the
small denoiser; if reshaped ≈ iso with a large denoiser, that is itself the honest result
(the prior is a small-model aid, not a structure model). Add full-empirical as the ablation
that says whether the graph basis is worth keeping at all.
