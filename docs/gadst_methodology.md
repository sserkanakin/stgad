# GAD-ST: methodology record (figures, numbers, and where everything lives)

> **⚠️ SUPERSEDED FACTS — read `docs/methodology_experiments.md` for the current chapter.**
> This file is the *figure/number ledger* and is kept for provenance, but three headline
> choices below were corrected after auditing GAD's reference code (github.com/vimalkb7/gad):
> **(1)** γ is now **0.8** (GAD's value), not 0.04 — 0.04 gave stiffness 51 and an
> EM-unfriendly reverse; GAD's 0.8 gives stiffness 3.5. **(2)** the denoiser is GAD's
> **GCNN** (~17k params), not UGnet — matched across all arms. **(3)** the comparison is a
> **four-arm** design (GAD-ST, VPD, VED, isotropic control) in **two regimes** (unconditional
> generation + conditional forecasting), tags **`20260716_gadst`** and
> **`20260716_gadst_cond`**. The "iso wins" numbers in §3.1 predate the fix and used a
> data-fitted empirical prior, not GAD-ST. Everything in the sections below should be read
> through that lens.

Single source of truth for the methodology chapter (ch. 4) and its figures/tables.
Experiment tags: **`20260715_gadst_res`** (headline: residual diffusion) and
`20260715_rawgad` (raw-signal arm, ablation/secondary). Branch `raw-signal-gad`.

## 1. The model, in five lines

1. **Target**: GWN forecast residual `R = Y − Ŷ_GWN`, per-horizon standardized
   (`R̃ = R / s_h`, 12 scalars from train; growth re-enters samples at de-normalization).
2. **Graph**: normalized **Cartesian product** `G_ST = G_road □ P_12` — one graph on
   N·12 vertices; eigenvectors `V_S ⊗ V_T`; eigenvalues `λ_ST = (λ_S + λ_T)/2 ∈ [0, 2]`.
3. **Forward = heat equation** (GAD Eq. 4): `dx = −c(t)(L_ST + γI)x dt + √(2c(t))·σ dw`,
   γ = 0.04 fixed on all datasets, FCPS schedule (c_min=0.1, S=7, α=4).
4. **Stationary prior**: GMRF `N(0, σ²(L_ST + γI)^{-1})`; σ² a single scalar matched
   to the standardized-residual variance. Two scalars total (γ, σ²); no fitted covariance.
5. **Reverse**: history/forecast-conditioned UGnet predicts x₀; SNR-weighted MSE
   (GAD's loss); ancestral sampler (exact, ≈25 steps) + EM (GAD-parity, floor ≈266 steps).
   Iso control: identical everything, white noise σ²I, ULS schedule, same total heat.

## 2. Hyperparameters — values and their justification

| knob | value | justification |
|---|---|---|
| γ (Laplacian shift) | **0.04, fixed** | GAD's small-γ regime (paper Fig. 1 uses shifted λ₁=0.1); matches the data's low-λ energy; one value everywhere |
| c_min, S=c₀, α | 0.1, 7, 4 | GAD-literal (exp_gasde.py); mixing 2·S·θ_min = 11.2 ⇒ slowest mode retains e^{−11.2} ≈ 1e−5 |
| product type | Cartesian | only product with spectrum ⊂ [0,2] at GAD spread; frequencies add (monotone smoothness); Kronecker folds frequencies (rough×rough→smooth) and measured worse than iso on 3/4 datasets; parametric sweep (Mazzola Eq. 2.25) selects s₁₁≈0 |
| norm | perhorizon (12 scalars) | residual std grows 1.7–2.9× across horizons; stationary priors are flat; standardize → flat (0.98–1.02×), de-normalized draws grow 1.73×/1.25× vs data 1.70×/1.27× |
| sampler steps | anc 25 (headline), EM 1000 | ancestral exact; EM stability floor θ_max·S ≈ 266–287 |
| noise floor c̄_lo | 0.0008 | fixed 4% final-sample noise |
| stiffness θ_max/θ_min | 47–51 | = GAD's own spatial-only regime; graded forward cascade (no early collapse) |

## 3. Validation results (all measured; scripts in repo)

### 3.1 Pre-flight, residual arm, all 6 datasets (`scratch preflight_all.py` logic)
| dataset | λ range | NLL adv vs iso (nats/dim) | low-λ data/prior | growth data/prior | fwd→stat err | stiffness |
|---|---|---|---|---|---|---|
| metrla | [0.00, 1.85] | **+0.301** | 0.58 / 0.49 | 1.70 / 1.73 | 0.000 | 47 |
| pemsbay | [0.00, 1.94] | **+0.623** | 0.60 / 0.51 | 2.78 / 2.63 | 0.004 | 50 |
| pems04f | [0.00, 2.00] | +0.018 | 0.41 / 0.63 | 1.23 / 1.20 | 0.003 | 51 |
| pems04s | [0.00, 2.00] | **+0.395** | 0.37 / 0.63 | 2.88 / 2.55 | 0.001 | 51 |
| pems08f | [0.00, 1.99] | +0.177 | 0.46 / 0.59 | 1.27 / 1.25 | 0.001 | 51 |
| pems08s | [0.00, 1.99] | **+0.478** | 0.36 / 0.59 | 2.69 / 2.28¹ | 0.001 | 51 |

¹ 15% growth mismatch = path-graph boundary effect (end horizons have slightly
different marginal variance), largest on the steepest-growth dataset; benign.

### 3.2 Correctness of forward / reverse / loss (numerical)
- forward closed-form vs brute-force SDE integration (20k steps): moment match (MC precision).
- ancestral reverse with oracle denoiser: recovers x₀ to **0.00e+00** (machine precision);
  transition-kernel identity `a_d²·s' + s_d = s(c)` holds exactly.
- EM reverse drift = GAD paper Eq. 6 with Tweedie score (Eq. 7), verified line-by-line.
- loss = GAD's SNR-weighted x₀-MSE (`w = 1/(1+2v̄)`), masked.
- paper check: implementation matches GAD (arXiv 2510.05036) Eq. 4/5/6/7 + FCPS verbatim.

### 3.3 Product-type comparison (parametric product graph, Mazzola/Isufi Eq. 2.24–2.25)
Held-out val NLL on residuals (lower = better); Kronecker-graph is *bounded* (stiff ≈ 51)
but folds frequencies → worse than iso on 3/4:
| dataset | iso | Cartesian | Kronecker | strong | best learned s=(time,space,kron) |
|---|---|---|---|---|---|
| metrla | 0.461 | 0.099 | 0.325 | 0.168 | (0.88, 0.12, 0.00) → 0.002 |
| pems08s | 1.523 | 1.024 | 1.820 | 1.214 | (1, 0, 0) → 0.204 |
| pems08f | 1.100 | 0.886 | 1.133 | 0.910 | (0.75, 0.12, 0.25) → 0.836 |
| pems04f | 1.114 | 1.102 | 1.342 | 1.072 | (0.50, 0.00, 0.38) → 1.017 |

(The learned s₁₁ ≈ 0 = empirical justification for Cartesian; the weighted-Cartesian
s₀₁ > s₁₀ is a possible ablation, cf. GiFlow's τ-weighted Kronecker sum.)

### 3.4 Why not the old priors (context for the methodology narrative)
- covariance-Kronecker Σ_T⊗Σ_S (r2): precisions multiply → spread ~2600 → stiff, forward
  collapses by t≈0.2; best *fit* but kron≈iso when trained (denoiser absorbs temporal structure).
- empirical Σ_T: bimodal θ spectrum (one slow mode + tiny rest) — same collapse.
- GAD-ST moves the only remaining asymmetry into the noise *geometry*, symmetric
  standardization handles the temporal profile for both arms.

## 4. Figures (all regenerable; scripts in `scripts/`)

| figure | path | script | shows |
|---|---|---|---|
| forward analysis, GAD Fig-1 style (×6 ds) | `results/figures/gadst_res/forward_gadstyle_<ds>_gwn.png` | `plot_forward_gad_style.py --backbone gwn --norm perhorizon` | mean decay + cov growth per mode, ULS vs FCPS; smooth modes protected; stationary variances differ ~50× (≠ white) |
| prior validation 7-panel (×6 ds) | `results/figures/gadst_res/gadst_checks_<ds>_gwn.png` | `plot_gadst_checks.py --backbone gwn --norm perhorizon` | A temporal corr (data/kron/iso) · B per-horizon std · C spatial spectrum · D Fiedler-sorted example fields · E/F forward cascade (physical λ ≤ 2) · G stationarity: →0 to our prior, ≫0 to white |
| raw-arm versions | scratchpad (regenerate with `--backbone raw --norm global` if needed) | same scripts | same panels for the raw-signal arm |

## 5. Citable supports
- **GAD** (Rozada et al., arXiv 2510.05036): the forward/backward machinery, FCPS, GMRF stationary — our Eq.-by-Eq. base. Isufi co-author.
- **Mazzola MSc thesis (TU Delft 2020, sup. Isufi)**: parametric product graph (Eq. 2.24–2.25) — framework for the product-type selection; our sweep instantiates it.
- **GiFlow (ICLR'26 under review)**: prior on the weighted Kronecker sum (= Cartesian Laplacian); **Theorem 1: transport cost from a graph-informed prior ≤ isotropic Gaussian** — the theory for why kron should beat iso; also motivates the few-step evaluation axis.
- **DiffSTG / CSDI / HySRD**: metric conventions (19-quantile normalized CRPS, masked MAE/RMSE at y=0) — verified identical to our eval code.

## 6. Experiments ledger
| tag | what | status |
|---|---|---|
| 20260708_r2 | residual, fitted Kronecker-covariance prior vs iso | done; kron≈iso (the motivating negative) |
| 20260715_rawgad | raw signal, multiplicative product prior vs iso | trained (Denis: kron>iso on all metrics, MAE beats DiffSTG); evals chained |
| **20260715_gadst_res** | **residual, Cartesian GAD-ST prior vs iso, perhorizon** | launching (6 ds × 2 arms × 2 clusters, evals chained) |
| post-hoc | few-step evals: ancestral @5/10 steps on the same checkpoints | after training; the GAD/GiFlow few-step axis |
