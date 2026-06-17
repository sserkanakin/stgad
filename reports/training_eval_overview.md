# Training & Evaluation Overview — diffusion prior over forecast residuals

**Experiment tag:** `20260612` · **branch:** `structured-t-prior` · **backbone:** Graph WaveNet (`gwn`)
**Grid:** 3 datasets (metrla, pems08f, pemsbay) × 2 priors (isotropic, kronecker) × 3 seeds (42/43/44) = **18 trained runs, 18 test evals — all complete.**
**Sampler used for the test eval:** full stochastic reverse SDE (`eta=1`), `n_steps=50`, `k=50` samples/window, 90% intervals (`alpha=0.1`). The `(n_steps, k)` were grounded by the sampling ablation, not guessed.

---

## 1. What we are modeling, and the goal of training

We do **probabilistic** traffic forecasting in two stages:

1. **Backbone (Graph WaveNet).** A deterministic point forecaster already trained. It outputs `Y_hat` for each (window, node, horizon). Its mistakes are the **residuals** `R0 = Y_true − Y_hat`.
2. **Diffusion model over the residuals.** A score/denoising network `R0Denoiser` learns the *distribution* of those residuals, conditioned on the backbone forecast `Y_hat`. At inference we draw `K` residual samples from the reverse SDE and add them back: `forecast = Y_hat + R0_sample`. The spread of those samples is the predictive uncertainty.

Because every metric is a difference of forecasts, the backbone term cancels exactly:
`forecast_err = (Y_hat + R0_hat) − (Y_hat + R0_true) = R0_hat − R0_true`.
So **scoring residuals = scoring the final forecast** — `Y_hat` never enters the numbers below.

**The thing being tested in this experiment** is the *noise prior* of the diffusion process:

- **`isotropic`** — the baseline: white Gaussian noise, `Σ = σ²I`.
- **`kronecker`** — the thesis contribution: a **structured spatio-temporal** prior `Σ = Σ_T ⊗ Σ_S`, with the spatial factor `Σ_S = (L + γI)⁻¹` tied to the graph Laplacian `L` (these runs use `--spatial structured` on the *repaired/connected* graph). The schedule constant `c_0` is auto-derived from the fitted prior (`5/θ_min`): kronecker ≈ 193, isotropic ≈ 5 for metrla.

**Training objective (both priors, identical otherwise).** Standard denoising:
draw `t ~ U[t_eps, T]`, forward-noise `R_t`, predict the clean residual `R0_hat = denoiser(R_t, t, Y_hat)`, minimize **masked MSE** `‖R0_hat − R0‖²` in unit-variance space (residuals divided by a single train-set scalar `sigma_r`). Adam, lr 1e-3, batch 32, ≤200 epochs, early stop on val loss (patience 15), ~1.04M params.

> **The goal is a calibrated, sharp predictive distribution of the forecast error — not a better point estimate.** A well-trained backbone leaves residuals that are largely *aleatoric* (irreducible). The diffusion model's job is to characterize that uncertainty (intervals, quantiles, the joint spatial shape of the error field), and the open research question is whether the structured Kronecker prior buys calibration/fidelity over plain isotropic noise.

---

## 2. The during-training validation evals (what we added, and what each measures)

Every run logs three families of validation metrics. They live in the training logs (`logs/20260612/diffusion/...`) and wandb.

| Eval | Cadence | Space / sampler | What it answers |
|---|---|---|---|
| **val denoising MSE** | every epoch | unit space | Is the denoiser learning the score? (this is the early-stop signal) |
| **reconstruction eval** → `recon-MAE/MSE`, `Winkler90`, `coverage` | every 5 ep, 1 fixed val batch | raw units, `eta=0`, K=20 | Does the reverse process invert the forward process, and are the point/interval estimates sane? |
| **distribution comparison** → `W1` (sliced Wasserstein-1) + `energy` | every 20 ep, 512 clean val windows | raw units, `eta=1` | Does the *generated residual cloud* look like the *true residual cloud*, per horizon, across nodes? |

A few things worth knowing about these diagnostics:

- **val denoising MSE** is in *unit* (z-scored) space, so the absolute number is not comparable across priors — kronecker's noised targets are simply harder to denoise (it starts at ~0.43 vs isotropic's ~0.22). What matters is that *each* run's loss falls monotonically.
- The **reconstruction eval is a single 32-window val batch with `eta=0` (deterministic)** — cheap and noisy, meant as a "is anything broken" probe, not the verdict. Its Winkler/coverage **under-disperse on purpose** (`eta=0` collapses the noise), which is exactly why the *test* eval switched to `eta=1`. Do not compare these coverage numbers to the test ones.
- The **W1 / energy distribution comparison uses `eta=1`** (full stochastic SDE) and is the diagnostic that actually tracks generative quality.

---

## 3. How much did training improve things?

All values below are **means over the 3 seeds**. Raw units = mph (metrla, pemsbay) / vehicle-flow (pems08f).

### 3a. The denoiser learns (val loss)

| dataset | prior | epochs (best) | val denoise-MSE ep0 → best |
|---|---|---|---|
| metrla | iso | 50 (35) | 0.224 → 0.178  (−20%) |
| metrla | kron | 43 (28) | 0.436 → 0.366  (−16%) |
| pems08f | iso | 56 (41) | 0.269 → 0.213  (−21%) |
| pems08f | kron | 36 (21) | 0.425 → 0.353  (−17%) |
| pemsbay | iso | 33 (18) | 0.223 → 0.186  (−17%) |
| pemsbay | kron | 52 (37) | 0.470 → 0.400  (−15%) |

Every run trains cleanly and early-stops well before the 200-epoch cap. (Kronecker's loss is on a different scale — see §2 — so the −16% is the relevant figure, not the absolute value.)

### 3b. The point error (recon-MAE) barely moves — and that is the expected result

The **recon-MAE / MSE** of the sample mean stay essentially flat over training, hovering at the **irreducible residual magnitude** of the backbone:

| dataset | prior | recon gen-MAE first → last | W1 first → last | energy (ED) first → last |
|---|---|---|---|---|
| metrla | iso | 5.58 → 5.65 (flat) | 1.72 → **1.27** | 2.97 → **2.08** |
| metrla | kron | 5.59 → 5.48 | 1.52 → 1.38 | 2.65 → 2.41 |
| pems08f | iso | 16.18 → 14.22 | 9.20 → 9.86 | 5.98 → 6.31 |
| pems08f | kron | 16.44 → 14.14 | 12.13 → 11.05 | 7.91 → 6.81 |
| pemsbay | iso | 1.30 → 1.18 | 1.30 → **0.98** | 2.72 → **1.76** |
| pemsbay | kron | 1.33 → 1.19 | 1.54 → **0.90** | 3.25 → **1.78** |

**Read this carefully — it is the central story:**

- **The point MAE was never going to improve much.** The residual is mostly noise the backbone couldn't predict; the conditional *mean* of that noise is ≈0. So the sample mean barely beats "predict zero residual" (= use the backbone as-is). On metrla it's even fractionally worse; on pems08f/pemsbay the conditional mean shaves a little (16.2→14.2, 1.30→1.18). The recon-MAE flatness is a feature, not a failure.
- **The real learning signal is the distribution comparison.** W1 and energy distance drop substantially as training proceeds (metrla-iso W1 1.72→1.27, ED 2.97→2.08; pemsbay-kron W1 1.54→0.90, ED 3.25→1.78). **That** is the diffusion model learning to reproduce the shape and spread of the residual field — which is the whole point.
- pems08f is the exception where W1/ED do *not* cleanly improve (iso even drifts up 9.2→9.9) — its residual cloud is the hardest to match, and it is also where kronecker fails worst on test (§5).

---

## 4. The latest test eval — every metric explained

Run on the **held-out test split**, full reverse SDE (`eta=1`), `k=50` samples/window, `n_steps=50`, 90% central intervals. All in raw units, masked to valid observations.

### 4a. The metrics

**Point accuracy (of the K-sample posterior mean):**
- **MAE / MSE / RMSE** — `|mean(samples) − R0_true|` and its square, averaged over valid (window, node, horizon). The sample mean is the **L2-optimal** estimate, so watch RMSE more than MAE here.

**Interval calibration & sharpness (the 90% predictive interval `[q5, q95]` of the K samples):**
- **Winkler score** (a.k.a. interval score) — `width + (2/α)·(distance the truth falls outside)`. Lower is better; it rewards *narrow* intervals but penalizes a miss by `2/α = 20×` the shortfall. The single number that trades off sharpness against coverage.
- **Coverage** — fraction of true residuals that land inside the 90% interval. **Target = 0.90.** Below 0.90 = intervals too narrow (overconfident); above = too wide.

**Full predictive-distribution quality (per point):**
- **CRPS** (continuous ranked probability score, energy form) — `E|X − y| − ½·E|X − X'|` estimated from the K samples. A *proper* scoring rule: rewards the entire predictive distribution being both accurate and well-spread, in raw units. Lower is better. (Reduces to MAE if the prediction is a point mass, so CRPS < MAE means the spread is earning its keep.)

**Joint distributional fidelity (cloud-vs-cloud, across nodes):**
- **W1 — sliced Wasserstein-1** — generate a cloud of residual *vectors* in node-space `ℝ^N`, project onto random directions, average the 1-D Wasserstein distance to the true cloud, per horizon. Asks: do generated residual **patterns over the sensor network** look real? (not just marginally per point).
- **Energy distance** — `√(2·E‖X−Y‖ − E‖X−X'‖ − E‖Y−Y'‖)` over the same `ℝ^N` node vectors. A second, complementary joint two-sample distance.

The first three families are **per-point / marginal**; W1 and energy are the **joint** checks that the spatial error field is realistic.

### 4b. Results (mean ± std over 3 seeds; **bold = better of the two priors**, lower is better except coverage→0.90)

| dataset | prior | MAE | RMSE | CRPS | coverage | Winkler90 | W1 | energy |
|---|---|---|---|---|---|---|---|---|
| metrla | iso  | **3.42** | **6.30** | **2.52** | **0.770** | **24.96** | **1.53** | **2.43** |
| metrla | kron | 3.61 | 6.57 | 2.64 | 0.769 | 26.42 | 1.79 | 2.63 |
| pems08f | iso  | **17.10** | **26.34** | **12.62** | **0.772** | **117.2** | **5.73** | **4.06** |
| pems08f | kron | 17.29 | 26.66 | 13.11 | 0.640 | 136.7 | 7.11 | 5.11 |
| pemsbay | iso  | 1.79 | 3.77 | 1.34 | **0.754** | **13.83** | 0.87 | 1.93 |
| pemsbay | kron | **1.76** | **3.77** | **1.33** | 0.711 | 14.25 | **0.88**¹ | **1.91** |

¹ pemsbay W1 is a tie (iso 0.873 vs kron 0.876); kron wins CRPS/MAE/energy there.

### 4c. Point accuracy vs the backbone alone

The "predict-zero-residual" baseline = **the backbone GWN by itself** (test split, raw units):

| dataset | backbone MAE₀ | backbone RMSE₀ | diffusion-mean MAE (iso) | diffusion-mean RMSE (iso) |
|---|---|---|---|---|
| metrla | 3.31 | 6.54 | 3.42 (+3%) | **6.30 (−4%)** |
| pems08f | 17.28 | 27.43 | 17.10 (−1%) | **26.34 (−4%)** |
| pemsbay | 1.68 | 3.78 | 1.79 (+6%) | 3.77 (≈0%) |

Exactly as predicted in §3b: the K-sample mean **modestly lowers RMSE** (it's the squared-loss-optimal estimate — averaging 50 draws shrinks toward the conditional mean) while **MAE is flat-to-slightly-worse** (L1 would want the median, and the small shrinkage bias costs a little there). **Point forecasting is a wash; the diffusion model earns its place on the distribution, not the mean.**

### 4d. Per-horizon (test MAE, mean over seeds)

| dataset | prior | h3 (15min) | h6 (30min) | h12 (60min) |
|---|---|---|---|---|
| metrla | iso | 3.05 | 3.40 | 4.03 |
| metrla | kron | 3.11 | 3.65 | 4.38 |
| pems08f | iso | 15.76 | 16.95 | 19.64 |
| pems08f | kron | **15.56** | 17.39 | 20.01 |
| pemsbay | iso | 1.48 | 1.88 | 2.19 |
| pemsbay | kron | **1.44** | **1.85** | **2.16** |

Error grows with horizon as expected. Kronecker is competitive (even best) at the short horizon on the big graphs but loses ground as the horizon lengthens — its relative weakness is a long-horizon one.

---

## 5. Verdict: isotropic vs kronecker

**On the test set, the isotropic baseline matches or beats the structured Kronecker prior almost everywhere.**

- **metrla, pems08f:** isotropic wins on *every* metric. On pems08f the gap is large and calibration-driven — kron coverage collapses to **0.64** (vs iso 0.77) and its Winkler/W1/energy are 15–25% worse.
- **pemsbay:** the only dataset where kron is competitive — it edges iso on MAE/RMSE/CRPS/energy, ties on W1 — but **still under-covers worse** (0.71 vs 0.75).

This **contradicts the earlier validation-set finding** (notes: `normalization-breakthrough`, where on pems08f *val* the Kronecker prior beat isotropic with coverage rising to 0.88). On the **test** split with the corrected `eta=1` sampler and the grounded `(n_steps=50, k=50)`, that advantage does not hold up. Worth reconciling before drawing thesis conclusions — candidates: val↔test distribution shift, the schedule-`c_0` fix (commit `8803302`), or the `eta`/`k`/`n_steps` change between the during-training probe and the test eval.

---

## 6. The cross-cutting issue: under-coverage

**Every run under-covers** — test coverage ranges **0.64–0.77 against a 0.90 target**, for *both* priors and *all three* datasets. The 90% predictive intervals are systematically too narrow. This is the single biggest open problem and it is **not** a prior-structure issue (isotropic under-covers too); it is a calibration/dispersion issue of the diffusion sampler itself. Switching the test eval to `eta=1` (full stochastic SDE) helped relative to the `eta=0` during-training probe, but did not close the gap. Likely next levers: more reverse steps, a noise/temperature adjustment at sampling time, or post-hoc interval recalibration.

---

## 7. Provenance & caveats

- **All 18 jobs COMPLETED (exit 0).** Last finished 2026-06-13 05:39 (pemsbay/iso/seed44). Results synced to the Mac and byte-for-byte consistent with the cluster.
- **Eval sampler is grounded, not guessed:** `(n_steps=50, k=50)` come from the sampling ablation (`results/analysis/ablation_*_kronecker_seed42.json`) via `pick_knee.py`.
- **Schedule bug fixed before these runs:** eval rebuilds the reverse schedule from the fitted prior (`c_0 = 5/θ_min`), not `config.json`'s ignored CLI default — earlier evals were invalid (commit `8803302`).
- **During-training recon numbers are diagnostic only** — single 32-window val batch, `eta=0`, K=20. The §4 test numbers (full test set, `eta=1`, K=50) are the ones to cite.
- **Units:** metrla/pemsbay in mph, pems08f in vehicle-flow; metrla/pemsbay residuals are de-normalized by `sigma_r` (metrla ≈ 6.16). Horizons h3/h6/h12 = 15/30/60 min at 5-min resolution.

---

## 8. One-paragraph summary

We trained a residual diffusion model on top of a Graph WaveNet backbone, comparing a structured spatio-temporal **Kronecker** noise prior against an **isotropic** baseline across three datasets and three seeds. Training is healthy everywhere (val denoising loss −15–21%, clean early-stop). As designed, the model does **not** improve point forecasts — the K-sample mean ties the backbone on MAE and shaves a few % off RMSE — while the **distributional** metrics it was built for (sliced-W1, energy distance) improve markedly over training. On the held-out test set, however, the **isotropic prior matches or beats Kronecker on nearly every metric**, with Kronecker only competitive on pemsbay; and **all runs under-cover** their 90% intervals (coverage 0.64–0.77 vs 0.90). The structured prior is not paying off on test, and interval calibration is the key open problem — both reversing earlier val-set impressions and pointing to the next round of work.