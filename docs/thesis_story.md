# The thesis story — complete, connected, defensible

*Rewritten 2026-07-23 (late). REVISION: adaptive recalibration REMOVED from the
contributions per user decision — it is calibration post-processing, not novelty. The
recal results remain on disk (results/analysis/adaptive_recal/) and may appear, at most,
as a one-paragraph calibration note. The thesis stands on the system, the controlled
characterization, and the mechanism.*

---

## The one-sentence thesis

**In diffusion-based traffic forecasting, graph structure belongs in the network, not in
the noise: a small white-noise diffusion refining a point forecaster beats larger
structured-noise models, and a controlled study — culminating in a best-case test on
provably graph-smooth signals with an analytic mechanism — shows why structured forward
noise cannot improve a capable conditional diffusion, paying off only at low capacity and
at few sampling steps.**

Two research questions:

- **RQ1 — System:** can a *small* diffusion refine a deterministic forecaster into
  state-of-the-art probabilistic forecasts? (Yes.)
- **RQ2 — Structure:** does graph-/data-structured forward noise (GAD-style) improve the
  diffusion — and if not, why, and where does it retain value? (No for accuracy at
  capacity, with a measured law, an analytic mechanism, and two bounded regimes where it
  does pay: low capacity and few-step sampling.)

## The contributions (what we claim)

1. **A SOTA probabilistic forecasting system at a fraction of the size** — predict–refine
   with a 17k-parameter denoiser beating DiffSTG/SpecSTG/HySRD on all six benchmarks.
2. **The first controlled decomposition of graph-aware diffusion for forecasting** — a
   matched-isotropic control (which GAD's own paper lacks), a backbone-capacity ladder, a
   denoiser-capacity ladder, both conditioning regimes, and a spatio-temporal extension of
   GAD's heat prior (Cartesian product graph — the only construction that remains a true
   space-time heat equation).
3. **The mechanism:** the Wiener-optimal-denoiser bound showing that a data-matched prior
   concentrates noise in the signal's own subspace and is therefore *intrinsically* harder
   to denoise (+20–28% at matched power) — "a good data model is a bad corruption." This
   explains every negative result at once and reconciles GAD's published wins
   (unconditional, few-step, no matched control).
4. **The boundary map of where structure pays:** in proportion to the structure the network
   lacks (mlp −31%, gcn ≈1% incl. the consistent gcn+past win band with −8/−10% joint
   metrics on flow, gwn 0), and in few-step sampling robustness (structured priors lose
   1.5–4% going 100→5 steps vs iso's 4–6%; weak-backbone absolute wins widen at 5 steps).

## Chapter: Methodology

1. **Predict–refine pipeline** (backbone point forecast + residual diffusion; ensemble =
   forecast + sampled residuals). Motivation: the backbone absorbs the predictable part so
   a tiny denoiser suffices (17k vs DiffSTG's ~1M).
2. **The backbone ladder** (mlp 312 / gcn 17k / gwn 309k) as confounder control — the
   ladder varies the residual's information content with the diffusion FIXED.
3. **The denoiser-capacity ladder** on the signal arm (997k/101k/38k/16k UGnet variants,
   16k ≈ GAD's own GCNN) — the second axis of the decomposition.
4. **The prior family**, engineered for clean comparison: iso (scale-matched white — over-
   generous vs PriorGrad Prop. 2's entropy-matched control), empirical Kronecker, reshaped,
   and the GAD-literal Cartesian heat prior Σ=σ²(L_ST+γI)⁻¹ on the space⊕time product
   graph (γ by spectral match; forward/backward verified against GAD Fig. 1/Fig. 2;
   multiplicative/Kronecker-strong variants rejected as non-heat, spectrum-degenerate).
   All arms share machinery (denoiser, loss, schedule family, sampler, normalization).
5. **Conditioning** per the literature standard (past window; DiffSTG's time features are
   dead code in their repo — audited); forecast-only kept as an ablation axis.
6. **Locked evaluation protocol** (test, ancestral@50, k=100, y==0 masked, 19-quantile
   CRPS + MAE/RMSE + coverage/PIT/Winkler + joint energy/W1, multi-seed) and the
   protocol-noise lesson (in-training k=4 margins of 1–2% do not survive the locked
   protocol) as an explicit methodological point.

## Chapter: Experiments — the arc

**E1 — The system result (RQ1).** gwn + 17k denoiser beats every cited baseline on every
test set: PEMS08F 0.0497 vs HySRD 0.0547 / SpecSTG 0.0572 / DiffSTG 0.0692; PEMS08S 0.0174
vs 0.0204; PEMS04F 0.0663 vs 0.0723; PEMS04S 0.0205 vs 0.0241; metrla 0.0432 / pemsbay
0.0209 (DiffSTG's own code: metrla 0.1599). Past-conditioning is the standard config; best
cells gwn-past 0.0496/0.0172/0.0662/0.0202. Sampling wall-time logged (gen_seconds) — the
efficiency column vs ~1M-param baselines.

**E2 — The ladder law (RQ2 on residuals).** Structured-prior gains track the structure the
backbone leaves behind (measured spectral spread 10.8→1.1 dB): mlp-fo up to −31%; gcn ≈1%
— including the one consistent raw win band, gcn+past empirical beating iso on every
dataset's gcn cells with the largest margins on joint metrics (energy −8%, W1 −10%,
pems04f) — and gwn 0 (white residual). Past-conditioning cuts mlp by 43–64% and erases its
prior advantage: capacity, conditioning, and prior are substitutable information channels,
and the learned channels dominate. No static fix (whitened loss, γ, shrinkage, EM) flips a
capable cell.

**E3 — The best-case control (RQ2 on signals).** On the raw signal — provably graph-smooth
(spread 6.5–7.3 dB vs the white gwn residual's 0.3), with the heat prior fitting 0.29–0.58
nats/coord better than iso, forward/backward verified, audited DiffSTG-equivalent iso —
matched white noise still wins or ties across 6 datasets × 4 capacities, including −6 to
−16% at GAD's own 16k size. **Mechanism:** the Wiener bound — the optimal denoiser removes
graph-correlated noise 20–28% worse than white at matched power because the prior's noise
occupies the signal's subspace. Pure math, no training in the argument.

**E3b — Where structure retains value: few sampling steps.** Structured priors degrade
1.5–4% from 100→5 reverse steps vs iso's 4–6%; on weak backbones the absolute win widens
(rsh 0.179 vs iso 0.264 at 5 steps, pems08f/mlp). Consistent with the Wiener curves
converging at low noise — kron's handicap vanishes exactly in GAD's claimed regime.
Structured noise is a fast-sampling property, not an accuracy property. (Raw-signal
few-step curves: running.)

## Chapter: Discussion

- **One mechanism, three channels:** capacity, conditioning, and prior are substitutes;
  learned channels are state-dependent and strictly stronger for structure; the prior's
  static structure only pays when the others are starved (low capacity) or when step
  budgets are tiny (few-step). The best-case control closes the loop: even a provably
  better-fitting prior loses at capacity, for computable reasons.
- **Graph structure belongs in the operator** — backbone and denoiser are both
  graph-convolutional, and that is where every observed gain from the graph actually lives.
- **Related work, reconciled:** GAD (unconditional, few-step, no matched-iso control —
  our E3 supplies the control and reproduces their few-step property); PriorGrad (helps
  most at small capacity — our 16k rung bounds it); HySRD (the dual move — reshape data to
  fit white noise — implicitly agrees structure does not belong in the noise); SpecSTG
  (isotropic noise despite spectral branding); Double-Diffusion (graph-heat prior *mean*,
  not covariance).
- **Calibration note (not a contribution):** all arms under-cover on test (0.6–0.83 vs
  0.90) due to heavy tails and train→test drift; standard val-fitted recalibration
  (incl. time-of-day/per-mode variants we explored) improves coverage; Student-t noise is
  the principled future fix.
- **Limitations:** raw-signal arm is single-seed with in-training selection metrics
  (protocol evals pending); pems08s idiosyncratically favors iso everywhere; HySRD's
  implementation is paywalled; heavy-tailed residuals bound Gaussian calibration.

## Defense one-liners

- *"Maybe your structured arm is buggy."* → The prior fits the data 0.3–0.6 nats better
  and still loses; forward/backward reproduce GAD Fig. 1/2; the iso control is audited
  DiffSTG-equivalent; and the Wiener bound predicts the ordering with no training in the
  argument.
- *"Is iso a fair control?"* → Same machinery, variance-matched; the entropy-matched
  control of PriorGrad Prop. 2 would be weaker than ours.
- *"Isn't a negative result thin for a thesis?"* → It is a *characterization* with a SOTA
  system attached: the first matched-control decomposition of graph-aware diffusion for
  forecasting, an analytic mechanism, and a boundary map (capacity, conditioning, steps)
  of where structure pays. Fields advance on exactly such studies.
- *"Why believe the system numbers?"* → Locked protocol identical to the baselines'
  conventions, 3–5 seeds, DiffSTG re-run with its own code.

## What must still run to lock the manuscript

1. Protocol-grade evals (test, ancestral@50, k=100) for the raw-signal ladder cells.
2. Raw-signal few-step curves (running on sakin; dkrylov half needs re-auth).
3. MIS at α=0.05 for exact DiffSTG comparability; metrla/pemsbay seed top-up (3 seeds).
