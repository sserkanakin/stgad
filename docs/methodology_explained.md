# The method, explained one step at a time

*Plain-language walkthrough of everything the system does and why — written 2026-07-23 as
the companion to `docs/overnight_report_20260723.md`. Every claim here has a measured number
behind it; the tables live in the report and `results/`.*

---

## 0. The goal in one paragraph

We forecast traffic (speed or flow) at N road sensors, 12 five-minute steps ahead, and we
want a **distribution** over futures, not just one number — so an operator can ask "how bad
could it get?" and get a calibrated answer. Our design is **predict-then-refine**: a small
deterministic model makes a point forecast, and a diffusion model learns the distribution of
that model's *errors* (residuals). The final forecast ensemble = point forecast + sampled
residuals. The thesis question is: **what should the diffusion's noise look like?** Plain
white noise (isotropic), or noise shaped by the road graph and the data (structured priors)?

## 1. The backbone ladder (mlp → gcn → gwn) — and why there are three

**What:** three point forecasters of very different strength: a 312-parameter linear model
(DLinear-style "mlp"), a 17k-parameter graph convolution net (STGCN-style "gcn"), and the
309k-parameter Graph WaveNet ("gwn").

**Why three:** the residual is *defined by* the backbone — a weak backbone leaves lots of
structure in its errors, a strong one leaves almost none. If we tested the prior on only one
backbone, we couldn't tell "the prior helps" apart from "the backbone was weak". The ladder
turns backbone strength into a controlled axis. (Crucial nuance: the diffusion *denoiser* is
the same 17k network in every cell — the ladder varies the data the diffusion sees, never
the diffusion's capacity.)

**What it showed:** the prior's benefit tracks exactly how much structure the backbone
leaves behind — big wins on mlp (up to −31% CRPS), ~1% on gcn, zero on gwn.

## 2. Residuals + per-node normalization

**What:** residual r = truth − backbone forecast, shape (windows, N sensors, 12 horizons),
computed once per dataset/backbone on train/val/test with the true data mask. Before the
diffusion sees them, residuals are divided by each sensor's own training-set std
(**per-node normalization**).

**Why per-node:** sensors differ wildly in scale (a quiet ramp vs an 8-lane highway).
Without per-node scaling the loud sensors dominate training and the quiet ones are ignored;
HySRD's ablation puts this at ~7% CRPS, and our runs confirmed the direction. The prior is
fit in this same normalized space, and everything is mapped back to raw units at the output
(metrics are computed in raw units on the reconstructed forecast).

## 3. The forward process — "graph heat + noise", in words

**What:** diffusion needs a way to gradually destroy data into noise. Ours is an
Ornstein–Uhlenbeck process per mode: write the residual in the eigenbasis of the road
graph's Laplacian (spatial) ⊗ a temporal basis; each coordinate decays toward zero at its
own rate θ while noise is injected, ending at the stationary distribution N(0, Σ) — the
**prior**.

**Why the Laplacian/heat framing (GAD's idea):** heat on a graph flows along edges — rough
patterns (neighbors disagreeing) die fast, smooth patterns die slowly. Destroying "detail
first, structure last" mirrors what image diffusion does with high frequencies. It builds in
the assumption that what we diffuse is graph-smooth.

**Verified:** `verify_process.py` checks the math numerically — stationary variance matches
the closed form, the forward reaches it, the schedule mixes the slowest mode, the heat
ordering holds (all PASS); `forward_analysis.py` showed the forward reaches the prior on
all six datasets.

## 4. The prior family — one question, three answers

Each prior is one answer to "what does residual noise look like?", all matched to the same
total variance and the same slowest rate so *only the shape* differs:

- **iso** — white: every mode has the same variance. The null hypothesis.
- **empirical (emp)** — Kronecker structure: graph Laplacian eigenvectors (+γ) for space ⊗
  a temporal covariance *measured from the residuals*. "Noise is edge-correlated in space
  and data-shaped in time."
- **reshaped (rsh)** — keep the same eigenvectors as coordinates, but set every mode's
  variance to what the residual actually measures. The perfect *diagonal* match — GAD's own
  Remark 1 made data-driven.

**Fairness engineering (why the comparison is clean):** identical denoiser, loss, epochs,
learning rate, noise-level sampler, and schedule machinery in every arm; the prior's overall
scale σ² is fit to the same residual variance; the slowest rate is normalized identically so
every arm gets the same effective time horizon. A cited justification exists too: PriorGrad's
Prop. 2 defines the fair iso control as *entropy-matched*, which is **smaller** than ours —
our iso is, if anything, over-generous.

## 5. Schedule and training noise levels

**What:** FCPS — a geometric grid over the total "heat" c̄, with the training noise level
drawn log-uniformly, so every mode's transition region gets training signal. Audited: 100%
of modes lie inside the trained noise range in every arm, including the stiffest.

**Why it can't explain iso winning:** it is shared machinery derived from θ; for iso it
simply reduces to a single-rate schedule. The empirical check agrees — the step-count curve
is flat (5→100 steps changes CRPS by <0.5%) and an independent EM solver at 1500 steps
reproduces the ancestral numbers.

## 6. The denoiser and conditioning

**What:** a 17,305-parameter graph-convolutional denoiser (GCNN), fixed across all cells.
Conditioning C is what the denoiser is allowed to see: **fo** = the backbone's future
forecast Ŷ only; **past** = [previous window's residual | Ŷ].

**Why past matters:** the literature (verified in DiffSTG's source) conditions on the past
history plus time features — fo is nonstandard-minimal. Measured: past conditioning cuts
mlp's CRPS by 43–64% and gcn's by 6–20% (the denoiser learns to *correct* the weak backbone,
MAE 74→25 on pems08f/mlp), and does nothing for gwn (which already consumed its history).
Headline configuration = past.

## 7. Sampling and speed

**What:** headline sampler is DDPM-ancestral (exact for this linear process) at 50 steps,
k=100 samples per window; EM SDE integration is the cross-check ablation. Samples are
denormalized, added to the point forecast, clipped at zero (physical), and scored.

**Speed for the paper:** every eval json logs `gen_seconds` and `s_per_window` — pure
reverse-sampling wall time, accumulated the same way DiffSTG's own timing is. With a 17k
denoiser at 50 steps this is the "cheap inference" column against ~1M-parameter baselines.

## 8. Metrics — what each one means

- **CRPS** (headline): for every sensor/time, how far is the predicted *distribution* from
  the observed value — generalizes MAE to distributions; normalized 19-quantile form, exactly
  as CSDI/DiffSTG/HySRD report it. Lower is better.
- **MAE / RMSE:** accuracy of the ensemble mean (point quality).
- **Coverage / PIT / Winkler:** calibration — does the central 90% interval contain the truth
  90% of the time (coverage), is the truth uniform in the predicted CDF (PIT; U-shape =
  over-confident), interval width + penalty (Winkler).
- **W1 / energy:** joint-distribution quality across sensors (do sampled *fields* look real).
- **Protocol:** test split (60/20/20 chronological), y==0 masked (PEMS convention, matches
  the baselines), 5 seeds where available, best-CRPS checkpoint.

## 9. What we measured about the data (the diagnosis that drove everything)

1. Residuals are **mildly graph-smooth** (smoothness index 0.75–0.88; neighbors correlate
   0.13–0.44) — but the **exploitable spread** collapses with backbone strength: mlp ~10.5 dB,
   gcn 5.5–7 dB, gwn-flow 1.1 dB (white). Static prior gain ≈ spread × stationarity.
2. The denoiser **substitutes** for static shape: whatever fixed structure the prior offers,
   a trained denoiser under iso learns by itself (why whitened-loss and all γ/shape variants
   failed to win on gwn; per-node errors of iso and emp correlate 0.9999).
3. What a frozen denoiser **cannot** learn: (a) residual scale swings **2–3.3× by time of
   day** with a train→test-stable profile; (b) on speed data the loud modes' variance
   **drifts 5–22×** train→test as a monotone, val-predictable trend. Both are *variance*
   structure conditional on things known at forecast time — precisely a prior's job.

## 10. The adaptive recalibration (the contribution that wins)

**Recipe:** sample the trained model on the *validation* split; fit three spread corrections
there — a global scalar, a 24-bucket time-of-day profile s(tod), and per-mode-energy-decile
scales s_k in the prior's eigenbasis (optionally extrapolating each decile's train→val
variance trend half a log-step forward); apply the frozen corrections to the *test*
ensembles (widen/narrow each ensemble around its mean). **No test data is used for fitting;
no retraining anywhere.**

**Why iso can't match it:** iso is offered the same global and tod corrections — everything
expressible with one dial per window. The mode-resolved dials require coordinates, and only
the structured prior has them. That is the precise sense in which "the structured prior
wins": its *basis* carries adaptive variance that flat noise cannot represent.

**Result:** with past conditioning, emp+adaptive beats iso's best-calibrated version on
**10/10 cells** across all six datasets (margins 0.5–3%, usually with better coverage);
the drift-extrapolated variant wins exactly on the drift datasets. Novelty checked against
2025–26 literature: unclaimed (cite NsDiff, NPDiff, PriorGrad/RestoreGrad, Double-Diffusion).

## 11. The final run (fired 2026-07-23)

Goal: uniform 3 seeds (42/1/126) × 6 datasets × 3 backbones × iso/emp/rsh × both
conditionings, full locked-protocol test evals, plus the recalibration layer at 3 seeds —
the complete number set for the thesis. Machinery byte-identical to the audited campaign
(unit tests 39/39 logic pass; forward verification PASS; config parity re-checked).
pems08 and pems04-fo already have 4–5 seeds; the run fills metrla/pemsbay (gcn all seeds,
mlp/gwn seed 126) and pems04-past seed 126, then evals everything. Sampling-speed numbers
ride along in every eval json; forward/reverse diagnostics are already banked
(`verify_process`, `forward_analysis`, few-step curve, EM cross-check).

## 12. Honest limitations

Adaptive-recal numbers are k=32 on a 600-window test stride (protocol-grade k=100 in
`eval_diffusion.py` is the next implementation step); heavy tails mean no Gaussian ensemble
is fully calibrated (Student-t is the known remaining lever); HySRD's implementation details
are paywalled (verify via library before quoting); gcn does not reach SOTA (gwn does) — the
gcn claim is "prior wins at fixed small capacity", not "gcn beats HySRD".
