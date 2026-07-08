# Decision log

One entry per locked decision; newest first. Methodology decisions reference the
evidence that forced them.

## 2026-07-08 — training protocol: sample-based selection, EMA, loss demoted to dashboard

**Decision.** Selection, early stopping and the LR plateau key off **sampled
forecast CRPS** (fixed clean val windows, fixed noise seed, k=4, ancestral
reverse at 50 steps, computed every epoch on **EMA(0.999) weights**). The raw
denoising validation loss is logged as a dashboard only and drives nothing.
`best_crps.pt` (EMA weights) is the shipped model; `best.pt` (min val loss, raw
weights) is kept as an ablation/dashboard artifact. Patience 15 on sampled CRPS,
ceiling 200 epochs.

**Why.** Two independent lines of evidence:
1. *Analytic floor*: the denoising val loss is ~93% irreducible — the observed
   metrla plateaus (kron 1.023, iso 0.894) match the per-eigenmode Gaussian
   Wiener floor (1.020 / 0.927) computed from the prior spectra and the c̄
   sampler. A monitor that saturates at the floor cannot indicate convergence.
2. *Lineage audit* (repos pulled 2026-07-08): DiffSTG early-stops on sampled
   val MAE with a frozen seed (denoising loss never consulted); CSDI has no
   early stop at all (fixed 200 epochs, all-t deterministic val loss for
   checkpointing only); HySRD-Net uses OneCycleLR (fixed budget by construction);
   the EDM/CorrDiff/GenCast lineage uses fixed budgets with EMA weights as the
   model. **No reference early-stops on the raw denoising loss.** Our 2026-07-06/07
   runs stopped at ~25-30 epochs because GWN's early-stopping protocol was pointed
   at a saturated signal — the monitor stopped, not the model.

**Enabler.** The closed-form ancestral sampler (exact per-eigenmode OU posterior,
no EM stability floor) makes per-epoch sampled validation affordable (~10-25 s
vs ~7 min for an EM sweep). Sampled CRPS in raw units is also prior-agnostic,
resolving the kron-vs-iso loss incomparability at the selection level.

**Methodology untouched:** forward SDE, FCPS/ULS schedules, γ=0.8, ρ=0.1, S=7,
x0-prediction, UGnet, EM as the GAD-parity headline sampler for reported evals.

## 2026-07-08 — round-2 recipe additions (submitted as EXP=20260708_r2)

- **SNR loss default** (GAD's rate-weighted MSE): only arm whose sampled CRPS
  improved through training; best metrla full-val CRPS 0.0451 vs 0.0483 (mse).
- **Past-residual conditioning** (`--cond-past`): conditioning = [R_prev | Ŷ_z];
  the previous window's residuals are fully observed at forecast time (no
  leakage) and carry the volatility signal (HySRD-Net's only conditioning source).
- **Masked-input zeroing**: metrla holds residuals of ≈ −11σ at offline-sensor
  coordinates (7%); they entered the network input and graph convs smeared them
  into scored neighbors. Zeroed at input; loss mask unchanged; prior fit was
  already masked.
- **Per-node normalization arm** (`--norm pernode`): per-sensor train std
  (window-level residual std varies ~3×); HySRD's ablation shows global-vs-node
  normalization costs 5-7% CRPS. Run as an arm, not a default (changes the prior).
- **Ancestral sampler** (`--sampler ancestral`): DDPM-ancestral analogue, exact
  for the linear per-mode OU SDE (bridge identity verified numerically); no
  θ_max·S stability floor → ~50-100 steps replace 1500-3500 EM steps. Final-val
  cross-validates it against EM on every run (sweep + full-val, both checkpoints).
