# Evaluation suite

Measures whether the full pipeline — GWN backbone + diffusion residual model —
produces a **calibrated predictive distribution** that beats the backbone and
simple baselines, and separately checks that the **diffusion machinery is
correct**. The method itself is documented in `docs/method_walkthrough.md`.

Everything is `Y_sample = Ŷ + R_sample`: we add the diffusion residual samples
back onto the GWN point forecast, then score the resulting distribution against
the truth.

## Layout

| File | Role |
|---|---|
| `src/eval/metrics.py` | masked metric kernels + streaming `MetricAccumulator` |
| `src/eval/samplers.py` | `DiffusionSampler`, `PriorOnlySampler`, `PerHorizonGaussianSampler` |
| `src/eval/io.py` | `load_eval_checkpoint` (rebuilds the model from `config.json`) |
| `scripts/eval/generate_samples.py` | **GPU** — sample + reduce to metrics (streaming) |
| `scripts/eval/evaluate_forecasts.py` | **CPU** — baselines + GWN point + comparison tables |
| `scripts/eval/diffusion_diagnostics.py` | correctness checks (oracle, forward marginal, sample quality) |
| `scripts/eval/make_figures.py` | all figures |
| `tests/test_eval_metrics.py` | metric kernels vs closed forms |

Why two scripts? A full split at K=100 samples is tens of GB, so we never store
raw samples. `generate_samples.py` runs the expensive reverse sampler and
**reduces each batch to metric sufficient-statistics on the fly**, persisting
only the finalized metrics and a small *sample bank* (12 windows kept in full,
for the fan-chart / spatial figures). `evaluate_forecasts.py` then adds the cheap
analytic baselines and assembles the tables — it mirrors the diffusion run's
window count and K so every method is scored on identical data.

## Running it

```bash
# 1. GPU: diffusion samples + metrics (DAIC, or locally for a smoke check)
python scripts/eval/generate_samples.py \
    --ckpt-dir results/checkpoints/diffusion/<run_id> --split test \
    --n-samples 100 --n-steps 100 --eta 1.0
#    add  --smoke  for a fast plumbing run (meaningless numbers)

# 2. CPU: baselines + GWN point + comparison tables
python scripts/eval/evaluate_forecasts.py --dataset metrla --split test

# 3. correctness diagnostics (oracle + forward marginal need no checkpoint;
#    pass --ckpt-dir to also score generated-sample quality)
python scripts/eval/diffusion_diagnostics.py --dataset metrla --ckpt-dir <run_id>

# 4. figures
python scripts/eval/make_figures.py --dataset metrla --split test
```

On DAIC a single `submit_generate_samples.sbatch` job runs all four steps — see
`scripts/daic/README.md`.

Outputs land in `results/eval/{dataset}/`:
`metrics_{split}.json` (diffusion), `metrics_all_{split}.json` (every method),
`comparison_{split}.md` / `.csv`, `diagnostics.json`, `samples_{split}.npz`,
and `figures/`.

## The metrics

`η` (reverse stochasticity) defaults to **1.0** — the exact ancestral sampler —
which is what gives calibrated spread. `η=0` is deterministic and collapses the
distribution; use it only for repeatable trajectories.

| Metric | What it measures | Good |
|---|---|---|
| **CRPS** | per-cell distributional accuracy (generalises MAE to distributions) | ↓ |
| **Energy score** | multivariate CRPS over the whole (sensor×horizon) field — sensitive to the joint spatial/temporal structure the Kronecker prior provides | ↓ |
| **Variogram score** | pairwise dependence calibration (order p=0.5, over a logged random pair subset) | ↓ |
| **PIT histogram** | rank of the truth in the ensemble — flat ⇒ calibrated | flat |
| **ACE** | mean \|empirical − nominal\| coverage over the reliability curve | ↓ |
| **PINAW90** | 90% interval width ÷ data range — sharpness | ↓ at fixed coverage |
| **PredStd** | mean predictive standard deviation | — (context) |
| **MAE / RMSE / MAPE** | point accuracy of the ensemble **mean** vs GWN | ↓ |

Calibration vs sharpness is a trade-off: read **ACE/PIT** (is the uncertainty
honest?) together with **PINAW/PredStd** (is it tight?). A point forecast like
GWN has no distribution, so its CRPS/energy/coverage cells show `—`.

### Baselines
- **GWN (point)** — the backbone alone; the bar everything must clear on MAE/RMSE.
- **Prior-only Gaussian** — `Ŷ +` a draw from the Kronecker prior, *no denoiser*.
  Isolates how much the *learned* model adds over the structured covariance.
- **Per-horizon Gaussian** — `Ŷ ±` independent per-horizon noise; the simplest
  uncertainty model (no spatial/temporal correlation).

## The figures (`results/eval/{dataset}/figures/`)

| Figure | Reads it as |
|---|---|
| `reliability.png` | empirical vs nominal coverage; on the diagonal = calibrated |
| `pit_histogram.png` | diffusion PIT; flat = calibrated, ∪/∩ shapes = under/over-dispersed |
| `crps_vs_horizon.png` | CRPS per step, all methods — does diffusion win, and where? |
| `sharpness.png` | interval width vs nominal coverage per method |
| `fan_chart.png` | the intuitive one — truth, GWN point, and diffusion median + 50/90% bands for one sensor over the horizon |
| `spatial_std.png` | per-sensor predictive uncertainty at one window/horizon |
| `reverse_trajectory.png` | ‖R_t‖ and ‖R̂₀‖ along the reverse sampler — convergence sanity |
| `generated_quality.png` | generated vs real residual variance + horizon-correlation R_H |

## Correctness diagnostics (`diagnostics.json`)

1. **Oracle reconstruction** — feeding the reverse sampler the true R₀ must
   reconstruct it to `< 1e-3` even under the stiff production schedule (PASS/FAIL).
2. **Forward-marginal exactness** — Monte-Carlo draws from the forward sampler
   must match the closed-form per-mode mean (z-score) and variance (ratio).
   *This check found and now guards against the eigenbasis rotate-back bug in
   `forward_fcps.sample_R_t_given_R0_torch` (see `tests/test_forward_fcps.py`).*
3. **Generated-sample quality** (with `--ckpt-dir`) — residuals from the trained
   model vs real held-out residuals: per-horizon variance and R_H match.
   Prior-vs-empirical fidelity is covered separately by
   `scripts/section3/exp34_36_reconcile_sample_psd.py`.
