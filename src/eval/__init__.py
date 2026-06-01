"""Evaluation suite for the diffusion forecasting pipeline.

Two layers, mirroring the questions we need to answer for the supervisor:

  * Headline forecasting quality — does ``Y_sample = Ŷ + R_sample`` beat the
    GWN backbone and simple baselines? Scored with CRPS, energy / variogram
    score, calibration (PIT, reliability), sharpness, and point metrics.
  * Diffusion correctness — oracle reconstruction, forward-marginal exactness,
    and whether generated residuals reproduce real residual statistics.

The shared, unit-tested core lives here (`metrics`, `samplers`, `io`); the
``scripts/eval/`` scripts wire it together. See ``metrics.MetricAccumulator``
for the streaming reduction that keeps memory bounded (a full split × K samples
would be tens of GB, so we never persist raw samples).
"""
