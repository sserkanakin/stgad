# Forward-process analysis (GAD-style) — how to read it

Checks that our graph-aware forward SDE transports a clean residual field `R₀` to the correct
stationary prior `N(0, Σ_T ⊗ Σ_S)`. Method borrowed from the GAD reference
(Rozada et al. 2025, `signal_metrics.py`), vendored verbatim at `src/eval/gad_signal_metrics.py`.

## What we compute

Reduce each residual field (shape `(N, H)`, H horizons treated as the feature dim `D`) to graph
**signal metrics**, then compare *distributions* of those scalars:

- **Total Variation** `TV = xᵀLx` — graph roughness.
- **Spectral Centroid** `SC = Σλᵢ|uᵢ|² / Σ|uᵢ|²` — where spectral energy sits.
- **Degree Correlation** `corr(deg, x)` — reported but ≈0 (residuals carry no degree signal), so it is
  not a useful convergence curve; its absolute distance is still tiny (≤0.006).

For a sweep of noise levels `c̄` (geometric, from a ~4% floor to full mixing) we forward-sample
`R_c̄ = sample_R_t_from_c_bar(R₀, c̄)` and measure the Wasserstein-1 / MMD / KL / KS distance between
its metric distribution and the same metrics on `sample_prior(...)`. A correct forward makes every
distance → 0 as `c̄ → c̄_hi`.

Two graph operators, both reported (same verdict):
- **combinatorial** `L = D − A` (symmetrized) — GAD-faithful, what `exp_gasde.py` measures with.
- **normalized** `L_sym` (self-loops stripped) — our prior's own operator, the self-consistency test.

## How to run

```
# per-dataset json + printed table
python scripts/forward_analysis.py --dataset metrla --n-samples 512 --n-steps 7
# cross-dataset summary table (reads the jsons)
python scripts/forward_analysis_summary.py
# the three figures (caches per-sample features under results/forward_analysis/feats/)
python scripts/forward_analysis_plots.py
```

Everything runs locally on cached residuals (`results/residuals/<ds>/gwn/seed42/`) — no GPU/DAIC.
Outputs land in `results/forward_analysis/` (jsons, `figs/`, `feats/`).

## The result — the forward is correct

All six datasets: mean TV lands on the prior to within **0.04–0.47%**, W1(TV) collapses **~100–260×**,
W1(SC) **~10–40×**. Figures:

- `figs/convergence_normalized.png` — W1-to-prior (fraction of start) → 0, per dataset, TV + SC.
- `figs/morph_tv_normalized.png` — the TV distribution sliding off the data onto the prior.
- `figs/snapshot_metrla.png` — one residual field dissolving into a prior draw.

One honest wrinkle worth keeping: the *direction* of TV motion is dataset-dependent — metr-la's residuals
are **rougher** than the prior (TV falls), pems-bay's are **smoother** (TV rises). Both converge, but it is a
real signal about how graph-structured each dataset's residual is (cf. `spectral_analysis`).
