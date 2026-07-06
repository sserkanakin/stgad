"""forward-process analysis, GAD-style (rozada et al. 2025).

verifies our forward sde transports clean residuals to the correct stationary
distribution N(0, Σ_T ⊗ Σ_S). we reduce each residual field to GAD's three graph
signal metrics — total variation (xᵀLx), spectral centroid, degree correlation —
and, at a sweep of noise levels c̄, compare the metric distribution of the
forward-noised residuals to the same metrics on samples drawn straight from the
prior. a correct forward makes every distance shrink to ~0 as c̄ → c̄_hi.

the H horizons are treated as the feature dim D that gad's signal_metrics averages
over. we report the table under two graph operators:
  combinatorial  L = D - A            (gad-faithful; what exp_gasde.py measures with)
  normalized     L_sym (self-loops stripped, our prior's own operator)

everything runs on cached residuals — no gpu / daic needed.
"""
from __future__ import annotations

import numpy as np
import torch

from src.diffusion.prior import load_prior, _load_residuals_train
from src.diffusion import forward_fcps as ff
from src.data.adjacency import load_adjacency, compute_normalized_laplacian
from src.eval.gad_signal_metrics import (
    SignalMetricsEvaluator,
    FeatureConfig,
    DistanceConfig,
    extract_feature_set,
    _eig_L,
)

METRICS = ("tv", "sc", "degcorr")            # gad's three graph signal metrics
_FEAT = FeatureConfig(metrics=METRICS)
_EVAL = SignalMetricsEvaluator(feature_cfg=_FEAT, distance_cfg=DistanceConfig())


def build_operators(dataset: str) -> dict:
    """the two graph operators the metrics are measured with. each is
    (L [N,N], eig=(evals,evecs), degrees [N]) ready for gad's signal_metrics."""
    adj = load_adjacency(dataset)                       # may be asymmetric (metr-la)
    A = 0.5 * (adj + adj.T)                             # gad symmetrises before L
    deg = A.sum(1)                                      # combinatorial degree (operator-free)
    deg_t = torch.as_tensor(deg, dtype=torch.float64)

    L_comb = np.diag(deg) - A
    L_norm = compute_normalized_laplacian(load_adjacency(dataset))  # strips self-loops

    out = {}
    for name, L in (("combinatorial", L_comb), ("normalized", L_norm)):
        Lt = torch.as_tensor(L, dtype=torch.float64)
        out[name] = {"L": Lt, "eig": _eig_L(Lt), "deg": deg_t}
    return out


def _features(X: np.ndarray, op: dict) -> dict:
    """gad signal-metric scalars for a batch of residual fields X (B,N,H)."""
    Xt = torch.as_tensor(X, dtype=torch.float64)
    return extract_feature_set(Xt, op["L"], degrees=op["deg"], eig=op["eig"], cfg=_FEAT)


def run(dataset: str, seed: int, backbone: str, n_samples: int, n_steps: int) -> dict:
    prior = load_prior(dataset, prior="kronecker", seed=seed,
                       spatial="structured", backbone=backbone)
    R, mask = _load_residuals_train(dataset, seed=seed, backbone=backbone)  # unit space (B,N,H)

    rng = np.random.default_rng(0)
    idx = rng.choice(R.shape[0], size=min(n_samples, R.shape[0]), replace=False)
    R0 = R[idx]                                         # clean residuals, unit space
    B = R0.shape[0]

    c_lo, c_hi = ff.noise_level_bounds(prior)
    grid = ff.reverse_c_bar_grid(c_lo, c_hi, n_steps)[::-1]   # ascending c̄_lo → c̄_hi

    ops = build_operators(dataset)
    R_prior = ff.sample_prior((B,), prior, rng)         # the stationary target, same B

    # forward-noised residuals at each level (shared across operators)
    R_at = {"data": R0}
    for cb in grid:
        R_at[float(cb)] = ff.sample_R_t_from_c_bar(R0, float(cb), prior, rng)

    result = {
        "dataset": dataset, "backbone": backbone, "seed": seed,
        "n_samples": B, "c_lo": c_lo, "c_hi": c_hi,
        "grid": [float(c) for c in grid], "operators": {},
    }

    for op_name, op in ops.items():
        F_prior = _features(R_prior, op)
        means = {}                                      # metric means per level (trajectory)
        dists = {}                                      # distance-to-prior per level
        # data + prior reference means
        for lvl, X in R_at.items():
            F = _features(X, op)
            means[str(lvl)] = {m: float(F[m].mean()) for m in METRICS}
        means["prior"] = {m: float(F_prior[m].mean()) for m in METRICS}
        # distances forward(c̄) vs prior
        for cb in grid:
            F = _features(R_at[float(cb)], op)
            cmp = _EVAL.compare(F, F_prior)             # metric -> {mmd,wasserstein,gaussian_kl,ks}
            dists[str(float(cb))] = cmp
        result["operators"][op_name] = {"means": means, "distances": dists}

    return result


# ─── pretty printing ─────────────────────────────────────────────────────────


def _fmt(x: float) -> str:
    return f"{x:9.3g}"


def print_report(res: dict) -> None:
    print(f"\n{'='*78}\nforward analysis — {res['dataset']} ({res['backbone']}, seed{res['seed']}), "
          f"B={res['n_samples']}\nc̄ range [{res['c_lo']:.3g}, {res['c_hi']:.3g}]\n{'='*78}")
    grid = res["grid"]
    for op_name, op in res["operators"].items():
        print(f"\n### operator: {op_name}")
        # trajectory: metric means data → c̄ levels → prior
        print("\n metric means (should walk from `data` to `prior`):")
        print(f"  {'level':>12} | " + " ".join(f"{m:>9}" for m in METRICS))
        order = ["data"] + [str(c) for c in grid] + ["prior"]
        for lvl in order:
            label = lvl if lvl in ("data", "prior") else f"c̄={float(lvl):.3g}"
            row = op["means"][lvl]
            print(f"  {label:>12} | " + " ".join(_fmt(row[m]) for m in METRICS))
        # distances to prior (wasserstein-1, the headline; → 0 means forward is right)
        print("\n wasserstein-1 distance to prior (→ 0 as c̄→c̄_hi ⇒ forward correct):")
        print(f"  {'c̄':>12} | " + " ".join(f"{m:>9}" for m in METRICS))
        for c in grid:
            d = op["distances"][str(float(c))]
            print(f"  {c:12.3g} | " + " ".join(_fmt(d[m]["wasserstein"]) for m in METRICS))
