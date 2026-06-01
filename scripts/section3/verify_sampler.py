"""Numerical verification of the reverse-sampler claims (reviewer follow-up).

Three decisive checks, all on a scalar Ornstein–Uhlenbeck eigenmode (the
reverse sampler acts independently per mode, so a scalar test validates the
per-mode update directly). Pure numpy, runs in <1s.

A. Step-count invariance of the η=1 exact ancestral sampler.
   If our update is the EXACT reverse transition, the output distribution is
   independent of n_steps. We feed the analytic oracle posterior mean
   E[c0|c_t] (a Gaussian computation, no network) and check Var(c_0) ≈ v0 for
   n_steps ∈ {5, 20, 100, 1000}. Constant ⇒ exact.

B. EM convergence + blow-up.
   Euler–Maruyama on the same reverse SDE, same oracle score, must (i) converge
   to v0 only as δ→0 and (ii) diverge for stiff c·θ at coarse δ. Demonstrates
   EM is the approximation and the exact update is the target.

C. η=1 vs EM-limit agreement.
   For a non-stiff mode where EM is stable, the fine-δ EM output variance must
   match the exact sampler's. If they agree, "η=1 = ancestral limit of the
   reverse SDE" is confirmed.

D. c̄-distribution skew under t ~ Uniform.
   Histogram of c̄(t) for t~U(0,1) at α ∈ {1.5, 3, 7}: shows how aggressively
   α concentrates training mass near c̄≈0 (the reviewer's bimodality concern).

Usage:
    python scripts/section3/verify_sampler.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np


# ── forward marginal factors (scalar mode), matching forward_fcps.py ──────────
def H_of(theta: float, c_bar: float) -> float:
    return np.exp(-theta * c_bar)


def s2_of(theta: float, c_bar: float) -> float:
    return (1.0 - np.exp(-2.0 * theta * c_bar)) / theta


def fcps_c_bar(t, c_min=0.05, c_0=20000.0, alpha=7.0, T=1.0):
    u = np.asarray(t) / T
    return c_min * np.asarray(t) + (c_0 - c_min * T) * np.power(u, alpha + 1)


def fcps_c(t, c_min=0.05, c_0=20000.0, alpha=7.0, T=1.0):
    u = np.asarray(t) / T
    k = (c_0 - c_min * T) * (alpha + 1) / T
    return c_min + k * np.power(u, alpha)


def oracle_C0(x_cur, H_cur, s2_cur, v0):
    """Analytic E[c0 | c_t] for data c0 ~ N(0, v0), c_t = H·c0 + N(0, s2)."""
    return (H_cur * v0 / (H_cur ** 2 * v0 + s2_cur)) * x_cur


# ── A/C: exact η-ancestral sampler (mirrors reverse.py:158-189, scalar) ───────
def _cbar_uniform_ts(n_steps, c0_kwargs):
    """Time grid spaced uniformly in c̄ (finer near t=T where c̄ ramps)."""
    T = c0_kwargs["T"]
    fine = np.linspace(0.0, T, 200_001)
    cb = fcps_c_bar(fine, **c0_kwargs)
    targets = np.linspace(cb[-1], 0.0, n_steps + 1)        # descending in c̄
    return np.interp(targets[::-1], cb, fine)[::-1]          # → descending in t


def exact_reverse(theta, v0, n_steps, n_samples, eta, rng, c0_kwargs, grid="t"):
    T = c0_kwargs["T"]
    ts = (np.linspace(T, 0.0, n_steps + 1) if grid == "t"
          else _cbar_uniform_ts(n_steps, c0_kwargs))
    # start from the TRUE marginal at t=T
    c_bar_T = float(fcps_c_bar(T, **c0_kwargs))
    H_T, s2_T = H_of(theta, c_bar_T), s2_of(theta, c_bar_T)
    var_T = H_T ** 2 * v0 + s2_T
    x = rng.standard_normal(n_samples) * np.sqrt(var_T)

    for i in range(n_steps):
        c_bar_cur = float(fcps_c_bar(ts[i], **c0_kwargs))
        c_bar_next = float(fcps_c_bar(ts[i + 1], **c0_kwargs))
        H_cur, s2_cur = H_of(theta, c_bar_cur), s2_of(theta, c_bar_cur)
        H_next, s2_next = H_of(theta, c_bar_next), s2_of(theta, c_bar_next)

        C0_hat = oracle_C0(x, H_cur, s2_cur, v0)
        eps_hat = (x - H_cur * C0_hat) / np.sqrt(max(s2_cur, 1e-20))

        ratio = (s2_next / max(s2_cur, 1e-20)) * (H_cur / max(H_next, 1e-20)) ** 2
        sig_tilde2 = eta ** 2 * s2_next * max(1.0 - ratio, 0.0)
        drift_var = max(s2_next - sig_tilde2, 0.0)
        x = H_next * C0_hat + np.sqrt(drift_var) * eps_hat
        if eta > 0:
            x = x + np.sqrt(sig_tilde2) * rng.standard_normal(n_samples)
    return x


# ── B/C: Euler–Maruyama on the reverse SDE, same oracle score ─────────────────
def em_reverse(theta, v0, n_steps, n_samples, rng, c0_kwargs):
    """Reverse SDE: dx = [cθx - 2c·x/Var_t]·δ + sqrt(2cδ)·z, stepping t down.
    Uses the analytic marginal score (-x/Var_t), so any error is pure
    discretization, not score error."""
    T = c0_kwargs["T"]
    ts = np.linspace(T, 0.0, n_steps + 1)
    c_bar_T = float(fcps_c_bar(T, **c0_kwargs))
    H_T, s2_T = H_of(theta, c_bar_T), s2_of(theta, c_bar_T)
    var_T = H_T ** 2 * v0 + s2_T
    x = rng.standard_normal(n_samples) * np.sqrt(var_T)

    for i in range(n_steps):
        t_cur = ts[i]
        delta = ts[i] - ts[i + 1]
        c = float(fcps_c(t_cur, **c0_kwargs))
        c_bar = float(fcps_c_bar(t_cur, **c0_kwargs))
        var_t = H_of(theta, c_bar) ** 2 * v0 + s2_of(theta, c_bar)
        drift = c * theta * x - 2.0 * c * x / var_t
        x = x + drift * delta + np.sqrt(2.0 * c * delta) * rng.standard_normal(n_samples)
        if not np.all(np.isfinite(x)):
            return x  # blew up
    return x


def main() -> int:
    rng = np.random.default_rng(0)
    N_S = 200_000
    print("=" * 72)
    print("A. Step-count invariance of η=1 exact ancestral sampler")
    print("   (mild mode θ=1, data var v0=2.0, c_0 chosen so θ·c_0=5)")
    print("=" * 72)
    theta = 1.0
    kw = dict(c_min=0.05, c_0=5.0, alpha=7.0, T=1.0)
    print("   case (i) data var v0=2.0 ≠ stationary 1/θ=1.0  (mismatch)")
    for n in (5, 20, 100, 1000):
        x0 = exact_reverse(theta, 2.0, n, N_S, eta=1.0, rng=rng, c0_kwargs=kw)
        print(f"   n_steps={n:5d}   Var(c_0)={x0.var():.4f}   (target 2.0)")
    print("   case (ii) data var v0=1.0 == stationary 1/θ  (matched)")
    for n in (5, 20, 100, 1000):
        x0 = exact_reverse(theta, 1.0, n, N_S, eta=1.0, rng=rng, c0_kwargs=kw)
        print(f"   n_steps={n:5d}   Var(c_0)={x0.var():.4f}   (target 1.0)")

    print()
    print("=" * 72)
    print("B. EM blow-up on a STIFF mode (θ=50, c_0=20000 → c·θ huge)")
    print("=" * 72)
    theta_stiff = 50.0
    kw_stiff = dict(c_min=0.05, c_0=20000.0, alpha=7.0, T=1.0)
    c_T = fcps_c(1.0, **kw_stiff)
    print(f"   c(T)·θ = {c_T * theta_stiff:.3e}   (EM needs c·θ·δ ≲ 2)")
    x0_exact = exact_reverse(theta_stiff, 2.0 / theta_stiff, 100, N_S, eta=1.0,
                             rng=rng, c0_kwargs=kw_stiff)
    print(f"   exact η=1, n=100:  Var={x0_exact.var():.4e}  "
          f"target={2.0/theta_stiff:.4e}  finite={np.all(np.isfinite(x0_exact))}")
    for n in (100, 1000, 10_000):
        x0 = em_reverse(theta_stiff, 2.0 / theta_stiff, n, 20_000, rng, kw_stiff)
        ok = np.all(np.isfinite(x0))
        v = x0.var() if ok else float("nan")
        need = c_T * theta_stiff / 2.0
        print(f"   EM n={n:6d}:  finite={ok!s:5}  Var={v:.4e}  "
              f"(stable n≈{need:.0f})")

    print()
    print("=" * 72)
    print("C. η=1 exact vs fine-δ EM agreement (non-stiff θ=2, must match)")
    print("=" * 72)
    theta_c, v0_c = 2.0, 1.5
    kw_c = dict(c_min=0.05, c_0=2.5, alpha=7.0, T=1.0)  # θ·c_0=5, gentle
    xe = exact_reverse(theta_c, v0_c, 100, N_S, eta=1.0, rng=rng, c0_kwargs=kw_c)
    xm = em_reverse(theta_c, v0_c, 200_000, 50_000, rng, kw_c)
    print(f"   target v0={v0_c}")
    print(f"   exact η=1 (n=100):     Var={xe.var():.4f}")
    print(f"   EM fine (n=200000):    Var={xm.var():.4f}")
    print(f"   relative gap: {abs(xe.var() - xm.var()) / v0_c:.2%}")

    print()
    print("=" * 72)
    print("D. c̄(t) distribution under t ~ Uniform(0,1)  [α concentration]")
    print("=" * 72)
    t = rng.uniform(0, 1, 500_000)
    for alpha in (1.5, 3.0, 7.0):
        cb = fcps_c_bar(t, c_min=0.05, c_0=20000.0, alpha=alpha, T=1.0)
        frac = cb / 20000.0  # fraction of total diffusion reached
        q = np.quantile(frac, [0.5, 0.9, 0.99])
        below_1pct = float(np.mean(frac < 0.01))
        print(f"   α={alpha:4.1f}:  median c̄/c_0={q[0]:.4f}  "
              f"p90={q[1]:.3f}  p99={q[2]:.3f}  "
              f"P(c̄<1% of c_0)={below_1pct:.1%}")

    print()
    print("=" * 72)
    print("E. Reverse grid spacing: t-uniform vs c̄-uniform (n=100, α=7)")
    print("   does spacing steps in c̄ cut the finite-step variance bias?")
    print("=" * 72)
    theta_e, v0_e = 1.0, 1.0
    kw_e = dict(c_min=0.05, c_0=5.0, alpha=7.0, T=1.0)
    xt = exact_reverse(theta_e, v0_e, 100, N_S, 1.0, rng, kw_e, grid="t")
    xc = exact_reverse(theta_e, v0_e, 100, N_S, 1.0, rng, kw_e, grid="cbar")
    print(f"   target v0={v0_e}")
    print(f"   t-uniform grid:   Var={xt.var():.4f}  (bias {(xt.var()-v0_e)/v0_e:+.1%})")
    print(f"   c̄-uniform grid:   Var={xc.var():.4f}  (bias {(xc.var()-v0_e)/v0_e:+.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
