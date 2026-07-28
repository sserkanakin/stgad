"""pre-flight gate for the GTCNN product-graph diffusion campaign.

Nothing gets submitted unless every check here passes. Per (dataset, product form) we
verify the three things that can silently be wrong:

  A. the OPERATOR is what we claim
     A1 each factor is a valid simple graph (symmetric, non-negative, no self-loops)
     A2 each factor is a SINGLE connected component (after graph repair)
     A3 the product GSO reproduces the product GRAPH's edge set (GTCNN Sec. 2.1)
     A4 the spectrum is a valid heat spectrum: lam in [0,2], theta = lam+gamma > 0,
        theta_min == gamma exactly (so every arm shares one mixing number)

  B. the SCHEDULE mixes and the solver is stable
     B1 GAD's slow-mode mixing condition 2*c_0*theta_min >= 5
     B2 the EM stability floor max(200, theta_max*c_0) is reported so the step grid
        can be chosen honestly (the ancestral sampler has no floor)

  C. the FORWARD PROCESS actually lands on the stationary distribution we designed
     C1 stationary draws have per-mode variance sigma^2/theta        (the design target)
     C2 running the forward SDE from real data to t=T reaches that same second moment
     C3 the terminal law is isotropic AFTER whitening by the prior (a real distributional
        check, not just a variance match) -- if the prior were wrong this fails while C1
        still passes

usage:
    python scripts/verify_product_graph.py --forms prod_cartesian prod_strong
    python scripts/verify_product_graph.py --all          # every dataset x every form
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.adjacency import compute_normalized_laplacian, load_adjacency  # noqa: E402
from src.diffusion.forward_fcps import (  # noqa: E402
    FCPSSchedule, _rotate_to_eigenbasis, sample_prior, sample_R_t_given_R0,
)


from src.diffusion.prior import PRODUCT_FORMS, load_prior  # noqa: E402
from src.diffusion.reverse import min_em_steps  # noqa: E402
from src.utils import paths  # noqa: E402


def _rot(X, prior):
    """BLAS rotation to the eigenbasis: V_S^T X V_T, the same map as the production
    _rotate_to_eigenbasis but via matmul instead of a naive einsum (which is ~50x slower
    and makes this gate unusable at campaign scale). check A0 asserts the two agree to
    machine precision before any bulk use, so the production path is still verified."""
    return np.matmul(np.matmul(prior.V_S.T, X), prior.V_T)

DATASETS = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]
# the campaign's five arms: the isotropic control + the four product-graph forms
ALL_FORMS = ["iso"] + list(PRODUCT_FORMS)
OK, NO = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
# gamma is the LOW value the heat cascade needs (GAD Fig.1 regime), NOT the residual
# arm's 0.8. c_0 is DERIVED by check B3, not taken from GAD's rule of thumb: the
# requirement is c_0 >= ln(4*r/VAR_TOL)/(2*gamma) = 5*ln(80*r) at gamma=0.1, where r is
# the raw signal's worst data/target per-mode variance ratio (~90-280 measured). that
# gives ~45-55; 100 clears it with room and still keeps the EM floor at max(200, 2.1*c_0).
GAMMA, C_0 = 0.1, 100.0
MIX_MIN = 5.0                 # GAD's slow-mode mixing condition
VAR_TOL = 0.05                # 5% relative error on second moments
WHITEN_TOL = 0.15             # 15% on the whitened-variance spread (finite-sample)


def _factors(dataset, H=12):
    """the two factor graphs, as simple graphs, plus their normalized-Laplacian spectra."""
    A_S = np.asarray(load_adjacency(dataset), dtype=np.float64).copy()
    np.fill_diagonal(A_S, 0.0)
    A_S = (A_S + A_S.T) / 2.0
    A_T = np.zeros((H, H))
    i = np.arange(H - 1)
    A_T[i, i + 1] = A_T[i + 1, i] = 1.0
    return A_S, A_T


def _n_components(A):
    """BFS component count. the spectral version (zero eigenvalues of the Laplacian) is
    equivalent but costs an O(n^3) eigvalsh, which is minutes on the NT x NT product."""
    n = A.shape[0]
    nbr = [np.nonzero(A[i])[0] for i in range(n)]
    seen = np.zeros(n, bool)
    comps = 0
    for s in range(n):
        if seen[s]:
            continue
        comps += 1
        seen[s] = True
        stack = [s]
        while stack:
            for v in nbr[stack.pop()]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return comps


def _product_adj(form, A_S, A_T):
    """the product GRAPH's adjacency on the NT product node set."""
    N, H = A_S.shape[0], A_T.shape[0]
    I_N, I_T = np.eye(N), np.eye(H)
    if form == "prod_cartesian":
        return np.kron(A_T, I_N) + np.kron(I_T, A_S)
    if form == "prod_strong":
        return np.kron(A_T, A_S) + np.kron(A_T, I_N) + np.kron(I_T, A_S)
    if form == "prod_tensor":
        return np.kron(A_T, A_S)
    return None                                   # prod_param is a fitted spectrum


def _product_gso(form, L_S, L_T):
    """the GSO GTCNN actually builds, from the FACTOR shift operators (their Sec. 2.1)."""
    N, H = L_S.shape[0], L_T.shape[0]
    I_N, I_T = np.eye(N), np.eye(H)
    if form == "prod_cartesian":
        return np.kron(L_T, I_N) + np.kron(I_T, L_S)
    if form == "prod_strong":
        return np.kron(L_T, L_S) + np.kron(L_T, I_N) + np.kron(I_T, L_S)
    if form == "prod_tensor":
        return np.kron(L_T, L_S)
    return None


def check(dataset, form, seed=42, backbone="raw", n_win=512, rng_seed=0):
    res, notes = {}, []
    is_iso = form == "iso"
    A_S, A_T = _factors(dataset)
    L_S = compute_normalized_laplacian(A_S.copy())
    L_T = compute_normalized_laplacian(A_T.copy())

    # ── A. the operator ───────────────────────────────────────────────────────
    res["A1 factors are simple graphs"] = all(
        np.allclose(A, A.T) and (A >= 0).all() and np.allclose(np.diag(A), 0.0)
        for A in (A_S, A_T))
    res["A2 factors singly connected"] = (_n_components(A_S) == 1 and _n_components(A_T) == 1)

    A_P, S_P = (None, None) if is_iso else (_product_adj(form, A_S, A_T),
                                            _product_gso(form, L_S, L_T))
    if is_iso:
        notes.append("A3 skipped: the isotropic control has no graph (theta is constant)")
    elif A_P is None:
        notes.append("A3 skipped: prod_param is a FITTED spectrum, not a fixed product graph")
    else:
        off = lambda M: np.abs(M - np.diag(np.diag(M))) > 1e-12    # noqa: E731
        match = bool(np.array_equal(off(S_P), off(A_P) > 0))
        # prod_tensor is a NEGATIVE CONTROL and fails this by construction, so record it
        # as a known structural property rather than a blocking failure: the Laplacian's
        # nonzero diagonal makes L_T (x) L_S connect (i,t)-(j,t) and (i,t)-(i,s) pairs
        # that the tensor product GRAPH does not have. GTCNN's eq. is still applied
        # literally -- the mismatch is with the graph reading of it, which is the point.
        if form == "prod_tensor":
            extra = int((off(S_P) & ~(off(A_P) > 0)).sum())
            notes.append(f"A3 (informational, expected): tensor GSO has {extra} edges the "
                         f"tensor product graph lacks — Laplacian diagonal leaks. "
                         f"KNOWN: this arm is the negative control.")
        else:
            res["A3 GSO == product graph edges"] = match
        res["A2b product singly connected"] = (_n_components(A_P) == 1)

    prior = (load_prior(dataset, prior="isotropic", seed=seed, backbone=backbone) if is_iso
             else load_prior(dataset, prior="kronecker", seed=seed, backbone=backbone,
                             temporal=form, gamma=GAMMA))
    th = prior.theta
    tmin, tmax = float(th.min()), float(th.max())
    if is_iso:
        # the control's only structural requirement is that theta is genuinely constant
        res["A4 theta is constant (no structure)"] = bool(tmax - tmin < 1e-12 and tmin > 0)
    else:
        res["A4 theta = lam+gamma in [g,2+g]"] = bool(
            abs(tmin - GAMMA) < 1e-9 and abs(tmax - (2.0 + GAMMA)) < 1e-6 and tmin > 0)
    # how many modes sit at the very bottom -- the tensor form pins N+H-1 of them there,
    # which means the prior CANNOT distinguish them (all get maximum variance)
    n_deg = int((th < GAMMA + 1e-9).sum())
    notes.append(f"modes at theta_min: {n_deg}/{th.size} "
                 f"({100*n_deg/th.size:.1f}%)" + ("  <-- DEGENERATE" if n_deg > 1 else ""))

    # ── B. schedule + solver ──────────────────────────────────────────────────
    sched = FCPSSchedule(c_0=C_0)
    mix = 2.0 * C_0 * tmin
    res[f"B1 mixing 2*c0*theta_min >= {MIX_MIN:g}"] = bool(mix >= MIX_MIN - 1e-9)
    em_floor = min_em_steps(prior, sched)
    notes.append(f"mixing={mix:.2f}  stiffness={tmax/tmin:.1f}  EM floor={em_floor} "
                 f"(ancestral has none)")

    # ── B3. is c_0 big enough for THIS data? ──────────────────────────────────
    # GAD's 2*c_0*theta_min >= 5 implicitly assumes the per-mode data variance is
    # comparable to sigma^2. on the RAW signal it is not: the graph-DC x time-DC mode is
    # the mean traffic level and carries ~700x the mean per-mode variance, so the slow
    # mode is still far from stationary at mixing=5. derive the c_0 that actually erases
    # it instead of trusting the rule of thumb.
    d = np.load(paths.residual_npz(dataset, "train", seed, backbone))
    R_all = d["residuals"].astype(np.float64)
    R_all = R_all / R_all.std()                            # unit space, as training uses
    # the variance ratio is a population statistic; a few thousand windows pin it to well
    # inside the margin we take on c_0, and rotating all ~24k metrla windows would make
    # this gate too slow to run before every submission.
    _r = np.random.default_rng(rng_seed)
    R_all = R_all[_r.choice(len(R_all), size=min(4096, len(R_all)), replace=False)]
    v_data = (_rot(R_all, prior) ** 2).mean(0)
    target = prior.sigma2_scale / th                       # designed per-mode variance
    need = np.log(np.maximum(v_data / target, 1.0) / (VAR_TOL / 4.0)) / (2.0 * th)
    c0_req = float(np.ceil(max(need.max(), MIX_MIN / (2.0 * tmin))))
    res[f"B3 c_0={C_0:g} erases the slow mode"] = bool(C_0 >= c0_req)
    notes.append(f"data/target variance ratio: max={float((v_data/target).max()):.1f} "
                 f"(mode {np.unravel_index(int((v_data/target).argmax()), th.shape)}) "
                 f"-> c_0 >= {c0_req:g} required, using {C_0:g}")

    # A0: the fast rotation used below must equal the PRODUCTION rotation exactly, so the
    # checks certify the code path training actually uses, not a re-implementation.
    _rng0 = np.random.default_rng(rng_seed)
    _X = _rng0.standard_normal((8, prior.N, prior.H))
    res["A0 fast rotation == production rotation"] = bool(
        np.abs(_rot(_X, prior) - _rotate_to_eigenbasis(_X, prior)).max() < 1e-9)

    # ── C. forward process / stationary law ───────────────────────────────────
    # all three use PER-MODE RELATIVE error (scale-free). the max over ~2000 modes is
    # dominated by Monte-Carlo noise, so the max tolerance is the MC floor 4*sqrt(2/n)
    # plus VAR_TOL, while the median must meet VAR_TOL outright.
    # C1 is a sanity check on sample_prior (already covered tightly by
    # tests/test_prior_form_mixing.py), so it runs at a modest draw count; C2/C3 are the
    # checks that actually bite and they use the real data windows.
    rng = np.random.default_rng(rng_seed)
    n_draw = 4096
    mc = 4.0 * np.sqrt(2.0 / n_draw)

    def _relerr(X):
        v = (_rot(X, prior) ** 2).mean(0)
        r = np.abs(v / target - 1.0)
        return float(np.median(r)), float(r.max())

    m1, x1 = _relerr(sample_prior((n_draw,), prior, rng))                     # C1
    res[f"C1 stationary var == sigma^2/theta ({VAR_TOL:.0%})"] = bool(
        m1 < VAR_TOL and x1 < VAR_TOL + mc)

    idx = rng.choice(len(R_all), size=min(n_win, len(R_all)), replace=False)  # C2
    R_T = sample_R_t_given_R0(R_all[idx], sched.T, prior, sched, rng)
    mc2 = 4.0 * np.sqrt(2.0 / len(idx))
    m2, x2 = _relerr(R_T)
    res[f"C2 forward(t=T) reaches it ({VAR_TOL:.0%})"] = bool(
        m2 < VAR_TOL and x2 < VAR_TOL + mc2)

    # C3 checks the EIGENBASIS, not the spectrum: the prior claims the modes are
    # INDEPENDENT, so the terminal sample's mode-by-mode correlation matrix must be
    # ~diagonal. a wrong rotation leaves the variances right and this one wrong.
    Cm = _rot(R_T, prior).reshape(len(idx), -1)
    sub = rng.choice(Cm.shape[1], size=min(256, Cm.shape[1]), replace=False)
    corr = np.corrcoef(Cm[:, sub].T)
    np.fill_diagonal(corr, 0.0)
    e3 = float(np.abs(corr).max())
    # sample correlation of independent series has sd ~ 1/sqrt(n); allow 5 sd
    res["C3 terminal modes are uncorrelated"] = bool(e3 < max(WHITEN_TOL, 5.0 / np.sqrt(len(idx))))
    notes.append(f"rel err  C1 med/max={m1:.4f}/{x1:.4f}  C2 med/max={m2:.4f}/{x2:.4f}"
                 f"  (MC floor {mc:.4f}/{mc2:.4f})   C3 max|corr|={e3:.4f}")

    allok = all(res.values())
    print(f"\n=== {dataset} · {form} === {'ALL PASS' if allok else 'HAS FAILURES'}")
    for k, v in res.items():
        print(f"   [{OK if v else NO}] {k}")
    for n in notes:
        print(f"    · {n}")
    return allok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--forms", nargs="+", default=ALL_FORMS, choices=ALL_FORMS)
    p.add_argument("--all", action="store_true", help="every dataset x every arm")
    p.add_argument("--backbone", default="raw")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    ds_list = DATASETS if a.all else a.datasets
    forms = ALL_FORMS if a.all else a.forms

    allok = True
    for ds in ds_list:
        for f in forms:
            try:
                allok &= check(ds, f, a.seed, a.backbone)
            except Exception as e:                       # a build failure is a FAIL
                print(f"\n=== {ds} · {f} === ERROR\n   {type(e).__name__}: {e}")
                allok = False
    print(f"\n{'='*60}\n{'ALL CHECKS PASS' if allok else 'SOME CHECKS FAILED'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
