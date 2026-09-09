"""How close is the prior to the residual field, and what does it change? (docs/prior_analysis_plan_20260907.md, §A-§C)

everything is measured on the FRESH residual caches (results/residuals/<ds>/gwn/seed42), in the
unit space the diffusion runs in (per-sensor scale), on train (where the prior is fitted) and
on test (where it is not). five priors are compared at the prior level, no training involved:
    iso          sigma^2 I
    temporal     Sigma_T (x) I
    spatial      I (x) Sigma_S(matched graph)
    road08       Sigma_T (x) Sigma_S(road graph only, gamma = 0.8: the old GAD prior)
    matched      Sigma_T (x) Sigma_S(road + global edge, (beta, gamma) matched)   <- the arm in the campaign
writes results/analysis/prior/<ds>.json and prints a summary.

usage: python scripts/analysis/prior_fidelity.py pems08f [pems08s ...]      (BACKBONE=mlp|gcn|gwn, default gwn)
"""
import sys, os, json, math
sys.path.insert(0, "."); os.environ.setdefault("RESULTS_ROOT", "results")
import numpy as np
from scipy.stats import spearmanr, kurtosis
from scipy.sparse.csgraph import shortest_path
from scipy.cluster.vq import kmeans2
from src.utils import paths
from src.diffusion import prior as P
from src.data.residual_dataset import train_residual_sigma
from src.data.adjacency import load_adjacency, compute_normalized_laplacian

OUT = paths.ROOT / "results" / "analysis" / "prior"; OUT.mkdir(parents=True, exist_ok=True)
BB = os.environ.get("BACKBONE", "gwn")             # the residual field depends on the backbone; the prior is fitted per backbone
rng = np.random.default_rng(0)


def unit_residuals(ds, split):
    """(r, mask, sigma_i): residual / per-sensor train scale, exactly the diffusion's space."""
    d = np.load(paths.residual_npz(ds, split, 42, BB))
    R = d["residuals"].astype(np.float32); M = d["mask"].astype(bool)
    sig = np.asarray(train_residual_sigma(ds, 42, BB, norm="pernode")).reshape(-1)
    return (R / sig.reshape(1, -1, 1).astype(np.float32)), M, sig


def build_priors(ds, R, M):
    """the five priors as (name, V_S, V_T, var (N,H) per-mode variance, gamma, beta). all scaled to
    the same total variance = mean r^2 on train, like load_prior does."""
    var_data = float((R[M] ** 2).mean()); N, H = R.shape[1], R.shape[2]
    delta, R_H, S_T = P._fit_sigma_T(R, M)
    mu_T, V_T = np.linalg.eigh(S_T); mu_T = np.clip(mu_T, 1e-12, None); mu_T = 0.9 * mu_T + 0.1 * mu_T.mean()
    beta, gamma, ratio = P.match_global_edge(ds, 42, BB, "pernode", True, mu_T, var_data)
    mu_Sm, V_Sm, _ = P._spatial_factor(ds, True, beta, gamma)
    mu_S8, V_S8, _ = P._spatial_factor(ds, True, 0.0, 0.8)
    def scaled(mS, mT):
        th = (1.0 / mS)[:, None] / mT[None, :]; th, s2 = P._normalize_scale(th); s2 *= var_data / float((s2 / th).mean())
        return s2 / th
    ones_S, ones_T = np.ones(N), np.ones(H)
    priors = {
        "iso":      (np.eye(N), np.eye(H), np.full((N, H), var_data), 0.0, 0.0),
        "temporal": (np.eye(N), V_T, scaled(ones_S, mu_T), 0.0, 0.0),
        "spatial":  (V_Sm, np.eye(H), scaled(mu_Sm, ones_T), gamma, beta),
        "road08":   (V_S8, V_T, scaled(mu_S8, mu_T), 0.8, 0.0),
        "matched":  (V_Sm, V_T, scaled(mu_Sm, mu_T), gamma, beta),
    }
    return priors, dict(beta=beta, gamma=gamma, match_ratio_train=ratio, delta_h=delta.tolist(), R_H=R_H.tolist(), var_data=var_data)


def mode_second_moment(R, M, V_S, V_T):
    """data second moment per mode (N,H) in the prior's basis; masked entries zeroed."""
    Rz = np.where(M, R, 0.0)
    C = np.einsum("ki,bih,hj->bkj", V_S.T, Rz, V_T)
    return (C ** 2).mean(axis=0)


def spatial_cov(V_S, var):
    """Sigma_S-like (N,N) covariance of the prior at horizon-average variance: V diag(mean_h var) V^T."""
    return (V_S * var.mean(axis=1)) @ V_S.T


def cov_h(V_S, var, h):
    return (V_S * var[:, h]) @ V_S.T


def main(ds):
    Rtr, Mtr, sig = unit_residuals(ds, "train"); Rte, Mte, _ = unit_residuals(ds, "test")
    N, H = Rtr.shape[1], Rtr.shape[2]
    priors, info = build_priors(ds, Rtr, Mtr)
    out = {"dataset": ds, "N": N, "H": H, **info}
    A = load_adjacency(ds, repair=True).astype(float); np.fill_diagonal(A, 0.0)
    L = compute_normalized_laplacian(A); lam, V = np.linalg.eigh(L); o = np.argsort(lam); lam, V = lam[o], V[:, o]
    hops = shortest_path((A > 0).astype(float), unweighted=True, directed=False); hops[~np.isfinite(hops)] = 99

    def corr_matrix(R, M):
        ok = M.reshape(len(M), -1).mean(1) > 0.5
        X = np.where(M, R, 0.0)[ok].transpose(0, 2, 1).reshape(-1, N); X = X - X.mean(0)
        s = np.sqrt((X ** 2).mean(0)); s[s == 0] = 1
        return (X.T @ X) / len(X) / np.outer(s, s), ok
    Ctr, oktr = corr_matrix(Rtr, Mtr); Cte, okte = corr_matrix(Rte, Mte)
    off = ~np.eye(N, dtype=bool)

    # ── §A the field ────────────────────────────────────────────────────────────
    out["A1_delta_h_train"] = info["delta_h"]
    out["A1_delta_h_test"] = ((np.where(Mte, Rte, 0) ** 2).sum((0, 1)) / np.maximum(Mte.sum((0, 1)), 1)).tolist()
    out["A2_corr_by_hops"] = {"hops": [1, 2, 3, 4, "5+"],
                              "train": [float(Ctr[(hops == k) & off].mean()) for k in (1, 2, 3, 4)] + [float(Ctr[(hops >= 5) & off].mean())],
                              "test": [float(Cte[(hops == k) & off].mean()) for k in (1, 2, 3, 4)] + [float(Cte[(hops >= 5) & off].mean())]}
    u = np.ones(N) / math.sqrt(N)
    def common_share(R, M, ok):
        Rz = np.where(M, R, 0.0)[ok]; return float(((Rz * u[None, :, None]).sum(1) ** 2).sum() / (Rz ** 2).sum())
    w_pc, v_pc = np.linalg.eigh(Ctr); pc1 = v_pc[:, -1]
    out["A3_common_share"] = {"train": common_share(Rtr, Mtr, oktr), "test": common_share(Rte, Mte, okte), "independent": 1.0 / N,
                              "pc1_overlap_with_constant": float(abs(pc1 @ u)), "pc1_share_of_corr": float(w_pc[-1] / w_pc.sum())}
    def total_sd(R, M, ok):
        return (np.where(M, R, 0.0) * sig[None, :, None]).sum(1)[ok].std(0)
    out["A4_total_sd"] = {"train": total_sd(Rtr, Mtr, oktr).tolist(), "test": total_sd(Rte, Mte, okte).tolist(),
                          "independent": float(np.sqrt((sig ** 2).sum() * info["var_data"]))}
    k_sensor = np.array([kurtosis(Rtr[:, i, :][Mtr[:, i, :]]) for i in range(N)])
    qs = np.linspace(0.001, 0.999, 41); allr = Rtr[Mtr]; allr = allr / allr.std()
    from scipy.stats import norm as _norm
    out["A5_tails"] = {"kurtosis_median": float(np.median(k_sensor)), "kurtosis_q90": float(np.quantile(k_sensor, 0.9)),
                       "qq": {"p": qs.tolist(), "data": np.quantile(allr, qs).tolist(), "gauss": _norm.ppf(qs).tolist()}}
    vtr = (np.where(Mtr, Rtr, 0) ** 2).sum((0, 2)) / np.maximum(Mtr.sum((0, 2)), 1)
    vte = (np.where(Mte, Rte, 0) ** 2).sum((0, 2)) / np.maximum(Mte.sum((0, 2)), 1)
    out["A6_test_train_var_ratio"] = {"overall": float((Rte[Mte] ** 2).mean() / (Rtr[Mtr] ** 2).mean()),
                                      "per_sensor_median": float(np.median(vte / np.maximum(vtr, 1e-9))),
                                      "per_horizon": (np.array(out["A1_delta_h_test"]) / np.array(info["delta_h"])).tolist()}

    # ── §B closeness ────────────────────────────────────────────────────────────
    # B1 graph-spectral variance: project on the ROAD laplacian's eigenvectors (frequency = lambda_k)
    def spectral(R, M, ok):
        Rz = np.where(M, R, 0.0)[ok]; C = np.einsum("ki,bih->bkh", V.T, Rz); return (C ** 2).mean((0, 2))
    B1 = {"lambda": lam.tolist(), "data_train": spectral(Rtr, Mtr, oktr).tolist(), "data_test": spectral(Rte, Mte, okte).tolist()}
    for name, (VS, VT, var, g, b) in priors.items():
        S = spatial_cov(VS, var); B1[name] = np.einsum("ik,ij,jk->k", V, S, V).tolist()     # v_k^T Sigma v_k
    out["B1_spectral"] = B1
    # B2 joint modes, B5 NLL, B6 whitening flatness -- in each prior's own basis
    B2, B5, B6 = {}, {}, {}
    for name, (VS, VT, var, g, b) in priors.items():
        d_tr = mode_second_moment(Rtr, Mtr, VS, VT); d_te = mode_second_moment(Rte, Mte, VS, VT)
        v = np.maximum(var, 1e-12)
        rho, _ = spearmanr(v.ravel(), d_tr.ravel())
        ratio = (d_tr / v).ravel(); top = np.argsort(-d_tr.ravel())[: int(0.95 * d_tr.size)]
        B2[name] = {"spearman_train": float(rho), "spearman_test": float(spearmanr(v.ravel(), d_te.ravel())[0]),
                    "ratio_within_2x_top95": float(np.mean((ratio[top] > 0.5) & (ratio[top] < 2.0))),
                    "log_ratio_q10_q50_q90": np.quantile(np.log(ratio[top]), [0.1, 0.5, 0.9]).tolist()}
        B5[name] = {"nll_train": float(0.5 * np.mean(np.log(v) + d_tr / v) + 0.5 * math.log(2 * math.pi)),
                    "nll_test": float(0.5 * np.mean(np.log(v) + d_te / v) + 0.5 * math.log(2 * math.pi))}
        x = (d_te / v).ravel(); B6[name] = float(np.exp(np.mean(np.log(x))) / np.mean(x))   # 1 = perfectly white after whitening
    out["B2_modes"], out["B5_nll"], out["B6_whitening_flatness_test"] = B2, B5, B6
    # B3 separability of the empirical covariance (train, in unit space): nearest kronecker product
    # cap the window count for the full NH x NH covariance (memory: pems-bay has 36k train windows)
    idx_ok = np.where(oktr)[0]; idx_ok = idx_ok[np.linspace(0, len(idx_ok) - 1, min(len(idx_ok), 12000)).astype(int)]
    X = np.where(Mtr[idx_ok], Rtr[idx_ok], 0.0).reshape(len(idx_ok), -1).astype(np.float32)      # vec: (N,H) row-major -> index i*H+h
    Cfull = np.zeros((N * H, N * H), np.float64)
    for s in range(0, len(X), 2048):
        xb = X[s:s + 2048].astype(np.float64); Cfull += xb.T @ xb
    Cfull /= len(X)
    Rt = Cfull.reshape(N, H, N, H).transpose(0, 2, 1, 3).reshape(N * N, H * H)               # van loan-pitsianis rearrangement
    sv = np.linalg.svd(Rt, compute_uv=False)
    out["B3_separability"] = {"kron_rank1_fraction": float(sv[0] ** 2 / (sv ** 2).sum()), "rank2_fraction": float((sv[:2] ** 2).sum() / (sv ** 2).sum())}
    del Cfull, Rt, X
    # B4 multi-scale variance: aggregates = sensors, 1-hop neighbourhoods, spectral clusters (4,16,64), network
    def clusters(k):
        feats = V[:, 1:1 + max(2, int(math.ceil(math.log2(k))) + 1)]
        _, lab = kmeans2(feats, k, minit="++", seed=0); return [np.where(lab == c)[0] for c in range(k) if (lab == c).sum() > 0]
    aggs = {"sensor": [np.array([i]) for i in range(N)], "1-hop": [np.where((hops[i] <= 1))[0] for i in range(N)],
            "cluster64": clusters(64), "cluster16": clusters(16), "cluster4": clusters(4), "network": [np.arange(N)]}
    def agg_ratio(R, M, ok, VS, var, sets):
        Rz = np.where(M, R, 0.0)[ok] * sig[None, :, None]
        num, den = [], []
        for S_ in sets:
            w = np.zeros(N); w[S_] = sig[S_]
            data_sd = Rz[:, S_, :].sum(1).std(0)                                              # (H,)
            prior_sd = np.array([math.sqrt(max(w @ cov_h(VS, var, h) @ w, 1e-12)) for h in range(H)])
            num.append(prior_sd); den.append(data_sd)
        return float(np.mean(np.array(num) / np.maximum(np.array(den), 1e-9)))
    B4 = {"scales": list(aggs.keys()), "mean_size": [float(np.mean([len(s) for s in v_])) for v_ in aggs.values()]}
    for name, (VS, VT, var, g, b) in priors.items():
        B4[name + "_train"] = [agg_ratio(Rtr, Mtr, oktr, VS, var, s) for s in aggs.values()]
        B4[name + "_test"] = [agg_ratio(Rte, Mte, okte, VS, var, s) for s in aggs.values()]
    out["B4_multiscale_ratio"] = B4
    # B7 the match surface + test ratio at the chosen pair
    mu_T_ = priors["matched"][1]; VS_m, VT_m, var_m, g_m, b_m = priors["matched"]
    tot_te = (np.where(Mte, Rte, 0.0) * sig[None, :, None]).sum(1)[okte].std(0)
    out["B7_match"] = {"beta": b_m, "gamma": g_m, "ratio_train": info["match_ratio_train"],
                       "ratio_test": float(np.mean(np.array([math.sqrt(sig @ cov_h(VS_m, var_m, h) @ sig) for h in range(H)]) / tot_te)),
                       "neighbour_corr_prior": float(spatial_cov(VS_m, var_m / var_m.mean())[(hops == 1) & off].mean()),
                       "neighbour_corr_data": out["A2_corr_by_hops"]["train"][0]}

    # ── §C what the prior changes ───────────────────────────────────────────────
    def draws(VS, VT, var, n=2000):
        Z = rng.standard_normal((n, N, H)) * np.sqrt(var)[None]; return np.einsum("ik,bkh,jh->bij", VS, Z, VT)
    e1 = np.where(hops == 1); pairs = (e1[0][:2000], e1[1][:2000])
    def derived(Rf):
        tot = (Rf * sig[None, :, None]).sum(1); nd = Rf[:, pairs[0], :] - Rf[:, pairs[1], :]; inc = Rf[:, :, 1:] - Rf[:, :, :-1]
        return {"total_sd": float(tot.std()), "neighbour_diff_sd": float(nd.std()), "horizon_increment_sd": float(inc.std()), "point_sd": float(Rf.std())}
    C2 = {"data_train": derived(np.where(Mtr, Rtr, 0.0)[oktr]), "data_test": derived(np.where(Mte, Rte, 0.0)[okte])}
    for name, (VS, VT, var, g, b) in priors.items():
        C2[name] = derived(draws(VS, VT, var, 800))
    out["C2_derived_quantities"] = C2
    th_m = 1.0 / (priors["matched"][2] / priors["matched"][2].mean())      # relative precision, matched
    out["C3_theta"] = {"matched_theta_quantiles": np.quantile(P.load_prior(ds, "kronecker", 42, backbone=BB, norm="pernode", global_edge="match").theta, [0, .1, .5, .9, 1]).tolist()}

    (OUT / (f"{ds}.json" if BB == "gwn" else f"{ds}_{BB}.json")).write_text(json.dumps(out, indent=1))
    # ── summary ─────────────────────────────────────────────────────────────────
    print(f"\n=== {ds} [{BB}]  N={N}  beta={b_m} gamma={g_m:.3f}  total-var ratio test/train {out['A6_test_train_var_ratio']['overall']:.2f}")
    print(f"  A2 corr by hops (train): {[round(x,3) for x in out['A2_corr_by_hops']['train']]}   A3 common share {100*out['A3_common_share']['train']:.1f}% (test {100*out['A3_common_share']['test']:.1f}%, indep {100/N:.1f}%)")
    print(f"  A4 total sd train h1..h12 {out['A4_total_sd']['train'][0]:.0f}..{out['A4_total_sd']['train'][-1]:.0f}  indep {out['A4_total_sd']['independent']:.0f}   A5 kurtosis median {out['A5_tails']['kurtosis_median']:.1f}")
    print(f"  B3 kronecker rank-1 fraction of the covariance: {100*out['B3_separability']['kron_rank1_fraction']:.1f}%")
    print(f"  {'prior':9s} {'NLL test':>9s} {'spearman':>9s} {'within2x':>9s} {'flat':>6s} | multiscale ratio (test): " + " ".join(f"{s[:7]:>7s}" for s in B4["scales"]))
    for name in priors:
        print(f"  {name:9s} {B5[name]['nll_test']:9.4f} {B2[name]['spearman_test']:9.3f} {B2[name]['ratio_within_2x_top95']:9.2f} {B6[name]:6.2f} | " + " ".join(f"{x:7.2f}" for x in B4[name + '_test']))
    print(f"  B7 matched: ratio train {out['B7_match']['ratio_train']:.3f} test {out['B7_match']['ratio_test']:.3f}; neighbour corr prior {out['B7_match']['neighbour_corr_prior']:.3f} vs data {out['B7_match']['neighbour_corr_data']:.3f}")
    print(f"  C2 sd of: total / neighbour diff / horizon increment — data {C2['data_train']['total_sd']:.0f}/{C2['data_train']['neighbour_diff_sd']:.2f}/{C2['data_train']['horizon_increment_sd']:.2f}"
          f"  iso {C2['iso']['total_sd']:.0f}/{C2['iso']['neighbour_diff_sd']:.2f}/{C2['iso']['horizon_increment_sd']:.2f}"
          f"  matched {C2['matched']['total_sd']:.0f}/{C2['matched']['neighbour_diff_sd']:.2f}/{C2['matched']['horizon_increment_sd']:.2f}")


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["pems08f", "pems08s", "pems04f", "pems04s", "metrla", "pemsbay"]):
        main(ds)
