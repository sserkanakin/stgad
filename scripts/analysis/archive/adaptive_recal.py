"""Adaptive spread recalibration: can conditional/mode-resolved variance win where static priors lost?

For each arm (iso/emp/rsh, same trained checkpoints), sample a VAL and a TEST window subset,
fit recalibration ON VAL ONLY, apply to the TEST ensembles, and score. Variants:

  none    raw ensembles (baseline = what the DAIC evals measured)
  glob    one global spread scalar (all iso can express natively)
  tod     s(tod), 24 hourly buckets (conditional scale)
  mode    s_k per mode-energy decile in the prior eigenbasis (structured handle)
  mode+x  mode with the train->val per-decile variance trend extrapolated half a log-step
  tod+mode[+x]  composed

The honest comparison: "iso + glob" (best flat-expressible) vs "emp + tod+mode+x" (what the
structured basis enables). All fits use train/val only — no test leakage.

usage: .venv/bin/python scripts/analysis/adaptive_recal.py --dataset pems08s [--backbone gwn]
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# PROJECT_DIR lets the script run from OUTSIDE the repo (e.g. parked on windowshome when
# linuxhome is at quota); default = repo-relative as usual.
ROOT = Path(os.environ.get("PROJECT_DIR") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT))
from src.utils import paths                                                       # noqa: E402
from src.data.residual_dataset import make_dataset, train_residual_sigma, sigma_to_torch  # noqa: E402
from src.diffusion.reverse import reverse_sample_ancestral                        # noqa: E402
from src.diffusion.prior import _load_residuals_train, _spatial_graph_eig, _fit_sigma_T   # noqa: E402
from src.eval.crps import _QUANTILES                                              # noqa: E402

spec = importlib.util.spec_from_file_location("eval_diffusion", ROOT / "scripts" / "eval_diffusion.py")
ed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ed)

ARMS_BASE = [("iso", "isotropic"), ("emp", "kronecker"), ("rsh", "kronecker")]
N_DECILES = 10
TOD_BUCKETS = 24


@torch.no_grad()
def sample_split(arm, prior, args, device, split, n_windows, t0_offset):
    """ensembles + truth for a stride-subset of a split. returns raw-unit arrays."""
    denoiser, prior_obj, schedule, cfg, _ = ed.load_run(
        args.dataset, prior, args.seed, f"{args.exp}/{arm}", device, args.backbone,
        ckpt_name="best_crps.pt")
    res_seed = cfg.get("residual_seed", args.seed)
    sigma_r = train_residual_sigma(args.dataset, res_seed, args.backbone, norm=cfg.get("norm", "global"))
    sig = sigma_to_torch(sigma_r, device)
    ds = make_dataset(args.dataset, split, res_seed, args.backbone, sigma_r,
                      cond_past=bool(cfg.get("cond_past")),
                      zero_masked=bool(cfg.get("zero_masked_input")))
    B = ds.R0.shape[0]
    idx = np.linspace(0, B - 1, min(n_windows, B)).round().astype(int)
    tod = ((idx + t0_offset + 12) % 288) // (288 // TOD_BUCKETS)
    fc_all, yt_all, mk_all = [], [], []
    chunk = max(1, args.gen_batch // args.k)
    for s in range(0, len(idx), chunk):
        sel = idx[s:s + chunk]
        R0 = ds.R0[sel].to(device) * sig
        Y = ds.Y_hat[sel].to(device)
        Mk = ds.mask[sel].to(device)
        yt = ds.y_true_raw[sel].to(device)
        yp = yt - R0
        b = R0.shape[0]
        samp, _ = reverse_sample_ancestral(prior_obj, schedule, denoiser,
                                           Y.repeat(args.k, 1, 1), n_steps=args.n_steps)
        fc = (yp.unsqueeze(0) + samp.view(args.k, b, -1, prior_obj.H) * sig).clamp(min=0.0)
        fc_all.append(fc.cpu()); yt_all.append(yt.cpu())
        mk_all.append((Mk & (yt.abs() > 1e-6)).cpu())
    return (torch.cat(fc_all, 1), torch.cat(yt_all), torch.cat(mk_all), tod)


def crps_cov(fc, yt, mk, qs):
    q_pred = torch.quantile(fc, qs, dim=0)
    yb = yt.unsqueeze(0)
    pin = ((q_pred - yb) * ((yb <= q_pred).to(fc.dtype) - qs.view(-1, 1, 1, 1))).abs()
    num = float((2.0 * pin.sum(0) * mk).sum()); den = float((yt.abs() * mk).sum())
    lo = torch.quantile(fc, 0.05, dim=0); hi = torch.quantile(fc, 0.95, dim=0)
    cov = float((((yt >= lo) & (yt <= hi)) & mk.bool()).sum()) / max(float(mk.sum()), 1.0)
    return num / (den * 19.0), cov


def project(D, V_S, V_T):
    return torch.einsum("kn,sbnh->sbkh", V_S.T, D) @ V_T


def backproject(P, V_S, V_T):
    return torch.einsum("nk,sbkh->sbnh", V_S, P @ V_T.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pems08s")
    ap.add_argument("--backbone", default="gwn")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp", default="20260721_ladder")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--n-steps", type=int, default=25)
    ap.add_argument("--val-windows", type=int, default=400)
    ap.add_argument("--test-windows", type=int, default=600)
    ap.add_argument("--gen-batch", type=int, default=1024)
    ap.add_argument("--cond", choices=("fo", "past"), default="fo",
                    help="which conditioning block's checkpoints to use (arm dirs get -past)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    qs = _QUANTILES.clone()
    suffix = "" if args.cond == "fo" else "-past"
    arms = [(name + suffix, prior) for name, prior in ARMS_BASE]

    # shared analysis frame: prior-convention eigenbasis + per-mode train/val variances
    Rtr, Mtr = _load_residuals_train(args.dataset, 42, args.backbone, norm="pernode")
    lam, V_S = _spatial_graph_eig(args.dataset, Rtr, "road", 15, repair=True)
    _, _, ST = _fit_sigma_T(Rtr, Mtr)
    _, V_T = np.linalg.eigh(ST)
    V_St, V_Tt = torch.tensor(V_S, dtype=torch.float32), torch.tensor(V_T, dtype=torch.float32)

    def mode_vars(split):
        d = np.load(paths.residual_npz(args.dataset, split, 42, args.backbone))
        sig = np.asarray(train_residual_sigma(args.dataset, 42, args.backbone, norm="pernode")).reshape(1, -1, 1)
        R = d["residuals"] / sig; M = d["mask"].astype(bool)
        C = np.matmul(np.matmul(V_S.T, np.where(M, R, 0.0)), V_T)
        return np.clip((C ** 2).mean(0), 1e-12, None), R.shape[0]

    v_tr, B_tr = mode_vars("train")
    v_va, B_va = mode_vars("val")
    _, B_te = mode_vars("test")
    t0_val, t0_test = B_tr + 23, B_tr + 23 + B_va + 23   # raw-step offsets of the splits
    dec = np.digitize(v_tr.ravel(), np.quantile(v_tr.ravel(), np.linspace(0, 1, N_DECILES + 1)[1:-1]))
    dec_t = torch.tensor(dec.reshape(v_tr.shape))
    # per-decile trend extrapolation: continue half the train->val log-growth
    g = np.array([np.exp(0.5 * np.log(np.clip(
        v_va.ravel()[dec == i].mean() / v_tr.ravel()[dec == i].mean(), 0.25, 4.0)))
        for i in range(N_DECILES)])

    results = {}
    for arm, prior in arms:
        t0 = time.time()
        fcv, ytv, mkv, todv = sample_split(arm, prior, args, device, "val", args.val_windows, t0_val)
        fct, ytt, mkt, todt = sample_split(arm, prior, args, device, "test", args.test_windows, t0_test)
        print(f"[{arm}] sampled val+test in {time.time()-t0:.0f}s", flush=True)

        # ── fit on val ───────────────────────────────────────────
        mv = fcv.mean(0); Dv = fcv - mv.unsqueeze(0); Tv = (ytv - mv)
        mk = mkv.bool()
        s_glob = float(torch.sqrt((Tv ** 2)[mk].mean() / (Dv ** 2).mean(0)[mk].mean()))
        s_tod = np.ones(TOD_BUCKETS)
        for b in range(TOD_BUCKETS):
            w = torch.tensor(todv == b)
            if w.any():
                m2 = mk[w]
                s_tod[b] = float(torch.sqrt((Tv[w] ** 2)[m2].mean() / (Dv[:, w] ** 2).mean(0)[m2].mean()))
        Pt = project(Tv.unsqueeze(0), V_St, V_Tt)[0]          # (b,N,H) true dev in modes
        Pe = project(Dv, V_St, V_Tt)                          # (k,b,N,H)
        s_mode = np.ones(N_DECILES)
        for i in range(N_DECILES):
            sel = (dec_t == i)
            s_mode[i] = float(torch.sqrt((Pt ** 2)[:, sel].mean() / (Pe ** 2).mean(0)[:, sel].mean()))

        # ── apply to test ────────────────────────────────────────
        # each per-bucket fit already contains the global correction; for COMPOSED variants
        # use shape-only profiles (geometric mean 1) times the single global scale, so the
        # level is corrected exactly once.
        mt = fct.mean(0); Dt = fct - mt.unsqueeze(0)
        gm = lambda x: np.exp(np.log(np.clip(x, 1e-6, None)).mean())
        tod_shape = s_tod / gm(s_tod); mode_shape = s_mode / gm(s_mode)
        stod_w = torch.tensor(s_tod[todt], dtype=torch.float32).view(1, -1, 1, 1)
        stodsh_w = torch.tensor(tod_shape[todt], dtype=torch.float32).view(1, -1, 1, 1)
        smode = torch.tensor(s_mode[dec], dtype=torch.float32).view(v_tr.shape)
        smodesh = torch.tensor(mode_shape[dec], dtype=torch.float32).view(v_tr.shape)
        smodesh_x = torch.tensor((mode_shape * g)[dec], dtype=torch.float32).view(v_tr.shape)

        def modescale(D, sm):
            return backproject(project(D, V_St, V_Tt) * sm, V_St, V_Tt)

        variants = {
            "none": Dt,
            "glob": Dt * s_glob,
            "tod": Dt * stod_w,
            "mode": modescale(Dt, smode),
            "mode+x": modescale(Dt, smode * torch.tensor(g[dec], dtype=torch.float32).view(v_tr.shape)),
            "tod+mode": modescale(Dt, smodesh) * stodsh_w * s_glob,
            "tod+mode+x": modescale(Dt, smodesh_x) * stodsh_w * s_glob,
        }
        res = {}
        for name, D in variants.items():
            crps, cov = crps_cov((mt.unsqueeze(0) + D).clamp(min=0.0), ytt, mkt, qs)
            res[name] = {"crps": crps, "cov": cov}
            print(f"  [{arm}] {name:10s} crps={crps:.4f} cov={cov:.2f}", flush=True)
        res["_fit"] = {"s_glob": s_glob, "s_tod": s_tod.tolist(), "s_mode": s_mode.tolist(),
                       "g_extrap": g.tolist()}
        results[arm] = res

    tag = "" if args.cond == "fo" else "_past"
    out = paths.analysis_dir("adaptive_recal") / f"{args.dataset}_{args.backbone}{tag}_seed{args.seed}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
