"""shared machinery for the two val-set diagnostics shown in the per-dataset
notebooks. everything here reuses the *production* code paths (load_run from the
offline eval, reverse_sample, the same masked metrics and W1/energy distances), so
the numbers the notebook prints ARE the numbers training and eval_diffusion report.

the two evals:

  1. reconstruction (eval 1) -- forward-noise the TRUE val residuals all the way to
     t=T (fully noised), then run the deterministic (eta=0) reverse sampler back to
     t=0. we report the masked MAE/MSE of the reconstruction and the MAE-vs-t curve
     along the trajectory. this is the "can the denoiser invert the forward
     process" check -- it is oracle-seeded (it starts from the true signal's noise),
     so it is a best case / sanity floor, not generative skill.

  2. generative distribution (eval 2) -- draw PURE-PRIOR samples (no true signal)
     and run the full stochastic (eta=1) reverse sde, then compare the generated
     residual cloud to the true val residual cloud, per horizon, with sliced
     wasserstein-1 + energy distance. we also report the K-sample-mean MAE/MSE and
     the Winkler-90 interval score + empirical coverage from the same samples.

all metrics are de-normalized to raw units (x sigma_r) -> mph (speed) or veh (flow),
so they are comparable to the backbone and across the two priors. the backbone term
cancels (forecast_err = (Y_hat+R0_hat) - (Y_hat+R0_true) = R0_hat - R0_true), so we
score residuals directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

# put the repo root on sys.path so `src` / `scripts` import regardless of the
# kernel's working directory (notebooks/ or the repo root).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from src.data.residual_dataset import ResidualDataset, residual_path, train_residual_sigma
from src.diffusion.forward_fcps import prior_to_torch, sample_R_t_given_R0_torch
from src.diffusion.reverse import reverse_sample
from src.eval.distribution import energy_distance, sliced_wasserstein1
from src.eval.intervals import winkler_score
from src.utils import paths
# the canonical loader: rebuilds the SAME prior + schedule the run trained with
# (incl. the c_0 = 5/theta_min fix), loads best.pt. importing it is side-effect
# free (no wandb), so the notebook stays exactly in sync with offline eval.
from scripts.eval_diffusion import empirical_crps, load_run

DATASETS = ("metrla", "pems08f", "pemsbay")
PRIORS = ("kronecker", "isotropic")
HL_HORIZONS = (3, 6, 12)                       # horizons highlighted in plots
PRIOR_LABEL = {"kronecker": "Kronecker (structured)", "isotropic": "isotropic (baseline)"}
UNIT = {"metrla": "mph", "pemsbay": "mph", "pems08f": "veh"}


# ─── device / rng ────────────────────────────────────────────────────────────


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_generator(device: torch.device, seed: int):
    # mps has no torch.Generator support for the ops we use; fall back to None
    if device.type == "mps":
        return None
    return torch.Generator(device=device.type).manual_seed(seed)


# ─── discover + load trained runs ────────────────────────────────────────────


def discover_runs(dataset: str) -> dict:
    """scan results/diffusion/** for trained checkpoints of `dataset`.

    returns {(prior, seed): {exp, ckpt_dir, mtime}}, keeping the newest best.pt per
    (prior, seed). handles both layouts: results/diffusion/<exp>/<ds>/<prior>/seedN
    and the un-namespaced results/diffusion/<ds>/<prior>/seedN.
    """
    base = paths.RESULTS / "diffusion"
    out: dict = {}
    if not base.exists():
        return out
    for ck in base.rglob("best.pt"):
        parts = ck.relative_to(base).parts            # (...exp, ds, prior, seedN, best.pt)
        prior = next((p for p in parts if p in PRIORS), None)
        if prior is None:
            continue
        i = parts.index(prior)
        if i == 0 or parts[i - 1] != dataset:
            continue
        seed_part = parts[i + 1] if i + 1 < len(parts) else ""
        if not seed_part.startswith("seed"):
            continue
        try:
            seed = int(seed_part[4:])
        except ValueError:
            continue
        exp = "/".join(parts[: i - 1]) or None        # everything before the dataset
        key = (prior, seed)
        mt = ck.stat().st_mtime
        if key not in out or mt > out[key]["mtime"]:
            out[key] = {"exp": exp, "ckpt_dir": ck.parent, "mtime": mt}
    return out


def load_everything(dataset: str, prior: str, seed: int, exp, device: torch.device) -> dict:
    """load a trained run + the val split it should be evaluated on, bundled."""
    denoiser, prior_obj, schedule, cfg, epoch = load_run(dataset, prior, seed, exp, device)
    sigma_r = train_residual_sigma(dataset, seed)
    val = ResidualDataset(residual_path(dataset, "val", seed), sigma_r)
    return {
        "denoiser": denoiser, "prior": prior_obj, "prior_t": prior_to_torch(prior_obj, device),
        "schedule": schedule, "cfg": cfg, "epoch": epoch, "sigma_r": sigma_r,
        "val": val, "seed": seed, "exp": exp,
    }


def load_runs(dataset: str, *, device: torch.device, seed=None, priors=PRIORS, exp=None) -> dict:
    """discover + load every requested prior for a dataset. prints one status line
    per prior and returns {prior: run-dict}; priors with no checkpoint are skipped
    (so the notebook degrades cleanly when checkpoints aren't synced locally)."""
    found = discover_runs(dataset)
    runs: dict = {}
    for prior in priors:
        cands = [(s, info) for (p, s), info in found.items()
                 if p == prior and (seed is None or s == seed)
                 and (exp is None or info["exp"] == exp)]
        if not cands:
            tail = f" for seed {seed}" if seed is not None else ""
            print(f"  [{prior:9s}] no checkpoint found{tail} under results/diffusion/ "
                  f"-- train it or sync from DAIC.")
            continue
        s, info = sorted(cands)[0]                     # smallest seed for determinism
        print(f"  [{prior:9s}] loading exp={info['exp']} seed={s}  ({info['ckpt_dir']})")
        runs[prior] = load_everything(dataset, prior, s, info["exp"], device)
    return runs


# ─── small masked helpers (mirror scripts/train_diffusion.py) ────────────────


def _masked_sums_h(err: torch.Tensor, mask: torch.Tensor):
    """sum of err*mask and count of mask, per horizon. err/mask (B, N, H) -> (H,)."""
    m = mask.float()
    return (err * m).sum(dim=(0, 1)).cpu(), m.sum(dim=(0, 1)).cpu()


def _subsample(a: np.ndarray, cap: int) -> np.ndarray:
    """deterministic thinning to <= cap points (for plotting big residual clouds)."""
    if a.shape[0] <= cap:
        return a
    idx = np.linspace(0, a.shape[0] - 1, cap).astype(int)
    return a[idx]


def _progress(tag, done, total, verbose):
    if verbose:
        print(f"    {tag}: {done}/{total} windows", end="\r", flush=True)


# ─── eval 1: reconstruction (fully noise -> denoise) ─────────────────────────


@torch.no_grad()
def reconstruction_eval(run: dict, *, device: torch.device, n_steps=100, log_every=10,
                        max_windows=1024, chunk=128, generator=None, verbose=True) -> dict:
    """forward-noise the true val residuals to t=T, then deterministically (eta=0)
    reverse-denoise to t=0. returns overall + per-horizon masked MAE/MSE (raw units)
    and the dataset-wide MAE/MSE-vs-t denoising curve."""
    denoiser, prior, prior_t = run["denoiser"], run["prior"], run["prior_t"]
    schedule, val, sigma_r = run["schedule"], run["val"], run["sigma_r"]
    H = prior.H
    n_total = min(max_windows, len(val)) if max_windows else len(val)

    num_mae = torch.zeros(H); num_mse = torch.zeros(H); cnt = torch.zeros(H)     # final t=0
    c_t: dict = {}; c_mae: dict = {}; c_mse: dict = {}; c_cnt: dict = {}          # the curve

    for start in range(0, n_total, chunk):
        end = min(start + chunk, n_total)
        R0 = val.R0[start:end].to(device)
        Y = val.Y_hat[start:end].to(device)
        Mk = val.mask[start:end].to(device)
        b = R0.shape[0]
        R0_phys = R0 * sigma_r

        # fully noise the TRUE residuals to the top of the schedule
        t_top = torch.full((b,), schedule.T, device=device, dtype=R0.dtype)
        R_T = sample_R_t_given_R0_torch(R0, t_top, prior_t, schedule, generator=generator)

        def _cb(i, t_cur, R0_hat, _M=Mk, _R=R0_phys):
            if i % log_every:
                return
            err = R0_hat * sigma_r - _R
            sm, cm = _masked_sums_h(err.abs(), _M)
            ss, _ = _masked_sums_h(err.pow(2), _M)
            c_t[i] = t_cur
            c_mae[i] = c_mae.get(i, 0.0) + sm
            c_mse[i] = c_mse.get(i, 0.0) + ss
            c_cnt[i] = c_cnt.get(i, 0.0) + cm

        R0_recon, _ = reverse_sample(prior, schedule, denoiser, Y, n_steps=n_steps,
                                     eta=0.0, R_T=R_T, generator=generator, step_callback=_cb)
        err = R0_recon * sigma_r - R0_phys
        sm, cm = _masked_sums_h(err.abs(), Mk); num_mae += sm; cnt += cm
        ss, _ = _masked_sums_h(err.pow(2), Mk); num_mse += ss
        _progress("recon", end, n_total, verbose)

    if verbose:
        print()
    cnt = cnt.clamp(min=1.0)
    # curve sorted T -> 0, with the exact t=0 reconstruction appended
    curve = []
    for i in sorted(c_t, key=lambda k: c_t[k], reverse=True):
        d = c_cnt[i].sum().clamp(min=1.0)
        curve.append((c_t[i], float(c_mae[i].sum() / d), float(c_mse[i].sum() / d)))
    curve.append((0.0, float(num_mae.sum() / cnt.sum()), float(num_mse.sum() / cnt.sum())))

    return {
        "mae": float(num_mae.sum() / cnt.sum()), "mse": float(num_mse.sum() / cnt.sum()),
        "mae_h": (num_mae / cnt).tolist(), "mse_h": (num_mse / cnt).tolist(),
        "curve": curve, "n_windows": int(n_total), "n_steps": n_steps,
    }


# ─── eval 2: generative distribution (prior -> denoise -> vs val) ────────────


@torch.no_grad()
def generative_eval(run: dict, *, device: torch.device, k=50, n_steps=100, eta=1.0,
                    alpha=0.1, max_windows=512, gen_batch=1024, dist_windows=512,
                    dist_samples=4, w1_proj=128, plot_cap=40000, generator=None,
                    verbose=True) -> dict:
    """pure-prior samples through the full stochastic (eta) reverse sde, compared to
    the true val residuals. part A: K samples/window -> sample-mean MAE/MSE, Winkler
    interval + coverage, CRPS. part B: complete-case windows -> sliced-W1 + energy
    per horizon, plus subsampled true/gen value clouds for the distribution plots."""
    denoiser, prior, schedule = run["denoiser"], run["prior"], run["schedule"]
    val, sigma_r = run["val"], run["sigma_r"]
    N, H = prior.N, prior.H

    # ── part A: point accuracy + Winkler interval (K samples per window) ──
    n_total = min(max_windows, len(val)) if max_windows else len(val)
    chunk = max(1, gen_batch // k)
    num_mae = torch.zeros(H); num_mse = torch.zeros(H); cnt = torch.zeros(H)
    num_w = torch.zeros(H); num_cov = torch.zeros(H); num_crps = torch.zeros(H)
    for start in range(0, n_total, chunk):
        end = min(start + chunk, n_total)
        R0 = val.R0[start:end].to(device) * sigma_r
        Y = val.Y_hat[start:end].to(device)
        Mk = val.mask[start:end].to(device)
        b = R0.shape[0]
        samp, _ = reverse_sample(prior, schedule, denoiser, Y.repeat(k, 1, 1),
                                 n_steps=n_steps, eta=eta, R_T=None, generator=generator)
        samp = samp.view(k, b, N, H) * sigma_r
        mean = samp.mean(dim=0)
        sm, cm = _masked_sums_h((mean - R0).abs(), Mk); num_mae += sm; cnt += cm
        ss, _ = _masked_sums_h((mean - R0).pow(2), Mk); num_mse += ss
        lo = torch.quantile(samp, alpha / 2.0, dim=0)
        hi = torch.quantile(samp, 1.0 - alpha / 2.0, dim=0)
        sw, _ = _masked_sums_h(winkler_score(R0, lo, hi, alpha), Mk); num_w += sw
        inside = ((R0 >= lo) & (R0 <= hi)).float()
        sc, _ = _masked_sums_h(inside, Mk); num_cov += sc
        scr, _ = _masked_sums_h(empirical_crps(samp, R0), Mk); num_crps += scr
        _progress("gen point", end, n_total, verbose)
    if verbose:
        print()
    cnt = cnt.clamp(min=1.0)

    # ── part B: distribution fidelity on complete-case (fully observed) windows ──
    M_all = val.mask
    valid = M_all.reshape(M_all.shape[0], -1).all(dim=1)
    vidx = valid.nonzero(as_tuple=True)[0]
    n_w = min(dist_windows, vidx.numel())
    sel = vidx[:n_w]
    R0v = val.R0[sel].to(device) * sigma_r
    Yv = val.Y_hat[sel].to(device)
    wb = max(1, gen_batch // dist_samples)
    parts = []
    for s in range(0, n_w, wb):
        e = min(s + wb, n_w)
        g, _ = reverse_sample(prior, schedule, denoiser, Yv[s:e].repeat(dist_samples, 1, 1),
                              n_steps=n_steps, eta=eta, R_T=None, generator=generator)
        parts.append(g.view(dist_samples, e - s, N, H))
        _progress("gen dist", e, n_w, verbose)
    if verbose:
        print()
    gen = torch.cat(parts, dim=1) * sigma_r                       # (dist_samples, n_w, N, H)

    w1_h, ed_h, dist_plots = [], [], {}
    for h in range(H):
        true_c = R0v[:, :, h]                                     # (n_w, N)
        gen_c = gen[:, :, :, h].reshape(dist_samples * n_w, N)    # (dist_samples*n_w, N)
        w1_h.append(sliced_wasserstein1(gen_c, true_c, w1_proj, generator))
        ed_h.append(energy_distance(gen_c, true_c))
        if (h + 1) in HL_HORIZONS:
            tv = _subsample(true_c.reshape(-1).cpu().numpy(), plot_cap)
            gv = _subsample(gen_c.reshape(-1).cpu().numpy(), plot_cap)
            dist_plots[h + 1] = (tv, gv)

    overall = lambda num: float(num.sum() / cnt.sum())
    return {
        "mae": overall(num_mae), "mse": overall(num_mse),
        "winkler": overall(num_w), "coverage": overall(num_cov), "crps": overall(num_crps),
        "mae_h": (num_mae / cnt).tolist(), "mse_h": (num_mse / cnt).tolist(),
        "winkler_h": (num_w / cnt).tolist(), "coverage_h": (num_cov / cnt).tolist(),
        "w1": float(np.mean(w1_h)), "energy": float(np.mean(ed_h)),
        "w1_h": w1_h, "energy_h": ed_h, "dist_plots": dist_plots,
        "alpha": alpha, "k": k, "eta": eta,
        "n_windows": int(n_total), "n_dist_windows": int(n_w),
    }


# ─── tables + plots ──────────────────────────────────────────────────────────


def reconstruction_table(res_by_prior: dict):
    import pandas as pd
    rows = {PRIOR_LABEL[p]: {"recon MAE": r["mae"], "recon MSE": r["mse"],
                             "recon RMSE": r["mse"] ** 0.5, "val windows": r["n_windows"]}
            for p, r in res_by_prior.items()}
    return pd.DataFrame(rows).T.round(4)


def generative_table(res_by_prior: dict):
    import pandas as pd
    rows = {}
    for p, r in res_by_prior.items():
        pct = int(round((1 - r["alpha"]) * 100))
        rows[PRIOR_LABEL[p]] = {
            "gen MAE": r["mae"], "gen MSE": r["mse"], "gen RMSE": r["mse"] ** 0.5,
            f"Winkler{pct}": r["winkler"], "coverage": r["coverage"], "CRPS": r["crps"],
            "W1": r["w1"], "energy": r["energy"],
        }
    return pd.DataFrame(rows).T.round(4)


def plot_reconstruction(res_by_prior: dict, dataset: str):
    import matplotlib.pyplot as plt
    unit = UNIT[dataset]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for prior, r in res_by_prior.items():
        ts = [t for t, _, _ in r["curve"]]
        maes = [m for _, m, _ in r["curve"]]
        ax[0].plot(ts, maes, marker="o", ms=3, label=PRIOR_LABEL[prior])
    ax[0].invert_xaxis()                                          # T on the left, 0 on the right
    ax[0].set_xlabel("diffusion time t   (T -> 0, the denoising direction)")
    ax[0].set_ylabel(f"masked MAE [{unit}]")
    ax[0].set_title("MAE as the reverse sampler denoises a fully-noised batch")
    ax[0].grid(alpha=.3); ax[0].legend()

    H = len(next(iter(res_by_prior.values()))["mae_h"])
    x = np.arange(1, H + 1); w = 0.38
    for j, (prior, r) in enumerate(res_by_prior.items()):
        ax[1].bar(x + (j - 0.5) * w, r["mae_h"], width=w, label=PRIOR_LABEL[prior])
    ax[1].set_xlabel("horizon (steps ahead)"); ax[1].set_ylabel(f"recon MAE [{unit}]")
    ax[1].set_title("reconstruction MAE per horizon  (at t=0)")
    ax[1].set_xticks(x); ax[1].grid(alpha=.3, axis="y"); ax[1].legend()
    fig.suptitle(f"{dataset} — eval 1: fully noise the true val residuals, then denoise back",
                 fontweight="bold")
    fig.tight_layout(); plt.show()


def plot_generative(res_by_prior: dict, dataset: str):
    import matplotlib.pyplot as plt
    unit = UNIT[dataset]
    priors = list(res_by_prior)
    hs = list(HL_HORIZONS)
    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(1, len(hs) + 1)
    for c, h in enumerate(hs):
        ax = fig.add_subplot(gs[0, c])
        tv, _ = res_by_prior[priors[0]]["dist_plots"][h]          # true cloud is prior-independent
        ax.hist(tv, bins=80, density=True, histtype="stepfilled", alpha=.35,
                color="0.4", label="val (true)")
        for prior in priors:
            _, gv = res_by_prior[prior]["dist_plots"][h]
            ax.hist(gv, bins=80, density=True, histtype="step", lw=1.6, label=PRIOR_LABEL[prior])
        ax.set_title(f"horizon {h}"); ax.set_xlabel(f"residual [{unit}]")
        if c == 0:
            ax.set_ylabel("density"); ax.legend(fontsize=8)
    axw = fig.add_subplot(gs[0, len(hs)])
    H = len(res_by_prior[priors[0]]["w1_h"])
    x = np.arange(1, H + 1); w = 0.38
    for j, prior in enumerate(priors):
        axw.bar(x + (j - 0.5) * w, res_by_prior[prior]["w1_h"], width=w, label=PRIOR_LABEL[prior])
    axw.set_title("sliced-W1 per horizon  (lower = better)")
    axw.set_xlabel("horizon"); axw.set_ylabel(f"W1 distance [{unit}]")
    axw.grid(alpha=.3, axis="y"); axw.legend(fontsize=8)
    fig.suptitle(f"{dataset} — eval 2: prior samples vs the true val distribution",
                 fontweight="bold")
    fig.tight_layout(); plt.show()
