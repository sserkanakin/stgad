"""evaluate trained diffusion checkpoints on the VAL or TEST split (--split).

for each (dataset, prior, seed) this loads best.pt, generates reverse-sde samples
on the split's residuals, and reports per-horizon + overall (all in raw units;
forecast_err = Y_hat + R0_hat - (Y_hat + R0_true) = R0_hat - R0_true, so the
backbone term cancels and we score residuals):

  - MAE / MSE of the K-sample mean            (point forecast accuracy)
  - Winkler interval score + empirical coverage at level alpha   (calibration)
  - sliced wasserstein-1 + energy distance vs the true residual cloud   (fidelity)

samples come from GAD's Euler–Maruyama reverse SDE (the only sampler — stochastic /
ancestral, so the intervals are properly dispersed).

writes <eval_dir>/{split}_metrics_steps{n}.json (per-steps filename, so parallel
step-count evals never clobber) and, with --all, prints a kron-vs-iso table.

usage:
    python scripts/eval_diffusion.py --dataset metrla --prior kronecker --seed 42 --split val
    python scripts/eval_diffusion.py --all
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np

# reduce fragmentation for the big-N generation batches (set before torch init)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

# eval is sampling-based (k draws of a stochastic reverse sde), so the tiny
# precision loss of TF32 matmuls is far below the monte-carlo noise floor but
# buys ~1.5-2x on the denoiser's graph-conv / linear layers on ampere/ada gpus.
# no-op on cpu/mps and on pre-ampere cards.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from src.data.adjacency import load_adjacency
from src.data.residual_dataset import (
    make_dataset, sigma_to_torch, train_residual_sigma,
)
from src.diffusion.denoiser import GCNNDenoiser, R0Denoiser
from src.diffusion.forward_fcps import FCPSSchedule, prior_to_torch
from src.diffusion.prior import load_prior
from src.diffusion.reverse import min_em_steps, reverse_sample_ancestral, reverse_sample_em
from src.eval.distribution import energy_distance, sliced_wasserstein1
from src.eval.intervals import winkler_score
from src.utils import paths
from src.utils.logging_setup import get_logger, setup_logging

# shared timestamped logger. when eval runs inside the combined traineval job the logger
# is already configured (lines land in the run's train.log too); standalone, main() sets
# it up for stdout. formatted comparison tables stay on plain print (kept clean).
log = get_logger()

_HL = (3, 6, 12)  # highlighted horizons (1-indexed)


# ─── small masked helpers (per horizon over (B, N, H)) ───────────────────────


def _masked_sums(err, mask):
    """sum of err*mask and count of mask, per horizon. err/mask (B, N, H)."""
    m = mask.float()
    return (err * m).sum(dim=(0, 1)), m.sum(dim=(0, 1))


from src.eval.crps import empirical_crps, quantile_crps_num   # noqa: E402  (shared w/ training)


# ─── load a trained run ──────────────────────────────────────────────────────


def load_run(dataset, prior, seed, exp, device, backbone="gwn", ckpt_name="best.pt"):
    ckpt_dir = paths.diffusion_dir(dataset, prior, seed, exp=exp, backbone=backbone)
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    ckpt = torch.load(ckpt_dir / ckpt_name, map_location=device, weights_only=False)

    # stage-1 (residual/prior) seed vs stage-2 (diffusion) seed: the ladder trains all
    # diffusion seeds on ONE residual set (residual_seed, pinned at 42), so the residuals,
    # sigma_r and the fitted prior all live under res_seed, NOT the diffusion seed. old
    # configs predate the split -> fall back to the diffusion seed.
    res_seed = cfg.get("residual_seed", seed)

    # rebuild the SAME prior the run trained with (γ, ρ, graph repair and residual
    # norm from config; defaults keep back-compat with older configs).
    # the RESOLVED values training recorded (a matched β/γ must not be re-fitted here)
    beta = float(cfg.get("global_edge_beta", 0.0) or 0.0)
    prior_obj = load_prior(dataset, prior, res_seed,
                           gamma=cfg.get("gamma_resolved", cfg.get("gamma")),
                           shrink_t=cfg.get("shrink_t"),
                           backbone=backbone,
                           repair=cfg.get("repair_graph"),
                           norm=cfg.get("norm", "global"),
                           global_edge=(beta or None))
    # rebuild the schedule exactly as training used it, INCLUDING the floor the
    # denoiser was trained down to (old configs have no c_lo -> the historical 8e-4).
    schedule = FCPSSchedule(c_min=cfg.get("c_min", 0.1), c_0=cfg.get("c_0", 7.0),
                            alpha=cfg.get("alpha", 4.0), T=1.0,
                            c_lo_override=cfg.get("c_lo"))
    log.info(f"[eval] schedule (FCPS): S=c_0={schedule.c_0:.2f} α={schedule.alpha:.1f} "
             f"c_min={schedule.c_min:.2f} c_lo={schedule.c_lo:.3g} "
             f"θ_min={float(prior_obj.theta.min()):.4f} θ_max={float(prior_obj.theta.max()):.1f} "
             f"β={beta:.4g} γ={prior_obj.gamma:.4g}")
    if cfg.get("denoiser") == "gcnn":
        denoiser = GCNNDenoiser(
            N=prior_obj.N, H=prior_obj.H, adjacency=load_adjacency(dataset, global_edge=beta),
            K=cfg.get("gcnn_hops", 5), n_layers=cfg.get("gcnn_layers", 3),
            hidden=cfg.get("gcnn_hidden", 24),
            # cond-past runs train with n_cond=2 ([R_past | Yhat]); without this the
            # state_dict load fails on the first conv (3 vs 2 input channels).
            n_cond=(2 if cfg.get("cond_past") else 1),
        ).to(device)
    else:
        denoiser = R0Denoiser(
            N=prior_obj.N, H=prior_obj.H, adjacency=load_adjacency(dataset, global_edge=beta),
            d_h=cfg["d_h"], channel_multipliers=tuple(cfg["channel_multipliers"]),
            n_blocks=cfg["n_blocks"], device=str(device),
            T_h=(prior_obj.H if cfg.get("cond_past") else 0),
        ).to(device)
    denoiser.load_state_dict(ckpt["model_state_dict"])
    denoiser.eval()
    return denoiser, prior_obj, schedule, cfg, ckpt.get("epoch")


# ─── the test eval ───────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(dataset, prior, seed, *, exp, device, k, n_steps, alpha,
             w1_windows, w1_samples, w1_proj, gen_batch, max_windows, generator,
             backbone="gwn", mask_zero=True, split="test", sampler="em",
             ckpt_name="best.pt", save_diagnostics=False,
             pit_bins=20, n_examples=6, n_example_sensors=4,
             spread_scale=1.0, fit_spread_grid=None, anc_c_lo=None, common_mode="learned"):
    denoiser, prior_obj, schedule, cfg, ckpt_epoch = load_run(
        dataset, prior, seed, exp, device, backbone, ckpt_name=ckpt_name)
    N, H = prior_obj.N, prior_obj.H
    # residuals/sigma/dataset are keyed by the STAGE-1 seed (see load_run); the diffusion
    # seed still names the checkpoint dir, the eval output dir and the sampling RNG.
    res_seed = cfg.get("residual_seed", seed)

    if n_steps is None:
        # ancestral: the count training derived for this prior (constant resolution per
        # decade of noise); em: its historical 1000
        n_steps = int(cfg.get("anc_steps_derived", 50)) if sampler == "ancestral" else 1000
        log.info(f"[eval] n_steps not given -> {n_steps} ({'derived' if sampler == 'ancestral' else 'default'})")
    if sampler == "em":
        # EM stability floor: explicit EM NaNs when the step count is too low for this
        # prior's stiffness (θ_max·S). the ancestral sampler has no such floor.
        floor = min_em_steps(prior_obj, schedule)
        if n_steps < floor:
            log.info(f"[eval] EM stability floor: n_steps {n_steps} → {floor} (θ_max·S)")
            n_steps = floor

    import time
    from src.utils.logging_setup import Throttle

    # reverse process: GAD's Euler–Maruyama SDE (headline, GAD parity) or the
    # closed-form ancestral sampler (exact for the linear SDE, ~20x fewer steps).
    # progress: ONE self-contained line every ~3 min for the whole eval (the
    # throttle is shared by the reverse steps and the chunk loop below).
    thr = Throttle(180.0)

    def _reverse(Y_cond, tag=""):
        def _cb(i, _c, _R):
            if thr.ready():
                log.info(f"  [eval {dataset}/{prior}] {tag} · {sampler} step "
                         f"{i + 1}/{n_steps}")
        fn = reverse_sample_em if sampler == "em" else reverse_sample_ancestral
        kw = {} if (sampler == "em" or anc_c_lo is None) else {"c_lo": anc_c_lo}
        if sampler != "em" and common_mode != "learned":
            kw["common_mode"] = common_mode
        s, _ = fn(prior_obj, schedule, denoiser, Y_cond,
                  n_steps=n_steps, R_T=None, generator=generator,
                  step_callback=_cb, **kw)
        return s

    # same residual scale + dataset flags the run trained with (from config.json)
    sigma_r = train_residual_sigma(dataset, res_seed, backbone, norm=cfg.get("norm", "global"))
    sig = sigma_to_torch(sigma_r, device)                     # (1,1) or (N,1), broadcasts
    ds = make_dataset(dataset, split, res_seed, backbone, sigma_r,
                      cond_past=bool(cfg.get("cond_past")),
                      zero_masked=bool(cfg.get("zero_masked_input")))
    R0_all, Y_all, M_all = ds.R0, ds.Y_hat, ds.mask
    Ytrue_all = ds.y_true_raw                                 # raw ground truth (y_pred+residual)
    # match SpecSTG/DiffSTG: drop y==0 points from the metrics. for the speed datasets
    # the stored mask already excludes zeros (null_val=0); for the flow datasets
    # (null_val=nan) the stored mask keeps them, so this is where it bites.
    if mask_zero:
        M_all = M_all & (Ytrue_all.abs() > 1e-6)
    n_total = R0_all.shape[0] if not max_windows else min(max_windows, R0_all.shape[0])

    # cap trajectories per reverse_sample call so big-N datasets (pemsbay N=325)
    # don't OOM: process chunk windows at a time with chunk*k <= gen_batch.
    chunk = max(1, gen_batch // k)

    nchunks = (n_total + chunk - 1) // chunk
    log.info(f"[eval] {dataset}/{prior}: {n_total} {split} windows, k={k}, chunk={chunk}, "
             f"{nchunks} chunks, n_steps={n_steps}")
    t0 = time.time()

    # ── point accuracy + Winkler: K samples per window, chunked over windows ──
    mae_n = torch.zeros(H); mse_n = torch.zeros(H); cnt = torch.zeros(H)
    mape_n = torch.zeros(H); wink_n = torch.zeros(H); cov_n = torch.zeros(H); crps_n = torch.zeros(H)
    # JOINT metrics -- where a covariance prior is the only lever. per-point scores are
    # blind to correlation; anything that SUMS errors is not: the network total per
    # horizon and the hour-ahead cumulative per sensor. for each: fair CRPS, 90%
    # coverage, and predictive spread / actual error (1 = the sum's uncertainty is right;
    # white noise sits near 1/sqrt(N) by construction).
    jt = {"tot": dict(crps=0.0, cov=0.0, sp2=0.0, er2=0.0, n=0),
          "cum": dict(crps=0.0, cov=0.0, sp2=0.0, er2=0.0, n=0)}
    def _joint(key, S, T):                                   # S (k, ...) samples, T (...) truth
        a = jt[key]
        a["crps"] += float(empirical_crps(S, T).sum()); a["n"] += T.numel()
        lo_, hi_ = torch.quantile(S, alpha / 2, dim=0), torch.quantile(S, 1 - alpha / 2, dim=0)
        a["cov"] += float(((T >= lo_) & (T <= hi_)).float().sum())
        a["sp2"] += float(S.var(dim=0, unbiased=True).sum()); a["er2"] += float((T - S.mean(0)).pow(2).sum())
    qnum_n = torch.zeros(H); qden_n = torch.zeros(H)
    if fit_spread_grid is not None:
        fit_qnum = [torch.zeros(H) for _ in fit_spread_grid]
        fit_cov = [torch.zeros(1) for _ in fit_spread_grid]
    # ── calibration diagnostics (figure data; free to accumulate from the samples) ──
    #   PIT histogram per horizon (flat = calibrated) + reliability curve (empirical
    #   coverage of central intervals at many nominal levels) + a few example fan charts.
    diag_levels = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    ex_qprobs = torch.tensor([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    pit_hist = torch.zeros(H, pit_bins)
    reliab_hit = torch.zeros(len(diag_levels), H)
    examples = None
    gen_seconds = 0.0    # pure reverse-sampling wall time (excludes metric computation),
                         # the DiffSTG-comparable number: their csv 'time' = np.sum(time_lst)
                         # accumulated the same way around their sampler on this cluster.
    for ci, start in enumerate(range(0, n_total, chunk)):
        end = min(start + chunk, n_total)
        R0 = R0_all[start:end].to(device) * sig           # raw residual
        Y = Y_all[start:end].to(device)                   # conditioning (z-scored)
        Mk = M_all[start:end].to(device)
        yt = Ytrue_all[start:end].to(device)              # raw ground-truth signal
        yp = yt - R0                                      # raw backbone forecast (0 for 'raw')
        b = R0.shape[0]
        _t0 = time.time()
        samp = _reverse(Y.repeat(k, 1, 1), tag=f"point {ci + 1}/{nchunks}")
        if device.type == "cuda":
            torch.cuda.synchronize()
        gen_seconds += time.time() - _t0
        samp = samp.view(k, b, N, H) * sig                # raw residual samples
        # reconstruct the forecast and clip to [0,∞) (non-negative flow/speed; matches DiffSTG).
        # all metrics are now in raw FORECAST space (= residual space + y_pred, then clip).
        fc = (yp.unsqueeze(0) + samp).clamp(min=0.0)      # (k, b, N, H) raw forecast samples
        # ensemble spread recalibration (val-fitted, applied identically to every arm):
        # inflate/deflate around the ensemble mean; fixes the universal test-time
        # under-dispersion (val→test residual-variance growth up to 1.7x).
        if spread_scale != 1.0:
            _m = fc.mean(dim=0, keepdim=True)
            fc = (_m + spread_scale * (fc - _m)).clamp(min=0.0)
        if fit_spread_grid is not None:
            # score the WHOLE grid from this one set of samples: quantiles of the
            # scaled ensemble are the scaled quantiles, so lo/hi come for free.
            _m = fc.mean(dim=0, keepdim=True)
            _lo0 = torch.quantile(fc, alpha / 2.0, dim=0)
            _hi0 = torch.quantile(fc, 1.0 - alpha / 2.0, dim=0)
            for si, s in enumerate(fit_spread_grid):
                fc_s = (_m + s * (fc - _m)).clamp(min=0.0)
                fit_qnum[si] += quantile_crps_num(fc_s, yt, Mk).cpu()
                lo_s = (_m[0] + s * (_lo0 - _m[0])).clamp(min=0.0)
                hi_s = _m[0] + s * (_hi0 - _m[0])
                sn, _ = _masked_sums(((yt >= lo_s) & (yt <= hi_s)).float(), Mk)
                fit_cov[si] += sn.sum().cpu()
        mean = fc.mean(dim=0)
        sn, cn = _masked_sums((mean - yt).abs(), Mk);  mae_n += sn.cpu(); cnt += cn.cpu()
        sn, _ = _masked_sums((mean - yt).pow(2), Mk);  mse_n += sn.cpu()
        sn, _ = _masked_sums((mean - yt).abs() / yt.abs().clamp(min=1e-6), Mk);  mape_n += sn.cpu()
        lo = torch.quantile(fc, alpha / 2.0, dim=0)
        hi = torch.quantile(fc, 1.0 - alpha / 2.0, dim=0)
        sn, _ = _masked_sums(winkler_score(yt, lo, hi, alpha), Mk);  wink_n += sn.cpu()
        sn, _ = _masked_sums(((yt >= lo) & (yt <= hi)).float(), Mk);  cov_n += sn.cpu()
        sn, _ = _masked_sums(empirical_crps(fc, yt), Mk);  crps_n += sn.cpu()   # raw-unit energy CRPS
        _joint("tot", (fc * Mk.unsqueeze(0)).sum(2), (yt * Mk).sum(1))         # network total (k,b,H) / (b,H)
        _joint("cum", (fc * Mk.unsqueeze(0)).sum(3), (yt * Mk).sum(2))         # cumulative     (k,b,N) / (b,N)
        qnum_n += quantile_crps_num(fc, yt, Mk).cpu()                           # normalized-CRPS num
        sn, _ = _masked_sums(yt.abs(), Mk);  qden_n += sn.cpu()                 # normalized-CRPS denom

        if save_diagnostics:
            mk = Mk.bool()
            # PIT = empirical CDF of the samples evaluated at the truth (per masked point)
            pit = (fc <= yt.unsqueeze(0)).float().mean(0)                       # (b,N,H) in [0,1]
            for h in range(H):
                v = pit[:, :, h][mk[:, :, h]]
                if v.numel():
                    pit_hist[h] += torch.histc(v.float().cpu(), bins=pit_bins, min=0.0, max=1.0)
            # reliability: coverage of the central interval at each nominal level
            L = len(diag_levels)
            probs = torch.cat([(1 - diag_levels) / 2, 1 - (1 - diag_levels) / 2]).to(device)
            Q = torch.quantile(fc, probs, dim=0)                               # (2L,b,N,H)
            lo_L, hi_L = Q[:L], Q[L:]
            hit = ((yt.unsqueeze(0) >= lo_L) & (yt.unsqueeze(0) <= hi_L)).float()
            hit = hit * mk.unsqueeze(0).float()
            reliab_hit += hit.sum(dim=(1, 2)).cpu()                            # (L,H)
            # example fan charts from the FIRST chunk's first few windows/sensors
            if examples is None:
                ne = min(n_examples, b)
                sidx = torch.linspace(0, N - 1, min(n_example_sensors, N)).long()
                sub = fc[:, :ne][:, :, sidx]                                   # (k,ne,S,H)
                exq = torch.quantile(sub, ex_qprobs.to(device), dim=0)        # (nq,ne,S,H)
                examples = {
                    "ex_qprobs": ex_qprobs.numpy(),
                    "ex_quantiles": exq.cpu().numpy(),                        # (nq,ne,S,H)
                    "ex_ytrue": yt[:ne][:, sidx].cpu().numpy(),               # (ne,S,H)
                    "ex_yhat": yp[:ne][:, sidx].cpu().numpy(),                # backbone point fc
                    "ex_samples": fc[:min(30, k), :ne][:, :, sidx].cpu().numpy(),
                    "ex_mask": Mk[:ne][:, sidx].cpu().numpy().astype("uint8"),
                    "ex_sensor_idx": sidx.numpy(), "ex_window_idx": np.arange(ne),
                }

        if thr.ready() or ci == nchunks - 1:
            el = time.time() - t0
            remaining = el / (ci + 1) * (nchunks - ci - 1)   # don't shadow the sde `eta`
            log.info(f"  [eval {dataset}/{prior}] point {ci+1}/{nchunks} "
                     f"({100*(ci+1)/nchunks:.0f}%) · elapsed {el/60:.0f}m · eta {remaining/60:.0f}m")

    log.info(f"  [eval {dataset}/{prior}] point phase done in {(time.time()-t0)/60:.0f}m; W1/energy...")
    cnt = cnt.clamp(min=1.0)
    mae_h = (mae_n / cnt); mse_h = (mse_n / cnt); mape_h = (mape_n / cnt)
    wink_h = (wink_n / cnt); cov_h = (cov_n / cnt); crps_h = (crps_n / cnt)
    qcrps_h = qnum_n / (qden_n.clamp(min=1e-8) * 19.0)    # normalized quantile-CRPS per horizon
    overall = lambda num: float(num.sum() / cnt.sum())

    # ── distribution fidelity: complete-case windows, sliced-W1 + energy ──────
    valid = M_all.reshape(M_all.shape[0], -1).all(dim=1)
    vidx = valid.nonzero(as_tuple=True)[0]
    n_w1 = min(w1_windows, vidx.numel())
    sel = vidx[:n_w1]
    w1_frac = float(valid.float().mean())
    R0v = R0_all[sel].to(device) * sig; Yv = Y_all[sel].to(device)   # de-norm R0 to raw
    # sub-batch windows so w1_samples * sub <= gen_batch (same OOM guard)
    wb = max(1, gen_batch // w1_samples)
    n_w1_chunks = (n_w1 + wb - 1) // wb
    t1 = time.time()
    gen_parts = []
    for wi, s in enumerate(range(0, n_w1, wb)):
        e = min(s + wb, n_w1)
        g = _reverse(Yv[s:e].repeat(w1_samples, 1, 1), tag=f"W1 {wi + 1}/{n_w1_chunks}")
        gen_parts.append(g.view(w1_samples, e - s, N, H))
        if thr.ready() or wi == n_w1_chunks - 1:
            el = time.time() - t1
            remaining = el / (wi + 1) * (n_w1_chunks - wi - 1)
            log.info(f"  [eval {dataset}/{prior}] W1 {wi + 1}/{n_w1_chunks} "
                     f"({100 * (wi + 1) / n_w1_chunks:.0f}%) · elapsed {el/60:.0f}m · "
                     f"eta {remaining/60:.0f}m")
    gen = torch.cat(gen_parts, dim=1) * sig        # (w1_samples, n_w1, N, H), raw units
    w1_h, ed_h = [], []
    for h in range(H):
        true_c = R0v[:, :, h]
        gen_c = gen[:, :, :, h].reshape(w1_samples * n_w1, N)
        w1_h.append(sliced_wasserstein1(gen_c, true_c, w1_proj, generator))
        ed_h.append(energy_distance(gen_c, true_c))

    res = {
        "dataset": dataset, "backbone": backbone, "prior": prior, "seed": seed,
        "exp": exp, "ckpt_epoch": ckpt_epoch, "split": split, "ckpt": ckpt_name,
        "n_test_windows": int(n_total), "k_samples": k, "n_steps": n_steps,
        "sampler": sampler, "alpha": alpha, "mask_zero": bool(mask_zero),
        "common_mode": common_mode,
        # terminal noise floor of the ancestral grid. None = the sampler default
        # (0.02); an explicit value is recorded so a re-sampled run is never
        # confused with the locked-protocol one (it also renames the output).
        "anc_c_lo": (None if sampler == "em" else anc_c_lo),
        "mae": overall(mae_n), "mse": overall(mse_n), "rmse": float((mse_n.sum()/cnt.sum())**0.5),
        "mape": overall(mape_n),
        "winkler": overall(wink_n), "coverage": overall(cov_n),
        "crps_energy": overall(crps_n),                              # raw-unit fair/energy CRPS
        # joint: network total per horizon / hour-ahead cumulative per sensor
        **{f"joint_{k}_{m}": v for k, a in jt.items() for m, v in (
            ("crps", a["crps"] / max(a["n"], 1)), ("coverage", a["cov"] / max(a["n"], 1)),
            ("spread_ratio", float(np.sqrt(a["sp2"] / max(a["er2"], 1e-12)))))},
        "crps": float(qnum_n.sum() / (qden_n.sum().clamp(min=1e-8) * 19.0)),  # normalized (papers/Vimal)
        "mae_h": mae_h.tolist(), "mse_h": mse_h.tolist(), "mape_h": mape_h.tolist(),
        "winkler_h": wink_h.tolist(), "coverage_h": cov_h.tolist(),
        "crps_h": qcrps_h.tolist(), "crps_energy_h": crps_h.tolist(),
        "w1": float(np.mean(w1_h)), "energy": float(np.mean(ed_h)),
        "w1_h": w1_h, "energy_h": ed_h,
        "w1_clean_windows": int(n_w1), "w1_frac_of_test": w1_frac,
        # sampling speed (point-metrics pass only; W1's extra generation excluded).
        # s_per_window is at THIS k and n_steps — divide by k for per-sample cost.
        "gen_seconds": round(gen_seconds, 2),
        "s_per_window": round(gen_seconds / max(n_total, 1), 4),
        "spread_scale": spread_scale,
    }
    if fit_spread_grid is not None:
        tot_cnt = float(cnt.sum())
        fit_crps = [float(q.sum() / (qden_n.sum().clamp(min=1e-8) * 19.0)) for q in fit_qnum]
        best_i = int(np.argmin(fit_crps))
        res["spread_fit"] = {
            "grid": [float(s) for s in fit_spread_grid],
            "crps": fit_crps,
            "coverage": [float(c.item()) / tot_cnt for c in fit_cov],
            "best": float(fit_spread_grid[best_i]),
            "best_crps": fit_crps[best_i],
        }
    if save_diagnostics:
        # reliability: empirical central-interval coverage vs nominal (per horizon + overall)
        reliab_cov = (reliab_hit / cnt).tolist()                         # (L,H)
        reliab_overall = (reliab_hit.sum(1) / cnt.sum()).tolist()        # (L,)
        pit_all = pit_hist.sum(0)                                        # (pit_bins,)
        res["_diag"] = {
            "pit_hist": pit_hist.numpy(),                # (H, pit_bins)
            "pit_hist_all": pit_all.numpy(),             # (pit_bins,)
            "pit_bins": np.array([pit_bins]),
            "reliab_levels": diag_levels.numpy(),        # (L,)
            "reliab_cov_h": np.array(reliab_cov),        # (L, H)
            "reliab_cov": np.array(reliab_overall),      # (L,)
            "dataset": np.array([dataset]), "prior": np.array([prior]),
            "exp": np.array([exp]), "sampler": np.array([sampler]),
            "ckpt": np.array([ckpt_name]), "n_steps": np.array([n_steps]),
            **(examples or {}),
        }
    return res


def _save(res):
    out = paths.eval_dir(res["dataset"], res["prior"], res["seed"], exp=res.get("exp"),
                         backbone=res.get("backbone", "gwn"))
    out.mkdir(parents=True, exist_ok=True)
    # per-(split, ckpt, sampler, steps) filename so parallel evals never clobber
    ck = "bestcrps" if res.get("ckpt") == "best_crps.pt" else "best"
    stem = f"{res.get('split', 'test')}_metrics_{ck}_{res.get('sampler', 'em')}{res['n_steps']}"
    # a non-default ancestral floor is a DIFFERENT protocol, so it gets its own
    # name; the locked-protocol filenames must stay byte-identical.
    if res.get("anc_c_lo") is not None:
        stem += f"_clo{res['anc_c_lo']:g}"
    if res.get("common_mode", "learned") != "learned":                     # common-mode rule = its own protocol
        stem += f"_cm{res['common_mode']}"
    # recalibrated / fit runs get their own names so they NEVER clobber the raw eval.
    if res.get("spread_fit") is not None:
        stem += "_sfit"
    elif res.get("spread_scale", 1.0) != 1.0:
        stem += "_srecal"
    # calibration/example figure data (large arrays) go in a companion .npz, not the json
    diag = res.pop("_diag", None)
    if diag is not None:
        np.savez_compressed(out / f"{stem.replace('_metrics_', '_diag_')}.npz", **diag)
    path = out / f"{stem}.json"
    path.write_text(json.dumps(res, indent=2))
    return path


def _print_one(res):
    print(f"\n=== {res['dataset']}/{res['prior']} (best.pt epoch {res['ckpt_epoch']}, "
          f"{res['n_test_windows']} {res.get('split', 'test')} windows, K={res['k_samples']}, "
          f"sampler={res.get('sampler','em')}, n_steps={res['n_steps']}) ===")
    print(f"  MAE={res['mae']:.4f}  RMSE={res['mse']**0.5:.4f}  MAPE={res['mape']:.4f}  "
          f"Winkler{int((1-res['alpha'])*100)}={res['winkler']:.4f}  coverage={res['coverage']:.3f}")
    print(f"  CRPS(norm)={res['crps']:.4f}  CRPS(energy)={res.get('crps_energy', float('nan')):.4f}  "
          f"W1={res['w1']:.4f}  energy={res['energy']:.4f}  "
          f"({res['w1_clean_windows']} clean windows = {res['w1_frac_of_test']:.2f} of test)")
    hl = "  ".join(f"h{h}:MAE={res['mae_h'][h-1]:.2f}" for h in _HL if h <= len(res['mae_h']))
    print(f"  per-horizon  {hl}")


def _print_comparison(results):
    by = {(r["dataset"], r["prior"]): r for r in results}
    print("\n" + "=" * 78)
    print("KRONECKER vs ISOTROPIC on TEST (lower=better; coverage target = 1-alpha)")
    print("=" * 78)
    print(f"{'dataset':9s} {'metric':9s} {'kronecker':12s} {'isotropic':12s} {'winner':8s} {'Δ%':>7s}")
    for ds in ["metrla", "pems08f", "pemsbay"]:
        rk, ri = by.get((ds, "kronecker")), by.get((ds, "isotropic"))
        if not (rk and ri):
            continue
        for metric, lower in [("mae", True), ("mse", True), ("winkler", True),
                              ("coverage", None), ("w1", True), ("energy", True)]:
            kv, iv = rk[metric], ri[metric]
            if lower is None:  # coverage: closer to (1-alpha) wins
                tgt = 1 - rk["alpha"]
                win = "kron" if abs(kv - tgt) < abs(iv - tgt) else "iso"
                delta = ""
            else:
                win = "kron" if kv < iv else "iso"
                delta = f"{100*(kv-iv)/iv:+.1f}" if iv else ""
            print(f"{ds:9s} {metric:9s} {kv:<12.4f} {iv:<12.4f} {win:8s} {delta:>7s}")
        print("-" * 60)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=("metrla", "pemsbay", "pems04f", "pems04s",
                                          "pems08f", "pems08s"))
    p.add_argument("--prior", choices=("kronecker", "isotropic"))
    p.add_argument("--backbone", choices=("gwn", "gcn", "mlp"), default="gwn",
                   help="which backbone's diffusion runs to eval (namespaces results)")
    p.add_argument("--exp", default="dev", help="experiment tag (matches training --exp)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", choices=("test", "val"), default="test",
                   help="which split to evaluate (val = the pre-test protocol)")
    p.add_argument("--all", action="store_true", help="eval all datasets x both priors")
    p.add_argument("--k", type=int, default=8, help="samples per window (Vimal uses 8)")
    p.add_argument("--n-steps", type=int, default=None,
                   help="reverse steps. default: ancestral -> the count training derived for this "
                        "prior (config anc_steps_derived, constant resolution per noise decade); "
                        "em -> 1000 (stiff priors need many; GAD uses ~5000)")
    p.add_argument("--alpha", type=float, default=0.1, help="0.1 -> central 90%% interval")
    p.add_argument("--w1-windows", type=int, default=512)
    p.add_argument("--w1-samples", type=int, default=4)
    p.add_argument("--w1-proj", type=int, default=128)
    p.add_argument("--gen-batch", type=int, default=1024,
                   help="max trajectories per reverse_sample call (caps gpu memory)")
    p.add_argument("--max-windows", type=int, default=0, help="cap test windows (0=all)")
    p.add_argument("--no-mask-zero", dest="mask_zero", action="store_false",
                   help="keep y==0 points in the metrics (default: drop them, SpecSTG/DiffSTG convention)")
    p.add_argument("--sampler", choices=("em", "ancestral"), default="em",
                   help="em = GAD parity (needs >= theta_max*S steps); ancestral = "
                        "closed-form OU posterior, exact for the linear SDE, ~100 steps")
    p.add_argument("--ckpt", choices=("best.pt", "best_crps.pt"), default="best.pt",
                   help="which checkpoint to evaluate (val-loss or sweep-CRPS selection)")
    p.add_argument("--common-mode", choices=("learned", "gauss"), default="learned",
                   help="ancestral only: x0_hat source for the network-wide spatial mode -- the "
                        "denoiser (learned) or the prior's optimal filter (gauss). see reverse.py")
    p.add_argument("--anc-c-lo", type=float, default=None,
                   help="terminal noise floor of the ancestral grid (default 0.02). the "
                        "grid is geometric from c_0 down to this value and the sampler "
                        "returns R0_hat there, discarding the noise still present. the "
                        "variance-weighted fraction still un-denoised, "
                        "sum_m var_m (1-e^{-2 theta_m c_lo}) / sum_m var_m, is 3.15%% for "
                        "the isotropic prior in every cell (theta=0.8 flat) but 19-28%% for "
                        "the structured one, so the default floor resolves the two arms to "
                        "fidelities differing by 6-9x. training draws go down to 8e-4, so "
                        "that is the matched value. non-default values write their own "
                        "_clo<value> filename and never clobber the locked protocol.")
    p.add_argument("--spread-scale", type=float, default=1.0,
                   help="ensemble spread recalibration: fc <- mean + s*(fc-mean). 1.0 = off.")
    p.add_argument("--auto-spread", default=None,
                   help="comma grid, e.g. '1.0,1.2,1.4,1.6,1.8': fit s on VAL (min CRPS), "
                        "then evaluate --split at the best s. applied uniformly to every arm.")
    p.add_argument("--save-diagnostics", action="store_true",
                   help="also dump calibration + example figure data to a companion "
                        "<split>_diag_*.npz (PIT histogram, reliability curve, example fan charts)")
    args = p.parse_args()

    setup_logging(None)        # standalone: timestamped lines to stdout (traineval configures its own)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    g = None if device.type == "mps" else torch.Generator(device=device.type).manual_seed(args.seed)

    if args.all:
        jobs = [(ds, pr) for ds in ("metrla", "pemsbay", "pems04f", "pems04s",
                                    "pems08f", "pems08s")
                for pr in ("kronecker", "isotropic")]
    else:
        if not (args.dataset and args.prior):
            p.error("give --dataset and --prior, or --all")
        jobs = [(args.dataset, args.prior)]

    print(f"[eval] device={device}  sampler={args.sampler}  ckpt={args.ckpt}  "
          f"k={args.k}  n_steps={args.n_steps}")
    results = []
    for ds, pr in jobs:
        # per-run eval.log INSIDE the dataset's eval folder (mirrors train.log), next to
        # the metrics json — foldered under <exp>/<arm>/<ds>/<backbone>/<prior>/seed<s>/eval/
        # so each run's progress + final numbers are easy to find, not just in the flat
        # slurm log. one file per (split, ckpt, sampler, steps) so parallel EM/anc don't clash.
        ed = paths.eval_dir(ds, pr, args.seed, exp=args.exp, backbone=args.backbone)
        ed.mkdir(parents=True, exist_ok=True)
        ck = "bestcrps" if args.ckpt == "best_crps.pt" else "best"
        clo_tag = "" if args.anc_c_lo is None else f"_clo{args.anc_c_lo:g}"
        logf = ed / f"{args.split}_eval_{ck}_{args.sampler}{args.n_steps}{clo_tag}.log"
        setup_logging(logf)     # stdout (slurm .out) AND this run's eval.log
        log.info(f"=== eval {ds}/{pr} · exp={args.exp} · {args.split} · sampler={args.sampler} "
                 f"n_steps={args.n_steps} · ckpt={args.ckpt} · k={args.k} · device={device} ===")
        common = dict(exp=args.exp, device=device, k=args.k,
                      n_steps=args.n_steps, alpha=args.alpha,
                      w1_windows=args.w1_windows, w1_samples=args.w1_samples,
                      w1_proj=args.w1_proj, gen_batch=args.gen_batch,
                      max_windows=args.max_windows, generator=g, backbone=args.backbone,
                      mask_zero=args.mask_zero, sampler=args.sampler,
                      ckpt_name=args.ckpt, anc_c_lo=args.anc_c_lo, common_mode=args.common_mode)
        spread = args.spread_scale
        if args.auto_spread:
            # phase 1: fit the spread scalar on VAL (grid scored from one sample set),
            # phase 2 below evaluates the target split at the best s. no test leakage.
            grid = [float(x) for x in args.auto_spread.split(",")]
            fit = evaluate(ds, pr, args.seed, split="val", spread_scale=1.0,
                           fit_spread_grid=grid, save_diagnostics=False, **common)
            _save(fit)
            spread = fit["spread_fit"]["best"]
            log.info(f"[auto-spread] val-fitted s={spread}  "
                     f"(grid crps: {['%.4f' % c for c in fit['spread_fit']['crps']]})")
        res = evaluate(ds, pr, args.seed, split=args.split, spread_scale=spread,
                       save_diagnostics=args.save_diagnostics, **common)
        _print_one(res)
        # record the headline numbers in eval.log too (_print_one only hits stdout)
        log.info(f"[done] {ds}/{pr} MAE={res['mae']:.4f} RMSE={res['rmse']:.4f} "
                 f"CRPS={res['crps']:.4f} Winkler={res['winkler']:.4f} cov={res['coverage']:.3f} "
                 f"W1={res['w1']:.4f} energy={res['energy']:.4f} | JOINT total: CRPS={res['joint_tot_crps']:.2f} "
                 f"cov={res['joint_tot_coverage']:.3f} spread/err={res['joint_tot_spread_ratio']:.2f} | "
                 f"cumulative: CRPS={res['joint_cum_crps']:.2f} cov={res['joint_cum_coverage']:.3f} "
                 f"spread/err={res['joint_cum_spread_ratio']:.2f}")
        saved = _save(res)
        log.info(f"[saved] {saved}")
        print(f"  wrote {saved}")
        results.append(res)

    if len(results) > 1:
        _print_comparison(results)


if __name__ == "__main__":
    main()
