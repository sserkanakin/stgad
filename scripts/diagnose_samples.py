"""are the model's samples distributionally correct? (Wasserstein / KL / covariance)

two checks, in GAD's empirical-cov-vs-target style:

  --check forward : does the FORWARD SDE reach the analytic stationary prior
                    Σ = Σ_T ⊗ Σ_S ? validates the NOISE MODEL (no trained net needed).
                    draws prior samples, compares empirical spatial/temporal covariance
                    to the analytic factors.

  --check reverse : do the REVERSE-generated residual samples match the REAL residuals?
                    THE real question — dispersion (coverage) + covariance structure.
                    loads best.pt, generates K samples on clean val windows, pools them,
                    and compares to the real residual cloud (and to the prior, for context).

metrics (raw residual units), between an empirical cloud and a reference:
  var_ratio  mean(diag C_gen) / mean(diag C_ref) — <1 = UNDER-dispersed (⇒ under-coverage),
             ~1 = well-spread. the single most telling number for the coverage issue.
  fro_rel    ||C_gen − C_ref||_F / ||C_ref||_F                (covariance shape error)
  gauss_w2   2-Wasserstein (Bures) between N(0,C_gen) and N(0,C_ref)
  gauss_kl   KL( N(0,C_gen) || N(0,C_ref) )
  sw1        sliced-Wasserstein-1 between the two node-vector clouds (distribution-free)

usage:
    python scripts/diagnose_samples.py --check forward --dataset metrla
    python scripts/diagnose_samples.py --check reverse --dataset metrla --prior kronecker \
        --backbone gwn --exp dev --seed 42 --k 100 --windows 256 --n-steps 1000
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.diffusion.forward_fcps import prior_to_torch, sample_R_t_from_c_bar_torch
from src.diffusion.prior import load_prior
from src.eval.distribution import sliced_wasserstein1


# ─── Gaussian distances between zero-mean N(0, C) ────────────────────────────

def _sym_sqrt(C: torch.Tensor) -> torch.Tensor:
    """symmetric PSD square root via eigendecomposition."""
    w, V = torch.linalg.eigh(0.5 * (C + C.T))
    w = w.clamp(min=0)
    return (V * w.sqrt()) @ V.T


def gaussian_w2(C_gen: torch.Tensor, C_ref: torch.Tensor) -> float:
    """2-Wasserstein (Bures) between N(0,C_gen) and N(0,C_ref) (zero means):
    W2² = Tr(C_gen + C_ref − 2 (C_ref^½ C_gen C_ref^½)^½)."""
    Rr = _sym_sqrt(C_ref)
    mid = _sym_sqrt(Rr @ C_gen @ Rr)
    w2sq = torch.trace(C_gen + C_ref - 2.0 * mid).clamp(min=0)
    return float(w2sq.sqrt())


def gaussian_kl(C_gen: torch.Tensor, C_ref: torch.Tensor, eps: float = 1e-6) -> float:
    """KL( N(0,C_gen) || N(0,C_ref) ). ridge-regularised for conditioning."""
    k = C_gen.shape[0]
    I = torch.eye(k, dtype=C_gen.dtype, device=C_gen.device)
    Cg = C_gen + eps * I
    Cr = C_ref + eps * I
    Cr_inv = torch.linalg.inv(Cr)
    tr = torch.trace(Cr_inv @ Cg)
    logdet = torch.logdet(Cr) - torch.logdet(Cg)
    return float(0.5 * (tr - k + logdet))


def _cov(rows: torch.Tensor) -> torch.Tensor:
    """cov of a (M, k) matrix of row-vectors (centred), symmetrised."""
    Xc = rows - rows.mean(dim=0, keepdim=True)
    C = (Xc.T @ Xc) / (rows.shape[0] - 1)
    return 0.5 * (C + C.T)


def spatial_rows(X: torch.Tensor) -> torch.Tensor:
    """(..., N, H) → (M, N) node-vector rows (each horizon/sample is one replicate)."""
    return X.reshape(-1, X.shape[-2], X.shape[-1]).permute(0, 2, 1).reshape(-1, X.shape[-2])


def temporal_rows(X: torch.Tensor) -> torch.Tensor:
    """(..., N, H) → (M, H) horizon-vector rows."""
    return X.reshape(-1, X.shape[-2], X.shape[-1]).reshape(-1, X.shape[-1])


def compare(gen: torch.Tensor, ref: torch.Tensor, *, n_proj: int, gen_=None) -> dict:
    """all metrics between a generated cloud and a reference cloud, both (..., N, H) raw.
    covariance / Gaussian metrics run on CPU float64 (MPS has no float64); the caller may
    pass GPU float32 tensors."""
    gen = gen.detach().to("cpu", torch.float64)
    ref = ref.detach().to("cpu", torch.float64)
    gs, rs = spatial_rows(gen), spatial_rows(ref)      # (Mg, N), (Mr, N)
    gt, rt = temporal_rows(gen), temporal_rows(ref)    # (Mg, H), (Mr, H)
    Cg_s, Cr_s = _cov(gs), _cov(rs)
    Cg_t, Cr_t = _cov(gt), _cov(rt)
    var_ratio = float(torch.diagonal(Cg_s).mean() / torch.diagonal(Cr_s).mean())
    # sliced-W1 on the node clouds (float32, cpu; matched sample counts by truncation)
    m = min(gs.shape[0], rs.shape[0])
    sw1 = sliced_wasserstein1(gs[:m].float(), rs[:m].float(), n_proj, None)
    return {
        "var_ratio": var_ratio,
        "spatial": {"fro_rel": float(torch.norm(Cg_s - Cr_s) / (torch.norm(Cr_s) + 1e-12)),
                    "gauss_w2": gaussian_w2(Cg_s, Cr_s), "gauss_kl": gaussian_kl(Cg_s, Cr_s)},
        "temporal": {"fro_rel": float(torch.norm(Cg_t - Cr_t) / (torch.norm(Cr_t) + 1e-12)),
                     "gauss_w2": gaussian_w2(Cg_t, Cr_t), "gauss_kl": gaussian_kl(Cg_t, Cr_t)},
        "sliced_w1": float(sw1),
    }


def _print(title: str, d: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"  var_ratio (gen/ref) = {d['var_ratio']:.3f}   "
          f"({'UNDER-dispersed' if d['var_ratio'] < 0.8 else 'OVER-dispersed' if d['var_ratio'] > 1.25 else 'OK'})")
    for k in ("spatial", "temporal"):
        s = d[k]
        print(f"  {k:8s}: fro_rel={s['fro_rel']:.3f}  W2={s['gauss_w2']:.3f}  KL={s['gauss_kl']:.3f}")
    print(f"  sliced_W1 (node clouds) = {d['sliced_w1']:.4f}")


# ─── forward check: prior sampler vs analytic Σ_T ⊗ Σ_S ──────────────────────

def check_forward(args, device, gen_) -> None:
    # small + exact: run entirely on CPU float64 (no GPU / MPS float64 issues).
    prior = load_prior(args.dataset, args.prior, args.seed, backbone=args.backbone)
    cpu = torch.device("cpu")
    prior_t = prior_to_torch(prior, device=cpu, dtype=torch.float64)
    N, H = prior.N, prior.H
    n = args.windows * max(1, args.k // 8)
    R0 = torch.zeros(n, N, H, dtype=torch.float64)
    c_hi = torch.full((n,), 5.0 / float(prior.theta.min()), dtype=torch.float64)
    samp = sample_R_t_from_c_bar_torch(R0, c_hi, prior_t, generator=None)   # ~ N(0, Σ_stat)

    # analytic marginal factor covariances read off the prior's eigenbasis:
    #   Σ_S = V_S diag(mean_h σ²_mode) V_Sᵀ ,  Σ_T = V_T diag(mean_i σ²_mode) V_Tᵀ
    VS = torch.as_tensor(prior.V_S, dtype=torch.float64)
    VT = torch.as_tensor(prior.V_T, dtype=torch.float64)
    s2 = torch.as_tensor(prior.sigma2, dtype=torch.float64)                 # (N,H) per-mode var
    Cs_ref = (VS * s2.mean(dim=1)) @ VS.T
    Ct_ref = (VT * s2.mean(dim=0)) @ VT.T
    Cs_gen = _cov(spatial_rows(samp)); Ct_gen = _cov(temporal_rows(samp))
    d = {"var_ratio": float(torch.diagonal(Cs_gen).mean() / torch.diagonal(Cs_ref).mean()),
         "spatial": {"fro_rel": float(torch.norm(Cs_gen - Cs_ref) / (torch.norm(Cs_ref) + 1e-12)),
                     "gauss_w2": gaussian_w2(Cs_gen, Cs_ref), "gauss_kl": gaussian_kl(Cs_gen, Cs_ref)},
         "temporal": {"fro_rel": float(torch.norm(Ct_gen - Ct_ref) / (torch.norm(Ct_ref) + 1e-12)),
                      "gauss_w2": gaussian_w2(Ct_gen, Ct_ref), "gauss_kl": gaussian_kl(Ct_gen, Ct_ref)},
         "sliced_w1": float("nan")}
    print(f"[forward] {args.dataset}/{args.prior}: {n} prior draws vs analytic Σ_T⊗Σ_S "
          f"(θ_min={float(prior.theta.min()):.3f})")
    _print("FORWARD prior vs analytic (noise-model sanity — expect fro_rel small, var_ratio~1)", d)


# ─── reverse check: generated residuals vs real residuals ────────────────────

def check_reverse(args, device, gen_) -> None:
    from src.data.residual_dataset import make_dataset, train_residual_sigma
    from src.diffusion.reverse import reverse_sample_em
    # heavy import path shared with eval
    try:
        from eval_diffusion import load_run
    except ModuleNotFoundError:
        from scripts.eval_diffusion import load_run

    denoiser, prior, schedule, cfg, ep = load_run(
        args.dataset, args.prior, args.seed, args.exp, device, args.backbone)
    sigma_r = train_residual_sigma(args.dataset, args.seed, args.backbone)
    ds = make_dataset(args.dataset, "val", args.seed, args.backbone, sigma_r)
    full = ds.mask.reshape(len(ds), -1).all(dim=1)
    idx = full.nonzero(as_tuple=True)[0][: args.windows]
    R0 = ds.R0[idx].to(device) * sigma_r                                   # real residuals (raw)
    Y = ds.Y_hat[idx].to(device)
    W, N, H = R0.shape

    # generate K samples per window (raw units)
    chunk = max(1, 1024 // args.k)
    gens = []
    for s in range(0, W, chunk):
        e = min(s + chunk, W)
        samp, _ = reverse_sample_em(prior, schedule, denoiser, Y[s:e].repeat(args.k, 1, 1),
                                    n_steps=args.n_steps, generator=gen_)
        gens.append(samp.view(args.k, e - s, N, H) * sigma_r)
    gen = torch.cat(gens, dim=1)                                           # (K, W, N, H) raw
    print(f"[reverse] {args.dataset}/{args.prior}/{args.backbone} best.pt(ep {ep}): "
          f"K={args.k} × {W} windows, {args.n_steps} EM steps")

    d_data = compare(gen, R0, n_proj=args.n_proj)
    _print("GENERATED vs REAL residuals (the real test — var_ratio<1 ⇒ under-coverage)", d_data)
    # context: also compare generated to the PRIOR (are samples following data or prior?).
    # prior draws on CPU float64 (MPS lacks float64); gen is moved to cpu inside compare().
    prior_t = prior_to_torch(prior, device=torch.device("cpu"), dtype=torch.float64)
    R0z = torch.zeros(gen.shape[1] * 4, N, H, dtype=torch.float64)
    c_hi = torch.full((R0z.shape[0],), 5.0 / float(prior.theta.min()), dtype=torch.float64)
    prior_draws = sample_R_t_from_c_bar_torch(R0z, c_hi, prior_t) * sigma_r
    d_prior = compare(gen, prior_draws, n_proj=args.n_proj)
    _print("GENERATED vs PRIOR (var_ratio~1 here ⇒ samples look like the prior, not the data)", d_prior)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", choices=("forward", "reverse"), default="reverse")
    p.add_argument("--dataset", required=True)
    p.add_argument("--prior", default="kronecker")
    p.add_argument("--backbone", default="gwn")
    p.add_argument("--exp", default="dev")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=100, help="samples/window for the reverse check")
    p.add_argument("--windows", type=int, default=256, help="clean val windows to use")
    p.add_argument("--n-steps", type=int, default=1000, help="EM reverse steps (auto-floored)")
    p.add_argument("--n-proj", type=int, default=256, help="projections for sliced-W1")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    gen_ = None if device.type == "mps" else torch.Generator(device=device.type).manual_seed(0)
    print(f"[diagnose] device={device}  check={args.check}")
    if args.check == "forward":
        check_forward(args, device, gen_)
    else:
        check_reverse(args, device, gen_)


if __name__ == "__main__":
    main()
