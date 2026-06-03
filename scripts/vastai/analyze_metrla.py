"""Diagnose why the metrla diffusion model trails baselines, using only saved
artifacts (no GPU). CENTER = point accuracy; SPREAD = dispersion vs truth."""
import json
import numpy as np

S = np.load("results/eval/metrla/samples_test.npz", allow_pickle=True)
R = np.load("results/residuals/gwn_metrla_test_seed0.npz", allow_pickle=True)
M = json.load(open("results/eval/metrla/metrics_test.json"))

# --- sample bank (12 windows) for bias/correlation of the predicted residual
samp = S["Y_sample"]            # (K, b, N, H)
yt_b = S["y_true"]; yh_b = S["y_hat"]; mk_b = S["mask"].astype(bool)
K, b, N, H = samp.shape
ens = samp.mean(0)             # (b,N,H)
add_resid = ens - yh_b         # residual the model adds to Y_hat
true_resid_b = yt_b - yh_b


def mae(a, c, m):
    return float(np.abs(a - c)[m].mean())


print(f"sample bank: K={K} windows={b} N={N} H={H}")
print("\n=== CENTER (point accuracy) ===")
print(f"  [full 512-win eval]  MAE gwn_point (Y_hat) : 3.4829   (from comparison table)")
print(f"  [full 512-win eval]  MAE diffusion ens     : {M['overall']['mae']:.4f}")
print(f"  [12-win bank]  MAE Y_hat                    : {mae(yh_b, yt_b, mk_b):.4f}")
print(f"  [12-win bank]  MAE diffusion ens-mean       : {mae(ens, yt_b, mk_b):.4f}")

# Does the predicted residual actually track the true residual, or just add noise?
a = add_resid[mk_b]; t = true_resid_b[mk_b]
corr = float(np.corrcoef(a, t)[0, 1])
print("\n=== Is the predicted residual signal or noise? ===")
print(f"  corr(added_resid, true_resid) : {corr:+.3f}   (0 = pure noise, 1 = perfect)")
print(f"  std(added_resid)              : {a.std():.3f}")
print(f"  std(true_resid)               : {t.std():.3f}")
print(f"  mean(added_resid)             : {a.mean():+.3f}   (bias)")

# --- SPREAD per horizon: model pred_std (full eval) vs true residual std (all 6831)
res = R["residuals"]; rmask = R["mask"].astype(bool)
model_std = M["per_horizon"]["pred_std"]
print("\n=== SPREAD per horizon (full data) ===")
print(" h   model_std   true_resid_std   ratio")
for h in range(H):
    mh = rmask[..., h]
    tstd = float(res[..., h][mh].std())
    ms = model_std[h]
    print(f"{h+1:2d}   {ms:8.3f}   {tstd:12.3f}   {ms/tstd if tstd else 0:5.2f}")

# finite-K Monte-Carlo error in the ensemble mean
mc = (samp.std(0) / np.sqrt(K))[mk_b].mean()
print(f"\n=== finite-K MC error in ens-mean at K={K}: {mc:.3f} per cell (≈ /2 at 4x K) ===")
