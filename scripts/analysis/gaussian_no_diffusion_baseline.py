"""GWN + a plain Gaussian residual, NO diffusion.

usage: RESULTS_ROOT=$PWD/archive/20260902/results python scripts/analysis/gaussian_no_diffusion_baseline.py

result 2026-09-06: 0.0681 (pems04f) / 0.0223 (pems04s) -- beats every published method
(0.0723 / 0.0241). iso-diffusion improves on this by 2-5%; the kron prior by +-1% on
top of iso. per-point CRPS on GWN is won by the backbone plus a standard deviation.

GWN + a plain Gaussian residual, NO diffusion: r_ih ~ N(0, sigma_i^2) with the
per-sensor TRAIN std. Scored with the thesis's normalised quantile-CRPS (19 quantiles,
divided by sum|y|, y==0 masked) so it is directly comparable to tab:exp_sota.
If this already beats the published numbers, per-point CRPS is not where the diffusion
-- let alone the prior -- earns its keep."""
import sys, warnings; sys.path.insert(0,'.'); warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import norm
from src.utils import paths
from src.data.residual_dataset import train_residual_sigma
QS=np.linspace(0.05,0.95,19)
print(f"{'dataset':9s}{'GWN+Gaussian (no diffusion)':>30s}{'iso diffusion':>15s}{'kron diffusion':>16s}{'best published':>16s}")
pub={"pems04f":0.0723,"pems04s":0.0241,"pems08f":None,"pems08s":None,"metrla":None,"pemsbay":None}
ours={"pems08f":(0.0497,0.0502),"pems08s":(0.0175,0.0173),"pems04f":(0.0663,0.0668),
      "pems04s":(0.0204,0.0206),"metrla":(0.0429,0.0438),"pemsbay":(0.0208,0.0211)}
for ds in ("pems08f","pems08s","pems04f","pems04s","metrla","pemsbay"):
    d=np.load(paths.residual_npz(ds,"test",42,"gwn")); r=d["residuals"].astype(np.float64); yp=d["y_pred"].astype(np.float64)
    m=d["mask"].astype(bool); yt=yp+r
    m&=(yt!=0)                                          # PEMS convention: drop y==0
    sig=np.asarray(train_residual_sigma(ds,42,"gwn",norm="pernode")).reshape(1,-1,1)
    num=0.0
    for q in QS:
        qhat=yp+sig*norm.ppf(q)
        pin=np.abs((qhat-yt)*((yt<=qhat).astype(float)-q))
        num+=2*(pin*m).sum()
    crps=num/(19*(np.abs(yt)*m).sum())
    i,k=ours[ds]; b=pub[ds]
    print(f"{ds:9s}{crps:30.4f}{i:15.4f}{k:16.4f}{(f'{b:.4f}' if b else '   (n/a here)'):>16s}")
