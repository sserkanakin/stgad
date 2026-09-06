"""unit-free joint-calibration statistics shared by the DiffSTG scorer and ours.
S (K,B,N,H) samples, Y (B,N,H) truth, same units. K small (8) -> use the sample RANGE:
a calibrated model puts the truth inside [min,max] of K samples (K-1)/(K+1) of the time."""
import numpy as np
def stats(S, Y):
    K=S.shape[0]; nominal=(K-1)/(K+1)
    def range_cov(s,y): return float(((y>=s.min(0))&(y<=s.max(0))).mean())
    def ratio(s,y):                                   # predictive spread / actual error, of a SUM
        m=s.mean(0); err=y-m; return float(s.std(0).mean()/np.sqrt((err**2).mean()))
    out={"nominal":nominal, "pt_cov":range_cov(S,Y)}
    T,Yt=S.sum(2),Y.sum(1)                            # network total per horizon  (K,B,H),(B,H)
    C,Yc=S.sum(3),Y.sum(2)                            # cumulative per sensor      (K,B,N),(B,N)
    out.update(tot_cov=range_cov(T,Yt), tot_ratio=ratio(T,Yt), cum_cov=range_cov(C,Yc), cum_ratio=ratio(C,Yc))
    return out
def show(label,o):
    print(f"{label:28s} pt-cov {o['pt_cov']:.3f} (nom {o['nominal']:.3f}) | TOTAL cov {o['tot_cov']:.3f} spread/err {o['tot_ratio']:.2f} | CUM cov {o['cum_cov']:.3f} spread/err {o['cum_ratio']:.2f}")
