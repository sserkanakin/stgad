"""Choose (beta, gamma) so the prior puts the RIGHT spread on the network total
(ratio = 1), instead of by likelihood (which over-concentrates). Then report the
held-out fit at that choice vs the shipped gamma=0.8 road prior, so we know what
matching the total costs in fit. All offline, GWN cells."""
import sys, warnings; sys.path.insert(0,'.'); warnings.filterwarnings("ignore")
import numpy as np
from src.utils import paths
from src.data.residual_dataset import train_residual_sigma
from src.data.adjacency import load_adjacency, compute_normalized_laplacian
from src.diffusion.prior import load_prior, _fit_sigma_T, _normalize_scale, SHRINK_T
CAP=3000

def unit(ds,bb,split):
    d=np.load(paths.residual_npz(ds,split,42,bb)); R=d["residuals"].astype(np.float64); M=d["mask"].astype(bool)
    s=np.asarray(train_residual_sigma(ds,42,bb,norm="pernode")); R=R/s.reshape(-1,1)
    k=M.reshape(len(M),-1).all(1); R=R[k]; return R[np.linspace(0,len(R)-1,min(CAP,len(R))).astype(int)]
def modevar(R,VS,VT):
    b,n,h=R.shape; X=R.transpose(1,0,2).reshape(n,b*h)
    C=(VS.T@X).reshape(n,b,h).transpose(1,0,2).reshape(b*n,h)@VT; return (C**2).reshape(b,n,h).mean(0)
def nll(v,vh): v=np.maximum(v,1e-12); return 0.5*float(np.mean(np.log(v)+vh/v))
def sc(v,vtr): return v*(vtr.mean()/max(v.mean(),1e-12))

print(f"{'ds':9s}| {'beta*':>6s}{'gamma*':>8s}{'total ratio':>12s} | {'fit vs iso: g=0.8':>18s}{'matched':>9s}{'delta':>7s}")
print("-"*80)
for ds in ("pems08f","pems08s","pems04f","pems04s","metrla","pemsbay"):
    bb="gwn"
    sig=np.asarray(train_residual_sigma(ds,42,bb,norm="pernode"))
    dt=np.load(paths.residual_npz(ds,"test",42,bb)); Rt=dt["residuals"].astype(np.float64); Mt=dt["mask"].astype(bool)
    tot=np.where(Mt,Rt,0.0).sum(axis=1); ok=Mt.reshape(len(Mt),-1).mean(1)>0.5; truth=tot[ok].std(axis=0)
    # temporal factor once
    d=np.load(paths.residual_npz(ds,"train",42,bb)); R=d["residuals"].astype(np.float64); M=d["mask"].astype(bool)
    R=R/sig.reshape(-1,1); _,_,ST=_fit_sigma_T(R,M); muT,VT=np.linalg.eigh(ST); muT=np.clip(muT,1e-12,None)
    muT=(1-SHRINK_T)*muT+SHRINK_T*muT.mean()
    A0=load_adjacency(ds).astype(float); np.fill_diagonal(A0,0); N=A0.shape[0]; w=A0[A0>0].mean()
    best=None
    for beta in (0.003,0.01,0.03,0.1,0.3):
        A=A0+beta*w*(np.ones((N,N))-np.eye(N))
        lam,VS=np.linalg.eigh(compute_normalized_laplacian(A)); o=np.argsort(lam); lam,VS=lam[o],VS[:,o]
        for g in np.geomspace(1e-3,3,40):
            q=lam+g; tau2=N/(1/q).sum(); th=q[:,None]/(tau2*muT[None,:]); th,s2=_normalize_scale(th); v=s2/th
            r=np.mean([np.sqrt(sig@(VS*v[:,h])@VS.T@sig)/truth[h] for h in range(v.shape[1])])
            if best is None or abs(r-1)<abs(best[0]-1): best=(r,beta,g,lam,VS,v)
    r,beta,g,lam,VS,v=best
    tr,va=unit(ds,bb,"train"),unit(ds,bb,"val")
    vtr,vva=modevar(tr,VS,VT),modevar(va,VS,VT); base=nll(sc(np.ones_like(vtr),vtr),vva)
    n_match=base-nll(sc(v,vtr),vva)
    pk=load_prior(ds,"kronecker",42,backbone=bb,norm="pernode")
    vtr0,vva0=modevar(tr,pk.V_S,VT),modevar(va,pk.V_S,VT); base0=nll(sc(np.ones_like(vtr0),vtr0),vva0)
    n_08=base0-nll(sc(pk.sigma2,vtr0),vva0)
    print(f"{ds:9s}| {beta:6.3f}{g:8.4f}{r:12.2f} | {n_08:18.3f}{n_match:9.3f}{n_match-n_08:+7.3f}", flush=True)
