"""Where does a COVARIANCE prior win by construction? On aggregates.

usage:  DS=pems08s BB=gwn NW=128 RESULTS_ROOT=$PWD/archive/20260902/results \
        OUTPUT_ROOT=$PWD/archive/20260902/results python scripts/analysis/aggregate_metrics_check.py
needs the trained checkpoints under OUTPUT_ROOT/20260728_histcond/{iso,emp}-hist/. runs on
the Mac in ~2 min at 64 windows x K=16; 256 x 32 is DAIC territory.

CRPS scores every (sensor, horizon) point on its own and cannot see correlation.
Anything that SUMS errors -- the network total, the hour-ahead cumulative flow -- is
exactly where iso's independence assumption breaks: its spread on a sum of N terms
collapses by sqrt(N). Test both priors' trained checkpoints (pems08f / GWN, the
strongest backbone, where CRPS ties) on: per-point CRPS, the network-total error per
horizon, and the 12-step cumulative error per sensor. Corrected ancestral sampler,
fair CRPS. 'today' = the coded sampler at c_lo=0.02, for reference.
"""
import sys, warnings, json; sys.path.insert(0,'.'); warnings.filterwarnings("ignore")
import numpy as np, torch
from src.data.residual_dataset import make_dataset, train_residual_sigma, sigma_to_torch
from src.diffusion.forward_fcps import prior_to_torch
from src.diffusion.reverse import reverse_sample_ancestral, _rotate_to_eig_torch, _rotate_from_eig_torch
sys.path.insert(0, 'scripts'); from eval_diffusion import load_run

import os; DS, BB, SEED = os.environ.get("DS","pems08f"), os.environ.get("BB","gwn"), 42
NW, K, STEPS = int(os.environ.get("NW",64)), 16, 50
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(0)

def anc_corrected(prior, schedule, denoiser, Y, n_steps, c_lo):
    """reverse_sample_ancestral with the posterior-variance term restored."""
    V_S=torch.as_tensor(prior.V_S,device=dev,dtype=torch.float32); V_T=torch.as_tensor(prior.V_T,device=dev,dtype=torch.float32)
    theta=torch.as_tensor(prior.theta,device=dev,dtype=torch.float32).unsqueeze(0); s2=float(prior.sigma2_scale)
    v=s2/theta; B=Y.shape[0]
    C=torch.randn(B,prior.N,prior.H,device=dev)*v.sqrt()
    grid=np.geomspace(schedule.c_0,c_lo,n_steps+1)
    for i in range(n_steps+1):
        c=float(grid[i]); R_t=_rotate_from_eig_torch(C,V_S,V_T)
        t=torch.full((B,),float(schedule.t_of_c_bar(c)),device=dev)
        R0_hat=denoiser(R_t,t,Y)
        if i==n_steps: return R0_hat
        cp=float(grid[i+1])
        a_c=torch.exp(-theta*c); s_c=torch.clamp(s2*(1-torch.exp(-2*theta*c))/theta,min=1e-20)
        a_d=torch.exp(-theta*(c-cp)); s_d=torch.clamp(s2*(1-torch.exp(-2*theta*(c-cp)))/theta,min=1e-20)
        a_p=torch.exp(-theta*cp); s_p=torch.clamp(s2*(1-torch.exp(-2*theta*cp))/theta,min=1e-20)
        vp=s_d*s_p/(a_d*a_d*s_p+s_d)
        mean=vp*(a_d*C/s_d+a_p*_rotate_to_eig_torch(R0_hat,V_S,V_T)/s_p)
        sig_p2=1.0/(a_c*a_c/s_c+1.0/v)
        var=vp+(vp*a_p/s_p)**2*sig_p2
        C=mean+var.sqrt()*torch.randn(C.shape,device=dev)

def fair_crps(S, y):                      # S (K,...), y (...)
    Kk=S.shape[0]; t1=(S-y[None]).abs().mean(0)
    xs,_=torch.sort(S,0); idx=torch.arange(1,Kk+1,device=S.device,dtype=S.dtype)
    w=(2*idx-Kk-1)/(Kk*(Kk-1)); return t1-(w.view(-1,*([1]*y.ndim))*xs).sum(0)
def cov90(S,y):
    lo,hi=torch.quantile(S,0.05,dim=0),torch.quantile(S,0.95,dim=0); return ((y>=lo)&(y<=hi)).float().mean().item()
def spread(S): return (torch.quantile(S,0.95,dim=0)-torch.quantile(S,0.05,dim=0)).mean().item()

results={}
for arm,prior_name in (("iso","isotropic"),("kron","kronecker")):
    exp=f"20260728_histcond/{'iso' if arm=='iso' else 'emp'}-hist"
    den,pr,sch,cfg,_=load_run(DS,prior_name,SEED,exp,dev,backbone=BB,ckpt_name="best_crps.pt")
    sig_r=train_residual_sigma(DS,SEED,BB,norm=cfg.get("norm","pernode"))
    ds=make_dataset(DS,"test",SEED,BB,sig_r,cond_past=bool(cfg.get("cond_past")),zero_masked=True)
    sel=torch.linspace(0,len(ds)-1,NW).long()
    Y=ds.Y_hat[sel].to(dev); M=ds.mask[sel].to(dev); yt=ds.y_true_raw[sel].to(dev); yp=ds.y_pred_raw[sel].to(dev)
    sig=sigma_to_torch(sig_r,device=dev)
    for proto,(fn,clo) in {"today":(None,0.02),"fixed":(anc_corrected,8e-4)}.items():
        with torch.no_grad():
            if fn is None:
                s,_=reverse_sample_ancestral(pr,sch,den,Y.repeat(K,1,1),n_steps=STEPS,c_lo=clo)
            else:
                s=fn(pr,sch,den,Y.repeat(K,1,1),STEPS,clo)
        S=s.view(K,NW,pr.N,pr.H)*sig                          # raw residual samples (K,B,N,H)
        truth=(yt-yp)                                          # raw residual truth (B,N,H)
        m=M.float()
        # 1. per-point (what CRPS reports)
        crps_pt=((fair_crps(S,truth)*m).sum()/m.sum()).item()
        # 2. network TOTAL error per horizon (sum over sensors) -- what a planner sees
        S_tot=(S*m[None]).sum(2); T_tot=(truth*m).sum(1)     # (K,B,H),(B,H)
        # 3. hour-ahead CUMULATIVE error per sensor (sum over the 12 horizons)
        S_cum=(S*m[None]).sum(3); T_cum=(truth*m).sum(2)     # (K,B,N),(B,N)
        results[(arm,proto)]=dict(
            crps_point=crps_pt, cov_point=cov90(S,truth),
            crps_total=fair_crps(S_tot,T_tot).mean().item(), cov_total=cov90(S_tot,T_tot),
            spread_total=spread(S_tot), true_sd_total=T_tot.std().item(),
            crps_cum=fair_crps(S_cum,T_cum).mean().item(), cov_cum=cov90(S_cum,T_cum),
            spread_cum=spread(S_cum), true_sd_cum=T_cum.std().item())
        print(f"{arm:5s} {proto:6s} done", flush=True)

print(f"\n{DS} / {BB} (strongest backbone) · {NW} test windows · K={K} · {STEPS} steps\n")
for proto in ("today","fixed"):
    print(f"=== {proto.upper()}  ({'coded sampler, c_lo=0.02' if proto=='today' else 'corrected sampler, c_lo=8e-4'}) ===")
    print(f"{'':28s}{'iso':>10s}{'kron':>10s}{'kron better by':>16s}")
    a,b=results[("iso",proto)],results[("kron",proto)]
    def row(lab,key,lower=True,pct=True):
        x,y=a[key],b[key]; d=(x-y)/abs(x)*100 if lower else (y-x)*100
        print(f"{lab:28s}{x:10.3f}{y:10.3f}{d:+15.1f}{'%' if pct else 'pp'}")
    row("per-point CRPS (vehicles)","crps_point")
    row("per-point 90% coverage","cov_point",lower=False,pct=False)
    print(f"{'--- network TOTAL per horizon':28s}")
    row("  CRPS","crps_total")
    row("  90% coverage","cov_total",lower=False,pct=False)
    print(f"{'  90% band width vs truth sd':28s}{a['spread_total']/a['true_sd_total']:10.2f}{b['spread_total']/b['true_sd_total']:10.2f}     (calibrated ≈ 3.3)")
    print(f"{'--- hour-ahead CUMULATIVE/sensor':28s}")
    row("  CRPS","crps_cum")
    row("  90% coverage","cov_cum",lower=False,pct=False)
    print(f"{'  90% band width vs truth sd':28s}{a['spread_cum']/a['true_sd_cum']:10.2f}{b['spread_cum']/b['true_sd_cum']:10.2f}     (calibrated ≈ 3.3)")
    print()
json.dump({f"{k[0]}/{k[1]}":v for k,v in results.items()},open(f"results/analysis/aggregate_check_{DS}_{BB}.json","w"),indent=1)
