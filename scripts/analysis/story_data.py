"""Collect the verified numbers behind the interactive thesis story page into results/analysis/story_data.json.
sources: results/analysis/prior/<ds>.json, results/20260907_matched/**/test_metrics_*.json, matched configs,
results/analysis/gaussian_baseline.json, the draft's published-baseline table, the chain diagnostic json,
and the archived 20260728_histcond test evals (old protocol). nothing is typed in by hand except the chain
numbers of the first (killed) run, which are copied from its log."""
import json, glob, re, collections, numpy as np
from pathlib import Path
DS=["pems08f","pems08s","pems04f","pems04s","metrla","pemsbay"]
out={"datasets":DS,"names":{"pems08f":"PEMS08 flow","pems08s":"PEMS08 speed","pems04f":"PEMS04 flow","pems04s":"PEMS04 speed","metrla":"METR-LA","pemsbay":"PEMS-BAY"}}
out["prior"]={}
for ds in DS:
    p=Path(f"results/analysis/prior/{ds}.json")
    if not p.exists(): continue
    j=json.load(open(p))
    lam=np.array(j["B1_spectral"]["lambda"]); bins=np.linspace(0,2,25); idx=np.clip(np.digitize(lam,bins)-1,0,23); xc=(0.5*(bins[1:]+bins[:-1])).round(3).tolist()
    def binned(v): v=np.array(v); return [float(v[idx==b].mean()) if (idx==b).any() else None for b in range(24)]
    B1=j["B1_spectral"]
    out["prior"][ds]={"N":j["N"],"beta":j["beta"],"gamma":round(j["gamma"],4),"match_ratio_train":round(j["match_ratio_train"],3),
        "delta_h":[round(x,3) for x in j["A1_delta_h_train"]],"corr_hops":[round(x,4) for x in j["A2_corr_by_hops"]["train"]],
        "common_share":{k:round(v,4) for k,v in j["A3_common_share"].items() if k in ("train","test","independent")},
        "total_sd":{"train":[round(x,1) for x in j["A4_total_sd"]["train"]],"test":[round(x,1) for x in j["A4_total_sd"]["test"]],"independent":round(j["A4_total_sd"]["independent"],1)},
        "kurtosis_median":round(j["A5_tails"]["kurtosis_median"],2),"var_ratio_test_train":round(j["A6_test_train_var_ratio"]["overall"],2),
        "spectrum":{"x":xc,"data_test":binned(B1["data_test"]),"dc":{"data":B1["data_test"][0],**{k:B1[k][0] for k in ("iso","road08","matched","temporal")}},**{k:binned(B1[k]) for k in ("iso","road08","matched","temporal")}},
        "multiscale":{"scales":j["B4_multiscale_ratio"]["scales"],**{k:[round(x,3) for x in j["B4_multiscale_ratio"][k+"_test"]] for k in ("iso","temporal","spatial","road08","matched")}},
        "nll_test":{k:round(v["nll_test"],4) for k,v in j["B5_nll"].items()},"separability":round(j["B3_separability"]["kron_rank1_fraction"],3),
        "match":{"ratio_test":round(j["B7_match"]["ratio_test"],3),"nb_corr_prior":round(j["B7_match"]["neighbour_corr_prior"],3),"nb_corr_data":round(j["B7_match"]["neighbour_corr_data"],3)},
        "derived":{k:{kk:round(vv,3) for kk,vv in v.items()} for k,v in j["C2_derived_quantities"].items() if k in ("data_train","iso","matched","road08")}}
cfg={r["dataset"]+"/"+r["arm"]:r for r in json.load(open("results/analysis/cell_config_preview.json"))} if Path("results/analysis/cell_config_preview.json").exists() else {}
for f in glob.glob("results/20260907_matched/*/*/gwn/*/seed42/train/config.json"):
    c=json.load(open(f)); ds=c["dataset"]; arm="kron" if c["prior"]=="kronecker" else "iso"
    cfg.setdefault(ds+"/"+arm,{}).update({"dataset":ds,"arm":arm,"beta":c.get("global_edge_beta",0.0),"gamma":c.get("gamma_resolved",0.0),"c_lo":c["c_lo"],"anc_steps":c["anc_steps_derived"],"ratio":c.get("global_edge_match_ratio")})
out["config"]={k:{kk:v[kk] for kk in ("beta","gamma","theta_max","c_lo","anc_steps","f_at_0_02","ratio") if kk in v} for k,v in cfg.items()}
cells=collections.defaultdict(list)
for f in glob.glob("results/20260907_matched/*/*/gwn/*/seed*/eval/test_metrics_bestcrps_ancestral*.json"):
    m=json.load(open(f)); p=f.split("/"); ds=p[3]; arm="kron" if p[5]=="kronecker" else "iso"; seed=p[6]
    if f.endswith("_cmgauss.json"): arm+="+dc"
    cells[(ds,arm)].append({"seed":seed,**{k:round(m[k],5) for k in ("crps","mae","coverage","winkler","joint_tot_crps","joint_tot_coverage","joint_tot_spread_ratio","joint_cum_crps","joint_cum_coverage","joint_cum_spread_ratio") if m.get(k) is not None},"crps_h":[round(x,5) for x in m["crps_h"]],"coverage_h":[round(x,4) for x in m["coverage_h"]]})
out["cells"]={f"{ds}/{arm}":v for (ds,arm),v in cells.items()}
out["gaussian"]=json.load(open("results/analysis/gaussian_baseline.json"))
tex=open("tudelft_thesis_draft/mainmatter/5_experiments.tex").read(); pub={}
for name,key in (("PEMS04 flow","pems04f"),("PEMS04 speed","pems04s"),("PEMS08 flow","pems08f"),("PEMS08 speed","pems08s")):
    i=tex.find("\\emph{"+name+"}"); blk=tex[i:i+1800]; pub[key]={}
    for meth in ("TimeGrad","DiffSTG","SpecSTG","HySRD-Net"):
        m=re.search(meth+r"\s*&\s*(.+?)\\\\",blk)
        if m:
            vals=[float(x) for x in re.findall(r"[\d.]+",m.group(1).replace("\\textbf{","").replace("}",""))]; pub[key][meth]={"mae":vals[4],"crps":vals[8]}
out["published"]=pub
out["chain"]={"pems08f":{"iso":{"learned":{"spread":0.481,"cov":0.535},"gauss":{"spread":0.367,"cov":0.443},"hybrid":{"spread":0.479,"cov":0.533}},
                          "kron":{"learned":{"spread":0.576,"cov":0.600},"gauss":{"spread":0.987,"cov":0.844},"hybrid":{"spread":0.996,"cov":0.841}},"truth_total_sd":707}}
ck=Path("results/analysis/chain_common_mode_pems08f_s42_kron.json")
if ck.exists():
    j=json.load(open(ck))["kron"]; out["chain"]["pems08f"]["trace"]={m:[[round(c,5),round(s,1),round(x,1)] for c,s,x in j[m]["trace"]] for m in ("gauss","hybrid") if m in j}
old=collections.defaultdict(list)
for f in glob.glob("archive/20260902/results/20260728_histcond/*-hist/*/gwn/*/seed*/eval/test_metrics_*.json"):
    m=json.load(open(f)); p=f.split("/"); arm="kron" if p[4].startswith("emp") else "iso"; ds=p[5]
    if m.get("crps") is not None: old[(ds,arm)].append(m)
out["old"]={f"{ds}/{arm}":{"crps":round(float(np.mean([m["crps"] for m in v])),5),"coverage":round(float(np.mean([m["coverage"] for m in v])),4),"winkler":round(float(np.mean([m["winkler"] for m in v])),3),"n":len(v)} for (ds,arm),v in old.items()}
Path("results/analysis/story_data.json").write_text(json.dumps(out))
print("prior:",list(out["prior"]),"| cells:",{k:len(v) for k,v in out["cells"].items() if "+dc" in k},"| KB",Path("results/analysis/story_data.json").stat().st_size//1024)
