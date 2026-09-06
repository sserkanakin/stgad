"""Automatic end-of-run summary for the reshape experiment (submit_reshape.sh).

Walks EXP/<arm>/<ds>/<backbone>/<prior>/seed*/train/validation_history.json and builds one
readable markdown summary: best sampled CRPS (+ MAE) per (dataset, arm), the best arm per
dataset, and the KEY contrast — does the whitened loss make the structured prior beat iso?

Runs standalone (stdlib only) so it can be a tiny no-GPU dependency job on DAIC.

usage: python scripts/analysis/summarize_reshape.py --exp /path/to/OUTPUT_ROOT/20260720_reshape
"""
import argparse
import json
from pathlib import Path

# arm -> (human label, group). groups order the columns.
ARMS = [
    ("ddpm",  "DDPM (baseline)",   "baseline"),
    ("iso",   "iso",               "snr"),
    ("emp",   "empirical",         "snr"),
    ("rsh",   "reshaped",          "snr"),
    ("iso_w", "iso + whitened",    "whitened"),
    ("emp_w", "empirical + whit.", "whitened"),
    ("rsh_w", "reshaped + whit.",  "whitened"),
]
DATASETS = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]
BACKBONES = ["mlp", "gcn", "gwn"]


def find_vh(exp: Path, arm: str, ds: str, bb: str):
    """the validation_history.json for one (arm, ds, backbone), whatever prior subdir it's in."""
    hits = sorted((exp / arm / ds / bb).glob("*/seed*/train/validation_history.json"))
    return hits[0] if hits else None


def read(vh: Path):
    try:
        d = json.loads(vh.read_text())
    except Exception:
        return None
    hist = d.get("history", [])
    ep = d.get("best_crps_epoch")
    row = hist[ep] if isinstance(hist, list) and ep is not None and ep < len(hist) else (hist[-1] if hist else {})
    return {"crps": d.get("best_crps"), "mae": row.get("val_mae"), "rmse": row.get("val_rmse"),
            "w1": row.get("val_w1"), "epoch": ep, "n_ep": len(hist)}


def fmt(v, p=4):
    return f"{v:.{p}f}" if isinstance(v, (int, float)) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="the EXP output dir (OUTPUT_ROOT/<EXP>)")
    ap.add_argument("--metric", default="crps", choices=("crps", "mae", "rmse"))
    args = ap.parse_args()
    exp = Path(args.exp)

    data = {}  # (ds, bb, arm) -> metrics
    for ds in DATASETS:
        for bb in BACKBONES:
            for arm, _, _ in ARMS:
                vh = find_vh(exp, arm, ds, bb)
                if vh:
                    data[(ds, bb, arm)] = read(vh)

    if not data:
        print(f"no runs found under {exp}"); return
    cells_present = [(ds, bb) for ds in DATASETS for bb in BACKBONES
                     if any((ds, bb, a) in data for a, _, _ in ARMS)]

    L = [f"# Reshape experiment summary — `{exp.name}`", ""]
    L.append("Best sampled **CRPS** per (dataset, backbone, arm). Lower is better. "
             "**bold** = best arm on that row.")
    L.append("")
    hdr = "| dataset | bb | " + " | ".join(lbl for _, lbl, _ in ARMS) + " |"
    L.append(hdr); L.append("|" + "---|" * (len(ARMS) + 2))
    for ds, bb in cells_present:
        vals = {arm: (data.get((ds, bb, arm)) or {}).get("crps") for arm, _, _ in ARMS}
        best = min((v for v in vals.values() if isinstance(v, (int, float))), default=None)
        cs = []
        for arm, _, _ in ARMS:
            v = vals[arm]; s = fmt(v)
            if isinstance(v, (int, float)) and best is not None and abs(v - best) < 1e-9:
                s = f"**{s}**"
            cs.append(s)
        L.append(f"| {ds} | {bb} | " + " | ".join(cs) + " |")

    # the key contrast: does whitening flip structured above iso? (per dataset × backbone)
    L += ["", "## Does the whitened loss make the structured prior beat iso?", "",
          "| dataset | bb | iso (best) | empirical snr → whit | reshaped snr → whit | structured beats iso? |",
          "|---|---|--:|--:|--:|:--|"]
    for ds, bb in cells_present:
        g = lambda a: (data.get((ds, bb, a)) or {}).get("crps")
        iso = min([x for x in (g("iso"), g("iso_w")) if isinstance(x, (int, float))], default=None)
        emp_s, emp_w = g("emp"), g("emp_w")
        rsh_s, rsh_w = g("rsh"), g("rsh_w")
        best_struct = min([x for x in (emp_w, rsh_w) if isinstance(x, (int, float))], default=None)
        verdict = "—"
        if isinstance(best_struct, (int, float)) and isinstance(iso, (int, float)):
            verdict = f"YES ({100*(iso-best_struct)/iso:+.1f}% vs iso)" if best_struct < iso else "no"
        L.append(f"| {ds} | {bb} | {fmt(iso)} | {fmt(emp_s)} → {fmt(emp_w)} | {fmt(rsh_s)} → {fmt(rsh_w)} | {verdict} |")

    n_done = sum(1 for v in data.values() if v and v.get("crps") is not None)
    L += ["", f"_runs found: {n_done}/{len(cells_present) * len(ARMS)} · metric = best per-epoch sampled "
          f"ancestral CRPS (val, EMA) · MAE/RMSE/W1 at that epoch in each validation_history.json._"]
    text = "\n".join(L) + "\n"
    out = exp / "SUMMARY.md"
    out.write_text(text)
    print(text)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
