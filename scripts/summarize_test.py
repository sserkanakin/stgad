#!/usr/bin/env python3
"""aggregate every TEST-split eval into thesis-ready tables + figure data.

idempotent: re-run any time (or in a watch loop) and it picks up whatever
`*_metrics_*.json` files exist so far — the table auto-updates as jobs finish.
the raw per-run JSONs are the permanent record (every scalar AND per-horizon
array), so nothing here needs the GPU again.

walks   <root>/<exp>/<arm>/<ds>/<backbone>/<prior>/seed<seed>/eval/test_metrics_*.json
writes  <root>/<exp>/test_leaderboard.{csv,md}   one row per run, all scalar metrics
        <root>/<exp>/test_perhorizon.csv         long format (metric x horizon) for figures
        <root>/<exp>/test_kron_vs_iso.md         base-pair deltas per dataset (RQ2)

usage:  python scripts/summarize_test.py --exp 20260708_r2
        python scripts/summarize_test.py --root out --exp 20260708_r2   # ./out symlink
        watch -n120 python scripts/summarize_test.py --exp 20260708_r2  # live refresh
"""
import argparse, csv, glob, json, math, os
from pathlib import Path

# scalar metrics surfaced in the leaderboard (key -> header, lower-is-better unless noted)
SCALARS = [
    ("mae", "MAE"), ("rmse", "RMSE"), ("mape", "MAPE"),
    ("crps", "CRPS"), ("crps_energy", "CRPS_en"),
    ("winkler", "Winkler"), ("coverage", "cov"),  # cov: closer to 1-alpha is better
    ("w1", "W1"), ("energy", "energy"),
]
# per-horizon arrays to unroll for figures. rmse handled specially from mse_h.
PERH = ["mae_h", "mse_h", "mape_h", "winkler_h", "coverage_h",
        "crps_h", "crps_energy_h", "w1_h", "energy_h"]
HL = (3, 6, 12)  # highlighted horizons


def arm_of(payload, jf):
    # arm is the exp leaf: exp="20260708_r2/kron_base" -> "kron_base"
    exp = payload.get("exp", "")
    if "/" in exp:
        return exp.split("/")[-1]
    # fallback: derive from the path (.../<exp>/<arm>/<ds>/...)
    return "?"


def load(root: Path, exp: str):
    rows, perh = [], []
    pat = str(root / exp / "**" / "eval" / "test_metrics_*.json")
    files = sorted(glob.glob(pat, recursive=True))
    for jf in files:
        try:
            p = json.loads(Path(jf).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        arm = arm_of(p, jf)
        base = {"arm": arm, "dataset": p.get("dataset"), "prior": p.get("prior"),
                "sampler": p.get("sampler"), "ckpt": p.get("ckpt"),
                "n_steps": p.get("n_steps"), "k": p.get("k_samples"),
                "split": p.get("split"), "windows": p.get("n_test_windows"),
                "_json": jf}
        row = dict(base)
        for k, _ in SCALARS:
            row[k] = p.get(k)
        rows.append(row)
        # per-horizon long format
        H = len(p.get("mae_h", []) or [])
        for h in range(H):
            rec = {**base, "horizon": h + 1}
            for arr in PERH:
                v = (p.get(arr) or [None] * H)[h]
                name = arr[:-2]  # strip "_h"
                if name == "mse":
                    rec["rmse"] = math.sqrt(v) if v is not None else None
                else:
                    rec[name] = v
            perh.append(rec)
    return rows, perh, files


def write_leaderboard(root: Path, exp: str, rows):
    rows = sorted(rows, key=lambda r: (str(r["dataset"]), str(r["sampler"]),
                                       r["crps"] if r["crps"] is not None else 9e9))
    cols = ["arm", "dataset", "prior", "sampler", "ckpt", "n_steps", "k", "windows"] \
        + [k for k, _ in SCALARS]
    csv_path = root / exp / "test_leaderboard.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["json"])
        for r in rows:
            w.writerow([r.get(c) for c in cols] + [r["_json"]])
    # markdown
    md = ["# test leaderboard — " + exp, "",
          "one row per (arm, dataset, sampler). lower is better except `cov` "
          "(target = 1-alpha = 0.90). CRPS = normalized 19-quantile (paper convention).", ""]
    hdr = "| " + " | ".join(["arm", "dataset", "sampler", "steps"] +
                            [h for _, h in SCALARS]) + " |"
    md += [hdr, "|" + "---|" * (4 + len(SCALARS))]
    for r in rows:
        def fmt(k):
            v = r.get(k)
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        md.append("| " + " | ".join(
            [r["arm"], str(r["dataset"]), str(r["sampler"]), str(r["n_steps"])] +
            [fmt(k) for k, _ in SCALARS]) + " |")
    (root / exp / "test_leaderboard.md").write_text("\n".join(md) + "\n")
    return csv_path


def write_perhorizon(root: Path, exp: str, perh):
    if not perh:
        return None
    cols = ["arm", "dataset", "prior", "sampler", "ckpt", "horizon",
            "mae", "rmse", "mape", "winkler", "coverage", "crps", "crps_energy", "w1", "energy"]
    path = root / exp / "test_perhorizon.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(perh, key=lambda r: (str(r["dataset"]), str(r["arm"]),
                                             str(r["sampler"]), r["horizon"])):
            w.writerow(r)
    return path


def write_kron_vs_iso(root: Path, exp: str, rows):
    # base-pair RQ2 deltas: kron_base vs iso_base, per (dataset, sampler)
    by = {}
    for r in rows:
        if r["arm"] in ("kron_base", "iso_base"):
            by.setdefault((r["dataset"], r["sampler"]), {})[r["arm"]] = r
    md = ["# kron_base vs iso_base on TEST (RQ2) — " + exp, "",
          "Δ% = (kron - iso)/iso; negative = kron better (except coverage, "
          "where closer to 0.90 wins).", ""]
    for (ds, smp), pair in sorted(by.items()):
        k, i = pair.get("kron_base"), pair.get("iso_base")
        if not (k and i):
            continue
        md += [f"## {ds}  ({smp})", "",
               "| metric | kron_base | iso_base | Δ% |", "|---|---|---|---|"]
        for key, name in SCALARS:
            kv, iv = k.get(key), i.get(key)
            if kv is None or iv is None:
                continue
            if key == "coverage":
                d = f"kron {abs(kv-0.9):.3f} vs iso {abs(iv-0.9):.3f} from .90"
            else:
                d = f"{100*(kv-iv)/iv:+.1f}%" if iv else ""
            md += [f"| {name} | {kv:.4f} | {iv:.4f} | {d} |"]
        md.append("")
    (root / exp / "test_kron_vs_iso.md").write_text("\n".join(md) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", required=True, help="experiment tag, e.g. 20260708_r2")
    ap.add_argument("--root", default=os.environ.get("OUTPUT_ROOT", "out"),
                    help="output root holding <exp>/ (default $OUTPUT_ROOT or ./out)")
    a = ap.parse_args()
    root = Path(a.root)
    rows, perh, files = load(root, a.exp)
    if not files:
        print(f"[summarize_test] no test_metrics_*.json under {root/a.exp} yet")
        return
    lb = write_leaderboard(root, a.exp, rows)
    ph = write_perhorizon(root, a.exp, perh)
    write_kron_vs_iso(root, a.exp, rows)
    print(f"[summarize_test] {len(files)} run(s) -> {lb}")
    print(f"                 per-horizon -> {ph}")
    print(f"                 RQ2 deltas  -> {root/a.exp/'test_kron_vs_iso.md'}")


if __name__ == "__main__":
    main()
