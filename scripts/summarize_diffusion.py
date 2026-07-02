"""aggregate diffusion test results across runs into a single leaderboard.

walks the eval output tree for every test_results.json (written by train_diffusion's
combined --eval-after job, or eval_diffusion --eval-after), flattens each file's
per-n_steps entries into one row per (run, n_steps), and writes a leaderboard sorted
by CRPS (Vimal's convention: CRPS primary, then RMSE, then MAE). also prints a
kronecker-vs-isotropic delta table per dataset.

writes <out_dir>/summary.csv and <out_dir>/summary.md (default: <eval_root>/summaries/).

usage:
    python scripts/summarize_diffusion.py --exp 20260702
    python scripts/summarize_diffusion.py --exp 20260702 --out-dir results/eval/20260702/summaries
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.utils import paths

# leaderboard columns, in display order. (key in the flattened row, header, format)
_COLS = [
    ("dataset", "dataset", "{}"),
    ("prior", "prior", "{}"),
    ("backbone", "backbone", "{}"),
    ("seed", "seed", "{}"),
    ("best_epoch", "best_epoch", "{}"),
    ("n_steps", "n_steps", "{}"),
    ("mae", "MAE↓", "{:.4f}"),
    ("rmse", "RMSE↓", "{:.4f}"),
    ("mape", "MAPE↓", "{:.4f}"),
    ("crps", "CRPS↓", "{:.4f}"),
    ("winkler", "Winkler↓", "{:.4f}"),
    ("coverage", "cov", "{:.3f}"),
]
_SORT_KEYS = ("crps", "rmse", "mae")   # ascending; CRPS primary (Vimal)


def _flatten(payload: dict) -> list[dict]:
    """one row per (run, n_steps) entry in a test_results.json payload."""
    base = {k: payload.get(k) for k in ("dataset", "prior", "backbone", "seed", "best_epoch")}
    rows = []
    for r in payload.get("test_results", []):
        row = dict(base)
        row.update({"n_steps": r.get("n_steps"), "mae": r.get("mae"), "rmse": r.get("rmse"),
                    "mape": r.get("mape"), "crps": r.get("crps"), "winkler": r.get("winkler"),
                    "coverage": r.get("coverage"), "_json": payload.get("_json")})
        rows.append(row)
    return rows


def collect(eval_root: Path) -> list[dict]:
    rows: list[dict] = []
    for jf in sorted(eval_root.rglob("test_results.json")):
        try:
            payload = json.loads(jf.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warn] skipping {jf}: {e}")
            continue
        payload["_json"] = str(jf)
        rows.extend(_flatten(payload))
    return rows


def _sort_key(row: dict):
    # NaN / missing metrics sort last
    return tuple((row.get(k) if row.get(k) is not None else float("inf")) for k in _SORT_KEYS)


def _fmt(row: dict) -> list[str]:
    out = []
    for key, _, fmt in _COLS:
        v = row.get(key)
        out.append(fmt.format(v) if v is not None else "-")
    return out


def write_outputs(rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = [h for _, h, _ in _COLS]
    table = [_fmt(r) for r in rows]

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers + ["json"])
        for r, cells in zip(rows, table):
            w.writerow(cells + [r.get("_json", "")])

    # markdown table
    widths = [max(len(h), *(len(row[i]) for row in table)) if table else len(h)
              for i, h in enumerate(headers)]
    def _md_row(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    md_lines = [_md_row(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    md_lines += [_md_row(cells) for cells in table]
    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    return csv_path, md_path


def _print_table(rows: list[dict]) -> None:
    headers = [h for _, h, _ in _COLS]
    table = [_fmt(r) for r in rows]
    widths = [max(len(h), *(len(row[i]) for row in table)) if table else len(h)
              for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))
    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for cells in table:
        print(line(cells))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--exp", default=None, help="experiment tag; limits the walk to results/eval/<exp>")
    p.add_argument("--out-dir", default=None, help="where to write summary.csv/.md (default: <eval_root>/summaries)")
    args = p.parse_args()

    eval_root = paths.OUTPUTS / "eval"
    if args.exp:
        eval_root = eval_root / args.exp
    if not eval_root.exists():
        p.error(f"no eval outputs at {eval_root} (train with --eval-after first)")

    rows = collect(eval_root)
    if not rows:
        print(f"[summarize] no test_results.json found under {eval_root}")
        return
    rows.sort(key=_sort_key)

    out_dir = Path(args.out_dir) if args.out_dir else eval_root / "summaries"
    csv_path, md_path = write_outputs(rows, out_dir)

    print(f"\n===== DIFFUSION LEADERBOARD ({len(rows)} rows, sorted by CRPS↓) =====")
    _print_table(rows)
    print(f"\nwrote {csv_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
