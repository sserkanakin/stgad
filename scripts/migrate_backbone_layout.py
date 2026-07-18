"""Migrate the old two-tree backbone layout into the new self-contained one.

OLD (scattered):
    RESULTS_ROOT/backbone/<ds>/<bb>/seed<s>/{best.pt,config.json,history.json}
    RESULTS_ROOT/residuals/<ds>/<bb>/seed<s>/{train,val,test}.npz
NEW (self-contained cell):
    RESULTS_ROOT/residuals/<ds>/<bb>/seed<s>/{train,val,test}.npz
                                           + model/{best.pt,config.json,history.json}

Run wherever the artifacts live (on the cluster before rsync, or locally after).
Idempotent: skips cells already migrated. Prints a plan; pass --apply to move.

usage:
    # on DAIC, before pulling:
    source scripts/daic/env.sh
    RESULTS_ROOT=$DAIC_DATA_DIR python scripts/migrate_backbone_layout.py --apply
    # or locally after rsyncing the old trees into ./results:
    python scripts/migrate_backbone_layout.py --apply
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import paths                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually move (default: dry-run)")
    args = ap.parse_args()

    old_root = paths.RESULTS / "backbone"
    if not old_root.exists():
        print(f"no old backbone/ tree at {old_root} — nothing to migrate "
              f"(already clean, or wrong RESULTS_ROOT).")
        return

    moved = skipped = 0
    for seed_dir in sorted(old_root.glob("*/*/seed*")):
        # backbone/<ds>/<bb>/seed<s>
        rel = seed_dir.relative_to(old_root)                      # <ds>/<bb>/seed<s>
        ds, bb, seedname = rel.parts
        seed = int(seedname.replace("seed", ""))
        dest = paths.backbone_dir(ds, seed, bb)                   # residuals/<ds>/<bb>/seed<s>/model
        files = [p for p in seed_dir.iterdir() if p.is_file()]
        if dest.exists() and any(dest.iterdir()):
            print(f"  skip  {rel}  (model/ already populated)")
            skipped += 1
            continue
        print(f"  move  {rel}/{{{','.join(p.name for p in files)}}}  ->  "
              f"{dest.relative_to(paths.RESULTS)}/")
        if args.apply:
            dest.mkdir(parents=True, exist_ok=True)
            for p in files:
                shutil.move(str(p), str(dest / p.name))
            moved += 1

    if args.apply:
        # remove the now-empty backbone tree
        for p in sorted(old_root.glob("*/*/seed*"), reverse=True):
            try:
                p.rmdir()
            except OSError:
                pass
        for p in sorted(old_root.glob("*/*"), reverse=True) + sorted(old_root.glob("*")):
            try:
                p.rmdir()
            except OSError:
                pass
        try:
            old_root.rmdir()
        except OSError:
            print(f"  note: {old_root} not empty (left non-model files); inspect manually")
        print(f"\nmigrated {moved} cells, skipped {skipped}. old backbone/ tree removed.")
    else:
        print(f"\nDRY-RUN: {moved if args.apply else 'N'} cells would move, {skipped} already done.")
        print("re-run with --apply to perform the moves.")


if __name__ == "__main__":
    main()
