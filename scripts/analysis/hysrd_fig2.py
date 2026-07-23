"""Reproduce HySRD-Net's Fig. 2 (historical trend-residual decomposition) from
residual caches -- for their re-implemented ST-TrendNet ("hysrd" arm) and for
our own backbones (mlp/gcn/gwn).

their fig 2: black = ground truth, blue = historical trend (only the FIRST
forecast step of each sliding window, sec 4.3), orange = residual. panels:
pems04f nodes 120 (idx 6760) & 299 (idx 14134), pems08s nodes 154 (idx 3044)
& 16 (idx 16041); short panels span 1440 min, long ones 4500 min.

our caches store exactly this: y_pred[:, n, 0] over consecutive stride-1
windows IS the 1-step trend series, residuals[:, n, 0] the matching residual,
truth = trend + residual. flow splits cut the TIME axis 60/20/20 then window
each segment (anchors [11, T_seg-12)), so original index i maps to window
b = i - seg_start - 12. panels that cross a segment boundary are clipped
(node 299's 4500-min span starts 528 steps into test -> ~2645 min shown).

usage:
  .venv/bin/python scripts/analysis/hysrd_fig2.py --backbones gwn hysrd
"""
import argparse
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import paths                                          # noqa: E402

# (dataset, node, "current time" original index, span in steps) -- their fig 2
PANELS = [
    ("pems04f", 120, 6760, 288),
    ("pems04f", 299, 14134, 900),
    ("pems08s", 154, 3044, 288),
    ("pems08s", 16, 16041, 900),
]
# their visual language (also palette slots 1/2): truth black, trend blue,
# residual orange
C_TRUTH, C_TREND, C_RES = "#1a1a19", "#2a78d6", "#eb6834"


def segment_bounds(dataset, backbone):
    """segment start indices + window counts, derived from cache shapes
    (T_seg = B_seg + 23 for 12-in/12-out stride-1 windows)."""
    out, start = {}, 0
    for split in ("train", "val", "test"):
        B = np.load(paths.residual_npz(dataset, split, 42, backbone))["residuals"].shape[0]
        out[split] = (start, B)
        start += B + 23
    return out


def panel_series(dataset, backbone, node, idx_end, span):
    """1-step trend + residual series ending at original index idx_end."""
    for split, (seg_start, B) in segment_bounds(dataset, backbone).items():
        b_end = idx_end - seg_start - 12
        if 0 <= b_end < B:
            b0 = max(0, b_end - span + 1)
            d = np.load(paths.residual_npz(dataset, split, 42, backbone))
            trend = d["y_pred"][b0:b_end + 1, node, 0]
            res = d["residuals"][b0:b_end + 1, node, 0]
            mins = (np.arange(b0, b_end + 1) - b_end) * 5.0
            return mins, trend + res, trend, res, split
    raise SystemExit(f"index {idx_end} not reachable in any {dataset} segment")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["hysrd", "gwn"])
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = paths.analysis_dir("hysrd_check")
    for bb in args.backbones:
        if not paths.residual_npz(PANELS[0][0], "train", 42, bb).exists():
            print(f"[{bb}] no residual cache yet, skipping")
            continue
        fig, axes = plt.subplots(2, 2, figsize=(13, 6.5), dpi=150)
        for ax, (ds, node, idx, span) in zip(axes.flat, PANELS):
            if not paths.residual_npz(ds, "train", 42, bb).exists():
                ax.text(0.5, 0.5, f"{ds}: cache pending", ha="center",
                        va="center", transform=ax.transAxes, color="#888888")
                ax.set_axis_off()
                continue
            mins, truth, trend, res, split = panel_series(ds, bb, node, idx, span)
            ax.plot(mins, truth, color=C_TRUTH, lw=0.9, label="ground truth")
            ax.plot(mins, trend, color=C_TREND, lw=1.2, label="trend (1-step)")
            ax.plot(mins, res, color=C_RES, lw=0.9, label="residual")
            ax.axhline(0, color="#bbbbbb", lw=0.6, ls=":")
            ax.set_title(f"{ds} - node {node}  (idx {idx}, {split})", fontsize=10)
            ax.set_xlabel("time (mins relative to current time)", fontsize=8)
            ax.set_ylabel("value / residual", fontsize=8)
            ax.grid(True, color="#ececec", lw=0.5)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=8)
        axes.flat[0].legend(fontsize=8, loc="center right", framealpha=0.95,
                            edgecolor="#dddddd")
        label = "HySRD ST-TrendNet (re-impl)" if bb == "hysrd" else f"our {bb} backbone"
        fig.suptitle(f"Fig.2-style trend-residual decomposition -- {label}", fontsize=11)
        fig.tight_layout()
        p = out_dir / f"fig2_{bb}.png"
        fig.savefig(p)
        plt.close(fig)
        print(f"[{bb}] {p}")


if __name__ == "__main__":
    main()
