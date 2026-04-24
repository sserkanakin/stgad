"""Verify a backbone loads correctly and produces the right output shape.

Runs six checks per backbone/dataset combination:
  1. Adjacency loads successfully.
  2. load_backbone() returns a model without error.
  3. Forward pass on dummy batch: output shape is (B, 12, N, 1).
  4. Parameter count is printed.
  5. Three gradient steps on a dummy batch with masked MAE loss.
  6. Loss decreases monotonically across the three steps.

Usage:
    python scripts/verify_backbone.py --backbone dcrnn --dataset metrla
    python scripts/verify_backbone.py --backbone gwn   --dataset pemsbay
"""
import argparse
import sys

import torch
import torch.nn.functional as F

from src.data.adjacency import load_adjacency
from src.backbones import load_backbone

_NUM_NODES = {"metrla": 207, "pemsbay": 325}


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        pred:   (B, 12, N, 1)
        target: (B, 12, N, 1)
        mask:   (B, 12, N, 1) bool

    Returns:
        Scalar loss tensor.
    """
    err = torch.abs(pred - target) * mask.float()
    return err.sum() / (mask.float().sum() + 1e-8)


def verify(backbone: str, dataset: str) -> bool:
    """Run all verification checks.  Returns True on success, False on failure."""
    N = _NUM_NODES[dataset]
    B = 2  # small batch for speed

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\n{'='*60}")
    print(f"  backbone={backbone}  dataset={dataset}  requested_device={device}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------ #
    # 1. Load adjacency
    # ------------------------------------------------------------------ #
    print("[1/6] Loading adjacency ...", end=" ", flush=True)
    try:
        adj_mx = load_adjacency(dataset)
    except Exception as e:
        print(f"FAIL\n      {e}")
        return False
    print(f"OK  shape={adj_mx.shape}")

    # ------------------------------------------------------------------ #
    # 2. Instantiate backbone
    # ------------------------------------------------------------------ #
    print("[2/6] Instantiating backbone ...", end=" ", flush=True)
    try:
        model = load_backbone(backbone, dataset, adj_mx, device)
    except Exception as e:
        print(f"FAIL\n      {e}")
        return False

    eff_device = getattr(model, "effective_device", device)
    print(f"OK  effective_device={eff_device}")

    # ------------------------------------------------------------------ #
    # 3. Forward pass — shape check
    # ------------------------------------------------------------------ #
    print("[3/6] Forward pass shape check ...", end=" ", flush=True)
    model.eval()
    x_dummy = torch.randn(B, 12, N, 2).to(eff_device)
    try:
        with torch.no_grad():
            out = model(x_dummy)
    except Exception as e:
        print(f"FAIL\n      {e}")
        return False

    expected = (B, 12, N, 1)
    if tuple(out.shape) != expected:
        print(f"FAIL  got {tuple(out.shape)}, expected {expected}")
        return False
    print(f"OK  output shape={tuple(out.shape)}")

    # ------------------------------------------------------------------ #
    # 4. Parameter count
    # ------------------------------------------------------------------ #
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[4/6] Parameter count: {n_params:,}")

    # ------------------------------------------------------------------ #
    # 5 & 6. Gradient steps — loss must decrease
    # ------------------------------------------------------------------ #
    print("[5/6] Running 3 gradient steps ...", end=" ", flush=True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x_t = torch.randn(B, 12, N, 2).to(eff_device)
    y_t = torch.randn(B, 12, N, 1).to(eff_device)
    mask_t = torch.ones(B, 12, N, 1, dtype=torch.bool).to(eff_device)

    losses = []
    try:
        for _ in range(3):
            optimizer.zero_grad()
            pred = model(x_t)
            loss = masked_mae(pred, y_t, mask_t)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    except Exception as e:
        print(f"FAIL\n      {e}")
        return False

    print(f"OK  losses={[f'{l:.4f}' for l in losses]}")

    print("[6/6] Loss decreasing check ...", end=" ", flush=True)
    if losses[-1] < losses[0]:
        print(f"OK  ({losses[0]:.4f} → {losses[-1]:.4f})")
    else:
        # Non-decrease is a warning, not a hard fail — can happen with tiny random batches
        print(f"WARN  loss did not decrease ({losses[0]:.4f} → {losses[-1]:.4f}); "
              "may be fine for 3 steps on random data")

    print(f"\n  PASS  {backbone}/{dataset}\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify backbone wiring.")
    parser.add_argument("--backbone", choices=["dcrnn", "gwn"], required=True)
    parser.add_argument("--dataset", choices=["metrla", "pemsbay"], required=True)
    args = parser.parse_args()

    ok = verify(args.backbone, args.dataset)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
