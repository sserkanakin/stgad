"""Extract backbone residuals (y_true - y_pred) and save to results/residuals/.

Usage:
    python scripts/extract_residuals.py --dataset metrla --seed 0

Implemented in Phase 2. Stub only at Phase 0.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Extract backbone residuals.")
    parser.add_argument("--dataset", choices=["metrla", "pemsbay"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raise NotImplementedError(
        f"Residual extraction not yet implemented (Phase 2). "
        f"Dataset: {args.dataset}, seed: {args.seed}"
    )


if __name__ == "__main__":
    main()
