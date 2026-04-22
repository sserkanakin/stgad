"""Fit Kronecker prior parameters from training-set residuals.

Usage:
    python scripts/fit_prior.py --dataset metrla --seed 0

Implemented in Phase 4. Stub only at Phase 0.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Fit prior parameters from residuals.")
    parser.add_argument("--dataset", choices=["metrla", "pemsbay"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raise NotImplementedError(
        f"Prior fitting not yet implemented (Phase 4). "
        f"Dataset: {args.dataset}, seed: {args.seed}"
    )


if __name__ == "__main__":
    main()
