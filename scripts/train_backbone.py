"""Train the STID backbone and save the best checkpoint.

Usage:
    python scripts/train_backbone.py --dataset metrla --seed 0 [--epochs N] [--smoke]

Implemented in Phase 2. Stub only at Phase 0.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train STID backbone.")
    parser.add_argument("--dataset", choices=["metrla", "pemsbay"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke-test mode: batch_size=16, skip wandb, 1 epoch."
    )
    args = parser.parse_args()
    raise NotImplementedError(
        f"Backbone training not yet implemented (Phase 2). "
        f"Dataset: {args.dataset}, seed: {args.seed}, epochs: {args.epochs}"
    )


if __name__ == "__main__":
    main()
