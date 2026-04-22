"""Train the conditional residual diffusion model.

Usage:
    python scripts/train_diffusion.py --dataset metrla --seed 0 [--epochs N]

Implemented in Phase 5/6. Stub only at Phase 0.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train the diffusion model.")
    parser.add_argument("--dataset", choices=["metrla", "pemsbay"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    raise NotImplementedError(
        f"Diffusion training not yet implemented (Phase 5). "
        f"Dataset: {args.dataset}, seed: {args.seed}, epochs: {args.epochs}"
    )


if __name__ == "__main__":
    main()
