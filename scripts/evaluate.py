"""Full evaluation on the test set: all deterministic and probabilistic metrics.

Usage:
    python scripts/evaluate.py --dataset metrla --seed 0

Implemented in Phase 7/8. Stub only at Phase 0.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on test set.")
    parser.add_argument("--dataset", choices=["metrla", "pemsbay"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raise NotImplementedError(
        f"Evaluation not yet implemented (Phase 7). "
        f"Dataset: {args.dataset}, seed: {args.seed}"
    )


if __name__ == "__main__":
    main()
