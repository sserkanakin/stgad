"""Preprocess raw .h5 files into sliding-window .npz files.

Usage:
    python scripts/preprocess.py --dataset metrla
    python scripts/preprocess.py --dataset pemsbay

Implemented in Phase 1. Stub only at Phase 0.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw traffic data.")
    parser.add_argument(
        "--dataset", choices=["metrla", "pemsbay"], required=True,
        help="Dataset to preprocess."
    )
    args = parser.parse_args()
    raise NotImplementedError(f"Preprocessing not yet implemented (Phase 1). Dataset: {args.dataset}")


if __name__ == "__main__":
    main()
