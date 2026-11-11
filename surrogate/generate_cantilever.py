#!/usr/bin/env python3
"""Generate fixed training and reference-test data for the Cantilever problem."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def cantilever_tip_deflection(x: np.ndarray) -> np.ndarray:
    """Dimensionless tip deflection: 4 F L^3 / (E b h^3)."""
    force, length, modulus, width, height = x.T
    return 4.0 * force * length**3 / (modulus * width * height**3)


def sample_inputs(rng: np.random.Generator, samples: int) -> np.ndarray:
    # Dimensionless factors around a nominal physical design. Keeping the
    # domain explicit makes this deterministic simulator easy to audit.
    lower = np.array([0.5, 0.8, 0.8, 0.7, 0.7])
    upper = np.array([1.5, 1.2, 1.2, 1.3, 1.3])
    return rng.uniform(lower, upper, size=(samples, len(lower)))


def write_dataset(path: Path, x: np.ndarray) -> None:
    frame = pd.DataFrame(x, columns=["force", "length", "modulus", "width", "height"])
    frame["target"] = cantilever_tip_deflection(x)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression="gzip")


def main() -> None:
    """Parse generation options and write both fixed dataset tables."""

    parser = argparse.ArgumentParser(
        description="Generate deterministic Cantilever surrogate data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cantilever"),
        help="directory for train.tsv.gz and test.tsv.gz",
    )
    parser.add_argument(
        "--train-size", type=int, default=400, help="number of training observations"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=4000,
        help="number of reference-test observations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260810,
        help="seed for consecutive training and reference-test draws",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    write_dataset(args.output_dir / "train.tsv.gz", sample_inputs(rng, args.train_size))
    write_dataset(args.output_dir / "test.tsv.gz", sample_inputs(rng, args.test_size))


if __name__ == "__main__":
    main()

