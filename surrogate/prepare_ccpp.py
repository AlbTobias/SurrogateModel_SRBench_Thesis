#!/usr/bin/env python3
"""Prepare fixed training and reference-test data for the UCI CCPP problem."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd


SOURCE_URL = "https://archive.ics.uci.edu/static/public/294/data.csv"
SOURCE_SHA256 = "3c1fc11025f8424f8d95802d8b7086dffd3f73a552c6dcab3d973620986194b2"
SOURCE_COLUMNS = ("AT", "V", "AP", "RH", "PE")
FEATURE_COLUMNS = (
    "ambient_temperature",
    "exhaust_vacuum",
    "ambient_pressure",
    "relative_humidity",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(path: Path) -> None:
    """Download the authoritative UCI CSV once and verify its content hash."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".download")
        with urlopen(SOURCE_URL, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(path)
    actual_hash = sha256(path)
    if actual_hash != SOURCE_SHA256:
        raise ValueError(
            f"Unexpected CCPP source SHA-256 for {path}: {actual_hash}; "
            f"expected {SOURCE_SHA256}"
        )


def load_source(path: Path) -> pd.DataFrame:
    """Load, validate, and rename the pinned UCI CCPP table."""

    frame = pd.read_csv(path)
    if tuple(frame.columns) != SOURCE_COLUMNS:
        raise ValueError(
            f"Unexpected CCPP columns {tuple(frame.columns)}; expected {SOURCE_COLUMNS}"
        )
    if frame.shape != (9568, 5):
        raise ValueError(f"Unexpected CCPP shape {frame.shape}; expected (9568, 5)")
    if frame.isna().any().any():
        raise ValueError("CCPP source unexpectedly contains missing values")
    frame.columns = (*FEATURE_COLUMNS, "target")
    return frame


def split_dataset(
    frame: pd.DataFrame, seed: int, train_size: int, test_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select deterministic, disjoint rows without altering their values."""

    if train_size + test_size > len(frame):
        raise ValueError(
            "Requested training and reference-test sizes exceed the source dataset"
        )
    indices = np.random.default_rng(seed).permutation(len(frame))
    train = frame.iloc[indices[:train_size]].reset_index(drop=True)
    test = frame.iloc[indices[train_size : train_size + test_size]].reset_index(drop=True)
    return train, test


def write_dataset(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression="gzip")


def main() -> None:
    """Parse preparation options and write both fixed dataset tables."""

    parser = argparse.ArgumentParser(
        description="Prepare deterministic UCI CCPP surrogate data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ccpp"),
        help="directory for the cached source and generated tables",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="optional pinned source CSV; download it when omitted",
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
        "--seed", type=int, default=20260815, help="deterministic split seed"
    )
    args = parser.parse_args()

    source = args.source or args.output_dir / "source.csv"
    if args.source is None:
        download_source(source)
    elif sha256(source) != SOURCE_SHA256:
        raise ValueError(f"Supplied source does not match the pinned CCPP dataset: {source}")
    train, test = split_dataset(
        load_source(source), args.seed, args.train_size, args.test_size
    )
    write_dataset(args.output_dir / "train.tsv.gz", train)
    write_dataset(args.output_dir / "test.tsv.gz", test)


if __name__ == "__main__":
    main()
