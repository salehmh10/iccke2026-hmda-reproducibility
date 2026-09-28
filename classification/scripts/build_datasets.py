"""Build common split and train-only imbalance index manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data import (
    build_dataset_variants,
    load_and_prepare,
    load_data_config,
    make_stratified_split,
    persist_split_manifest,
    persist_variant_manifests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/data.yaml"),
        help="Repository-relative data configuration path.",
    )
    parser.add_argument("--data", type=Path, help="Optional raw-path override.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Repository-relative output directory for compact manifests.",
    )
    parser.add_argument("--seed", type=int, help="Optional configured-seed override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_data_config(args.config)
    data_path = args.data or config.raw_path
    seed = args.seed if args.seed is not None else config.seed
    prepared = load_and_prepare(data_path)
    splits = make_stratified_split(
        prepared.modeling,
        seed=seed,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )
    split_manifest = persist_split_manifest(
        splits,
        prepared.modeling,
        args.output_dir,
        source_sha256=prepared.source_sha256,
    )
    variants = build_dataset_variants(prepared.modeling, splits, seed=seed)
    variant_manifests = persist_variant_manifests(
        variants, prepared.modeling, splits, args.output_dir
    )
    print(
        json.dumps(
            {
                "source_sha256": prepared.source_sha256,
                "raw_rows": prepared.quality.raw_rows,
                "modeling_rows": prepared.quality.modeling_rows,
                "duplicates_removed": prepared.quality.duplicate_rows_removed,
                "split_manifest": split_manifest.as_posix(),
                "variant_manifests": [path.as_posix() for path in variant_manifests],
                "split_rows": {
                    "train": len(splits.train),
                    "validation": len(splits.validation),
                    "test": len(splits.test),
                },
                "variant_train_rows": {
                    name: len(variant.train_indices) for name, variant in variants.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
