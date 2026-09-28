"""Inspect the raw HMDA extract through the deterministic data contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data import load_and_prepare, load_data_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/data.yaml"),
        help="Repository-relative data configuration path.",
    )
    parser.add_argument("--data", type=Path, help="Optional raw-path override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_data_config(args.config)
    prepared = load_and_prepare(args.data or config.raw_path)
    result = prepared.quality.to_dict()
    result["source_sha256"] = prepared.source_sha256
    result["target_mapping"] = {
        "loan_approved=0": "target_denied=1",
        "loan_approved=1": "target_denied=0",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
