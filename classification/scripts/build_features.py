"""Fit and serialize HMDA features using the persisted split-index manifest.

The source CSV is loaded through the validated data pipeline, but only rows in
the persisted training-index manifest are passed to ``fit``. Validation/test
rows are passed only to ``transform``. No transformed dataset is duplicated.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy import sparse


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data import load_and_prepare, load_split_manifest  # noqa: E402
from src.features import build_feature_pipeline  # noqa: E402


def _repository_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError(f"Path must remain inside the repository: {value}") from error
    return resolved


def _assert_finite(matrix: object, *, split_name: str) -> None:
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all():
        raise ValueError(f"{split_name} feature matrix contains NaN or infinity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-csv",
        default="hmda_classification_stratified_500k.csv",
        help="Repository-relative source CSV path.",
    )
    parser.add_argument(
        "--manifest-dir",
        default="data/manifests",
        help="Repository-relative directory containing the common split manifest.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/preprocessors/hmda_feature_pipeline.joblib",
        help="Repository-relative serialized pipeline destination.",
    )
    parser.add_argument(
        "--metadata-output",
        help="Optional metadata JSON path (default: next to --output).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacement of existing pipeline/metadata artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = _repository_path(args.raw_csv)
    manifest_dir = _repository_path(args.manifest_dir)
    output_path = _repository_path(args.output)
    metadata_path = (
        _repository_path(args.metadata_output)
        if args.metadata_output
        else output_path.with_suffix(".metadata.json")
    )
    existing_outputs = [path for path in (output_path, metadata_path) if path.exists()]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing artifacts without --overwrite: "
            f"{existing_outputs}"
        )
    manifest_path = manifest_dir / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {manifest_path}")

    prepared = load_and_prepare(raw_path)
    manifest_metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_metadata["source_sha256"] != prepared.source_sha256:
        raise ValueError("Split manifest source hash does not match the source CSV")
    splits = load_split_manifest(manifest_dir)
    splits.validate(prepared.modeling)

    pipeline = build_feature_pipeline()
    train_matrix = pipeline.fit_transform(prepared.modeling.loc[splits.train])
    _assert_finite(train_matrix, split_name="train")

    split_shapes: dict[str, list[int]] = {
        "train": [int(train_matrix.shape[0]), int(train_matrix.shape[1])]
    }
    for name in ("validation", "test"):
        matrix = pipeline.transform(prepared.modeling.loc[getattr(splits, name)])
        _assert_finite(matrix, split_name=name)
        if matrix.shape[1] != train_matrix.shape[1]:
            raise AssertionError(f"{name} transformed feature count changed")
        split_shapes[name] = [int(matrix.shape[0]), int(matrix.shape[1])]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit_split": "train",
        "source": str(raw_path.relative_to(REPOSITORY_ROOT)),
        "source_sha256": prepared.source_sha256,
        "split_manifest": str(manifest_path.relative_to(REPOSITORY_ROOT)),
        "split_index_sha256": splits.hashes(),
        "transformed_shapes": split_shapes,
        "feature_count": int(train_matrix.shape[1]),
        "feature_names": pipeline.get_feature_names_out().tolist(),
        "validation_and_test_fit": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
