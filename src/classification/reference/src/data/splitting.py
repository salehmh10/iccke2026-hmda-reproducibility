"""Leakage-safe common train/validation/test split manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .schema import ANALYTICAL_TARGET

DEFAULT_SEED = 20260809


def hash_indices(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    seed: int = DEFAULT_SEED
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    test_fraction: float = 0.20

    def validate(self, modeling: pd.DataFrame) -> None:
        sets = [set(part.tolist()) for part in (self.train, self.validation, self.test)]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise AssertionError("Train, validation, and test indices overlap")
        if sets[0] | sets[1] | sets[2] != set(modeling.index.tolist()):
            raise AssertionError("Split indices do not exactly cover the modeling view")
        for name, values in zip(("train", "validation", "test"), sets):
            labels = set(modeling.loc[list(values), ANALYTICAL_TARGET].astype(int).tolist())
            if labels != {0, 1}:
                raise AssertionError(f"{name} does not contain both classes")

    def hashes(self) -> dict[str, str]:
        return {
            "train": hash_indices(self.train),
            "validation": hash_indices(self.validation),
            "test": hash_indices(self.test),
        }


def make_stratified_split(
    modeling: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
) -> SplitIndices:
    """Make the documented random fallback split (the extract has no time field)."""

    fractions = train_fraction + validation_fraction + test_fraction
    if not np.isclose(fractions, 1.0):
        raise ValueError(f"Split fractions must total 1.0, got {fractions}")
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("Every split fraction must be positive")
    if ANALYTICAL_TARGET not in modeling:
        raise ValueError(f"Missing analytical target {ANALYTICAL_TARGET}")

    source_indices = modeling.index.to_numpy(dtype=np.int64, copy=True)
    target = modeling[ANALYTICAL_TARGET].to_numpy(dtype=np.int8, copy=False)
    holdout_fraction = validation_fraction + test_fraction
    train_idx, holdout_idx, _, holdout_y = train_test_split(
        source_indices,
        target,
        train_size=train_fraction,
        test_size=holdout_fraction,
        stratify=target,
        random_state=seed,
        shuffle=True,
    )
    relative_validation_fraction = validation_fraction / holdout_fraction
    validation_idx, test_idx = train_test_split(
        holdout_idx,
        train_size=relative_validation_fraction,
        test_size=1.0 - relative_validation_fraction,
        stratify=holdout_y,
        random_state=seed,
        shuffle=True,
    )
    result = SplitIndices(
        train=np.sort(train_idx.astype(np.int64, copy=False)),
        validation=np.sort(validation_idx.astype(np.int64, copy=False)),
        test=np.sort(test_idx.astype(np.int64, copy=False)),
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    result.validate(modeling)
    return result


def _class_counts(modeling: pd.DataFrame, indices: np.ndarray) -> dict[str, int]:
    counts = modeling.loc[indices, ANALYTICAL_TARGET].value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def persist_split_manifest(
    splits: SplitIndices,
    modeling: pd.DataFrame,
    output_dir: str | Path,
    *,
    source_sha256: str,
) -> Path:
    """Persist compact indices and a deterministic, self-verifying manifest."""

    splits.validate(modeling)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    file_names = {
        "train": "split_train.npy",
        "validation": "split_validation.npy",
        "test": "split_test.npy",
    }
    for name, file_name in file_names.items():
        np.save(destination / file_name, getattr(splits, name), allow_pickle=False)

    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "strategy": "stratified_random_fallback_no_temporal_field",
        "seed": splits.seed,
        "fractions": {
            "train": splits.train_fraction,
            "validation": splits.validation_fraction,
            "test": splits.test_fraction,
        },
        "index_basis": "zero_based_raw_csv_data_row_number",
        "source_sha256": source_sha256,
        "modeling_rows": len(modeling),
        "modeling_index_sha256": hash_indices(
            np.sort(modeling.index.to_numpy(dtype=np.int64))
        ),
        "duplicate_policy": "drop_exact_full_raw_row_keep_first_before_split",
        "target": {
            "name": ANALYTICAL_TARGET,
            "mapping": {"loan_approved=0": 1, "loan_approved=1": 0},
            "positive_class": "denial",
        },
        "splits": {
            name: {
                "file": file_name,
                "rows": int(len(getattr(splits, name))),
                "index_sha256": splits.hashes()[name],
                "class_counts": _class_counts(modeling, getattr(splits, name)),
            }
            for name, file_name in file_names.items()
        },
    }
    manifest_path = destination / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def load_split_manifest(output_dir: str | Path) -> SplitIndices:
    destination = Path(output_dir)
    manifest = json.loads((destination / "split_manifest.json").read_text(encoding="utf-8"))
    arrays: dict[str, np.ndarray] = {}
    for name in ("train", "validation", "test"):
        details = manifest["splits"][name]
        values = np.load(destination / details["file"], allow_pickle=False)
        if len(values) != details["rows"] or hash_indices(values) != details["index_sha256"]:
            raise DataIntegrityError(f"Persisted {name} indices fail manifest validation")
        arrays[name] = values
    fractions = manifest["fractions"]
    return SplitIndices(
        seed=int(manifest["seed"]),
        train_fraction=float(fractions["train"]),
        validation_fraction=float(fractions["validation"]),
        test_fraction=float(fractions["test"]),
        **arrays,
    )


class DataIntegrityError(RuntimeError):
    """Raised when a persisted manifest does not match its array artifact."""
