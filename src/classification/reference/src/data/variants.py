"""Train-only imbalance strategies represented by compact index adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight

from .schema import ANALYTICAL_TARGET
from .splitting import DEFAULT_SEED, SplitIndices, hash_indices

VARIANT_NAMES = ("original_weighted", "oversampled", "undersampled")


def _hash_sample_weights(weights: np.ndarray) -> str:
    values = np.asarray(weights, dtype="<f8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class DatasetVariant:
    name: str
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    sample_weights: np.ndarray | None
    class_weights: dict[int, float] | None
    sampler: str
    seed: int

    def train_target(self, modeling: pd.DataFrame) -> np.ndarray:
        return modeling.loc[self.train_indices, ANALYTICAL_TARGET].to_numpy(dtype=np.int8)

    def class_counts(self, modeling: pd.DataFrame) -> dict[int, int]:
        labels, counts = np.unique(self.train_target(modeling), return_counts=True)
        return {int(label): int(count) for label, count in zip(labels, counts)}


def _resampled_source_indices(
    train_indices: np.ndarray,
    target: np.ndarray,
    sampler: RandomOverSampler | RandomUnderSampler,
) -> np.ndarray:
    # Sampling local row numbers, instead of the feature matrix, prevents large
    # duplicated datasets and makes the exact realization reloadable.
    local_rows = np.arange(len(train_indices), dtype=np.int64).reshape(-1, 1)
    resampled_local, _ = sampler.fit_resample(local_rows, target)
    return train_indices[resampled_local[:, 0]].astype(np.int64, copy=False)


def build_dataset_variants(
    modeling: pd.DataFrame,
    splits: SplitIndices,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, DatasetVariant]:
    """Build all three adapters while leaving validation/test untouched."""

    splits.validate(modeling)
    train = splits.train.copy()
    validation = splits.validation.copy()
    test = splits.test.copy()
    target = modeling.loc[train, ANALYTICAL_TARGET].to_numpy(dtype=np.int8)
    classes = np.unique(target)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError("Training data must contain both binary classes")

    balanced_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=target
    )
    class_weights = {
        int(label): float(weight) for label, weight in zip(classes, balanced_weights)
    }
    sample_weights = np.asarray(
        [class_weights[int(label)] for label in target], dtype=np.float64
    )

    oversampled = _resampled_source_indices(
        train,
        target,
        RandomOverSampler(random_state=seed, sampling_strategy="auto"),
    )
    undersampled = _resampled_source_indices(
        train,
        target,
        RandomUnderSampler(random_state=seed, sampling_strategy="auto"),
    )
    variants = {
        "original_weighted": DatasetVariant(
            "original_weighted", train, validation, test, sample_weights,
            class_weights, "class_weight_balanced", seed
        ),
        "oversampled": DatasetVariant(
            "oversampled", oversampled, validation, test, None,
            None, "RandomOverSampler", seed
        ),
        "undersampled": DatasetVariant(
            "undersampled", undersampled, validation, test, None,
            None, "RandomUnderSampler", seed
        ),
    }
    validate_dataset_variants(modeling, splits, variants)
    return variants


def validate_dataset_variants(
    modeling: pd.DataFrame,
    splits: SplitIndices,
    variants: dict[str, DatasetVariant],
) -> None:
    if set(variants) != set(VARIANT_NAMES):
        raise AssertionError(f"Expected variants {VARIANT_NAMES}, got {tuple(variants)}")
    allowed_train = set(splits.train.tolist())
    for variant in variants.values():
        if not set(variant.train_indices.tolist()).issubset(allowed_train):
            raise AssertionError(f"{variant.name} contains a non-training row")
        if not np.array_equal(variant.validation_indices, splits.validation):
            raise AssertionError(f"{variant.name} changed validation indices")
        if not np.array_equal(variant.test_indices, splits.test):
            raise AssertionError(f"{variant.name} changed test indices")

    original = variants["original_weighted"]
    if not np.array_equal(original.train_indices, splits.train):
        raise AssertionError("Weighted-original variant changed training rows")
    if original.sample_weights is None or len(original.sample_weights) != len(splits.train):
        raise AssertionError("Weighted-original variant has invalid sample weights")
    if not np.isfinite(original.sample_weights).all() or (original.sample_weights <= 0).any():
        raise AssertionError("Weighted-original sample weights must be finite and positive")

    for name in ("oversampled", "undersampled"):
        counts = variants[name].class_counts(modeling)
        if set(counts) != {0, 1} or counts[0] != counts[1]:
            raise AssertionError(f"{name} is not exactly class-balanced: {counts}")


def persist_variant_manifests(
    variants: dict[str, DatasetVariant],
    modeling: pd.DataFrame,
    splits: SplitIndices,
    output_dir: str | Path,
) -> list[Path]:
    validate_dataset_variants(modeling, splits, variants)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    common_hashes = splits.hashes()
    for name in VARIANT_NAMES:
        variant = variants[name]
        train_file = f"variant_{name}_train_indices.npy"
        np.save(destination / train_file, variant.train_indices, allow_pickle=False)
        weights_file: str | None = None
        weights_sha256: str | None = None
        if variant.sample_weights is not None:
            weights_file = f"variant_{name}_sample_weights.npy"
            np.save(destination / weights_file, variant.sample_weights, allow_pickle=False)
            weights_sha256 = _hash_sample_weights(variant.sample_weights)
        manifest = {
            "manifest_version": 1,
            "name": name,
            "seed": variant.seed,
            "sampler": variant.sampler,
            "scope": "train_only",
            "target": ANALYTICAL_TARGET,
            "train_indices_file": train_file,
            "sample_weights_file": weights_file,
            "sample_weights_sha256": weights_sha256,
            "input_train_rows": int(len(splits.train)),
            "output_train_rows": int(len(variant.train_indices)),
            "input_train_index_sha256": common_hashes["train"],
            "output_train_index_sha256": hash_indices(variant.train_indices),
            "validation_index_sha256": common_hashes["validation"],
            "test_index_sha256": common_hashes["test"],
            "class_counts": {
                str(label): count
                for label, count in variant.class_counts(modeling).items()
            },
            "class_weights": (
                {str(label): weight for label, weight in variant.class_weights.items()}
                if variant.class_weights is not None else None
            ),
        }
        output_path = destination / f"variant_{name}_manifest.json"
        output_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output_paths.append(output_path)
    return output_paths


def load_variant(
    name: str,
    output_dir: str | Path,
    splits: SplitIndices,
) -> DatasetVariant:
    if name not in VARIANT_NAMES:
        raise ValueError(f"Unknown variant: {name}")
    destination = Path(output_dir)
    manifest = json.loads(
        (destination / f"variant_{name}_manifest.json").read_text(encoding="utf-8")
    )
    split_hashes = splits.hashes()
    if manifest["input_train_index_sha256"] != split_hashes["train"]:
        raise RuntimeError(f"Persisted {name} was built from a different training split")
    if manifest["validation_index_sha256"] != split_hashes["validation"]:
        raise RuntimeError(f"Persisted {name} references a different validation split")
    if manifest["test_index_sha256"] != split_hashes["test"]:
        raise RuntimeError(f"Persisted {name} references a different test split")
    train = np.load(destination / manifest["train_indices_file"], allow_pickle=False)
    if (
        len(train) != manifest["output_train_rows"]
        or hash_indices(train) != manifest["output_train_index_sha256"]
    ):
        raise RuntimeError(f"Persisted {name} indices fail manifest validation")
    weights = None
    if manifest["sample_weights_file"]:
        weights = np.load(destination / manifest["sample_weights_file"], allow_pickle=False)
        if (
            len(weights) != len(train)
            or _hash_sample_weights(weights) != manifest["sample_weights_sha256"]
        ):
            raise RuntimeError(f"Persisted {name} weights fail manifest validation")
    class_weights = manifest["class_weights"]
    return DatasetVariant(
        name=name,
        train_indices=train,
        validation_indices=splits.validation.copy(),
        test_indices=splits.test.copy(),
        sample_weights=weights,
        class_weights=(
            {int(label): float(weight) for label, weight in class_weights.items()}
            if class_weights is not None else None
        ),
        sampler=manifest["sampler"],
        seed=int(manifest["seed"]),
    )
